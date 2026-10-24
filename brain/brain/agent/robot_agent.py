"""One decision step of a robot's agent: an event comes in, the model calls OpenCV 5 tools, then orders actions.

The reflex brain in Unity keeps driving, mapping and avoiding heat on its own. The recorder calls this step only on
events worth a decision (a warm candidate, poor visibility, rising heat, being stuck, a partner's message, a periodic
check). Every model reply, tool call and refusal is written to a trace, so a run can be audited and replayed.

    agent = RobotAgent(backend, trace)
    out = agent.step(request)       # request as sent by the recorder, see STEP_FIELDS
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from brain.agent.llm import Reply, append_jsonl, request_key
from brain.agent import tools as _tools
from brain.agent.tools import ALL_TOOLS, AgentContext, ToolRegistry, survivors_to_rescue
from brain.agent.team import TeamState

# Model rounds per step. Scan, inspect, act and finish took exactly 4 rounds with one tool call per reply, so a 4-round
# limit cut the step before finish: in held-out runs 1 and 2, 60 of 84 steps hit it, 41 lost their summary and 2 lost a
# decision (issue list 23 Oct, item 10). 6 leaves two rounds of slack; each round costs about 3 s of wall time (the
# simulation clock waits), and STEP_DEADLINE_S still bounds the step.
MAX_ROUNDS = 6
STEP_DEADLINE_S = 75.0     # past this the step ends with what it has; the reflex layer keeps the robot safe meanwhile
REPEAT_WINDOW_S = 12.0     # the reflex brain re-raises the same candidate every few seconds; one decision covers it

SYSTEM = """You are the decision layer of a quadruped search-and-rescue robot inside a burning building.

Goal: find every person in the building and get them out alive, while keeping your robot able to work; it may be the
only thing that can reach them. Nobody has given you a map, a floor plan or a list of rooms: everything you know you
sensed yourself or heard on the radio. How you search, in what order, and what you tell others is your judgement.

What executes your decisions: a reflex controller plans paths on the robot's own LiDAR and camera map, keeps clear of
walls, brakes for obstacles and backs away from heat on its own. If you give no movement order, it continues its own
ranked search (frontiers are ranked by code: travel time, rooms not yet entered, corridor openings not yet looked
through), and sweeps its camera over floor it has not yet searched. It counts floor as searched once its camera has
seen it within 4 m, so a small room seen whole from its doorway may not be walked into. You need not re-order the search every step: order a move only when your own
evidence changes where to go (a person, fire on the route, a proposal you accepted). It leaves the building when its own heat rules say so, whatever you ordered.

Limits the tools enforce (an action that breaks one is refused, and the refusal is returned to you):
- Every action except go_to, explore_frontier on proven floor, turn_to_face, send_team_message and finish cites the
  evidence_id of a perception result from this step, of the kind its description names.
- confirm_survivor needs an inspect_candidate verdict of person taken within 5 m, or, up to 8 m, two person verdicts on
  the same spot from standpoints at least 0.5 m apart that agree within 0.5 m; reject_candidate needs not_person. extract_survivor
  needs the person within the rescue reach and the grip cone (numbers in its description), cited either by the evidence
  that confirmed them (survivors_to_rescue) or by a person verdict taken within that reach. The robot never confirms a
  person by itself.
- explore_frontier on a frontier with proven_indoors false needs a check_indoors result saying indoors within its
  bearing tolerance of that frontier. go_to is refused where no ceiling has been seen.
- No exploring or go_to during a heat event (avoid_fire, retreat).
- turn_to_face is refused when the point is already inside the thermal camera's view or under the robot, and after
  two turns toward the same spot within 90 s.

Operator rule (no tool enforces it; the incident commander requires it): request_human_approval before entering a room
where fire is spreading toward you.

Facts about your robot and the context:
- Positions are in YOUR OWN map frame, metres from where you entered: y is the direction you faced on entry, x to its
  right. Your partner's numbers are in its own frame, so its coordinates are not positions in yours. Once partners
  shows maps_linked, your robot matched LiDAR fingerprints with that partner: its position and the area it searched
  are converted into your frame, and your frontier list leaves its searched area out.
- Heat, from your own air sensor 0.6 m above the floor: air_c is the air now (C), air_trend_c_per_min its trend over
  the last seconds, heat_budget how much heat the robot can still take (1 full, 0 empty), and heat_seconds_left how
  long the budget lasts at this air (absent below 50 C, where it does not drain). The budget drains by 0.002 per second
  for every degree above 50 C: at 100 C it lasts about 10 s, at 380 C under 2 s. The air near the floor can jump from
  40 C to over 300 C within seconds when the hot smoke layer comes down, so a fast rising trend is the warning, at any
  temperature. nearest_fire is the closest floor your own thermal map measured burning, anywhere in your map, with
  bearing, distance and burning area; absent when your map holds none.
- looks_not_finished: warm candidates you set out to look at closer whose approach was dropped (dropped_because: a
  heat event, no path to it, or timed out), not yet confirmed or rejected by you or confirmed by the team.
- frontiers: floor your robot can reach that no camera on the team has searched. Each has id, x, y, path_m (metres along
  the planned route; path_s, when present, the estimated seconds of walking), unsearched_m2 (unsearched floor right beside the frontier), mapped_unsearched_m2
  (indoor floor your own sensors have mapped beyond it, up to the LiDAR's range, that no camera has searched; floor
  never mapped at all is not counted), proven_indoors (your own ceiling or wall evidence says it is inside) and its
  distance from your partner's claimed area.
- Floor counts as searched only where the camera's depth rays came within 0.3 m of it; floor seen past furniture from
  above is mapped but not searched.
- The colour camera sees about 37 degrees either side of straight ahead; turn_to_face brings a bearing into view.
  A person lying within the arm's reach is below the camera's view, so an inspection there may say uncertain.
- lidar gives the nearest obstacle in eight directions. The robot's body radius is 0.25 m.
- A warm candidate with on_wall 1 is a thermal-only warm patch on a long mapped wall (the structure around it runs on
  for 2 m or more). Fire heats walls: in 15 recorded runs 1 of 151 such patches was a person. Inspect it from where you
  stand if you like; turning or walking to it is refused while frontiers are offered, unless an inspection shows a colour
  person or a shape standing off the wall. Colour candidates were people 13 times in 16: always look at those.
- previously_rejected 1 on a warm candidate: you rejected a warm region at that spot before, and the robot now sees it
  from a new viewpoint (at least 45 degrees round or half the distance) or the colour detector saw a person there.
- survivors_to_rescue: people already confirmed whom nobody has lifted, with bearing, distance, in_view and the
  evidence that confirmed them. Getting a person out means extract_survivor within the rescue reach.
  A confirmation stays until that person is lifted or withdraw_survivor removes it; the robot's own reflexes never drop it.
- no_frontier: your map has no reachable unsearched floor; the robot returns after a short scan unless you act.
  fire_blocks_frontier, when present, is where the fire margin cuts the route to unsearched floor. fire_blocks_exit,
  when present: no route to a known way out is open, and this is where the fire margin first cuts the shortest one
  (bearing, distance, exit_path_m). blocked_unsearched
  is unsearched floor your planner cannot reach, with its bearing, distance and the nearest reachable approach point.
  water_l is the water left for spray_water.
- go_to, approach_candidate and turn_to_face end with an agent_goal_reached event when done.
- mission_time_left_s: seconds until the robot's operating limit, when it must be out of the building.
- frontiers lists up to 8 options in the code's ranked order, option 0 first; some are the frontiers opening onto the
  largest unsearched areas the map shows, so not every option is close by. Take option 0 unless your evidence says otherwise.
- rooms, when present: the rooms your own map shows, segmented by code at door-width openings (ids are local to your
  robot and can change when a room splits or merges): id, centre x and y, covered_share (share of its sensed indoor floor
  any camera on the team has searched), entered (your robot has stood in it), dwell_s, indoor_floor_m2. A frontier's
  room_id says which room it serves. A room with entered false or covered_share well under 1 is not searched yet.
- last_orders_outcome: what the brain did with the movement orders of your previous step (reached, refused with its
  reason, no path, expired before arrival, overridden by a heat event, under way). An order refused for a reason that
  still holds is not worth repeating.
- confirmed_people: every person the team has already confirmed, with bearing, distance and in_view. A warm shape or a
  person verdict within 1 m of one of them is that same person: do not confirm or report her again, and do not radio a
  "new survivor"; search elsewhere.
- context.scan_thermal is this step's thermal scan, already run by code on the newest frame: use its hot spots and
  person-sized regions and cite its id as evidence; do not spend a round calling scan_thermal again.
- Each perception tool returns the same result if called again in this step. scan_thermal is how you find people and
  fire the robot has not raised yet: one scan on a check-in, an arrival or a new candidate is worth its cost, and a warm
  person-sized region deserves inspect_candidate at once, even during a retreat (inspecting and confirming from where
  the robot stands is always allowed). An inspection that lists other_people saw a whole person elsewhere in the view
  with the colour detector: inspect that bearing and range next. not_person with nothing warm at the spot means the spot
  was in full view and empty, with no person anywhere in the colour image: reject it rather than look again. assess_visibility is for the torch and track_fire_spread for a decision about
  fire or water; the context already carries nearest_fire, fire_near_robot, fire_blocks_frontier, fire_blocks_exit and
  spray_option from the robot's own thermal map, so neither is needed every step.
- send_team_message reaches your partner and the commander only within radio range and through at most one wall;
  what you do not radio, they do not know. Your partner is woken only by a proposal or an answer to its proposal, not
  by findings, so a message is not a way to get its attention; send findings when you have a new one, not every step.
- team: your radio conversation with your partner, kept by code across your steps. awaiting_your_answer lists plans
  the partner proposed that need answer_proposal before they expire; your_proposals shows the status of plans you
  proposed; agreements_in_force are plans both robots accepted, with your part and the partner's; partner_findings are
  its recent reports; your_last_intent is what you set out to do at your previous step. propose_plan suggests a
  division of work. Until your maps are linked, a place in your partner's words is not a position in your frame.
- spray_option, when present: the fire your own thermal map measures where it cuts your way out (or to unsearched
  floor). Water opens a way into rooms not yet searched; never go to a fire that blocks nothing. Bearing, distance, whether it is in the camera's view now (spray_water needs a scan_thermal hot spot within its
  aim tolerance) and within the nozzle's reach, its burning area and estimated heat release, the heat one stream removes,
  water left, knock_down_s_one_robot (seconds of this robot's spray that cool the burning floor below about 300 C, null
  when one stream cannot outpace the fire) and code's assessment. The stream spreads over a wider patch the further it
  travels, so closer spraying puts more of it on the flames. Spraying is expected, not optional, when spray_option
  blocks "frontier" or "exit" and its assessment says this robot alone can knock the fire down: if it says to
  approach first, call approach_fire (it walks to spray_from_m short of "at" and faces it), then on arrival scan_thermal
  and spray_water at its hot spot. Do not walk away from a fire that cuts your way to unsearched floor while the water
  could open it.

Finish every step with one sentence saying what you decided and why: call finish in the same reply as your last action
(one reply may hold several tool calls), not in a round of its own."""


_RESCUE_TEXT = [
    ("find every person in the building and get them out alive", "find every person in the building and report where they are (a reported person needs no robot beside them: the crews outside have the position)"),
    (""" extract_survivor
  needs the person within the rescue reach and the grip cone (numbers in its description), cited either by the evidence
  that confirmed them (survivors_to_rescue) or by a person verdict taken within that reach. The robot never confirms a
  person by itself.""", """ The robot never confirms a person by itself."""),
    ("""- survivors_to_rescue: people already confirmed whom nobody has lifted, with bearing, distance, in_view and the
  evidence that confirmed them. Getting a person out means extract_survivor within the rescue reach.
  A confirmation stays until that person is lifted or withdraw_survivor removes it; the robot's own reflexes never drop it.""",
     """- A confirmation stays until withdraw_survivor removes it; the robot's own reflexes never drop it."""),
]


_LEAVES = "It leaves the building when its own heat rules say so, whatever you ordered."
_STAYS = ("It never leaves the building once inside: when heat builds it moves to the coolest floor it has measured and "
          "searches on from there while its air allows, whatever you ordered; holding still is its own decision at its "
          "hard heat limits and needs no order from you. Spraying its water on the fire is your decision.")


def system_prompt() -> str:
    """The system prompt for this step: with carrying people out switched off (actuators.json rescue.enabled), the goal,
    limits and facts say nothing about lifting, so the model is never told of a tool it does not have. With stay_inside
    (brain.json, the owner's decision of 22 Oct) the robot never leaves, and the model is told so."""
    text = SYSTEM.replace(_LEAVES, _STAYS) if _tools._CONFIG.get("stay_inside") else SYSTEM
    if _tools.RESCUE_ENABLED:
        return text
    for old, new in _RESCUE_TEXT:
        if text.count(old) != 1:
            raise RuntimeError(f"system prompt changed; rescue text not found once: {old[:60]!r}")
        text = text.replace(old, new)
    return text


class RobotAgent:
    def __init__(self, backend, trace_path: Path, max_rounds: int = MAX_ROUNDS):
        self.backend = backend
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_rounds = max_rounds

    def _trace(self, rec: dict) -> None:
        append_jsonl(self.trace_path, rec)

    _recent: dict = {}
    _ordered: dict = {}   # robot -> time of the last step that gave the brain an order

    def _repeat(self, req: dict) -> bool:
        ev = req.get("event") or {}
        n = ev.get("numbers") or {}
        key = (req["robot"], ev.get("kind"))
        here = (n.get("x_m"), n.get("y_m"))
        last = self._recent.get(key)
        self._recent[key] = (float(req["t"]), here)
        if ev.get("kind") in ("periodic", None) or last is None:
            return False
        lt, lpos = last
        # An event with a position repeats only at the same spot: arriving at a second place within the window is a new
        # decision (held-out seed 914 runs 1 to 3: five agent_goal_reached events at new places were dropped, so the reflex
        # chose the next frontier alone). Events without a position repeat by time alone.
        # An arrival answers the order that sent the robot there. After a new order it is a new decision even at the same
        # spot (seed 929, robot A: ordered at 44.0 s to turn and face s4 again, arrived at 46.8 s, and the arrival was
        # dropped as a repeat of the one at 38.2 s that faced the same point; the agent never inspected her again).
        if ev.get("kind") == "agent_goal_reached" and self._ordered.get(req["robot"], float("-inf")) > lt:
            return False
        positional = None not in here and None not in lpos
        near = positional and ((here[0] - lpos[0]) ** 2 + (here[1] - lpos[1]) ** 2) ** 0.5 < 1.5
        return float(req["t"]) - lt < REPEAT_WINDOW_S and (near if positional else ev.get("kind") != "survivor_candidate")

    _run: str | None = None
    _last_t: dict = {}

    @classmethod
    def _new_run(cls, req: dict) -> None:
        """Memory kept across steps (recent events, close looks per spot, people lifted) belongs to one recording. The
        server outlives recordings: in run 24 robot A's approach to a person it had just confirmed was refused as "already
        looked closer 2 times at this spot", counting looks from runs 21 to 23 at the same place."""
        # Every recording is written to the same folder and renamed afterwards, so a new one is also recognised by its
        # clock starting again: a robot's time going backwards.
        run, robot, t = str(Path(req.get("robot_dir") or ".").resolve().parent), req.get("robot"), float(req["t"])
        restarted = t < cls._last_t.get(robot, float("-inf")) - 1.0
        cls._last_t[robot] = t
        if run != cls._run or restarted:
            cls._run = run; cls._last_t = {robot: t}
            cls._recent.clear(); cls._ordered.clear(); ToolRegistry._looks.clear(); ToolRegistry._lifted.clear(); ToolRegistry._confirmed.clear()
            ToolRegistry._pending.clear(); ToolRegistry._unfinished.clear(); TeamState.reset()
            ToolRegistry._inspection_histories.clear(); ToolRegistry._faces.clear(); ToolRegistry._wall_spots.clear(); ToolRegistry._confirm_walks.clear(); ToolRegistry._person_views.clear()
            cls._last_orders.clear()

    _last_orders: dict = {}   # robot -> (t, orders) of its last step that moved the robot

    @staticmethod
    def _radio_needs_decision(team: TeamState, req: dict, new_messages: list[str]) -> bool:
        """A radio text alone is not a decision. Only a proposal still open for this robot's answer, or a new answer to
        a plan it proposed, needs the model now; findings wait in partner_findings for the next real step. Seed 932: 80 of
        116 robot steps were woken by radio texts, 78 of them by findings and commander broadcasts, and each paused the
        simulation 12.6 s (median) to say nothing new."""
        t = float(req["t"])
        if any(p["status"] == "pending" and t <= p["expires_t"] for p in team.received.values()):
            return True
        # A close look under way is finished on any wake-up: seed 932, robot A ordered an approach at 45.0 s and confirmed
        # the person on the radio-woken step at 49.0 s, before the arrival event.
        # Not while the brain is retreating or returning: the look cannot be driven then, and the radio-woken steps only
        # said "I remain in retreat" (seed 932 replay, robot B, eight steps between 108 and 144 s).
        if req.get("state") not in ("retreat", "return"):
            if ToolRegistry._pending.get(req["robot"]) is not None:
                return True
            # A walk to confirm a person seen from too far: seed 932, robot A confirmed s4 on such a radio-woken step.
            walk = ToolRegistry._confirm_walks.get(req["robot"])
            if walk is not None and t - walk["t"] <= float(_tools._PERCEPTION["pending_candidate_s"]):
                return True
        new = set(new_messages)
        for m in req.get("inbox") or []:
            mid = str(m.get("message_id") or f"{m.get('from')}@{m.get('t')}:{str(m.get('text'))[:40]}")
            if mid in new and m.get("kind") in ("proposal", "accept", "reject"):
                return True
        return False

    def step(self, req: dict) -> dict:
        start = time.perf_counter()
        self._new_run(req)
        team = TeamState.of(req["robot"])
        # A radio message not seen before is always decided on, even when the event looks like a repeat: a partner's
        # proposal or report arriving within REPEAT_WINDOW_S of the last step was dropped (Codex design review, 22 Oct).
        new_messages = team.ingest(req.get("inbox") or [], float(req["t"]))
        if (req.get("event") or {}).get("kind") == "team_message" and not self._radio_needs_decision(team, req, new_messages):
            out = {"step": f"{req['robot']}@{float(req['t']):.3f}", "orders": [], "refusals": [], "rounds": 0,
                   "summary": None, "evidence_ids": [], "latency_s": 0.0, "radio_text_only": True}
            self._trace({"kind": "step_skipped", **out, "event": req.get("event"),
                         "why": "radio text only: no proposal to answer and no answer to this robot's proposal"})
            return out
        if self._repeat(req) and not new_messages:
            out = {"step": f"{req['robot']}@{float(req['t']):.3f}", "orders": [], "refusals": [], "rounds": 0,
                   "summary": None, "evidence_ids": [], "latency_s": 0.0, "repeat_of_recent_event": True}
            self._trace({"kind": "step_skipped", **out, "event": req.get("event")})
            return out
        pose = req["pose"]
        ctx = AgentContext(robot=req["robot"], t=float(req["t"]), x=float(pose["x"]), y=float(pose["y"]),
                           heading_deg=float(pose["heading_deg"]),
                           frame_times=[float(s) for s in req.get("frame_times", [])],
                           water_l=float(req.get("water_l") or 0.0), carrying_person=bool(req.get("carrying_person")),
                           event_kind=(req.get("event") or {}).get("kind", ""), torch=req.get("torch", "wide"))
        ctx.frontiers = list(req.get("frontiers") or [])
        ctx.goal = list(req["goal"]) if req.get("goal") else None
        ctx.event = dict(req.get("event") or {})
        ctx.survivors = [list(p) for p in req.get("survivors") or []]
        ctx.maps_linked = any(p.get("maps_linked") for p in req.get("partners") or [])
        ev_now, ev_numbers = req.get("event") or {}, (req.get("event") or {}).get("numbers") or {}
        last = self._last_orders.get(req["robot"])
        outcomes = _tools.order_outcomes(Path(req["robot_dir"]), last[0], last[1], ctx.t) if last else []
        ctx.last_orders = outcomes
        tools = ToolRegistry(Path(req["robot_dir"]), ctx, Path(req.get("debug_dir") or Path(self.trace_path).parent / "agent_debug"))
        context = {k: req[k] for k in ("robot", "t", "pose", "torch", "mode", "state", "goal", "heat_budget", "air_c", "air_trend_c_per_min", "heat_seconds_left", "nearest_fire", "fire_near_robot", "event",
                                       "inbox", "orders", "survivors", "frontiers", "lidar", "partners", "blocked_unsearched", "fire_blocks_frontier", "fire_blocks_exit", "mission_time_left_s",
                                       "water_l", "carrying_person", "rooms") if k in req}
        if _tools.RESCUE_ENABLED:
            context["survivors_to_rescue"] = survivors_to_rescue(req["robot"], pose, req.get("survivors") or [])
        # Every person the team has confirmed, marked for the model with bearing, distance and whether the camera sees the
        # spot now. Seed 934: both robots confirmed one person twice and radioed "new survivor"; the raw survivors list of
        # coordinates was in the context and the model did not tie it to the person in front of it.
        if req.get("survivors"):
            context["confirmed_people"] = [{k: v for k, v in s.items() if k not in ("approach", "confirmed_by")}
                                           for s in survivors_to_rescue(req["robot"], pose, req.get("survivors") or [])]
        context["team"] = team.context(ctx.t)
        spray = _tools.spray_option(req)
        if spray is not None:
            context["spray_option"] = spray
        ctx.spray_at = spray.get("at") if spray is not None else None
        pending = tools.pending()
        if pending is not None:
            context["pending_candidate"] = pending
        unfinished = tools.unfinished()
        if unfinished:
            context["looks_not_finished"] = unfinished
        if outcomes:
            context["last_orders_outcome"] = outcomes
        # The step's thermal scan runs in code before the first round, so the model can inspect or act in round 1. Seed
        # 934: 62 of 72 steps spent a whole round (about 3 s) on scan_thermal first, steps averaged 15 s and reached 28 s
        # at 6 rounds. It is this step's evidence like any other scan; calling scan_thermal again returns the same result.
        # Local recordings only: through the recording transport (remote_frames, set by the server for run:// and pull-v1)
        # frames are fetched on request, and a scan the model did not ask for broke that protocol (8 transport tests).
        scan = {"error": "remote frames: no pre-scan"} if req.get("remote_frames") else None
        try:
            scan = scan or tools.call("scan_thermal", {})
        except Exception as e:                      # no usable frame: the model can still scan, or act without it
            scan = {"error": repr(e)}
        if scan.get("id") and not scan.get("error"):
            context["scan_thermal"] = _compact(scan)
        messages = [{"role": "system", "content": system_prompt()},
                    {"role": "user", "content": "Decision needed. Context:\n" + json.dumps(context, default=str)}]
        step_id = f"{req['robot']}@{float(req['t']):.3f}"
        # The full context the model saw, so every decision can be audited against what the robot knew at that moment.
        self._trace({"kind": "step_start", "step": step_id, "t": req["t"], "robot": req["robot"], "event": req.get("event"),
                     "context": context})
        rounds = 0
        while rounds < self.max_rounds and tools.finished is None:
            if time.perf_counter() - start > STEP_DEADLINE_S:
                self._trace({"kind": "step_deadline", "step": step_id, "after_rounds": rounds})
                break
            rounds += 1
            key = request_key(messages, _tools.tool_specs(ctx.event_kind))
            try:
                reply: Reply = self.backend.chat(messages, _tools.tool_specs(ctx.event_kind))
            except Exception as e:                       # model down or replay miss: the reflex brain carries on
                self._trace({"kind": "model_error", "step": step_id, "error": repr(e)})
                break
            self._trace({"kind": "model_reply", "step": step_id, "round": rounds, "request_key": key, "model": reply.model,
                         "latency_s": round(reply.latency_s, 3), "text": reply.text,
                         "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in reply.tool_calls]})
            if not reply.tool_calls:
                tools.finished = reply.text.strip() or "no action"
                break
            messages.append({"role": "assistant", "content": reply.text,
                             "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in reply.tool_calls]})
            for call in reply.tool_calls:
                result = tools.call(call.name, call.arguments)
                self._trace({"kind": "tool", "step": step_id, "round": rounds, "name": call.name, "arguments": call.arguments,
                             "result_id": result.get("id"), "verdict": result.get("verdict"),
                             **({"reasons": result.get("reasons"), "cues": result.get("cues"),
                                 "candidate": result.get("candidate")} if call.name == "inspect_candidate" else {}),
                             "refused": result.get("refused", False), "reason": result.get("reason"), "error": result.get("error")})
                messages.append({"role": "tool", "name": call.name, "content": json.dumps(_compact(result), default=str)})
        if tools.finished is None and (tools.orders or tools.refusals):
            # Out of rounds or time before finish: the orders already stand, so record what was decided rather than nothing.
            tools.finished = "(no summary: the step ended before finish) " + "; ".join(
                [o.get("type", "order") for o in tools.orders] + [f"refused {r['tool']}" for r in tools.refusals])
        team.note_intent(ctx.t, tools.finished, tools.orders)
        if tools.orders: self._ordered[req["robot"]] = ctx.t
        if any(o.get("type") in ("go_to", "explore_frontier") for o in tools.orders):
            self._last_orders[req["robot"]] = (ctx.t, list(tools.orders))
        elif last and all(o["outcome"].startswith("under way") for o in outcomes):
            pass   # the earlier order is still being driven: keep watching it
        else:
            self._last_orders.pop(req["robot"], None)
        out = {"step": step_id, "orders": tools.orders, "refusals": tools.refusals, "summary": tools.finished,
               "rounds": rounds, "evidence_ids": list(tools.evidence), "latency_s": round(time.perf_counter() - start, 3)}
        self._trace({"kind": "step_end", **out})
        return out


def _compact(result: dict) -> dict:
    """What the model sees of a tool result: drop bulky arrays, keep ids, verdicts, numbers and evidence text."""
    drop = {"intrinsics", "rgb_to_thermal", "stamps", "mask", "profile", "jamb_pixels"}
    out = {}
    for k, v in result.items():
        if k in drop:
            continue
        if isinstance(v, list) and len(v) > 8 and not all(isinstance(x, dict) for x in v):
            continue
        out[k] = v
    return out
