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

MAX_ROUNDS = 4
STEP_DEADLINE_S = 75.0     # past this the step ends with what it has; the reflex layer keeps the robot safe meanwhile
REPEAT_WINDOW_S = 12.0     # the reflex brain re-raises the same candidate every few seconds; one decision covers it

SYSTEM = """You are the decision layer of a quadruped search-and-rescue robot inside a burning building.

Goal: find every person in the building and get them out alive, while keeping your robot able to work; it may be the
only thing that can reach them. Nobody has given you a map, a floor plan or a list of rooms: everything you know you
sensed yourself or heard on the radio. How you search, in what order, and what you tell others is your judgement.

What executes your decisions: a reflex controller plans paths on the robot's own LiDAR and camera map, keeps clear of
walls, brakes for obstacles and backs away from heat on its own. If you give no movement order, it picks the nearest
worthwhile frontier itself. It leaves the building when its own heat rules say so, whatever you ordered.

Limits the tools enforce (an action that breaks one is refused, and the refusal is returned to you):
- Every action except go_to, explore_frontier on proven floor, turn_to_face, send_team_message and finish cites the
  evidence_id of a perception result from this step, of the kind its description names.
- confirm_survivor needs an inspect_candidate verdict of person; reject_candidate needs not_person. extract_survivor
  needs the person within the rescue reach and the grip cone (numbers in its description), cited either by the evidence
  that confirmed them (survivors_to_rescue) or by a person verdict taken within that reach. The robot never confirms a
  person by itself.
- explore_frontier on a frontier with proven_indoors false needs a check_indoors result saying indoors within its
  bearing tolerance of that frontier. go_to is refused where no ceiling has been seen.
- No exploring or go_to during a heat event (avoid_fire, retreat).

Operator rule (no tool enforces it; the incident commander requires it): request_human_approval before entering a room
where fire is spreading toward you.

Facts about your robot and the context:
- Positions are in YOUR OWN map frame, metres from where you entered: y is the direction you faced on entry, x to its
  right. Your partner's numbers are in its own frame, so its coordinates are not positions in yours. Once partners
  shows maps_linked, your robot matched LiDAR fingerprints with that partner: its position and the area it searched
  are converted into your frame, and your frontier list leaves its searched area out.
- frontiers: floor your robot can reach that no camera on the team has searched. Each has id, x, y, path_m (planned
  distance, with heat and wall costs), unsearched_m2 (unsearched floor right beside the frontier), mapped_unsearched_m2
  (indoor floor your own sensors have mapped beyond it, up to the LiDAR's range, that no camera has searched; floor
  never mapped at all is not counted), proven_indoors (your own ceiling or wall evidence says it is inside) and its
  distance from your partner's claimed area.
- Floor counts as searched only where the camera's depth rays came within 0.3 m of it; floor seen past furniture from
  above is mapped but not searched.
- The colour camera sees about 37 degrees either side of straight ahead; turn_to_face brings a bearing into view.
  A person lying within the arm's reach is below the camera's view, so an inspection there may say uncertain.
- lidar gives the nearest obstacle in eight directions. The robot's body radius is 0.25 m.
- A warm candidate with on_wall 1 is a thermal-only warm patch on a long mapped wall (the structure around it runs on
  for 2 m or more). Fire heats walls; people can also lie against them.
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
- frontiers lists up to 8 options, nearest first; some are the frontiers opening onto the largest unsearched areas the map
  shows, so not every option is close by.
- Each perception tool returns the same result if called again in this step.
- send_team_message reaches your partner and the commander only within radio range and through at most one wall;
  what you do not radio, they do not know.
- team: your radio conversation with your partner, kept by code across your steps. awaiting_your_answer lists plans
  the partner proposed that need answer_proposal before they expire; your_proposals shows the status of plans you
  proposed; agreements_in_force are plans both robots accepted, with your part and the partner's; partner_findings are
  its recent reports; your_last_intent is what you set out to do at your previous step. propose_plan suggests a
  division of work. Until your maps are linked, a place in your partner's words is not a position in your frame.
- spray_option, when present: the fire your own thermal map measures where it cuts your way out (or to unsearched
  floor): bearing, distance, whether it is in the camera's view now (spray_water needs a scan_thermal hot spot within its
  aim tolerance) and within the nozzle's reach, its burning area and estimated heat release, the heat one stream removes,
  water left, and code's comparison of the two.

Finish every step with one sentence saying what you decided and why."""


_RESCUE_TEXT = [
    ("find every person in the building and get them out alive", "find every person in the building and report where they are"),
    (""" extract_survivor
  needs the person within the rescue reach and the grip cone (numbers in its description), cited either by the evidence
  that confirmed them (survivors_to_rescue) or by a person verdict taken within that reach. The robot never confirms a
  person by itself.""", """ The robot never confirms a person by itself."""),
    ("""- survivors_to_rescue: people already confirmed whom nobody has lifted, with bearing, distance, in_view and the
  evidence that confirmed them. Getting a person out means extract_survivor within the rescue reach.
  A confirmation stays until that person is lifted or withdraw_survivor removes it; the robot's own reflexes never drop it.""",
     """- A confirmation stays until withdraw_survivor removes it; the robot's own reflexes never drop it."""),
]


def system_prompt() -> str:
    """The system prompt for this step: with carrying people out switched off (actuators.json rescue.enabled), the goal,
    limits and facts say nothing about lifting, so the model is never told of a tool it does not have."""
    if _tools.RESCUE_ENABLED:
        return SYSTEM
    text = SYSTEM
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
            cls._recent.clear(); ToolRegistry._looks.clear(); ToolRegistry._lifted.clear(); ToolRegistry._confirmed.clear()
            ToolRegistry._pending.clear(); TeamState.reset()

    def step(self, req: dict) -> dict:
        start = time.perf_counter()
        self._new_run(req)
        team = TeamState.of(req["robot"])
        # A radio message not seen before is always decided on, even when the event looks like a repeat: a partner's
        # proposal or report arriving within REPEAT_WINDOW_S of the last step was dropped (Codex design review, 22 Oct).
        new_messages = team.ingest(req.get("inbox") or [], float(req["t"]))
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
        ctx.survivors = [list(p) for p in req.get("survivors") or []]
        ctx.maps_linked = any(p.get("maps_linked") for p in req.get("partners") or [])
        tools = ToolRegistry(Path(req["robot_dir"]), ctx, Path(req.get("debug_dir") or Path(self.trace_path).parent / "agent_debug"))
        context = {k: req[k] for k in ("robot", "t", "pose", "torch", "mode", "state", "goal", "heat_budget", "air_c", "event",
                                       "inbox", "orders", "survivors", "frontiers", "lidar", "partners", "blocked_unsearched", "fire_blocks_frontier", "fire_blocks_exit", "mission_time_left_s",
                                       "water_l", "carrying_person") if k in req}
        if _tools.RESCUE_ENABLED:
            context["survivors_to_rescue"] = survivors_to_rescue(req["robot"], pose, req.get("survivors") or [])
        context["team"] = team.context(ctx.t)
        spray = _tools.spray_option(req)
        if spray is not None:
            context["spray_option"] = spray
        pending = tools.pending()
        if pending is not None:
            context["pending_candidate"] = pending
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
            key = request_key(messages, _tools.tool_specs())
            try:
                reply: Reply = self.backend.chat(messages, _tools.tool_specs())
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
                             "refused": result.get("refused", False), "reason": result.get("reason"), "error": result.get("error")})
                messages.append({"role": "tool", "name": call.name, "content": json.dumps(_compact(result), default=str)})
        team.note_intent(ctx.t, tools.finished, tools.orders)
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
