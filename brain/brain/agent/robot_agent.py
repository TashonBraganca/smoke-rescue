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
from brain.agent.tools import ALL_TOOLS, AgentContext, ToolRegistry

MAX_ROUNDS = 4
STEP_DEADLINE_S = 75.0     # past this the step ends with what it has; the reflex layer keeps the robot safe meanwhile
REPEAT_WINDOW_S = 12.0     # the reflex brain re-raises the same candidate every few seconds; one decision covers it

SYSTEM = """You are the decision layer of a quadruped search-and-rescue robot inside a burning building.
You decide where the robot searches and what it does there. A reflex controller executes your choice: it plans
the path on the robot's own LiDAR and camera map, keeps clear of walls, brakes for obstacles and backs away from
heat. You decide which area to search next, which warm shape is a person, where fire is spreading, how to light
the scene, what to tell your partner, and when to use the arm.

Rules:
- Look before you act. Use the OpenCV perception tools; never claim a survivor, reject a candidate, avoid an area,
  change the torch, or use the arm without citing the evidence_id of a perception result from this step.
- A survivor needs inspect_candidate with verdict person. A not_person verdict means reject_candidate.
- If inspect_candidate says uncertain, do not confirm; go_to a closer viewpoint or inspect from another bearing.
- Tell your partner what changes their plan: confirmed survivors, spreading fire, blocked routes, areas you cover.
- Ask for human approval before entering a room where fire is spreading toward you.
- Positions are in YOUR OWN map frame, in metres from where you entered: y is the direction you faced on entry, x
  to its right. Nobody has given you a map or a building outline; everything you know you mapped yourself or heard
  on the radio. Your partner's coordinates are in its own frame, so do not reuse numbers from its messages as
  positions; describe places in words (the room past the hallway, the door on your left) when you radio. Once
  partners shows maps_linked, your robot has matched LiDAR fingerprints with that partner: its position and the area
  it searched are already converted into your frame and your frontier list leaves its searched area out.
- Be brief. Call the tools you need, then finish with one sentence saying what you decided and why.

Choosing where to search (you, not the reflex layer, pick every search target):
- The context lists frontiers: unsearched floor the robot can reach, each with id, x, y, path_m, unsearched_m2 and
  how far it is from your partner's claim. After area_searched, no_progress, goal_blacklisted, collision_hold, and at every periodic check-in
  when nothing more urgent is happening, call explore_frontier with one id and a one-line reason.
- Prefer more unsearched area per metre of path; skip areas your partner said they cover; do not pick a frontier
  next to a spot that just failed (goal_blacklisted, collision_hold).
- The lidar entry gives the nearest obstacle in eight directions; under 0.4 m means the robot is in a tight spot,
  so prefer a frontier on the open side.
- If the list is empty, say so to your partner and the commander: your reachable area is searched.

Playbook by event kind:
- survivor_view_ready: the robot has stopped facing the warm shape at close range. This is your moment to decide:
  it is straight ahead, so inspect_candidate at bearing 0 and the event's range_m, then confirm_survivor or
  reject_candidate citing that result. The robot never confirms a person by itself.
- survivor_candidate: scan_thermal, then inspect_candidate on the warm region; confirm_survivor or reject_candidate
  with its evidence_id; if uncertain, approach_candidate citing that inspection (never compute coordinates
  yourself). If the colour image is dark, set the torch first so the second look can use both sensors. Tell your
  partner.
- avoid_fire or retreat: track_fire_spread, then avoid_area around the hot area citing that evidence. Never go_to
  toward heat. If fire blocks the way to unsearched floor or to a person and water_l is above 0, scan_thermal and
  spray_water at the hot spot's bearing citing that scan (a few seconds at a time), then look again.
- after confirm_survivor: if the person is within reach (inspect from close, approach_candidate first if needed),
  extract_survivor citing that inspection; the robot tows them out to the nearest way out and may come back in. If fire spreads toward unexplored rooms, tell your partner and the commander.
- sensor_switch (colour camera poor): assess_visibility, then set_torch citing it. Dark image (low brightness,
  low contrast, little haze) or torch off: wide. Thick smoke with glare: low. Long corridor: narrow. Clear: wide.
  After changing the torch, call assess_visibility again; if the colour image is now usable, scan_thermal to look
  for people with both sensors.
- stuck: measure_opening straight ahead (bearing 0); if not passable, open_door or clear_debris citing it.
- no_frontier (your map has no reachable unsearched floor; the robot returns after a short scan unless you act):
  if blocked_unsearched is given, your map shows unsearched floor you cannot reach at that bearing and distance:
  go_to its approach point, then measure_opening at bearing 0; if not passable because of light debris or a closed
  door, clear_debris or open_door citing it. Otherwise read the LiDAR summary and your camera. If a closed door or
  light debris blocks the way on, measure_opening toward it and open_door or clear_debris citing that. If fire blocks it, tell your partner and the commander and
  let the robot return. Never invent a target. Apply this whenever the context's frontiers list is empty, whatever
  the event: another event (a warm shape, say) may have arrived at the same moment.
- Every step: if the context says the torch is off or the event is periodic, call assess_visibility first; if the
  colour image is not usable and the torch is off, set_torch wide citing it.
- periodic (radio check-in): scan_thermal, act on what it finds, then send_team_message with one line for your
  partner and the commander: where you are searching, what you found (survivors, fire, blocked doors), and what
  you will do next, so the two of you do not search the same rooms.
- When your inbox has a partner message, use it: do not re-search what they covered, and answer if they ask.
Each perception tool returns the same result if called again in this step, so call it once."""


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
        near = None not in here and None not in lpos and ((here[0] - lpos[0]) ** 2 + (here[1] - lpos[1]) ** 2) ** 0.5 < 1.5
        return float(req["t"]) - lt < REPEAT_WINDOW_S and (near or ev.get("kind") != "survivor_candidate")

    def step(self, req: dict) -> dict:
        start = time.perf_counter()
        if self._repeat(req):
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
        tools = ToolRegistry(Path(req["robot_dir"]), ctx, Path(req.get("debug_dir") or Path(self.trace_path).parent / "agent_debug"))
        context = {k: req[k] for k in ("robot", "t", "pose", "torch", "mode", "state", "goal", "heat_budget", "air_c", "event",
                                       "inbox", "orders", "survivors", "frontiers", "lidar", "partners", "blocked_unsearched", "water_l", "carrying_person") if k in req}
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": "Decision needed. Context:\n" + json.dumps(context, default=str)}]
        step_id = f"{req['robot']}@{float(req['t']):.3f}"
        self._trace({"kind": "step_start", "step": step_id, "t": req["t"], "robot": req["robot"], "event": req.get("event")})
        rounds = 0
        while rounds < self.max_rounds and tools.finished is None:
            if time.perf_counter() - start > STEP_DEADLINE_S:
                self._trace({"kind": "step_deadline", "step": step_id, "after_rounds": rounds})
                break
            rounds += 1
            key = request_key(messages, ALL_TOOLS)
            try:
                reply: Reply = self.backend.chat(messages, ALL_TOOLS)
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
