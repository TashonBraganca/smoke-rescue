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

from brain.agent.llm import Reply, request_key
from brain.agent.tools import ALL_TOOLS, AgentContext, ToolRegistry

MAX_ROUNDS = 4
REPEAT_WINDOW_S = 12.0     # the reflex brain re-raises the same candidate every few seconds; one decision covers it

SYSTEM = """You are the decision layer of a quadruped search-and-rescue robot inside a burning building.
A reflex controller already drives, maps, avoids obstacles and backs away from heat. You decide what matters:
which warm shape is a person, where fire is spreading, how to light the scene, what to tell your partner, and
when to use the arm.

Rules:
- Look before you act. Use the OpenCV perception tools; never claim a survivor, reject a candidate, avoid an area,
  change the torch, or use the arm without citing the evidence_id of a perception result from this step.
- A survivor needs inspect_candidate with verdict person. A not_person verdict means reject_candidate.
- If inspect_candidate says uncertain, do not confirm; go_to a closer viewpoint or inspect from another bearing.
- Tell your partner what changes their plan: confirmed survivors, spreading fire, blocked routes, areas you cover.
- Ask for human approval before entering a room where fire is spreading toward you.
- Positions are building coordinates in metres: x to the right, y forward, as given in the context.
- Be brief. Call the tools you need, then finish with one sentence saying what you decided and why.

Playbook by event kind:
- survivor_candidate: scan_thermal, then inspect_candidate on the warm region; confirm_survivor or reject_candidate
  with its evidence_id; if uncertain, approach_candidate citing that inspection (never compute coordinates
  yourself). If the colour image is dark, set the torch first so the second look can use both sensors. Tell your
  partner.
- avoid_fire or retreat: track_fire_spread, then avoid_area around the hot area citing that evidence. Never go_to
  toward heat. If fire spreads toward unexplored rooms, tell your partner and the commander.
- sensor_switch (colour camera poor): assess_visibility, then set_torch citing it. Dark image (low brightness,
  low contrast, little haze) or torch off: wide. Thick smoke with glare: low. Long corridor: narrow. Clear: wide.
  After changing the torch, call assess_visibility again; if the colour image is now usable, scan_thermal to look
  for people with both sensors.
- stuck: measure_opening straight ahead (bearing 0); if not passable, open_door or clear_debris citing it.
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
        with self.trace_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

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
                           search_area=tuple(req["search_area"]) if req.get("search_area") else None,
                           frame_times=[float(s) for s in req.get("frame_times", [])],
                           event_kind=(req.get("event") or {}).get("kind", ""), torch=req.get("torch", "wide"))
        tools = ToolRegistry(Path(req["robot_dir"]), ctx, Path(req.get("debug_dir") or Path(self.trace_path).parent / "agent_debug"))
        context = {k: req[k] for k in ("robot", "t", "pose", "torch", "mode", "state", "goal", "heat_budget", "air_c", "event",
                                       "inbox", "orders", "survivors", "search_area") if k in req}
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": "Decision needed. Context:\n" + json.dumps(context, default=str)}]
        step_id = f"{req['robot']}@{float(req['t']):.3f}"
        self._trace({"kind": "step_start", "step": step_id, "t": req["t"], "robot": req["robot"], "event": req.get("event")})
        rounds = 0
        while rounds < self.max_rounds and tools.finished is None:
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
