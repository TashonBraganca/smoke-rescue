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

MAX_ROUNDS = 6

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
- Be brief. Call the tools you need, then finish with one sentence saying what you decided and why."""


class RobotAgent:
    def __init__(self, backend, trace_path: Path, max_rounds: int = MAX_ROUNDS):
        self.backend = backend
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_rounds = max_rounds

    def _trace(self, rec: dict) -> None:
        with self.trace_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def step(self, req: dict) -> dict:
        start = time.perf_counter()
        pose = req["pose"]
        ctx = AgentContext(robot=req["robot"], t=float(req["t"]), x=float(pose["x"]), y=float(pose["y"]),
                           heading_deg=float(pose["heading_deg"]),
                           search_area=tuple(req["search_area"]) if req.get("search_area") else None,
                           frame_times=[float(s) for s in req.get("frame_times", [])])
        tools = ToolRegistry(Path(req["robot_dir"]), ctx, Path(req.get("debug_dir") or Path(self.trace_path).parent / "agent_debug"))
        context = {k: req[k] for k in ("robot", "t", "pose", "mode", "state", "goal", "heat_budget", "air_c", "event",
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
