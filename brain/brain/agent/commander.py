"""The incident commander agent: stands at the command post outside, hears robot reports over the same lossy radio,
splits the building into sectors and decides human approvals.

Approvals: in "human" mode a pending request waits for a person (the web page posts to /approvals/<id>); in
"simulated" mode a fixed, logged operator rule answers, and every answer says it came from the rule, not a person.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from brain.agent.llm import request_key

SYSTEM = """You are the incident commander for two search robots in a burning building. You stand outside at the
command post and hear their radio reports, which may arrive late or not at all. Assign each robot a rectangular
sector of the building so they do not search the same rooms twice; move a robot when its partner already covered an
area, when fire blocks a sector, or when a survivor needs a second look. Keep sectors inside the building outline.
Be brief: call assign_sector or broadcast as needed, then finish with one sentence."""

NUM = {"type": "number"}
TOOLS = [
    {"type": "function", "function": {"name": "assign_sector", "description": "Give a robot a search sector (building metres).",
     "parameters": {"type": "object", "properties": {"robot": {"type": "string"}, "x0": NUM, "y0": NUM, "x1": NUM, "y1": NUM,
                    "reason": {"type": "string"}}, "required": ["robot", "x0", "y0", "x1", "y1", "reason"]}}},
    {"type": "function", "function": {"name": "broadcast", "description": "Radio a short message to all robots.",
     "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "finish", "description": "End this step with a one-sentence summary.",
     "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
]


def simulated_operator(request: dict) -> dict:
    """Fixed, logged stand-in for a person during unattended recordings: refuse entry toward spreading fire."""
    ev = request.get("evidence") or {}
    toward = bool(ev.get("spreading_toward_camera")) or (ev.get("speed_m_s") or 0) > 0.3
    hot = any((h.get("range_m") or 99) < 2.0 for h in ev.get("hotspots", []))
    approve = not (toward or hot)
    return {"approved": approve, "by": "simulated_operator_rule",
            "reason": "approved: no fire front moving toward the robot and no hotspot within 2 m" if approve
            else "refused: fire is spreading toward the robot or a hotspot is within 2 m"}


class Commander:
    def __init__(self, backend, trace_path: Path, search_area: tuple[float, float, float, float]):
        self.backend = backend
        self.trace_path = Path(trace_path)
        self.area = search_area

    def _trace(self, rec: dict) -> None:
        with self.trace_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def step(self, req: dict) -> dict:
        start = time.perf_counter()
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": "Situation:\n" + json.dumps({"t": req["t"], "building": self.area,
                     "reports": req.get("reports", []), "robots": req.get("robots", {}),
                     "survivors": req.get("survivors", []), "sectors": req.get("sectors", {})}, default=str)}]
        step_id = f"commander@{float(req['t']):.3f}"
        orders, refusals, summary = [], [], None
        for rnd in range(1, 5):
            key = request_key(messages, TOOLS)
            try:
                reply = self.backend.chat(messages, TOOLS)
            except Exception as e:
                self._trace({"kind": "model_error", "step": step_id, "error": repr(e)})
                break
            self._trace({"kind": "model_reply", "step": step_id, "round": rnd, "request_key": key, "model": reply.model,
                         "latency_s": round(reply.latency_s, 3), "text": reply.text,
                         "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in reply.tool_calls]})
            if not reply.tool_calls:
                summary = reply.text.strip()
                break
            messages.append({"role": "assistant", "content": reply.text,
                             "tool_calls": [{"name": c.name, "arguments": c.arguments} for c in reply.tool_calls]})
            for c in reply.tool_calls:
                a = c.arguments or {}
                if c.name == "finish":
                    summary = a.get("summary", "")
                    result = {"ok": True}
                elif c.name == "assign_sector":
                    x0, x1 = sorted((float(a.get("x0", 0)), float(a.get("x1", 0))))
                    y0, y1 = sorted((float(a.get("y0", 0)), float(a.get("y1", 0))))
                    X0, Y0, X1, Y1 = self.area
                    if x1 - x0 < 2 or y1 - y0 < 2 or x0 < X0 - .5 or y0 < Y0 - .5 or x1 > X1 + .5 or y1 > Y1 + .5:
                        result = {"refused": True, "reason": "sector must be at least 2 x 2 m and inside the building"}
                        refusals.append({"tool": c.name, "args": a, **result})
                    else:
                        orders.append({"type": "assign_sector", "robot": a.get("robot"), "rect": [x0, y0, x1, y1],
                                       "reason": a.get("reason", ""), "t": req["t"]})
                        result = {"ok": True}
                elif c.name == "broadcast":
                    orders.append({"type": "broadcast", "text": str(a.get("text", ""))[:280], "t": req["t"]})
                    result = {"ok": True}
                else:
                    result = {"error": f"unknown tool {c.name}"}
                self._trace({"kind": "tool", "step": step_id, "round": rnd, "name": c.name, "arguments": a, **result})
                messages.append({"role": "tool", "name": c.name, "content": json.dumps(result)})
            if summary is not None:
                break
        out = {"step": step_id, "orders": orders, "refusals": refusals, "summary": summary,
               "latency_s": round(time.perf_counter() - start, 3)}
        self._trace({"kind": "step_end", **out})
        return out
