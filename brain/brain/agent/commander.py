"""The incident commander agent: stands at the command post outside, hears robot reports over the same lossy radio,
relays what changes the other robot's plan and decides human approvals. It has no map and assigns no areas.

Approvals: in "human" mode a pending request waits for a person (the web page posts to /approvals/<id>); in
"simulated" mode a fixed, logged operator rule answers, and every answer says it came from the rule, not a person.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from brain.agent.llm import append_jsonl, request_key

SYSTEM = """You are the incident commander for two search robots inside a burning building whose layout nobody knows.
You stand outside at the command post and hear their radio reports, which may arrive late or not at all. You have
no map, no floor plan and no building outline: everything you know comes from those reports. Your job is to relay
what one robot learned that changes the other's plan (a survivor found, fire spreading, a route blocked, an area
already searched), and to set priorities in words. You never assign coordinates or areas; each robot decides where
to search from its own map. Be brief: broadcast only when it helps, then finish with one sentence."""

TOOLS = [
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
    def __init__(self, backend, trace_path: Path):
        self.backend = backend
        self.trace_path = Path(trace_path)

    def _trace(self, rec: dict) -> None:
        append_jsonl(self.trace_path, rec)

    def step(self, req: dict) -> dict:
        start = time.perf_counter()
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": "Situation:\n" + json.dumps({"t": req["t"],
                     "reports": req.get("reports", []), "robots": req.get("robots", {})}, default=str)}]
        step_id = f"commander@{float(req['t']):.3f}"
        # What reached the post, so an audit can tell "nothing arrived" from "arrived and ignored".
        self._trace({"kind": "step_start", "step": step_id, "t": req["t"], "robot": "commander",
                     "reports": req.get("reports", [])})
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
