"""The agent server runs both robots' steps at the same time, so the sim waits for one model call, not two."""

import asyncio
import json
import time

import httpx

from brain.agent import server
from brain.agent.llm import Reply, ToolCall


class Slow:
    model = "slow"

    def chat(self, messages, tools):
        time.sleep(.6)
        return Reply("", [ToolCall("finish", {"summary": "done"})], .6, self.model)


def test_two_robot_steps_run_concurrently_and_trace_lines_stay_whole(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_backend", Slow())
    trace = tmp_path / "trace.jsonl"

    def req(robot):
        return {"robot_dir": str(tmp_path), "t": 1.0, "trace_path": str(trace),
                "pose": {"x": 0.0, "y": 0.0, "heading_deg": 0.0}, "event": {"kind": f"periodic_{robot}"}}

    async def both():
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://agents") as client:
            start = time.perf_counter()
            replies = await asyncio.gather(*(client.post(f"/robot/{r}/step", json=req(r)) for r in ("robot_A", "robot_B")))
            return time.perf_counter() - start, replies

    elapsed, replies = asyncio.run(both())
    assert all(r.status_code == 200 for r in replies)
    assert elapsed < 1.0, f"steps ran one after the other ({elapsed:.2f} s for two 0.6 s model calls)"
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert {r["step"] for r in rows if r["kind"] == "step_end"} == {"robot_A@1.000", "robot_B@1.000"}
