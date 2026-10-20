"""Agent runtime guardrails with a scripted model: actions without matching OpenCV evidence are refused in code."""

import json
from pathlib import Path

import pytest

from brain.agent import tools as T
from brain.agent.llm import Reply, ReplayBackend, ToolCall, request_key
from brain.agent.robot_agent import RobotAgent


class Scripted:
    model = "scripted"

    def __init__(self, turns):
        self.turns = list(turns)

    def chat(self, messages, tools):
        return Reply("", self.turns.pop(0) if self.turns else [ToolCall("finish", {"summary": "done"})], 0.0, self.model)


def fake_vision(monkeypatch, verdict="person"):
    monkeypatch.setattr(T.vision, "load_frame", lambda d, t: object())
    monkeypatch.setattr(T.vision, "inspect_candidate", lambda f, b, r, output_dir: {"id": "ev1", "tool": "inspect_candidate", "verdict": verdict})
    monkeypatch.setattr(T.vision, "assess_visibility", lambda f: {"id": "vis1", "tool": "assess_visibility", "usable_rgb": False})


def request(tmp_path):
    return {"robot": "robot_A", "robot_dir": str(tmp_path), "t": 10.0, "pose": {"x": 5.0, "y": 5.0, "heading_deg": 0.0},
            "event": {"kind": "survivor_candidate"}, "search_area": [0, 0, 40, 28]}


def run(tmp_path, turns):
    agent = RobotAgent(Scripted(turns), tmp_path / "trace.jsonl")
    return agent.step(request(tmp_path)), [json.loads(l) for l in (tmp_path / "trace.jsonl").read_text().splitlines()]


def test_survivor_claim_without_evidence_is_refused(tmp_path, monkeypatch):
    fake_vision(monkeypatch)
    out, _ = run(tmp_path, [[ToolCall("confirm_survivor", {"x": 5, "y": 8, "evidence_id": "made_up"})]])
    assert out["orders"] == [] and out["refusals"][0]["tool"] == "confirm_survivor"


def test_person_verdict_allows_confirmation_at_that_position(tmp_path, monkeypatch):
    fake_vision(monkeypatch, "person")
    out, trace = run(tmp_path, [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 3})],
                                [ToolCall("confirm_survivor", {"x": 5, "y": 8, "evidence_id": "ev1"})]])
    assert [o["type"] for o in out["orders"]] == ["confirm_survivor"]
    assert out["orders"][0]["evidence_tool"] == "inspect_candidate"
    assert any(r["kind"] == "tool" and r["name"] == "inspect_candidate" for r in trace)


@pytest.mark.parametrize("verdict", ["not_person", "uncertain"])
def test_non_person_verdict_cannot_confirm(tmp_path, monkeypatch, verdict):
    # Control for the test above: same calls, only the OpenCV verdict differs, and the order is refused.
    fake_vision(monkeypatch, verdict)
    out, _ = run(tmp_path, [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 3})],
                            [ToolCall("confirm_survivor", {"x": 5, "y": 8, "evidence_id": "ev1"})]])
    assert out["orders"] == [] and len(out["refusals"]) == 1


def test_confirmation_far_from_the_inspected_spot_is_refused(tmp_path, monkeypatch):
    fake_vision(monkeypatch, "person")
    out, _ = run(tmp_path, [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 3})],
                            [ToolCall("confirm_survivor", {"x": 20, "y": 20, "evidence_id": "ev1"})]])
    assert out["orders"] == []


def test_torch_needs_visibility_evidence_and_go_to_stays_inside(tmp_path, monkeypatch):
    fake_vision(monkeypatch)
    out, _ = run(tmp_path, [[ToolCall("set_torch", {"mode": "low", "evidence_id": "ev1"}),
                             ToolCall("go_to", {"x": 50, "y": 5, "reason": "outside"})],
                            [ToolCall("assess_visibility", {})],
                            [ToolCall("set_torch", {"mode": "low", "evidence_id": "vis1"})]])
    assert [o["type"] for o in out["orders"]] == ["set_torch"]
    assert {r["tool"] for r in out["refusals"]} == {"set_torch", "go_to"}


def test_replay_backend_reproduces_a_recorded_step(tmp_path, monkeypatch):
    fake_vision(monkeypatch, "person")
    turns = [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 3})],
             [ToolCall("confirm_survivor", {"x": 5, "y": 8, "evidence_id": "ev1"})]]
    first, _ = run(tmp_path, turns)
    replay = RobotAgent(ReplayBackend(tmp_path / "trace.jsonl"), tmp_path / "replay.jsonl").step(request(tmp_path))
    assert replay["orders"] == first["orders"]


def test_request_key_changes_with_content():
    m = [{"role": "user", "content": "a"}]
    assert request_key(m, []) != request_key([{"role": "user", "content": "b"}], [])
