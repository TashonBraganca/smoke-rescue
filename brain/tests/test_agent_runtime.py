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
    monkeypatch.setattr(T.vision, "track_fire_spread", lambda frames: {"id": "fire1", "tool": "track_fire_spread"})
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


def test_approach_candidate_stops_short_of_the_inspected_spot(tmp_path, monkeypatch):
    fake_vision(monkeypatch, "uncertain")
    out, _ = run(tmp_path, [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 5})],
                            [ToolCall("approach_candidate", {"evidence_id": "ev1"})]])
    (o,) = out["orders"]
    assert o["type"] == "go_to" and o["x"] == pytest.approx(5.0) and o["y"] == pytest.approx(8.5)   # 5 m ahead, 1.5 m short


def test_no_go_to_during_a_heat_event(tmp_path, monkeypatch):
    fake_vision(monkeypatch)
    req = request(tmp_path) | {"event": {"kind": "avoid_fire"}}
    agent = RobotAgent(Scripted([[ToolCall("go_to", {"x": 6, "y": 6, "reason": "toward"})]]), tmp_path / "t.jsonl")
    out = agent.step(req)
    assert out["orders"] == [] and out["refusals"][0]["tool"] == "go_to"


def test_third_close_look_at_one_spot_is_refused(tmp_path, monkeypatch):
    fake_vision(monkeypatch, "uncertain")
    T.ToolRegistry._looks.clear()
    for k in range(3):
        req = request(tmp_path) | {"t": 10.0 + k}
        out = RobotAgent(Scripted([[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 5})],
                                   [ToolCall("approach_candidate", {"evidence_id": "ev1"})]]), tmp_path / "t.jsonl").step(req)
    assert out["orders"] == [] and "already looked" in out["refusals"][0]["reason"]


def test_explore_frontier_uses_the_offered_coordinates_and_refuses_invented_ids(tmp_path, monkeypatch):
    fake_vision(monkeypatch)
    frontiers = [{"id": 0, "x": 7.5, "y": 3.2, "path_m": 4.0, "unsearched_m2": 6.0, "in_my_sector": True},
                 {"id": 1, "x": 12.0, "y": 9.1, "path_m": 9.5, "unsearched_m2": 14.0, "in_my_sector": False}]
    req = request(tmp_path) | {"event": {"kind": "area_searched"}, "frontiers": frontiers}
    turns = [[ToolCall("explore_frontier", {"frontier_id": 7, "reason": "invented"}),
              ToolCall("explore_frontier", {"frontier_id": 1, "reason": "most unsearched floor", "x": 99, "y": 99})]]
    out = RobotAgent(Scripted(turns), tmp_path / "t.jsonl").step(req)
    assert [o["type"] for o in out["orders"]] == ["explore_frontier"]
    assert (out["orders"][0]["x"], out["orders"][0]["y"]) == (12.0, 9.1)      # from the brain's list, not the model
    assert out["refusals"][0]["tool"] == "explore_frontier" and "not one of the frontiers" in out["refusals"][0]["reason"]


def test_no_exploring_during_a_heat_event(tmp_path, monkeypatch):
    fake_vision(monkeypatch)
    req = request(tmp_path) | {"event": {"kind": "retreat"}, "frontiers": [{"id": 0, "x": 7.5, "y": 3.2}]}
    out = RobotAgent(Scripted([[ToolCall("explore_frontier", {"frontier_id": 0, "reason": "go"})]]), tmp_path / "t.jsonl").step(req)
    assert out["orders"] == [] and out["refusals"][0]["tool"] == "explore_frontier"
