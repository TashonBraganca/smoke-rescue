"""The agent loops measured in the one-door runs, and the code that breaks them: radio texts that woke a step (80 of 116 in
seed 932), turns toward a point already in view (32 of 164 face orders), third turns at one spot, turns and walks to
thermal-only wall patches (1 person in 151), movement orders replaced before arrival (121 of 420), and a model that never
heard what the brain did with its last order."""

import json
import math

from brain.agent import tools as T
from brain.agent.commander import Commander
from brain.agent.llm import Reply, ToolCall
from brain.agent.robot_agent import RobotAgent


class Scripted:
    model = "scripted"

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls = 0

    def chat(self, messages, tools):
        self.calls += 1
        return Reply("", self.turns.pop(0) if self.turns else [ToolCall("finish", {"summary": "done"})], 0.0, self.model)


def fake_vision(monkeypatch, verdict="uncertain", cues=None):
    monkeypatch.setattr(T.vision, "load_frame", lambda d, t: object())
    monkeypatch.setattr(T.vision, "inspect_candidate", lambda f, b, r, output_dir: {
        "id": "ev1", "tool": "inspect_candidate", "verdict": verdict, "candidate": {"bearing_deg": b, "range_m": r},
        "cues": cues or {}})
    monkeypatch.setattr(T.vision, "scan_thermal", lambda f: {"id": "scan1", "tool": "scan_thermal", "hotspots": [], "people": []})


def request(tmp_path, t=10.0, event=None, inbox=None, heading=0.0):
    return {"robot": "robot_A", "robot_dir": str(tmp_path), "t": t, "pose": {"x": 0.0, "y": 0.0, "heading_deg": heading},
            "event": event or {"kind": "periodic"}, "inbox": inbox or []}


def trace(tmp_path):
    return [json.loads(l) for l in (tmp_path / "trace.jsonl").read_text().splitlines()]


def _fresh():
    T.ToolRegistry._faces.clear(); T.ToolRegistry._wall_spots.clear(); T.ToolRegistry._looks.clear(); T.ToolRegistry._pending.clear()
    RobotAgent._last_orders.clear(); RobotAgent._recent.clear()


# ----- radio texts -------------------------------------------------------------------------------------------------

def test_a_radio_finding_alone_does_not_wake_the_model(tmp_path, monkeypatch):
    _fresh(); fake_vision(monkeypatch)
    model = Scripted([[ToolCall("explore_frontier", {"frontier_id": 0, "reason": "x"})]])
    inbox = [{"message_id": "robot_B-3", "from": "robot_B", "t": 9.0, "kind": "finding", "text": "west wall warm"}]
    out = RobotAgent(model, tmp_path / "trace.jsonl").step(request(tmp_path, event={"kind": "team_message"}, inbox=inbox))
    assert model.calls == 0 and out["orders"] == [] and out.get("radio_text_only")
    assert trace(tmp_path)[-1]["kind"] == "step_skipped"


def test_a_proposal_or_an_answer_wakes_the_model(tmp_path, monkeypatch):
    _fresh(); fake_vision(monkeypatch)
    model = Scripted([])
    inbox = [{"message_id": "robot_B-p4", "from": "robot_B", "t": 9.0, "kind": "proposal", "proposal_id": "robot_B-p4",
              "my_part": "left side", "your_part": "right side", "text": "split", "expires_t": 29.0}]
    RobotAgent(model, tmp_path / "trace.jsonl").step(request(tmp_path, event={"kind": "team_message"}, inbox=inbox))
    assert model.calls == 1


def test_a_radio_text_still_wakes_the_step_while_a_close_look_is_pending(tmp_path, monkeypatch):
    _fresh(); fake_vision(monkeypatch)
    T.ToolRegistry._pending["robot_A"] = {"x": 3.0, "y": 3.0, "t": 9.0, "evidence_id": "ev0"}
    model = Scripted([])
    inbox = [{"message_id": "robot_B-5", "from": "robot_B", "t": 9.0, "kind": "finding", "text": "nothing new"}]
    RobotAgent(model, tmp_path / "trace.jsonl").step(request(tmp_path, event={"kind": "team_message"}, inbox=inbox))
    assert model.calls == 1


def test_the_commander_makes_no_call_without_reports(tmp_path):
    model = Scripted([])
    out = Commander(model, tmp_path / "trace.jsonl").step({"t": 30.0, "reports": [], "robots": ["robot_A", "robot_B"]})
    assert model.calls == 0 and out["orders"] == [] and out.get("no_reports")
    Commander(model, tmp_path / "trace.jsonl").step({"t": 60.0, "reports": [{"from": "robot_A", "text": "person at the door"}]})
    assert model.calls == 1


# ----- turning in place --------------------------------------------------------------------------------------------

def test_a_turn_toward_a_point_already_in_the_thermal_view_is_refused(tmp_path, monkeypatch):
    _fresh(); fake_vision(monkeypatch)
    # 4 m ahead, 5 deg right: a lying body (0.85 m half length, 12 deg at 4 m) fits inside the 25 deg half view.
    model = Scripted([[ToolCall("turn_to_face", {"x": 4 * math.sin(math.radians(5)), "y": 4 * math.cos(math.radians(5)), "reason": "centre it"})]])
    out = RobotAgent(model, tmp_path / "trace.jsonl").step(request(tmp_path))
    assert out["orders"] == [] and "already in view" in out["refusals"][0]["reason"]


def test_a_turn_toward_a_point_outside_the_view_is_an_order(tmp_path, monkeypatch):
    _fresh(); fake_vision(monkeypatch)
    model = Scripted([[ToolCall("turn_to_face", {"x": 4 * math.sin(math.radians(40)), "y": 4 * math.cos(math.radians(40)), "reason": "look"})]])
    out = RobotAgent(model, tmp_path / "trace.jsonl").step(request(tmp_path))
    assert [o["type"] for o in out["orders"]] == ["go_to"] and out["orders"][0]["face_x"] > 0


def test_a_third_turn_toward_one_spot_within_the_memory_is_refused(tmp_path, monkeypatch):
    _fresh(); fake_vision(monkeypatch)
    x, y = 4 * math.sin(math.radians(40)), 4 * math.cos(math.radians(40))
    outs = []
    for t in (10.0, 30.0, 50.0):
        model = Scripted([[ToolCall("turn_to_face", {"x": x, "y": y, "reason": "again"})]])
        outs.append(RobotAgent(model, tmp_path / "trace.jsonl").step(request(tmp_path, t=t)))
    assert [len(o["orders"]) for o in outs] == [1, 1, 0] and "already turned to face" in outs[2]["refusals"][0]["reason"]


# ----- what the brain did with the last order ----------------------------------------------------------------------

def _log(tmp_path, events):
    (tmp_path / "decisions.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))


def test_order_outcomes_read_the_brain_log():
    orders = [{"type": "go_to", "x": 2.0, "y": 3.0}, {"type": "explore_frontier", "x": -5.0, "y": 1.0},
              {"type": "go_to", "x": 8.0, "y": 8.0}, {"type": "go_to", "x": 0.0, "y": 9.0}]
    import tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp())
    _log(d, [{"t": 11.0, "kind": "agent_order", "reason": "Refused go_to: no ceiling has been seen over that point.", "numbers": {"x_m": 2.0, "y_m": 3.0}},
             {"t": 12.0, "kind": "replan", "reason": "Agent goal has no known free path; back to exploring.", "numbers": {"x_m": -5.0, "y_m": 1.0}},
             {"t": 13.0, "kind": "agent_order_expired", "reason": "hold lapsed", "numbers": {"x_m": 8.0, "y_m": 8.0, "remaining_m": 2.5}},
             {"t": 9.0, "kind": "agent_goal_reached", "reason": "earlier arrival, before this order", "numbers": {"x_m": 0.0, "y_m": 9.0}}])
    out = T.order_outcomes(d, 10.0, orders)
    assert out[0]["outcome"].startswith("refused by the brain: no ceiling")
    assert out[1]["outcome"] == "dropped: no known free path to it"
    assert out[2]["outcome"] == "expired before arrival, 2.5 m short"
    assert out[3]["outcome"].startswith("under way")


def test_the_next_step_sees_the_outcome_and_keeps_a_live_order(tmp_path, monkeypatch):
    _fresh(); fake_vision(monkeypatch)
    seen = []

    class Capture(Scripted):
        def chat(self, messages, tools):
            seen.append(json.loads(messages[1]["content"].split("Context:\n", 1)[1]))
            return super().chat(messages, tools)

    agent = RobotAgent(Capture([[ToolCall("explore_frontier", {"frontier_id": 0, "reason": "go"})]]), tmp_path / "trace.jsonl")
    req = {**request(tmp_path), "frontiers": [{"id": 0, "x": -5.0, "y": 1.0, "path_m": 6.0, "proven_indoors": True}]}
    out1 = agent.step(req)
    assert [o["type"] for o in out1["orders"]] == ["explore_frontier"]
    _log(tmp_path, [{"t": 12.0, "kind": "replan", "reason": "Agent goal has no known free path; back to exploring.", "numbers": {"x_m": -5.0, "y_m": 1.0}}])
    agent2 = RobotAgent(Capture([]), tmp_path / "trace.jsonl")
    agent2.step({**req, "t": 30.0})
    assert seen[-1]["last_orders_outcome"][0]["outcome"] == "dropped: no known free path to it"


def test_a_move_still_under_way_is_not_replaced_on_a_periodic_step(tmp_path, monkeypatch):
    _fresh(); fake_vision(monkeypatch)
    req = {**request(tmp_path), "frontiers": [{"id": 0, "x": -5.0, "y": 1.0, "path_m": 6.0, "proven_indoors": True},
                                               {"id": 1, "x": 5.0, "y": 1.0, "path_m": 6.0, "proven_indoors": True}]}
    agent = RobotAgent(Scripted([[ToolCall("explore_frontier", {"frontier_id": 0, "reason": "go"})]]), tmp_path / "trace.jsonl")
    agent.step(req)
    _log(tmp_path, [{"t": 11.0, "kind": "goal", "reason": "Agent order: explore", "numbers": {"x_m": -5.0, "y_m": 1.0}}])
    out = RobotAgent(Scripted([[ToolCall("explore_frontier", {"frontier_id": 1, "reason": "other"})]]), tmp_path / "trace.jsonl").step({**req, "t": 30.0})
    assert out["orders"] == [] and "still being driven" in out["refusals"][0]["reason"]
    # A new candidate is a reason to change course: the same second order on a survivor_candidate event is accepted.
    out = RobotAgent(Scripted([[ToolCall("explore_frontier", {"frontier_id": 1, "reason": "other"})]]), tmp_path / "trace.jsonl").step(
        {**req, "t": 50.0, "event": {"kind": "survivor_candidate", "numbers": {"x_m": 9.0, "y_m": 9.0}}})
    assert [o["type"] for o in out["orders"]] == ["explore_frontier"]


# ----- tools offered per event ----------------------------------------------------------------------------------------

def test_heat_events_offer_only_the_tools_that_can_act_there():
    names = {t["function"]["name"] for t in T.tool_specs("retreat")}
    assert "go_to" not in names and "explore_frontier" not in names and "turn_to_face" not in names
    assert {"scan_thermal", "inspect_candidate", "confirm_survivor", "reject_candidate", "spray_water", "avoid_area",
            "send_team_message", "finish"} <= names
    everything = {t["function"]["name"] for t in T.tool_specs("survivor_candidate")}
    assert "go_to" in everything and "explore_frontier" in everything and len(everything) > len(names)


def test_partner_findings_carry_their_age():
    from brain.agent.team import TeamState
    team = TeamState.of("robot_A")
    team.ingest([{"message_id": "robot_B-9", "from": "robot_B", "t": 40.0, "kind": "finding", "text": "fire in the hall"}], 41.0)
    ctx = team.context(70.0)
    assert ctx["partner_findings"][0]["age_s"] == 30.0 and ctx["partner_last_heard_s"] == 30.0


def test_a_pending_look_does_not_wake_radio_steps_during_a_retreat(tmp_path, monkeypatch):
    _fresh(); fake_vision(monkeypatch)
    T.ToolRegistry._pending["robot_A"] = {"x": 3.0, "y": 3.0, "t": 9.0, "evidence_id": "ev0"}
    model = Scripted([])
    inbox = [{"message_id": "robot_B-6", "from": "robot_B", "t": 9.0, "kind": "finding", "text": "nothing new"}]
    RobotAgent(model, tmp_path / "trace.jsonl").step({**request(tmp_path, event={"kind": "team_message"}, inbox=inbox), "state": "retreat"})
    assert model.calls == 0
