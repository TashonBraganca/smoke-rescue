"""The team conversation across steps: proposals, answers, agreements, findings and intent, all kept and stamped by code.

Codex design review (22 Oct): each step started from nothing, radio texts carried no proposal, acceptance or expiry,
and a partner's message arriving within the repeat window of an earlier event was dropped. These tests use a scripted
model, so they check the protocol, not whether a live model proposes useful plans.
"""

import json

from brain.agent import tools as T
from brain.agent.llm import Reply, ToolCall
from brain.agent.robot_agent import RobotAgent
from brain.agent.team import PROPOSAL_TTL_S, TeamState


class Capture:
    """Scripted model that records the context it was shown."""
    model = "scripted"

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    def chat(self, messages, tools):
        self.seen.append(json.loads(messages[1]["content"].split("Context:\n", 1)[1]))
        return Reply("", self.turns.pop(0) if self.turns else [ToolCall("finish", {"summary": "done"})], 0.0, self.model)


def request(run, robot, t, inbox=(), event="periodic", frontiers=()):
    d = run / robot; d.mkdir(parents=True, exist_ok=True)
    return {"robot": robot, "robot_dir": str(d), "t": t, "pose": {"x": 0.0, "y": 0.0, "heading_deg": 0.0},
            "event": {"kind": event}, "inbox": list(inbox), "frontiers": list(frontiers)}


def step(run, robot, t, turns, inbox=(), event="periodic", frontiers=()):
    model = Capture(turns)
    out = RobotAgent(model, run / "trace.jsonl").step(request(run, robot, t, inbox, event, frontiers))
    return out, model.seen


def radio(order, sender):
    """What the recorder delivers: the order's message fields, from the sender (positions dropped: frames not linked)."""
    return {k: v for k, v in order.items() if k not in ("type", "robot")} | {"from": sender}


def test_a_proposal_reaches_the_partner_and_its_acceptance_comes_back_as_an_agreement(tmp_path):
    out, _ = step(tmp_path, "robot_A", 10.0, [[ToolCall("propose_plan", {
        "my_part": "search past the kitchen door", "your_part": "search the rooms off your hallway", "why": "split the house"})]])
    [proposal] = [o for o in out["orders"] if o["type"] == "send_team_message"]
    assert proposal["kind"] == "proposal" and proposal["proposal_id"] and proposal["expires_t"] == 10.0 + PROPOSAL_TTL_S
    out_b, seen_b = step(tmp_path, "robot_B", 12.0, [[ToolCall("answer_proposal", {
        "proposal_id": proposal["proposal_id"], "accept": True, "reason": "my hallway is clear"})]], inbox=[radio(proposal, "robot_A")])
    awaiting = seen_b[0]["team"]["awaiting_your_answer"]
    assert [p["proposal_id"] for p in awaiting] == [proposal["proposal_id"]]
    assert awaiting[0]["your_part"] == "search the rooms off your hallway"
    [answer] = [o for o in out_b["orders"] if o["type"] == "send_team_message"]
    assert answer["kind"] == "accept" and answer["reply_to"] == proposal["proposal_id"]
    _, seen_a = step(tmp_path, "robot_A", 15.0, [], inbox=[radio(answer, "robot_B")])
    [agreement] = seen_a[0]["team"]["agreements_in_force"]
    assert agreement["your_part"] == "search past the kitchen door" and agreement["partner_part"] == "search the rooms off your hallway"
    assert seen_a[0]["team"]["your_proposals"][0]["status"] == "accepted"


def test_an_unknown_or_expired_proposal_cannot_be_answered(tmp_path):
    out, _ = step(tmp_path, "robot_B", 5.0, [[ToolCall("answer_proposal", {"proposal_id": "robot_A-p99", "accept": True, "reason": "x"})]])
    assert out["refusals"] and "no proposal" in out["refusals"][0]["reason"]
    late = {"message_id": "robot_A-7", "from": "robot_A", "t": 5.0, "kind": "proposal", "proposal_id": "robot_A-p8",
            "expires_t": 5.0 + PROPOSAL_TTL_S, "my_part": "a", "your_part": "b", "text": "why"}
    out, _ = step(tmp_path, "robot_B", 5.0 + PROPOSAL_TTL_S + 1, [[ToolCall("answer_proposal", {"proposal_id": "robot_A-p8", "accept": True, "reason": "x"})]],
                  inbox=[late])
    assert out["refusals"] and "expired" in out["refusals"][0]["reason"]
    assert not [o for o in out["orders"] if o["type"] == "send_team_message"]


def test_a_new_radio_message_is_decided_even_when_the_event_repeats(tmp_path):
    step(tmp_path, "robot_A", 10.0, [], event="no_progress")
    skipped, _ = step(tmp_path, "robot_A", 12.0, [], event="no_progress")
    assert skipped.get("repeat_of_recent_event"), "control: the same event with nothing new is a repeat"
    msg = {"message_id": "robot_B-3", "from": "robot_B", "t": 13.0, "kind": "finding", "text": "fire across the hallway"}
    decided, seen = step(tmp_path, "robot_A", 14.0, [], inbox=[msg], event="no_progress")
    assert not decided.get("repeat_of_recent_event")
    assert seen[0]["team"]["partner_findings"][0]["text"] == "fire across the hallway"


def test_a_new_recording_forgets_the_conversation(tmp_path):
    msg = {"message_id": "robot_B-1", "from": "robot_B", "t": 3.0, "kind": "finding", "text": "kitchen searched"}
    _, seen = step(tmp_path / "run1", "robot_A", 4.0, [], inbox=[msg])
    assert seen[0]["team"]["partner_findings"]
    _, seen = step(tmp_path / "run2", "robot_A", 1.0, [])
    assert seen[0]["team"]["partner_findings"] == [] and seen[0]["team"]["your_last_intent"] is None


def test_the_last_intent_is_the_orders_the_tools_accepted_not_the_words(tmp_path, monkeypatch):
    monkeypatch.setattr(T.vision, "load_frame", lambda d, t: object())
    step(tmp_path, "robot_A", 10.0, [[ToolCall("explore_frontier", {"frontier_id": 999, "reason": "not offered"})],
                                     [ToolCall("finish", {"summary": "heading to the big room"})]])
    intent = TeamState.of("robot_A").intent
    assert intent["orders"] == [], "a refused order is not an intent"
    assert intent["summary"] == "heading to the big room"
    offered = [{"id": 3, "x": 1.0, "y": 4.0, "path_m": 4.2, "unsearched_m2": 6.0, "proven_indoors": True}]
    step(tmp_path, "robot_A", 30.0, [[ToolCall("explore_frontier", {"frontier_id": 3, "reason": "largest unsearched area"})]],
         frontiers=offered)
    assert TeamState.of("robot_A").intent["orders"] == [{"type": "explore_frontier", "x": 1.0, "y": 4.0, "frontier_id": 3}]


def test_the_spray_option_compares_the_blocking_fire_with_the_water(tmp_path):
    # Held-out seed 921: robot A, cut off by a burning wood pile, had 12 L and no measure of whether it could open the
    # way. One stream removes flow x cooling x efficiency; the fire releases burning area x burning_kw_per_m2.
    one = T.ONE_ROBOT_COOLS_KW
    small = 0.5 * one / T.BURNING_KW_PER_M2; mid = 1.5 * one / T.BURNING_KW_PER_M2; big = 3 * one / T.BURNING_KW_PER_M2
    def opt(area, water=12.0, bearing=-90.7):
        return T.spray_option({"fire_blocks_exit": {"bearing_deg": bearing, "distance_m": 1.66, "burning_m2": area}, "water_l": water})
    assert "this robot's stream" in opt(small)["assessment"]
    assert "two robots" in opt(mid)["assessment"]
    assert "more than two" in opt(big)["assessment"]
    assert opt(small)["in_camera_view"] is False and opt(small, bearing=10)["in_camera_view"] is True
    assert "no water" in opt(small, water=0)["assessment"]
    assert "no floor measured burning" in opt(0.0)["assessment"]
    assert T.spray_option({"water_l": 12}) is None, "control: nothing blocks, no option"


def test_a_plan_names_a_place_only_when_the_maps_are_linked(tmp_path):
    target = {"x": 2.0, "y": 6.5}
    req = request(tmp_path, "robot_A", 20.0)
    req["partners"] = [{"robot": "robot_B", "maps_linked": False}]
    model = Capture([[ToolCall("propose_plan", {"my_part": "spray from the left", "your_part": "spray from the right",
                                                "why": "fire too big for one", "target": target})]])
    out = RobotAgent(model, tmp_path / "trace.jsonl").step(req)
    assert out["refusals"] and "not linked" in out["refusals"][0]["reason"]
    req = request(tmp_path, "robot_A", 22.0); req["partners"] = [{"robot": "robot_B", "maps_linked": True}]
    model = Capture([[ToolCall("propose_plan", {"my_part": "spray from the left", "your_part": "spray from the right",
                                                "why": "fire too big for one", "target": target})]])
    out = RobotAgent(model, tmp_path / "trace.jsonl").step(req)
    [proposal] = [o for o in out["orders"] if o["type"] == "send_team_message"]
    assert proposal["target"] == target
    _, seen_b = step(tmp_path, "robot_B", 24.0, [], inbox=[radio(proposal, "robot_A")])
    assert seen_b[0]["team"]["awaiting_your_answer"][0]["target"] == target
