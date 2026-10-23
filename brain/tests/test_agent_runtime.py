import math
"""Agent runtime guardrails with a scripted model: actions without matching OpenCV evidence are refused in code."""

import json
from pathlib import Path

import pytest

from brain.agent import tools as T
from brain.agent.llm import Reply, ReplayBackend, ToolCall, request_key
from brain.agent.robot_agent import RobotAgent
from brain.agent.team import TeamState


class Scripted:
    model = "scripted"

    def __init__(self, turns):
        self.turns = list(turns)

    def chat(self, messages, tools):
        return Reply("", self.turns.pop(0) if self.turns else [ToolCall("finish", {"summary": "done"})], 0.0, self.model)


def fake_vision(monkeypatch, verdict="person"):
    monkeypatch.setattr(T.vision, "load_frame", lambda d, t: object())
    monkeypatch.setattr(T.vision, "inspect_candidate", lambda f, b, r, output_dir: {"id": "ev1", "tool": "inspect_candidate", "verdict": verdict,
                                                                                  "candidate": {"bearing_deg": b, "range_m": r}})
    monkeypatch.setattr(T.vision, "track_fire_spread", lambda frames: {"id": "fire1", "tool": "track_fire_spread"})
    monkeypatch.setattr(T.vision, "assess_visibility", lambda f: {"id": "vis1", "tool": "assess_visibility", "usable_rgb": False})


def request(tmp_path):
    return {"robot": "robot_A", "robot_dir": str(tmp_path), "t": 10.0, "pose": {"x": 5.0, "y": 5.0, "heading_deg": 0.0},
            "event": {"kind": "survivor_candidate"}}


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


def test_torch_needs_visibility_evidence(tmp_path, monkeypatch):
    fake_vision(monkeypatch)
    out, _ = run(tmp_path, [[ToolCall("set_torch", {"mode": "low", "evidence_id": "ev1"})],
                            [ToolCall("assess_visibility", {})],
                            [ToolCall("set_torch", {"mode": "low", "evidence_id": "vis1"})]])
    assert [o["type"] for o in out["orders"]] == ["set_torch"]
    assert {r["tool"] for r in out["refusals"]} == {"set_torch"}


def test_a_building_outline_never_reaches_the_model(tmp_path, monkeypatch):
    # No prior knowledge (Tashon, 2026-10-20): even if a request carried an outline, the model must not see it.
    # Whether a go_to target is inside is decided by the brain from the robot's own ceiling evidence.
    fake_vision(monkeypatch)
    seen = []

    class Capture(Scripted):
        def chat(self, messages, tools):
            seen.append(messages[1]["content"])   # the context the robot sends; the system prompt may say "no outline"
            return super().chat(messages, tools)

    req = {**request(tmp_path), "search_area": [0, 0, 20, 14], "outline": [[0, 0], [20, 14]]}
    out = RobotAgent(Capture([[ToolCall("go_to", {"x": 50, "y": 5, "reason": "far"})]]), tmp_path / "t.jsonl").step(req)
    assert seen and all('"search_area"' not in m and '"outline"' not in m and "[0, 0, 20, 14]" not in m for m in seen)
    assert [o["type"] for o in out["orders"]] == ["go_to"]


def test_replay_backend_reproduces_a_recorded_step(tmp_path, monkeypatch):
    fake_vision(monkeypatch, "person")
    turns = [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 3})],
             [ToolCall("confirm_survivor", {"x": 5, "y": 8, "evidence_id": "ev1"})]]
    first, _ = run(tmp_path, turns)
    # A replay runs in a fresh process from the start of the recording, so the team conversation starts empty too: the
    # first step's intent would otherwise change the replayed step's context and miss the recorded reply.
    TeamState.reset()
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
    # 5 m ahead, stopping the brain's standoff short of it, which must be inside the arm's reach so a lift can follow.
    assert o["type"] == "go_to" and o["x"] == pytest.approx(5.0) and o["y"] == pytest.approx(10.0 - T.APPROACH_STANDOFF_M)
    # A walk to a person (not this uncertain look) ends inside the arm's measured reach so a lift can follow.
    assert T.RESCUE_APPROACH_M < float(T._RESCUE["reach_m"])


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


def test_walking_to_a_person_the_inspection_saw_is_never_refused_as_another_look(tmp_path, monkeypatch):
    # Held-out seed 914 run 4: two close looks at a spot, then a "person" verdict and confirmation; approach_candidate
    # refused, so robot B never reached the arm's range. The control: the same calls with an "uncertain" verdict.
    for verdict, refused in (("person", False), ("uncertain", True)):
        fake_vision(monkeypatch, verdict)
        T.ToolRegistry._looks.clear(); T.ToolRegistry._pending.clear()
        for k in range(3):
            req = request(tmp_path) | {"t": 30.0 + k}
            out = RobotAgent(Scripted([[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 5})],
                                       [ToolCall("approach_candidate", {"evidence_id": "ev1"})]]), tmp_path / "t.jsonl").step(req)
        assert bool(out["refusals"]) is refused, verdict
        if not refused:
            (o,) = out["orders"]
            assert o["type"] == "go_to" and (o["face_x"], o["face_y"]) == pytest.approx((5.0, 10.0))
            assert T.ToolRegistry._pending.get("robot_A") is None   # nothing left to resolve: exploring stays allowed


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


def test_commander_trace_records_the_reports_that_reached_the_post(tmp_path):
    # Agent run 9: the commander said "no update" at 30 s although a survivor had been confirmed at 13 s. The trace
    # held no inputs, so "nothing arrived" could not be told from "arrived and ignored".
    from brain.agent.commander import Commander

    trace = tmp_path / "trace.jsonl"
    report = {"from": "robot_A", "t": 13.0, "text": "Confirmed survivor in my current area."}
    Commander(Scripted([]), trace).step({"t": 30.0, "reports": [report], "robots": ["robot_A", "robot_B"]})
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    start = [r for r in rows if r["kind"] == "step_start"]
    assert len(start) == 1 and start[0]["reports"] == [report]
    assert start[0]["step"] == "commander@30.000"


def test_a_survivor_is_confirmed_only_from_a_close_inspection(tmp_path, monkeypatch):
    # Slow-fire run: a confirmation from 6.55 m at the image edge landed 1.55 m from the person (a duplicate scored
    # as false). Beyond person_confirm_max_range_m the agent must approach first; within it, confirming works.
    fake_vision(monkeypatch, "person")
    monkeypatch.setattr(T.vision, "inspect_candidate", lambda f, b, r, output_dir: {
        "id": f"ev{r:g}", "tool": "inspect_candidate", "verdict": "person", "params": {"bearing_deg": b, "range_m": r},
        "candidate": {"bearing_deg": b, "range_m": r}})
    far, _ = run(tmp_path, [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 6.5})],
                            [ToolCall("confirm_survivor", {"x": 5, "y": 11.5, "evidence_id": "ev6.5"})]])
    assert far["orders"] == [] and far["refusals"][0]["tool"] == "confirm_survivor"
    assert "approach_candidate" in far["refusals"][0]["reason"]
    near, _ = run(tmp_path, [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 3})],
                             [ToolCall("confirm_survivor", {"x": 5, "y": 8, "evidence_id": "ev3"})]])
    assert [o["type"] for o in near["orders"]] == ["confirm_survivor"], "control: 3 m"


def test_a_person_with_no_measured_range_is_not_confirmed_at_a_guessed_spot(tmp_path, monkeypatch):
    # Held-out seed 920, robot B: a person far through two doorways, no depth on the body, inspected at a range the model
    # guessed (1.3 m), and confirmed there, 5.6 m from anyone. Control: the same calls with a measured range confirm.
    fake_vision(monkeypatch, "person")
    monkeypatch.setattr(T.vision, "inspect_candidate", lambda f, b, r, output_dir: {
        "id": "ev1", "tool": "inspect_candidate", "verdict": "person", "candidate": {"bearing_deg": b, "range_m": None}})
    out, _ = run(tmp_path, [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 1.3})],
                            [ToolCall("confirm_survivor", {"x": 5, "y": 6.3, "evidence_id": "ev1"})]])
    assert out["orders"] == [] and "measured no range" in out["refusals"][0]["reason"]
    fake_vision(monkeypatch, "person")
    out, _ = run(tmp_path, [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 1.3})],
                            [ToolCall("confirm_survivor", {"x": 5, "y": 6.3, "evidence_id": "ev1"})]])
    assert [o["type"] for o in out["orders"]] == ["confirm_survivor"], "control: measured range"


def test_blocked_unsearched_floor_reaches_the_model(tmp_path, monkeypatch):
    fake_vision(monkeypatch)
    seen = []

    class Capture(Scripted):
        def chat(self, messages, tools):
            seen.append(messages[1]["content"])
            return super().chat(messages, tools)

    blocked = {"bearing_deg": 4.0, "distance_m": 2.1, "unsearched_m2": 6.5, "approach": {"x": 5.0, "y": 6.2}}
    req = {**request(tmp_path), "event": {"kind": "no_frontier"}, "frontiers": [], "blocked_unsearched": blocked}
    RobotAgent(Capture([]), tmp_path / "t.jsonl").step(req)
    assert seen and '"blocked_unsearched": {"bearing_deg": 4.0' in seen[0]


def test_spray_water_needs_a_measured_hot_spot_on_that_bearing_and_water(tmp_path, monkeypatch):
    # The nozzle may only be aimed at a fire the robot's own thermal scan measured, within aim tolerance and range,
    # and only while there is water. A long request is clipped to max_spray_s.
    fake_vision(monkeypatch)
    monkeypatch.setattr(T.vision, "scan_thermal", lambda f: {"id": "scan1", "tool": "scan_thermal",
                        "hotspots": [{"bearing_deg": 5.0, "range_m": 3.2}], "people": []})

    def spray(bearing, water, seconds=4):
        agent = RobotAgent(Scripted([[ToolCall("scan_thermal", {})],
                                     [ToolCall("spray_water", {"bearing_deg": bearing, "seconds": seconds, "evidence_id": "scan1"})]]),
                           tmp_path / f"t{bearing}{water}.jsonl")
        spray.t += 30   # a fresh event each call; the agent skips a repeat of a recent one
        return agent.step({**request(tmp_path), "t": spray.t, "event": {"kind": "avoid_fire"}, "water_l": water})
    spray.t = 0

    ok = spray(0, 12)
    assert [o["type"] for o in ok["orders"]] == ["spray_water"] and ok["orders"][0]["seconds"] == 4
    assert spray(40, 12)["refusals"][0]["tool"] == "spray_water", "40 deg off the measured fire"
    assert "empty" in spray(0, 0)["refusals"][0]["reason"]
    assert spray(0, 12, seconds=60)["orders"][0]["seconds"] == 6.0, "clipped to max_spray_s"


def test_extract_survivor_needs_a_close_person_verdict_and_a_free_sled(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "RESCUE_ENABLED", True)   # rescue is switched off in actuators.json; this tests the code behind the switch
    fake_vision(monkeypatch, "person")
    monkeypatch.setattr(T.vision, "inspect_candidate", lambda f, b, r, output_dir: {
        "id": f"ev{r:g}", "tool": "inspect_candidate", "verdict": "person", "candidate": {"bearing_deg": b, "range_m": r}})

    def extract(range_m, carrying=False):
        extract.t += 30
        agent = RobotAgent(Scripted([[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": range_m})],
                                     [ToolCall("extract_survivor", {"evidence_id": f"ev{range_m:g}"})]]), tmp_path / f"x{extract.t}.jsonl")
        return agent.step({**request(tmp_path), "t": extract.t, "carrying_person": carrying})
    extract.t = 0
    # Within the arm's measured reach (actuators.json rescue.reach_m, 0.85 m); 1.2 m was accepted before and the arm
    # then refused it (seed 914 run 6).
    assert [o["type"] for o in extract(0.7)["orders"]] == ["extract_survivor"]
    assert extract(1.2)["refusals"][0]["tool"] == "extract_survivor", "beyond the arm's reach: approach first"
    assert "already towing" in extract(0.7, carrying=True)["refusals"][0]["reason"]



class Recording(Scripted):
    """Scripted backend that keeps the messages it was sent, so a test can read what the model was told."""
    def __init__(self, turns):
        super().__init__(turns); self.seen = []

    def chat(self, messages, tools):
        self.seen.append([dict(m) for m in messages]); return super().chat(messages, tools)


def test_a_confirmation_says_how_to_get_the_person_out(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "RESCUE_ENABLED", True)   # rescue is switched off in actuators.json; this tests the code behind the switch
    # Run 15: robot A confirmed a person 2.7 m away and searched on. The confirm result names the next step: an approach
    # when the inspection was beyond the arm's reach, a lift when it was within it.
    def told(range_m):
        T.ToolRegistry._confirmed.clear()   # each call confirms the same spot afresh (one person per survivor_merge_m)
        monkeypatch.setattr(T.vision, "load_frame", lambda d, t: object())
        monkeypatch.setattr(T.vision, "inspect_candidate", lambda f, b, r, output_dir: {
            "id": "ev1", "tool": "inspect_candidate", "verdict": "person", "candidate": {"range_m": range_m}})
        backend = Recording([[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": range_m})],
                             [ToolCall("confirm_survivor", {"x": 5, "y": 5 + range_m, "evidence_id": "ev1"})]])
        out = RobotAgent(backend, tmp_path / f"{range_m}.jsonl").step(request(tmp_path))
        assert [o["type"] for o in out["orders"]] == ["confirm_survivor"]
        return [m for m in backend.seen[-1] if m.get("role") == "tool" and m.get("name") == "confirm_survivor"][-1]["content"]
    assert "approach_candidate" in told(2.7) and "extract_survivor" not in told(2.7)
    assert "extract_survivor" in told(0.7) and "extract_survivor" not in told(1.2)


def test_confirmed_people_not_yet_lifted_are_listed_with_bearing_and_approach():
    # Run 19: robot A confirmed a person, a heat alarm interrupted the approach, and nothing brought it back.
    T.ToolRegistry._lifted.clear()
    pose = {"x": 0.0, "y": 0.0, "heading_deg": 90.0}          # facing +x
    out = T.survivors_to_rescue("robot_A", pose, [[3.0, 0.0], [0.0, -5.0]])
    assert [s["x"] for s in out] == [3.0, 0.0]                 # nearest first
    assert out[0]["bearing_deg"] == pytest.approx(0.0) and out[0]["distance_m"] == pytest.approx(3.0)
    assert out[0]["approach"]["x"] == pytest.approx(3.0 - T.RESCUE_APPROACH_M)
    assert out[1]["bearing_deg"] == pytest.approx(90.0)        # -y is to the right of a robot facing +x
    T.ToolRegistry._lifted["robot_A"] = [(3.1, 0.1)]
    assert [s["x"] for s in T.survivors_to_rescue("robot_A", pose, [[3.0, 0.0], [0.0, -5.0]])] == [0.0]
    T.ToolRegistry._lifted.clear()


def test_turn_to_face_is_a_go_to_where_the_robot_stands_with_a_face_point(tmp_path, monkeypatch):
    # Run 20: robot A was within reach of a confirmed person 106 to 148 deg off its heading; inspections there looked
    # outside the picture. turn_to_face sends a go_to at the robot's own position with the person as the face point.
    # Control: during a heat event it is refused.
    fake_vision(monkeypatch)
    out, _ = run(tmp_path, [[ToolCall("turn_to_face", {"x": 2.0, "y": 3.0, "reason": "person to lift"})]])
    (o,) = out["orders"]
    req = request(tmp_path)
    assert o["type"] == "go_to" and (o["face_x"], o["face_y"]) == (2.0, 3.0)
    assert (o["x"], o["y"]) == (round(req["pose"]["x"], 2), round(req["pose"]["y"], 2))
    RobotAgent._recent.clear()   # another test's avoid_fire at this place would be skipped as a repeat
    heat = RobotAgent(Scripted([[ToolCall("turn_to_face", {"x": 2.0, "y": 3.0, "reason": "r"})]]), tmp_path / "t.jsonl").step(
        request(tmp_path) | {"t": 20.0, "event": {"kind": "avoid_fire"}})   # a new step time: steps are keyed by robot and t
    assert heat["orders"] == [] and heat["refusals"][0]["tool"] == "turn_to_face"


def test_turn_to_face_is_refused_for_a_point_already_in_view_or_under_the_robot(tmp_path, monkeypatch):
    # One-door runs 926 to 929 and 932: of 89 turn_to_face orders, 24 aimed at a point already inside the thermal view
    # (seed 932 robot A: the same patch faced twice in one step at 62.7 s) and 16 at points under 1 m away.
    fake_vision(monkeypatch)
    def face(x, y, t):
        RobotAgent._recent.clear()
        return RobotAgent(Scripted([[ToolCall("turn_to_face", {"x": x, "y": y, "reason": "r"})]]), tmp_path / f"{t}.jsonl").step(
            request(tmp_path) | {"t": t})
    ahead = face(5.0, 8.0, 30.0)                                   # robot at (5, 5) facing +y: dead ahead
    assert ahead["orders"] == [] and "already in view" in ahead["refusals"][0]["reason"]
    edge = face(5.0 + 3 * math.tan(math.radians(T.THERMAL_HALF_HFOV_DEG - 1)), 8.0, 31.0)
    assert edge["orders"] == [], "inside the thermal half view by 1 deg"
    under = face(5.2, 4.9, 32.0)
    assert under["orders"] == [] and "under this robot" in under["refusals"][0]["reason"]
    side = face(8.0, 5.0, 33.0)                                    # control: 90 deg to the right is out of view
    assert [o["type"] for o in side["orders"]] == ["go_to"] and side["refusals"] == []
    beyond = face(5.0 + 3 * math.tan(math.radians(T.THERMAL_HALF_HFOV_DEG + 2)), 8.0, 34.0)
    assert [o["type"] for o in beyond["orders"]] == ["go_to"], "control: just outside the thermal view the turn is allowed"


def test_approach_candidate_faces_the_inspected_spot_on_arrival(tmp_path, monkeypatch):
    fake_vision(monkeypatch, "uncertain")
    T.ToolRegistry._looks.clear()
    out, _ = run(tmp_path, [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 5})],
                            [ToolCall("approach_candidate", {"evidence_id": "ev1"})]])
    (o,) = out["orders"]
    assert (o["face_x"], o["face_y"]) == pytest.approx((5.0, 10.0))


def test_a_thermal_only_warm_patch_on_a_wall_waits_until_the_frontiers_are_searched(tmp_path, monkeypatch):
    # Over 15 runs 151 thermal-only candidates on a long wall held 1 person; seed 932 robot A spent four orders on one
    # wall patch. Controls: no frontiers left, a patch not on a wall, and a colour sighting at the spot all allow it.
    fake_vision(monkeypatch, "uncertain")
    def step(t, numbers, frontiers, calls):
        RobotAgent._recent.clear(); T.ToolRegistry._looks.clear(); T.ToolRegistry._pending.clear()
        req = request(tmp_path) | {"t": t, "event": {"kind": "survivor_candidate", "numbers": numbers}, "frontiers": frontiers}
        return RobotAgent(Scripted(calls), tmp_path / f"{t}.jsonl").step(req)
    T.ToolRegistry._wall_spots.clear()
    wall = {"x_m": 5.0, "y_m": 10.0, "on_wall": 1.0, "rgb": 0.0}
    frontier = [{"id": "f1", "x": 9.0, "y": 1.0}]
    approach = [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 5})], [ToolCall("approach_candidate", {"evidence_id": "ev1"})]]
    out = step(40.0, wall, frontier, approach)
    assert out["orders"] == [] and "1 in 151" in out["refusals"][0]["reason"], out
    face = step(41.0, wall, frontier, [[ToolCall("turn_to_face", {"x": 5.3, "y": 10.0, "reason": "r"})]])
    assert face["orders"] == [] and "1 in 151" in face["refusals"][0]["reason"]
    assert [o["type"] for o in step(42.0, wall, [], approach)["orders"]] == ["go_to"], "control: no frontier left"
    T.ToolRegistry._wall_spots.clear()
    assert [o["type"] for o in step(43.0, wall | {"on_wall": 0.0}, frontier, approach)["orders"]] == ["go_to"], "control: not on a wall"
    step(44.0, wall, frontier, [[ToolCall("finish", {"summary": "s"})]])
    step(45.0, wall | {"on_wall": 0.0, "rgb": 1.0}, frontier, [[ToolCall("finish", {"summary": "s"})]])
    assert [o["type"] for o in step(46.0, {"x_m": 0.0, "y_m": 0.0}, frontier, approach)["orders"]] == ["go_to"], "a colour sighting clears the spot"


def test_people_to_rescue_say_whether_they_are_in_the_camera_view():
    T.ToolRegistry._lifted.clear()
    assert 37.0 < T.CAMERA_HALF_HFOV_DEG < 38.0               # 640 px wide, fx 415.7
    pose = {"x": 0.0, "y": 0.0, "heading_deg": 0.0}           # facing +y
    side, ahead = T.survivors_to_rescue("robot_A", pose, [[0.5, 2.0], [-1.5, 0.2]])   # nearest first
    assert ahead["in_view"] is True and side["in_view"] is False


def test_memory_of_close_looks_and_lifts_resets_with_a_new_recording(tmp_path, monkeypatch):
    # Run 24: the server kept looks from runs 21 to 23, so a fresh run's first approach to a person was refused.
    # Control: within one recording, the third close look at a spot is still refused.
    fake_vision(monkeypatch, "uncertain")
    def look(run_dir, t):
        d = tmp_path / run_dir / "robot_A"; d.mkdir(parents=True, exist_ok=True)
        req = request(tmp_path) | {"robot_dir": str(d), "t": t}
        return RobotAgent(Scripted([[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 5})],
                                    [ToolCall("approach_candidate", {"evidence_id": "ev1"})]]), tmp_path / "t.jsonl").step(req)
    for k in range(2):
        look("run_a", 10.0 + k)
    assert look("run_a", 12.0)["refusals"], "control: third look in the same recording"
    assert look("run_b", 10.0)["orders"], "a new recording starts with no looks"
    for k in range(2):
        look("run_b", 20.0 + k)
    assert look("run_b", 1.0)["orders"], "same folder, clock restarted: a new recording"


def test_lift_may_cite_the_confirming_inspection_once_within_reach_and_facing(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "RESCUE_ENABLED", True)   # rescue is switched off in actuators.json; this tests the code behind the switch
    # Run 25: robot A stood 1.34 m from the person it had confirmed, facing them, and the re-inspection was uncertain:
    # a person lying that close is below the colour camera's view. The lift may cite the confirming inspection when the
    # robot is within reach and facing them. Controls: beyond reach, and facing away, are refused.
    fake_vision(monkeypatch, "person")
    T.ToolRegistry._confirmed.clear(); RobotAgent._recent.clear()
    out, _ = run(tmp_path, [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 3})],
                            [ToolCall("confirm_survivor", {"x": 5, "y": 8, "evidence_id": "ev1"})]])
    assert [o["type"] for o in out["orders"]] == ["confirm_survivor"]
    def lift(x, y, heading, t):
        req = request(tmp_path) | {"t": t, "pose": {"x": x, "y": y, "heading_deg": heading}}
        return RobotAgent(Scripted([[ToolCall("extract_survivor", {"evidence_id": "ev1"})]]), tmp_path / "t.jsonl").step(req)
    near = lift(5.0, 7.3, 0.0, 20.0)                 # 0.7 m short of them (inside the 0.85 m reach), facing +y
    assert [o["type"] for o in near["orders"]] == ["extract_survivor"]
    # The order names the person: the arm lifts only the body at that spot.
    assert (near["orders"][0]["target_x"], near["orders"][0]["target_y"]) == (5.0, 8.0)
    T.ToolRegistry._lifted.clear()
    assert lift(5.0, 4.0, 0.0, 30.0)["refusals"], "control: 4 m away"
    assert lift(5.0, 7.3, 180.0, 40.0)["refusals"], "control: facing away"


def test_the_lift_description_states_both_ways_the_check_accepts():
    # Held-out seed 914 run 5: the description named only a fresh person verdict taken within the reach, which a lying
    # person below the camera's view cannot give; both agents re-inspected confirmed people for 70 s and lifted no one.
    from brain.agent.robot_agent import SYSTEM
    d = next(t for t in T.ALL_TOOLS if t["function"]["name"] == "extract_survivor")["function"]["description"]
    assert "survivors_to_rescue" in d and "confirmed" in d
    assert f"{float(T._RESCUE['reach_m']):.1f} m" in d and f"{float(T._RESCUE['grip_cone_deg']):.0f} deg" in d
    assert "evidence\n  that confirmed them" in SYSTEM or "evidence that confirmed them" in SYSTEM.replace("\n  ", " ")


def test_walking_back_to_a_person_this_robot_confirmed_is_not_another_uncertain_look(tmp_path, monkeypatch):
    # Run 5, robot A: it confirmed a person, then cited a newer uncertain look at them and was refused after two looks.
    # Control: the same uncertain looks at a spot nobody confirmed are still limited. With the rescue arm on only.
    monkeypatch.setattr(T, "RESCUE_ENABLED", True)
    for confirmed, refused in ((True, False), (False, True)):
        T.ToolRegistry._looks.clear(); T.ToolRegistry._pending.clear(); T.ToolRegistry._confirmed.clear()
        if confirmed:
            T.ToolRegistry._confirmed["robot_A"] = [{"id": "old", "x": 5.0, "y": 10.0}]
        fake_vision(monkeypatch, "uncertain")
        for k in range(3):
            req = request(tmp_path) | {"t": 50.0 + k}
            out = RobotAgent(Scripted([[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 5})],
                                       [ToolCall("approach_candidate", {"evidence_id": "ev1"})]]), tmp_path / "t.jsonl").step(req)
        assert bool(out["refusals"]) is refused, confirmed
    T.ToolRegistry._confirmed.clear()


def test_an_approach_to_an_uncertain_candidate_is_finished_before_exploring(tmp_path, monkeypatch):
    # Hidden-corner run 3: both agents ordered an approach and dropped it for a frontier on the next area_searched event,
    # so nobody was confirmed. explore_frontier is refused while the approach is open; the context shows it. It opens
    # again when the approach fails (goal_blacklisted), times out (pending_candidate_s), or the spot is rejected.
    frontier = {"id": 0, "x": 0.0, "y": 0.0, "path_m": 5.0, "unsearched_m2": 4.0, "proven_indoors": True}
    explore = [[ToolCall("explore_frontier", {"frontier_id": 0, "reason": "more area"})]]

    def step(t, kind, turns, verdict="uncertain"):
        fake_vision(monkeypatch, verdict)
        req = request(tmp_path) | {"t": t, "event": {"kind": kind}, "frontiers": [frontier]}
        return RobotAgent(Scripted(turns), tmp_path / "t.jsonl").step(req)

    def open_approach(t):
        T.ToolRegistry._pending.clear(); T.ToolRegistry._looks.clear()
        out = step(t, "survivor_candidate", [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 5})],
                                             [ToolCall("approach_candidate", {"evidence_id": "ev1"})]])
        assert [o["type"] for o in out["orders"]] == ["go_to"]

    open_approach(10.0)
    held = step(14.0, "area_searched", explore)
    assert held["orders"] == [] and "unresolved warm candidate" in held["refusals"][0]["reason"]
    last = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines() if '"step_start"' in l][-1]
    assert last["context"]["pending_candidate"]["distance_m"] == pytest.approx(5.0)

    open_approach(10.0)
    assert [o["type"] for o in step(14.0, "goal_blacklisted", explore)["orders"]] == ["explore_frontier"], "failed approach"

    open_approach(10.0)
    late = 10.0 + float(T._PERCEPTION["pending_candidate_s"]) + 1
    assert [o["type"] for o in step(late, "area_searched", explore)["orders"]] == ["explore_frontier"], "timed out"

    open_approach(10.0)
    rejected = step(20.0, "agent_goal_reached", [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 5})],
                                                 [ToolCall("reject_candidate", {"x": 5.0, "y": 10.0, "evidence_id": "ev1"})],
                                                 [ToolCall("explore_frontier", {"frontier_id": 0, "reason": "next"})]],
                    verdict="not_person")
    assert [o["type"] for o in rejected["orders"]] == ["reject_candidate", "explore_frontier"], "resolved by a reject"


def test_a_person_this_robot_carried_out_is_not_confirmed_again(tmp_path, monkeypatch):
    # Held-out run 2: robot B's agent confirmed the person B had set down in the yard as a new survivor. A confirmation
    # within survivor_merge_m of a set-down point in the robot's own decision log is refused with the reason. Controls:
    # a set-down point 7 m away, and one logged after the step's time, do not block it.
    fake_vision(monkeypatch, "person")
    def confirm(set_down_x, set_down_y, at_t):
        T.ToolRegistry._confirmed.clear()   # each call confirms the same spot afresh (one person per survivor_merge_m)
        line = {"t": at_t, "robot": "robot_A", "kind": "person_carried_out", "reason": "Set the person down.",
                "numbers": {"distance_to_exit_m": 2.2, "person_x_m": set_down_x, "person_y_m": set_down_y}}
        (tmp_path / "decisions.jsonl").write_text(json.dumps({"t": 1.0, "kind": "replan", "numbers": {}}) + "\n" + json.dumps(line) + "\n")
        out, _ = run(tmp_path, [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 3})],
                                [ToolCall("confirm_survivor", {"x": 5, "y": 8, "evidence_id": "ev1"})]])
        return out
    near = confirm(5.2, 8.3, 5.0)
    assert near["orders"] == [] and "already rescued" in near["refusals"][0]["reason"]
    assert [o["type"] for o in confirm(5.0, 1.0, 5.0)["orders"]] == ["confirm_survivor"], "control: set down 7 m away"
    assert [o["type"] for o in confirm(5.2, 8.3, 50.0)["orders"]] == ["confirm_survivor"], "control: logged after this step"


def test_an_event_at_a_new_place_is_not_dropped_as_a_repeat(tmp_path):
    # Held-out seed 914 runs 1 to 3: arriving at a second place within 12 s was dropped as a repeat of the first arrival,
    # so the reflex picked the next frontier alone. Positional events repeat only at the same spot; others by time.
    agent = RobotAgent(Scripted([]), tmp_path / "t.jsonl")
    RobotAgent._recent.clear(); RobotAgent._ordered.clear()   # class-level memory: earlier tests' steps must not leak in
    def ev(t, kind, x=None):
        n = {} if x is None else {"x_m": x, "y_m": 0.0}
        return {"robot": "robot_A", "t": t, "event": {"kind": kind, "numbers": n}}
    assert not agent._repeat(ev(0, "agent_goal_reached", 0.0))
    assert not agent._repeat(ev(8, "agent_goal_reached", 4.0)), "a new place is a new decision"
    assert agent._repeat(ev(10, "agent_goal_reached", 4.5)), "the same place within the window repeats"
    assert not agent._repeat(ev(0, "sensor_switch"))
    assert agent._repeat(ev(5, "sensor_switch")), "events without a position repeat by time"


def test_an_arrival_after_a_new_order_is_not_a_repeat(tmp_path):
    # Seed 929, robot A: ordered at 44.0 s to face s4 again, it arrived at 46.8 s; the arrival faced the same point as the
    # one at 38.2 s and was dropped as a repeat, so the agent never inspected her again. Control: with no order in between
    # the same arrival still repeats.
    def agent_with(order_at):
        agent = RobotAgent(Scripted([]), tmp_path / "t.jsonl")
        RobotAgent._recent.clear(); RobotAgent._ordered.clear()
        ev = lambda t: {"robot": "robot_A", "t": t, "event": {"kind": "agent_goal_reached", "numbers": {"x_m": -4.1, "y_m": 3.9}}}
        assert not agent._repeat(ev(38.2))
        if order_at is not None:
            RobotAgent._ordered["robot_A"] = order_at
        return agent._repeat(ev(46.8))
    assert not agent_with(44.0), "a new order since the last arrival makes this arrival a new decision"
    assert agent_with(None), "control: the same arrival with no order in between repeats"
    assert agent_with(30.0), "control: an order before the last arrival does not count"


def test_where_fire_blocks_the_way_out_reaches_the_model(tmp_path):
    # Held-out seed 914 run 5: robot A held at its coolest floor while its air rose to 185 C; its agent was never told where
    # fire cut its way out. The context whitelist drops unknown fields, so the new one must be passed through.
    seen = {}

    class Model:
        model = "m"

        def chat(self, messages, tools):
            seen["ctx"] = json.dumps(messages)
            return Reply("", [ToolCall("finish", {"summary": "x"})], 0.0, "m")

    RobotAgent(Model(), tmp_path / "t.jsonl").step(request(tmp_path) | {"fire_blocks_exit": {"bearing_deg": 170.5, "distance_m": 2.1, "exit_path_m": 9.5}})
    assert "fire_blocks_exit" in seen["ctx"] and "170.5" in seen["ctx"]


def test_with_rescue_switched_off_nothing_about_lifting_reaches_the_model(tmp_path, monkeypatch):
    # 22 Oct: carrying people out is switched off (actuators.json rescue.enabled). The lift tool, the rescue list and the
    # prompt's rescue text must not reach the model, and a scripted lift is refused. Control: switched on, all three appear.
    from brain.agent import robot_agent as R
    fake_vision(monkeypatch, "person")
    monkeypatch.setattr(T, "RESCUE_ENABLED", False)
    assert "extract_survivor" not in [t["function"]["name"] for t in T.tool_specs()]
    assert "extract" not in R.system_prompt() and "survivors_to_rescue" not in R.system_prompt()
    out, trace = run(tmp_path, [[ToolCall("extract_survivor", {"evidence_id": "ev1"})]])
    assert out["orders"] == [] and "switched off" in out["refusals"][0]["reason"]
    start = [r for r in trace if r["kind"] == "step_start"][-1]
    assert "survivors_to_rescue" not in start.get("context", {})
    monkeypatch.setattr(T, "RESCUE_ENABLED", True)
    assert "extract_survivor" in [t["function"]["name"] for t in T.tool_specs()] and "extract_survivor" in R.system_prompt()



def test_every_radio_message_carries_the_state_the_tools_accepted(tmp_path, monkeypatch):
    # Held-out seed 917, robot B: confirm_survivor was refused (inspection from beyond the confirm range) and the next
    # call radioed "Confirmed survivor". The text now carries the robot's own state, written by code. Control: after an
    # accepted confirmation the state lists that person.
    T.ToolRegistry._confirmed.clear(); RobotAgent._recent.clear()
    fake_vision(monkeypatch, "uncertain")
    out, _ = run(tmp_path, [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 3})],
                            [ToolCall("confirm_survivor", {"x": 5, "y": 8, "evidence_id": "ev1"})],
                            [ToolCall("send_team_message", {"text": "Confirmed survivor at (5, 8)."})]])
    (msg,) = [o for o in out["orders"] if o["type"] == "send_team_message"]
    assert "this robot has confirmed nobody" in msg["text"] and "confirmation was refused this step" in msg["text"]
    T.ToolRegistry._confirmed.clear(); RobotAgent._recent.clear()
    fake_vision(monkeypatch, "person")
    req = request(tmp_path) | {"t": 40.0}
    out = RobotAgent(Scripted([[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 3})],
                               [ToolCall("confirm_survivor", {"x": 5, "y": 8, "evidence_id": "ev1"})],
                               [ToolCall("send_team_message", {"text": "Found one."})]]), tmp_path / "t2.jsonl").step(req)
    (msg,) = [o for o in out["orders"] if o["type"] == "send_team_message"]
    assert "(5.0, 8.0)" in msg["text"] and "refused" not in msg["text"]
    T.ToolRegistry._confirmed.clear()


def test_an_approach_the_brain_refused_is_no_longer_pending(tmp_path, monkeypatch):
    # Codex review #28: the agent kept "walking to look closer" for up to 60 s after the brain refused the walk (for
    # example "no ceiling has been seen over that point") or found no path. The brain's decision log says so; the
    # pending approach ends. Control: with nothing in the log, it stays pending.
    fake_vision(monkeypatch, "uncertain")
    run(tmp_path, [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 5})], [ToolCall("approach_candidate", {"evidence_id": "ev1"})]])
    reg = T.ToolRegistry(tmp_path, T.AgentContext(robot="robot_A", t=12.0, x=5.0, y=5.0, heading_deg=0.0), tmp_path)
    assert reg.pending() is not None, "control: nothing says the walk failed"
    (tmp_path / "decisions.jsonl").write_text(json.dumps({"t": 10.0, "robot": "robot_A", "kind": "agent_order",
        "reason": "Refused go_to: no ceiling has been seen over that point, so it is not known to be inside."}) + "\n")
    reg = T.ToolRegistry(tmp_path, T.AgentContext(robot="robot_A", t=13.0, x=5.0, y=5.0, heading_deg=0.0), tmp_path)
    assert reg.pending() is None



def test_approaching_from_too_close_steps_back_to_the_viewing_distance(tmp_path, monkeypatch):
    # One-door seed 923: robot A stood about 1 m from a woman (inside its camera's blind floor) and re-centred six times;
    # approach_candidate's point was its own position, because the standoff was floored at zero. Now a robot 1 m from
    # the spot steps back to the standoff on the same line. Control: from 5 m it still stops the standoff short.
    def target(range_m):
        fake_vision(monkeypatch, "uncertain")
        out, _ = run(tmp_path, [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": range_m})],
                                [ToolCall("approach_candidate", {"evidence_id": "ev1"})]])
        T.ToolRegistry._looks.clear()
        (o,) = out["orders"]; return o
    near = target(1.0)                                  # robot at (5, 5) facing +y; the spot is at (5, 6)
    assert near["y"] == pytest.approx(6.0 - T.APPROACH_STANDOFF_M) and near["y"] < 5.0, near
    far = target(5.0)
    assert far["y"] == pytest.approx(10.0 - T.APPROACH_STANDOFF_M), "control"


def test_a_clipped_body_closer_than_the_cameras_floor_view_says_so():
    # The clipped inspections at 1 m gave the agent no reason why they kept failing.
    from brain.agent import vision as v
    assert 2.0 < v._blind_floor_m() < 2.3, "the 1.6 m mast sees the floor from about 2.16 m"


def test_the_commander_is_told_what_robots_can_do_for_a_found_person(tmp_path, monkeypatch):
    # One-door seed 924: with carrying switched off, the commander told robot A to "protect and maintain access to the
    # confirmed survivor" and A held beside her for about 30 s instead of searching.
    from brain.agent import commander as C

    class Seen(Scripted):
        def chat(self, messages, tools):
            self.system = messages[0]["content"]
            return super().chat(messages, tools)

    for enabled, said in ((False, "cannot lift, carry, shield or guard anyone"), (True, "carry them out")):
        monkeypatch.setattr(C, "RESCUE_ENABLED", enabled)
        model = Seen([])
        C.Commander(model, tmp_path / "trace.jsonl").step({"t": 48.0, "reports": [{"from": "robot_A", "text": "Survivor confirmed by the door."}], "robots": ["robot_A", "robot_B"]})
        assert said in model.system


def test_an_approach_dropped_for_heat_is_kept_as_a_look_not_finished(tmp_path, monkeypatch):
    # One-door seed 924, robot A at 37.6 s: a "person" verdict from 4.6 m, an approach ordered, then a heat event backed
    # the robot away; the next step had no trace of the candidate and she was never found. Control: once rejected, it
    # is gone; and a candidate the team has confirmed since is not listed.
    from brain.agent.robot_agent import RobotAgent as Agent
    fake_vision(monkeypatch, "uncertain")

    class Seen(Scripted):
        def chat(self, messages, tools):
            self.context = json.loads(messages[1]["content"].split("Context:\n", 1)[1])
            return super().chat(messages, tools)

    def step(t, event, turns, survivors=()):
        model = Seen(turns)
        req = request(tmp_path) | {"t": t, "event": {"kind": event}, "survivors": [list(s) for s in survivors]}
        Agent(model, tmp_path / "t.jsonl").step(req)
        return model.context

    step(10.0, "survivor_candidate", [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 4.6})],
                                      [ToolCall("approach_candidate", {"evidence_id": "ev1"})]])
    assert "pending_candidate" in step(12.0, "periodic", [])
    seen = step(14.0, "avoid_fire", [])
    [look] = seen["looks_not_finished"]
    assert look["dropped_because"] == "avoid_fire" and abs(look["y"] - 9.6) < .05 and "pending_candidate" not in seen
    assert step(20.0, "periodic", [], survivors=[(5.0, 9.6)]).get("looks_not_finished") is None, "confirmed by the team since"
    assert step(22.0, "periodic", [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 4.6})]])["looks_not_finished"]
    fake_vision(monkeypatch, "not_person")
    step(24.0, "periodic", [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 4.6})],
                            [ToolCall("reject_candidate", {"x": 5.0, "y": 9.6, "evidence_id": "ev1"})]])
    assert step(26.0, "periodic", []).get("looks_not_finished") is None, "rejected: no longer listed"


def test_the_agent_is_told_the_robot_stays_inside_when_configured(monkeypatch):
    # Owner's decision (22 Oct): robots never leave by the entrance; the prompt must not say they leave on heat.
    from brain.agent import robot_agent as RA
    monkeypatch.setitem(T._CONFIG, "stay_inside", True)
    text = RA.system_prompt()
    assert "never leaves the building" in text and "It leaves the building when" not in text
    monkeypatch.setitem(T._CONFIG, "stay_inside", False)
    assert "It leaves the building when its own heat rules say so" in RA.system_prompt(), "control"


def test_a_fire_heating_the_robot_is_a_spray_option(tmp_path):
    # Seed 929, robot B: air 30 to 211 C beside a fire in its own room; the spray option existed only for a fire blocking
    # a way out or a frontier, so its agent was never offered its water. Control: with no fire near, no option.
    from brain.agent.tools import spray_option
    near = {"bearing_deg": 12.0, "distance_m": 3.1, "burning_m2": 1.5}
    opt = spray_option({"fire_near_robot": near, "water_l": 12.0})
    assert opt is not None and opt["blocks"] == "near" and opt["within_nozzle_reach"] and opt["in_camera_view"]
    assert spray_option({"water_l": 12.0}) is None, "control: no fire near the robot, no option"
    blocked = {"bearing_deg": 0.0, "distance_m": 2.0, "burning_m2": 1.0}
    assert spray_option({"fire_blocks_exit": blocked, "fire_near_robot": near, "water_l": 12.0})["blocks"] == "exit", "a blocked route still comes first"


def test_without_the_rescue_arm_walking_to_a_confirmed_person_is_refused(tmp_path, monkeypatch):
    # One-door seed 930, robot A at 50.5 s: it walked back toward the person it confirmed at 0 s "to lift them" with the
    # rescue arm off, and lost 6 s before the route failed. Without the arm a confirmed person is reported and stays put.
    # Controls: a person confirmed by nobody can still be approached for a closer look; with the arm on, the walk is allowed.
    def approach(confirmed, rescue):
        monkeypatch.setattr(T, "RESCUE_ENABLED", rescue)
        RobotAgent._new_run(request(tmp_path) | {"t": 49.0})   # start this recording first, so its reset does not clear the setup
        T.ToolRegistry._looks.clear(); T.ToolRegistry._pending.clear(); T.ToolRegistry._confirmed.clear()
        if confirmed:
            T.ToolRegistry._confirmed["robot_A"] = [{"id": "old", "x": 5.0, "y": 10.0}]
        fake_vision(monkeypatch, "person")
        out = RobotAgent(Scripted([[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 5})],
                                   [ToolCall("approach_candidate", {"evidence_id": "ev1"})]]), tmp_path / "t.jsonl").step(request(tmp_path) | {"t": 50.0})
        T.ToolRegistry._confirmed.clear()
        return out
    refused = approach(True, False)
    assert refused["orders"] == [] and "already confirmed" in refused["refusals"][0]["reason"]
    allowed = approach(False, False)
    assert [o["type"] for o in allowed["orders"]] == ["go_to"], "control: an unconfirmed person can be approached"
    assert "closer look to confirm" in allowed["orders"][0]["reason"]
    assert [o["type"] for o in approach(True, True)["orders"]] == ["go_to"], "control: with the arm on, walk to lift"


def test_a_closer_look_at_a_flat_warm_surface_is_refused(tmp_path, monkeypatch):
    # Seed 932: in the 96 s before the fire took the house, robot A spent about 37 s and robot B about 17 s walking up to
    # warm shapes on walls. An inspection that measured the warm region flat (enough depth, a large plane, no protrusion,
    # no colour person) refuses approach_candidate. Controls: a protruding shape, and too little depth to tell, are allowed.
    def approach(cues):
        RobotAgent._new_run(request(tmp_path) | {"t": 49.0})
        T.ToolRegistry._looks.clear(); T.ToolRegistry._pending.clear(); T.ToolRegistry._confirmed.clear()
        fake_vision(monkeypatch, "uncertain")
        real = T.vision.inspect_candidate
        def with_cues(*a, **k):
            r = real(*a, **k); r["cues"] = cues; return r
        monkeypatch.setattr(T.vision, "inspect_candidate", with_cues)
        return RobotAgent(Scripted([[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 3})],
                                    [ToolCall("approach_candidate", {"evidence_id": "ev1"})]]), tmp_path / "t.jsonl").step(request(tmp_path) | {"t": 50.0})
    flat = {"depth": {"valid_fraction": .9, "nonplanar": False}, "plane": {"large_plane": True}, "rgb_dnn": {"hits": []}}
    out = approach(flat)
    assert out["orders"] == [] and "flat surface" in out["refusals"][0]["reason"]
    body = {"depth": {"valid_fraction": .9, "nonplanar": True}, "plane": {"large_plane": False}, "rgb_dnn": {"hits": []}}
    assert [o["type"] for o in approach(body)["orders"]] == ["go_to"], "control: a protruding shape is approached"
    unknown = {"depth": {"valid_fraction": .3, "nonplanar": False}, "plane": {"large_plane": False}, "rgb_dnn": {"hits": []}}
    assert [o["type"] for o in approach(unknown)["orders"]] == ["go_to"], "control: too little depth to tell is approached"
    colour = {"depth": {"valid_fraction": .9, "nonplanar": False}, "plane": {"large_plane": True}, "rgb_dnn": {"hits": [{"confidence": .8}]}}
    assert [o["type"] for o in approach(colour)["orders"]] == ["go_to"], "control: a colour person is approached"


def test_air_heat_time_and_nearest_fire_reach_the_model(tmp_path, monkeypatch):
    # Held-out run 2 (seed 913), robot A: the model saw only heat_budget, which read 1.0 six seconds before it was empty,
    # while the air jumped from 44 to 375 C; it radioed "no fire seen" at 378 C. Unity now sends the robot's own air,
    # trend, heat time left and nearest measured fire; they must reach the model's context. Control: a field the
    # whitelist does not name never does.
    fake_vision(monkeypatch, "uncertain")
    fire = {"bearing_deg": -20.0, "distance_m": 7.5, "burning_m2": 0.6}
    req = request(tmp_path) | {"t": 3.0, "event": {"kind": "periodic"}, "air_c": 96.4, "air_trend_c_per_min": 140.0,
                               "heat_seconds_left": 9.1, "nearest_fire": fire, "fire_near_robot": None, "not_a_field": 1}
    RobotAgent(Scripted([[ToolCall("finish", {"summary": "ok"})]]), tmp_path / "t.jsonl").step(req)
    ctx = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines() if '"step_start"' in l][-1]["context"]
    assert ctx["air_c"] == 96.4 and ctx["air_trend_c_per_min"] == 140.0 and ctx["heat_seconds_left"] == 9.1
    assert ctx["nearest_fire"] == fire and "fire_near_robot" in ctx
    assert "not_a_field" not in ctx, "control: unnamed fields are not passed"


def test_close_looks_count_per_shape_not_per_grid_square_and_a_twice_looked_shape_can_be_dropped(tmp_path, monkeypatch):
    # Held-out run 2, robot A: 5 close looks at one blob from 80 to 108 s; its measured position wandered over three
    # 1.5 m squares, so the two-look limit never fired, and only not_person could reject it. Now looks within 1.0 m
    # (survivor_merge_m) are one shape: the third from the same side is refused, and reject_candidate may cite the
    # uncertain look. Controls: a shape 2 m away is a new shape; a third look from 90 deg round is allowed; one look is
    # not enough to drop an uncertain shape.
    fake_vision(monkeypatch, "uncertain")

    def look(t, rng, pose=(5.0, 5.0, 0.0), bearing=0.0, then="approach_candidate"):
        req = request(tmp_path) | {"t": t, "pose": {"x": pose[0], "y": pose[1], "heading_deg": pose[2]}}
        second = (ToolCall("approach_candidate", {"evidence_id": "ev1"}) if then == "approach_candidate" else
                  ToolCall("reject_candidate", {"x": pose[0], "y": pose[1] + rng, "evidence_id": "ev1"}))
        return RobotAgent(Scripted([[ToolCall("inspect_candidate", {"bearing_deg": bearing, "range_m": rng})], [second]]),
                          tmp_path / "t.jsonl").step(req)

    def fresh():
        T.ToolRegistry._looks.clear(); T.ToolRegistry._pending.clear()

    fresh()
    for k, rng in enumerate((4.45, 5.0)):                     # shape at y 9.45 then 10.0: grid squares 6 and 7
        assert look(10.0 + k, rng)["orders"], "first two looks go ahead"
    third = look(12.0, 5.3)                                   # y 10.3: within 1 m of both
    assert third["orders"] == [] and "already looked closer 2 times at this shape" in third["refusals"][0]["reason"]
    assert look(13.0, 5.3, then="reject")["orders"][0]["type"] == "reject_candidate", "a twice-looked shape can be dropped"

    fresh()
    look(10.0, 5.0); look(11.0, 5.0)
    assert look(12.0, 7.0)["orders"], "control: a shape 2 m further on is a new shape"
    fresh()
    look(10.0, 5.0); look(11.0, 5.0)
    assert look(12.0, 5.0, pose=(10.0, 10.0, -90.0), bearing=0.0)["orders"], "control: a third look from 90 deg round"
    fresh()
    look(10.0, 5.0)
    out = look(11.0, 5.0, then="reject")
    assert out["orders"] == [] and out["refusals"][0]["tool"] == "reject_candidate", "control: one look cannot drop it"


def test_a_five_round_step_reaches_finish_and_a_cut_step_still_logs_its_decision(tmp_path, monkeypatch):
    # Held-out runs 1 and 2: with 4 rounds, scan, inspect, act and a separate finish filled the limit; 41 of 84 steps
    # lost their summary and 2 a decision. Control: the old 4-round limit cuts the same step before finish, and the
    # orders it gave are still named in the summary.
    fake_vision(monkeypatch, "uncertain")
    T.ToolRegistry._looks.clear(); T.ToolRegistry._pending.clear()
    turns = lambda: [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 5})], [ToolCall("approach_candidate", {"evidence_id": "ev1"})],
                     [ToolCall("send_team_message", {"text": "warm shape ahead, looking closer"})],
                     [ToolCall("send_team_message", {"text": "still looking"})], [ToolCall("finish", {"summary": "closer look ordered"})]]
    out = RobotAgent(Scripted(turns()), tmp_path / "t.jsonl").step(request(tmp_path))
    assert out["summary"] == "closer look ordered" and out["rounds"] == 5
    T.ToolRegistry._looks.clear(); T.ToolRegistry._pending.clear()
    cut = RobotAgent(Scripted(turns()), tmp_path / "t.jsonl", max_rounds=4).step(request(tmp_path) | {"t": 40.0})
    assert cut["rounds"] == 4 and cut["summary"].startswith("(no summary") and "go_to" in cut["summary"], "control: 4 rounds"


def test_a_warm_surface_measured_flat_can_be_rejected(tmp_path, monkeypatch):
    # Seed 934, robot A: approach_candidate refused a flat warm surface with "reject it with reject_candidate", and
    # reject_candidate then refused because the verdict was only uncertain; the robot looped 4 times, a full step each.
    # Both now share _flat_warm_surface. Control: a protruding (non-flat) uncertain shape still cannot be rejected on one look.
    def reject(cues):
        RobotAgent._new_run(request(tmp_path) | {"t": 49.0})
        T.ToolRegistry._looks.clear(); T.ToolRegistry._pending.clear(); T.ToolRegistry._confirmed.clear()
        fake_vision(monkeypatch, "uncertain")
        real = T.vision.inspect_candidate
        def with_cues(*a, **k):
            r = real(*a, **k); r["cues"] = cues; return r
        monkeypatch.setattr(T.vision, "inspect_candidate", with_cues)
        return RobotAgent(Scripted([[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 3})],
                                    [ToolCall("reject_candidate", {"x": 5.0, "y": 8.0, "evidence_id": "ev1"})]]), tmp_path / "t.jsonl").step(request(tmp_path) | {"t": 50.0})
    flat = {"depth": {"valid_fraction": .9, "nonplanar": False}, "plane": {"large_plane": True}, "rgb_dnn": {"hits": []}}
    assert [o["type"] for o in reject(flat)["orders"]] == ["reject_candidate"], "a flat warm surface is rejected"
    body = {"depth": {"valid_fraction": .9, "nonplanar": True}, "plane": {"large_plane": False}, "rgb_dnn": {"hits": []}}
    out = reject(body)
    assert out["orders"] == [] and out["refusals"][0]["tool"] == "reject_candidate", "control: a protruding shape is not"


def test_a_person_already_confirmed_nearby_is_not_confirmed_again(tmp_path, monkeypatch):
    # Seed 934, robot A confirmed the study survivor twice (18.8 s and 193.6 s, 0.46 m apart) and radioed a new survivor.
    # Within survivor_merge_m of this robot's or the team's confirmations, confirm_survivor is refused. Control: a person
    # 3 m from the known one is confirmed.
    def confirm(survivors, rng):
        RobotAgent._new_run(request(tmp_path) | {"t": 49.0})
        T.ToolRegistry._looks.clear(); T.ToolRegistry._pending.clear(); T.ToolRegistry._confirmed.clear()
        fake_vision(monkeypatch, "person")
        return RobotAgent(Scripted([[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": rng})],
                                    [ToolCall("confirm_survivor", {"x": 5.0, "y": 5.0 + rng, "evidence_id": "ev1"})]]),
                          tmp_path / "t.jsonl").step(request(tmp_path) | {"t": 50.0, "survivors": survivors})
    dup = confirm([[5.3, 8.2]], 3.0)
    assert dup["orders"] == [] and "already confirmed" in dup["refusals"][0]["reason"]
    assert [o["type"] for o in confirm([[5.0, 11.0]], 3.0)["orders"]] == ["confirm_survivor"], "control: 3 m away is someone else"


def test_the_step_scan_is_in_the_first_rounds_context_and_citable(tmp_path, monkeypatch):
    # Seed 934: 62 of 72 steps spent their first round (about 3 s) on scan_thermal; steps averaged 15 s. Code now runs the
    # scan before round 1, so its id is evidence from the start. Control: a frame that cannot be scanned leaves it out
    # and the step still finishes.
    fake_vision(monkeypatch, "uncertain")
    monkeypatch.setattr(T.vision, "scan_thermal", lambda f: {"id": "scan1", "tool": "scan_thermal", "hotspots": [], "people": []})
    out = RobotAgent(Scripted([[ToolCall("finish", {"summary": "nothing warm"})]]), tmp_path / "t.jsonl").step(request(tmp_path))
    start = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines() if '"step_start"' in l][-1]
    assert start["context"]["scan_thermal"]["id"] == "scan1" and "scan1" in out["evidence_ids"] and out["rounds"] == 1
    def broken(f): raise AttributeError("no frame")
    monkeypatch.setattr(T.vision, "scan_thermal", broken)
    out = RobotAgent(Scripted([[ToolCall("finish", {"summary": "no frame"})]]), tmp_path / "u.jsonl").step(request(tmp_path) | {"t": 20.0})
    start = [json.loads(l) for l in (tmp_path / "u.jsonl").read_text().splitlines() if '"step_start"' in l][-1]
    assert "scan_thermal" not in start["context"] and out["summary"] == "no frame", "control: no scan, step still runs"


def test_confirmed_people_are_marked_for_the_model_with_bearing_and_distance(tmp_path, monkeypatch):
    # Seed 934: both robots confirmed one person twice; the context held only raw survivor coordinates. Each confirmed
    # person is now listed with bearing, distance and in_view. Control: with nobody confirmed, no list.
    fake_vision(monkeypatch, "uncertain")
    RobotAgent(Scripted([[ToolCall("finish", {"summary": "ok"})]]), tmp_path / "t.jsonl").step(request(tmp_path) | {"survivors": [[5.0, 8.0]]})
    ctx = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines() if '"step_start"' in l][-1]["context"]
    (p,) = ctx["confirmed_people"]
    assert (p["x"], p["y"], p["distance_m"], p["bearing_deg"], p["in_view"]) == (5.0, 8.0, 3.0, 0.0, True)
    RobotAgent(Scripted([[ToolCall("finish", {"summary": "ok"})]]), tmp_path / "u.jsonl").step(request(tmp_path) | {"t": 30.0})
    ctx = [json.loads(l) for l in (tmp_path / "u.jsonl").read_text().splitlines() if '"step_start"' in l][-1]["context"]
    assert "confirmed_people" not in ctx, "control: nobody confirmed"


def test_an_explore_order_for_the_goal_the_robot_already_has_sends_no_order(tmp_path, monkeypatch):
    # Seed 934, robot B: 11 of 15 explore orders named the brain's own first frontier; each made a fresh goal and a turn.
    # An explore order within 1 m of the robot's current goal now sends nothing. Control: a frontier 4 m from the goal
    # is ordered as before.
    fake_vision(monkeypatch, "uncertain")
    T.ToolRegistry._pending.clear(); T.ToolRegistry._looks.clear()
    frontiers = [{"id": 0, "x": 5.0, "y": 9.0, "path_m": 4.0, "unsearched_m2": 6.0, "proven_indoors": True},
                 {"id": 1, "x": 9.0, "y": 9.0, "path_m": 5.0, "unsearched_m2": 6.0, "proven_indoors": True}]
    def explore(fid, t, kind):
        req = request(tmp_path) | {"t": t, "event": {"kind": kind}, "frontiers": frontiers, "goal": [5.2, 9.1]}
        return RobotAgent(Scripted([[ToolCall("explore_frontier", {"frontier_id": fid, "reason": "search on"})]]), tmp_path / "t.jsonl").step(req)
    same = explore(0, 10.0, "area_searched")
    assert same["orders"] == [] and same["refusals"] == [], "already heading there: nothing sent, nothing refused"
    other = explore(1, 20.0, "agent_goal_reached")   # a new event, so the step is not skipped as a repeat
    assert [o["type"] for o in other["orders"]] == ["explore_frontier"], "control: a different frontier is ordered"


def test_an_approach_to_a_deferred_wall_patch_does_not_hold_exploring_back(tmp_path, monkeypatch):
    # Seed 934 (24 Oct), robot A at 145 to 167 s: approach_candidate was refused for a thermal-only patch on a long wall
    # ("search the offered frontiers first") and explore_frontier for the approach already under way to it ("finish that
    # first"), six steps running. The approach to a deferred wall patch is dropped, kept among the unfinished sightings,
    # and exploring is allowed. Control: an approach to a patch that is not on a wall still holds exploring back.
    fake_vision(monkeypatch, "uncertain")
    T.ToolRegistry._pending.clear(); T.ToolRegistry._looks.clear(); T.ToolRegistry._wall_spots.clear(); T.ToolRegistry._unfinished.clear()
    frontier = [{"id": 1, "x": 9.0, "y": 1.0, "path_m": 6.0, "unsearched_m2": 6.0, "proven_indoors": True}]
    def step(t, numbers, calls):
        RobotAgent._recent.clear()
        req = request(tmp_path) | {"t": t, "event": {"kind": "survivor_candidate", "numbers": numbers}, "frontiers": frontier}
        return RobotAgent(Scripted(calls), tmp_path / f"{t}.jsonl").step(req)
    approach = [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 5})], [ToolCall("approach_candidate", {"evidence_id": "ev1"})]]
    explore = [[ToolCall("explore_frontier", {"frontier_id": 1, "reason": "search on"})]]
    patch = {"x_m": 5.0, "y_m": 10.0, "on_wall": 0.0, "rgb": 0.0}
    assert [o["type"] for o in step(40.0, patch, approach)["orders"]] == ["go_to"]
    held = step(41.0, patch | {"x_m": 5.1}, explore)
    assert held["orders"] == [] and "finish that first" in held["refusals"][0]["reason"], "control: not on a wall"
    freed = step(42.0, patch | {"on_wall": 1.0}, explore)
    assert [o["type"] for o in freed["orders"]] == ["explore_frontier"] and freed["refusals"] == [], freed
    assert T.ToolRegistry._pending.get("robot_A") is None
    assert any("long wall" in u["dropped_because"] for u in T.ToolRegistry._unfinished["robot_A"])
