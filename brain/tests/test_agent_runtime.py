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
    assert T.APPROACH_STANDOFF_M < float(T._RESCUE["reach_m"])


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
    fake_vision(monkeypatch, "person")
    monkeypatch.setattr(T.vision, "inspect_candidate", lambda f, b, r, output_dir: {
        "id": f"ev{r:g}", "tool": "inspect_candidate", "verdict": "person", "candidate": {"bearing_deg": b, "range_m": r}})

    def extract(range_m, carrying=False):
        extract.t += 30
        agent = RobotAgent(Scripted([[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": range_m})],
                                     [ToolCall("extract_survivor", {"evidence_id": f"ev{range_m:g}"})]]), tmp_path / f"x{extract.t}.jsonl")
        return agent.step({**request(tmp_path), "t": extract.t, "carrying_person": carrying})
    extract.t = 0
    assert [o["type"] for o in extract(1.2)["orders"]] == ["extract_survivor"]
    assert extract(3.0)["refusals"][0]["tool"] == "extract_survivor", "too far: approach first"
    assert "already towing" in extract(1.2, carrying=True)["refusals"][0]["reason"]



class Recording(Scripted):
    """Scripted backend that keeps the messages it was sent, so a test can read what the model was told."""
    def __init__(self, turns):
        super().__init__(turns); self.seen = []

    def chat(self, messages, tools):
        self.seen.append([dict(m) for m in messages]); return super().chat(messages, tools)


def test_a_confirmation_says_how_to_get_the_person_out(tmp_path, monkeypatch):
    # Run 15: robot A confirmed a person 2.7 m away and searched on. The confirm result names the next step: an approach
    # when the inspection was beyond the arm's reach, a lift when it was within it.
    def told(range_m):
        monkeypatch.setattr(T.vision, "load_frame", lambda d, t: object())
        monkeypatch.setattr(T.vision, "inspect_candidate", lambda f, b, r, output_dir: {
            "id": "ev1", "tool": "inspect_candidate", "verdict": "person", "candidate": {"range_m": range_m}})
        backend = Recording([[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": range_m})],
                             [ToolCall("confirm_survivor", {"x": 5, "y": 5 + range_m, "evidence_id": "ev1"})]])
        out = RobotAgent(backend, tmp_path / f"{range_m}.jsonl").step(request(tmp_path))
        assert [o["type"] for o in out["orders"]] == ["confirm_survivor"]
        return [m for m in backend.seen[-1] if m.get("role") == "tool" and m.get("name") == "confirm_survivor"][-1]["content"]
    assert "approach_candidate" in told(2.7) and "extract_survivor" not in told(2.7)
    assert "extract_survivor" in told(1.2)


def test_confirmed_people_not_yet_lifted_are_listed_with_bearing_and_approach():
    # Run 19: robot A confirmed a person, a heat alarm interrupted the approach, and nothing brought it back.
    T.ToolRegistry._lifted.clear()
    pose = {"x": 0.0, "y": 0.0, "heading_deg": 90.0}          # facing +x
    out = T.survivors_to_rescue("robot_A", pose, [[3.0, 0.0], [0.0, -5.0]])
    assert [s["x"] for s in out] == [3.0, 0.0]                 # nearest first
    assert out[0]["bearing_deg"] == pytest.approx(0.0) and out[0]["distance_m"] == pytest.approx(3.0)
    assert out[0]["approach"]["x"] == pytest.approx(3.0 - T.APPROACH_STANDOFF_M)
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


def test_approach_candidate_faces_the_inspected_spot_on_arrival(tmp_path, monkeypatch):
    fake_vision(monkeypatch, "uncertain")
    T.ToolRegistry._looks.clear()
    out, _ = run(tmp_path, [[ToolCall("inspect_candidate", {"bearing_deg": 0, "range_m": 5})],
                            [ToolCall("approach_candidate", {"evidence_id": "ev1"})]])
    (o,) = out["orders"]
    assert (o["face_x"], o["face_y"]) == pytest.approx((5.0, 10.0))


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
