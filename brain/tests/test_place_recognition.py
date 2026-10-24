"""Visual place recognition: payload codec, matcher, run reset and endpoints."""
import base64
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from brain.agent import place

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "data/recordings/unity_house_hidden_run1"
A, B = str(RUN / "robot_A"), str(RUN / "robot_B")
needs_data = pytest.mark.skipif(not RUN.exists(), reason="recording missing")


def test_payload_round_trip_and_size():
    rng = np.random.default_rng(1)
    desc = rng.integers(0, 256, (10, 32), dtype=np.uint8)
    xyh = np.round(rng.uniform(-5, 5, (10, 3)), 2)
    raw = place.encode_payload(desc, xyh)
    assert raw[:3] == b"VP\x01" and len(raw) == 5 + 38 * 10
    kf = place.decode_payload(raw)
    np.testing.assert_array_equal(kf.desc, desc)
    np.testing.assert_allclose(kf.xyh, xyh, atol=1e-9)
    assert place.encode_payload(kf.desc, kf.xyh) == raw
    with pytest.raises(ValueError):
        place.decode_payload(raw[:-1])


def _truth(recording, robot, t):
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import place_eval as pe
    return pe, pe.truth_planar(recording, robot, np.array([t]), place.load_settings()["mount"])


@needs_data
def test_self_match_is_identity_and_far_negative_rejected():
    db = place.PlaceDatabase()
    first = db.keyframe(A, 100.0)
    assert first["ok"] and first["id"] == 0
    own = db.match(first["payload_b64"])
    assert own["matched"] and own["mine_id"] == 0
    assert abs(own["dx_m"]) < .02 and abs(own["dy_m"]) < .02 and abs(own["dyaw_deg"]) < .5
    # robot_B at t=5 is far from robot_A at t=100 in the recorded truth (checked below), so it must not match.
    _, (oa, _, _, _) = _truth("unity_house_hidden_run1", "robot_A", 100.0)
    _, (ob, _, _, _) = _truth("unity_house_hidden_run1", "robot_B", 5.0)
    assert np.hypot(*(oa[0] - ob[0])) > 6.0
    far = place.PlaceDatabase().keyframe(B, 5.0)
    assert far["ok"]
    assert db.match(far["payload_b64"])["matched"] is False


TONIGHT = ROOT / "data/recordings/unity_house_onedoor_seed934_brain_final_1954/robot_A"


@pytest.mark.skipif(not TONIGHT.exists(), reason="recording missing")
def test_candidates_limit_the_match_to_the_pose_prior():
    # The brain passes only own views near where the partner stood (or, for a loop closure, its own older views of
    # this place). Here view 0 is the true match: allowed, it matches; excluded, it cannot be returned; an empty
    # candidate list reports that nothing lies near the prior. Control: no candidates searches everything.
    db = place.PlaceDatabase()
    first = db.keyframe(str(TONIGHT), 100.0)
    other = db.keyframe(str(TONIGHT), 160.0)
    assert first["ok"] and other["ok"]
    assert db.match(first["payload_b64"])["mine_id"] == 0
    assert db.match(first["payload_b64"], candidates=[0])["mine_id"] == 0
    excluded = db.match(first["payload_b64"], candidates=[other["id"]])
    assert not (excluded["matched"] and excluded["mine_id"] == 0)
    empty = db.match(first["payload_b64"], candidates=[])
    assert empty["matched"] is False and "pose prior" in empty["reason"]


@needs_data
def test_real_positive_pair_recovers_true_pose():
    # heldout_run1: robot_B t=138 seen from robot_A t=160, 0.27 m apart; found by tools/place_eval.py (41 inliers).
    rec = ROOT / "data/recordings/unity_house_heldout_run1"
    mine, partner = place.PlaceDatabase(), place.PlaceDatabase()
    assert mine.keyframe(str(rec / "robot_A"), 160.0)["ok"]
    sent = partner.keyframe(str(rec / "robot_B"), 138.0)
    m = mine.match(sent["payload_b64"])
    assert m["matched"] and m["inliers"] >= 16
    pe, (oa, ha, _, _) = _truth("unity_house_heldout_run1", "robot_A", 160.0)
    _, (ob, hb, _, _) = _truth("unity_house_heldout_run1", "robot_B", 138.0)
    true = pe.true_relative(ob[0], hb[0], oa[0], ha[0])
    err = pe.pose_error((m["dx_m"], m["dy_m"], m["dyaw_deg"]), true)
    assert err[0] < 0.2 and err[1] < 3.0, (err, m, true)  # eval p95 for accepted pairs: 0.10 m, 1.5 deg


def test_too_few_features_is_ok_false(tmp_path):
    db = place.PlaceDatabase()
    flat = np.full((480, 640, 3), 90, np.uint8)
    kf, usable = place.make_keyframe(flat, np.full((480, 640), 2.0, np.float32),
                                     np.array([[415.692, 0, 319.5], [0, 415.692, 239.5], [0, 0, 1]]), db.settings)
    assert kf is None and usable < db.settings["place_min_features"]


@needs_data
def test_run_reset_clears_database():
    db = place.PlaceDatabase()
    assert db.keyframe(A, 100.0)["id"] == 0 and db.keyframe(A, 101.0)["id"] == 1
    assert db.keyframe(A, 50.0)["id"] == 0  # t went backwards: new run
    assert len(db.keyframes) == 1


@needs_data
def test_endpoints_round_trip():
    from brain.agent import server
    with TestClient(server.app) as client:
        r = client.post("/place/t_A/keyframe", json={"robot_dir": A, "t": 100.0}).json()
        assert r["ok"] and r["id"] == 0 and len(base64.b64decode(r["payload_b64"])) == 5 + 38 * r["features"]
        m = client.post("/place/t_A/match", json={"robot_dir": A, "t": 100.0, "payload_b64": r["payload_b64"]}).json()
        assert m["matched"] and abs(m["dx_m"]) < .02
        assert client.post("/place/t_A/keyframe", json={"robot_dir": A}).status_code == 400
        empty = client.post("/place/t_other/match", json={"robot_dir": B, "t": 1.0, "payload_b64": r["payload_b64"]}).json()
        assert empty["matched"] is False


def test_database_identity_changes_when_service_ids_restart(tmp_path, monkeypatch):
    settings = place.load_settings()
    rng = np.random.default_rng(2)
    kf = place.Keyframe(rng.integers(0, 256, (24, 32), dtype=np.uint8), rng.uniform(-1, 1, (24, 3)))

    class Frame:
        rgb, depth_m, K = None, None, None
        def __getitem__(self, key):
            return {"stamps": {"rgb": 10.0}}[key]

    monkeypatch.setattr(place.vision, "load_frame", lambda *_: Frame())
    monkeypatch.setattr(place, "make_keyframe", lambda *_: (kf, len(kf)))
    db = place.PlaceDatabase(settings)
    first = db.keyframe(tmp_path / "run" / "A", 10)
    second = db.keyframe(tmp_path / "run" / "A", 11)
    assert first["database_id"] == second["database_id"]
    assert first["id"] == 0 and second["id"] == 1
    assert first["t"] == 10.0
    # A backwards clock restarts service ids. A caller must not reuse the old capture pose for a fresh id 0.
    restarted = db.keyframe(tmp_path / "run" / "A", 5)
    assert restarted["id"] == 0 and restarted["database_id"] != first["database_id"]
    assert db.match(restarted["payload_b64"])["database_id"] == restarted["database_id"]
    other = place.PlaceDatabase(settings).keyframe(tmp_path / "run" / "A", 5)
    assert other["database_id"] != restarted["database_id"]
    from brain.agent import server
    monkeypatch.setattr(place, "_robot", lambda robot: db)
    with TestClient(server.app) as client:
        endpoint = client.post("/place/A/keyframe", json={"robot_dir": str(tmp_path / "run" / "A"), "t": 20}).json()
        assert endpoint["ok"] and endpoint["database_id"] == restarted["database_id"]
        assert endpoint["t"] == 10  # actual frame stamp, even when the caller requested t=20
        matched = client.post("/place/A/match", json={"robot_dir": str(tmp_path / "run" / "A"), "t": 20,
                              "payload_b64": endpoint["payload_b64"]}).json()
        assert matched["matched"] and matched["database_id"] == endpoint["database_id"]


def test_repeated_texture_descriptors_cannot_fix_a_place_pose():
    settings = place.load_settings()
    # The nearest and second-nearest Hamming distances tie on identical tiles, even with identical 3D layouts.
    desc = np.zeros((96, 32), np.uint8)
    positions = np.column_stack((np.linspace(-1, 1, 96), np.linspace(0, 4, 96), np.ones(96)))
    repeated = place.Keyframe(desc, positions)
    assert place.match_keyframes(repeated, [(0, repeated)], settings)["matched"] is False
    rng = np.random.default_rng(3)
    distinct = place.Keyframe(rng.integers(0, 256, (96, 32), dtype=np.uint8), positions)
    accepted = place.match_keyframes(distinct, [(0, distinct)], settings)
    assert accepted["matched"] and accepted["inliers"] == 96  # control: distinct descriptors at the same points


def test_place_endpoints_report_worker_failure_as_unavailable(monkeypatch, tmp_path):
    from brain.agent import server
    def down(*args):
        raise RuntimeError("place worker timed out")
    monkeypatch.setattr(server.place, "keyframe", down)
    with TestClient(server.app) as client:
        reply = client.post("/place/A/keyframe", json={"robot_dir": str(tmp_path), "t": 1})
        assert reply.status_code == 503 and "timed out" in reply.json()["detail"]
        monkeypatch.setattr(server.place, "keyframe", lambda *args: {"ok": False, "reason": "no usable image"})
        control = client.post("/place/A/keyframe", json={"robot_dir": str(tmp_path), "t": 1})
        assert control.status_code == 200 and control.json()["ok"] is False
