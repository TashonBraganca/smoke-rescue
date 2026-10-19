import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import json

from brain.evaluator import compute_ate, evaluate_run, validate_ate
from brain.oracle import load_trajectory


@pytest.fixture
def ground_truth(tmp_path):
    t = np.arange(400) / 30
    rows = np.column_stack((t, np.sin(t), np.cos(t), t / 5,
                            np.zeros((400, 3)), np.ones(400)))
    folder = tmp_path / "oracle"
    folder.mkdir()
    np.savetxt(folder / "robot_A_groundtruth.txt", rows)
    return load_trajectory(tmp_path, "robot_A"), rows


def test_ground_truth_as_estimate_has_near_zero_ate(ground_truth):
    truth, rows = ground_truth
    result = compute_ate(truth, rows)
    assert result["ate_rmse_m"] < 1e-12
    assert result["matched_poses"] == 400


def test_zigzag_ten_centimetres_survives_se3_alignment(ground_truth):
    truth, rows = ground_truth
    rows[:, 1] += np.where(np.arange(len(rows)) % 2, .1, -.1)
    rotation = Rotation.from_euler("xyz", [20, 10, 30], degrees=True)
    rows[:, 1:4] = rotation.apply(rows[:, 1:4]) + [4, 2, 1]
    rows[:, 4:8] = rotation.as_quat()
    result = compute_ate(truth, rows)
    assert .08 < result["ate_rmse_m"] < .12
    assert result["matched_poses"] == 400
    assert result["alignment_scale"] == 1


def test_timestamp_association_uses_twenty_milliseconds(ground_truth):
    truth, rows = ground_truth
    rows[:, 0] += .019
    assert compute_ate(truth, rows)["matched_poses"] > 390
    rows[:, 0] += 100
    with pytest.raises(ValueError, match="100"):
        compute_ate(truth, rows)


def test_se3_alignment_does_not_correct_scale(ground_truth):
    truth, rows = ground_truth
    rows[:, 1:4] *= 2
    assert compute_ate(truth, rows)["ate_rmse_m"] > .2


@pytest.mark.parametrize("error,count", [(0, 400), (float("nan"), 400),
                                        (float("inf"), 400), (.1, 99)])
def test_run_sanity_gates(error, count):
    with pytest.raises(ValueError):
        validate_ate({"ate_rmse_m": error, "matched_poses": count})


def test_run_report_preserves_per_robot_contract_and_saves_evidence(ground_truth, tmp_path):
    truth, rows = ground_truth
    recording = tmp_path
    (recording / "meta.json").write_text(json.dumps({"run_id": "test", "robots": ["robot_A"]}))
    robot = recording / "robot_A"
    robot.mkdir()
    (robot / "camera.yaml").write_text(
        "Camera.fx: 2\nCamera.fy: 2\nCamera.cx: 0\nCamera.cy: 0\n"
        "Camera.width: 2\nCamera.height: 2\nCamera.fps: 30\nDepthMapFactor: 1000\n"
    )
    (robot / "associations.txt").write_text("".join(
        f"{t:.9f} rgb/{t:.9f}.png {t:.9f} depth/{t:.9f}.png\n" for t in rows[:, 0]))
    output = tmp_path / "outputs"
    (output / "robot_A").mkdir(parents=True)
    rows[:, 1] += np.where(np.arange(len(rows)) % 2, .1, -.1)
    np.savetxt(output / "robot_A/CameraTrajectory.txt", rows)
    np.savetxt(output / "robot_A/KeyFrameTrajectory.txt", rows[::10])
    report = evaluate_run(recording, output)
    assert set(report["per_robot"]["robot_A"]) == {
        "ate_rmse_m", "frames", "keyframes", "tracking_lost_s"}
    assert report["per_robot"]["robot_A"]["frames"] == 400
    assert report["per_robot"]["robot_A"]["keyframes"] == 40
    details = json.loads((output / "ate_details.json").read_text())
    assert details["robot_A"]["matched_poses"] == 400
    assert (output / "report.md").is_file()
    assert (output / "evaluation.rrd").stat().st_size > 0
