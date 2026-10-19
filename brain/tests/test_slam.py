from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import json
import subprocess

from brain.trajectory import match_timestamps, read_tum, tracking_summary
from brain.slam_runner import align_keyframes
from brain import slam_runner
from brain.io import RecordingReader


def poses(times):
    rows = np.zeros((len(times), 8))
    rows[:, 0] = times
    rows[:, 7] = 1
    return rows


def test_tracking_counts_missing_frames_and_uses_timestamp_intervals():
    result = tracking_summary(np.array([0, .1, .3, .4]), poses([0, .3]), 10)
    assert result["frames"] == 4
    assert result["tracked_frames"] == 2
    assert result["tracking_lost_frames"] == 2
    assert result["tracking_lost_s"] == pytest.approx(.3)


def test_tracking_raises_below_half_coverage():
    with pytest.raises(ValueError, match="50%"):
        tracking_summary(np.arange(4), poses([0]), 1)


def test_timestamps_cannot_reuse_frames_or_match_unrelated_times():
    with pytest.raises(ValueError):
        match_timestamps(np.array([0., .1]), np.array([0., 0.]))
    with pytest.raises(ValueError):
        match_timestamps(np.array([0., .1]), np.array([.05]))


def test_tum_rejects_nan_and_out_of_order_rows(tmp_path):
    path = tmp_path / "poses.txt"
    for rows in (poses([1, 0]), poses([0, float("nan")])):
        np.savetxt(path, rows)
        with pytest.raises(ValueError):
            read_tum(path)


def test_keyframes_are_rebased_to_camera_trajectory(tmp_path):
    camera = poses([0, 1, 2])
    camera[:, 1:4] = [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
    rotation = Rotation.from_euler("z", 35, degrees=True)
    keyframes = camera[[0, 2]].copy()
    keyframes[:, 1:4] = rotation.apply(keyframes[:, 1:4]) + [4, 5, 6]
    keyframes[:, 4:8] = rotation.as_quat()
    path = tmp_path / "KeyFrameTrajectory.txt"
    np.savetxt(path, keyframes)
    align_keyframes(camera, path)
    np.testing.assert_allclose(read_tum(path), camera[[0, 2]], atol=1e-6)


def test_runner_mounts_only_one_robot_readonly_and_records_results(tmp_path, monkeypatch):
    robot = Path(__file__).parents[2] / "data/fixtures/tiny_2robots/robot_A"
    output = tmp_path / "output"
    reader = RecordingReader(robot)
    trajectory = poses([a.rgb_timestamp for a in reader.associations])
    trajectory[:, 1] = np.arange(len(trajectory)) * .01
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        np.savetxt(output / "CameraTrajectory.txt", trajectory)
        np.savetxt(output / "KeyFrameTrajectory.txt", trajectory[::2])

    monkeypatch.setattr(slam_runner.subprocess, "check_output", lambda *a, **k: "sha256:test\n")
    monkeypatch.setattr(slam_runner.subprocess, "run", execute)
    paths = slam_runner.run_orbslam3(robot, output)
    assert all(path.exists() for path in paths)
    assert f"type=bind,source={robot.resolve()},target=/recording,readonly" in commands[0]
    assert commands[0].count("--mount") == 2
    meta = json.loads((output / "slam_meta.json").read_text())
    assert meta["status"] == "ok"
    assert meta["tracking_lost_frames"] == 0
    assert meta["keyframes"] == 8
    assert meta["wall_time_s"] > 0
    manifest = json.loads((output / "keyframes.json").read_text())
    assert manifest[1]["frame_index"] == 2
    with pytest.raises(FileExistsError):
        slam_runner.run_orbslam3(robot, output)


def test_runner_failure_preserves_log_and_failure_metadata(tmp_path, monkeypatch):
    robot = Path(__file__).parents[2] / "data/fixtures/tiny_2robots/robot_A"
    monkeypatch.setattr(slam_runner.subprocess, "check_output", lambda *a, **k: "sha256:test\n")

    def crash(command, **kwargs):
        kwargs["stdout"].write("simulated container crash\n")
        raise subprocess.CalledProcessError(139, command)

    monkeypatch.setattr(slam_runner.subprocess, "run", crash)
    with pytest.raises(subprocess.CalledProcessError):
        slam_runner.run_orbslam3(robot, tmp_path)
    assert "simulated container crash" in (tmp_path / "slam.log").read_text()
    assert json.loads((tmp_path / "slam_meta.json").read_text())["status"] == "failed"


def test_runner_rejects_output_ancestor_that_would_expose_other_recordings(tmp_path):
    robot = tmp_path / "recording/robot_A"
    with pytest.raises(ValueError, match="disjoint"):
        slam_runner.run_orbslam3(robot, tmp_path)


def test_tracking_states_report_loss_separately_from_exported_poses(tmp_path):
    path = tmp_path / 'tracking.csv'
    path.write_text('timestamp,state,imu_samples,imu_first,imu_last,imu_initialized\n'
                    '0,1,0,0,0,0\n0.1,2,21,0,0.1,0\n'
                    '0.2,3,20,0.105,0.2,1\n0.3,4,20,0.205,0.3,1\n')
    result = slam_runner.tracking_state_summary(path, np.array([0., .1, .2, .3]), 10)
    assert result['state_counts'] == {'1': 1, '2': 1, '3': 1, '4': 1}
    assert result['state_tracking_lost_s'] == pytest.approx(.2)
    assert result['imu_initialized'] is True
    assert result['imu_first_initialized_frame_s'] == .2


def test_crash_with_partial_tracking_row_keeps_original_error(tmp_path, monkeypatch):
    robot = Path(__file__).parents[2] / 'data/fixtures/tiny_2robots/robot_A'
    monkeypatch.setattr(slam_runner.subprocess, 'check_output', lambda *a, **k: 'sha256:test\n')

    def crash(command, **kwargs):
        (tmp_path / 'tracking.csv').write_text('timestamp,state\n0,')
        kwargs['stdout'].write('Creation of new map with id: 0\n')
        raise subprocess.CalledProcessError(125, command)

    monkeypatch.setattr(slam_runner.subprocess, 'run', crash)
    with pytest.raises(subprocess.CalledProcessError):
        slam_runner.run_orbslam3(robot, tmp_path)
    metadata = json.loads((tmp_path / 'slam_meta.json').read_text())
    assert metadata['status'] == 'failed'
    assert metadata['maps_created'] == 1
    assert '125' in metadata['error']
    assert metadata['tracking_diagnostics_error']
