"""Container wiring checks complement the numerical calibration tests."""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from brain import slam_runner
from brain.io import RecordingReader

ROOT = Path(__file__).parents[2]


def test_inertial_runner_passes_settings_and_keeps_single_robot_mount(tmp_path, monkeypatch):
    robot = tmp_path / 'unity_clear_seed7/robot_A'
    shutil.copytree(ROOT / 'data/fixtures/tiny_2robots/robot_A', robot)
    (robot / 'imu.txt').write_text('0 0 -9.81 0 0 0 0\n')
    output = tmp_path / 'output'
    reader = RecordingReader(robot)
    poses = np.zeros((len(reader), 8))
    poses[:, 0] = [a.rgb_timestamp for a in reader.associations]
    poses[:, 7] = 1
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        np.savetxt(output / 'CameraTrajectory.txt', poses)
        np.savetxt(output / 'KeyFrameTrajectory.txt', poses[::2])
        kwargs['stdout'].write('Creation of new map with id: 0\n')

    monkeypatch.setattr(slam_runner.subprocess, 'check_output', lambda *a, **k: 'sha256:test\n')
    monkeypatch.setattr(slam_runner.subprocess, 'run', execute)
    slam_runner.run_orbslam3(robot, output, 'rgbd_inertial')
    assert commands[0][-2:] == ['rgbd_inertial', '/output/inertial.yaml']
    assert commands[0].count('--mount') == 2
    assert f'type=bind,source={robot},target=/recording,readonly' in commands[0]
    metadata = json.loads((output / 'slam_meta.json').read_text())
    assert metadata['mode'] == 'rgbd_inertial'
    assert metadata['maps_created'] == 1
    assert metadata['calibration']['T_body_camera'][1][3] == -.5


def test_missing_imu_fails_before_launching_container(tmp_path):
    robot = tmp_path / 'recording/robot_A'
    shutil.copytree(ROOT / 'data/fixtures/tiny_2robots/robot_A', robot)
    (robot / 'imu.txt').unlink(missing_ok=True)
    with pytest.raises(ValueError, match='imu.txt'):
        slam_runner.run_orbslam3(robot, tmp_path / 'output', 'rgbd_inertial')
