"""Physical mount checks without trajectory truth."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from brain import slam_runner, pipeline

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize('height,forward', [(0.5, 0.41), (1.6, 0.25)])
def test_camera_forward_point_in_body_frame(height, forward):
    transform = slam_runner.body_camera_transform(
        {'height_m': height, 'forward_offset_m': forward, 'pitch_down_deg': 10.0})
    np.testing.assert_allclose(transform @ [0, 0, 2, 1],
                               [0, -height + .3472963553, forward + 1.9696155060, 1], atol=1e-9)
    np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-12)
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1)
    np.testing.assert_allclose(transform @ [0, 1, 0, 1],
                               [0, -height + .9848077530, forward - .1736481777, 1], atol=1e-9)


@pytest.mark.parametrize('run,height,forward', [('unity_clear_seed7', .5, .41),
                                               ('unity_clear', 1.6, .25),
                                               ('unity_haze', 1.6, .25)])
def test_generated_yaml_uses_recording_mount_and_sensor_noise(tmp_path, run, height, forward):
    robot = tmp_path / run / 'robot_A'
    robot.mkdir(parents=True)
    (robot / 'camera.yaml').write_text((ROOT / 'data/fixtures/tiny_2robots/robot_A/camera.yaml').read_text())
    output = tmp_path / 'settings.yaml'
    calibration = slam_runner.write_inertial_settings(robot, output)
    fs = cv2.FileStorage(str(output), cv2.FILE_STORAGE_READ)
    transform = fs.getNode('IMU.T_b_c1').mat()
    assert transform.dtype == np.float32
    np.testing.assert_allclose(transform[:3, 3], [0, -height, forward])
    sensor = json.loads((ROOT / 'config/sensors.json').read_text())['imu']
    for key, source in [('NoiseAcc', 'accel_noise_density'), ('NoiseGyro', 'gyro_noise_density'),
                        ('AccWalk', 'accel_bias_random_walk'), ('GyroWalk', 'gyro_bias_random_walk'),
                        ('Frequency', 'rate_hz')]:
        assert fs.getNode('IMU.' + key).real() == sensor[source]
    np.testing.assert_allclose(calibration['T_body_camera'], transform, atol=1e-12)
    fs.release()


def test_recorded_mount_overrides_current_config(tmp_path):
    robot = tmp_path / 'recording/robot_A'
    robot.mkdir(parents=True)
    (robot / 'camera.yaml').write_text((ROOT / 'data/fixtures/tiny_2robots/robot_A/camera.yaml').read_text())
    (robot.parent / 'meta.json').write_text(json.dumps({'camera': {'mount': {
        'height_m': .8, 'forward_offset_m': .3, 'pitch_down_deg': 0}}}))
    calibration = slam_runner.write_inertial_settings(robot, tmp_path / 'imu.yaml')
    np.testing.assert_allclose(np.array(calibration['T_body_camera'])[:3, 3], [0, -.8, .3])


def test_unknown_recording_requires_mount_metadata(tmp_path):
    robot = tmp_path / 'unknown/robot_A'
    robot.mkdir(parents=True)
    with pytest.raises(ValueError, match='mount'):
        slam_runner.write_inertial_settings(robot, tmp_path / 'imu.yaml')


def test_cache_changes_with_mode_and_imu(tmp_path):
    robot = tmp_path / 'unity_clear_seed7/robot_A'
    robot.mkdir(parents=True)
    for name in ('camera.yaml', 'associations.txt', 'imu.txt'):
        (robot / name).write_text('original')
    (robot / 'rgb').mkdir()
    (robot / 'depth').mkdir()
    rgbd = pipeline.input_hash(robot)
    inertial = pipeline.input_hash(robot, 'rgbd_inertial')
    assert inertial != rgbd
    (robot / 'imu.txt').write_text('changed')
    assert pipeline.input_hash(robot, 'rgbd_inertial') != inertial
    assert pipeline.input_hash(robot) == rgbd


def test_cache_changes_when_recorded_mount_changes(tmp_path):
    robot = tmp_path / 'unity_clear_seed7/robot_A'
    robot.mkdir(parents=True)
    for name in ('camera.yaml', 'associations.txt', 'imu.txt'):
        (robot / name).write_text('same')
    original = pipeline.input_hash(robot, 'rgbd_inertial')
    (robot.parent / 'meta.json').write_text(json.dumps({'camera': {'mount': {
        'height_m': 1.6, 'forward_offset_m': .25, 'pitch_down_deg': 10}}}))
    assert pipeline.input_hash(robot, 'rgbd_inertial') != original
