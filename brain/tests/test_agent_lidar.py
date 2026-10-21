"""Range fallback controls use synthetic recordings with the real LiDAR file format."""
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from brain.agent import vision as v

ROOT = Path(__file__).resolve().parents[2]


def recording(folder, *, scan_m=2., real_depth=False, curved=True, stamp=0., bearing=0.):
    folder.mkdir(parents=True)
    calibration = ('%YAML:1.0\n---\nCamera1.fx: 120.0\nCamera1.fy: 120.0\n'
                   'Camera1.cx: 79.5\nCamera1.cy: 59.5\nCamera.width: 160\n'
                   'Camera.height: 120\nCamera.fps: 15\nRGBD.DepthMapFactor: 1000.0\n'
                   'T_rgb_thermal.translation_m: [0.0, 0.0, 0.0]\n')
    for name in ('camera.yaml', 'thermal.yaml'):
        (folder / name).write_text(calibration)
    yy, xx = np.indices((120, 160))
    centre = int(round(79.5 + 120 * math.tan(math.radians(bearing))))
    mask = (yy >= 30) & (yy < 90) & (xx >= centre - 11) & (xx < centre + 11)
    thermal = np.full(mask.shape, 2932, np.uint16)
    thermal[mask] = np.round((303.15 + 5 * (yy[mask] - 30) / 60) * 10).astype(np.uint16)
    # A rounded body's shoulders return farther than its centre at 2 m.
    angles = -math.pi + np.arange(720) * (2 * math.pi / 720)
    distance_from_centre = angles + math.radians(bearing)
    ranges = scan_m + .25 * np.minimum((distance_from_centre / .1) ** 2, 1) if curved else np.full(720, scan_m)
    header = ('# planar LiDAR, ranges in millimetres, 0 = no return\n'
              '# beams 720 angle_min_rad -3.141592653589793 increment_rad 0.008726646259971648 '
              'mount_height_m 0.45 mount_forward_m 0.0 min_m 0.15 max_m 12.0\n')
    (folder / 'lidar.txt').write_text(header + f'{stamp} ' + ' '.join(map(str, np.rint(ranges * 1000).astype(int))) + '\n')
    depth = np.zeros(mask.shape, np.uint16)
    if real_depth:
        # Optical z for the same rounded body, derived from pitched camera rays.
        x = (xx - 79.5) / 120
        forward = math.cos(math.radians(10)) - (yy - 59.5) / 120 * math.sin(math.radians(10))
        a = np.arctan2(-x, forward)
        beam = np.rint((a + math.pi) / (2 * math.pi / 720)).astype(int) % 720
        z = (ranges[beam] - .25 * np.cos(a)) / np.hypot(x, forward)
        depth[mask] = np.rint(z[mask] * 1000).astype(np.uint16)
    images = {'rgb': np.full((120, 160, 3), 70, np.uint8), 'thermal': thermal, 'depth': depth}
    for sensor, image in images.items():
        (folder / sensor).mkdir()
        assert cv2.imwrite(str(folder / sensor / '0.png'), image)
        (folder / f'{sensor}.txt').write_text(f'0 {sensor}/0.png\n')
    return v.load_frame(folder, 0.)


def inspect(frame, folder, bearing=0.):
    return v.inspect_candidate(frame, bearing, 2., output_dir=folder)


def test_lidar_body_at_two_metres_matches_registered_depth(tmp_path):
    measured = inspect(recording(tmp_path / 'depth', real_depth=True), tmp_path / 'debug')
    fallback = inspect(recording(tmp_path / 'lidar'), tmp_path / 'debug')
    assert fallback['verdict'] == measured['verdict'] == 'person'
    assert fallback['cues']['range_source'] == 'lidar'
    assert fallback['cues']['lidar']['beam_range_m'] == pytest.approx(2., abs=.1)
    assert fallback['candidate']['range_m'] == pytest.approx(measured['candidate']['range_m'], abs=.002)
    assert any('LiDAR' in reason for reason in fallback['reasons'])
    assert np.isnan(v._depth(v.load_frame(tmp_path / 'lidar', 0.))).all()


def test_missing_lidar_preserves_uncertain_and_depth_provenance(tmp_path):
    recording(tmp_path / 'missing')
    (tmp_path / 'missing/lidar.txt').unlink()
    v.load_frame.cache_clear()
    result = inspect(v.load_frame(tmp_path / 'missing', 0.), tmp_path / 'debug')
    assert result['verdict'] == 'uncertain'
    assert 'Too little registered depth to judge shape.' in result['reasons']
    assert result['cues']['range_source'] == 'depth'


def test_wall_behind_low_person_is_rejected_by_size(tmp_path):
    result = inspect(recording(tmp_path / 'wall', scan_m=7.), tmp_path / 'debug')
    assert result['verdict'] == 'not_person'
    assert result['cues']['range_source'] == 'lidar'
    assert not result['cues']['size']['plausible']
    assert any('body-size bounds' in reason for reason in result['reasons'])


def test_valid_registered_depth_ignores_conflicting_lidar(tmp_path):
    frame = recording(tmp_path / 'measured', real_depth=True)
    before = inspect(frame, tmp_path / 'debug')
    path = tmp_path / 'measured/lidar.txt'
    lines = path.read_text().splitlines()
    lines[-1] = '0 ' + ' '.join(['7000'] * 720)
    path.write_text('\n'.join(lines) + '\n')
    v.load_frame.cache_clear()
    after = inspect(v.load_frame(tmp_path / 'measured', 0.), tmp_path / 'debug')
    assert before['verdict'] == after['verdict'] == 'person'
    assert before['cues'] == after['cues']
    assert after['cues']['range_source'] == 'depth'
    assert 'lidar' not in after['cues']


@pytest.mark.parametrize('bearing', [-20., 20.])
def test_off_axis_lidar_uses_left_positive_beams_and_pitch(tmp_path, bearing):
    frame = recording(tmp_path / 'off_axis', bearing=bearing)
    result = inspect(frame, tmp_path / 'debug', bearing)
    assert result['verdict'] == 'person'
    assert result['cues']['range_source'] == 'lidar'
    assert result['cues']['lidar']['beam_range_m'] < 2.2


def test_flat_scan_does_not_bypass_nonplanar_shape_check(tmp_path):
    result = inspect(recording(tmp_path / 'flat', curved=False), tmp_path / 'debug')
    assert result['verdict'] == 'uncertain'
    assert result['cues']['range_source'] == 'lidar'
    assert not result['cues']['depth']['nonplanar']


@pytest.mark.parametrize('stamp', [-.500001, .500001])
def test_stale_scan_cannot_supply_range(tmp_path, stamp):
    result = inspect(recording(tmp_path / 'stale', stamp=stamp), tmp_path / 'debug')
    assert result['verdict'] == 'uncertain'
    assert result['cues']['range_source'] == 'depth'
    assert 'lidar' not in result['cues']


def test_nearest_scan_ties_earlier_and_refreshes_after_append(tmp_path):
    folder = tmp_path / 'nearest'
    recording(folder, stamp=-.5)
    path = folder / 'lidar.txt'
    first = path.read_text()
    path.write_text(first + '.5 ' + ' '.join(['7000'] * 720) + '\n')
    v.load_frame.cache_clear()
    earlier = v.load_frame(folder, 0.)
    assert inspect(earlier, tmp_path / 'debug')['verdict'] == 'person'
    assert earlier['stamps']['lidar'] == -.5
    path.write_text(first + '.1 ' + ' '.join(['7000'] * 720) + '\n')
    v.load_frame.cache_clear()
    closer = v.load_frame(folder, 0.)
    assert inspect(closer, tmp_path / 'debug')['verdict'] == 'not_person'
    assert closer['stamps']['lidar'] == .1
    assert closer['id'] != earlier['id']


@pytest.mark.parametrize('range_m', [0., .149, 12.001])
def test_missing_or_out_of_sensor_range_beams_are_unusable(tmp_path, range_m):
    result = inspect(recording(tmp_path / 'invalid', scan_m=range_m, curved=False), tmp_path / 'debug')
    assert result['verdict'] == 'uncertain'
    assert result['cues']['range_source'] == 'depth'
    assert 'lidar' not in result['cues']


def test_sparse_real_depth_is_preserved_during_fill(tmp_path):
    frame = recording(tmp_path / 'sparse')
    # Keep one measured column at a different range from the LiDAR.
    depth = v._depth(frame).copy()
    depth[30:90, 80] = 1.5
    frame._registered = depth
    mask = frame.thermal > 3000
    filled, cue = v._lidar_component_depth(frame, mask, depth, .5)
    np.testing.assert_array_equal(filled[30:90, 80], depth[30:90, 80])
    assert np.isnan(filled[~mask]).all()
    # At pixel (79, 30), the pitched ray has horizontal length 1.027505.
    # Its 2 m beam and 0.25 m forward correction give optical z = 1.703158 m.
    assert filled[30, 79] == pytest.approx(1.703158, abs=.00001)
    assert cue['filled_pixels'] == int(mask.sum()) - 60
    assert inspect(frame, tmp_path / 'debug')['cues']['range_source'] == 'lidar'


def test_recorded_visible_survivor_with_sparse_registered_depth(tmp_path):
    from brain.agent.eval_vision import match_component, survivor_visibility
    from brain.evaluator import gt_pose_at, load_floorplan, load_survivors, load_trajectory

    run = ROOT / 'data/recordings/unity_house_hidden_run1'
    if not (run / 'robot_B/lidar.txt').exists():
        pytest.skip('local hidden-house recording absent')
    frame = v.load_frame(run / 'robot_B', 59.)
    survivor = next(s for s in load_survivors(run) if s['id'] == 's3')
    visibility = survivor_visibility(frame, gt_pose_at(load_trajectory(run, 'robot_B'), 59.),
                                     survivor, load_floorplan(run))
    assert visibility['visible']
    result = v.inspect_candidate(frame, -2.896164, 7.717141, output_dir=tmp_path)
    assert match_component(result['candidate'], visibility)
    assert result['cues']['depth']['registered_fraction'] == pytest.approx(.242857, abs=.000001)
    assert result['cues']['depth']['valid_fraction'] == 1.
    assert result['cues']['range_source'] == 'lidar'
    assert result['cues']['lidar']['filled_pixels'] == 477
    assert result['cues']['lidar']['t'] == pytest.approx(59.001667)
    # A real person behind the desk remains rejected by the existing recess check.
    # This recording predates severe frame-wide smoke dropout; it tests real sparse registration.
    assert result['verdict'] == 'not_person'
    assert result['cues']['window']['opening']
