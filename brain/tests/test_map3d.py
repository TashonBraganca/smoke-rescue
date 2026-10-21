"""Depth-only map behavior, native safety, and offline accuracy evaluation."""
import base64
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
RECORDING = ROOT / 'data/recordings/unity_house_hidden_run1'
SENSORS = json.loads((ROOT / 'config/sensors.json').read_text())


def config():
    return json.loads((ROOT / 'config/map3d.json').read_text())


def new_map(**kwargs):
    from brain.agent.map3d import RobotMap3D
    return RobotMap3D(**kwargs)


@pytest.fixture
def scene(tmp_path):
    """An independently ray-traced 0.8 m box and floor in the initial robot frame."""
    cam = SENSORS['camera']
    h, w = cam['height'], cam['width']
    v, u = np.indices((h, w))
    px = (u - cam['cx']) / cam['fx']
    py = (v - cam['cy']) / cam['fy']
    p = math.radians(cam['mount']['pitch_down_deg'])
    rays = np.stack((px, math.cos(p) - py * math.sin(p),
                     -math.sin(p) - py * math.cos(p)), axis=-1)
    centre = np.array([0., cam['mount']['forward_offset_m'], cam['mount']['height_m']])
    with np.errstate(divide='ignore', invalid='ignore'):
        floor = -centre[2] / rays[..., 2]
        a = (np.array([.3, 2., 0.]) - centre) / rays
        b = (np.array([1.3, 3., .8]) - centre) / rays
    entry = np.minimum(a, b).max(axis=-1)
    exit = np.maximum(a, b).min(axis=-1)
    box = np.where((entry > 0) & (entry <= exit), entry, np.inf)
    depth = np.minimum(np.where(floor > 0, floor, np.inf), box)
    depth[(depth < SENSORS['depth']['min_m']) | (depth > SENSORS['depth']['max_m'])] = 0
    depth = np.rint(depth * 1000).astype(np.uint16)
    folder = tmp_path / 'run1/robot_B'
    write_depth(folder, depth, stamps=(0., 1., 2.))
    return folder, depth


def write_depth(folder, depth, stamps=(0.,)):
    (folder / 'depth').mkdir(parents=True)
    cam = SENSORS['camera']
    (folder / 'camera.yaml').write_text('%YAML:1.0\n---\n' + '\n'.join(
        f'{k}: {v}' for k, v in {'Camera1.fx': cam['fx'], 'Camera1.fy': cam['fy'],
            'Camera1.cx': cam['cx'], 'Camera1.cy': cam['cy'], 'Camera.width': depth.shape[1],
            'Camera.height': depth.shape[0], 'Camera.fps': cam['rate_hz'],
            'RGBD.DepthMapFactor': 1000.}.items()))
    rows = []
    for t in stamps:
        name = f'depth/{t:.6f}.png'
        assert cv2.imwrite(str(folder / name), depth)
        rows.append(f'{t:.6f} {name}')
    (folder / 'depth.txt').write_text('\n'.join(rows) + '\n')


ZERO = dict(x=0., y=0., heading_deg=0.)


def test_box_top_floor_and_optical_mount(scene):
    folder, _ = scene
    with new_map() as m:
        empty, unseen = m.layers(.1, [-4., -1.], [80, 70])
        assert empty.dtype == np.dtype('<u2') and not empty.any() and not unseen.any()
        m.integrate(folder, 0., ZERO)
        top, seen = m.layers(.1, [-4., -1.], [80, 70])
        # Interior of the visible box top, x=0.5..1.1, y=2.2..2.6.
        assert np.count_nonzero(top[32:36, 45:51]) >= 15
        assert np.max(np.abs(top[32:36, 45:51].astype(int) - 80)) <= 10
        assert not top[30:40, 25:35].any(), 'flat floor is not an obstacle'
        assert seen[33, 47] and seen[22, 42], 'both surface and free columns are observed'
        assert not seen[2, 2], 'cells outside the observed rays remain unknown'
        points, normals = m.points_normals()
        assert len(points) > 100 and points.shape == normals.shape
        assert np.isfinite(points).all() and np.isfinite(normals).all()
        assert np.allclose(np.linalg.norm(normals, axis=1), 1., atol=.01)
        front = points[(points[:, 0] > .4) & (points[:, 0] < 1.2) &
                       (points[:, 2] > .2) & (points[:, 2] < .6)]
        assert len(front) > 50 and abs(np.median(front[:, 1]) - 2.) < .06


def test_right_positive_turn_and_translation(scene):
    folder, _ = scene
    with new_map() as m:
        m.integrate(folder, 0., dict(x=1., y=2., heading_deg=90.))
        top, seen = m.layers(.1, [-1., -1.], [80, 80])
        # (x,y) maps to (1+y,2-x), so the box moves to x=3..4,y=.7..1.7.
        assert np.count_nonzero(top[19:25, 42:46]) >= 15
        assert not top[32:36, 15:21].any()
        assert seen[21, 43]


@pytest.mark.parametrize('restart', ['parent', 'clock'])
def test_new_recording_resets_surfaces_and_coverage(scene, tmp_path, restart):
    folder, depth = scene
    other = tmp_path / 'run2/robot_B'
    write_depth(other, np.zeros_like(depth), stamps=(3.,))
    with new_map() as m:
        m.integrate(folder, 2., ZERO)
        assert m.layers(.1, [-4., -1.], 80)[0].any()
        if restart == 'parent':
            m.integrate(other, 3., ZERO)
        else:
            # The same folder is reused by the recorder with a restarted clock.
            for image in (folder / 'depth').glob('*.png'):
                cv2.imwrite(str(image), np.zeros_like(depth))
            m.integrate(folder, 1., ZERO)
        top, seen = m.layers(.1, [-4., -1.], 80)
        assert not top.any() and not seen.any()


def test_coverage_stops_at_depth_and_accumulates(scene, tmp_path):
    folder, depth = scene
    # Only the camera's central pixel column can observe anything.
    depth[:, :319] = 0
    depth[:, 320:] = 0
    for image in (folder / 'depth').glob('*.png'):
        cv2.imwrite(str(image), depth)
    with new_map() as m:
        m.integrate(folder, 0., ZERO)
        _, seen = m.layers(.1, [-1., -1.], [40, 100])
        assert seen[25, 9:11].any()
        assert not seen[:, 15:].any()
        assert not seen[95:].any()
        m.integrate(folder, 1., dict(x=2., y=0., heading_deg=0.))
        _, later = m.layers(.1, [-1., -1.], [40, 100])
        assert later[25, 9:11].any() and later[25, 29:31].any()


def test_export_contains_points_and_normals(scene, tmp_path):
    with new_map() as m:
        m.integrate(scene[0], 0., ZERO)
        path = tmp_path / 'map.ply'
        m.export(path)
        with path.open('rb') as f:
            header = []
            while (line := f.readline().decode().strip()) != 'end_header':
                header.append(line)
            count = int(next(s for s in header if s.startswith('element vertex')).split()[-1])
            data = np.fromfile(f, '<f4').reshape(-1, 6)
        assert count == len(data) > 100
        assert all(f'property float {n}' in header for n in ['x', 'y', 'z', 'nx', 'ny', 'nz'])
        points, normals = m.points_normals()
        np.testing.assert_array_equal(data, np.c_[points, normals])


def test_endpoints_round_trip_and_robot_isolation(scene, tmp_path):
    from brain.agent import server
    with TestClient(server.app) as client:
        req = dict(robot_dir=str(scene[0]), t=0., pose=ZERO)
        r = client.post('/map3d/test_B/integrate', json=req)
        assert r.status_code == 200 and r.json()['ok'] and r.json()['ms'] > 0
        layers = dict(cell_m=.1, origin=[-4., -1.], size=[81, 71])
        r = client.post('/map3d/test_B/layers', json=layers)
        assert r.status_code == 200
        out = r.json()
        top = np.frombuffer(base64.b64decode(out['top_cm']), '<u2').reshape(71, 81)
        bits = np.unpackbits(np.frombuffer(base64.b64decode(out['seen']), np.uint8), bitorder='little')
        seen = bits[:71 * 81].reshape(71, 81)
        assert top[33, 47] in range(70, 91) and seen[33, 47] and not seen[2, 2]
        assert not bits[71 * 81:].any() and out['ms'] > 0
        expected_top, expected_seen = server._maps3d['test_B'].layers(**layers)
        np.testing.assert_array_equal(top, expected_top)
        np.testing.assert_array_equal(seen, expected_seen)
        empty = client.post('/map3d/test_A/layers', json=layers).json()
        assert not np.frombuffer(base64.b64decode(empty['top_cm']), '<u2').any()
        path = tmp_path / 'http.ply'
        r = client.post('/map3d/test_B/export', json=dict(path=str(path)))
        assert r.status_code == 200 and r.json()['ok'] and path.stat().st_size > 100
        bad = client.post('/map3d/test_B/integrate', json={**req, 't': 20.})
        assert bad.status_code == 400
        assert client.get('/health').status_code == 200


def test_native_worker_death_returns_error_and_can_restart(scene):
    with new_map() as m:
        m.integrate(scene[0], 0., ZERO)
        os.kill(m._process.pid, signal.SIGKILL)
        m._process.join()
        with pytest.raises(RuntimeError, match='worker'):
            m.layers(.1, [0., 0.], 10)
        m.integrate(scene[0], 1., ZERO)
        assert m.layers(.1, [-4., -1.], 80)[0].any()
        os.kill(m._process.pid, signal.SIGKILL)
        m._process.join()
        m.integrate(scene[0], 2., ZERO)
        assert m.layers(.1, [-4., -1.], 80)[0].any()


def test_recorded_sequence_accuracy_and_budget(tmp_path):
    report = recorded_report(tmp_path / 'direct_report.json')
    assert report['frames'] == 15 and report['points'] > 1000
    assert report['median_distance_m'] < config()['accuracy_median_bound_m'], report
    assert report['integrate_max_ms'] < config()['integrate_budget_ms'], report
    assert report['layers_max_ms'] < config()['layers_budget_ms'], report
    print(json.dumps(report, sort_keys=True))


def test_recorded_sequence_native_subprocess(tmp_path):
    result = subprocess.run([sys.executable, __file__, str(tmp_path / 'report.json')],
                            capture_output=True, text=True, timeout=config()['worker_timeout_s'] * 2)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['frames'] == 15 and report['points'] > 1000
    assert report['median_distance_m'] < config()['accuracy_median_bound_m'], report
    assert report['integrate_p95_ms'] < config()['integrate_budget_ms'], report
    assert report['layers_p95_ms'] < config()['layers_budget_ms'], report
    print(json.dumps(report, sort_keys=True))


def recorded_report(path):
    """Truth is opened only here, after integrating the robot's own estimates."""
    states = [json.loads(l) for l in (RECORDING / 'robot_B/brain_state.jsonl').read_text().splitlines()][:30:2]
    timings, layer_times = [], []
    with new_map() as m:
        for s in states:
            start = time.perf_counter()
            m.integrate(RECORDING / 'robot_B', s['t'],
                        dict(x=s['x'], y=s['y'], heading_deg=-math.degrees(s['yaw_rad'])))
            timings.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            m.layers(.1, [-10., -5.], [200, 200])
            layer_times.append((time.perf_counter() - start) * 1000)
        points, normals = m.points_normals()
    truth_pose = np.loadtxt(RECORDING / 'oracle/robot_B_groundtruth.txt')
    initial = truth_pose[0]
    initial_rotation = Rotation.from_quat(initial[4:]).as_matrix()
    right = initial_rotation[:, 0]
    forward = initial_rotation[:, 2].copy()
    forward[1] = 0
    forward /= np.linalg.norm(forward)
    offset = SENSORS['camera']['mount']['forward_offset_m']
    origin = initial[1:4] - offset * forward
    origin[1] = 0.
    basis = np.column_stack([right, forward, [0., -1., 0.]])
    with (RECORDING / 'oracle/house_truth.ply').open('rb') as f:
        while f.readline().strip() != b'end_header':
            pass
        truth = np.fromfile(f, '<f4').reshape(-1, 3)
    truth = (truth - origin) @ basis
    distance = cKDTree(truth).query(points, workers=1)[0]
    position_errors, yaw_errors = [], []
    for s in states:
        gt = truth_pose[np.argmin(abs(truth_pose[:, 0] - s['t']))]
        fwd = Rotation.from_quat(gt[4:]).as_matrix()[:, 2].copy()
        fwd[1] = 0.
        fwd /= np.linalg.norm(fwd)
        body = (gt[1:4] - offset * fwd - origin) @ basis
        position_errors.append(float(np.linalg.norm(body[:2] - [s['x'], s['y']])))
        true_yaw = math.atan2(right @ fwd, forward @ fwd)
        delta = -s['yaw_rad'] - true_yaw
        yaw_errors.append(abs(math.atan2(math.sin(delta), math.cos(delta))))
    report = dict(frames=len(states), points=len(points), median_distance_m=float(np.median(distance)),
                  p95_distance_m=float(np.percentile(distance, 95)),
                  pose_median_m=float(np.median(position_errors)), pose_max_m=max(position_errors),
                  heading_median_deg=math.degrees(float(np.median(yaw_errors))),
                  heading_max_deg=math.degrees(max(yaw_errors)),
                  cold_integrate_ms=timings[0],
                  integrate_median_ms=float(np.median(timings[1:])), integrate_p95_ms=float(np.percentile(timings[1:], 95)),
                  integrate_max_ms=max(timings[1:]), layers_median_ms=float(np.median(layer_times)),
                  layers_p95_ms=float(np.percentile(layer_times, 95)), layers_max_ms=max(layer_times))
    Path(path).write_text(json.dumps(report))
    return report


def test_depth_lookup_tie_then_refresh_live_index(scene):
    folder, depth = scene
    # Tie at 0.5 must use the earlier zero frame, like vision.load_frame.
    cv2.imwrite(str(folder / 'depth/0.000000.png'), np.zeros_like(depth))
    with new_map() as m:
        m.integrate(folder, .5, ZERO)
        assert not m.layers(.1, [-4., -1.], 80)[1].any()
        m.integrate(folder, 1., ZERO)
        assert m.layers(.1, [-4., -1.], 80)[0].any()
        # The live recorder appends its sensor index between requests.
        cv2.imwrite(str(folder / 'depth/3.000000.png'), depth)
        with (folder / 'depth.txt').open('a') as f:
            f.write('3.000000 depth/3.000000.png\n')
        m.integrate(folder, 3., dict(x=10., y=0., heading_deg=0.))
        assert m.layers(.1, [6., -1.], 80)[0].any()
        assert m.layers(.1, [-4., -1.], 80)[0].any(), 'growth retains the old map'


def test_recording_intrinsics_and_depth_factor_override_sensor_defaults(scene):
    folder, depth = scene
    with new_map() as reference:
        reference.integrate(folder, 0., ZERO)
        expected = reference.layers(.1, [-4., -1.], 80)
    calibration = folder / 'camera.yaml'
    calibration.write_text(calibration.read_text().replace('1000.0', '2000.0'))
    for image in (folder / 'depth').glob('*.png'):
        cv2.imwrite(str(image), depth * 2)
    with new_map() as m:
        m.sensors['camera'].update(fx=1., fy=1., cx=0., cy=0.)
        m.integrate(folder, 0., ZERO)
        actual = m.layers(.1, [-4., -1.], 80)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


@pytest.mark.parametrize('bad_depth', [0, 100, 8001])
def test_missing_and_out_of_sensor_range_depth_never_marks_seen(scene, bad_depth):
    folder, depth = scene
    for image in (folder / 'depth').glob('*.png'):
        cv2.imwrite(str(image), np.full_like(depth, bad_depth))
    with new_map() as m:
        m.integrate(folder, 0., ZERO)
        top, seen = m.layers(.1, [-4., -1.], 80)
        assert not top.any() and not seen.any()


def test_surface_does_not_mark_occluded_columns(scene):
    folder, depth = scene
    # A depth return at 2 m optical z for every pixel forms an opaque wall.
    for image in (folder / 'depth').glob('*.png'):
        cv2.imwrite(str(image), np.full_like(depth, 2000))
    with new_map() as m:
        m.integrate(folder, 0., ZERO)
        _, seen = m.layers(.1, [-4., -1.], 80)
        assert seen[25, 40]
        assert not seen[40:, :].any(), 'camera did not see behind the wall'


@pytest.mark.parametrize('bad', ['time', 'pose', 'stale', 'shape', 'dtype', 'path', 'calibration'])
def test_bad_frame_rejected_without_losing_existing_map(scene, bad):
    folder, depth = scene
    with new_map() as m:
        m.integrate(folder, 0., ZERO)
        t, pose = 1., ZERO
        if bad == 'time':
            t = float('nan')
        elif bad == 'pose':
            pose = {**ZERO, 'x': float('inf')}
        elif bad == 'stale':
            t = 20.
        elif bad in ('shape', 'dtype'):
            image = depth[:-1] if bad == 'shape' else depth.astype(np.uint8)
            cv2.imwrite(str(folder / 'depth/1.000000.png'), image)
        elif bad == 'path':
            (folder / 'depth.txt').write_text('1.0 ../outside.png\n')
        elif bad == 'calibration':
            p = folder / 'camera.yaml'
            p.write_text(p.read_text().replace('Camera1.fx: 415.692', 'Camera1.fx: 0.0'))
        with pytest.raises(ValueError):
            m.integrate(folder, t, pose)
        assert m.layers(.1, [-4., -1.], 80)[0].any()


@pytest.mark.parametrize('args', [(0., [0., 0.], 10), (.1, [float('nan'), 0.], 10),
                                   (.1, [0., 0.], [2.5, 10]), (.1, [0., 0.], [-1, 10])])
def test_bad_grid_rejected(args):
    with new_map() as m:
        with pytest.raises(ValueError):
            m.layers(*args)


def test_coverage_small_and_unaligned_cells_preserves_observed_area(scene):
    with new_map() as m:
        m.integrate(scene[0], 0., ZERO)
        top, seen = m.layers(.025, [.493, 2.193], [20, 20])
        assert seen.all()
        assert top.any() and top.shape == (20, 20)


def test_worker_timeout_is_bounded_and_loses_no_other_robot(scene):
    with new_map() as m:
        m.integrate(scene[0], 0., ZERO)
        m.settings['worker_timeout_s'] = .01
        os.kill(m._process.pid, signal.SIGSTOP)
        with pytest.raises(RuntimeError, match='timed out'):
            m.layers(.1, [0., 0.], 10)
        assert m._process is None


def test_endpoint_native_death_is_503_and_shutdown_closes_workers(scene):
    from brain.agent import server
    with TestClient(server.app) as client:
        assert client.post('/map3d/crash/integrate', json=dict(
            robot_dir=str(scene[0]), t=0., pose=ZERO)).status_code == 200
        worker = server._maps3d['crash']._process
        os.kill(worker.pid, signal.SIGKILL)
        worker.join()
        response = client.post('/map3d/crash/layers', json=dict(cell_m=.1, origin=[0, 0], size=10))
        assert response.status_code == 503
        assert client.get('/health').status_code == 200
        assert client.post('/map3d/crash/integrate', json=dict(
            robot_dir=str(scene[0]), t=1., pose=ZERO)).status_code == 200
        live = server._maps3d['crash']._process
    assert not live.is_alive() and not server._maps3d


def test_downsampling_preserves_pixel_centres_and_box_height(scene):
    with new_map() as m:
        m.settings['depth_stride'] = 2
        m.integrate(scene[0], 0., ZERO)
        top, seen = m.layers(.1, [-4., -1.], [80, 70])
        assert np.count_nonzero(top[32:36, 45:51]) >= 15
        assert np.max(np.abs(top[32:36, 45:51].astype(int) - 80)) <= 10
        assert not top[30:40, 25:35].any()
        assert seen[33, 47]


def test_floor_height_threshold_excludes_small_floor_errors(scene):
    folder, depth = scene
    cam = SENSORS['camera']
    v, u = np.indices(depth.shape)
    pitch = math.radians(cam['mount']['pitch_down_deg'])
    down = math.sin(pitch) + (v - cam['cy']) / cam['fy'] * math.cos(pitch)
    with np.errstate(divide='ignore', invalid='ignore'):
        plane = (cam['mount']['height_m'] - .08) / down
    plane[(plane < SENSORS['depth']['min_m']) | (plane > SENSORS['depth']['max_m'])] = 0
    raw = np.rint(plane * 1000).astype(np.uint16)
    for image in (folder / 'depth').glob('*.png'):
        cv2.imwrite(str(image), raw)
    with new_map() as m:
        m.integrate(folder, 0., ZERO)
        top, seen = m.layers(.1, [-4., -1.], 80)
        assert not top.any() and seen.any()


def test_nonfinite_pose_rejected_before_native():
    from brain.agent.map3d import camera_pose
    with pytest.raises(ValueError, match='pose must be finite'):
        camera_pose(dict(x=float('inf'), y=0., heading_deg=0.), SENSORS['camera']['mount'])


def test_changed_camera_calibration_starts_a_new_volume(scene):
    folder, depth = scene
    with new_map() as m:
        m.integrate(folder, 0., ZERO)
        assert m.layers(.1, [-4., -1.], 80)[0].any()
        calibration = folder / 'camera.yaml'
        calibration.write_text(calibration.read_text().replace('Camera1.fx: 415.692', 'Camera1.fx: 420.0'))
        cv2.imwrite(str(folder / 'depth/1.000000.png'), np.zeros_like(depth))
        m.integrate(folder, 1., ZERO)
        top, seen = m.layers(.1, [-4., -1.], 80)
        assert not top.any() and not seen.any()


if __name__ == '__main__':
    recorded_report(sys.argv[1])
