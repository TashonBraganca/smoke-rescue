"""Analytic recording tests with deliberately wrong frame conventions."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest


@pytest.fixture
def recording(tmp_path):
    rec = tmp_path / 'square'
    robot, truth = rec / 'robot_A', rec / 'oracle'
    robot.mkdir(parents=True)
    truth.mkdir()
    (robot / 'brain_map').mkdir()
    times = np.arange(0, 40.5, .5)
    y = .7 + .12 * np.minimum(times, 10) + .12 * np.maximum(times - 26, 0)
    yaw = .3
    states = [dict(t=float(t), x=1.2, y=float(v), yaw_rad=yaw, state='explore',
                   path=[[1.2, float(v)], [1.2, 5.]]) for t, v in zip(times, y)]
    (robot / 'brain_state.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in states))
    (robot / 'decisions.jsonl').write_text(json.dumps(dict(t=15, kind='stuck', numbers={})) + '\n')
    forward = np.array([-np.sin(yaw), np.cos(yaw)])
    poses = [[t, 1.2 + .25 * forward[0], -1.6, v + .25 * forward[1],
              0, -np.sin(yaw / 2), 0, np.cos(yaw / 2)] for t, v in zip(times, y)]
    np.savetxt(truth / 'robot_A_groundtruth.txt', poses)
    u = np.linspace(0, 6, 1201)
    xy = np.concatenate([np.c_[u, u * 0], np.c_[u, u * 0 + 6], np.c_[u * 0, u], np.c_[u * 0 + 6, u]])
    points = np.c_[xy[:, 0], np.full(len(xy), -.6), xy[:, 1]]
    with (truth / 'house_truth.ply').open('wb') as f:
        f.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n'.encode())
        f.write(points.astype('<f4').tobytes())
    (truth / 'floorplan_used.json').write_text(json.dumps(dict(doors=[dict(id='gap', wall='z', at=[3, 2], width=1, state='open')])))
    (truth / 'collisions.json').write_text('{"robot_A": 2.5}')
    angles = -np.pi + np.arange(180) * 2 * np.pi / 180
    lines = [f'# analytic rays\nbeams 180 angle_min_rad {-np.pi} increment_rad {2*np.pi/180} mount_height_m .6 mount_forward_m .12 min_m .05 max_m 20\n']
    for s in states:
        origin = np.array([s['x'], s['y']]) + .12 * forward
        directions = np.c_[-np.sin(yaw + angles), np.cos(yaw + angles)]
        with np.errstate(divide='ignore', invalid='ignore'):
            distances = np.stack([(wall - origin[axis]) / directions[:, axis] for axis in (0, 1) for wall in (0., 6.)])
        distances[distances <= 0] = np.inf
        ranges = np.rint(distances.min(axis=0) * 1000).astype(int)
        lines.append(str(s['t']) + ' ' + ' '.join(map(str, ranges)) + '\n')
    (robot / 'lidar.txt').write_text(''.join(lines))
    grid = np.full((90, 90), 128, np.uint8)
    grid[20:81, 20:81] = 255
    grid[20, 20:81] = grid[80, 20:81] = 0
    grid[20:81, 20] = grid[20:81, 80] = 0
    grid[40, 45:55] = 0
    cv2.imwrite(str(robot / 'brain_map/40.png'), np.flipud(grid))
    (robot / 'brain_map.txt').write_text('40 1.2 3.58 .3 brain_map/40.png\n')
    (rec / 'meta.json').write_text('{"robots": ["robot_A"]}')
    return rec


def test_lidar_gate_and_angle_negative_control(recording, tmp_path):
    from brain.nav_eval import evaluate
    report = evaluate(recording, tmp_path / 'good')['robot_A']
    assert report['lidar']['true_pose']['passed']
    assert report['lidar']['true_pose']['median_m'] < .005
    assert report['localization']['position_m']['max'] < 1e-6
    lidar = recording / 'robot_A/lidar.txt'
    text = lidar.read_text().replace('angle_min_rad -3.141592653589793', 'angle_min_rad 3.141592653589793')
    lidar.write_text(text.replace('increment_rad 0.03490658503988659', 'increment_rad -0.03490658503988659'))
    assert not evaluate(recording, tmp_path / 'bad')['robot_A']['lidar']['true_pose']['passed']
    result = subprocess.run([sys.executable, '-m', 'brain.nav_eval', str(recording), '--out', str(tmp_path / 'cli')], capture_output=True, text=True)
    assert result.returncode == 1
    assert 'LIDAR CONVENTION FAILURE' in result.stdout


def test_png_row_negative_control(recording, tmp_path):
    from brain.nav_eval import evaluate
    good = evaluate(recording, tmp_path / 'good')['robot_A']
    png = recording / 'robot_A/brain_map/40.png'
    cv2.imwrite(str(png), np.flipud(cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)))
    bad = evaluate(recording, tmp_path / 'bad')['robot_A']
    assert good['map']['precision'] > .95
    assert bad['map']['precision'] < .6
    assert bad['lidar']['true_pose']['passed']  # PNG rows cannot affect the independent LiDAR gate.


def test_stalls_doors_collisions_and_artifacts(recording, tmp_path):
    from brain.nav_eval import evaluate
    out = tmp_path / 'eval'
    r = evaluate(recording, out)['robot_A']
    assert any(s['start_t'] <= 10 and s['duration_s'] >= 15 and s['events'].get('stuck') == 1 for s in r['stalls'])
    assert r['map']['doors']['closed'] == ['gap']
    assert r['collision_seconds'] == 2.5
    for name in ['robot_A/report.json', 'robot_A/errors.png', 'robot_A/overlay.png', 'overlay.png']:
        assert (out / name).is_file()


def load_watcher():
    spec = importlib.util.spec_from_file_location('watch_run', Path(__file__).parents[2] / 'tools/watch_run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_watcher_partial_lines_counts_and_snapshot(recording, tmp_path):
    watcher = load_watcher()
    events = recording / 'robot_A/decisions.jsonl'
    with events.open('a') as f:
        f.write(''.join(json.dumps(dict(t=35, kind='collision_hold')) + '\n' for _ in range(31)))
        f.write(json.dumps(dict(t=39, kind='localization', numbers=dict(mean_fit=.01, matched=42))) + '\n')
        f.write('{"t":40,"kind":')
    w = watcher.RunWatcher(recording, tmp_path / 'snap')
    line = w.lines()[0]
    assert 'collision_hold=31' in line and 'SPAM' in line and 'mean_fit' in line
    assert 'collision_hold=0' in w.lines()[0]
    assert cv2.imread(str(tmp_path / 'snap/robot_A_latest.png')).shape[:2] == (270, 270)
    with events.open('a') as f:
        f.write('"stuck"}\n')
    assert 'stuck=1' in w.lines()[0]


def test_watcher_finished_and_inactivity(recording, capsys):
    watcher = load_watcher()
    assert watcher.watch(recording, every=.01) == 0
    assert 'RUN FINISHED' in capsys.readouterr().out
    (recording / 'meta.json').unlink()
    tick = iter([0., 601.])
    assert watcher.watch(recording, every=.01, clock=lambda: next(tick), sleep=lambda _: None) == 2
    assert 'NO PROGRESS' in capsys.readouterr().out


def test_commented_lidar_header_from_unity(recording, tmp_path):
    from brain.nav_eval import evaluate
    path = recording / 'robot_A/lidar.txt'
    path.write_text(path.read_text().replace('\nbeams ', '\n# beams '))
    assert evaluate(recording, tmp_path / 'commented')['robot_A']['lidar']['true_pose']['passed']


def test_unknown_door_and_free_crossing(recording):
    from brain.nav_eval import door_status
    from brain.nav_io import latest_map
    _, grid = latest_map(recording / 'robot_A')
    door = dict(wall='z', at=[3, 2], width=1)
    grid[40, 45:55] = 128
    assert door_status(grid, door) == 'not_observed'
    grid[40, 45:55] = 255
    assert door_status(grid, door) == 'open'
    grid[35, 45:55] = 0  # All cells at least 0.4 m south of the wall are blocked.
    assert door_status(grid, door) == 'closed'


def test_watcher_pinned_complete_and_loop():
    watcher = load_watcher()
    states = [dict(t=t, x=1., y=1., yaw_rad=0., state='explore') for t in range(16)]
    assert watcher.is_pinned(states)
    assert not watcher.is_pinned(states[1:])
    states[-1]['state'] = 'complete'
    assert not watcher.is_pinned(states)
    states[-1]['state'] = 'explore'
    states[5]['x'] = 2.
    assert not watcher.is_pinned(states)


def test_missing_lidar_is_unverified_not_a_pass(recording, tmp_path):
    from brain.nav_eval import main
    (recording / 'robot_A/lidar.txt').unlink()
    assert main([str(recording), '--out', str(tmp_path / 'missing')]) == 2


def test_yaw_wrap_does_not_interpolate_through_zero():
    from brain.nav_io import interpolate_poses
    result = interpolate_poses(np.array([0., 1.]), np.array([[0., 0., np.radians(179)], [0., 0., np.radians(-179)]]), np.array([.5]))
    assert abs(abs(result[0, 2]) - np.pi) < 1e-9
    with pytest.raises(ValueError, match='outside'):
        interpolate_poses(np.array([0., 1.]), np.zeros((2, 3)), np.array([2.]))


def test_watcher_never_opens_hidden_truth(recording, monkeypatch):
    watcher = load_watcher()
    original = Path.open
    def guarded_open(path, *args, **kwargs):
        assert 'oracle' not in path.parts, 'watcher tried to open hidden truth'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    assert watcher.RunWatcher(recording).lines()
    assert all('/oracle/' not in entry[0] for entry in watcher.file_signature(recording))


def test_brain_projection_measures_drift_separately(recording, tmp_path):
    from brain.nav_eval import evaluate
    path = recording / 'robot_A/brain_state.jsonl'
    states = [json.loads(line) for line in path.read_text().splitlines()]
    for s in states:
        s['x'] += .6
    path.write_text(''.join(json.dumps(s) + '\n' for s in states))
    r = evaluate(recording, tmp_path / 'drift')['robot_A']
    assert r['lidar']['true_pose']['passed']
    assert r['lidar']['brain_pose']['median_m'] > .05
    assert r['localization']['position_m']['median'] == pytest.approx(.6)
    assert r['localization']['first_over_0_2_m_t'] == 0


def test_stall_detector_does_not_call_a_loop_or_completion_a_stall(recording):
    from brain.nav_eval import detect_stalls
    times = np.arange(0., 21.)
    states = [dict(t=t, x=0., y=0., yaw_rad=0., state='explore') for t in times]
    poses = np.c_[np.sin(times * np.pi / 5), np.zeros((len(times), 2))]
    assert detect_stalls(states, times, poses, []) == []
    for state in states:
        state['state'] = 'complete'
    assert detect_stalls(states, times, np.zeros_like(poses), []) == []


def test_no_returns_are_reported_and_gate_is_unverified(recording, tmp_path):
    from brain.nav_eval import evaluate
    path = recording / 'robot_A/lidar.txt'
    lines = path.read_text().splitlines()
    path.write_text('\n'.join(lines[:2] + [line.split()[0] + ' ' + ' '.join(['0'] * 180) for line in lines[2:]]) + '\n')
    lidar = evaluate(recording, tmp_path / 'no_returns')['robot_A']['lidar']
    assert lidar['no_return_share'] == 1
    assert lidar['true_pose']['status'] == 'no_returns'
    assert not lidar['true_pose']['passed']


def test_two_robot_overlay_and_phantom_clusters(recording, tmp_path):
    import shutil
    from brain.nav_eval import evaluate
    shutil.copytree(recording / 'robot_A', recording / 'robot_B')
    shutil.copyfile(recording / 'oracle/robot_A_groundtruth.txt', recording / 'oracle/robot_B_groundtruth.txt')
    (recording / 'meta.json').write_text('{"robots":["robot_A","robot_B"]}')
    r = evaluate(recording, tmp_path / 'both')
    assert set(r) == {'robot_A', 'robot_B'}
    assert r['robot_A']['map']['phantom_cells'] == 10
    assert r['robot_A']['map']['phantom_clusters'][0]['centre_xy'] == pytest.approx([3., 2.05])
    assert cv2.imread(str(tmp_path / 'both/overlay.png')).shape[:2] == (630, 630)


def test_snapshots_do_not_hide_recording_changes(recording):
    watcher = load_watcher()
    snap = recording / 'robot_A'
    signature = watcher.file_signature(recording, snap)
    assert any(name.endswith('brain_state.jsonl') for name, _, _ in signature)
    watcher.RunWatcher(recording, snap).lines()
    assert watcher.file_signature(recording, snap) == signature
