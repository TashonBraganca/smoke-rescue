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


@pytest.fixture
def local_recording(recording):
    """Two independently rotated starts observing the same analytic square."""
    from brain.nav_io import rotation, sample_grid
    original = [json.loads(l) for l in (recording / 'robot_A/brain_state.jsonl').read_text().splitlines()]
    source = np.flipud(cv2.imread(str(recording / 'robot_A/brain_map/40.png'), 0))
    source[39:42, 44:56] = 0
    config = json.loads((Path(__file__).parents[2] / 'config/brain.json').read_text())
    cell, size = config['cell_m'], config['grid_size_m']
    n = round(size / cell)
    yy, xx = np.indices((n, n))
    own = -size / 2 + (np.stack([xx, yy], axis=-1) + .5) * cell
    scenario = {'search_area': [0, 0, 6, 6], 'robots': []}
    for name, x, yaw in [('robot_A', 1.2, .3), ('robot_B', 4.8, -.7)]:
        rd = recording / name
        (rd / 'brain_map').mkdir(parents=True, exist_ok=True)
        start = np.array([x, .7])
        scenario['robots'].append(dict(id=name, spawn=dict(at=start.tolist(), yaw=float(-np.degrees(yaw)))))
        world = [dict(s, x=x, yaw_rad=yaw) for s in original]
        states = []
        for s in world:
            xy = (np.array([s['x'], s['y']]) - start) @ rotation(yaw)
            path = (np.array([[s['x'], s['y']], [x, 5.]]) - start) @ rotation(yaw)
            states.append(dict(s, x=float(xy[0]), y=float(xy[1]), yaw_rad=0., path=path.tolist(), goal=path[-1].tolist()))
        (rd / 'brain_state.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in states))
        forward = np.array([-np.sin(yaw), np.cos(yaw)])
        poses = [[s['t'], x + .25 * forward[0], -1.6, s['y'] + .25 * forward[1],
                  0, -np.sin(yaw / 2), 0, np.cos(yaw / 2)] for s in world]
        np.savetxt(recording / 'oracle' / f'{name}_groundtruth.txt', poses)
        angles = -np.pi + np.arange(180) * 2 * np.pi / 180
        lines = [f'# beams 180 angle_min_rad {-np.pi} increment_rad {2*np.pi/180} mount_height_m .6 mount_forward_m .12 min_m .05 max_m 20\n']
        for s in world:
            origin = np.array([x, s['y']]) + .12 * forward
            directions = np.c_[-np.sin(yaw + angles), np.cos(yaw + angles)]
            with np.errstate(divide='ignore', invalid='ignore'):
                distances = np.stack([(wall - origin[axis]) / directions[:, axis] for axis in (0, 1) for wall in (0., 6.)])
            distances[distances <= 0] = np.inf
            lines.append(str(s['t']) + ' ' + ' '.join(map(str, np.rint(distances.min(axis=0) * 1000).astype(int))) + '\n')
        (rd / 'lidar.txt').write_text(''.join(lines))
        world_cells = own @ rotation(yaw).T + start
        grid = sample_grid(source, world_cells, np.array([-2., -2.]), cell)
        cv2.imwrite(str(rd / 'brain_map/40.png'), np.flipud(grid))
        (rd / 'brain_map.txt').write_text(f"40 {states[-1]['x']} {states[-1]['y']} 0 brain_map/40.png\n")
        # Sender B into receiver A. Explicit formula independent of frame_link_report.
        dx, dy = 3.6 * np.cos(.3), -3.6 * np.sin(.3)
        event = dict(t=12., kind='frame_linked', reason='matched robot_B', numbers=dict(dx_m=float(dx), dy_m=float(dy), dyaw_deg=float(np.degrees(-1.))))
        (rd / 'decisions.jsonl').write_text(json.dumps(event) + '\n' if name == 'robot_A' else '')
    scenario_path = recording / 'scenario.json'
    scenario_path.write_text(json.dumps(scenario))
    (recording / 'meta.json').write_text(json.dumps(dict(robots=['robot_A', 'robot_B'], scenario=str(scenario_path))))
    return recording


def test_local_frames_maps_lidar_and_link(local_recording, tmp_path, capsys):
    from brain.nav_eval import evaluate, print_table
    reports = evaluate(local_recording, tmp_path / 'local')
    for r in reports.values():
        assert r['frame_check']['recorded_frame'] == 'local'
        assert r['frame_check']['passed']
        assert r['localization']['position_m']['max'] < 1e-6
        assert r['localization']['yaw_deg']['max'] < 1e-6
        assert r['lidar']['true_pose']['median_m'] < .005
        assert r['lidar']['brain_pose']['median_m'] < .005
        assert r['map']['precision'] > .85
        assert r['map']['doors']['closed'] == ['gap']
    link = reports['robot_A']['frame_link']
    assert link['t'] == 12
    assert link['sender'] == 'robot_B'
    assert link['position_error_m'] < 1e-10
    assert link['yaw_error_deg'] < 1e-10
    assert reports['robot_B']['frame_link']['t'] is None
    print_table(reports)
    assert 'linked at 12.00s' in capsys.readouterr().out


def test_local_start_sign_and_missing_transform_controls(local_recording, tmp_path, monkeypatch):
    import brain.nav_eval as nav
    good = nav.evaluate(local_recording, tmp_path / 'good')
    scenario_path = local_recording / 'scenario.json'
    scenario = json.loads(scenario_path.read_text())
    for robot in scenario['robots']:
        robot['spawn']['yaw'] *= -1
    scenario_path.write_text(json.dumps(scenario))
    wrong = nav.evaluate(local_recording, tmp_path / 'sign')
    assert wrong['robot_B']['localization']['position_m']['max'] > 1
    assert wrong['robot_B']['localization']['yaw_deg']['max'] > 70
    assert wrong['robot_B']['map']['precision'] < good['robot_B']['map']['precision'] - .2
    assert wrong['robot_A']['frame_link']['position_error_m'] > 1
    for robot in scenario['robots']:
        robot['spawn']['yaw'] *= -1
    scenario_path.write_text(json.dumps(scenario))
    monkeypatch.setattr(nav, 'placed_states', lambda states, transform: states)
    missing = nav.evaluate(local_recording, tmp_path / 'missing_transform')
    assert missing['robot_B']['localization']['position_m']['max'] > 4
    assert missing['robot_B']['lidar']['brain_pose']['median_m'] > .05
    assert missing['robot_B']['lidar']['true_pose']['passed']


def test_watcher_keeps_link_status_without_truth(local_recording, tmp_path, monkeypatch):
    watcher = load_watcher()
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert 'oracle' not in path.parts
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    w = watcher.RunWatcher(local_recording, tmp_path / 'snap')
    for lines in [w.lines(), w.lines()]:
        assert 'linked at 12.0s' in lines[0] and 'dyaw=-57.30deg' in lines[0]
        assert 'not linked' in lines[1]
    config = json.loads((Path(__file__).parents[2] / 'config/brain.json').read_text())
    side = 3 * round(config['grid_size_m'] / config['cell_m'])
    assert cv2.imread(str(tmp_path / 'snap/robot_A_latest.png')).shape[:2] == (side, side)


def test_shared_frame_detected_from_spawn(recording, tmp_path):
    from brain.nav_eval import evaluate
    from brain.nav_io import recording_frame, placed_states
    states = [json.loads(l) for l in (recording / 'robot_A/brain_state.jsonl').read_text().splitlines()]
    scenario = dict(robots=[dict(id='robot_A', spawn=dict(at=[1.2, .7], yaw=float(-np.degrees(.3))))])
    path = recording / 'scenario.json'
    path.write_text(json.dumps(scenario))
    (recording / 'meta.json').write_text(json.dumps(dict(robots=['robot_A'], scenario=str(path))))
    transform, frame = recording_frame(states, scenario, 'robot_A')
    assert frame == 'shared'
    np.testing.assert_array_equal(transform, [0, 0, 0])
    assert placed_states(states, transform) == states
    report = evaluate(recording, tmp_path / 'shared')['robot_A']
    assert report['localization']['position_m']['max'] < 1e-6
    assert report['map']['precision'] > .95


def test_link_errors_measure_translation_and_wrapped_yaw(local_recording):
    from brain.nav_eval import frame_link_report
    from brain.nav_io import load_scenario
    meta = json.loads((local_recording / 'meta.json').read_text())
    event = json.loads((local_recording / 'robot_A/decisions.jsonl').read_text())
    event['numbers']['dx_m'] += .3
    event['numbers']['dy_m'] += .4
    event['numbers']['dyaw_deg'] += 365
    link = frame_link_report([event], 'robot_A', meta['robots'], load_scenario(meta))
    assert link['position_error_m'] == pytest.approx(.5)
    assert link['yaw_error_deg'] == pytest.approx(5)


def test_indoor_report_counts_marked_cells_outside_the_building():
    import numpy as np
    from brain.nav_eval import indoor_report

    building = np.zeros((40, 40), bool); building[10:30, 10:30] = True
    indoor = np.zeros((40, 40), np.uint8); indoor[12:28, 12:28] = 255
    searched = np.zeros((40, 40), np.uint8)
    clean = indoor_report(indoor, building, searched, cell=.1)
    assert clean['indoor_cells'] == 256 and clean['indoor_cells_outside'] == 0
    # A threshold one cell past the wall face is tolerated; a strip of yard 1 m out is not.
    indoor[20, 30] = 128; indoor[5, 12:22] = 128; searched[5, 12:16] = 255
    leaked = indoor_report(indoor, building, searched, cell=.1)
    assert leaked['indoor_cells_outside'] == 10 and leaked['searched_cells_outside'] == 4
    assert abs(leaked['indoor_outside_share'] - 10 / 267) < 1e-9


def test_sample_grid_fill_marks_cells_outside_the_source_map():
    import numpy as np
    from brain.nav_io import sample_grid

    grid = np.full((2, 2), 255, np.uint8)
    xy = np.array([[[.05, .05], [5.0, 5.0]]])
    assert sample_grid(grid, xy, np.zeros(2), .1).tolist() == [[255, 128]]
    assert sample_grid(grid, xy, np.zeros(2), .1, fill=0).tolist() == [[255, 0]]


def test_a_stall_is_explained_only_by_a_short_deliberate_stop():
    from brain.nav_eval import detect_stalls
    times = np.arange(0., 41.)
    states = [dict(t=t, x=0., y=0., yaw_rad=0., state='explore') for t in times]
    still = np.zeros((len(times), 3))
    sweep = [dict(t=2., kind='sweep_started')]
    long = detect_stalls(states, times, still, sweep)
    assert len(long) == 1 and long[0]['duration_s'] > 20 and not long[0]['explained']   # outlasts the sweep
    short = detect_stalls(states[:15], times[:15], still[:15], sweep)
    assert len(short) == 1 and short[0]['explained'] and short[0]['reason'] == ['sweep_started']
    unexplained = detect_stalls(states[:15], times[:15], still[:15], [dict(t=2., kind='replan')])
    assert len(unexplained) == 1 and not unexplained[0]['explained']


def test_cells_on_a_survivor_are_not_phantoms():
    import numpy as np
    from brain.nav_eval import map_accuracy

    grid = np.full((60, 60), 255, np.uint8)
    grid[10, 5:55] = 0                                  # a wall the truth has
    grid[40:43, 30:33] = 0                              # a person lying on the floor
    points = np.array([[(x + .5) * .1, -1.0, (10 + .5) * .1] for x in range(5, 55)])   # 1 m up (y is negative up)
    plain, _ = map_accuracy(grid, points, {}, origin=np.zeros(2), cell=.1)
    people, _ = map_accuracy(grid, points, {}, origin=np.zeros(2), cell=.1, people_xy=[(3.15, 4.15)])
    assert plain['phantom_cells'] == 9 and plain['precision'] < 1
    assert people['phantom_cells'] == 0 and people['precision'] == 1 and people['occupied_cells_on_people'] == 9
