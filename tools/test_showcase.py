"""Sensor geometry and temporal selection for the audience map."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import showcase


@pytest.fixture
def live(tmp_path):
    for robot in ('robot_A', 'robot_B'):
        rd = tmp_path / robot
        (rd / 'brain_map').mkdir(parents=True)
        (rd / 'brain_map.txt').write_text(
            '# t x y yaw_rad file\n5 0 0 0 brain_map/5.png\n10 0 0 0 brain_map/10.png\n')
        grid = np.full((40, 40), 128, np.uint8)
        grid[5, 5] = 0 if robot == 'robot_A' else 255
        grid[30, 30] = 255
        cv2.imwrite(str(rd / 'brain_map/5.png'), grid)
        cv2.imwrite(str(rd / 'brain_map/10.png'), np.zeros_like(grid))
        (rd / 'lidar.txt').write_text(
            '# beams 4 angle_min_rad 0 increment_rad 1.5707963267948966 '
            'mount_height_m 0.4 mount_forward_m 0.2 min_m 0.1 max_m 8\n'
            '5 1000 2000 0 1000\n10 500 500 500 500\n')
    states = {r: [dict(t=4, x=0, y=0, yaw_rad=np.deg2rad(170)),
                  dict(t=6, x=2, y=0, yaw_rad=np.deg2rad(-170))]
              for r in ('robot_A', 'robot_B')}
    scenario = tmp_path / 'scenario.json'
    scenario.write_text(json.dumps({'search_area': [-1, -1, 1, 1]}))
    return showcase.LiveMaps(tmp_path, {'scenario': str(scenario)}, list(states), states)


def test_live_maps_not_available_without_snapshots(tmp_path):
    assert not showcase.LiveMaps.available(tmp_path, ['robot_A'])


def test_snapshot_selection_merge_priority_and_north(live):
    assert live.map_indices(4) == (-1, -1)
    assert live.map_indices(9.99) == (0, 0)
    assert live.map_indices(10) == (1, 1)
    base = live.base(6, (400, 400))
    # PNG row zero stays north; occupied wins over the other robot's free cell.
    assert tuple(base[55, 55]) == showcase.TEXT
    assert not np.array_equal(base[305, 305], base[55, 55])
    assert live.pixels(np.array([[0, 1], [0, -1]]), (400, 400))[0, 1] < 200
    assert live.pixels(np.array([[0, 1], [0, -1]]), (400, 400))[1, 1] > 200


def test_lidar_uses_sweep_time_brain_pose_and_mount(live):
    pose = live.pose('robot_A', 5)
    np.testing.assert_allclose(pose[:2], [1, 0])
    assert abs(abs(pose[2]) - np.pi) < 1e-7
    origin, hits = live.sweep('robot_A', 9)
    np.testing.assert_allclose(origin, [1, -.2], atol=1e-7)
    np.testing.assert_allclose(hits, [[1, -1.2], [3, -.2], [0, -.2]], atol=1e-7)
    assert live.sweep('robot_A', 4) is None
    assert live.pose('robot_A', 3) is None


def test_map_and_lidar_cache_survive_source_removal(live):
    first = live.base(6, (400, 400)).copy()
    for rd in live.rec.iterdir():
        if rd.is_dir():
            (rd / 'brain_map/5.png').unlink()
            (rd / 'lidar.txt').unlink()
    np.testing.assert_array_equal(live.base(7, (400, 400)), first)
    assert live.sweep('robot_A', 7) is not None
    assert live.render(7, (616, 414)).size == (414, 414)


def test_free_tints_and_unknown_before_first_map(live):
    before = live.base(4, (400, 400))
    assert np.all(before == showcase.PANEL)
    # Change B's map to unknown here to distinguish the free-space ownership.
    path = live.maps['robot_B'][0][1]
    grid = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    grid[30, 30] = 128
    cv2.imwrite(str(path), grid)
    after = live.base(6, (400, 400))
    tint = np.array(showcase.ROBOT[0]) * .22 + np.array(showcase.PANEL) * .78
    np.testing.assert_array_equal(after[305, 305], tint.astype(np.uint8))


def test_one_robot_without_map_or_lidar_is_supported(live):
    rd = live.rec / 'robot_B'
    (rd / 'brain_map.txt').unlink()
    (rd / 'lidar.txt').unlink()
    p = showcase.LiveMaps(live.rec, {'scenario': str(live.rec / 'scenario.json')},
                          live.robots, live.states)
    assert p.map_indices(6) == (0, -1)
    assert p.sweep('robot_B', 6) is None
    assert p.render(6, (616, 424)).size == (424, 424)


def test_lidar_body_left_and_no_return(live):
    live.poses['robot_A'][:, 2] = 0
    origin, hits = live.sweep('robot_A', 5)
    np.testing.assert_allclose(origin, [1, .2], atol=1e-7)
    np.testing.assert_allclose(hits, [[1, 1.2], [-1, .2], [2, .2]], atol=1e-7)


def test_panel_never_opens_oracle(live, monkeypatch):
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert 'oracle' not in path.parts
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded)
    p = showcase.LiveMaps(live.rec, {'scenario': str(live.rec / 'scenario.json')},
                          live.robots, live.states)
    p.render(6, (616, 424))


@pytest.fixture
def local_live(tmp_path):
    scenario = dict(search_area=[0, 0, 10, 10], robots=[
        dict(id='robot_A', spawn=dict(at=[3, 4], yaw=90)),
        dict(id='robot_B', spawn=dict(at=[8, 6], yaw=-90))])
    sp = tmp_path / 'scenario.json'
    sp.write_text(json.dumps(scenario))
    meta = dict(robots=['robot_A', 'robot_B'], scenario=str(sp))
    (tmp_path / 'meta.json').write_text(json.dumps(meta))
    config = json.loads((Path(__file__).parents[1] / 'config/brain.json').read_text())
    size, cell = config['grid_size_m'], config['cell_m']
    n = round(size / cell)
    states = {}
    for r in meta['robots']:
        rd = tmp_path / r
        (rd / 'brain_map').mkdir(parents=True)
        grid = np.full((n, n), 128, np.uint8)
        x, y = np.floor((np.array([1., 2.]) + size / 2) / cell + 1e-8).astype(int)
        grid[y, x] = 0
        cv2.imwrite(str(rd / 'brain_map/5.png'), np.flipud(grid))
        (rd / 'brain_map.txt').write_text('5 1 2 0 brain_map/5.png\n')
        (rd / 'lidar.txt').write_text('# beams 4 angle_min_rad 0 increment_rad 1.5707963267948966 mount_height_m .4 mount_forward_m .2 min_m .1 max_m 8\n5 1000 2000 0 1000\n')
        states[r] = [dict(t=0., x=0., y=0., yaw_rad=0., state='explore'),
                     dict(t=5., x=1., y=2., yaw_rad=0., state='explore', path=[[0, 0], [1, 2]], goal=[1, 2])]
        (rd / 'brain_state.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in states[r]))
        event = dict(t=4., kind='frame_linked', reason='robot_B fingerprint', numbers=dict(dx_m=-2, dy_m=5, dyaw_deg=180))
        (rd / 'decisions.jsonl').write_text(json.dumps(event) + '\n' if r == 'robot_A' else '')
    return showcase.LiveMaps(tmp_path, meta, meta['robots'], states)


def test_local_placement_pose_sweep_and_grid(local_live):
    p = local_live
    np.testing.assert_allclose(p.pose('robot_A', 5), [5, 3, -np.pi / 2])
    np.testing.assert_allclose(p.pose('robot_B', 5), [6, 7, np.pi / 2])
    origin, hits = p.sweep('robot_A', 5)
    np.testing.assert_allclose(origin, [5.2, 3])
    np.testing.assert_allclose(hits, [[6.2, 3], [5.2, 5], [5.2, 2]])
    image = p.base(5, (600, 600))
    for xy in [[5.05, 2.95], [5.95, 7.05]]:
        x, y = p.pixels(xy, (600, 600))
        np.testing.assert_array_equal(image[y, x], showcase.TEXT)
    # A missing start transform cannot leave a wall near its local coordinate.
    x, y = p.pixels([1.05, 2.05], (600, 600))
    np.testing.assert_array_equal(image[y, x], showcase.PANEL)
    assert p.render(5, (600, 600)).size == (600, 600)


def test_local_context_transforms_trail_path_goal_and_radio(local_live, tmp_path, monkeypatch):
    from contextpanel import ContextPanel
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert 'oracle' not in path.parts
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    topdown = tmp_path / 'plan.png'
    cv2.imwrite(str(topdown), np.full((280, 400, 3), 100, np.uint8))
    p = ContextPanel(local_live.rec, topdown)
    for r, expected, start in [('robot_A', [5, 3], [3, 4]), ('robot_B', [6, 7], [8, 6])]:
        state = p.state[r][-1]
        np.testing.assert_allclose([state['x'], state['y']], expected)
        np.testing.assert_allclose(state['path'], [start, expected])
        np.testing.assert_allclose(state['goal'], expected)
    p.render(5)
    radio = showcase.Radio(local_live.rec, None, local_live.robots)
    links = [item for item in radio.items if item[2] == 'frame_link']
    assert len(links) == 1
    t, side, kind, text, chips = links[0]
    assert t == 4 and side == 0
    assert text == 'A linked its map with B from matched LiDAR fingerprints'
    assert chips == [('LiDAR frame linked', showcase.GOOD)]
    assert 'error' not in text and 'cm' not in text


def test_display_wrong_sign_and_missing_transform_controls(local_live):
    p = local_live
    sp = p.rec / 'scenario.json'
    scenario = json.loads(sp.read_text())
    for robot in scenario['robots']:
        robot['spawn']['yaw'] *= -1
    sp.write_text(json.dumps(scenario))
    wrong = showcase.LiveMaps(p.rec, {'scenario': str(sp)}, p.robots, p.states)
    assert np.linalg.norm(wrong.pose('robot_A', 5)[:2] - [5, 3]) > 4
    x, y = p.pixels([5.05, 2.95], (600, 600))
    np.testing.assert_array_equal(wrong.base(5, (600, 600))[y, x], showcase.PANEL)
    p.frames['robot_A'] = (np.zeros(3), 'local')
    np.testing.assert_array_equal(p.base(5, (600, 600))[y, x], showcase.PANEL)
