"""Team-map export, evaluator geometry, and viewer provenance checks."""
import gzip
import importlib.util
from pathlib import Path
import struct
import base64
from dataclasses import replace

import numpy as np
import pytest


def write_grid(path, occupied=.9, searched=.8, at=5):
    arrays = [np.array([occupied, .1, .5, .5], '<f4'), np.array([0, searched, 0, 0], '<f4'),
              np.array([1, 2, 0, 0], '<u4'), np.array([0, 2, 0, 0], '<u4'),
              np.array([0, 2, -1, -1], '<f4'), np.array([-1, 2, -1, -1], '<f4')]
    raw = b'TMAP' + struct.pack('<BifffdB', 1, 2, .2, 0, 0, at, 2)
    raw += struct.pack('<H', 1) + b'A' + struct.pack('<H', 1) + b'B'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(raw + b''.join(a.tobytes() for a in arrays)))
    return path


def test_export_reader_keeps_probabilities_sources_ages_and_row_order(tmp_path):
    from brain.team_map import load_team_map
    grid = load_team_map(write_grid(tmp_path / 'map.gz'))
    assert grid.sources == ('A', 'B')
    assert grid.occupancy[0, 0] == pytest.approx(.9)
    assert grid.searched_source_mask[0, 1] == 2
    assert grid.search_age_s[0, 1] == 2
    assert grid.occupancy[1, 0] == .5
    with pytest.raises(ValueError):
        (tmp_path / 'bad.gz').write_bytes(gzip.compress(b'TMAP\x02'))
        load_team_map(tmp_path / 'bad.gz')


def test_latest_export_never_uses_a_future_or_unfinished_snapshot(tmp_path):
    from brain.team_map import latest_team_map
    rd = tmp_path / 'robot_A'
    write_grid(rd / 'brain_map/5.000000.team.bin.gz', at=5)
    write_grid(rd / 'brain_map/10.000000.team.bin.gz', at=10)
    assert latest_team_map(rd, 4) is None
    assert latest_team_map(rd, 9).at == 5
    (rd / 'brain_map/10.000000.team.bin.gz').write_bytes(b'partial')
    assert latest_team_map(rd, 11).at == 5


def test_team_accuracy_measures_wall_distance_and_camera_searched_floor(tmp_path):
    from brain.team_map import load_team_map
    from brain.nav_eval import team_map_accuracy
    grid = load_team_map(write_grid(tmp_path / 'map.gz'))
    plan = dict(rooms=[dict(id='room', rect=[0, 0, 2, 2])], doors=[])
    points = np.array([[0, -.5, 0], [0, -.5, 1], [2, -.5, 0], [2, -.5, 1]], float)
    measured = team_map_accuracy(grid, np.zeros(3), plan, {}, points)
    assert measured['occupied_within_0_2_m_wall_share'] == 1
    assert measured['searched_floor_m2'] == pytest.approx(.04)
    shifted = team_map_accuracy(grid, np.array([.7, .7, 0]), plan, {}, points)
    assert shifted['occupied_within_0_2_m_wall_share'] == 0
    plan['doors'] = [dict(wall='z', at=[.1, 0], width=1)]
    assert team_map_accuracy(grid, np.zeros(3), plan, {}, points)['occupied_within_0_2_m_wall_share'] == 1  # west wall still nearby
    plan['doors'] = [dict(wall='z', at=[1, 0], width=.8)]
    at_door = replace(grid, origin=np.array([.9, 0]))
    assert team_map_accuracy(at_door, np.zeros(3), plan, {}, points)['occupied_within_0_2_m_wall_share'] == 0


def test_shared_pad_frame_is_reported_without_a_discovery_event():
    from brain.nav_eval import frame_link_report
    scenario = dict(robots=[dict(id='A', launch_pad=[0, 0, 0]), dict(id='B', launch_pad=[0, -1.2, 0])])
    linked = frame_link_report([], 'A', ['A', 'B'], scenario)
    assert linked['status'] == 'shared_at_launch'
    assert linked['reported_transform'] == dict(dx_m=0., dy_m=0., dyaw_deg=0.)
    assert frame_link_report([], 'A', ['A', 'B'], {})['status'] == 'not_linked'


def load_showcase():
    spec = importlib.util.spec_from_file_location('showcase', Path(__file__).parents[2] / 'tools/showcase.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_viewer_shades_exported_confidence_and_ignores_legacy_png(tmp_path):
    from brain.team_map import load_team_map
    module = load_showcase()
    low = load_team_map(write_grid(tmp_path / 'low.gz', occupied=.55, searched=.1))
    high = load_team_map(write_grid(tmp_path / 'high.gz', occupied=.95, searched=.9))
    assert np.linalg.norm(module.team_map_rgb(high)[0, 0]) > np.linalg.norm(module.team_map_rgb(low)[0, 0])
    assert np.linalg.norm(module.team_map_rgb(high)[0, 1]) > np.linalg.norm(module.team_map_rgb(low)[0, 1])
    assert np.array_equal(module.team_map_rgb(high)[1, 0], module.PANEL)


def test_live_maps_uses_one_receivers_export_without_reading_binary_pngs(tmp_path, monkeypatch):
    from brain.team_map import load_team_map
    module = load_showcase()
    robot = 'robot_A'
    path = write_grid(tmp_path / robot / 'brain_map/5.000000.team.bin.gz')
    rd = tmp_path / robot
    (rd / 'brain_map.txt').write_text('5 0 0 0 brain_map/5.000000.png\n')
    (rd / 'brain_map/5.000000.png').write_bytes(b'legacy data must not colour the confidence map')
    monkeypatch.setattr(module, 'load_scenario', lambda _: dict(search_area=[0, 0, .4, .4],
                        robots=[dict(id=robot, launch_pad=[0, 0, 0], spawn=dict(at=[0, 0], yaw=0))]))
    states = {robot: [dict(t=0, x=0, y=0, yaw_rad=0)]}
    live = module.LiveMaps(tmp_path, {}, [robot], states)
    live.bounds = np.array([0, 0, .4, .4])
    def forbidden(*args, **kwargs):
        raise AssertionError('legacy PNG was read while a team export was available')
    monkeypatch.setattr(module.cv2, 'imread', forbidden)
    image = live.base(5, (2, 2))
    assert np.array_equal(image, np.flipud(module.team_map_rgb(load_team_map(path))))
    assert "robot_A's team belief" in live.belief_caption(5)


def test_headless_evaluator_scores_legacy_and_export_with_the_same_wall_rule(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parents[2] / 'tools'))
    from headless_report import headless_team_maps
    snapshot = dict(robot='A', grid_size=2, cell_m=.2, origin=[0, 0],
                    occupancy=base64.b64encode(bytes([255, 1, 0, 0])).decode(),
                    searched=base64.b64encode(bytes([0, 1, 0, 0])).decode())
    world = dict(rooms=[dict(poly=[[0, 0], [2, 0], [2, 2], [0, 2]])], prisms=[], segments=[[0, 0, 0, 2, 0, 2.8]])
    result = dict(maps=[snapshot], duration_s=5)
    old = headless_team_maps(tmp_path, result, world)['A']
    write_grid(tmp_path / 'A.team.bin.gz')
    new = headless_team_maps(tmp_path, result, world)['A']
    assert old['occupied_within_0_2_m_wall_share'] == new['occupied_within_0_2_m_wall_share'] == 1
    assert old['searched_floor_m2'] == pytest.approx(new['searched_floor_m2'])
    assert old['representation'] == 'legacy_binary_no_confidence_or_age'
    assert new['representation'] == 'confidence_export'
