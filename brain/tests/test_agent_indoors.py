"""Direction checks use sensor images; only these evaluator tests read labels."""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from brain.agent import tools as T
from brain.agent import vision as v

ROOT = Path(__file__).resolve().parents[2]
RECORDING = ROOT / 'data/recordings/unity_house_hidden_run1'


def room_frame(kind='room', ceiling=3., t=0., back_m=7., side_m=2.):
    """Intersect optical rays with a floor, ceiling and three walls."""
    k = np.array([[415.692, 0, 319.5], [0, 415.692, 239.5], [0, 0, 1.]])
    yy, xx = np.indices((480, 640))
    pitch = math.radians(10)
    right = (xx - 319.5) / 415.692
    up = -math.sin(pitch) - (yy - 239.5) / 415.692 * math.cos(pitch)
    forward = math.cos(pitch) - (yy - 239.5) / 415.692 * math.sin(pitch)
    with np.errstate(divide='ignore', invalid='ignore'):
        floor = np.where(up < 0, -1.6 / up, np.inf)
        roof = np.where(up > 0, (ceiling - 1.6) / up, np.inf)
        sides = np.where(right != 0, side_m / np.abs(right), np.inf)
        back = back_m / forward
    depth = floor if kind == 'sky' else np.minimum.reduce([floor, roof, sides, back])
    if kind == 'canopy':
        depth = np.minimum(floor, roof)
    depth = np.where(depth < 20., depth, 0.).astype(np.float32)
    if kind == 'facade':
        depth[:] = 1.
    if kind == 'missing':
        depth[:] = 0.
    return v.Frame(t=t, rgb=np.full((480, 640, 3), 100, np.uint8), depth_m=depth,
                   thermal=np.full((480, 640), 2932, np.uint16), K=k, thermal_K=k,
                   rgb_to_thermal=np.eye(4), stamps={'rgb': t, 'depth': t, 'thermal': t})


def test_horizontal_ceiling_height_walls_and_debug_evidence(tmp_path):
    f = room_frame()
    r = v.check_indoors(f, 0., output_dir=tmp_path)
    assert r['verdict'] == 'indoors'
    assert r['ceiling']['height_above_floor_m'] == pytest.approx(3., abs=.02)
    assert r['ceiling']['upward_hit_fraction'] > .1
    assert r['ceiling']['range_m'] > 0
    assert len(r['walls']) == 3 and any(w['hit_fraction'] > 0 for w in r['walls'])
    assert {'id', 'tool', 't', 'evidence', 'reasons'} <= r.keys()
    assert cv2.imread(r['debug_crop']).shape[:2] == f.depth_m.shape
    assert r == v.check_indoors(f, 0., output_dir=tmp_path)
    json.dumps(r, allow_nan=False)


def test_sky_needs_lower_returns_and_facade_is_not_indoor(tmp_path):
    sky = v.check_indoors(room_frame('sky'), 0., output_dir=tmp_path)
    assert sky['verdict'] == 'outdoors'
    assert sky['void']['upward_no_return_fraction'] == 1.
    assert sky['void']['lower_return_fraction'] > .5
    assert v.check_indoors(room_frame('missing'), 0., output_dir=tmp_path)['verdict'] == 'unknown'
    assert v.check_indoors(room_frame('facade'), 0., output_dir=tmp_path)['verdict'] == 'unknown'


def test_canopy_without_walls_is_unknown(tmp_path):
    r = v.check_indoors(room_frame('canopy'), 0., output_dir=tmp_path)
    assert r['ceiling']['upward_hit_fraction'] > .1
    assert r['verdict'] == 'unknown'


def test_tall_ceiling_has_no_height_band_and_sensor_range_is_respected(tmp_path):
    sensors = json.loads((ROOT / 'config/sensors.json').read_text())
    sensors['depth']['max_m'] = 20.
    f = room_frame(ceiling=6., back_m=18., side_m=1.5)
    r = v.check_indoors(f, 0., output_dir=tmp_path, sensors=sensors)
    assert r['verdict'] == 'indoors'
    assert r['ceiling']['height_above_floor_m'] == pytest.approx(6., abs=.02)
    assert v.check_indoors(f, 0., output_dir=tmp_path)['verdict'] == 'unknown'
    assert r['id'] != v.check_indoors(f, 0., output_dir=tmp_path)['id']


@pytest.mark.parametrize('bearing', [90., -90., 180.])
def test_unseen_bearing_is_unknown(bearing, tmp_path):
    assert v.check_indoors(room_frame(), bearing, output_dir=tmp_path)['verdict'] == 'unknown'


@pytest.mark.parametrize('bearing', [float('nan'), float('inf')])
def test_nonfinite_bearing_rejected(bearing, tmp_path):
    with pytest.raises(ValueError, match="bearing must be finite"):
        v.check_indoors(room_frame(), bearing, output_dir=tmp_path)


def test_verdict_thresholds_come_from_config(tmp_path):
    f = room_frame()
    config = json.loads((ROOT / 'config/perception.json').read_text())
    assert v.check_indoors(f, 0., output_dir=tmp_path)['verdict'] == 'indoors'
    config['min_ceiling_upward_fraction'] = 1.
    assert v.check_indoors(f, 0., output_dir=tmp_path, config=config)['verdict'] != 'indoors'
    for key, value in config.items():
        if isinstance(value, (int, float)):
            assert config[key + '_reason']


@pytest.mark.parametrize('robot', ['A', 'B'])
@pytest.mark.parametrize('bearing', [-35., -30., 0., 30., 35.])
def test_recorded_outside_doorway_and_facade(robot, bearing, tmp_path):
    # Oracle labels camera location only. An outside camera can look indoors.
    oracle = RECORDING / 'oracle'
    pose = np.loadtxt(oracle / f'robot_{robot}_groundtruth.txt')[0]
    x0, z0, x1, z1 = json.loads((oracle / 'floorplan_used.json').read_text())['footprint']
    assert not (x0 <= pose[1] <= x1 and z0 <= pose[3] <= z1)
    r = v.check_indoors(v.load_frame(RECORDING / f'robot_{robot}', 0.), bearing, output_dir=tmp_path)
    if bearing == 0.:
        assert r['verdict'] == 'indoors'
        assert r['ceiling']['height_above_floor_m'] == pytest.approx(2.8, abs=.05)
    else:
        assert r['verdict'] == 'unknown'
        assert r['walls'][1]['hit_fraction'] > .5


@pytest.mark.parametrize('robot', ['A', 'B'])
@pytest.mark.parametrize('t', [60., 70., 80., 90., 100., 110., 120.])
def test_recorded_interval_labels_and_verdicts(robot, t, tmp_path):
    oracle = RECORDING / 'oracle'
    poses = np.loadtxt(oracle / f'robot_{robot}_groundtruth.txt')
    pose = poses[np.argmin(abs(poses[:, 0] - t))]
    x0, z0, x1, z1 = json.loads((oracle / 'floorplan_used.json').read_text())['footprint']
    inside = x0 <= pose[1] <= x1 and z0 <= pose[3] <= z1
    assert inside == (robot == 'B')
    r = v.check_indoors(v.load_frame(RECORDING / f'robot_{robot}', t), 0., output_dir=tmp_path)
    # Walls fill the view at A60 and B60/80/90/100/110; absence of a ceiling proves nothing.
    expected = 'indoors' if (robot == 'A' and t != 60.) or (robot == 'B' and t in [70., 120.]) else 'unknown'
    assert r['verdict'] == expected
    if expected == 'indoors':
        assert r['ceiling']['height_above_floor_m'] == pytest.approx(2.8, abs=.15)


def registry(tmp_path, monkeypatch, *, proven=False, heading=90.):
    ctx = T.AgentContext('robot_A', 0., 4., 5., heading,
                         frontiers=[{'id': 3, 'x': 8., 'y': 5., 'proven_indoors': proven}])
    reg = T.ToolRegistry(tmp_path, ctx, tmp_path / 'debug')
    monkeypatch.setattr(reg, '_frame', lambda: room_frame())
    return reg


def test_unproven_frontier_requires_evidence(tmp_path, monkeypatch):
    reg = registry(tmp_path, monkeypatch)
    r = reg.call('explore_frontier', {'frontier_id': 3, 'reason': 'search'})
    assert r.get('refused') and not reg.orders


@pytest.mark.parametrize('kind', ['sky', 'missing'])
def test_outdoors_or_unknown_frontier_evidence_refused(kind, tmp_path, monkeypatch):
    reg = registry(tmp_path, monkeypatch)
    monkeypatch.setattr(reg, '_frame', lambda: room_frame(kind))
    ev = reg.call('check_indoors', {'bearing_deg': 0.})
    assert ev.get('verdict') in {'outdoors', 'unknown'}
    r = reg.call('explore_frontier', {'frontier_id': 3, 'reason': 'search', 'evidence_id': ev['id']})
    assert r.get('refused') and not reg.orders


@pytest.mark.parametrize('bearing,allowed', [(0., True), (5., True), (5.1, False), (-5.1, False), (20., False)])
def test_indoor_frontier_requires_matching_bearing(bearing, allowed, tmp_path, monkeypatch):
    reg = registry(tmp_path, monkeypatch)
    ev = reg.call('check_indoors', {'bearing_deg': bearing})
    assert ev.get('verdict') == 'indoors'
    r = reg.call('explore_frontier', {'frontier_id': 3, 'reason': 'search', 'evidence_id': ev['id']})
    assert bool(r.get('ok')) == allowed
    if allowed:
        assert reg.orders[0]['unproven'] is True
        assert reg.orders[0]['evidence_id'] == ev['id']
        assert reg.orders[0]['evidence_tool'] == 'check_indoors'
        assert (reg.orders[0]['x'], reg.orders[0]['y']) == (8., 5.)
    else:
        assert r.get('refused') and not reg.orders


@pytest.mark.parametrize('proven', [True, 'absent'])
def test_proven_frontiers_unchanged(proven, tmp_path, monkeypatch):
    reg = registry(tmp_path, monkeypatch, proven=True)
    if proven == 'absent':
        del reg.ctx.frontiers[0]['proven_indoors']
    assert reg.call('explore_frontier', {'frontier_id': 3, 'reason': 'search'})['ok']
    assert 'unproven' not in reg.orders[0]


def test_frontier_cannot_cite_another_tool_or_another_step(tmp_path, monkeypatch):
    reg = registry(tmp_path, monkeypatch)
    reg.evidence['other'] = {'id': 'other', 'tool': 'assess_visibility', 'verdict': 'indoors'}
    for evidence_id in ['other', 'invented']:
        assert reg.call('explore_frontier', {'frontier_id': 3, 'reason': 'search', 'evidence_id': evidence_id}).get('refused')
    assert not reg.orders


def test_perception_schema_exposes_bearing_and_optional_action_evidence():
    by_name = {t['function']['name']: t['function'] for t in T.ALL_TOOLS}
    assert by_name['check_indoors']['parameters']['required'] == ['bearing_deg']
    action = by_name['explore_frontier']['parameters']
    assert 'evidence_id' in action['properties'] and 'evidence_id' not in action['required']
