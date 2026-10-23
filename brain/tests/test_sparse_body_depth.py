"""Missing rays must not erase measured body shape or become invented body points."""
import json
from copy import copy
from pathlib import Path
import numpy as np
import pytest
from brain.agent import vision as V
from test_agent_vision import scene


def test_distributed_camera_points_support_shape_below_fraction_gate(tmp_path, monkeypatch):
    f = scene()
    # Each fifth source ray survives, in every part of the body. No range is invented.
    yy, xx = np.indices(f.depth_m.shape)
    f.depth_m[(xx + 2 * yy) % 5 != 0] = 0
    monkeypatch.setattr(V, 'detect_people_rgb', lambda frame: [])
    r = V.inspect_candidate(f, 0, 3, output_dir=tmp_path)
    assert r['cues']['depth']['registered_fraction'] < .6
    assert r['verdict'] == 'person'
    assert r['cues']['depth']['camera_support']['sufficient']
    assert r['cues']['range_source'] == 'depth'


def test_a_sparse_planar_patch_stays_unconfirmed(tmp_path, monkeypatch):
    f = scene('wall')
    yy, xx = np.indices(f.depth_m.shape)
    f.depth_m[(xx + 2 * yy) % 5 != 0] = 0
    monkeypatch.setattr(V, 'detect_people_rgb', lambda frame: [])
    assert V.inspect_candidate(f, 0, 3, output_dir=tmp_path)['verdict'] != 'person'


def test_depth_in_only_one_part_cannot_certify_the_whole_warm_body(tmp_path, monkeypatch):
    f = scene()
    f.depth_m[:60] = 0
    monkeypatch.setattr(V, 'detect_people_rgb', lambda frame: [])
    r = V.inspect_candidate(f, 0, 3, output_dir=tmp_path)
    assert r['verdict'] == 'uncertain'


def test_recorded_bed_view_uses_camera_shape_without_bedside_lidar(tmp_path, monkeypatch):
    rec = Path(__file__).resolve().parents[2] / 'data/recordings/unity_house_onedoor_seed929_run1/robot_A'
    if not rec.exists(): pytest.skip('local recording absent')
    monkeypatch.setattr(V, 'detect_people_rgb', lambda frame: [])
    for scan in (True, False):
        f = copy(V.load_frame(rec, 38.21666666666667))
        if not scan: f.lidar = None
        r = V.inspect_candidate(f, 9.8, 3.16, output_dir=tmp_path)
        assert r['verdict'] == 'person'
        assert r['candidate']['range_m'] == pytest.approx(3.264, abs=.002)
        assert r['cues']['range_source'] == 'depth'
        assert r['cues']['depth']['registered_fraction'] < .6
        assert r['cues']['depth']['camera_support']['sufficient']
