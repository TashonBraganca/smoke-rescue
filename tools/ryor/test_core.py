"""Regression checks for placement, replay integrity, fallbacks and agent order boundaries."""
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/ryor')]
from placement import validate, setup_key
from terra import Worker


@pytest.fixture
def house():
    return json.loads((ROOT / 'data/runs/ryor/gallery.json').read_text())['houses'][0]


@pytest.fixture
def scene():
    return {'people': [[1.75, 3.5], [5.25, 3], [-2.25, 8.25]], 'fire': [-.25, 2.5]}


def test_generator_path_validator_accepts_clear_floor(house, scene):
    assert validate(house, scene) == scene


@pytest.mark.parametrize('change, message', [
    ({'people': [[-3, 3]]}, 'walls'),
    ({'people': [[-2, 2.5]]}, 'furniture'),
    ({'people': [[-2.25, 5.5]]}, 'doorway'),
    ({'people': [[1.75, 3.5], [2, 3.5]]}, 'apart'),
    ({'people': [[float('nan'), 3]]}, 'finite'),
    ({'people': [[float('inf'), 3]]}, 'finite'),
    ({'people': [[1, 2]] * 6}, 'five'),
    ({'fire': [99, 99]}, 'inside'),
])
def test_bad_placements_are_rejected(house, scene, change, message):
    with pytest.raises(ValueError, match=message):
        validate(house, dict(scene, **change))


def test_cache_identity_covers_content_engine_and_setup(scene):
    key = setup_key('family_ranch__r2__medium', scene, 'engine-A')
    assert key == setup_key('family_ranch__r2__medium', json.loads(json.dumps(scene)), 'engine-A')
    assert key != setup_key('family_ranch__r1__medium', scene, 'engine-A')
    assert key != setup_key('family_ranch__r2__medium', scene, 'engine-B')
    assert key != setup_key('family_ranch__r2__medium', dict(scene, fire=[0, 3]), 'engine-A')


@pytest.fixture
def api(monkeypatch, house):
    import boto3
    monkeypatch.setattr(boto3, 'client', lambda *a, **kw: SimpleNamespace())
    for key, val in dict(RYOR_BUCKET='test', RYOR_TABLE='test', RYOR_CLUSTER='test', RYOR_FAMILY='test',
                         RYOR_SUBNETS='test', RYOR_SG='test', RYOR_ENGINE='test').items():
        monkeypatch.setenv(key, val)
    spec = importlib.util.spec_from_file_location('ryor_api_test', ROOT / 'deploy/aws/site_lambda/ryor_api.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, 'gallery', lambda: {'houses': {house['id']: house},
                                                'difficulties': {'easy', 'medium', 'hard'}, 'mission_s': 180})
    return mod


def test_missing_cache_cannot_be_promised(api, monkeypatch):
    monkeypatch.setattr(api, 'ready_cache', lambda key: None)
    code, body = api.post({'body': json.dumps({'house': 'family_ranch', 'robots': 2, 'difficulty': 'medium'})})
    assert code == 409 and json.loads(body)['needs_prepare']
    assert 'mission' not in json.loads(body)


def replay():
    return [{'chunk': n, 't0': n * 10, 'frames': [
        {'t': i / 2, 'robots': [{'id': 'robot_A', 'x': 0, 'y': 2}], 'events': [], 'found': []}
        for i in range(n * 20, min(n * 20 + 20, 361))]} for n in range(19)]


def test_replay_validation_catches_corruption(api, monkeypatch):
    data = replay()
    monkeypatch.setattr(api, 'chunks', lambda *a: data)
    rec, status = {'setup': 'family_ranch__r1__easy'}, {'sim_t': 180, 'chunks': 19}
    assert api.validate(rec, status, '') is None
    data[0]['frames'].pop(2)
    assert 'missing' in api.validate(rec, status, '')
    data = replay(); data[0]['frames'][0]['robots'][0]['x'] = 100
    assert 'outside' in api.validate(rec, status, '')
    data = replay(); data[0]['frames'][0]['found'] = [{'by': 'robot_A', 't': 0}]
    assert 'before confirmation' in api.validate(rec, status, '')
    data = replay(); data[0]['frames'][0]['robots'][0]['x'] = float('nan')
    assert 'outside' in api.validate(rec, status, '')


@pytest.mark.parametrize('fid, expected', [(7, 'explore_frontier'), (999, 'unavailable')])
def test_terra_can_only_order_an_observed_frontier(tmp_path, monkeypatch, fid, expected):
    worker = Worker(tmp_path, 'test', 'test', None)
    monkeypatch.setattr(worker, 'reserve', lambda: True)
    reply = SimpleNamespace(tool_calls=[SimpleNamespace(name='explore_frontier', arguments={'frontier_id': fid,
                           'reason': 'Room 3 has unsearched floor and no peer.'})], raw={}, latency_s=.1)
    monkeypatch.setattr(worker, 'load_backend', lambda: SimpleNamespace(chat=lambda *a: reply))
    obs = {'robot': 'robot_A', 't': 15, 'frontiers': [{'id': 7, 'x': 2, 'y': 5}]}
    worker.decide(obs)
    answer = json.loads((tmp_path / 'robot_A.order.json').read_text())
    assert answer['type'] == expected
    if expected == 'explore_frontier':
        assert [answer['x'], answer['y']] == [2, 5]


def test_terra_budget_refusal_never_calls_model(tmp_path, monkeypatch):
    worker = Worker(tmp_path, 'test', 'test', None)
    monkeypatch.setattr(worker, 'reserve', lambda: False)
    monkeypatch.setattr(worker, 'load_backend', lambda: pytest.fail('Budget refusal must not call the model'))
    worker.decide({'robot': 'robot_A', 't': 15, 'frontiers': []})
    answer = json.loads((tmp_path / 'robot_A.order.json').read_text())
    assert answer['type'] == 'unavailable' and 'budget' in answer['reason']
