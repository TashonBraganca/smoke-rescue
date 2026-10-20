"""Synthetic metric geometry and negative controls for the C# port."""
import json
from pathlib import Path

import numpy as np
import pytest

from brain.agent.vision import Frame
from brain import warm
from brain.warm_eval import predictions, score

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT/'config/brain.json').read_text())
MOUNT = dict(height_m=1.6, forward_offset_m=.25, pitch_down_deg=10.)


def scene(kind='person'):
    h, w = 128, 160
    k = np.array([[120., 0, 79.5], [0, 120., 63.5], [0, 0, 1.]])
    depth = np.full((h, w), 3.8, np.float32)
    temp = np.full((h, w), 24., np.float32)
    yy, xx = np.indices((h, w))
    top = 30 if kind == 'high' else 80
    mask = (yy >= top) & (yy < top+22) & (xx >= 60) & (xx < 100)
    # A curved object stands in front of the background.
    depth[mask] = 3.0-.22*np.sin((xx[mask]-60)/39*np.pi)*np.sin((yy[mask]-top)/21*np.pi)
    temp[mask] = 35
    if kind == 'wall':
        depth[:] = 3.
    elif kind == 'low_contrast':
        temp[~mask] = 34.8
        # Keep the background out of the absolute band without inventing a cool rim.
        # Invalid background depth leaves temperature observable but prevents merging.
        depth[~mask] = 0
        # Supply distant valid ring samples separated by >0.5 m.
        depth[(~mask) & (xx % 2 == 0)] = 4.0
    elif kind == 'window':
        depth[~mask] = 1.8
    elif kind == 'floor':
        p = np.radians(MOUNT['pitch_down_deg'])
        depth = (MOUNT['height_m']/(np.sin(p)+(yy-k[1, 2])/k[1, 1]*np.cos(p))).astype(np.float32)
        depth[depth <= 0] = 0
    elif kind == 'saturated':
        temp[:] = 90.
    elif kind == 'missing':
        depth[:] = 0
    return Frame(t=0, rgb=np.zeros((h, w, 3), np.uint8), depth_m=depth,
                 thermal=np.rint((temp+273.15)*10).astype(np.uint16), K=k, thermal_K=k,
                 rgb_to_thermal=np.eye(4), stamps=dict(rgb=0., depth=0., thermal=0.))


def component(kind):
    blobs = warm.detect(scene(kind), CONFIG, MOUNT)
    assert len(blobs) == 1
    return blobs[0]


def test_person_kept_by_every_rule():
    assert all(warm.gates(component('person')).values())


@pytest.mark.parametrize(('kind', 'rule'), [('high', 'height'), ('wall', 'separate'), ('window', 'separate'), ('low_contrast', 'contrast')])
def test_target_rule_and_removal_negative_control(kind, rule):
    gates = warm.gates(component(kind))
    assert not gates[rule]
    assert all(value for name, value in gates.items() if name != rule), gates
    # Removing this rule admits the target false object; the complete gate rejects it.
    assert not all(gates.values())
    assert all(warm.gates(component('person')).values())


def test_heated_floor_height_rejects():
    blob = component('floor')
    assert abs(blob['height_m'][1]) < 1e-5
    assert not warm.gates(blob)['height']
    assert warm.gates(blob, {'person_floor_clearance_m': -1e-4})['height']


@pytest.mark.parametrize('kind', ['missing', 'saturated'])
def test_rules_do_not_claim_recall_without_an_observable_blob(kind):
    assert warm.detect(scene(kind), CONFIG, MOUNT) == []


def test_depth_edges_split_four_connected_blobs():
    frame = scene('wall')
    frame.depth_m[:, 80:] += 1.
    c = CONFIG | {'person_min_area_m2': .01}
    assert len(warm.detect(frame, c, MOUNT)) == 2
    assert len(warm.detect(frame, c | {'blob_depth_jump_m': 2.}, MOUNT)) == 1


def test_no_diagonal_connectivity():
    frame = scene('wall')
    frame.thermal[:] = 0
    frame.thermal[80, 79] = frame.thermal[81, 80] = 3082
    c = CONFIG | {'person_min_pixels': 1, 'person_min_area_m2': 0.}
    assert len(warm.detect(frame, c, MOUNT)) == 2


def test_projection_uses_mount_and_ignores_translation_like_csharp():
    f = scene()
    _, valid, p, u, v = warm.matched_geometry(f, MOUNT)
    y, x = 90, 70
    z = f.depth_m[y, x]
    angle = np.radians(10)
    down = (y-63.5)/120*z
    assert p[y, x] == pytest.approx([(x-79.5)/120*z, .25+z*np.cos(angle)-down*np.sin(angle),
                                    1.6-z*np.sin(angle)-down*np.cos(angle)], abs=1e-6)
    assert valid[y, x] and (u[y, x], v[y, x]) == (x, y)
    original = warm.detect(f, CONFIG, MOUNT)[0]['xy']
    f.rgb_to_thermal[0, 3] = -1.
    assert warm.detect(f, CONFIG, MOUNT)[0]['xy'] == pytest.approx(original)
    moved = warm.detect(f, CONFIG, MOUNT, (2., 3., np.pi/2))[0]['xy']
    assert moved == pytest.approx([2-original[1], 3+original[0]])


def test_empty_predictions_have_undefined_precision_and_zero_recall():
    rows = [dict(label=True, combined=False), dict(label=False, combined=False)]
    assert score(rows, 'combined') == dict(tp=0, fp=0, fn=1, positives=1, precision=None, recall=0.)


def test_sensor_module_has_no_truth_access():
    from test_oracle_isolation import scan_strict_source
    assert not scan_strict_source((ROOT/'brain/brain/warm.py').read_text())


@pytest.mark.parametrize(('kind', 'rule'), [('high', 'height'), ('wall', 'separate'), ('window', 'separate'), ('low_contrast', 'contrast')])
def test_filter_api_with_real_ablation(kind, rule):
    assert not warm.detect_filtered(scene(kind), CONFIG, MOUNT)
    other = tuple(r for r in ('height', 'separate', 'contrast') if r != rule)
    assert len(warm.detect_filtered(scene(kind), CONFIG, MOUNT, rules=other)) == 1
    assert len(warm.detect_filtered(scene(), CONFIG, MOUNT)) == 1


def test_graph_components_equal_independent_queue_reference():
    from collections import deque
    rng = np.random.default_rng(12)
    frame = scene('wall')
    frame.thermal[:] = np.where(rng.random(frame.thermal.shape) < .45, 3082, 2900)
    frame.depth_m[:] = rng.choice([0., 2., 2.3, 4.], frame.depth_m.shape)
    z, valid, _, _, _ = warm.matched_geometry(frame, MOUNT)
    good = valid & (frame.thermal == 3082)
    seen = set(); expected = []
    for y, x in zip(*np.nonzero(good)):
        if (y, x) in seen:
            continue
        q = deque([(y, x)]); seen.add((y, x)); blob = set()
        while q:
            a, b = q.popleft(); blob.add(a*good.shape[1]+b)
            for c, d in [(a-1, b), (a+1, b), (a, b-1), (a, b+1)]:
                if (0 <= c < good.shape[0] and 0 <= d < good.shape[1] and good[c, d]
                        and (c, d) not in seen and abs(z[a, b]-z[c, d]) <= CONFIG['blob_depth_jump_m']):
                    seen.add((c, d)); q.append((c, d))
        expected.append(blob)
    config = CONFIG | {'person_min_pixels': 1, 'person_min_area_m2': 0., 'person_max_area_m2': 100.}
    actual = warm.detect(frame, config, MOUNT, features=False)
    assert [set(np.flatnonzero(c['mask'])) for c in actual] == expected


def test_capture_pose_replays_turn_instead_of_interpolating_it(tmp_path):
    from brain.warm_eval import capture_pose
    states = [dict(t=0., x=0., y=0., yaw_rad=0.), dict(t=.5, x=0., y=0., yaw_rad=0.)]
    np.savetxt(tmp_path/'odom.txt', [[.1, 0, 0, 0], [.2, 0, 0, 0], [.5, 0, 0, 0]])
    # Gyro y is opposite to brain yaw. The robot turns left then right.
    np.savetxt(tmp_path/'imu.txt', [[.1, 0, 0, 0, 0, -1., 0], [.2, 0, 0, 0, 0, -1., 0], [.5, 0, 0, 0, 0, 2/3, 0]])
    pose, method = capture_pose(tmp_path, states, .2)
    assert pose[2] == pytest.approx(.2)
    assert 'replay' in method


def test_incomplete_run_is_reported_without_reading_images(tmp_path, monkeypatch):
    from brain import warm_eval
    monkeypatch.setattr(warm_eval, 'ROOT', tmp_path)
    assert warm_eval.main(['--recordings', 'pending', '--out', str(tmp_path/'out')]) == 0
    summary = json.loads((tmp_path/'out/report.json').read_text())
    assert summary['frames_evaluated'] == 0
    assert summary['skipped'][0]['run'] == 'pending'


def test_relative_segmentation_can_find_cool_person_outside_absolute_band():
    frame = scene()
    frame.thermal[:] = np.where(frame.thermal > 3000, 3302, 3532).astype(np.uint16)
    # A 57 C apparent body against 80 C surroundings lies outside the legacy band.
    assert not warm.detect(frame, CONFIG, MOUNT)
    blobs = warm.detect(frame, CONFIG, MOUNT, segmentation='contrast')
    assert any(np.linalg.norm(c['xy']-component('person')['xy']) < .2 for c in blobs)


def test_evaluator_places_local_event_and_reads_truth_only_in_evaluator(tmp_path, monkeypatch):
    import cv2
    import yaml
    from brain import warm_eval
    from brain.nav_io import transform_xy
    rec = tmp_path/'recording'; folder = rec/'robot_A'; folder.mkdir(parents=True)
    frame = scene(); blob = component('person')
    for sensor, image in [('rgb', frame.rgb), ('depth', np.rint(frame.depth_m*1000).astype(np.uint16)), ('thermal', frame.thermal)]:
        (folder/sensor).mkdir()
        assert cv2.imwrite(str(folder/sensor/'0.000000.png'), image)
        (folder/(sensor+'.txt')).write_text(f'0.000000 {sensor}/0.000000.png\n')
    camera = {'Camera1.fx':120., 'Camera1.fy':120., 'Camera1.cx':79.5, 'Camera1.cy':63.5,
              'Camera.width':160, 'Camera.height':128, 'Camera.fps':15., 'RGBD.DepthMapFactor':1000.}
    (folder/'camera.yaml').write_text(yaml.safe_dump(camera))
    (folder/'thermal.yaml').write_text(yaml.safe_dump(camera | {'T_rgb_thermal.translation_m':[.03, 0., 0.]}))
    states = [dict(t=0., x=0., y=0., yaw_rad=0.), dict(t=.5, x=0., y=0., yaw_rad=0.)]
    (folder/'brain_state.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in states))
    event = dict(t=0., kind='survivor_candidate', numbers=dict(x_m=float(blob['xy'][0]), y_m=float(blob['xy'][1]), area_m2=blob['area_m2']))
    (folder/'decisions.jsonl').write_text(json.dumps(event)+'\n')
    (rec/'meta.json').write_text(json.dumps(dict(robots=['robot_A'], camera_mount=MOUNT)))
    scenario = {'robots':[{'id':'robot_A', 'spawn':{'at':[5., 6.], 'yaw':90.}}]}
    monkeypatch.setattr(warm_eval, 'load_scenario', lambda meta: scenario)
    xy = transform_xy(blob['xy'], [5., 6., -np.pi/2])
    truth = rec/'oracle'; truth.mkdir()
    (truth/'survivors.json').write_text(json.dumps([dict(id='s1', position_house=[xy[0], 0., xy[1]], pose='lying_on_side')]))
    (truth/'floorplan_used.json').write_text('{"rooms":[], "doors":[]}')
    report, rows = warm_eval.evaluate_run(rec, tmp_path/'out', event_only=True)
    result = report['events'][0]
    assert result['building_xy'] == pytest.approx(xy)
    assert result['matched'] and result['real']
    assert result['rules']['combined']
    assert report['survivor_candidate']['precision'] == 1.
    assert (tmp_path/'out/candidates.png').exists()
    assert (tmp_path/'out/events.csv').exists()
    assert not rows


def test_proposed_keys_have_reasons_and_config_overrides_work():
    assert set(warm.PROPOSED) == set(warm.PROPOSED_REASONS)
    assert not warm.detect_filtered(scene('high'), CONFIG, MOUNT)
    assert len(warm.detect_filtered(scene('high'), CONFIG | {'person_top_max_m': 3.}, MOUNT)) == 1


def test_unknown_background_cannot_confirm_an_object():
    frame = scene()
    frame.depth_m[frame.thermal < 3000] = 0
    blob = warm.detect(frame, CONFIG, MOUNT)[0]
    assert blob['ring_plane_support'] is None
    assert not warm.gates(blob)['separate']


def test_recess_boundary_is_rejected_inclusive():
    blob = component('person')
    blob['ring_depth_delta_m'] = -.8
    assert not warm.gates(blob)['separate']
    blob['ring_depth_delta_m'] = -.799
    assert warm.gates(blob)['separate']


def test_match_rounds_half_pixels_to_even():
    frame = scene('wall')
    # Thermal x=0 maps to RGB u=0.5. Mathf.RoundToInt uses the even integer.
    frame.thermal_K[0, 2] = 0.
    frame.K[0, 2] = .5
    frame.thermal_K[1, 2] = 0.
    frame.K[1, 2] = .5
    _, _, _, u, v = warm.matched_geometry(frame, MOUNT)
    assert list(u[0, :4]) == [0, 2, 2, 4]
    assert list(v[:4, 0]) == [0, 2, 2, 4]


def test_candidate_persistence_is_not_person_evidence():
    tracker = warm.CandidateTracker(CONFIG)
    blob = dict(xy=np.array([1., 2.]), area_m2=.3)
    assert tracker.observe(0., [blob, blob]) == []
    assert tracker.observe(.1, [blob]) == []
    events = tracker.observe(.2, [blob])
    assert len(events) == 1 and events[0]['frames'] == 3
    assert tracker.observe(.3, [blob]) == []
    # A stationary non-person receives exactly the same votes.
    assert len(warm.CandidateTracker(CONFIG).tracks) == 0


def test_candidate_votes_expire_and_confirmed_positions_do_not_vote():
    tracker = warm.CandidateTracker(CONFIG)
    blob = dict(xy=np.array([1., 2.]), area_m2=.3)
    tracker.observe(0., [blob]); tracker.observe(.1, [blob])
    assert tracker.observe(2., [blob]) == []
    assert len(tracker.tracks[0]['observations']) == 1
    tracker.confirmed_positions.append(blob['xy'])
    assert tracker.observe(2.1, [blob]) == []
    assert len(tracker.tracks[0]['observations']) == 1
