"""Synthetic checks for localization, unique recall and candidate voting."""
import numpy as np
import pytest
from brain import rgb_people_eval as E


def detection(conf=.5, z=3., bearing=0.):
    return dict(confidence=conf, range_m=z, bearing_deg=bearing, bbox_rgb=[40, 40, 20, 20])


def test_projection_uses_optical_depth_and_vertical_box_center():
    k = np.array([[100., 0, 50], [0, 100., 50], [0, 0, 1]])
    p = np.eye(4); p[:3, 3] = [10, -1.6, 20]
    assert E.place_detection(detection(z=3, bearing=45), p, k) == pytest.approx([13, -1.6, 23])
    assert E.place_detection(detection(z=None), p, k) is None


def test_nearest_match_inclusive_one_metre_and_missing_depth():
    survivors = [dict(id='a', position_house=[0, 0, 3]), dict(id='b', position_house=[4, 0, 3])]
    assert E.nearest_match([1, -1, 3], survivors)['survivor'] == 'a'
    assert E.nearest_match([1.001, 0, 3], survivors)['survivor'] is None
    assert E.nearest_match(None, survivors)['survivor'] is None
    assert E.nearest_match([0, 0, 3], [])['survivor'] is None


def test_thresholds_and_duplicates_do_not_inflate_recall():
    frame = dict(run='r', robot='a', t=0., condition='dark', visible=[dict(survivor='s', range_m=3.)],
                 detections=[dict(confidence=.5, survivor='s', range_m=3., range_error_m=.2),
                             dict(confidence=.7, survivor='s', range_m=3., range_error_m=.3),
                             dict(confidence=.6, survivor=None, range_m=1.)])
    m = E.metrics([frame], .5)
    assert (m['tp'], m['fp'], m['hits'], m['visible']) == (2, 1, 1, 1)
    assert m['recall'] == 1.
    assert E.metrics([frame], .7)['precision'] == 1.
    assert E.metrics([frame], .71)['recall'] == 0.
    assert E.metrics([frame], .5, band='under_2m')['fp'] == 1
    assert E.metrics([frame], .5, band='2_to_4m')['visible'] == 1


def test_repetition_uses_separate_seconds_and_robot_tracks():
    frames = []
    for robot, t in [('a', 0.), ('a', .2), ('b', 1.), ('a', 2.), ('a', 7.)]:
        frames.append(dict(run='r', robot=robot, t=t, visible=[dict(survivor='s', range_m=3.)],
                           detections=[dict(confidence=.7, survivor='s', world=[0, 0, 3], range_m=3.)]))
    replay = E.replay_candidates(frames, .6, repeats=2, window_s=3.)
    assert len(replay['events']) == 1
    assert replay['events'][0]['t'] == 2.
    assert replay['events'][0]['robot'] == 'a'
    assert replay['precision'] == 1.


def test_visibility_requires_horizontal_fov_range_and_wall_los():
    k = np.array([[100., 0, 50], [0, 100., 50], [0, 0, 1]])
    survivors = [dict(id='front', position_house=[0, 0, 3]),
                 dict(id='behind', position_house=[0, 0, -3]),
                 dict(id='far', position_house=[0, 0, 7]),
                 dict(id='side', position_house=[4, 0, 3])]
    assert [v['survivor'] for v in E.visible_survivors(np.eye(4), k, 100, survivors, dict(rooms=[], doors=[]))] == ['front']


def test_missing_depth_is_false_and_empty_denominators_stay_unknown():
    frame = dict(run='r', robot='a', t=0., condition='dark', visible=[],
                 detections=[dict(confidence=.6, survivor=None, range_m=None)])
    assert E.metrics([frame], .5)['precision'] == 0.
    assert E.metrics([frame], .5)['recall'] is None
    assert E.metrics([frame], .7)['precision'] is None
    assert E.metrics([frame], .5, band='missing_depth')['fp'] == 1


def test_closed_wall_removes_visible_survivor_but_open_door_restores_it():
    k = np.array([[100., 0, 50], [0, 100., 50], [0, 0, 1]])
    survivor = [dict(id='s', position_house=[0, 0, 3])]
    plan = dict(rooms=[dict(rect=[-1, -1, 1, 2])], doors=[])
    assert E.visible_survivors(np.eye(4), k, 100, survivor, plan) == []
    plan['doors'] = [dict(wall='z', at=[0, 2], width=1, state='open')]
    assert len(E.visible_survivors(np.eye(4), k, 100, survivor, plan)) == 1
    plan['doors'][0]['state'] = 'closed'
    assert E.visible_survivors(np.eye(4), k, 100, survivor, plan) == []


def test_candidate_cannot_combine_distant_sightings_or_expired_votes():
    def f(t, x):
        return dict(run='r', robot='a', t=t, visible=[],
                    detections=[dict(confidence=.7, survivor=None, world=[x, 0, 3], range_m=3.)])
    assert E.replay_candidates([f(0, 0), f(1, 2)], .6, repeats=2, window_s=3)['events'] == []
    assert E.replay_candidates([f(0, 0), f(4, 0)], .6, repeats=2, window_s=3)['events'] == []
    result = E.replay_candidates([f(0, 0), f(3, 0)], .6, repeats=2, window_s=3)
    assert result['fp'] == 1
    assert result['precision'] == 0.


def test_same_second_and_same_frame_duplicates_cannot_raise_candidate():
    d = dict(confidence=.7, survivor='s', world=[0, 0, 3], range_m=3.)
    frames = [dict(run='r', robot='a', t=t, visible=[], detections=[d, d]) for t in [0., .2, .8]]
    assert E.replay_candidates(frames, .6, repeats=2)['events'] == []


def test_sampling_phase_can_expose_short_detection_bursts():
    frames = [dict(run='r', robot='a', t=t, visible=[], detections=[]) for t in (0., .2, 1., 1.2)]
    assert [f['t'] for f in E.one_hz(frames, phase_s=.2)] == [.2, 1.2]


def test_projection_rotates_camera_ray_into_building_axes():
    k = np.array([[100., 0, 50], [0, 100., 50], [0, 0, 1]])
    p = np.eye(4)
    p[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    p[:3, 3] = [10, -1.6, 20]
    assert E.place_detection(detection(), p, k) == pytest.approx([13, -1.6, 20])


def test_checkpoint_recovery_discards_only_interrupted_tail(tmp_path):
    path = tmp_path/'frames.jsonl'
    path.write_text('{"t":0}\n{"t":1}\n{"t":')
    assert E.read_checkpoint(path, repair=True) == [{'t': 0}, {'t': 1}]
    assert path.read_text() == '{"t":0}\n{"t":1}\n'
    path.write_text('broken\n{"t":1}\n')
    with pytest.raises(ValueError):
        E.read_checkpoint(path, repair=True)


def test_replay_keeps_full_rate_visibility_denominator():
    frames = [dict(run='r', robot='a', t=t, detections=[],
                   visible=[] if t == 1. else [dict(survivor='s', range_m=3.)])
              for t in (0., .5, 1., 1.5, 2.)]
    result = E.replay_candidates(frames, .5)
    assert result['sampled_frames'] == 3
    assert result['episodes'] == 1
    assert result['episode_recall'] == 0.


def test_phase_summary_weights_events_instead_of_averaging_precision():
    phases = [dict(tp=1, fp=1, episodes=2, episode_hits=1, precision=.5, episode_recall=.5),
              dict(tp=9, fp=0, episodes=4, episode_hits=3, precision=1., episode_recall=.75)]
    result = E.phase_summary(phases)
    assert result['precision'] == pytest.approx(10/11)
    assert result['episode_recall'] == pytest.approx(4/6)
    assert result['phase_count'] == 2
    assert result['worst_phase_precision'] == .5


def test_coincident_boxes_do_not_create_duplicate_candidate_tracks():
    d = dict(confidence=.7, survivor='s', world=[0, 0, 3], range_m=3.)
    frames = [dict(run='r', robot='a', t=t, visible=[], detections=[d, d]) for t in (0., 1.)]
    assert len(E.replay_candidates(frames, .6, repeats=2)['events']) == 1


def test_operating_point_does_not_hide_a_bad_recording_in_pooled_precision():
    grid = [dict(threshold=.35, repeats=2, window_s=2., precision=.99,
                 worst_recording_precision=.7, episode_recall=.9),
            dict(threshold=.6, repeats=2, window_s=2., precision=.98,
                 worst_recording_precision=.98, episode_recall=.7)]
    best, meets_target = E.choose_operating_point(grid)
    assert meets_target
    assert best['threshold'] == .6
