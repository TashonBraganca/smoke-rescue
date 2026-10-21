"""Sensor geometry and temporal selection for the audience map."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import showcase


def test_readme_action_evidence_scope():
    readme = (Path(__file__).parents[1] / 'README.md').read_text()
    agents = next(line for line in readme.splitlines() if line.startswith('- **Agents**'))
    for action in ('confirm_survivor', 'reject_candidate', 'avoid_area', 'set_torch',
                   'open_door', 'clear_debris', 'spray_water', 'extract_survivor',
                   'go_to', 'approach_candidate', 'turn_to_face', 'explore_frontier',
                   'send_team_message', 'request_human_approval'):
        assert f'`{action}`' in agents
    for disclosure in ('indoors', 'offered id', 'arm', 'facing', 'below', 'Unity', 'own spot'):
        assert disclosure in agents
    assert 'Every action' not in agents


def test_readme_discloses_training_overlap():
    readme = (Path(__file__).parents[1] / 'README.md').read_text()
    paragraph = readme.split('## Evaluation\n', 1)[1].split('\n## ', 1)[0]
    for disclosure in ('config/brain.json', 'fixed spots', 'same frames', 'held-out layout', 'separately'):
        assert disclosure in paragraph


@pytest.fixture
def review_recording(tmp_path):
    (tmp_path / 'meta.json').write_text(json.dumps({'robots': ['robot_A', 'robot_B']}))
    oracle = tmp_path / 'oracle'
    oracle.mkdir()
    (oracle / 'survivors.json').write_text(json.dumps([
        {'id': 'person1', 'position_house': [2, 0, 3]},
        {'id': 'person2', 'position_house': [8, 0, 9]}]))
    for r in ('robot_A', 'robot_B'):
        (tmp_path / r).mkdir()
        (tmp_path / r / 'brain_state.jsonl').write_text('')
        (oracle / f'{r}_groundtruth.txt').write_text('0 0 0 0 0 0 0 1\n20 0 0 0 0 0 0 1\n')
    # B confirms first even though metadata lists A first.
    events = {'robot_A': [(4, 2, 3), (6, 12, 13), (8, 2, 3), (10, 8, 9)],
              'robot_B': [(2, 2, 3), (3, 12, 13), (7, 20, 20)]}
    for r, rows in events.items():
        (tmp_path / r / 'decisions.jsonl').write_text(''.join(json.dumps(dict(
            t=t, kind='survivor_confirmed', numbers=dict(x_m=x, y_m=y))) + '\n'
            for t, x, y in rows))
    (tmp_path / 'rescues.jsonl').write_text(json.dumps(dict(
        t=5, event='released', person='person1', x=12, z=13, outside_building=True)) + '\n')
    return tmp_path


def test_scoring_counts_duplicates_moved_people_and_false_confirmations(review_recording):
    from brain.compare_runs import score
    result = score(review_recording)
    assert result['survivors_found'] == result['distinct_people_confirmed'] == 2
    assert result['confirmations'] == 7
    assert result['duplicate_confirmations'] == 2
    assert result['moved_confirmations'] == 1
    assert result['false_confirmations'] == 3
    assert result['confirmation_precision'] == pytest.approx(2 / 7)
    assert result['first_found_s'] == 2
    assert result['people_carried_out'] == 1
    details = result['confirmation_details']
    assert [d['t'] for d in details] == [2, 3, 4, 6, 7, 8, 10]
    assert not details[0]['duplicate'] and details[2]['duplicate']
    assert details[3]['matched_after_rescue'] and details[3]['survivor'] == 'person1'
    assert details[1]['survivor'] is None and details[5]['survivor'] is None


def test_survivor_matching_reads_evaluation_config(review_recording, monkeypatch):
    from brain import compare_runs
    original = Path.read_text
    def configured(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        if path.name == 'evaluation.json':
            config = json.loads(text)
            config['survivor_match_m'] = .1
            return json.dumps(config)
        return text
    path = review_recording / 'robot_A/decisions.jsonl'
    path.write_text(json.dumps(dict(t=1, kind='survivor_confirmed', numbers=dict(x_m=2.5, y_m=3))) + '\n')
    (review_recording / 'robot_B/decisions.jsonl').write_text('')
    assert compare_runs.score(review_recording)['survivors_found'] == 1
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'read_text', configured)
        assert compare_runs.score(review_recording)['survivors_found'] == 0


def test_video_counter_uses_evaluator_matches_at_display_time(review_recording):
    # The video needs only confirmation truth, not trajectory/coverage evaluation.
    for path in (review_recording / 'oracle').glob('*groundtruth.txt'):
        path.unlink()
    counts = showcase.score_confirmations(review_recording)
    assert showcase.confirmation_label(counts, 'robot_A', 1) == 'confirmations 0; distinct people 0'
    assert showcase.confirmation_label(counts, 'robot_A', 4) == 'confirmations 1; distinct people 1'
    assert showcase.confirmation_label(counts, 'robot_A', 8) == 'confirmations 3; distinct people 1'
    assert showcase.confirmation_label(counts, 'robot_A', 10) == 'confirmations 4; distinct people 2'


def test_video_labels_true_floor_plan(monkeypatch):
    from PIL import Image, ImageDraw
    rendered = []
    original = ImageDraw.ImageDraw.text
    def capture(draw, xy, text, *args, **kwargs):
        rendered.append(text)
        return original(draw, xy, text, *args, **kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw, 'text', capture)
    draw = ImageDraw.Draw(Image.new('RGB', (1920, 1080)))
    showcase.evaluator_plan_panel(draw, (672, 568, 1312, 1064), 'Shared context')
    assert any("evaluator's view of the true floor plan" in text for text in rendered)
    assert any('robots never see it' in text for text in rendered)


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


@pytest.fixture
def placed_scored_recording(tmp_path):
    rec = tmp_path / 'scored'
    rec.mkdir()
    scenario = tmp_path / 'scenario.json'
    scenario.write_text(json.dumps(dict(robots=[
        dict(id='robot_A', spawn=dict(at=[10, 20], yaw=90)),
        dict(id='robot_B', spawn=dict(at=[10, 20], yaw=90))])))
    (rec / 'meta.json').write_text(json.dumps(dict(
        robots=['robot_A', 'robot_B'], scenario=str(scenario))))
    (rec / 'oracle').mkdir()
    (rec / 'oracle/survivors.json').write_text(json.dumps([
        dict(id='person_1', position_house=[10, 0, 20]),
        dict(id='person_2', position_house=[20, 0, 20])]))
    # Local [0, x] rotates into house [10 + x, 20]. B logs first in time.
    events = {'robot_A': [(2, 0), (4, 10), (6, 30), (8, 20), (9, 0)],
              'robot_B': [(1, 0), (7, 20)]}
    for robot, rows in events.items():
        rd = rec / robot
        rd.mkdir()
        (rd / 'brain_state.jsonl').write_text(
            json.dumps(dict(t=0, x=0, y=0, yaw_rad=0)) + '\n')
        (rd / 'decisions.jsonl').write_text(''.join(json.dumps(dict(
            t=t, kind='survivor_confirmed', numbers=dict(x_m=0, y_m=x))) + '\n'
            for t, x in rows))
        np.savetxt(rec / 'oracle' / f'{robot}_groundtruth.txt',
                   [[0, 10, 0, 20, 0, 0, 0, 1], [10, 10, 0, 20, 0, 0, 0, 1]])
    # A release in the house frame, followed by a second move of the same person.
    (rec / 'rescues.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in [
        dict(t=5, event='released', person='person_1', x=30, z=20, outside_building=True),
        dict(t=8.5, event='released', person='person_1', x=40, z=20, outside_building=True)]))
    return rec


def test_scoring_counts_duplicates_and_released_positions(placed_scored_recording):
    from brain.compare_runs import score
    report = score(placed_scored_recording)
    assert report['survivors_found'] == 2
    assert report['confirmations'] == 7
    assert report['false_confirmations'] == 2
    assert report['duplicate_confirmations'] == 3
    assert report['moved_confirmations'] == 2
    assert report['confirmation_precision'] == pytest.approx(2 / 7)
    assert report['first_found_s'] == 1
    assert report['people_carried_out'] == 1
    rows = report['confirmation_details']
    assert [r['t'] for r in rows] == [1, 2, 4, 6, 7, 8, 9]
    assert [r['duplicate'] for r in rows] == [False, True, False, False, True, True, False]
    assert [r['matched_after_rescue'] for r in rows] == [False, False, False, False, True, True, False]
    assert rows[-1]['survivor'] is None  # The original spot is empty after release.


def test_scoring_uses_configured_match_radius(placed_scored_recording, monkeypatch):
    import brain.compare_runs as scoring
    config_path = Path(__file__).parents[1] / 'config/evaluation.json'
    config = json.loads(config_path.read_text())
    assert config['survivor_match_m_reason'].strip()
    rd = placed_scored_recording / 'robot_A'
    (rd / 'decisions.jsonl').write_text(json.dumps(dict(
        t=2, kind='survivor_confirmed', numbers=dict(x_m=0, y_m=1.2))) + '\n')
    (placed_scored_recording / 'robot_B/decisions.jsonl').write_text('')
    assert scoring.score(placed_scored_recording)['survivors_found'] == 0
    original = Path.read_text
    def changed(path, *args, **kwargs):
        if path.resolve() == config_path.resolve():
            return json.dumps(config | dict(survivor_match_m=1.3))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', changed)
    assert scoring.score(placed_scored_recording)['survivors_found'] == 1


def test_scoring_reads_complete_release_without_final_newline(placed_scored_recording):
    from brain.compare_runs import score
    path = placed_scored_recording / 'rescues.jsonl'
    path.write_text(path.read_text().rstrip())
    path = placed_scored_recording / 'robot_A/decisions.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]['numbers']['y_m'] = 30
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    last = score(placed_scored_recording)['confirmation_details'][-1]
    assert last['survivor'] == 'person_1'
    assert last['matched_after_rescue']


def test_video_counter_separates_events_from_distinct_people(placed_scored_recording):
    report = showcase.score_confirmations(placed_scored_recording)
    assert showcase.confirmation_label(report, 'robot_A', 1) == 'confirmations 0; distinct people 0'
    assert showcase.confirmation_label(report, 'robot_A', 6) == 'confirmations 3; distinct people 2'
    assert showcase.confirmation_label(report, 'robot_A', 9) == 'confirmations 5; distinct people 2'
    assert showcase.confirmation_label(report, 'robot_B', 9) == 'confirmations 2; distinct people 1'
    assert showcase.confirmation_label(report, None, 9) == 'confirmations 7; distinct people 2'


def test_video_labels_true_plan_as_evaluator_only():
    from PIL import Image, ImageDraw
    image = Image.new('RGB', (640, 500), showcase.BG)
    class RecordingDraw:
        def __init__(self):
            self.draw = ImageDraw.Draw(image)
            self.texts = []
        def text(self, xy, text, **kwargs):
            self.texts.append(text)
            return self.draw.text(xy, text, **kwargs)
        def __getattr__(self, name):
            return getattr(self.draw, name)
    for title in ['Shared context', 'SLAM map']:
        draw = RecordingDraw()
        showcase.evaluator_plan_panel(draw, (0, 0, 640, 500), title)
        assert "evaluator's view of the true floor plan; robots never see it" in draw.texts
