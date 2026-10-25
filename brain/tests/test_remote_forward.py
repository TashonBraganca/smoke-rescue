"""The AWS forwarder: Unity's local calls reach the real agent server through uploads and pull-v1 and give the same
answers as a local server; AWS failures never become empty answers."""
import asyncio
import json
from pathlib import Path

import httpx
import pytest

from brain.agent import remote_forward as rf
from brain.agent import server, vision
from brain.agent.llm import Reply, ToolCall
from brain.agent.recording_transport import SMALL_FILES
from test_recording_transport import call, local  # noqa: F401  (fixture)
from test_pull_transport import history

SECRET = ('X-Origin-Verify', 'test-origin-value')
PUBLIC = ('/health', '/vision/', '/perceive/people')


class Remote:
    """The real agent server app behind an ALB-like header rule, recording every request the forwarder makes."""
    def __init__(self):
        self.requests = []

    def __call__(self, request: httpx.Request):
        path = request.url.path
        self.requests.append((request.method, path, dict(request.headers), request.content))
        public = path.startswith(PUBLIC) or (path.startswith('/place/') and path.endswith('/match'))
        if not public and request.headers.get(SECRET[0]) != SECRET[1]:
            return httpx.Response(403, json={'detail': 'forbidden'})
        r = call(request.method, path + (('?' + request.url.query.decode()) if request.url.query else ''),
                 content=request.content, headers={k: v for k, v in request.headers.items() if k.lower() in ('content-type', 'x-recording-source')})
        return httpx.Response(r.status_code, content=r.content, headers={'content-type': r.headers.get('content-type', 'application/json')})


def through(method, path, **kwargs):
    async def send():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=rf.app), base_url='http://fwd') as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(send())


@pytest.fixture
def remote(monkeypatch):
    fake = Remote()
    fwd = rf.Forwarder('http://aws.test', SECRET, transport=httpx.MockTransport(fake))
    monkeypatch.setattr(rf, '_fwd', fwd)
    monkeypatch.setitem(rf.FORWARD, 'poll_interval_s', 0.01)
    monkeypatch.setitem(rf.FORWARD, 'step_deadline_s', 15.0)
    fake.fwd = fwd
    return fake


def fingerprint(frame, min_conf=.25):
    # A deterministic stand-in for YOLO that changes with every RGB and depth pixel of the frame it was given.
    return [{'rgb_sum': int(frame.rgb.astype('int64').sum()), 'depth_sum': float(frame.depth_m.sum()), 'range_m': 1.0,
             'bearing_deg': 0.0, 'confidence': .9}]


def test_people_through_forwarder_equal_local_server_on_the_same_frame(local, remote, monkeypatch):
    history(local)                      # frames at 0, 0.5 and 1 with different pixels
    monkeypatch.setattr(vision, 'detect_people_rgb', fingerprint)
    body = {'robot_dir': str(local), 't': .5, 'min_conf': .25}
    expected = call('POST', '/perceive/people', json=body)
    assert expected.status_code == 200, expected.text
    got = through('POST', '/perceive/people', json=body)
    assert got.status_code == 200, got.text
    assert got.headers['x-smoke-served-by'] == 'aws'
    assert got.json()['people'] == expected.json()['people']
    assert got.json()['t'] == expected.json()['t'] == .5
    puts = [r for r in remote.requests if r[0] == 'PUT']
    assert puts and all(r[2].get(SECRET[0].lower()) == SECRET[1] for r in puts)
    posted = [json.loads(r[3]) for r in remote.requests if r[1] == '/perceive/people']
    assert posted[-1]['robot_dir'].startswith('run://')
    # Only the frame the call reads went up, never the other stamps.
    remote_dir = next((server.RUN_DIR / 'recordings').glob('*/robot_A'))
    assert sorted(p.name for p in (remote_dir / 'rgb').iterdir()) == ['0.5.png']
    # A second call at the same stamp uploads nothing new.
    before = len(puts)
    assert through('POST', '/perceive/people', json=body).json()['people'] == expected.json()['people']
    assert len([r for r in remote.requests if r[0] == 'PUT']) == before


def test_uploads_hold_only_allowlisted_robot_files(local, remote, monkeypatch):
    history(local)
    (local.parent / 'oracle').mkdir()
    (local.parent / 'oracle' / 'survivors.json').write_text('{}')
    monkeypatch.setattr(vision, 'detect_people_rgb', fingerprint)
    assert through('POST', '/perceive/people', json={'robot_dir': str(local), 't': 1.}).status_code == 200
    import io
    import zipfile
    for method, path, headers, content in remote.requests:
        if method == 'PUT':
            names = zipfile.ZipFile(io.BytesIO(content)).namelist()
            assert all(n in SMALL_FILES or n.split('/')[0] in ('rgb', 'depth', 'thermal') for n in names), names
            assert not any('oracle' in n or 'truth' in n for n in names)


def test_agent_step_uses_pull_and_measures_the_same_frames(local, remote, monkeypatch):
    history(local)
    scan = vision.scan_thermal(vision.load_frame(str(local), 1.))
    fire = vision.track_fire_spread([vision.load_frame(str(local), t) for t in (0., .5, 1.)])

    class Decide:
        model = 'test'
        def __init__(self):
            self.turn = 0
        def chat(self, messages, tools):
            self.turn += 1
            if self.turn <= 2:
                return Reply('', [ToolCall(('scan_thermal', 'track_fire_spread')[self.turn - 1], {})], 0, self.model)
            self.measured = [json.loads(m['content']) for m in messages if m['role'] == 'tool']
            return Reply('', [ToolCall('finish', {'summary': 'measured remotely'})], 0, self.model)

    model = Decide()
    monkeypatch.setattr(server, '_backend', model)
    body = {'robot_dir': str(local), 't': 1., 'frame_times': [0., .5, 1.], 'pose': {'x': 0, 'y': 0, 'heading_deg': 0},
            'trace_path': '/Volumes/not-on-aws/agent_trace.jsonl', 'debug_dir': '/Volumes/not-on-aws/agent_debug'}
    got = through('POST', '/robot/robot_A/step', json=body)
    assert got.status_code == 200, got.text
    assert got.json()['summary'] == 'measured remotely', json.dumps(got.json())
    # The latest-frame tool keeps the local evidence id (X-Recording-Source); fire tracking on the pull path loads
    # depth+thermal only, so its frame id differs by design (test_pull_transport), but every measurement is equal.
    assert model.measured[0]['id'] == scan['id']
    for key in ('areas_m2', 'hot_pixels', 'growth_px_s', 'growth_m2_s', 'speed_m_s', 'direction_deg', 'hot_depth_rate_m_s'):
        assert model.measured[1][key] == fire[key]
    step = json.loads(next(r[3] for r in remote.requests if r[1] == '/robot/robot_A/step'))
    assert step['transport'] == 'pull-v1' and step['robot_dir'].startswith('run://')
    assert 'trace_path' not in step and 'debug_dir' not in step
    # The history frames came through the pull demand, after the decision-time triple was pushed.
    assert any(r[1].startswith('/pull/') for r in remote.requests)
    remote_dir = next((server.RUN_DIR / 'recordings').glob('*/robot_A'))
    for name in ('depth/0.png', 'thermal/0.png', 'depth/0.5.png', 'thermal/0.5.png'):
        assert (remote_dir / name).read_bytes() == (local / name).read_bytes()
    assert not (remote_dir / 'rgb' / '0.5.png').exists()


def test_aws_down_fails_loudly_never_with_empty_people(local, monkeypatch):
    def down(request):
        raise httpx.ConnectError('connection refused', request=request)
    monkeypatch.setattr(rf, '_fwd', rf.Forwarder('http://aws.test', SECRET, transport=httpx.MockTransport(down)))
    got = through('POST', '/perceive/people', json={'robot_dir': str(local), 't': 0.})
    assert got.status_code == 502
    assert got.headers['x-smoke-served-by'] == 'none'
    assert 'people' not in got.json()
    assert through('GET', '/health').status_code == 503


def test_aws_5xx_goes_to_the_fallback_and_says_so(local, monkeypatch):
    monkeypatch.setattr(vision, 'detect_people_rgb', fingerprint)
    def broken(request):
        return httpx.Response(503, json={'detail': 'overloaded'})
    fwd = rf.Forwarder('http://aws.test', SECRET, fallback_url='http://local.test', transport=httpx.MockTransport(broken))
    fwd.local = httpx.Client(transport=httpx.MockTransport(Remote()))
    monkeypatch.setattr(rf, '_fwd', fwd)
    body = {'robot_dir': str(local), 't': 0., 'min_conf': .25}
    got = through('POST', '/perceive/people', json=body)
    assert got.status_code == 200, got.text
    assert got.headers['x-smoke-served-by'] == 'fallback'
    assert got.json()['people'] == call('POST', '/perceive/people', json=body).json()['people']


def test_remote_4xx_is_returned_unchanged_not_hidden(local, remote):
    # Nothing near t=40 s: the remote server's own 400 must reach Unity, not a fallback or an empty list.
    got = through('POST', '/perceive/people', json={'robot_dir': str(local), 't': 40.})
    assert got.status_code == 400
    assert got.headers['x-smoke-served-by'] == 'aws'


def test_model_routes_carry_the_origin_header_and_no_local_trace_path(remote, monkeypatch):
    seen = {}

    class Talk:
        model = 'test'
        def chat(self, messages, tools):
            seen['called'] = True
            return Reply('', [ToolCall('finish', {'summary': 'ok'})], 0, self.model)

    monkeypatch.setattr(server, '_backend', Talk())
    got = through('POST', '/commander/step', json={'t': 30., 'reports': [], 'robots': ['robot_A'],
                                                    'trace_path': '/Volumes/not-on-aws/agent_trace.jsonl'})
    assert got.status_code != 403, got.text
    sent = json.loads(next(r[3] for r in remote.requests if r[1] == '/commander/step'))
    assert 'trace_path' not in sent


def test_cumulative_index_picks_the_same_frame_as_the_full_index(tmp_path):
    rows = [(i / 15, f'{i / 15:.6f} rgb/{i}.png', f'rgb/{i}.png') for i in range(300)]
    for t in (0., 3.3, 7.04, 19.95):
        full = rf.nearest(rows, t, .5)
        subset = [r for r in rows if r is full or int(r[2][4:-4]) % 7 == 0]
        assert rf.nearest(subset, t, .5) == full
    assert rf.nearest(rows, 25., .5) is None


def test_a_step_the_remote_model_never_answered_is_a_failure_not_an_empty_200(remote, monkeypatch):
    class Down:
        model = 'test'
        def chat(self, messages, tools):
            raise httpx.HTTPStatusError('400 Bad Request', request=None, response=None)
    monkeypatch.setattr(server, '_backend', Down())
    got = through('POST', '/commander/step', json={'t': 30., 'reports': [{'from': 'robot_A', 't': 29., 'text': 'hello'}],
                                                    'robots': ['robot_A']})
    assert got.status_code == 502, got.text
    assert rf.model_failed({'summary': None, 'orders': [], 'refusals': [], 'rounds': 1})
    assert not rf.model_failed({'summary': None, 'orders': [], 'refusals': [], 'rounds': 0, 'repeat_of_recent_event': True})
    assert not rf.model_failed({'summary': 'ok', 'orders': [], 'refusals': []})
