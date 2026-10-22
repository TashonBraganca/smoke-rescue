"""Remote recordings must expose the same sensor evidence without accepting truth files."""
import asyncio
import io
import json
import stat
import zipfile
from pathlib import Path

import httpx
import pytest

from brain.agent import server, vision
from brain.agent.llm import Reply, ToolCall
from test_agent_lidar import recording


def archive(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_STORED) as out:
        for name, content in files.items():
            out.writestr(name, content)
    return stream.getvalue()


def batch(folder):
    return archive({str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob('*') if p.is_file()})


def call(method, path, **kwargs):
    async def send():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url='http://test') as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(send())


@pytest.fixture
def local(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'RUN_DIR', tmp_path / 'server')
    folder = tmp_path / 'local' / 'robot_A'
    recording(folder, real_depth=True)
    return folder


def upload(local, **kwargs):
    return call('PUT', '/runs/test-run/robot_A/files', content=batch(local),
                headers={'Content-Type': 'application/zip', 'X-Recording-Source': str(local)}, **kwargs)


def test_uploaded_tools_are_byte_identical_to_local_including_evidence_ids(local, monkeypatch):
    response = upload(local)
    assert response.status_code == 200, response.text
    assert response.json()['robot_dir'] == 'run://test-run/robot_A'
    from brain.agent.tools import AgentContext, ToolRegistry
    expected = ToolRegistry(local, AgentContext('robot_A', 0., 0., 0., 0.)).call('scan_thermal', {})

    class Scan:
        model = 'test'
        def __init__(self):
            self.turn = 0
        def chat(self, messages, tools):
            self.turn += 1
            if self.turn == 1:
                return Reply('', [ToolCall('scan_thermal', {})], 0., self.model)
            actual = json.loads(messages[-1]['content'])
            assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
            return Reply('', [ToolCall('finish', {'summary': 'same sensor evidence'})], 0., self.model)

    monkeypatch.setattr(server, '_backend', Scan())
    result = call('POST', '/robot/robot_A/step', json={
        'robot_dir': 'run://test-run/robot_A', 't': 0., 'frame_times': [0.],
        'trace_path': '/not-a-server-path/trace.jsonl', 'debug_dir': '/not-a-server-path/debug',
        'pose': {'x': 0., 'y': 0., 'heading_deg': 0.}})
    assert result.status_code == 200, result.text
    assert result.json()['summary'] == 'same sensor evidence'
    assert result.json()['evidence_ids'] == [expected['id']]


@pytest.mark.parametrize('name', ['../camera.yaml', '/tmp/camera.yaml', 'rgb/../../camera.yaml',
                                  'rgb\\0.png', 'oracle/truth.json', 'truth.json', 'ground_truth/0.png',
                                  'rgb/oracle.png', 'rgb/truth.png', 'scene.json', 'meta.json'])
def test_refuses_traversal_and_truth_without_partial_publication(local, name):
    response = call('PUT', '/runs/test-run/robot_A/files', content=archive({'camera.yaml': b'valid', name: b'secret'}))
    assert response.status_code == 400, response.text
    assert not list(server.RUN_DIR.rglob('camera.yaml'))


def test_refuses_archive_symlink(local):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as out:
        link = zipfile.ZipInfo('rgb/0.png')
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        out.writestr(link, '/etc/passwd')
    assert call('PUT', '/runs/test-run/robot_A/files', content=stream.getvalue()).status_code == 400


def test_missing_uploaded_frame_is_an_error_before_detection_or_agent(local, monkeypatch):
    (local / 'depth' / '0.png').unlink()
    assert upload(local).status_code == 200
    for route in ('/perceive/people', '/robot/robot_A/step', '/map3d/robot_A/integrate', '/place/robot_A/keyframe'):
        response = call('POST', route, json={'robot_dir': 'run://test-run/robot_A', 't': 0.,
                                            'pose': {'x': 0., 'y': 0., 'heading_deg': 0.}})
        assert response.status_code == 400, (route, response.text)
        assert 'missing' in response.json()['detail'].lower()
        assert 'depth/0.png' in response.json()['detail']


def test_history_window_is_required_not_just_the_latest_frame(local):
    assert upload(local).status_code == 200
    # The index advertises a newer frame that has not arrived. Both the history and latest must be checked.
    payload = archive({s + '.txt': f'0 {s}/0.png\n1 {s}/1.png\n' for s in ('rgb', 'depth', 'thermal')})
    assert call('PUT', '/runs/test-run/robot_A/files', content=payload).status_code == 200
    response = call('POST', '/robot/robot_A/step', json={'robot_dir': 'run://test-run/robot_A', 't': 0., 'frame_times': [0., 1.]})
    assert response.status_code == 400
    assert 'missing' in response.json()['detail'].lower()


@pytest.mark.parametrize('uri', ['run://../robot_A', 'run://test-run/../robot_A', 'run://test-run/robot_A/extra',
                                  'run://test-run/robot_A?file=truth', 'run://test-run/%2e%2e', 'run:/test-run/robot_A'])
def test_refuses_malformed_run_identifiers(local, uri):
    response = call('POST', '/perceive/people', json={'robot_dir': uri, 't': 0.})
    assert response.status_code == 400


def test_unknown_run_is_a_clear_error(local):
    response = call('POST', '/perceive/people', json={'robot_dir': 'run://absent/robot_A', 't': 0.})
    assert response.status_code == 404
    assert 'upload' in response.json()['detail'].lower()


def test_retry_keeps_images_immutable_and_run_source_stable(local):
    assert upload(local).status_code == 200
    assert upload(local).status_code == 200
    changed = call('PUT', '/runs/test-run/robot_A/files', content=archive({'rgb/0.png': b'changed'}))
    assert changed.status_code == 400
    assert 'immutable' in changed.json()['detail']
    changed = call('PUT', '/runs/test-run/robot_A/files', content=archive({'rgb.txt': '0 rgb/0.png\n'}),
                   headers={'X-Recording-Source': '/another/robot_A'})
    assert changed.status_code == 400


def test_server_refuses_existing_folder_symlink(local, tmp_path):
    outside = tmp_path / 'outside'
    outside.mkdir()
    root = server.RUN_DIR / 'recordings'
    root.mkdir(parents=True)
    (root / 'test-run').symlink_to(outside, target_is_directory=True)
    assert upload(local).status_code == 400
    assert not list(outside.iterdir())


def test_remote_map_export_can_be_downloaded_without_a_mac_output_path(local):
    assert upload(local).status_code == 200
    pose = {'x': 0., 'y': 0., 'heading_deg': 0.}
    try:
        response = call('POST', '/map3d/robot_A/integrate', json={
            'robot_dir': 'run://test-run/robot_A', 't': 0., 'pose': pose})
        assert response.status_code == 200, response.text
        response = call('POST', '/map3d/robot_A/export', json={
            'robot_dir': 'run://test-run/robot_A', 't': 0., 'path': '/some/mac/map3d.ply'})
        assert response.status_code == 200, response.text
        assert response.json()['download_url'] == '/runs/test-run/robot_A/map3d.ply'
        downloaded = call('GET', response.json()['download_url'])
        assert downloaded.status_code == 200
        assert downloaded.content.startswith(b'ply\n')
        assert b'property float nx' in downloaded.content
    finally:
        for volume in server._maps3d.values():
            volume.close()
        server._maps3d.clear()


def test_track_fire_spread_remote_history_equals_local(local, monkeypatch):
    import cv2
    for sensor in ('rgb', 'depth', 'thermal'):
        image = cv2.imread(str(local / sensor / '0.png'), cv2.IMREAD_UNCHANGED)
        if sensor == 'thermal':
            image[25:95, 60:100] = 5500
        assert cv2.imwrite(str(local / sensor / '1.png'), image)
        (local / (sensor + '.txt')).write_text(f'0 {sensor}/0.png\n1 {sensor}/1.png\n')
    assert upload(local).status_code == 200
    from brain.agent.tools import AgentContext, ToolRegistry
    expected = ToolRegistry(local, AgentContext('robot_A', 1., 0., 0., 0., frame_times=[0., 1.])).call('track_fire_spread', {})
    assert 'error' not in expected

    class Spread:
        model = 'test'
        def __init__(self):
            self.turn = 0
        def chat(self, messages, tools):
            self.turn += 1
            if self.turn == 1:
                return Reply('', [ToolCall('track_fire_spread', {})], 0., self.model)
            assert json.dumps(json.loads(messages[-1]['content']), sort_keys=True) == json.dumps(expected, sort_keys=True)
            return Reply('', [ToolCall('finish', {'summary': 'same history'})], 0., self.model)

    monkeypatch.setattr(server, '_backend', Spread())
    response = call('POST', '/robot/robot_A/step', json={
        'robot_dir': 'run://test-run/robot_A', 't': 1., 'frame_times': [0., 1.],
        'pose': {'x': 0., 'y': 0., 'heading_deg': 0.}})
    assert response.status_code == 200, response.text
    assert response.json()['evidence_ids'] == [expected['id']]
    assert response.json()['summary'] == 'same history'


def test_expanded_zip_limit_is_checked_before_publication(local, monkeypatch):
    from brain.agent import recording_transport
    monkeypatch.setattr(recording_transport, 'MAX_BATCH_BYTES', 32)
    response = call('PUT', '/runs/test-run/robot_A/files', content=archive({'camera.yaml': b'x' * 33}))
    assert response.status_code == 413
    assert not list(server.RUN_DIR.rglob('camera.yaml'))


def test_invalid_zip_is_a_client_error(local):
    assert call('PUT', '/runs/test-run/robot_A/files', content=b'not a zip').status_code == 400
