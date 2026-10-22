"""The measurement must count skipped history frames and exclude unrelated files."""
import importlib.util
from pathlib import Path

from test_agent_lidar import recording

spec = importlib.util.spec_from_file_location('measure_transport', Path(__file__).parents[2] / 'tools/measure_recording_transport.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_person_cursor_does_not_hide_missing_history_images(tmp_path):
    folder = tmp_path / 'robot_A'
    recording(folder)
    for sensor in m.SENSORS:
        for stamp in ('0.5', '1'):
            (folder / sensor / (stamp + '.png')).write_bytes((folder / sensor / '0.png').read_bytes())
        (folder / (sensor + '.txt')).write_text(f'0 {sensor}/0.png\n0.5 {sensor}/0.5.png\n1 {sensor}/1.png\n')
    (folder / 'scene.json').write_text('must not send')
    sent, hashes = set(), {}
    first = m.plan_batch(folder, 0., sent=sent, hashes=hashes)
    assert {n for n in first if n.endswith('.png')} == {'rgb/0.png', 'depth/0.png', 'thermal/0.png'}
    second = m.plan_batch(folder, 1., sent=sent, hashes=hashes)
    assert {n for n in second if n.endswith('.png')} == {'rgb/1.png', 'depth/1.png', 'thermal/1.png'}
    step = m.plan_batch(folder, 1., history=True, sent=sent, hashes=hashes)
    assert set(step) == {'rgb/0.5.png', 'depth/0.5.png', 'thermal/0.5.png'}
    size = sum((folder / n).stat().st_size for n in step)
    measured = m.measure(step)
    assert measured['payload_bytes'] == size
    assert measured['zip_bytes'] == size + 22 + sum(76 + 2 * len(n) for n in step)
    assert measured['upload_seconds']['5'] == round(measured['zip_bytes'] * 8 / 5_000_000, 6)
    assert m.plan_batch(folder, 1., history=True, sent=sent, hashes=hashes) == {}
