"""Audit the fifteen source runs and the exact static files served by Try it."""
import argparse
import hashlib
import json
from pathlib import Path

from judge_gate import gate

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def verify(site):
    evidence = ROOT / 'handoff/backend_evidence'
    summary = read(evidence / 'rebuild_summary.json')
    snapshot = read(evidence / 'engine_snapshot.json')
    for name, digest in snapshot['sha256'].items():
        assert hashlib.sha256((Path(snapshot['stage']) / name).read_bytes()).hexdigest() == digest, name + ': snapshot changed'
    results = read(evidence / 'rebuild_gate.json')
    source = read(Path(summary['stage']) / 'data/runs/ryor/gallery.json')
    public = site / 'public/rescue-data'
    gallery = read(public / 'gallery.json')
    assert len(results) == 15, 'Not all fifteen setups were evaluated'
    assert read(public / 'missions.json')['engine'] == summary['commit']
    assert [h['id'] for h in gallery['houses']] == [h['id'] for h in source['houses']]
    published, withheld = {}, []
    for house in gallery['houses']:
        expected = []
        truth = next(h for h in source['houses'] if h['id'] == house['id'])
        for n in (1, 2, 3):
            key = f'{house["id"]}__r{n}__easy'
            result = results[key]
            folder = Path(result.get('path', ROOT / 'data/runs/ryor-judge' / summary['commit'][:12] / key))
            status = read(folder / 'status.json')
            assert status['state'] == 'done', f'{key}: simulation did not finish'
            assert status['wall']['started'] >= snapshot['batch_started_unix_s'], key + ': stale recording'
            request = read(folder / 'run/request.json')['runs'][0]
            assert request['head_start_s'] == 0 and request['visibility_m'] == 6 and request['settings'] == {}, key + ': not unchanged EASY settings'
            assert request['duration_s'] == 360 and not request.get('terra'), key + ': wrong mission mode'
            assert all(request[k] for k in ('fire', 'time_limit', 'robot_collision', 'capsule_walls')), key + ': disabled simulation checks'
            chunks = [read(folder / f'chunk_{i:04d}.json') for i in range(status['chunks'])]
            frames = [f for chunk in chunks for f in chunk['frames']]
            assert [f['t'] for f in frames] == [i / 2 for i in range(source['mission_s'] * 2 + 1)], key
            for frame in frames:
                assert [p['id'] for p in frame['robots']] == ['robot_' + c for c in 'ABC'[:n]], key
            proof = gate(truth, frames, status, summary['commit'])
            assert proof['passed'] == result['passed'], key + ': gate result changed'
            target = public / (key + '.json')
            if not proof['passed']:
                assert not target.exists(), key + ': failed replay was published'
                withheld.append(key)
                continue
            recording = read(target)
            assert recording['validated'] is True and recording['setup'] == key
            assert recording['difficulty'] == 'easy' and recording['duration_s'] == source['mission_s']
            assert recording['engine']['git'] == summary['commit']
            assert recording['score'] == status['score']
            assert recording['frames'][-1]['t'] == source['mission_s']
            assert len(recording['frames']) == len(frames)
            for raw, shown in zip(frames, recording['frames']):
                assert all(raw[k] == shown[k] for k in ('t', 'robots', 'fire', 'found', 'events')), key
                assert len(shown['mapped']) == len(house['rooms_poly'])
                assert all(0 <= v <= 100 for v in shown['mapped']), key
            assert all(recording['gate'][k] == proof[k] for k in recording['gate']), key
            assert house['scores'][str(n)] == status['score'] | {'rooms_searched': recording['rooms_searched']}
            expected.append(n)
            published[key] = {'frames': len(frames), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
        assert house['available_replays'] == expected, house['id']
        assert set(house['scores']) == {str(n) for n in expected}, house['id']
    assert {p.stem for p in public.glob('*__r*__easy.json')} == set(published)
    audit = {'engine': summary['commit'], 'evaluated': 15, 'published': published, 'withheld': withheld}
    (evidence / 'static_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(f'PASS 15 complete source runs; {len(published)} published and {len(withheld)} withheld; static files match source frames')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', type=Path, default=ROOT.parent / 'showcase/site-next')
    verify(parser.parse_args().site)
