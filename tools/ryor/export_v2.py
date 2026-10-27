"""Export five houses and fifteen validated EASY recordings for the static main site.

Reuses completed simulator runs. This script only reads AWS objects. Validation runs
on every source frame before the smaller presentation files are written.
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SELECTED = [('family_ranch', 'Three-bed ranch'), ('long_ranch', 'L-shaped ranch'),
            ('onedoor_934', 'One-door house'), ('hx_035feb0125', 'Bungalow'),
            ('hx_cb9d3d6726', 'Corner house')]


def room_grid(poly, grid, contains):
    cm, n = grid['cell_m'], grid['size']
    ox, oy = grid['origin']
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    cells, total = set(), 0
    for y in range(int((min(ys)-oy)/cm), int((max(ys)-oy)/cm)+1):
        for x in range(int((min(xs)-ox)/cm), int((max(xs)-ox)/cm)+1):
            if contains(poly, ox+(x+.5)*cm, oy+(y+.5)*cm):
                total += 1
                if 0 <= x < n and 0 <= y < n:
                    cells.add(y*n+x)
    return cells, total


def export(site):
    for k, v in dict(RYOR_BUCKET='smoke-rescue-ryor-runs-968000946565', RYOR_TABLE='smoke-rescue-ryor',
                     RYOR_CLUSTER='smoke-rescue', RYOR_FAMILY='smoke-rescue-ryor', RYOR_SUBNETS='unused',
                     RYOR_SG='unused', AWS_DEFAULT_REGION='us-east-1').items():
        os.environ[k] = v
    sys.path[:0] = [str(ROOT / 'tools/ryor'), str(ROOT / 'tools')]
    spec = importlib.util.spec_from_file_location('v2_api', ROOT / 'deploy/aws/site_lambda/ryor_api.py')
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    from replay import inside
    gallery = json.loads((ROOT / 'data/runs/ryor/gallery.json').read_text())
    houses = []
    for hid, name in SELECTED:
        h = dict(next(h for h in gallery['houses'] if h['id'] == hid))
        h['name'] = name
        houses.append(h)
    api.gallery = lambda: {'houses': {h['id']: h for h in houses}, 'mission_s': 180}
    dest = site / 'public/rescue-data'
    raw = ROOT / 'data/runs/ryor/v2-source'
    dest.mkdir(parents=True, exist_ok=True)
    results = {}
    for house in houses:
        # Membership is used to count mapped floor, not to guide the robot.
        room_cells = [room_grid(room['poly'], gallery['grid'], inside) for room in house['rooms_poly']]
        for n in (1, 2, 3):
            key = f"{house['id']}__r{n}__easy"
            folder = raw / key
            folder.mkdir(parents=True, exist_ok=True)
            def read(name):
                p = folder / name
                if not p.exists():
                    body = api.s3.get_object(Bucket=os.environ['RYOR_BUCKET'], Key=f'ryor/cache/{key}/{name}')['Body'].read()
                    p.write_bytes(body)
                return json.loads(p.read_text())
            request = json.loads((ROOT / f'data/runs/ryor/setups/{key}.json').read_text())
            settings = request['runs'][0]
            if request['ryor']['difficulty'] != 'easy' or settings['head_start_s'] != 0 or settings['visibility_m'] != 6.0:
                raise ValueError(f'{key}: not the fixed EASY settings')
            status = read('status.json')
            if status.get('state') != 'done':
                raise ValueError(f'{key}: source recording is not complete')
            with ThreadPoolExecutor(max_workers=8) as pool:
                chunks = list(pool.map(read, [f'chunk_{i:04d}.json' for i in range(status['chunks'])]))
            api.chunks = lambda *args: chunks
            why = api.validate({'setup': key}, status, '')
            if why:
                raise ValueError(f'{key}: {why}')
            seen, frames, events = set(), [], []
            for chunk in chunks:
                for f in chunk['frames']:
                    for cells in f['seen'].values():
                        seen.update(cells)
                    mapped = [math.floor(100*len(seen & cells)/max(1, total)) for cells, total in room_cells]
                    frame = {k: f[k] for k in ('t', 'robots', 'fire', 'found', 'events')}
                    frame['mapped'] = mapped
                    frames.append(frame)
                    events.extend(f['events'])
            run = {'setup': key, 'difficulty': 'easy', 'engine': status.get('engine'), 'score': status['score'],
                   'frames': frames, 'rooms_searched': sum(p >= 80 for p in frames[-1]['mapped']), 'validated': True}
            body = json.dumps(run, separators=(',', ':')).encode()
            (dest / f'{key}.json').write_bytes(body)
            results[key] = {'valid': True, 'frames': len(frames), 'first_t': frames[0]['t'], 'last_t': frames[-1]['t'],
                            'score': status['score'], 'rooms_searched': run['rooms_searched'],
                            'settings': {'head_start_s': settings['head_start_s'], 'visibility_m': settings['visibility_m']},
                            'source': f's3://{os.environ["RYOR_BUCKET"]}/ryor/cache/{key}/',
                            'source_sha256': hashlib.sha256(json.dumps(chunks, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                            'validator_sha256': hashlib.sha256(inspect.getsource(api.validate).encode()).hexdigest(),
                            'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body), 'engine': status.get('engine')}
            print(key, status['score'], len(body), flush=True)
        house['scores'] = {str(n): results[f"{house['id']}__r{n}__easy"]['score'] | {'rooms_searched': results[f"{house['id']}__r{n}__easy"]['rooms_searched']} for n in (1,2,3)}
    (dest / 'gallery.json').write_text(json.dumps({'houses': houses, 'mission_s': 180, 'difficulty': 'easy', 'citation': gallery['citation']}, separators=(',', ':')))
    (dest / 'houseexpo-license.txt').write_bytes((ROOT / 'deploy/aws/rescue_page/houseexpo-license.txt').read_bytes())
    evidence = ROOT / 'handoff/ryor_v2_evidence'
    evidence.mkdir(exist_ok=True)
    (evidence / 'recordings_audit.json').write_text(json.dumps(results, indent=2))
    print(f'{len(results)}/15 EASY recordings validated and exported')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', type=Path, default=Path('/Volumes/Tashon/opencv/showcase/site-next'))
    export(parser.parse_args().site)
