"""Evaluator measurement for warm-shape candidates: how far is each candidate from the nearest wall on the robot's own map,
and is it a real person? Over every recording with truth: thermal candidates (survivor_candidate without rgb) and colour
candidates, labelled person when within 1 m of a true survivor (placed with the true spawn), with the distance to the
nearest occupied cell of the robot's own map snapshot at or before the candidate time.

    uv run python ../tools/candidate_wall_eval.py ../data/recordings/unity_house_*
"""
import json, math, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'brain'))
from brain.nav_io import load_scenario, map_geometry, recording_frame, read_json_lines

rows = []
for rec in map(Path, sys.argv[1:]):
    try:
        meta = json.loads((rec / 'meta.json').read_text()); scenario = load_scenario(meta)
        people = [(p['id'], np.array([p['position_house'][0], p['position_house'][2]])) for p in json.loads((rec / 'oracle' / 'survivors.json').read_text())]
    except Exception:
        continue
    for robot in meta['robots']:
        d = rec / robot
        if not (d / 'decisions.jsonl').exists() or not (d / 'brain_map.txt').exists():
            continue
        states = read_json_lines(d / 'brain_state.jsonl')
        try:
            T, frame = recording_frame(states, scenario, robot)
        except Exception:
            continue
        snaps = [(float(l.split()[0]), l.split()[4]) for l in (d / 'brain_map.txt').read_text().splitlines() if l[:1].isdigit()]
        cache = {}
        for line in (d / 'decisions.jsonl').read_text().splitlines():
            e = json.loads(line)
            if e['kind'] != 'survivor_candidate' or 'x_m' not in e.get('numbers', {}):
                continue
            n = e['numbers']; t = e['t']
            prior = [s for s in snaps if s[0] <= t + 1e-6]
            if not prior:
                continue
            st, rel = prior[-1]
            if rel not in cache:
                img = cv2.imread(str(d / rel), 0)
                if img is None:
                    continue
                occ = np.flipud(img) == 0
                dist = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 5)
                cache[rel] = (dist, img.shape, occ)
            dist, shape = cache[rel][0], cache[rel][1]
            origin, cell = map_geometry(shape, frame)
            i, j = int((n['x_m'] - origin[0]) / cell), int((n['y_m'] - origin[1]) / cell)
            if not (0 <= i < shape[1] and 0 <= j < shape[0]):
                continue
            c, s = math.cos(T[2]), math.sin(T[2])
            h = T[:2] + np.array([c * n['x_m'] - s * n['y_m'], s * n['x_m'] + c * n['y_m']])
            near = min(float(np.linalg.norm(h - q)) for _, q in people)
            # Extent of the occupied structure under the candidate: the connected occupied component (8-neighbour) touching
            # a 0.3 m disc around it, within a 6 m window. A wall runs on; a body is a compact blob.
            img = cache[rel][2] if len(cache[rel]) > 2 else None
            if img is None:
                occ = np.flipud(cv2.imread(str(d / rel), 0)) == 0; cache[rel] = (dist, shape, occ); img = occ
            w = int(3.0 / cell); y0, y1, x0, x1 = max(0, j - w), min(shape[0], j + w), max(0, i - w), min(shape[1], i + w)
            win = img[y0:y1, x0:x1].astype(np.uint8)
            nlab, lab = cv2.connectedComponents(win, connectivity=8)
            r = int(.3 / cell); cj, ci = j - y0, i - x0
            ids = set(lab[max(0, cj - r):cj + r + 1, max(0, ci - r):ci + r + 1].ravel()) - {0}
            extent = 0.0
            for k in ids:
                ys, xs = np.nonzero(lab == k)
                extent = max(extent, float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min())) * cell)
            rows.append(dict(run=rec.name, robot=robot, t=t, rgb=int(n.get('rgb', 0)), person=near <= 1.0, wall_m=float(dist[j, i]) * cell,
                             area=n.get('area_m2'), near_m=near, extent=extent))
thermal = [r for r in rows if not r['rgb']]
print(f'{len(rows)} candidates ({len(thermal)} thermal) from {len(set(r["run"] for r in rows))} runs')
for name, group in (('thermal', thermal), ('colour', [r for r in rows if r['rgb']])):
    for label in (True, False):
        g = sorted(r['wall_m'] for r in group if r['person'] == label)
        if g:
            q = lambda p: g[min(len(g) - 1, int(p * len(g)))]
            print(f'{name:8s} {"person" if label else "not   "} n={len(g):4d}  wall distance m: p05 {q(.05):.2f}  p25 {q(.25):.2f}  median {q(.5):.2f}  p75 {q(.75):.2f}')
for thr in (.1, .15, .2, .25, .3):
    lost = sum(1 for r in thermal if r['person'] and r['wall_m'] <= thr); kept_false = sum(1 for r in thermal if not r['person'] and r['wall_m'] > thr)
    cut = sum(1 for r in thermal if not r['person'] and r['wall_m'] <= thr)
    print(f'drop thermal candidates within {thr:.2f} m of a mapped wall: removes {cut} non-person, loses {lost} person; non-person left {kept_false}')

print('thermal candidates that are real people:', [(r['run'][-12:], r['robot'], round(r['t'], 1), round(r['near_m'], 2), round(r['extent'], 1)) for r in thermal if r['person']])
for label in (True, False):
    g = sorted(r['extent'] for r in thermal if r['person'] == label)
    if g: print(f'thermal {"person" if label else "not   "} occupied-structure extent m: min {g[0]:.1f} p25 {g[len(g)//4]:.1f} median {g[len(g)//2]:.1f} max {g[-1]:.1f}')
for thr in (2.0, 2.5, 3.0, 4.0):
    cut = sum(1 for r in thermal if not r['person'] and r['extent'] >= thr); lost = sum(1 for r in thermal if r['person'] and r['extent'] >= thr)
    print(f'drop thermal candidates on a structure >= {thr} m long: removes {cut}/{sum(1 for r in thermal if not r["person"])} non-person, loses {lost}/2 person')
