"""Evaluator check: when was each true person actually visible to a robot's colour camera within 4 m?

For every robot and every sampled frame, each person's body points (at their recorded position, several heights) are
projected into the camera with the TRUE camera pose (oracle/<robot>_groundtruth.txt, Unity world: x right, y up, z
forward; the oracle writes the optical pose in the house frame, x east, y down, z north) and the recorded depth image decides occlusion: a point counts as seen when it is in the image, within range,
and the depth there is not more than 0.4 m nearer than the point. Compares the visible spans with the robots' candidate
and confirmation events (placed with the true spawn pose). Evaluator only: robots never read any of this.

    uv run python ../tools/person_visibility.py ../data/recordings/<run> [--every 4] [--range 4]
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

rec = Path(sys.argv[1])
every = int(sys.argv[sys.argv.index('--every') + 1]) if '--every' in sys.argv else 4
max_range = float(sys.argv[sys.argv.index('--range') + 1]) if '--range' in sys.argv else 4.0
meta = json.loads((rec / 'meta.json').read_text())
scenario = json.loads((Path(__file__).resolve().parents[1] / 'config' / meta['scenario']).read_text())
spawn = {r['id']: (np.array(r['spawn']['at'], float), -math.radians(r['spawn']['yaw'])) for r in scenario['robots']}
people = json.loads((rec / 'oracle' / 'survivors.json').read_text())
HEIGHTS = (.15, .3, .45, .7)


def quat_matrix(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def house(robot, x, y):
    o, yaw = spawn[robot]; c, s = math.cos(yaw), math.sin(yaw)
    return o + np.array([c * x - s * y, s * x + c * y])


report = {p['id']: {} for p in people}
for robot in meta['robots']:
    d = rec / robot
    cam = {k.split(':')[0].strip(): float(k.split(':')[1]) for k in (d / 'camera.yaml').read_text().splitlines()
           if k.startswith('Camera1.f') or k.startswith('Camera1.c')}
    fx, fy, cx, cy = cam['Camera1.fx'], cam['Camera1.fy'], cam['Camera1.cx'], cam['Camera1.cy']
    gt = np.array([[float(v) for v in l.split()[:8]] for l in (rec / 'oracle' / f'{robot}_groundtruth.txt').read_text().splitlines() if l[:1].isdigit()])
    depth_rows = [l.split() for l in (d / 'depth.txt').read_text().splitlines() if l[:1].isdigit()]
    for k, (ts, rel) in enumerate(depth_rows[::every]):
        t = float(ts)
        if t < gt[0, 0] or t > gt[-1, 0]:
            continue
        i = int(np.argmin(np.abs(gt[:, 0] - t)))
        if abs(gt[i, 0] - t) > .05:
            continue
        pos, R = gt[i, 1:4], quat_matrix(*gt[i, 4:8])
        depth = None
        for p in people:
            px, _, pz = p['position_house']
            seen = 0
            for h in HEIGHTS:
                c = R.T @ (np.array([px, -h, pz]) - pos)          # house frame y is down; optical frame x right, y down, z ahead
                if c[2] < .3 or c[2] > max_range:
                    continue
                u, v = cx + fx * c[0] / c[2], cy + fy * c[1] / c[2]
                if not (0 <= u < 640 and 0 <= v < 480):
                    continue
                if depth is None:
                    depth = cv2.imread(str(d / rel), cv2.IMREAD_UNCHANGED)
                z = depth[int(v), int(u)] * .001
                if z > 0 and z >= c[2] - .4:   # no depth reading (smoke, out of range) is unknown, not seen
                    seen += 1
            if seen >= 2:
                report[p['id']].setdefault(robot, []).append((t, float(np.linalg.norm(pos[[0, 2]] - [px, pz]))))
frame_dt = every / 15.0
finds = {}
for robot in meta['robots']:
    for l in (rec / robot / 'decisions.jsonl').read_text().splitlines():
        e = json.loads(l)
        if e['kind'] in ('survivor_candidate', 'survivor_confirmed') and 'x_m' in e.get('numbers', {}):
            finds.setdefault(robot, []).append((e['t'], e['kind'], house(robot, e['numbers']['x_m'], e['numbers']['y_m'])))
for p in people:
    px, _, pz = p['position_house']
    print(f"{p['id']} {p['room']:<10} ({px:5.1f},{pz:5.1f}) {p['pose']}")
    for robot in meta['robots']:
        spans = report[p['id']].get(robot, [])
        near = [(t, k) for t, k, q in finds.get(robot, []) if np.linalg.norm(q - [px, pz]) <= 1.0]
        cand = min((t for t, k in near if k == 'survivor_candidate'), default=None)
        conf = min((t for t, k in near if k == 'survivor_confirmed'), default=None)
        if spans:
            print(f"   {robot}: visible within {max_range:.0f} m for {len(spans) * frame_dt:5.1f} s, first at {spans[0][0]:6.1f} s "
                  f"(closest {min(r for _, r in spans):.1f} m); candidate {cand if cand is None else round(cand, 1)}, confirmed {conf if conf is None else round(conf, 1)}")
        else:
            print(f"   {robot}: never visible within {max_range:.0f} m" + (f"; candidate {round(cand, 1)}" if cand else ''))
