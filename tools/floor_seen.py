"""Evaluator check: how much of each room's floor did a robot's camera actually see, within search range?

The robots' own searched layer marks floor under every depth ray, including rays that pass high over a table to the wall
behind it, so it counts floor behind furniture as searched (held-out seed 914 run 1: a person slumped by the kitchen
island lay on "searched" floor and was never once in view). This tool counts a floor cell as seen only where a recorded
depth pixel, placed with the TRUE camera pose, lands within --low metres of the floor (the floor itself, or something
lying on it) and within --range of the camera: a person lying there would have been in view. Evaluator only.

    uv run python ../tools/floor_seen.py ../data/recordings/<run> [--every 2] [--range 4] [--low 0.4]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'brain'))
from brain.nav_eval import truth_occupancy
from brain.nav_io import (latest_map, load_scenario, map_geometry, read_json_lines, recording_frame, rotation,
                          sample_grid, searched_for)
from brain.oracle import load_floorplan, load_house_points


def arg(name, default):
    return float(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def quat_matrix(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main(recording):
    rec = Path(recording)
    every, max_range, low, stride = int(arg('--every', 2)), arg('--range', 4.0), arg('--low', .4), 4
    meta = json.loads((rec / 'meta.json').read_text())
    scenario, plan = load_scenario(meta), load_floorplan(rec)
    area = scenario.get('search_area')
    rooms = [r for r in plan['rooms'] if not area or (r['rect'][0] >= area[0] - 1e-6 and r['rect'][1] >= area[1] - 1e-6 and
                                                     r['rect'][2] <= area[2] + 1e-6 and r['rect'][3] <= area[3] + 1e-6)]
    cell = .1
    x0, y0 = min(r['rect'][0] for r in rooms) - 1, min(r['rect'][1] for r in rooms) - 1
    x1, y1 = max(r['rect'][2] for r in rooms) + 1, max(r['rect'][3] for r in rooms) + 1
    shape = (int(np.ceil((y1 - y0) / cell)), int(np.ceil((x1 - x0) / cell)))
    origin = np.array([x0, y0])
    occupied = truth_occupancy(load_house_points(rec), shape, origin, cell)
    yy, xx = np.indices(shape); xy = origin + (np.stack([xx, yy], -1) + .5) * cell
    seen, claimed = np.zeros(shape, bool), np.zeros(shape, bool)
    for robot in meta['robots']:
        d = rec / robot
        cam = {k.split(':')[0].strip(): float(k.split(':')[1]) for k in (d / 'camera.yaml').read_text().splitlines()
               if k.startswith('Camera1.f') or k.startswith('Camera1.c')}
        fx, fy, cx, cy = cam['Camera1.fx'], cam['Camera1.fy'], cam['Camera1.cx'], cam['Camera1.cy']
        gt = np.array([[float(v) for v in l.split()[:8]] for l in (rec / 'oracle' / f'{robot}_groundtruth.txt').read_text().splitlines() if l[:1].isdigit()])
        vv, uu = np.mgrid[0:480:stride, 0:640:stride]
        rays = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu, float)], -1).reshape(-1, 3)
        for ts, rel in [l.split() for l in (d / 'depth.txt').read_text().splitlines() if l[:1].isdigit()][::every]:
            t = float(ts); i = int(np.argmin(np.abs(gt[:, 0] - t)))
            if abs(gt[i, 0] - t) > .05:
                continue
            depth = cv2.imread(str(d / rel), cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue
            z = depth[::stride, ::stride].reshape(-1) * .001
            ok = (z > .3) & (z <= max_range)          # 0 is no reading (smoke, out of range): unknown, not seen
            pts = gt[i, 1:4] + (rays[ok] * z[ok, None]) @ quat_matrix(*gt[i, 4:8]).T   # house frame: x east, y down, z north
            pts = pts[-pts[:, 1] <= low]
            col = ((pts[:, 0] - origin[0]) / cell).astype(int); row = ((pts[:, 2] - origin[1]) / cell).astype(int)
            inside = (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
            seen[row[inside], col[inside]] = True
        snap = latest_map(d)
        if snap is not None and (searched := searched_for(d, snap[0])) is not None:
            transform, frame = recording_frame(read_json_lines(d / 'brain_state.jsonl'), scenario, robot)
            src_origin, src_cell = map_geometry(snap[1].shape, frame)
            claimed |= sample_grid(searched, (xy - transform[:2]) @ rotation(transform[2]), src_origin, src_cell, fill=0) == 255
    print(f'{"room":<12} {"floor m2":>8} {"seen":>6} {"claimed":>8}   (seen: floor in a depth image within {max_range:.0f} m, '
          f'true pose; claimed: the robots\' own searched layer)')
    tot = [0, 0, 0]
    for r in rooms:
        a, b, c, e = r['rect']
        floor = (xy[..., 0] >= a) & (xy[..., 0] < c) & (xy[..., 1] >= b) & (xy[..., 1] < e) & ~occupied
        n = int(floor.sum()); s, k = int((seen & floor).sum()), int((claimed & floor).sum())
        tot[0] += n; tot[1] += s; tot[2] += k
        print(f'{r["id"]:<12} {n * cell * cell:8.1f} {s / max(n, 1):6.0%} {k / max(n, 1):8.0%}')
    print(f'{"all":<12} {tot[0] * cell * cell:8.1f} {tot[1] / max(tot[0], 1):6.0%} {tot[2] / max(tot[0], 1):8.0%}')
    unseen_claimed = int((claimed & ~seen & ~occupied).sum())
    print(f'claimed but never seen: {unseen_claimed * cell * cell:.1f} m2')


if __name__ == '__main__':
    main(sys.argv[1])
