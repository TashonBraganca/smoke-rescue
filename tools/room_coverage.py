"""Per-room searched floor for a finished recording, evaluator side (reads truth: the floor plan and spawn poses).

Each robot's last searched mask is placed with its true spawn pose; a room's share is the fraction of its free floor
(room rectangle minus true walls and furniture, as in nav_eval.floor_mask) that either robot's camera searched.

    uv run python ../tools/room_coverage.py ../data/recordings/<run>
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'brain'))
from brain.nav_eval import truth_occupancy
from brain.nav_io import (latest_map, load_scenario, map_geometry, read_json_lines, recording_frame, rotation,
                          sample_grid, searched_for)
from brain.oracle import load_floorplan, load_house_points


def main(recording):
    rec = Path(recording)
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
    team = np.zeros(shape, bool); per = {}; mixed = []
    for robot in meta['robots']:
        snap = latest_map(rec / robot)
        if snap is None:
            continue
        t, grid = snap
        searched = searched_for(rec / robot, t)
        if searched is None:
            continue
        transform, frame = recording_frame(read_json_lines(rec / robot / 'brain_state.jsonl'), scenario, robot)
        src_origin, src_cell = map_geometry(grid.shape, frame)
        own = (xy - transform[:2]) @ rotation(transform[2])
        mask = sample_grid(searched, own, src_origin, src_cell, fill=0) == 255
        team |= mask
        # Per robot: its own cameras' floor. The searched layer also holds floor a linked partner sent, so without the
        # own-only layer (recordings before 22 Oct 14:40) the robot's column includes its partner's floor.
        own_layer = searched_for(rec / robot, t, 'own_searched')
        if own_layer is not None:
            mask = sample_grid(own_layer, own, src_origin, src_cell, fill=0) == 255
        elif any(r.get('launch_pad') is not None for r in scenario.get('robots', [])):
            mixed.append(robot)
        per[robot] = mask
    rows = []
    for r in rooms:
        a, b, c, d = r['rect']
        floor = (xy[..., 0] >= a) & (xy[..., 0] < c) & (xy[..., 1] >= b) & (xy[..., 1] < d) & ~occupied
        n = int(floor.sum())
        rows.append((r['id'], n * cell * cell, float((team & floor).sum() / max(1, n)),
                     {k: float((v & floor).sum() / max(1, n)) for k, v in per.items()}))
    total = sum(n for _, n, _, _ in rows)
    print(f'{"room":<12} {"floor m2":>8} {"team":>6}  ' + '  '.join(f'{k:>8}' for k in per))
    for name, m2, share, by in rows:
        print(f'{name:<12} {m2:8.1f} {share:6.0%}  ' + '  '.join(f'{by[k]:8.0%}' for k in per))
    print(f'{"all":<12} {total:8.1f} {sum(m2 * s for _, m2, s, _ in rows) / max(total, 1e-9):6.0%}')
    if mixed:
        print(f'note: {", ".join(mixed)} columns include floor received from the partner (no own-only layer in this '
              'recording); only the team column measures coverage')


if __name__ == '__main__':
    main(sys.argv[1])
