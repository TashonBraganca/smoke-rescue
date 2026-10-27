"""World-side room-block editing. The API and mission task run these same checks.

Blocks keep their generator dimensions. Translation snaps relative to each original
room on a half-metre grid. The union of the original rooms fixes the outer outline.
The robot brain receives only the resulting sensor observations.
"""
import copy
import math

from validate_building import inside, overlap, boundary_side
from validate_ranches import furniture_clear, room_paths
from placement import validate as validate_placement

SNAP_M = .5  # Requested room-drag grid; existing plan origins need not lie on this grid.
EPS = 1e-5  # Compiled gate geometry retains six decimal places.
DOOR_M = 1.0  # Ranch generators use one-metre clear openings, above the 0.9 m minimum.


def area(rect):
    a, b, c, d = rect
    return (c - a) * (d - b)


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def walls(rooms, doors):
    """Merge shared room edges and cut real openings, as in the existing house viewer."""
    lines = {}
    for r in rooms:
        a, b, c, d = r['rect']
        for axis, fixed, lo, hi in [('z', b, a, c), ('z', d, a, c), ('x', a, b, d), ('x', c, b, d)]:
            lines.setdefault((axis, fixed), []).append([lo, hi])
    result, lintels = [], []
    for (axis, fixed), intervals in lines.items():
        merged = []
        for lo, hi in sorted(intervals):
            if merged and lo <= merged[-1][1] + EPS:
                merged[-1][1] = max(hi, merged[-1][1])
            else:
                merged.append([lo, hi])
        for lo, hi in merged:
            pieces = [[lo, hi]]
            for door in doors:
                k, j = (1, 0) if axis == 'z' else (0, 1)
                if door['wall'] != axis or abs(door['at'][k] - fixed) > EPS:
                    continue
                left, right = door['at'][j] - door['width'] / 2, door['at'][j] + door['width'] / 2
                if left < lo - EPS or right > hi + EPS:
                    continue
                lintels.append([left, fixed, right, fixed] if axis == 'z' else [fixed, left, fixed, right])
                pieces = [[a, min(b, left)] for a, b in pieces if a < left] + [[max(a, right), b] for a, b in pieces if b > right]
            for a, b in pieces:
                if b - a > EPS:
                    result.append([a, fixed, b, fixed] if axis == 'z' else [fixed, a, fixed, b])
    return result, lintels


def build(house, submitted):
    original = house['placement_geometry']
    old = {r['id']: r for r in original['rooms']}
    if not isinstance(submitted, list) or len(submitted) != len(old):
        raise ValueError('Keep every room in the fixed house outline.')
    if any(not isinstance(r, dict) or set(r) != {'id', 'rect'} or not isinstance(r['id'], str) for r in submitted):
        raise ValueError('Each room needs its original name and four rectangle coordinates.')
    if {r['id'] for r in submitted} != set(old):
        raise ValueError('Keep every room once. The room names are fixed.')
    rooms, moves = [], {}
    for value in submitted:
        r = copy.deepcopy(old[value['id']]); box = value['rect']
        if not isinstance(box, list) or len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 50 for v in box):
            raise ValueError('Room coordinates must be finite points within the house.')
        before = r['rect']; dx, dy = box[0] - before[0], box[1] - before[1]
        if abs((box[2] - box[0]) - (before[2] - before[0])) > EPS or abs((box[3] - box[1]) - (before[3] - before[1])) > EPS:
            raise ValueError(r['name'] + ' must keep its generator dimensions. Slide or swap the block.')
        if any(abs(v / SNAP_M - round(v / SNAP_M)) > EPS for v in (dx, dy)):
            raise ValueError('Move rooms in 0.5 m steps from their original positions.')
        if abs(sum(intersection(box, t['rect']) for t in old.values()) - area(box)) > EPS:
            raise ValueError(r['name'] + ' leaves the fixed house outline.')
        r['rect'] = [round(v, 6) for v in box]; rooms.append(r); moves[r['id']] = [dx, dy]
    for i, r in enumerate(rooms):
        if any(overlap(r['rect'], t['rect']) for t in rooms[i + 1:]):
            raise ValueError(r['name'] + ' overlaps another room and leaves a gap. Swap matching blocks or undo.')
    # Fixed block areas, no overlap and containment in the original union prove no missing floor.
    if abs(sum(area(r['rect']) for r in rooms) - sum(area(r['rect']) for r in old.values())) > EPS:
        raise ValueError('The rooms must fill the fixed outline without gaps.')
    furniture = copy.deepcopy(original['furniture'])
    for f in furniture:
        dx, dy = moves.get(f['room'], [0, 0]); f['at'] = [f['at'][0] + dx, f['at'][1] + dy]
    fire = original['footprints']
    front = copy.deepcopy(next(d for d in original['doors'] if 'outside' in d['between']))
    entry = next((r for r in rooms if boundary_side(front, r['rect'])), None)
    if not entry:
        raise ValueError('Keep a room connected to the fixed front door.')
    front['between'] = ['outside', entry['id']]; front['width'] = max(.9, front['width'])
    doors = [front]

    def clear(door, pair):
        plan = {'doors': [door]}
        for r in pair:
            objects = [f for f in furniture if f['room'] == r['id']]
            if any(not furniture_clear(f, r, plan, fire, objects[:i]) for i, f in enumerate(objects)):
                return False
            if room_paths(r, [door], objects, fire, require_all=True):
                return False
        return True

    if not clear(front, [entry]):
        raise ValueError('Furniture blocks the fixed front door or its 0.9 m path.')
    for i, r in enumerate(rooms):
        for t in rooms[i + 1:]:
            a, b = r['rect'], t['rect']; shared = None
            for axis, ai, bi, low, high in [('x', 2, 0, 1, 3), ('x', 0, 2, 1, 3), ('z', 3, 1, 0, 2), ('z', 1, 3, 0, 2)]:
                lo, hi = max(a[low], b[low]), min(a[high], b[high])
                if abs(a[ai] - b[bi]) < EPS and hi - lo >= DOOR_M + .2:
                    shared = axis, a[ai], lo, hi; break
            if not shared:
                continue
            axis, fixed, lo, hi = shared
            # Prefer the old opening when it still belongs to this pair, then search clear approaches.
            candidates = [d['at'][1 if axis == 'x' else 0] for d in original['doors']
                          if set(d['between']) == {r['id'], t['id']} and d['wall'] == axis]
            candidates += [(lo + hi) / 2] + [lo + .6 + .25 * j for j in range(int((hi - lo - 1.2) / .25) + 1)]
            for q in candidates:
                if not lo + .5 + EPS <= q <= hi - .5 - EPS:
                    continue
                door = dict(id='edit_%d' % len(doors), wall=axis, at=[fixed, q] if axis == 'x' else [q, fixed],
                            width=DOOR_M, between=[r['id'], t['id']])
                if clear(door, [r, t]):
                    doors.append(door); break
    graph = {r['id']: set() for r in rooms}; graph['outside'] = set()
    for d in doors:
        a, b = d['between']; graph[a].add(b); graph[b].add(a)
    reached, queue = {'outside'}, ['outside']
    while queue:
        for rid in graph[queue.pop()] - reached:
            reached.add(rid); queue.append(rid)
    if len(reached) != len(graph):
        raise ValueError('Every room needs a reachable door at least 0.9 m wide with clear approaches.')
    result = copy.deepcopy(house)
    result['placement_geometry'].update(rooms=rooms, doors=doors, furniture=furniture)
    result['walls'], result['low_walls'] = walls(rooms, doors)
    result['rooms_poly'] = [dict(id=r['id'], name=r['name'], kind=r['kind'], poly=[r['rect'][:2], [r['rect'][2], r['rect'][1]], r['rect'][2:], [r['rect'][0], r['rect'][3]]]) for r in rooms]
    result['furniture'] = [[[p[0] + moves[f['room']][0], p[1] + moves[f['room']][1]] for p in poly]
                           for f, poly in zip(original['furniture'], house['furniture'])]
    return result


def generated_placement(house):
    """Use shared placement/path checks to find a deterministic clear scene for an edit."""
    geometry = house['placement_geometry']; candidates = []
    for r in geometry['rooms']:
        a, b, c, d = r['rect']
        for y in range(math.ceil((b + .75) * 2), math.floor((d - .75) * 2) + 1):
            for x in range(math.ceil((a + .75) * 2), math.floor((c - .75) * 2) + 1):
                candidates.append((r['id'], [x / 2, y / 2]))
    fire = None
    for rid, p in candidates:
        try:
            fire = validate_placement(house, {'people': [], 'fire': p})['fire']; break
        except ValueError:
            pass
    if fire is None:
        raise ValueError('No clear fire origin remains. Undo the room move.')
    people, used = [], set()
    for rid, p in candidates:
        if rid in used:
            continue
        try:
            checked = validate_placement(house, {'people': people + [p], 'fire': fire})
            people = checked['people']; used.add(rid)
            if len(people) >= min(5, house['people']):
                break
        except ValueError:
            pass
    if not people:
        raise ValueError('No clear person placement remains. Undo the room move.')
    return {'people': people, 'fire': fire}


def key(base, rooms, placement, engine):
    import hashlib
    import json
    content = dict(base=base, rooms=sorted(rooms, key=lambda r: r['id']), placement=placement, engine=engine, layout_validator=1)
    return base + '__p' + hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:20]


def apply_world(root, run_dir, world, house):
    """Rebuild evaluation geometry and the fire plan in one gate coordinate frame."""
    import json
    plan = json.loads((root / 'config' / world['fire']['floorplan']).resolve().read_text())
    fire = json.loads((root / 'config' / world['fire']['config']).resolve().read_text())
    geometry = house['placement_geometry']
    by_id = {r['id']: r for r in geometry['rooms']}
    for r in plan['rooms']:
        r['rect'] = by_id[r['id']]['rect']
    rects = [r['rect'] for r in geometry['rooms']]
    plan['footprint'] = [min(r[0] for r in rects), min(r[1] for r in rects), max(r[2] for r in rects), max(r[3] for r in rects)]
    plan['footprint_rects'] = rects
    plan['doors'] = [dict(d, state='removed') for d in geometry['doors']]
    plan['landmarks'] = [dict(f) for f in geometry['furniture']]
    plan['fires'], plan['emergency_lights'] = [], []
    # Retain the original prism's material and measured height while moving its footprint.
    for source, edited, poly in zip(world['prisms'], geometry['furniture'], house['furniture']):
        old_kind = source['kind']; source['poly'] = poly
        props = copy.deepcopy(fire['furniture'].get(old_kind, {}))
        if not props:
            raise ValueError('The original furniture material is missing.')
        props['size_m'] = geometry['footprints']['furniture'][edited['kind']]['size_m']
        fire['furniture'][edited['kind']] = props
    world['segments'] = [w + [0.0, world['ceiling_m']] for w in house['walls']] + [w + [world['lintel_m'], world['ceiling_m']] for w in house['low_walls']]
    world['rooms'] = [{k: r[k] for k in ('id', 'kind', 'poly')} for r in house['rooms_poly']]
    world['gate'] = {'origin': [0, 0], 'yaw_deg': 0}
    world['fire']['obstacles'] = []
    path = run_dir / 'floorplan.json'; path.write_text(json.dumps(plan))
    world['fire']['floorplan'] = str(path)
    # entry.py writes the requested ignition after this reconstruction.
    path = run_dir / 'layout_fire.json'; path.write_text(json.dumps(fire))
    world['fire']['config'] = str(path)
    return world
