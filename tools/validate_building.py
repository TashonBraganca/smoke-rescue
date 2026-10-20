"""Check JSON geometry without Unity and draw data/evidence/building_plan.png.

Run with python3 tools/validate_building.py. Geometry checks use declared fire
footprints, not imported meshes. Pillow is used only to draw the evidence image.
Frozen pre-extension hashes protect the original house, props and fire settings.
The original narrow hallway, two narrow doors and three clipped fire footprints
and four original wall-clearance shortfalls are reported separately. Any new
violation fails with a nonzero exit code.
"""

from collections import deque
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-6
ORIGINAL_ROOMS = ('living', 'foyer', 'kitchen', 'hallway', 'bedroom_1',
                  'bathroom', 'closet', 'study', 'bedroom_2', 'laundry')
# Populated from the untouched files before the extension was written.
REFERENCE = {
    "rooms": "989ce33f43692ee68c4be9c91fcaf277ddf6226d8a4357b0d6cb4c95f0069f45",
    "doors": "a75fcc628b9f7a06d41186f6e644963f8598fb5858afef29b848cf513647a2a4",
    "landmarks": "78aebd491ca320cdc5086f6d115a47b11329e415ee2b933aa2aa797a6f6dc927",
    "fires": "0b0b187e486a9486a40923c1a276a413b14a488792e40f3571102e8ab91ddd6e",
    "emergency_lights": "b83959bcbd619c9d62afd68ac2660d7f9caf52f7053064f189d3b3caffe46ea4",
    "props": "3059704ee7cbb0b7c109a735841006b54c7d4e1347577016e27bae03b0a8de38",
    "materials": "f7ca4ecf98337de2ef9bbeb75c72575421475977457f42e8789dabb4eb01c8c0",
    "fire": "2ac2781cd629f0558241a0756036c329c5514b321a21edf4f66010f59f4d23f8",
    "new_kinds": [
        "apartment_bed",
        "apartment_counter",
        "apartment_fridge",
        "apartment_modern_sofa",
        "hall_plant",
        "office_meeting_table",
        "office_task_chair",
        "office_workstation",
        "steel_storage_shelf",
        "supply_boxes",
        "utility_washer",
        "workshop_bench",
        "workshop_drawers"
    ],
    "new_materials": [
        "apartment_plaster_cream",
        "office_plaster_blue",
        "workshop_concrete_ochre"
    ],
    "poses": [
        "collapsed_at_desk",
        "curled_on_bed",
        "lying_on_side",
        "slumped_against_wall"
    ],
    "survivors": "5f1c6dc88d59a3c40af5657955d2746a5fd752fcf47fd712817098ed9e9db1a2",
    "clear_robot_A": "dc399de76db051eb1b26b48ecece285f996bd822358707e8a2328e533edd00a8",
    "clear_robot_B": "91045b5b636bdb788dab9f12bfd17f9c226057c24fde85a691f55138c3d9bdab"
}
WALL_ART = {'large_landscape_painting', 'framed_family_photos',
            'bedroom1_art_dartboard', 'bedroom1_art_clock'} | {
                'hallway_art_' + c for c in 'abcdefgh'}
LEGACY_MOUNTS = {
    'brick_fireplace': {'west'}, 'coat_rack_with_coats': {'west', 'south'},
    'console_table_with_vase': {'west'}, 'tall_steel_fridge': {'east'},
    'stove_and_hood': {'south'}, 'grandfather_clock': {'east'},
    'blue_double_bed': {'north'}, 'tall_wardrobe': {'west'},
    'clawfoot_bathtub': {'north'}, 'sink_with_mirror': {'east'},
    'clothes_racks_both_sides': {'west', 'east'},
    'floor_to_ceiling_bookshelves': {'west'},
    'wooden_desk_with_lamp': {'north'}, 'antique_globe': {'east'},
    'bunk_bed': {'north'}, 'toy_chest_and_toys': {'west'},
    'washer_and_dryer': {'east'}, 'water_heater_and_pipes': {'north', 'west'},
}


def read(name):
    return json.loads((ROOT / 'config' / (name + '.json')).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def merged_scenario(name):
    child = read(name)
    if 'inherits' not in child:
        return child
    parent = merged_scenario(Path(child['inherits']).stem)
    parent.update({k: v for k, v in child.items()
                   if k not in ('inherits', 'extra_robots')})
    parent['robots'] += child.get('extra_robots', [])
    return parent


def inside(at, rect, margin=0):
    x, z = at
    a, b, c, d = rect
    return a + margin - EPS <= x <= c - margin + EPS and b + margin - EPS <= z <= d - margin + EPS


def overlap(a, b):
    return min(a[2], b[2]) - max(a[0], b[0]) > EPS and min(a[3], b[3]) - max(a[1], b[1]) > EPS


def footprint(lm, fire):
    w, d = fire['furniture'][lm['kind']]['size_m']
    angle = math.radians(lm.get('yaw', 0))
    hx = (abs(math.cos(angle)) * w + abs(math.sin(angle)) * d) / 2
    hz = (abs(math.sin(angle)) * w + abs(math.cos(angle)) * d) / 2
    x, z = lm['at']
    return [x - hx, z - hz, x + hx, z + hz]


def boundary_side(door, rect):
    x, z = door['at']
    a, b, c, d = rect
    half = door['width'] / 2
    if door['wall'] == 'z' and a - EPS <= x - half and x + half <= c + EPS:
        if abs(z - b) < EPS:
            return 'south'
        if abs(z - d) < EPS:
            return 'north'
    if door['wall'] == 'x' and b - EPS <= z - half and z + half <= d + EPS:
        if abs(x - a) < EPS:
            return 'west'
        if abs(x - c) < EPS:
            return 'east'
    return None


def wall_clearances(box, rect):
    return dict(west=box[0] - rect[0], south=box[1] - rect[1],
                east=rect[2] - box[2], north=rect[3] - box[3])


def window_spans(plan):
    """Mirror HouseBuilder.WindowSpans, including its ordered candidate search."""
    fp = plan['footprint']
    result = []
    for room in plan['rooms']:
        x0, z0, x1, z1 = room['rect']
        edges = [('z', z0, x0, x1), ('z', z1, x0, x1),
                 ('x', x0, z0, z1), ('x', x1, z0, z1)]
        for axis, at, lo, hi in edges:
            bounds = (fp[1], fp[3]) if axis == 'z' else (fp[0], fp[2])
            if at not in bounds or hi - lo < 2.2:
                continue
            mid = (lo + hi) / 2
            for pos in (mid, mid - 1.2, mid + 1.2, mid - 2.2, mid + 2.2):
                if pos - 0.6 < lo + 0.4 or pos + 0.6 > hi - 0.4:
                    continue
                point = [pos, at] if axis == 'z' else [at, pos]
                if any(math.dist(d['at'], point) < 0.6 + d['width'] / 2 + 0.8
                       for d in plan['doors']):
                    continue
                if any(math.dist(l['at'], point) < 1.2 for l in plan['landmarks']):
                    continue
                result.append(dict(room=room['id'], wall=axis, at=point, width=1.2))
                break
    return result


def route_crossings(route, spawn, plan):
    """Check straight route segments at wall crossings; Unity must test turns."""
    failures = []
    prev = spawn
    walls = set()
    for r in plan['rooms']:
        a, b, c, d = r['rect']
        walls.update([('x', a, b, d), ('x', c, b, d),
                      ('z', b, a, c), ('z', d, a, c)])
    for index, point in enumerate(route):
        for axis, line, lo, hi in walls:
            across = 0 if axis == 'x' else 1
            along = 1 - across
            delta = point[across] - prev[across]
            if abs(delta) < EPS:
                continue
            t = (line - prev[across]) / delta
            if not EPS < t < 1 - EPS:
                continue
            pos = prev[along] + t * (point[along] - prev[along])
            if not lo + EPS < pos < hi - EPS:
                continue
            cuts = [d for d in plan['doors'] if d['wall'] == axis and
                    abs(d['at'][across] - line) < EPS and d['state'] in ('open', 'removed') and
                    abs(d['at'][along] - pos) + 0.18 <= d['width'] / 2 + EPS]
            if not cuts:
                failures.append(f'segment {index} crosses {axis}={line:g} at {pos:g} without a capsule-width door gap')
        prev = point
    return sorted(set(failures))


def room_paths(room, doors, obstacles, fire):
    """Find 0.9 m paths on a 0.1 m grid within each new room.

    Inflate declared footprints by 0.45 m and include wall thickness. Door
    endpoints sit 0.75 m inside the room. This checks the room's free floor,
    separately from the exact boundary gaps checked above. Wall art sits above
    the robot; it does not occupy floor space. Imported mesh bounds still need
    a Unity check.
    """
    x0, z0, x1, z1 = room['rect']
    step, inset, radius = 0.1, 0.75, 0.45
    nx, nz = round((x1 - x0 - 2 * inset) / step), round((z1 - z0 - 2 * inset) / step)
    boxes = [footprint(o, fire) for o in obstacles if o['kind'] not in WALL_ART]

    def point(key):
        return (x0 + inset + key[0] * step, z0 + inset + key[1] * step)

    def nearest(p):
        return (max(0, min(nx, round((p[0] - x0 - inset) / step))),
                max(0, min(nz, round((p[1] - z0 - inset) / step))))

    def free(key):
        x, z = point(key)
        for a, b, c, d in boxes:
            dx, dz = max(a - x, 0, x - c), max(b - z, 0, z - d)
            if dx * dx + dz * dz < radius * radius - EPS:
                return False
        return True

    centre = nearest([(x0 + x1) / 2, (z0 + z1) / 2])
    if not free(centre):
        return ['room centre is obstructed at 0.9 m path width']
    seen, queue = {centre}, deque([centre])
    while queue:
        ix, iz = queue.popleft()
        for key in ((ix - 1, iz), (ix + 1, iz), (ix, iz - 1), (ix, iz + 1)):
            if 0 <= key[0] <= nx and 0 <= key[1] <= nz and key not in seen and free(key):
                seen.add(key)
                queue.append(key)
    failures = []
    for door in doors:
        side = boundary_side(door, room['rect'])
        if side is None:
            failures.append(door['id'] + ' has no valid boundary approach')
            continue
        dx, dz = dict(south=(0, inset), north=(0, -inset), west=(inset, 0), east=(-inset, 0))[side]
        key = nearest([door['at'][0] + dx, door['at'][1] + dz])
        if key not in seen:
            failures.append(door['id'] + ' has no 0.9 m path to room centre')
    return failures


def draw_plan(plan, fire, scenarios, windows):
    from PIL import Image, ImageDraw, ImageFont

    scale, left, top = 32, 110, 130
    width, height = 1510, 1200
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=14)
    small = ImageFont.load_default(size=12)
    title = ImageFont.load_default(size=24)
    xy = lambda p: (left + p[0] * scale, top + (28 - p[1]) * scale)
    draw.text((left, 20), 'Smoke Rescue | 40 x 28 m | north up', font=title, fill='#17212B')
    draw.text((left, 56), 'South-west house retained. Apartment east. Office north. Workshop north-east.', font=font, fill='#17212B')
    for room in plan['rooms']:
        rid = room['id']
        color = '#EEE4CD' if rid in ORIGINAL_ROOMS else '#DCEBCD' if rid.startswith('apt') else '#D5E5F1' if rid.startswith('office') else '#F2D8BC'
        a, b, c, d = room['rect']
        draw.rectangle([xy([a, d]), xy([c, b])], fill=color, outline='#283139', width=3)
    for lm in plan['landmarks']:
        if lm['kind'] not in fire['furniture']:
            continue
        box = footprint(lm, fire)
        draw.rectangle([xy([box[0], box[3]]), xy([box[2], box[1]])], fill='#B7A18C', outline='#80634B')
    for o in scenarios['haze']['obstacles']:
        if o['kind'] not in fire['furniture']:
            continue
        box = footprint(o, fire)
        draw.rectangle([xy([box[0], box[3]]), xy([box[2], box[1]])], fill='#A94F38', outline='#792E1B')
    for win in windows:
        x, z = win['at']
        delta = [0.6, 0] if win['wall'] == 'z' else [0, 0.6]
        draw.line([xy([x - delta[0], z - delta[1]]), xy([x + delta[0], z + delta[1]])], fill='#5A9BC0', width=5)
    for door in plan['doors']:
        x, z = door['at']
        h = door['width'] / 2
        delta = [h, 0] if door['wall'] == 'z' else [0, h]
        ends = [xy([x - delta[0], z - delta[1]]), xy([x + delta[0], z + delta[1]])]
        draw.line(ends, fill='white', width=7)
        draw.line(ends, fill='#197448', width=3)
        if 'outside' in door['between']:
            side = boundary_side(door, plan['footprint'])
            if side is None:
                continue
            offsets = dict(south=(0, 16), north=(0, -26), west=(-18, -18), east=(18, -18))
            ox, oy = offsets[side]
            px, py = xy(door['at'])
            anchor = 'rm' if side == 'west' else 'lm' if side == 'east' else 'mm'
            label = door['id'].replace('d_', '')
            draw.text((px + ox, py + oy), label, font=small, fill='#125235', anchor=anchor)
    for room in plan['rooms']:
        a, b, c, d = room['rect']
        label = room['id']
        size = 14
        room_font = font
        while draw.textlength(label, font=room_font) > (c - a) * scale - 8 and size > 8:
            size -= 1
            room_font = ImageFont.load_default(size=size)
        px, py = xy([(a + c) / 2, (b + d) / 2])
        label_box = draw.textbbox((px, py), label, font=room_font, anchor='mm')
        draw.rectangle([label_box[0] - 2, label_box[1] - 2, label_box[2] + 2, label_box[3] + 2], fill='white')
        draw.text((px, py), label, font=room_font, fill='#17212B', anchor='mm')
    for scenario_name, color in [('haze', '#BA302A'), ('clear', '#425AA8')]:
        for rb in scenarios[scenario_name]['robots']:
            if scenario_name == 'clear' and rb['id'] == 'robot_A':
                continue
            x, z = rb['spawn']['at']
            px, py = xy([x, z])
            angle = math.radians(rb['spawn']['yaw'])
            end = xy([x + 0.9 * math.sin(angle), z + 0.9 * math.cos(angle)])
            draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=color)
            draw.line([(px, py), end], fill=color, width=3)
            label = ('clear B' if scenario_name == 'clear' else rb['id'])
            draw.text((px, py + 19), label, font=small, fill=color, anchor='mm')
    for tick in range(0, 41, 5):
        px, _ = xy([tick, 0])
        draw.text((px, top - 50), str(tick), font=small, fill='#30373B', anchor='mm')
    for tick in range(0, 29, 2):
        _, py = xy([0, tick])
        draw.text((30, py), str(tick), font=small, fill='#30373B', anchor='mm')
    draw.text((left, 1090), 'Green gaps = doors. Blue = generated windows. Brown = declared furniture footprints. Rust = scenario obstacles.', font=font, fill='#17212B')
    draw.text((left, 1115), 'Red arrows = autonomous spawns. Blue arrow = clear-air B. Room sizes include wall thickness.', font=font, fill='#17212B')
    draw.text((left, 1140), 'JSON checks do not prove mesh bounds, curved robot motion, frame rate, smoke timing or survivor body clearance.', font=font, fill='#17212B')
    path = ROOT / 'data/evidence/building_plan.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def validate():
    plan, props, materials, fire = [read(n) for n in ('floorplan', 'props', 'materials', 'fire')]
    scenarios = dict(haze=merged_scenario('scenario_haze_2agents'), clear=merged_scenario('scenario_clear_2agents'))
    errors, exceptions = [], []
    check_count = 0

    def check(ok, message):
        nonlocal check_count
        check_count += 1
        if not ok:
            errors.append(message)

    rooms = {r['id']: r for r in plan['rooms']}
    check(len(rooms) == len(plan['rooms']), 'Duplicate room ID')
    check(plan['footprint'] == [0, 0, 40, 28], 'Footprint must be 0..40 by 0..28')
    check(len({d['id'] for d in plan['doors']}) == len(plan['doors']), 'Duplicate door ID')
    for group, value in [('rooms', plan['rooms'][:10]), ('landmarks', plan['landmarks'][:34]),
                         ('fires', plan['fires']), ('emergency_lights', plan['emergency_lights'][:3])]:
        check(digest(value) == REFERENCE[group], 'Original ' + group + ' changed')
    old_doors = json.loads(json.dumps(plan['doors'][:14]))
    old_doors[2]['between'] = ['outside', 'laundry']
    check(digest(old_doors) == REFERENCE['doors'], 'Original doors changed beyond the permitted d_back connection')
    check(plan['doors'][2]['between'] == ['laundry', 'office_link'], 'd_back must join laundry and office_link while still swinging into laundry')
    for name, current, nested, added in [
        ('props', props, 'landmarks', REFERENCE['new_kinds']),
        ('materials', materials, 'materials', REFERENCE['new_materials']),
        ('fire', fire, 'furniture', REFERENCE['new_kinds']),
    ]:
        original = json.loads(json.dumps(current))
        for key in added:
            original[nested].pop(key, None)
        if name == 'props':
            original['survivors_posed'].pop('s5', None)
            original['survivors_posed'].pop('s6', None)
        if name == 'fire':
            for key in REFERENCE['new_materials']:
                original['surfaces'].pop(key, None)
        check(digest(original) == REFERENCE[name], 'Original ' + name + ' settings changed')

    area = 0
    block_areas = dict(house=0, apartment=0, office=0, workshop=0)
    pairs = []
    grid = fire['cell_size_m']
    for i, room in enumerate(plan['rooms']):
        rid, rect = room['id'], room['rect']
        a, b, c, d = rect
        check(c > a and d > b and inside([a, b], plan['footprint']) and inside([c, d], plan['footprint']), rid + ' is empty or outside footprint')
        check(all(abs(v / grid - round(v / grid)) < EPS for v in rect), rid + ' does not align to fire grid')
        size = (c - a) * (d - b)
        area += size
        block = 'house' if rid in ORIGINAL_ROOMS else 'apartment' if rid.startswith('apt') else 'office' if rid.startswith('office') else 'workshop'
        block_areas[block] += size
        if rid in ORIGINAL_ROOMS and min(c - a, d - b) < 2.4:
            exceptions.append(f'{rid}: preserved {min(c-a,d-b):g} m width below new 2.4 m rule')
        else:
            check(min(c - a, d - b) >= 2.4, rid + ' narrower than 2.4 m')
        for other in plan['rooms'][:i]:
            check(not overlap(rect, other['rect']), rid + ' overlaps ' + other['id'])
        for field in ('floor', 'walls', 'ceiling'):
            name = room[field]
            check(name in materials['materials'], f'{rid}: missing material {name}')
            check(name in fire['surfaces'] and fire['surfaces'].get(name) in fire['materials'], f'{rid}: missing fire surface {name}')
        if rid not in ORIGINAL_ROOMS:
            pairs.append((room['floor'], room['walls']))
            check(any(l['room'] == rid for l in plan['landmarks']), rid + ' has no landmark')
            check(any(inside(l['at'], rect) for l in plan['emergency_lights']), rid + ' has no emergency light')
    check(area == 1120 and all(v == 280 for v in block_areas.values()), 'Rooms must tile four 280 m2 blocks')
    check(len(pairs) == len(set(pairs)), 'New rooms must use distinct floor/wall pairs')

    graph = {rid: set() for rid in rooms}
    exterior = []
    opposite = dict(north='south', south='north', east='west', west='east')
    for i, door in enumerate(plan['doors']):
        rid_a, rid_b = door['between']
        check(door['wall'] in ('x', 'z') and door['state'] in ('open', 'removed', 'half', 'closed'), door['id'] + ' invalid axis/state')
        check(rid_a != rid_b and all(r == 'outside' or r in rooms for r in (rid_a, rid_b)), door['id'] + ' invalid room reference')
        if any(r != 'outside' and r not in rooms for r in (rid_a, rid_b)):
            continue
        sides = [boundary_side(door, plan['footprint'] if rid == 'outside' else rooms[rid]['rect']) for rid in (rid_a, rid_b)]
        check(all(sides), door['id'] + ' does not lie fully on both named boundaries')
        check(sides[0] == sides[1] if 'outside' in door['between'] else opposite.get(sides[0]) == sides[1], door['id'] + ' does not separate its named rooms')
        if door['width'] < 0.9 and i < 14:
            exceptions.append(f"{door['id']}: preserved {door['width']:g} m door below new 0.9 m rule")
        else:
            check(door['width'] >= 0.9, door['id'] + ' narrower than 0.9 m')
        for prev in plan['doors'][:i]:
            if prev['wall'] != door['wall']:
                continue
            across = 0 if door['wall'] == 'x' else 1
            along = 1 - across
            if abs(prev['at'][across] - door['at'][across]) < EPS:
                check(abs(prev['at'][along] - door['at'][along]) >= (prev['width'] + door['width']) / 2 + 0.14 - EPS, door['id'] + ' clashes with another door casing')
        if 'outside' in door['between']:
            exterior.append(door)
        elif door['state'] in ('open', 'removed'):
            graph[rid_a].add(rid_b)
            graph[rid_b].add(rid_a)
    for entry in exterior:
        interior = next(r for r in entry['between'] if r != 'outside')
        seen, queue = {interior}, deque([interior])
        while queue:
            for nxt in graph[queue.popleft()] - seen:
                seen.add(nxt)
                queue.append(nxt)
        check(entry['state'] in ('open', 'removed') and len(seen) == len(rooms), entry['id'] + ' cannot reach every room internally')
    check(len({boundary_side(d, plan['footprint']) for d in exterior}) == 4, 'Entries must cover four cardinal sides')

    for index, landmark in enumerate(plan['landmarks']):
        rid, kind = landmark['room'], landmark['kind']
        check(rid in rooms and inside(landmark['at'], rooms[rid]['rect']), f'Landmark {index} anchor outside room')
        check(kind in props['landmarks'], 'Missing prop kind ' + kind)
        check(kind in fire['furniture'], 'Missing fire furniture ' + kind)
        if rid not in rooms or kind not in fire['furniture']:
            continue
        box = footprint(landmark, fire)
        clearances = wall_clearances(box, rooms[rid]['rect'])
        if index in (10, 14, 32):
            exceptions.append(f'{rid}/{kind}: preserved estimated fire footprint extends {abs(min(clearances.values())):.3f} m beyond room; anchor inside')
        else:
            check(min(clearances.values()) >= -EPS, f'{rid}/{kind}: footprint outside room')
        allowed = set()
        if kind in WALL_ART or kind == 'sink_with_mirror':
            allowed.add({0: 'south', 90: 'west', 180: 'north', 270: 'east'}[int(landmark['yaw']) % 360])
        if index < 34:
            allowed |= LEGACY_MOUNTS.get(kind, set())
        for side, distance in clearances.items():
            if index < 34 and distance < 0.5 - EPS and side not in allowed:
                exceptions.append(f'{rid}/{kind}: preserved {distance:.3f} m clearance to unrelated {side} wall, below 0.5 m')
            else:
                face_inset = 0
                if index >= 34:
                    edge = dict(west=0, south=1, east=2, north=3)[side]
                    external = abs(rooms[rid]['rect'][edge] - plan['footprint'][edge]) < EPS
                    face_inset = plan['exterior_wall_thickness'] if external else plan['interior_wall_thickness'] / 2
                check(distance >= 0.5 + face_inset - EPS or side in allowed, f'{rid}/{kind}: {distance-face_inset:.3f} m from unrelated {side} wall face')
    for kind, definition in fire['furniture'].items():
        check(definition['material'] in fire['materials'] and len(definition['size_m']) == 2 and min(definition['size_m']) > 0 and bool(definition['justification']), 'Invalid fire furniture ' + kind)
    asset_count = 0
    for kind in {l['kind'] for l in plan['landmarks']} | {o['kind'] for o in scenarios['haze']['obstacles']}:
        for part in props['landmarks'].get(kind, []):
            rel = part.get('downloaded') or 'models/' + part.get('polyhaven', 'blender_' + part.get('blender', '')) + '.glb'
            check((ROOT / 'assets_cache' / rel).is_file(), f'{kind}: missing model {rel}')
            asset_count += 1
            if kind in REFERENCE['new_kinds']:
                check(part.get('fit_width', 0) > 0, kind + ' needs fit_width')
                check('downloaded' not in part or bool(part.get('credit')), kind + ' needs downloaded model credit')
    for name in REFERENCE['new_materials']:
        spec = materials['materials'].get(name, {})
        check(spec.get('source') in ('polyhaven', 'ambientcg') and bool(spec.get('id')), name + ' missing or unsupported source')

    for label, scenario in scenarios.items():
        check(len({s['id'] for s in scenario['survivors']}) == len(scenario['survivors']), label + ' duplicate survivor ID')
        for s in scenario['survivors']:
            check(s['room'] in rooms and inside(s['at'], rooms[s['room']]['rect']), label + ' survivor outside named room: ' + s['id'])
            check(s['pose'] in REFERENCE['poses'], label + ' unknown survivor pose: ' + s['pose'])
            part = props['survivors_posed'].get(s['id'], {})
            check(bool(part) and (ROOT / 'assets_cache' / part.get('downloaded', 'missing')).is_file(), 'Missing posed survivor mesh ' + s['id'])
        for rb in scenario['robots']:
            at = rb['spawn']['at']
            check(not inside(at, plan['footprint']), label + ' spawn inside footprint: ' + rb['id'])
            near = [d for d in exterior if math.dist(d['at'], at) <= 2]
            check(bool(near), label + ' spawn farther than 2 m from an exterior door: ' + rb['id'])
            if near:
                entry = min(near, key=lambda d: math.dist(d['at'], at))
                heading = math.degrees(math.atan2(entry['at'][0] - at[0], entry['at'][1] - at[1])) % 360
                check(abs((rb['spawn']['yaw'] - heading + 180) % 360 - 180) < 1, label + ' spawn does not face door: ' + rb['id'])
            for failure in route_crossings(rb['route'], at, plan):
                check(False, label + '/' + rb['id'] + ': ' + failure)
        for obstacle in scenario['obstacles']:
            check(obstacle['room'] in rooms and obstacle['kind'] in props['landmarks'] and obstacle['kind'] in fire['furniture'], label + ' invalid obstacle')
            if obstacle['room'] not in ORIGINAL_ROOMS:
                rect = rooms[obstacle['room']]['rect']
                box = footprint(obstacle, fire)
                distances = wall_clearances(box, rect)
                check(min(distances.values()) >= 0.075 - EPS, 'New obstacle outside clear wall faces')
                free_width = max(box[1] - (rect[1] + 0.075), rect[3] - 0.075 - box[3])
                check(free_width >= 0.9, obstacle['room'] + ' obstacle blocks corridor')
                print(f"PASS {obstacle['room']} obstacle leaves {free_width:.3f} m clear on one side using declared footprint")
    haze, clear = scenarios['haze'], scenarios['clear']
    check(haze['duration_s'] == 300 and haze['survivor_placement'] == 'random' and len(haze['survivors']) == 6 and all(r['control'] == 'autonomous' for r in haze['robots']), 'Haze scenario requirements not met')
    check(len(haze['obstacles']) == 7, 'Expected three added obstacles')
    check(digest(clear['survivors']) == REFERENCE['survivors'], 'Clear baseline fixed survivors changed')
    check(digest(clear['robots'][0]) == REFERENCE['clear_robot_A'], 'Clear baseline robot A changed')
    old_b = json.loads(json.dumps(clear['robots'][1]))
    old_b['spawn'] = dict(at=[18.0, 15.2], yaw=180)
    old_b['route'] = old_b['route'][2:]
    check(digest(old_b) == REFERENCE['clear_robot_B'], 'Clear baseline robot B original route changed')
    check(all(inside(q, [0, 0, 20, 14]) for rb in clear['robots'] for q in rb['route']), 'Clear baseline route leaves original house')
    check(clear['duration_s'] == 160 and clear['survivor_placement'] == 'fixed', 'Clear baseline needs 160 s for west ingress and fixed placement')
    for ignition in fire['ignitions']:
        check(ignition['room'] in rooms and inside(ignition['at'], rooms[ignition['room']]['rect']), 'Ignition outside room')

    for room in plan['rooms'][10:]:
        doors = [d for d in plan['doors'] if room['id'] in d['between']]
        objects = [o for o in plan['landmarks'] + haze['obstacles'] if o['room'] == room['id'] and o['kind'] in fire['furniture']]
        for failure in room_paths(room, doors, objects, fire):
            check(False, room['id'] + ': ' + failure)
        check_count += 1

    windows = window_spans(plan)
    path = draw_plan(plan, fire, scenarios, windows)
    status = 'PASS' if not errors else 'CHECKED'
    print(f'{status} {len(rooms)} rooms, {area:g} m2, four 280 m2 blocks; no overlaps; all room edges on {grid:g} m fire grid')
    print(f'{status} {len(plan["doors"])} door spans and connections; {len(exterior)} entries on four sides each reach all {len(rooms)} rooms without going outside')
    print(f'{status} {len(plan["landmarks"])} landmark anchors, new footprint clearance, prop/fire kinds and surface mappings; {asset_count} referenced model parts exist')
    print(f'{status} {len(plan["emergency_lights"])} emergency lights; every new room has a light and distinct floor/wall pair')
    print(f'{status} 0.9 m floor paths in all 20 new rooms with scenario debris, using declared footprints and wall thickness')
    print(f'{status} spawn position/yaw, survivor references and route wall crossings in both merged scenarios; {len(windows)} exterior windows predicted from WindowSpans')
    print(f'{status} original rooms, landmarks, fires, lights, door geometry, props and fire settings match frozen pre-extension hashes')
    print(f'CHECKS {check_count}; ERRORS {len(errors)}; PRESERVED EXCEPTIONS {len(exceptions)}')
    for message in exceptions:
        print('PRESERVED EXCEPTION ' + message)
    for message in errors:
        print('FAIL ' + message)
    print('ROOMS ' + '; '.join(f'{r["id"]} {(r["rect"][2]-r["rect"][0]):g} x {(r["rect"][3]-r["rect"][1]):g} m, {r["name"]}' for r in plan['rooms'][10:]))
    for d in plan['doors']:
        if d['id'] in ('d_back', 'd_house_apartment', 'd_apartment_workshop', 'd_office_workshop'):
            print(f'BLOCK DOOR {d["id"]} {d["between"]} at {d["at"]}, width {d["width"]:g} m, {d["state"]}')
    for d in exterior:
        print(f'ENTRY {d["id"]} at {d["at"]}, width {d["width"]:g} m')
    for label, scenario in scenarios.items():
        for rb in scenario['robots']:
            print(f'SPAWN {label}/{rb["id"]} {rb["spawn"]}')
    for kind in REFERENCE['new_kinds']:
        sources = []
        for part in props['landmarks'].get(kind, []):
            sources.append((part.get('downloaded') or 'https://polyhaven.com/a/' + part['polyhaven']) + '; ' + part.get('credit', 'Poly Haven CC0') + f'; fit_width {part["fit_width"]:g} m')
        print('NEW KIND ' + kind + ': ' + ' | '.join(sources))
    for name in REFERENCE['new_materials']:
        print('NEW MATERIAL ' + name + ' ' + json.dumps(materials['materials'].get(name, {})))
    print('EVIDENCE ' + str(path))
    print('LIMITS JSON geometry and estimated footprints only. Unity must verify imported bounds, colliders, curved route motion, random survivor body clearance, fire timing and performance.')
    print('UNITY HouseBuilderTests hardcodes 280 m2; change its area assertion to the footprint area of 1120 m2. North/east original windows become internal walls and their thickness changes with the footprint. The config cannot freeze those generated windows or wall thicknesses.')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(validate())
