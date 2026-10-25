"""Attach validator geometry to the existing gallery without disturbing running precomputations."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/runs/ryor'


def enrich(gallery):
    for house in gallery['houses']:
        world = json.loads((DATA / 'worlds' / (house['id'] + '__r2.json')).read_text())
        plan = json.loads((ROOT / 'config' / world['fire']['floorplan']).resolve().read_text())
        fire = json.loads((ROOT / 'config' / world['fire']['config']).resolve().read_text())
        angle = math.radians(world['gate']['yaw_deg'])
        ca, sa = math.cos(angle), math.sin(angle)
        gx, gy = world['gate']['origin']

        def gate(p):
            x, y = p[0] - gx, p[1] - gy
            return [round(x * ca - y * sa, 6), round(x * sa + y * ca, 6)]

        rooms = [dict(id=r['id'], name=r.get('name', r['id']), kind=w['kind'],
                      rect=[min(p[0] for p in w['poly']), min(p[1] for p in w['poly']),
                            max(p[0] for p in w['poly']), max(p[1] for p in w['poly'])])
                 for r in plan['rooms'] for w in world['rooms'] if r['id'] == w['id']]
        doors = []
        for d in plan['doors']:
            wall = d['wall'] if abs(sa) < .5 else ('z' if d['wall'] == 'x' else 'x')
            doors.append(dict(id=d['id'], at=gate(d['at']), width=d['width'], wall=wall, between=d['between']))
        furniture, footprints = [], {'furniture': {'person_body': {'size_m': [.9, .9]}}}
        for i, obj in enumerate(world['prisms']):
            xs, ys = [p[0] for p in obj['poly']], [p[1] for p in obj['poly']]
            kind = 'box_%d' % i
            footprints['furniture'][kind] = {'size_m': [max(xs) - min(xs), max(ys) - min(ys)]}
            at = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2]
            from validate_building import inside
            room = next((r for r in rooms if inside(at, r['rect'])), None)
            furniture.append(dict(kind=kind, at=at, yaw=0, room=room['id'] if room else 'outside'))
        combustible = [r['id'] for r in plan['rooms'] if any(
            fire['materials'][fire['surfaces'][r[s]]]['fuel_kg_m2'] > 0 for s in ('floor', 'ceiling'))]
        ignitions = [gate(i['at']) for i in fire['ignitions']]
        house['fire_origins'] = ignitions
        house['door'] = next(d['at'] for d in doors if 'outside' in d['between'])
        house['placement_geometry'] = dict(rooms=rooms, doors=doors, furniture=furniture,
                                            footprints=footprints, combustible_rooms=combustible)
    return gallery


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(ROOT / 'tools'))
    path = DATA / 'gallery.json'
    gallery = enrich(json.loads(path.read_text()))
    path.write_text(json.dumps(gallery, separators=(',', ':')) + '\n')
    print('Attached shared placement geometry to %d houses.' % len(gallery['houses']))
