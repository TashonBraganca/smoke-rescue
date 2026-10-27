"""World-side placement checks. The API and engine import this same module.

Door clearance and body/path checks call the existing ranch validator. Nothing
from this module or the submitted scene is sent to a robot or a language model.
"""
import hashlib
import json
import math

from validate_building import inside, footprint
from validate_ranches import room_paths

# These are the generator's wall, furniture and door margins in validate_ranches.py.
WALL_M, FURNITURE_M, DOOR_M = .6, .9, 1.2
PERSON_SEPARATION_M = 1.2  # Two conservative 0.9 m body envelopes need distinct floor.
MAX_PEOPLE = 5  # The judge's requested maximum; zero people is a useful empty-house test.
SNAP_M = .25  # Match the fire solver grid and canonicalise pointer/keyboard coordinates.
BODY_M = .9  # validate_ranches.py uses this conservative body envelope for path checks.


def point(value):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError('Choose a point on the floor plan.')
    if any(type(n) not in (int, float) or not math.isfinite(n) or abs(n) > 100 for n in value):
        raise ValueError('Placement coordinates must be finite points on the floor plan.')
    return [round(round(n / SNAP_M) * SNAP_M, 2) for n in value]


def validate(house, submitted):
    if not isinstance(submitted, dict) or set(submitted) != {'people', 'fire'}:
        raise ValueError('Place people and one fire origin, or choose Random.')
    people = submitted['people']
    if not isinstance(people, list) or len(people) > MAX_PEOPLE:
        raise ValueError('Place up to five people.')
    result = {'people': [point(p) for p in people], 'fire': point(submitted['fire'])}
    geometry = house['placement_geometry']
    rooms, doors, furniture = geometry['rooms'], geometry['doors'], geometry['furniture']
    fire_cfg = geometry['footprints']
    bodies, fire_body = [], None
    for i, p in enumerate(result['people'] + [result['fire']]):
        is_fire = i == len(result['people'])
        label = 'The fire' if is_fire else 'Person %d' % (i + 1)
        room = next((r for r in rooms if inside(p, r['rect'], WALL_M)), None)
        if not room:
            raise ValueError(label + ' must be inside a room, at least 0.6 m from its walls.')
        if any(inside(p, [a - FURNITURE_M, b - FURNITURE_M, c + FURNITURE_M, d + FURNITURE_M])
               for a, b, c, d in (footprint(f, fire_cfg) for f in furniture)):
            raise ValueError(label + ' is too close to furniture. Choose clear floor.')
        if any(math.dist(p, d['at']) < DOOR_M for d in doors):
            raise ValueError(label + ' blocks a doorway. Keep 1.2 m clear.')
        if is_fire:
            if room['id'] not in geometry['combustible_rooms']:
                raise ValueError('The fire needs combustible floor. Choose a bedroom or living room.')
            if any(math.dist(p, q) < PERSON_SEPARATION_M for q in result['people']):
                raise ValueError('Keep the fire at least 1.2 m from each person.')
            # Reserve the same conservative envelope around the ignition. Distance from a door alone
            # does not prove that the room's only 0.9 m circulation path remains clear.
            fire_body = {'kind': 'person_body', 'at': p, 'yaw': 0, 'room': room['id']}
        else:
            if any(math.dist(p, q) < PERSON_SEPARATION_M for q in result['people'][:i]):
                raise ValueError('People must be at least 1.2 m apart.')
            bodies.append({'kind': 'person_body', 'at': p, 'yaw': 0, 'room': room['id']})
    for room in rooms:
        targets = [p for p in bodies if p['room'] == room['id']]
        hazards = [fire_body] if fire_body['room'] == room['id'] else []
        if not targets and not hazards:
            continue
        obstacles = [f for f in furniture if f['room'] == room['id']]
        issues = room_paths(room, [d for d in doors if room['id'] in d['between']],
                            obstacles + targets + hazards, fire_cfg, targets=targets, require_all=True)
        if issues:
            raise ValueError('A person or the fire blocks the 0.9 m path through ' + room.get('name', room['id']) + '.')
    return result


def setup_key(base, placement, engine):
    # Include actual scene bytes, engine and validator revision. Changing any invalidates the recording.
    data = {'base': base, 'placement': placement, 'engine': engine, 'validator': 2}
    digest = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:20]
    return base + '__p' + digest
