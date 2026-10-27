"""Evaluator-only release gate. Geometry never enters the robot brain."""
from replay import inside


def gate(house, frames, status, engine_git, require_rooms=True):
    score = status['score']
    n = len(frames[-1]['robots'])
    entered = {room['id'] for room in house['rooms_poly']
               if any(inside(room['poly'], p['x'], p['y']) for f in frames for p in f['robots'])}
    missing = [r['id'] for r in house['rooms_poly'] if r['id'] not in entered]
    all_out = score.get('robots_out') == n and score.get('all_out') is True and all(p.get('out') for p in frames[-1]['robots'])
    engine_matches = status.get('engine', {}).get('git') == engine_git
    false = score.get('false_confirmations')
    result = dict(all_rooms_entered=not missing, missing_rooms=missing, rooms_entered=len(entered),
                  all_out=all_out, false_confirmations=false, engine_git=engine_git, engine_matches=engine_matches)
    # All five houses use the same room-entry requirement.
    result['passed'] = all_out and false == 0 and engine_matches and not missing
    return result
