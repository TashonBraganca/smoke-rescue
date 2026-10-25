"""Replay a recording's robot calls through the AWS forwarder and a local agent server, and compare the answers.

    brain/.venv/bin/python tools/replay_remote_forward.py data/recordings/unity_house_onedoor_seed934_full_BEST_0318 \
        --remote http://127.0.0.1:8095 --local http://127.0.0.1:8091 --out data/runs/remote_forward_25oct/replay.json

The calls are the ones Unity's recorder makes (RecordingSession.Agents.cs, VisualPlaceClient.cs), with the bodies it
sends: person scans with a 3D map integrate after each, a height layer every 5th scan, a place keyframe every 30 s,
partner place matches, agent steps at the times the recording logged agent orders, one commander step and a map
export. Times come from decisions.jsonl (candidate and glance events, agent orders) plus a regular 30 s grid; poses from
brain_state.jsonl. Both servers receive each call at the same moment; latencies are measured per target.
Never point --local at the server a live recording is using (8080): run a separate one (e.g. port 8091).
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import statistics
import time

import httpx

PEOPLE_GRID_S = 30.0
KEYFRAME_EVERY_S = 30.0
LAYERS_EVERY = 5
CELL_M, GRID = 0.1, 200


def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def stamps(robot_dir, sensor):
    out = []
    for line in (robot_dir / f'{sensor}.txt').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            out.append(float(line.split()[0]))
    return out


def snap(ts, t):
    return min(ts, key=lambda s: abs(s - t))


def pose_at(states, t):
    s = min(states, key=lambda r: abs(r['t'] - t))
    return s, {'x': s['x'], 'y': s['y'], 'heading_deg': -math.degrees(s['yaw_rad'])}


def plan(rec: Path, steps_per_robot: int):
    events = []
    for robot_dir in sorted(p for p in rec.iterdir() if p.is_dir() and p.name.startswith('robot_')):
        robot = robot_dir.name
        decisions = rows(robot_dir / 'decisions.jsonl')
        states = rows(robot_dir / 'brain_state.jsonl')
        rgb = stamps(robot_dir, 'rgb')
        thermal = stamps(robot_dir, 'thermal')
        end = min(rgb[-1], states[-1]['t'])
        seen = [d['t'] for d in decisions if d['kind'] in ('survivor_candidate', 'person_glance', 'survivor_view_ready')]
        grid = [5.0 + PEOPLE_GRID_S * k for k in range(int((end - 5) // PEOPLE_GRID_S) + 1)]
        scan_t = sorted({snap(rgb, t) for t in seen + grid if 0 < t <= end})
        last_kf = -1e9
        for i, t in enumerate(scan_t):
            s, pose = pose_at(states, t)
            if t - last_kf >= KEYFRAME_EVERY_S:
                last_kf = t
                events.append((t, robot, 'keyframe', {'robot_dir': str(robot_dir.resolve()), 't': t}))
            events.append((t, robot, 'people', {'robot_dir': str(robot_dir.resolve()), 't': t, 'min_conf': 0.25}))
            events.append((t, robot, 'integrate', {'robot_dir': str(robot_dir.resolve()), 't': t, 'pose': pose}))
            if i % LAYERS_EVERY == LAYERS_EVERY - 1:
                events.append((t, robot, 'layers', {'cell_m': CELL_M, 'origin': [pose['x'] - CELL_M * GRID / 2, pose['y'] - CELL_M * GRID / 2], 'size': GRID}))
        orders = sorted({round(d['t'], 6) for d in decisions if d['kind'] == 'agent_order' and d['t'] > 0})[:steps_per_robot]
        for t in orders:
            t = snap(thermal, t)
            s, pose = pose_at(states, t)
            hist = [x for x in thermal if x <= t][-45:]
            events.append((t, robot, 'step', {
                'robot_dir': str(robot_dir.resolve()), 't': t, 'mission_time_left_s': round(550 - t), 'pose': pose,
                'mode': s['mode'], 'state': s['state'], 'heat_budget': s['heat_budget'], 'torch': 'wide',
                'goal': s['goal'], 'event': {'kind': 'periodic', 'reason': 'Regular check-in.', 'numbers': {}},
                'frame_times': hist, 'inbox': [], 'survivors': s.get('survivors') or [], 'frontiers': [],
                'water_l': 12.0, 'carrying_person': False, 'partners': [{'robot': 'robot_B' if robot == 'robot_A' else 'robot_A', 'maps_linked': False, 'position': None}],
            }))
    events.sort(key=lambda e: (e[0], e[1], ['keyframe', 'people', 'integrate', 'layers', 'step'].index(e[2])))
    return events


def call(client, base, route, robot, body, trace_dir):
    path = {'people': '/perceive/people', 'integrate': f'/map3d/{robot}/integrate', 'layers': f'/map3d/{robot}/layers',
            'keyframe': f'/place/{robot}/keyframe', 'match': f'/place/{robot}/match', 'step': f'/robot/{robot}/step',
            'commander': '/commander/step', 'export': f'/map3d/{robot}/export'}[route]
    body = dict(body)
    if route in ('step', 'commander'):
        body['trace_path'] = str(trace_dir / 'agent_trace.jsonl')
        body['debug_dir'] = str(trace_dir / 'agent_debug')
    t0 = time.perf_counter()
    try:
        r = client.post(base + path, json=body, timeout=130)
        out = {'status': r.status_code, 's': round(time.perf_counter() - t0, 3), 'served_by': r.headers.get('x-smoke-served-by')}
        try:
            out['json'] = r.json()
        except ValueError:
            out['text'] = r.text[:300]
    except httpx.HTTPError as exc:
        out = {'status': 0, 's': round(time.perf_counter() - t0, 3), 'error': str(exc)}
    return out


def same_people(a, b):
    if a == b:
        return True, 'identical'
    if len(a) != len(b):
        return False, f'{len(a)} vs {len(b)} people'
    worst = 0.0
    for p, q in zip(a, b):
        for k in set(p) | set(q):
            if isinstance(p.get(k), (int, float)) and isinstance(q.get(k), (int, float)):
                worst = max(worst, abs(p[k] - q[k]))
            elif p.get(k) != q.get(k) and not isinstance(p.get(k), list):
                return False, f'field {k} differs'
    return worst < 0.02, f'same count, max numeric difference {worst:.4f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('recording', type=Path)
    ap.add_argument('--remote', default='http://127.0.0.1:8095')
    ap.add_argument('--local', default='http://127.0.0.1:8091')
    ap.add_argument('--steps', type=int, default=3)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--only', default='', help='comma list of routes to replay (default all)')
    a = ap.parse_args()
    if a.local.rstrip('/').endswith(':8080'):
        raise SystemExit('refusing to replay against :8080, the live recording server')
    events = plan(a.recording, a.steps)
    if a.only:
        keep = set(a.only.split(','))
        events = [e for e in events if e[2] in keep]
    trace = a.out.parent / 'local_trace'
    trace.mkdir(parents=True, exist_ok=True)
    rc, lc = httpx.Client(), httpx.Client()
    results, last_payload = [], {}
    pool = ThreadPoolExecutor(2)

    def both(route, robot, body, t):
        fr = pool.submit(call, rc, a.remote, route, robot, body, trace)
        fl = pool.submit(call, lc, a.local, route, robot, body, trace)
        res = {'t': t, 'robot': robot, 'route': route, 'remote': fr.result(), 'local': fl.result()}
        results.append(res)
        r, l = res['remote'], res['local']
        print(f"{t:8.3f} {robot} {route:9s} aws {r['status']} {r['s']:6.2f}s  local {l['status']} {l['s']:6.2f}s", flush=True)
        return res

    for t, robot, route, body in events:
        res = both(route, robot, body, t)
        if route == 'keyframe':
            for side in ('remote', 'local'):
                j = res[side].get('json') or {}
                if j.get('ok'):
                    last_payload.setdefault(side, {})[robot] = j['payload_b64']
            # The partner matches the newest keyframe against its own views, as VisualPlaceWorker does.
            partner = 'robot_B' if robot == 'robot_A' else 'robot_A'
            if all(robot in last_payload.get(s, {}) for s in ('remote', 'local')) and not a.only:
                pdir = a.recording.resolve() / partner
                fr = pool.submit(call, rc, a.remote, 'match', partner, {'robot_dir': str(pdir), 't': t, 'payload_b64': last_payload['remote'][robot]}, trace)
                fl = pool.submit(call, lc, a.local, 'match', partner, {'robot_dir': str(pdir), 't': t, 'payload_b64': last_payload['local'][robot]}, trace)
                results.append({'t': t, 'robot': partner, 'route': 'match', 'remote': fr.result(), 'local': fl.result()})
    if not a.only:
        last_t = max(e[0] for e in events)
        both('commander', None, {'t': last_t, 'reports': [], 'robots': ['robot_A', 'robot_B']}, last_t)
        for robot in ('robot_A', 'robot_B'):
            ply = a.out.parent / f'{robot}_remote_map3d.ply'
            res = {'t': last_t, 'robot': robot, 'route': 'export',
                   'remote': call(rc, a.remote, 'export', robot, {'path': str(a.recording.resolve() / robot / 'map3d.ply.remote_replay')}, trace)}
            results.append(res)

    # ----- comparison -----
    summary = {}
    for res in results:
        b = summary.setdefault(res['route'], {'n': 0, 'remote_ok': 0, 'local_ok': 0, 'equal': 0, 'diffs': [], 'remote_s': [], 'local_s': []})
        b['n'] += 1
        r, l = res['remote'], res.get('local', {})
        b['remote_ok'] += r.get('status') == 200
        b['local_ok'] += l.get('status') == 200
        b['remote_s'].append(r['s'])
        if 's' in l:
            b['local_s'].append(l['s'])
        rj, lj = r.get('json') or {}, l.get('json') or {}
        if res['route'] == 'people' and r.get('status') == 200 and l.get('status') == 200:
            ok, why = same_people(rj.get('people'), lj.get('people'))
            b['equal'] += ok
            if not ok or why != 'identical':
                b['diffs'].append({'t': res['t'], 'robot': res['robot'], 'why': why, 'aws': rj.get('people'), 'local': lj.get('people')})
        elif res['route'] == 'keyframe' and r.get('status') == 200 and l.get('status') == 200:
            eq = rj.get('payload_b64') == lj.get('payload_b64') and rj.get('t') == lj.get('t')
            b['equal'] += eq
            if not eq:
                b['diffs'].append({'t': res['t'], 'robot': res['robot'], 'aws_features': rj.get('features'), 'local_features': lj.get('features'), 'aws_t': rj.get('t'), 'local_t': lj.get('t')})
        elif res['route'] == 'match' and r.get('status') == 200 and l.get('status') == 200:
            keys = ('matched', 'mine_id', 'inliers', 'dx_m', 'dy_m', 'dyaw_deg', 'reason')
            eq = all(rj.get(k) == lj.get(k) for k in keys)
            b['equal'] += eq
            if not eq:
                b['diffs'].append({'t': res['t'], 'aws': {k: rj.get(k) for k in keys}, 'local': {k: lj.get(k) for k in keys}})
        elif res['route'] == 'layers' and r.get('status') == 200 and l.get('status') == 200:
            ra = base64.b64decode(rj['top_cm']); la = base64.b64decode(lj['top_cm'])
            import numpy as np
            x, y = np.frombuffer(ra, '<u2').astype(int), np.frombuffer(la, '<u2').astype(int)
            eq = ra == la and rj['seen'] == lj['seen']
            b['equal'] += eq
            if not eq:
                b['diffs'].append({'t': res['t'], 'robot': res['robot'], 'cells_differ': int((x != y).sum()), 'max_cm': int(abs(x - y).max())})
        elif res['route'] in ('step', 'commander') and r.get('status') == 200 and l.get('status') == 200:
            b['diffs'].append({'t': res['t'], 'robot': res['robot'],
                               'aws': {k: rj.get(k) for k in ('orders', 'evidence_ids', 'rounds', 'latency_s', 'summary')},
                               'local': {k: lj.get(k) for k in ('orders', 'evidence_ids', 'rounds', 'latency_s', 'summary')}})
        elif res['route'] == 'integrate' and r.get('status') == 200 and l.get('status') == 200:
            b['equal'] += rj.get('ok') == lj.get('ok') == True
    for b in summary.values():
        for side in ('remote_s', 'local_s'):
            s = b.pop(side)
            b[side.replace('_s', '_median_s')] = round(statistics.median(s), 3) if s else None
            b[side.replace('_s', '_p90_s')] = round(sorted(s)[int(0.9 * (len(s) - 1))], 3) if s else None
            b[side.replace('_s', '_max_s')] = round(max(s), 3) if s else None
    try:
        summary['forwarder_stats'] = rc.get(a.remote + '/forwarder/stats', timeout=10).json()
    except (httpx.HTTPError, ValueError):
        pass
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({'summary': summary, 'calls': results}, indent=1, default=str))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'diffs'} if isinstance(v, dict) and 'n' in v else v
                      for k, v in summary.items()}, indent=1))


if __name__ == '__main__':
    main()
