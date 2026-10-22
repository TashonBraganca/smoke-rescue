#!/usr/bin/env python3
"""Check observed trigger responses. Missing evidence is FAIL, never an inferred success."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def temperatures(folder):
    path = folder / 'air_temp.txt'
    return [tuple(map(float, line.split()[:2])) for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('#')] if path.exists() else []


def shared_gate_from_scenario(recording):
    """Evaluator setup only: determine whether both estimated target positions use the launch-pad frame."""
    meta=Path(recording)/'meta.json'
    if not meta.is_file():return False
    name=Path(json.loads(meta.read_text()).get('scenario','')).name
    folder=Path(__file__).resolve().parents[1]/'config'
    robots=None; seen=set()
    while name and name not in seen:
        seen.add(name); path=folder/name
        if not path.is_file():return False
        scenario=json.loads(path.read_text())
        if robots is None:robots=scenario.get('robots')
        name=scenario.get('inherits')
    return bool(robots and all(len(r.get('launch_pad') or [])==3 for r in robots))


def check(recording, fixes, config=None, shared_gate=None):
    config = config or json.loads((Path(__file__).resolve().parents[1] / 'config/brain.json').read_text())
    folders = [p for p in sorted(Path(recording).glob('robot_*')) if p.is_dir()]
    logs = {p.name: jsonl(p / 'decisions.jsonl') for p in folders}
    air = {p.name: temperatures(p) for p in folders}
    states = {p.name: jsonl(p / 'brain_state.jsonl') for p in folders}
    water = jsonl(Path(recording) / 'water.jsonl')
    trace = jsonl(Path(recording) / 'agent_trace.jsonl')

    def events(kind, text=''):
        return [(robot, e) for robot, rows in logs.items() for e in rows
                if e.get('kind') == kind and text.lower() in e.get('reason', '').lower()]

    def warm(robot, t):
        return any(abs(at-t) <= config['frame_max_age_s'] and c >= config['air_warm_c'] for at, c in air[robot])

    def moved(robot, t):
        samples = [s for s in states[robot] if t <= s.get('t', -1) <= t+config['stuck_window_s']*4]
        positions = [[s['x'], s['y']] for s in samples if 'x' in s and 'y' in s]
        positions = [p for p in positions if isinstance(p, list) and len(p) >= 2]
        return bool(positions and any((p[0]-positions[0][0])**2+(p[1]-positions[0][1])**2 > config['cell_m']**2 for p in positions[1:]))

    retreat = [(r,e) for r,e in events('retreat')+events('emergency_exit') if warm(r,e['t']) and moved(r,e['t'])]
    reentries=[]
    for r,e in events('reenter'):
        exits=[x for x in logs[r] if x['t'] < e['t'] and x.get('kind')=='mission_complete'
               and ('heat' in x.get('reason','').lower() or 'air' in x.get('reason','').lower())]
        if exits and e['t']-exits[-1]['t'] >= config['reenter_after_heat_s']:
            reentries.append((r,e))
    cut=[(r,e) for r,e in events('escape_closing') if e.get('numbers',{}).get('route_points')==0]
    fire_cut=[(r,e) for r,e in cut if any(x.get('kind')=='step_start' and x.get('robot')==r and
              e['t']<=x.get('t',-1)<=e['t']+config['frame_max_age_s'] and
              ((x.get('context') or {}).get('event') or {}).get('kind')=='escape_closing' and
              (x.get('context') or {}).get('fire_blocks_exit') for x in trace)]
    deadline=[(r,e) for r,e in fire_cut if any(x['t']<=e['t'] and x.get('kind')=='goal' and 'deadline' in x.get('reason','') for x in logs[r])]
    hot_exit=[(r,e) for r,e in events('emergency_exit') if warm(r,e['t']) and moved(r,e['t'])]
    hot_cut=[(r,e) for r,e in events('stay_overridden')+events('replan','coolest reachable floor')
             if warm(r,e['t']) and moved(r,e['t']) and any(rr==r and ee['t']<=e['t'] for rr,ee in cut)]
    spray_orders=events('agent_order','spray_water at')
    # Arm reach precedes water flow. Match a pulse to the last successful spray order, rather than a camera-age window.
    sprays=[(r,e) for r,e in spray_orders if any(w.get('robot')==r and w['t']>=e['t'] and
            w.get('flow_lps',0)>0 and w.get('seconds',0)>0 and
            not any(rr==r and e['t']<other['t']<=w['t'] for rr,other in spray_orders) for w in water)]
    meetings=[(r,e) for r,e in events('rendezvous') if e.get('numbers',{}).get('out_of_contact_s',0)>=config['rendezvous_after_s'] and e.get('numbers',{}).get('unshared_cells',0)>=config['rendezvous_min_unshared_cells']]
    if shared_gate is None:shared_gate=shared_gate_from_scenario(recording)
    nozzle=json.loads((Path(__file__).resolve().parents[1]/'config/actuators.json').read_text())['nozzle']
    starts=[e for e in trace if e.get('kind')=='step_start' and e.get('robot')!='commander' and 't' in e]
    ends={e.get('step'):e for e in trace if e.get('kind')=='step_end'}
    def spray_target(pulse):
        recent=[e for e in starts if e.get('robot')==pulse.get('robot') and e['t']<=pulse['t']]
        if not recent:return None
        step=max(recent,key=lambda e:e['t']); context=step.get('context') or {}
        option=context.get('spray_option') or {}; pose=context.get('pose') or {}
        if 'two robots spraying together' not in option.get('assessment',''):return None
        if any(key not in pose for key in ('x','y','heading_deg')) or option.get('distance_m') is None:return None
        bearing=float(option['bearing_deg']); distance=float(option['distance_m'])
        key=step.get('step',f"{step['robot']}@{float(step['t']):.3f}")
        aimed=any(o.get('type')=='spray_water' and abs((float(o.get('bearing_deg',1e9))-bearing+180)%360-180)<=nozzle['aim_tolerance_deg']
                  for o in (ends.get(key) or {}).get('orders',[]))
        if not aimed or not 0<distance<=nozzle['range_m']:return None
        heading=math.radians(float(pose['heading_deg'])+bearing)
        return (float(pose['x'])+math.sin(heading)*distance,float(pose['y'])+math.cos(heading)*distance)
    together=False
    if shared_gate:
        for a in water:
            ta=spray_target(a)
            if ta is None:continue
            for b in water:
                tb=spray_target(b)
                if tb is None or a.get('robot')==b.get('robot'):continue
                if (a.get('flow_lps',0)>0 and b.get('flow_lps',0)>0 and
                    max(a['t'],b['t'])<min(a['t']+a['seconds'],b['t']+b['seconds']) and
                    math.hypot(ta[0]-tb[0],ta[1]-tb[1])<=config['burning_radius_m']):together=True
    evidence={'12':retreat,'14':reentries,'36':fire_cut,'43':deadline,'55':hot_exit,'66':hot_cut,
              '19':sprays,'23':sprays,'79':sprays,'57':meetings,'59':together}
    descriptions={'12':'heated retreat followed by measured motion','14':'heat exit followed by re-entry after the full cool wait',
                  '36':'way out lost with fire proven to cut an otherwise geometric exit route','43':'deadline return followed by lost way out','55':'hot emergency exit followed by motion',
                  '66':'cut off and hot, safety response followed by motion','19':'evidenced spray reached the world','23':'evidenced spray reached the world',
                  '79':'evidenced spray reached the world','57':'45 s radio loss with enough unshared map','59':'both agents assess and aim at the same gate-frame fire patch, with overlapping water flow'}
    return {str(f):{'pass':bool(evidence.get(str(f))), 'criterion':descriptions.get(str(f),'unknown fix'),
                    'evidence':[(r,round(e['t'],3),e['kind']) for r,e in evidence[str(f)]] if isinstance(evidence.get(str(f)),list) else []} for f in fixes}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording',type=Path)
    parser.add_argument('--scenario',type=Path)
    parser.add_argument('--fixes',nargs='+')
    args=parser.parse_args()
    if not args.recording.is_dir():
        parser.error(f'recording is absent: {args.recording}')
    scenario=args.scenario
    if scenario is None and (args.recording/'meta.json').is_file():
        name=Path(json.loads((args.recording/'meta.json').read_text())['scenario']).name
        scenario=Path(__file__).resolve().parents[1]/'config'/name
    fixes=args.fixes or (json.loads(scenario.read_text()).get('target_fixes',[]) if scenario else [])
    if not fixes:
        parser.error('provide --fixes or a scenario with target_fixes')
    results=check(args.recording,fixes)
    for fix,result in results.items():
        print(f"#{fix} {'PASS' if result['pass'] else 'FAIL'} {result['criterion']}; evidence={result['evidence']}")
    return 0 if all(r['pass'] for r in results.values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
