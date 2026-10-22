"""Offline audit. Truth stays in evaluator code, outside the robot packages.

Run from brain with uv run python ../tools/eval/people_coverage.py --out DIR.
Inspection matching is reported at 1 m and 1.5 m; neither is a runtime prior.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'brain'))
from brain.agent import vision as V
from brain.agent import eval_vision as E
from brain.evaluator import load_survivors, load_floorplan, load_trajectory, gt_pose_at
from brain.nav_io import load_scenario, recording_frame, transform_xy, read_json_lines

def inspections(rec, out):
    trace = read_json_lines(rec / 'agent_trace.jsonl')
    starts = {r['step']: r for r in trace if r['kind'] == 'step_start'}
    survivors, plan = load_survivors(rec), load_floorplan(rec)
    scenario = load_scenario(json.loads((rec / 'meta.json').read_text()))
    trajectories = {r: load_trajectory(rec, r) for r in ('robot_A', 'robot_B')}
    transforms = {r: recording_frame(read_json_lines(rec / r / 'brain_state.jsonl'), scenario, r)[0] for r in trajectories}
    rows = []
    for call in trace:
        if call['kind'] != 'tool' or call['name'] != 'inspect_candidate':
            continue
        s = starts[call['step']]; robot, t = s['robot'], s['t']; a = call['arguments']
        context = s.get('context') or s.get('reports') or {}
        frame = V.load_frame(rec / robot, t)
        result = V.inspect_candidate(frame, **a, output_dir=out / 'crops')
        pose = context['pose']; bearing = math.radians(pose['heading_deg'] + a['bearing_deg'])
        xy = np.array([pose['x'], pose['y']]) + a['range_m'] * np.array([math.sin(bearing), math.cos(bearing)])
        xy = transform_xy(xy, transforms[robot])
        distances = [float(np.linalg.norm(xy - np.array(p['position_house'])[::2])) for p in survivors]
        p = survivors[int(np.argmin(distances))]; distance = min(distances)
        truth_pose = gt_pose_at(trajectories[robot], frame['stamps']['thermal'])
        vis = E.survivor_visibility(frame, truth_pose, p, plan)
        cand = result.get('candidate')
        projected = bool(cand and 'bbox' in cand and E.match_component(cand, vis))
        row = dict(run=rec.name, robot=robot, t=t, arguments=a, logged_verdict=call.get('verdict'),
                   nearest_person=p['id'], requested_error_m=distance, target_within_1m=distance<=1,
                   target_within_1_5m=distance<=1.5, component_matches_projected_body=projected,
                   visible=bool(vis and vis['visible']), event=s.get('event'), pose=pose,
                   result=result)
        rows.append(row)
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--recordings', type=Path, default=Path(__file__).resolve().parents[2]/'data_main/recordings')
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in range(922,927):
        rec=a.recordings/f'unity_house_onedoor_seed{seed}_run1'
        sub=inspections(rec,a.out); rows+=sub
        print(rec.name, len(sub),Counter(r['logged_verdict'] for r in sub), flush=True)
    (a.out/'inspections.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    print('all',len(rows), 'strict targets',sum(r['target_within_1m'] for r in rows),'loose targets',sum(r['target_within_1_5m'] for r in rows),flush=True)
if __name__=='__main__': main()
