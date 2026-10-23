"""Offline room/time audit. Truth enters only through the evaluator readers.

Run with the project's Python from the repository root. No recording is changed.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'brain'))
from brain.nav_eval import body_trajectory
from brain.nav_io import (read_json_lines, load_scenario, recording_frame, transform_xy,
                          map_geometry, sample_grid, searched_for, interpolate_poses)
from brain.oracle import load_floorplan


class Activity:
    """Replay logged control transitions. Cause remains an inference, not command telemetry."""

    def __init__(self, face_budget_s, probe_arrival_m):
        self.face_budget_s = face_budget_s
        self.probe_arrival_m = probe_arrival_m
        self.sweep = self.scan = self.viewing = self.route_look = False
        self.probe_target = None
        self.face_until = -1
        self.order_until = -1
        self.order = None

    def apply(self, event):
        t, kind, reason = event['t'], event['kind'], event['reason']
        numbers = event['numbers']
        if kind == 'sweep_started':
            self.sweep = True
        elif kind == 'area_searched':
            self.sweep = False
        elif kind in ('doorway_probe_started', 'local_evidence_started'):
            self.probe_target = np.array([numbers['x_m'], numbers['y_m']])
        elif kind == 'doorway_probe_done':
            self.probe_target = None
        elif kind == 'route_watch':
            self.route_look = True
        elif kind == 'route_watch_done':
            self.route_look = False
        elif kind == 'agent_order' and reason.startswith('go_to:'):
            self.face_until = -1
            self.order_until = t + numbers['hold_s']
            self.order = 'candidate' if 'closer look at an uncertain warm region' in reason else 'move'
        elif kind == 'agent_goal_reached':
            self.face_until = self.order_until = -1
        elif kind == 'goal':
            self.viewing = 'At survivor viewing distance' in reason
            if 'turn to face the point' in reason:
                self.face_until = t + self.face_budget_s
            if 'Periodic' not in reason:
                self.scan = 'No reachable frontier is visible' in reason
        if kind in ('retreat', 'mission_complete', 'stuck'):
            self.sweep = False
            self.face_until = self.order_until = -1

    def cause(self, t, state, at, has_goal):
        # Follow RobotBrain.Step's priority. A probe still travelling to its standpoint
        # is a heading change, not yet a camera look.
        if state in ('recover', 'retreat', 'return', 'complete'):
            return 'other_safety_or_recovery'
        if self.route_look:
            return 'scan_sweep_evidence'
        if state == 'survivor':
            return 'survivor_facing' if self.viewing else 'turn_then_drive'
        if self.probe_target is not None:
            return ('scan_sweep_evidence' if np.linalg.norm(np.asarray(at) - self.probe_target)
                    <= self.probe_arrival_m else 'turn_then_drive')
        if self.sweep:
            return 'scan_sweep_evidence'
        if t <= self.face_until:
            return 'agent_face'
        if self.scan and not has_goal:
            return 'scan_sweep_evidence'
        return 'turn_then_drive' if has_goal else 'unresolved'


def audit(rec):
    meta = json.loads((rec / 'meta.json').read_text())
    if not meta.get('complete', False):
        raise ValueError(f'{rec}: recording is incomplete')
    rooms = load_floorplan(rec)['rooms']
    def room(xy):
        for r in rooms:
            a,b,c,d = r['rect']
            if a <= xy[0] < c and b <= xy[1] < d:
                return r['id']
        return 'outside'
    trace = read_json_lines(rec / 'agent_trace.jsonl')
    starts = {x['step']: x for x in trace if x['kind'] == 'step_start'}
    scenario = load_scenario(meta)
    config = json.loads((Path(__file__).resolve().parents[2] / 'config/brain.json').read_text())
    out = {}
    for robot in meta['robots']:
        rd = rec / robot
        states = read_json_lines(rd / 'brain_state.jsonl')
        events = read_json_lines(rd / 'decisions.jsonl')
        transform, frame = recording_frame(states, scenario, robot)
        truth_t, truth_pose = body_trajectory(rec, robot)
        ts = np.array([s["t"] for s in states])
        poses = interpolate_poses(truth_t, truth_pose, ts)
        # Body truth is used to prevent scan-matching pose jitter counting as walking.
        # Existing nav_eval uses the same camera-to-body conversion.
        dt = np.diff(ts); mid = (ts[:-1] + ts[1:]) / 2
        middle_poses = interpolate_poses(truth_t, truth_pose, mid)
        speed = np.linalg.norm(np.diff(poses[:,:2],axis=0),axis=1) / dt
        yaw = np.abs(np.degrees(np.diff(poses[:,2]))) / dt
        turning = (speed <= .05) & (yaw > 5)
        rows = defaultdict(Counter); totals = Counter({"turn_"+k+"_s":0 for k in ("agent_face","survivor_facing","scan_sweep_evidence","turn_then_drive","other_safety_or_recovery","unresolved")}); modes = Counter()
        ei = si = 0
        activity = Activity(config['sweep_max_s'], min(config['goal_tolerance_m'], config['cell_m'] / 2))
        for i,t in enumerate(mid):
            while ei < len(events) and events[ei]['t'] <= t:
                activity.apply(events[ei]); ei += 1
            while si+1<len(states) and states[si+1]['t']<=t: si+=1
            state=states[si]['state']; name=room(middle_poses[i,:2]); row=rows[name]
            row['dwell_s']+=dt[i]; totals['dwell_s']+=dt[i];modes[state]+=dt[i]
            if speed[i]>.05: row['walking_s']+=dt[i]
            if activity.sweep:row['sweep_s']+=dt[i]
            if activity.probe_target is not None:row['evidence_look_s']+=dt[i]
            if t <= activity.face_until:row['agent_face_s']+=dt[i]
            if state=='survivor' or (activity.order == 'candidate' and t <= activity.order_until):row['candidate_s']+=dt[i]
            if turning[i]:
                # Attribution follows the control priority in RobotBrain.Step. Events do not
                # log each command's owner; recovery and ambiguous intervals stay explicit.
                cause = activity.cause(t, state, [states[si]['x'], states[si]['y']], bool(states[si].get('goal')))
                row['turn_s']+=dt[i];row['turn_'+cause+'_s']+=dt[i];totals['turn_s']+=dt[i];totals['turn_'+cause+'_s']+=dt[i]
        tools=[]; opportunities=[]
        for x in trace:
            if x['kind']!='tool' or x['step'] not in starts:continue
            s=starts[x['step']]
            if s.get('robot')!=robot:continue
            t=s['t']; name=room(interpolate_poses(truth_t, truth_pose, np.array([t]))[0,:2])
            if x['name'] in ('turn_to_face','approach_candidate','explore_frontier','inspect_candidate'):
                rows[name][x['name']+'_calls']+=1
                if x.get('refused') or x.get('error'):rows[name][x['name']+'_refused']+=1
                tools.append(dict(t=t,name=x['name'],refused=bool(x.get('refused')),arguments=x['arguments']))
        for s in starts.values():
            if s.get('robot')!=robot:continue
            t=s['t']; fs=s.get('context',{}).get('frontiers',[])
            entered={room(p[:2]) for p in poses[ts<=t]}
            proven=[f for f in fs if f.get('proven_indoors')]
            new=[(f,room(transform_xy([f['x'],f['y']],transform))) for f in proven]
            new=[(f,n) for f,n in new if n not in entered and n!='outside']
            calls=[x for x in trace if x.get('step')==s['step'] and x['kind']=='tool']
            chosen=[x for x in calls if x['name']=='explore_frontier' and not x.get('refused') and not x.get('error')]
            if new:
                ids=[f['id'] for f,n in new]
                context = s.get('context', {})
                existing_goal = context.get('goal')
                goal_room = room(transform_xy(existing_goal, transform)) if existing_goal is not None else None
                goal_events = [e for e in events if e['kind'] == 'goal' and e['t'] <= t]
                opportunities.append(dict(t=t,unentered_options=[dict(id=f['id'],room=n,path_m=f['path_m']) for f,n in new],
                    first_proven=proven[0]['id'], first_proven_unentered=proven[0]['id'] in ids,
                    selected=[x['arguments'].get('frontier_id') for x in chosen],
                    selected_unentered=any(x['arguments'].get('frontier_id') in ids for x in chosen),
                    state=context.get('state'), existing_goal=existing_goal, existing_goal_room=goal_room,
                    existing_goal_unentered=goal_room is not None and goal_room != 'outside' and goal_room not in entered,
                    latest_goal_reason=goal_events[-1]['reason'] if goal_events else None,
                    brain_move_refusals=[e['reason'] for e in events if e['kind']=='agent_order'
                                        and e['reason'].startswith('Refused go_to:') and abs(e['t']-t)<1e-6],
                    actions=[x['name'] for x in calls if not x.get('refused') and not x.get('error')],
                    selected_reasons=[x['arguments'].get('reason') for x in chosen],
                    refusals=[dict(name=x['name'], reason=x.get('reason') or x.get('error'))
                              for x in calls if x.get('refused') or x.get('error')]))
        for e in events:
            if e['kind'] in ('sweep_started','local_evidence_started','doorway_probe_started'):
                at = interpolate_poses(truth_t, truth_pose, np.array([e['t']]))[0,:2]
                rows[room(at)][e['kind']+'_count']+=1
        options=[e for e in events if e['kind']=='frontier_options']
        # Snapshot evidence in each evaluator room, regardless of camera coverage. This
        # describes what was sensed; it is never fed back to the running robot.
        maps=[]
        for line in (rd/'brain_map.txt').read_text().splitlines():
            if line.startswith('#') or not line.strip():continue
            parts=line.split();t=float(parts[0]); image=cv2.imread(str(rd/parts[4]),0)
            if image is None:continue
            image=np.flipud(image); indoor=searched_for(rd,t,'indoor'); searched=searched_for(rd,t)
            if indoor is None or searched is None:continue
            origin,cell=map_geometry(image.shape,frame)
            yy,xx=np.nonzero((image==255)&(indoor>0));xy=transform_xy(origin+(np.c_[xx,yy]+.5)*cell,transform)
            counts=Counter(room(p) for p in xy);unseen=Counter(room(p) for p in xy[searched[yy,xx]!=255])
            yy_all,xx_all=np.nonzero(indoor>0)
            all_xy=transform_xy(origin+(np.c_[xx_all,yy_all]+.5)*cell,transform)
            indoor_counts=Counter(room(p) for p in all_xy)
            maps.append(dict(t=t,indoor_m2={k:v*cell*cell for k,v in counts.items()},
                             all_indoor_m2={k:v*cell*cell for k,v in indoor_counts.items()},
                             unsearched_indoor_m2={k:v*cell*cell for k,v in unseen.items()}))
        for r in rooms:
            rows[r["id"]]["dwell_s"] += 0
        out[robot]=dict(total=dict(totals),distance_m=float(np.linalg.norm(np.diff(poses[:,:2],axis=0),axis=1).sum()),
            state_s=dict(modes),rooms={k:dict(v) for k,v in rows.items()},tools=tools,opportunities=opportunities,
            interval_s=float(np.median(dt)), frame=frame, transform=transform.tolist(),
            options=dict(samples=len(options),with_proven=sum(e['numbers']['frontiers']>0 for e in options),
                with_claim_rejections=sum(e['numbers']['cells_claimed']>0 for e in options),
                with_visited_rejections=sum(e['numbers']['cells_visited']>0 for e in options),
                with_blacklist_rejections=sum(e['numbers']['cells_blacklisted']>0 for e in options)),maps=maps)
    return out


def render(result):
    lines = ["All times are simulation seconds. Turning means true body translation at most 0.05 m/s and absolute yaw above 5 degrees/s over each 0.5 s state interval. Dwell uses the interpolated true body midpoint. Motion causes follow logged state and event transitions, rather than per-command owner telemetry. Activity durations overlap. Tool counts include refused attempts. The room names and true body poses come only from evaluator readers."]
    def areas(values):
        return json.dumps({k: round(v, 2) for k, v in values.items()})
    for seed, robots in result.items():
        for robot, r in robots.items():
            total = r['total']
            lines.append(f"Run {seed} {robot} travelled {r['distance_m']:.2f} m over {total['dwell_s']:.1f} s and turned in place for {total['turn_s']:.1f} s. Its turn attribution was {total['turn_turn_then_drive_s']:.1f} s toward travel goals, {total['turn_scan_sweep_evidence_s']:.1f} s in scan, sweep or evidence looks, {total['turn_agent_face_s']:.1f} s facing under agent orders, {total['turn_survivor_facing_s']:.1f} s facing a survivor, {total['turn_other_safety_or_recovery_s']:.1f} s in safety or recovery states, and {total['turn_unresolved_s']:.1f} s unresolved. Its state durations in seconds were {areas(r['state_s'])}.")
            zero = []
            for name, row in r['rooms'].items():
                if not any(row.values()):
                    zero.append(name)
                    continue
                v = lambda key: row.get(key, 0)
                place = 'Outside evaluator rooms' if name == 'outside' else f'In evaluator room {name}'
                lines.append(f"{place}, {robot} dwelled for {v('dwell_s'):.1f} s and turned in place for {v('turn_s'):.1f} s. It had {v('walking_s'):.1f} s of translation, {v('agent_face_s'):.1f} s of logged agent facing, {v('sweep_s'):.1f} s of sweep activity, {v('evidence_look_s'):.1f} s of evidence approach/look activity, and {v('candidate_s'):.1f} s with an active survivor or candidate approach order. Its tool attempt counts were {v('turn_to_face_calls')} for turn_to_face, {v('explore_frontier_calls')} for explore_frontier, {v('inspect_candidate_calls')} for inspect_candidate and {v('approach_candidate_calls')} for approach_candidate. The corresponding refused/error counts were {v('turn_to_face_refused')}, {v('explore_frontier_refused')}, {v('inspect_candidate_refused')} and {v('approach_candidate_refused')}. The brain's start counts there were {v('sweep_started_count')} for sweeps and {v('doorway_probe_started_count') + v('local_evidence_started_count')} for evidence approaches.")
            lines.append('The evaluator rooms with zero dwell and no logged activity were '+', '.join(zero)+'.')
            op = r['opportunities']
            lines.append(f"On {len(op)} agent decision steps, at least one offered proven-indoor endpoint projected into a room this robot had not entered. The first offered proven endpoint belonged to that set on {sum(x['first_proven_unentered'] for x in op)} steps. The agent made a tool-accepted explore call toward that set on {sum(x['selected_unentered'] for x in op)} steps, chose another offered endpoint on {sum(bool(x['selected']) and not x['selected_unentered'] for x in op)} steps, and made no tool-accepted explore call on {sum(not x['selected'] for x in op)} steps. A tool-accepted call is a tool result without refusal or error, not proof that C# accepted the move or that the robot arrived.")
            no_explore = Counter()
            for x in op:
                if x['selected']:
                    continue
                if any('heat' in str(f['reason']) or 'reflex' in str(f['reason']) for f in x['refusals']): action='heat_refusal'
                elif 'approach_candidate' in x['actions']: action='candidate_approach'
                elif 'turn_to_face' in x['actions']: action='face_order'
                elif 'go_to' in x['actions']: action='other_goto'
                elif 'inspect_candidate' in x['actions']: action='inspect_without_move'
                elif 'avoid_area' in x['actions']: action='avoid_area'
                else: action='other'
                no_explore[action] += 1
            lines.append('Those no-explore steps fall into the following categories, in heat-refusal, candidate approach, face order, other go_to, inspection, avoidance and other priority: '+json.dumps(no_explore)+'. These categories describe the step, rather than proving why the brain stayed in a room.')
            lines.append(f"Of those no-explore steps, {sum(not x['selected'] and x['existing_goal'] is not None for x in op)} already had a brain goal, and {sum(not x['selected'] and x['existing_goal_unentered'] for x in op)} had a goal endpoint projected into an unentered room. No new explore call does not mean the brain stopped. On {sum(x['first_proven_unentered'] and bool(x['selected']) and not x['selected_unentered'] for x in op)} steps the first offered proven endpoint was in an unentered room but the agent chose an endpoint outside that proven set. C# logged a move refusal at the decision timestamp on {sum(bool(x['brain_move_refusals']) for x in op)} opportunity steps.")
            filters = r['options']
            lines.append(f"The brain logged {filters['samples']} option enumerations. Of these, {filters['with_proven']} had proven options, {filters['with_claim_rejections']} rejected at least one boundary cell under partner claims, {filters['with_visited_rejections']} rejected visited cells and {filters['with_blacklist_rejections']} rejected blacklisted cells. These are aggregate filter counters. They do not identify the evaluator rooms of the rejected cells.")
            for m in r['maps'][-1:]:
                lines.append(f"At {m['t']:.1f} s, recorded indoor-and-free floor area by evaluator room in square metres was {areas(m['indoor_m2'])}. Its camera-unsearched portion was {areas(m['unsearched_indoor_m2'])}. Counting every positive indoor pixel, including pixels whose recorded occupancy was not free, gave {areas(m['all_indoor_m2'])}.")
    return '\n'.join(lines) + '\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recordings',type=Path,default=Path('data/recordings'));p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[926,927,928]); a=p.parse_args()
    result={str(seed):audit(a.recordings/f'unity_house_onedoor_seed{seed}_run1') for seed in a.seeds}
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'measurements.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.out/'measurements.txt').write_text(render(result))
    print(render(result))
