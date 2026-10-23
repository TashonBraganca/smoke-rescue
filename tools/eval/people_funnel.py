"""Evaluator-only person funnel and camera-tool replay. Never imported by robots.

From brain: uv run python ../tools/eval/people_funnel.py --out /tmp/funnel.
Only completed recordings are read. Counts are sampled frames, not all frames.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'brain'))
from brain.agent import vision as V
from brain.agent.tools import ToolRegistry, AgentContext
from brain.agent import eval_vision as E
from brain.evaluator import load_survivors, load_floorplan, load_trajectory, gt_pose_at
from brain.nav_io import load_scenario, recording_frame, transform_xy, read_json_lines

# Evaluation tolerance only, the existing survivor merge distance. Never a robot prior.
MATCH_M = 1.05


def nearest(xy, survivors):
    distances = [float(np.linalg.norm(np.asarray(xy) - np.array(s['position_house'])[::2])) for s in survivors]
    i = int(np.argmin(distances))
    return survivors[i]['id'], distances[i]


def point(pose, bearing, rng):
    angle = math.radians(pose['heading_deg'] + bearing)
    return [pose['x'] + rng * math.sin(angle), pose['y'] + rng * math.cos(angle)]


def thermal_visibility(frame, pose, survivor, plan):
    """Use the same visibility test in the calibrated thermal optical frame."""
    proxy = SimpleNamespace(K=frame.thermal_K, depth_m=V._depth(frame),
                            thermal_K=frame.thermal_K, rgb_to_thermal=np.eye(4))
    return E.survivor_visibility(proxy, pose @ np.linalg.inv(frame.rgb_to_thermal), survivor, plan)


def owner(result, visibility):
    candidate = result.get('candidate') or {}
    if 'bbox' not in candidate:
        return None, False
    matched = [sid for sid, vis in visibility.items() if E.match_component(candidate, vis)]
    seen = [sid for sid in matched if visibility[sid]['visible']]
    return (seen[0] if seen else None), bool(matched and not seen)


def rgb_owner(frame, pose, survivors, plan, result):
    """Semantic RGB match without using a possibly wrong measured depth as its label."""
    hits = result.get('cues', {}).get('rgb_dnn', {}).get('hits', [])
    for survivor in survivors:
        vis = E.survivor_visibility(frame, pose, survivor, plan)
        if not vis or not vis['visible']:
            continue
        pts = E._body_points(survivor['position_house'], survivor.get('pose'))
        cam = (pose[:3,:3].T @ (pts-pose[:3,3]).T).T
        cam = cam[cam[:,2] > 0]
        uv = (cam @ frame.K.T)
        uv = uv[:,:2]/uv[:,2,None]
        for hit in hits:
            if E.match_component({'bbox':hit['bbox_rgb'], 'range_m':None}, vis | {'uv_thermal':uv}):
                return survivor['id']
    return None


def completed_recordings(root, seeds):
    selected, skipped = [], []
    for seed in seeds:
        # The live recorder has not yet added the run1 suffix to 928.
        paths = [root / f'unity_house_onedoor_seed{seed}_run1', root / f'unity_house_onedoor_seed{seed}']
        rec = next((p for p in paths if (p / 'meta.json').exists()), None)
        if rec is None:
            skipped.append(f'{seed}: absent')
        elif json.loads((rec / 'meta.json').read_text()).get('complete') is not True:
            skipped.append(f'{seed}: incomplete, metadata only read')
        else:
            selected.append(rec)
    return selected, skipped


def departure(person, robot, t, states, decisions, transform):
    later = [s for s in states[robot] if s['t'] >= t]
    if not later:
        return {'reason': 'no later pose'}
    at = np.array(person['position_house'])[::2]
    initial = float(np.linalg.norm(transform_xy([later[0]['x'], later[0]['y']], transform) - at))
    left = next((s for s in later if np.linalg.norm(transform_xy([s['x'], s['y']], transform) - at) > initial + MATCH_M), None)
    stop = left['t'] if left else later[-1]['t']
    events = [e for e in decisions[robot] if t <= e['t'] <= stop]
    fire = next((e for e in events if e['kind'] in ('avoid_fire', 'retreat', 'escape_closing')), None)
    other = next((e for e in events if e['kind'] == 'goal' and 'person_x_m' in e.get('numbers', {}) and
                  np.linalg.norm(transform_xy([e['numbers']['person_x_m'], e['numbers']['person_y_m']], transform) - at) > MATCH_M), None)
    agent = next((e for e in events if e['kind'] == 'agent_order'), None)
    cause = fire or other or agent
    return {'left': left is not None, 't': left['t'] if left else None,
            'reason': ('fire' if fire else 'other candidate' if other else 'agent choice' if agent else 'other motion') if left else 'no departure before recording end',
            'evidence': cause, 'observed_until': stop}


def check_confirmation(rec, out, row, arguments=None):
    """Exercise the real action guard, separately from the historical agent's choices."""
    pose = row['pose']
    reg = ToolRegistry(rec/row['robot'], AgentContext(row['robot'],row['t'],pose['x'],pose['y'],pose['heading_deg']),out/'crops')
    result = dict(row['fused'],x=round(row['map_position'][0],2),y=round(row['map_position'][1],2),requested=row['arguments'])
    reg.evidence[result['id']] = result
    args = dict(arguments) if arguments else {'x':result['x'],'y':result['y']}
    args['evidence_id'] = result['id']
    return reg.call('confirm_survivor',args)


def inspection_errors(trace):
    """Keep failed calls visible without inventing a camera observation for them."""
    return [r for r in trace if r.get('kind') == 'tool' and r.get('name') == 'inspect_candidate' and r.get('error')]


def replay_inspections(rec, out, survivors, plan, transforms, trajectories, trace):
    starts = {r['step']: r for r in trace if r['kind'] == 'step_start'}
    histories = {r: V.InspectionHistory() for r in transforms}
    ToolRegistry._confirmed.clear(); ToolRegistry._pending.clear(); ToolRegistry._unfinished.clear()
    rows, evidence = [], {}
    for call in trace:
        if call.get('kind') != 'tool' or call.get('name') != 'inspect_candidate':
            continue
        if call.get('error'):
            continue
        start = starts[call['step']]
        robot, t = start['robot'], start['t']
        pose = start['context']['pose']
        args = call['arguments']
        frame = V.load_frame(rec / robot, t)
        raw = V.inspect_candidate(frame, **args, output_dir=out / 'crops')
        candidate = raw.get('candidate') or {}
        requested = point(pose, args['bearing_deg'], args['range_m'])
        measured = point(pose, candidate.get('bearing_deg', args['bearing_deg']), candidate.get('range_m') or args['range_m'])
        fused = histories[robot].observe(raw, position=measured, observer=(pose['x'], pose['y']))
        sid, error = nearest(transform_xy(requested, transforms[robot]), survivors)
        measured_sid, measured_error = nearest(transform_xy(measured, transforms[robot]), survivors)
        gt = gt_pose_at(trajectories[robot], frame['stamps']['rgb'])
        visibility = {s['id']: E.survivor_visibility(frame, gt, s, plan) for s in survivors}
        component_owner, ambiguous = owner(raw, visibility)
        # Spatially aimed calls and component matches are separate, inspectable labels.
        target = sid if error <= MATCH_M else None
        rgb_person = rgb_owner(frame, gt, survivors, plan, raw)
        label = component_owner or target or rgb_person
        row = {'run': rec.name, 'robot': robot, 't': t, 'step': call['step'], 'arguments': args,
               'requested_person': target, 'nearest_person': sid, 'requested_error_m': error,
               'measured_person': measured_sid if measured_error <= MATCH_M else None, 'measured_error_m': measured_error,
               'component_person': component_owner, 'rgb_person': rgb_person, 'ambiguous_component': ambiguous,
               'label_person': label, 'logged_verdict': call.get('verdict'), 'logged_reasons': call.get('reasons'),
               'raw': raw, 'fused': fused, 'pose': pose, 'map_position': measured,
               'following_tools': []}
        row['confirmation_at_returned_position'] = check_confirmation(rec,out,row)
        rows.append(row)
        evidence[(call['step'], call.get('result_id'))] = row
    confirmations = []
    for call in trace:
        if call.get('kind') != 'tool':
            continue
        if call['name'] in ('confirm_survivor', 'reject_candidate', 'approach_candidate'):
            inspection = evidence.get((call['step'], call['arguments'].get('evidence_id')))
            if inspection:
                inspection['following_tools'].append(call)
        if call['name'] == 'confirm_survivor':
            start = starts[call['step']]
            robot = start['robot']
            sid, error = nearest(transform_xy([call['arguments']['x'], call['arguments']['y']], transforms[robot]), survivors)
            confirmations.append({'robot': robot, 't': start['t'], 'person': sid if error <= MATCH_M else None,
                                  'error_m': error, 'call': call,
                                  'current_guard': check_confirmation(rec,out,evidence[(call['step'],call['arguments'].get('evidence_id'))],call['arguments'])
                                  if (call['step'],call['arguments'].get('evidence_id')) in evidence else {'refused':True,'reason':'inspection not found'}})
    return rows, confirmations


def nonperson_time(trace, inspections, decisions, states, transforms, survivors):
    """Attribute bounded command intervals, never pretend sensor calls log their CPU duration.

    A movement interval ends at the next accepted movement, heat interruption,
    reached/failed event, or recording end. Inspection wall latency is a step
    total shared by all its calls, so report it separately without summing it.
    """
    starts = {r['step']: r for r in trace if r['kind'] == 'step_start'}
    ends = {r['step']: r for r in trace if r['kind'] == 'step_end'}
    by_evidence = {(r['step'], r['raw']['id']): r for r in inspections}
    # Replay IDs differ with output_dir. Use logged IDs to resolve commands.
    for call in trace:
        if call.get('kind') == 'tool' and call.get('name') == 'inspect_candidate':
            match = next(r for r in inspections if r['step'] == call['step'] and r['arguments'] == call['arguments'])
            by_evidence[(call['step'], call.get('result_id'))] = match
    moves, last_inspection = [], {}
    movement = {'approach_candidate', 'turn_to_face', 'go_to', 'explore_frontier', 'stay', 'extract_survivor'}
    for call in trace:
        if call.get('kind') != 'tool' or call.get('step') not in starts:
            continue
        start = starts[call['step']]
        if start.get('robot') not in transforms:
            continue
        robot, t = start['robot'], start['t']
        if call['name'] == 'inspect_candidate':
            last_inspection[robot] = by_evidence.get((call['step'], call.get('result_id')))
        if call['name'] not in movement or call.get('refused') or call.get('error'):
            continue
        row = by_evidence.get((call['step'], call['arguments'].get('evidence_id')))
        if call['name'] == 'turn_to_face':
            row = last_inspection.get(robot)
            # A turn belongs to that candidate only when issued in the same decision step.
            if row and row['step'] != call['step']:
                row = None
        moves.append({'robot': robot, 't': t, 'tool': call['name'], 'inspection': row})
    intervals = []
    for i, move in enumerate(moves):
        row = move['inspection']
        if move['tool'] not in ('approach_candidate', 'turn_to_face') or row is None or row['label_person'] or row['ambiguous_component']:
            continue
        robot = move['robot']
        # The Python tool can queue a move which Unity refuses. Count only an
        # accepted movement order in the real brain log at the same step time.
        at = [e for e in decisions[robot] if abs(e['t']-move['t'])<=1e-6 and e['kind']=='agent_order' and
              (e['reason'].startswith('go_to:') or e['reason'].startswith('explore') or e['reason'].startswith('Refused go_to'))]
        if not at or not at[-1]['reason'].startswith('go_to:'):
            continue
        stop = states[robot][-1]['t']
        event = next((e for e in decisions[robot] if move['t'] < e['t'] <= stop and
                     (e['kind'] in {'agent_goal_reached','avoid_fire','retreat','mission_complete'} or
                      (e['kind']=='agent_order' and (e['reason'].startswith('go_to:') or e['reason'].startswith('explore'))) or
                      (e['kind']=='replan' and e['reason'].startswith('Agent goal has no known free path')))),None)
        if event:
            stop = event['t']
        poses = [s for s in states[robot] if move['t']<=s['t']<=stop]
        distance = sum(math.hypot(b['x']-a['x'],b['y']-a['y']) for a,b in zip(poses,poses[1:]))
        turning = sum(abs((b['yaw_rad']-a['yaw_rad']+math.pi)%(2*math.pi)-math.pi) for a,b in zip(poses,poses[1:]))
        intervals.append({'robot':robot,'start':move['t'],'end':stop,'seconds':max(0.,stop-move['t']),
                          'tool':move['tool'],'reasons':row['raw']['reasons'],'map_position':row['map_position'],
                          'observed_travel_m':distance,'observed_turn_deg':math.degrees(turning),
                          'end_event':event})
    # The reflex also approaches candidates before the model acts. Report these
    # state intervals separately because they can overlap agent command intervals.
    reflex = []
    for robot, samples in states.items():
        for a,b in zip(samples,samples[1:]):
            if a.get('state')!='survivor':
                continue
            selected = [e for e in decisions[robot] if e['t']<=a['t'] and e['kind']=='goal' and 'person_x_m' in e.get('numbers',{})]
            if not selected:
                continue
            e = selected[-1]; n = e['numbers']
            sid,error = nearest(transform_xy([n['person_x_m'],n['person_y_m']],transforms[robot]),survivors)
            if error>MATCH_M:
                reflex.append({'robot':robot,'start':a['t'],'end':b['t'],'seconds':b['t']-a['t'],
                               'candidate':e,'nearest_person_error_m':error})
    negatives = [r for r in inspections if not r['label_person'] and not r['ambiguous_component']]
    steps = {r['step'] for r in negatives}
    return {'inspections': len(negatives), 'verdicts': dict(Counter(r['fused']['verdict'] for r in negatives)),
            'steps_with_negative_inspection': len(steps),
            'step_wall_latency_s_shared_with_other_work': sum(ends.get(s, {}).get('latency_s', 0) for s in steps),
            'movement_seconds': dict_sum(intervals), 'intervals': intervals,
            'reflex_negative_focus_s':sum(r['seconds'] for r in reflex),'reflex_intervals':reflex,
            'reasons': dict(Counter(reason for r in negatives for reason in r['raw']['reasons']))}


def dict_sum(intervals):
    out = Counter()
    for i in intervals:
        out[i['tool']] += i['seconds']
    return dict(out)


def evaluate_run(rec, out, stride, sample_inspections=True):
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((rec / 'meta.json').read_text())
    if meta.get('complete') is not True:
        raise ValueError('refuse incomplete recording')
    survivors, plan = load_survivors(rec), load_floorplan(rec)
    scenario = load_scenario(meta)
    robots = meta['robots']
    states = {r: read_json_lines(rec / r / 'brain_state.jsonl') for r in robots}
    transforms = {r: recording_frame(states[r], scenario, r)[0] for r in robots}
    trajectories = {r: load_trajectory(rec, r) for r in robots}
    decisions = {r: read_json_lines(rec / r / 'decisions.jsonl') for r in robots}
    trace = read_json_lines(rec / 'agent_trace.jsonl')
    people = {s['id']: {'id': s['id'], 'room': next((room['id'] for room in plan['rooms'] if room['rect'][0] <= s['position_house'][0] <= room['rect'][2] and room['rect'][1] <= s['position_house'][2] <= room['rect'][3]), 'outside'),
                       'views': [], 'brain_candidates': [], 'inspections': [], 'confirm_attempts': [], 'confirmed': []} for s in survivors}
    for robot in robots:
        for event in decisions[robot]:
            n = event.get('numbers', {})
            if event['kind'] in ('survivor_candidate', 'survivor_confirmed') and 'x_m' in n:
                sid, error = nearest(transform_xy([n['x_m'], n['y_m']], transforms[robot]), survivors)
                if error <= MATCH_M:
                    people[sid]['brain_candidates' if event['kind'] == 'survivor_candidate' else 'confirmed'].append({'robot': robot, 'error_m': error, **event})
    eval_rows = []
    for robot in robots:
        stamps = np.atleast_1d(np.loadtxt(rec / robot / 'thermal.txt', usecols=0))
        for frame_index, t in enumerate(stamps[::stride]):
            if frame_index % 25 == 0:
                print(f'{rec.name} {robot}: frame {frame_index} t={t:.1f}', flush=True)
            frame = V.load_frame(rec / robot, float(t))
            gt = gt_pose_at(trajectories[robot], frame['stamps']['rgb'])
            # Register once for all survivors.
            proxy = SimpleNamespace(K=frame.thermal_K, depth_m=V._depth(frame), thermal_K=frame.thermal_K, rgb_to_thermal=np.eye(4))
            thermal_pose = gt @ np.linalg.inv(frame.rgb_to_thermal)
            for survivor in survivors:
                rgb = E.survivor_visibility(frame, gt, survivor, plan)
                thermal = E.survivor_visibility(proxy, thermal_pose, survivor, plan)
                seen = [name for name, v in [('rgb', rgb), ('thermal', thermal)] if v and v['visible']]
                if seen:
                    people[survivor['id']]['views'].append({'robot': robot, 't': float(t), 'sensors': seen,
                        'range_m': min(v['range_m'] for v in (rgb, thermal) if v and v['visible']),
                        'robot_room': next((room['id'] for room in plan['rooms'] if room['rect'][0] <= gt[0,3] <= room['rect'][2] and room['rect'][1] <= gt[2,3] <= room['rect'][3]), 'outside')})
            if sample_inspections:
                eval_rows += E.label_frame(frame, gt, survivors, plan, rec.name, robot, float(t), out / 'eval_crops')
        print(f'{rec.name} {robot}: sampled {len(stamps[::stride])} frames', flush=True)
    inspections, confirms = replay_inspections(rec, out, survivors, plan, transforms, trajectories, trace)
    for row in inspections:
        if row['label_person']:
            people[row['label_person']]['inspections'].append(row)
    for row in confirms:
        if row['person']:
            people[row['person']]['confirm_attempts'].append(row)
    for survivor in survivors:
        p = people[survivor['id']]
        p['views'].sort(key=lambda r:r['t'])
        p['brain_candidates'].sort(key=lambda r:r['t'])
        if p['inspections']:
            last = p['inspections'][-1]
            p['departure'] = departure(survivor, last['robot'], last['t'], states, decisions, transforms[last['robot']])
        else:
            p['departure'] = {'reason': 'no aimed inspection'}
    negatives = nonperson_time(trace, inspections, decisions, states, transforms, survivors)
    return {'run': rec.name, 'stride': stride, 'people': list(people.values()), 'inspections': inspections,
            'confirmations': confirms, 'eval_rows': eval_rows, 'nonperson_time': negatives,
            'failed_inspection_calls': inspection_errors(trace),
            'eval_scores': E.summarize(eval_rows, n_boot=0) if eval_rows else {}}


def replay_cached_run(cached, rec, out):
    """Reuse only visibility sampling, then replay every real inspection and action."""
    if json.loads((rec/'meta.json').read_text()).get('complete') is not True:
        raise ValueError('refuse incomplete recording')
    survivors,plan = load_survivors(rec),load_floorplan(rec)
    scenario = load_scenario(json.loads((rec/'meta.json').read_text()))
    robots = json.loads((rec/'meta.json').read_text())['robots']
    states = {r:read_json_lines(rec/r/'brain_state.jsonl') for r in robots}
    decisions = {r:read_json_lines(rec/r/'decisions.jsonl') for r in robots}
    transforms = {r:recording_frame(states[r],scenario,r)[0] for r in robots}
    trajectories = {r:load_trajectory(rec,r) for r in robots}
    trace = read_json_lines(rec/'agent_trace.jsonl')
    inspections,confirmations = replay_inspections(rec,out,survivors,plan,transforms,trajectories,trace)
    for p in cached['people']:
        p['inspections'] = [r for r in inspections if r['label_person']==p['id']]
        p['confirm_attempts'] = [r for r in confirmations if r['person']==p['id']]
    cached.update(inspections=inspections,confirmations=confirmations,
                  failed_inspection_calls=inspection_errors(trace),
                  nonperson_time=nonperson_time(trace,inspections,decisions,states,transforms,survivors))
    return cached


def repair_eval_selections(run,rec,out,*,recheck_all_person=False):
    """Re-evaluate frames whose returned body differed from the requested component.

    Earlier eval_vision versions charged those person verdicts to the requested
    warm patch. Keep the old and corrected rows as evidence of the attribution fix.
    """
    rows = run['eval_rows']
    keys = {(r['robot'],r['t']) for r in rows if
            (recheck_all_person and (r.get('inspect') or r.get('returned_person'))) or
            (not r['label'] and r.get('inspect') and not r.get('same_component') and 'returned_person' not in r)}
    survivors,plan = load_survivors(rec),load_floorplan(rec)
    replacements,repairs = {},[]
    for robot,t in sorted(keys):
        frame = V.load_frame(rec/robot,t)
        pose = gt_pose_at(load_trajectory(rec,robot),frame['stamps']['rgb'])
        fresh = E.label_frame(frame,pose,survivors,plan,rec.name,robot,t,out/'eval_crops')
        original = [r for r in rows if (r['robot'],r['t'])==(robot,t)]
        replacements[(robot,t)] = fresh
        repairs.append({'robot':robot,'t':t,'original':original,'corrected':fresh})
    kept = [r for r in rows if (r['robot'],r['t']) not in keys]
    run['eval_rows'] = sorted(kept+[r for rs in replacements.values() for r in rs],key=lambda r:(r['robot'],r['t'],not r['label']))
    run['eval_scores'] = E.summarize(run['eval_rows'],n_boot=0)
    run.setdefault('selection_repairs',[]).extend(repairs)
    return run


def audit_confirmation_rows(run,rec,out):
    """Check every sampled returned-person result against the real action guard.

    Raw unmatched camera verdicts and false accepted positions are different
    metrics. In particular, distant colour people can lie beyond the depth
    visibility labeller's range while the confirmation range still refuses them.
    """
    keys = {(r['robot'],r['t'],tuple(box)) for r in run['eval_rows']
            if r.get('inspect') or r.get('returned_person') for box in r.get('bboxes',[])}
    survivors,plan = load_survivors(rec),load_floorplan(rec)
    scenario = load_scenario(json.loads((rec/'meta.json').read_text()))
    audit = []
    for robot,t,box in sorted(keys):
        frame = V.load_frame(rec/robot,t)
        scan = V.scan_thermal(frame)
        comp = next(c for c in scan['people'] if tuple(c['bbox'])==box)
        args = {'bearing_deg':comp['bearing_deg'],'range_m':comp['range_m'] or 3.}
        result = V.inspect_candidate(frame,**args,output_dir=out/'guard_crops')
        if result['verdict']!='person':
            continue
        states = read_json_lines(rec/robot/'brain_state.jsonl')
        transform = recording_frame(states,scenario,robot)[0]
        times = np.array([s['t'] for s in states])
        pose = {'x':float(np.interp(t,times,[s['x'] for s in states])),
                'y':float(np.interp(t,times,[s['y'] for s in states])),
                'heading_deg':float(-np.degrees(np.interp(t,times,np.unwrap([s['yaw_rad'] for s in states]))))}
        c = result.get('candidate') or {}
        xy = point(pose,c.get('bearing_deg',args['bearing_deg']),c.get('range_m') or args['range_m'])
        sid,error = nearest(transform_xy(xy,transform),survivors)
        row = {'robot':robot,'t':t,'pose':pose,'arguments':args,'fused':result,'map_position':xy}
        guard = check_confirmation(rec,out,row)
        audit.append({'robot':robot,'t':t,'requested_bbox':list(box),'candidate':result.get('candidate'),
                      'verdict':result['verdict'],'nearest_person':sid,'position_error_m':error,
                      'guard':guard,'false_confirmation':bool(guard.get('ok') and error>MATCH_M)})
    run['sampled_confirmation_audit'] = audit
    return run


def render(runs, skipped):
    lines = ['Person funnel. Visibility counts are sampled robot frames, union of RGB and thermal, with per-sensor counts. Times are simulation seconds. Range is optical depth. Aimed calls use requested position within 1.05 m OR a component matched to the projected body; both labels remain in JSON. Brain candidates use the gate frame from nav_io.recording_frame. Historical logged verdicts and current replay verdicts differ because these runs span earlier code versions.',
             'run person room first_view frames rgb thermal closest_m closest_camera_room first_candidate inspections logged_person replay_person confirm_attempts refused confirmed departure']
    drops = Counter()
    for run in runs:
        for p in run['people']:
            views, ins = p['views'], p['inspections']
            first = lambda rows: f"{min(r['t'] for r in rows):.2f}" if rows else '-'
            closest = f"{min(v['range_m'] for v in views):.2f}" if views else '-'
            lines.append(f"{run['run']} {p['id']} {p['room']} {first(views)} {len(views)} {sum('rgb' in v['sensors'] for v in views)} {sum('thermal' in v['sensors'] for v in views)} {closest} {min(views,key=lambda v:v['range_m'])['robot_room'] if views else '-'} {first(p['brain_candidates'])} {len(ins)} {sum(r['logged_verdict']=='person' for r in ins)} {sum(r['fused']['verdict']=='person' for r in ins)} {len(p['confirm_attempts'])} {sum(r['call'].get('refused',False) for r in p['confirm_attempts'])} {len(p['confirmed'])} {p['departure']['reason']}")
            stage = ('confirmed' if p['confirmed'] else 'verdict_to_confirmation' if any(r['logged_verdict']=='person' for r in ins) else 'inspection_to_verdict' if ins else 'candidate_to_inspection' if p['brain_candidates'] else 'view_to_candidate' if views else 'never_in_sampled_view')
            drops[stage] += 1
            for r in ins:
                lines.append(f"  inspect {r['robot']} t={r['t']:.3f} bearing={r['arguments']['bearing_deg']:.2f} range={r['arguments']['range_m']:.2f} logged={r['logged_verdict']} replay={r['raw']['verdict']} fused={r['fused']['verdict']} requested_error={r['requested_error_m']:.3f} component_person={r['component_person']} reasons={json.dumps(r['raw']['reasons'])} cues={json.dumps(r['raw']['cues'],sort_keys=True)}")
            for c in p['confirm_attempts']:
                lines.append('  confirm '+json.dumps(c,sort_keys=True))
            lines.append('  departure '+json.dumps(p['departure'],sort_keys=True))
        n = run['nonperson_time']
        lines.append(f"{run['run']} non-person workload {json.dumps(n,sort_keys=True)}")
        lines.append(f"{run['run']} eval_vision sampled scores {json.dumps(run['eval_scores'],sort_keys=True)}")
    lines.append('Aggregate exclusive final drop-off counts '+json.dumps(drops,sort_keys=True))
    negative = [r for run in runs for r in run['inspections'] if not r['label_person'] and not r['ambiguous_component']]
    lines.append(f"Actual non-person inspections n={len(negative)} logged_person={sum(r['logged_verdict']=='person' for r in negative)} raw_person={sum(r['raw']['verdict']=='person' for r in negative)} fused_person={sum(r['fused']['verdict']=='person' for r in negative)}")
    lines.append('Skipped '+json.dumps(skipped))
    lines.append('Limits. Visibility is a geometric body-volume and depth test, not proof that RGB pixels contain a recognizable person. Stride can miss short views. Non-person inspection CPU time is not logged. Movement durations are command-attribution intervals, not measured actuator effort. A turn without evidence in the same step stays unattributed. Departure cause is a logged correlate, not an inferred intention. Recorded replay cannot prove a changed route or more people found in a new run.')
    return '\n'.join(lines)+'\n'


def cache_rgb_per_frame():
    """Reuse the deterministic colour inference while one frame supplies several crops."""
    original = V.detect_people_rgb
    cache = {}
    def detect(frame):
        key = frame['id']
        if key not in cache:
            cache.clear()
            cache[key] = original(frame)
        return cache[key]
    V.detect_people_rgb = detect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings', type=Path, default=ROOT/'data/recordings')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=30)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(922,929)))
    parser.add_argument('--no-eval-rows', action='store_true')
    parser.add_argument('--replay-from',type=Path,help='reuse cached visibility and eval rows, replay real inspections only')
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('stride must be positive')
    # Read-only asset fallback for worktrees which deliberately omit large data.
    if not V.PERSON_ONNX.exists() and (ROOT/'data/models/onnx/yolo11m.onnx').exists():
        V.PERSON_ONNX = ROOT/'data/models/onnx/yolo11m.onnx'
    args.out.mkdir(parents=True, exist_ok=True)
    recs, skipped = completed_recordings(args.recordings, args.seeds)
    cache_rgb_per_frame()
    runs = []
    for rec in recs:
        if args.replay_from:
            cached=json.loads((args.replay_from/f'{rec.name}.json').read_text())
            run=replay_cached_run(cached,rec,args.out/rec.name)
        else:
            run = evaluate_run(rec, args.out/rec.name, args.stride, not args.no_eval_rows)
        if run.get('eval_rows'):
            run=repair_eval_selections(run,rec,args.out/rec.name)
            run=audit_confirmation_rows(run,rec,args.out/rec.name)
        else:
            run['sampled_confirmation_audit']=[]
        (args.out/f'{rec.name}.json').write_text(json.dumps(run,allow_nan=False)+'\n')
        runs.append(run)
    (args.out/'funnel.txt').write_text(render(runs,skipped))
    print(render(runs,skipped).split('Aggregate exclusive')[1].split('Limits.')[0],flush=True)


if __name__ == '__main__':
    main()
