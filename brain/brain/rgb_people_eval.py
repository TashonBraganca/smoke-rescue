"""Offline RGB person localization audit. Run from brain with python -m brain.rgb_people_eval.

Truth readers are confined to this evaluator. Detection and candidate association use
sensor outputs only. Range is optical z; localization uses the RGB box centre ray.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from brain.agent.vision import Frame, _index, detect_people_rgb, assess_visibility, PERSON_ONNX
from brain.agent.eval_vision import line_of_sight, _body_points
from brain.io import read_camera
from brain.oracle import load_survivors, load_floorplan, load_trajectory

ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = (.25, .35, .5, .6, .7)
BANDS = ('under_2m', '2_to_4m', '4_to_6m', 'over_6m', 'missing_depth')


def range_band(distance):
    if distance is None:
        return 'missing_depth'
    return 'under_2m' if distance < 2 else '2_to_4m' if distance < 4 else '4_to_6m' if distance <= 6 else 'over_6m'


def place_detection(detection, pose, k):
    """Back-project median optical z through the clipped box centre, then use true pose."""
    z = detection['range_m']
    if z is None or not np.isfinite(z) or z <= 0:
        return None
    x, y, w, h = detection['bbox_rgb']
    point = [z * math.tan(math.radians(detection['bearing_deg'])),
             z * (y + h / 2 - k[1, 2]) / k[1, 1], z, 1.]
    return (pose @ point)[:3].tolist()


def nearest_match(world, survivors, tolerance_m=1.):
    """Nearest survivor in building x,z. Height is not part of the candidate location."""
    if world is None or not survivors:
        return dict(survivor=None, nearest_survivor=None, error_m=None)
    distances = [float(np.linalg.norm(np.asarray(world)[[0, 2]] - np.asarray(s['position_house'])[[0, 2]])) for s in survivors]
    i = int(np.argmin(distances))
    return dict(survivor=survivors[i]['id'] if distances[i] <= tolerance_m else None,
                nearest_survivor=survivors[i]['id'], error_m=distances[i])


def visible_survivors(pose, k, width, survivors, plan):
    """Requested recall set: centre in horizontal FOV, planar distance <=6 m and floor-plan LOS.

    This deliberately does not require a vertical FOV or a depth/body-occlusion test.
    A person behind furniture can remain in this denominator. Labels describe
    geometric viewing opportunities, not human-reviewed RGB visibility.
    """
    out = []
    for s in survivors:
        position = np.asarray(s['position_house'])
        cam = pose[:3, :3].T @ (position - pose[:3, 3])
        distance = float(np.linalg.norm((position - pose[:3, 3])[[0, 2]]))
        u = k[0, 0] * cam[0] / cam[2] + k[0, 2] if cam[2] > 0 else -1
        if distance <= 6 and cam[2] > 0 and 0 <= u < width and line_of_sight(pose[[0, 2], 3], position[[0, 2]], plan):
            out.append(dict(survivor=s['id'], range_m=distance, optical_z_m=float(cam[2]), u=float(u)))
    return out


def metrics(frames, threshold, *, band=None, condition=None):
    tp = fp = hits = visible = 0
    errors, optical, radial = [], [], []
    for f in frames:
        if condition is not None and f['condition'] != condition:
            continue
        detections = [d for d in f['detections'] if d['confidence'] >= threshold]
        # Detection precision is localization precision. Duplicate detections count
        # independently; each visible survivor contributes at most one recall hit.
        selected = [d for d in detections if band is None or range_band(d['range_m']) == band]
        tp += sum(d['survivor'] is not None for d in selected)
        fp += sum(d['survivor'] is None for d in selected)
        for d in selected:
            if d['survivor'] is not None:
                for name, values in [('range_error_m', errors), ('optical_error_m', optical), ('radial_error_m', radial)]:
                    if d.get(name) is not None:
                        values.append(d[name])
        targets = [v for v in f['visible'] if band is None or range_band(v['range_m']) == band]
        visible += len(targets)
        hits += sum(any(d['survivor'] == v['survivor'] for d in detections) for v in targets)
    def error_summary(values):
        return dict(n=len(values), bias_m=float(np.mean(values)), mae_m=float(np.mean(np.abs(values))),
                    median_abs_m=float(np.median(np.abs(values))), p90_abs_m=float(np.percentile(np.abs(values), 90))) if values else dict(n=0)
    return dict(tp=tp, fp=fp, hits=hits, visible=visible, fn=visible-hits,
                precision=tp/(tp+fp) if tp+fp else None, recall=hits/visible if visible else None,
                range_error=error_summary(errors), optical_error=error_summary(optical), radial_error=error_summary(radial))


def episodes(frames):
    """Contiguous visible episodes per recording, robot and survivor; gaps >1.5 s split."""
    grouped = {}
    for f in sorted(frames, key=lambda f: (f['run'], f['robot'], f['t'])):
        for v in f['visible']:
            key = (f['run'], f['robot'], v['survivor'])
            groups = grouped.setdefault(key, [])
            if not groups or f['t'] - groups[-1]['end'] > 1.5:
                groups.append(dict(run=key[0], robot=key[1], survivor=key[2], start=f['t'], end=f['t']))
            else:
                groups[-1]['end'] = f['t']
    return [e for group in grouped.values() for e in group]


def one_hz(frames, phase_s=0.):
    selected = {}
    for f in sorted(frames, key=lambda f: (f['run'], f['robot'], f['t'])):
        second = int(math.floor(f['t'] - phase_s + 1e-5))
        if second >= 0:
            selected.setdefault((f['run'], f['robot'], second), f)
    return list(selected.values())


def replay_candidates(frames, threshold, repeats=2, window_s=3., phase_s=0.):
    """First saved frame per sim second, spatial votes <=1 m, no truth in association.

    A track expires after window_s without a sighting. It raises one event until
    expiry. Only separate sampled seconds vote. Tracks never cross robot or run.
    Event labels use the current detection's nearest-truth match. These are fixed
    trajectory results, without inspect_candidate or changed robot actions.
    """
    sampled = one_hz(frames, phase_s=phase_s)
    events, tracks, previous = [], [], None
    for f in sampled:
        key = (f['run'], f['robot'])
        if key != previous:
            tracks = []; previous = key
        tracks = [tr for tr in tracks if f['t'] - tr['last'] <= window_s + 1e-5]
        used = set()
        for d in sorted(f['detections'], key=lambda d: -d['confidence']):
            if d['confidence'] < threshold or d.get('world') is None:
                continue
            xy = np.asarray(d['world'])[[0, 2]]
            choices = [(float(np.linalg.norm(xy-tr['xy'])), i) for i, tr in enumerate(tracks)]
            dist, idx = min(choices, default=(float('inf'), -1))
            if dist <= 1. and idx in used:
                continue
            if dist > 1.:
                tracks.append(dict(xy=xy, times=[], last=f['t'], raised=False)); idx = len(tracks)-1
            tr = tracks[idx]; used.add(idx)
            tr['times'] = [t for t in tr['times'] if f['t'] - t <= window_s + 1e-5] + [f['t']]
            tr['last'] = f['t']; tr['xy'] = xy
            if len(tr['times']) >= repeats and not tr['raised']:
                tr['raised'] = True
                events.append(dict(run=f['run'], robot=f['robot'], t=f['t'], survivor=d['survivor'],
                                   world=d['world'], confidence=d['confidence']))
    eps = episodes(frames)
    hits = sum(any(e['run'] == ep['run'] and e['robot'] == ep['robot'] and e['survivor'] == ep['survivor']
                   and ep['start'] - 1e-5 <= e['t'] <= ep['end'] + 1e-5 for e in events) for ep in eps)
    tp = sum(e['survivor'] is not None for e in events)
    opportunities = {(ep['run'], ep['robot'], ep['survivor']) for ep in eps}
    found = opportunities & {(e['run'], e['robot'], e['survivor']) for e in events if e['survivor'] is not None}
    return dict(phase_s=phase_s, visible_survivor_robot_pairs=len(opportunities), found_survivor_robot_pairs=len(found),
                survivor_robot_recall=len(found)/len(opportunities) if opportunities else None, threshold=threshold, repeats=repeats, window_s=window_s, events=events, tp=tp, fp=len(events)-tp,
                precision=tp/len(events) if events else None, episodes=len(eps), episode_hits=hits,
                episode_recall=hits/len(eps) if eps else None, sampled_frames=len(sampled))


def phase_summary(replays):
    """Counts sum alternative sampling phases, never independent new recordings."""
    keys = ('tp', 'fp', 'episodes', 'episode_hits', 'visible_survivor_robot_pairs', 'found_survivor_robot_pairs')
    out = {key: sum(r.get(key, 0) for r in replays) for key in keys}
    n = len(replays)
    precision = [r['precision'] for r in replays if r['precision'] is not None]
    recalls = [r['episode_recall'] for r in replays if r['episode_recall'] is not None]
    out.update(phase_count=n, precision=out['tp']/(out['tp']+out['fp']) if out['tp']+out['fp'] else None,
               episode_recall=out['episode_hits']/out['episodes'] if out['episodes'] else None,
               worst_phase_precision=min(precision) if precision else None,
               min_phase_episode_recall=min(recalls) if recalls else None,
               max_phase_episode_recall=max(recalls) if recalls else None,
               tp_per_phase_mean=out['tp']/n if n else None, fp_per_phase_mean=out['fp']/n if n else None,
               episodes_per_phase=out['episodes']/n if n else None,
               survivor_robot_recall=out['found_survivor_robot_pairs']/out['visible_survivor_robot_pairs']
               if out['visible_survivor_robot_pairs'] else None)
    return out


def choose_operating_point(grid, target_precision=.95):
    """Require the precision target in every recording that raises candidates."""
    eligible = [r for r in grid if r['repeats'] >= 2 and r['precision'] is not None
                and r['precision'] >= target_precision
                and r['worst_recording_precision'] >= target_precision]
    meets_target = bool(eligible)
    if not eligible:
        eligible = [r for r in grid if r['repeats'] >= 2 and r['precision'] is not None]
    def rank(r):
        if meets_target:
            quality = (r['episode_recall'] or 0, r['precision'], r['worst_recording_precision'])
        else:
            quality = (r['worst_recording_precision'], r['precision'], r['episode_recall'] or 0)
        return (*quality, -r['repeats'], -r['window_s'], -r['threshold'])
    return max(eligible, key=rank, default=None), meets_target


def discover(recordings):
    eligible, skipped = [], []
    for rec in sorted(recordings.iterdir()):
        if not rec.is_dir():
            continue
        if not (rec/'meta.json').exists():
            skipped.append(dict(run=rec.name, reason='no meta.json')); continue
        survivors = load_survivors(rec)
        if survivors is None or not list(rec.glob('*/rgb/*.png')):
            skipped.append(dict(run=rec.name, reason='no survivor truth or RGB PNGs')); continue
        eligible.append(rec)
    return eligible, skipped


def load_rgb_depth(folder, t, rgb_path, camera, depth_index):
    stamp, depth_path = min(depth_index, key=lambda row: (abs(row[0]-t), row[0]))
    if abs(stamp-t) > .5:
        raise ValueError(f'{folder} no depth within .5 s at {t}')
    rgb = cv2.imread(str(rgb_path)); depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None or depth.dtype != np.uint16:
        raise ValueError(f'Unreadable RGB or uint16 depth at {folder} {t}')
    k = np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1.]])
    if rgb.shape[:2] != (camera.height, camera.width) or depth.shape != rgb.shape[:2]:
        raise ValueError('image size differs from camera.yaml')
    # RGB evaluation needs no thermal image. The placeholder is never read by the
    # detector or assess_visibility and permits RGB-D-only recordings.
    return Frame(t=t, rgb=cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), depth_m=depth.astype(np.float32)/camera.depth_factor,
                 thermal=np.zeros((1, 1), np.uint16), K=k, thermal_K=k,
                 rgb_to_thermal=np.eye(4), stamps=dict(rgb=t, depth=stamp, thermal=t), source=str(folder))


def input_signature(rec, robot):
    files = [rec/'meta.json', rec/robot/'rgb.txt', rec/robot/'depth.txt', rec/robot/'camera.yaml', PERSON_ONNX]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def read_checkpoint(path, repair=False):
    data = path.read_bytes() if path.exists() else b''
    if data and not data.endswith(b'\n'):
        if not repair:
            raise ValueError(f'Interrupted checkpoint tail {path}')
        data = data[:data.rfind(b'\n')+1]
        path.write_bytes(data)
    return [json.loads(line) for line in data.splitlines()]


def evaluate_robot(job, max_new_frames=None):
    rec, robot, out, threads = job
    cv2.setNumThreads(threads)
    import faulthandler
    faulthandler.dump_traceback_later(120, repeat=True)
    folder = rec/robot
    cache = out/'frames'/f'{rec.name}__{robot}.jsonl'
    cache.parent.mkdir(parents=True, exist_ok=True)
    signature = input_signature(rec, robot)
    sigfile = cache.with_suffix('.inputs.json')
    if cache.exists() and (not sigfile.exists() or json.loads(sigfile.read_text()) != signature):
        raise ValueError(f'Inputs changed for {cache}; select a fresh --out directory')
    sigfile.write_text(json.dumps(signature, indent=2))
    existing = {}
    if cache.exists():
        for row in read_checkpoint(cache, repair=True):
            existing[row['t']] = row
    survivors = load_survivors(rec)
    if isinstance(survivors, dict):
        survivors = survivors['survivors']
    plan = load_floorplan(rec)
    if plan is None:
        raise ValueError(f'{rec.name}: floorplan required for LOS')
    trajectory = load_trajectory(rec, robot)
    camera = read_camera(folder/'camera.yaml')
    indices = _index(folder, 'rgb'); depths = _index(folder, 'depth')
    if {p.resolve() for _, p in indices} != {p.resolve() for p in (folder/'rgb').glob('*.png')}:
        raise ValueError(f'RGB index does not enumerate every PNG in {folder}')
    max_pose_dt = 0.
    processed = 0
    with cache.open('a') as stream:
        for i, (t, path) in enumerate(indices):
            if t in existing:
                continue
            j = int(np.argmin(np.abs(trajectory.timestamps-t)))
            dt = float(abs(trajectory.timestamps[j]-t)); max_pose_dt = max(max_pose_dt, dt)
            if dt > .1:
                raise ValueError(f'{rec.name}/{robot}: true pose {dt}s away at {t}')
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat(trajectory.quaternions_xyzw[j]).as_matrix()
            pose[:3, 3] = trajectory.positions[j]
            frame = load_rgb_depth(folder, t, path, camera, depths)
            visible = visible_survivors(pose, frame.K, frame.rgb.shape[1], survivors, plan)
            detections = detect_people_rgb(frame, min_conf=.25)
            for d in detections:
                world = place_detection(d, pose, frame.K)
                d.update(world=world, **nearest_match(world, survivors))
                if d['survivor'] is not None:
                    s = next(s for s in survivors if s['id'] == d['survivor'])
                    delta = np.asarray(s['position_house']) - pose[:3, 3]
                    true_range = float(np.linalg.norm(delta[[0, 2]]))
                    true_z = float((pose[:3, :3].T @ delta)[2])
                    d.update(true_distance_m=true_range, true_optical_z_m=true_z,
                             range_error_m=d['range_m']-true_range, optical_error_m=d['range_m']-true_z,
                             radial_error_m=float(np.linalg.norm((np.asarray(world)-pose[:3, 3])[[0, 2]]))-true_range)
            visibility = assess_visibility(frame)
            brightness = float(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY).mean())
            condition = ('dark' if brightness < 40 else 'lit') + ('_usable' if visibility['usable_rgb'] else '_degraded')
            row = dict(run=rec.name, robot=robot, t=t, rgb=str(path.relative_to(ROOT)),
                       depth_t=frame['stamps']['depth'], pose_dt_s=dt, visible=visible, detections=detections,
                       condition=condition, mean_brightness=brightness, visibility={k: visibility[k] for k in
                       ('laplacian_variance', 'rms_contrast', 'dark_channel_haze', 'usable_rgb')})
            stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
            processed += 1
            if i % 150 == 0:
                print(f'{rec.name}/{robot}: {i+1}/{len(indices)} RGB frames', flush=True)
            if max_new_frames is not None and processed >= max_new_frames:
                break
    if signature != input_signature(rec, robot):
        raise ValueError(f'Recording indices changed during evaluation: {rec.name}/{robot}')
    rows = read_checkpoint(cache)
    expected = {t for t, _ in indices}
    if len(rows) != len({r['t'] for r in rows}) or not {r['t'] for r in rows} <= expected:
        raise ValueError(f'Duplicate or unknown frame cache {cache}')
    complete = len(rows) == len(indices)
    if not complete and max_new_frames is None:
        raise ValueError(f'Incomplete frame cache {cache}')
    faulthandler.cancel_dump_traceback_later()
    return dict(run=rec.name, robot=robot, frames=len(rows), complete=complete, max_pose_dt_s=max(r['pose_dt_s'] for r in rows))


def run_chunk(job, limit):
    print('Batch '+json.dumps(evaluate_robot(job, max_new_frames=limit)), flush=True)


def run_guarded(job, limit, timeout_s):
    """Bound native inference lifetime. Retry a timed-out batch with one thread."""
    rec, robot, out, threads = job
    expected = len(_index(rec/robot, 'rgb'))
    path = out/'frames'/f'{rec.name}__{robot}.jsonl'
    context = multiprocessing.get_context('spawn')
    while len(read_checkpoint(path, repair=True)) < expected:
        before = len(read_checkpoint(path))
        child = context.Process(target=run_chunk, args=((rec, robot, out, threads), limit))
        child.start()
        try:
            child.join(timeout_s)
            if child.is_alive():
                child.terminate(); child.join()
                if threads == 1:
                    raise TimeoutError(f'Single-thread inference timed out {rec.name}/{robot}')
                print(f'Native inference timeout {rec.name}/{robot}; retry with one thread', flush=True)
                threads = 1
                continue
            if child.exitcode != 0:
                raise RuntimeError(f'Inference worker failed {rec.name}/{robot}, exit {child.exitcode}')
        finally:
            if child.is_alive():
                child.terminate(); child.join()
        if len(read_checkpoint(path)) <= before:
            raise RuntimeError(f'No inference progress {rec.name}/{robot}')
    print(f'Finished {rec.name}/{robot}: {expected} RGB frames', flush=True)


def tile(f, detection=None, survivor=None, miss_threshold=.25):
    image = cv2.imread(str(ROOT/f['rgb']))
    if detection is not None:
        x, y, w, h = detection['bbox_rgb']
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 3)
        error = detection['error_m']
        error_text = 'no depth' if error is None else f'{error:.2f}m'
        detail = f'FALSE conf={detection["confidence"]:.3f} nearest={error_text}'
    else:
        rec = ROOT/'data/recordings'/f['run']
        svs = load_survivors(rec)
        if isinstance(svs, dict):
            svs = svs['survivors']
        s = next(s for s in svs if s['id'] == survivor)
        traj = load_trajectory(rec, f['robot']); j = int(np.argmin(abs(traj.timestamps-f['t'])))
        rot = Rotation.from_quat(traj.quaternions_xyzw[j]).as_matrix()
        cam = read_camera(rec/f['robot']/'camera.yaml')
        points = (rot.T @ (_body_points(s['position_house'], s.get('pose'))-traj.positions[j]).T).T
        points = points[points[:, 2] > .1]
        if len(points):
            uv = np.column_stack([cam.fx*points[:, 0]/points[:, 2]+cam.cx, cam.fy*points[:, 1]/points[:, 2]+cam.cy])
            uv = np.clip(uv, -10000, 10000).astype(np.int32)
            cv2.polylines(image, [cv2.convexHull(uv)], True, (0, 220, 255), 2)
        detail = f'MISSED {survivor} at threshold {miss_threshold:.2f}; yellow truth body volume'
    image = cv2.resize(image, (480, 360))
    canvas = np.full((430, 480, 3), 240, np.uint8); canvas[70:] = image
    cv2.putText(canvas, f'{f["run"]} {f["robot"]} {f["t"]:.3f}s', (4, 18), cv2.FONT_HERSHEY_SIMPLEX, .40, (0, 0, 0), 1)
    cv2.putText(canvas, detail[:100], (4, 39), cv2.FONT_HERSHEY_SIMPLEX, .37, (0, 0, 0), 1)
    if detection is not None and detection.get('description'):
        cv2.putText(canvas, detection['description'][:86], (4, 59), cv2.FONT_HERSHEY_SIMPLEX, .32, (0, 0, 0), 1)
    return canvas


def write_sheet(tiles, path):
    if not tiles:
        tiles = [np.full((430, 480, 3), 240, np.uint8)]
        cv2.putText(tiles[0], 'No examples', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 1)
    while len(tiles) % 5:
        tiles.append(np.full_like(tiles[0], 240))
    if not cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i+5]) for i in range(0, len(tiles), 5)])):
        raise OSError(f'Cannot write {path}')


def trajectory_overlaps(runs):
    """Report equal camera-pose prefixes; RGB/smoke need not be equal."""
    entries = []
    for run in runs:
        rec = ROOT/'data/recordings'/run
        for index in sorted(rec.glob('*/rgb.txt')):
            traj = load_trajectory(rec, index.parent.name)
            rotations = Rotation.from_quat(traj.quaternions_xyzw).as_matrix().reshape(-1, 9)
            entries.append((run, index.parent.name, np.column_stack([traj.timestamps, traj.positions, rotations])))
    overlaps = []
    for i, (a, ra, x) in enumerate(entries):
        for b, rb, y in entries[i+1:]:
            if a == b or ra != rb:
                continue
            n = min(len(x), len(y))
            equal = np.all(np.isclose(x[:n], y[:n], atol=1e-5, rtol=0), axis=1)
            different = np.flatnonzero(~equal)
            prefix = int(different[0]) if len(different) else n
            if prefix > 1 and x[prefix-1, 0] - x[0, 0] >= 1.:
                overlaps.append(dict(run_a=a, run_b=b, robot=ra, equal_prefix_poses=prefix,
                                     through_s=float(x[prefix-1, 0])))
    return overlaps


def summarize(frames, out, runs, skipped):
    overall = {str(t): metrics(frames, t) for t in THRESHOLDS}
    report = dict(runs=runs, skipped=skipped, opencv=cv2.__version__, frames=len(frames),
                  metrics=overall,
                  recording_frame_counts={r: sum(f['run'] == r for f in frames) for r in runs},
                  max_pose_dt_s=max((f.get('pose_dt_s', 0.) for f in frames), default=0.),
                  max_rgb_depth_dt_s=max((abs(f['t']-f.get('depth_t', f['t'])) for f in frames), default=0.),
                  by_range={b: {str(t): metrics(frames, t, band=b) for t in THRESHOLDS} for b in BANDS},
                  by_condition={c: {str(t): metrics(frames, t, condition=c) for t in THRESHOLDS}
                                for c in sorted({f['condition'] for f in frames})},
                  by_run={r: {str(t): metrics([f for f in frames if f['run'] == r], t) for t in THRESHOLDS} for r in runs},
                  by_haze_proxy={name: {str(t): metrics([f for f in frames if (f['visibility']['dark_channel_haze'] >= .2) == high], t)
                                        for t in THRESHOLDS} for name, high in [('below_0.2', False), ('at_least_0.2', True)]})
    all_s, visible_s, positions, visible_positions, windows, positive_windows = set(), set(), set(), set(), set(), set()
    position_lookup = {}
    report['recording_conditions'] = {}
    for run in runs:
        ss = load_survivors(ROOT/'data/recordings'/run)
        if isinstance(ss, dict): ss = ss['survivors']
        meta = json.loads((ROOT/'data/recordings'/run/'meta.json').read_text())
        report['recording_conditions'][run] = {k: meta.get(k) for k in ('seed', 'smoke_level', 'smoke_rooms', 'exposure')}
        for s in ss:
            all_s.add((run, s['id'])); positions.add(tuple(s['position_house']))
            position_lookup[(run, s['id'])] = tuple(s['position_house'])
    for f in frames:
        windows.add((f['run'], f['robot'], int(f['t']//10)))
        for v in f['visible']:
            visible_s.add((f['run'], v['survivor']))
            visible_positions.add(position_lookup[(f['run'], v['survivor'])])
            positive_windows.add((f['run'], f['robot'], v['survivor'], int(f['t']//10)))
    report['sample_sizes'] = dict(recording_survivor_ids=len(all_s), unique_survivor_positions=len(positions),
        visible_recording_survivor_ids=sorted('/'.join(v) for v in visible_s),
        distinct_10s_robot_windows=len(windows), distinct_positive_10s_survivor_robot_windows=len(positive_windows),
        distinct_positive_10s_robot_windows=len({(r, robot, block) for r, robot, survivor, block in positive_windows}),
        visible_episodes=len(episodes(frames)), visible_unique_positions=len(visible_positions))
    report['trajectory_overlaps'] = trajectory_overlaps(runs)
    fps = max(read_camera(index.parent/'camera.yaml').fps for run in runs
              for index in (ROOT/'data/recordings'/run).glob('*/rgb.txt'))
    phase_offsets = [p/round(fps) for p in range(round(fps))]
    configurations = [(t, n, w) for t in THRESHOLDS for n in (1, 2, 3) for w in (2., 3., 5.) if n <= w+1]
    phase_results, grid, zero_phase = {}, [], []
    for threshold, repeats, window in configurations:
        cases = [replay_candidates(frames, threshold, repeats=repeats, window_s=window, phase_s=phase)
                 for phase in phase_offsets]
        key = (threshold, repeats, window)
        phase_results[key] = cases
        zero_phase.append({k: v for k, v in cases[0].items() if k != 'events'})
        by_recording = {}
        for run in runs:
            events = [e for case in cases for e in case['events'] if e['run'] == run]
            tp = sum(e['survivor'] is not None for e in events)
            by_recording[run] = dict(tp=tp, fp=len(events)-tp, precision=tp/len(events) if events else None)
        rates = [r['precision'] for r in by_recording.values() if r['precision'] is not None]
        grid.append(dict(threshold=threshold, repeats=repeats, window_s=window, **phase_summary(cases),
                         by_recording=by_recording, worst_recording_precision=min(rates) if rates else None))
    report['candidate_replays'] = zero_phase
    report['candidate_phase_grid'] = grid
    report['sampling_phase_offsets_s'] = phase_offsets
    best, meets_target = choose_operating_point(grid)
    report['operating_point_policy'] = 'At least two sightings and 95% event precision both pooled and within each recording across sampling phases; then maximize episode recall.'
    report['recommendation'] = dict(best) if best else None
    if best:
        report['recommendation']['meets_0.95_event_precision_target'] = meets_target
        cases = phase_results[(best['threshold'], best['repeats'], best['window_s'])]
        (out/'recommended_candidate_events.json').write_text(json.dumps(
            [dict(phase_s=r['phase_s'], events=r['events']) for r in cases], indent=2))
        report['recommendation_sampling_phases'] = [{k: v for k, v in r.items() if k != 'events'} for r in cases]
    false = sorted([(f, d) for f in frames for d in f['detections'] if d['survivor'] is None], key=lambda p: -p[1]['confidence'])
    reviews_path = out/'visual_review.json'
    reviews = json.loads(reviews_path.read_text()) if reviews_path.exists() else {}
    false_rows = []
    for f, d in false:
        if d['confidence'] <= .5:
            continue
        key = f'{f["run"]}/{f["robot"]}/{f["t"]:.6f}/' + ','.join(str(n) for n in d['bbox_rgb'])
        false_rows.append(dict(key=key, run=f['run'], robot=f['robot'], t=f['t'], confidence=d['confidence'],
            bbox_rgb=json.dumps(d['bbox_rgb']), range_m=d['range_m'], nearest_survivor=d['nearest_survivor'],
            nearest_distance_m=d['error_m'], description=reviews.get(key, 'Unreviewed RGB detection; object identity unresolved.'), rgb=f['rgb']))
    fields = ['key', 'run', 'robot', 't', 'confidence', 'bbox_rgb', 'range_m', 'nearest_survivor', 'nearest_distance_m', 'description', 'rgb']
    with (out/'false_above_0.5.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(false_rows)
    top_false = []
    for f, d in false[:30]:
        key = f'{f["run"]}/{f["robot"]}/{f["t"]:.6f}/' + ','.join(str(n) for n in d['bbox_rgb'])
        top_false.append((f, dict(d, description=reviews.get(key, 'RGB review pending.'))))
    write_sheet([tile(f, detection=d) for f, d in top_false], out/'false_top30.png')
    (out/'false_top30.json').write_text(json.dumps([dict(run=f['run'], robot=f['robot'], t=f['t'], **d) for f, d in top_false], indent=2))
    miss_threshold = .25
    report['missed_contact_threshold'] = miss_threshold
    missed = [(f, v) for f in frames for v in f['visible'] if not any(d['confidence'] >= miss_threshold and d['survivor'] == v['survivor'] for d in f['detections'])]
    # Spread examples across survivor/robot/10s windows before filling remaining slots.
    picked, seen = [], set()
    for f, v in sorted(missed, key=lambda p: p[1]['range_m']):
        key = (f['run'], f['robot'], v['survivor'], int(f['t']//10))
        if key not in seen:
            seen.add(key); picked.append((f, v))
        if len(picked) == 30: break
    if len(picked) < 30:
        keys = {(f['run'], f['robot'], f['t'], v['survivor']) for f, v in picked}
        picked += [(f, v) for f, v in missed if (f['run'], f['robot'], f['t'], v['survivor']) not in keys][:30-len(picked)]
    write_sheet([tile(f, survivor=v['survivor'], miss_threshold=miss_threshold) for f, v in picked], out/'missed_top30.png')
    (out/'missed_top30.json').write_text(json.dumps([dict(run=f['run'], robot=f['robot'], t=f['t'], threshold=miss_threshold, **v) for f, v in picked], indent=2))
    checks = out/'confirmation_checks/results.json'
    if checks.exists():
        report['selected_confirmation_checks'] = json.loads(checks.read_text())
        report['confirmation_check_note'] = 'Selected false detections only, not a full confirmation-stage precision or recall audit.'
    report['false_above_0.5'] = len(false_rows)
    report['false_above_0.5_time_windows'] = len({(r['run'], r['robot'], int(r['t']//10)) for r in false_rows})
    report['visible_episodes'] = episodes(frames)
    report['condition_frame_counts'] = {c: sum(f['condition'] == c for f in frames) for c in report['by_condition']}
    report['limitations'] = [
        'All frames are evaluated, including overlapping short recordings. Counts are correlated and not independent trials.',
        'Survivor IDs can repeat across recordings; unique positions are an upper bound on distinct physical scene instances, not people.',
        'LOS and horizontal FOV are geometric opportunity labels. Furniture, vertical clipping, smoke and darkness can hide a survivor in that set.',
        'Precision uses nearest survivor within 1 m in building x,z, without visibility gating. Recall counts each visible survivor at most once.',
        'Precision range bands use predicted optical depth; recall bands use true planar distance. They cannot be paired as identical populations.',
        'Raw range error compares returned optical depth with true planar distance as requested. Optical and reconstructed planar errors are also reported.',
        'Dark means mean grayscale below 40/255. assess_visibility is an image usability proxy, not a physical smoke measurement.',
        'Candidate replay samples first RGB frame per second. Recall uses full-rate visible episodes, including opportunities missed by sampling. Association uses no truth; confirmation and changed trajectories are not modeled.',
        'The operating point is chosen on these recordings and has no held-out validation. No population-level precision guarantee is supported.',
        'Phase-grid counts sum alternative one-second sampling offsets, not new observations. Expected event precision and episode recall assume an equally likely fixed offset. Association uses perfect true poses.',
        'Range error is conditional on passing the 1 m localization match; errors that fail that match count as false detections instead.',
        'inspect_candidate can accept the same RGB detection after finding no warm component. This is not independent evidence. Selected diagnostic results are separate from the detector and candidate metrics.'
    ]
    smoke_settings = sorted({c['smoke_level'] for c in report['recording_conditions'].values()}, key=str)
    report['by_recorded_smoke_setting'] = {str(level): {str(t): metrics(
        [f for f in frames if report['recording_conditions'][f['run']]['smoke_level'] == level], t)
        for t in THRESHOLDS} for level in smoke_settings}
    report['recorded_smoke_note'] = 'Run default from meta.json, not measured room-level smoke density or optical extinction.'
    report['threshold_tables'] = []
    for category, groups in [('overall', {'all': overall}), ('range', report['by_range']),
                             ('condition', report['by_condition']), ('haze_proxy', report['by_haze_proxy']),
                             ('recorded_smoke_setting', report['by_recorded_smoke_setting']), ('run', report['by_run'])]:
        for group, scores in groups.items():
            for threshold, m in scores.items():
                report['threshold_tables'].append(dict(category=category, group=group, threshold=threshold,
                    **{k: m[k] for k in ('tp', 'fp', 'hits', 'visible', 'precision', 'recall')},
                    range_mae_m=m['range_error'].get('mae_m'), range_bias_m=m['range_error'].get('bias_m')))
    with (out/'thresholds.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report['threshold_tables'][0]))
        writer.writeheader(); writer.writerows(report['threshold_tables'])
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    def fmt(value):
        return 'NA' if value is None else f'{value:.4f}'
    lines = ['threshold  TP  FP  hit/views  precision  recall']
    for t, m in overall.items():
        lines.append(f'{t:>9} {m["tp"]:5} {m["fp"]:5} {m["hits"]:5}/{m["visible"]:<5} {fmt(m["precision"])} {fmt(m["recall"])}')
    for category in ('range', 'condition', 'haze_proxy', 'recorded_smoke_setting'):
        lines.append(category+' group threshold TP FP hit/views precision recall range_MAE_m')
        for row in report['threshold_tables']:
            if row['category'] != category:
                continue
            lines.append(f'{row["group"]} {row["threshold"]} {row["tp"]} {row["fp"]} {row["hits"]}/{row["visible"]} '
                         f'{fmt(row["precision"])} {fmt(row["recall"])} {fmt(row["range_mae_m"])}')
    lines.append('Recommendation '+json.dumps(report['recommendation']))
    lines.append('Sample sizes '+json.dumps(report['sample_sizes']))
    lines.append('Recordings '+', '.join(runs))
    lines.append('Output '+str(out))
    (out/'summary.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/rgb_people_eval')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--runs', nargs='+', help='default is every eligible recording')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--chunk-frames', type=int, default=100, help='restart native inference after this many frames; 0 disables')
    parser.add_argument('--timeout-s', type=float, default=180., help='per-process timeout for chunked inference')
    args = parser.parse_args(argv)
    if not PERSON_ONNX.exists():
        raise FileNotFoundError(PERSON_ONNX)
    recordings, skipped = discover(ROOT/'data/recordings')
    if args.runs:
        recordings = [r for r in recordings if r.name in args.runs]
        if set(args.runs) != {r.name for r in recordings}:
            raise ValueError('Requested recording is not eligible')
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    jobs = [(r, p.parent.name, out, args.threads) for r in recordings for p in sorted(r.glob('*/rgb.txt'))]
    print('Eligible recordings '+', '.join(r.name for r in recordings), flush=True)
    jobs.sort(key=lambda job: (job[0].name not in {'unity_house_reflex_check', 'unity_dark_torch_agents'}, job[0].name, job[1]))
    if not args.summarize_only:
        if args.workers == 1:
            for job in jobs:
                if args.chunk_frames:
                    run_guarded(job, args.chunk_frames, args.timeout_s)
                else:
                    print('Finished '+json.dumps(evaluate_robot(job)), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(evaluate_robot, job) for job in jobs]
                for future in as_completed(futures):
                    print('Finished '+json.dumps(future.result()), flush=True)
    frames = []
    for rec, robot, _, _ in jobs:
        path = out/'frames'/f'{rec.name}__{robot}.jsonl'
        rows = read_checkpoint(path)
        expected = _index(rec/robot, 'rgb')
        if len(rows) != len(expected) or {r['t'] for r in rows} != {t for t, _ in expected}:
            raise ValueError(f'Incomplete frame results {path}')
        if json.loads(path.with_suffix('.inputs.json').read_text()) != input_signature(rec, robot):
            raise ValueError(f'Input signature changed {path}')
        frames.extend(rows)
    summarize(frames, out, [r.name for r in recordings], skipped)


if __name__ == '__main__':
    main()
