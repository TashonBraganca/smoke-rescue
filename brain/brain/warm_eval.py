"""Offline warm-blob audit. Run uv run python -m brain.warm_eval.

Building transforms and truth stay here; warm.py contains sensor-only code.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
from brain import warm
from brain.agent.vision import load_frame, _index
from brain.agent.eval_vision import survivor_visibility, match_component
from brain.evaluator import gt_pose_at
from brain.oracle import load_survivors, load_floorplan, load_trajectory
from brain.nav_io import (read_json_lines, poses_from_states, interpolate_poses,
                          load_scenario, recording_frame, transform_xy, rotation)
ROOT = Path(__file__).resolve().parents[2]
RUNS = ('unity_house_reflex_check', 'unity_house_reflex_v1', 'unity_dark_torch_agents')
KEYS = ('baseline', 'height', 'separate', 'contrast', 'geometry', 'combined')


def serial(comp):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in comp.items() if k != 'mask'}


def score(rows, key):
    tp = sum(r['label'] and r[key] for r in rows)
    fp = sum(not r['label'] and r[key] for r in rows)
    positive = sum(r['label'] for r in rows)
    return dict(tp=int(tp), fp=int(fp), fn=int(positive-tp), positives=int(positive),
                precision=tp/(tp+fp) if tp+fp else None, recall=tp/positive if positive else None)


def predictions(comp):
    g = warm.gates(comp)
    return dict(baseline=True, **g, geometry=g['height'] and g['separate'], combined=all(g.values()))


def label_alternative(components, visibility, survivors, base):
    """Use identical visibility and ambiguity rules for the contrast experiment."""
    rows, hits = [], {s['id']: [] for s in survivors}
    for comp in components:
        owners = [s['id'] for s in survivors if visibility[s['id']] is not None and match_component(comp, visibility[s['id']])]
        visible = [sid for sid in owners if visibility[sid]['visible']]
        if visible:
            for sid in visible:
                hits[sid].append(comp)
        elif not owners:
            rows.append(dict(**base, label=False, **predictions(comp), blob=serial(comp)))
    for s in survivors:
        vis = visibility[s['id']]
        if vis is not None and vis['visible']:
            rows.append(dict(**base, label=True, survivor=s['id'],
                             **{key: any(predictions(c)[key] for c in hits[s['id']]) for key in KEYS},
                             blobs=[serial(c) for c in hits[s['id']]]))
    return rows


def capture_pose(folder, states, t):
    """Replay interval-ending motion for legacy logs without LiDAR corrections.

    Interpolating 2 Hz yaw through a reversal can displace a distant blob by 67 cm.
    LiDAR correction histories are not logged at capture rate, so corrected runs
    retain state interpolation and explicitly report that source of uncertainty.
    """
    times, poses = poses_from_states(states)
    clipped = float(np.clip(t, times[0], times[-1]))
    if (folder/'lidar.txt').exists() or not (folder/'imu.txt').exists() or not (folder/'odom.txt').exists():
        return interpolate_poses(times, poses, np.array([clipped]))[0], '2 Hz state interpolation'
    index = max(0, int(np.searchsorted(times, clipped, side='right'))-1)
    start, pose = times[index], poses[index].copy()
    odom = np.loadtxt(folder/'odom.txt', comments='#', ndmin=2)
    imu = np.loadtxt(folder/'imu.txt', comments='#', ndmin=2)
    bounds = np.unique(np.r_[start, clipped, odom[(odom[:, 0]>start)&(odom[:, 0]<clipped), 0],
                             imu[(imu[:, 0]>start)&(imu[:, 0]<clipped), 0]])
    for a, b in zip(bounds[:-1], bounds[1:]):
        o = odom[min(np.searchsorted(odom[:, 0], b-1e-8), len(odom)-1)]
        i = imu[min(np.searchsorted(imu[:, 0], b-1e-8), len(imu)-1)]
        dt = b-a; yaw = -i[5]; half = yaw*dt/2
        scale = dt*np.sinc(half/np.pi)
        c, s = np.cos(pose[2]+half), np.sin(pose[2]+half)
        right, forward = -o[2]*scale, o[1]*scale
        pose += [c*right-s*forward, s*right+c*forward, yaw*dt]
    return pose, 'interval-ending odometry and IMU replay from preceding state'


def contact_tile(frame, comp, title, subtitle):
    mask = comp['mask'] if comp else np.ones(frame.thermal.shape, bool)
    y, x = np.nonzero(mask)
    x0, x1 = max(0, x.min()-12), min(mask.shape[1], x.max()+13)
    y0, y1 = max(0, y.min()-12), min(mask.shape[0], y.max()+13)
    temp = frame.thermal.astype(float)/10-273.15
    colour = cv2.applyColorMap(np.clip((temp-15)/45*255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(colour, contours, -1, (0, 255, 0), 1)
    a, b = frame.thermal_K, frame.K
    ux = np.rint((np.array([x0, x1-1])-a[0, 2])/a[0, 0]*b[0, 0]+b[0, 2]).astype(int)
    vy = np.rint((np.array([y0, y1-1])-a[1, 2])/a[1, 1]*b[1, 1]+b[1, 2]).astype(int)
    ux = np.clip(ux, 0, frame.rgb.shape[1]-1); vy = np.clip(vy, 0, frame.rgb.shape[0]-1)
    rgb = cv2.cvtColor(frame.rgb[vy[0]:vy[1]+1, ux[0]:ux[1]+1], cv2.COLOR_RGB2BGR)
    tile = np.full((305, 720, 3), 240, np.uint8)
    for offset, image in [(0, colour[y0:y1, x0:x1]), (360, rgb)]:
        scale = min(350/image.shape[1], 195/image.shape[0])
        width, height = max(1, round(image.shape[1]*scale)), max(1, round(image.shape[0]*scale))
        left, top = offset+(360-width)//2, 53+(195-height)//2
        tile[top:top+height, left:left+width] = cv2.resize(image, (width, height))
    measurements = ''
    if comp is not None and 'height_m' in comp:
        lo, _, hi = comp['height_m']
        measurements = f'T50={comp["temperature_c"][1]:.1f} C, height5..95={lo:.2f}..{hi:.2f} m, area={comp["area_m2"]:.3f} m2'
    for text, yy in [(title, 19), (subtitle, 40), (measurements, 270), ('Thermal 15..60 C, green mask | RGB, same direction; aspect preserved', 295)]:
        cv2.putText(tile, text, (5, yy), cv2.FONT_HERSHEY_SIMPLEX, .43, (15, 15, 15), 1, cv2.LINE_AA)
    return tile


def describe(comp):
    if comp is None:
        return 'No reproduced blob; identity unresolved.'
    support = comp['ring_plane_support']
    plane = comp['plane_rms_m'] <= comp['plane_tolerance_m'] and support is not None and support >= .65
    if plane and comp['plane_up_normal'] > .85:
        return ('Warm horizontal plane consistent with ceiling; RGB review required.' if comp['height_m'][1] > 2. else
                'Warm horizontal plane consistent with floor; RGB review required.')
    if plane and comp['plane_up_normal'] < .35:
        return 'Warm vertical plane consistent with wall, frame or window; RGB review required.'
    if comp['height_m'][2] > 1.25:
        return 'High warm structure; wall/frame/furniture identity requires RGB review.'
    return 'Low warm object; RGB review required to distinguish person, debris and furniture.'


def evaluate_run(recording, out, *, stride=1, event_only=False):
    metadata_bytes = (recording/'meta.json').read_bytes()
    meta = json.loads(metadata_bytes)
    index_files = [recording/r/(sensor+'.txt') for r in meta['robots'] for sensor in ('rgb', 'depth', 'thermal')]
    before = {str(path.relative_to(recording)): hashlib.sha256(path.read_bytes()).hexdigest() for path in index_files}
    config = json.loads((ROOT/'config/brain.json').read_text())
    sensors = json.loads((ROOT/'config/sensors.json').read_text())
    mount = meta.get('camera_mount', sensors['camera']['mount'])
    scenario = load_scenario(meta)
    survivors = load_survivors(recording)
    if isinstance(survivors, dict):
        survivors = survivors['survivors']
    floorplan = load_floorplan(recording)
    truth_xy = np.array([s['position_house'][::2] for s in survivors])
    review_file = out/'visual_review.json'
    reviews = json.loads(review_file.read_text()) if review_file.exists() else {}
    events_out, rows, tiles, robots, positive_tiles = [], [], [], {}, []
    example_groups = set()
    frames_count = 0
    alternative_rows = []
    for robot in meta['robots']:
        folder = recording/robot
        states = read_json_lines(folder/'brain_state.jsonl')
        transform, convention = recording_frame(states, scenario, robot) if states else (np.zeros(3), 'no brain states; frame-only evaluation')
        events = [e for e in read_json_lines(folder/'decisions.jsonl') if e['kind'] in {'survivor_candidate', 'survivor_confirmed'}]
        robots[robot] = dict(frame=convention, building_transform=transform.tolist())
        if events and not states:
            raise ValueError(f'{robot}: cannot place candidate events without brain states')
        for event in events:
            t, n = event['t'], event['numbers']
            frame = load_frame(folder, t)
            pose, pose_method = capture_pose(folder, states, frame['stamps']['depth'])
            components = warm.detect(frame, config, mount, pose, depth_limits=(sensors['depth']['min_m'], sensors['depth']['max_m']), noise_coeff=sensors['depth']['sigma_coeff_mm_per_m2']*.001)
            own = np.array([n['x_m'], n['y_m']]); xy = transform_xy(own, transform)
            distance = np.linalg.norm(truth_xy-xy, axis=1)
            comp = min(components, key=lambda c: np.linalg.norm(c['xy']-own), default=None)
            delta = float(np.linalg.norm(comp['xy']-own)) if comp else None
            area_error = abs(comp['area_m2']-n['area_m2'])/n['area_m2'] if comp and 'area_m2' in n else None
            matched = comp is not None and delta <= .3 and (area_error is None or area_error <= .02)
            real = bool(distance.min() <= 1.)
            row = dict(run=recording.name, robot=robot, t=t, kind=event['kind'], robot_xy=own.tolist(), building_xy=xy.tolist(),
                       nearest_survivor=survivors[int(distance.argmin())]['id'], nearest_distance_m=float(distance.min()), real=real,
                       frame_stamps=frame['stamps'], pose_method=pose_method, matched=matched, centroid_error_m=delta, relative_area_error=area_error,
                       blob=serial(comp) if comp else None, rules=predictions(comp) if comp else None, description=describe(comp))
            if not matched:
                row['parity_note'] = ('Confirmation is a moving track, not a detector output.' if event['kind']=='survivor_confirmed' else
                                      '2 Hz state interpolation and nearest saved sensor frame can differ from the live update; inspect area parity and timestamps.')
            row['visual_review'] = reviews.get(f'{robot}/{t:.3f}')
            events_out.append(row)
            title = f'{recording.name.removeprefix("unity_")} {robot} t={t:.3f} {event["kind"].removeprefix("survivor_")}'
            subtitle = f'{"REAL" if real else "FALSE"} ({xy[0]:.2f}, {xy[1]:.2f}) nearest={distance.min():.2f}m match={matched}'
            tiles.append(contact_tile(frame, comp, title, subtitle))
            print(json.dumps({k: v for k, v in row.items() if k not in {'blob', 'rules'}}), flush=True)
        if event_only:
            continue
        trajectory = load_trajectory(recording, robot)
        for i, (t, _) in enumerate(_index(folder, 'thermal')[::stride]):
            frame = load_frame(folder, t)
            pose = gt_pose_at(trajectory, t)
            visibility = {s['id']: survivor_visibility(frame, pose, s, floorplan) for s in survivors}
            components = warm.detect(frame, config, mount, depth_limits=(sensors['depth']['min_m'], sensors['depth']['max_m']), noise_coeff=sensors['depth']['sigma_coeff_mm_per_m2']*.001)
            hits = {s['id']: [] for s in survivors}
            base = dict(run=recording.name, robot=robot, t=t)
            for comp in components:
                owners = [s['id'] for s in survivors if visibility[s['id']] is not None and match_component(comp, visibility[s['id']])]
                visible_owners = [sid for sid in owners if visibility[sid]['visible']]
                if visible_owners:
                    for sid in visible_owners:
                        hits[sid].append(comp)
                elif not owners:
                    rows.append(dict(**base, label=False, group=f'{recording.name}/{robot}/negative/{int(t//10)}', **predictions(comp), blob=serial(comp)))
            for s in survivors:
                vis = visibility[s['id']]
                if vis is None or not vis['visible']:
                    continue
                matched = hits[s['id']]
                values = {key: any(predictions(c)[key] for c in matched) for key in KEYS}
                group = f'{recording.name}/{robot}/{s["id"]}/{int(t//10)}'
                uv = np.rint(vis['uv_thermal']).astype(int)
                inside = (uv[:, 0]>=0)&(uv[:, 0]<frame.thermal.shape[1])&(uv[:, 1]>=0)&(uv[:, 1]<frame.thermal.shape[0])
                uv = uv[inside]
                temperatures = frame.thermal[uv[:, 1], uv[:, 0]].astype(float)/10-273.15
                temperatures = temperatures[temperatures > -273.15]
                row = dict(**base, label=True, survivor=s['id'], group=group,
                           **values, blobs=[serial(c) for c in matched],
                           thermal_in_image_fraction=float(inside.mean()) if len(inside) else 0.,
                           body_volume_temp_p10_p50_p90_c=np.percentile(temperatures, [10, 50, 90]).tolist() if len(temperatures) else None,
                           body_volume_band_fraction=float(np.mean((temperatures>=config['person_min_c'])&(temperatures<=config['person_max_c']))) if len(temperatures) else None,
                           visibility={k: v for k, v in vis.items() if k != 'uv_thermal'})
                rows.append(row)
                if group not in example_groups and len(positive_tiles) < 8 and len(uv) >= 3:
                    example_groups.add(group)
                    mask = np.zeros(frame.thermal.shape, np.uint8)
                    cv2.fillConvexPoly(mask, cv2.convexHull(uv.astype(np.int32)), 1)
                    positive_tiles.append(contact_tile(frame, dict(mask=mask.astype(bool)),
                        f'{recording.name} {robot} {s["id"]} t={t:.3f}',
                        f'TRUTH BODY VOLUME, baseline hit={values["baseline"]}, filtered hit={values["combined"]}'))
            alternative = warm.detect(frame, config, mount, segmentation='contrast',
                                      depth_limits=(sensors['depth']['min_m'], sensors['depth']['max_m']), noise_coeff=sensors['depth']['sigma_coeff_mm_per_m2']*.001)
            alternative_rows.extend(label_alternative(alternative, visibility, survivors, base))
            frames_count += 1
            if i % 300 == 0:
                print(f'{recording.name} {robot}: {i*stride} frames scanned, {len(rows)} labelled rows', flush=True)
    after = {str(path.relative_to(recording)): hashlib.sha256(path.read_bytes()).hexdigest() for path in index_files}
    if after != before or (recording/'meta.json').read_bytes() != metadata_bytes:
        raise ValueError(f'{recording.name}: recording changed during evaluation; discard partial results')
    out.mkdir(parents=True, exist_ok=True)
    if tiles:
        if len(tiles) % 2:
            tiles.append(np.full_like(tiles[0], 240))
        sheet = np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, len(tiles), 2)])
        if not cv2.imwrite(str(out/'candidates.png'), sheet):
            raise OSError('contact sheet write failed')
    if positive_tiles:
        if len(positive_tiles) % 2:
            positive_tiles.append(np.full_like(positive_tiles[0], 240))
        if not cv2.imwrite(str(out/'visible_survivors.png'), np.vstack([np.hstack(positive_tiles[i:i+2]) for i in range(0, len(positive_tiles), 2)])):
            raise OSError('positive sheet write failed')
    report = dict(run=recording.name, frames_evaluated=frames_count, stride=stride, robots=robots,
                  settings=dict(baseline=config, proposed=warm.PROPOSED, mount=mount), events=events_out,
                  input_index_sha256=before, metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
                  completed_metadata=meta, event_parity_tolerances=dict(centroid_m=.3, relative_area=.02),
                  detector_parity_note='Graph components equal four-connected depth-edge flood fill. Double reductions differ from C# float accumulation; state interpolation remains approximate in LiDAR-corrected runs.')
    for kind in ('survivor_candidate', 'survivor_confirmed'):
        subset = [e for e in events_out if e['kind']==kind]
        report[kind] = dict(total=len(subset), real=sum(e['real'] for e in subset),
                            precision=sum(e['real'] for e in subset)/len(subset) if subset else None, reproduced=sum(e['matched'] for e in subset))
    report['event_filter_metrics'] = {kind: {key: score([dict(label=e['real'], **(e['rules'] or {k: False for k in KEYS})) for e in events_out if e['kind']==kind], key) for key in KEYS} for kind in ('survivor_candidate', 'survivor_confirmed')}
    report['relative_contrast_metrics'] = {key: score(alternative_rows, key) for key in KEYS}
    report['relative_contrast_note'] = 'Absolute temperature band replaced by absolute difference from a normalized 61px square local mean. Geometry combines height and separation; combined also demands the original positive ring contrast.'
    (out/'contrast_rows.json').write_text(json.dumps(alternative_rows, allow_nan=False))
    report['metrics'] = {key: score(rows, key) for key in KEYS}
    report['thermal_eligible_metrics'] = {key: score([row for row in rows if not row['label'] or row.get('thermal_in_image_fraction', 0.) >= .5], key) for key in KEYS}
    report['fixed_trajectory_candidate_replay'] = {key: replay_candidates(rows, recording, config, key) for key in ('height', 'geometry', 'combined')}
    report['replay_note'] = 'Candidate votes only, with interpolated logged poses. Excludes ambiguous components omitted from frame metrics and does not simulate confirmation, changed routes, or turning commands.'
    report['visible_survivors'] = sorted({r['survivor'] for r in rows if r['label']})
    report['positive_groups'] = len({r['group'] for r in rows if r['label']})
    with (out/'events.csv').open('w', newline='') as stream:
        fields = ['robot', 't', 'kind', 'building_x_m', 'building_y_m', 'nearest_survivor', 'nearest_distance_m', 'real', 'matched', 'relative_area_error', 'centroid_error_m', 'description', 'visual_review']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for event in events_out:
            writer.writerow({key: event.get(key) for key in fields} | dict(building_x_m=event['building_xy'][0], building_y_m=event['building_xy'][1]))
    (out/'rows.json').write_text(json.dumps(rows, allow_nan=False))
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report, rows


def replay_candidates(rows, recording, config, key):
    """Fixed recorded trajectory, 2 Hz interpolated poses, no new dwell or routes."""
    events = []
    for robot in sorted({r['robot'] for r in rows}):
        states = read_json_lines(recording/robot/'brain_state.jsonl')
        if not states:
            continue
        times, poses = poses_from_states(states)
        tracker = warm.CandidateTracker(config)
        detections_by_time = {}
        for row in rows:
            if row['robot'] != robot or not row[key]:
                continue
            blobs = row.get('blobs', []) if row['label'] else [row['blob']]
            pose = interpolate_poses(times, poses, np.array([np.clip(row['t'], times[0], times[-1])]))[0]
            for blob in blobs:
                if predictions(blob)[key]:
                    xy = np.asarray(blob['xy']) @ rotation(pose[2]).T + pose[:2]
                    detections_by_time.setdefault(row['t'], []).append(dict(xy=xy, area_m2=blob['area_m2']))
        for stamp, detections in sorted(detections_by_time.items()):
            for event in tracker.observe(stamp, detections):
                events.append(dict(robot=robot, **event))
    return events


def porting_spec():
    proposed = {}
    for key, value in warm.PROPOSED.items():
        proposed[key] = value
        proposed[key+'_reason'] = warm.PROPOSED_REASONS[key]
    return dict(
        status='Provisional sensor-only false-alarm filter. Real-person retention is not established. Do not treat a surviving blob as a verified person.',
        target='ThermalPerception.Detect(ThermalFrame thermal, RgbdFrame rgb, SensorConfig s, BrainConfig c, BrainPose pose)',
        reference='brain/brain/warm.py detect_filtered. Use rules=(height,) for the minimum candidate filter; rules=(height,separate,contrast) is the measured experimental combination.',
        recommendation='Height alone matches the combined filter on the requested completed recordings. Prototype it as a candidate filter only. Plane, recess and contrast gates add no benefit to that target result and are not justified for deployment by this sample.',
        inputs=['Thermal uint16 Kelvin*10 image and intrinsics', 'Synchronized RGB-D axial depth and intrinsics from camera.yaml',
                'Camera mount height, forward offset and downward pitch', 'Depth range and noise sigma coefficient from SensorConfig',
                'Robot own-frame capture-time pose from motion.At, never true pose or building geometry'],
        baseline='Retain Match, row-major four-neighbor flood fill, per-edge blob_depth_jump_m, pixel count, summed z*z/(thermal.fx*thermal.fy) area and area-weighted pose.Transform centroid. Ignore the 3 cm translation for parity with existing C#.',
        backprojection='down=(v-cy)/fy*z; right=(u-cx)/fx*z; forward=mount.forward_offset_m+z*cos(pitch)-down*sin(pitch); height=mount.height_m-z*sin(pitch)-down*cos(pitch).',
        insertion='After each baseline component passes pixel/area bounds, before result.Add. Rejected blobs must not vote in ObservePeople. Keep existing tracking, approach and dwell logic. Repeated static-wall sightings are not independent evidence of a person.',
        functions={
            'BlobGeometry': 'Collect right/forward/height and temperature for every valid component pixel. Use unweighted linear percentiles q5/q50/q95 with index (n-1)*q and interpolation between sorted neighbors. The existing centroid remains weighted by optical pixel area.',
            'HeightPlausible': 'Keep iff height q50 >= person_floor_clearance_m and height q95 <= person_top_max_m.',
            'BackgroundRing': 'Square dilation radius person_background_ring_px around the exact mask, clipped to the image, minus the mask. Use original matched valid depth for geometry and nonzero thermal samples for temperature, counted independently.',
            'FitPlane': 'Compute mean of component 3D points and covariance sum((p-mean)*(p-mean)^T)/N. Use the unit eigenvector of the smallest eigenvalue as the plane normal; RMS=sqrt(max(0,minimum eigenvalue)). Compute tolerance=max(person_plane_min_tolerance_m,person_plane_noise_multiplier*s.depth.sigma_coeff_mm_per_m2*0.001*median(z)^2).',
            'SeparateObject': 'Require at least person_min_background_pixels valid ring depth samples. support=number with abs(dot(ring_point-mean,normal))<=tolerance divided by valid ring depth count. Reject when RMS<=tolerance and support>=person_plane_support_fraction. Also reject when median(component_z)-median(ring_z)>=person_max_recess_m.',
            'ThermalContrast': 'Require at least person_min_background_pixels nonzero thermal ring pixels. Keep iff median(component Celsius)-median(ring Celsius)>=person_contrast_min_c. This is an experimental positive-contrast gate; a real person may be cooler than a heated room.',
            'Decision': 'The experimental combined filter requires HeightPlausible AND SeparateObject AND ThermalContrast. Report each gate separately. Unknown background fails the combined filter rather than supplying positive evidence.',
        },
        experimental_alternative='Contrast segmentation replaces the absolute band with abs(T-local mean)>=person_contrast_min_c, using normalized box sums over person_local_contrast_window_px square and at least person_min_background_pixels nonzero temperatures. Retain depth connectivity and area gates. It is measured separately and is not recommended by the observed results.',
        numeric_port_note='Python uses float32 for thermal conversion and matching, and double precision for component reductions and covariance. Existing C# float accumulation yields tiny area differences. Use double accumulators for the new statistics. Projection and blob parity tolerances are 0.3 m with 2 Hz logged pose interpolation and 2 percent area; actual observed errors are reported per event.',
        proposed_config=proposed,
        adoption_limit='Synthetic person retention proves the rule implementations and ablations, not real-world recall. Validate on held-out recordings containing isolated, partially occluded and thermally saturated real survivor views before treating this as a detector fix.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings', nargs='+', default=RUNS)
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/warm_eval')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--event-only', action='store_true')
    args = parser.parse_args(argv)
    if args.stride < 1:
        parser.error('stride must be positive')
    reports, rows, skipped = [], [], []
    started = time.monotonic()
    for name in args.recordings:
        recording = ROOT/'data/recordings'/name
        if not (recording/'meta.json').exists():
            skipped.append(dict(run=name, reason='meta.json absent; recording in progress, not evaluated'))
            print(json.dumps(skipped[-1]), flush=True)
            continue
        report, rr = evaluate_run(recording, args.out/name, stride=args.stride, event_only=args.event_only)
        reports.append(report); rows.extend(rr)
    summary = dict(runs=[r['run'] for r in reports], skipped=skipped, stride=args.stride,
                   frames_evaluated=sum(r['frames_evaluated'] for r in reports), elapsed_s=time.monotonic()-started,
                   proposed=warm.PROPOSED, metrics={key: score(rows, key) for key in KEYS},
                   thermal_eligible_metrics={key: score([row for row in rows if not row['label'] or row.get('thermal_in_image_fraction', 0.) >= .5], key) for key in KEYS},
                   relative_contrast_metrics={key: score([row for report in reports for row in json.loads((args.out/report['run']/'contrast_rows.json').read_text())], key) for key in KEYS},
                   event_metrics_note='Event filters apply to the recorded event-time blob, not a replay of tracking, routing or dwell.',
                   event_filter_metrics={kind: {key: score([dict(label=e['real'], **(e['rules'] or {k: False for k in KEYS})) for r in reports for e in r['events'] if e['kind']==kind], key) for key in KEYS} for kind in ('survivor_candidate', 'survivor_confirmed')},
                   limitations=['Frame rows are correlated, not independent trials. Thresholds are in-sample proposals.',
                                'Visibility uses eval_vision RGB body-volume, depth occlusion and floor-plan line-of-sight labels. Thermal-eligible metrics also require half of the unoccluded projected samples inside the narrower thermal image.',
                                'Event positions include robot localization error. Real means within 1 m of truth, not semantic proof.',
                                'Frame metrics measure detections, not a closed-loop rerun of exploration or confirmation dwell.',
                                'Exact furniture/window identity needs RGB review; geometry descriptions are inferences.'])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'porting_spec.json').write_text(json.dumps(porting_spec(), indent=2)+'\n')
    (args.out/'report.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    import rerun as rr
    rr.init('smoke-rescue-warm-eval'); rr.save(args.out/'warm_eval.rrd')
    for report in reports:
        for name in ('candidates', 'visible_survivors'):
            path = args.out/report['run']/(name+'.png')
            if path.exists():
                rr.log('evaluation/'+report['run']+'/'+name, rr.Image(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)), static=True)
    rr.disconnect()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
