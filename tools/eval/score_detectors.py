"""Evaluator-only scoring of saved person-detector boxes against projected oracle footprints.

Inputs
  raw    modeleval2_raw.json: {model: {"rec|robot|t": [[x0,y0,x1,y1,conf], ...]}} (boxes kept at conf >= 0.1)
  views  views2.json: per colour frame, every survivor whose footprint projects into the image, with the
         projected footprint box, in-image fraction, depth-occlusion fraction, range and pose. It was built from
         oracle/survivors.json and oracle/*_groundtruth.txt (camera optical pose) by visib2.py with a disc of
         radius 0.7 m and pose specific heights, the same idea as brain.agent.eval_vision._body_points.

A visible view is a (frame, person) pair with in_image >= 0.8, occluded <= 0.25, range <= 6.5 m. Everything here is a
proxy: a box that overlaps a projected footprint is not a confirmed person, and the footprint box is wider than the body.

Run from brain: UV_NO_SYNC=1 uv run python ../tools/eval/score_detectors.py --out scores.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path

import numpy as np

RAW = '/Volumes/Tashon/opencv/codex_briefs/research_23oct/modeleval2_raw.json'
VIEWS = '/tmp/scratch'
OUT = '/tmp/scratch'
CONFS = (0.25, 0.35, 0.5)
LIVE_WINDOW_S = 3.0
LIVE_SIGHTINGS = 2
ID_FRAC = 0.3  # a footprint with at least this much of its points in the image can own a box


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for k successes in n trials. (nan, nan) when n is 0."""
    if n <= 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {'k': int(k), 'n': int(n), 'rate': (k / n if n else None), 'wilson95': [None if math.isnan(lo) else lo, None if math.isnan(hi) else hi]}


def iou_ioa(det, gt):
    """IoU and intersection over detection area for xyxy boxes."""
    ix = max(0.0, min(det[2], gt[2]) - max(det[0], gt[0]))
    iy = max(0.0, min(det[3], gt[3]) - max(det[1], gt[1]))
    inter = ix * iy
    ad = max(0.0, det[2] - det[0]) * max(0.0, det[3] - det[1])
    ag = max(0.0, gt[2] - gt[0]) * max(0.0, gt[3] - gt[1])
    union = ad + ag - inter
    return (inter / union if union > 0 else 0.0), (inter / ad if ad > 0 else 0.0)


def is_match(det, gt, mode='primary') -> bool:
    """primary: IoU >= 0.3 or at least half the detection inside the footprint box. strict: IoU >= 0.3 only."""
    iou, ioa = iou_ioa(det, gt)
    return iou >= 0.3 or (mode == 'primary' and ioa >= 0.5)


def key_of(e) -> str:
    return f"{e['rec']}|{e['robot']}|{e['t']}"


def is_visible(q) -> bool:
    return q['in_image'] >= 0.8 and q['occluded'] <= 0.25 and q['range'] <= 6.5


def one_hz_subsample(frames):
    """Keep the first frame in each whole second per (rec, robot). The raw file has no every-15th-frame series
    (its frames are 0.8 s apart where contiguous), so this is the closest available stand-in for the robot's 1 Hz."""
    seen, out = set(), []
    for e in sorted(frames, key=lambda e: (e['rec'], e['robot'], e['t'])):
        b = (e['rec'], e['robot'], math.floor(e['t']))
        if b not in seen:
            seen.add(b)
            out.append(e)
    return out


def live_rule(sight_times, first_visible_t, window=LIVE_WINDOW_S, need=LIVE_SIGHTINGS):
    """sight_times: sorted times of frames where the model sighted the person. Returns the time of the frame that
    completes `need` sightings inside `window` seconds, or None."""
    s = sorted(sight_times)
    for i in range(need - 1, len(s)):
        if s[i] - s[i - need + 1] <= window:
            return s[i]
    return None


def detections(raw_model, e, conf):
    return [d for d in raw_model.get(key_of(e), []) if d[4] >= conf]


def score_model(raw_model, frames, conf, mode='primary', boot=0, rng=None):
    pos = [e for e in frames if any(is_visible(q) for q in e['persons'])]
    neg = [e for e in frames if not e['persons']]

    def person_hit(dets, q):
        return any(is_match(d, q['box'], mode) for d in dets)

    views = []  # (rec, robot, pid, pose, t, hit)
    fire_pos = tp_frames = boxes_pos = tp_boxes = 0
    for e in pos:
        dets = detections(raw_model, e, conf)
        owners = [q for q in e['persons'] if q['in_image'] >= ID_FRAC]
        boxes_pos += len(dets)
        tp_boxes += sum(1 for d in dets if any(is_match(d, q['box'], mode) for q in owners))
        if dets:
            fire_pos += 1
            tp_frames += any(any(is_match(d, q['box'], mode) for q in owners) for d in dets)
        for q in e['persons']:
            if is_visible(q):
                views.append((e['rec'], e['robot'], q['pid'], q['pose'], e['t'], person_hit(dets, q)))
    fire_neg = sum(1 for e in neg if detections(raw_model, e, conf))
    boxes_neg = sum(len(detections(raw_model, e, conf)) for e in neg)

    out = {
        'recall_per_view': rate(sum(v[5] for v in views), len(views)),
        'precision_per_frame': rate(tp_frames, fire_pos + fire_neg),
        'precision_per_box': rate(tp_boxes, boxes_pos + boxes_neg),
        'neg_frames_with_any_box': rate(fire_neg, len(neg)),
        'false_positives_per_negative_frame': {'boxes': boxes_neg, 'n': len(neg), 'mean': boxes_neg / len(neg) if neg else None},
    }
    inst = collections.defaultdict(list)
    for v in views:
        inst[v[:3]].append(v[5])
    out['recall_per_instance_any_view'] = rate(sum(any(h) for h in inst.values()), len(inst))
    out['recall_per_instance_mean_view_fraction'] = {'n': len(inst), 'mean': float(np.mean([np.mean(h) for h in inst.values()])) if inst else None}
    by_pose = collections.defaultdict(list)
    for v in views:
        by_pose[v[3]].append(v[5])
    out['recall_per_view_by_pose'] = {p: rate(sum(h), len(h)) for p, h in sorted(by_pose.items())}
    pose_inst = collections.defaultdict(dict)
    for k, h in inst.items():
        pose_inst[next(v[3] for v in views if v[:3] == k)][k] = any(h)
    out['recall_per_instance_by_pose'] = {p: rate(sum(d.values()), len(d)) for p, d in sorted(pose_inst.items())}
    if boot and inst:
        keys = list(inst)
        stats = []
        for _ in range(boot):
            samp = [inst[keys[rng.randrange(len(keys))]] for _ in keys]
            tot = sum(len(h) for h in samp)
            stats.append(sum(sum(h) for h in samp) / tot)
        out['recall_per_view']['cluster_bootstrap95_by_instance'] = [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]

    # 1 Hz series
    f1 = one_hz_subsample(frames)
    pos1 = [e for e in f1 if any(is_visible(q) for q in e['persons'])]
    v1 = [(e, q) for e in pos1 for q in e['persons'] if is_visible(q)]
    h1 = [person_hit(detections(raw_model, e, conf), q) for e, q in v1]
    out['recall_per_view_1hz'] = rate(sum(h1), len(h1))
    i1 = collections.defaultdict(list)
    for (e, q), h in zip(v1, h1):
        i1[(e['rec'], e['robot'], q['pid'])].append(h)
    out['recall_per_instance_1hz_any_view'] = rate(sum(any(h) for h in i1.values()), len(i1))
    neg1 = [e for e in f1 if not e['persons']]
    b1 = sum(len(detections(raw_model, e, conf)) for e in neg1)
    out['false_positives_per_negative_frame_1hz'] = {'boxes': b1, 'n': len(neg1), 'mean': b1 / len(neg1) if neg1 else None}

    # live rule on the 1 Hz series
    sight = collections.defaultdict(list)
    firstvis = {}
    for e in f1:
        dets = detections(raw_model, e, conf)
        for q in e['persons']:
            k = (e['rec'], e['robot'], q['pid'])
            if is_visible(q):
                firstvis.setdefault(k, e['t'])
            if q['in_image'] >= ID_FRAC and person_hit(dets, q):
                sight[k].append(e['t'])
    delays, made, single = [], 0, 0
    for k, t0 in firstvis.items():
        ts = [t for t in sight.get(k, []) if t >= t0]
        t_c = live_rule(ts, t0)
        if t_c is not None:
            made += 1
            delays.append(t_c - t0)
        elif ts:
            single += 1
    out['live_rule'] = {
        'candidate_made': rate(made, len(firstvis)),
        'seen_once_but_no_candidate': single,
        'never_sighted': len(firstvis) - made - single,
        'delay_s_after_first_visible_1hz_frame': {
            'n': len(delays),
            'median': float(np.median(delays)) if delays else None,
            'p25': float(np.percentile(delays, 25)) if delays else None,
            'p75': float(np.percentile(delays, 75)) if delays else None,
            'max': float(max(delays)) if delays else None,
        },
    }
    return out


def load_frames(raw, views):
    keys = set(next(iter(raw.values())))
    frames = [e for e in views if key_of(e) in keys]
    return frames, keys


def checks(raw, views, frames):
    names = list(raw)
    nf = {m: len(v) for m, v in raw.items()}
    keysets = [set(v) for v in raw.values()]
    pos = [e for e in frames if any(is_visible(q) for q in e['persons'])]
    neg = [e for e in frames if not e['persons']]
    vv = [(e, q) for e in pos for q in e['persons'] if is_visible(q)]
    gaps = []
    g = collections.defaultdict(list)
    for e in pos:
        g[(e['rec'], e['robot'])].append(e['t'])
    for ts in g.values():
        ts.sort()
        gaps += [b - a for a, b in zip(ts, ts[1:])]
    return {
        'models': names,
        'frames_per_model': nf,
        'all_models_same_keys': all(k == keysets[0] for k in keysets),
        'raw_keys_found_in_views': len(frames), 'raw_keys': len(keysets[0]),
        'positive_frames': len(pos), 'negative_frames_without_any_projected_person': len(neg),
        'other_frames': len(frames) - len(pos) - len(neg),
        'visible_views': len(vv),
        'person_instances': len({(e['rec'], e['robot'], q['pid']) for e, q in vv}),
        'recordings': len({e['rec'] for e in frames}),
        'frames_with_at_least_one_box_at_0.1': {m: sum(1 for b in raw[m].values() if b) for m in names},
        'median_gap_s_between_positive_frames_same_run': float(np.median(gaps)) if gaps else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=RAW)
    ap.add_argument('--views', default=VIEWS)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args()
    raw = json.load(open(a.raw))
    views = json.load(open(a.views))
    frames, _ = load_frames(raw, views)
    chk = checks(raw, views, frames)
    assert len(raw) >= 7 and chk['raw_keys'] == 1212 and chk['raw_keys_found_in_views'] == 1212, chk
    rng = random.Random(0)
    res = {'checks': chk, 'confs': list(CONFS), 'match': 'primary: IoU>=0.3 or >=50% of the box inside the projected footprint; strict: IoU>=0.3',
           'scores': {}, 'scores_strict_match': {}}
    for m in raw:
        res['scores'][m] = {str(c): score_model(raw[m], frames, c, 'primary', a.boot, rng) for c in CONFS}
        res['scores_strict_match'][m] = {str(c): score_model(raw[m], frames, c, 'strict') for c in CONFS}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps(chk, indent=1))
    for m in raw:
        for c in CONFS:
            s = res['scores'][m][str(c)]
            print(f"{m:16s} conf {c}: view recall {s['recall_per_view']['rate']:.3f} (n={s['recall_per_view']['n']}) inst {s['recall_per_instance_any_view']['rate']:.3f} "
                  f"FP/neg {s['false_positives_per_negative_frame']['mean']:.3f} prec/frame {s['precision_per_frame']['rate']:.3f} 1Hz {s['recall_per_view_1hz']['rate']:.3f} "
                  f"live {s['live_rule']['candidate_made']['k']}/{s['live_rule']['candidate_made']['n']}")


if __name__ == '__main__':
    main()
