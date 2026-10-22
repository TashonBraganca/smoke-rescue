"""Evaluate the deterministic camera tools of brain.agent.vision against hidden truth.

This is an evaluator, not robot code. Truth comes only through brain.evaluator, which
re-exports the readers in the truth package. vision.py never sees any of it.

Labelling. A survivor is "visible" in a frame when a cloud of sample points filling a
body-sized volume around the truth position projects into the frame with measured
depth that agrees with the point's own range (not occluded by a closer surface).
A warm component from scan_thermal is a true candidate when its centroid falls inside
the projected hull of those hit points at a compatible range. Components that match no
survivor are negatives. Components near a survivor that is only partly visible are
skipped as ambiguous, never counted either way.

Rows have the keys group, label, raw, size and inspect. Positive rows are one per
(frame, visible survivor); a raw hit means some component matched it. Negative rows are
one per unmatched component. Frames are strongly correlated, so the bootstrap
resamples whole groups (recording, robot, survivor or time block), not rows.

Usage: uv run python -m brain.agent.eval_vision [--stride 5] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from brain.agent import vision as V
from brain.evaluator import gt_pose_at, load_floorplan, load_survivors, load_trajectory

ROOT = Path(__file__).resolve().parents[3]
RECORDINGS = ROOT / "data" / "recordings"
# unity_haze_test20 is excluded: its trajectories equal the first 20 s of unity_haze to the last digit, so its
# frames would be near-duplicates and inflate n.
DEFAULT_RUNS = ("unity_haze", "unity_haze_auto_test20")
# Body volume per pose: (radius_m, top_height_m). Lying poses are wide and low.
BODY = {"lying_on_side": (.6, .5), "curled_on_bed": (.6, 1.0),
        "slumped_against_wall": (.5, 1.1), "collapsed_at_desk": (.55, 1.1)}
MATCH_RANGE_M = .8


def _body_points(position, pose):
    radius, top = BODY.get(pose, (.55, 1.0))
    xs = np.linspace(-radius, radius, 5)
    pts = []
    for dx in xs:
        for dz in xs:
            if dx * dx + dz * dz <= radius * radius:
                for h in np.linspace(.1, top, 4):
                    pts.append((position[0] + dx, -h, position[2] + dz))
    return np.array(pts)


def wall_edges(floorplan):
    """Room rectangle edges as (axis, coordinate, lo, hi) plus door openings as (axis, coordinate, lo, hi).

    axis "x" means the edge runs along z at fixed x. Doors in state closed stay walls.
    """
    edges, doors = [], []
    for room in floorplan["rooms"]:
        x0, z0, x1, z1 = room["rect"]
        edges += [("x", x0, z0, z1), ("x", x1, z0, z1), ("z", z0, x0, x1), ("z", z1, x0, x1)]
    for d in floorplan["doors"]:
        if d["state"] == "closed":
            continue
        if d["wall"] == "z":
            doors.append(("z", d["at"][1], d["at"][0] - d["width"] / 2, d["at"][0] + d["width"] / 2))
        else:
            doors.append(("x", d["at"][0], d["at"][1] - d["width"] / 2, d["at"][1] + d["width"] / 2))
    return edges, doors


def line_of_sight(cam_xz, target_xz, floorplan, stop_short_m=.35):
    """True when the straight floor-plan segment crosses no wall except through a door opening."""
    edges, doors = wall_edges(floorplan)
    a = np.asarray(cam_xz, float)
    b = np.asarray(target_xz, float)
    length = np.linalg.norm(b - a)
    if length < 1e-6:
        return True
    b = a + (b - a) * max(0., length - stop_short_m) / length
    for axis, c, lo, hi in edges:
        i = 0 if axis == "x" else 1
        j = 1 - i
        if abs(b[i] - a[i]) < 1e-9 or (a[i] - c) * (b[i] - c) >= 0:
            continue
        s = (c - a[i]) / (b[i] - a[i])
        cross = a[j] + s * (b[j] - a[j])
        if not lo + 1e-6 < cross < hi - 1e-6:
            continue
        if any(d[0] == axis and abs(d[1] - c) < 1e-6 and d[2] <= cross <= d[3] for d in doors):
            continue
        return False
    return True


def survivor_visibility(frame, pose_c2w, survivor, floorplan=None):
    """Project the body volume into the frame, then decide whether the survivor is seen.

    Visible means: the floor-plan line of sight is clear, at least half the sample points
    fall inside the image between 0.3 and 8 m, and at most 40 percent of those points are
    hidden behind a closer valid depth reading (more than 0.6 m nearer than the point).
    Far geometry often has no depth, so invalid depth does not count as occlusion.
    Returns None when nothing projects, else a dict with the thermal-image hull points.
    """
    pts = _body_points(survivor["position_house"], survivor.get("pose"))
    cam = (pose_c2w[:3, :3].T @ (pts - pose_c2w[:3, 3]).T).T
    z = cam[:, 2]
    in_range = (z > .3) & (z < 8.)
    if not in_range.any():
        return None
    k, depth = frame.K, frame.depth_m
    zs = np.where(in_range, z, 1.)
    u = np.rint(k[0, 0] * cam[:, 0] / zs + k[0, 2]).astype(int)
    v = np.rint(k[1, 1] * cam[:, 1] / zs + k[1, 2]).astype(int)
    inside = in_range & (u >= 0) & (u < depth.shape[1]) & (v >= 0) & (v < depth.shape[0])
    if inside.sum() < 3:
        return None
    d = np.full(len(pts), np.nan)
    d[inside] = depth[v[inside], u[inside]]
    occluded = inside & np.isfinite(d) & (d > 0) & (d < z - .6)
    seen = inside & ~occluded
    t_cam = (np.asarray(frame.rgb_to_thermal) @ np.vstack([cam.T, np.ones(len(cam))]))[:3].T
    kt = frame.thermal_K
    zt = np.maximum(t_cam[:, 2], 1e-6)
    ut = kt[0, 0] * t_cam[:, 0] / zt + kt[0, 2]
    vt = kt[1, 1] * t_cam[:, 1] / zt + kt[1, 2]
    clear = True
    if floorplan is not None:
        clear = line_of_sight((pose_c2w[0, 3], pose_c2w[2, 3]), survivor["position_house"][::2], floorplan)
    in_frac = float(inside.sum() / len(pts))
    occ_frac = float(occluded.sum() / inside.sum())
    return {"visible": bool(clear and in_frac >= .5 and occ_frac <= .4 and seen.sum() >= 3),
            "line_of_sight": bool(clear), "in_image_fraction": in_frac, "occluded_fraction": occ_frac,
            "range_m": float(np.median(z[seen])) if seen.any() else float(np.median(z[inside])),
            "uv_thermal": np.column_stack([ut[seen], vt[seen]]) if seen.any() else np.zeros((0, 2))}


def match_component(comp, vis):
    """True when the component centroid lies in the hit hull (grown by 4 px) at a compatible range."""
    if vis is None or len(vis["uv_thermal"]) < 3:
        return False
    x, y, w, h = comp["bbox"]
    cx, cy = x + w / 2, y + h / 2
    hull = cv2.convexHull(vis["uv_thermal"].astype(np.float32))
    if cv2.pointPolygonTest(hull, (float(cx), float(cy)), True) < -4:
        return False
    return comp["range_m"] is None or abs(comp["range_m"] - vis["range_m"]) <= MATCH_RANGE_M


def inspection_selection(result, frame, pose, survivors, visibility):
    """Label the object the tool returned, independently of the requested warm patch.

    A colour fallback can select a different body. Counting its person verdict
    against the original wall patch creates a false positive in the evaluator.
    Preserve that change of selection in each negative row.
    """
    candidate = result.get("candidate") or {}
    if "bbox" in candidate:
        matches = [s["id"] for s in survivors if match_component(candidate,visibility[s["id"]])]
    else:
        hits = result.get("cues",{}).get("rgb_dnn",{}).get("hits",[])
        if candidate.get("bbox_rgb"):
            hits = [h for h in hits if h["bbox_rgb"]==candidate["bbox_rgb"]]
        elif candidate.get("source")=="rgb_dnn":
            hits = [h for h in hits if h.get("bearing_deg")==candidate.get("bearing_deg") and h.get("range_m")==candidate.get("range_m")]
        matches = []
        for survivor in survivors:
            vis = visibility[survivor["id"]]
            if vis is None or not vis.get("line_of_sight",False):
                continue
            points = _body_points(survivor["position_house"],survivor.get("pose"))
            cam = (pose[:3,:3].T @ (points-pose[:3,3]).T).T
            cam = cam[cam[:,2]>0]
            uv = cam @ frame.K.T
            uv = uv[:,:2]/uv[:,2,None]
            if any(match_component({"bbox":hit["bbox_rgb"],"range_m":None},vis | {"uv_thermal":uv}) for hit in hits):
                matches.append(survivor["id"])
    visible = [sid for sid in matches if visibility[sid]["visible"]]
    return (visible[0] if visible else None), bool(matches and not visible)


def _rate(k, n):
    return None if n == 0 else k / n


def _scores(rows, key):
    tp = sum(1 for r in rows if r["label"] and r[key])
    fp = sum(1 for r in rows if not r["label"] and r[key])
    pos = sum(1 for r in rows if r["label"])
    return tp, fp, pos


def summarize(rows, n_boot=2000, seed=0):
    """Precision and recall for raw scan, size gate and inspect_candidate with cluster bootstrap CIs."""
    out = {"n_rows": len(rows), "n_positive": sum(1 for r in rows if r["label"]),
           "n_negative": sum(1 for r in rows if not r["label"]),
           "n_groups": len({r["group"] for r in rows})}
    keys = [k for k in ("raw", "size", "inspect") if rows and k in rows[0]]
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    names = sorted(groups)
    rng = np.random.default_rng(seed)
    boots = {k: {"precision": [], "recall": []} for k in keys}
    for _ in range(n_boot if len(names) > 1 else 0):
        pick = rng.integers(0, len(names), len(names))
        sample = [r for i in pick for r in groups[names[i]]]
        for k in keys:
            tp, fp, pos = _scores(sample, k)
            p, rc = _rate(tp, tp + fp), _rate(tp, pos)
            if p is not None:
                boots[k]["precision"].append(p)
            if rc is not None:
                boots[k]["recall"].append(rc)
    for k in keys:
        tp, fp, pos = _scores(rows, k)
        entry = {"tp": tp, "fp": fp, "fn": pos - tp, "predicted_positive": tp + fp,
                 "precision": _rate(tp, tp + fp), "recall": _rate(tp, pos)}
        for m in ("precision", "recall"):
            b = boots[k][m]
            entry[m + "_ci95"] = ([float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
                                  if len(b) >= 100 else None)
        out[k] = entry
    return out


def label_frame(frame, pose, survivors, floorplan, run, robot, t, out_dir):
    """Run scan_thermal and inspect_candidate on one frame and return labelled rows."""
    scan = V.scan_thermal(frame)
    visibility = {s["id"]: survivor_visibility(frame, pose, s, floorplan) for s in survivors}
    comps = []
    for comp in scan["people"]:
        call = V.inspect_candidate(frame, comp["bearing_deg"], comp["range_m"] or 3., output_dir=out_dir)
        same = call["candidate"] is not None and call["candidate"].get("bbox") == comp["bbox"]
        selected, ambiguous_selection = inspection_selection(call,frame,pose,survivors,visibility)
        comps.append({"comp": comp, "verdict": call["verdict"], "same_component": same,
                      "selected_person":selected,"ambiguous_selection":ambiguous_selection,
                      "size": bool(comp.get("size_plausible"))})
    rows, block = [], int(t // 10)
    hit_by = {s["id"]: [] for s in survivors}
    negatives = []
    for c in comps:
        owner, ambiguous = None, False
        for s in survivors:
            vis = visibility[s["id"]]
            if vis is not None and match_component(c["comp"], vis):
                if vis["visible"]:
                    owner = s["id"]
                else:
                    ambiguous = True
        if owner:
            hit_by[owner].append(c)
        elif not ambiguous:
            negatives.append(c)
    base = {"run": run, "robot": robot, "t": float(t)}
    for s in survivors:
        vis = visibility[s["id"]]
        if vis is None or not vis["visible"]:
            continue
        matched = hit_by[s["id"]]
        uv = vis["uv_thermal"]
        inside = uv[(uv[:, 0] >= 0) & (uv[:, 0] < frame.thermal.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < frame.thermal.shape[0])].astype(int)
        apparent = float(np.percentile(frame.thermal[inside[:, 1], inside[:, 0]], 90)) / 10 - 273.15 if len(inside) else None
        bearing = float(np.degrees(np.arctan2(np.median(uv[:, 0]) - frame.thermal_K[0, 2], frame.thermal_K[0, 0])))
        aimed = V.inspect_candidate(frame, bearing, vis["range_m"], output_dir=out_dir)
        rows.append({**base, "apparent_p90_c": apparent, "aimed_verdict": aimed["verdict"],
                     "aimed_reason": aimed["reasons"][:1], "group": f"{run}/{robot}/{s['id']}/{block}", "label": True, "survivor": s["id"],
                     "raw": bool(matched), "size": any(c["size"] for c in matched),
                     "inspect": any(c["verdict"] == "person" and
                                    (c["same_component"] or c["selected_person"]==s["id"]) for c in matched) or
                                any(c["verdict"]=="person" and c["selected_person"]==s["id"] for c in comps),
                     "verdicts": [c["verdict"] for c in matched],
                     "bboxes": [c["comp"]["bbox"] for c in matched], "range_m": vis["range_m"]})
    for c in negatives:
        rows.append({**base, "group": f"{run}/{robot}/neg/{block}", "label": False, "survivor": None,
                     "raw": True, "size": c["size"], "inspect": c["verdict"] == "person" and not c["selected_person"] and not c["ambiguous_selection"],
                     "returned_person":c["verdict"]=="person","selected_person":c["selected_person"],
                     "ambiguous_selection":c["ambiguous_selection"],
                     "verdicts": [c["verdict"]], "bboxes": [c["comp"]["bbox"]],
                     "range_m": c["comp"]["range_m"], "mean_c": c["comp"]["mean_c"],
                     "same_component": c["same_component"]})
    return rows


def collect(runs, stride, out_dir):
    rows = []
    for run in runs:
        rec = RECORDINGS / run
        survivors, plan = load_survivors(rec), load_floorplan(rec)
        for robot in sorted(p.name for p in rec.glob("robot_*")):
            traj = load_trajectory(rec, robot)
            stamps = np.loadtxt(rec / robot / "thermal.txt", usecols=0)
            for t in stamps[::stride]:
                frame = V.load_frame(rec / robot, float(t))
                rows += label_frame(frame, gt_pose_at(traj, float(t)), survivors, plan, run, robot, float(t), out_dir)
    return rows


def door_views(frame, pose, plan, max_range=5., max_bearing=20., max_incidence=15.):
    """Doors seen nearly head-on with a clear line of sight: the cases where the measured width is the true width."""
    views = []
    for d in plan["doors"]:
        if d["state"] == "closed" or d["between"][0] == d["between"][1]:
            continue
        centre = np.array([d["at"][0], -1.0, d["at"][1]])
        normal = np.array([0., 0., 1.]) if d["wall"] == "z" else np.array([1., 0., 0.])
        cam = pose[:3, :3].T @ (centre - pose[:3, 3])
        if not 1. <= cam[2] <= max_range:
            continue
        bearing = float(np.degrees(np.arctan2(cam[0], cam[2])))
        n_cam = pose[:3, :3].T @ normal
        incidence = float(np.degrees(np.arccos(min(1., abs(n_cam @ cam) / np.linalg.norm(cam)))))
        if abs(bearing) > max_bearing or incidence > max_incidence:
            continue
        if not line_of_sight((pose[0, 3], pose[2, 3]), (d["at"][0], d["at"][1]), plan, stop_short_m=.3):
            continue
        views.append({"door": d["id"], "true_width_m": float(d["width"]), "bearing_deg": bearing,
                      "range_m": float(cam[2]), "incidence_deg": incidence,
                      "optical_y_m": float((pose[:3, :3].T @ (np.array([d["at"][0], -.9, d["at"][1]]) - pose[:3, 3]))[1])})
    # A farther door seen through a nearer opening is not what the tool measures at that bearing: keep the nearest.
    views.sort(key=lambda w: w["range_m"])
    kept = []
    for w in views:
        if all(abs(w["bearing_deg"] - k["bearing_deg"]) > 4. for k in kept):
            kept.append(w)
    for w in kept:
        w["n_aligned"] = sum(1 for o in views if abs(o["bearing_deg"] - w["bearing_deg"]) <= 4.)
    return kept


def evaluate_openings(runs, stride):
    out = []
    for run in runs:
        rec = RECORDINGS / run
        plan = load_floorplan(rec)
        for robot in sorted(p.name for p in rec.glob("robot_*")):
            traj = load_trajectory(rec, robot)
            stamps = np.loadtxt(rec / robot / "thermal.txt", usecols=0)
            for t in stamps[::stride]:
                frame = V.load_frame(rec / robot, float(t))
                for view in door_views(frame, gt_pose_at(traj, float(t)), plan):
                    res = V.measure_opening(frame, view["bearing_deg"], body_height_m=view["optical_y_m"])
                    out.append({**view, "run": run, "robot": robot, "t": float(t),
                                "measured_width_m": res["clear_width_m"], "passable": res["passable"]})
    return out


def strata(rows):
    """Recall of each method on slices of the visible survivors, plus the raw false alarm load."""
    pos = [r for r in rows if r["label"]]
    cuts = {"range_lt_3m": [r for r in pos if r["range_m"] < 3.],
            "range_3_to_5m": [r for r in pos if 3. <= r["range_m"] < 5.],
            "range_ge_5m": [r for r in pos if r["range_m"] >= 5.],
            "apparent_temp_in_28_40c": [r for r in pos if r["apparent_p90_c"] is not None and 28. <= r["apparent_p90_c"] <= 40.],
            "apparent_temp_outside_band": [r for r in pos if r["apparent_p90_c"] is None or not 28. <= r["apparent_p90_c"] <= 40.]}
    out = {}
    for name, sub in cuts.items():
        out[name] = {"n": len(sub), "raw_hits": sum(r["raw"] for r in sub), "size_hits": sum(r["size"] for r in sub),
                     "inspect_person": sum(r["inspect"] for r in sub),
                     "aimed_at_truth": {v: sum(1 for r in sub if r["aimed_verdict"] == v)
                                        for v in ("person", "not_person", "uncertain")}}
    neg = [r for r in rows if not r["label"]]
    out["negatives"] = {"n": len(neg), "verdict_counts": {v: sum(1 for r in neg if r["verdicts"][0] == v)
                                                          for v in ("person", "not_person", "uncertain")},
                        "size_gate_passes": sum(r["size"] for r in neg)}
    out["survivors_in_positives"] = sorted({f"{r['run']}/{r['survivor']}" for r in pos})
    return out


def summarize_openings(rows, n_boot=2000, seed=0):
    """Width error of measure_opening on head-on doors; ambiguous views (nested openings) are excluded."""
    clean = [r for r in rows if r["n_aligned"] == 1]
    measured = [r for r in clean if r["measured_width_m"] is not None]
    err = np.array([abs(r["measured_width_m"] - r["true_width_m"]) for r in measured])
    out = {"views_total": len(rows), "views_unambiguous": len(clean), "measured": len(measured),
           "coverage": _rate(len(measured), len(clean)),
           "groups": sorted({f"{r['run']}/{r['robot']}/{r['door']}" for r in measured})}
    if len(err):
        groups = {}
        for r, e in zip(measured, err):
            groups.setdefault(f"{r['run']}/{r['robot']}/{r['door']}", []).append(e)
        names = sorted(groups)
        rng = np.random.default_rng(seed)
        med, w5 = [], []
        for _ in range(n_boot if len(names) > 1 else 0):
            pick = rng.integers(0, len(names), len(names))
            sample = np.concatenate([groups[names[i]] for i in pick])
            med.append(float(np.median(sample)))
            w5.append(float(np.mean(sample <= .05)))
        out.update({"median_abs_error_m": float(np.median(err)), "p90_abs_error_m": float(np.percentile(err, 90)),
                    "within_5cm": float(np.mean(err <= .05)), "within_10cm": float(np.mean(err <= .10)),
                    "median_abs_error_ci95": [float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))] if med else None,
                    "within_5cm_ci95": [float(np.percentile(w5, 2.5)), float(np.percentile(w5, 97.5))] if w5 else None,
                    "by_door": {k: {"n": len(v), "true_width_m": next(r["true_width_m"] for r in measured
                                    if f"{r['run']}/{r['robot']}/{r['door']}" == k),
                                    "median_abs_error_m": float(np.median(v))} for k, v in sorted(groups.items())}})
    return out


def timing(run="unity_haze", robot="robot_A", frames=60, warmup=5):
    """Per-call wall time on real frames, single thread, first calls discarded as warmup."""
    import tempfile
    rec = RECORDINGS / run / robot
    stamps = np.loadtxt(rec / "thermal.txt", usecols=0)
    picks = [float(t) for t in stamps[:: max(1, len(stamps) // frames)][:frames]]
    out_dir = tempfile.mkdtemp()
    samples = {k: [] for k in ("load_frame", "scan_thermal", "inspect_candidate", "assess_visibility",
                               "measure_opening", "track_fire_spread_3_frames")}
    raw_load = V.load_frame.__wrapped__

    def clock(name, fn, *a, **k):
        t0 = time.perf_counter()
        out = fn(*a, **k)
        samples[name].append((time.perf_counter() - t0) * 1000.)
        return out

    for t in picks:
        frame = clock("load_frame", raw_load, rec, t)
        scan = clock("scan_thermal", V.scan_thermal, frame)
        comp = max(scan["people"], key=lambda c: c["pixels"], default=None)
        bearing, rng_m = (comp["bearing_deg"], comp["range_m"] or 3.) if comp else (0., 3.)
        clock("inspect_candidate", V.inspect_candidate, frame, bearing, rng_m, output_dir=out_dir)
        clock("assess_visibility", V.assess_visibility, frame)
        clock("measure_opening", V.measure_opening, frame, 0.)
    trio = [V.load_frame(rec, t) for t in picks[:3]]
    for _ in picks:
        clock("track_fire_spread_3_frames", V.track_fire_spread, trio)
    out = {}
    for name, xs in samples.items():
        xs = xs[warmup:]
        out[name] = {"n": len(xs), "median_ms": round(statistics.median(xs), 2),
                     "p95_ms": round(float(np.percentile(xs, 95)), 2)}
    return out


def _draw(frame, rows_for_frame, hull_points=None, label=""):
    c = frame.thermal.astype(float) / 10 - 273.15
    thermal = cv2.applyColorMap(np.clip((c - 20) / 25 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    thermal = cv2.resize(thermal, (640, 512), interpolation=cv2.INTER_NEAREST)
    colour = {"person": (0, 255, 0), "not_person": (255, 160, 0), "uncertain": (0, 220, 255)}
    for verdict, bbox in rows_for_frame:
        x, y, w, h = bbox
        cv2.rectangle(thermal, (2 * x, 2 * y), (2 * (x + w), 2 * (y + h)), colour[verdict], 2)
        cv2.putText(thermal, verdict, (2 * x, max(12, 2 * y - 4)), cv2.FONT_HERSHEY_SIMPLEX, .5, colour[verdict], 1)
    if hull_points is not None and len(hull_points) >= 3:
        hull = cv2.convexHull(hull_points.astype(np.float32))
        cv2.polylines(thermal, [(hull * 2).astype(np.int32)], True, (255, 255, 255), 1)
    rgb = cv2.resize(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), (640, 512))
    image = np.hstack([thermal, rgb])
    cv2.putText(image, label, (8, 502), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
    cv2.putText(image, label, (8, 502), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1)
    return image


def write_examples(rows, out_dir):
    """Annotated PNGs. Left thermal (20 to 45 C), right RGB. Boxes are warm components coloured by verdict;
    the white outline is the projected body volume of a visible survivor from truth."""
    out_dir.mkdir(parents=True, exist_ok=True)
    picked = {}
    neg = [r for r in rows if not r["label"]]
    pos = [r for r in rows if r["label"]]
    picked["true_negative"] = next((r for r in neg if r["verdicts"][0] == "not_person" and r["range_m"]
                                    and r["range_m"] < 4 and r["bboxes"][0][2] * r["bboxes"][0][3] > 400), neg[0] if neg else None)
    picked["false_negative_raw_hit_rejected"] = next((r for r in pos if r["raw"] and not r["inspect"]), None)
    picked["false_negative_missed_by_scan"] = next((r for r in pos if not r["raw"] and r["range_m"] < 3.5), None)
    picked["false_positive"] = next((r for r in neg if r["inspect"]), None)
    picked["true_positive"] = next((r for r in pos if r["inspect"]), None)
    written = {}
    for kind, row in picked.items():
        if row is None:
            written[kind] = None
            continue
        rec = RECORDINGS / row["run"]
        frame = V.load_frame(rec / row["robot"], row["t"])
        plan, survivors = load_floorplan(rec), load_survivors(rec)
        pose = gt_pose_at(load_trajectory(rec, row["robot"]), row["t"])
        hull = None
        for sv in survivors:
            vis = survivor_visibility(frame, pose, sv, plan)
            if vis and vis["visible"] and (row["label"] and sv["id"] == row["survivor"]):
                hull = vis["uv_thermal"]
        scan = V.scan_thermal(frame)
        drawn = []
        import tempfile
        for comp in scan["people"]:
            if comp["pixels"] >= 30 or list(comp["bbox"]) in row["bboxes"]:
                verdict = V.inspect_candidate(frame, comp["bearing_deg"], comp["range_m"] or 3.,
                                              output_dir=tempfile.mkdtemp())["verdict"]
                drawn.append((verdict, comp["bbox"]))
        label = f"{kind}  {row['run']} {row['robot']} t={row['t']:.2f}s"
        path = out_dir / f"{kind}.png"
        cv2.imwrite(str(path), _draw(frame, drawn, hull, label))
        written[kind] = {"path": str(path), "run": row["run"], "robot": row["robot"], "t": row["t"],
                         "survivor": row.get("survivor"), "verdicts": row["verdicts"]}
    return written


def synthetic_examples(out_dir):
    """Real data has no confirmed person, so show the two classifier outcomes on the synthetic unit-test scenes."""
    import tempfile
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for kind, scene in (("synthetic_true_positive", "body"), ("synthetic_true_negative_wall", "wall")):
        frame = synthetic_scene(scene)
        comp = V.scan_thermal(frame)["people"][0]
        res = V.inspect_candidate(frame, comp["bearing_deg"], comp["range_m"], output_dir=tempfile.mkdtemp())
        path = out_dir / f"{kind}.png"
        cv2.imwrite(str(path), _draw(frame, [(res["verdict"], comp["bbox"])], None, f"{kind} (synthetic) verdict={res['verdict']}"))
        written[kind] = {"path": str(path), "verdict": res["verdict"]}
    return written


def synthetic_scene(kind="body", t=0.):
    """Warm patch on a 3 m wall: a bulging body-sized blob, a flat wall patch, or an opening."""
    k = np.array([[120., 0, 79.5], [0, 120., 59.5], [0, 0, 1.]])
    depth = np.full((120, 160), 3., np.float32)
    thermal = np.full((120, 160), 2932, np.uint16)
    yy, xx = np.mgrid[30:90, 69:91]
    thermal[30:90, 69:91] = np.round((303.15 + 5 * (yy - 30) / 60) * 10).astype(np.uint16)
    if kind != "wall":
        depth[30:90, 69:91] -= .25 * np.sin((xx - 68) / 24 * np.pi) * np.sin((yy - 29) / 62 * np.pi)
    if kind == "window":
        depth[30:90, 69:91] += 2.
    rgb = np.full((120, 160, 3), 70, np.uint8)
    rgb[30:90, 69:91] = 160
    return V.Frame(t=t, rgb=rgb, depth_m=depth, thermal=thermal, K=k, thermal_K=k,
                   rgb_to_thermal=np.eye(4), stamps={"rgb": t, "depth": t, "thermal": t})


LIMITS = [
    "Few survivors are ever in view: positives come from two survivors in one house, and consecutive frames "
    "of one pass are highly correlated, so intervals are wide and the result is directional only.",
    "Survivors are labelled by projecting a body volume around the true position with true camera poses and a "
    "floor-plan line of sight. The volume is a disc, not the true pose, so label boxes are generous.",
    "In the recorded thermal frames a survivor is rarely warmer than its surroundings (smoke attenuates the "
    "apparent temperature with range and heated rooms sit near body temperature), so the 28 to 40 C band "
    "rarely isolates a body for any method.",
    "Cue thresholds were adjusted after looking at these same frames (in-sample), so the inspect numbers are "
    "optimistic about precision and cannot show generalisation.",
    "Negatives are warm non-person regions found in the data. The broken-window warm air case from an earlier "
    "run is not in these recordings, so it is covered only by the synthetic window test.",
    "unity_haze_test20 is skipped because it repeats the first 20 s of unity_haze.",
]


def main(argv=None):
    import tempfile
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=5, help="evaluate every Nth thermal frame")
    ap.add_argument("--runs", nargs="*", default=list(DEFAULT_RUNS))
    ap.add_argument("--out", default=str(ROOT / "data" / "runs" / "agent_vision_eval"))
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = tempfile.mkdtemp()
    started = time.time()
    rows = collect(args.runs, args.stride, scratch)
    opening_rows = evaluate_openings(args.runs, max(1, args.stride - 2))
    report = {
        "opencv": cv2.__version__, "runs": args.runs, "stride_frames": args.stride,
        "frames_evaluated": len({(r["run"], r["robot"], r["t"]) for r in rows}),
        "labelling": {"match_range_m": MATCH_RANGE_M, "body_volumes": BODY},
        "people": summarize(rows, args.boot), "people_strata": strata(rows),
        "openings": summarize_openings(opening_rows, args.boot),
        "timing_ms": timing(), "examples": {}, "limitations": LIMITS,
    }
    report["examples"] = {**write_examples(rows, out / "examples"), **synthetic_examples(out / "examples")}
    report["elapsed_s"] = round(time.time() - started, 1)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (out / "rows.json").write_text(json.dumps(rows, allow_nan=False))
    print(json.dumps({k: report[k] for k in ("people", "people_strata", "openings", "timing_ms")}, indent=2))
    return report


if __name__ == "__main__":
    main()
