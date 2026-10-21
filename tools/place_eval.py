"""Evaluate visual place recognition (brain/brain/agent/place.py) against ground truth.

Evaluation side: this script reads data/recordings/*/oracle through brain.oracle. The service never does.

Keyframes are sampled every sample_step_s from robot_A and robot_B of three recordings. For each pair of keyframes
the partner payload is built from one and a one-keyframe database from the other, the real matcher runs, and the
returned relative pose is compared with the true relative planar pose of the two body frames.

    cd brain && uv run python ../tools/place_eval.py --out ../data/evidence/place_recognition_eval.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from multiprocessing import get_context
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "brain"))

from brain.agent import place, vision  # noqa: E402
from brain.oracle import load_trajectory  # noqa: E402

RECORDINGS = ["unity_house_hidden_run1", "unity_house_hidden_run3", "unity_house_heldout_run1"]
ROBOTS = ["robot_A", "robot_B"]


# ---------------------------------------------------------------- keyframes

def source_hash():
    """Cache key part: the extraction code and the settings that decide which features exist."""
    return hashlib.sha1((ROOT / "brain/brain/agent/place.py").read_bytes()).hexdigest()[:12]


def extract_recording_robot(args):
    """All usable features (strongest first) at every sampled time. Returns arrays indexed by sample."""
    recording, robot, step, settings = args
    folder = ROOT / "data/recordings" / recording / robot
    stamps = [float(line.split()[0]) for line in (folder / "rgb.txt").read_text().splitlines()
              if line.strip() and not line.startswith("#")]
    times = np.arange(0.0, stamps[-1] + 1e-6, step)
    # Keep every usable feature so any features-per-keyframe setting can be applied later by slicing.
    wide = dict(settings, place_features_per_keyframe=100000)
    out = {"t": [], "t_frame": [], "desc": [], "xyh": [], "usable": []}
    for t in times:
        frame = vision.load_frame(str(folder), float(t))
        desc, xyh = place.extract_features(frame.rgb, frame.depth_m, frame.K, wide)
        out["t"].append(float(t))
        out["t_frame"].append(float(frame["stamps"]["rgb"]))
        out["desc"].append(desc)
        out["xyh"].append(xyh)
        out["usable"].append(len(desc))
    return (recording, robot), out


def load_all(settings, step, cache_dir, workers=6):
    """Extract (or reload) features for every recording and robot. The cache key covers code and settings."""
    key = hashlib.sha1(json.dumps([source_hash(), step, settings["place_detect_features"],
                                   settings["place_max_depth_m"], settings["place_orb_scale_factor"],
                                   settings["place_orb_levels"], cv2.__version__], sort_keys=True).encode()).hexdigest()[:12]
    cache = Path(cache_dir) / f"features_{key}.npz" if cache_dir else None
    jobs = [(r, b, step, settings) for r in RECORDINGS for b in ROBOTS]
    if cache and cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {tuple(k.split("|")): v.item() for k, v in z.items()}
    with get_context("spawn").Pool(workers) as pool:
        result = dict(pool.map(extract_recording_robot, jobs))
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **{"|".join(k): np.array(v, dtype=object) for k, v in result.items()})
    return result


# ---------------------------------------------------------------- ground truth

def truth_planar(recording, robot, t_frames, mount):
    """True planar body pose of each keyframe in the house floor plane (east, north).

    The truth file holds the camera optical pose, camera-to-world, in the house frame (x east, y down, z north).
    Heading is the optical +z axis projected on the floor, counter-clockwise from east. The body origin lies
    forward_offset_m behind the camera along that heading; pitch only tilts the camera, so the heading is the body's."""
    tr = load_trajectory(ROOT / "data/recordings" / recording, robot)
    index = np.searchsorted(tr.timestamps, t_frames)
    index = np.clip(index, 1, len(tr.timestamps) - 1)
    pick = np.where(np.abs(tr.timestamps[index - 1] - t_frames) <= np.abs(tr.timestamps[index] - t_frames), index - 1, index)
    assert np.abs(tr.timestamps[pick] - t_frames).max() < 1e-3, "truth timestamps do not line up with frame stamps"
    R = Rotation.from_quat(tr.quaternions_xyzw[pick]).as_matrix()
    forward = R[:, :, 2]
    heading = np.arctan2(forward[:, 2], forward[:, 0])
    cam = tr.positions[pick][:, [0, 2]]
    fwd = np.column_stack((np.cos(heading), np.sin(heading)))
    origin = cam - mount["forward_offset_m"] * fwd
    return origin, heading, tr.positions[pick], R


def wrap_deg(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def true_relative(origin_p, heading_p, origin_m, heading_m):
    """Partner body pose in my body frame (dx, dy, dyaw_deg), body x right, y forward, yaw counter-clockwise.

    A body point b maps to world as origin + R(heading - 90 deg) b, so partner to mine is
    R(heading_p - heading_m) p + R(-(heading_m - 90 deg)) (origin_p - origin_m)."""
    psi = heading_m - math.pi / 2
    d = np.asarray(origin_p) - np.asarray(origin_m)
    c, s = math.cos(-psi), math.sin(-psi)
    return (c * d[0] - s * d[1], s * d[0] + c * d[1], float(wrap_deg(math.degrees(heading_p - heading_m))))


def pose_error(est, true):
    return float(math.hypot(est[0] - true[0], est[1] - true[1])), float(abs(wrap_deg(est[2] - true[2])))


# ---------------------------------------------------------------- keyframes and pairs

def build_keyframes(samples, settings):
    """Keyframe (or None when rejected) for each sample, built the way the service builds them."""
    n = int(settings["place_features_per_keyframe"])
    out = []
    for desc, xyh in zip(samples["desc"], samples["xyh"]):
        if len(desc) < int(settings["place_min_features"]):
            out.append(None)
            continue
        out.append(place.decode_payload(place.encode_payload(desc[:n], xyh[:n])))
    return out


def ordered(ref):
    return f"{ref[0]}/{ref[1]}"


def make_pairs(truth, keyframes, settings, same_gap_s, pos_dist, pos_yaw, neg_dist, rng, same_cap):
    """Positive and negative keyframe pairs from true poses. truth[ref] = (origin, heading, t)."""
    pairs = []
    for rec in RECORDINGS:
        for kind, (pr, mr) in (("cross", (ROBOTS[1], ROBOTS[0])), ("cross_rev", (ROBOTS[0], ROBOTS[1])),
                               ("same_A", (ROBOTS[0], ROBOTS[0])), ("same_B", (ROBOTS[1], ROBOTS[1]))):
            po, ph, pt = truth[(rec, pr)]
            mo, mh, mt = truth[(rec, mr)]
            kp, km = keyframes[(rec, pr)], keyframes[(rec, mr)]
            ok_p = np.array([k is not None for k in kp])
            ok_m = np.array([k is not None for k in km])
            D = np.hypot(po[:, None, 0] - mo[None, :, 0], po[:, None, 1] - mo[None, :, 1])
            dyaw = np.abs(wrap_deg(np.degrees(ph[:, None] - mh[None, :])))
            valid = ok_p[:, None] & ok_m[None, :]
            if kind.startswith("same"):
                # Different times only: the partner is the later keyframe, mine the earlier one, at least same_gap_s apart.
                valid &= (pt[:, None] - mt[None, :]) >= same_gap_s
            pos = valid & (D <= pos_dist) & (dyaw < pos_yaw)
            neg = valid & (D > neg_dist)
            for label, mask in (("pos", pos), ("neg", neg)):
                idx = np.argwhere(mask)
                # Revisit positives are dominated by stationary periods, so only they are subsampled.
                if label == "pos" and kind.startswith("same") and len(idx) > same_cap:
                    idx = idx[np.sort(rng.choice(len(idx), same_cap, replace=False))]
                for i, j in idx:
                    pairs.append((rec, kind, label, int(i), int(j), float(D[i, j]), float(dyaw[i, j])))
    return pairs


def permissive(settings):
    """Same matcher, but every candidate is reported, so min_inliers and max_rmse can be applied afterwards."""
    return dict(settings, place_min_inliers=2, place_max_rmse_m=1e9)


def run_pairs(pairs, keyframes, truth, settings):
    """Run the matcher on each pair. Row: partner pose in mine from truth, the matcher output, and the pair label."""
    loose = permissive(settings)
    rows = []
    for rec, kind, label, i, j, dist, dyaw_true in pairs:
        pr, mr = {"cross": (ROBOTS[1], ROBOTS[0]), "cross_rev": (ROBOTS[0], ROBOTS[1]),
                  "same_A": (ROBOTS[0], ROBOTS[0]), "same_B": (ROBOTS[1], ROBOTS[1])}[kind]
        po, ph, pt = truth[(rec, pr)]
        mo, mh, mt = truth[(rec, mr)]
        out = place.match_keyframes(keyframes[(rec, pr)][i], [(0, keyframes[(rec, mr)][j])], loose)
        true = true_relative(po[i], ph[i], mo[j], mh[j])
        row = {"rec": rec, "kind": kind, "label": label, "i": i, "j": j, "dist": dist, "yaw_diff": dyaw_true,
               "t_partner": float(pt[i]), "t_mine": float(mt[j]), "true": true, "matched": out["matched"]}
        if out["matched"]:
            est = (out["dx_m"], out["dy_m"], out["dyaw_deg"])
            row.update(inliers=out["inliers"], rmse=out["rmse_m"], est=est, err=pose_error(est, true))
        else:
            row.update(inliers=0, rmse=None, est=None, err=None)
        rows.append(row)
    return rows


def pairs_worker(args):
    pairs, keyframes, truth, settings = args
    return run_pairs(pairs, keyframes, truth, settings)


def accepted(row, min_inliers, max_rmse):
    return bool(row["matched"] and row["inliers"] >= min_inliers and row["rmse"] <= max_rmse)


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - h) / d, 4), round((c + h) / d, 4)]


def summarize(rows, min_inliers, max_rmse, trans_ok_m=0.5, rot_ok_deg=10.0):
    """Counts and error percentiles for a set of pair rows at one acceptance setting."""
    pos = [r for r in rows if r["label"] == "pos"]
    neg = [r for r in rows if r["label"] == "neg"]
    acc_pos = [r for r in pos if accepted(r, min_inliers, max_rmse)]
    acc_neg = [r for r in neg if accepted(r, min_inliers, max_rmse)]
    errs = np.array([r["err"] for r in acc_pos]) if acc_pos else np.zeros((0, 2))
    correct = [r for r in acc_pos if r["err"][0] <= trans_ok_m and r["err"][1] <= rot_ok_deg]

    def pct(col, q):
        return round(float(np.percentile(errs[:, col], q)), 3) if len(errs) else None

    return {"n_pos": len(pos), "n_neg": len(neg), "accepted_pos": len(acc_pos), "recall": round(len(acc_pos) / len(pos), 4) if pos else None,
            "recall_ci95": wilson(len(acc_pos), len(pos)), "false_accepts": len(acc_neg),
            "false_accept_rate": round(len(acc_neg) / len(neg), 6) if neg else None,
            "false_accept_ci95_upper": wilson(len(acc_neg), len(neg))[1],
            "accepted_pos_correct": len(correct), "accepted_pos_wrong_pose": len(acc_pos) - len(correct),
            "trans_err_m_median": pct(0, 50), "trans_err_m_p95": pct(0, 95),
            "rot_err_deg_median": pct(1, 50), "rot_err_deg_p95": pct(1, 95)}


def run_database(keyframes, truth, settings, recordings, pos_dist, pos_yaw, neg_dist, trans_ok_m=0.5, rot_ok_deg=10.0):
    """Database mode, as the service runs it: each partner keyframe is matched against the receiver's whole database.

    A returned match is correct when its pose is within trans_ok_m and rot_ok_deg of the true relative pose to the
    keyframe it names. Queries are grouped by what the database truly holds: a place (an own keyframe within
    pos_dist and pos_yaw) or none (every own keyframe farther than neg_dist)."""
    kept = int(settings["place_keyframes_kept"])
    rows = []
    for rec in recordings:
        for pr, mr in ((ROBOTS[1], ROBOTS[0]), (ROBOTS[0], ROBOTS[1])):
            po, ph, pt = truth[(rec, pr)]
            mo, mh, mt = truth[(rec, mr)]
            own_all = [(j, k) for j, k in enumerate(keyframes[(rec, mr)]) if k is not None][-kept:]
            own_ids = np.array([j for j, _ in own_all])
            for i, kp in enumerate(keyframes[(rec, pr)]):
                if kp is None:
                    continue
                D = np.hypot(mo[own_ids, 0] - po[i, 0], mo[own_ids, 1] - po[i, 1])
                yaw = np.abs(wrap_deg(np.degrees(ph[i] - mh[own_ids])))
                group = "place" if ((D <= pos_dist) & (yaw < pos_yaw)).any() else ("none" if (D > neg_dist).all() else "between")
                out = place.match_keyframes(kp, own_all, settings)
                row = {"rec": rec, "direction": f"{pr}->{mr}", "i": i, "t_partner": float(pt[i]), "group": group,
                       "matched": out["matched"], "min_dist": float(D.min()) if len(D) else None}
                if out["matched"]:
                    j = out["mine_id"]
                    true = true_relative(po[i], ph[i], mo[j], mh[j])
                    est = (out["dx_m"], out["dy_m"], out["dyaw_deg"])
                    err = pose_error(est, true)
                    row.update(mine_id=j, inliers=out["inliers"], rmse=out["rmse_m"], err=err,
                               correct=bool(err[0] <= trans_ok_m and err[1] <= rot_ok_deg),
                               t_mine=float(mt[j]), true=true, est=est)
                rows.append(row)
    return rows


def summarize_database(rows):
    out = {"queries": len(rows)}
    for group in ("place", "none", "between"):
        g = [r for r in rows if r["group"] == group]
        ok = [r for r in g if r["matched"] and r["correct"]]
        bad = [r for r in g if r["matched"] and not r["correct"]]
        out[group] = {"n": len(g), "accepted_correct": len(ok), "accepted_wrong": len(bad),
                      "rate_correct": round(len(ok) / len(g), 4) if g else None,
                      "rate_correct_ci95": wilson(len(ok), len(g)), "rate_wrong": round(len(bad) / len(g), 6) if g else None}
    out["accepted_wrong_total"] = sum(1 for r in rows if r["matched"] and not r["correct"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/evidence/place_recognition_eval.json"))
    ap.add_argument("--cache", default=str(ROOT / "data/runs/place_eval_cache"))
    args = ap.parse_args()
    s = place.load_settings()
    data = load_all(s, 1.0, args.cache)
    truth = {}
    for (rec, rob), v in data.items():
        o, h, _, _ = truth_planar(rec, rob, np.array(v["t_frame"]), s["mount"])
        truth[(rec, rob)] = (o, h, np.array(v["t_frame"]))
    kfs = {k: build_keyframes(v, s) for k, v in data.items()}
    # Convention check: a keyframe against itself has identity truth and the matcher returns identity.
    ident = true_relative(truth[(RECORDINGS[0], "robot_A")][0][50], truth[(RECORDINGS[0], "robot_A")][1][50],
                          truth[(RECORDINGS[0], "robot_A")][0][50], truth[(RECORDINGS[0], "robot_A")][1][50])
    selfm = place.match_keyframes(kfs[(RECORDINGS[0], "robot_A")][50], [(0, kfs[(RECORDINGS[0], "robot_A")][50])], s)
    pairs = make_pairs(truth, kfs, s, 10.0, 1.5, 30.0, 6.0, np.random.default_rng(0), 300)
    rows = run_pairs(pairs, kfs, truth, s)
    db = run_database(kfs, truth, s, RECORDINGS, 1.5, 30.0, 6.0)
    dev = RECORDINGS[:2]
    out = {"settings": {k: v for k, v in s.items() if k.startswith("place_")},
           "payload_bytes": place.HEADER_BYTES + place.RECORD_BYTES * int(s["place_features_per_keyframe"]),
           "identity_check": {"true_relative_same_frame": ident, "matcher_self_match": selfm},
           "definitions": "positive: true camera positions within 1.5 m and floor viewing directions within 30 deg; negative: more than 6 m apart; same-robot pairs at least 10 s apart; truth read through brain.oracle",
           "keyframes": {f"{r}/{b}": {"sampled": len(v["usable"]), "rejected_too_few_features": int((np.array(v["usable"]) < s["place_min_features"]).sum())}
                         for (r, b), v in data.items()},
           "pair_mode": {}, "database_mode": {}}
    out["keyframes_total"] = {"sampled": sum(x["sampled"] for x in out["keyframes"].values()),
                              "rejected": sum(x["rejected_too_few_features"] for x in out["keyframes"].values())}
    negmax = max(r["inliers"] for r in rows if r["label"] == "neg")
    out["max_inliers_among_negatives"] = int(negmax)
    for kmin in sorted({7, int(s["place_min_inliers"])}):
        for name, recs in (("dev_run1_run3", dev), ("heldout_run1", RECORDINGS[2:]), ("all", RECORDINGS)):
            for cls, kinds in (("cross_B_to_A", ("cross",)), ("cross_A_to_B", ("cross_rev",)), ("same_robot_revisit", ("same_A", "same_B"))):
                sub = [r for r in rows if r["rec"] in recs and r["kind"] in kinds]
                out["pair_mode"][f"min_inliers={kmin}/{name}/{cls}"] = summarize(sub, kmin, s["place_max_rmse_m"])
        for name, recs in (("dev_run1_run3", dev), ("heldout_run1", RECORDINGS[2:]), ("all", RECORDINGS)):
            sub = [r for r in db if r["rec"] in recs and not (r["matched"] and r["inliers"] < kmin)]
            out["database_mode"][f"min_inliers={kmin}/{name}"] = summarize_database(sub)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("keyframes_total", "max_inliers_among_negatives", "payload_bytes")}))
    for k, v in out["pair_mode"].items():
        if "/all/cross_B" in k or "/heldout" in k and "cross_B" in k:
            print(k, {x: v[x] for x in ("n_pos", "n_neg", "recall", "false_accepts", "trans_err_m_median", "trans_err_m_p95", "rot_err_deg_median", "rot_err_deg_p95")})
    for k, v in out["database_mode"].items():
        print(k, v["place"], v["accepted_wrong_total"])


if __name__ == "__main__":
    main()
