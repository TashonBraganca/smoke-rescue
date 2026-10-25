#!/usr/bin/env python3
"""Relocalization upgrade test on the TUM pairs of tools/real_data_check.py: PnP, better matching, temporal consistency.

Every variant starts from the same keyframes (place.extract_features, production settings, payload-quantised partner
points) and the same pair list and split as tools/fence_fix_compare.py. The 'ours' rows must reproduce the committed
numbers (all 1518/2414 accepted, 70 wrong, 0/6500 false; held-out 1203/1601, 46 wrong, 0/3394 false) or the run halts.

Variants:
  ours           place.good_matches (kNN ratio 0.85, one-to-one) + height filter + place.ransac_rigid, >= 10 inliers,
                 RMSE <= 0.06 m. Unchanged production path.
  ours_wide      ours with a wider planar RANSAC threshold, RMSE gate and min inliers, tuned: the fair rival to PnP,
                 which tolerates own depth noise at range by working in pixels.
  ours_range     ours, then reject when the median planar range of the own-side inliers exceeds R metres (the lead from
                 the fence fix: wrong-pose inliers sat far away on tuning data).
  mutual         cv2.BFMatcher(NORM_HAMMING, crossCheck=True): mutual nearest neighbours, no ratio test; then height
                 filter and place.ransac_rigid; min inliers tuned.
  mutual_ratio   the ratio-test set of ours, keeping only pairs that are also mutual nearest neighbours; min inliers tuned.
  pnp_sqpnp      cv2.solvePnPRansac(SOLVEPNP_SQPNP): partner keyframe 3D body points (the radio payload, cm-quantised)
                 against own keyframe 2D keypoint pixels, on the same ratio-test + height-filtered match set.
  pnp_magsac     the same with flags=cv2.USAC_MAGSAC (OpenCV's USAC PnP path).
                 PnP gives the own camera pose in the partner body frame. It is converted to the planar link with the
                 own keyframe's camera-to-body rotation B and mount height h (the same tilt and height the 2D pipeline
                 already uses to build own body points): M = B R_pnp, t = B t_pnp + h e_z. dyaw = atan2(M10, M00),
                 (dx, dy) = t[:2]. Both body frames are gravity aligned, so the residual tilt of M and t_z must be near
                 zero for a right pose; both are offered to the tuner as optional gates. Inliers are recounted with the
                 final pose at the reprojection threshold.
  temporal_next  ours on (i, j), accepted only when ours also accepts the next keyframe pair (i+1, j+1) and the two
                 agree: pose of partner keyframe i+1 in own keyframe j predicted two ways, M(i,j) * O_a(i->i+1) and
                 O_b(j->j+1) * M(i+1,j+1), within link_agree_m 0.3 m and link_agree_deg 4 deg (config/brain.json).
                 O_a, O_b are each robot's own odometry between its consecutive keyframes; on TUM they come from the
                 ground-truth planar poses (0.33 s apart), which is a stand-in for the robot's own SLAM pose over one
                 keyframe step. Neighbours must be at most 1.0 s apart and both have keyframes, else the pair is rejected.
  temporal_either the same, with (i+1, j+1) or (i-1, j-1).

Thresholds are picked on TUNING only (fr2_pioneer_slam + fr1_room): 0 false accepts and recall >= ours on tuning,
then fewest wrong poses, then most recall. Reported on HELD-OUT (fr2_pioneer_360 + fr3_long_office_household) first.
Two tuning protocols. v1 (the fence-fix protocol): tuning negatives are the committed sample (1500 of pioneer_slam's
12023 within-sequence negatives; fr1_room has none over 4 m) plus tuning cross pairs. v2: every within-sequence tuning
negative must also stay unaccepted. v2 was added after a fixed-mount diagnostic found PnP false accepts on unsampled
pioneer_slam negatives, which was after I had seen the v1 held-out result; both picks are reported. Held-out false
accepts are counted on the committed sample and on every within-sequence held-out negative.

Ground truth that enters a variant (not only scoring): the per-frame camera tilt and height (every variant, as in
real_data_check), and for temporal_* the planar odometry between consecutive keyframes of the same robot.

    nice -n 10 brain/.venv/bin/python -I tools/reloc_compare.py [--no-write]
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_data_check as rdc  # noqa: E402
from real_data_check import place  # noqa: E402
from fence_fix_compare import HELD_OUT, TUNING  # noqa: E402

LINK_AGREE_M, LINK_AGREE_DEG = 0.3, 4.0      # config/brain.json link_agree_m / link_agree_deg
NEIGHBOUR_MAX_DT_S = 1.0
PNP_THR_PX = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
PNP_MIN_INL = [8, 10, 12, 15, 20, 25, 30]
PNP_TILT_DEG = [1.0, 2.0, 3.0, 5.0, math.inf]
PNP_DZ_M = [0.05, 0.1, 0.2, 0.3, math.inf]
PNP_FLAGS = {"pnp_sqpnp": cv2.SOLVEPNP_SQPNP, "pnp_magsac": cv2.USAC_MAGSAC}
MATCH_MIN_INL = list(range(10, 21))
RANGE_GRID = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, math.inf]
WIDE_THR_M = [0.1, 0.15, 0.2, 0.25, 0.3]
WIDE_RMSE_M = [0.06, 0.1, 0.15]
WIDE_MIN_INL = [10, 12, 15]


# ------------------------------------------------------------------------------------------------------- keyframes
def keyframe_px(frame, K, settings):
    """rdc.keyframe plus what stays on the robot: own keypoint pixels (n, 2, unquantised), camera-to-body rotation
    B (3, 3) and mount height h, from the same ground-truth tilt and height rdc.keyframe uses."""
    bgr = cv2.imread(str(frame["rgb"]), cv2.IMREAD_COLOR)
    d = cv2.imread(str(frame["depth"]), cv2.IMREAD_UNCHANGED)
    if bgr is None or d is None or d.dtype != np.uint16 or d.shape != bgr.shape[:2]:
        raise ValueError(f"bad frame {frame['rgb']}")
    depth_m = d.astype(np.float32) / rdc.DEPTH_SCALE
    depth_m[d == 0] = np.nan
    s = dict(settings)
    s["mount"] = {"height_m": 0.0, "forward_offset_m": 0.0, "pitch_down_deg": 0.0}
    Kmat = np.array([[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1.0]])
    desc, xyh = place.extract_features(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth_m, Kmat, s)
    if len(desc) < int(settings["place_min_features"]):
        return None
    cam = np.column_stack((xyh[:, 0], -xyh[:, 2], xyh[:, 1]))   # optical x right, y down, z forward (m)
    R, h = frame["R"], float(frame["c"][2])
    kf = place.decode_payload(place.encode_payload(desc, rdc.to_body(cam, R, h)))
    px = np.column_stack((K[0] * cam[:, 0] / cam[:, 2] + K[2], K[1] * cam[:, 1] / cam[:, 2] + K[3]))
    x, y = rdc.body_axes(R)
    B = np.vstack((x, y, rdc.UP)) @ R
    return {"kf": kf, "px": px, "B": B, "h": h, "K": Kmat, "cam": cam, "bgr": bgr}


def check_keyframes(name, frames, K, kx, settings):
    """Halting gates on every 10th frame: identical to rdc.keyframe; pixels are the ORB keypoint positions found by an
    independent detectAndCompute; pixels are inside the 640 x 480 image; B is a rotation."""
    worst_px = 0.0
    for i in range(0, len(frames), 10):
        ref = rdc.keyframe(frames[i], K, settings)
        mine = kx[i]
        if (ref is None) != (mine is None):
            raise SystemExit(f"{name} frame {i}: keyframe presence differs from rdc.keyframe")
        if ref is None:
            continue
        if not (np.array_equal(ref.desc, mine["kf"].desc) and np.array_equal(ref.xyh, mine["kf"].xyh)):
            raise SystemExit(f"{name} frame {i}: keyframe differs from rdc.keyframe")
        px = mine["px"]
        if not ((px[:, 0] >= 0) & (px[:, 0] < 640) & (px[:, 1] >= 0) & (px[:, 1] < 480)).all():
            raise SystemExit(f"{name} frame {i}: own pixel outside the image")
        B = mine["B"]
        if abs(np.linalg.det(B) - 1) > 1e-9 or np.abs(B @ B.T - np.eye(3)).max() > 1e-9:
            raise SystemExit(f"{name} frame {i}: B is not a rotation")
        gray = cv2.cvtColor(mine["bgr"], cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=int(settings["place_detect_features"]),
                             scaleFactor=float(settings["place_orb_scale_factor"]), nlevels=int(settings["place_orb_levels"]))
        kps, dd = orb.detectAndCompute(gray, None)
        pts = np.array([k.pt for k in kps])
        for n in range(len(px)):
            same = np.flatnonzero((dd == mine["kf"].desc[n]).all(axis=1))
            if not len(same):
                raise SystemExit(f"{name} frame {i}: kept descriptor {n} not found in an independent ORB run")
            worst_px = max(worst_px, float(np.min(np.hypot(*(pts[same] - px[n]).T))))
    if worst_px > 1e-3:
        raise SystemExit(f"{name}: recovered own pixels are up to {worst_px:.4f} px from the ORB keypoints")
    return worst_px


# ---------------------------------------------------------------------------------------------------------- matching
def height_filter(pairs, a, b, settings):
    if not len(pairs):
        return pairs
    dh = np.abs(a.xyh[pairs[:, 0], 2] - b.xyh[pairs[:, 1], 2])
    return pairs[dh <= settings["place_height_tolerance_m"]]


def mutual_pairs(a, b):
    if len(a.desc) < 1 or len(b.desc) < 1:
        return np.zeros((0, 2), int)
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(a.desc, b.desc)
    return np.array([(x.queryIdx, x.trainIdx) for x in m], int).reshape(-1, 2)


def match_sets(a, b, settings):
    knn = place.good_matches(a.desc, b.desc, settings["place_ratio_test"])
    mut = mutual_pairs(a, b)
    mutual_set = {tuple(p) for p in mut}
    mr = np.array([p for p in knn if tuple(p) in mutual_set], int).reshape(-1, 2)
    return {"knn": height_filter(knn, a, b, settings), "mutual": height_filter(mut, a, b, settings),
            "mutual_ratio": height_filter(mr, a, b, settings)}


# ---------------------------------------------------------------------------------------------------------- fits
def rigid_fit(a, b, pairs, settings, thr=None):
    """place.ransac_rigid on a match set: dict(n, rmse, yaw deg, t, range) or None. range = median planar range of
    the own-side inliers (m)."""
    if len(pairs) < 2:
        return None
    p, q = a.xyh[pairs[:, 0], :2], b.xyh[pairs[:, 1], :2]
    fit = place.ransac_rigid(p, q, settings if thr is None else dict(settings, place_ransac_threshold_m=thr))
    if fit is None:
        return None
    angle, t, mask, rmse = fit
    return {"n": int(mask.sum()), "rmse": rmse, "yaw": rdc.wrap(math.degrees(angle)), "t": np.asarray(t, float),
            "range": float(np.median(np.hypot(*q[mask].T)))}


def pnp_fit(a, own, pairs, flag, thr_px, settings):
    """Partner 3D body points vs own 2D pixels -> planar link. dict(n, yaw, t, tilt deg, dz m) or None."""
    if len(pairs) < 6:
        return None
    obj = a.xyh[pairs[:, 0]].astype(np.float64)
    img = own["px"][pairs[:, 1]].astype(np.float64)
    cv2.setRNGSeed(int(settings["place_ransac_seed"]))
    try:
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, own["K"], None, iterationsCount=int(settings["place_ransac_iterations"]),
                                                 reprojectionError=float(thr_px), confidence=0.99, flags=flag)
    except cv2.error:
        return None
    if not ok or inl is None or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
        return None
    Rp = cv2.Rodrigues(rvec)[0]
    proj = cv2.projectPoints(obj, rvec, tvec, own["K"], None)[0].reshape(-1, 2)
    depth = (obj @ Rp.T + tvec.ravel())[:, 2]
    n = int(((np.hypot(*(proj - img).T) <= thr_px) & (depth > 0)).sum())
    M = own["B"] @ Rp
    t = own["B"] @ tvec.ravel() + np.array([0.0, 0.0, own["h"]])
    return {"n": n, "yaw": rdc.wrap(math.degrees(math.atan2(M[1, 0], M[0, 0]))), "t": t[:2].copy(),
            "tilt": math.degrees(math.acos(float(np.clip(M[2, 2], -1, 1)))), "dz": float(t[2])}


def check_pnp_frames(kx, settings):
    """Self-match gate for the camera/body conversion: a keyframe matched against itself must give the identity link.
    With unquantised body points (B cam + h e_z) the link must be exact (1e-3 deg, 1e-3 m): this tests the frame
    conversion. With the radio's cm-quantised points the spread is reported, not gated: rounding a point 0.8 m away
    by up to 5 mm tilts the solved pose by up to about 0.8 deg (fr3, measured), which a robot would see too."""
    worst, tested, quant = 0.0, 0, []
    for name, ks in kx.items():
        for k in ks[::25]:
            if k is None:
                continue
            pairs = np.column_stack((np.arange(len(k["kf"])), np.arange(len(k["kf"]))))
            exact = place.Keyframe(k["kf"].desc, k["cam"] @ k["B"].T + np.array([0.0, 0.0, k["h"]]))
            for flag in PNP_FLAGS.values():
                f = pnp_fit(exact, k, pairs, flag, 3.0, settings)
                g = pnp_fit(k["kf"], k, pairs, flag, 3.0, settings)
                if f is None or g is None:
                    raise SystemExit(f"{name}: PnP self-match failed")
                err = max(abs(f["yaw"]), float(np.hypot(*f["t"])), f["tilt"], abs(f["dz"])) / 1e-3
                worst = max(worst, err)
                tested += 1
                quant.append((abs(g["yaw"]), float(np.hypot(*g["t"])), g["tilt"]))
                if err > 1:
                    raise SystemExit(f"{name}: PnP self-match on exact points is not the identity: {f}")
    return tested, worst, np.max(np.array(quant), axis=0)


# ---------------------------------------------------------------------------------------------------------- pairs
def collect(settings):
    """The pairs of fence_fix_compare.collect, same seeded draw, keeping (sequence, i, j) for the temporal variants."""
    rng = np.random.default_rng(rdc.SEED)
    seq, out, gate_px = {}, [], {}
    for name in rdc.SEQUENCES:
        frames = rdc.load_sequence(name)
        K = rdc.INTRINSICS[name.split("_")[2]]
        kx = [keyframe_px(f, K, settings) for f in frames]
        gate_px[name] = check_keyframes(name, frames, K, kx, settings)
        for k in kx:
            if k is not None:
                del k["bgr"]
        seq[name] = (frames, kx)
        pos, neg = rdc.pairs_for(frames, rng)
        split = "held" if name in HELD_OUT else "tune"
        # Every other within-sequence negative (> 4 m apart), outside the committed sample: split tune_x / held_x.
        cap, rdc.NEG_PER_SEQ = rdc.NEG_PER_SEQ, 10 ** 9
        _, every = rdc.pairs_for(frames, np.random.default_rng(0))
        rdc.NEG_PER_SEQ = cap
        sampled = set(neg)
        for i, j in every:
            if (i, j) not in sampled and kx[i] is not None and kx[j] is not None:
                out.append({"split": split + "_x", "kind": "neg", "name": name, "sa": name, "sb": name, "i": i, "j": j,
                            "truth": None})
        for i, j in pos:
            if kx[i] is not None and kx[j] is not None:
                out.append({"split": split, "kind": "pos", "name": name, "sa": name, "sb": name, "i": i, "j": j,
                            "truth": rdc.truth_relative(frames[i], frames[j])})
        for i, j in neg:
            if kx[i] is not None and kx[j] is not None:
                out.append({"split": split, "kind": "neg", "name": name, "sa": name, "sb": name, "i": i, "j": j, "truth": None})
    names = list(seq)
    valid = {n: [i for i, k in enumerate(seq[n][1]) if k is not None] for n in names}
    combos = [(a, b) for a in names for b in names if a != b and not (a in rdc.PIONEER and b in rdc.PIONEER)]
    for _ in range(rdc.CROSS_N):
        a, b = combos[int(rng.integers(len(combos)))]
        i = valid[a][int(rng.integers(len(valid[a])))]
        j = valid[b][int(rng.integers(len(valid[b])))]
        split = "held" if {a, b} <= HELD_OUT else "tune" if {a, b} <= TUNING else "mixed"
        out.append({"split": split, "kind": "neg", "name": "cross", "sa": a, "sb": b, "i": i, "j": j, "truth": None})
    return seq, out, gate_px


def ours_ok(f, settings):
    return f is not None and f["n"] >= settings["place_min_inliers"] and f["rmse"] <= settings["place_max_rmse_m"]


def se2(yaw_deg, t):
    a = math.radians(yaw_deg)
    return np.array([[math.cos(a), -math.sin(a), t[0]], [math.sin(a), math.cos(a), t[1]], [0, 0, 1.0]])


def odom(frames, i, k):
    """Planar pose of keyframe k in keyframe i's body frame (the robot's own odometry; from ground truth on TUM)."""
    yaw, t = rdc.truth_relative(frames[k], frames[i])
    return se2(yaw, t)


def evaluate(seq, records, settings):
    rows = []
    ours_cache = {}

    def ours_pair(sa, i, sb, j):
        key = (sa, i, sb, j)
        if key not in ours_cache:
            a, b = seq[sa][1][i]["kf"], seq[sb][1][j]["kf"]
            ours_cache[key] = rigid_fit(a, b, height_filter(place.good_matches(a.desc, b.desc, settings["place_ratio_test"]),
                                                            a, b, settings), settings)
        return ours_cache[key]

    for r in records:
        A, Bk = seq[r["sa"]][1][r["i"]], seq[r["sb"]][1][r["j"]]
        a, b = A["kf"], Bk["kf"]
        t0 = time.perf_counter()
        knn = height_filter(place.good_matches(a.desc, b.desc, settings["place_ratio_test"]), a, b, settings)
        r["t_knn"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        sets = match_sets(a, b, settings)
        r["t_sets"] = time.perf_counter() - t0
        if not np.array_equal(sets["knn"], knn):
            raise SystemExit("match_sets knn differs from the production match set")
        t0 = time.perf_counter()
        r["ours"] = rigid_fit(a, b, knn, settings)
        r["t_ours"] = time.perf_counter() - t0
        ours_cache[(r["sa"], r["i"], r["sb"], r["j"])] = r["ours"]
        for thr in WIDE_THR_M:
            t0 = time.perf_counter()
            r[("ours_wide", thr)] = r["ours"] if thr == settings["place_ransac_threshold_m"] else rigid_fit(a, b, knn, settings, thr)
            r[("t", "ours_wide", thr)] = r["t_ours"] if thr == settings["place_ransac_threshold_m"] else time.perf_counter() - t0
        for v in ("mutual", "mutual_ratio"):
            t0 = time.perf_counter()
            r[v] = rigid_fit(a, b, sets[v], settings)
            r["t_" + v] = time.perf_counter() - t0
        for v, flag in PNP_FLAGS.items():
            for thr in PNP_THR_PX:
                t0 = time.perf_counter()
                r[(v, thr)] = pnp_fit(a, Bk, knn, flag, thr, settings)
                r[("t", v, thr)] = time.perf_counter() - t0
        # Temporal: does the neighbouring keyframe pair agree?
        for lab, step in (("next", 1), ("prev", -1)):
            r["temporal_" + lab] = False
            if not ours_ok(r["ours"], settings):
                continue
            fa, fb = seq[r["sa"]][0], seq[r["sb"]][0]
            i2, j2 = r["i"] + step, r["j"] + step
            if not (0 <= i2 < len(fa) and 0 <= j2 < len(fb)):
                continue
            if abs(fa[i2]["t"] - fa[r["i"]]["t"]) > NEIGHBOUR_MAX_DT_S or abs(fb[j2]["t"] - fb[r["j"]]["t"]) > NEIGHBOUR_MAX_DT_S:
                continue
            if seq[r["sa"]][1][i2] is None or seq[r["sb"]][1][j2] is None:
                continue
            f2 = ours_pair(r["sa"], i2, r["sb"], j2)
            if not ours_ok(f2, settings):
                continue
            pred1 = se2(r["ours"]["yaw"], r["ours"]["t"]) @ odom(fa, r["i"], i2)
            pred2 = odom(fb, r["j"], j2) @ se2(f2["yaw"], f2["t"])
            d = np.linalg.inv(pred2) @ pred1
            r["temporal_" + lab] = (float(np.hypot(d[0, 2], d[1, 2])) <= LINK_AGREE_M
                                    and abs(math.degrees(math.atan2(d[1, 0], d[0, 0]))) <= LINK_AGREE_DEG)
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------------------------------------- scoring
def err_of(r, f):
    yaw, tt = r["truth"]
    return abs(rdc.wrap(f["yaw"] - yaw)), float(np.hypot(f["t"][0] - tt[0], f["t"][1] - tt[1]))


def decision(r, variant, p, settings):
    """(accepted, inliers, fit) for one row under a variant and its parameters p."""
    if variant == "ours":
        f = r["ours"]
        return ours_ok(f, settings), (f["n"] if f else 0), f
    if variant == "ours_wide":
        f = r[("ours_wide", p["thr"])]
        ok = f is not None and f["n"] >= p["min_inl"] and f["rmse"] <= p["rmse"]
        return ok, (f["n"] if f else 0), f
    if variant == "ours_range":
        f = r["ours"]
        return ours_ok(f, settings) and f["range"] <= p["range"], (f["n"] if f else 0), f
    if variant in ("mutual", "mutual_ratio"):
        f = r[variant]
        ok = f is not None and f["n"] >= p["min_inl"] and f["rmse"] <= settings["place_max_rmse_m"]
        return ok, (f["n"] if f else 0), f
    if variant in PNP_FLAGS:
        f = r[(variant, p["thr"])]
        ok = f is not None and f["n"] >= p["min_inl"] and f["tilt"] <= p["tilt"] and abs(f["dz"]) <= p["dz"]
        return ok, (f["n"] if f else 0), f
    if variant in ("temporal_next", "temporal_either"):
        ok = r["temporal_next"] or (variant == "temporal_either" and r["temporal_prev"])
        f = r["ours"]
        return ok, (f["n"] if f else 0), f
    raise KeyError(variant)


def ms_of(r, variant, p):
    if variant in PNP_FLAGS or variant == "ours_wide":
        return r["t_knn"] + r[("t", variant, p["thr"])]
    if variant in ("mutual", "mutual_ratio"):
        return r["t_sets"] + r["t_" + variant]
    return r["t_knn"] + r["t_ours"]


def score(rows, variant, p, settings):
    acc, false, maxneg, npos, nneg = [], 0, 0, 0, 0
    for r in rows:
        ok, n, f = decision(r, variant, p, settings)
        if r["kind"] == "pos":
            npos += 1
            if ok:
                acc.append(err_of(r, f))
        else:
            nneg += 1
            false += ok
            maxneg = max(maxneg, n)
    wrong = sum(e[0] > rdc.GROSS_YAW_DEG or e[1] > rdc.GROSS_T_M for e in acc)
    ms = 1000 * float(np.mean([ms_of(r, variant, p) for r in rows])) if rows else float("nan")
    return {"n": npos, "k": len(acc), "yaw": [e[0] for e in acc], "trans": [e[1] for e in acc], "wrong": wrong,
            "neg": nneg, "false": false, "maxneg": maxneg, "ms": ms}


def grid_for(variant):
    if variant == "ours_wide":
        return [{"thr": a, "rmse": b, "min_inl": c} for a in WIDE_THR_M for b in WIDE_RMSE_M for c in WIDE_MIN_INL]
    if variant == "ours_range":
        return [{"range": x} for x in RANGE_GRID]
    if variant in ("mutual", "mutual_ratio"):
        return [{"min_inl": m} for m in MATCH_MIN_INL]
    if variant in PNP_FLAGS:
        return [{"thr": a, "min_inl": b, "tilt": c, "dz": d} for a in PNP_THR_PX for b in PNP_MIN_INL
                for c in PNP_TILT_DEG for d in PNP_DZ_M]
    return [{}]


def tune(tune_rows, variant, settings, base):
    """0 false accepts and recall >= ours on tuning; then fewest wrong poses, then most recall."""
    grid = [(p, score(tune_rows, variant, p, settings)) for p in grid_for(variant)]
    ok = [g for g in grid if g[1]["false"] == 0 and g[1]["k"] >= base["k"]]
    if ok:
        return min(ok, key=lambda g: (g[1]["wrong"], -g[1]["k"]))[0], "fewest wrong with 0 false and recall >= ours", grid
    zero = [g for g in grid if g[1]["false"] == 0]
    if zero:
        return max(zero, key=lambda g: (g[1]["k"], -g[1]["wrong"]))[0], "no point kept recall >= ours; highest-recall zero-false point", grid
    return min(grid, key=lambda g: (g[1]["false"], -g[1]["k"]))[0], "no zero-false point; fewest false accepts", grid


def fmt(label, s):
    lo, hi = rdc.wilson(s["k"], s["n"])
    return (f"  {label:<26} {s['n']:>5} {100 * s['k'] / s['n']:5.1f} % ({100 * lo:4.1f}-{100 * hi:4.1f}) "
            f"{rdc.pct(s['yaw'], 50):5.2f}/{rdc.pct(s['yaw'], 90):5.2f}  {rdc.pct(s['trans'], 50):.3f}/{rdc.pct(s['trans'], 90):.3f}  "
            f"{s['wrong']:>4}/{s['k']:<5} {s['false']:>3}/{s['neg']:<5} {s['maxneg']:>3}  {s['ms']:6.2f}")


VARIANTS = ["ours", "ours_wide", "ours_range", "mutual", "mutual_ratio", "pnp_sqpnp", "pnp_magsac", "temporal_next", "temporal_either"]


def main():
    settings = dict(place.load_settings())
    t0 = time.time()
    cache = None
    if "--cache" in sys.argv[1:]:   # development only: rows keyed on the source of everything that produces them
        import hashlib, inspect, pickle
        src = "".join(inspect.getsource(f) for f in (collect, evaluate, keyframe_px, check_keyframes, check_pnp_frames,
                                                     rigid_fit, pnp_fit, match_sets, mutual_pairs, height_filter, odom))
        src += Path(place.__file__).read_text() + Path(rdc.__file__).read_text() + repr(sorted(settings.items()))
        src += repr((PNP_THR_PX, WIDE_THR_M, list(PNP_FLAGS.items())))
        cache = Path(__file__).resolve().parents[1] / "data" / "real" / f"reloc_rows_{hashlib.sha1(src.encode()).hexdigest()[:12]}.pkl"
    if cache is not None and cache.exists():
        seq, every, gate_px, pnp_tested, pnp_worst, pnp_quant = pickle.loads(cache.read_bytes())
        print(f"rows from cache {cache.name}", file=sys.stderr)
    else:
        seq, records, gate_px = collect(settings)
        pnp_tested, pnp_worst, pnp_quant = check_pnp_frames({n: v[1] for n, v in seq.items()}, settings)
        every = evaluate(seq, records, settings)
        if cache is not None:
            cache.write_bytes(pickle.dumps((seq, every, gate_px, pnp_tested, pnp_worst, pnp_quant)))
    rows = [r for r in every if not r["split"].endswith("_x")]
    tune_rows = [r for r in rows if r["split"] == "tune"]
    held_rows = [r for r in rows if r["split"] == "held"]
    tune_all = tune_rows + [r for r in every if r["split"] == "tune_x"]
    held_xneg = [r for r in held_rows if r["kind"] == "neg"] + [r for r in every if r["split"] == "held_x"]
    tune_xneg = [r for r in tune_rows if r["kind"] == "neg"] + [r for r in every if r["split"] == "tune_x"]

    allv = score(rows, "ours", {}, settings)
    if (allv["n"], allv["k"], allv["wrong"], allv["false"], allv["neg"]) != (2414, 1518, 70, 0, 6500):
        raise SystemExit(f"harness does not reproduce ours (2414, 1518, 70, 0, 6500): {allv}")
    hv = score(held_rows, "ours", {}, settings)
    if (hv["n"], hv["k"], hv["wrong"], hv["false"], hv["neg"]) != (1601, 1203, 46, 0, 3394):
        raise SystemExit(f"harness does not reproduce held-out ours (1601, 1203, 46, 0, 3394): {hv}")

    base = score(tune_rows, "ours", {}, settings)
    fixed = {"ours": ({}, "production"), "temporal_next": ({}, "brain tolerances, not tuned"),
             "temporal_either": ({}, "brain tolerances, not tuned")}
    picks, picks_v1, grids = dict(fixed), dict(fixed), {}
    for v in ("ours_wide", "ours_range", "mutual", "mutual_ratio", "pnp_sqpnp", "pnp_magsac"):
        p, note, grid = tune(tune_rows, v, settings, base)
        picks_v1[v] = (p, note)
        p, note, grid = tune(tune_all, v, settings, base)
        picks[v], grids[v] = (p, note), grid
    RESULT.update(rows=rows, picks=picks, grids=grids, settings=settings, seq=seq)

    L = ["", "RELOCALIZATION UPGRADE: PnP, mutual matching, temporal consistency (tools/reloc_compare.py, "
         + time.strftime("%Y-%m-%d %H:%M %Z") + f", {time.time() - t0:.0f} s, OpenCV {cv2.__version__})",
         "Same keyframes, pairs and split as the fence fix. Variants are defined in the tool's docstring. Thresholds picked",
         "on tuning only (fr2_pioneer_slam + fr1_room); held-out = fr2_pioneer_360 + fr3_long_office_household.",
         f"Gates passed: 'ours' all 1518/2414, 70 wrong, 0/6500 false and held-out 1203/1601, 46 wrong, 0/3394 false;",
         f"keyframes with pixels equal rdc.keyframe on every 10th frame and own pixels match an independent ORB run",
         f"(max {max(gate_px.values()):.1e} px); PnP self-match on unquantised points is the identity link on {pnp_tested}",
         f"solves (worst {pnp_worst:.2f} of a 1e-3 deg / 1e-3 m tolerance); with cm-quantised radio points the self-match",
         f"is off by up to {pnp_quant[0]:.2f} deg yaw, {pnp_quant[1]:.3f} m, {pnp_quant[2]:.2f} deg tilt (quantisation, not gated).",
         "OpenCV build: opencv-python 5.0.0 without contrib: cv2.xfeatures2d is absent, so BEBLID and TEBLID cannot be",
         "tested (not installed, per the brief). ORB is the only 32-byte binary descriptor in this build.",
         f"Within-sequence negatives: committed sample {sum(r['kind'] == 'neg' and r['name'] != 'cross' for r in rows)}, every one "
         f"{sum(r['kind'] == 'neg' and r['name'] != 'cross' for r in every)} (tuning {len(tune_xneg)} with tuning cross pairs, held-out {len(held_xneg)} with held-out cross pairs).",
         "Picked parameters, v2 protocol (used in every table below), then v1:"]
    for v in VARIANTS:
        p, note = picks[v]
        L.append(f"    {v}: {p if p else '-'} ({note})" + (f"; v1 {picks_v1[v][0]} ({picks_v1[v][1]})" if picks_v1[v][0] != p else ""))
    L.append("  variant / subset            pos  recall (Wilson 95 % CI)  yaw med/p90 deg  trans med/p90 m  wrong/acc  "
             "false/neg maxneg  ms/match")
    for label, rs in (("held-out", held_rows), ("pooled", rows), ("tuning", tune_rows)):
        for v in VARIANTS:
            L.append(fmt(f"{v} {label}", score(rs, v, picks[v][0], settings)))
    for name in sorted(TUNING | HELD_OUT):
        sub = [r for r in rows if r["name"] == name]
        L.append(f"    {name}: " + "; ".join(
            f"{v} {s['k']}/{s['n']} {s['wrong']}w" for v, s in ((v, score(sub, v, picks[v][0], settings)) for v in VARIANTS)))
    held = {v: score(held_rows, v, picks[v][0], settings) for v in VARIANTS}
    held_x = {v: score(held_xneg, v, picks[v][0], settings) for v in VARIANTS}
    L.append("  Every within-sequence negative (plus the cross pairs of that split), false accepts and max inliers:")
    for v in VARIANTS:
        sx = score(tune_xneg, v, picks[v][0], settings)
        L.append(f"    {v}: held-out {held_x[v]['false']}/{held_x[v]['neg']} false, maxneg {held_x[v]['maxneg']}; "
                 f"tuning {sx['false']}/{sx['neg']} false, maxneg {sx['maxneg']}")
    for v in VARIANTS:
        if picks_v1[v][0] != picks[v][0]:
            s1, sx1 = score(held_rows, v, picks_v1[v][0], settings), score(held_xneg, v, picks_v1[v][0], settings)
            st1 = score(tune_xneg, v, picks_v1[v][0], settings)
            L.append(f"    v1 pick {v} {picks_v1[v][0]}: held-out {s1['k']}/{s1['n']} acc, {s1['wrong']} wrong, {s1['false']}/{s1['neg']} "
                     f"false; every held-out negative {sx1['false']}/{sx1['neg']} false (maxneg {sx1['maxneg']}); every tuning negative "
                     f"{st1['false']}/{st1['neg']} false (maxneg {st1['maxneg']})")
    L += extra_lines(rows, tune_rows, held_rows, picks, settings)
    L += mechanism_lines(held_rows, picks, settings, seq)
    L += fixed_mount_lines(picks, settings)
    n_gate, worst_gate = shipped_gate(seq, rows, picks, settings)
    L.append(f"  Shipped-path gate passed: place.match_keyframes with place_fit=pnp_sqpnp at the picked thresholds gives the "
             f"harness decision on all {n_gate} pairs (largest pose difference {worst_gate:.1e}); place_fit=ransac_rigid gives "
             f"ours on every pair; a decoded payload on the own side falls back to the planar fit.")
    cfg = place.load_settings()
    p = picks["pnp_sqpnp"][0]
    same = (cfg["place_pnp_reproj_px"], cfg["place_pnp_min_inliers"], cfg["place_pnp_max_tilt_deg"], cfg["place_pnp_max_dz_m"]) \
        == (p["thr"], p["min_inl"], p["tilt"], p["dz"])
    if cfg["place_fit"] == "pnp_sqpnp" and not same:
        raise SystemExit(f"config/perception.json PnP thresholds differ from the tuned {p}")
    L.append(f"  config/perception.json place_fit = {cfg['place_fit']}; its place_pnp_* thresholds "
             + ("equal the v2 pick." if same else f"differ from the v2 pick {p}."))
    winners = [v for v in VARIANTS[1:] if held[v]["k"] >= held["ours"]["k"] and held[v]["wrong"] < held["ours"]["wrong"]
               and held[v]["false"] == 0 and held_x[v]["false"] == 0]
    RESULT["winners"] = winners
    L.append(f"  Ship rule on held-out (recall >= ours, fewer wrong poses, 0 false accepts incl. every held-out negative): "
             + (f"PASS for {', '.join(winners)}" if winners else "FAIL for every variant")
             + f" (ours {held['ours']['k']} acc / {held['ours']['wrong']} wrong / {held['ours']['false']} false; "
             + "; ".join(f"{v} {held[v]['k']} / {held[v]['wrong']} / {held[v]['false']}" for v in VARIANTS[1:]) + ").")
    text = "\n".join(L) + "\n"
    print(text)
    if "--no-write" not in sys.argv[1:]:
        with open(rdc.EVIDENCE, "a") as fh:
            fh.write(text)


RESULT: dict = {}


def own_keyframe(k):
    """The own keyframe as place.make_keyframe stores it: payload values plus own pixels, camera and mount."""
    K = k["K"]
    return place.Keyframe(k["kf"].desc, k["kf"].xyh, k["px"], (K[0, 0], K[1, 1], K[0, 2], K[1, 2]), k["B"],
                          np.array([0.0, 0.0, k["h"]]))


def shipped_gate(seq, rows, picks, settings):
    """place.match_keyframes must give the harness decisions on every pair: place_fit=pnp_sqpnp at the picked
    thresholds equals pnp_sqpnp here (pose within 1e-6), place_fit=ransac_rigid equals ours, and a partner payload
    without pixels on the own side falls back to the planar fit even with the switch on."""
    p = picks["pnp_sqpnp"][0]
    s_pnp = dict(settings, place_fit="pnp_sqpnp", place_pnp_reproj_px=p["thr"], place_pnp_min_inliers=p["min_inl"],
                 place_pnp_max_tilt_deg=p["tilt"], place_pnp_max_dz_m=p["dz"])
    s_rig = dict(settings, place_fit="ransac_rigid")
    bad = {"pnp": 0, "rigid": 0, "fallback": 0, "fit_label": 0}
    worst = 0.0
    for r in rows:
        a = seq[r["sa"]][1][r["i"]]["kf"]
        k = seq[r["sb"]][1][r["j"]]
        own = own_keyframe(k)
        m = place.match_keyframes(a, [(0, own)], s_pnp)
        ok, _, f = decision(r, "pnp_sqpnp", p, settings)
        bad["pnp"] += m["matched"] != ok
        if m["matched"] and ok:
            worst = max(worst, abs(rdc.wrap(m["dyaw_deg"] - f["yaw"])), abs(m["dx_m"] - f["t"][0]), abs(m["dy_m"] - f["t"][1]))
            bad["fit_label"] += m["fit"] != "pnp_sqpnp"
        m2 = place.match_keyframes(a, [(0, own)], s_rig)
        bad["rigid"] += m2["matched"] != ours_ok(r["ours"], settings)
        bad["fit_label"] += m2["matched"] and m2["fit"] != "ransac_rigid"
        m3 = place.match_keyframes(a, [(0, k["kf"])], s_pnp)
        bad["fallback"] += m3["matched"] != ours_ok(r["ours"], settings)
    if any(bad.values()) or worst > 1e-6:
        raise SystemExit(f"place.match_keyframes disagrees with the harness: {bad}, worst pose difference {worst}")
    return len(rows), worst


def fixed_mount_lines(picks, settings):
    """Pioneer sequences with one fixed mount (sequence-median tilt and height) for every frame: no per-frame truth
    enters the features or the PnP conversion. Positives as before; negatives from rdc.pairs_for with a fresh seed."""
    L = []
    for name in sorted(rdc.PIONEER):
        frames = rdc.load_sequence(name)
        K = rdc.INTRINSICS[name.split("_")[2]]
        R, h, pd, rd = rdc.fixed_mount_for(frames)
        kx = []
        for f in frames:
            k = keyframe_px(dict(f, R=R, c=np.array([f["c"][0], f["c"][1], h])), K, settings)
            if k is not None:
                del k["bgr"]
            kx.append(k)
        seq = {name: (frames, kx)}
        pos, neg = rdc.pairs_for(frames, np.random.default_rng(rdc.SEED))
        recs = [{"split": "fixed", "kind": kind, "name": name, "sa": name, "sb": name, "i": i, "j": j,
                 "truth": rdc.truth_relative(frames[i], frames[j]) if kind == "pos" else None}
                for kind, lst in (("pos", pos), ("neg", neg)) for i, j in lst if kx[i] is not None and kx[j] is not None]
        rows = []
        for r in recs:
            a, b = kx[r["i"]], kx[r["j"]]
            knn = height_filter(place.good_matches(a["kf"].desc, b["kf"].desc, settings["place_ratio_test"]), a["kf"], b["kf"], settings)
            r["ours"] = rigid_fit(a["kf"], b["kf"], knn, settings)
            r["t_knn"] = r["t_ours"] = 0.0
            q = picks["ours_wide"][0]
            r[("ours_wide", q["thr"])] = rigid_fit(a["kf"], b["kf"], knn, settings, q["thr"])
            r[("t", "ours_wide", q["thr"])] = 0.0
            for v, flag in PNP_FLAGS.items():
                thr = picks[v][0]["thr"]
                r[(v, thr)] = pnp_fit(a["kf"], b, knn, flag, thr, settings)
                r[("t", v, thr)] = 0.0
            rows.append(r)
        L.append(f"    fixed mount {name} (pitch down {pd:.1f} deg, roll {rd:.1f} deg, height {h:.2f} m): " + "; ".join(
            f"{v} {s_['k']}/{s_['n']} acc {s_['wrong']}w, {s_['false']}/{s_['neg']} false, maxneg {s_['maxneg']}"
            for v, s_ in ((v, score(rows, v, picks[v][0], settings)) for v in ("ours", "ours_wide", "pnp_sqpnp", "pnp_magsac"))))
    return L


def mechanism_lines(held_rows, picks, settings, seq):
    """Held-out positives: why ours is wrong where PnP is right. Under the TRUE planar link, how many matches lie
    within place_ransac_threshold_m in the planar (own-depth) test, and how far away the own points are."""
    p = picks["pnp_sqpnp"][0]
    groups = {"ours wrong, pnp_sqpnp right": [], "both right": []}
    thr = settings["place_ransac_threshold_m"]
    for r in held_rows:
        if r["kind"] != "pos":
            continue
        ok_o, _, fo = decision(r, "ours", {}, settings)
        ok_p, _, fp = decision(r, "pnp_sqpnp", p, settings)
        if not (ok_o and ok_p):
            continue
        eo, ep = err_of(r, fo), err_of(r, fp)
        wo, wp = eo[0] > rdc.GROSS_YAW_DEG or eo[1] > rdc.GROSS_T_M, ep[0] > rdc.GROSS_YAW_DEG or ep[1] > rdc.GROSS_T_M
        if wp:
            continue
        a, b = seq[r["sa"]][1][r["i"]]["kf"], seq[r["sb"]][1][r["j"]]["kf"]
        knn = height_filter(place.good_matches(a.desc, b.desc, settings["place_ratio_test"]), a, b, settings)
        yaw, tt = r["truth"]
        res = np.hypot(*(place._apply(math.radians(yaw), tt, a.xyh[knn[:, 0], :2]) - b.xyh[knn[:, 1], :2]).T)
        groups["ours wrong, pnp_sqpnp right" if wo else "both right"].append(
            (int((res <= thr).sum()), fo["n"], fp["n"], float(np.median(np.hypot(*b.xyh[knn[:, 1], :2].T))),
             float(np.median(res[res <= 1.0])) if (res <= 1.0).any() else float("nan")))
    L = []
    for g, v in groups.items():
        m = np.array(v).reshape(-1, 5)
        if not len(m):
            continue
        L.append(f"    held-out {g} (n={len(m)}): matches within {thr} m of the TRUE planar link, median {np.median(m[:, 0]):.0f} "
                 f"(ours needs {settings['place_min_inliers']}); ours inliers median {np.median(m[:, 1]):.0f}; pnp inliers median "
                 f"{np.median(m[:, 2]):.0f}; own match range median {np.median(m[:, 3]):.1f} m; planar residual under truth "
                 f"(matches within 1 m) median {np.nanmedian(m[:, 4]):.3f} m")
    return L


def extra_lines(rows, tune_rows, held_rows, picks, settings):
    """Diagnostics: PnP ablations on held-out, and what PnP does on the pairs ours gets wrong."""
    L = []
    for v in PNP_FLAGS:
        p = picks[v][0]
        for lab, q in (("no tilt/dz gates", dict(p, tilt=math.inf, dz=math.inf)), ("tilt gate only", dict(p, dz=math.inf)),
                       ("dz gate only", dict(p, tilt=math.inf))):
            s = score(held_rows, v, q, settings)
            L.append(f"    held-out {v} {lab}: {s['k']}/{s['n']} acc, {s['wrong']} wrong, {s['false']} false, maxneg {s['maxneg']}")
    p = picks["pnp_sqpnp"][0]
    for m in (12, 15):
        for lab, rs in (("held-out", held_rows), ("tuning", tune_rows)):
            s = score(rs, "pnp_sqpnp", dict(p, min_inl=m), settings)
            L.append(f"    {lab} pnp_sqpnp at min inliers {m}: {s['k']}/{s['n']} acc, {s['wrong']} wrong, {s['false']} false, maxneg {s['maxneg']}")
    # On held-out positives ours accepts with a wrong pose: what does each PnP variant say?
    for v in PNP_FLAGS:
        p = picks[v][0]
        hit = [r for r in held_rows if r["kind"] == "pos" and ours_ok(r["ours"], settings)
               and (lambda e: e[0] > rdc.GROSS_YAW_DEG or e[1] > rdc.GROSS_T_M)(err_of(r, r["ours"]))]
        rej = sum(not decision(r, v, p, settings)[0] for r in hit)
        right = sum(decision(r, v, p, settings)[0] and (lambda e: e[0] <= rdc.GROSS_YAW_DEG and e[1] <= rdc.GROSS_T_M)(
            err_of(r, decision(r, v, p, settings)[2])) for r in hit)
        L.append(f"    held-out pairs where ours accepts a wrong pose (n={len(hit)}): {v} rejects {rej}, accepts with a right pose {right}")
    return L


if __name__ == "__main__":
    main()
