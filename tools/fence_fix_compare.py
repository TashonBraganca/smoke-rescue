#!/usr/bin/env python3
"""Can an OpenCV-based 2D fit beat place.py's ransac_rigid on the TUM pairs of tools/real_data_check.py?

The new fit (cv2_unique): a hypothesis from cv2.estimateAffinePartial2D(method=cv2.RANSAC), a strict rigid refit on its
inliers with place.fit_rigid, then two gates against repetitive structure:
  * uniqueness: remove the winner's inliers, fit the rest again the same way, and reject when the runner-up's inlier
    count divided by the winner's is above a ratio threshold (a fence of identical panels gives near-equal rivals);
  * spread: the inliers' planar body-frame points (x right, y forward, metres; the radio payload carries no image
    coordinates) must cover a convex hull (cv2.convexHull + cv2.contourArea) of at least a minimum area. The smaller of
    the partner-side and own-side areas is used.

The two thresholds are picked on the TUNING sequences only and then reported on the HELD-OUT sequences, which were not
looked at while choosing. Match sets (kNN + ratio test + one-to-one + height filter, exactly as place.match_keyframes)
are computed once per pair and every fit runs on the same sets. The 'ours' rows must reproduce the earlier
real_data_check numbers, and when cv2_unique ships in place.py the shipped code path is re-run through
place.match_keyframes and must give the same accept decisions; both are checked and the run halts if they do not.

    nice -n 10 brain/.venv/bin/python -I tools/fence_fix_compare.py [--no-write]
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
from rigid_fit_compare import cv2_rigid  # noqa: E402

TUNING = {"rgbd_dataset_freiburg2_pioneer_slam", "rgbd_dataset_freiburg1_room"}
HELD_OUT = {"rgbd_dataset_freiburg2_pioneer_360", "rgbd_dataset_freiburg3_long_office_household"}
RATIO_GRID = [round(x, 2) for x in np.arange(0.30, 1.0001, 0.05)] + [math.inf]
AREA_GRID = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0]


def hull_area(pts):
    if len(pts) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(np.asarray(pts, np.float32).reshape(-1, 1, 2))))


def unique_fit(p, q, settings):
    """Raw cv2_unique quantities for one match set: the winner (angle, t, mask, rmse) or None, the runner-up's
    inlier count, and the inlier hull area in m^2. place.cv2_unique_rigid applies the thresholds once shipped."""
    if hasattr(place, "cv2_unique_parts"):
        return place.cv2_unique_parts(p, q, settings)
    win = _cv2_refit(p, q, settings)
    if win is None:
        return None, 0, 0.0
    mask = win[2]
    rest = ~mask
    run = _cv2_refit(p[rest], q[rest], settings) if rest.sum() >= 2 else None
    runner = int(run[2].sum()) if run is not None else 0
    return win, runner, min(hull_area(p[mask]), hull_area(q[mask]))


def _cv2_refit(p, q, settings):
    """cv2.estimateAffinePartial2D RANSAC hypothesis, then rigid refits on its inliers (place.fit_rigid) until the
    inlier set at place_ransac_threshold_m stops changing. Same return as place.ransac_rigid."""
    if len(p) < 2:
        return None
    thr = float(settings["place_ransac_threshold_m"])
    cv2.setRNGSeed(int(settings["place_ransac_seed"]))
    M, inl = cv2.estimateAffinePartial2D(np.asarray(p, np.float32).reshape(-1, 1, 2), np.asarray(q, np.float32).reshape(-1, 1, 2),
                                         method=cv2.RANSAC, ransacReprojThreshold=thr,
                                         maxIters=int(settings["place_ransac_iterations"]), confidence=0.99)
    if M is None or inl is None or not np.isfinite(M).all():
        return None
    mask = inl.ravel().astype(bool)
    for _ in range(int(settings["place_refine_iterations"]) + 1):
        if mask.sum() < 2:
            return None
        a, t = place.fit_rigid(p[mask], q[mask])
        err = np.hypot(*(place._apply(a, t, p) - q).T)
        new = err <= thr
        if np.array_equal(new, mask):
            break
        mask = new
    if not mask.any():
        return None
    return a, t, mask, float(np.sqrt(np.mean(err[mask] ** 2)))


# ---------------------------------------------------------------------------------------------------------- pairs
def match_set(a, b, settings):
    """The height-consistent one-to-one match set place.match_keyframes builds for partner a against own b."""
    pairs = place.good_matches(a.desc, b.desc, settings["place_ratio_test"])
    if len(pairs) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2))
    dh = np.abs(a.xyh[pairs[:, 0], 2] - b.xyh[pairs[:, 1], 2])
    pairs = pairs[dh <= settings["place_height_tolerance_m"]]
    return a.xyh[pairs[:, 0], :2], b.xyh[pairs[:, 1], :2]


def collect(settings):
    """Every positive and negative pair of rigid_fit_compare, in the same seeded order: (split, kind, truth, a, b)."""
    rng = np.random.default_rng(rdc.SEED)
    kfs_by, out = {}, []
    for name in rdc.SEQUENCES:
        frames = rdc.load_sequence(name)
        K = rdc.INTRINSICS[name.split("_")[2]]
        kfs = [rdc.keyframe(f, K, settings) for f in frames]
        kfs_by[name] = kfs
        pos, neg = rdc.pairs_for(frames, rng)
        split = "held" if name in HELD_OUT else "tune"
        for i, j in pos:
            if kfs[i] is not None and kfs[j] is not None:
                out.append((split, "pos", name, rdc.truth_relative(frames[i], frames[j]), kfs[i], kfs[j]))
        for i, j in neg:
            if kfs[i] is not None and kfs[j] is not None:
                out.append((split, "neg", name, None, kfs[i], kfs[j]))
    names = list(kfs_by)
    combos = [(a, b) for a in names for b in names if a != b and not (a in rdc.PIONEER and b in rdc.PIONEER)]
    for _ in range(rdc.CROSS_N):   # the draw of rdc.run_cross, so the cross pairs are the same ones
        a, b = combos[int(rng.integers(len(combos)))]
        ka = [k for k in kfs_by[a] if k is not None]
        kb = [k for k in kfs_by[b] if k is not None]
        x, y = ka[int(rng.integers(len(ka)))], kb[int(rng.integers(len(kb)))]
        split = "held" if {a, b} <= HELD_OUT else "tune" if {a, b} <= TUNING else "mixed"
        out.append((split, "neg", "cross", None, x, y))
    return out


def evaluate(records, settings):
    """Per pair: match set once (timed), then each fit (timed) with its raw outputs."""
    rows = []
    for split, kind, name, truth, a, b in records:
        t0 = time.perf_counter()
        p, q = match_set(a, b, settings)
        t_match = time.perf_counter() - t0
        r = {"split": split, "kind": kind, "name": name, "truth": truth, "t_match": t_match, "a": a, "b": b}
        for v, fn in (("ours", place.ransac_rigid), ("cv2_ransac", lambda p, q, s: cv2_rigid(p, q, s, cv2.RANSAC))):
            t0 = time.perf_counter()
            fit = fn(p, q, settings) if len(p) >= 2 else None
            r["t_" + v] = time.perf_counter() - t0
            r[v] = fit
        t0 = time.perf_counter()
        r["cv2_unique"], r["runner"], r["area"] = unique_fit(p, q, settings) if len(p) >= 2 else (None, 0, 0.0)
        r["t_cv2_unique"] = time.perf_counter() - t0
        rows.append(r)
    return rows


def decide(r, variant, settings, ratio=math.inf, area=0.0):
    """(accepted, inliers, pose error (yaw deg, trans m) or None)."""
    fit = r[variant]
    if fit is None:
        return False, 0, None
    angle, t, mask, rmse = fit
    n = int(mask.sum())
    ok = n >= settings["place_min_inliers"] and rmse <= settings["place_max_rmse_m"]
    if variant == "cv2_unique":
        ok = ok and r["runner"] / n <= ratio and r["area"] >= area
    err = None
    if r["truth"] is not None:
        yaw, tt = r["truth"]
        err = (abs(rdc.wrap(math.degrees(angle) - yaw)), float(np.hypot(t[0] - tt[0], t[1] - tt[1])))
    return ok, n, err


def score(rows, variant, settings, ratio=math.inf, area=0.0):
    pos = [r for r in rows if r["kind"] == "pos"]
    neg = [r for r in rows if r["kind"] == "neg"]
    acc = [e for ok, _, e in (decide(r, variant, settings, ratio, area) for r in pos) if ok]
    nd = [decide(r, variant, settings, ratio, area) for r in neg]
    wrong = sum(e[0] > rdc.GROSS_YAW_DEG or e[1] > rdc.GROSS_T_M for e in acc)
    ms = 1000 * np.mean([r["t_match"] + r["t_" + variant] for r in rows]) if rows else float("nan")
    return {"n": len(pos), "k": len(acc), "yaw": [e[0] for e in acc], "trans": [e[1] for e in acc], "wrong": wrong,
            "neg": len(neg), "false": sum(ok for ok, _, _ in nd), "maxneg": max([n for _, n, _ in nd], default=0), "ms": ms}


def fmt(label, s):
    lo, hi = rdc.wilson(s["k"], s["n"])
    return (f"  {label:<26} {s['n']:>5} {100 * s['k'] / s['n']:5.1f} % ({100 * lo:4.1f}-{100 * hi:4.1f}) "
            f"{rdc.pct(s['yaw'], 50):5.2f}/{rdc.pct(s['yaw'], 90):5.2f}  {rdc.pct(s['trans'], 50):.3f}/{rdc.pct(s['trans'], 90):.3f}  "
            f"{s['wrong']:>4}/{s['k']:<5} {s['false']:>3}/{s['neg']:<5} {s['maxneg']:>3}  {s['ms']:6.2f}")


def tune(rows, settings):
    """Grid over (ratio, area) on tuning rows: 0 false accepts and recall >= ours are required; fewest wrong poses
    wins, then most recall, then the loosest thresholds."""
    base = score(rows, "ours", settings)
    grid = []
    for ratio in RATIO_GRID:
        for area in AREA_GRID:
            s = score(rows, "cv2_unique", settings, ratio, area)
            grid.append((ratio, area, s))
    ok = [g for g in grid if g[2]["false"] == 0 and g[2]["k"] >= base["k"]]
    pick = min(ok, key=lambda g: (g[2]["wrong"], -g[2]["k"], -g[0], g[1])) if ok else None
    return base, grid, pick


def main():
    settings = dict(place.load_settings())
    t0 = time.time()
    rows = evaluate(collect(settings), settings)
    tune_rows = [r for r in rows if r["split"] == "tune"]
    held_rows = [r for r in rows if r["split"] == "held"]

    # Gate: 'ours' on all rows must reproduce the committed real_data_check / rigid_fit_compare numbers.
    allv = score(rows, "ours", settings)
    if (allv["n"], allv["k"], allv["wrong"], allv["false"], allv["neg"]) != (2414, 1518, 70, 0, 6500):
        raise SystemExit(f"harness does not reproduce ours (2414, 1518, 70, 0, 6500): {allv}")
    cvr = score(rows, "cv2_ransac", settings)
    if (cvr["k"], cvr["wrong"], cvr["false"]) != (1594, 126, 3):
        raise SystemExit(f"harness does not reproduce cv2_ransac (1594, 126, 3): {cvr}")

    base, grid, pick = tune(tune_rows, settings)
    if pick is None:
        best = min(grid, key=lambda g: (g[2]["false"], -g[2]["k"]))
        ratio, area = best[0], best[1]
        note = "no tuning point kept recall >= ours with 0 false accepts; reporting the highest-recall zero-false point"
    else:
        ratio, area = pick[0], pick[1]
        note = "picked on tuning: fewest wrong poses with 0 false accepts and recall >= ours"

    shipped = hasattr(place, "cv2_unique_parts")
    if shipped:   # the shipped switch must give the same decisions as the harness at the config thresholds
        s = dict(settings, place_rigid_fit="cv2_unique")
        if (s["place_unique_max_ratio"], s["place_min_inlier_area_m2"]) != (ratio, area):
            raise SystemExit(f"config thresholds {s['place_unique_max_ratio']}, {s['place_min_inlier_area_m2']} "
                             f"differ from the tuned {ratio}, {area}")
        bad = 0
        for r in rows:
            m = place.match_keyframes(r["a"], [(0, r["b"])], s)
            bad += m["matched"] != decide(r, "cv2_unique", settings, ratio, area)[0]
        if bad:
            raise SystemExit(f"place.match_keyframes with place_rigid_fit=cv2_unique disagrees with the harness on {bad} pairs")

    L = ["", "FENCE FIX: OpenCV fit with uniqueness and spread gates (tools/fence_fix_compare.py, "
         + time.strftime("%Y-%m-%d %H:%M %Z") + f", {time.time() - t0:.0f} s)",
         "cv2_unique = cv2.estimateAffinePartial2D(RANSAC, 0.1 m, 500 iterations, seed 0) hypothesis, rigid refit on its",
         "inliers with place.fit_rigid until the inlier set is stable; uniqueness: the same fit on the matches left after",
         "removing the winner's inliers gives a runner-up, reject when runner-up inliers / winner inliers > ratio; spread:",
         "the smaller of the partner-side and own-side convex hull areas (cv2.convexHull + cv2.contourArea) of the",
         "inliers' planar body points must be >= area. Planar body points, not image points: the radio payload has no",
         "image coordinates. Same match sets and accept rule (>= 10 inliers, rmse <= 0.06 m) for every variant.",
         "Tuning split: fr2_pioneer_slam + fr1_room (and cross-sequence negatives between them). Held-out split:",
         "fr2_pioneer_360 + fr3_long_office_household (and cross negatives between them). Cross negatives that mix a",
         "tuning and a held-out sequence count only in the pooled rows.",
         f"Grid: ratio {RATIO_GRID[0]} to {RATIO_GRID[-2]} step 0.05 plus off; area {AREA_GRID} m^2.",
         f"Thresholds: ratio {ratio}, area {area} m^2 ({note}).",
         "Harness gate passed: 'ours' all = 1518/2414 accepted, 70 wrong, 0/6500 false; cv2_ransac all = 1594, 126, 3"
         " (the committed numbers)." + (" Shipped-path gate passed: place.match_keyframes with place_rigid_fit"
                                        "=cv2_unique gives the same accept decision on every pair." if shipped else ""),
         "ms/match = kNN match set + fit, per pair, averaged; machine was shared with other jobs, so treat it as rough.",
         "  variant / subset            pos  recall (Wilson 95 % CI)  yaw med/p90 deg  trans med/p90 m  wrong/acc  "
         "false/neg maxneg  ms/match"]
    for label, rs in (("held-out", held_rows), ("tuning", tune_rows), ("pooled", rows)):
        for v in ("ours", "cv2_ransac", "cv2_unique"):
            kw = {"ratio": ratio, "area": area} if v == "cv2_unique" else {}
            L.append(fmt(f"{v} {label}", score(rs, v, settings, **kw)))
    held = {v: score(held_rows, v, settings, **({"ratio": ratio, "area": area} if v == "cv2_unique" else {}))
            for v in ("ours", "cv2_unique")}
    for name in sorted(TUNING | HELD_OUT):
        sub = [r for r in rows if r["name"] == name]
        s0 = score(sub, "ours", settings); s1 = score(sub, "cv2_unique", settings, ratio, area)
        L.append(f"    {name}: ours {s0['k']}/{s0['n']} acc, {s0['wrong']} wrong; cv2_unique {s1['k']}/{s1['n']} acc, {s1['wrong']} wrong")
    # Gate ablations on held-out, so the effect of each gate is visible.
    for lab, kw in (("no gates", {}), ("uniqueness only", {"ratio": ratio}), ("spread only", {"area": area})):
        s = score(held_rows, "cv2_unique", settings, **kw)
        L.append(f"    held-out cv2_unique {lab}: {s['k']}/{s['n']} acc, {s['wrong']} wrong, {s['false']} false")
    near = sorted(grid, key=lambda g: (g[2]["false"] > 0, g[2]["wrong"], -g[2]["k"]))[:6]
    L.append("    tuning grid, best six by (false, wrong, -recall): " + "; ".join(
        f"r {g[0]} a {g[1]}: {g[2]['k']}/{g[2]['n']} {g[2]['wrong']}w {g[2]['false']}f" for g in near)
        + f"; ours tuning {base['k']}/{base['n']} {base['wrong']}w {base['false']}f")
    # Why the gates separate so little: on TUNING positives that pass the base rule, right vs wrong poses.
    groups = {"right": [], "wrong": []}
    for r in tune_rows:
        ok, n, e = decide(r, "cv2_unique", settings)
        if r["kind"] == "pos" and ok:
            mask = r["cv2_unique"][2]
            far = float(np.median(np.hypot(*match_set(r["a"], r["b"], settings)[1][mask].T)))
            groups["wrong" if e[0] > rdc.GROSS_YAW_DEG or e[1] > rdc.GROSS_T_M else "right"].append(
                (r["runner"] / n, r["area"], far))
    for lab, g in groups.items():
        g = np.array(g)
        L.append(f"    tuning {lab} poses before the gates (n={len(g)}): runner-up ratio p10/p50/p90 "
                 + "/".join(f"{x:.2f}" for x in np.percentile(g[:, 0], [10, 50, 90]))
                 + "; hull area m^2 " + "/".join(f"{x:.2f}" for x in np.percentile(g[:, 1], [10, 50, 90]))
                 + "; median inlier range m " + "/".join(f"{x:.1f}" for x in np.percentile(g[:, 2], [10, 50, 90])))
    win = (held["cv2_unique"]["k"] >= held["ours"]["k"] and held["cv2_unique"]["wrong"] < held["ours"]["wrong"]
           and held["cv2_unique"]["false"] == 0)
    L.append(f"  Ship rule on held-out (recall >= ours, fewer wrong poses, 0 false accepts): {'PASS' if win else 'FAIL'} "
             f"(ours {held['ours']['k']} acc / {held['ours']['wrong']} wrong / {held['ours']['false']} false; cv2_unique "
             f"{held['cv2_unique']['k']} / {held['cv2_unique']['wrong']} / {held['cv2_unique']['false']})."
             + ("" if win or shipped else " place.py is unchanged and no config switch was added."))
    text = "\n".join(L) + "\n"
    print(text)
    if sys.argv[1:] != ["--no-write"]:
        with open(rdc.EVIDENCE, "a") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
