#!/usr/bin/env python3
"""Compare place.py's own 2D rigid RANSAC with OpenCV 5's cv2.estimateAffinePartial2D (USAC_MAGSAC and RANSAC) on the
TUM pairs of tools/real_data_check.py. Same features, height filter and accept rule (place_min_inliers,
place_max_rmse_m); only the fit changes. place.py is not modified: the cv2 fits are swapped in for this run only.

    nice -n 10 brain/.venv/bin/python -I tools/rigid_fit_compare.py

Appends its table to data/evidence/real_data_check_25oct.txt.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_data_check as rdc  # noqa: E402
from real_data_check import place  # noqa: E402

import math  # noqa: E402

import cv2  # noqa: E402

VARIANTS = ["ours", "cv2_usac", "cv2_ransac"]
OURS = place.ransac_rigid


def cv2_rigid(p, q, settings, method):
    """OpenCV robust fit with a fixed RNG seed. RANSAC: cv2.estimateAffinePartial2D (rotation, translation, uniform
    scale), pose read from its matrix. USAC_MAGSAC: OpenCV 5.0 supports it only in cv2.estimateAffine2D (6 DOF), so its
    inlier set is refit as a rigid transform with place.fit_rigid. Either way inliers and RMSE are recomputed with the
    final transform at place_ransac_threshold_m. Same return as place.ransac_rigid."""
    if len(p) < (3 if method == cv2.USAC_MAGSAC else 2):
        return None
    thr = float(settings["place_ransac_threshold_m"])
    cv2.setRNGSeed(int(settings["place_ransac_seed"]))
    src, dst = np.asarray(p, np.float32).reshape(-1, 1, 2), np.asarray(q, np.float32).reshape(-1, 1, 2)
    kwargs = dict(method=method, ransacReprojThreshold=thr, maxIters=int(settings["place_ransac_iterations"]), confidence=0.99)
    if method == cv2.USAC_MAGSAC:
        M, inl = cv2.estimateAffine2D(src, dst, **kwargs)
        if M is None or inl is None or int(inl.sum()) < 2:
            return None
        sel = inl.ravel().astype(bool)
        angle, t = place.fit_rigid(p[sel], q[sel])
        c, s = math.cos(angle), math.sin(angle)
        pred = np.column_stack((c * p[:, 0] - s * p[:, 1] + t[0], s * p[:, 0] + c * p[:, 1] + t[1]))
    else:
        M, _ = cv2.estimateAffinePartial2D(src, dst, **kwargs)
        if M is None or not np.isfinite(M).all():
            return None
        angle, t = math.atan2(M[1, 0], M[0, 0]), M[:, 2].astype(np.float64)
        pred = p @ M[:, :2].T + M[:, 2]
    err = np.hypot(*(pred - q).T)
    mask = err <= thr
    if not mask.any():
        return None
    return angle, t, mask, float(np.sqrt(np.mean(err[mask] ** 2)))


FITS = {"ours": OURS,
        "cv2_usac": lambda p, q, s: cv2_rigid(p, q, s, cv2.USAC_MAGSAC),
        "cv2_ransac": lambda p, q, s: cv2_rigid(p, q, s, cv2.RANSAC)}
WRONG = (rdc.GROSS_YAW_DEG, rdc.GROSS_T_M)


def run(variant):
    settings = dict(place.load_settings())
    rng = np.random.default_rng(rdc.SEED)
    rdc.KF_CACHE.clear()
    timing = [0.0, 0]
    original = place.match_keyframes

    def timed(*a, **k):
        t0 = time.perf_counter()
        r = original(*a, **k)
        timing[0] += time.perf_counter() - t0
        timing[1] += 1
        return r

    place.match_keyframes = timed
    place.ransac_rigid = FITS[variant]   # match_keyframes looks ransac_rigid up in the module at call time
    try:
        results = [rdc.run_place(s, settings, rng) for s in rdc.SEQUENCES]
        cross = rdc.run_cross(settings, rng)
    finally:
        place.match_keyframes = original
        place.ransac_rigid = OURS
    return results, cross, 1000 * timing[0] / max(1, timing[1])


def row(label, rs, cross=None, ms=None):
    k = sum(r["pos_accepted"] for r in rs); n = sum(r["pos_n"] for r in rs)
    lo, hi = rdc.wilson(k, n)
    ye = sum((r["yaw_err"] for r in rs), []); te = sum((r["trans_err"] for r in rs), [])
    wrong = sum(len(r["gross"]) for r in rs)
    nn = sum(r["neg_n"] for r in rs) + (cross["n"] if cross else 0)
    nf = sum(r["neg_false"] for r in rs) + (cross["false"] if cross else 0)
    nmax = max([r["neg_max_inliers"] or 0 for r in rs] + ([cross["max_inliers"]] if cross else []))
    return (f"  {label:<26} {n:>5} {100 * k / n:5.1f} % ({100 * lo:4.1f}-{100 * hi:4.1f}) "
            f"{rdc.pct(ye, 50):5.2f}/{rdc.pct(ye, 90):5.2f}  {rdc.pct(te, 50):.3f}/{rdc.pct(te, 90):.3f}  "
            f"{wrong:>4}/{k:<5} {nf:>3}/{nn:<5} {nmax:>3}" + (f"  {ms:6.2f}" if ms is not None else ""))


def main():
    out = {v: run(v) for v in VARIANTS}
    head = ("  variant / subset            pos  recall (Wilson 95 % CI)  yaw med/p90 deg  trans med/p90 m  wrong/acc  "
            "false/neg maxneg  ms/match")
    L = ["", "RIGID FIT COMPARISON (tools/rigid_fit_compare.py, " + time.strftime("%Y-%m-%d %H:%M %Z") + ")",
         "Same TUM pairs, features, height filter and accept rule (min_inliers 10, max_rmse 0.06 m); only the 2D fit changes.",
         "cv2_ransac: cv2.estimateAffinePartial2D(method=cv2.RANSAC, ransacReprojThreshold=place_ransac_threshold_m,",
         "maxIters=place_ransac_iterations), pose read from the 4-DOF matrix (scale not forced to 1).",
         "cv2_usac: OpenCV 5.0 rejects USAC_MAGSAC in estimateAffinePartial2D ('Unknown or unsupported robust estimation",
         "method'), so this uses cv2.estimateAffine2D(method=cv2.USAC_MAGSAC, same threshold and iterations) for the",
         "inlier set and refits a rigid transform on it with place.fit_rigid. Both seed cv2.setRNGSeed(place_ransac_seed);",
         "inliers and RMSE are recomputed with the final transform at 0.1 m. 'ours' also applies",
         "place_ransac_min_baseline_m and its refit loop; the cv2 variants do not.",
         "Negatives = within-sequence > 4 m plus 2000 cross-sequence pairs. ms/match = place.match_keyframes wall time,",
         "one keyframe against one, averaged over every positive and negative call (includes kNN matching).",
         head]
    for v in VARIANTS:
        results, cross, ms = out[v]
        L.append(row(v + " all", results, cross, ms))
        L.append(row(v + " Pioneer hall", [r for r in results if r["name"] in rdc.PIONEER]))
        L.append(row(v + " offices", [r for r in results if r["name"] not in rdc.PIONEER]))
    text = "\n".join(L) + "\n"
    print(text)
    if sys.argv[1:] != ["--no-write"]:
        with open(rdc.EVIDENCE, "a") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
