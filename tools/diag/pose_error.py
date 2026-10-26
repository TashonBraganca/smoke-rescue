#!/usr/bin/env python3
"""Pose error of each robot's brain estimate against the simulator's ground truth, in the shared gate frame.

    brain/.venv/bin/python tools/diag/pose_error.py data/recordings/<run> [gate_x gate_z]

Ground truth is the camera optical pose in the house frame (oracle/robot_*_groundtruth.txt); the body is 0.25 m behind
the camera along its heading. The gate frame puts robot A's launch pad at the origin facing +y with x to its right, so
gate = (house_x - gate_x, house_z - gate_z). Defaults are read from the scenario in meta.json when possible.
"""
import bisect, json, math, sys
from pathlib import Path
import numpy as np

run = Path(sys.argv[1]); gx, gz = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (None, None)
if gx is None:
    meta = json.loads((run / "meta.json").read_text()); sc = json.loads((Path("config") / meta.get("scenario_file", meta.get("scenario", ""))).read_text()) if meta.get("scenario_file") or meta.get("scenario") else None
    a = next(r for r in sc["robots"] if r["id"] == "robot_A"); gx, gz = a["spawn"]["at"]
for rid in ("robot_A", "robot_B"):
    gt = [l.split() for l in open(run / "oracle" / f"{rid}_groundtruth.txt") if l[0].isdigit()]
    ts = [float(r[0]) for r in gt]
    errs = []
    for line in open(run / rid / "brain_state.jsonl"):
        s = json.loads(line); i = bisect.bisect_left(ts, s["t"])
        if i >= len(ts): break
        _, tx, ty, tz, qx, qy, qz, qw = map(float, gt[i])
        # camera forward (+z of the optical frame) in the house frame
        fx = 2 * (qx * qz + qw * qy); fz = 1 - 2 * (qx * qx + qy * qy); n = math.hypot(fx, fz) or 1
        bx, bz = tx - .25 * fx / n, tz - .25 * fz / n
        errs.append((s["t"], math.hypot(s["x"] - (bx - gx), s["y"] - (bz - gz)), s["x"], s["y"], bx - gx, bz - gz))
    e = np.array([r[1] for r in errs])
    worst = max(errs, key=lambda r: r[1])
    print(f"{rid}: n={len(e)} median {np.median(e):.2f} m  p90 {np.percentile(e, 90):.2f} m  max {e.max():.2f} m at t={worst[0]:.1f} "
          f"(brain {worst[2]:.2f},{worst[3]:.2f} true {worst[4]:.2f},{worst[5]:.2f})  first {errs[0][1]:.2f} m")
    for t0 in range(0, int(errs[-1][0]) + 1, 30):
        w = [r[1] for r in errs if t0 <= r[0] < t0 + 30]
        if w: print(f"   {t0:4d}-{t0+30:<4d} s  median {np.median(w):.2f}  max {max(w):.2f}")
