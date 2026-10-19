"""C2 texture gate: OpenCV ORB keypoints on every sensor frame of a recording.

    brain/.venv/bin/python tools/orb_gate.py data/recordings/<run_id> [--every 1]

Pass: median >= 400 keypoints per frame for every robot, and no run of frames below 150
(BUILD_ORDER C2). Prints per-robot statistics and every low stretch with its time span, and writes
<recording>/../../evidence/<run_id>_orb_gate.json.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

MEDIAN_MIN = 400
STRETCH_MIN = 150


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", type=Path)
    ap.add_argument("--every", type=int, default=1)
    args = ap.parse_args()
    meta = json.loads((args.recording / "meta.json").read_text())
    orb = cv2.ORB_create(nfeatures=1250)   # same feature budget as camera.yaml
    result, ok = {"run_id": meta["run_id"], "robots": {}}, True
    for robot in meta["robots"]:
        rows = [l.split() for l in (args.recording / robot / "rgb.txt").read_text().splitlines() if l and not l.startswith("#")]
        times, counts = [], []
        for t, rel in rows[:: args.every]:
            g = cv2.imread(str(args.recording / robot / rel), cv2.IMREAD_GRAYSCALE)
            times.append(float(t))
            counts.append(len(orb.detect(g, None)))
        counts = np.asarray(counts)
        low = counts < STRETCH_MIN
        stretches, start = [], None
        for i, flag in enumerate(low):
            if flag and start is None:
                start = i
            if (not flag or i == len(low) - 1) and start is not None:
                end = i if flag else i - 1
                stretches.append([times[start], times[end]])
                start = None
        median = float(np.median(counts))
        robot_ok = median >= MEDIAN_MIN and not stretches
        ok &= robot_ok
        result["robots"][robot] = {"frames": len(counts), "median": median, "p5": float(np.percentile(counts, 5)),
                                   "min": int(counts.min()), "low_stretches_s": stretches, "pass": robot_ok}
        print(f"{robot}: {len(counts)} frames, median {median:.0f}, p5 {np.percentile(counts, 5):.0f}, min {counts.min()}, "
              f"stretches below {STRETCH_MIN}: {stretches or 'none'} -> {'PASS' if robot_ok else 'FAIL'}")
    result["pass"] = ok
    out = args.recording.parent.parent / "evidence" / f"{meta['run_id']}_orb_gate.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
