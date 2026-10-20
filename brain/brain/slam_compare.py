"""OpenCV 5.0.0 vs OpenCV 4.5.4 builds of the same ORB-SLAM3 on the same recording, with repeats.

ORB-SLAM3 is multithreaded and not bit-reproducible, so a difference only counts if it exceeds the run-to-run spread.

    uv run python -m brain.slam_compare ../data/recordings/unity_clear_seed7 robot_A --repeats 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from brain.evaluator import compute_ate
from brain.oracle import load_trajectory
from brain.slam_runner import run_orbslam3
from brain.trajectory import read_tum

IMAGES = ("smoke-rescue-orbslam3", "smoke-rescue-orbslam3-cv5")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("recording", type=Path)
    ap.add_argument("robot")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("../data/runs/cv5_compare_v2"))
    a = ap.parse_args()
    truth = load_trajectory(a.recording, a.robot)
    report_path = a.out / f"{a.recording.name}_{a.robot}.json"
    rows = json.loads(report_path.read_text())["runs"] if report_path.exists() else []
    for rep in range(1, a.repeats + 1):
        for image in IMAGES:                                    # interleaved, so machine load affects both alike
            if any(r["image"] == image and r["repeat"] == rep for r in rows):
                continue
            out = a.out / image / a.robot / f"repeat{rep}"
            t0 = time.perf_counter()
            run_orbslam3(a.recording / a.robot, out, image=image)
            wall = time.perf_counter() - t0
            est = read_tum(out / "CameraTrajectory.txt")
            ate = compute_ate(truth, est)
            rows.append({"image": image, "repeat": rep, "ate_rmse_m": ate["rmse"] if "rmse" in ate else ate.get("ate_rmse_m"),
                         "matched_poses": ate.get("matched", ate.get("matched_poses")), "frames_exported": int(len(est)),
                         "wall_s": round(wall, 1), "out": str(out)})
            a.out.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps({"recording": str(a.recording), "robot": a.robot, "runs": rows}, indent=1))
            print(json.dumps(rows[-1]), flush=True)
    for image in IMAGES:
        v = [r["ate_rmse_m"] for r in rows if r["image"] == image]
        print(f"{image:28s} n={len(v)} ATE mean {statistics.mean(v):.3f} m  min {min(v):.3f}  max {max(v):.3f}")


if __name__ == "__main__":
    main()
