"""Compare fresh, sequential RGB-D runs using the project's unchanged ATE evaluator.

Run from the repository root with PYTHONPATH=brain and brain/.venv/bin/python.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np

from brain.evaluator import compute_ate
from brain.oracle import load_trajectory


def compare(session: str, recordings: list[str], repeats: int) -> Path:
    if repeats < 2:
        raise ValueError("at least two repeats are required")
    if not session or Path(session).name != session:
        raise ValueError("session must be one path component")
    root = Path(__file__).resolve().parents[1]
    output = root / "data/runs/cv5_compare"
    report = output / f"comparison_{session}.json"
    if report.exists():
        raise FileExistsError(report)
    images = ["smoke-rescue-orbslam3", "smoke-rescue-orbslam3-cv5"]
    image_ids = {image: subprocess.check_output(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True
    ).strip() for image in images}
    results = []
    for recording in recordings:
        source = root / "data/recordings" / recording / "robot_A"
        truth = load_trajectory(source.parent, "robot_A")
        fingerprints = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                        for name in ("associations.txt", "camera.yaml")}
        for repeat in range(1, repeats + 1):
            for image in images:
                dest = output / image / "robot_A" / f"{recording}_{session}_repeat{repeat}"
                dest.mkdir(parents=True, exist_ok=False)
                command = ["docker", "run", "--rm", "--platform", "linux/arm64",
                           "--network", "none", "--workdir", "/output",
                           "--mount", f"type=bind,src={source},dst=/recording,readonly",
                           "--mount", f"type=bind,src={dest},dst=/output",
                           image, "/recording", "/output"]
                print(f"Starting {recording} {image} repeat {repeat}", flush=True)
                started = time.perf_counter()
                with (dest / "slam.log").open("x") as log:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                wall = time.perf_counter() - started
                row = {"recording": recording, "robot": "robot_A", "image": image,
                       "image_id": image_ids[image], "repeat": repeat,
                       "command": command, "input_sha256": fingerprints,
                       "wall_time_s": wall, "exit_code": process.returncode,
                       "output": str(dest.relative_to(root))}
                (dest / "run.json").write_text(json.dumps(row, indent=2) + "\n")
                process.check_returncode()
                raw = np.loadtxt(dest / "CameraTrajectory.txt", ndmin=2)
                # Match slam_runner.dedupe_tum without rewriting original exports.
                _, indices = np.unique(raw[:, 0], return_index=True)
                estimate = raw[indices]
                row.update(compute_ate(truth, estimate))
                row["exported_poses"] = len(estimate)
                row["duplicate_export_rows"] = len(raw) - len(estimate)
                log_text = (dest / "slam.log").read_text()
                row["maps_created"] = log_text.count("Creation of new map with id")
                with (dest / "tracking.csv").open() as stream:
                    states = [int(r["state"]) for r in csv.DictReader(stream)]
                row["input_frames"] = len(states)
                row["tracked_frames"] = states.count(2)
                row["state_counts"] = {str(s): states.count(s) for s in sorted(set(states))}
                (dest / "metrics.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
                results.append(row)
                report.write_text(json.dumps({"runs": results}, indent=2, allow_nan=False) + "\n")
                print(f"ATE {row['ate_rmse_m']:.9f} m, tracked {row['tracked_frames']}, "
                      f"maps {row['maps_created']}, wall {wall:.3f} s", flush=True)
    summary = []
    for recording in recordings:
        for image in images:
            rows = [r for r in results if r["recording"] == recording and r["image"] == image]
            entry = {"recording": recording, "image": image, "n": len(rows)}
            for metric in ("ate_rmse_m", "tracked_frames", "maps_created", "wall_time_s"):
                values = [r[metric] for r in rows]
                entry[metric] = {"mean": statistics.mean(values), "min": min(values),
                                 "max": max(values), "range": max(values) - min(values),
                                 "sample_sd": statistics.stdev(values)}
            summary.append(entry)
    report.write_text(json.dumps({"runs": results, "summary": summary}, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--recordings", nargs="+", default=["unity_clear_seed7"])
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    print(compare(args.session, args.recordings, args.repeats))
