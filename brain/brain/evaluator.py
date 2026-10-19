"""Evaluate per-robot translational ATE using evo and rigid SE(3) alignment."""

import argparse
import json
from pathlib import Path

import numpy as np
import rerun as rr
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D

from brain.io import RecordingReader
from brain.oracle import Trajectory, load_trajectory
from brain.trajectory import read_tum, tracking_summary, validate_tum


def compute_ate(truth: Trajectory, estimate: np.ndarray) -> dict:
    estimate = validate_tum(estimate)
    reference = validate_tum(np.column_stack((truth.timestamps, truth.positions,
                                              truth.quaternions_xyzw)))

    def evo_trajectory(rows):
        return PoseTrajectory3D(positions_xyz=rows[:, 1:4],
                                orientations_quat_wxyz=rows[:, [7, 4, 5, 6]],
                                timestamps=rows[:, 0])

    try:
        ref, est = sync.associate_trajectories(evo_trajectory(reference),
                                              evo_trajectory(estimate), max_diff=.02)
    except sync.SyncException as error:
        raise ValueError("fewer than 100 matched poses: no timestamps within 0.02 s") from error
    if est.num_poses < 100:
        raise ValueError(f"fewer than 100 matched poses: {est.num_poses}")
    rotation, translation, scale = est.align(ref, correct_scale=False)
    metric = metrics.APE(metrics.PoseRelation.translation_part)
    metric.process_data((ref, est))
    error = float(metric.get_statistic(metrics.StatisticsType.rmse))
    return {"ate_rmse_m": error, "matched_poses": est.num_poses,
            "association_max_diff_s": .02,
            "max_matched_timestamp_difference_s": float(np.max(abs(ref.timestamps - est.timestamps))),
            "alignment": "SE(3)", "alignment_scale": float(scale),
            "alignment_rotation": rotation.tolist(),
            "alignment_translation_m": translation.tolist()}


def validate_ate(result: dict) -> None:
    error = result["ate_rmse_m"]
    if not np.isfinite(error) or error <= 0:
        raise ValueError(f"ATE must be finite and greater than zero, got {error}")
    if result["matched_poses"] < 100:
        raise ValueError(f"fewer than 100 matched poses: {result['matched_poses']}")


def evaluate_run(recording: Path | str, out_dir: Path | str) -> dict:
    recording, out_dir = Path(recording), Path(out_dir)
    metadata = json.loads((recording / "meta.json").read_text())
    report = {"run_id": metadata["run_id"], "per_robot": {}}
    details = {}
    for robot in metadata["robots"]:
        robot_output = out_dir / robot
        reader = RecordingReader(recording / robot)
        trajectory = read_tum(robot_output / "CameraTrajectory.txt")
        tracking = tracking_summary(np.array([a.rgb_timestamp for a in reader.associations]),
                                    trajectory, reader.camera.fps)
        result = compute_ate(load_trajectory(recording, robot), trajectory)
        validate_ate(result)
        keyframes = read_tum(robot_output / "KeyFrameTrajectory.txt")
        report["per_robot"][robot] = {
            "ate_rmse_m": result["ate_rmse_m"], "frames": len(reader),
            "keyframes": len(keyframes), "tracking_lost_s": tracking["tracking_lost_s"]}
        details[robot] = result
        print(f"{robot}: ATE {result['ate_rmse_m']:.9f} m, "
              f"{result['matched_poses']} matched poses, "
              f"{tracking['tracking_lost_s']:.6f} s missing poses", flush=True)
    # Keep the frozen per_robot format intact. Association and alignment evidence
    # goes in a separate file; unimplemented merge/map metrics are not fabricated.
    (out_dir / "ate_details.json").write_text(json.dumps(details, indent=2, allow_nan=False) + "\n")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    paragraphs = [f"Run {report['run_id']}. ATE uses evo with SE(3) alignment, "
                  "no scale correction, and timestamp association within 0.02 seconds."]
    rr.init("smoke-rescue-c10-ate", spawn=False)
    rr.save(str(out_dir / "evaluation.rrd"))
    for robot, values in report["per_robot"].items():
        paragraphs.append(
            f"{robot} has ATE RMSE {values['ate_rmse_m']:.9f} m over "
            f"{details[robot]['matched_poses']} matched poses. "
            f"The recording has {values['frames']} frames and "
            f"{values['keyframes']} surviving keyframes. "
            f"Missing exported poses account for {values['tracking_lost_s']:.6f} seconds.")
        for metric, value in values.items():
            rr.log(f"metrics/{robot}/{metric}", rr.Scalars(value))
        rr.log(f"metrics/{robot}/matched_poses", rr.Scalars(details[robot]["matched_poses"]))
    rr.disconnect()
    paragraphs.append("Only per-robot ATE is evaluated here. Merge error, false merges, "
                      "coverage and map accuracy have not been computed.")
    (out_dir / "report.md").write_text("\n".join(paragraphs) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    evaluate_run(args.recording, args.out_dir)


if __name__ == "__main__":
    main()
