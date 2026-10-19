"""Evaluate per-robot translational ATE using evo and rigid SE(3) alignment."""

import argparse
import json
from pathlib import Path

import numpy as np
import rerun as rr
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D

from brain.io import RecordingReader
from brain.oracle import Trajectory, load_floorplan, load_survivors, load_trajectory, truth_folder
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



def alignment_matrix(result: dict) -> np.ndarray:
    """4x4 that maps the robot's SLAM map frame into the ground-truth (house) frame, from the ATE alignment."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(result["alignment_rotation"])
    T[:3, 3] = np.asarray(result["alignment_translation_m"])
    return T


def transform_error(T_est: np.ndarray, T_true: np.ndarray) -> tuple[float, float]:
    """Translation (m) and rotation (deg) of T_true^-1 T_est."""
    d = np.linalg.inv(T_true) @ T_est
    angle = float(np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1))))
    return float(np.linalg.norm(d[:3, 3])), angle


def gt_pose_at(truth: Trajectory, t: float) -> np.ndarray:
    """Ground-truth camera-to-house pose at the nearest truth sample to t."""
    from scipy.spatial.transform import Rotation
    i = int(np.argmin(np.abs(truth.timestamps - t)))
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(truth.quaternions_xyzw[i]).as_matrix()
    T[:3, 3] = truth.positions[i]
    return T


def local_true_transform(slam_a: np.ndarray, t_a: float, gt_a: Trajectory, slam_b: np.ndarray, t_b: float, gt_b: Trajectory) -> np.ndarray:
    """True T_A_B at the matched pair of keyframes: maps B's map into A's map so that B's keyframe lands where ground
    truth says it is relative to A's keyframe. Independent of drift elsewhere on either trajectory."""
    from brain.merge import pose_matrix
    Sa = pose_matrix(slam_a[int(np.argmin(np.abs(slam_a[:, 0] - t_a)))])
    Sb = pose_matrix(slam_b[int(np.argmin(np.abs(slam_b[:, 0] - t_b)))])
    return Sa @ np.linalg.inv(gt_pose_at(gt_a, t_a)) @ gt_pose_at(gt_b, t_b) @ np.linalg.inv(Sb)


def evaluate_merge(out_dir: Path, details: dict, pair: tuple[str, str], recording: Path | None = None) -> dict:
    """Merge error against T_true = align_A^-1 align_B. A proxy: it includes each robot's own SLAM drift."""
    merge_dir = out_dir / "merge"
    if not (merge_dir / "merge.json").exists():
        return {"merges": [], "false_merges": None, "note": "no merge output"}
    m = json.loads((merge_dir / "merge.json").read_text())
    cands = [json.loads(l) for l in (merge_dir / "candidates.jsonl").read_text().splitlines() if l.strip()]
    a, b = pair
    T_true = np.linalg.inv(alignment_matrix(details[a])) @ alignment_matrix(details[b])
    false_merges, per_match = 0, []
    local = recording is not None
    if local:
        gt = {r: load_trajectory(recording, r) for r in pair}
        traj = {r: read_tum(out_dir / r / "KeyFrameTrajectory.txt") for r in pair}
    for c in cands:
        if c.get("accepted") and "T_A_B" in c:
            err_m, err_deg = transform_error(np.asarray(c["T_A_B"]), T_true)
            row = {"t": c["t"], "msg_id": c["msg_id"], "query_kf": c["query_kf"], "match_kf": c["match_kf"],
                   "error_m": err_m, "error_deg": err_deg, "used_in_pose_graph": c.get("used_in_pose_graph")}
            if local:
                ta = c["match_t"] if c["to"] == a else c["query_t"]
                tb = c["query_t"] if c["to"] == a else c["match_t"]
                T_loc = local_true_transform(traj[a], ta, gt[a], traj[b], tb, gt[b])
                row["local_error_m"], row["local_error_deg"] = transform_error(np.asarray(c["T_A_B"]), T_loc)
                false_merges += int(row["local_error_m"] > 0.5)
            else:
                false_merges += int(err_m > 0.5)
            per_match.append(row)
    merges = []
    if m.get("merged"):
        err_m, err_deg = transform_error(np.asarray(m["T_A_B"]), T_true)
        first_m, first_deg = transform_error(np.asarray(m["T_A_B_first"]), T_true)
        entry = {"t": m["t_merge"], "pair": list(pair), "accepted_matches": m["accepted"],
                 "rejected_candidates": m["rejected"], "transform_error_m": err_m, "transform_error_deg": err_deg,
                 "first_match_error_m": first_m, "first_match_error_deg": first_deg}
        used = [r for r in per_match if r.get("used_in_pose_graph") and "local_error_m" in r]
        if used:
            # Headline merge error: registration error at the matched places (median over pose-graph matches).
            entry["local_error_m"] = float(np.median([r["local_error_m"] for r in used]))
            entry["local_error_deg"] = float(np.median([r["local_error_deg"] for r in used]))
            entry["global_error_m"], entry["global_error_deg"] = err_m, err_deg
            entry["transform_error_m"], entry["transform_error_deg"] = entry["local_error_m"], entry["local_error_deg"]
        merges.append(entry)
    (out_dir / "merge_details.json").write_text(json.dumps({"T_true_A_B": T_true.tolist(), "matches": per_match}, indent=2) + "\n")
    return {"merges": merges, "false_merges": false_merges}


def evaluate_map(recording: Path, out_dir: Path, details: dict, pair: tuple[str, str]) -> dict:
    """Median distance from merged map points (in the house frame) to the true surfaces, and floor coverage."""
    import open3d as o3d
    from scipy.spatial import cKDTree
    result = {"coverage_pct": None, "map_median_error_m": None}
    merged = out_dir / "merge" / "merged_map.ply"
    truth = truth_folder(recording) / "house_truth.ply"
    if merged.exists() and truth.exists():
        pts = np.asarray(o3d.io.read_point_cloud(str(merged)).points)
        T = alignment_matrix(details[pair[0]])
        pts_h = pts @ T[:3, :3].T + T[:3, 3]
        tp = np.asarray(o3d.io.read_point_cloud(str(truth)).points)
        dist, _ = cKDTree(tp).query(pts_h, k=1)
        result["map_median_error_m"] = float(np.median(dist))
        result["map_points"] = int(len(pts_h))
        plan = load_floorplan(recording)
        if plan:
            # Coverage proxy: a 0.25 m floor cell counts as seen if a merged-map point lies above it and
            # within 3 m (horizontally) of the camera of the robot that mapped the house.
            cells = set()
            for room in plan["rooms"]:
                x0, z0, x1, z1 = room["rect"]
                for x in np.arange(x0 + 0.125, x1, 0.25):
                    for z in np.arange(z0 + 0.125, z1, 0.25):
                        cells.add((round(float(x), 3), round(float(z), 3)))
            cams = np.vstack([load_trajectory(recording, r).positions for r in pair])
            near = cKDTree(cams[:, [0, 2]]).query(pts_h[:, [0, 2]], k=1)[0] <= 3.0
            seen = {(round(float(np.floor(x / 0.25) * 0.25 + 0.125), 3), round(float(np.floor(z / 0.25) * 0.25 + 0.125), 3))
                    for x, z in pts_h[near][:, [0, 2]]}
            result["coverage_pct"] = 100.0 * len(cells & seen) / len(cells)
    return result

def evaluate_survivors(recording: Path, out_dir: Path, details: dict, robot_a: str) -> dict | None:
    """Match merged survivor detections (robot A map frame) to the oracle's survivors within 1 m on the floor plane."""
    path = out_dir / "merge" / "survivors_merged.json"
    truth = load_survivors(recording)
    if not path.exists() or not truth or robot_a not in details:
        return None
    found = json.loads(path.read_text())
    T = alignment_matrix(details[robot_a])
    est = [(f["id"], T[:3, :3] @ np.asarray(f["position_a"]) + T[:3, 3]) for f in found]
    matches, used = [], set()
    for s in truth:
        q = np.asarray(s["position_house"], float)
        best = min(((np.linalg.norm((p - q)[[0, 2]]), fid, p) for fid, p in est if fid not in used), default=None, key=lambda x: x[0])
        if best and best[0] <= 1.0:
            used.add(best[1])
            matches.append({"survivor": s["id"], "detection": best[1], "error_m": float(best[0])})
    false = [{"detection": fid, "position_house": p.round(2).tolist()} for fid, p in est if fid not in used]
    return {"total": len(truth), "found": len(matches), "matches": matches, "false_detections": false}


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
    pair = tuple(metadata["robots"][:2])
    if len(pair) == 2:
        report.update(evaluate_merge(out_dir, details, pair, recording))
        report.update(evaluate_map(recording, out_dir, details, pair))
        surv = evaluate_survivors(recording, out_dir, details, pair[0])
        if surv is not None:
            report["survivors"] = surv
    # Association and alignment evidence goes in a separate file.
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
    for mg in report.get("merges", []):
        paragraphs.append(
            f"Maps of {mg['pair'][0]} and {mg['pair'][1]} merged at t = {mg['t']:.2f} s after {mg['accepted_matches']} accepted "
            f"and {mg['rejected_candidates']} rejected candidates. Transform error {mg['transform_error_m'] * 100:.1f} cm and "
            f"{mg['transform_error_deg']:.2f} degrees (median over the matches used in the pose graph). "
            "Merge error is measured at the matched places: ground truth says where each matched keyframe sits relative to the other, independent of drift elsewhere."
            + (f" Against whole-trajectory alignment, which also counts each robot's accumulated drift, it is {mg['global_error_m'] * 100:.1f} cm and {mg['global_error_deg']:.2f} degrees." if 'global_error_m' in mg else ""))
    if not report.get("merges"):
        paragraphs.append("No merge happened in this run.")
    if report.get("false_merges") is not None:
        paragraphs.append(f"False merges (accepted matches whose registration is more than 50 cm wrong at the matched place): {report['false_merges']}.")
    if report.get("map_median_error_m") is not None:
        paragraphs.append(f"Merged map median distance to the true surfaces: {report['map_median_error_m'] * 100:.1f} cm.")
    if report.get("survivors"):
        sv = report["survivors"]
        errs = ", ".join(f"{m['survivor']} {m['error_m'] * 100:.0f} cm" for m in sv["matches"]) or "none"
        paragraphs.append(f"Survivors found by the robots' person detector: {sv['found']} of {sv['total']} (position error: {errs}); "
                          f"{len(sv['false_detections'])} false detection(s).")
    if report.get("coverage_pct") is not None:
        paragraphs.append(f"Floor coverage (proxy: floor cells with a merged-map point within 3 m of a camera): {report['coverage_pct']:.1f} percent.")
    (out_dir / "report.md").write_text("\n".join(paragraphs) + "\n")
    return report


def compute_place_recall(truth_a: Trajectory, truth_b: Trajectory,
                         times_a: np.ndarray, times_b: np.ndarray,
                         descriptors_a: np.ndarray, descriptors_b: np.ndarray) -> dict:
    """Score B-to-A retrieval using truth only to label the returned candidates."""
    from brain.place import retrieve

    def sample(truth, times):
        times = np.asarray(times)
        right = np.clip(np.searchsorted(truth.timestamps, times), 0, len(truth.timestamps) - 1)
        left = np.maximum(right - 1, 0)
        indices = np.where(abs(truth.timestamps[left] - times)
                           <= abs(truth.timestamps[right] - times), left, right)
        differences = abs(truth.timestamps[indices] - times)
        return truth.positions[indices], truth.quaternions_xyzw[indices], differences, differences <= .02

    if len(times_a) != len(descriptors_a) or len(times_b) != len(descriptors_b):
        raise ValueError("descriptor counts must match keyframe timestamps")
    positions_a, rotations_a, dt_a, valid_a = sample(truth_a, times_a)
    positions_b, rotations_b, dt_b, valid_b = sample(truth_b, times_b)
    rotations_a = rotations_a / np.linalg.norm(rotations_a, axis=1, keepdims=True)
    rotations_b = rotations_b / np.linalg.norm(rotations_b, axis=1, keepdims=True)
    queries = []
    for i, (position, rotation, descriptor) in enumerate(zip(positions_b, rotations_b, descriptors_b)):
        distances = np.linalg.norm(positions_a - position, axis=1)
        angles = np.degrees(2 * np.arccos(np.clip(abs(rotations_a @ rotation), 0, 1)))
        true_matches = np.flatnonzero((distances <= .5) & (angles <= 30.) & valid_a & valid_b[i])
        top = retrieve(descriptor, descriptors_a)
        hit = bool(set(top) & set(true_matches.tolist()))
        queries.append({"query_index": i, "timestamp": float(times_b[i]),
                        "eligible": bool(len(true_matches)), "hit": hit,
                        "true_matches": true_matches.tolist(), "top_5": top,
                        "truth_available": bool(valid_b[i]),
                        "top_5_distances_m": [float(distances[j]) if valid_a[j] and valid_b[i] else None for j in top],
                        "top_5_angles_deg": [float(angles[j]) if valid_a[j] and valid_b[i] else None for j in top]})
    eligible = sum(q["eligible"] for q in queries)
    hits = sum(q["hit"] for q in queries if q["eligible"])
    return {"direction": "robot_B to robot_A", "translation_threshold_m": .5,
            "rotation_threshold_deg": 30., "total_queries": len(queries),
            "database_keyframes": len(descriptors_a), "eligible_queries": eligible,
            "truth_labels_complete": bool(valid_a.all() and valid_b.all()),
            "unlabelled_database_indices": np.flatnonzero(~valid_a).tolist(),
            "unlabelled_query_indices": np.flatnonzero(~valid_b).tolist(),
            "truth_association_max_diff_s": .02,
            "hits_at_5": hits, "recall_at_5": hits / eligible if eligible else None,
            "max_truth_timestamp_difference_s": float(max(dt_a.max(), dt_b.max())),
            "queries": queries}


def evaluate_place(recording: Path | str, out_dir: Path | str) -> dict:
    """C8 acceptance, separate from the unfinished C10 merge metrics."""
    from brain.place import acceptance_gate
    recording, out_dir = Path(recording), Path(out_dir)
    manifests = [json.loads((out_dir / robot / "keyframes.json").read_text())
                 for robot in ("robot_A", "robot_B")]
    times_a, times_b = [np.array([row["timestamp"] for row in rows]) for rows in manifests]
    descriptors_a, descriptors_b = [np.load(out_dir / robot / "descriptors.npy")
                                     for robot in ("robot_A", "robot_B")]
    provenance = [json.loads((out_dir / robot / "descriptors.json").read_text())
                  for robot in ("robot_A", "robot_B")]
    if provenance[0]["centers_sha256"] != provenance[1]["centers_sha256"]:
        raise ValueError("robots' descriptors use different vocabularies")
    recall = compute_place_recall(load_trajectory(recording, "robot_A"),
                                 load_trajectory(recording, "robot_B"), times_a, times_b,
                                 descriptors_a, descriptors_b)
    # A permutation control uses the same query vectors with image IDs scrambled.
    # It exposes chance recall when a broad pose threshold labels most of a room.
    shuffled = compute_place_recall(load_trajectory(recording, "robot_A"),
                                   load_trajectory(recording, "robot_B"), times_a, times_b,
                                   descriptors_a[np.random.default_rng(1729).permutation(len(descriptors_a))],
                                   descriptors_b)
    recall["shuffled_database_recall_at_5"] = shuffled["recall_at_5"]
    # Identify literal shared images in the public sequence halves. These are
    # valid overlap, but must not conceal poor retrieval on different views.
    import hashlib
    hashes_a = {hashlib.sha256((recording / "robot_A" / row["rgb"]).read_bytes()).hexdigest()
                for row in manifests[0]}
    for query, row in zip(recall["queries"], manifests[1]):
        query["identical_rgb_in_database"] = (hashlib.sha256(
            (recording / "robot_B" / row["rgb"]).read_bytes()).hexdigest() in hashes_a)
    distinct = [q for q in recall["queries"] if q["eligible"] and not q["identical_rgb_in_database"]]
    recall["distinct_image_eligible_queries"] = len(distinct)
    recall["distinct_image_hits_at_5"] = sum(q["hit"] for q in distinct)
    recall["distinct_image_recall_at_5"] = sum(q["hit"] for q in distinct) / len(distinct) if distinct else None
    attempts = json.loads((out_dir / "c8_timing.json").read_text())
    timing = attempts[-1]["timing"]
    reasons = []
    if timing["device"] != "mps":
        reasons.append("MPS timing unavailable; CPU timing does not satisfy the C8 gate")
    if not recall["truth_labels_complete"]:
        reasons.append("Some keyframes lack truth within 0.02 s; recall covers only labelled poses")
    if recall["eligible_queries"] == 0 or recall["recall_at_5"] < .8:
        reasons.append("Retrieval recall threshold not met")
    if timing["mean_ms"] >= 150:
        reasons.append("Descriptor mean latency is not below 150 ms")
    result = {"component": "C8", "passed": acceptance_gate(recall, timing),
              "recall_at_5": recall["recall_at_5"], "eligible_queries": recall["eligible_queries"],
              "hits_at_5": recall["hits_at_5"], "descriptor_mean_ms": timing["mean_ms"],
              "truth_labels_complete": recall["truth_labels_complete"],
              "unlabelled_database_indices": recall["unlabelled_database_indices"],
              "device": timing["device"], "model": attempts[-1]["config"]["model"],
              "mps_timing_verified": timing["device"] == "mps",
              "reasons": reasons}
    (out_dir / "c8_recall.json").write_text(json.dumps(recall, indent=2, allow_nan=False) + "\n")
    (out_dir / "c8_acceptance.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--place-only", action="store_true", help="evaluate the C8 retrieval gate")
    args = parser.parse_args()
    if args.place_only:
        result = evaluate_place(args.recording, args.out_dir)
        raise SystemExit(0 if result["passed"] else 1)
    evaluate_run(args.recording, args.out_dir)


if __name__ == "__main__":
    main()
