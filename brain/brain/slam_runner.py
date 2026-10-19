"""Run headless ORB-SLAM3 on one robot's RGB-D recording."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy.spatial.transform import Rotation

from brain.io import RecordingReader
from brain.trajectory import match_timestamps, read_tum, tracking_summary


def body_camera_transform(mount: dict) -> np.ndarray:
    """Map optical camera points into the IMU body frame, with S on both frames."""
    height, forward, pitch = (float(mount[k]) for k in
                              ("height_m", "forward_offset_m", "pitch_down_deg"))
    if not np.isfinite([height, forward, pitch]).all():
        raise ValueError("camera mount must be finite")
    reflection = np.diag([1., -1., 1.])
    transform = np.eye(4)
    transform[:3, :3] = reflection @ Rotation.from_euler("x", pitch, degrees=True).as_matrix() @ reflection
    transform[:3, 3] = [0, -height, forward]
    return transform


def inertial_calibration(robot_dir: Path) -> dict:
    """Use recorded calibration first; only named Unity runs have known fallbacks."""
    config_path = Path(__file__).resolve().parents[2] / "config/sensors.json"
    sensors = json.loads(config_path.read_text())
    meta_path = robot_dir.parent / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    mount = meta.get("camera", {}).get("mount") or meta.get("camera_mount")
    source = "meta.json"
    if mount is None:
        run_id = meta.get("run_id", robot_dir.parent.name)
        if run_id == "unity_clear_seed7":
            mount = {"height_m": .50, "forward_offset_m": .41, "pitch_down_deg": 10.}
            source = "unity_clear_seed7 measured mount"
        elif run_id in ("unity_clear", "unity_haze"):
            mount = sensors["camera"]["mount"]
            source = "config/sensors.json camera.mount"
        else:
            raise ValueError("recording needs camera.mount in meta.json for inertial calibration")
    imu = meta.get("imu", sensors["imu"])
    for key in ("rate_hz", "accel_noise_density", "gyro_noise_density",
                "accel_bias_random_walk", "gyro_bias_random_walk"):
        if not np.isfinite(imu[key]) or imu[key] <= 0:
            raise ValueError(f"IMU calibration must be positive and finite: {key}")
    return {"mount": mount, "mount_source": source, "imu": imu,
            "T_body_camera": body_camera_transform(mount).tolist()}


def write_inertial_settings(robot_dir: Path, output: Path) -> dict:
    """Keep camera calibration intact and append the pinned ORB-SLAM3 IMU keys."""
    calibration = inertial_calibration(robot_dir)
    camera = (robot_dir / "camera.yaml").read_text()
    if "IMU." in camera:
        raise ValueError("camera.yaml already contains IMU settings")
    if 'File.version: "1.0"' not in camera:
        raise ValueError('inertial mode requires File.version: "1.0" camera settings')
    imu = calibration["imu"]
    rows = [", ".join(f"{value:.12g}" for value in row) for row in calibration["T_body_camera"]]
    text = camera.rstrip() + "\n\n# Camera optical to IMU body, p_body = T_b_c1 * p_camera.\n"
    # Converter::toSophus reads cv::Mat entries with at<float> in this version.
    text += "IMU.T_b_c1: !!opencv-matrix\n   rows: 4\n   cols: 4\n   dt: f\n   data: [" + ",\n          ".join(rows) + "]\n"
    for name, key in (("NoiseGyro", "gyro_noise_density"), ("NoiseAcc", "accel_noise_density"),
                      ("GyroWalk", "gyro_bias_random_walk"), ("AccWalk", "accel_bias_random_walk"),
                      ("Frequency", "rate_hz")):
        # ORB-SLAM3 converts these continuous densities at the configured rate.
        text += f"IMU.{name}: {float(imu[key]):.12f}\n"
    text += "IMU.InsertKFsWhenLost: 1\n"
    output.write_text(text)
    return calibration


def align_keyframes(camera: np.ndarray, keyframe_path: Path) -> dict:
    """Put atlas-frame keyframes in the same first-camera frame as the trajectory."""
    keyframes = read_tum(keyframe_path)
    indices = match_timestamps(camera[:, 0], keyframes[:, 0])
    anchor = camera[indices[0]]
    rotation = (Rotation.from_quat(anchor[4:8])
                * Rotation.from_quat(keyframes[0, 4:8]).inv())
    translation = anchor[1:4] - rotation.apply(keyframes[0, 1:4])
    keyframes[:, 1:4] = rotation.apply(keyframes[:, 1:4]) + translation
    keyframes[:, 4:8] = (rotation * Rotation.from_quat(keyframes[:, 4:8])).as_quat()
    differences = np.linalg.norm(keyframes[:, 1:4] - camera[indices, 1:4], axis=1)
    angles = (Rotation.from_quat(camera[indices, 4:8]).inv()
              * Rotation.from_quat(keyframes[:, 4:8])).magnitude()
    if np.max(differences) > .002 or np.max(angles) > .01:
        raise ValueError("keyframes and trajectory disagree after rigid frame conversion")
    np.savetxt(keyframe_path, keyframes, fmt="%.9f")
    return {"rotation_xyzw": rotation.as_quat().tolist(),
            "translation_m": translation.tolist(),
            "max_keyframe_position_difference_m": float(np.max(differences))}


def dedupe_tum(path: Path) -> int:
    """Rewrite a TUM file keeping only the first row of each timestamp, in time order. Returns rows dropped."""
    lines = [l for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    seen, keep = set(), []
    for l in lines:
        t = l.split()[0]
        if t not in seen:
            seen.add(t)
            keep.append(l)
    keep.sort(key=lambda l: float(l.split()[0]))
    path.write_text("\n".join(keep) + "\n")
    return len(lines) - len(keep)


def tracking_state_summary(path: Path, frame_times: np.ndarray, fps: float) -> dict:
    """Count tracker failures even when ORB-SLAM3 exports predicted poses."""
    if not path.exists():
        return {}
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return {}
    timestamps = np.array([float(row["timestamp"]) for row in rows])
    indices = match_timestamps(frame_times, timestamps)
    intervals = np.r_[np.diff(frame_times), 1 / fps][indices]
    states = np.array([int(row["state"]) for row in rows])
    result = {"state_counts": {str(int(state)): int(np.sum(states == state))
                               for state in np.unique(states)},
              "state_tracking_lost_s": float(intervals[np.isin(states, [3, 4])].sum()),
              "state_not_initialized_s": float(intervals[states == 1].sum()),
              "state_tracking_lost_definition": "RECENTLY_LOST=3 or LOST=4, interval to next input frame"}
    if "imu_initialized" in rows[0]:
        initialized = np.array([row["imu_initialized"] == "1" for row in rows])
        result["imu_initialized"] = bool(initialized.any())
        result["imu_first_initialized_frame_s"] = float(timestamps[initialized][0]) if initialized.any() else None
    return result


def run_orbslam3(robot_dir: Path | str, out_dir: Path | str,
                mode: str = "rgbd") -> tuple[Path, Path]:
    if mode not in ("rgbd", "rgbd_inertial"):
        raise ValueError(f"unknown SLAM mode: {mode}")
    robot_dir, out_dir = Path(robot_dir).resolve(), Path(out_dir).resolve()
    if (out_dir == robot_dir or robot_dir in out_dir.parents
            or out_dir in robot_dir.parents):
        raise ValueError("input and output folders must be disjoint")
    reader = RecordingReader(robot_dir)
    times = np.array([a.rgb_timestamp for a in reader.associations])
    out_dir.mkdir(parents=True, exist_ok=True)
    camera_path = out_dir / "CameraTrajectory.txt"
    keyframe_path = out_dir / "KeyFrameTrajectory.txt"
    if any((out_dir / name).exists() for name in
           ("CameraTrajectory.txt", "KeyFrameTrajectory.txt", "slam_meta.json", "slam.log")):
        raise FileExistsError(f"refusing to mix a new SLAM run with existing outputs: {out_dir}")
    calibration = None
    if mode == "rgbd_inertial":
        if not (robot_dir / "imu.txt").is_file():
            raise ValueError("inertial mode requires imu.txt")
        calibration = write_inertial_settings(robot_dir, out_dir / "inertial.yaml")
    docker = shutil.which("docker") or str(Path.home() / ".orbstack/bin/docker")
    image = "smoke-rescue-orbslam3"
    image_id = subprocess.check_output(
        [docker, "image", "inspect", image, "--format", "{{.Id}}"], text=True
    ).strip()
    command = [docker, "run", "--rm", "--network", "none",
               "--mount", f"type=bind,source={robot_dir},target=/recording,readonly",
               "--mount", f"type=bind,source={out_dir},target=/output",
               image, "/recording", "/output"]
    if mode == "rgbd_inertial":
        command += [mode, "/output/inertial.yaml"]
    meta = {"backend": "orbslam3", "mode": mode, "calibration": calibration,
            "image_id": image_id, "command": command,
            "robot_dir": str(robot_dir), "frames": len(reader),
            "tracking_lost_definition": "missing exported pose, including startup; interval to next input frame, final interval 1/camera.fps",
            "status": "running"}
    started = time.perf_counter()
    try:
        print(f"Running ORB-SLAM3 on {robot_dir.name}: {len(reader)} frames", flush=True)
        with (out_dir / "slam.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        meta["wall_time_s"] = time.perf_counter() - started
        # When ORB-SLAM3 loses tracking it starts a new map in its Atlas and writes the lost frames with a repeated
        # timestamp. Keep the first row per timestamp and record how many maps were created, so a reset is reported
        # (and visible in ATE) instead of crashing the pipeline.
        meta["maps_created"] = (out_dir / "slam.log").read_text().count("Creation of new map with id")
        meta["duplicate_rows_dropped"] = dedupe_tum(camera_path) + dedupe_tum(keyframe_path)
        camera = read_tum(camera_path)
        meta.update(tracking_summary(times, camera, reader.camera.fps))
        shutil.copyfile(keyframe_path, out_dir / "KeyFrameTrajectory.raw.txt")
        meta["keyframe_frame_conversion"] = align_keyframes(camera, keyframe_path)
        keyframes = read_tum(keyframe_path)
        indices = match_timestamps(times, keyframes[:, 0])
        manifest = [{"keyframe_id": i, "frame_index": int(index),
                     "timestamp": float(row[0]),
                     "rgb": str(reader.associations[index].rgb_path),
                     "depth": str(reader.associations[index].depth_path)}
                    for i, (row, index) in enumerate(zip(keyframes, indices))]
        (out_dir / "keyframes.json").write_text(json.dumps(manifest, indent=2) + "\n")
        meta.update(keyframes=len(keyframes), status="ok")
    except Exception as error:
        meta.update(status="failed", error=str(error))
        raise
    finally:
        if (out_dir / "slam.log").exists():
            meta["maps_created"] = (out_dir / "slam.log").read_text().count("Creation of new map with id")
        try:
            meta.update(tracking_state_summary(out_dir / "tracking.csv", times, reader.camera.fps))
        except (ValueError, TypeError, KeyError) as error:
            # A crashed container can leave an incomplete CSV row. Keep its
            # original failure and log even when diagnostic parsing fails.
            meta["tracking_diagnostics_error"] = str(error)
        meta.setdefault("wall_time_s", time.perf_counter() - started)
        (out_dir / "slam_meta.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
        print(json.dumps(meta, indent=2), flush=True)
    return camera_path, keyframe_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--mode", choices=("rgbd", "rgbd_inertial"), default="rgbd")
    args = parser.parse_args()
    run_orbslam3(args.robot_dir, args.out_dir, args.mode)


if __name__ == "__main__":
    main()
