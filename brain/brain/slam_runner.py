"""Run headless ORB-SLAM3 on one robot's RGB-D recording."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scipy.spatial.transform import Rotation

from brain.io import RecordingReader
from brain.trajectory import match_timestamps, read_tum, tracking_summary


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


def run_orbslam3(robot_dir: Path | str, out_dir: Path | str) -> tuple[Path, Path]:
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
    docker = shutil.which("docker") or str(Path.home() / ".orbstack/bin/docker")
    image = "smoke-rescue-orbslam3"
    image_id = subprocess.check_output(
        [docker, "image", "inspect", image, "--format", "{{.Id}}"], text=True
    ).strip()
    command = [docker, "run", "--rm", "--network", "none",
               "--mount", f"type=bind,source={robot_dir},target=/recording,readonly",
               "--mount", f"type=bind,source={out_dir},target=/output",
               image, "/recording", "/output"]
    meta = {"backend": "orbslam3", "image_id": image_id, "command": command,
            "robot_dir": str(robot_dir), "frames": len(reader),
            "tracking_lost_definition": "missing exported pose, including startup; interval to next input frame, final interval 1/camera.fps",
            "status": "running"}
    started = time.perf_counter()
    try:
        print(f"Running ORB-SLAM3 on {robot_dir.name}: {len(reader)} frames", flush=True)
        with (out_dir / "slam.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        meta["wall_time_s"] = time.perf_counter() - started
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
        meta.setdefault("wall_time_s", time.perf_counter() - started)
        (out_dir / "slam_meta.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
        print(json.dumps(meta, indent=2), flush=True)
    return camera_path, keyframe_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    run_orbslam3(args.robot_dir, args.out_dir)


if __name__ == "__main__":
    main()
