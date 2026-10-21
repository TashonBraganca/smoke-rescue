"""Back-project estimated keyframe poses into a 3 cm voxel point cloud."""

import argparse
import json
from pathlib import Path

import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation

from brain.io import RecordingReader
from brain.pointcloud import PointCloud, write_cloud
from brain.trajectory import match_timestamps, read_tum


def build_dense_map(robot_dir: Path | str, keyframe_path: Path | str,
                    output_path: Path | str) -> Path:
    reader = RecordingReader(robot_dir)
    camera = reader.camera
    if min(camera.fx, camera.fy, camera.depth_factor) <= 0:
        raise ValueError("camera focal lengths and depth factor must be positive")
    keyframes = read_tum(keyframe_path)
    times = np.array([a.rgb_timestamp for a in reader.associations])
    indices = match_timestamps(times, keyframes[:, 0])
    # Average points and colours in fixed, world-frame 3 cm cells. Accumulating
    # sums and counts keeps memory bounded by occupied cells rather than pixels.
    cells = np.empty((0, 3), dtype=np.int64)
    sums = np.empty((0, 6))
    counts = np.empty(0, dtype=np.int64)
    v, u = np.indices((camera.height, camera.width))
    rays = np.stack(((u - camera.cx) / camera.fx,
                     (v - camera.cy) / camera.fy, np.ones_like(u)), axis=-1)
    for pose, index in zip(keyframes, indices):
        frame = reader.read_frame(int(index))
        valid = frame.depth > 0
        points = rays[valid] * frame.depth_m[valid, None]
        points = Rotation.from_quat(pose[4:8]).apply(points) + pose[1:4]
        values = np.column_stack((points, frame.rgb[valid] / 255.))
        new_cells = np.floor(points / .03).astype(np.int64)
        combined = np.concatenate((cells, new_cells))
        unique, inverse = np.unique(combined, axis=0, return_inverse=True)
        weights = np.r_[counts, np.ones(len(points), dtype=np.int64)]
        values = np.concatenate((sums, values))
        sums = np.column_stack([np.bincount(inverse, weights=values[:, j],
                                           minlength=len(unique)) for j in range(6)])
        counts = np.bincount(inverse, weights=weights, minlength=len(unique)).astype(np.int64)
        cells = unique
    if not len(cells):
        raise ValueError("keyframe depth contains no valid points")
    averages = sums / counts[:, None]
    output_path = Path(output_path)
    write_cloud(output_path, PointCloud(averages[:, :3], averages[:, 3:]))
    meta = {"voxel_size_m": .03, "points": len(cells), "keyframes": len(keyframes),
            "pose_source": str(Path(keyframe_path).resolve()),
            "camera_source": str((Path(robot_dir) / "camera.yaml").resolve()),
            "method": "point and RGB averages in fixed world-frame voxels"}
    output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    rr.init("smoke-rescue-c7", spawn=False)
    rr.save(str(output_path.with_suffix(".rrd")))
    entity = f"world/{Path(robot_dir).name}"
    rr.log(f"{entity}/map", rr.Points3D(averages[:, :3], colors=averages[:, 3:]), static=True)
    rr.log(f"{entity}/trajectory", rr.LineStrips3D([keyframes[:, 1:4]]), static=True)
    rr.disconnect()
    print(json.dumps(meta, indent=2), flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    build_dense_map(args.robot_dir, args.out_dir / "KeyFrameTrajectory.txt", args.out_dir / "map.ply")


if __name__ == "__main__":
    main()
