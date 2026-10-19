"""C12 phase 2: survivor detection from each robot's own camera, placed in its own SLAM map.

Robot-side module: reads the robot's recording (RGB, depth, camera.yaml), its SLAM trajectory and the merge
transform. Never reads ground truth.

For every 3rd frame (5 Hz) a COCO person detector (YOLO11m) runs on the RGB image. Each person box is placed in
3D with the median valid depth of its central 40 percent and the SLAM camera pose at that frame. Detections within
0.8 m are clustered; a survivor is a cluster seen in at least 3 frames. Robot B's survivors are moved into robot A's
map with the merge transform and clusters within 1 m are merged.

    uv run python -m brain.detect <recording_dir> <run_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from brain.io import RecordingReader
from brain.merge import pose_matrix
from brain.trajectory import read_tum

WEIGHTS = Path(__file__).resolve().parents[1] / "yolo11m.pt"
CONF = 0.30
EVERY = 3
CLUSTER_M = 0.8
MERGE_M = 1.3        # scenario survivors are at least 3 m apart
MIN_FRAMES = 3


def detect_robot(recording: Path, run: Path, robot: str, model) -> list[dict]:
    reader = RecordingReader(recording / robot)
    cam = reader.camera
    traj = read_tum(run / robot / "CameraTrajectory.txt")
    times = np.array([a.rgb_timestamp for a in reader.associations])
    dets = []
    idx = list(range(0, len(reader), EVERY))
    for start in range(0, len(idx), 16):
        batch = idx[start:start + 16]
        frames = [reader.read_frame(i) for i in batch]
        results = model.predict([np.ascontiguousarray(f.rgb[:, :, ::-1]) for f in frames], conf=CONF, classes=[0], verbose=False, device="mps")
        for i, f, res in zip(batch, frames, results):
            j = int(np.argmin(np.abs(traj[:, 0] - times[i])))
            if abs(traj[j, 0] - times[i]) > 1e-3:
                continue   # no SLAM pose for this frame
            T = pose_matrix(traj[j])
            for b in res.boxes:
                x0, y0, x1, y1 = [float(v) for v in b.xyxy[0]]
                cxb, cyb = (x0 + x1) / 2, (y0 + y1) / 2
                w, h = x1 - x0, y1 - y0
                patch = f.depth_m[int(cyb - 0.2 * h):int(cyb + 0.2 * h) + 1, int(cxb - 0.2 * w):int(cxb + 0.2 * w) + 1]
                valid = patch[(patch > 0.3) & (patch < 6.0)]
                if valid.size < 10:
                    continue
                z = float(np.median(valid))
                p_cam = np.array([(cxb - cam.cx) * z / cam.fx, (cyb - cam.cy) * z / cam.fy, z, 1.0])
                p_map = (T @ p_cam)[:3]
                dets.append({"t": float(times[i]), "frame": i, "conf": float(b.conf), "box": [x0, y0, x1, y1], "p": p_map.tolist()})
    clusters: list[dict] = []
    for d in sorted(dets, key=lambda d: d["t"]):
        p = np.asarray(d["p"])
        best = min(clusters, key=lambda c: np.linalg.norm(c["sum"] / c["n"] - p), default=None)
        if best is not None and np.linalg.norm(best["sum"] / best["n"] - p) < CLUSTER_M:
            best["sum"] += p; best["n"] += 1; best["conf"].append(d["conf"])
        else:
            clusters.append({"sum": p.copy(), "n": 1, "first_seen_t": d["t"], "conf": [d["conf"]]})
    survivors = [{"id": f"{robot[-1]}{k}", "position_map": (c["sum"] / c["n"]).tolist(), "n_frames": c["n"],
                  "first_seen_t": c["first_seen_t"], "mean_conf": float(np.mean(c["conf"]))}
                 for k, c in enumerate(c for c in clusters if c["n"] >= MIN_FRAMES)]
    (run / robot / "detections.json").write_text(json.dumps(dets, indent=1))
    (run / robot / "survivors.json").write_text(json.dumps(survivors, indent=2))
    return survivors


def run(recording: Path, run_dir: Path) -> dict:
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    meta = json.loads((recording / "meta.json").read_text())
    robots = meta["robots"][:2]
    per_robot = {r: detect_robot(recording, run_dir, r, model) for r in robots}
    merge = json.loads((run_dir / "merge" / "merge.json").read_text()) if (run_dir / "merge" / "merge.json").exists() else {}
    T_A_B = np.asarray(merge["T_A_B"]) if merge.get("merged") else None
    merged: list[dict] = []
    for r in robots:
        for s in per_robot[r]:
            p = np.asarray(s["position_map"])
            if r != robots[0]:
                if T_A_B is None:
                    continue   # maps never merged: B's survivors cannot be placed in A's frame
                p = T_A_B[:3, :3] @ p + T_A_B[:3, 3]
            hit = next((m for m in merged if np.linalg.norm(np.asarray(m["position_a"]) - p) < MERGE_M), None)
            if hit:
                hit["seen_by"] = sorted(set(hit["seen_by"]) | {r})
                hit["first_seen_t"] = min(hit["first_seen_t"], s["first_seen_t"])
                hit["n_frames"] += s["n_frames"]
            else:
                merged.append({"id": f"S{len(merged) + 1}", "position_a": p.tolist(), "seen_by": [r],
                               "first_seen_t": s["first_seen_t"], "n_frames": s["n_frames"]})
    (run_dir / "merge").mkdir(exist_ok=True)
    (run_dir / "merge" / "survivors_merged.json").write_text(json.dumps(merged, indent=2))
    return {"per_robot": {r: len(v) for r, v in per_robot.items()}, "merged": len(merged)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recording", type=Path)
    ap.add_argument("run", type=Path)
    a = ap.parse_args()
    print(json.dumps(run(a.recording.resolve(), a.run.resolve()), indent=2))


if __name__ == "__main__":
    main()
