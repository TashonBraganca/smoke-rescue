"""OpenCV-5-native RGB-D SLAM, run beside ORB-SLAM3 on the same recordings and scored the same way.

    uv run python -m brain.cvslam ../data/recordings/unity_clear_seed7/robot_A ../data/runs/cvslam/robot_A

Pipeline, all image work in OpenCV 5.0:
  1. Frame-to-frame RGB-D odometry with cv2.Odometry (photometric + ICP, OdometryType_RGB_DEPTH). Frame-to-keyframe
     tracking was measured first and under-measured motion by about half (path 32.5 m vs 13.99 m true in 600 frames).
  2. A keyframe every 0.25 m or 12 degrees, used only for loop closure and the pose graph.
  3. Loop closure: ORB features per keyframe (cv2.ORB, cv2.BFMatcher with Hamming distance), 3D points from depth,
     cv2.solvePnPRansac between a keyframe and older keyframes that lie within 3 m on the current estimate.
  4. Pose graph over keyframes (odometry edges + verified loop edges), solved with scipy least squares, because the
     cv2.detail_PoseGraph Python binding exposes no methods in opencv-python 5.0.0.93.
  5. Every frame's pose = its keyframe's optimised pose times its own odometry offset.
Output matches ORB-SLAM3's: CameraTrajectory.txt and KeyFrameTrajectory.txt (TUM, camera-to-world, first camera =
world), plus cvslam_meta.json. Reads only the robot's own camera, depth and calibration; never oracle/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from brain.io import read_camera

KEY_TRANS_M, KEY_ROT_DEG = 0.25, 12.0
LOOP_RADIUS_M, LOOP_MIN_GAP, LOOP_MIN_INLIERS = 3.0, 25, 40


def _assoc(robot_dir: Path) -> list[tuple[float, Path, Path]]:
    rows = []
    for line in (robot_dir / "associations.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        t_rgb, rgb, _, depth = line.split()[:4]
        rows.append((float(t_rgb), robot_dir / rgb, robot_dir / depth))
    return rows


def _angle_deg(T: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(T[:3, :3]).as_rotvec())))


class Keyframe:
    __slots__ = ("idx", "frame", "pose", "kps", "desc", "pts3d")

    def __init__(self, idx, frame, pose):
        self.idx, self.frame, self.pose = idx, frame, pose
        self.kps = self.desc = self.pts3d = None


def run(robot_dir: Path, out_dir: Path, mode: str = "RGB_DEPTH", loops: bool = True, max_frames: int | None = None,
        invert: bool = True) -> dict:
    robot_dir, out_dir = Path(robot_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cam = read_camera(robot_dir / "camera.yaml")
    K = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]], np.float32)
    settings = cv2.OdometrySettings()
    settings.setCameraMatrix(K)
    # Defaults reject motions over 0.15 m / 15 deg per call; keyframes are up to 0.25 m / 12 deg apart.
    settings.setMaxTranslation(0.6)
    settings.setMaxRotation(30)
    settings.setMaxDepth(8.0)
    odom = cv2.Odometry(getattr(cv2, f"OdometryType_{mode}"), settings, cv2.OdometryAlgoType_COMMON)
    orb = cv2.ORB_create(1000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    rows = _assoc(robot_dir)[:max_frames]
    t0 = time.perf_counter()

    def load(i):
        gray = cv2.imread(str(rows[i][1]), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(str(rows[i][2]), cv2.IMREAD_UNCHANGED).astype(np.float32) / cam.depth_factor
        depth[(depth < 0.2) | (depth > 8.0)] = 0
        f = cv2.OdometryFrame(depth, gray)
        return gray, depth, f

    def features(kf: Keyframe, gray, depth):
        kps, desc = orb.detectAndCompute(gray, None)
        keep, pts = [], []
        for j, k in enumerate(kps or []):
            u, v = int(round(k.pt[0])), int(round(k.pt[1]))
            z = depth[min(v, depth.shape[0] - 1), min(u, depth.shape[1] - 1)]
            if z > 0:
                keep.append(j)
                pts.append(((k.pt[0] - cam.cx) * z / cam.fx, (k.pt[1] - cam.cy) * z / cam.fy, z))
        kf.kps = [kps[j] for j in keep]
        kf.desc = desc[keep] if desc is not None and keep else None
        kf.pts3d = np.array(pts, np.float32)

    keyframes: list[Keyframe] = []
    rel: list[tuple[int, np.ndarray]] = []         # per frame: (keyframe index, T_kf_cam)
    odo_edges: list[tuple[int, int, np.ndarray]] = []
    loop_edges: list[tuple[int, int, np.ndarray]] = []
    failures = 0

    gray, depth, f = load(0)
    odom.prepareFrame(f)
    kf = Keyframe(0, f, np.eye(4))
    features(kf, gray, depth)
    keyframes.append(kf)
    rel.append((0, np.eye(4)))
    prev_f, pose = f, np.eye(4)
    for i in range(1, len(rows)):
        gray, depth, f = load(i)
        # Frame-to-frame: small baselines converge; frame-to-keyframe under-measured motion by about half.
        odom.prepareFrames(prev_f, f)
        ok, Rt = odom.compute(prev_f, f)
        if ok and np.isfinite(Rt).all():
            # Rt maps points from the previous camera into this one, so this camera's pose in the previous is inv(Rt).
            step = np.linalg.inv(Rt) if invert else Rt
        else:
            failures += 1
            step = np.eye(4)                     # hold still for one frame rather than guess
        pose = pose @ step
        prev_f = f
        k = len(keyframes) - 1
        T_kf = np.linalg.inv(keyframes[k].pose) @ pose
        if np.linalg.norm(T_kf[:3, 3]) > KEY_TRANS_M or _angle_deg(T_kf) > KEY_ROT_DEG:
            new = Keyframe(i, f, pose.copy())
            features(new, gray, depth)
            odo_edges.append((k, k + 1, T_kf))
            keyframes.append(new)
            rel.append((k + 1, np.eye(4)))
            if loops:
                _close_loops(keyframes, loop_edges, matcher, K)
        else:
            rel.append((k, T_kf))
    t_track = time.perf_counter() - t0

    poses = _optimise([k.pose for k in keyframes], odo_edges, loop_edges) if loop_edges else [k.pose for k in keyframes]
    traj = [(rows[i][0], poses[k] @ T) for i, (k, T) in enumerate(rel)]
    _write_tum(out_dir / "CameraTrajectory.txt", traj)
    _write_tum(out_dir / "KeyFrameTrajectory.txt", [(rows[k.idx][0], poses[j]) for j, k in enumerate(keyframes)])
    meta = {"opencv": cv2.__version__, "odometry": f"cv2.Odometry {mode} COMMON, frame-to-frame",
            "frames": len(rows), "keyframes": len(keyframes), "odometry_failures": failures,
            "loop_edges": len(loop_edges), "loops_enabled": loops,
            "pose_graph": "scipy least_squares (cv2.detail_PoseGraph has no Python methods in 5.0.0.93)" if loop_edges else "none",
            "tracking_s": round(t_track, 1), "ms_per_frame": round(1000 * t_track / max(1, len(rows)), 1),
            "settings": {"key_trans_m": KEY_TRANS_M, "key_rot_deg": KEY_ROT_DEG, "loop_radius_m": LOOP_RADIUS_M,
                         "loop_min_gap_keyframes": LOOP_MIN_GAP, "loop_min_inliers": LOOP_MIN_INLIERS}}
    (out_dir / "cvslam_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def _close_loops(kfs, loop_edges, matcher, K):
    cur = kfs[-1]
    if cur.desc is None or len(cur.kps) < 50:
        return
    pos = cur.pose[:3, 3]
    best = None
    for j, old in enumerate(kfs[:-LOOP_MIN_GAP]):
        if old.desc is None or np.linalg.norm(old.pose[:3, 3] - pos) > LOOP_RADIUS_M:
            continue
        m = matcher.match(old.desc, cur.desc)
        if len(m) < LOOP_MIN_INLIERS:
            continue
        obj = np.float32([old.pts3d[x.queryIdx] for x in m])
        img = np.float32([cur.kps[x.trainIdx].pt for x in m])
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, reprojectionError=3.0, iterationsCount=200,
                                                 flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or inl is None or len(inl) < LOOP_MIN_INLIERS:
            continue
        if best is None or len(inl) > best[0]:
            T_cur_old = np.eye(4)
            T_cur_old[:3, :3], T_cur_old[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
            best = (len(inl), j, np.linalg.inv(T_cur_old))   # pose of cur in old's camera frame
    if best:
        loop_edges.append((best[1], len(kfs) - 1, best[2]))


def _optimise(poses, odo, loops):
    n = len(poses)
    x0 = np.concatenate([np.r_[Rotation.from_matrix(P[:3, :3]).as_rotvec(), P[:3, 3]] for P in poses[1:]])

    def mat(v):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rotation.from_rotvec(v[:3]).as_matrix(), v[3:]
        return T

    def resid(x):
        P = [poses[0]] + [mat(x[6 * k:6 * k + 6]) for k in range(n - 1)]
        out = []
        for (i, j, Z), w in [(e, 1.0) for e in odo] + [(e, 2.0) for e in loops]:
            E = np.linalg.inv(Z) @ np.linalg.inv(P[i]) @ P[j]
            out.append(w * np.r_[Rotation.from_matrix(E[:3, :3]).as_rotvec() * 3.0, E[:3, 3]])
        return np.concatenate(out)

    sol = least_squares(resid, x0, loss="huber", f_scale=0.1, max_nfev=200)
    return [poses[0]] + [mat(sol.x[6 * k:6 * k + 6]) for k in range(n - 1)]


def _write_tum(path: Path, traj):
    with path.open("w") as f:
        for t, T in traj:
            q = Rotation.from_matrix(T[:3, :3]).as_quat()   # x y z w
            f.write(f"{t:.6f} {T[0,3]:.6f} {T[1,3]:.6f} {T[2,3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("robot_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--mode", default="RGB_DEPTH", choices=("RGB_DEPTH", "DEPTH", "RGB"))
    ap.add_argument("--no-loops", action="store_true")
    a = ap.parse_args()
    print(json.dumps(run(a.robot_dir, a.out_dir, a.mode, not a.no_loops), indent=1))


if __name__ == "__main__":
    main()
