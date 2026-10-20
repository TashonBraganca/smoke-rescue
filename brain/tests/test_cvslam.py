"""OpenCV-5-native SLAM on a synthetic RGB-D sequence with known camera motion.

A pinhole camera moves inside a 6 x 3 x 8 m box whose walls carry a fixed random texture; depth and colour are ray
cast exactly, so any error comes from cvslam, not from the data.
"""

import cv2
import numpy as np
import pytest

from brain import cvslam

W, H, F = 320, 240, 220.0
ROOM = np.array([[-3.0, -1.5, -1.0], [3.0, 1.5, 7.0]])      # x right, y down, z forward (camera optical frame)
_rng = np.random.default_rng(3)
_TEX = cv2.GaussianBlur((_rng.random((512, 512)) * 255).astype(np.uint8), (0, 0), 1.2)


def render(R: np.ndarray, t: np.ndarray):
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    d = np.stack([(u - W / 2 + .5) / F, (v - H / 2 + .5) / F, np.ones_like(u, float)], -1) @ R.T
    best = np.full((H, W), np.inf)
    tex = np.zeros((H, W))
    for axis in range(3):
        for side in range(2):
            plane = ROOM[side, axis]
            with np.errstate(divide="ignore", invalid="ignore"):
                s = (plane - t[axis]) / d[..., axis]
            p = t + s[..., None] * d
            ok = (s > 0) & (s < best)
            a, b = [k for k in range(3) if k != axis]
            ok &= (p[..., a] >= ROOM[0, a] - 1e-6) & (p[..., a] <= ROOM[1, a] + 1e-6)
            ok &= (p[..., b] >= ROOM[0, b] - 1e-6) & (p[..., b] <= ROOM[1, b] + 1e-6)
            best[ok] = s[ok]
            iu = ((p[..., a] * 64 + axis * 97 + side * 31) % 512).astype(int)
            iv = ((p[..., b] * 64 + axis * 53) % 512).astype(int)
            tex[ok] = _TEX[iv[ok], iu[ok]]
    depth = best * (d @ R[:, 2])                                # z along the camera axis
    gray = tex.astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), (depth * 1000).astype(np.uint16)


def yaw(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def write_sequence(tmp_path, poses):
    d = tmp_path / "robot"
    (d / "rgb").mkdir(parents=True)
    (d / "depth").mkdir()
    lines = []
    for i, (R, t) in enumerate(poses):
        rgb, depth = render(R, t)
        ts = f"{i / 15:.6f}"
        cv2.imwrite(str(d / "rgb" / f"{ts}.png"), rgb)
        cv2.imwrite(str(d / "depth" / f"{ts}.png"), depth)
        lines.append(f"{ts} rgb/{ts}.png {ts} depth/{ts}.png")
    (d / "associations.txt").write_text("\n".join(lines) + "\n")
    (d / "camera.yaml").write_text("%YAML:1.0\n---\n" + "\n".join([
        f"Camera1.fx: {F}", f"Camera1.fy: {F}", f"Camera1.cx: {W / 2 - .5}", f"Camera1.cy: {H / 2 - .5}",
        f"Camera.width: {W}", f"Camera.height: {H}", "Camera.fps: 15", "RGBD.DepthMapFactor: 1000.0"]) + "\n")
    return d


def trajectory(out):
    return np.loadtxt(out / "CameraTrajectory.txt")


def test_odometry_recovers_a_known_path(tmp_path):
    poses = [(yaw(0.5 * i), np.array([0.01 * i, 0.0, 0.03 * i])) for i in range(40)]   # 1.2 m forward, 20 deg turn
    d = write_sequence(tmp_path, poses)
    meta = cvslam.run(d, tmp_path / "out", loops=False)
    est = trajectory(tmp_path / "out")
    assert meta["odometry_failures"] == 0
    true_end = poses[-1][1]
    # World = first camera, which is the synthetic origin, so positions compare directly.
    assert np.linalg.norm(est[-1, 1:4] - true_end) < 0.05, (est[-1, 1:4], true_end)


def assert_loops_match_truth(meta, poses):
    """Each loop's transform (pose of the later keyframe in the earlier one's camera) must equal the true one."""
    for loop in meta["loops"]:
        (Ri, ti), (Rj, tj) = poses[loop["from_frame"]], poses[loop["to_frame"]]
        true = np.eye(4)
        true[:3, :3], true[:3, 3] = Ri.T @ Rj, Ri.T @ (tj - ti)
        Z = np.array(loop["T_from_to"])
        rot_err = np.degrees(np.arccos(np.clip((np.trace(true[:3, :3].T @ Z[:3, :3]) - 1) / 2, -1, 1)))
        # PnP on a 320x240 image is good to a few cm; a false loop is off by metres or tens of degrees.
        tol = 0.03 + 0.1 * np.linalg.norm(true[:3, 3])
        assert np.linalg.norm(Z[:3, 3] - true[:3, 3]) < tol and rot_err < 3.0, (loop["from_frame"], loop["to_frame"], rot_err)


def test_loops_found_on_return_are_geometrically_correct(tmp_path, monkeypatch):
    monkeypatch.setattr(cvslam, "KEY_TRANS_M", 0.15)
    monkeypatch.setattr(cvslam, "LOOP_MIN_GAP", 4)
    monkeypatch.setattr(cvslam, "LOOP_MIN_INLIERS", 30)
    out_and_back = [(np.eye(3), np.array([0.0, 0.0, 0.05 * i])) for i in range(30)]
    out_and_back += [(np.eye(3), np.array([0.0, 0.0, 0.05 * (29 - i)])) for i in range(1, 30)]
    meta = cvslam.run(write_sequence(tmp_path / "a", out_and_back), tmp_path / "a_out", loops=True)
    assert meta["loop_edges"] >= 1
    assert_loops_match_truth(meta, out_and_back)
    # Turning in place: overlapping views may close loops, but every loop must carry the true rotation.
    turn = [(yaw(3 * i), np.zeros(3)) for i in range(59)]
    meta = cvslam.run(write_sequence(tmp_path / "b", turn), tmp_path / "b_out", loops=True)
    assert meta["keyframes"] >= 8
    assert_loops_match_truth(meta, turn)
