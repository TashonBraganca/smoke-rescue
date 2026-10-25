#!/usr/bin/env python3
"""Short clips and stills of the OpenCV 5 steps on a real recording, for the video edit.

    brain/.venv/bin/python tools/opencv_clips.py data/recordings/<run> --out data/evidence/opencv_clips

Writes yolo_*, thermal_*, rgbd_* and map3d_* (.mp4 and a .png still each). Every overlay is computed by the robot-side
code (brain.agent.vision) on the robot's own recorded frames; nothing reads ground truth.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "brain"))
from brain.agent import vision  # noqa: E402

FPS = 15
FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(img, text, org=(14, 30), scale=.7):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)


def stamps(robot_dir, t0, t1, step):
    s = [float(l.split()[0]) for l in open(robot_dir / "rgb.txt") if l[0].isdigit()]
    out, nxt = [], t0
    for t in s:
        if t0 <= t <= t1 and t >= nxt:
            out.append(t); nxt = t + step
    return out


def writer(path, size):
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)


def h264(path):
    """Re-encode to H.264 so browsers and editors play it; keep mp4v if ffmpeg is missing."""
    ff = shutil.which("ffmpeg")
    if not ff:
        return
    tmp = path.with_suffix(".h264.mp4")
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264", "-crf", "24", "-pix_fmt", "yuv420p",
                    str(tmp)], check=True)
    tmp.replace(path)


def yolo_clip(rd, ts, out):
    first = None
    for i, t in enumerate(ts):
        f = vision.load_frame(str(rd), t)
        img = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR)
        people = vision.detect_people_rgb(f, min_conf=0.25)
        for p in people:
            bx, by, bw, bh = [int(v) for v in p["bbox_rgb"]]; x0, y0, x1, y1 = bx, by, bx + bw, by + bh
            cv2.rectangle(img, (x0, y0), (x1, y1), (60, 220, 60), 3)
            d = p.get("range_m") or p.get("depth_m")
            label(img, f"person {p.get('confidence', 0):.2f}" + (f"  {d:.1f} m" if d else ""), (x0 + 4, max(24, y0 - 8)), .6)
        label(img, f"OpenCV 5 dnn: YOLO11m on the robot's camera   t = {t:.1f} s   people: {len(people)}")
        if first is None:
            first = writer(out / "yolo_detections.mp4", (img.shape[1], img.shape[0]))
        first.write(img)
        if people and not (out / "yolo_detections.png").exists():
            cv2.imwrite(str(out / "yolo_detections.png"), img)
    first.release()


def thermal_clip(rd, ts, out):
    w = None; best = (-1, None)
    for t in ts:
        f = vision.load_frame(str(rd), t)
        c = f.thermal.astype(float) / 10 - 273.15
        norm = np.clip((c - 15) / (450 - 15), 0, 1)
        img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        img = cv2.resize(img, (640, 512), interpolation=cv2.INTER_NEAREST)
        sx, sy = 640 / c.shape[1], 512 / c.shape[0]
        for (lo, hi), col, name in (((28, 40), (90, 230, 90), "body heat 28-40 C"), ((200, 1e9), (60, 60, 255), "fire >= 200 C")):
            m = ((c >= lo) & (c <= hi) & (f.thermal > 0)).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cs = [(cc * [sx, sy]).astype(np.int32) for cc in cs if cv2.contourArea(cc) >= 8]
            cv2.drawContours(img, cs, -1, col, 2)
        label(img, f"OpenCV threshold + contours on thermal   t = {t:.1f} s   max {c.max():.0f} C")
        label(img, "green: body heat 28-40 C   red: fire >= 200 C", (14, 500), .55)
        if w is None:
            w = writer(out / "thermal_threshold.mp4", (640, 512))
        w.write(img)
        if c.max() > best[0]: best = (c.max(), img.copy())
    w.release()
    cv2.imwrite(str(out / "thermal_threshold.png"), best[1])


def rgbd_clip(rd, ts, out):
    w = None
    for t in ts:
        f = vision.load_frame(str(rd), t)
        rgb = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR)
        d = np.nan_to_num(f.depth_m, nan=0.0)
        dn = (np.clip(d / 8.0, 0, 1) * 255).astype(np.uint8)
        dep = cv2.applyColorMap(255 - dn, cv2.COLORMAP_TURBO); dep[d <= 0] = 0
        img = np.hstack([rgb, cv2.resize(dep, (rgb.shape[1], rgb.shape[0]))])
        label(img, f"RGB-D: colour and depth (0 to 8 m) from the robot's camera   t = {t:.1f} s")
        if w is None:
            w = writer(out / "rgbd.mp4", (img.shape[1], img.shape[0]))
        w.write(img)
    w.release()
    cv2.imwrite(str(out / "rgbd.png"), img)


def map3d_clip(rd, out, n=160000):
    ply = rd / "map3d.ply"
    raw = ply.read_bytes(); head = raw[:raw.index(b"end_header\n") + 11]
    count = int([l for l in head.decode().splitlines() if l.startswith("element vertex")][0].split()[2])
    pts = np.frombuffer(raw[len(head):], dtype="<f4", count=count * 6).reshape(count, 6)[:, :3]
    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = pts[np.random.default_rng(0).choice(len(pts), min(n, len(pts)), replace=False)]
    c = pts - np.median(pts, axis=0)
    zmin, zmax = np.percentile(c[:, 2], [2, 98])
    col = cv2.applyColorMap((np.clip((c[:, 2] - zmin) / (zmax - zmin + 1e-6), 0, 1) * 255).astype(np.uint8).reshape(-1, 1),
                            cv2.COLORMAP_VIRIDIS).reshape(-1, 3)
    W, H = 1280, 720; w = writer(out / "map3d_tsdf.mp4", (W, H)); scale = 34
    for k in range(FPS * 10):
        a = np.radians(k * 360 / (FPS * 10)); el = np.radians(55)
        x = c[:, 0] * np.cos(a) - c[:, 1] * np.sin(a); y = c[:, 0] * np.sin(a) + c[:, 1] * np.cos(a)
        u = (W / 2 + x * scale).astype(int); v = (H / 2 - (y * np.cos(el) + c[:, 2] * np.sin(el)) * scale).astype(int)
        depth = y * np.sin(el) - c[:, 2] * np.cos(el); order = np.argsort(-depth)
        img = np.full((H, W, 3), 18, np.uint8); ok = (u[order] >= 0) & (u[order] < W) & (v[order] >= 0) & (v[order] < H)
        img[v[order][ok], u[order][ok]] = col[order][ok]
        label(img, f"OpenCV 5 HashTSDF 3D map built live by one robot: {count:,} surface points (colour = height)")
        w.write(img)
        if k == 0:
            cv2.imwrite(str(out / "map3d_tsdf.png"), img)
    w.release()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("rec", type=Path); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--person", default="robot_A:24:38"); ap.add_argument("--fire", default="robot_B:28:42")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    r, t0, t1 = a.person.split(":"); rd = a.rec / r
    yolo_clip(rd, stamps(rd, float(t0), float(t1), 0.2), a.out)
    rgbd_clip(rd, stamps(rd, float(t0), float(t1), 1 / FPS), a.out)
    r2, f0, f1 = a.fire.split(":"); rd2 = a.rec / r2
    thermal_clip(rd2, stamps(rd2, float(f0), float(f1), 1 / FPS), a.out)
    map3d_clip(rd, a.out)
    for p in a.out.glob("*.mp4"):
        h264(p)
    for p in sorted(a.out.iterdir()):
        print(p.name, p.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
