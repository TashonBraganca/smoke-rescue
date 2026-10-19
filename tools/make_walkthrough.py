"""Front-camera walkthrough video from a recording: each robot's RGB on top, its depth and ORB keypoints below.

    brain/.venv/bin/python tools/make_walkthrough.py data/recordings/unity_haze [--out data/evidence/walkthrough_unity_haze.mp4]

Reads only the robots' own sensor output (rgb/, depth/, associations.txt), the same data SLAM sees.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

import re

W, H = 640, 480


def orb_from_yaml(camera_yaml: Path) -> cv2.ORB:
    """An OpenCV ORB with the extractor settings ORB-SLAM3 reads from this robot's camera.yaml."""
    txt = camera_yaml.read_text()
    get = lambda k: float(re.search(rf"ORBextractor\.{k}:\s*([0-9.]+)", txt).group(1))
    return cv2.ORB_create(int(get("nFeatures")), get("scaleFactor"), int(get("nLevels")), fastThreshold=int(get("iniThFAST")))


def frames(robot_dir: Path) -> list[tuple[float, Path, Path]]:
    out = []
    for line in (robot_dir / "associations.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        t, rgb, _, depth = line.split()
        out.append((float(t), robot_dir / rgb, robot_dir / depth))
    return out


def depth_colour(path: Path) -> np.ndarray:
    mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    z = np.clip(mm.astype(np.float32) / 6000.0, 0, 1)
    img = cv2.applyColorMap((255 * (1 - z)).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[mm == 0] = 0
    return img


def label(img: np.ndarray, text: str, colour: tuple[int, int, int]) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.rectangle(img, (0, 0), (8, 30), colour, -1)
    cv2.putText(img, text, (16, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recording", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--fps", type=float, default=15.0)
    a = ap.parse_args()
    meta = json.loads((a.recording / "meta.json").read_text())
    robots = meta["robots"]
    out = a.out or Path("data/evidence") / f"walkthrough_{meta['run_id']}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    tracks = {r: frames(a.recording / r) for r in robots}
    colours = [(60, 120, 255), (255, 170, 40), (90, 220, 90)]   # BGR: orange-red A, blue B, green C
    clock = sorted({round(t, 4) for fr in tracks.values() for t, _, _ in fr})
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W * len(robots)}x{H + H // 2}", "-r", str(a.fps), "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)], stdin=subprocess.PIPE)
    orbs = {r: orb_from_yaml(a.recording / r / "camera.yaml") for r in robots}
    idx = {r: 0 for r in robots}
    cache: dict[str, tuple] = {}
    for t in clock:
        cols = []
        for k, r in enumerate(robots):
            fr = tracks[r]
            while idx[r] + 1 < len(fr) and fr[idx[r] + 1][0] <= t + 1e-4:
                idx[r] += 1
            ti, rgb_p, dep_p = fr[idx[r]]
            if cache.get(r, (None,))[0] != ti:
                rgb = cv2.imread(str(rgb_p))
                dep = cv2.resize(depth_colour(dep_p), (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
                kps = orbs[r].detect(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), None)
                orb = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) // 2
                for kp in kps:
                    cv2.circle(orb, (int(kp.pt[0]), int(kp.pt[1])), 3, (80, 255, 120), 1, cv2.LINE_AA)
                orb = cv2.resize(orb, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
                cv2.putText(orb, f"ORB {len(kps)}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
                cache[r] = (ti, rgb, dep, orb)
            _, rgb, dep, orb = cache[r]
            top = rgb.copy()
            label(top, f"{r.replace('_', ' ')}  front camera  t = {t:6.1f} s", colours[k % len(colours)])
            bottom = np.zeros((H // 2, W, 3), np.uint8)
            bottom[:, :W // 2] = dep
            bottom[:, W // 2:] = orb
            cv2.putText(bottom, "depth 0-6 m", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cols.append(np.vstack([top, bottom]))
        ff.stdin.write(np.hstack(cols).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{out}: {len(clock)} frames, {len(clock) / a.fps:.0f} s")


if __name__ == "__main__":
    main()
