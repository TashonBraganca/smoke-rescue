"""Front-camera walkthrough video from a recording: each robot's RGB on top; depth, thermal and ORB keypoints below.

With --run, the narrator's sentences (narration.jsonl) are shown under each robot's view for 6 s after each one.

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


IRONBOW = np.array([(0, 0, 0), (92, 0, 33), (153, 0, 140), (64, 51, 217), (0, 140, 250), (38, 217, 255), (230, 255, 255)],
                   np.float32)   # BGR stops: black, deep purple, magenta, red, orange, yellow, white


def ironbow(dk: np.ndarray) -> np.ndarray:
    """Thermal frame (kelvin x 10) to ironbow false colour with plateau histogram equalisation, like a thermal camera."""
    lo, hi = np.percentile(dk, [1, 99])
    if hi - lo < 100:
        # Nearly uniform scene (a wall at one temperature): equalising would stretch 0.05 K sensor noise over the whole
        # palette. Thermal cameras keep a minimum span instead; use a linear 10 K window around the median.
        mid = float(np.median(dk))
        v = np.clip((dk.astype(np.float32) - (mid - 50)) / 100.0, 0, 1) * (len(IRONBOW) - 1)
    else:
        hist = np.bincount(dk.ravel(), minlength=65536).astype(np.float64)
        hist = np.minimum(hist, max(4, dk.size // 300))
        lut = np.cumsum(hist) / max(hist.sum(), 1)
        v = lut[dk] * (len(IRONBOW) - 1)
    i = np.minimum(v.astype(int), len(IRONBOW) - 2)
    f = (v - i)[..., None]
    return (IRONBOW[i] * (1 - f) + IRONBOW[i + 1] * f).astype(np.uint8)


def thermal_frames(robot_dir: Path) -> dict[float, Path]:
    index = robot_dir / "thermal.txt"
    if not index.exists():
        return {}
    out = {}
    for line in index.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        t, rel = line.split()
        out[round(float(t), 4)] = robot_dir / rel
    return out


def narration(run_dir: Path | None, robot: str) -> list[tuple[float, str]]:
    if run_dir is None or not (run_dir / robot / "narration.jsonl").exists():
        return []
    rows = [json.loads(l) for l in (run_dir / robot / "narration.jsonl").read_text().splitlines() if l.strip()]
    return sorted((float(r["t"]), r["text"]) for r in rows)


def wrap(text: str, width: int = 70) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + ([cur] if cur else [])


def label(img: np.ndarray, text: str, colour: tuple[int, int, int]) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.rectangle(img, (0, 0), (8, 30), colour, -1)
    cv2.putText(img, text, (16, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recording", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--run", type=Path, help="run folder with <robot>/narration.jsonl to caption the video")
    ap.add_argument("--map", action="store_true", help="add the 2D house map panel (needs --run with SLAM and merge outputs)")
    ap.add_argument("--chat", action="store_true", help="add the team radio chat panel (needs --run)")
    ap.add_argument("--context", action="store_true", help="add the shared-context panel (brain estimates on the building plan)")
    a = ap.parse_args()
    meta = json.loads((a.recording / "meta.json").read_text())
    robots = meta["robots"]
    out = a.out or Path("data/evidence") / f"walkthrough_{meta['run_id']}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    tracks = {r: frames(a.recording / r) for r in robots}
    colours = [(60, 120, 255), (255, 170, 40), (90, 220, 90)]   # BGR: orange-red A, blue B, green C
    clock = sorted({round(t, 4) for fr in tracks.values() for t, _, _ in fr})
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{W * len(robots) + (640 if a.map else 0) + (640 if a.context else 0) + (640 if a.chat else 0)}x{H + 160}", "-r", str(a.fps), "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)], stdin=subprocess.PIPE)
    orbs = {r: orb_from_yaml(a.recording / r / "camera.yaml") for r in robots}
    thermals = {r: thermal_frames(a.recording / r) for r in robots}
    captions = {r: narration(a.run, r) for r in robots}
    minimap = None
    if a.map:
        from minimap import MiniMap
        minimap = MiniMap(a.recording, a.run)
    context = None
    if a.context:
        from contextpanel import ContextPanel
        context = ContextPanel(a.recording)
    chat = None
    if a.chat:
        from chatpanel import ChatPanel
        chat = ChatPanel(a.recording, a.run)
    PW, PH = W // 3, 160   # three bottom panels (even height for the encoder)
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
                dep = cv2.resize(depth_colour(dep_p), (PW, PH), interpolation=cv2.INTER_NEAREST)
                cv2.putText(dep, "depth 0-6 m", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                kps = orbs[r].detect(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), None)
                orb = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) // 2
                for kp in kps:
                    cv2.circle(orb, (int(kp.pt[0]), int(kp.pt[1])), 3, (80, 255, 120), 1, cv2.LINE_AA)
                orb = cv2.resize(orb, (PW, PH), interpolation=cv2.INTER_AREA)
                cv2.putText(orb, f"ORB corners {len(kps)}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
                tp = thermals[r].get(round(ti, 4))
                if tp is not None and tp.exists():
                    dk = cv2.imread(str(tp), cv2.IMREAD_UNCHANGED)
                    th = cv2.resize(ironbow(dk), (PW, PH), interpolation=cv2.INTER_NEAREST)
                    hot = dk.max() / 10 - 273.15
                    cv2.putText(th, f"thermal, max {hot:.0f} C", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    th = np.zeros((PH, PW, 3), np.uint8)
                    cv2.putText(th, "no thermal", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
                cache[r] = (ti, rgb, dep, orb, th)
            _, rgb, dep, orb, th = cache[r]
            top = rgb.copy()
            label(top, f"{r.replace('_', ' ')}  front camera  t = {t:6.1f} s", colours[k % len(colours)])
            said = [txt for ct, txt in captions[r] if ct <= t < ct + 6.0]
            if said:
                lines = wrap(said[-1])
                y0 = H - 12 - 22 * len(lines)
                cv2.rectangle(top, (0, y0 - 8), (W, H), (0, 0, 0), -1)
                for n, line in enumerate(lines):
                    cv2.putText(top, line, (12, y0 + 14 + 22 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
            bottom = np.zeros((PH, W, 3), np.uint8)
            bottom[:, :PW] = dep
            bottom[:, PW:2 * PW] = th
            bottom[:, 2 * PW:3 * PW] = orb
            cols.append(np.vstack([top, bottom]))
        if minimap is not None:
            cols.append(minimap.render(t))
        if context is not None:
            cols.append(context.render(t))
        if chat is not None:
            cols.append(chat.render(t))
        ff.stdin.write(np.hstack(cols).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{out}: {len(clock)} frames, {len(clock) / a.fps:.0f} s")


if __name__ == "__main__":
    main()
