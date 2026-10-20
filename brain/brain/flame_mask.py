"""Hide fire from SLAM: a robot-side filter that blacks out flame pixels, found with the thermal camera, before ORB-SLAM3.

Flames, embers and lit smoke move and flicker, so corners on them look like landmarks that jump around. On the burning
house run, robot A gained 23 degrees of heading error in 8 s while turning beside the living-room fire. This filter uses
only the robot's own sensors. Each colour pixel is looked up in the thermal image (both cameras face the same way; the
3 cm offset between them is ignored) and masked when it reads above 250 C. Outside the thermal camera's narrower view a
strict colour rule is used instead (very bright, saturated, orange-yellow). A colour-only rule was tried first and
masked fire-lit walls and floodlight hotspots too (31 percent of a plain hallway frame). Masked pixels are dilated, set
to black, and their depth invalidated, so ORB-SLAM3 finds no corners there.

    uv run python -m brain.flame_mask <robot_dir> <out_dir>

Writes a complete robot folder (rgb/ masked, depth/ masked, the text files copied) that slam_runner reads as usual.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


HOT_C = 250.0


def _yaml(path: Path) -> dict:
    vals = {}
    for line in path.read_text().splitlines():
        if ":" in line and not line.startswith(("%", "#")):
            k, v = line.split(":", 1)
            try:
                vals[k.strip()] = float(v.strip())
            except ValueError:
                pass
    return vals


def thermal_lookup(rgb_cal: dict, th_cal: dict, w: int, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every colour pixel, the thermal pixel that sees the same direction, and whether it is inside the thermal view."""
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x = (u - rgb_cal["Camera1.cx"]) / rgb_cal["Camera1.fx"]
    y = (v - rgb_cal["Camera1.cy"]) / rgb_cal["Camera1.fy"]
    ut = np.round(th_cal["Camera1.cx"] + th_cal["Camera1.fx"] * x).astype(int)
    vt = np.round(th_cal["Camera1.cy"] + th_cal["Camera1.fy"] * y).astype(int)
    inside = (ut >= 0) & (vt >= 0) & (ut < int(th_cal["Camera.width"])) & (vt < int(th_cal["Camera.height"]))
    return np.clip(ut, 0, int(th_cal["Camera.width"]) - 1), np.clip(vt, 0, int(th_cal["Camera.height"]) - 1), inside


def flame_mask(bgr: np.ndarray, thermal_dk: np.ndarray | None = None, lookup=None, dilate_px: int = 9) -> np.ndarray:
    """True where the image shows fire. With thermal: above 250 C inside the thermal view, strict colour rule outside it."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    colour = (hsv[..., 2] > 245) & (hsv[..., 1] > 130) & (hsv[..., 0] < 35)
    if thermal_dk is not None and lookup is not None:
        ut, vt, inside = lookup
        hot = (thermal_dk[vt, ut].astype(np.float32) / 10.0 - 273.15) > HOT_C
        m = np.where(inside, hot, colour)
    else:
        m = colour
    m = m.astype(np.uint8)
    if dilate_px > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)))
    return m.astype(bool)


def mask_robot(robot_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "rgb").mkdir(exist_ok=True)
    (out_dir / "depth").mkdir(exist_ok=True)
    for f in robot_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, out_dir / f.name)
    fractions = []
    has_thermal = (robot_dir / "thermal.yaml").exists()
    lookup = None
    for line in (robot_dir / "associations.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        t, rgb_rel, _, depth_rel = line.split()
        bgr = cv2.imread(str(robot_dir / rgb_rel), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(robot_dir / depth_rel), cv2.IMREAD_UNCHANGED)
        dk = None
        if has_thermal:
            if lookup is None:
                lookup = thermal_lookup(_yaml(robot_dir / "camera.yaml"), _yaml(robot_dir / "thermal.yaml"), bgr.shape[1], bgr.shape[0])
            tp = robot_dir / "thermal" / f"{t}.png"
            dk = cv2.imread(str(tp), cv2.IMREAD_UNCHANGED) if tp.exists() else None
        m = flame_mask(bgr, dk, lookup if dk is not None else None)
        bgr[m] = 0
        depth[m] = 0
        cv2.imwrite(str(out_dir / rgb_rel), bgr)
        cv2.imwrite(str(out_dir / depth_rel), depth)
        fractions.append(float(m.mean()))
    f = np.array(fractions)
    stats = {"frames": int(f.size), "frames_with_fire": int((f > 0.001).sum()),
             "mean_masked_fraction": float(f.mean()), "max_masked_fraction": float(f.max())}
    (out_dir / "flame_mask_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("robot_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    a = ap.parse_args()
    print(json.dumps(mask_robot(a.robot_dir.resolve(), a.out_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
