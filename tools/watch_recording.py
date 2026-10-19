"""Watch a recording live while Unity writes it: each robot's RGB and depth stream into a Rerun window.

    brain/.venv/bin/python tools/watch_recording.py data/recordings/unity_clear

Polls the robot folders for new frames (by file name, which is the sim timestamp) and logs them on a
sim_time timeline. Stops when meta.json appears (the recorder writes it last) and all frames are shown.
"""

import sys
import time
from pathlib import Path

import cv2
import rerun as rr
import rerun.blueprint as rrb


def main() -> None:
    rec = Path(sys.argv[1]).resolve()
    robots = ["robot_A", "robot_B"]
    rr.init(f"smoke_rescue_live_{rec.name}", spawn=True)
    rr.send_blueprint(rrb.Blueprint(rrb.Grid(*[rrb.Spatial2DView(origin=f"{r}/rgb", name=f"{r} camera") for r in robots],
                                             *[rrb.Spatial2DView(origin=f"{r}/depth", name=f"{r} depth") for r in robots])))
    rr.log("info", rr.TextDocument(f"Live view of {rec.name} as Unity records it. Frames appear as they are written to disk."), static=True)
    shown = {r: set() for r in robots}
    while True:
        new = 0
        for r in robots:
            folder = rec / r / "rgb"
            if not folder.exists():
                continue
            names = sorted(p.name for p in folder.glob("*.png"))
            for name in names[-30:]:
                if name in shown[r]:
                    continue
                rgb_path, depth_path = folder / name, rec / r / "depth" / name
                if not depth_path.exists() or time.time() - rgb_path.stat().st_mtime < 0.5:
                    continue   # still being written
                bgr = cv2.imread(str(rgb_path))
                depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                if bgr is None or depth is None:
                    continue
                rr.set_time("sim_time", duration=float(name[:-4]))
                rr.log(f"{r}/rgb", rr.Image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).compress(jpeg_quality=80))
                rr.log(f"{r}/depth", rr.DepthImage(depth, meter=1000.0, colormap=rr.components.Colormap.Turbo))
                shown[r].add(name)
                new += 1
        if (rec / "meta.json").exists() and new == 0:
            break
        time.sleep(0.3)
    print("recording finished; live view complete")


if __name__ == "__main__":
    main()
