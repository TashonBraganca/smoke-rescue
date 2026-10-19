"""One command from a recording to a scored run: C7 SLAM, dense maps, C8 fingerprints, C9 merge, C10 report, C11 replay.

    cd brain && uv run python -m brain.pipeline ../data/recordings/<run_id> [--out ../data/runs/<run_id>] [--device mps]

Every stage is the module's own command line, so a stage can be rerun alone. SLAM is skipped when its inputs are
unchanged: the cache key hashes associations.txt, camera.yaml and the size of every image, never just the path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def input_hash(robot_dir: Path) -> str:
    h = hashlib.sha256()
    for name in ("associations.txt", "camera.yaml"):
        h.update((robot_dir / name).read_bytes())
    for sub in ("rgb", "depth"):
        for p in sorted((robot_dir / sub).glob("*.png")):
            h.update(p.name.encode())
            h.update(str(p.stat().st_size).encode())
    return h.hexdigest()


def step(name: str, args: list[str]) -> float:
    print(f"[pipeline] {name}: {' '.join(args)}", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, "-m", *args], check=True, cwd=ROOT / "brain")
    dt = time.time() - t0
    print(f"[pipeline] {name} done in {dt:.0f} s", flush=True)
    return dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recording", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--force-slam", action="store_true")
    ap.add_argument("--no-viewer", action="store_true")
    args = ap.parse_args()
    rec = args.recording.resolve()
    meta = json.loads((rec / "meta.json").read_text())
    out = (args.out or ROOT / "data" / "runs" / meta["run_id"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    timings = {}

    for robot in meta["robots"]:
        rdir, odir = rec / robot, out / robot
        key = input_hash(rdir)
        cache = odir / "pipeline_input_hash.txt"
        if not args.force_slam and cache.exists() and cache.read_text() == key and (odir / "CameraTrajectory.txt").exists():
            print(f"[pipeline] {robot}: SLAM inputs unchanged, reusing {odir}", flush=True)
        else:
            odir.mkdir(parents=True, exist_ok=True)
            timings[f"slam_{robot}"] = step(f"C7 SLAM {robot}", ["brain.slam_runner", str(rdir), str(odir)])
            cache.write_text(key)
        timings[f"map_{robot}"] = step(f"dense map {robot}", ["brain.dense_map", str(rdir), str(odir)])

    assets = ROOT / "data" / "runs" / "place_assets"
    timings["place"] = step("C8 fingerprints", ["brain.place", str(rec), "--out", str(out),
                                                "--neutral-images", str(assets / "coco128/images/train2017"),
                                                "--assets", str(assets), "--device", args.device])
    timings["merge"] = step("C9 merge", ["brain.merge", str(rec), str(out)])
    timings["evaluate"] = step("C10 evaluate", ["brain.evaluator", str(rec), str(out)])
    if not args.no_viewer:
        timings["viewer"] = step("C11 replay", ["brain.viewer", str(out), "--recording", str(rec)])
    (out / "pipeline_timings.json").write_text(json.dumps(timings, indent=2) + "\n")
    print((out / "report.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
