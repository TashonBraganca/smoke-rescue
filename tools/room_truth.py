#!/usr/bin/env python3
"""Compare each robot's room ledger with the real floor plan (evaluator side; robots never see the world file).

    python tools/room_truth.py data/runs/headless/<exp>

For every run folder (<seed>_<variant>/result.json) and robot: ledger rooms of at least 2 m2, slivers under 1 m2,
real rooms with exactly one ledger room centred in them, and ledger rooms bigger than 1.3 x the real room their centre
lies in (several real rooms merged into one). Prints per run and the mean per variant.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def inside(p, poly):
    x, y = p; ok = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            ok = not ok
    return ok


def area(poly):
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]))) / 2


def score(run: Path, root: Path) -> dict:
    r = json.loads((run / "result.json").read_text())
    world = json.loads((root / r["world"]).read_text()) if not Path(r["world"]).is_absolute() else json.loads(Path(r["world"]).read_text())
    real = [(q["kind"], q["poly"], area(q["poly"])) for q in world["rooms"]]
    out = {}
    for m in r["maps"]:
        rooms = m["rooms"]
        big = [q for q in rooms if q["floor_m2"] >= 2]
        sliver = sum(q["floor_m2"] < 1 for q in rooms)
        hits = defaultdict(int); merged = 0
        for q in big:
            home = next((k for k, (_, poly, _) in enumerate(real) if inside((q["x"], q["y"]), poly)), None)
            if home is None:
                continue
            hits[home] += 1
            if q["floor_m2"] > 1.3 * real[home][2]:
                merged += 1
        out[m["robot"]] = {"ledger_rooms": len(rooms), "rooms_2m2": len(big), "slivers": sliver,
                           "real_matched_once": sum(1 for v in hits.values() if v == 1), "real_rooms": len(real), "merged": merged}
    return out


def main(folder: Path):
    root = Path(__file__).resolve().parents[1]
    by = defaultdict(list)
    for run in sorted(p for p in folder.iterdir() if (p / "result.json").exists()):
        seed, variant = run.name.split("_", 1)
        s = score(run, root)
        for robot, v in s.items():
            by[variant].append(v)
            print(f"{run.name:28s} {robot}  ledger {v['ledger_rooms']:4d}  >=2m2 {v['rooms_2m2']:3d}  slivers {v['slivers']:4d}  "
                  f"real matched once {v['real_matched_once']}/{v['real_rooms']}  merged {v['merged']}")
    for variant, rows in by.items():
        mean = {k: round(sum(r[k] for r in rows) / len(rows), 1) for k in rows[0]}
        print(variant, mean)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
