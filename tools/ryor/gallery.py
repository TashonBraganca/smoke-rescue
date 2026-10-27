#!/usr/bin/env python3
"""Build the Run-your-own-rescue gallery: one world per house and robot count, one request per setup, and gallery.json.

    brain/.venv/bin/python tools/ryor/gallery.py [--hx 24] [--real <real_layouts dir>]

Evaluator side only: the brains never read these files. Writes data/runs/ryor/ (git-ignored, because the HouseExpo-derived
plans stay out of git; see handoff/real_layouts_report_25oct.txt on the SUNCG question):
  worlds/<house>__r<n>.json      compiled world with 1, 2 or 3 robots
  houses/                        floor plan and fire files of the HouseExpo houses (copied, paths rewritten)
  setups/<house>__r<n>__<diff>.json   the simulator request for that setup ("out" is filled in at run time)
  gallery.json                   what the page draws (tools/ryor/CONTRACT.txt section 2)
Houses: ours (family ranch, long ranch, the one-door house of seed 934) and the first --hx HouseExpo houses in the converter's
own shuffled order (tools/bench/real_layouts.py, data/runs/real_layouts_25oct/manifest.json). People and fire are the ones the
existing generators placed; nothing is chosen by hand.
Every setup runs with parallel_robots (tools/ryor/patch_headless.py): live and cached runs use the same mode, so a setup
always gives the same mission. Robots: 1 = robot_A only; 2 = robot_A and robot_B as compiled; 3 = adds robot_C on the next pad of the same queue, 1.2 m behind
robot_B (the queue spacing every compiled world uses: B's pad is 1.2 m behind A's).
"""
import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "bench"))
from headless_world import compile_world, load_scenario  # noqa: E402
from make_requests import fire_grid_plan  # noqa: E402

CONFIG = ROOT / "config"
OUT = ROOT / "data/runs/ryor"
MISSION_S = 180
LEAVE_BY_S = 140
LEAVE_BY_REASON = ("The brain searches until mission_time_limit_s and only then walks out (return_home_early off, owner 24 Oct). "
                   "140 s of a 180 s mission leaves 40 s for the walk out, so the replay ends with robots leaving. Three-bed ranch, "
                   "medium, 2 robots, limit 170 s: both were still inside at 180 s.")
DIFFICULTIES = {
    "easy": {"head_start_s": 0, "visibility_m": 6.0, "label": "Fire just started. Light smoke: people visible up to 6 m."},
    "medium": {"head_start_s": 90, "visibility_m": 4.0, "label": "Fire burning for 90 s before entry. Smoke: people visible up to 4 m."},
    "hard": {"head_start_s": 180, "visibility_m": 2.5, "label": "Fire burning for 3 min before entry. Thick smoke: people visible up to 2.5 m."},
}
VISIBILITY_REASON = ("6 m is the brain's own detection limit (person_rgb_max_range_m), so easy adds no smoke limit. 4 and 2.5 m are "
                     "illustrative smoke levels, not measured from a smoke model: headless has no rendered camera.")
OURS = [("family_ranch", "Three-bed ranch", "scenario_family_ranch_full.json"),
        ("long_ranch", "L-shaped ranch", "scenario_long_ranch_full.json"),
        ("onedoor_934", "One-door house", "scenario_house_onedoor_seed934_full.json")]
HOUSEEXPO_LICENCE = "MIT (HouseExpo, Copyright (c) 2019 Tingguang Li)"
CITATION = ("HouseExpo: T. Li, D. Ho, C. Li, D. Zhu, C. Wang, M. Q.-H. Meng, \"HouseExpo: A Large-scale 2D Indoor Layout Dataset for "
            "Learning-based Algorithms on Mobile Robots\", IROS 2020. https://github.com/TeaganLi/HouseExpo, MIT licence. Layouts "
            "converted to our floor-plan format by tools/bench/real_layouts.py.")
KIND_NAMES = {"bedroom": "bedroom", "bathroom": "bathroom", "kitchen": "kitchen", "living": "living room", "study": "study",
              "corridor": "corridor", "closet": "closet", "laundry": "utility room"}


def with_robots(world, n):
    w = copy.deepcopy(world)
    a, b = w["robots"][0], w["robots"][1]
    if n == 1:
        w["robots"] = [a]
    elif n == 3:
        c = copy.deepcopy(b)
        step = [b["pad"][0] - a["pad"][0], b["pad"][1] - a["pad"][1]]
        c["id"] = "robot_C"
        c["at"] = [round(b["at"][0] + step[0], 4), round(b["at"][1] + step[1], 4)]
        c["pad"] = [b["pad"][0] + step[0], b["pad"][1] + step[1], b["pad"][2]]
        w["robots"] = [a, b, c]
    return w


def poly_area(poly):
    return abs(sum(poly[i][0] * poly[i - 1][1] - poly[i - 1][0] * poly[i][1] for i in range(len(poly)))) / 2


def house_entry(hid, name, source, source_id, licence, world):
    walls = [[round(v, 3) for v in s[:4]] for s in world["segments"] if s[5] - s[4] > 1.5]
    low = [[round(v, 3) for v in s[:4]] for s in world["segments"] if s[5] - s[4] <= 1.5]
    furniture = [[[round(x, 3), round(y, 3)] for x, y in p["poly"]] for p in world["prisms"]]
    rooms = [{"id": r["id"], "kind": r["kind"], "name": KIND_NAMES.get(r["kind"], r["kind"]),
              "poly": [[round(x, 3), round(y, 3)] for x, y in r["poly"]]} for r in world["rooms"]]
    xs = [v for w in walls for v in (w[0], w[2])] + [r["at"][0] for r in world["robots"]]
    ys = [v for w in walls for v in (w[1], w[3])] + [r["at"][1] for r in world["robots"]] + [-2.4]
    return {"id": hid, "name": name, "source": source, "source_id": source_id, "licence": licence,
            "rooms": len(rooms), "floor_m2": round(sum(poly_area(r["poly"]) for r in rooms), 1), "people": len(world["people"]),
            "bounds": [round(min(xs), 2), round(min(ys), 2), round(max(xs), 2), round(max(ys), 2)],
            "walls": walls, "low_walls": low, "furniture": furniture, "rooms_poly": rooms,
            "pads": [[r["pad"][0], r["pad"][1]] for r in with_robots(world, 3)["robots"]],
            "people_truth": [{"id": p["id"], "x": p["at"][0], "y": p["at"][1]} for p in world["people"]]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hx", type=int, default=24, help="how many HouseExpo houses")
    ap.add_argument("--real", type=Path, default=Path("/Volumes/Tashon/opencv/smoke-rescue/data/runs/real_layouts_25oct"))
    args = ap.parse_args()
    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ("worlds", "houses", "setups", "fires"):
        (OUT / sub).mkdir(parents=True)
    rel = lambda p: str(Path(p).resolve().relative_to(ROOT))
    worlds, entries = {}, []
    for hid, name, scen in OURS:
        path = CONFIG / scen
        sc = load_scenario(path)
        sc.setdefault("floorplan_file", "floorplan.json")   # compile_world's own defaults
        sc["fire"] = {"config_file": "fire.json", **sc.get("fire", {})}
        world = compile_world(path.resolve())
        fp = fire_grid_plan(sc, OUT, hid)
        if fp:
            world["fire"]["floorplan"] = fp
        world["ryor"] = {"house": hid, "scenario": scen}
        worlds[hid] = world
        entries.append(house_entry(hid, name, "ours", scen, "ours", world))
    manifest = json.loads((args.real / "manifest.json").read_text())
    seen = []
    for run in manifest["runs"]:
        if run.get("variant") != "full" or run["house"] in seen:
            continue
        seen.append(run["house"])
        if len(seen) > args.hx:
            break
        hid = run["house"]
        world = json.loads((ROOT.parent / "smoke-rescue" / run["world"]).read_text()) if not (ROOT / run["world"]).exists() \
            else json.loads((ROOT / run["world"]).read_text())
        for key in ("floorplan", "config"):
            src = (CONFIG / world["fire"][key]).resolve() if (CONFIG / world["fire"][key]).exists() else \
                (args.real / "houses" / Path(world["fire"][key]).name)
            dst = OUT / "houses" / src.name
            shutil.copy(src, dst)
            world["fire"][key] = os.path.relpath(dst, CONFIG)
        world["ryor"] = {"house": hid, "source": world.get("bench", {}).get("source")}
        worlds[hid] = world
        src_id = world.get("bench", {}).get("source", hid).replace("HouseExpo ", "")
        n = len(entries) - len(OURS) + 1
        entries.append(house_entry(hid, f"House {n:02d}", "HouseExpo", src_id, HOUSEEXPO_LICENCE, world))
    setups = []
    for hid, world in worlds.items():
        for n in (1, 2, 3):
            w = with_robots(world, n)
            wp = OUT / "worlds" / f"{hid}__r{n}.json"
            wp.write_text(json.dumps(w))
            for diff, d in DIFFICULTIES.items():
                key = f"{hid}__r{n}__{diff}"
                req = {"runs": [{"world": rel(wp), "out": "", "duration_s": MISSION_S, "time_limit": True, "fire": True,
                                 "robot_collision": True, "capsule_walls": True,
                                 "settings": {"mission_time_limit_s": LEAVE_BY_S}, "head_start_s": d["head_start_s"],
                                 "visibility_m": d["visibility_m"], "replay_every_s": 0.5,
                                 "parallel_robots": True}],
                       "ryor": {"setup": key, "house": hid, "robots": n, "difficulty": diff}}
                (OUT / "setups" / f"{key}.json").write_text(json.dumps(req, indent=1) + "\n")
                setups.append(key)
    gallery = {"v": 1, "mission_s": MISSION_S, "leave_by_s": LEAVE_BY_S, "leave_by_reason": LEAVE_BY_REASON,
               "difficulties": DIFFICULTIES, "visibility_reason": VISIBILITY_REASON, "robots": [1, 2, 3],
               "grid": {"cell_m": 0.1, "origin": [-25.0, -25.0], "size": 500},
               "houses": entries, "citation": CITATION}
    (OUT / "gallery.json").write_text(json.dumps(gallery, separators=(",", ":")) + "\n")
    print(f"{len(entries)} houses ({len(OURS)} ours, {len(entries) - len(OURS)} HouseExpo), {len(setups)} setups -> {rel(OUT)}")


if __name__ == "__main__":
    main()
