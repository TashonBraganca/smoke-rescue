#!/usr/bin/env python3
"""Convert real HouseExpo floor plans (SUNCG-derived) into benchmark houses and requests (evaluator side only).

    brain/.venv/bin/python -I tools/bench/real_layouts.py data/real/layouts_src/houseexpo_json/json data/runs/real_layouts_25oct
    brain/.venv/bin/python -I tools/bench/real_layouts.py <json_dir> <out> --count 2 --duration 60    (validation)

Input: HouseExpo JSON (id, bbox, verts = free-space outline in metres, room_category = {category: [[x1,y1,x2,y2], ...]}).
HouseExpo has no doors, so doors are read from the outline: an opening is where free floor crosses a shared wall line.
The files are read as data only (json.load); nothing from the download is imported or executed.

Selection: every file name sorted, shuffled with random.Random(--seed); keep houses whose room count is 6..15 and convert
in that order until --count houses pass every check. Each conversion step below either succeeds or rejects the house
with a named reason; the reasons are counted in <out>/real_layouts_stats.json.

Conversion (no house-specific constants; every constant has a *_REASON):
 1. Grid of CELL m over the house bbox, snapped outward to the grid; x -> our x, HouseExpo y -> our z, shifted to (0, 0).
 2. Each cell takes the label of the smallest-area room bbox that contains its centre (SUNCG bboxes of irregular rooms
    overlap; the smaller one is the more specific room).
 3. A room is kept as the bounding rectangle of its cells if the cells fill >= RECT_FILL of it; rectangles must not
    overlap. Our floor-plan format and HouseBuilder only build rectangular rooms that tile a rectangular footprint.
 4. Cells covered by no room rectangle are split into maximal rectangles and become corridor rooms only if >= FREE_SHARE
    of their cells lie on HouseExpo free floor; otherwise the house is rejected (not rectangular enough).
 5. Doors: for each pair of rooms sharing an edge, the longest run along it where every sample across WALL_PROBE_M on
    both sides is free floor; a run of >= MIN_OPENING_M becomes one door, width clamped to the format's 0.8..1.0 m.
 6. One entrance and the robots' pads, furniture, people, fire origin: the existing generator's own functions and rules
    (tools/random_layout make_doors entrance rule, place_furniture, people, fire draw, validate).
Snap shift: |snapped edge - original bbox edge| for every kept room edge; the maximum is reported. Room edges then lie on the
0.25 m fire grid, so make_requests.fire_grid_plan's extra snap is not needed (checked per house).
"""
import argparse
import copy
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "bench"))
import random_layout as rl  # noqa: E402
from headless_world import compile_world, load_scenario  # noqa: E402
from make_requests import COMMON_SETTINGS, aligned  # noqa: E402

CONFIG = ROOT / "config"
CELL = 0.25
CELL_REASON = "Fire solver cell_size_m in config/fire.json; room edges must sit on it (FireSim rule, see make_requests.fire_grid_plan)."
RECT_FILL = 0.90
RECT_FILL_REASON = "A room is accepted as a rectangle if its cells fill 90 % of their bounding box; below that the room is L-shaped or merged and a rectangle would misstate it."
FREE_SHARE = 0.5
FREE_SHARE_REASON = "Leftover area becomes corridor only if most of it is real HouseExpo floor; otherwise it is outside the house and our rectangular footprint cannot express it."
WALL_PROBE_M = 0.20
WALL_PROBE_REASON = "Probe 0.20 m each side of the snapped wall line: covers the 0.125 m maximum snap shift plus a typical SUNCG wall half-thickness."
MIN_OPENING_M = 0.6
MIN_OPENING_REASON = "Openings narrower than 0.6 m are wall artefacts in the outline, not doorways; real interior doors are 0.7 m or wider."
OVERLAP_TRIM_CELLS = 1
OVERLAP_TRIM_REASON = "SUNCG room boxes include the wall, so neighbours often share one 0.25 m strip; trimming one cell is within the snap error, deeper overlap is a real shape mismatch."
ROOMS_MIN, ROOMS_MAX = 6, 15
ROOMS_REASON = "Owner brief item C: single-floor houses of 6 to 15 rooms."
VARIANTS = ["full", "one_robot"]
VARIANTS_REASON = "Owner brief item C: full vs one_robot only."
KIND = {"Bedroom": "bedroom", "Child_Room": "bedroom", "Guest_Room": "bedroom", "Room": "bedroom",
        "Living_Room": "living", "Dining_Room": "living", "Kitchen": "kitchen", "Bathroom": "bathroom", "Toilet": "bathroom",
        "Office": "study", "Gym": "study", "Hallway": "corridor", "Hall": "corridor", "Entryway": "corridor", "Lobby": "corridor",
        "Storage": "closet", "Wardrobe": "closet", "Boiler_room": "laundry", "Garage": "laundry"}
KIND_REASON = ("SUNCG labels mapped to our eight kinds. 'Room' (unlabelled) counts as bedroom, the most common habitable room. Garage counts as laundry (an unfurnished-for-fire utility space; we have no garage kind). "
               "Balcony, Terrace, Loggia, Aeration and elevators are not indoor single-floor living space: houses with them are rejected.")
VALIDATE_IGNORED = "Room count or duplicate ids"
VALIDATE_IGNORED_REASON = "random_layout.validate hard-codes 7..12 rooms for generated houses; the brief asks for 6..15, so only that check is replaced by ROOMS_MIN..ROOMS_MAX plus a duplicate-id check."


def unique_rooms(h):
    """SUNCG gives a multi-use room once per label with the same box (e.g. Kitchen and Dining_Room). One box is one room;
    its label is the first of its labels in KIND order (unknown labels first, so such houses are still rejected)."""
    order = {k: i for i, k in enumerate(KIND)}
    by_box = {}
    for cat, bs in h["room_category"].items():
        for b in bs:
            by_box.setdefault(tuple(b), []).append(cat)
    return [(min(cats, key=lambda c: order.get(c, -1)), list(b)) for b, cats in by_box.items()]


class Reject(Exception):
    pass


def snapv(v):
    return round(round(v / CELL) * CELL, 6)


def rectangles(mask):
    """Greedy maximal rectangles over a boolean grid (rows = z, cols = x). Returns [(r0, c0, r1, c1)] exclusive ends."""
    m = mask.copy()
    out = []
    while m.any():
        r0, c0 = map(int, np.argwhere(m)[0])
        c1 = c0
        while c1 < m.shape[1] and m[r0, c1]:
            c1 += 1
        r1 = r0 + 1
        while r1 < m.shape[0] and m[r1, c0:c1].all():
            r1 += 1
        m[r0:r1, c0:c1] = False
        out.append((r0, c0, r1, c1))
    return out


def convert(h, rng):
    """HouseExpo dict -> (plan, fire, scenario dict, stats). Raises Reject with a reason."""
    rooms_src = unique_rooms(h)
    bad = sorted({c for c, _ in rooms_src if c not in KIND})
    if bad:
        raise Reject(f"non-living-space category {bad[0]}")
    (bx0, by0), (bx1, by1) = h["bbox"]["min"], h["bbox"]["max"]
    x0, z0 = math.floor(bx0 / CELL) * CELL, math.floor(by0 / CELL) * CELL
    nx, nz = math.ceil((bx1 - x0) / CELL - 1e-9), math.ceil((by1 - z0) / CELL - 1e-9)
    cx = x0 + (np.arange(nx) + 0.5) * CELL
    cz = z0 + (np.arange(nz) + 0.5) * CELL
    GX, GZ = np.meshgrid(cx, cz)
    outline = MPath(np.array(h["verts"], float))
    free_at = lambda pts: outline.contains_points(np.asarray(pts, float).reshape(-1, 2))
    free = free_at(np.c_[GX.ravel(), GZ.ravel()]).reshape(nz, nx)
    area = [(b[2] - b[0]) * (b[3] - b[1]) for _, b in rooms_src]
    label = -np.ones((nz, nx), int)
    best = np.full((nz, nx), np.inf)
    for i, (_, b) in enumerate(rooms_src):
        m = (GX >= b[0]) & (GX <= b[2]) & (GZ >= b[1]) & (GZ <= b[3]) & (area[i] < best)
        label[m], best[m] = i, area[i]
    boxes = []
    for i, (cat, b) in enumerate(rooms_src):
        cells = np.argwhere(label == i)
        if len(cells) == 0:
            raise Reject("room hidden inside another room's box")
        r0, c0 = cells.min(0)
        r1, c1 = cells.max(0) + 1
        if len(cells) < RECT_FILL * (r1 - r0) * (c1 - c0):
            raise Reject("room not rectangular")
        boxes.append([int(r0), int(c0), int(r1), int(c1)])
    # Neighbouring rectangles that overlap by at most OVERLAP_TRIM_CELLS (a shared wall counted by both) are trimmed:
    # the larger room gives up the strip. Deeper overlaps are rejected.
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            a, b = boxes[i], boxes[j]
            if i == j or (a[2] - a[0]) * (a[3] - a[1]) < (b[2] - b[0]) * (b[3] - b[1]):
                continue
            dr, dc = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
            if dr <= 0 or dc <= 0:
                continue
            if min(dr, dc) > OVERLAP_TRIM_CELLS:
                raise Reject("room rectangles overlap")
            if dr <= dc:   # thin in rows: move a's row edge that lies inside b
                if b[0] <= a[0] < b[2]: a[0] = b[2]
                elif b[0] < a[2] <= b[2]: a[2] = b[0]
                else: raise Reject("room rectangles overlap")
            else:
                if b[1] <= a[1] < b[3]: a[1] = b[3]
                elif b[1] < a[3] <= b[3]: a[3] = b[1]
                else: raise Reject("room rectangles overlap")
            if a[2] <= a[0] or a[3] <= a[1]:
                raise Reject("room rectangles overlap")
    rects, shift = [], 0.0
    taken = np.zeros((nz, nx), bool)
    for (cat, b), (r0, c0, r1, c1) in zip(rooms_src, boxes):
        if taken[r0:r1, c0:c1].any():
            raise Reject("room rectangles overlap")
        taken[r0:r1, c0:c1] = True
        rect = [c0 * CELL, r0 * CELL, c1 * CELL, r1 * CELL]
        orig = [b[0] - x0, b[1] - z0, b[2] - x0, b[3] - z0]
        # Shift counted only on edges within one cell of the room's own bbox edge (a trimmed or cut edge is not a snap).
        for k in range(4):
            if abs(rect[k] - orig[k]) <= CELL:
                shift = max(shift, abs(rect[k] - orig[k]))
        rects.append((KIND[cat], cat, rect))
    filler = 0
    for r0, c0, r1, c1 in rectangles(~taken):
        if free[r0:r1, c0:c1].mean() < FREE_SHARE:
            raise Reject("footprint not rectangular (area outside the house)")
        rects.append(("corridor", "filler", [c0 * CELL, r0 * CELL, c1 * CELL, r1 * CELL]))
        filler += (r1 - r0) * (c1 - c0)
    if not ROOMS_MIN <= len(rects) <= ROOMS_MAX:
        raise Reject("room count outside 6..15 after conversion")
    fp = [0.0, 0.0, nx * CELL, nz * CELL]
    names = Counter()
    rooms = []
    for j, (kind, cat, rect) in enumerate(rects):
        names[cat] += 1
        rooms.append(dict(id=f"r{j+1:02}", name=f"{cat.replace('_', ' ')} {names[cat]}", kind=kind, source_label=cat,
                          rect=[snapv(v) for v in rect], floor="carpet_beige", walls="paint_warm_grey_textured", ceiling="plaster_white"))
    # Doors from the outline.
    doors = []
    probe = np.arange(-WALL_PROBE_M, WALL_PROBE_M + 1e-9, 0.05)
    for i, a in enumerate(rooms):
        for b in rooms[i + 1:]:
            ra, rb = a["rect"], b["rect"]
            for axis, cross, along in (("x", 0, 1), ("z", 1, 0)):
                lo, hi = max(ra[along], rb[along]), min(ra[along + 2], rb[along + 2])
                if hi - lo < 1.6:   # same minimum shared edge as random_layout.shared_edge
                    continue
                if abs(ra[cross + 2] - rb[cross]) < 1e-6:
                    line = ra[cross + 2]
                elif abs(rb[cross + 2] - ra[cross]) < 1e-6:
                    line = rb[cross + 2]
                else:
                    continue
                ts = np.arange(lo + 0.05, hi - 0.05 + 1e-9, 0.05)
                pts = []
                for t in ts:
                    for o in probe:
                        p = (line + o, t) if axis == "x" else (t, line + o)
                        pts.append((p[0] + x0, p[1] + z0))
                ok = free_at(pts).reshape(len(ts), len(probe)).all(1)
                best_run, run_start = (0, 0), None
                for k, v in enumerate(list(ok) + [False]):
                    if v and run_start is None:
                        run_start = k
                    if not v and run_start is not None:
                        if k - run_start > best_run[1] - best_run[0]:
                            best_run = (run_start, k)
                        run_start = None
                length = (best_run[1] - best_run[0]) * 0.05
                if length < MIN_OPENING_M:
                    continue
                width = min(1.0, max(0.8, round(length, 2)))
                mid = (ts[best_run[0]] + ts[best_run[1] - 1]) / 2
                mid = min(max(mid, lo + width / 2 + 0.05), hi - width / 2 - 0.05)
                pos = round(mid, 3)
                doors.append(dict(id=f"d{len(doors)+1:02}", between=[a["id"], b["id"]], wall=axis,
                                  at=[line, pos] if axis == "x" else [pos, line], width=width, state="removed",
                                  opening_m=round(length, 2)))
    # Entrance: the generator's rule (corridor, else living room; an exterior side; position drawn at least 0.65 m from ends).
    entry_room = next((r for r in rooms if r["kind"] == "corridor" and r["source_label"] != "filler"),
                      next((r for r in rooms if r["kind"] == "living"), None))
    if entry_room is None:
        raise Reject("no corridor or living room for the entrance")
    r = entry_room["rect"]
    opts = [(ax, line, lo, hi, n) for ax, line, lo, hi, n in
            [("x", r[0], r[1], r[3], [1, 0]), ("x", r[2], r[1], r[3], [-1, 0]), ("z", r[1], r[0], r[2], [0, 1]), ("z", r[3], r[0], r[2], [0, -1])]
            if line in ([fp[0], fp[2]] if ax == "x" else [fp[1], fp[3]]) and hi - lo >= 1.3]
    if not opts:
        raise Reject("entrance room has no exterior wall")
    axis, line, lo, hi, inward = rng.choice(opts)
    pos = round(rng.uniform(lo + 0.65, hi - 0.65), 3)
    entry = dict(id=f"d{len(doors)+1:02}", between=["outside", entry_room["id"]], wall=axis,
                 at=[line, pos] if axis == "x" else [pos, line], width=1.0, state="removed")
    doors.append(entry)
    return rooms, doors, entry, inward, fp, dict(max_snap_m=round(shift, 4), filler_m2=round(filler * CELL * CELL, 2),
                                                  doors_clamped=sum(1 for d in doors if d.get("opening_m", 1) > 1.0))


def build(hid, h, seed):
    """Mirror of random_layout.generate around a converted geometry: furniture, pads, people, fire, validate."""
    rng = random.Random(seed)
    rooms, doors, entry, inward, fp, stats = convert(h, rng)
    base = json.loads((CONFIG / "fire.json").read_text())
    catalog = json.loads(rl.CATALOG.read_text())
    fire = copy.deepcopy(base)
    for kind, item in catalog["kinds"].items():
        fire["furniture"][kind]["size_m"] = item["conservative_size_m"]
        fire["furniture"][kind]["size_m_reason"] = "Conservative envelope of props.json mesh bounds and existing fuel estimate; tools/random_layout_assets.json records asset hashes."
    plan = dict(version=1, units="m", origin="south-west corner, Unity world (0,0,0)", footprint=fp,
                wall_height=2.8, exterior_wall_thickness=.25, interior_wall_thickness=.15, door_height=2.1,
                rooms=rooms, doors=doors, landmarks=[], fires=[], emergency_lights=[],
                generator=dict(source="HouseExpo (https://github.com/TeaganLi/HouseExpo), built from SUNCG", house_id=hid, seed=seed,
                               converter="tools/bench/real_layouts.py", size_basis=rl.SIZE_BASIS, kind_map_reason=KIND_REASON),
                clear_path_rule="0.9 m clear door-to-centre paths; 1.2 m door keepout plus furniture radius.")
    try:
        rl.place_furniture(plan, fire, rng)
    except ValueError as e:
        raise Reject(f"furniture: {str(e).split(' in ')[0]}")
    yaw = math.degrees(math.atan2(inward[0], inward[1])) % 360
    robots = [dict(id=rid, color=["#F2A900", "#2FA3E6"][i], spawn=dict(at=[round(entry["at"][k] - inward[k] * (1.2 + 1.2 * i), 4) for k in range(2)], yaw=yaw),
                   control="autonomous", cruise_speed=.5, doorway_speed=.3, route=[], look_around_at=[], torch="wide", launch_pad=[0, -1.2 * i, 0])
              for i, rid in enumerate(["robot_A", "robot_B"])]
    name = f"hx_{hid[:10]}"
    sc = dict(id=name, description="Real HouseExpo layout. Autonomous robots get sensors and local launch-pad poses only.",
              seed=seed, duration_s=480, time_limit=False, floorplan_file=f"floorplan_{name}.json", search_area=fp[:],
              dark_rooms=[], sim_rate_hz=15, robots=robots, survivor_placement="random", survivors=[],
              smoke=dict(model="fluid", global_haze=.15, rooms={"default": .15}, ramp_per_minute=0),
              fire=dict(spread=True, head_start_s=0, config_file=f"fire_{name}.json"), obstacles=[], warm_objects=[],
              radio=dict(range_m=12, max_walls=1, broadcast_period_s=2, budget_bytes_per_s=2048),
              launch=dict(door=entry["id"], pad_spacing_m=1.2, placement_sigma_m=0, placement_sigma_deg=0,
                          reason="Two surveyed pads outside the only entrance, as tools/random_layout.generate."))
    try:
        sc["survivors"] = rl.people(plan, fire, sc)
    except ValueError:
        raise Reject("people: fair placement exhausted")
    combustible = [r for r in rooms if r["kind"] in ("living", "kitchen", "bedroom", "study")]
    if not combustible:
        raise Reject("no combustible room")
    room = rng.choice(combustible)
    fuels = [l for l in plan["landmarks"] if l["room"] == room["id"] and fire["materials"][fire["furniture"][l["kind"]]["material"]]["fuel_kg_m2"] > 0]
    if not fuels:
        raise Reject("fire: no combustible furniture in the drawn room")
    origin = rng.choice(fuels)["at"][:]
    fire["ignitions"] = [dict(room=room["id"], at=origin, time_s=0, radius_m=.3, at_reason="Seeded combustible furniture location, drawn as tools/random_layout.generate.")]
    fire["model_note"] = "Real HouseExpo layout; one seeded combustible origin. Solver and material assumptions inherited from config/fire.json."
    plan["fires"] = [dict(id="f01", room=room["id"], at=origin, intensity=1)]
    plan["emergency_lights"] = [dict(at=rl.centre(r["rect"]), note=r["id"]) for r in rooms]
    if len({r["id"] for r in rooms}) != len(rooms):
        raise Reject("validate: duplicate room ids")
    try:
        rl.validate(plan, fire, sc)
    except ValueError as e:
        errs = [x for x in str(e).split("; ") if x != VALIDATE_IGNORED]
        if errs:
            raise Reject("validate: " + errs[0].split(": ")[-1])
    edges = list(plan["footprint"]) + [v for r in rooms for v in r["rect"]]
    if not all(aligned(v, fire["cell_size_m"]) for v in edges):
        raise Reject("internal: edge off the fire grid")
    return name, plan, fire, sc, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--duration", type=float, default=600)
    ap.add_argument("--render", type=int, default=6, help="draw this many converted plans to <out>/plans/")
    args = ap.parse_args()
    out = (ROOT / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
    if not out.is_relative_to(ROOT):
        raise SystemExit("output folder must be inside the repo")
    for sub in ("houses", "worlds", "requests", "runs", "plans"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    files = sorted(args.src.glob("*.json"))
    random.Random(args.seed).shuffle(files)
    rel = lambda p: str(Path(p).resolve().relative_to(ROOT))
    reasons, tried, kept, manifest, house_stats = Counter(), 0, [], [], []
    for f in files:
        if len(kept) >= args.count:
            break
        h = json.loads(f.read_text())
        n = len(unique_rooms(h))
        if not ROOMS_MIN <= n <= ROOMS_MAX:
            continue
        tried += 1
        try:
            name, plan, fire, sc, st = build(f.stem, h, args.seed + tried)
        except Reject as e:
            reasons[str(e)] += 1
            continue
        kept.append(name)
        hd = out / "houses"
        for kind, val in (("floorplan", plan), ("fire", fire)):
            (hd / f"{kind}_{name}.json").write_text(json.dumps(val, indent=1) + "\n")
        sc["floorplan_file"] = os.path.relpath(hd / f"floorplan_{name}.json", CONFIG)
        sc["fire"]["config_file"] = os.path.relpath(hd / f"fire_{name}.json", CONFIG)
        spath = hd / f"scenario_{name}.json"
        spath.write_text(json.dumps(sc, indent=1) + "\n")
        house_stats.append(dict(house=name, source=f.name, rooms=len(plan["rooms"]), doors=len(plan["doors"]),
                                footprint=plan["footprint"], **st))
        if len(kept) <= args.render:
            rl.render(plan, fire, sc, out / "plans" / f"{name}.png")
        world = compile_world(spath.resolve())
        world["bench"] = {"house": name, "seed_index": 0, "people_seed": sc["seed"], "fire_floorplan": world["fire"]["floorplan"],
                          "source": f"HouseExpo {f.stem}"}
        for variant in VARIANTS:
            w = copy.deepcopy(world)
            if variant == "one_robot":
                w["robots"] = [r for r in w["robots"] if r["id"] != "robot_B"]
            w["bench"] = dict(world["bench"], variant=variant)
            run_id = f"{name}_s0_{variant}"
            wpath = out / "worlds" / f"{run_id}.json"
            wpath.write_text(json.dumps(w))
            run = {"world": rel(wpath), "out": rel(out / "runs" / run_id), "duration_s": args.duration, "time_limit": True,
                   "fire": True, "robot_collision": True, "capsule_walls": True, "settings": dict(COMMON_SETTINGS)}
            (out / "requests" / f"{run_id}.json").write_text(json.dumps({"runs": [run]}, indent=2) + "\n")
            manifest.append({"run_id": run_id, "house": name, "seed_index": 0, "people_seed": sc["seed"], "variant": variant,
                             "world": run["world"], "out": run["out"], "fire_config": sc["fire"]["config_file"],
                             "fire_origin": fire["ignitions"][0]["at"], "people": [p["at"] for p in w["people"]]})
    (out / "manifest.json").write_text(json.dumps({"variants": VARIANTS, "baseline_settings": {}, "common_settings": COMMON_SETTINGS,
                                                   "source": "HouseExpo", "runs": manifest}, indent=1) + "\n")
    stats = dict(selection_seed=args.seed, tried=tried, kept=len(kept),
                 rejected=dict(reasons.most_common()), max_snap_m=max((s["max_snap_m"] for s in house_stats), default=None),
                 houses=house_stats)
    (out / "real_layouts_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(f"tried {tried}, kept {len(kept)}, requests {len(manifest)} -> {rel(out)}")
    for k, v in reasons.most_common():
        print(f"  rejected {v:5d}  {k}")
    print(f"max snap shift {stats['max_snap_m']} m")


if __name__ == "__main__":
    main()
