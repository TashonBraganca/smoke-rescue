#!/usr/bin/env python3
"""Write benchmark worlds and one headless request per run (evaluator side only; the brains never read these files).

    brain/.venv/bin/python tools/bench/make_requests.py data/runs/bench_25oct
    brain/.venv/bin/python tools/bench/make_requests.py data/runs/bench_val --houses random_5001,family_ranch --seeds 1 --duration 60

Houses: the 28 seeded houses (config/scenario_random_5001..5028.json) and the two ranches (config/scenario_family_ranch_full.json,
config/scenario_long_ranch_full.json, compiled from the current config, so the newest long ranch). Worlds are compiled with
tools/headless_world.compile_world, the same compiler the earlier headless runs used.

Seeds per house. Seed 0 is the house as configured: its own people and its own fire origin. Seed k > 0 draws new people with
the generator's own placement rule (tools/random_layout.people, PlacementRandom seeded with house_seed * 10 + k) and a new fire
origin the way tools/random_layout.generate picks one (random.Random(house_seed * 10 + k): a combustible room, then one of its
fuel-bearing furniture items). The fire file for k > 0 is written under <out>/fires/ and the world points at it relative to
config/, since the simulator joins config/ with that path. Nothing under config/ is written.

Variants, each changing one thing against full:
  full      : the world as compiled, both robots, radio from the scenario, no brain overrides.
  no_radio  : world radio.range_m = 0. The simulator delivers a message only when distance <= range_m, so nothing is delivered.
  one_robot : robot_B removed from world.robots.
  baseline  : world identical to full; request settings switch off the brain features added since 23 Oct (BASELINE_OFF).
Every request: duration_s (600 unless --duration), time_limit true, fire true, robot_collision true, capsule_walls true,
settings.mission_time_limit_s 480.
"""
import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from headless_world import compile_world, load_scenario  # noqa: E402
import random_layout  # noqa: E402

CONFIG = ROOT / "config"
VARIANTS = ["full", "no_radio", "one_robot", "baseline"]
COMMON_SETTINGS = {"mission_time_limit_s": 480}
# Brain features added since 23 Oct, every one a BrainConfig field set true (or 1.5 m) in config/brain.json.
BASELINE_OFF = {
    "room_memory": False,
    "door_claims": False,
    "room_entered_near_seat": False,
    "room_memory_same_space": False,
    "leave_waits_for_team": False,
    "walk_out_past_fire": False,
    "end_clear_way_out_m": 0.0,
}
COMBUSTIBLE_KINDS = ("living", "kitchen", "bedroom", "study")


def houses():
    out = [(f"random_{s}", CONFIG / f"scenario_random_{s}.json") for s in range(5001, 5029)]
    out += [("family_ranch", CONFIG / "scenario_family_ranch_full.json"), ("long_ranch", CONFIG / "scenario_long_ranch_full.json")]
    for name, path in out:
        if not path.exists():
            raise SystemExit(f"missing scenario for {name}: {path}")
    return out


def aligned(v, cell):
    return abs(v / cell - round(v / cell)) <= 1e-4


def fire_grid_plan(sc, out, name):
    """The seeded houses' room edges (e.g. 18.843 m) do not fall on the fire solver's 0.25 m grid, and FireSim refuses such a
    plan ("Footprint and room edges must align to the fire grid"). For the fire solver only, write a copy of the floor plan
    with the footprint, every room edge and every door's wall line moved to the nearest grid line (at most cell/2, 0.125 m).
    Walls, furniture, people and robots in the world are untouched. Returns the path relative to config/, or None when the
    plan already fits the grid (the ranches)."""
    plan = json.loads((CONFIG / sc["floorplan_file"]).read_text())
    cell = json.loads((CONFIG / sc["fire"]["config_file"]).read_text())["cell_size_m"]
    edges = list(plan["footprint"]) + [v for r in plan["rooms"] for v in r["rect"]]
    if all(aligned(v, cell) for v in edges):
        return None
    snap = lambda v: round(round(v / cell) * cell, 6)
    plan["footprint"] = [snap(v) for v in plan["footprint"]]
    for r in plan["rooms"]:
        r["rect"] = [snap(v) for v in r["rect"]]
    for d in plan["doors"]:
        k = 1 if d["wall"] == "z" else 0   # wall "z": runs along x, wall line is z (at[1]); "x": wall line is x (at[0])
        d["at"][k] = snap(d["at"][k])
    plan["bench_note"] = f"Fire-solver copy of {sc['floorplan_file']}: edges snapped to the {cell} m fire grid (tools/bench/make_requests.py)."
    path = out / "fires" / f"floorplan_{name}_firegrid.json"
    path.write_text(json.dumps(plan, indent=1) + "\n")
    return os.path.relpath(path, CONFIG)


def reseed(sc, k):
    """People and fire origin for seed index k > 0, through the generator's own rules. Returns survivors, fire dict, seed."""
    plan = json.loads((CONFIG / sc["floorplan_file"]).read_text())
    fire = json.loads((CONFIG / sc["fire"]["config_file"]).read_text())
    new_seed = int(sc["seed"]) * 10 + k
    sc2 = copy.deepcopy(sc)
    sc2["seed"] = new_seed
    survivors = random_layout.people(plan, fire, sc2)
    rng = random.Random(new_seed)
    rooms = [r for r in plan["rooms"] if r["kind"] in COMBUSTIBLE_KINDS]
    for _ in range(200):
        room = rng.choice(rooms)
        fuels = [l for l in plan.get("landmarks", []) if l.get("room") == room["id"] and l.get("kind") in fire["furniture"]
                 and fire["materials"][fire["furniture"][l["kind"]]["material"]]["fuel_kg_m2"] > 0]
        if fuels:
            break
    else:
        raise ValueError(f"{sc['id']}: no combustible furniture for seed {new_seed}")
    origin = rng.choice(fuels)["at"][:]
    first = fire["ignitions"][0] if fire.get("ignitions") else {"time_s": 0, "radius_m": 0.3}
    fire["ignitions"] = [dict(room=room["id"], at=origin, time_s=first.get("time_s", 0), radius_m=first.get("radius_m", 0.3),
                              at_reason=f"Benchmark seed {new_seed}: combustible furniture drawn as tools/random_layout.generate does.")]
    return survivors, fire, new_seed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, help="output folder inside the repo, e.g. data/runs/bench_25oct")
    ap.add_argument("--houses", default="", help="comma list of house names (default: all 30)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--duration", type=float, default=600)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    args = ap.parse_args()
    out = (ROOT / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
    if not out.is_relative_to(ROOT):
        raise SystemExit("output folder must be inside the repo: requests use repo-relative paths")
    variants = [v for v in args.variants.split(",") if v]
    bad = set(variants) - set(VARIANTS)
    if bad:
        raise SystemExit(f"unknown variants {bad}")
    wanted = set(filter(None, args.houses.split(",")))
    chosen = [h for h in houses() if not wanted or h[0] in wanted]
    if wanted - {h[0] for h in chosen}:
        raise SystemExit(f"unknown houses {wanted - {h[0] for h in chosen}}")
    for sub in ("worlds", "requests", "fires", "runs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rel = lambda p: str(Path(p).resolve().relative_to(ROOT))
    manifest = []
    for name, scen in chosen:
        sc = load_scenario(scen)
        fire_plan = fire_grid_plan(sc, out, name)
        for k in range(args.seeds):
            if k == 0:
                world = compile_world(scen.resolve())
                fire_cfg, people_seed = world["fire"]["config"], int(sc["seed"])
            else:
                survivors, fire, people_seed = reseed(sc, k)
                world = compile_world(scen.resolve(), survivors)
                fire_path = out / "fires" / f"fire_{name}_s{k}.json"
                fire_path.write_text(json.dumps(fire, indent=2) + "\n")
                fire_cfg = os.path.relpath(fire_path, CONFIG)
                world["fire"]["config"] = fire_cfg
                world["fire"]["seed"] = people_seed
            if fire_plan:
                world["fire"]["floorplan"] = fire_plan
            world["bench"] = {"house": name, "seed_index": k, "people_seed": people_seed, "fire_floorplan": world["fire"]["floorplan"]}
            for variant in variants:
                w = copy.deepcopy(world)
                settings = dict(COMMON_SETTINGS)
                if variant == "no_radio":
                    w["radio"]["range_m"] = 0
                elif variant == "one_robot":
                    w["robots"] = [r for r in w["robots"] if r["id"] != "robot_B"]
                elif variant == "baseline":
                    settings.update(BASELINE_OFF)
                w["bench"] = dict(world["bench"], variant=variant)
                run_id = f"{name}_s{k}_{variant}"
                wpath = out / "worlds" / f"{run_id}.json"
                wpath.write_text(json.dumps(w))
                run = {"world": rel(wpath), "out": rel(out / "runs" / run_id), "duration_s": args.duration, "time_limit": True,
                       "fire": True, "robot_collision": True, "capsule_walls": True, "settings": settings}
                (out / "requests" / f"{run_id}.json").write_text(json.dumps({"runs": [run]}, indent=2) + "\n")
                manifest.append({"run_id": run_id, "house": name, "seed_index": k, "people_seed": people_seed, "variant": variant,
                                 "world": run["world"], "out": run["out"], "fire_config": fire_cfg,
                                 "fire_origin": json.loads((CONFIG / fire_cfg).read_text())["ignitions"][0]["at"],
                                 "people": [p["at"] for p in w["people"]]})
    (out / "manifest.json").write_text(json.dumps({"variants": variants, "baseline_settings": BASELINE_OFF,
                                                   "common_settings": COMMON_SETTINGS, "runs": manifest}, indent=1) + "\n")
    print(f"{len(manifest)} runs: {len(chosen)} houses x {args.seeds} seeds x {len(variants)} variants -> {rel(out)}")


if __name__ == "__main__":
    main()
