#!/usr/bin/env python3
"""Compile a floor plan and scenario into a plain world file for the headless brain simulator (EvaluatorOnly).

The output is the WORLD the simulator ray-casts against: wall and lintel segments, furniture prisms, people, robot
pads and evaluation rooms, all in the robots' gate frame (robot A's launch pad at the origin, facing +y, x to its right),
which is the frame the robot brains start in on a one-door launch. The brains never read this file; the simulator uses it
to produce their sensor data and to score them.

    python tools/headless_world.py config/scenario_random_5001.json data/runs/headless/world_5001.json
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"


def load_scenario(path: Path) -> dict:
    s = json.loads(path.read_text())
    if s.get("inherits"):
        base = load_scenario(CONFIG / s["inherits"])
        base.update({k: v for k, v in s.items() if k != "inherits"})
        return base
    return s


def furniture_heights() -> dict:
    """Height of each furniture kind from the measured mesh bounds the layout generator keeps; 0.8 m when unknown."""
    path = ROOT / "tools" / "random_layout_assets.json"
    heights = {}
    if path.exists():
        for kind, item in json.loads(path.read_text())["kinds"].items():
            tops = [a["max_m"][1] - a["min_m"][1] for a in item.get("assets", []) if a.get("max_m") and a.get("min_m")]
            if tops:
                heights[kind] = max(tops)
    return heights


# Lowest mounting height (model origin above the floor, config/props.json offset y) at which a landmark hangs clear of the
# robot body and its 0.45 m LiDAR plane: the pictures, clocks and dartboard hang 1.25 to 1.75 m up the wall.
HUNG_MIN_M = 1.0


def hung_kinds() -> set:
    """Landmark kinds whose every model part hangs on a wall at HUNG_MIN_M or more: no floor prism. Seed 934 with one door
    per room (24 Oct): pictures 1.35 to 1.7 m up, modelled as 0.8 m floor prisms, narrowed the hallway beside the rubble and
    the wood pile to 0.54 to 0.64 m and sealed four rooms off in headless only."""
    landmarks = json.loads((CONFIG / "props.json").read_text()).get("landmarks", {})
    out = set()
    for kind, parts in landmarks.items():
        if not isinstance(parts, list) or not parts:
            continue
        ys = [(p.get("offset") or [0, 0, 0])[1] for p in parts if isinstance(p, dict)]
        if ys and all(isinstance(y, (int, float)) and y >= HUNG_MIN_M for y in ys):
            out.add(kind)
    return out


def compile_world(scenario_path: Path, survivors: list | None = None) -> dict:
    sc = load_scenario(scenario_path)
    plan = json.loads((CONFIG / sc.get("floorplan_file", "floorplan.json")).read_text())
    fire = json.loads((CONFIG / sc.get("fire", {}).get("config_file", "fire.json")).read_text())
    wall_h, door_h = float(plan["wall_height"]), float(plan["door_height"])
    states = sc.get("door_states") or {}
    robots = sorted(sc["robots"], key=lambda r: r["id"])
    a = robots[0]["spawn"]
    h = math.radians(a["yaw"])
    fwd, right, origin = (math.sin(h), math.cos(h)), (math.cos(h), -math.sin(h)), a["at"]

    def gate(p):
        d = (p[0] - origin[0], p[1] - origin[1])
        return [round(d[0] * right[0] + d[1] * right[1], 4), round(d[0] * fwd[0] + d[1] * fwd[1], 4)]

    # Walls: every room edge, less its open door gaps; a lintel spans each gap above door height.
    segments, seen = [], set()
    doors = [d for d in plan["doors"] if states.get(d["id"], d.get("state", "open")) in ("open", "removed")]
    for room in plan["rooms"]:
        x0, z0, x1, z1 = room["rect"]
        for p, q in [((x0, z0), (x1, z0)), ((x1, z0), (x1, z1)), ((x1, z1), (x0, z1)), ((x0, z1), (x0, z0))]:
            key = tuple(sorted([tuple(round(v, 3) for v in p), tuple(round(v, 3) for v in q)]))
            if key in seen:
                continue
            seen.add(key)
            horizontal = abs(p[1] - q[1]) < 1e-6
            lo, hi = sorted([p[0], q[0]] if horizontal else [p[1], q[1]])
            line = p[1] if horizontal else p[0]
            cuts = []
            for d in doors:
                along, across = (d["at"][0], d["at"][1]) if d["wall"] == "z" else (d["at"][1], d["at"][0])
                if (d["wall"] == "z") == horizontal and abs(across - line) < 1e-3 and lo - 1e-3 <= along <= hi + 1e-3:
                    cuts.append((along - d["width"] / 2, along + d["width"] / 2))
            start = lo
            for c0, c1 in sorted(cuts):
                pieces = [(start, c0, 0.0, wall_h), (c0, c1, door_h, wall_h)]
                for s0, s1, hmin, hmax in pieces:
                    if s1 - s0 > 1e-3:
                        e0, e1 = ((s0, line), (s1, line)) if horizontal else ((line, s0), (line, s1))
                        segments.append(gate(e0) + gate(e1) + [hmin, hmax])
                start = c1
            if hi - start > 1e-3:
                e0, e1 = ((start, line), (hi, line)) if horizontal else ((line, start), (line, hi))
                segments.append(gate(e0) + gate(e1) + [0.0, wall_h])

    heights = furniture_heights()
    hung = hung_kinds()
    prisms = []
    for lm in plan.get("landmarks", []) + sc.get("obstacles", []):
        kind = lm.get("kind")
        if kind not in fire.get("furniture", {}) or kind in hung:
            continue
        w, d = fire["furniture"][kind]["size_m"]
        yaw = math.radians(lm.get("yaw", 0))
        c, s = math.cos(yaw), math.sin(yaw)
        x, z = lm["at"]
        corners = [(x + c * u + s * v, z - s * u + c * v) for u, v in [(-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2)]]
        prisms.append({"kind": kind, "poly": [gate(p) for p in corners], "height_m": round(heights.get(kind, 0.8), 3)})

    people = survivors if survivors is not None else sc.get("survivors", [])
    return {
        "name": sc.get("id", scenario_path.stem), "ceiling_m": wall_h, "lintel_m": door_h,
        "segments": segments, "prisms": prisms,
        "people": [{"id": p["id"], "at": gate(p["at"]), "pose": p.get("pose", "lying_on_side"), "room": p.get("room")} for p in people],
        "robots": [{"id": r["id"], "at": gate(r["spawn"]["at"]),
                    "heading_deg": round((r["spawn"]["yaw"] - a["yaw"] + 180) % 360 - 180, 3),
                    "pad": r.get("launch_pad", [0, 0, 0])} for r in robots],
        "rooms": [{"id": r["id"], "kind": r.get("kind", r["id"]),
                   "poly": [gate(p) for p in [(r["rect"][0], r["rect"][1]), (r["rect"][2], r["rect"][1]),
                                              (r["rect"][2], r["rect"][3]), (r["rect"][0], r["rect"][3])]]} for r in plan["rooms"]],
        "radio": sc.get("radio", {"range_m": 12, "max_walls": 1, "broadcast_period_s": 2}),
        # For the simulator's fire: house coordinates of the gate frame (robot A's pad and heading) and the scenario's fire.
        "gate": {"origin": origin, "yaw_deg": a["yaw"]},
        "fire": {"floorplan": sc.get("floorplan_file", "floorplan.json"), "config": sc.get("fire", {}).get("config_file", "fire.json"),
                 "seed": sc.get("seed", 0), "head_start_s": sc.get("fire", {}).get("head_start_s", 0.0),
                 "spread": bool(sc.get("fire", {}).get("spread", False)), "obstacles": sc.get("obstacles", [])},
        "source": {"scenario": str(scenario_path.relative_to(ROOT)) if scenario_path.is_relative_to(ROOT) else str(scenario_path)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--survivors", type=Path, help="oracle survivors.json for scenarios whose people Unity places at run time")
    args = ap.parse_args()
    people = None
    if args.survivors:
        raw = json.loads(args.survivors.read_text())
        people = raw.get("survivors", raw) if isinstance(raw, dict) else raw
    world = compile_world(args.scenario.resolve(), people)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(world))
    print(f"{world['name']}: {len(world['segments'])} wall segments, {len(world['prisms'])} furniture, "
          f"{len(world['people'])} people, robots {[(r['id'], r['at'], r['heading_deg']) for r in world['robots']]}")


if __name__ == "__main__":
    main()
