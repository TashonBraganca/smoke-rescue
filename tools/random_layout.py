#!/usr/bin/env python3
"""Seeded WORLD-only houses. Robots never import this tool or its outputs.

Run: python tools/random_layout.py 5001 --render-dir /tmp/plans
Only the three generated config basenames are written. Existing configs stay intact.
Size references and assumptions are recorded in SIZE_BASIS and in each plan.
"""
from collections import deque
import argparse
import copy
import json
import math
from pathlib import Path
import random
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_building import inside, overlap, footprint, boundary_side, room_paths, window_spans

ROOT = Path(__file__).resolve().parents[1]
SIZE_BASIS = {
    "source": "https://www.iccsafe.org/wp-content/uploads/Session-41-and-67-2021-IRC-Plan-Review.pdf",
    "habitable": "2021 IRC R304.1/R304.2: 6.5 m2 and 2.134 m minimum clear dimension; kitchen exception not used.",
    "hall": "2021 IRC R311.6: 0.914 m minimum hallway width. Generated corridors are 1.65-2.0 m before wall deductions.",
    "simulation": "SIMULATION_SPEC 3.4 and floorplan.json clear_path_rule: 0.9 m door-to-centre path, 1.2 m door keepout.",
    "assumptions": "Bathroom/laundry/closet minimum clear dimension 1.5 m and area 3 m2 are generator choices, not code requirements. Door gaps 0.9-1.0 m, entrance 1.0 m, walls 0.25/0.15 m and height 2.8 m follow this simulation. This is not a building-code compliance check.",
}

# Bounds enclose both fire.json fuel estimates and props.json models. They are
# refreshed from glTF POSITION accessors by random_layout_asset_bounds.py.
CATALOG = Path(__file__).with_name("random_layout_assets.json")
FURNITURE = {
    "living": ["apartment_modern_sofa", "console_table_with_vase"],
    "kitchen": ["apartment_counter", "stove_and_hood", "tall_steel_fridge"],
    "bedroom": ["blue_double_bed", "tall_wardrobe"],
    "bathroom": ["clawfoot_bathtub", "sink_with_mirror"],
    "study": ["wooden_desk_with_lamp", "antique_globe"],
    "closet": ["tall_wardrobe", "supply_boxes"],
    "laundry": ["utility_washer", "supply_boxes"],
    "corridor": [],
}


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


class PlacementRandom:
    """System.Random(int), the seeded compatibility algorithm used by Unity.

    SplitMix64 is Scenario.MixSeed from 9f74c16. The 56-entry subtractive
    generator below matches .NET's seeded Random, not Python's Mersenne Twister.
    """
    def __init__(self, seed):
        mask = (1 << 64) - 1
        z = (seed + 0x9E3779B97F4A7C15) & mask
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        seed = (z ^ (z >> 31)) & 0x7FFFFFFF
        self.a = [0] * 56
        mj = 161803398 - seed
        self.a[55] = mj
        mk = 1
        for i in range(1, 55):
            ii = (21 * i) % 55
            self.a[ii] = mk
            mk = mj - mk
            if mk < 0:
                mk += 2147483647
            mj = self.a[ii]
        for _ in range(4):
            for i in range(1, 56):
                self.a[i] -= self.a[1 + (i + 30) % 55]
                if self.a[i] < 0:
                    self.a[i] += 2147483647
        self.i, self.j = 0, 21

    def draw(self):
        self.i = self.i + 1 if self.i < 55 else 1
        self.j = self.j + 1 if self.j < 55 else 1
        n = self.a[self.i] - self.a[self.j]
        if n == 2147483647:
            n -= 1
        if n < 0:
            n += 2147483647
        self.a[self.i] = n
        return n / 2147483647


def centre(rect):
    return [(rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2]


def distance_segment(p, a, b):
    dx, dz = b[0] - a[0], b[1] - a[1]
    t = max(0, min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dz) / max(dx * dx + dz * dz, 1e-12)))
    return math.dist(p, [a[0] + t * dx, a[1] + t * dz])


def bands(length, n, rng, minimum=2.7):
    weights = [rng.uniform(0.5, 1.5) for _ in range(n)]
    free = length - minimum * n
    cuts = [0]
    for w in weights[:-1]:
        cuts.append(round(cuts[-1] + minimum + free * w / sum(weights), 3))
    return cuts + [length]


def geometry(rng):
    w, h = round(rng.uniform(12, 24), 3), round(rng.uniform(10, 18), 3)
    hall = rng.random() < 0.75
    raw = []
    if hall:
        # A long central hall, uneven room bays on both sides. The longer axis
        # can be north/south; the transpose changes the entrance axis too.
        transpose = h >= 12 and w <= 20 and rng.random() < 0.5
        length, depth = (h, w) if transpose else (w, h)
        low = rng.randint(2, min(4, int(length / 3.8)))
        high = rng.randint(1, min(4, int((length-3.2) / 2.8)))
        hall_w = round(rng.uniform(1.65, 2.0), 3)
        z0 = round((depth - hall_w) * rng.uniform(0.43, 0.57), 3)
        raw.append(("corridor", [0, z0, length, round(z0 + hall_w, 3)]))
        cuts = bands(length, low, rng, 3.5)
        for i,(a,b) in enumerate(zip(cuts,cuts[1:])):
            raw.append((["living","kitchen","study","bedroom"][i], [a,0,b,z0]))
        # Wet rooms share a narrow service bay rather than occupying an entire
        # bedroom-sized strip. This keeps bathrooms and laundry near 6-15 m2.
        service_w = round(rng.uniform(2.8,3.3),3)
        north = round(z0+hall_w,3)
        cuts = bands(length-service_w, high, rng, 2.8)
        for i,(a,b) in enumerate(zip(cuts,cuts[1:])):
            raw.append(("study" if i==2 else "bedroom",[a,north,b,depth]))
        services = ["bathroom","laundry"]
        if depth-north >= 6.3 and 1+low+high+3 <= 12:
            services.append("closet")
        heights = [north + (depth-north)*i/len(services) for i in range(len(services)+1)]
        for kind,a,b in zip(services,heights,heights[1:]):
            raw.append((kind,[round(length-service_w,3),round(a,3),length,round(b,3)]))
        if transpose:
            raw = [(k, [r[1], r[0], r[3], r[2]]) for k, r in raw]
    else:
        # Living and kitchen are the circulation spaces. Rear bays can have an
        # ensuite, closet or utility room at their outer end, never a corridor.
        front = round(rng.uniform(3.8, min(5.8, h - 6)), 3)
        split = round(w * rng.uniform(0.5, 0.65), 3)
        raw = [("living", [0, 0, split, front]), ("kitchen", [split, 0, w, front])]
        n = rng.randint(3, min(5, int(w / 2.8)))
        cuts = bands(w, n, rng, 2.8)
        extra = rng.randint(max(2, 7 - 2 - n), min(n, 12 - 2 - n))
        divide = set(rng.sample(range(n), extra))
        service = ["bathroom", "laundry", "closet", "bathroom", "closet"]
        for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
            kind = "bedroom" if i < 2 or i > 2 else "study"
            if i in divide:
                cut = round(front + (h - front) * rng.uniform(0.56, 0.63), 3)
                raw.append((kind, [a, front, b, cut]))
                raw.append((service.pop(0), [a, cut, b, h]))
            else:
                raw.append((kind, [a, front, b, h]))
    # Reflection changes which side the room types and utility dead ends occupy.
    if rng.random() < 0.5:
        raw = [(k, [round(w-r[2], 3), r[1], round(w-r[0], 3), r[3]]) for k, r in raw]
    if rng.random() < 0.5:
        raw = [(k, [r[0], round(h-r[3], 3), r[2], round(h-r[1], 3)]) for k, r in raw]
    rng.shuffle(raw)
    rooms = []
    for i, (kind, rect) in enumerate(raw, 1):
        wet = kind in ("bathroom", "laundry")
        floor = "ceramic_tiles_checker" if wet else rng.choice(["wood_oak_planks", "wood_herringbone", "carpet_beige", "wood_dark_walnut"])
        walls = "ceramic_tiles_large_white" if wet else rng.choice(["wallpaper_deep_green", "paint_warm_grey_textured", "wallpaper_blue_floral", "wallpaper_yellow_stripes"])
        rooms.append(dict(id=f"r{i:02}", name=kind.capitalize(), kind=kind, rect=rect, floor=floor, walls=walls, ceiling="plaster_white"))
    return [0, 0, w, h], rooms


def shared_edge(a, b):
    for axis, cross, along in [("x", 0, 1), ("z", 1, 0)]:
        lo, hi = max(a[along], b[along]), min(a[along+2], b[along+2])
        if hi - lo < 1.6:
            continue
        if abs(a[cross+2] - b[cross]) < 1e-6:
            return axis, a[cross+2], lo, hi
        if abs(b[cross+2] - a[cross]) < 1e-6:
            return axis, b[cross+2], lo, hi
    return None


def make_doors(rooms, fp, rng):
    edges = []
    for i, a in enumerate(rooms):
        for b in rooms[i+1:]:
            edge = shared_edge(a["rect"], b["rect"])
            if edge:
                # Corridors and living/kitchen get first choice. Bathrooms and
                # closets stay dead ends where the adjacency allows it.
                cost = rng.random()
                if a["kind"] in ("living", "kitchen") or b["kind"] in ("living", "kitchen"):
                    cost -= 3
                if a["kind"] == "corridor" or b["kind"] == "corridor":
                    cost -= 6
                if a["kind"] in ("bathroom", "closet", "laundry") or b["kind"] in ("bathroom", "closet", "laundry"):
                    cost += 1
                edges.append((cost, a["id"], b["id"], edge))
    groups = {r["id"]: r["id"] for r in rooms}
    def root(k):
        while groups[k] != k:
            k = groups[k]
        return k
    doors = []
    for _, a, b, (axis, line, lo, hi) in sorted(edges):
        if root(a) == root(b):
            continue
        groups[root(a)] = root(b)
        pos = round(rng.uniform(lo + 0.8, hi - 0.8), 3)
        doors.append(dict(id=f"d{len(doors)+1:02}", between=[a, b], wall=axis,
                          at=[line, pos] if axis == "x" else [pos, line],
                          width=rng.choice([0.9, 0.95, 1.0]), state="removed"))
    # A single exterior entrance into the corridor or living room.
    entry_room = next((r for r in rooms if r["kind"] == "corridor"), next(r for r in rooms if r["kind"] == "living"))
    r = entry_room["rect"]
    options = []
    for axis, line, lo, hi, normal in [("x", r[0], r[1], r[3], [1, 0]), ("x", r[2], r[1], r[3], [-1, 0]),
                                       ("z", r[1], r[0], r[2], [0, 1]), ("z", r[3], r[0], r[2], [0, -1])]:
        if line in ([fp[0], fp[2]] if axis == "x" else [fp[1], fp[3]]):
            options.append((axis, line, lo, hi, normal))
    axis, line, lo, hi, inward = rng.choice(options)
    pos = round(rng.uniform(lo + 0.65, hi - 0.65), 3)
    entry = dict(id=f"d{len(doors)+1:02}", between=["outside", entry_room["id"]], wall=axis,
                 at=[line, pos] if axis == "x" else [pos, line], width=1.0, state="removed")
    doors.append(entry)
    return doors, entry, inward


def furniture_clear(lm, room, plan, fire, previous):
    box = footprint(lm, fire)
    # Exterior walls sit entirely inside the footprint. Use 0.30 m at all
    # walls so the same test also covers the 0.15 m interior wall and finishes.
    if not inside([box[0], box[1]], room["rect"], 0.30) or not inside([box[2], box[3]], room["rect"], 0.30):
        return False
    if any(overlap([box[0]-.10, box[1]-.10, box[2]+.10, box[3]+.10], footprint(o, fire)) for o in previous):
        return False
    radius = math.hypot(box[2]-box[0], box[3]-box[1]) / 2
    doors = [d for d in plan["doors"] if room["id"] in d["between"]]
    return all(math.dist(lm["at"], d["at"]) >= 1.2 + radius and
               distance_segment(lm["at"], d["at"], centre(room["rect"])) >= 0.45 + radius for d in doors)


def place_furniture(plan, fire, rng):
    for room in plan["rooms"]:
        a, b, c, d = room["rect"]
        kinds = FURNITURE[room["kind"]]
        for kind in kinds:
            candidates = []
            # Same four walls and 0.4 m scan used by FurnitureBuilder.PlaceFiller,
            # with random wall and starting order rather than south-first bias.
            for yaw, axis in [(0, "south"), (270, "east"), (180, "north"), (90, "west")]:
                w, depth = fire["furniture"][kind]["size_m"]
                for step in range(math.ceil(max(c-a, d-b) / 0.4)):
                    along = 0.6 + w / 2 + step * 0.4
                    if axis == "south": p = [a+along, b+.32+depth/2]
                    elif axis == "east": p = [c-.32-depth/2, b+along]
                    elif axis == "north": p = [c-along, d-.32-depth/2]
                    else: p = [a+.32+depth/2, d-along]
                    candidates.append(dict(room=room["id"], kind=kind, at=[round(x, 4) for x in p], yaw=yaw))
            rng.shuffle(candidates)
            previous = [l for l in plan["landmarks"] if l["room"] == room["id"]]
            lm = next((lm for lm in candidates if furniture_clear(lm, room, plan, fire, previous)), None)
            if lm is not None:
                plan["landmarks"].append(lm)
        if not any(l["room"] == room["id"] for l in plan["landmarks"]):
            # Small service rooms (about 3 x 2.3 m) leave no spot clear of the 1.2 m door keepout for a tub or washer;
            # requiring one rejected every hall layout over 50 seeds. They may stay empty. Habitable rooms carry the
            # fire's fuel and the robots' visual landmarks, so they still need furniture.
            if room["kind"] in ("living", "kitchen", "bedroom", "study"):
                raise ValueError(f"No usable furniture position in {room['id']}")
            plan.setdefault("generator_notes", []).append(f"{room['id']} ({room['kind']}) has no furniture: none fits clear of its doors")


def people(plan, fire, sc):
    """Preview of the existing fair placement, with a conservative .9 m body box.

    Unity still executes Scenario.PlaceSurvivorsRandomly with scene colliders.
    These fields are safe fallback positions and a reproducible plan preview.
    """
    rng = PlacementRandom(sc["seed"])
    fp = plan["footprint"]
    avoid = [(l["at"], .9) for l in plan["landmarks"]] + [(d["at"], 1.2) for d in plan["doors"]] + [(r["spawn"]["at"], 1.2) for r in sc["robots"]]
    boxes = [footprint(l, fire) for l in plan["landmarks"]]
    result = []
    for i in range(5):
        for _ in range(20000):
            p = [f32(f32(fp[0]) + f32(f32(rng.draw()) * f32(fp[2]-fp[0]))),
                 f32(f32(fp[1]) + f32(f32(rng.draw()) * f32(fp[3]-fp[1])))]
            room = next((r for r in plan["rooms"] if inside(p, r["rect"], .6)), None)
            if room is None or any(math.dist(p, q) < radius for q, radius in avoid) or any(math.dist(p, s["at"]) < 3 for s in result):
                continue
            if any(inside(p, [a-.9, b-.9, c+.9, d+.9]) for a,b,c,d in boxes):
                continue
            result.append(dict(id=f"s{i+1}", room=room["id"], at=p, yaw=f32(rng.draw()*360), pose="lying_on_side"))
            break
        else:
            raise ValueError("Fair placement exhausted 20000 attempts")
    return result


def generate(seed):
    if not isinstance(seed, int) or not -(2**31) <= seed < 2**31:
        raise ValueError("Seed must fit a signed 32-bit Unity integer")
    rng = random.Random(seed)
    base = json.loads((ROOT / "config/fire.json").read_text())
    catalog = json.loads(CATALOG.read_text())
    for attempt in range(100):
        fire = copy.deepcopy(base)
        for kind, item in catalog["kinds"].items():
            fire["furniture"][kind]["size_m"] = item["conservative_size_m"]
            fire["furniture"][kind]["size_m_reason"] = "Conservative envelope of props.json mesh bounds and existing fuel estimate; tools/random_layout_assets.json records asset hashes."
        fp, rooms = geometry(rng)
        doors, entry, inward = make_doors(rooms, fp, rng)
        plan = dict(version=1, units="m", origin="south-west corner, Unity world (0,0,0)", footprint=fp,
                    wall_height=2.8, exterior_wall_thickness=.25, interior_wall_thickness=.15, door_height=2.1,
                    rooms=rooms, doors=doors, landmarks=[], fires=[], emergency_lights=[],
                    generator=dict(seed=seed, attempt=attempt, size_basis=SIZE_BASIS),
                    clear_path_rule="0.9 m clear door-to-centre paths; 1.2 m door keepout plus furniture radius.")
        try:
            place_furniture(plan, fire, rng)
            yaw = math.degrees(math.atan2(inward[0], inward[1])) % 360
            robots = []
            for i, rid in enumerate(["robot_A", "robot_B"]):
                # Pad coordinates are local to the launch line, never house x/z.
                p = [round(entry["at"][k] - inward[k] * (1.2 + 1.2*i), 4) for k in range(2)]
                robots.append(dict(id=rid, color=["#F2A900", "#2FA3E6"][i], spawn=dict(at=p, yaw=yaw),
                                   control="autonomous", cruise_speed=.5, doorway_speed=.3, route=[], look_around_at=[],
                                   torch="wide", launch_pad=[0, -1.2*i, 0]))
            sc = dict(id=f"random_{seed}", description="Unseen seeded world. Autonomous robots get sensors and local launch-pad poses only.",
                      seed=seed, duration_s=480, time_limit=False, floorplan_file=f"floorplan_random_{seed}.json", search_area=fp[:],
                      dark_rooms=[], sim_rate_hz=15, robots=robots, survivor_placement="random", survivors=[],
                      smoke=dict(model="fluid", global_haze=.15, rooms={"default": .15}, ramp_per_minute=0),
                      fire=dict(spread=True, head_start_s=0, config_file=f"fire_random_{seed}.json"),
                      obstacles=[], warm_objects=[], radio=dict(range_m=12, max_walls=1, broadcast_period_s=2, budget_bytes_per_s=2048),
                      launch=dict(door=entry["id"], pad_spacing_m=1.2, placement_sigma_m=0, placement_sigma_deg=0,
                                  reason="Two surveyed pads outside the only entrance. Exact placement is an explicit simulation assumption; no house coordinates enter the brain."))
            sc["survivors"] = people(plan, fire, sc)
            combustible = [r for r in rooms if r["kind"] in ("living", "kitchen", "bedroom", "study")]
            room = rng.choice(combustible)
            fuels = [l for l in plan["landmarks"] if l["room"] == room["id"] and fire["materials"][fire["furniture"][l["kind"]]["material"]]["fuel_kg_m2"] > 0]
            if not fuels:
                raise ValueError("No combustible furniture in fire room")
            origin = rng.choice(fuels)["at"][:]
            fire["ignitions"] = [dict(room=room["id"], at=origin, time_s=0, radius_m=.3,
                                      at_reason="Seeded combustible furniture location in this generated house.")]
            fire["model_note"] = "Generated house; one seeded combustible origin. Solver and material assumptions inherited from config/fire.json. Not a validated fire engineering model."
            plan["fires"] = [dict(id="f01", room=room["id"], at=origin, intensity=1)]
            plan["emergency_lights"] = [dict(at=centre(r["rect"]), note=r["id"]) for r in rooms]
            validate(plan, fire, sc)
            return plan, fire, sc
        except ValueError:
            continue
    raise ValueError(f"Seed {seed}: no valid furnished house after 100 attempts")


def validate(plan, fire, sc):
    errors = []
    rooms = {r["id"]: r for r in plan["rooms"]}
    fp = plan["footprint"]
    if len(rooms) != len(plan["rooms"]) or not 7 <= len(rooms) <= 12:
        errors.append("Room count or duplicate ids")
    area = 0
    for i, r in enumerate(plan["rooms"]):
        rect = r["rect"]
        w, h = rect[2]-rect[0], rect[3]-rect[1]
        if min(w,h) <= 0 or not inside(rect[:2], fp) or not inside(rect[2:], fp):
            errors.append(f"{r['id']} outside footprint or inverted")
        area += w*h
        if any(overlap(rect, other["rect"]) for other in plan["rooms"][i+1:]):
            errors.append(f"{r['id']} overlaps another room")
        clear_w, clear_h = w-.4, h-.4  # conservative two exterior/interior deductions
        if r["kind"] == "corridor":
            if min(clear_w,clear_h) < .914: errors.append("Corridor too narrow")
        elif r["kind"] in ("bathroom", "laundry", "closet"):
            if min(clear_w,clear_h) < 1.5 or clear_w*clear_h < 3: errors.append("Utility room too small")
        elif min(clear_w,clear_h) < 2.134 or clear_w*clear_h < 6.5:
            errors.append("Habitable room too small")
    if abs(area - (fp[2]-fp[0])*(fp[3]-fp[1])) > .002:
        errors.append("Rooms leave a hole in the footprint")
    # A tiled rectangle plus complete room edges is exactly HouseBuilder's wall
    # construction. Windows start at .85 m; they are not floor-level exits.
    graph = {rid: set() for rid in ["outside", *rooms]}
    for d in plan["doors"]:
        if not .8 <= d["width"] <= 1 or d["state"] not in ("removed", "open"):
            errors.append(f"{d['id']} invalid width/state")
        if len(set(d["between"])) != 2 or any(r not in graph for r in d["between"]):
            errors.append("Unknown or duplicate door endpoints"); continue
        for rid in d["between"]:
            if boundary_side(d, fp if rid == "outside" else rooms[rid]["rect"]) is None:
                errors.append(f"{d['id']} cuts no boundary of {rid}")
        a,b = d["between"]
        if a != "outside" and b != "outside":
            side_a, side_b = boundary_side(d, rooms[a]["rect"]), boundary_side(d, rooms[b]["rect"])
            if (side_a,side_b) not in [("east","west"),("west","east"),("south","north"),("north","south")]:
                errors.append("Door endpoints are not on opposite sides")
        graph[a].add(b); graph[b].add(a)
    exterior = [d for d in plan["doors"] if "outside" in d["between"]]
    if len(exterior) != 1: errors.append("Expected exactly one entrance")
    seen, queue = {"outside"}, deque(["outside"])
    while queue:
        for k in graph[queue.popleft()] - seen:
            seen.add(k); queue.append(k)
    if len(seen) != len(graph): errors.append("Unreachable room")
    for r in plan["rooms"]:
        lm = [l for l in plan["landmarks"] if l["room"] == r["id"]]
        for i, l in enumerate(lm):
            if not furniture_clear(l, r, plan, fire, lm[:i]): errors.append(f"{r['id']} furniture blocks clear path or wall")
        # Reuse the existing building validator's independent 0.1 m flood fill.
        errors += [f"{r['id']}: {e}" for e in room_paths(r, [d for d in plan["doors"] if r["id"] in d["between"]], lm, fire)]
    for i, s in enumerate(sc["survivors"]):
        if s["room"] not in rooms or not inside(s["at"], rooms[s["room"]]["rect"], .6): errors.append("Person off safe floor")
        if any(inside(s["at"], [a-.9,b-.9,c+.9,d+.9]) for a,b,c,d in [footprint(l, fire) for l in plan["landmarks"]]): errors.append("Person intersects furniture body envelope")
        if any(math.dist(s["at"], t["at"]) < 3-1e-5 for t in sc["survivors"][:i]): errors.append("People too close")
        if any(math.dist(s["at"], d["at"]) < 1.2 for d in plan["doors"]): errors.append("Person blocks door")
    for f in fire["ignitions"]:
        if f["room"] not in rooms or not inside(f["at"], rooms.get(f["room"], {"rect": [0,0,0,0]})["rect"], .25):
            errors.append("Fire outside combustible room"); continue
        if not any(l["room"] == f["room"] and inside(f["at"], footprint(l, fire)) and fire["materials"][fire["furniture"][l["kind"]]["material"]]["fuel_kg_m2"] > 0 for l in plan["landmarks"]):
            errors.append("Fire origin has no combustible fuel")
    for r in sc["robots"]:
        if inside(r["spawn"]["at"], fp): errors.append("Robot spawn inside house")
    if errors:
        raise ValueError("; ".join(errors))


def write(seed, config_dir=ROOT / "config", render_dir=None):
    plan, fire, sc = generate(seed)
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    for name, value in [("floorplan",plan),("fire",fire),("scenario",sc)]:
        (config_dir / f"{name}_random_{seed}.json").write_text(json.dumps(value, indent=2) + "\n")
    if render_dir is not None:
        render(plan, fire, sc, Path(render_dir) / f"random_{seed}.png")
    return plan, fire, sc


def render(plan, fire, sc, target):
    from PIL import Image, ImageDraw, ImageFont
    fp = plan["footprint"]
    scale, margin = 48, 110
    image = Image.new("RGB", (round(fp[2]*scale+2*margin), round(fp[3]*scale+2*margin+80)), "#faf9f4")
    draw = ImageDraw.Draw(image)
    font, small = ImageFont.load_default(size=16), ImageFont.load_default(size=12)
    def xy(p): return (margin+p[0]*scale, margin+(fp[3]-p[1])*scale)
    colors = dict(corridor="#eee8d8",living="#f0cfaa",kitchen="#e9d59b",bedroom="#cadbe5",bathroom="#b9ddd8",study="#d8d0e4",closet="#ded7ca",laundry="#cce0c0")
    draw.text((20,18), f"Seed {sc['seed']} | {fp[2]:.2f} x {fp[3]:.2f} m | {len(plan['rooms'])} rooms | north up", fill="#142432", font=font)
    draw.text((20,44), "Furniture = conservative mesh/fuel envelope. People = fair-placement preview; Unity repeats placement with scene colliders.", fill="#142432", font=small)
    for r in plan["rooms"]:
        a,b,c,d = r["rect"]
        draw.rectangle([xy([a,d]),xy([c,b])],fill=colors[r["kind"]],outline="#26323a",width=6)
    for l in plan["landmarks"]:
        a,b,c,d = footprint(l,fire)
        draw.rectangle([xy([a,d]),xy([c,b])],fill="#998573",outline="#574b41",width=2)
    for win in window_spans(plan):
        x,z=win["at"]
        p = [[x-.6,z],[x+.6,z]] if win["wall"]=="z" else [[x,z-.6],[x,z+.6]]
        draw.line([xy(v) for v in p],fill="#479bb0",width=5)
    for d in plan["doors"]:
        x,z=d["at"]; h=d["width"]/2
        p = [[x-h,z],[x+h,z]] if d["wall"]=="z" else [[x,z-h],[x,z+h]]
        draw.line([xy(v) for v in p],fill="#faf9f4",width=9)
        draw.line([xy(v) for v in p],fill="#247447",width=3)
        q=xy(d["at"]);draw.text((q[0]+5,q[1]+4),d["id"],fill="#17472c",font=small)
        if "outside" in d["between"]: draw.ellipse((q[0]-11,q[1]-11,q[0]+11,q[1]+11),outline="#247447",width=3)
    for r in plan["rooms"]:
        p=xy(centre(r["rect"])); a,b,c,d=r["rect"]
        draw.multiline_text((p[0],p[1]-15),f"{r['id']} {r['name']}\n{(c-a)*(d-b):.1f} m2",anchor="mm",align="center",fill="#172632",font=font)
    for s in sc["survivors"]:
        x,y=xy(s["at"]);draw.ellipse((x-7,y-7,x+7,y+7),fill="#ad3a76",outline="white",width=2);draw.text((x+8,y-15),s["id"],fill="#7a1550",font=small)
    for f in fire["ignitions"]:
        x,y=xy(f["at"]);draw.polygon([(x,y-12),(x+10,y+8),(x-10,y+8)],fill="#e64d26");draw.text((x+12,y-8),"fire",fill="#a12910",font=small)
    for r in sc["robots"]:
        x,y=xy(r["spawn"]["at"]);draw.rectangle((x-6,y-8,x+6,y+8),fill=r["color"],outline="#172632");draw.text((x+9,y-6),r["id"][-1],fill="#172632",font=small)
    draw.text((20,image.height-28),"Green gaps = doors; circled gap = only entrance. Blue = windows above 0.85 m. Pink = people. Triangle = ignition.",fill="#172632",font=small)
    target.parent.mkdir(parents=True,exist_ok=True);image.save(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seeds",type=int,nargs="+")
    parser.add_argument("--config-dir",type=Path,default=ROOT/"config")
    parser.add_argument("--render-dir",type=Path)
    args=parser.parse_args()
    for seed in args.seeds:
        plan,fire,sc=write(seed,args.config_dir,args.render_dir)
        print(f"seed {seed}: validated {len(plan['rooms'])} rooms, {len(plan['doors'])} doors, {len(plan['landmarks'])} furniture groups, 5 people; {sc['floorplan_file']}")
