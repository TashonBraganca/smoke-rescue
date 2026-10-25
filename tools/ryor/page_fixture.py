#!/usr/bin/env python3
"""Fake "Run your own rescue" data for developing the page (tools/ryor/CONTRACT.txt sections 2 and 3).

NOT a simulation. Robots walk a scripted route through a grid of rectangular rooms so the page has walls, room tints,
moving robots, growing searched areas, fire, people found, a map link and robots walking out to draw. The real data
comes from the headless brain; this file only exists so the page can be built and tested before that data is ready.

    python3 tools/ryor/page_fixture.py OUT_DIR
writes OUT_DIR/gallery.json and OUT_DIR/cache/<setup key>/{status.json,chunk_NNNN.json}.
"""
import json
import math
import random
import sys
from pathlib import Path

MISSION_S = 180.0
DT = 0.5
CELL = 0.1
ORIGIN = -25.0
DIFF = {
    "easy": {"head_start_s": 0, "visibility_m": 6.0, "label": "Fire starts as the robots enter. Light smoke."},
    "medium": {"head_start_s": 90, "visibility_m": 4.0, "label": "Fire has burned for 90 s. Smoke cuts sight to 4 m."},
    "hard": {"head_start_s": 180, "visibility_m": 2.5, "label": "Fire has burned for 3 min. Smoke cuts sight to 2.5 m."},
}
HOUSES = [
    # id, name, source, cols, rows, room w, room d, people, seed
    ("family_ranch", "Family ranch", "ours", 3, 2, 4.2, 3.8, 3, 11),
    ("long_ranch", "Long ranch", "ours", 5, 1, 3.6, 4.4, 3, 12),
    ("onedoor_934", "One-door house 934", "ours", 3, 3, 3.4, 3.2, 4, 13),
    ("hx_035feb0125", "HouseExpo 035feb0125", "HouseExpo", 2, 2, 4.6, 4.0, 2, 14),
    ("hx_0a1b2c3d4e", "HouseExpo 0a1b2c3d4e", "HouseExpo", 4, 2, 3.2, 3.6, 4, 15),
    ("hx_77e0c4a9f1", "HouseExpo 77e0c4a9f1", "HouseExpo", 3, 2, 5.0, 3.4, 3, 16),
]
KINDS = ["living", "kitchen", "bedroom", "bath", "bedroom", "hall", "dining", "bedroom", "study", "laundry"]
GAP = 0.9


def build_house(hid, name, source, cols, rows, rw, rd, people, seed):
    W, D = cols * rw, rows * rd
    # entrance in the middle of the bottom wall of the room that contains x = 0
    ent_col = cols // 2
    x0 = -(ent_col + 0.5) * rw
    y0 = 1.5
    walls = []

    def hwall(y, xa, xb, gaps):
        cuts = sorted(gaps)
        cur = xa
        for g in cuts:
            walls.append([round(cur, 2), round(y, 2), round(g - GAP / 2, 2), round(y, 2)])
            cur = g + GAP / 2
        walls.append([round(cur, 2), round(y, 2), round(xb, 2), round(y, 2)])

    def vwall(x, ya, yb, gaps):
        cur = ya
        for g in sorted(gaps):
            walls.append([round(x, 2), round(cur, 2), round(x, 2), round(g - GAP / 2, 2)])
            cur = g + GAP / 2
        walls.append([round(x, 2), round(cur, 2), round(x, 2), round(yb, 2)])

    hwall(y0, x0, x0 + W, [0.0])
    hwall(y0 + D, x0, x0 + W, [])
    vwall(x0, y0, y0 + D, [])
    vwall(x0 + W, y0, y0 + D, [])
    doors = {}  # (room a, room b) -> gap point
    for r in range(rows):
        for c in range(1, cols):
            x = x0 + c * rw
            gy = y0 + r * rd + rd / 2
            vwall(x, y0 + r * rd, y0 + (r + 1) * rd, [gy])
            doors[((c - 1, r), (c, r))] = (x, gy)
    for r in range(1, rows):
        y = y0 + r * rd
        gaps = []
        for c in range(cols):
            gx = x0 + c * rw + rw / 2
            gaps.append(gx)
            doors[((c, r - 1), (c, r))] = (gx, y)
        hwall(y, x0, x0 + W, gaps)
    rooms = []
    rnd = random.Random(seed)
    for r in range(rows):
        for c in range(cols):
            k = KINDS[(r * cols + c + seed) % len(KINDS)]
            if (c, r) == (ent_col, 0):
                k = "hall"
            xa, ya = x0 + c * rw, y0 + r * rd
            rooms.append({"id": f"r{r}{c}", "kind": k, "cell": (c, r),
                          "poly": [[round(xa, 2), round(ya, 2)], [round(xa + rw, 2), round(ya, 2)],
                                   [round(xa + rw, 2), round(ya + rd, 2)], [round(xa, 2), round(ya + rd, 2)]]})
    cand = [rm for rm in rooms if rm["cell"] != (ent_col, 0)]
    rnd.shuffle(cand)
    ppl = []
    for i, rm in enumerate(cand[:people]):
        (xa, ya), (xb, yb) = rm["poly"][0], rm["poly"][2]
        ppl.append({"id": f"c{i + 1}", "x": round(rnd.uniform(xa + 0.6, xb - 0.6), 2),
                    "y": round(rnd.uniform(ya + 0.6, yb - 0.6), 2), "cell": rm["cell"]})
    fire_room = cand[-1] if len(cand) > people else cand[0]
    (fa, fb), (fc, fd) = fire_room["poly"][0], fire_room["poly"][2]
    fire = (round((fa + fc) / 2 + 0.4, 2), round((fb + fd) / 2 - 0.3, 2))
    house = {
        "id": hid, "name": name, "source": source,
        "source_id": hid[3:] if source == "HouseExpo" else None,
        "licence": "MIT (HouseExpo, Li et al. 2019)" if source == "HouseExpo" else "ours",
        "rooms": len(rooms), "floor_m2": round(W * D, 1), "people": people,
        "bounds": [round(x0, 2), round(y0, 2), round(x0 + W, 2), round(y0 + D, 2)],
        "walls": walls,
        "rooms_poly": [{"id": r["id"], "kind": r["kind"], "poly": r["poly"]} for r in rooms],
        "door": [0.0, y0], "pads": [[0.0, 0.0], [0.0, -1.2], [0.0, -2.4]],
    }
    meta = {"x0": x0, "y0": y0, "rw": rw, "rd": rd, "cols": cols, "rows": rows, "ent": (ent_col, 0),
            "doors": doors, "people": ppl, "fire": fire, "fire_cell": fire_room["cell"], "rooms": rooms}
    return house, meta


def center(meta, cell):
    c, r = cell
    return (meta["x0"] + (c + 0.5) * meta["rw"], meta["y0"] + (r + 0.5) * meta["rd"])


def route(meta, a, b):
    """BFS over the room grid; returns gap points then the target centre."""
    if a == b:
        return []
    prev = {a: None}
    q = [a]
    while q:
        cur = q.pop(0)
        if cur == b:
            break
        c, r = cur
        for n in [(c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)]:
            if 0 <= n[0] < meta["cols"] and 0 <= n[1] < meta["rows"] and n not in prev:
                prev[n] = cur
                q.append(n)
    path = [b]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    path.reverse()
    pts = []
    for u, v in zip(path, path[1:]):
        key = (u, v) if (u, v) in meta["doors"] else (v, u)
        pts.append(meta["doors"][key])
        pts.append(center(meta, v))
    return pts


def cells_in_disc(x, y, rad, bounds):
    out = []
    ix0, ix1 = int((x - rad - ORIGIN) / CELL), int((x + rad - ORIGIN) / CELL)
    iy0, iy1 = int((y - rad - ORIGIN) / CELL), int((y + rad - ORIGIN) / CELL)
    for iy in range(iy0, iy1 + 1):
        cy = ORIGIN + (iy + 0.5) * CELL
        if not bounds[1] <= cy <= bounds[3]:
            continue
        for ix in range(ix0, ix1 + 1):
            cx = ORIGIN + (ix + 0.5) * CELL
            if bounds[0] <= cx <= bounds[2] and (cx - x) ** 2 + (cy - y) ** 2 <= rad * rad:
                out.append(iy * 500 + ix)
    return out


def simulate(house, meta, n, diff):
    d = DIFF[diff]
    rnd = random.Random(hash((house["id"], n, diff)) & 0xffff)
    ids = ["robot_A", "robot_B", "robot_C"][:n]
    see = min(2.4, d["visibility_m"] / 2)
    find_r = {"easy": 2.2, "medium": 1.7, "hard": 1.2}[diff]
    speed = 0.3
    fire_r0 = 0.6 + 0.004 * d["head_start_s"]
    fire_room = meta["fire_cell"]
    rooms = [rm["cell"] for rm in meta["rooms"]]
    ent = meta["ent"]
    ex, ey = center(meta, ent)
    order = sorted(rooms, key=lambda c: abs(c[0] - ent[0]) + abs(c[1] - ent[1]) + rnd.random() * 0.1)
    plans = {}
    for k, rid in enumerate(ids):
        mine = [c for i, c in enumerate(order) if i % n == k] or [ent]
        if n > 1 and k == 0:
            mine = [c for c in mine if c != ent] or mine
        pts = [(0.0, 1.0), (0.0, 2.2), (ex, ey)]
        cur = ent
        for c in mine:
            pts += route(meta, cur, c)
            cx, cy = center(meta, c)
            ox, oy = meta["rw"] * 0.28, meta["rd"] * 0.28
            pts += [(cx - ox, cy - oy), (cx + ox, cy - oy), (cx + ox, cy + oy), (cx - ox, cy + oy), (cx, cy)]
            cur = c
        plans[rid] = {"pts": pts, "cur": cur}
    pad = {rid: tuple(house["pads"][i]) for i, rid in enumerate(ids)}
    st = {rid: {"x": pad[rid][0], "y": pad[rid][1], "yaw": 90.0, "i": 0, "state": "wait", "out": False,
                "home": None, "seen": set()} for rid in ids}
    found = {}
    linked = set()
    fire_marked = set()
    frames = []
    t = 0.0
    while t <= MISSION_S - DT / 2:
        fr = {"t": round(t, 1), "robots": [], "seen": {}, "fire": [], "found": [], "events": []}
        fr_r = fire_r0 + 0.006 * t
        fx, fy = meta["fire"]
        fire = [[fx, fy, round(fr_r, 2)]]
        if fr_r > 1.2:
            fire.append([round(fx - 0.9, 2), round(fy + 0.5, 2), round(fr_r * 0.6, 2)])
        fr["fire"] = fire
        for k, rid in enumerate(ids):
            s = st[rid]
            start = 2.0 + 8.0 * k
            if s["state"] == "wait" and t >= start:
                s["state"] = "explore"
            if s["state"] in ("explore", "return"):
                if s["state"] == "explore" and t >= 125 and not s["home"]:
                    s["state"] = "return"
                    back = route(meta, plans[rid]["cur"], ent) + [(0.0, 2.2), (0.0, 1.0), pad[rid]]
                    s["home"] = back
                    s["i"] = 0
                    fr["events"].append({"robot": rid, "kind": "mission_complete",
                                         "text": f"{rid[-1]} is heading out to be out by 170 s"})
                pts = s["home"] if s["state"] == "return" else plans[rid]["pts"]
                if s["i"] < len(pts):
                    tx, ty = pts[s["i"]]
                    dx, dy = tx - s["x"], ty - s["y"]
                    dist = math.hypot(dx, dy)
                    step = speed * DT
                    if dist <= step:
                        s["x"], s["y"] = tx, ty
                        s["i"] += 1
                    else:
                        s["x"] += dx / dist * step
                        s["y"] += dy / dist * step
                    if dist > 1e-3:
                        s["yaw"] = math.degrees(math.atan2(dy, dx))
                elif s["state"] == "return":
                    s["state"] = "out"
                    s["out"] = True
                    fr["events"].append({"robot": rid, "kind": "robot_out", "text": f"{rid[-1]} is back outside"})
                else:
                    # finished its rooms early: head out
                    s["state"] = "return"
                    s["home"] = route(meta, plans[rid]["cur"], ent) + [(0.0, 2.2), (0.0, 1.0), pad[rid]]
                    s["i"] = 0
                    fr["events"].append({"robot": rid, "kind": "mission_complete",
                                         "text": f"{rid[-1]} has searched its rooms and is walking out"})
            fr["robots"].append({"id": rid, "x": round(s["x"], 2), "y": round(s["y"], 2), "yaw": round(s["yaw"], 1),
                                 "state": s["state"] if s["state"] != "wait" else "launch", "out": s["out"]})
            if s["y"] > house["bounds"][1] and not s["out"]:
                new = [c for c in cells_in_disc(s["x"], s["y"], see, house["bounds"]) if c not in s["seen"]]
                s["seen"].update(new)
                if new:
                    fr["seen"][rid] = new
                for p in meta["people"]:
                    if p["id"] in found:
                        continue
                    if math.hypot(p["x"] - s["x"], p["y"] - s["y"]) <= find_r and math.hypot(p["x"] - fx, p["y"] - fy) > fr_r:
                        found[p["id"]] = rid
                        ex_, ey_ = p["x"] + rnd.uniform(-0.25, 0.25), p["y"] + rnd.uniform(-0.25, 0.25)
                        fr["found"].append({"id": p["id"], "x": round(ex_, 2), "y": round(ey_, 2), "by": rid})
                        fr["events"].append({"robot": rid, "kind": "survivor_confirmed",
                                             "text": f"{rid[-1]} found a person and marked the spot"})
                        for o in ids:
                            if o != rid and frozenset((o, rid)) in linked:
                                fr["events"].append({"robot": o, "kind": "survivor_from_peer",
                                                     "text": f"{o[-1]} got the person's spot from {rid[-1]}"})
                c = (int((s["x"] - meta["x0"]) // meta["rw"]), int((s["y"] - meta["y0"]) // meta["rd"]))
                if c == fire_room and c not in fire_marked and fr_r > 0.8:
                    fire_marked.add(c)
                    fr["events"].append({"robot": rid, "kind": "room_marked_fire",
                                         "text": f"{rid[-1]} saw flames and marked the room as on fire"})
        for a in ids:
            for b in ids:
                if a < b and frozenset((a, b)) not in linked and t > 20:
                    sa, sb = st[a], st[b]
                    if sa["y"] > 2 and sb["y"] > 2 and math.hypot(sa["x"] - sb["x"], sa["y"] - sb["y"]) < 3.5:
                        linked.add(frozenset((a, b)))
                        fr["events"].append({"robot": a, "kind": "map_shared",
                                             "text": f"{a[-1]} and {b[-1]} recognised the same walls and joined their maps"})
        frames.append(fr)
        t += DT
    union = set()
    for s in st.values():
        union |= s["seen"]
    floor_cells = house["floor_m2"] / (CELL * CELL)
    out_t = max((f["t"] for f in frames if all(r["out"] for r in f["robots"])), default=None)
    first_all_out = next((f["t"] for f in frames if all(r["out"] for r in f["robots"])), None)
    score = {"people_found": len(found), "people_total": len(meta["people"]),
             "time_s": first_all_out if first_all_out is not None else MISSION_S,
             "coverage_pct": round(100 * len(union) / floor_cells, 1),
             "robots_out": sum(1 for s in st.values() if s["out"]), "false_confirmations": 0}
    return frames, score


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "page_fixture_out")
    (out / "cache").mkdir(parents=True, exist_ok=True)
    houses = []
    for spec in HOUSES:
        house, meta = build_house(*spec)
        houses.append(house)
        for n in (1, 2, 3):
            for diff in DIFF:
                key = f"{house['id']}__r{n}__{diff}"
                frames, score = simulate(house, meta, n, diff)
                d = out / "cache" / key
                d.mkdir(parents=True, exist_ok=True)
                nchunks = 0
                for ci in range(int(MISSION_S // 10)):
                    fr = [f for f in frames if 10 * ci <= f["t"] < 10 * ci + 10]
                    (d / f"chunk_{ci:04d}.json").write_text(json.dumps({"v": 1, "chunk": ci, "t0": 10 * ci, "frames": fr},
                                                                       separators=(",", ":")))
                    nchunks += 1
                (d / "status.json").write_text(json.dumps({
                    "v": 1, "mission": f"cache:{key}", "setup": key, "state": "done", "source": "recorded",
                    "chunks": nchunks, "sim_t": MISSION_S, "updated": "2026-10-26T00:00:00Z", "error": None,
                    "score": score, "engine": {"git": "fixture", "image": "fixture"}}))
    (out / "gallery.json").write_text(json.dumps({
        "v": 1, "mission_s": MISSION_S, "difficulties": DIFF, "houses": houses,
        "citation": "HouseExpo: Li, T., Ho, D., Li, C., Zhu, D., Wang, C., Meng, M. Q.-H. (2019). HouseExpo: A Large-scale "
                    "2D Indoor Layout Dataset for Learning-based Algorithms on Mobile Robots. MIT licence. "
                    "https://github.com/TeaganLi/HouseExpo"}, indent=1))
    print("wrote", out, len(houses), "houses")


if __name__ == "__main__":
    main()
