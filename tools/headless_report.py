#!/usr/bin/env python3
"""Score a headless simulator run (evaluator side): stuck, turning in place, idle and outside time per robot, rooms in the
order they were entered, camera-searched share of each room, first sighting and confirmation of each person, radio, and
a top-down picture.

    python tools/headless_report.py data/runs/headless/5001_default [more run dirs]
"""
import base64
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUCK_KINDS = ("stuck", "collision_hold", "no_progress", "goal_blacklisted", "collision_risk")


def inside(poly, q):
    x, y, hit = q[0], q[1], False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[i - 1]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def room_of(world, q):
    return next((r["id"] for r in world["rooms"] if inside(r["poly"], q)), None)


def report(run_dir: Path) -> dict:
    result = json.loads((run_dir / "result.json").read_text())
    world = json.loads((ROOT / result["world"]).read_text())
    rows, bad = [], 0
    for l in (run_dir / "trace.jsonl").read_text().splitlines():
        try:
            rows.append(json.loads(l)) if l.strip() else None
        except json.JSONDecodeError:
            bad += 1
    kinds = {r["id"]: r.get("kind", r["id"]) for r in world["rooms"]}
    out = {"run": str(run_dir), "settings": result.get("settings"), "sim_s": result["duration_s"],
           "wall_s": round(result.get("wall_s", 0)), "bad_trace_lines": bad, "robots": {}, "rooms": {}, "people": {}}

    # Per robot motion, from pose samples about every 0.53 s.
    for rid in sorted({r["robot"] for r in rows if r["kind"] == "pose"}):
        poses = [r for r in rows if r["kind"] == "pose" and r["robot"] == rid]
        acc, entered, states = Counter(), [], Counter()
        for a, b in zip(poses, poses[1:]):
            dt = b["t"] - a["t"]
            v, w = abs(a["v"]), abs(a["w_dps"])
            states[a["state"]] += dt
            room = room_of(world, (a["x"], a["y"]))
            if room is None:
                acc["outside_s"] += dt
            elif not entered or entered[-1][0] != room:
                entered.append((room, round(a["t"], 1)))
            if v < .05 and w > 5:
                acc["turning_in_place_s"] += dt; acc["turned_in_place_deg"] += w * dt
            elif v < .05 and a["state"] not in ("complete",):
                acc["still_s"] += dt
        events = Counter(r["kind"] for r in rows if r.get("robot") == rid and r["kind"] not in ("pose", "sighting", "radio"))
        m = next(x for x in result["maps"] if x["robot"] == rid)
        out["robots"][rid] = {
            "walked_m": round(m["walked_m"], 1), "final_state": m["final_state"],
            "peak_air_c": round(m.get("peak_air_c", 20)), "air_over_60_s": round(m.get("air_over_60_s", 0)),
            **{k: round(v, 1) for k, v in acc.items()},
            "turned_in_place_turns": round(acc["turned_in_place_deg"] / 360, 1),
            "states_s": {k: round(v) for k, v in states.most_common()},
            "rooms_in_order": [(f"{r}:{kinds[r]}", t) for r, t in entered],
            "stuck_events": {k: events[k] for k in STUCK_KINDS if events[k]},
            "room_claimed": events["room_claimed"], "room_yielded": events["room_yielded"],
            "events": dict(events.most_common(12)),
        }

    # Camera-searched floor, union of both robots' own searched masks, per room.
    masks = []
    for m in result["maps"]:
        n, cell, (ox, oy) = m["grid_size"], m["cell_m"], m["origin"]
        bits = base64.b64decode(m["own_searched"])
        masks.append((n, cell, ox, oy, bits))
    n, cell, ox, oy, _ = masks[0]
    searched = [any(mk[4][i] for mk in masks) for i in range(n * n)]
    per_robot = [[bool(mk[4][i]) for i in range(n * n)] for mk in masks]
    for room in world["rooms"]:
        xs, ys = [p[0] for p in room["poly"]], [p[1] for p in room["poly"]]
        total = hit = 0; by = [0] * len(masks)
        for row in range(max(0, int((min(ys) - oy) / cell)), min(n, int((max(ys) - oy) / cell) + 1)):
            for col in range(max(0, int((min(xs) - ox) / cell)), min(n, int((max(xs) - ox) / cell) + 1)):
                q = (ox + (col + .5) * cell, oy + (row + .5) * cell)
                if not inside(room["poly"], q):
                    continue
                total += 1; i = row * n + col; hit += searched[i]
                for k in range(len(masks)): by[k] += per_robot[k][i]
        out["rooms"][f"{room['id']}:{kinds[room['id']]}"] = {"searched_pct": round(100 * hit / max(1, total)),
                                                            "by_robot_pct": [round(100 * b / max(1, total)) for b in by]}
    all_cells = sum(1 for r in out["rooms"].values())
    out["house_searched_pct_mean_over_rooms"] = round(sum(r["searched_pct"] for r in out["rooms"].values()) / max(1, all_cells))

    # People: first sighting by any robot, and confirmations within 1.05 m (the evaluator's match distance).
    confirmed = [(r["t"], r["robot"], r["numbers"]["x_m"], r["numbers"]["y_m"]) for r in rows
                 if r["kind"] == "survivor_confirmed" and r.get("numbers") and "x_m" in r["numbers"]]
    for p in world["people"]:
        seen = [r["t"] for r in rows if r["kind"] == "sighting" and r["person"] == p["id"]]
        conf = [c for c in confirmed if math.dist((c[2], c[3]), p["at"]) <= 1.05]
        out["people"][f"{p['id']}:{kinds.get(p.get('room'), p.get('room'))}"] = {
            "first_sighting_s": round(min(seen), 1) if seen else None,
            "confirmed_s": round(min(c[0] for c in conf), 1) if conf else None}
    out["false_confirmations"] = sum(1 for c in confirmed if all(math.dist((c[2], c[3]), p["at"]) > 1.05 for p in world["people"]))
    radio = [r for r in rows if r["kind"] == "radio"]
    out["radio"] = {"sent": len(radio), "delivered": sum(r["ok"] for r in radio),
                    "with_room_claim": sum(1 for r in radio if r.get("room_claim")),
                    "mean_packet_bytes": sum(r.get('bytes', 0) for r in radio) / max(1, len(radio))}
    out['team_map_by_receiver'] = headless_team_maps(run_dir, result, world)
    draw(run_dir, world, rows, masks)
    return out


def polygon_mask(xy, poly):
    import numpy as np
    x, y = xy[..., 0], xy[..., 1]
    hit = np.zeros(x.shape, bool)
    for i, p in enumerate(poly):
        q = poly[i - 1]
        if p[1] == q[1]:
            continue
        hit ^= ((p[1] > y) != (q[1] > y)) & (x < (q[0] - p[0]) * (y - p[1]) / (q[1] - p[1]) + p[0])
    return hit


def headless_team_maps(run_dir, result, world):
    """Evaluate each received team belief against simulator geometry, outside RobotBrain."""
    import numpy as np
    sys.path.insert(0, str(ROOT / 'brain'))
    from brain.team_map import TeamGrid, load_team_map
    from brain.nav_eval import team_belief_metrics
    reports = {}
    segments = np.array([s[:4] for s in world['segments'] if s[4] <= .2 and s[5] >= 1.5], float).reshape(-1, 2, 2)
    for snapshot in result['maps']:
        path = run_dir / (snapshot['robot'] + '.team.bin.gz')
        if path.exists():
            grid = load_team_map(path)
            representation = 'confidence_export'
        else:
            n = snapshot['grid_size']
            occupancy = np.frombuffer(base64.b64decode(snapshot['occupancy']), np.uint8).reshape(n, n)
            searched = np.frombuffer(base64.b64decode(snapshot['searched']), np.uint8).reshape(n, n) > 0
            known = occupancy != 0
            grid = TeamGrid(n, snapshot['cell_m'], np.array(snapshot['origin']), result['duration_s'],
                            (snapshot['robot'] + ':legacy_fused',), np.where(occupancy == 255, 1., np.where(occupancy == 1, 0., .5)),
                            searched.astype(float), known.astype(np.uint32), searched.astype(np.uint32),
                            np.where(known, 0., -1.), np.where(searched, 0., -1.))
            representation = 'legacy_binary_no_confidence_or_age'
        yy, xx = np.indices((grid.size, grid.size))
        xy = grid.origin + (np.stack([xx, yy], axis=-1) + .5) * grid.cell_m
        floor = np.zeros((grid.size, grid.size), bool)
        for room in world['rooms']:
            floor |= polygon_mask(xy, room['poly'])
        for prism in world['prisms']:
            if prism['height_m'] >= .15:
                floor &= ~polygon_mask(xy, prism['poly'])
        for a, b in segments:
            d = b - a
            u = np.clip(((xy - a) * d).sum(axis=2) / max(1e-12, (d * d).sum()), 0, 1)
            floor &= np.linalg.norm(xy - a - u[..., None] * d, axis=2) > grid.cell_m * .5
        reports[snapshot['robot']] = team_belief_metrics(grid, xy, segments, floor) | dict(representation=representation, cell_m=grid.cell_m)
    return reports


def draw(run_dir, world, rows, masks):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    pts = [p for r in world["rooms"] for p in r["poly"]] + [r["at"] for r in world["robots"]]
    x0, x1 = min(p[0] for p in pts) - 2, max(p[0] for p in pts) + 2
    y0, y1 = min(p[1] for p in pts) - 2, max(p[1] for p in pts) + 2
    s = 40
    img = Image.new("RGB", (int((x1 - x0) * s), int((y1 - y0) * s)), "white"); d = ImageDraw.Draw(img)
    P = lambda q: ((q[0] - x0) * s, (y1 - q[1]) * s)
    colours = [(242, 169, 0), (47, 163, 230)]
    for k, (n, cell, ox, oy, bits) in enumerate(masks):
        for i, b in enumerate(bits):
            if b:
                q = (ox + (i % n) * cell, oy + (i // n + 1) * cell); a = P(q)
                c = colours[k % 2]; d.rectangle([a[0], a[1], a[0] + cell * s, a[1] + cell * s], fill=tuple(int(v * .35 + 255 * .65) for v in c))
    for seg in world["segments"]:
        d.line([P(seg[0:2]), P(seg[2:4])], fill=(150, 150, 150) if seg[4] > 1 else (30, 30, 30), width=3)
    for pr in world["prisms"]:
        d.polygon([P(q) for q in pr["poly"]], outline=(120, 90, 60))
    for k, rid in enumerate(sorted({r["robot"] for r in rows if r["kind"] == "pose"})):
        path = [P((r["x"], r["y"])) for r in rows if r["kind"] == "pose" and r["robot"] == rid]
        if len(path) > 1:
            d.line(path, fill=colours[k % 2], width=3)
    seen = {r["person"] for r in rows if r["kind"] == "sighting"}
    for p in world["people"]:
        a = P(p["at"]); c = (40, 160, 60) if p["id"] in seen else (220, 30, 30)
        d.ellipse([a[0] - 8, a[1] - 8, a[0] + 8, a[1] + 8], fill=c); d.text((a[0] + 10, a[1] - 6), p["id"], fill=c)
    for r in world["rooms"]:
        cx = sum(q[0] for q in r["poly"]) / 4; cy = sum(q[1] for q in r["poly"]) / 4
        d.text(P((cx, cy)), f"{r['id']} {r.get('kind', '')}", fill=(0, 0, 0))
    img.save(run_dir / "topdown.png")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(json.dumps(report(Path(arg).resolve()), indent=1))
