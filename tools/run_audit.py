#!/usr/bin/env python3
"""Evaluator-side audit of one recording: every robot and agent mistake with its time, checked against ground truth.

    brain/.venv/bin/python tools/run_audit.py data/recordings/<run> [--out data/runs/<dir>/audit.txt]

Reads the robots' decisions.jsonl, the agents' agent_trace.jsonl and the oracle folder. The oracle is read here, by the
evaluator, never by robot code. Positions in the robots' logs are in the shared gate frame (robot A's pad at the
origin, y along A's first heading); they are mapped to the house with robot A's first ground-truth pose.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROBOTS = ("robot_A", "robot_B")
CAMERA_FORWARD_M = json.loads((Path(__file__).resolve().parents[1] / "config/sensors.json").read_text())["camera"]["mount"]["forward_offset_m"]


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip().startswith("{")] if path.exists() else []


def truth(rec, robot):
    out = []
    for l in open(rec / "oracle" / f"{robot}_groundtruth.txt"):
        if l[0] == "#": continue
        t, x, y, z, qx, qy, qz, qw = map(float, l.split()[:8])
        h = math.atan2(2 * (qx * qz + qw * qy), 1 - 2 * (qx * qx + qy * qy))
        # Ground truth is the camera's pose, 0.25 m ahead of the body centre (camera mount forward_offset_m): turning in
        # place swings the camera at about 0.13 m/s, so standing and turning are judged on the body centre.
        out.append((t, x - CAMERA_FORWARD_M * math.sin(h), z - CAMERA_FORWARD_M * math.cos(h), h))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("rec", type=Path); ap.add_argument("--out", type=Path)
    a = ap.parse_args(); rec = a.rec
    lines = []; say = lines.append
    meta = json.loads((rec / "meta.json").read_text()) if (rec / "meta.json").exists() else {}
    plan = json.loads((rec / "oracle" / "floorplan_used.json").read_text())
    people = json.loads((rec / "oracle" / "survivors.json").read_text())
    P = {p["id"]: (p["position_house"][0], p["position_house"][2], p["room"], p.get("pose")) for p in people}
    tr = {r: truth(rec, r) for r in ROBOTS}
    x0, z0 = tr["robot_A"][0][1], tr["robot_A"][0][2]
    house = lambda bx, by: (x0 + bx, z0 + by)          # gate frame to house frame (A faced +z at its pad)
    dec = {r: rows(rec / r / "decisions.jsonl") for r in ROBOTS}
    agent = rows(rec / "agent_trace.jsonl")
    end = max(v[-1][0] for v in tr.values())
    say(f"AUDIT {rec.name}: {end:.0f} s simulated, complete={meta.get('complete')}, {len(P)} people, {len(plan['rooms'])} rooms")

    # ---------------- people: found, missed, passed close unseen ----------------
    say("\nPEOPLE")
    confirmed = defaultdict(list)
    for r in ROBOTS:
        for e in dec[r]:
            if e["kind"] == "survivor_confirmed":
                n = e["numbers"]; hx, hz = house(n["x_m"], n["y_m"])
                pid, d = min(((k, math.dist((hx, hz), v[:2])) for k, v in P.items()), key=lambda q: q[1])
                confirmed[pid if d <= 1.5 else "FALSE"].append((r, round(e["t"], 1), round(d, 2)))
    for pid, (px, pz, room, pose) in P.items():
        near = []
        for r in ROBOTS:
            for t, x, z, h in tr[r]:
                d = math.dist((x, z), (px, pz))
                if d <= 2.5:
                    off = abs((math.degrees(math.atan2(px - x, pz - z) - h) + 180) % 360 - 180)
                    near.append((r, t, d, off))
        closest = min((math.dist((x, z), (px, pz)) for r in ROBOTS for t, x, z, h in tr[r]), default=99)
        status = "confirmed " + str(confirmed[pid]) if pid in confirmed else "MISSED"
        say(f"  {pid} {room} ({pose}): {status}; closest robot {closest:.1f} m")
        if pid not in confirmed and near:
            facing = [n for n in near if n[3] <= 37]
            span = (min(n[1] for n in near), max(n[1] for n in near))
            say(f"    MISTAKE: a robot stood within 2.5 m for {len(near) * 0.0 + (span[1] - span[0]):.0f} s ({span[0]:.0f}-{span[1]:.0f} s) and never confirmed; "
                f"facing her (camera half-view 37 deg) in {len(facing)} of {len(near)} samples")
    if confirmed.get("FALSE"): say(f"  MISTAKE: confirmations more than 1.5 m from any person: {confirmed['FALSE']}")

    # ---------------- per robot: still, turning, outside, rooms, events ----------------
    for r in ROBOTS:
        say(f"\n{r.upper()}")
        t0 = tr[r]; still = []; walked = 0; outside = 0; rooms = set()
        X0, Z0, X1, Z1 = plan["footprint"]
        cur = None
        for (ta, xa, za, ha), (tb, xb, zb, hb) in zip(t0, t0[1:]):
            dt = tb - ta; d = math.dist((xa, za), (xb, zb)); walked += d
            if not (X0 <= xb <= X1 and Z0 <= zb <= Z1): outside += dt
            for rm in plan["rooms"]:
                rx0, rz0, rx1, rz1 = rm["rect"]
                if rx0 + .6 < xb < rx1 - .6 and rz0 + .6 < zb < rz1 - .6: rooms.add(rm["id"])
            if dt > 0 and d / dt < .05:
                cur = [ta, tb] if cur is None else [cur[0], tb]
            elif cur is not None:
                still.append(cur); cur = None
        if cur: still.append(cur)
        # Turning in place: heading change while the body moves under 0.05 m/s, and whole 360 deg turns made that way.
        turn_s = turn_deg = 0.0
        for (ta, xa, za, ha), (tb, xb, zb, hb) in zip(t0, t0[1:]):
            dt = tb - ta; dh = abs((math.degrees(hb - ha) + 180) % 360 - 180)
            if dt > 0 and math.dist((xa, za), (xb, zb)) / dt < .05 and dh / dt > 5: turn_s += dt; turn_deg += dh
        evs = [(e["t"], e["kind"], e.get("reason", "")) for e in dec[r]]
        def cause(t):
            c = [e for e in evs if e[0] <= t and e[1] in ("goal", "agent_order", "retreat", "no_frontier", "sweep_look", "survivor_confirmed", "stuck", "collision_hold")]
            return (c[-1][1] + ": " + c[-1][2][:80]) if c else "start"
        long = [s for s in still if s[1] - s[0] >= 5]
        say(f"  turning in place {turn_s:.0f} s, {turn_deg:.0f} deg ({turn_deg / 360:.1f} full turns)")
        say(f"  walked {walked:.0f} m, still {sum(b - a for a, b in still):.0f} s ({100 * sum(b - a for a, b in still) / end:.0f} %), outside {outside:.0f} s, rooms entered {len(rooms)}/{len(plan['rooms'])}")
        for a_, b_ in long: say(f"    MISTAKE still {b_ - a_:.0f} s at {a_:.0f}-{b_:.0f} s, after {cause(a_)}")
        k = Counter(e[1] for e in evs)
        say("  events " + ", ".join(f"{x} {k[x]}" for x in ("stuck", "collision_hold", "pinned", "goal_blacklisted", "no_progress", "retreat",
                                                            "retreat_reduced_clearance", "body_swing_blocked", "no_frontier", "survivor_candidate") if k[x]))
        fire_cut = [e for e in evs if e[1] == "goal_blacklisted" and "fire" in e[2]]
        if fire_cut: say(f"    MISTAKE fire cut {len(fire_cut)} routes ({fire_cut[0][0]:.0f}-{fire_cut[-1][0]:.0f} s)")
        wall = sum(1 for e in evs if e[1] == "survivor_candidate" and "long mapped wall" in e[2])
        if wall: say(f"  warm-wall candidates raised: {wall} of {k['survivor_candidate']} candidates")

    # ---------------- agents ----------------
    say("\nAGENTS")
    tools = [e for e in agent if e.get("kind") == "tool"]; steps = [e for e in agent if e.get("kind") == "step_end"]
    starts = {e["step"]: e for e in agent if e.get("kind") == "step_start"}
    c = Counter((e["name"], e.get("verdict")) for e in tools if not e.get("refused"))
    say("  tool calls: " + ", ".join(f"{n}{'/' + v if v else ''} {k}" for (n, v), k in c.most_common()))
    say(f"  steps {len(steps)}; hit 6 rounds {sum(e.get('rounds') == 6 for e in steps)}; no summary "
        f"{sum((e.get('summary') or '').startswith('(no summary') or e.get('summary') is None for e in steps)}; "
        f"model errors {sum(e.get('kind') == 'model_error' for e in agent)}; deadlines {sum(e.get('kind') == 'step_deadline' for e in agent)}")
    lat = [e.get("latency_s", 0) for e in agent if e.get("kind") == "model_reply"]
    if lat: say(f"  model latency per round: median {sorted(lat)[len(lat) // 2]:.1f} s, max {max(lat):.1f} s, {len(lat)} rounds")
    for e in tools:
        if e.get("refused"): say(f"    REFUSED {e['step']} {e['name']}: {(e.get('reason') or '')[:120]}")
    sprays = [e for e in tools if e["name"] == "spray_water" and not e.get("refused")]
    offers = sum(1 for s in starts.values() if (s.get("context") or {}).get("spray_option"))
    say(f"  spray offers {offers} steps, sprays {len(sprays)}")
    # rejections near a real person (evaluator): the rejected point within 1.5 m of a person
    for e in tools:
        if e["name"] == "reject_candidate" and not e.get("refused"):
            a_ = e.get("arguments") or {}
            if "x" in a_:
                hx, hz = house(float(a_["x"]), float(a_["y"]))
                pid, d = min(((k, math.dist((hx, hz), v[:2])) for k, v in P.items()), key=lambda q: q[1])
                if d <= 1.5: say(f"    MISTAKE {e['step']} rejected a spot {d:.1f} m from real person {pid}")
    out = "\n".join(lines)
    print(out)
    if a.out: a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(out + "\n")


if __name__ == "__main__":
    main()
