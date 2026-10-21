"""Score recordings against ground truth and compare them, e.g. the same scenario with and without the agent layer.

    uv run python -m brain.compare_runs ../data/recordings/unity_haze_auto_v1 ../data/recordings/unity_haze_agents_v1

Evaluator: reads oracle/ (survivor truth, true poses, heat at the body, collisions) and the robots' own decision logs.
Per run: survivors confirmed and how many are real (within 1.5 m of a true survivor), time to the first real one,
false confirmations, floor area searched (cells within 2.5 m of a robot's true path, a coverage proxy), the share of
that area both robots searched, heat exposure, collision time, and the agent layer's steps, orders and refusals.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from brain.nav_io import load_scenario, read_json_lines, recording_frame, transform_xy
from brain.oracle import load_survivors, load_trajectory

MATCH_M, SEARCH_RADIUS_M, CELL_M = 1.5, 2.5, 0.5


def _decisions(robot_dir: Path) -> list[dict]:
    p = robot_dir / "decisions.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def score(rec: Path) -> dict:
    rec = Path(rec)
    meta = json.loads((rec / "meta.json").read_text())
    robots = meta["robots"]
    truth_people = [(s["id"], np.array([s["position_house"][0], s["position_house"][2]])) for s in (load_survivors(rec) or [])]
    confirmations, found, false = [], {}, []
    scenario = load_scenario(meta)
    for r in robots:
        # Robots report positions in their own frames; place them on the plan with the true start (evaluation only).
        frame, _ = recording_frame(read_json_lines(rec / r / "brain_state.jsonl"), scenario, r)
        for e in _decisions(rec / r):
            if e["kind"] != "survivor_confirmed":
                continue
            n = e.get("numbers") or {}
            p = transform_xy(np.array([n.get("x_m", np.nan), n.get("y_m", np.nan)]), frame)
            d = [(float(np.linalg.norm(p - q)), sid) for sid, q in truth_people]
            best = min(d) if d else (np.inf, None)
            by_agent = "Agent confirmed" in (e.get("reason") or "")
            rec_c = {"robot": r, "t": e["t"], "x": p[0], "y": p[1], "nearest_m": round(best[0], 2), "survivor": best[1] if best[0] <= MATCH_M else None, "by_agent": by_agent}
            confirmations.append(rec_c)
            if rec_c["survivor"]:
                found.setdefault(rec_c["survivor"], e["t"])
            else:
                false.append(rec_c)
    cells = {}
    heat = {}
    for r in robots:
        tr = load_trajectory(rec, r)
        xy = tr.positions[:, [0, 2]]
        s = set()
        rad = int(np.ceil(SEARCH_RADIUS_M / CELL_M))
        for x, y in xy[::5]:
            cx, cy = int(np.floor(x / CELL_M)), int(np.floor(y / CELL_M))
            for dx in range(-rad, rad + 1):
                for dy in range(-rad, rad + 1):
                    if (dx * dx + dy * dy) * CELL_M * CELL_M <= SEARCH_RADIUS_M ** 2:
                        s.add((cx + dx, cy + dy))
        cells[r] = s
        hp = rec / "oracle" / f"{r}_heat.txt"
        if hp.exists():
            h = np.loadtxt(hp, comments="#", ndmin=2)
            dt = float(np.median(np.diff(h[:, 0]))) if len(h) > 1 else 0.0
            heat[r] = {"max_air_c": round(float(h[:, 1].max()), 1), "s_air_over_60c": round(float((h[:, 1] > 60).sum() * dt), 1),
                       "max_surface_within_1m_c": round(float(h[:, 2].max()), 1)}
    union = set().union(*cells.values()) if cells else set()
    both = set.intersection(*cells.values()) if len(cells) > 1 else set()
    coll = json.loads((rec / "oracle" / "collisions.json").read_text()) if (rec / "oracle" / "collisions.json").exists() else {}
    agent = {}
    if (rec / "agent_summary.json").exists():
        agent = json.loads((rec / "agent_summary.json").read_text())
        kinds = Counter()
        tp = rec / "agent_trace.jsonl"
        if tp.exists():
            for l in tp.read_text().splitlines():
                e = json.loads(l)
                if e.get("kind") == "step_end":
                    kinds.update(o["type"] for o in e.get("orders", []))
        agent["orders_by_type"] = dict(kinds)
    duration = max(float(load_trajectory(rec, r).timestamps[-1]) for r in robots)
    # People carried out: released from a sled outside every room of the floor plan (recorder log, world truth).
    rescued, released = set(), []
    rp = rec / "rescues.jsonl"
    if rp.exists():
        for line in rp.read_text().splitlines():
            e = json.loads(line)
            if e.get("event") == "released":
                released.append(e)
                if e.get("outside_building"):
                    rescued.add(e["person"])
    return {"recording": rec.name, "duration_s": round(duration, 1), "survivors_total": len(truth_people),
            "survivors_found": len(found), "first_found_s": round(min(found.values()), 1) if found else None,
            "confirmations": len(confirmations), "false_confirmations": len(false), "confirmation_details": confirmations,
            "area_searched_m2": round(len(union) * CELL_M ** 2, 1),
            "area_searched_by_both_m2": round(len(both) * CELL_M ** 2, 1),
            "double_coverage_share": round(len(both) / len(union), 3) if union else None,
            "people_carried_out": len(rescued), "releases": released,
            "heat": heat, "collision_s": coll, "agent": agent}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("recordings", type=Path, nargs="+")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    rows = [score(r) for r in a.recordings]
    keys = ["duration_s", "survivors_total", "survivors_found", "first_found_s", "confirmations", "false_confirmations",
            "area_searched_m2", "double_coverage_share", "people_carried_out", "collision_s"]
    print(f"{'':28s}" + "".join(f"{r['recording'][:26]:>28s}" for r in rows))
    for k in keys:
        print(f"{k:28s}" + "".join(f"{str(r[k])[:26]:>28s}" for r in rows))
    for r in rows:
        print(r["recording"], "heat", r["heat"], "agent", {k: v for k, v in r["agent"].items() if k != "url"})
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
