"""Score recordings against ground truth and compare them, e.g. the same scenario with and without the agent layer.

    uv run python -m brain.compare_runs ../data/recordings/unity_haze_auto_v1 ../data/recordings/unity_haze_agents_v1

Evaluator: reads oracle/ (survivor truth, true poses, heat at the body, collisions) and the robots' own decision logs.
Per run: distinct survivors confirmed within the configured match radius, time to the first real one,
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

EVALUATION_CONFIG = Path(__file__).resolve().parents[2] / "config" / "evaluation.json"


def evaluation_config() -> dict:
    return json.loads(EVALUATION_CONFIG.read_text())


# Preserve these imports for existing analysis callers.
_CONFIG = evaluation_config()
MATCH_M = _CONFIG["survivor_match_m"]
SEARCH_RADIUS_M = _CONFIG["search_radius_m"]
CELL_M = _CONFIG["coverage_cell_m"]


def _decisions(robot_dir: Path) -> list[dict]:
    p = robot_dir / "decisions.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def score_confirmations(rec: Path) -> dict:
    """Match time-ordered confirmations to each person's position at that time."""
    rec = Path(rec)
    meta = json.loads((rec / "meta.json").read_text())
    robots = meta["robots"]
    truth_people = [(s["id"], np.array([s["position_house"][0], s["position_house"][2]])) for s in (load_survivors(rec) or [])]
    match_m = float(evaluation_config()["survivor_match_m"])
    if not np.isfinite(match_m) or match_m <= 0:
        raise ValueError("survivor_match_m must be finite and positive")
    rescue_path = rec / "rescues.jsonl"
    rescue_events = [json.loads(line) for line in rescue_path.read_text().splitlines() if line.strip()] if rescue_path.exists() else []
    releases = sorted((e for e in rescue_events
                       if e.get("event") == "released"), key=lambda e: e["t"])
    rescued = {e["person"] for e in releases if e.get("outside_building")}
    confirmations = []
    scenario = load_scenario(meta)
    for r in robots:
        # Robots report positions in their own frames; place them on the plan with the true start (evaluation only).
        frame, _ = recording_frame(read_json_lines(rec / r / "brain_state.jsonl"), scenario, r)
        for e in _decisions(rec / r):
            if e["kind"] != "survivor_confirmed":
                continue
            n = e.get("numbers") or {}
            p = transform_xy(np.array([n.get("x_m", np.nan), n.get("y_m", np.nan)]), frame)
            by_agent = "Agent confirmed" in (e.get("reason") or "")
            confirmations.append({"robot": r, "t": e["t"], "x": p[0], "y": p[1], "by_agent": by_agent})
    confirmations.sort(key=lambda e: e["t"])
    positions = dict(truth_people)
    moved, found = set(), {}
    release_index = 0
    for confirmation in confirmations:
        while release_index < len(releases) and releases[release_index]["t"] <= confirmation["t"]:
            release = releases[release_index]
            sid = release["person"]
            if sid in positions and "x" in release and "z" in release:
                # Recorder release coordinates already use the house frame.
                positions[sid] = np.array([release["x"], release["z"]])
                moved.add(sid)
            release_index += 1
        point = np.array([confirmation["x"], confirmation["y"]])
        distances = [(float(np.linalg.norm(point - position)), sid)
                     for sid, position in positions.items()]
        best = min(distances) if distances else (np.inf, None)
        sid = best[1] if np.isfinite(best[0]) and best[0] <= match_m else None
        confirmation.update(nearest_m=round(best[0], 2), survivor=sid,
                            duplicate=sid is not None and sid in found,
                            matched_after_rescue=sid is not None and sid in moved,
                            moved=sid is not None and sid in moved)
        if sid is not None:
            found.setdefault(sid, confirmation["t"])
    return {"survivors_total": len(truth_people), "survivors_found": len(found),
            "distinct_people_confirmed": len(found),
            "first_found_s": round(min(found.values()), 1) if found else None,
            "confirmations": len(confirmations),
            "false_confirmations": sum(c["survivor"] is None for c in confirmations),
            "duplicate_confirmations": sum(c["duplicate"] for c in confirmations),
            "moved_confirmations": sum(c["matched_after_rescue"] for c in confirmations),
            # Each repeated confirmation stays in the denominator.
            "precision": len(found) / len(confirmations) if confirmations else None,
            "confirmation_precision": len(found) / len(confirmations) if confirmations else None,
            "confirmation_details": confirmations,
            "people_carried_out": len(rescued), "releases": releases}


def score(rec: Path) -> dict:
    rec = Path(rec)
    meta = json.loads((rec / "meta.json").read_text())
    robots = meta["robots"]
    confirmation_score = score_confirmations(rec)
    config = evaluation_config()
    search_radius, cell = config["search_radius_m"], config["coverage_cell_m"]
    cells = {}
    heat = {}
    for r in robots:
        tr = load_trajectory(rec, r)
        xy = tr.positions[:, [0, 2]]
        s = set()
        rad = int(np.ceil(search_radius / cell))
        for x, y in xy[::config["coverage_pose_stride"]]:
            cx, cy = int(np.floor(x / cell)), int(np.floor(y / cell))
            for dx in range(-rad, rad + 1):
                for dy in range(-rad, rad + 1):
                    if (dx * dx + dy * dy) * cell * cell <= search_radius ** 2:
                        s.add((cx + dx, cy + dy))
        cells[r] = s
        hp = rec / "oracle" / f"{r}_heat.txt"
        if hp.exists():
            h = np.loadtxt(hp, comments="#", ndmin=2)
            dt = float(np.median(np.diff(h[:, 0]))) if len(h) > 1 else 0.0
            heat[r] = {"max_air_c": round(float(h[:, 1].max()), 1), "s_air_over_60c": round(float((h[:, 1] > config["heat_exposure_c"]).sum() * dt), 1),
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
    return {"recording": rec.name, "duration_s": round(duration, 1), **confirmation_score,
            "area_searched_m2": round(len(union) * cell ** 2, 1),
            "area_searched_by_both_m2": round(len(both) * cell ** 2, 1),
            "double_coverage_share": round(len(both) / len(union), 3) if union else None,
            "heat": heat, "collision_s": coll, "agent": agent}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("recordings", type=Path, nargs="+")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    rows = [score(r) for r in a.recordings]
    keys = ["duration_s", "survivors_total", "survivors_found", "first_found_s", "confirmations",
            "duplicate_confirmations", "moved_confirmations", "false_confirmations", "confirmation_precision",
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
