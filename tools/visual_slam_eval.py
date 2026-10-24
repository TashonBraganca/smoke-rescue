#!/usr/bin/env python3
"""Evaluator-side report of live visual SLAM in one recording (visual_slam_live): camera place matches between the robots,
the refined link between their frames against ground truth, and visual loop closures.

    brain/.venv/bin/python tools/visual_slam_eval.py data/recordings/<run> [--out data/runs/<dir>/visual_slam.json]

Reads the robots' decisions.jsonl and, as the evaluator, the oracle ground truth. Robots start in one shared frame from
their launch pads; a pad placement error turns a robot's whole frame. The true link from a sender's frame into a
receiver's is the difference of their true launch headings (brain angles turn counter-clockwise, Unity's clockwise).
"""
import argparse
import json
import math
from pathlib import Path

import run_audit as audit


def events(rec, robot, kinds):
    return [e for e in audit.rows(rec / robot / "decisions.jsonl") if e["kind"] in kinds]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("rec", type=Path); ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    heading = {r: math.degrees(audit.truth(a.rec, r)[0][3]) for r in audit.ROBOTS}
    report = {"recording": a.rec.name, "true_launch_heading_deg": heading, "robots": {}}
    for r in audit.ROBOTS:
        partner = next(p for p in audit.ROBOTS if p != r)
        # Link from the partner's frame into this one: the partner's launch turn relative to this robot's, brain sign.
        truth = -(heading[partner] - heading[r])
        recog = events(a.rec, r, {"place_recognised"})
        refined = events(a.rec, r, {"frame_link_refined"})
        rejected = events(a.rec, r, {"visual_link_rejected"})
        loops = events(a.rec, r, {"visual_loop_closure"})
        row = {"true_link_yaw_deg": round(truth, 3), "link_error_before_deg": round(abs(truth), 3),
               "camera_matches_verified": len(recog), "candidates_rejected_by_bound": len(rejected)}
        if recog:
            ys = [((e["numbers"]["link_yaw_deg"] + 180) % 360) - 180 for e in recog]
            row["candidate_link_yaw_deg"] = {"min": round(min(ys), 3), "max": round(max(ys), 3),
                                             "median_error_deg": round(sorted(abs(y - truth) for y in ys)[len(ys) // 2], 3)}
        if refined:
            e = refined[0]; y = ((e["numbers"]["dyaw_deg"] + 180) % 360) - 180
            row["refined"] = {"t_s": round(e["t"], 1), "link_yaw_deg": round(y, 3), "link_error_after_deg": round(abs(y - truth), 3),
                              "dx_m": round(e["numbers"]["dx_m"], 3), "dy_m": round(e["numbers"]["dy_m"], 3),
                              "candidates": int(e["numbers"]["candidates"])}
        if loops:
            row["loop_closures"] = {"count": len(loops), "applied": sum(int(e["numbers"]["applied"]) for e in loops),
                                    "median_drift_m": round(sorted(e["numbers"]["drift_m"] for e in loops)[len(loops) // 2], 4),
                                    "median_drift_deg": round(sorted(abs(e["numbers"]["drift_deg"]) for e in loops)[len(loops) // 2], 3),
                                    "median_age_s": round(sorted(e["numbers"]["age_s"] for e in loops)[len(loops) // 2], 1)}
        report["robots"][r] = row
    text = json.dumps(report, indent=1)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(text)


if __name__ == "__main__":
    main()
