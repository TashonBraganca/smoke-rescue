"""Replay a recording's agent decision steps through the current agent code with a live or recorded model.

Each recorded step_start in agent_trace.jsonl carries the exact context the model saw. This tool rebuilds the request,
runs RobotAgent.step on it (tools run on the recorded frames, so inspections are real), and counts what the agent does
now: steps skipped before any model call, refusals by rule, orders by type, radio messages, rounds and latency. Set
against the recorded run it shows what the agent-layer changes do to the decisions, without a simulation. It is not a
closed-loop run: the robot does not move in response, so the next step's context is still the recorded one.

    cd brain && AGENT_BACKEND=azure AZURE_REASONING_EFFORT=medium uv run python ../tools/eval/agent_replay.py \
        ../data/recordings/unity_house_onedoor_seed932_nolimit_run1 --out /tmp/replay932 [--max-steps 40] [--robot robot_A]
    AGENT_BACKEND=openrouter AGENT_MODEL=<provider/model> ... (OPENROUTER_API_KEY in the environment)
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "brain"))

from brain.agent.llm import make_backend          # noqa: E402
from brain.agent.robot_agent import RobotAgent    # noqa: E402

CONTEXT_KEYS = ("robot", "t", "pose", "torch", "mode", "state", "goal", "heat_budget", "air_c", "event", "inbox", "orders",
                "survivors", "frontiers", "lidar", "partners", "blocked_unsearched", "fire_blocks_frontier", "fire_blocks_exit",
                "fire_near_robot", "mission_time_left_s", "water_l", "carrying_person", "rooms")


def thermal_times(robot_dir: Path) -> list[float]:
    out = []
    for line in (robot_dir / "thermal.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        out.append(float(line.split()[0]))
    return out


def recorded_summary(trace: list[dict]) -> dict:
    orders = collections.Counter(); steps = 0; skipped = 0; lat = []; tools = collections.Counter()
    for d in trace:
        robot = d.get("robot") or str(d.get("step", "")).split("@")[0]
        if not robot.startswith("robot"):
            continue
        if d["kind"] == "step_start": steps += 1
        if d["kind"] == "step_skipped": skipped += 1
        if d["kind"] == "tool": tools[d["name"]] += 1
        if d["kind"] == "step_end":
            lat.append(d.get("latency_s", 0))
            for o in d.get("orders") or []: orders[o.get("type")] += 1
    return {"steps_run": steps, "steps_skipped": skipped, "orders": dict(orders), "tools": dict(tools),
            "latency_median_s": round(statistics.median(lat), 1) if lat else None, "latency_sum_s": round(sum(lat))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("recording"); ap.add_argument("--out", required=True)
    ap.add_argument("--robot", default=None); ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--budget-usd", type=float, default=5.0, help="stop when an OpenRouter backend reports more spend than this")
    args = ap.parse_args()
    rec = Path(args.recording).resolve(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    trace = [json.loads(l) for l in (rec / "agent_trace.jsonl").read_text().splitlines()]
    starts = [d for d in trace if d["kind"] == "step_start" and str(d.get("robot", "")).startswith("robot")
              and (args.robot is None or d["robot"] == args.robot)]
    starts.sort(key=lambda d: float(d["t"]))
    if args.max_steps: starts = starts[:args.max_steps]
    times = {r: thermal_times(rec / r) for r in {d["robot"] for d in starts}}
    backend = make_backend()
    agent = RobotAgent(backend, out / "agent_trace.jsonl")
    results = []; refusals = collections.Counter(); orders = collections.Counter(); tools = collections.Counter()
    lat = []; rounds = []; skipped = collections.Counter(); spend = 0.0
    t0 = time.perf_counter()
    for i, d in enumerate(starts):
        ctx = d["context"]; robot = d["robot"]; t = float(d["t"])
        req = {k: ctx[k] for k in CONTEXT_KEYS if k in ctx}
        req.update({"robot": robot, "t": t, "robot_dir": str(rec / robot), "debug_dir": str(out / "agent_debug"),
                    "frame_times": [s for s in times[robot] if s <= t][-45:]})
        try:
            res = agent.step(req)
        except Exception as e:                      # a frame missing from the recording, or the model down
            res = {"error": repr(e), "orders": [], "refusals": [], "rounds": 0, "latency_s": 0.0}
        res["event"] = (ctx.get("event") or {}).get("kind"); res["robot"] = robot; res["t"] = t
        results.append(res)
        if res.get("radio_text_only"): skipped["radio text only"] += 1
        elif res.get("repeat_of_recent_event"): skipped["repeat of recent event"] += 1
        else:
            lat.append(res.get("latency_s", 0)); rounds.append(res.get("rounds", 0))
        for o in res.get("orders") or []: orders[o.get("type")] += 1
        for r in res.get("refusals") or []: refusals[(r.get("tool"), str(r.get("reason", ""))[:70])] += 1
        usage = getattr(backend, "usage", None)
        if usage:
            spend = sum(float(u.get("cost") or 0) for u in usage)
            if spend > args.budget_usd:
                print(f"stopping: OpenRouter spend {spend:.2f} USD over the budget", file=sys.stderr); break
        print(f"[{i + 1}/{len(starts)}] {robot}@{t:.1f} {res['event']}: orders={[o.get('type') for o in res.get('orders') or []]} "
              f"refusals={len(res.get('refusals') or [])} rounds={res.get('rounds')} lat={res.get('latency_s', 0):.1f}s", flush=True)
    for line in (out / "agent_trace.jsonl").read_text().splitlines():
        rec_ = json.loads(line)
        if rec_["kind"] == "tool": tools[rec_["name"]] += 1
    summary = {"recording": str(rec), "backend": getattr(backend, "model", "?"), "steps_offered": len(starts),
               "steps_run": len(lat), "steps_skipped": dict(skipped), "orders": dict(orders), "tools": dict(tools),
               "refusals": {f"{k[0]}: {k[1]}": v for k, v in refusals.most_common()},
               "latency_median_s": round(statistics.median(lat), 1) if lat else None, "latency_sum_s": round(sum(lat)),
               "rounds_mean": round(statistics.mean(rounds), 2) if rounds else None, "wall_s": round(time.perf_counter() - t0),
               "openrouter_spend_usd": round(spend, 3), "recorded": recorded_summary(trace)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "steps.json").write_text(json.dumps(results, indent=1, default=str))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
