#!/usr/bin/env python3
"""Agent behaviour across every recorded agent run: OpenCV tool calls, actions and the evidence they cite, guardrail
refusals by reason, and chains where a refused action was followed by a different action citing new OpenCV evidence.

    python3 tools/agent_trace_stats.py [data/recordings] [--out data/evidence/agent_trace_stats.txt]

Reads only agent_trace.jsonl (the agent layer's own log), never ground truth.
"""
import argparse
import collections
import json
import re
from pathlib import Path

PERCEPTION = {"scan_thermal", "inspect_candidate", "check_indoors", "assess_visibility", "track_fire_spread", "measure_opening"}


def load(path):
    rows = []
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root", nargs="?", type=Path, default=Path("data/recordings"))
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    traces = sorted(a.root.glob("*/agent_trace.jsonl"))
    calls = collections.Counter(); refused = collections.Counter(); reasons = collections.Counter()
    actions = cited = steps = 0; produced = set(); cited_ids = set(); chains = []
    for path in traces:
        rows = load(path)
        steps += sum(1 for r in rows if r.get("kind") == "step_start")
        last_refusal = {}
        for r in rows:
            if r.get("kind") != "tool":
                continue
            name = r.get("name"); calls[name] += 1
            robot = str(r.get("step", "")).split("@")[0]
            if name in PERCEPTION and r.get("result_id"):
                produced.add(r["result_id"])
            ev = (r.get("arguments") or {}).get("evidence_id")
            if name not in PERCEPTION and name not in {"finish", "send_team_message", "broadcast", "propose_plan"}:
                actions += 1
                if ev:
                    cited += 1; cited_ids.add(ev)
            if r.get("refused"):
                refused[name] += 1
                # Group reasons by their wording, without the numbers.
                reasons[re.sub(r"[-0-9.]+", "#", (r.get("reason") or "")[:70])] += 1
                last_refusal[robot] = r
            elif robot in last_refusal and name != "finish" and ev and ev != (last_refusal[robot].get("arguments") or {}).get("evidence_id"):
                f = last_refusal.pop(robot)
                if len(chains) < 6:
                    chains.append(f"{path.parent.name}: {f['step']} {f['name']} refused ({(f.get('reason') or '')[:90]}) -> "
                                  f"{r['step']} {name} citing new evidence {ev}")
    total = sum(calls.values()); perception = sum(calls[n] for n in PERCEPTION)
    lines = [f"Agent runs: {len(traces)}, agent steps: {steps}, tool calls: {total}",
             f"OpenCV perception calls: {perception} ({', '.join(f'{n} {calls[n]}' for n in sorted(PERCEPTION))})",
             f"Actions (orders to the robot): {actions}; citing an evidence id: {cited} ({100 * cited / max(1, actions):.1f} %)",
             f"Cited evidence ids that an OpenCV tool produced in a recorded call: {len(cited_ids & produced)} of {len(cited_ids)}",
             f"Guardrail refusals: {sum(refused.values())} ({', '.join(f'{k} {v}' for k, v in refused.most_common())})",
             "Refusal reasons (numbers masked):"] + [f"  {v:3d}  {k}" for k, v in reasons.most_common(10)] + \
            ["Refusal followed by a different action on new OpenCV evidence (examples):"] + [f"  {c}" for c in chains]
    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
