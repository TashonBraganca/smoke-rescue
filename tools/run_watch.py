#!/usr/bin/env python3
"""Watch a recording while Unity writes it: every period, log each robot's and each agent's new decisions, the
mistakes worth a look (refused orders, uncertain verdicts, stuck/held/blacklisted, retreats, fire cuts) and, as the
evaluator only, how close the robots have come to each person. Renders the showcase video once meta.json says complete.

    brain/.venv/bin/python tools/run_watch.py data/recordings/<run> data/runs/<watch_dir> [--video out.mp4]
"""
import argparse, json, math, subprocess, sys, time
from collections import Counter
from pathlib import Path

MISTAKE = {"stuck", "collision_hold", "pinned", "goal_blacklisted", "no_progress", "retreat", "retreat_reduced_clearance",
           "escape_closing", "stay_overridden", "no_frontier", "body_swing_blocked"}


def lines(path, start):
    if not path.exists(): return [], start
    with open(path) as f:
        f.seek(start); text = f.read(); end = f.tell()
    return [json.loads(l) for l in text.splitlines() if l.strip().startswith("{")], end


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("rec", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--video", type=Path); ap.add_argument("--period", type=float, default=20)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    log = open(a.out / "watch.log", "a"); offsets = {}; totals = Counter()
    def say(s): log.write(s + "\n"); log.flush()
    while True:
        meta = json.loads((a.rec / "meta.json").read_text()) if (a.rec / "meta.json").exists() else {}
        say(f"==== {time.strftime('%H:%M:%S')} sim {meta.get('sim_seconds_written')} s complete {meta.get('complete')}")
        for rob in ("robot_A", "robot_B"):
            new, offsets[rob] = lines(a.rec / rob / "decisions.jsonl", offsets.get(rob, 0))
            for e in new:
                k = e.get("kind"); totals[(rob, k)] += 1
                if k in MISTAKE: say(f"  MISTAKE {rob} {e['t']:.1f} {k}: {e.get('reason', '')[:110]}")
                elif k in ("survivor_confirmed", "agent_order", "survivor_from_peer", "entrance_locked"):
                    say(f"  {rob} {e['t']:.1f} {k}: {e.get('reason', '')[:110]}")
        new, offsets["agent"] = lines(a.rec / "agent_trace.jsonl", offsets.get("agent", 0))
        for e in new:
            k = e.get("kind")
            if k == "tool":
                totals[("agent", e.get("name"), e.get("verdict"), bool(e.get("refused")))] += 1
                if e.get("refused"): say(f"  AGENT-REFUSED {e.get('step')} {e.get('name')}: {(e.get('reason') or '')[:110]}")
                elif e.get("name") == "inspect_candidate": say(f"  agent {e.get('step')} inspect -> {e.get('verdict')}: {'; '.join(e.get('reasons') or [])[:110]}")
            elif k == "step_end":
                say(f"  agent {e.get('step')} rounds {e.get('rounds')} orders {[o.get('type') for o in e.get('orders', [])]} :: {(e.get('summary') or 'NO SUMMARY')[:120]}")
                if e.get("summary") is None or (e.get("summary") or "").startswith("(no summary"): totals[("agent", "no_summary")] += 1
            elif k in ("model_error", "step_deadline"):
                totals[("agent", k)] += 1; say(f"  AGENT-{k.upper()} {e.get('step')} {e.get('error', '')[:110]}")
        try:   # evaluator only: closest true approach to each person
            surv = json.loads((a.rec / "oracle" / "survivors.json").read_text()); near = {}
            for rob in ("robot_A", "robot_B"):
                for l in open(a.rec / "oracle" / f"{rob}_groundtruth.txt"):
                    if l[0] == "#": continue
                    v = l.split()
                    for p in surv:
                        d = math.dist((float(v[1]), float(v[3])), (p["position_house"][0], p["position_house"][2]))
                        near[p["id"]] = min(near.get(p["id"], 99), d)
            say("  closest approach (m): " + ", ".join(f"{p['id']} {p['room']} {near.get(p['id'], 99):.1f}" for p in surv))
        except Exception as ex: say(f"  (oracle not readable yet: {ex})")
        (a.out / "totals.json").write_text(json.dumps({" | ".join(map(str, k)): v for k, v in totals.items()}, indent=1))
        if meta.get("complete"):
            say("recording complete")
            if a.video:
                r = subprocess.run([sys.executable, "tools/showcase.py", str(a.rec), "--speed", "3", "--out", str(a.video), "--title", "Smoke Rescue: seed 934"], capture_output=True, text=True)
                say("render: " + (r.stdout.strip().splitlines() or ["?"])[-1] + (" ERR " + r.stderr[-300:] if r.returncode else ""))
            return
        time.sleep(a.period)


if __name__ == "__main__":
    main()
