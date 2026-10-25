#!/usr/bin/env python3
"""Score a benchmark folder made by make_requests.py and run by run_all.sh. Evaluator side only: reads trace.jsonl, result.json
and the world; no robot reads any of this.

    brain/.venv/bin/python tools/bench/score_bench.py data/runs/bench_25oct
Writes <bench>/score.txt and <bench>/score.json and prints the text.

Per run:
  rooms entered / total: evaluator rooms that contain at least one pose of any robot (as tools/score_ranches.py).
  usable floor coverage %: 0.1 m evaluator samples inside rooms, off furniture 0.15 m or taller and 0.125 m clear of full-height
      walls (the score_ranches.py usable floor), searched in the union of the robots' own_searched masks in result.json.
  people confirmed: people with a survivor_confirmed event within 1.5 m. False confirmations: survivor_confirmed events farther
      than 1.5 m from every person. Time to each confirmed person: first matching event time.
  T90: result.json holds only the final searched masks, so coverage over time is not recorded. T90 uses a swept-floor proxy:
      the usable samples within 2.0 m of a robot pose and in the same evaluator room as that pose, accumulated over time
      (poses every 1 s). T90 = first time the team's swept proxy reaches 90 % of the house's ceiling, where the ceiling is the
      largest final swept proxy among that house's full-variant runs (all seeds). Not reached: counted as duration_s (censored)
      in the means; the count of runs that reached it is reported.
  overlap %: usable samples in both robots' own_searched / usable samples in either (2-robot variants; one_robot has none).
  all robots out: every robot ended outside every room in state complete having entered a room (score_ranches.py walked_out).
Seen vs unseen: a seeded house is seen if its seed number (5001..5028) appears in any string of config/brain.json or
config/perception.json, or in any data/evidence/*.txt; ranches are unseen. Every match is listed so the list can be audited.
Aggregates per variant and split: mean with a nested bootstrap 95 % CI (10,000 resamples: houses with replacement, then seeds
within each house with replacement), and paired full minus variant per (house, seed) with the same bootstrap and a Wilcoxon
signed-rank p over per-house mean differences (houses are the independent units).
"""
import base64
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MATCH_M, SWEEP_M, CELL = 1.5, 2.0, 0.1
SEED_RE = re.compile(r"(?<!\d)50(0[1-9]|1\d|2[0-8])(?!\d)")
METRICS = [("rooms_entered_pct", "rooms %"), ("usable_coverage_pct", "cover %"), ("people_found_pct", "people %"),
           ("false_confirmations", "false"), ("mean_time_to_confirm_s", "t_conf s"), ("t90_s", "T90 s"),
           ("overlap_pct", "overlap %"), ("all_out", "all out")]


def inside_np(poly, x, y):
    hit = np.zeros(x.shape, bool)
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[i - 1]
        if y1 == y2:
            continue
        cross = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
        hit ^= cross
    return hit


def seg_dist(x, y, s):
    ax, ay, bx, by = s[:4]
    dx, dy = bx - ax, by - ay
    t = np.clip(((x - ax) * dx + (y - ay) * dy) / max(1e-12, dx * dx + dy * dy), 0, 1)
    return np.hypot(x - (ax + t * dx), y - (ay + t * dy))


_GRIDS = {}


def floor_grid(world):
    key = hashlib.sha1(json.dumps([world["rooms"], world["segments"], world["prisms"]]).encode()).hexdigest()
    if key in _GRIDS:
        return _GRIDS[key]
    xs, ys, room_of = [], [], []
    for ri, room in enumerate(world["rooms"]):
        px = [p[0] for p in room["poly"]]; py = [p[1] for p in room["poly"]]
        ix = np.arange(math.floor(min(px) * 10), math.ceil(max(px) * 10)); iy = np.arange(math.floor(min(py) * 10), math.ceil(max(py) * 10))
        gx, gy = np.meshgrid((ix + .5) / 10, (iy + .5) / 10)
        gx, gy = gx.ravel(), gy.ravel()
        m = inside_np(room["poly"], gx, gy)
        xs.append(gx[m]); ys.append(gy[m]); room_of.append(np.full(m.sum(), ri))
    x, y, room_of = np.concatenate(xs), np.concatenate(ys), np.concatenate(room_of)
    usable = np.ones(x.shape, bool)
    for q in world["prisms"]:
        if q["height_m"] >= .15:
            usable &= ~inside_np(q["poly"], x, y)
    for s in world["segments"]:
        if s[4] <= .2 and s[5] >= 1.5:
            usable &= seg_dist(x, y, s) >= .125
    _GRIDS[key] = (x[usable], y[usable], room_of[usable])
    return _GRIDS[key]


def room_index(world, px, py):
    out = np.full(px.shape, -1)
    for ri, room in enumerate(world["rooms"]):
        out[(out < 0) & inside_np(room["poly"], px, py)] = ri
    return out


def mask_hits(m, x, y):
    n, c, (ox, oy) = m["grid_size"], m["cell_m"], m["origin"]
    bits = np.frombuffer(base64.b64decode(m["own_searched"]), np.uint8)
    cx, cy = np.floor((x - ox) / c).astype(int), np.floor((y - oy) / c).astype(int)
    ok = (cx >= 0) & (cx < n) & (cy >= 0) & (cy < n)
    hit = np.zeros(x.shape, bool)
    hit[ok] = bits[cy[ok] * n + cx[ok]] != 0
    return hit


def score_run(entry):
    run = ROOT / entry["out"]
    result = json.loads((run / "result.json").read_text())
    world = json.loads((ROOT / entry["world"]).read_text())
    rows = [json.loads(l) for l in (run / "trace.jsonl").read_text().splitlines() if l.strip()]
    poses = [r for r in rows if r["kind"] == "pose"]
    x, y, room_of = floor_grid(world)
    px = np.array([p["x"] for p in poses]); py = np.array([p["y"] for p in poses])
    pr = room_index(world, px, py) if poses else np.array([], int)
    entered = set(pr[pr >= 0].tolist())
    hits = [mask_hits(m, x, y) for m in result["maps"]]
    union = np.logical_or.reduce(hits) if hits else np.zeros(x.shape, bool)
    overlap = None
    if len(hits) == 2:
        overlap = round(100 * float((hits[0] & hits[1]).sum()) / max(1, int(union.sum())), 2)
    confirmed = [r for r in rows if r["kind"] == "survivor_confirmed" and "x_m" in r.get("numbers", {})]
    found, false = {}, 0
    for e in confirmed:
        q = (e["numbers"]["x_m"], e["numbers"]["y_m"])
        near = [p for p in world["people"] if math.dist(q, p["at"]) <= MATCH_M]
        if not near:
            false += 1
        for p in near:
            found.setdefault(p["id"], e["t"])
    # Swept-floor proxy over time for T90 (see module docstring).
    swept = np.zeros(x.shape, bool); series, next_t = [], 0.0
    for p, ri in zip(poses, pr):
        if p["t"] + 1e-9 < next_t:
            continue
        next_t = p["t"] + 1.0
        if ri >= 0:
            swept |= (room_of == ri) & (np.hypot(x - p["x"], y - p["y"]) <= SWEEP_M)
        series.append((p["t"], 100 * float(swept.sum()) / max(1, len(x))))
    robots = {}
    for rid in sorted({p["robot"] for p in poses}):
        idx = [i for i, p in enumerate(poses) if p["robot"] == rid]
        last = poses[idx[-1]]; visited = {int(pr[i]) for i in idx if pr[i] >= 0}
        outside = pr[idx[-1]] < 0
        robots[rid] = {"rooms_entered": len(visited), "final_state": last["state"], "outside_at_end": bool(outside),
                       "walked_out": bool(visited) and bool(outside) and last["state"] == "complete"}
    n_people = len(world["people"])
    return {"run_id": entry["run_id"], "house": entry["house"], "seed_index": entry["seed_index"], "variant": entry["variant"],
            "duration_s": result["duration_s"], "settings": result.get("settings", {}), "robots_in_result": [m["robot"] for m in result["maps"]],
            "rooms_entered": len(entered), "rooms_total": len(world["rooms"]),
            "rooms_entered_pct": round(100 * len(entered) / max(1, len(world["rooms"])), 2),
            "usable_coverage_pct": round(100 * float(union.sum()) / max(1, len(x)), 2), "usable_samples": int(len(x)),
            "people_found": len(found), "people_total": n_people, "people_found_pct": round(100 * len(found) / max(1, n_people), 2),
            "time_to_person_s": {k: round(v, 2) for k, v in sorted(found.items())},
            "mean_time_to_confirm_s": round(sum(found.values()) / len(found), 2) if found else None,
            "false_confirmations": false, "confirm_events": len(confirmed), "overlap_pct": overlap,
            "all_out": float(bool(robots) and all(r["walked_out"] for r in robots.values())), "robots": robots,
            "swept_final_pct": round(series[-1][1], 2) if series else 0.0, "_series": series}


def seen_seeds():
    matches = []
    for name in ("config/brain.json", "config/perception.json"):
        def walk(v, key):
            if isinstance(v, dict):
                for k, w in v.items():
                    walk(w, k)
            elif isinstance(v, list):
                for w in v:
                    walk(w, key)
            elif isinstance(v, str):
                for m in SEED_RE.finditer(v):
                    matches.append((int(m.group(0)), f"{name}:{key}", v[max(0, m.start() - 50):m.end() + 30].replace("\n", " ")))
        walk(json.loads((ROOT / name).read_text()), "")
    for path in sorted((ROOT / "data/evidence").glob("*.txt")):
        for ln, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            for m in SEED_RE.finditer(line):
                matches.append((int(m.group(0)), f"{path.relative_to(ROOT)}:{ln}", line[max(0, m.start() - 50):m.end() + 30]))
    return sorted({s for s, _, _ in matches}), matches


def nested_boot(per_house, rng, n=10000):
    """per_house: list of 1-D arrays (seed values, NaN allowed). Returns mean, lo, hi."""
    per_house = [a[~np.isnan(a)] for a in per_house]
    per_house = [a for a in per_house if len(a)]
    if not per_house:
        return None, None, None
    h = len(per_house); s = max(len(a) for a in per_house)
    mat = np.full((h, s), np.nan); cnt = np.array([len(a) for a in per_house])
    for i, a in enumerate(per_house):
        mat[i, :len(a)] = a
    mean = float(np.mean([a.mean() for a in per_house]))   # houses weigh equally, as in the resampling
    hi_idx = rng.integers(0, h, (n, h))
    si_idx = np.floor(rng.random((n, h, s)) * cnt[hi_idx][:, :, None]).astype(int)
    draws = mat[hi_idx[:, :, None], si_idx]
    means = draws.reshape(n, -1).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return round(mean, 2), round(float(lo), 2), round(float(hi), 2)


def wilcoxon_p(per_house):
    d = np.array([np.nanmean(a) for a in per_house if np.any(~np.isnan(a))])
    d = d[d != 0]
    if len(d) < 1:
        return None
    from scipy.stats import wilcoxon
    try:
        return round(float(wilcoxon(d).pvalue), 4)
    except ValueError:
        return None


def main():
    bench = Path(sys.argv[1]); bench = bench if bench.is_absolute() else ROOT / bench
    if not (bench / "manifest.json").exists() and (bench.parent / "manifest.json").exists():
        bench = bench.parent
    man = json.loads((bench / "manifest.json").read_text())
    seeds, matches = seen_seeds()
    seen_houses = {f"random_{s}" for s in seeds}
    runs, missing = [], []
    for e in man["runs"]:
        if (ROOT / e["out"] / "result.json").exists() and (ROOT / e["out"] / "trace.jsonl").exists():
            runs.append(score_run(e))
        else:
            missing.append(e["run_id"])
    ceilings = {}
    for r in runs:
        if r["variant"] == "full":
            ceilings[r["house"]] = max(ceilings.get(r["house"], 0.0), r["swept_final_pct"])
    for r in runs:
        c = ceilings.get(r["house"])
        r["t90_ceiling_pct"] = c
        t = next((t for t, v in r["_series"] if c and v >= .9 * c), None) if c else None
        r["t90_reached"] = t is not None
        r["t90_s"] = round(t, 2) if t is not None else (r["duration_s"] if c else None)
        r["split"] = "seen" if r["house"] in seen_houses else "unseen"
        del r["_series"]
    rng = np.random.default_rng(20261025)
    variants = man["variants"]
    agg, lines = {}, []
    lines.append(f"Benchmark {bench.relative_to(ROOT)}: {len(runs)} runs scored, {len(missing)} missing" + (f" ({', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''})" if missing else ""))
    lines.append(f"Seen seeds (grep of config/brain.json, config/perception.json strings and data/evidence/*.txt): {seeds}")
    lines.append("")
    for split in ("unseen", "seen", "all"):
        sub = [r for r in runs if split == "all" or r["split"] == split]
        if not sub:
            continue
        houses = sorted({r["house"] for r in sub})
        lines.append(f"== {split}: {len(houses)} houses ({', '.join(houses)}) ==")
        lines.append(f"{'variant':10s} {'n':>3s} " + " ".join(f"{lab:>22s}" for _, lab in METRICS))
        agg[split] = {}
        for v in variants:
            vr = [r for r in sub if r["variant"] == v]
            row = {"n_runs": len(vr), "t90_reached": sum(r["t90_reached"] for r in vr)}
            for key, _ in METRICS:
                per_house = [np.array([np.nan if r[key] is None else float(r[key]) for r in vr if r["house"] == h]) for h in houses]
                row[key] = nested_boot(per_house, rng)
            agg[split][v] = row
            cells = [("-" if row[k][0] is None else f"{row[k][0]:.1f} [{row[k][1]:.1f},{row[k][2]:.1f}]") for k, _ in METRICS]
            lines.append(f"{v:10s} {len(vr):3d} " + " ".join(f"{c:>22s}" for c in cells))
        lines.append("paired full minus variant: mean [95 % CI] p(Wilcoxon, per-house means)")
        for v in variants:
            if v == "full":
                continue
            row = {}
            for key, _ in METRICS:
                per_house = []
                for h in houses:
                    ds = []
                    for r in sub:
                        if r["house"] == h and r["variant"] == v:
                            f = next((q for q in sub if q["house"] == h and q["seed_index"] == r["seed_index"] and q["variant"] == "full"), None)
                            ds.append(np.nan if f is None or f[key] is None or r[key] is None else float(f[key]) - float(r[key]))
                    per_house.append(np.array(ds))
                m = nested_boot(per_house, rng)
                row[key] = {"mean": m[0], "ci95": [m[1], m[2]], "wilcoxon_p": wilcoxon_p(per_house)}
            agg[split]["full_minus_" + v] = row
            cells = [("-" if row[k]["mean"] is None else f"{row[k]['mean']:+.1f} [{row[k]['ci95'][0]:+.1f},{row[k]['ci95'][1]:+.1f}] p{row[k]['wilcoxon_p'] if row[k]['wilcoxon_p'] is not None else '-'}") for k, _ in METRICS]
            lines.append(f"-{v:9s}     " + " ".join(f"{c:>22s}" for c in cells))
        lines.append("")
    lines.append("Per run: house seed variant | rooms in/total | cover % | people found/total | false | T90 s (reached) | overlap % | all out | robots in result")
    for r in sorted(runs, key=lambda r: (r["split"] != "unseen", r["house"], r["seed_index"], variants.index(r["variant"]))):
        lines.append(f"{r['split']:6s} {r['house']:13s} s{r['seed_index']} {r['variant']:9s} | {r['rooms_entered']}/{r['rooms_total']} | {r['usable_coverage_pct']:.1f} | "
                     f"{r['people_found']}/{r['people_total']} {r['time_to_person_s']} | {r['false_confirmations']} | {r['t90_s']} ({'yes' if r['t90_reached'] else 'no'}) | "
                     f"{r['overlap_pct']} | {int(r['all_out'])} | {r['robots_in_result']}")
    lines.append("")
    lines.append(f"Seen-seed matches ({len(matches)}), seed, source, context:")
    for s, src, ctx in matches:
        lines.append(f"  {s} {src}: ...{ctx}...")
    text = "\n".join(lines) + "\n"
    (bench / "score.txt").write_text(text)
    (bench / "score.json").write_text(json.dumps({"seen_seeds": seeds, "seen_matches": [{"seed": s, "source": src, "context": c} for s, src, c in matches],
                                                  "missing": missing, "aggregate": agg, "runs": runs,
                                                  "definitions": __doc__}, indent=1) + "\n")
    print(text)


if __name__ == "__main__":
    main()
