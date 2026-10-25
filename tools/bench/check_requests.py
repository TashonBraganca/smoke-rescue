#!/usr/bin/env python3
"""Check a make_requests.py folder: every variant differs from full in exactly its one intended thing; seeds differ.

    brain/.venv/bin/python tools/bench/check_requests.py data/runs/bench_25oct
Exits 1 on any unexpected difference.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_WORLD = {"full": set(), "no_radio": {"radio.range_m"}, "one_robot": {"robots"}, "baseline": set()}


def flat(x, prefix=""):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            out.update(flat(v, f"{prefix}{k}."))
        return out
    return {prefix[:-1]: json.dumps(x, sort_keys=True)}


def diff(a, b):
    fa, fb = flat(a), flat(b)
    return sorted(k for k in fa.keys() | fb.keys() if fa.get(k) != fb.get(k))


def main():
    out = Path(sys.argv[1])
    out = out if out.is_absolute() else ROOT / out
    man = json.loads((out / "manifest.json").read_text())
    by = {(r["house"], r["seed_index"], r["variant"]): r for r in man["runs"]}
    bad, lines = 0, []
    for (house, k, variant), r in sorted(by.items()):
        full = by.get((house, k, "full"))
        if full is None:
            continue
        wf, wv = (json.loads((ROOT / x["world"]).read_text()) for x in (full, r))
        qf, qv = (json.loads((out / "requests" / f"{x['run_id']}.json").read_text())["runs"][0] for x in (full, r))
        wd = [d for d in diff(wf, wv) if d != "bench.variant"]
        qd = diff(qf, qv)
        qd_settings = sorted(d for d in qd if d.startswith("settings."))
        qd_other = sorted(d for d in qd if not d.startswith("settings.") and d not in ("world", "out"))
        ok_world = set(wd) == EXPECTED_WORLD[variant]
        ok_req = not qd_other and (bool(qd_settings) == (variant == "baseline"))
        if variant == "one_robot":
            ok_world &= [x["id"] for x in wv["robots"]] == ["robot_A"] and len(wf["robots"]) == 2
        if variant == "no_radio":
            ok_world &= wv["radio"]["range_m"] == 0 and wf["radio"]["range_m"] > 0
        if not (ok_world and ok_req):
            bad += 1
        if variant != "full":
            lines.append(f"{'OK ' if ok_world and ok_req else 'BAD'} {house} s{k} {variant:9s} world diff {wd or '[]'}  request diff {qd_settings or '[]'}"
                         + (f" other {qd_other}" if qd_other else ""))
    # Seeds of one house must differ in people and fire origin, and only there.
    for house in sorted({r["house"] for r in man["runs"]}):
        fulls = sorted((r for r in man["runs"] if r["house"] == house and r["variant"] == "full"), key=lambda r: r["seed_index"])
        for a in fulls[1:]:
            wa, w0 = (json.loads((ROOT / x["world"]).read_text()) for x in (a, fulls[0]))
            d = diff(w0, wa)
            fire_moved = a["fire_origin"] != fulls[0]["fire_origin"]
            ok = d and set(d) <= {"people", "fire.config", "fire.seed", "bench.seed_index", "bench.people_seed"} and "people" in d
            bad += not ok
            lines.append(f"{'OK ' if ok else 'BAD'} {house} s0 vs s{a['seed_index']} world diff {d}; fire origin moved {fire_moved}")
    print("\n".join(lines))
    print(f"{len(man['runs'])} runs checked, {bad} unexpected")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
