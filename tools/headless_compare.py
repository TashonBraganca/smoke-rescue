#!/usr/bin/env python3
"""Compare headless simulator variants across houses (evaluator side).

    python tools/headless_compare.py data/runs/headless/exp1

Run folders are named <seed>_<variant>. Prints one row per variant (mean over houses, and the per-house values), so one
easy house cannot hide a bad one. Writes compare.json beside the runs.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from headless_report import report  # noqa: E402


def summary(r: dict) -> dict:
    rooms = [v["searched_pct"] for v in r["rooms"].values()]
    people = list(r["people"].values())
    robots = list(r["robots"].values())
    return {
        "room_mean_pct": round(sum(rooms) / len(rooms)),
        "rooms_under_50": sum(p < 50 for p in rooms), "rooms": len(rooms),
        "sighted": sum(p["first_sighting_s"] is not None for p in people),
        "confirmed": sum(p["confirmed_s"] is not None for p in people), "people": len(people),
        "false": r["false_confirmations"],
        "walked_m": round(sum(x["walked_m"] for x in robots), 1),
        "turning_s": round(sum(x.get("turning_in_place_s", 0) for x in robots)),
        "still_s": round(sum(x.get("still_s", 0) for x in robots)),
        "outside_s": round(sum(x.get("outside_s", 0) for x in robots)),
        "stuck_events": sum(sum(x["stuck_events"].values()) for x in robots),
        "yields": sum(x["room_yielded"] for x in robots),
        "peak_air_c": max(x.get("peak_air_c", 20) for x in robots), "over_60_s": sum(x.get("air_over_60_s", 0) for x in robots),
    }


def main(folder: Path):
    by_variant = defaultdict(dict)
    for run in sorted(p for p in folder.iterdir() if (p / "result.json").exists()):
        seed, variant = run.name.split("_", 1)
        r = report(run)
        (run / "report.json").write_text(json.dumps(r, indent=1))
        by_variant[variant][seed] = summary(r)
    keys = ["room_mean_pct", "rooms_under_50", "sighted", "confirmed", "false", "walked_m", "turning_s", "still_s", "outside_s", "stuck_events", "yields", "peak_air_c", "over_60_s"]
    print("variant".ljust(22) + "".join(k[:12].rjust(13) for k in keys))
    for variant, seeds in by_variant.items():
        mean = {k: round(sum(s[k] for s in seeds.values()) / len(seeds), 1) for k in keys}
        print(variant.ljust(22) + "".join(str(mean[k]).rjust(13) for k in keys))
        for seed, s in seeds.items():
            print(("  " + seed).ljust(22) + "".join(str(s[k]).rjust(13) for k in keys) + f"   rooms {s['rooms']}, people {s['people']}")
    (folder / "compare.json").write_text(json.dumps(by_variant, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
