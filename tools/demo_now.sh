#!/bin/sh
# Opens everything for a live demo: the results page, the robots' front-camera video, and the Rerun replay.
#   tools/demo_now.sh                       burning-house run
#   tools/demo_now.sh unity_clear           a specific run
RUN="${1:-unity_haze}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RRD="$ROOT/data/runs/$RUN/replay.rrd"
VIDEO="$ROOT/data/evidence/walkthrough_burning_house_map_3x.mp4"
[ -f "$RRD" ] || { echo "No replay at $RRD" >&2; exit 1; }

cat "$ROOT/data/runs/$RUN/report.md"
open ""
[ -f "$VIDEO" ] && open -a "QuickTime Player" "$VIDEO"
cd "$ROOT/brain" && exec uv run rerun "$RRD"
