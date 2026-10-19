#!/bin/sh
# Opens the demo replay in the Rerun viewer.
#   tools/demo.sh                 haze run (tomorrow's demo)
#   tools/demo.sh unity_clear
#   tools/demo.sh tum_fr3_long_office_2halves
RUN="${1:-unity_haze}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RRD="$ROOT/data/runs/$RUN/replay.rrd"
if [ ! -f "$RRD" ]; then
  echo "No replay at $RRD. Build it with: cd brain && uv run python -m brain.pipeline ../data/recordings/$RUN" >&2
  exit 1
fi
cat "$ROOT/data/runs/$RUN/report.md"
cd "$ROOT/brain" && exec uv run rerun "$RRD"
