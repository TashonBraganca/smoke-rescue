#!/bin/sh
# Score a finished recording against hidden truth (evaluator side): people found, false and carried out, heat
# exposure, agent orders (compare_runs); pose and map accuracy, indoor layer outside the building, links (nav_eval);
# searched floor per room; when each person was visible to a camera within 4 m.
#   tools/scorecard.sh data/recordings/<run>
set -e
run=$(cd "$1" && pwd)
cd "$(dirname "$0")/../brain"
echo "== compare_runs"; uv run python -m brain.compare_runs "$run"
echo "== nav_eval";     uv run python -m brain.nav_eval "$run" | grep -v "frame check"
echo "== searched floor per room"; uv run python ../tools/room_coverage.py "$run"
echo "== people visible to a camera within 4 m"; uv run python ../tools/person_visibility.py "$run" --every 4
echo "== radio to the command post"; python3 -c "
import json,sys
rows=[json.loads(l) for l in open('$run/command_post.jsonl')] if __import__('os').path.exists('$run/command_post.jsonl') else []
print(len(rows), 'command post records;', sum(1 for r in rows if r.get('heard') or r.get('kind')=='heard'), 'heard')"
