#!/bin/sh
# Copy one recording into run_logs/ for replay and analysis, then commit and push it (owner's rule, 23 Oct).
# Kept: robot_A/ and robot_B/ without their rgb/ and depth/ frame folders, all of oracle/, meta.json, the scenario
# json named in meta.json, and every agent trace (agent_trace.jsonl, agent_debug/, radio, timing). PNG and PLY files go through Git LFS (.gitattributes).
#   tools/publish_run.sh data/recordings/<run>
set -e
src=${1%/}; run=$(basename "$src"); repo=$(cd "$(dirname "$0")/.." && pwd); dst="$repo/run_logs/$run"
[ -f "$src/meta.json" ] || { echo "no meta.json in $src"; exit 1; }
scenario=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['scenario'])" "$src/meta.json")
mkdir -p "$dst"
for r in robot_A robot_B; do
  [ -d "$src/$r" ] && rsync -a --delete --exclude rgb/ --exclude depth/ "$src/$r/" "$dst/$r/"
done
rsync -a --delete "$src/oracle/" "$dst/oracle/"
cp "$src/meta.json" "$dst/meta.json"
# Every agent trace too (owner, 23 Oct: "keep a log of all the traces of the agents"): the model's context, replies, tool
# calls and verdicts per step, the inspection crops, the radio and the timing.
for f in agent_trace.jsonl agent_summary.json command_post.jsonl radio.txt brain_radio.txt perf.txt water.jsonl; do
  [ -f "$src/$f" ] && cp "$src/$f" "$dst/$f"
done
[ -d "$src/agent_debug" ] && rsync -a --delete "$src/agent_debug/" "$dst/agent_debug/"
cp "$repo/config/$scenario" "$dst/$scenario"
cd "$repo"
git add "run_logs/$run"
git commit -q -m "run_logs: $run (robot logs without rgb/depth frames, oracle, meta, $scenario)"
git push -q origin main
du -sh "$dst"; git log --oneline -1
