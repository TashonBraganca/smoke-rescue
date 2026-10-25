#!/bin/sh
# Run every benchmark request in a folder with the already-built headless runner, N at a time.
#   tools/bench/run_all.sh <requests_dir> <parallel>
# Env: DOTNET (default `dotnet` on PATH), HEADLESS_DLL (default tools/headless/bin/HeadlessRunner.dll), NICE (default 10).
# A run whose result.json exists is skipped. A run with a running.marker but no result.json is skipped by the simulator
# itself and reported here as "stale marker" (delete that marker by hand once no runner is alive). Failures go to
# <requests_dir>/../logs/failures.txt; each run's output goes to <requests_dir>/../logs/<run_id>.log. POSIX sh, Linux and macOS.
set -u
here=$(cd "$(dirname "$0")" && pwd)
SMOKE_REPO=$(cd "$here/../.." && pwd); export SMOKE_REPO
if [ "${1:-}" = "--one" ]; then mode=one; req_dir=$(cd "$(dirname "$2")" && pwd); par=1; else
  mode=all; [ $# -eq 2 ] || { echo "usage: $0 <requests_dir> <parallel>" >&2; exit 2; }
  req_dir=$(cd "$1" && pwd); par=$2; fi
dll=${HEADLESS_DLL:-$SMOKE_REPO/tools/headless/bin/HeadlessRunner.dll}
dotnet=${DOTNET:-dotnet}
nice_n=${NICE:-10}
export DOTNET_CLI_HOME=${DOTNET_CLI_HOME:-${TMPDIR:-/tmp}/headless-dotnet-home} DOTNET_NOLOGO=1 DOTNET_CLI_TELEMETRY_OPTOUT=1
[ -f "$dll" ] || { echo "no runner at $dll: build it with tools/headless/run.sh or dotnet build tools/headless/HeadlessRunner.csproj -c Release -o tools/headless/bin" >&2; exit 2; }
command -v "$dotnet" >/dev/null 2>&1 || [ -x "$dotnet" ] || { echo "dotnet not found: set DOTNET" >&2; exit 2; }
log_dir=$(dirname "$req_dir")/logs; mkdir -p "$log_dir"
[ "$mode" = all ] && echo "runner $dll ($(ls -l "$dll" | awk '{print $6, $7, $8}'))"
[ "$mode" = all ] && echo "dotnet $dotnet; repo $SMOKE_REPO; $par in parallel; logs $log_dir"

one() {
  req=$1; id=$(basename "$req" .json)
  out=$(sed -n 's/^ *"out": "\(.*\)",*$/\1/p' "$req" | head -1)
  [ -n "$out" ] || { echo "$id: no out path" >> "$log_dir/failures.txt"; echo "$id FAILED (no out)"; return 0; }
  if [ -f "$SMOKE_REPO/$out/result.json" ]; then echo "$id skipped (result.json exists)"; return 0; fi
  if [ -f "$SMOKE_REPO/$out/running.marker" ]; then echo "$id stale marker $out/running.marker" | tee -a "$log_dir/failures.txt"; return 0; fi
  start=$(date +%s)
  if nice -n "$nice_n" "$dotnet" "$dll" "$req" > "$log_dir/$id.log" 2>&1 && [ -s "$SMOKE_REPO/$out/result.json" ]; then
    echo "$id ok $(( $(date +%s) - start )) s"
  else
    echo "$id FAILED rc=$? after $(( $(date +%s) - start )) s, see $log_dir/$id.log" | tee -a "$log_dir/failures.txt"
  fi
}
if [ "$mode" = one ]; then one "$2"; exit 0; fi
ls "$req_dir"/*.json | xargs -n 1 -P "$par" "$here/run_all.sh" --one
n_ok=$(ls "$req_dir"/*.json | wc -l | tr -d ' ')
echo "done: $n_ok requests; failures listed in $log_dir/failures.txt (if it exists)"
