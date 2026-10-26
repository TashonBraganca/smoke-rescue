#!/bin/sh
# Stage the engine context and build the linux/arm64 image: tools/ryor/build_image.sh <tag>
# Tag it with the git commit so cached replays and live runs can be matched to one engine build.
set -e
here=$(cd "$(dirname "$0")" && pwd); root=$(cd "$here/../.." && pwd); tag=${1:?tag}
"$here/build_engine.sh"
ctx=$(mktemp -d "${TMPDIR:-/tmp}/ryor_ctx.XXXXXX"); mkdir -p "$ctx/ctx/tools/ryor" "$ctx/ctx/data/runs"
cp "$here/Dockerfile" "$ctx/"
cp "$here/entry.py" "$here/replay.py" "$here/placement.py" "$here/terra.py" "$here/layout.py" "$ctx/ctx/tools/ryor/"
cp "$root/tools/validate_ranches.py" "$root/tools/validate_building.py" "$ctx/ctx/tools/ryor/"
cp "$root/brain/brain/agent/llm.py" "$ctx/ctx/tools/ryor/agent_backend.py"
cp -R "$here/engine/bin" "$ctx/ctx/tools/ryor/engine_bin"; mkdir -p "$ctx/ctx/tools/ryor/engine"; mv "$ctx/ctx/tools/ryor/engine_bin" "$ctx/ctx/tools/ryor/engine/bin"
cp -R "$root/config" "$ctx/ctx/config"
cp -R "$root/data/runs/ryor" "$ctx/ctx/data/runs/ryor"; rm -rf "$ctx/ctx/data/runs/ryor/local" "$ctx/ctx/data/runs/ryor/cache"
docker build --platform linux/arm64 -t "smoke-rescue/ryor-engine:$tag" --build-arg TAG="$tag" "$ctx"
rm -rf "$ctx"
docker image ls "smoke-rescue/ryor-engine:$tag"
