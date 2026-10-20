#!/bin/bash
# Download Unity Labs CC0 flipbook zips into one folder per asset.
BASE=/Volumes/Tashon/opencv/smoke-rescue/assets_cache/vfx_candidates
LIST=/tmp/scratch
for u in $(cat $LIST); do
  n=$(basename "$u" -flipbooks.zip)
  d="$BASE/unitylabs_$n"
  [ -d "$d" ] && continue
  mkdir -p "$d"
  curl -sL -m 300 -o "$d/$n-flipbooks.zip" "$u" && unzip -q -o "$d/$n-flipbooks.zip" -d "$d/extracted" && echo "ok $n"
done
