#!/usr/bin/env bash
# AWS Batch entrypoint: s3 recording folder in, ORB-SLAM3 RGB-D run, s3 output folder out.
# Usage: slam_entrypoint.sh s3://bucket/recordings/<id>/robot_A s3://bucket/runs/<run>/robot_A
set -euo pipefail

in_uri="${1:?input s3 uri required}"
out_uri="${2:?output s3 uri required}"
bucket="${RUNS_BUCKET:?RUNS_BUCKET not set}"

case "$in_uri" in "s3://${bucket}/recordings/"*) ;; *) echo "input must be under s3://${bucket}/recordings/" >&2; exit 64 ;; esac
case "$out_uri" in "s3://${bucket}/runs/"*) ;; *) echo "output must be under s3://${bucket}/runs/" >&2; exit 64 ;; esac

rec=/work/recording
out=/work/output
mkdir -p "$rec" "$out"

echo "[slam] sync in: $in_uri"
aws s3 sync "$in_uri" "$rec" --only-show-errors --exclude "oracle/*"
for f in camera.yaml associations.txt; do
  [ -s "$rec/$f" ] || { echo "missing $f in recording" >&2; exit 65; }
done
[ -d "$rec/rgb" ] && [ -d "$rec/depth" ] || { echo "missing rgb/ or depth/" >&2; exit 65; }
echo "[slam] frames: $(find "$rec/rgb" -type f | wc -l)"

status=0
/usr/local/bin/smoke_rescue_rgbd "$rec" "$out" rgbd || status=$?
echo "[slam] runner exit code: $status"

echo "[slam] sync out: $out_uri"
aws s3 sync "$out" "$out_uri" --only-show-errors
exit "$status"
