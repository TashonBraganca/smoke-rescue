#!/usr/bin/env bash
# AWS Batch entrypoint: score SLAM outputs against the recording's oracle (ground truth stays in this job).
# Usage: pipeline_entrypoint.sh s3://bucket/recordings/<id> s3://bucket/runs/<id>
# The run folder holds robot_*/ SLAM outputs and receives the evaluator report files.
set -euo pipefail

rec_uri="${1:?recording s3 uri required}"
run_uri="${2:?run s3 uri required}"
bucket="${RUNS_BUCKET:?RUNS_BUCKET not set}"

case "$rec_uri" in "s3://${bucket}/recordings/"*) ;; *) echo "recording must be under s3://${bucket}/recordings/" >&2; exit 64 ;; esac
case "$run_uri" in "s3://${bucket}/runs/"*) ;; *) echo "run must be under s3://${bucket}/runs/" >&2; exit 64 ;; esac

aws s3 sync "$rec_uri" /work/recording --only-show-errors
aws s3 sync "$run_uri" /work/run --only-show-errors
[ -d /work/recording/oracle ] || { echo "recording has no oracle/ folder" >&2; exit 65; }

status=0
python -m brain.evaluator /work/recording /work/run || status=$?
echo "[pipeline] evaluator exit code: $status"

aws s3 sync /work/run "$run_uri" --only-show-errors
exit "$status"
