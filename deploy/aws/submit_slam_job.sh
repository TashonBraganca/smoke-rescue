#!/usr/bin/env bash
# Upload one robot recording to S3, run ORB-SLAM3 (OpenCV 5) on AWS Batch, wait, and download the result.
#   ./deploy/aws/submit_slam_job.sh data/recordings/unity_clear_seed7/robot_A my-run-001
# The oracle/ folder is never uploaded by this script.
set -euo pipefail

REC="${1:?usage: submit_slam_job.sh <robot_recording_dir> <run_id>}"
RUN_ID="${2:?run id required}"
PROJECT="${PROJECT_NAME:-smoke-rescue}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
export AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION"
[ -f "$REC/camera.yaml" ] && [ -f "$REC/associations.txt" ] || { echo "not a robot recording folder: $REC" >&2; exit 1; }

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="$PROJECT-runs-$ACCOUNT-$REGION"
NAME="$(basename "$REC")"
IN="s3://$BUCKET/recordings/$RUN_ID/$NAME"
OUT="s3://$BUCKET/runs/$RUN_ID/$NAME"

aws s3 sync "$REC" "$IN" --exclude "oracle/*" --only-show-errors
JOB="$(aws batch submit-job --job-name "slam-$RUN_ID-$NAME" --job-queue "$PROJECT-queue" \
  --job-definition "$PROJECT-orbslam3-cv5" \
  --parameters "input_s3_uri=$IN,output_s3_uri=$OUT" --query jobId --output text)"
echo "submitted job $JOB"
while :; do
  status="$(aws batch describe-jobs --jobs "$JOB" --query 'jobs[0].status' --output text)"
  echo "status: $status"
  case "$status" in SUCCEEDED) break ;; FAILED) echo "job failed; logs in CloudWatch group /aws/batch/$PROJECT" >&2; exit 1 ;; esac
  sleep 20
done
mkdir -p "data/runs/aws/$RUN_ID/$NAME"
aws s3 sync "$OUT" "data/runs/aws/$RUN_ID/$NAME" --only-show-errors
echo "results in data/runs/aws/$RUN_ID/$NAME"
