#!/usr/bin/env bash
# "Run your own rescue" (tools/ryor/CONTRACT.txt): stack smoke-rescue-ryor (runs bucket, engine ECR repo, Fargate ARM64
# task definition, DynamoDB, failure alarm), then the site Lambda that carries the mission API. Re-running is safe.
#   IMAGE_TAG=abc123 ./deploy/aws/deploy_ryor.sh      # the engine image pushed to smoke-rescue/ryor-engine
#   STANDIN=1 ./deploy/aws/deploy_ryor.sh             # test stand-in: aws-cli image running ryor_standin.sh
# Site Lambda switches pass through: RYOR_ENABLED, RYOR_CAP, RYOR_IP_LIMIT (see deploy_site_lambda.sh).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGION="${AWS_REGION:-us-east-1}"
if [ "${STANDIN:-0}" = 1 ]; then
  IMG="public.ecr.aws/aws-cli/aws-cli:${STANDIN_TAG:-2.17.0}"
  SCRIPT="$(grep -v '^#' "$HERE/ryor_standin.sh")"
else
  IMG=""; SCRIPT=""
fi
aws cloudformation deploy --region "$REGION" --stack-name smoke-rescue-ryor --template-file "$HERE/ryor.yaml" \
  --capabilities CAPABILITY_IAM --tags Project=smoke-rescue Environment=demo --no-fail-on-empty-changeset \
  --parameter-overrides ImageTag="${IMAGE_TAG:-latest}" ImageUri="$IMG" StandInScript="$SCRIPT" \
    StandInSleep="${STANDIN_SLEEP:-4}" Cpu="${CPU:-2048}" MemoryMb="${MEMORY_MB:-4096}"
# Two vCPUs let two live missions fit beside the inherited five four-vCPU precompute tasks under this account's quota.
aws cloudformation describe-stacks --region "$REGION" --stack-name smoke-rescue-ryor --query 'Stacks[0].Outputs' --output text
SITE_SYNC="${SITE_SYNC:-0}" "$HERE/deploy_site_lambda.sh"
