#!/usr/bin/env bash
# Delete every Smoke Rescue stack in reverse order. Buckets are emptied only with --yes.
#   ./deploy/aws/teardown.sh --yes
set -euo pipefail

PROJECT="${PROJECT_NAME:-smoke-rescue}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
YES=0
[ "${1:-}" = "--yes" ] && YES=1
export AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION"

command -v aws >/dev/null || { echo "aws CLI not found" >&2; exit 1; }
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)" || { echo "no valid AWS credentials" >&2; exit 1; }

if [ "$YES" != 1 ]; then
  echo "Dry run. With --yes this deletes stacks $PROJECT-budget, -storage, -batch, -agent in $REGION (account $ACCOUNT),"
  echo "including ALL objects in $PROJECT-runs-$ACCOUNT-$REGION and $PROJECT-site-$ACCOUNT-$REGION. Re-run with --yes."
  exit 0
fi

empty_bucket() {
  local b="$1" out
  aws s3api head-bucket --bucket "$b" 2>/dev/null || return 0
  echo "[teardown] emptying $b"
  for key in Versions DeleteMarkers; do
    while :; do
      out="$(aws s3api list-object-versions --bucket "$b" --max-items 1000 --output json \
        --query "{Objects: ${key}[].{Key:Key,VersionId:VersionId}}")"
      case "$out" in *'"Key"'*) ;; *) break ;; esac
      aws s3api delete-objects --bucket "$b" --delete "$out" --output text >/dev/null
    done
  done
}

delete_stack() {
  local s="$1"
  if aws cloudformation describe-stacks --stack-name "$s" >/dev/null 2>&1; then
    echo "[teardown] deleting $s"
    aws cloudformation delete-stack --stack-name "$s"
    aws cloudformation wait stack-delete-complete --stack-name "$s"
  fi
}

delete_stack "$PROJECT-budget"
empty_bucket "$PROJECT-site-$ACCOUNT-$REGION"
empty_bucket "$PROJECT-runs-$ACCOUNT-$REGION"
delete_stack "$PROJECT-storage"
delete_stack "$PROJECT-batch"
delete_stack "$PROJECT-agent"
# Secrets Manager keeps a deleted secret for 30 days, which would block redeploying the same name.
aws secretsmanager delete-secret --secret-id "$PROJECT/origin-verify" --force-delete-without-recovery >/dev/null 2>&1 || true
echo "[teardown] done. ECR repositories are emptied by CloudFormation (EmptyOnDelete)."
