#!/usr/bin/env bash
# Deploy Smoke Rescue to AWS with one command. Needs credentials in the environment or a profile.
#
#   ./deploy/aws/deploy.sh --email you@example.com
#
# Options (or the matching environment variables):
#   --email ADDR        budget and alarm email (ALERT_EMAIL), required
#   --region REGION     AWS region (AWS_REGION), default us-east-1
#   --model-id ID       Bedrock model or inference profile id (BEDROCK_MODEL_ID)
#   --budget USD        monthly budget limit (MONTHLY_BUDGET_USD), default 50
#   --skip-slam         do not build/push the ORB-SLAM3 OpenCV 5 image (slow, about 30 to 60 minutes)
#   --skip-pipeline     do not build/push the evaluation image
#   --results DIR       also upload DIR to the site under /results/
#
# Order: agent (scaled to 0) -> push agent image -> agent (scaled to 1) -> batch -> push batch images
#        -> storage and CloudFront -> budget and alarms -> static site. Re-running is safe.
# Run from anywhere; paths resolve from this file. Never writes credentials to disk.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PROJECT="${PROJECT_NAME:-smoke-rescue}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
EMAIL="${ALERT_EMAIL:-}"
MODEL_ID="${BEDROCK_MODEL_ID:-us.anthropic.claude-haiku-4-5-20251001-v1:0}"
BUDGET="${MONTHLY_BUDGET_USD:-50}"
SKIP_SLAM=0
SKIP_PIPELINE=0
RESULTS_DIR=""

die() { echo "error: $*" >&2; exit 1; }
log() { echo "[deploy] $*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --email) EMAIL="${2:?}"; shift 2 ;;
    --region) REGION="${2:?}"; shift 2 ;;
    --model-id) MODEL_ID="${2:?}"; shift 2 ;;
    --budget) BUDGET="${2:?}"; shift 2 ;;
    --skip-slam) SKIP_SLAM=1; shift ;;
    --skip-pipeline) SKIP_PIPELINE=1; shift ;;
    --results) RESULTS_DIR="${2:?}"; shift 2 ;;
    -h|--help) sed -n '2,19p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown option $1" ;;
  esac
done
[ -n "$EMAIL" ] || die "--email is required"
export AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION"

# ---- preflight
command -v aws >/dev/null || die "aws CLI not found (install AWS CLI v2)"
command -v docker >/dev/null || die "docker not found"
docker buildx version >/dev/null 2>&1 || die "docker buildx not available"
docker info >/dev/null 2>&1 || die "docker daemon is not running"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text 2>/dev/null)" \
  || die "no valid AWS credentials. Set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY or AWS_PROFILE."
if [ "$SKIP_SLAM" = 0 ] || [ "$SKIP_PIPELINE" = 0 ]; then
  driver="$(docker buildx inspect 2>/dev/null | awk '/^Driver:/{print $2}')"
  [ "$driver" = "docker" ] || die "active buildx driver is '$driver'; batch images build from a local base image and need the 'docker' driver (run: docker buildx use default)"
fi
case "$MODEL_ID" in
  us.*|eu.*|apac.*|au.*|jp.*|global.*) ;;
  *) log "model id has no geo prefix; treating it as an in-region foundation model" ;;
esac
log "account $ACCOUNT, region $REGION, project $PROJECT"

REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
RUNS_BUCKET="$PROJECT-runs-$ACCOUNT-$REGION"
SITE_BUCKET="$PROJECT-site-$ACCOUNT-$REGION"
ORIGIN_SECRET="$PROJECT/origin-verify"
AGENT_STACK="$PROJECT-agent"; BATCH_STACK="$PROJECT-batch"
STORAGE_STACK="$PROJECT-storage"; BUDGET_STACK="$PROJECT-budget"

# Bedrock ARNs the task role may invoke. Foundation model ARN uses a wildcard region because
# cross-region inference profiles route to other regions.
case "$MODEL_ID" in
  us.*|eu.*|apac.*|au.*|jp.*|global.*)
    MODEL_ARNS="arn:aws:bedrock:$REGION:$ACCOUNT:inference-profile/$MODEL_ID,arn:aws:bedrock:*::foundation-model/${MODEL_ID#*.}" ;;
  *) MODEL_ARNS="arn:aws:bedrock:$REGION::foundation-model/$MODEL_ID" ;;
esac

# CloudFront origin-facing prefix list; skip it if it would not fit the default 60-rule security group quota.
PREFIX_LIST="$(aws ec2 describe-managed-prefix-lists \
  --filters Name=prefix-list-name,Values=com.amazonaws.global.cloudfront.origin-facing \
  --query 'PrefixLists[0].[PrefixListId,MaxEntries]' --output text 2>/dev/null || true)"
PREFIX_ID="$(echo "$PREFIX_LIST" | awk '{print $1}')"; PREFIX_MAX="$(echo "$PREFIX_LIST" | awk '{print $2}')"
if [ -z "$PREFIX_ID" ] || [ "$PREFIX_ID" = "None" ] || [ "${PREFIX_MAX:-999}" -gt 60 ]; then
  log "CloudFront prefix list unavailable or too large; ALB relies on the origin secret header only"
  PREFIX_ID=""
fi

# ---- helpers
hash_of() { # hash_of path...  -> short content hash, no git needed
  find "$@" -type f -not -path '*/__pycache__/*' -not -name '*.pyc' -not -name '.DS_Store' | LC_ALL=C sort \
    | xargs shasum -a 256 | shasum -a 256 | cut -c1-12
}
stack_exists() { aws cloudformation describe-stacks --stack-name "$1" >/dev/null 2>&1; }
stack_out() { aws cloudformation describe-stacks --stack-name "$1" \
  --query "Stacks[0].Outputs[?OutputKey=='$2'].OutputValue" --output text; }
deploy_stack() { # deploy_stack name template [Key=Value ...]
  local name="$1" tpl="$2"; shift 2
  log "stack $name"
  aws cloudformation deploy --stack-name "$name" --template-file "$HERE/$tpl" \
    --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset \
    --tags Project="$PROJECT" Environment=demo \
    --parameter-overrides ProjectName="$PROJECT" "$@"
}
image_exists() { aws ecr describe-images --repository-name "$1" --image-ids imageTag="$2" >/dev/null 2>&1; }
ecr_login() { aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null; }

AGENT_TAG="agent-$(hash_of "$ROOT/brain/brain" "$ROOT/brain/uv.lock" "$ROOT/brain/pyproject.toml" "$HERE/Dockerfile.agent")"
SLAM_TAG="slam-$(hash_of "$ROOT/slam" "$HERE/batch/Dockerfile.slam" "$HERE/batch/slam_entrypoint.sh")"
PIPE_TAG="pipe-$(hash_of "$ROOT/brain/brain" "$ROOT/brain/uv.lock" "$HERE/Dockerfile.pipeline" "$HERE/batch/pipeline_entrypoint.sh")"

agent_params() { # agent_params desired tag
  echo "RunsBucketName=$RUNS_BUCKET" "DesiredCount=$1" "ImageTag=$2" "AgentModelId=$MODEL_ID" \
    "BedrockModelArns=$MODEL_ARNS" "CloudFrontPrefixListId=$PREFIX_ID" "OriginSecretName=$ORIGIN_SECRET"
}

# ---- 1. agent stack and image
if stack_exists "$AGENT_STACK"; then
  ecr_login
  if ! image_exists "$PROJECT/agent" "$AGENT_TAG"; then
    log "build agent image $AGENT_TAG"
    docker buildx build --platform linux/arm64 --load -f "$HERE/Dockerfile.agent" -t "$REGISTRY/$PROJECT/agent:$AGENT_TAG" "$ROOT"
    docker push "$REGISTRY/$PROJECT/agent:$AGENT_TAG"
  fi
  # shellcheck disable=SC2046
  deploy_stack "$AGENT_STACK" agent.yaml $(agent_params 1 "$AGENT_TAG")
else
  # shellcheck disable=SC2046
  deploy_stack "$AGENT_STACK" agent.yaml $(agent_params 0 bootstrap)
  ecr_login
  log "build agent image $AGENT_TAG"
  docker buildx build --platform linux/arm64 --load -f "$HERE/Dockerfile.agent" -t "$REGISTRY/$PROJECT/agent:$AGENT_TAG" "$ROOT"
  docker push "$REGISTRY/$PROJECT/agent:$AGENT_TAG"
  # shellcheck disable=SC2046
  deploy_stack "$AGENT_STACK" agent.yaml $(agent_params 1 "$AGENT_TAG")
fi
ALB_DNS="$(stack_out "$AGENT_STACK" AlbDnsName)"
VPC_ID="$(stack_out "$AGENT_STACK" VpcId)"
SUBNETS="$(stack_out "$AGENT_STACK" SubnetIds)"

# ---- 2. batch stack and images
slam_tag_param="unset"; pipe_tag_param="unset"
batch_params() { echo "VpcId=$VPC_ID" "SubnetIds=$SUBNETS" "RunsBucketName=$RUNS_BUCKET" "SlamImageTag=$slam_tag_param" "PipelineImageTag=$pipe_tag_param"; }
if ! stack_exists "$BATCH_STACK"; then
  # shellcheck disable=SC2046
  deploy_stack "$BATCH_STACK" batch.yaml $(batch_params)
fi
ecr_login
if [ "$SKIP_SLAM" = 0 ]; then
  if ! image_exists "$PROJECT/slam" "$SLAM_TAG"; then
    log "build OpenCV 5 + ORB-SLAM3 image (long)"
    docker buildx build --platform linux/arm64 --load -f "$ROOT/slam/Dockerfile.cv5" -t "$PROJECT-orbslam3-cv5:$SLAM_TAG" "$ROOT"
    docker buildx build --platform linux/arm64 --load -f "$HERE/batch/Dockerfile.slam" \
      --build-arg "BASE_IMAGE=$PROJECT-orbslam3-cv5:$SLAM_TAG" -t "$REGISTRY/$PROJECT/slam:$SLAM_TAG" "$HERE/batch"
    docker push "$REGISTRY/$PROJECT/slam:$SLAM_TAG"
  fi
  slam_tag_param="$SLAM_TAG"
fi
if [ "$SKIP_PIPELINE" = 0 ]; then
  if ! image_exists "$PROJECT/pipeline" "$PIPE_TAG"; then
    log "build pipeline image"
    docker buildx build --platform linux/amd64 --load -f "$HERE/Dockerfile.pipeline" -t "$REGISTRY/$PROJECT/pipeline:$PIPE_TAG" "$ROOT"
    docker push "$REGISTRY/$PROJECT/pipeline:$PIPE_TAG"
  fi
  pipe_tag_param="$PIPE_TAG"
fi
# shellcheck disable=SC2046
deploy_stack "$BATCH_STACK" batch.yaml $(batch_params)

# ---- 3. storage and CloudFront
deploy_stack "$STORAGE_STACK" storage.yaml "RunsBucketName=$RUNS_BUCKET" "SiteBucketName=$SITE_BUCKET" \
  "ApiOriginDomain=$ALB_DNS" "OriginSecretName=$ORIGIN_SECRET"
DIST_ID="$(stack_out "$STORAGE_STACK" DistributionId)"
SITE_URL="$(stack_out "$STORAGE_STACK" SiteUrl)"

# ---- 4. budget and alarms
deploy_stack "$BUDGET_STACK" budget.yaml "AlertEmail=$EMAIL" "MonthlyBudgetUsd=$BUDGET" \
  "AlbFullName=$(stack_out "$AGENT_STACK" AlbFullName)" "TargetGroupFullName=$(stack_out "$AGENT_STACK" TargetGroupFullName)" \
  "ClusterName=$(stack_out "$AGENT_STACK" ClusterName)" "ServiceName=$(stack_out "$AGENT_STACK" ServiceName)"

# ---- 5. static site
log "upload site"
aws s3 sync "$HERE/site/" "s3://$SITE_BUCKET/" --delete --exclude "results/*" --cache-control "public,max-age=300" --only-show-errors
if [ -n "$RESULTS_DIR" ]; then
  [ -d "$RESULTS_DIR" ] || die "--results $RESULTS_DIR is not a directory"
  aws s3 sync "$RESULTS_DIR" "s3://$SITE_BUCKET/results/" --delete --only-show-errors
fi
aws cloudfront create-invalidation --distribution-id "$DIST_ID" --paths "/*" --query Invalidation.Id --output text >/dev/null

echo
echo "Deployed. URL: $SITE_URL"
echo "Confirm the SNS subscription email sent to $EMAIL, then run: $HERE/smoke_test.sh $SITE_URL"
