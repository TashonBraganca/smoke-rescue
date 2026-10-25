#!/usr/bin/env bash
# HTTPS judge site on a Lambda Function URL (arm64) while CloudFront is blocked. Creates/updates stack smoke-rescue-site,
# then syncs deploy/aws/site into its private bucket. Re-running is safe. No docker, no change to other stacks.
# The function code (deploy/aws/site_lambda/) is zipped and uploaded to the ryor runs bucket under deploy/, so stack
# smoke-rescue-ryor must exist first (deploy_ryor.sh). Its outputs wire the /api/mission routes.
#   ./deploy/aws/deploy_site_lambda.sh
#   SITE_SYNC=0 ./deploy/aws/deploy_site_lambda.sh          # code and stack only, leave the static files alone
#   RYOR_ENABLED=0 RYOR_CAP=2 RYOR_DAILY_CAP=150 RYOR_IP_LIMIT=6 ./deploy/aws/deploy_site_lambda.sh   # mission API switches
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGION="${AWS_REGION:-us-east-1}"; STACK=smoke-rescue-site
out() { aws cloudformation describe-stacks --region "$REGION" --stack-name "$1" --query "Stacks[0].Outputs[?OutputKey=='$2'].OutputValue" --output text; }

RB="$(out smoke-rescue-ryor RunsBucketName)"
SUM="$(cat "$HERE"/site_lambda/*.py | shasum -a 256 | cut -c1-16)"
KEY="deploy/site_lambda/$SUM.zip"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
(cd "$HERE/site_lambda" && zip -q -X "$TMP/code.zip" index.py ryor_api.py)
aws s3 cp --region "$REGION" --quiet "$TMP/code.zip" "s3://$RB/$KEY"

aws cloudformation deploy --region "$REGION" --stack-name "$STACK" --template-file "$HERE/site_lambda.yaml" \
  --capabilities CAPABILITY_IAM --tags Project=smoke-rescue Environment=demo --no-fail-on-empty-changeset \
  --parameter-overrides CodeBucket="$RB" CodeKey="$KEY" \
    RyorPolicyArn="$(out smoke-rescue-ryor LambdaPolicyArn)" RyorBucket="$RB" \
    RyorTable="$(out smoke-rescue-ryor TableName)" RyorCluster="$(out smoke-rescue-ryor ClusterName)" \
    RyorSecurityGroup="$(out smoke-rescue-ryor TaskSecurityGroupId)" RyorEngine="$(out smoke-rescue-ryor ImageLabel)" \
    RyorEnabled="${RYOR_ENABLED:-1}" RyorCap="${RYOR_CAP:-2}" RyorDailyCap="${RYOR_DAILY_CAP:-150}" RyorIpLimit="${RYOR_IP_LIMIT:-6}"
BUCKET="$(out "$STACK" SiteBucketName)"
if [ "${SITE_SYNC:-1}" = 1 ]; then
  aws s3 sync --region "$REGION" --delete --exclude 'runs/README.txt' "$HERE/site/" "s3://$BUCKET/"
fi
echo "site: $(out "$STACK" SiteUrl)  code: s3://$RB/$KEY"
