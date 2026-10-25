#!/usr/bin/env bash
# HTTPS judge site on a Lambda Function URL (arm64) while CloudFront is blocked. Creates/updates stack smoke-rescue-site,
# then syncs deploy/aws/site into its private bucket. Re-running is safe. No docker, no change to other stacks.
#   ./deploy/aws/deploy_site_lambda.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGION="${AWS_REGION:-us-east-1}"; STACK=smoke-rescue-site
aws cloudformation deploy --region "$REGION" --stack-name "$STACK" --template-file "$HERE/site_lambda.yaml" \
  --capabilities CAPABILITY_IAM --tags Project=smoke-rescue Environment=demo --no-fail-on-empty-changeset
out() { aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }
BUCKET="$(out SiteBucketName)"
aws s3 sync --region "$REGION" --delete --exclude 'runs/README.txt' "$HERE/site/" "s3://$BUCKET/"
echo "site: $(out SiteUrl)"
