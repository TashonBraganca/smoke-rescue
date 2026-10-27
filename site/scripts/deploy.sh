#!/bin/zsh
# Build the static site, then publish it to the Cloudflare preview (same URL) and AWS Amplify.
set -e
cd /Volumes/Tashon/opencv/showcase/site-next
npx next build
test -f out/index.html
# the preview tunnel serves showcase/web/site on 127.0.0.1:8790; keep its URL by replacing the folder contents
rsync -a --delete --exclude lab.html --exclude preview/ --exclude voices/ out/ ../web/site/
APP=$(cat ../deploy_site/amplify_app_id.txt)
rm -f ../deploy_site/next_site.zip && (cd out && zip -qr -X ../../deploy_site/next_site.zip .)
J=$(aws amplify create-deployment --app-id $APP --branch-name main --region us-east-1 --output json)
JOB=$(echo "$J" | python3 -c "import json,sys;print(json.load(sys.stdin)['jobId'])")
URL=$(echo "$J" | python3 -c "import json,sys;print(json.load(sys.stdin)['zipUploadUrl'])")
curl -s -o /dev/null -w "amplify upload %{http_code}\n" -T ../deploy_site/next_site.zip "$URL"
aws amplify start-deployment --app-id $APP --branch-name main --job-id $JOB --region us-east-1 --query jobSummary.status --output text
for i in $(seq 1 40); do S=$(aws amplify get-job --app-id $APP --branch-name main --job-id $JOB --region us-east-1 --query job.summary.status --output text); [ "$S" = SUCCEED -o "$S" = FAILED ] && break; sleep 5; done
echo "amplify $S"
curl -s -o /dev/null -w "amplify site %{http_code}\n" https://main.$APP.amplifyapp.com/
curl -s -o /dev/null -w "tunnel site %{http_code}\n" https://brain-smooth-northeast-packard.trycloudflare.com/
