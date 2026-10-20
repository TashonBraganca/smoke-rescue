#!/usr/bin/env bash
# Smoke test the deployed CloudFront URL: GET /api/health and one POST /api/vision/<tool> with a bundled frame.
#   ./deploy/aws/smoke_test.sh https://dxxxxxxxx.cloudfront.net
# Environment: VISION_TOOL (default assess_visibility), VISION_FIELD (multipart field, default frame).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URL="${1:?usage: smoke_test.sh https://<cloudfront-domain>}"
URL="${URL%/}"
TOOL="${VISION_TOOL:-assess_visibility}"
FIELD="${VISION_FIELD:-frame}"
FRAME="$HERE/samples/sample_rgb.png"
body="$(mktemp)"; trap 'rm -f "$body"' EXIT
fail=0

check() { # check label expected_code actual_code
  if [ "$2" = "$3" ]; then echo "PASS $1 (HTTP $3)"; else echo "FAIL $1 (HTTP $3, wanted $2)"; head -c 400 "$body"; echo; fail=1; fi
}

code="$(curl -sS -m 30 -o "$body" -w '%{http_code}' "$URL/api/health" || echo 000)"
check "GET /api/health" 200 "$code"
cat "$body"; echo

code="$(curl -sS -m 90 -o "$body" -w '%{http_code}' -F "$FIELD=@$FRAME" "$URL/api/vision/$TOOL" || echo 000)"
check "POST /api/vision/$TOOL" 200 "$code"
head -c 600 "$body"; echo

# The site root is served from S3 through CloudFront.
code="$(curl -sS -m 30 -o /dev/null -w '%{http_code}' "$URL/" || echo 000)"
check "GET / serves the site" 200 "$code"

exit "$fail"
