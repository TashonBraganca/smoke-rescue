#!/usr/bin/env bash
# Check CloudFront by default, or a local agent / ALB with --direct.
#   ./deploy/aws/smoke_test.sh https://dxxxxxxxx.cloudfront.net
#   ./deploy/aws/smoke_test.sh http://localhost:8080 --direct
#   ./deploy/aws/smoke_test.sh http://<alb-dns> --direct --header-file /tmp/origin-header
# The header file contains X-Origin-Verify: <value>; its contents are never printed.
# VISION_FRAME may select another recorded PNG. VISION_TOOL defaults to assess_visibility.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URL="${1:?usage: smoke_test.sh URL [--direct] [--header-file PATH]}"; shift
URL="${URL%/}"
PREFIX=/api
SITE=1
headers=(--header "Accept: application/json")
while [ $# -gt 0 ]; do
  case "$1" in
    --direct) PREFIX=""; SITE=0; shift ;;
    --header-file) headers+=(--header "@${2:?}"); shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
TOOL="${VISION_TOOL:-assess_visibility}"
FIELD="${VISION_FIELD:-frame}"
FRAME="${VISION_FRAME:-$HERE/samples/sample_rgb.png}"
[ -s "$FRAME" ] || { echo "missing frame $FRAME" >&2; exit 1; }
body="$(mktemp)"; trap 'rm -f "$body"' EXIT
fail=0
check() {
  if [ "$2" = "$3" ]; then echo "PASS $1 (HTTP $3)"; else echo "FAIL $1 (HTTP $3, wanted $2)"; head -c 400 "$body"; echo; fail=1; fi
}
code="$(curl -sS --connect-timeout 10 -m 30 "${headers[@]}" -o "$body" -w '%{http_code}' "$URL$PREFIX/health")" || code=000
check "GET $PREFIX/health" 200 "$code"
cat "$body"; echo
if [ "$code" = 200 ]; then
  python3 - "$body" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))
assert h['status'] == 'ok', h
assert h['opencv'].split('.')[0] == '5', h
PY
fi
code="$(curl -sS --connect-timeout 10 -m 90 "${headers[@]}" -o "$body" -w '%{http_code}' -F "$FIELD=@$FRAME" "$URL$PREFIX/vision/$TOOL")" || code=000
check "POST $PREFIX/vision/$TOOL" 200 "$code"
head -c 600 "$body"; echo
if [ "$code" = 200 ]; then
  python3 - "$body" "$TOOL" <<'PY'
import json, sys
v = json.load(open(sys.argv[1]))
assert v['tool'] == sys.argv[2], v
assert v['opencv'].split('.')[0] == '5', v
assert v['ms'] >= 0, v
PY
fi
if [ "$SITE" = 1 ]; then
  code="$(curl -sS --connect-timeout 10 -m 30 -o "$body" -w '%{http_code}' "$URL/")" || code=000
  check "GET / serves the site" 200 "$code"
fi
exit "$fail"
