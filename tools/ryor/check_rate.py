"""Exceed the public IP limit with curl, save refusals, then stop only runs this check started."""
import json
import subprocess
import time
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[2]
URL = 'https://ajfjm7evl33gu6ptprty6hfthq0ukygx.lambda-url.us-east-1.on.aws'
ddb, ecs = [boto3.client(s, region_name='us-east-1') for s in ('dynamodb', 'ecs')]
body = json.dumps({'house': 'family_ranch', 'robots': 2, 'difficulty': 'medium', 'terra': False})
results, launched = [], []
try:
    # One curl process reuses its connection. Separate processes may leave through different proxy IPs.
    command = ['curl', '-4', '-sS', '-H', 'content-type: application/json', '-d', body, '-w', '\n%{http_code}\n']
    command += [URL + '/api/mission'] * 7  # Six starts are allowed per source IP per clock hour.
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()
    for n in range(7):
        raw, status = lines[n * 2:n * 2 + 2]
        reply = json.loads(raw)
        results.append({'request': n + 1, 'http': int(status), 'body': reply})
        print(n + 1, status, reply)
        if reply.get('source') == 'live':
            launched.append(reply['mission'])
    assert results[-1]['body'].get('reason') == 'rate_limited', 'The seventh request must be refused live.'
finally:
    for mid in launched:
        row = ddb.get_item(TableName='smoke-rescue-ryor', Key={'pk': {'S': 'm#' + mid}}, ConsistentRead=True)['Item']
        ecs.stop_task(cluster='smoke-rescue', task=row['task']['S'], reason='ryor public rate-limit verification complete')
        subprocess.run(['curl', '-sS', URL + '/api/mission/' + mid + '?from=19'], capture_output=True, check=True)
    (ROOT / 'handoff/ryor_evidence/rate_limit.json').write_text(json.dumps({'requests': results, 'test_runs_stopped': launched}, indent=1))
