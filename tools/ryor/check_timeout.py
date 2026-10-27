"""Age only this test mission and verify the scheduled stop without browser polling."""
import json
import time
from pathlib import Path

import boto3

from test_api import http

ddb, ecs = [boto3.client(name, region_name='us-east-1') for name in ('dynamodb', 'ecs')]
TABLE = 'smoke-rescue-ryor'
code, launch = http('POST', '/api/mission', {'house': 'family_ranch', 'robots': 1,
                                           'difficulty': 'medium', 'terra': False})
assert code == 200 and launch['source'] == 'live', launch
mid = launch['mission']
key = {'pk': {'S': 'm#' + mid}}
deadline = time.monotonic() + 120  # Allow Fargate startup before fault injection.
while time.monotonic() < deadline:
    code, reply = http('GET', '/api/mission/' + mid + '?from=19')
    assert code == 200 and reply['status']['state'] != 'failed', reply
    if reply['status']['chunks']:
        break
    time.sleep(2)
else:
    raise AssertionError('Test mission did not produce its first chunk')
row = ddb.get_item(TableName=TABLE, Key=key, ConsistentRead=True)['Item']
aged_at = time.time()
ddb.update_item(TableName=TABLE, Key=key, UpdateExpression='SET started = :old',
                ExpressionAttributeValues={':old': {'N': str(int(aged_at) - 601)}})
observed = []
# Do not call the public mission endpoint here. Only the scheduled Lambda can mark it failed.
while time.time() - aged_at < 100:
    current = ddb.get_item(TableName=TABLE, Key=key, ConsistentRead=True)['Item']
    task = ecs.describe_tasks(cluster='smoke-rescue', tasks=[row['task']['S']])['tasks'][0]
    observed.append({'s': round(time.time() - aged_at, 2), 'mission_state': current['state']['S'],
                     'task_state': task['lastStatus'], 'task_desired': task['desiredStatus']})
    if current['state']['S'] == 'failed' and task['desiredStatus'] == 'STOPPED':
        break
    time.sleep(3)
else:
    ecs.stop_task(cluster='smoke-rescue', task=row['task']['S'], reason='ryor timeout test cleanup')
    raise AssertionError('The scheduled backstop did not stop the artificially aged task')
code, reply = http('GET', '/api/mission/' + mid + '?from=19')
assert code == 200 and reply['status']['state'] == 'failed'
assert reply['status']['fallback'] == 'cache:family_ranch__r1__medium'
result = {'test': 'Artificially age this mission by 601 seconds; no mission GET during the wait.',
          'mission': mid, 'observations': observed, 'public_result': reply['status']}
Path('handoff/ryor_evidence/scheduled_timeout.json').write_text(json.dumps(result, indent=1))
print(json.dumps(result))
