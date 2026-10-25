"""Asynchronous headless decisions using the project's Azure agent backend.

Only brain observations enter the prompt. There are no images in this simulator,
so no OpenCV tools are exposed. The model can select a proven frontier or leave
the reflex brain searching. The existing AgentExplore path applies accepted orders.
"""
import json
import os
import threading
import time
from pathlib import Path

from botocore.exceptions import ClientError

MAX_STEPS = 20  # Owner's per-mission cap; four checkpoints x three robots use at most twelve.
DAILY_STEPS = 100  # Owner's site-wide UTC-day cap, reserved atomically before each model call.
INPUT_CHARS = 12000  # Bound prompt size to at most 12,000 UTF-8 bytes plus the fixed instructions.
REASON_CHARS = 360  # Keep decision captions readable on a phone.
TIMEOUT_S = 15  # Slow model calls never stop simulation; stale answers are rejected by the bridge.
SYSTEM = ('You are the search robot agent in Smoke Rescue. You have no images or ground-truth layout. '
          'The context is ONLY your own estimated pose, sensed room/map summary, confirmed people, sensed fire '
          'and received radio peers. Choose where to search next from the offered frontiers. Prefer unsearched '
          'indoor floor, avoid a peer already searching a room, and respect heat and the time left to exit. '
          'Use explore_frontier with one offered id and a short specific reason, or continue_search. '
          'You cannot confirm people, use camera tools, invent coordinates, or issue safety overrides. '
          'Reply with exactly one tool call. Write a plain ASCII sentence for the reason. '
          'The brain rechecks every order and keeps moving while you decide.')
TOOLS = [{'type': 'function', 'function': {'name': 'explore_frontier',
    'description': 'Search one of the indoor frontiers in this robot observation.', 'parameters': {
        'type': 'object', 'properties': {'frontier_id': {'type': 'integer'}, 'reason': {'type': 'string'}},
        'required': ['frontier_id', 'reason'], 'additionalProperties': False}}},
    {'type': 'function', 'function': {'name': 'continue_search',
    'description': 'Leave the reflex brain searching with its current plan.', 'parameters': {
        'type': 'object', 'properties': {'reason': {'type': 'string'}}, 'required': ['reason'],
        'additionalProperties': False}}}]


class Worker:
    def __init__(self, directory, mission, table, ddb, secret_client=None):
        self.directory, self.mission, self.table, self.ddb = Path(directory), mission, table, ddb
        self.secret_client = secret_client
        self.busy, self.steps, self.lock, self.backend = set(), 0, threading.Lock(), None
        self.closed = False

    def reserve(self):
        day, now = int(time.time() // 86400), int(time.time())
        try:
            # A transaction makes both caps hold across concurrent tasks and UTC day changes.
            self.ddb.transact_write_items(TransactItems=[{'Update': {
                'TableName': self.table, 'Key': {'pk': {'S': key}},
                'UpdateExpression': 'SET #t = :ttl ADD n :one',
                'ConditionExpression': 'attribute_not_exists(n) OR n < :cap',
                'ExpressionAttributeNames': {'#t': 'ttl'},
                'ExpressionAttributeValues': {':one': {'N': '1'}, ':cap': {'N': str(cap)},
                                              ':ttl': {'N': str(now + 3 * 86400)}}}}
                for key, cap in [('terra#day#%d' % day, DAILY_STEPS), ('terra#mission#' + self.mission, MAX_STEPS)]])
            return True
        except ClientError as e:
            if e.response['Error']['Code'] in ('TransactionCanceledException', 'ConditionalCheckFailedException'):
                return False
            raise

    def submit(self, observation):
        robot = observation['robot']
        with self.lock:
            if self.closed or robot in self.busy or self.steps >= MAX_STEPS:
                return
            self.busy.add(robot)
            self.steps += 1
        threading.Thread(target=self.decide, args=(observation,), daemon=True).start()

    def load_backend(self):
        # Credentials stay in memory. Reuse the same Azure backend as brain.agent.server.
        with self.lock:
            if self.backend is None:
                secret = self.secret_client.get_secret_value(SecretId=os.environ['TERRA_SECRET_ARN'])
                values = json.loads(secret['SecretString'])
                for key in ('AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_API_VERSION'):
                    if key in values:
                        os.environ[key] = values[key]
                os.environ['AZURE_OPENAI_MAX_ATTEMPTS'] = '1'  # No hidden retries past the daily reservation.
                os.environ['AZURE_OPENAI_TIMEOUT_S'] = str(TIMEOUT_S)
                from agent_backend import AzureOpenAIBackend
                self.backend = AzureOpenAIBackend(deployment='gpt-5.6-terra', reasoning_effort='low', timeout=TIMEOUT_S)
        return self.backend

    def decide(self, obs):
        robot = obs['robot']
        answer = {'asked_t': obs['t'], 'type': 'unavailable', 'model': 'gpt-5.6-terra'}
        try:
            if not self.reserve():
                answer['reason'] = 'The Terra decision budget is used. The brain keeps searching.'
            else:
                text = json.dumps(obs, separators=(',', ':'), ensure_ascii=True)
                if len(text) > INPUT_CHARS:
                    raise ValueError('observation exceeded input cap')
                reply = self.load_backend().chat([{'role': 'system', 'content': SYSTEM},
                                                   {'role': 'user', 'content': text}], TOOLS)
                if not reply.tool_calls:
                    raise ValueError('model returned no decision')
                call = reply.tool_calls[0]
                reason = str(call.arguments.get('reason', '')).replace('\u2014', ',')[:REASON_CHARS]
                if not reason:
                    raise ValueError('model returned no reason')
                answer.update(type='continue_search', reason=reason, usage=reply.raw.get('usage'), latency_s=reply.latency_s)
                if call.name == 'explore_frontier':
                    fid = call.arguments.get('frontier_id')
                    option = next((f for f in obs['frontiers'] if type(fid) is int and f['id'] == fid), None)
                    if option is None:
                        raise ValueError('model named a frontier absent from its observation')
                    answer.update(type='explore_frontier', x=option['x'], y=option['y'], frontier_id=fid)
                elif call.name != 'continue_search':
                    raise ValueError('model requested an unavailable tool')
        except Exception as e:
            # Do not log HTTP exception text: it may contain endpoint or credential details.
            answer.update(type='unavailable', reason='Terra did not answer in time or returned an invalid decision. The brain keeps searching.',
                          error_type=type(e).__name__)
        finally:
            with self.lock:
                if not self.closed:
                    path = self.directory / (robot + '.order.json')
                    tmp = path.with_suffix('.tmp')
                    tmp.write_text(json.dumps(answer))
                    tmp.replace(path)
                    with (self.directory / 'terra_audit.jsonl').open('a') as audit:
                        audit.write(json.dumps({'observation': obs, 'answer': answer}) + '\n')
                self.busy.discard(robot)
