# "Run your own rescue" mission API (tools/ryor/CONTRACT.txt sections 4 and 5), called from index.py.
#   POST /api/mission              validate the setup, per-IP limit, kill switch, global cap, then ecs RunTask
#   GET  /api/mission/<id>?from=N  status.json + up to 6 chunks; failures and the 600 s hard limit end in a fallback
#   GET  /api/mission/cache:<key>  the precomputed replay of one setup (ryor/cache/<key>/)
#   GET  /api/mission/health
# Every failure prints "RYOR_FAIL <mission> <reason>" (CloudWatch metric filter + alarm in deploy/aws/ryor.yaml).
import hashlib, json, math, os, re, secrets, time
from concurrent.futures import ThreadPoolExecutor
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
from placement import validate as validate_placement, setup_key
import layout as room_layout

E = os.environ
RB, TABLE, CLUSTER, FAMILY = E['RYOR_BUCKET'], E['RYOR_TABLE'], E['RYOR_CLUSTER'], E['RYOR_FAMILY']
SUBNETS, SG = E['RYOR_SUBNETS'].split(','), E['RYOR_SG']
ENABLED = E.get('RYOR_ENABLED', '1') == '1'
CAP, IP_LIMIT, HARD_S = int(E.get('RYOR_CAP', '2')), int(E.get('RYOR_IP_LIMIT', '6')), int(E.get('RYOR_HARD_S', '600'))
ENGINE = E.get('RYOR_ENGINE', '')
DAILY = int(E.get('RYOR_DAILY_CAP', '150'))  # live missions per UTC day, site-wide
PAGE = 6  # Bound each public response below Lambda's 6 MB payload limit.
# The launch lease must outlive the entire Lambda invocation, including AWS retries.
AWS_CONFIG = Config(connect_timeout=3, read_timeout=15, retries={'total_max_attempts': 2})
s3, ecs, ddb = [boto3.client(name, config=AWS_CONFIG) for name in ('s3', 'ecs', 'dynamodb')]
MID = re.compile(r'[a-z0-9]{8,40}\Z')
KEY = re.compile(r'([a-z0-9_]{1,40})__r([123])__(easy|medium|hard)(?:__p([a-f0-9]{20}))?\Z')
_memo = {}


def j(code, obj):
    return code, json.dumps(obj, separators=(',', ':')).encode()


def fail_line(mission, reason):
    print('RYOR_FAIL %s %s' % (mission, ' '.join(str(reason).split())[:300]))


def memo(name, ttl, fn):
    hit = _memo.get(name)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    val = fn()
    _memo[name] = (time.time(), val)
    return val


def s3_json(key):
    try:
        return json.loads(s3.get_object(Bucket=RB, Key=key)['Body'].read())
    except ClientError as e:
        if e.response['Error']['Code'] in ('NoSuchKey', '404'):
            return None
        raise


def gallery():
    def load():
        g = s3_json('ryor/gallery.json')
        if not g:
            return None
        return {'houses': {h['id']: h for h in g.get('houses', [])}, 'difficulties': set(g.get('difficulties', {})),
                'mission_s': float(g.get('mission_s', 180))}
    return memo('gallery', 60, load)  # Publish gallery and ready-cache manifest together.


def cache_setups():
    def count():
        n, tok = 0, None
        while True:
            kw = {'Bucket': RB, 'Prefix': 'ryor/cache/', 'Delimiter': '/'}
            if tok:
                kw['ContinuationToken'] = tok
            r = s3.list_objects_v2(**kw)
            n += len(r.get('CommonPrefixes', []))
            tok = r.get('NextContinuationToken')
            if not tok:
                return n
    return memo('cache_setups', 60, count)


def running_tasks():
    arns, tok = [], None
    while True:
        kw = {'cluster': CLUSTER, 'family': FAMILY, 'desiredStatus': 'RUNNING'}
        if tok:
            kw['nextToken'] = tok
        r = ecs.list_tasks(**kw)
        arns += r['taskArns']
        tok = r.get('nextToken')
        if not tok:
            return arns


def chunks(prefix, start, end):
    keys = ['%schunk_%04d.json' % (prefix, n) for n in range(start, end)]
    if not keys:
        return []
    with ThreadPoolExecutor(min(8, len(keys))) as ex:
        got = list(ex.map(s3_json, keys))
    out = []
    for c in got:  # stop at the first chunk not written yet
        if c is None:
            break
        for frame in c.get('frames', []):
            for event in frame.get('events', []):
                if event.get('kind') == 'agent_order':
                    # The brain log contains acceptances and refusals. Older captions said "applied" for both.
                    event['text'] = event.get('text', '').replace(' applied an agent order.', ' reported an agent order.')
        out.append(c)
    return out


# ------------------------------------------------------------------ DynamoDB: per-IP counter, launch lock, mission rows
def count_ip(ip):
    hour = int(time.time() // 3600)
    r = ddb.update_item(TableName=TABLE, Key={'pk': {'S': 'ip#%s#%d' % (ip, hour)}},
                        UpdateExpression='ADD n :one SET #t = :ttl', ExpressionAttributeNames={'#t': 'ttl'},
                        ExpressionAttributeValues={':one': {'N': '1'}, ':ttl': {'N': str((hour + 2) * 3600)}},
                        ReturnValues='UPDATED_NEW')
    return int(r['Attributes']['n']['N'])


def count_day():
    day = int(time.time() // 86400)
    r = ddb.update_item(TableName=TABLE, Key={'pk': {'S': 'day#%d' % day}},
                        UpdateExpression='ADD n :one SET #t = :ttl', ExpressionAttributeNames={'#t': 'ttl'},
                        ExpressionAttributeValues={':one': {'N': '1'}, ':ttl': {'N': str((day + 3) * 86400)}},
                        ReturnValues='UPDATED_NEW')
    return int(r['Attributes']['n']['N'])


def lock(token):
    now = int(time.time())
    for _ in range(25):
        try:
            ddb.put_item(TableName=TABLE, Item={'pk': {'S': 'lock'}, 'owner': {'S': token}, 'exp': {'N': str(now + 90)},
                                                 'ttl': {'N': str(now + 3600)}},
                         ConditionExpression='attribute_not_exists(pk) OR #e < :now',
                         ExpressionAttributeNames={'#e': 'exp'}, ExpressionAttributeValues={':now': {'N': str(now)}})
            return True
        except ClientError as e:
            if e.response['Error']['Code'] != 'ConditionalCheckFailedException':
                raise
            time.sleep(0.2)
            now = int(time.time())
    return False


def unlock(token):
    try:
        ddb.delete_item(TableName=TABLE, Key={'pk': {'S': 'lock'}}, ConditionExpression='#o = :t',
                        ExpressionAttributeNames={'#o': 'owner'}, ExpressionAttributeValues={':t': {'S': token}})
    except ClientError:
        pass


def record(mid):
    r = ddb.get_item(TableName=TABLE, Key={'pk': {'S': 'm#' + mid}}, ConsistentRead=True).get('Item')
    return {k: list(v.values())[0] for k, v in r.items()} if r else None


def set_state(mid, state, reason=None):
    """Move a mission to done or failed once. Returns False if it already left the running states."""
    names, vals = {'#s': 'state'}, {':s': {'S': state}, ':f': {'S': 'failed'}, ':d': {'S': 'done'}}
    expr = 'SET #s = :s'
    if reason:
        expr += ', reason = :r'
        vals[':r'] = {'S': reason[:500]}
    try:
        ddb.update_item(TableName=TABLE, Key={'pk': {'S': 'm#' + mid}}, UpdateExpression=expr,
                        ConditionExpression='attribute_exists(pk) AND #s <> :f AND #s <> :d',
                        ExpressionAttributeNames=names, ExpressionAttributeValues=vals)
        rec = record(mid)
        if rec and rec.get('slot'):
            release_slot(rec['slot'], mid)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
            return False
        raise


def take_slot(mid):
    # DynamoDB conditional writes include pending launches. ECS ListTasks is eventually consistent.
    now = int(time.time())
    for i in range(CAP):
        key = 'slot#%d' % i
        try:
            ddb.put_item(TableName=TABLE, Item={'pk': {'S': key}, 'owner': {'S': mid},
                         'exp': {'N': str(now + HARD_S + 120)}, 'ttl': {'N': str(now + 86400)}},
                         ConditionExpression='attribute_not_exists(pk) OR #e < :now',
                         ExpressionAttributeNames={'#e': 'exp'}, ExpressionAttributeValues={':now': {'N': str(now)}})
            return key
        except ClientError as e:
            if e.response['Error']['Code'] != 'ConditionalCheckFailedException':
                raise
    return None


def release_slot(slot, mid):
    try:
        ddb.delete_item(TableName=TABLE, Key={'pk': {'S': slot}}, ConditionExpression='#o = :m',
                        ExpressionAttributeNames={'#o': 'owner'}, ExpressionAttributeValues={':m': {'S': mid}})
    except ClientError as e:
        if e.response['Error']['Code'] != 'ConditionalCheckFailedException':
            raise


def fail(mid, rec, reason, stop=True):
    if set_state(mid, 'failed', reason):
        fail_line(mid, reason)
        if stop and rec.get('task'):
            try:
                ecs.stop_task(cluster=CLUSTER, task=rec['task'], reason=('ryor: ' + reason)[:255])
                return True
            except ClientError as e:
                print('stop_task', mid, e)
    return False


# ------------------------------------------------------------------ routes
def route(method, path, event):
    try:
        if method == 'POST' and path == '/api/mission':
            return post(event)
        if method == 'POST' and path == '/api/mission/validate':
            return post(event, validate_only=True)
        if method == 'GET' and path == '/api/mission/health':
            return health()
        if method == 'GET' and path.startswith('/api/mission/'):
            q = event.get('queryStringParameters') or {}
            try:
                frm = max(0, int(q.get('from', '0')))
            except ValueError:
                return j(400, {'error': 'from must be an integer'})
            return get(path[len('/api/mission/'):], frm)
        return j(405 if path == '/api/mission' else 404, {'error': 'not a mission route'})
    except Exception as e:  # never a stack trace to the page
        fail_line('-', 'api error %s %s %s: %s' % (method, path, type(e).__name__, e))
        return j(503, {'error': 'The mission service is unavailable. Your scene has not changed.'})


def post(event, validate_only=False):
    raw = event.get('body') or ''
    if event.get('isBase64Encoded'):
        import base64
        raw = base64.b64decode(raw).decode('utf-8', 'replace')
    try:
        if len(raw) > 24576:  # Room rectangles plus five points fit below this bounded public input.
            return j(400, {'error': 'The scene request is too large.'})
        body = json.loads(raw)
        house, robots, diff = body.get('house'), body.get('robots'), body.get('difficulty')
    except Exception:
        return j(400, {'error': 'body must be JSON {"house","robots","difficulty"}'})
    g = gallery()
    if g is None:
        fail_line('-', 'gallery missing: ryor/gallery.json')
        return j(503, {'error': 'gallery not available'})
    if not isinstance(house, str) or house not in g['houses']:
        return j(400, {'error': 'unknown house'})
    if type(robots) is not int or robots not in (1, 2, 3):
        return j(400, {'error': 'robots must be 1, 2 or 3'})
    if not isinstance(diff, str) or diff not in g['difficulties']:
        return j(400, {'error': 'unknown difficulty'})
    key = '%s__r%d__%s' % (house, robots, diff)
    base = key
    placement = None
    layout = None
    edited_house = None
    prepare = body.get('prepare') is True
    if body.get('layout') is not None:
        try:
            edited_house = room_layout.build(g['houses'][house], body['layout'])
            layout = sorted(body['layout'], key=lambda r: r['id'])
            placement = validate_placement(edited_house, body['placement']) if body.get('placement') is not None else room_layout.generated_placement(edited_house)
            key = room_layout.key(base, layout, placement, ENGINE)
        except (ValueError, KeyError) as e:
            return j(400, {'error': str(e)})
    elif body.get('placement') is not None:
        try:
            placement = validate_placement(g['houses'][house], body['placement'])
        except (ValueError, KeyError) as e:
            return j(400, {'error': str(e)})
        key = setup_key(base, placement, ENGINE)
    if validate_only:
        return j(200, {'valid': True, 'placement': placement, 'layout': layout, 'house': edited_house, 'setup': key})
    cached = ready_cache(key)
    if prepare and cached:
        return j(200, {'mission': 'cache:' + key, 'setup': key, 'source': 'recorded', 'prepared': True})
    if not prepare and not cached:
        return j(409, {'error': 'Prepare this scene first so its exact recorded fallback is ready.', 'setup': key,
                       'needs_prepare': True})

    def fallback(reason):
        if not cached:
            return j(429, {'error': 'Live slots are full. Your scene is kept. Try preparing it shortly.', 'reason': reason})
        return j(200, {'mission': 'cache:' + key, 'setup': key, 'source': 'recorded', 'reason': reason})

    ip = event['requestContext']['http'].get('sourceIp', '?')
    if count_ip(ip) > IP_LIMIT:
        return fallback('rate_limited')
    if not ENABLED:
        return fallback('disabled')
    sweep()
    mid = '%x%s' % (int(time.time()), secrets.token_hex(5))
    token = secrets.token_hex(8)
    if not lock(token):
        return fallback('busy')
    try:
        if len(running_tasks()) >= CAP:
            return fallback('busy')
        slot = take_slot(mid)
        if not slot:
            return fallback('busy')
        n_day = count_day()
        if n_day > DAILY:
            print('RYOR_DAILY_CAP %d > %d' % (n_day, DAILY))
            release_slot(slot, mid)
            return fallback('daily_limited')
        prefix = 'ryor/missions/%s/' % mid
        if placement:
            # The task revalidates these bytes before using them. No user-controlled path or URL enters its request.
            s3.put_object(Bucket=RB, Key='ryor/requests/' + key + '.json',
                          Body=json.dumps({'base': base, 'setup': key, 'placement': placement, 'layout': layout, 'engine': ENGINE}).encode(),
                          ContentType='application/json')
        env = [{'name': 'MISSION_ID', 'value': mid}, {'name': 'SETUP', 'value': key},
               {'name': 'BUCKET', 'value': RB}, {'name': 'PREFIX', 'value': prefix},
               {'name': 'PREPARE', 'value': '1' if prepare else '0'},
               {'name': 'TERRA_ENABLED', 'value': '1' if body.get('terra', True) is True and not prepare else '0'}]
        # Fargate offers two or four vCPUs here. Three independent sensing threads need the four-vCPU size.
        task_cpu, task_memory = ('4096', '8192') if robots == 3 else ('2048', '4096')
        try:
            r = ecs.run_task(cluster=CLUSTER, taskDefinition=FAMILY, launchType='FARGATE', count=1,
                             clientToken=mid,
                             startedBy='ryor-' + mid,
                             networkConfiguration={'awsvpcConfiguration': {
                                 'subnets': SUBNETS, 'securityGroups': [SG], 'assignPublicIp': 'ENABLED'}},
                             overrides={'cpu': task_cpu, 'memory': task_memory,
                                        'containerOverrides': [{'name': 'engine', 'environment': env}]})
            if r.get('failures') or not r.get('tasks'):
                raise RuntimeError(str(r.get('failures'))[:200])
        except Exception as e:
            fail_line(mid, 'run_task failed %s: %s' % (type(e).__name__, e))
            release_slot(slot, mid)
            return fallback('busy')
        now = int(time.time())
        ddb.put_item(TableName=TABLE, Item={
            'pk': {'S': 'm#' + mid}, 'setup': {'S': key}, 'task': {'S': r['tasks'][0]['taskArn']},
            'slot': {'S': slot},
            'ip': {'S': ip}, 'started': {'N': str(now)}, 'state': {'S': 'starting'},
            'prepare': {'S': '1' if prepare else '0'}, 'ttl': {'N': str(now + 7 * 86400)}})
    finally:
        unlock(token)
    print('RYOR_START %s %s %s' % (mid, key, r['tasks'][0]['taskArn'].rsplit('/', 1)[-1]))
    return j(200, {'mission': mid, 'setup': key, 'source': 'live', 'preparing': prepare, 'poll': '/api/mission/' + mid})


def ready_cache(key):
    """Only validated, completed recordings may be promised as a fallback."""
    def read():
        st = s3_json(cache_prefix(key) + 'status.json')
        if not st or st.get('state') != 'done' or not st.get('validated'):
            return None
        return st
    return memo('cache:' + key, 10, read)  # Brief negative caching lets a prepared scene become ready promptly.


def cache_prefix(key):
    return ('ryor/cache-custom/' if '__p' in key else 'ryor/cache/') + key + '/'


def get(mid, frm):
    if mid.startswith('cache:'):
        key = mid[6:]
        if not KEY.match(key):
            return j(404, {'error': 'unknown setup'})
        st = ready_cache(key)
        if st is None:
            fail_line(mid, 'no recorded run for this setup')
            return j(404, {'error': 'no recorded run for this setup'})
        st['source'] = 'recorded'
        return j(200, {'status': st, 'chunks': chunks(cache_prefix(key), frm, min(int(st.get('chunks', 0)), frm + PAGE))})
    if not MID.match(mid):
        return j(404, {'error': 'unknown mission'})
    rec = record(mid)
    if rec is None:
        return j(404, {'error': 'unknown mission'})
    prefix = 'ryor/missions/%s/' % mid
    st = s3_json(prefix + 'status.json')
    if rec['state'] not in ('done', 'failed'):
        reason = check(mid, rec, st, prefix)
        if reason:
            fail(mid, rec, reason)
        rec = record(mid)
    if rec['state'] == 'failed':
        st = st or synth(mid, rec)
        st.update(state='failed', error=rec.get('reason', 'failed'))
        if ready_cache(rec['setup']):
            st['fallback'] = 'cache:' + rec['setup']
        return j(200, {'status': st, 'chunks': []})
    st = st or synth(mid, rec)
    st['preparing'] = rec.get('prepare') == '1'
    return j(200, {'status': st, 'chunks': chunks(prefix, frm, min(int(st.get('chunks', 0)), frm + PAGE))})


def synth(mid, rec):
    return {'v': 1, 'mission': mid, 'setup': rec['setup'], 'state': 'starting', 'source': 'live', 'chunks': 0,
            'sim_t': 0.0, 'updated': None, 'error': None, 'score': None, 'engine': {'git': None, 'image': ENGINE}}


def check(mid, rec, st, prefix):
    """Reason the live mission has failed, or None. Marks a validated finished mission done."""
    if st and st.get('state') == 'failed':
        return 'engine: %s' % (st.get('error') or 'failed')
    if st and st.get('state') == 'done':
        why = validate(rec, st, prefix)
        if why:
            return 'validation: ' + why
        if rec.get('prepare') == '1':
            # Publish status last. Readers cannot see a ready recording with missing chunks.
            for n in range(int(st['chunks'])):
                name = 'chunk_%04d.json' % n
                s3.copy_object(Bucket=RB, Key=cache_prefix(rec['setup']) + name,
                               CopySource={'Bucket': RB, 'Key': prefix + name}, CacheControl='no-store',
                               MetadataDirective='REPLACE', ContentType='application/json')
            cache_st = dict(st, mission='cache', source='recorded', validated=True)
            s3.put_object(Bucket=RB, Key=cache_prefix(rec['setup']) + 'status.json',
                          Body=json.dumps(cache_st).encode(), ContentType='application/json', CacheControl='no-store')
            _memo.pop('cache:' + rec['setup'], None)
        set_state(mid, 'done')
        print('RYOR_DONE %s %s wall %ds' % (mid, rec['setup'], int(time.time()) - int(rec['started'])))
        return None
    age = int(time.time()) - int(rec['started'])
    if age > HARD_S:
        return 'hard time limit: running %d s > %d s' % (age, HARD_S)
    t = ecs.describe_tasks(cluster=CLUSTER, tasks=[rec['task']]).get('tasks', [])
    if t and t[0].get('lastStatus') == 'STOPPED':
        st2 = s3_json(prefix + 'status.json')  # the engine may have finished between the two reads
        if st2 and st2.get('state') in ('done', 'failed'):
            return check(mid, rec, st2, prefix)
        c = (t[0].get('containers') or [{}])[0]
        return 'task stopped before done: %s (exit %s)' % (t[0].get('stoppedReason', '?'), c.get('exitCode', '?'))
    return None


def validate(rec, st, prefix):
    """CONTRACT.txt section 5. Returns a reason string, or None when the replay is fit to show."""
    g = gallery()
    m = KEY.match(rec['setup'])
    house = g and g['houses'].get(m.group(1))
    if not house:
        return 'house not in gallery'
    mission_s = g['mission_s']
    n = int(st.get('chunks', 0))
    sim_t = float(st.get('sim_t') or 0)
    if not math.isfinite(sim_t) or sim_t != mission_s:
        return 'sim_t %.1f < %.0f' % (sim_t, mission_s)
    if not 1 <= n <= 19:  # 180 s, 10 s chunks, inclusive final frame.
        return 'invalid chunk count'
    got = chunks(prefix, 0, n)
    if len(got) != n or n == 0:
        return 'chunks: %d of %d readable' % (len(got), n)
    ids = {'robot_' + 'ABC'[i] for i in range(int(m.group(2)))}
    x0, y0, x1, y1 = house['bounds']
    x0, y0, x1, y1 = x0 - 3, y0 - 3, x1 + 3, y1 + 3
    halves = []
    confirmations = set()
    for index, c in enumerate(got):
        if c.get('chunk') != index or c.get('t0') != index * 10:
            return 'chunk index does not match its contents'
        for f in c.get('frames', []):
            h = float(f['t']) * 2
            if not math.isfinite(h) or h != int(h):
                return 'frame time %s not on the 0.5 s grid' % f['t']
            halves.append(int(h))
            seen = {r['id'] for r in f.get('robots', [])}
            if seen != ids or len(f.get('robots', [])) != len(ids):
                return 'robots at t=%s are %s, setup has %s' % (f['t'], sorted(seen), sorted(ids))
            for r in f['robots']:
                if not (x0 <= r['x'] <= x1 and y0 <= r['y'] <= y1):
                    return '%s at (%.1f, %.1f) outside house bounds + 3 m at t=%s' % (r['id'], r['x'], r['y'], f['t'])
            for ev in f.get('events', []):
                if ev.get('kind') == 'survivor_confirmed':
                    confirmations.add((ev.get('robot'), float(ev.get('t', f['t']))))
            for p in f.get('found', []):
                if (p.get('by'), p.get('t')) not in confirmations or p.get('t', -1) > f['t']:
                    return 'person marker appears before confirmation'
            for rid, cells in f.get('seen', {}).items():
                if rid not in ids or any(type(cell) is not int or not 0 <= cell < 250000 for cell in cells):
                    return 'invalid searched map cells'
            for fire in f.get('fire', []):
                if len(fire) != 3 or any(not math.isfinite(v) for v in fire) or fire[2] < 0:
                    return 'invalid fire marker'
    if halves != list(range(int(mission_s * 2) + 1)):
        return 'frames missing or out of order (%d frames)' % len(halves)
    if halves[-1] != int(sim_t * 2):
        return 'last frame t=%.1f before sim_t %.1f' % (halves[-1] / 2, sim_t)
    score = st.get('score')
    if not isinstance(score, dict):
        return 'score is missing'
    for field in ('people_found', 'people_total', 'robots_out', 'false_confirmations'):
        if type(score.get(field)) is not int or score[field] < 0:
            return 'invalid score ' + field
    if score['people_found'] > score['people_total'] or score['robots_out'] > len(ids):
        return 'score exceeds the people or robots in the scene'
    placement = st.get('placement')
    if m.group(4) and not isinstance(placement, dict):
        return 'custom placement is missing'
    expected_people = len(placement['people']) if m.group(4) else int(house['people'])
    if score['people_total'] != expected_people:
        return 'score has the wrong number of people'
    if score['robots_out'] != sum(bool(robot.get('out')) for robot in got[-1]['frames'][-1]['robots']):
        return 'score has the wrong number of robots outside'
    for field, maximum in [('coverage_pct', 100), ('time_s', mission_s)]:
        value = score.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
            return 'invalid score ' + field
    return None


def health():
    try:
        running = len(running_tasks())
        ok = gallery() is not None
        manifest = s3_json('ryor/cache_manifest.json') or {}
        n = int(manifest.get('ready', 0))
    except Exception as e:
        fail_line('-', 'health %s: %s' % (type(e).__name__, e))
        return j(503, {'ok': False, 'cap': CAP, 'engine': ENGINE})
    return j(200, {'ok': ok, 'running': running, 'cap': CAP, 'cache_setups': n, 'engine': ENGINE,
                   'cache_expected': int(manifest.get('total', 0)),
                   'ready_setups': manifest.get('setups', []),
                   'enabled': ENABLED, 'ip_limit_per_hour': IP_LIMIT, 'daily_cap': DAILY, 'hard_limit_s': HARD_S,
                   'terra_steps_per_mission': 20, 'terra_steps_per_utc_day': 100,
                   'pair_vcpu': 2, 'trio_vcpu': 4})


def sweep():
    """Stop stale task containers. Runs on each POST and once per minute even with no browser polling."""
    stopped = set()
    # Reconcile completed or failed owners even after ECS removes them from ListTasks.
    for i in range(CAP):
        item = ddb.get_item(TableName=TABLE, Key={'pk': {'S': 'slot#%d' % i}}, ConsistentRead=True).get('Item')
        if item:
            mid = item['owner']['S']
            rec = record(mid)
            if rec and rec['state'] not in ('done', 'failed'):
                prefix = 'ryor/missions/%s/' % mid
                why = check(mid, rec, s3_json(prefix + 'status.json'), prefix)
                if why:
                    if fail(mid, rec, why):
                        stopped.add(rec['task'])
            elif rec:
                release_slot('slot#%d' % i, mid)
    arns = running_tasks()
    for i in range(0, len(arns), 100):
        for t in ecs.describe_tasks(cluster=CLUSTER, tasks=arns[i:i + 100]).get('tasks', []):
            if t['taskArn'] in stopped:
                continue
            age = time.time() - t['createdAt'].timestamp()
            if age <= HARD_S:
                continue
            mid = (t.get('startedBy') or '').replace('ryor-', '', 1) or '-'
            reason = 'hard time limit: running %d s > %d s' % (age, HARD_S)
            rec = record(mid) if MID.match(mid) else None
            if rec and rec['state'] not in ('done', 'failed'):
                if fail(mid, rec, reason):
                    stopped.add(t['taskArn'])
            else:
                # A terminal database row must not keep a stuck container billable forever.
                # Completed simulations stay valid; the stale container still has to stop.
                if not rec:
                    fail_line(mid, reason)
                else:
                    print('RYOR_CLEANUP %s %s stale container' % (mid, rec['state']))
                ecs.stop_task(cluster=CLUSTER, task=t['taskArn'], reason='ryor: ' + reason)
                stopped.add(t['taskArn'])
    return {'running': len(arns), 'stopped': len(stopped)}
