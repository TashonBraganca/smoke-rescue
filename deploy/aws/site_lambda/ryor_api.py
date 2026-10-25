# "Run your own rescue" mission API (tools/ryor/CONTRACT.txt sections 4 and 5), called from index.py.
#   POST /api/mission              validate the setup, per-IP limit, kill switch, global cap, then ecs RunTask
#   GET  /api/mission/<id>?from=N  status.json + up to 6 chunks; failures and the 600 s hard limit end in a fallback
#   GET  /api/mission/cache:<key>  the precomputed replay of one setup (ryor/cache/<key>/)
#   GET  /api/mission/health
# Every failure prints "RYOR_FAIL <mission> <reason>" (CloudWatch metric filter + alarm in deploy/aws/ryor.yaml).
import json, os, re, secrets, time
from concurrent.futures import ThreadPoolExecutor
import boto3
from botocore.exceptions import ClientError

E = os.environ
RB, TABLE, CLUSTER, FAMILY = E['RYOR_BUCKET'], E['RYOR_TABLE'], E['RYOR_CLUSTER'], E['RYOR_FAMILY']
SUBNETS, SG = E['RYOR_SUBNETS'].split(','), E['RYOR_SG']
ENABLED = E.get('RYOR_ENABLED', '1') == '1'
CAP, IP_LIMIT, HARD_S = int(E.get('RYOR_CAP', '2')), int(E.get('RYOR_IP_LIMIT', '6')), int(E.get('RYOR_HARD_S', '600'))
ENGINE = E.get('RYOR_ENGINE', '')
DAILY = int(E.get('RYOR_DAILY_CAP', '150'))  # live missions per UTC day, site-wide
PAGE = 6
s3, ecs, ddb = boto3.client('s3'), boto3.client('ecs'), boto3.client('dynamodb')
MID = re.compile(r'[a-z0-9]{8,40}\Z')
KEY = re.compile(r'([a-z0-9_]{1,40})__r([123])__(easy|medium|hard)\Z')
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
    return memo('gallery', 300, load)


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
            ddb.put_item(TableName=TABLE, Item={'pk': {'S': 'lock'}, 'owner': {'S': token}, 'exp': {'N': str(now + 15)},
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
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
            return False
        raise


def fail(mid, rec, reason, stop=True):
    if set_state(mid, 'failed', reason):
        fail_line(mid, reason)
        if stop and rec.get('task'):
            try:
                ecs.stop_task(cluster=CLUSTER, task=rec['task'], reason=('ryor: ' + reason)[:255])
            except ClientError as e:
                print('stop_task', mid, e)


# ------------------------------------------------------------------ routes
def route(method, path, event):
    try:
        if method == 'POST' and path == '/api/mission':
            return post(event)
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
        return j(500, {'error': 'mission API error'})


def post(event):
    raw = event.get('body') or ''
    if event.get('isBase64Encoded'):
        import base64
        raw = base64.b64decode(raw).decode('utf-8', 'replace')
    try:
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
    if diff not in g['difficulties']:
        return j(400, {'error': 'unknown difficulty'})
    key = '%s__r%d__%s' % (house, robots, diff)

    def fallback(reason):
        return j(200, {'mission': 'cache:' + key, 'source': 'recorded', 'reason': reason})

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
        n_day = count_day()
        if n_day > DAILY:
            print('RYOR_DAILY_CAP %d > %d' % (n_day, DAILY))
            return fallback('busy')
        prefix = 'ryor/missions/%s/' % mid
        env = [{'name': 'MISSION_ID', 'value': mid}, {'name': 'SETUP', 'value': key},
               {'name': 'BUCKET', 'value': RB}, {'name': 'PREFIX', 'value': prefix}]
        try:
            r = ecs.run_task(cluster=CLUSTER, taskDefinition=FAMILY, launchType='FARGATE', count=1,
                             startedBy='ryor-' + mid,
                             networkConfiguration={'awsvpcConfiguration': {
                                 'subnets': SUBNETS, 'securityGroups': [SG], 'assignPublicIp': 'ENABLED'}},
                             overrides={'containerOverrides': [{'name': 'engine', 'environment': env}]})
            if r.get('failures') or not r.get('tasks'):
                raise RuntimeError(str(r.get('failures'))[:200])
        except Exception as e:
            fail_line(mid, 'run_task failed %s: %s' % (type(e).__name__, e))
            return fallback('busy')
        now = int(time.time())
        ddb.put_item(TableName=TABLE, Item={
            'pk': {'S': 'm#' + mid}, 'setup': {'S': key}, 'task': {'S': r['tasks'][0]['taskArn']},
            'ip': {'S': ip}, 'started': {'N': str(now)}, 'state': {'S': 'starting'}, 'ttl': {'N': str(now + 7 * 86400)}})
    finally:
        unlock(token)
    print('RYOR_START %s %s %s' % (mid, key, r['tasks'][0]['taskArn'].rsplit('/', 1)[-1]))
    return j(200, {'mission': mid, 'source': 'live', 'poll': '/api/mission/' + mid})


def get(mid, frm):
    if mid.startswith('cache:'):
        key = mid[6:]
        if not KEY.match(key):
            return j(404, {'error': 'unknown setup'})
        st = s3_json('ryor/cache/%s/status.json' % key)
        if st is None:
            fail_line(mid, 'no recorded run for this setup')
            return j(404, {'error': 'no recorded run for this setup'})
        st['source'] = 'recorded'
        return j(200, {'status': st, 'chunks': chunks('ryor/cache/%s/' % key, frm, min(int(st.get('chunks', 0)), frm + PAGE))})
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
        st.update(state='failed', error=rec.get('reason', 'failed'), fallback='cache:' + rec['setup'])
        return j(200, {'status': st, 'chunks': []})
    st = st or synth(mid, rec)
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
    if sim_t < mission_s:
        return 'sim_t %.1f < %.0f' % (sim_t, mission_s)
    got = chunks(prefix, 0, n)
    if len(got) != n or n == 0:
        return 'chunks: %d of %d readable' % (len(got), n)
    ids = {'robot_' + 'ABC'[i] for i in range(int(m.group(2)))}
    x0, y0, x1, y1 = house['bounds']
    x0, y0, x1, y1 = x0 - 3, y0 - 3, x1 + 3, y1 + 3
    halves = []
    for c in got:
        for f in c.get('frames', []):
            h = float(f['t']) * 2
            if h != int(h):
                return 'frame time %s not on the 0.5 s grid' % f['t']
            halves.append(int(h))
            seen = {r['id'] for r in f.get('robots', [])}
            if seen != ids:
                return 'robots at t=%s are %s, setup has %s' % (f['t'], sorted(seen), sorted(ids))
            for r in f['robots']:
                if not (x0 <= r['x'] <= x1 and y0 <= r['y'] <= y1):
                    return '%s at (%.1f, %.1f) outside house bounds + 3 m at t=%s' % (r['id'], r['x'], r['y'], f['t'])
    if not halves or halves[0] > 1 or halves != list(range(halves[0], halves[0] + len(halves))):
        return 'frames missing or out of order (%d frames)' % len(halves)
    if halves[-1] < int(sim_t * 2) - 1:
        return 'last frame t=%.1f before sim_t %.1f' % (halves[-1] / 2, sim_t)
    return None


def health():
    try:
        running = len(running_tasks())
        ok = gallery() is not None
        n = cache_setups()
    except Exception as e:
        fail_line('-', 'health %s: %s' % (type(e).__name__, e))
        return j(503, {'ok': False, 'cap': CAP, 'engine': ENGINE})
    return j(200, {'ok': ok, 'running': running, 'cap': CAP, 'cache_setups': n, 'engine': ENGINE,
                   'enabled': ENABLED, 'ip_limit_per_hour': IP_LIMIT, 'daily_cap': DAILY, 'hard_limit_s': HARD_S})


def sweep():
    """Stop every ryor task older than the hard limit and mark its mission failed. Runs on each POST and every 5 min."""
    arns = running_tasks()
    stopped = 0
    for i in range(0, len(arns), 100):
        for t in ecs.describe_tasks(cluster=CLUSTER, tasks=arns[i:i + 100]).get('tasks', []):
            age = time.time() - t['createdAt'].timestamp()
            if age <= HARD_S:
                continue
            mid = (t.get('startedBy') or '').replace('ryor-', '', 1) or '-'
            reason = 'hard time limit: running %d s > %d s' % (age, HARD_S)
            rec = record(mid) if MID.match(mid) else None
            if rec:
                fail(mid, rec, reason)
            else:
                fail_line(mid, reason)
                ecs.stop_task(cluster=CLUSTER, task=t['taskArn'], reason='ryor: ' + reason)
            stopped += 1
    return {'running': len(arns), 'stopped': stopped}
