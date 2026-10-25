#!/usr/bin/env python3
"""End-to-end test of the "Run your own rescue" mission API on the public site URL (CONTRACT.txt sections 4 and 5).

  python3 tools/ryor/test_api.py --seed-fixture        # once: test house zz_apitest + its recorded replay in the bucket
  python3 tools/ryor/test_api.py                       # health, bad setup, live run, kill test, cap, kill switch, rate limit
  python3 tools/ryor/test_api.py --only kill           # one test: health bad cache live kill cap disabled rate

Needs the aws CLI (account 968000946565) for the parts a browser cannot do: stopping the ECS task mid-run, flipping the
site stack's RyorCap / RyorEnabled parameters (restored afterwards), clearing this machine's per-IP counter, and
reading the RYOR_FAIL log lines. Standard library only.
"""
import argparse, json, subprocess, sys, time, urllib.error, urllib.request

URL = 'https://ajfjm7evl33gu6ptprty6hfthq0ukygx.lambda-url.us-east-1.on.aws'
BUCKET, TABLE, CLUSTER = 'smoke-rescue-ryor-runs-968000946565', 'smoke-rescue-ryor', 'smoke-rescue'
SITE_STACK, LOGS = 'smoke-rescue-site', '/aws/lambda/smoke-rescue-site'
FIX_HOUSE = 'zz_apitest'
results = []


def http(method, path, body=None):
    data = json.dumps(body).encode() if isinstance(body, (dict, list)) else body
    req = urllib.request.Request(URL + path, data=data, method=method, headers={'content-type': 'application/json'})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=75) as r:
            code, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        code, raw = e.code, e.read()
    try:
        obj = json.loads(raw)
    except ValueError:
        obj = raw.decode('utf-8', 'replace')
    short = json.dumps(obj)[:260] if not isinstance(obj, str) else obj[:200]
    print('  %s %s -> %d (%.2f s) %s' % (method, path, code, time.time() - t0, short))
    return code, obj


def aws(*args, parse=True):
    out = subprocess.run(['aws', '--region', 'us-east-1', *args], check=True, capture_output=True, text=True).stdout
    return json.loads(out) if parse and out.strip() else out


def check(name, ok, detail=''):
    results.append((name, bool(ok)))
    print('%s %s %s' % ('PASS' if ok else 'FAIL', name, detail))
    return ok


def post(house, robots, diff):
    return http('POST', '/api/mission', {'house': house, 'robots': robots, 'difficulty': diff})


def poll(mid, until, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        code, r = http('GET', '/api/mission/%s?from=0' % mid)
        if code == 200 and until(r['status']):
            return r
        time.sleep(4)
    return None


def site_params(**changes):
    """Update the site stack in place, changing only the named parameters; returns the previous values."""
    st = aws('cloudformation', 'describe-stacks', '--stack-name', SITE_STACK)['Stacks'][0]
    prev = {p['ParameterKey']: p['ParameterValue'] for p in st['Parameters']}
    params = []
    for k in prev:
        params.append('ParameterKey=%s,ParameterValue=%s' % (k, changes[k]) if k in changes
                      else 'ParameterKey=%s,UsePreviousValue=true' % k)
    print('  site stack: %s' % changes)
    aws('cloudformation', 'update-stack', '--stack-name', SITE_STACK, '--use-previous-template',
        '--capabilities', 'CAPABILITY_IAM', '--parameters', *params)
    aws('cloudformation', 'wait', 'stack-update-complete', '--stack-name', SITE_STACK, parse=False)
    time.sleep(5)
    return {k: prev[k] for k in changes}


def clear_ip_counters():
    """Delete this hour's per-IP counters for this machine's v4 and v6 addresses (whichever the Function URL saw)."""
    mine = set()
    for u in ('https://checkip.amazonaws.com', 'https://api64.ipify.org'):
        try:
            mine.add(urllib.request.urlopen(u, timeout=5).read().decode().strip())
        except Exception:
            pass
    hour = int(time.time() // 3600)
    for ip in mine:
        aws('dynamodb', 'delete-item', '--table-name', TABLE, '--key', json.dumps({'pk': {'S': 'ip#%s#%d' % (ip, hour)}}),
            parse=False)
    print('  cleared per-IP counters for %s' % sorted(mine))


def fail_lines(mid, since_ms):
    r = aws('logs', 'filter-log-events', '--log-group-name', LOGS, '--start-time', str(since_ms),
            '--filter-pattern', '"RYOR_FAIL %s"' % mid)
    return [e['message'].strip() for e in r.get('events', [])]


# ------------------------------------------------------------------ fixture
def seed_fixture():
    """Add test house zz_apitest to the bucket's gallery.json (keeping every other house) and a recorded replay for
    zz_apitest__r1__easy shaped like CONTRACT.txt section 3. The page's own gallery never lists this house."""
    import os, tempfile
    tmp = tempfile.mkdtemp()
    try:
        aws('s3', 'cp', 's3://%s/ryor/gallery.json' % BUCKET, tmp + '/g.json', parse=False)
        g = json.load(open(tmp + '/g.json'))
    except subprocess.CalledProcessError:
        local = os.path.join(os.path.dirname(__file__), '../../data/runs/ryor/gallery.json')
        g = json.load(open(local))
    g['houses'] = [h for h in g['houses'] if h['id'] != FIX_HOUSE] + [{
        'id': FIX_HOUSE, 'name': 'API test box (not in the page gallery)', 'source': 'ours', 'source_id': 'test',
        'licence': 'n/a', 'rooms': 1, 'floor_m2': 100.0, 'people': 0, 'bounds': [-5.0, -2.4, 5.0, 10.0],
        'walls': [[-5, -2.4, 5, -2.4], [5, -2.4, 5, 10], [5, 10, -5, 10], [-5, 10, -5, -2.4]],
        'rooms_poly': [{'id': 'r0', 'kind': 'living', 'poly': [[-5, -2.4], [5, -2.4], [5, 10], [-5, 10]]}],
        'door': [0.0, -2.4], 'pads': [[0, 0], [0, -1.2], [0, -2.4]]}]
    json.dump(g, open(tmp + '/g.json', 'w'), separators=(',', ':'))
    aws('s3', 'cp', '--quiet', '--content-type', 'application/json', tmp + '/g.json', 's3://%s/ryor/gallery.json' % BUCKET,
        parse=False)
    key = '%s__r1__easy' % FIX_HOUSE
    for c in range(18):
        frames = [{'t': (c * 20 + k) / 2, 'robots': [{'id': 'robot_A', 'x': 0.2, 'y': 1.0 + c * 0.4, 'yaw': 90,
                                                       'state': 'explore', 'out': False}],
                   'seen': {}, 'fire': [], 'found': [], 'events': []} for k in range(20)]
        json.dump({'v': 1, 'chunk': c, 't0': c * 10, 'frames': frames}, open('%s/chunk_%04d.json' % (tmp, c), 'w'))
    json.dump({'v': 1, 'mission': 'cache', 'setup': key, 'state': 'done', 'source': 'recorded', 'chunks': 18,
               'sim_t': 180.0, 'updated': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'error': None,
               'score': {'people_found': 0, 'people_total': 0, 'time_s': 180.0, 'coverage_pct': 0.0, 'robots_out': 0,
                         'false_confirmations': 0}, 'engine': {'git': 'fixture', 'image': 'fixture'}},
              open(tmp + '/status.json', 'w'))
    aws('s3', 'cp', '--quiet', '--recursive', '--exclude', 'g.json', '--content-type', 'application/json', tmp + '/',
        's3://%s/ryor/cache/%s/' % (BUCKET, key), parse=False)
    print('seeded gallery (%d houses) and ryor/cache/%s/' % (len(g['houses']), key))


# ------------------------------------------------------------------ tests
def t_health(a):
    code, r = http('GET', '/api/mission/health')
    check('health', code == 200 and r.get('ok') is True and all(k in r for k in ('running', 'cap', 'cache_setups', 'engine')),
          'running=%s cap=%s cache_setups=%s engine=%s' % (r.get('running'), r.get('cap'), r.get('cache_setups'), r.get('engine')))


def t_bad(a):
    c1, _ = post('no_such_house', 1, 'easy')
    c2, _ = post(a.house, 4, 'easy')
    c3, _ = post(a.house, 1, 'impossible')
    c4, _ = http('POST', '/api/mission', b'{not json')
    c5, _ = http('GET', '/api/mission/..%2f..%2fgallery')
    check('bad setup gives 400', (c1, c2, c3, c4) == (400, 400, 400, 400) and c5 == 404, str((c1, c2, c3, c4, c5)))


def t_cache(a):
    code, r = http('GET', '/api/mission/cache:%s?from=12' % a.key)
    ok = code == 200 and r['status'].get('source') == 'recorded' and len(r['chunks']) == min(6, r['status']['chunks'] - 12)
    check('recorded replay served', ok, 'chunks from 12: %s' % [c['chunk'] for c in r.get('chunks', [])] if code == 200 else '')


def t_live(a):
    code, r = post(a.house, a.robots, a.difficulty)
    if not check('live POST', code == 200 and r.get('source') == 'live', str(r)):
        return
    t0 = time.time()
    first = poll(r['mission'], lambda s: s['chunks'] >= 1 or s['state'] in ('done', 'failed'), 240)
    t_first = time.time() - t0
    end = poll(r['mission'], lambda s: s['state'] in ('done', 'failed'), 600)
    if not check('live mission done', end and end['status']['state'] == 'done',
                 'first chunk after %.0f s, done after %.0f s' % (t_first, time.time() - t0)):
        return
    got = []
    for frm in range(0, end['status']['chunks'], 6):
        _, rr = http('GET', '/api/mission/%s?from=%d' % (r['mission'], frm))
        got += rr['chunks']
    ts = [f['t'] for c in got for f in c['frames']]
    sim_t = end['status']['sim_t']
    contiguous = ts == [k / 2 for k in range(len(ts))] and ts[-1] >= sim_t - 0.5
    check('live replay complete', len(got) == end['status']['chunks'] and contiguous,
          '%d chunks, %d frames, t %s..%s, sim_t %s, score %s' % (len(got), len(ts), ts[0], ts[-1], sim_t,
                                                                  end['status'].get('score')))


def t_kill(a):
    since = int(time.time() * 1000) - 5000
    code, r = post(a.house, a.robots, a.difficulty)
    if not check('kill: live POST', code == 200 and r.get('source') == 'live', str(r)):
        return
    mid = r['mission']
    mid_run = poll(mid, lambda s: s['state'] == 'running' and s['chunks'] >= 2 or s['state'] in ('done', 'failed'), 300)
    if not check('kill: mission running', mid_run and mid_run['status']['state'] == 'running',
                 'chunks %s' % (mid_run and mid_run['status']['chunks'])):
        return
    item = aws('dynamodb', 'get-item', '--table-name', TABLE, '--key', json.dumps({'pk': {'S': 'm#' + mid}}))['Item']
    task = item['task']['S']
    aws('ecs', 'stop-task', '--cluster', CLUSTER, '--task', task, '--reason', 'test_api.py kill test')
    print('  stopped task %s mid-run' % task.rsplit('/', 1)[-1])
    t0 = time.time()
    end = poll(mid, lambda s: s['state'] == 'failed', 180)
    st = end and end['status']
    check('kill: GET answers failed with fallback', st and st.get('fallback') == 'cache:' + a.key,
          'after %.0f s: state=%s error=%s fallback=%s' % (time.time() - t0, st and st['state'], st and st.get('error'),
                                                           st and st.get('fallback')))
    if st:
        code, fb = http('GET', '/api/mission/%s?from=0' % st['fallback'])
        check('kill: fallback replay served', code == 200 and fb['status']['source'] == 'recorded' and len(fb['chunks']) > 0,
              '%d chunks, first t=%s' % (len(fb.get('chunks', [])), fb.get('chunks', [{}])[0].get('t0')))
    time.sleep(8)
    lines = fail_lines(mid, since)
    check('kill: RYOR_FAIL logged', len(lines) == 1, ' | '.join(l[:160] for l in lines))


def t_cap(a):
    prev = site_params(RyorCap='0')
    try:
        code, r = post(a.house, a.robots, a.difficulty)
        check('cap fallback', code == 200 and r == {'mission': 'cache:' + a.key, 'source': 'recorded', 'reason': 'busy'},
              str(r))
    finally:
        site_params(**prev)


def t_disabled_and_rate(a):
    prev = site_params(RyorEnabled='0')
    try:
        clear_ip_counters()
        reasons = [post(a.house, a.robots, a.difficulty)[1].get('reason') for _ in range(a.ip_limit + 1)]
        check('kill switch fallback', reasons[:a.ip_limit] == ['disabled'] * a.ip_limit, str(reasons))
        check('rate-limit fallback', reasons[a.ip_limit] == 'rate_limited', str(reasons))
    finally:
        clear_ip_counters()
        site_params(**prev)


TESTS = {'health': t_health, 'bad': t_bad, 'cache': t_cache, 'kill': t_kill, 'live': t_live, 'cap': t_cap,
         'rate': t_disabled_and_rate}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed-fixture', action='store_true')
    ap.add_argument('--only', nargs='*', choices=list(TESTS))
    ap.add_argument('--house', default=FIX_HOUSE)
    ap.add_argument('--robots', type=int, default=1)
    ap.add_argument('--difficulty', default='easy')
    ap.add_argument('--ip-limit', type=int, default=6)
    a = ap.parse_args()
    a.key = '%s__r%d__%s' % (a.house, a.robots, a.difficulty)
    if a.seed_fixture:
        seed_fixture()
        sys.exit(0)
    clear_ip_counters()
    for name in a.only or list(TESTS):
        print('== %s' % name)
        TESTS[name](a)
    print('\n%d passed, %d failed' % (sum(ok for _, ok in results), sum(not ok for _, ok in results)))
    sys.exit(0 if all(ok for _, ok in results) else 1)
