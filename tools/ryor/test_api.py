#!/usr/bin/env python3
"""Check the public mission API against real gallery data. Never upload synthetic public replays.

Run python3 tools/ryor/test_api.py for read-only checks, or add --live to launch
one real mission. Browser kill/placement checks are in browser_checks.cjs;
check_rate.py and test_core.py cover the limits and invalid placements.
"""
import argparse
import json
import time
import urllib.error
import urllib.request

URL = 'https://ajfjm7evl33gu6ptprty6hfthq0ukygx.lambda-url.us-east-1.on.aws'
POLL_S = 2  # Match the page's bounded polling cadence.
TIMEOUT_S = 700  # Cover the 600 s API hard limit and recorded-fallback delivery.


def http(method, path, body=None):
    req = urllib.request.Request(URL + path, method=method, headers={'content-type': 'application/json'},
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=70) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    code, health = http('GET', '/api/mission/health')
    assert code == 200 and health['ok']
    print('PASS public health', json.dumps(health))
    setup = {'house': 'family_ranch', 'robots': 2, 'difficulty': 'medium', 'terra': False}
    key = 'family_ranch__r2__medium'
    code, first = http('GET', '/api/mission/cache:' + key)
    assert code == 200 and first['status']['state'] == 'done' and first['status']['validated']
    chunks = list(first['chunks'])
    for start in range(len(chunks), first['status']['chunks'], 6):
        code, more = http('GET', '/api/mission/cache:' + key + '?from=%d' % start)
        assert code == 200
        chunks.extend(more['chunks'])
    times = [frame['t'] for chunk in chunks for frame in chunk['frames']]
    assert times == [n / 2 for n in range(361)]
    print('PASS real recorded replay, 361 consecutive frames, time 0 to 180 s')
    code, error = http('POST', '/api/mission', dict(setup, robots=4))
    assert code == 400 and 'robots' in error['error']
    code, error = http('POST', '/api/mission/validate', dict(setup, placement={'people': [[4, 5.5]], 'fire': [7.75, 2]}))
    assert code == 400 and '0.9 m path' in error['error']
    print('PASS invalid robot count and blocked 0.9 m door path')
    if not args.live:
        return
    started = time.monotonic()
    code, mission = http('POST', '/api/mission', setup)
    assert code == 200 and mission['source'] == 'live', mission
    first_frame = None
    while time.monotonic() - started < TIMEOUT_S:
        code, answer = http('GET', '/api/mission/' + mission['mission'] + '?from=19')
        assert code == 200
        status = answer['status']
        if status['chunks'] and first_frame is None:
            first_frame = round(time.monotonic() - started, 2)
        if status['state'] == 'failed':
            raise AssertionError(status)
        if status['state'] == 'done':
            print('PASS live mission', json.dumps({'mission': mission['mission'], 'first_chunk_s': first_frame,
                  'total_s': round(time.monotonic() - started, 2), 'status': status}))
            return
        time.sleep(POLL_S)
    raise AssertionError('Live mission exceeded the test deadline')


if __name__ == '__main__':
    main()
