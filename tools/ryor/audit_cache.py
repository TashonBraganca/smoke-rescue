"""Validate every frame of every default recording, then mark only checked results ready.

This uses the same API validator as live missions. Keep the per-setup verdict and
content digest in handoff/ryor_evidence/cache_audit.json. Run again after precompute.
"""
import hashlib
import inspect
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/ryor')]
for k, v in dict(RYOR_BUCKET='smoke-rescue-ryor-runs-968000946565', RYOR_TABLE='smoke-rescue-ryor',
                 RYOR_CLUSTER='smoke-rescue', RYOR_FAMILY='smoke-rescue-ryor', RYOR_SUBNETS='unused', RYOR_SG='unused',
                 AWS_DEFAULT_REGION='us-east-1').items():
    os.environ[k] = v
spec = importlib.util.spec_from_file_location('ryor_api', ROOT / 'deploy/aws/site_lambda/ryor_api.py')
api = importlib.util.module_from_spec(spec); spec.loader.exec_module(api)
gallery = json.loads((ROOT / 'data/runs/ryor/gallery.json').read_text())
api.gallery = lambda: {'houses': {h['id']: h for h in gallery['houses']},
                       'difficulties': set(gallery['difficulties']), 'mission_s': gallery['mission_s']}
BUCKET = os.environ['RYOR_BUCKET']
REPORT = ROOT / 'handoff/ryor_evidence/cache_audit.json'
PREVIOUS = json.loads(REPORT.read_text()) if REPORT.exists() else {}
VALIDATOR_SIGNATURE = hashlib.sha256(inspect.getsource(api.validate).encode()).hexdigest()


def audit(key):
    prefix = 'ryor/cache/' + key + '/'
    status = api.s3_json(prefix + 'status.json')
    if not status or status.get('state') != 'done':
        return key, {'valid': False, 'error': 'not ready'}
    status_digest = hashlib.sha256(json.dumps({k: v for k, v in status.items() if k != 'validated'},
                                             sort_keys=True).encode()).hexdigest()
    old = PREVIOUS.get(key, {})
    if old.get('valid') and old.get('status_digest') == status_digest and old.get('validator') == VALIDATOR_SIGNATURE:
        return key, old  # Immutable finished recordings need one full audit per published status, not repeated downloads.
    chunks = api.chunks(prefix, 0, int(status['chunks']))
    # Avoid downloading a second time during the complete validation pass.
    # Each worker calls a private validator module so its chunk reader cannot race another setup.
    namespace = dict(api.validate.__globals__, chunks=lambda *args: chunks)
    import types
    validate = types.FunctionType(api.validate.__code__, namespace)
    why = validate({'setup': key}, status, prefix)
    if why:
        return key, {'valid': False, 'error': why}
    digest = hashlib.sha256(json.dumps(chunks, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    frames = [f for c in chunks for f in c['frames']]
    if not status.get('validated'):
        status['validated'] = True
        api.s3.put_object(Bucket=BUCKET, Key=prefix + 'status.json', Body=json.dumps(status).encode(),
                          ContentType='application/json', CacheControl='no-store')
    return key, {'valid': True, 'frames': len(frames), 'first_t': frames[0]['t'], 'last_t': frames[-1]['t'],
                 'markers': sum(len(f.get('found', [])) for f in frames), 'sha256': digest,
                 'status_digest': status_digest,
                 'validator': VALIDATOR_SIGNATURE,
                 'wall': status.get('wall'), 'engine': status.get('engine')}


if __name__ == '__main__':
    keys = sorted(p.stem for p in (ROOT / 'data/runs/ryor/setups').glob('*.json'))
    with ThreadPoolExecutor(max_workers=4) as pool:  # Bound S3 pressure while inherited precompute continues.
        results = dict(pool.map(audit, keys))
    path = REPORT
    path.write_text(json.dumps(results, indent=1))
    passed = sum(r['valid'] for r in results.values())
    manifest = {'ready': passed, 'total': len(keys), 'setups': [k for k, r in results.items() if r['valid']],
                'checked_at': datetime.now(timezone.utc).isoformat()}
    api.s3.put_object(Bucket=BUCKET, Key='ryor/cache_manifest.json', Body=json.dumps(manifest).encode(),
                      ContentType='application/json', CacheControl='no-store')
    print('%d/%d complete replays validated; %s' % (passed, len(keys), path))
    print('Failures:', {k: r['error'] for k, r in results.items() if not r['valid'] and r['error'] != 'not ready'})
