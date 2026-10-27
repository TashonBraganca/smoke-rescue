"""Check every frame of the actual edited mission and its exact prepared fallback."""
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/ryor')]
os.environ.update(RYOR_BUCKET='smoke-rescue-ryor-runs-968000946565', RYOR_TABLE='smoke-rescue-ryor',
                  RYOR_CLUSTER='smoke-rescue', RYOR_FAMILY='smoke-rescue-ryor', RYOR_SUBNETS='unused',
                  RYOR_SG='unused', AWS_DEFAULT_REGION='us-east-1')
spec = importlib.util.spec_from_file_location('ryor_api', ROOT / 'deploy/aws/site_lambda/ryor_api.py')
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
gallery = json.loads((ROOT / 'data/runs/ryor/gallery.json').read_text())
a.gallery = lambda: {'houses': {h['id']: h for h in gallery['houses']}, 'mission_s': 180}
mid = '6adeaa48548633fae1'
status = a.s3_json('ryor/missions/' + mid + '/status.json')
result = {}
for label, prefix in [('live', 'ryor/missions/' + mid + '/'), ('fallback', a.cache_prefix(status['setup']))]:
    st = a.s3_json(prefix + 'status.json')
    why = a.validate({'setup': status['setup']}, st, prefix)
    assert why is None, why
    chunks = a.chunks(prefix, 0, st['chunks'])
    frames = [f for c in chunks for f in c['frames']]
    events = [e for f in frames for e in f.get('events', [])]
    result[label] = dict(setup=st['setup'], frames=len(frames), first_t=frames[0]['t'], last_t=frames[-1]['t'],
                         score=st['score'], wall=st['wall'], engine=st['engine'],
                         decisions=sum(e['kind'] == 'agent_decision' for e in events),
                         orders=sum(e['kind'] == 'agent_order' for e in events),
                         confirmations=sum(e['kind'] == 'survivor_confirmed' for e in events),
                         bedrooms={r['id']: r['rect'] for r in st['layout'] if r['id'] in ('fr_bed1', 'fr_bed2')},
                         validation='Every frame passed time, bounds, confirmation provenance, score and exact-geometry identity checks.')
assert result['live']['decisions'] and result['live']['orders']
assert result['fallback']['decisions'] == 0
(ROOT / 'handoff/ryor_evidence/final_edited_audit.json').write_text(json.dumps(result, indent=1))
print(json.dumps(result, indent=1))
