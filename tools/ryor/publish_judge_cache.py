"""Publish only gate-passing current-engine replays, with status written last.

Run after rebuild_judge.py. Uses the existing feature bucket, and writes no
synthetic or failed run. The owner deploys the main Next.js site separately.
"""
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/ryor'))


def main():
    import boto3
    from judge_gate import gate
    evidence=ROOT/'handoff/judge_flow_evidence'
    summary=json.loads((evidence/'rebuild_summary.json').read_text())
    g=json.loads((Path(summary['stage'])/'data/runs/ryor/gallery.json').read_text())
    audit=json.loads((evidence/'rebuild_gate.json').read_text())
    s3=boto3.client('s3',region_name='us-east-1');bucket='smoke-rescue-ryor-runs-968000946565'
    ready=[]
    def put(name,body):s3.put_object(Bucket=bucket,Key=name,Body=json.dumps(body,separators=(',',':')).encode(),ContentType='application/json',CacheControl='no-store')
    for key,result in audit.items():
        if not result.get('passed'):continue
        folder=Path(result['path']);st=json.loads((folder/'status.json').read_text())
        chunks=[json.loads((folder/f'chunk_{i:04d}.json').read_text()) for i in range(st['chunks'])]
        h=next(h for h in g['houses'] if key.startswith(h['id']+'__'))
        proof=gate(h,[f for c in chunks for f in c['frames']],st,summary['commit'],h['source']=='HouseExpo')
        if not proof['passed']:raise ValueError(key+': gate changed since rebuild')
        prefix='ryor/cache/'+key+'/'
        for i,c in enumerate(chunks):put(prefix+f'chunk_{i:04d}.json',c)
        st['gate']=proof
        put(prefix+'status.json',st);ready.append(key)
    # A failed setup gets no new status and is absent from the manifest. Old engine statuses
    # are rejected by the API's engine check even if an old object still exists privately.
    put('ryor/gallery.json',g)
    put('ryor/cache_manifest.json',{'ready':len(ready),'total':15,'setups':ready,'engine':summary['commit'][:12]})
    (evidence/'published_cache.json').write_text(json.dumps({'ready':ready,'engine':summary['commit']},indent=2)+'\n')
    print('Published',len(ready),'current-engine recordings; failed runs withheld')

if __name__=='__main__':main()
