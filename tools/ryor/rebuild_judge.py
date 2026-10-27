"""Snapshot overnight-25oct and rebuild the five current houses at EASY.

Runs in an isolated /tmp copy. The recording checkout is read-only. Failed runs
remain in the evidence directory and never replace a site's checked fallbacks.
"""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / 'smoke-rescue'
EVIDENCE = ROOT / 'handoff/backend_evidence'
DATA = ROOT / 'data/runs/ryor-judge'
HOUSES = [('family_ranch', 'Three-bed ranch', 'scenario_family_ranch_full.json'),
          ('long_ranch', 'L-shaped ranch', 'scenario_long_ranch_full.json'),
          ('onedoor_934', 'One-door house', 'scenario_house_onedoor_seed934_full.json')]
EXTRAS = [('hx_46bfe067cc', 'Bungalow'), ('hx_cb9d3d6726', 'Corner house')]
DOTNET = '/Applications/Unity/Hub/Editor/6000.6.6f1/Unity.app/Contents/Resources/Scripting/DotNetSdk/dotnet'


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(',', ':')))


def snapshot():
    marker = SOURCE / 'handoff/HOUSES8_MERGED.txt'
    commit = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'overnight-25oct'], text=True).strip()
    stage = Path(tempfile.mkdtemp(prefix='ryor-judge-'))
    archive = stage / 'source.tar'
    with archive.open('wb') as out:
        subprocess.run(['git', '-C', str(SOURCE), 'archive', commit, 'config', 'unity/SmokeRescue/Assets/SmokeRescue', 'tools'], stdout=out, check=True)
    with tarfile.open(archive) as t:
        t.extractall(stage, filter='data')
    archive.unlink()
    # Use the reviewed replay/build adapter with the newly pinned simulator sources.
    shutil.copytree(ROOT / 'tools/ryor', stage / 'tools/ryor', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('engine', 'site-next', '__pycache__', 'page_shots'))
    shutil.copytree(ROOT / 'tools/ryor/engine', stage / 'tools/ryor/engine', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('bin', 'obj', 'gen'))
    meta = dict(commit=commit, stage=str(stage), marker=marker.read_text() if marker.exists() else None,
                brain_config_sha256=hashlib.sha256((stage / 'config/brain.json').read_bytes()).hexdigest())
    write(EVIDENCE / 'engine_snapshot.json', meta)
    return stage, commit


def record_engine(stage):
    meta=json.loads((EVIDENCE/'engine_snapshot.json').read_text())
    files=['config/brain.json','tools/ryor/entry.py','tools/ryor/patch_headless.py',
           'tools/ryor/replay.py','tools/ryor/engine/bin/RyorRunner.dll']
    files += [str(p.relative_to(stage)) for p in sorted((stage/'unity/SmokeRescue/Assets/SmokeRescue/Brain').glob('*.cs'))]
    meta['sha256']={f:hashlib.sha256((stage/f).read_bytes()).hexdigest() for f in files}
    meta['local_precompute_hard_limit_s']=3600
    meta['batch_started_unix_s']=time.time()
    write(EVIDENCE/'engine_snapshot.json',meta)


def gallery(stage):
    sys.path[:0] = [str(stage / 'tools'), str(stage / 'tools/bench'), str(ROOT / 'tools/ryor')]
    import headless_world
    import make_requests
    # Imported module constants resolve config and assets against the snapshot.
    spec = importlib.util.spec_from_file_location('judge_gallery', stage / 'tools/ryor/gallery.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    out = stage / 'data/runs/ryor'
    for name in ('worlds', 'houses', 'setups', 'fires'):(out / name).mkdir(parents=True, exist_ok=True)
    original = json.loads((ROOT / 'data/runs/ryor/gallery.json').read_text())
    entries, worlds = [], {}
    for hid, name, scenario in HOUSES:
        p = stage / 'config' / scenario
        if not p.exists():
            raise FileNotFoundError(p)
        sc = headless_world.load_scenario(p)
        sc.setdefault('floorplan_file', 'floorplan.json')
        sc['fire'] = {'config_file': 'fire.json', **sc.get('fire', {})}
        world = headless_world.compile_world(p)
        fp = make_requests.fire_grid_plan(sc, out, hid)
        if fp:world['fire']['floorplan'] = fp
        h = mod.house_entry(hid, name, 'ours', p.name, 'ours', world)
        # The merge marker cancels the eight-room replacement. Keep the committed current houses.
        entries.append(h);worlds[hid] = world
    for hid, name in EXTRAS:
        h = dict(next(h for h in original['houses'] if h['id'] == hid));h['name'] = name
        if not 7 <= h['rooms'] <= 8:raise ValueError(hid + ': requires 7 to 8 rooms')
        world = json.loads((ROOT / 'data/runs/ryor/worlds' / (hid + '__r2.json')).read_text())
        for field in ('floorplan','config'):
            old = (ROOT / 'config' / world['fire'][field]).resolve()
            new = out / 'houses' / old.name
            shutil.copyfile(old, new)
            world['fire'][field] = os.path.relpath(new, stage / 'config')
        entries.append(h);worlds[hid] = world
    # Preserve each generator's people and fire identically across all robot counts.
    for hid, world in worlds.items():
        for n in (1,2,3):
            w = mod.with_robots(world,n)
            wp = out / 'worlds' / f'{hid}__r{n}.json';write(wp,w)
            key = f'{hid}__r{n}__easy'
            req = {'runs':[{'world':str(wp.relative_to(stage)),'out':'','duration_s':360,'time_limit':True,
                           'fire':True,'robot_collision':True,'capsule_walls':True,'settings':{},
                           'head_start_s':0,'visibility_m':6.0,'replay_every_s':0.5,'parallel_robots':True}],
                   'ryor':{'house':hid,'robots':n,'difficulty':'easy'}}
            write(out / 'setups' / (key+'.json'),req)
    g = dict(original, mission_s=360, houses=entries, setups=[f'{h["id"]}__r{n}__easy' for h in entries for n in (1,2,3)], difficulties={'easy':mod.DIFFICULTIES['easy']})
    write(out / 'gallery.json',g)
    # Attach fire origins and placement geometry for captions and the mission API.
    spec = importlib.util.spec_from_file_location('judge_enrich', ROOT / 'tools/ryor/enrich_gallery.py')
    enrich=importlib.util.module_from_spec(spec);spec.loader.exec_module(enrich)
    enrich.ROOT=stage;enrich.DATA=out;enrich.enrich(g)
    write(out / 'gallery.json',g)
    return g


def simulate(stage, commit, key):
    dest = DATA / commit[:12] / key
    env = dict(os.environ, SETUP=key, MISSION_ID='cache', LOCAL_OUT=str(dest), RUN_DIR=str(dest/'run'), BUCKET='',
               ENGINE_GIT=commit, ENGINE_TAG=commit[:12], DOTNET=DOTNET, HARD_LIMIT_S='3600', TERRA_ENABLED='0')
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/'engine.log').open('w') as log:
        p=subprocess.run([sys.executable,str(stage/'tools/ryor/entry.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
    return key, p.returncode, dest


def inspect_run(g, commit, key, code, dest):
    from judge_gate import gate
    from export_v2 import room_grid
    from replay import inside
    if code:return {'passed':False,'error':'engine exited '+str(code),'log':str(dest/'engine.log')},None
    status=json.loads((dest/'status.json').read_text())
    house=next(h for h in g['houses'] if key.startswith(h['id']+'__'))
    chunks=[json.loads((dest/f'chunk_{i:04d}.json').read_text()) for i in range(status['chunks'])]
    frames=[f for c in chunks for f in c['frames']]
    result=gate(house,frames,status,commit)
    if len(frames)!=int(g['mission_s']*2)+1 or [f['t'] for f in frames]!=[i*.5 for i in range(int(g['mission_s']*2)+1)]:
        result.update(passed=False,error='Incomplete frame sequence')
    result['score']=status['score'];result['path']=str(dest)
    result['rooms_total']=len(house['rooms_poly'])
    result['final_robots']=frames[-1]['robots']
    result['frames']=len(frames)
    result['causes']=[]
    result['never_entered_robots']=[p['id'] for p in frames[-1]['robots'] if not any(
        inside(room['poly'],q['x'],q['y']) for f in frames for q in f['robots'] if q['id']==p['id'] for room in house['rooms_poly'])]
    result['inside_at_end']=[p['id'] for p in frames[-1]['robots'] if any(inside(room['poly'],p['x'],p['y']) for room in house['rooms_poly'])]
    if result['missing_rooms']:result['causes'].append('Rooms never entered: '+', '.join(result['missing_rooms']))
    if result['inside_at_end']:result['causes'].append('Robots still inside at mission end: '+', '.join(result['inside_at_end']))
    if result['never_entered_robots']:result['causes'].append('No verified return because the robot never entered: '+', '.join(result['never_entered_robots']))
    if result['false_confirmations']:result['causes'].append(str(result['false_confirmations'])+' false confirmations')
    if not result['engine_matches']:result['causes'].append('Engine source does not match the pinned commit')
    if not result['passed']:return result,None
    seen=set();mapped=[]
    cells=[room_grid(room['poly'],g['grid'],inside) for room in house['rooms_poly']]
    for f in frames:
        for s in f['seen'].values():seen.update(s)
        f['mapped']=[int(100*len(seen & ids)/max(1,total)) for ids,total in cells]
        f.pop('seen',None)
    recording=dict(setup=key,duration_s=g['mission_s'],difficulty='easy',engine=status['engine'],frames=frames,score=status['score'],
                   validated=True,rooms_searched=sum(p>=80 for p in frames[-1]['mapped']),gate=result)
    # Keep only the small proof fields in public payloads.
    recording['gate']={k:result[k] for k in ('all_rooms_entered','all_out','false_confirmations','engine_git')}
    write(dest/'presentation.json',recording)
    return result,recording


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--site',type=Path,default=Path('/Volumes/Tashon/opencv/showcase/site-next'));args=p.parse_args()
    if args.resume:
        meta=json.loads((EVIDENCE/'engine_snapshot.json').read_text())
        stage,commit=Path(meta['stage']),meta['commit']
        g=json.loads((stage/'data/runs/ryor/gallery.json').read_text())
    else:
        stage,commit=snapshot()
        if not args.prepare_only:g=gallery(stage)
        subprocess.run([str(stage/'tools/ryor/build_engine.sh')],check=True)
        record_engine(stage)
    if args.prepare_only:
        print('Current source engine compiled at',stage,flush=True);return
    results={};recordings={}
    keys=g['setups']
    def run(key):
        dest=DATA/commit[:12]/key
        status=dest/'status.json'
        if args.resume and status.exists():
            st=json.loads(status.read_text())
            if st['state']=='done' and st['engine']['git']==commit:return key,0,dest
        return simulate(stage,commit,key)
    # Three workers bound local CPU use; each team already parallelizes sensing.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run,key) for key in keys]
        for future in concurrent.futures.as_completed(futures):
            key,code,dest=future.result();result,recording=inspect_run(g,commit,key,code,dest)
            results[key]=result
            if recording:recordings[key]=recording
            write(EVIDENCE/'rebuild_gate.json',results)
            print(key,json.dumps(result),flush=True)
    dest=args.site/'public/rescue-data'
    for h in g['houses']:
        h['mission_s']=g['mission_s']
        h['available_replays']=[n for n in (1,2,3) if f'{h["id"]}__r{n}__easy' in recordings]
        h['scores']={str(n):recordings[f'{h["id"]}__r{n}__easy']['score'] | {'rooms_searched':recordings[f'{h["id"]}__r{n}__easy']['rooms_searched']}
                     for n in (1,2,3) if f'{h["id"]}__r{n}__easy' in recordings}
    # Delete retired and failed presentation files, including the old-engine fallbacks.
    for old in dest.glob('*__r*__easy.json'):old.unlink()
    for key,r in recordings.items():write(dest/(key+'.json'),r)
    write(dest/'gallery.json',dict(houses=g['houses'],mission_s=g['mission_s'],difficulty='easy',citation=g['citation']))
    config=json.loads((dest/'missions.json').read_text());config['engine']=commit;write(dest/'missions.json',config)
    # Keep a versioned copy of data only. The UI source belongs to its owner.
    saved=ROOT/'tools/ryor/site-next/public/rescue-data'
    if saved.resolve() != dest.resolve():
        saved.mkdir(parents=True,exist_ok=True)
        for old in saved.glob('*__r*__easy.json'):old.unlink()
        for name in ['gallery.json','missions.json']+[key+'.json' for key in recordings]:
            shutil.copyfile(dest/name,saved/name)
    print(f'{len(recordings)}/15 passed; failed runs withheld. Snapshot {stage}',flush=True)
    write(EVIDENCE/'rebuild_summary.json',dict(commit=commit,stage=str(stage),passed=len(recordings),total=15))

if __name__=='__main__':main()
