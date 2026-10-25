#!/usr/bin/env python3
"""Score completed ranch headless runs. Evaluator-only, never read by a robot."""
import base64
import json
import math
from pathlib import Path
from headless_report import inside

ROOT=Path(__file__).resolve().parents[1]

def score(name, suffix=""):
    run=ROOT/'data/runs/houses_25oct'/(name+suffix)
    result=json.loads((run/'result.json').read_text())
    world=json.loads((ROOT/result['world']).read_text())
    rows=[json.loads(line) for line in (run/'trace.jsonl').read_text().splitlines() if line.strip()]
    poses=[r for r in rows if r['kind']=='pose']
    def room_at(p): return next((r['id'] for r in world['rooms'] if inside(r['poly'],p)),None)
    entered={room_at((r['x'],r['y'])) for r in poses}-{None}
    confirmed=[r for r in rows if r['kind']=='survivor_confirmed' and 'x_m' in r.get('numbers',{})]
    def match(event,person):
        n=event['numbers'];return math.dist((n['x_m'],n['y_m']),person['at'])<=1.05
    found=[p['id'] for p in world['people'] if any(match(e,p) for e in confirmed)]
    false=sum(not any(match(e,p) for p in world['people']) for e in confirmed)
    masks=[]
    for m in result['maps']:
        masks.append((m['grid_size'],m['cell_m'],m['origin'],base64.b64decode(m['own_searched'])))
    def searched(p):
        for n,c,(ox,oy),bits in masks:
            x,y=math.floor((p[0]-ox)/c),math.floor((p[1]-oy)/c)
            if 0<=x<n and 0<=y<n and bits[y*n+x]:return True
        return False
    # Common 0.1 m evaluator grid, not a room-averaged percentage. Both the
    # whole room rectangle and the usable floor are reported with their counts.
    per_room={}; gross_hit=gross_total=free_hit=free_total=0
    segments=[s for s in world['segments'] if s[4]<=.2 and s[5]>=1.5]
    def wall_distance(p,s):
        ax,ay,bx,by=s[:4]; dx,dy=bx-ax,by-ay
        t=max(0,min(1,((p[0]-ax)*dx+(p[1]-ay)*dy)/max(1e-12,dx*dx+dy*dy)))
        return math.dist(p,(ax+t*dx,ay+t*dy))
    for room in world['rooms']:
        xs=[p[0] for p in room['poly']];ys=[p[1] for p in room['poly']]
        counts=[0,0,0,0]
        for iy in range(math.floor(min(ys)*10),math.ceil(max(ys)*10)):
            for ix in range(math.floor(min(xs)*10),math.ceil(max(xs)*10)):
                p=((ix+.5)/10,(iy+.5)/10)
                if not inside(room['poly'],p):continue
                hit=searched(p); counts[0]+=hit;counts[1]+=1
                if any(q['height_m']>=.15 and inside(q['poly'],p) for q in world['prisms']):continue
                # Headless wall lines are zero thickness. Exclude 0.125 m
                # conservatively on each side, the baseline exterior half-width.
                if any(wall_distance(p,s)<.125 for s in segments):continue
                counts[2]+=hit;counts[3]+=1
        gh,gt,fh,ft=counts
        gross_hit+=gh;gross_total+=gt;free_hit+=fh;free_total+=ft
        per_room[room['id']]={'entered':room['id'] in entered,'gross_floor_pct':round(100*gh/max(1,gt),2),'usable_floor_pct':round(100*fh/max(1,ft),2),'usable_samples_searched':fh,'usable_samples':ft}
    robots={}
    for rid in sorted({p['robot'] for p in poses}):
        path=[p for p in poses if p['robot']==rid]; last=path[-1]
        visited=sorted({room_at((p['x'],p['y'])) for p in path}-{None})
        outside=room_at((last['x'],last['y'])) is None
        completed=[p['t'] for p in path if p['state']=='complete' and room_at((p['x'],p['y'])) is None]
        robots[rid]={'walked_out_at_s':min(completed) if completed and visited else None,'rooms_entered':visited,'final_pose_gate':[last['x'],last['y']],'final_state':last['state'],'outside_at_end':outside,'walked_out':bool(visited) and outside and last['state']=='complete'}
    return {'run':str(run.relative_to(ROOT)),'house':name,'duration_s':result['duration_s'],'rooms_entered':len(entered),'rooms_total':len(world['rooms']),'usable_floor_coverage_pct':round(100*free_hit/max(1,free_total),2),'gross_floor_coverage_pct':round(100*gross_hit/max(1,gross_total),2),'usable_samples_searched':free_hit,'usable_samples':free_total,'people_confirmed':len(found),'people_total':len(world['people']),'confirmed_ids':found,'false_confirmation_events':false,'both_walked_out':all(r['walked_out'] for r in robots.values()),'robots':robots,'rooms':per_room}

if __name__=='__main__':
    for name in ('family_ranch','long_ranch'):
        import sys
        suffix = sys.argv[1] if len(sys.argv)>1 else ''
        if suffix == '_final': suffix = '_wide' if name == 'family_ranch' else '_front_living'
        output=score(name, suffix)
        (ROOT/f'handoff/houses_evidence/metrics_{name}{sys.argv[2] if len(sys.argv)>2 else ""}.json').write_text(json.dumps(output,indent=2)+'\n')
        print(json.dumps(output,indent=2))
