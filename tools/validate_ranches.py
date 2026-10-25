#!/usr/bin/env python3
"""WORLD-side ranch geometry checks, including 0.9 m paths and body envelopes."""
from collections import deque
import json
import math
from pathlib import Path
from validate_building import inside, overlap, footprint, boundary_side, room_paths

ROOT = Path(__file__).resolve().parents[1]

def furniture_clear(lm, room, plan, fire, previous):
    box = footprint(lm,fire)
    if not inside(box[:2],room['rect'],.30) or not inside(box[2:],room['rect'],.30): return False
    if any(overlap([box[0]-.1,box[1]-.1,box[2]+.1,box[3]+.1],footprint(o,fire)) for o in previous): return False
    # Rectangular 0.9 m door approach, 1.2 m into the room. Avoid the generator's
    # enclosing-circle approximation, which rejects normal beds in 4 m rooms.
    for d in plan['doors']:
        if room['id'] not in d['between']: continue
        x,z=d['at']; side=boundary_side(d,room['rect'])
        keep={'south':[x-.45,z,x+.45,z+1.2],'north':[x-.45,z-1.2,x+.45,z],
              'west':[x,z-.45,x+1.2,z+.45],'east':[x-1.2,z-.45,x,z+.45]}[side]
        if overlap(box,keep): return False
    return not room_paths(room,[d for d in plan['doors'] if room['id'] in d['between']],previous+[lm],fire)

def validate(plan,fire,sc):
    rooms={r['id']:r for r in plan['rooms']}; fp=plan['footprint']
    assert len(rooms)==len(plan['rooms']), 'Duplicate room ids'
    area=0
    for i,r in enumerate(plan['rooms']):
        a,b,c,d=r['rect']; assert all(abs(v/.25-round(v/.25))<1e-6 for v in (a,b,c,d)), 'Room edge off fire grid'
        assert a<c and b<d and inside([a,b],fp) and inside([c,d],fp)
        assert not any(overlap(r['rect'],t['rect']) for t in plan['rooms'][i+1:]), 'Overlapping rooms'
        area+=(c-a)*(d-b)
    assert abs(area-(fp[2]-fp[0])*(fp[3]-fp[1]))<1e-6, 'Untiled footprint'
    graph={rid:set() for rid in ['outside',*rooms]}
    assert sum('outside' in d['between'] for d in plan['doors'])==1
    for d in plan['doors']:
        assert d['width']>=.9 and d['state'] in ('open','removed'), 'Blocked or narrow door'
        a,b=d['between']; assert a!=b and a in graph and b in graph
        sides=[]
        for rid in (a,b):
            side=boundary_side(d,fp if rid=='outside' else rooms[rid]['rect']); assert side, 'Door misses room boundary'
            sides.append(side)
        if 'outside' not in (a,b): assert tuple(sides) in [('east','west'),('west','east'),('north','south'),('south','north')]
        graph[a].add(b); graph[b].add(a)
    seen={'outside'}; q=deque(seen)
    while q:
        for rid in graph[q.popleft()]-seen: seen.add(rid); q.append(rid)
    assert len(seen)==len(graph),'Unreachable room'
    for r in plan['rooms']:
        furniture=[l for l in plan['landmarks'] if l['room']==r['id']]
        for i,lm in enumerate(furniture): assert furniture_clear(lm,r,plan,fire,furniture[:i]), f'Furniture blocks {r["id"]}'
        assert not room_paths(r,[d for d in plan['doors'] if r['id'] in d['between']],furniture,fire)
    assert 4<=len(sc['survivors'])<=5 and len({s['room'] for s in sc['survivors']})==len(sc['survivors'])
    for s in sc['survivors']:
        assert inside(s['at'],rooms[s['room']]['rect'],.6), 'Person near a wall'
        assert all(not inside(s['at'],[a-.9,b-.9,c+.9,d+.9]) for a,b,c,d in [footprint(l,fire) for l in plan['landmarks']]), 'Person intersects furniture envelope'
        assert all(math.dist(s['at'],d['at'])>=1.2 for d in plan['doors']), 'Person at door'
    assert all(not inside(r['spawn']['at'],fp) for r in sc['robots'])
    assert sc['search_area']==fp and sc['duration_s']==600 and sc['search_s']==480 and sc['time_limit']
    assert all(r['control']=='autonomous' and r['route']==[] for r in sc['robots'])
    assert 1<=len(fire['ignitions'])<=2
    for f in fire['ignitions']:
        assert inside(f['at'],rooms[f['room']]['rect'],.25)
        assert any(l['room']==f['room'] and inside(f['at'],footprint(l,fire)) and fire['materials'][fire['furniture'][l['kind']]['material']]['fuel_kg_m2']>0 for l in plan['landmarks'])
    def check_reasons(obj):
        if isinstance(obj,dict):
            for key,value in obj.items():
                if key.endswith('_reason'): continue
                if isinstance(value,(int,float)) and not isinstance(value,bool) or isinstance(value,list) and any(isinstance(x,(int,float)) for x in value): assert obj.get(key+'_reason'), f'Missing {key}_reason'
                check_reasons(value)
        elif isinstance(obj,list):
            for value in obj: check_reasons(value)
    for obj in (plan,fire,sc): check_reasons(obj)
    return area

if __name__=='__main__':
    for name in ('family_ranch','long_ranch'):
        p,f,s=[json.loads((ROOT/'config'/file).read_text()) for file in (f'floorplan_{name}.json',f'fire_{name}.json',f'scenario_{name}_full.json')]
        area=validate(p,f,s)
        print(f'{name}: PASS, {len(p["rooms"])} rooms, {area:g} m2, one entrance, all doors >= 0.9 m, all rooms reachable by 0.9 m paths, five people in separate rooms clear of furniture, numeric reasons present')
