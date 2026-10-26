#!/usr/bin/env python3
"""WORLD-side ranch geometry checks, including 0.9 m paths and body envelopes."""
from collections import deque
import json
import math
from pathlib import Path
from validate_building import inside, overlap, footprint, boundary_side

ROOT = Path(__file__).resolve().parents[1]

def room_paths(room, doors, obstacles, fire, targets=(), require_all=False):
    """Connect all door approaches to free interior floor on a 0.1 m grid.

    A fixture may occupy the mathematical centre. Choose the closest free
    sample instead. Walls retain the original conservative 0.75 m inset.
    """
    x0,z0,x1,z1=room['rect']
    step,inset,radius=.1,.75,.45
    nx,nz=round((x1-x0-2*inset)/step),round((z1-z0-2*inset)/step)
    if nx<0 or nz<0: return ['Room too narrow for a 0.9 m path']
    boxes=[footprint(o,fire) for o in obstacles]
    def point(k): return x0+inset+k[0]*step,z0+inset+k[1]*step
    def free(k):
        x,z=point(k)
        return all(max(a-x,0,x-c)**2+max(b-z,0,z-d)**2 >= radius**2-1e-6 for a,b,c,d in boxes)
    keys={(i,j) for i in range(nx+1) for j in range(nz+1) if free((i,j))}
    if not keys: return ['No 0.9 m clear interior floor']
    anchor=min(keys,key=lambda k:((k[0]-nx/2)**2+(k[1]-nz/2)**2,k))
    seen={anchor}; queue=deque(seen)
    while queue:
        i,j=queue.popleft()
        for k in ((i-1,j),(i+1,j),(i,j-1),(i,j+1)):
            if k in keys and k not in seen: seen.add(k);queue.append(k)
    failures=[]
    for door in doors:
        side=boundary_side(door,room['rect'])
        if side is None:
            failures.append(door['id']+' has no boundary approach');continue
        dx,dz=dict(south=(0,inset),north=(0,-inset),west=(inset,0),east=(-inset,0))[side]
        k=(max(0,min(nx,round((door['at'][0]+dx-x0-inset)/step))),max(0,min(nz,round((door['at'][1]+dz-z0-inset)/step))))
        if k not in seen: failures.append(door['id']+' has no 0.9 m path to clear interior floor')
    if require_all and seen!=keys:
        failures.append('Furniture or a body divides the 0.9 m clear floor')
    for target in targets:
        a,b,c,d=footprint(target,fire)
        if not any(max(a-point(k)[0],0,point(k)[0]-c)**2+max(b-point(k)[1],0,point(k)[1]-d)**2<=.95**2 for k in seen):
            failures.append('No 0.9 m path to within 0.95 m of a person body')
    return failures

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
    return not room_paths(room,[d for d in plan['doors'] if room['id'] in d['between']],previous+[lm],fire,require_all='footprint_rects' in plan)

def validate(plan,fire,sc):
    rooms={r['id']:r for r in plan['rooms']}; fp=plan['footprint']
    assert len(rooms)==len(plan['rooms']), 'Duplicate room ids'
    area=0
    for i,r in enumerate(plan['rooms']):
        a,b,c,d=r['rect']; assert all(abs(v/.25-round(v/.25))<1e-6 for v in (a,b,c,d)), 'Room edge off fire grid'
        assert a<c and b<d and inside([a,b],fp) and inside([c,d],fp)
        assert not any(overlap(r['rect'],t['rect']) for t in plan['rooms'][i+1:]), 'Overlapping rooms'
        area+=(c-a)*(d-b)
    tiles=plan.get('footprint_rects',[fp])
    assert all(inside(t[:2],fp) and inside(t[2:],fp) and t[0]<t[2] and t[1]<t[3] for t in tiles), 'Invalid footprint tile'
    assert all(not overlap(t,u) for i,t in enumerate(tiles) for u in tiles[i+1:]), 'Overlapping footprint tiles'
    assert all(abs(sum(max(0,min(r['rect'][2],t[2])-max(r['rect'][0],t[0]))*max(0,min(r['rect'][3],t[3])-max(r['rect'][1],t[1])) for t in tiles)-(r['rect'][2]-r['rect'][0])*(r['rect'][3]-r['rect'][1]))<1e-6 for r in rooms.values()), 'Room outside footprint tiles'
    assert abs(area-sum((t[2]-t[0])*(t[3]-t[1]) for t in tiles))<1e-6, 'Untiled footprint'
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
        assert not room_paths(r,[d for d in plan['doors'] if r['id'] in d['between']],furniture,fire,require_all='footprint_rects' in plan)
    assert 4<=len(sc['survivors'])<=5 and len({s['room'] for s in sc['survivors']})==len(sc['survivors'])
    for s in sc['survivors']:
        assert inside(s['at'],rooms[s['room']]['rect'],.6), 'Person near a wall'
        assert all(not inside(s['at'],[a-.9,b-.9,c+.9,d+.9]) for a,b,c,d in [footprint(l,fire) for l in plan['landmarks']]), 'Person intersects furniture envelope'
        assert all(math.dist(s['at'],d['at'])>=1.2 for d in plan['doors']), 'Person at door'
    body_fire=dict(fire,furniture=dict(fire['furniture'],person_body={'size_m':[.9,.9]}))
    for room in plan['rooms'] if 'footprint_rects' in plan else []:
        furniture=[l for l in plan['landmarks'] if l['room']==room['id']]
        bodies=[]
        for person in (s for s in sc['survivors'] if s['room']==room['id']):
            kind='person_'+person['id']
            body_fire['furniture'][kind]={'size_m':person['body_size_m']}
            body=dict(kind=kind,at=person['at'],yaw=person['yaw'])
            box=footprint(body,body_fire)
            assert inside(box[:2],room['rect'],.25) and inside(box[2:],room['rect'],.25), 'Person body crosses wall'
            assert all(not overlap(box,footprint(l,fire)) for l in furniture), 'Person body intersects furniture'
            bodies.append(body)
        assert not room_paths(room,[d for d in plan['doors'] if room['id'] in d['between']],furniture+bodies,body_fire,targets=bodies,require_all=True), 'Person blocks a 0.9 m room path'
    assert all(not inside(r['spawn']['at'],fp) for r in sc['robots'])
    assert sc['search_area']==fp and sc['duration_s']==600 and 0<sc['search_s']<=480 and sc['time_limit']
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
