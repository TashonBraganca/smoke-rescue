#!/usr/bin/env python3
"""Offline evaluator geometry audit. Reads project inputs and prints JSON only.

No simulator, Unity process, robot code, configuration, or model cache is changed.
Run with PYTHONDONTWRITEBYTECODE=1 and the brain Python environment.
"""
import argparse
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import struct
import sys

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
RYOR = ROOT.parent / "smoke-rescue-ryor"
SITE = ROOT.parent / "showcase/site-next"
sys.path.insert(0, str(ROOT / "tools"))
from headless_world import compile_world, load_scenario
from random_layout_asset_bounds import mesh_bounds

SOURCES = {}
MESH_POINTS = {}
STEP = .025


def read(path):
    path = Path(path)
    data = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def rot(yaw):
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return np.array([[c, s], [-s, c]])


def obb(name, centre, size, yaw=0, bottom=0, top=2.8):
    return dict(name=name, centre=list(map(float, centre)), size=list(map(float, size)),
                yaw=float(yaw), bottom=float(bottom), top=float(top))


def corners(b):
    return np.array(list(itertools.product((-.5, .5), repeat=2))) @ np.diag(b['size']) @ rot(b['yaw']).T + b['centre']


def distance(points, b):
    local = (points - b['centre']) @ rot(b['yaw'])
    return np.linalg.norm(np.maximum(abs(local) - np.array(b['size']) / 2, 0), axis=-1)


def clearance(points, boxes):
    out = np.full(points.shape[:-1], np.inf)
    for b in boxes:
        if b['bottom'] < .60 and b['top'] > .1:
            out = np.minimum(out, distance(points, b))
    return out


def ray_hits(starts, end, b):
    """Slab intersection for a batch of 3D segments against one yaw box."""
    sr = np.empty_like(starts)
    sr[:, [0, 2]] = (starts[:, [0, 2]] - b['centre']) @ rot(b['yaw'])
    sr[:, 1] = starts[:, 1]
    er = np.empty(3)
    er[[0, 2]] = (end[[0, 2]] - b['centre']) @ rot(b['yaw'])
    er[1] = end[1]
    limits = [(-b['size'][0]/2, b['size'][0]/2), (b['bottom'], b['top']),
              (-b['size'][1]/2, b['size'][1]/2)]
    lo, hi = np.zeros(len(starts)), np.ones(len(starts))
    for k, (a, z) in enumerate(limits):
        v = er[k] - sr[:, k]
        parallel = abs(v) < 1e-10
        safe = np.where(parallel, 1, v)
        u, w = (a-sr[:, k])/safe, (z-sr[:, k])/safe
        lo = np.maximum(lo, np.where(parallel, -np.inf, np.minimum(u, w)))
        hi = np.minimum(hi, np.where(parallel, np.inf, np.maximum(u, w)))
        hi = np.where(parallel & ((sr[:, k] < a) | (sr[:, k] > z)), -1, hi)
    return (hi >= lo) & (hi > 1e-6) & (lo < 1-1e-6)


def native_bounds(part, assets):
    rel = part_path(part)
    if rel in assets:
        a = assets[rel]
        lo, hi = np.array(a['min_m']), np.array(a['max_m'])
    else:
        p = ROOT / 'assets_cache' / rel
        SOURCES[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        lo, hi = mesh_bounds(p)
    # glTFast mirrors the glTF x axis on import.
    lo, hi = lo.copy(), hi.copy()
    lo[0], hi[0] = -hi[0], -lo[0]
    return lo, hi


def part_path(part):
    if part.get('downloaded'):return part['downloaded']
    if part.get('polyhaven'):return 'models/'+part['polyhaven']+'.glb'
    return 'models/blender_'+part.get('variant',part['blender'])+'.glb'


def furniture(plan, sc):
    props = read(ROOT / 'config/props.json')
    assets = {a['path']: a for k in read(ROOT/'tools/random_layout_assets.json')['kinds'].values() for a in k['assets']}
    out = []
    for i, lm in enumerate(plan.get('landmarks', []) + sc.get('obstacles', [])):
        if lm['kind'] == 'large_rug_patterned':
            continue
        last_top = 0
        for j, part in enumerate(props['landmarks'][lm['kind']]):
            lo, hi = native_bounds(part, assets)
            yaw = lm.get('yaw', 0) + part.get('yaw', 0)
            size, centre = hi-lo, (hi+lo)/2
            raw_points=mesh_bound_points(ROOT/'assets_cache'/part_path(part))
            xy_points=raw_points[:,[0,2]].copy();xy_points[:,0]*=-1
            placed_points=xy_points@rot(yaw).T
            actual_world_size=placed_points.max(axis=0)-placed_points.min(axis=0)
            factor = 1.
            if 'fit_width' in part:
                factor = part['fit_width'] / max(actual_world_size)
            off = part.get('offset', [0, 0, 0])
            oy = last_top if isinstance(off[1], str) else off[1]
            root_xy = np.array(lm['at']) + rot(lm.get('yaw', 0)) @ np.array(off)[[0, 2]].astype(float)
            xy = root_xy + rot(yaw) @ centre[[0, 2]] * factor
            bottom, top = lo[1]*factor+oy, hi[1]*factor+oy
            if part.get('on_floor'):
                top -= bottom
                bottom = 0
            last_top = top
            b=obb(f"{i}:{lm['room']}:{lm['kind']}:{j}", xy, size[[0, 2]]*factor, yaw, bottom, top)
            b['collider_size']=abs(rot(yaw).T@actual_world_size*factor).tolist()
            b['collider_centre']=(root_xy+(placed_points.max(axis=0)+placed_points.min(axis=0))/2*factor).tolist()
            out.append(b)
    return out


def saved_filler():
    """Read the eleven saved prefab root transforms and added box colliders."""
    p = ROOT/'unity/SmokeRescue/Assets/SmokeRescue/Scenes/House.unity'
    raw = p.read_text()
    SOURCES[str(p)] = hashlib.sha256(raw.encode()).hexdigest()
    docs = {m[1]: m[2] for m in re.findall(r'--- !u!(\d+) &(-?\d+)[^\n]*\n(.*?)(?=\n--- !u!|\Z)', raw, re.S)}
    out = []
    for doc in docs.values():
        if not doc.startswith('PrefabInstance:') or 'm_TransformParent: {fileID: 1147340248}' not in doc:
            continue
        values = dict(re.findall(r'propertyPath: (\S+)\n\s+value: ([^\n]*)', doc))
        name = values.get('m_Name', '')
        added = re.findall(r'addedObject: \{fileID: (\d+)\}', doc)
        for aid in added:
            col = docs[aid]
            if not col.startswith('BoxCollider:'):
                continue
            def xyz(field):
                return np.array([float(x) for x in re.search(field+r': \{x: ([^,]+), y: ([^,]+), z: ([^}]+)', col).groups()])
            qy, qw = [float(values.get('m_LocalRotation.'+k, v)) for k,v in [('y',0),('w',1)]]
            yaw = math.degrees(2*math.atan2(qy,qw))
            pos = np.array([float(values.get('m_LocalPosition.'+k, 0)) for k in 'xyz'])
            c, s = xyz('m_Center'), xyz('m_Size')
            xy = pos[[0,2]]+rot(yaw)@c[[0,2]]
            out.append(obb('saved_filler:'+name, xy, s[[0,2]], yaw, pos[1]+c[1]-s[1]/2, pos[1]+c[1]+s[1]/2))
    assert len(out) == 11, f'Expected eleven saved filler colliders, got {len(out)}'
    return out


def walls(plan, thickness=True, leaves=True):
    """Merge collinear room edges before cutting doors, as HouseBuilder does."""
    lines = {}
    for room in plan['rooms']:
        x0,z0,x1,z1 = room['rect']
        for horizontal, at, a,b in [(True,z0,x0,x1),(True,z1,x0,x1),(False,x0,z0,z1),(False,x1,z0,z1)]:
            lines.setdefault((horizontal,at), []).append((a,b))
    out=[]
    def occupied(x,z):
        return any(r['rect'][0]<x<r['rect'][2] and r['rect'][1]<z<r['rect'][3] for r in plan['rooms'])
    for (horizontal,at), spans in lines.items():
        stops=sorted(set(v for t in spans for v in t))
        # Split only to classify exterior versus interior, then cut the same door gaps.
        for a,b in zip(stops,stops[1:]):
            if not any(c<=a and d>=b for c,d in spans):continue
            mid=(a+b)/2
            low=occupied(mid,at-.01) if horizontal else occupied(at-.01,mid)
            high=occupied(mid,at+.01) if horizontal else occupied(at+.01,mid)
            exterior=low!=high
            t=(plan['exterior_wall_thickness'] if exterior else plan['interior_wall_thickness']) if thickness else 0
            centre=at+(t/2*(1 if high else -1) if exterior else 0)
            cuts=[]
            for d in plan['doors']:
                if (d['wall']=='z')!=horizontal or abs(d['at'][1 if horizontal else 0]-at)>.001:continue
                if d.get('state','open') not in ['open','removed']:continue
                p=d['at'][0 if horizontal else 1];cuts.append((p-d['width']/2,p+d['width']/2))
            cur=a
            for c,d in sorted(cuts)+[(b,b)]:
                if d<cur or c>b:continue
                end=min(c,b)
                if end>cur+1e-6:
                    xy=[(cur+end)/2,centre] if horizontal else [centre,(cur+end)/2]
                    size=[end-cur,t] if horizontal else [t,end-cur]
                    out.append(obb('wall',xy,size))
                cur=max(cur,d)
                if cur>=b:break
    if leaves:
        for d in plan['doors']:
            if d.get('state')!='open':continue
            horizontal=d['wall']=='z';swing=next(r for r in plan['rooms'] if r['id']==next(x for x in d['between'] if x!='outside'))
            a,b,c,e=swing['rect'];side=1 if ((b+e)/2 if horizontal else (a+c)/2)>d['at'][1 if horizontal else 0] else -1
            exterior='outside' in d['between'];t=plan['exterior_wall_thickness'] if exterior else plan['interior_wall_thickness']
            centre=np.array(d['at'],dtype=float)
            if exterior:centre[1 if horizontal else 0]+=side*t/2
            centre[0 if horizontal else 1]-=d['width']/2-.01
            centre[1 if horizontal else 0]+=side*(t/2+.025)
            yaw=-side*175 if horizontal else side*175
            size=[d['width']-.02,.04] if horizontal else [.04,d['width']-.02]
            centre+=rot(yaw)@np.array([size[0]/2 if horizontal else 0,0 if horizontal else size[1]/2])
            out.append(obb('leaf:'+d['id'],centre,size,yaw,0,plan['door_height']-.02))
    return out


def box_from_poly(name,poly,height,bottom=0):
    p=np.array(poly);e=p[1]-p[0];f=p[2]-p[1]
    return obb(name,p.mean(axis=0),[np.linalg.norm(e),np.linalg.norm(f)],math.degrees(math.atan2(-e[1],e[0])),bottom,height)


def world_boxes(world, inverse=False):
    def pt(p):
        return np.array(world['gate']['origin'])+rot(world['gate']['yaw_deg'])@p if inverse else np.array(p)
    boxes=[]
    for i,s in enumerate(world['segments']):
        if s[4]>.6:continue
        a,b=pt(s[:2]),pt(s[2:4]);v=b-a
        boxes.append(obb('world_wall:'+str(i),(a+b)/2,[np.linalg.norm(v),0],math.degrees(math.atan2(-v[1],v[0])),s[4],s[5]))
    for i,p in enumerate(world['prisms']):
        boxes.append(box_from_poly('world:'+str(i)+':'+p['kind'],[pt(v) for v in p['poly']],p['height_m']))
    return boxes


def people_boxes(people, props):
    out=[]
    for p in people:
        part=props['survivors_posed'][p['id']]
        path=ROOT/'assets_cache'/part['downloaded']
        lo,hi=mesh_bounds(path)
        size=hi-lo
        p['height']=float(size[1]/2+.01)
        b=obb('person:'+p['id'],p['at'],size[[0,2]],p.get('yaw',0),.01,float(size[1]+.01))
        points=mesh_bound_points(path)
        xy=points[:,[0,2]].copy();xy[:,0]*=-1
        world=xy@rot(p.get('yaw',0)).T
        world_size=world.max(axis=0)-world.min(axis=0)
        b['world_bounds_size']=world_size.tolist()
        b['collider_size']=abs(rot(p.get('yaw',0)).T@world_size).tolist()
        out.append(b)
    return out


def mesh_bound_points(path):
    """Keep each transformed accessor box separate before the placement yaw."""
    if str(path) in MESH_POINTS:return MESH_POINTS[str(path)]
    data=path.read_bytes();SOURCES[str(path)]=hashlib.sha256(data).hexdigest()
    size,kind=struct.unpack('<II',data[12:20]);doc=json.loads(data[20:20+size]);points=[]
    def walk(index,parent):
        node=doc['nodes'][index]
        if 'matrix' in node:m=np.array(node['matrix']).reshape((4,4),order='F')
        else:
            x,y,z,w=node.get('rotation',[0,0,0,1])
            r=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
            m=np.eye(4);m[:3,:3]=r@np.diag(node.get('scale',[1,1,1]));m[:3,3]=node.get('translation',[0,0,0])
        world=parent@m
        if 'mesh' in node:
            for prim in doc['meshes'][node['mesh']]['primitives']:
                a=doc['accessors'][prim['attributes']['POSITION']]
                for p in itertools.product(*zip(a['min'],a['max'])):points.append((world@[*p,1])[:3])
        for child in node.get('children',[]):walk(child,world)
    for i in doc['scenes'][doc.get('scene',0)]['nodes']:walk(i,np.eye(4))
    MESH_POINTS[str(path)]=np.array(points)
    return MESH_POINTS[str(path)]


def grid(plan, boxes, radius=.35):
    a,b,c,d=plan['footprint'];pad=plan['entry']
    x0,z0=min(a,pad[0]-1)-.1,min(b,pad[1]-1.5)-.1
    xx=np.arange(x0,max(c,pad[0]+1)+.1,STEP);zz=np.arange(z0,max(d,pad[1]+1)+.1,STEP)
    x,z=np.meshgrid(xx,zz);points=np.stack([x,z],axis=-1)
    domain=np.zeros(x.shape,bool)
    room_masks={}
    for room in plan['rooms']:
        a,b,c,d=room['rect'];m=(x>=a)&(x<=c)&(z>=b)&(z<=d);domain|=m;room_masks[room['id']]=m
    # Admit only the launch approach around this entrance, never an exterior shortcut.
    domain|=(abs(x-pad[0])<=.85)&(abs(z-pad[1])<=1.5)
    dist=clearance(points,boxes)
    free=domain&(dist>=radius+STEP/math.sqrt(2))
    labels,n=ndimage.label(free)
    # Seed just outside the selected opening, using the launch pad for our three houses.
    spawn=plan['spawn'];iz=np.argmin(abs(zz-spawn[1]));ix=np.argmin(abs(xx-spawn[0]));seed=labels[iz,ix]
    assert seed, (plan.get('id'), 'launch point occupied', spawn)
    reachable=labels==seed
    return dict(points=points,dist=dist,free=free,reachable=reachable,labels=labels,room_masks=room_masks,xx=xx,zz=zz)


def audit(plan,boxes,people,body_boxes=(),nav_boxes=None):
    navigation = (boxes if nav_boxes is None else nav_boxes)+(list(body_boxes) if nav_boxes is None else collider_boxes(body_boxes))
    g=grid(plan,navigation)
    out={'rooms':{},'doors':{},'people':{},'unreachable_components':[]}
    for rid,m in g['room_masks'].items():
        eligible=g['free']&m;reach=g['reachable']&m
        out['rooms'][rid]={'reachable':bool(reach.any()),'free_m2':round(float(eligible.sum())*STEP**2,3),
                           'reachable_m2':round(float(reach.sum())*STEP**2,3),
                           'stranded_m2':round(float((eligible&~g['reachable']).sum())*STEP**2,3)}
    pts=g['points']
    for d in plan['doors']:
        p=np.array(d['at']);axis=0 if d['wall']=='z' else 1;across=1-axis
        cut=(abs(pts[:,:,across]-p[across])<STEP*.55)&(abs(pts[:,:,axis]-p[axis])<d['width']/2)
        near=(abs(pts[:,:,across]-p[across])<.6)&(abs(pts[:,:,axis]-p[axis])<d['width']/2)
        approach=[]
        for sign in [-1,1]:
            m=near&((pts[:,:,across]-p[across])*sign>.4)
            approach.append(bool((m&g['reachable']).any()))
        exact=np.repeat(np.array([d['at']],float),801,axis=0)
        exact[:,axis]+=np.linspace(-d['width']/2,d['width']/2,801)
        dc=clearance(exact,navigation);best_door=int(np.argmax(dc))
        nearest=sorted((float(distance(exact[best_door:best_door+1],b)[0]),b['name'])
                       for b in navigation if b['bottom']<.6 and b['top']>.1)[:3]
        out['doors'][d['id']]={'width_m':d['width'],'threshold_reachable':bool((cut&g['reachable']).any()),
                              'both_approaches_reachable':all(approach),
                              'max_threshold_disc_diameter_m':round(float(dc[best_door])*2,3),
                              'best_threshold_point':exact[best_door].tolist(),
                              'nearest_obstacles':[(round(d,3),n) for d,n in nearest]}
    candidates=pts[g['reachable']]
    mount=read(ROOT/'config/sensors.json')['camera']['mount']
    for p in people:
        target=np.array(p['at']);h=p.get('height',.3)
        delta=target-candidates;body_range=np.linalg.norm(delta,axis=1)
        crange=body_range-mount['forward_offset_m']
        distance3=np.sqrt(crange**2+(mount['height_m']-h)**2)
        angle=np.degrees(np.arctan2(mount['height_m']-h,crange))
        valid=(crange>0)&(distance3<=4.5)&(body_range<=4.5)&(angle<39.9)&(angle>-19.9)
        sel=np.where(valid)[0];order=sel[np.argsort(distance3[sel],kind='stable')]
        los_boxes=boxes+[b for b in body_boxes if b['name']!='person:'+p['id']]
        best=None
        # Check all candidates in one vectorized pass. This also measures residual sightline opportunities.
        starts=np.zeros((len(order),3));starts[:,[0,2]]=candidates[order]+delta[order]/body_range[order,None]*mount['forward_offset_m'];starts[:,1]=mount['height_m']
        endpoint=np.array([target[0],h,target[1]])
        visible=np.ones(len(order),bool)
        for b in los_boxes:
            visible &= ~ray_hits(starts,endpoint,b)
        vi=np.flatnonzero(visible)
        if len(vi):
            visible_indices=order[vi]
            # Prefer framing five degrees above the lower image edge. Among those
            # points, choose the largest navigation clearance, then shortest range.
            framed=visible_indices[angle[visible_indices]<=35]
            pool=framed if len(framed) else visible_indices
            gaps=clearance(candidates[pool],navigation)
            k=pool[np.lexsort((distance3[pool],-gaps))[0]]
            best={'point':np.round(candidates[k],3).tolist(),'camera_range_3d_m':round(float(distance3[k]),3),
                                'camera_range_horizontal_m':round(float(crange[k]),3),'body_range_horizontal_m':round(float(body_range[k]),3),
                                'clearance_m':round(float(clearance(candidates[k:k+1],navigation)[0]),3),
                                'down_angle_deg':round(float(angle[k]),2),'framing_margin_pass':bool(len(framed))}
        out['people'][p['id']]={'at':p['at'],'body_centre_height_m':round(h,3),'best':best,'visible_standing_area_m2':round(len(vi)*STEP**2,3)}
    return out


def collider_boxes(boxes):
    """Mirror FurnitureBuilder.AddBoxCollider's inverse-vector size calculation."""
    out=copy.deepcopy(boxes)
    for b in out:
        if 'collider_size' in b:
            b['size']=b['collider_size']
            if 'collider_centre' in b:b['centre']=b['collider_centre']
            continue
        if b['name'][0].isdigit() or b['name'].startswith('person:'):
            r=rot(b['yaw'])
            b['size']=abs(r.T@(abs(r)@b['size'])).tolist()
    return out


def load_cases():
    cases=[]
    props=read(ROOT/'config/props.json')
    for name in ['house_onedoor_seed934','family_ranch','long_ranch']:
        sp=ROOT/f'config/scenario_{name}_full.json';read(sp);sc=load_scenario(sp)
        plan=read(ROOT/'config'/sc.get('floorplan_file','floorplan.json'))
        full_plan=copy.deepcopy(plan)
        read(ROOT/'config'/sc.get('fire',{}).get('config_file','fire.json'))
        # Seed 934 searches only the original ten rooms of the larger saved scene.
        if name=='house_onedoor_seed934':
            fp=sc['search_area'];plan=copy.deepcopy(plan)
            plan['rooms']=[r for r in plan['rooms'] if all([r['rect'][0]>=fp[0],r['rect'][1]>=fp[1],r['rect'][2]<=fp[2],r['rect'][3]<=fp[3]])]
            ids={r['id'] for r in plan['rooms']}|{'outside'}
            plan['doors']=[d for d in plan['doors'] if set(d['between'])<=ids]
            plan['landmarks']=[l for l in plan['landmarks'] if l['room'] in ids];plan['footprint']=fp
        plan['id']=name;plan['entry']=next(d['at'] for d in plan['doors'] if 'outside' in d['between']);plan['spawn']=sc['robots'][0]['spawn']['at']
        people=copy.deepcopy(sc['survivors'])
        if name=='house_onedoor_seed934':
            oracle=ROOT/'data/recordings/unity_house_onedoor_seed934_full_1520_5of5/oracle/survivors.json'
            people=[dict(id=p['id'],at=[p['position_house'][0],p['position_house'][2]],yaw=p['yaw'],room=p['room'],pose=p['pose']) for p in read(oracle)]
        bodies=people_boxes(people,props)
        furn=furniture(plan,sc)
        if name=='house_onedoor_seed934':furn+=saved_filler()
        all_boxes=walls(full_plan)+furn
        cases.append(dict(name=name,mode='unity_mesh_bounds',plan=plan,boxes=all_boxes,people=people,bodies=bodies,nav_boxes=collider_boxes(all_boxes)))
        world=compile_world(sp)
        hp=copy.deepcopy(sc['survivors'])
        for p in hp:p['height']=.3
        cases.append(dict(name=name,mode='headless_world',plan=plan,boxes=world_boxes(world,True),people=hp,bodies=[]))
    gallery=read(SITE/'public/rescue-data/gallery.json')
    for name in ['hx_46bfe067cc','hx_cb9d3d6726']:
        world=read(RYOR/f'data/runs/ryor/worlds/{name}__r2.json')
        source_plan=read(RYOR/f'data/runs/ryor/houses/floorplan_{name}.json')
        gh=next(h for h in gallery['houses'] if h['id']==name)
        plan=copy.deepcopy(gh['placement_geometry']);plan['landmarks']=plan.pop('furniture')
        plan.update(footprint=[min(r['rect'][0] for r in plan['rooms']),min(r['rect'][1] for r in plan['rooms']),max(r['rect'][2] for r in plan['rooms']),max(r['rect'][3] for r in plan['rooms'])],id=name,entry=gh['door'],spawn=[0,0],exterior_wall_thickness=.25,interior_wall_thickness=.15,door_height=2.1)
        people=copy.deepcopy(world['people'])
        for p in people:p['height']=.3
        # The deployed houses are headless worlds. Assert gallery geometry matches their authoritative world arrays.
        assert np.allclose([[p['x'],p['y']] for p in gh['people_truth']],[p['at'] for p in people])
        assert np.allclose(gh['walls'],[s[:4] for s in world['segments'] if s[4]==0],atol=.0006)
        assert np.allclose(gh['furniture'],[p['poly'] for p in world['prisms']],atol=.0006)
        assert len(plan['rooms'])==len(world['rooms'])==7
        cases.append(dict(name=name,mode='live_headless_world',plan=plan,boxes=world_boxes(world),people=people,bodies=[]))
        # Reconstruct mesh placement for a future Unity build, in the live gallery frame.
        native_furn=furniture(source_plan,{'obstacles':[]})
        inv=rot(world['gate']['yaw_deg']).T
        for b in native_furn:
            b['centre']=(inv@(np.array(b['centre'])-world['gate']['origin'])).tolist();b['yaw']-=world['gate']['yaw_deg']
        cases.append(dict(name=name,mode='unity_mesh_bounds_comparison',plan=plan,boxes=walls(plan)+native_furn,people=copy.deepcopy(people),bodies=[]))
    return cases


def checks():
    b=obb('test',[0,0],[2,2],0,0,2)
    assert np.allclose(distance(np.array([[0,0],[2,0],[2,2]]),b),[0,1,math.sqrt(2)])
    assert ray_hits(np.array([[-3,1,0],[-3,3,0]]),np.array([3,1,0]),b).tolist()==[True,True]
    assert not ray_hits(np.array([[-3,3,0]]),np.array([3,3,0]),b)[0]
    assert np.allclose(distance(np.array([[0,0]]),obb('rot',[0,0],[2,1],90)),[0])


def repair_proposals(cases):
    out={}
    for original in cases:
        if original['name']=='family_ranch' and original['mode']=='unity_mesh_bounds':
            c=copy.deepcopy(original)
            c['people'][0]['yaw']=90
            c['bodies']=people_boxes(c['people'],read(ROOT/'config/props.json'))
            result=audit(c['plan'],c['boxes'],c['people'],c['bodies'],c.get('nav_boxes'))
            assert all(r['reachable'] for r in result['rooms'].values())
            assert all(p['best'] for p in result['people'].values())
            out['family_ranch_person_yaw90']=result
            continue
        if original['name']!='house_onedoor_seed934':continue
        c=copy.deepcopy(original)
        c['boxes']=[b for b in c['boxes'] if 'obstacle_wood_pile' not in b['name']]
        if c['mode']=='unity_mesh_bounds':
            c['boxes']+=furniture({'landmarks':[]},{'obstacles':[dict(room='hallway',kind='obstacle_wood_pile',at=[5.4,7.2],yaw=0)]})
            for b in c['boxes']:
                if 'clothes_racks_both_sides' in b['name']:
                    b['centre']=[5.8 if b['name'].endswith(':0') else 7.2,13.2]
                    b['collider_centre']=b['centre'][:]
                # Remove existing wall intersections with small local translations.
                moves={'13:bedroom_1:tall_wardrobe':(.51,0),'14:bathroom:clawfoot_bathtub':(0,-.4),
                       '15:bathroom:sink_with_mirror':(-.22,0),'35:hallway:obstacle_rubble':(0,-.61),
                       '38:kitchen:wooden_desk_with_lamp':(-.3,.4)}
                for prefix,delta in moves.items():
                    if b['name'].startswith(prefix):
                        b['centre']=(np.array(b['centre'])+delta).tolist()
                        if 'collider_centre' in b:b['collider_centre']=(np.array(b['collider_centre'])+delta).tolist()
            c['nav_boxes']=collider_boxes(c['boxes'])
        else:
            c['boxes']+=[obb('proposed_wood_pile',[5.4,7.2],[1.3,.9],0,0,.8)]
            for b in c['boxes']:
                if 'clothes_racks_both_sides' in b['name']:b['centre']=[6.5,13.2];b['size']=[2.65,.5]
                if 'large_rug_patterned' in b['name']:b['top']=.02
                for kind,delta in [('tall_wardrobe',(.51,0)),('clawfoot_bathtub',(0,-.4)),('sink_with_mirror',(-.22,0))]:
                    # Only original-house fixtures, not the other building wings.
                    if kind in b['name'] and np.linalg.norm(np.array(b['centre'])-[.4,9.5])<.1 and kind=='tall_wardrobe':b['centre']=(np.array(b['centre'])+delta).tolist()
                    elif kind=='clawfoot_bathtub' and np.linalg.norm(np.array(b['centre'])-[6.5,10.4])<.1:b['centre']=(np.array(b['centre'])+delta).tolist()
                    elif kind=='sink_with_mirror' and np.linalg.norm(np.array(b['centre'])-[7.65,9])<.1:b['centre']=(np.array(b['centre'])+delta).tolist()
                if 'wooden_desk_with_lamp' in b['name'] and np.linalg.norm(np.array(b['centre'])-[18.9,1])<.1:b['centre']=(np.array(b['centre'])+[-.3,.4]).tolist()
        result=audit(c['plan'],c['boxes'],c['people'],c['bodies'],c.get('nav_boxes'))
        assert all(r['reachable'] for r in result['rooms'].values()),c['mode']
        assert all(d['both_approaches_reachable'] for d in result['doors'].values()),c['mode']
        out[c['mode']]=result
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--house');ap.add_argument('--mode');ap.add_argument('--step',type=float,default=.025);ap.add_argument('--repairs',action='store_true');args=ap.parse_args()
    global STEP
    STEP=args.step
    checks()
    out={}
    cases=load_cases()
    for case in cases:
        if args.house and case['name']!=args.house:continue
        if args.mode and case['mode']!=args.mode:continue
        key=case['name']+'/'+case['mode']
        print('Auditing '+key,file=sys.stderr,flush=True)
        out[key]=audit(case['plan'],case['boxes'],case['people'],case['bodies'],case.get('nav_boxes'))
        out[key]['geometry']=case['boxes']
        if case.get('nav_boxes') is not None:out[key]['collider_geometry']=case['nav_boxes']
    repaired=repair_proposals(cases) if args.repairs else {}
    print(json.dumps(dict(step_m=STEP,results=out,repair_proposals=repaired,sources=SOURCES),indent=2))


if __name__=='__main__':main()
