#!/usr/bin/env python3
"""Write the WORLD-only ranches. Never imported by robot code.

The default target is the L-shaped long ranch. Select family_ranch explicitly
to rebuild the original family house. Brains never read construction inputs.
"""
import copy
import json
import random
from pathlib import Path
import random_layout as g
from validate_ranches import furniture_clear, validate

ROOT = Path(__file__).resolve().parents[1]

def reasons(value, context):
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if key.endswith('_reason'):
                continue
            if isinstance(item, (int, float)) and not isinstance(item, bool) or isinstance(item, list) and any(isinstance(x, (int, float)) for x in item):
                value.setdefault(key + '_reason', context)
            reasons(item, context)
    elif isinstance(value, list):
        for item in value:
            reasons(item, context)

def build(name):
    if name == 'long_ranch':
        return build_long_v2()
    family = name == 'family_ranch'
    width = 15 if family else 18
    raw = [('hall', 'Hallway', 'corridor', [0,3.5,width,5.25])]
    if family:
        raw += [('living','Living room','living',[0,0,6,3.5]), ('kitchen','Kitchen and dining','kitchen',[6,0,12,3.5]), ('laundry','Laundry','laundry',[12,0,15,3.5])]
        raw += [(f'bed{i+1}',f'Bedroom {i+1}','bedroom',[i*3.5,5.25,(i+1)*3.5,11]) for i in range(3)]
        raw += [('bath1','Bathroom 1','bathroom',[10.5,5.25,12.75,11]), ('bath2','Bathroom 2','bathroom',[12.75,5.25,15,11])]
    else:
        raw += [('living','Living room','living',[0,0,5,3.5]), ('kitchen','Kitchen and dining','kitchen',[5,0,10,3.5]), ('family','Family room','living',[10,0,15,3.5]), ('utility','Utility','laundry',[15,0,18,3.5])]
        raw += [(f'bed{i+1}',f'Bedroom {i+1}','bedroom',[i*3.5,5.25,(i+1)*3.5,11]) for i in range(4)]
        raw += [('bath1','Bathroom 1','bathroom',[14,5.25,16,11]), ('bath2','Bathroom 2','bathroom',[16,5.25,18,11])]
    prefix = 'fr_' if family else 'lr_'
    rooms = []
    for rid, label, kind, rect in raw:
        wet = kind in ('bathroom','laundry')
        rooms.append(dict(id=prefix+rid,name=label,kind=kind,rect=rect,rect_reason=f'Selected {rect[2]-rect[0]} by {rect[3]-rect[1]} m bay in a tiled {width} by 11 m single-floor ranch. Hallway is 1.75 m before wall deductions.',floor='ceramic_tiles_checker' if wet else 'wood_oak_planks',walls='ceramic_tiles_large_white' if wet else 'paint_warm_grey_textured',ceiling='plaster_white'))
    doors = []
    hall = rooms[0]['id']
    for r in rooms[1:]:
        x0,z0,x1,z1 = r['rect']
        doors.append(dict(id='d_'+r['id'],between=[hall,r['id']],wall='z',at=[x0+.8 if r['kind']=='bedroom' else (x0+x1)/2,3.5 if z0==0 else 5.25],width=1.0,state='removed',at_reason='Hall opening centred in service and living bays, offset 0.8 m from the bedroom bay edge to leave a bed along the opposite wall. Door-to-centre paths stay clear.',width_reason='1 m clear opening exceeds the requested 0.9 m and avoids an open leaf narrowing the approach.'))
    entry_at = [3 if family else 2.5,0]
    entry_room = prefix+'living'
    doors.append(dict(id='d_front',between=['outside',entry_room],wall='z',at=entry_at,width=1.0,state='removed',at_reason='Only entrance, on the south living-room wall.',width_reason='Same 1 m entrance gap as seed 934; removed leaf keeps the launch route unobstructed.'))
    for d in doors:
        if 'outside' in d['between'] or prefix+'living' in d['between']:
            d['width'] = 1.6
            d['width_reason'] = 'A 1.6 m double front entrance, or living-to-hall cased opening, provides passing and turning room. Earlier 1 m openings passed static path checks but left one robot inside during full return tests. All other interior openings remain 1 m.'
    plan = dict(version=1,units='m',origin='South-west outer corner at Unity world (0,0,0), x east and z north.',footprint=[0,0,width,11],wall_height=2.8,exterior_wall_thickness=.25,interior_wall_thickness=.15,door_height=2.1,rooms=rooms,doors=doors,landmarks=[],fires=[],emergency_lights=[],clear_path_rule='0.9 m door-to-centre path and 1.2 m door keepout, checked with conservative furniture mesh bounds.',footprint_reason=f'{width*11} m2 gross area chosen within the brief range; walls are inside this rectangle.')
    fire = json.loads((ROOT/'config/fire.json').read_text())
    catalog = json.loads(g.CATALOG.read_text())
    for kind,item in catalog['kinds'].items():
        fire['furniture'][kind]['size_m'] = item['conservative_size_m']
        fire['furniture'][kind]['size_m_reason'] = 'Conservative envelope of measured mesh and baseline fuel bounds in tools/random_layout_assets.json.'
    previous_clear = g.furniture_clear
    try:
        g.furniture_clear = furniture_clear
        g.place_furniture(plan,fire,random.Random(2510 if family else 2511))
    finally:
        g.furniture_clear = previous_clear
    sc = json.loads((ROOT/'config/scenario_house_onedoor_seed934_full.json').read_text())
    sc.update(id=name,description='Full mission in a new ranch. Unchanged brain, sensors and local gate pads only. Layout is world-side evaluation data.',floorplan_file=f'floorplan_{name}.json',seed=2510 if family else 2511,search_area=plan['footprint'][:],survivor_placement='fixed',obstacles=[],warm_objects=[],dark_rooms=[prefix+'bed1',prefix+'bath1'])
    sc['seed_reason'] = 'World seed 2510 for family ranch and 2511 for long ranch, chosen as adjacent reproducible seeds for the 25 October layouts. This selects furnishing and safe placement candidates, plus world and sensor randomness; the brain receives no layout ID.'
    yaw = 0
    for i,r in enumerate(sc['robots']):
        # Keep seed 934 relative B placement error and local pads, rotated with the gate.
        r['spawn'] = dict(at=[entry_at[0] + (-.024 if i else 0),entry_at[1]-(2.402 if i else 1.2)] ,yaw=yaw+(2.39 if i else 0))
    sc['launch']['door'] = 'd_front'
    sc['launch']['reason'] = 'Same 1.2 m surveyed pad spacing and seeded B placement error as seed 934, rotated to this entrance. Brains receive only the local gate frame.'
    sc['smoke']['rooms'] = {prefix+'kitchen':.35,prefix+'living':.25,prefix+'bed2':.3,'default':.15}
    sc['fire'].update(config_file=f'fire_{name}.json',note='Baseline medium-growth model with one origin in a rear bedroom and the seed 934 head start.')
    # Existing fair-placement generator supplies clear candidates; choose one per room.
    pool = dict(sc,seed=sc['seed'])
    chosen = []
    for seed in range(sc['seed'],sc['seed']+100):
        pool['seed'] = seed
        for s in g.people(plan,fire,pool):
            if s['room'] == hall or any(s['room']==t['room'] or g.math.dist(s['at'],t['at'])<3 for t in chosen):
                continue
            chosen.append(s)
            if len(chosen)==5: break
        if len(chosen)==5: break
    assert len(chosen)==5
    poses = [('s1','lying_on_side'),('s2','slumped_against_wall'),('s3','collapsed_at_desk'),('s4','curled_on_bed'),('s6','slumped_against_wall')]
    for s,(sid,pose) in zip(chosen,poses): s.update(id=sid,pose=pose)
    sc['survivors'] = chosen
    sc['survivor_placement_reason'] = 'Five explicit safe positions in different rooms, chosen by the existing conservative placement checker. Fixed placement preserves the checked rooms and furniture clearance in Unity; no position is sent to a robot.'
    fuel = next(l for l in plan['landmarks'] if l['room']==prefix+('bed3' if family else 'bed4') and l['kind']=='blue_double_bed')
    fire['ignitions'] = [dict(room=fuel['room'],at=fuel['at'][:],time_s=0,radius_m=.3,at_reason='Origin inside the rear-bedroom bed fuel envelope, separated from the entrance by the hall; the initial front living-room fire test left one returning robot inside each house.')]
    fire['ignitions_reason'] = 'One rear-bedroom bed origin, away from the only entrance. Every later ignition must follow the unchanged fire solver.'
    for source in fire['sources']:
        if source['id'] == 'geometry':
            source.update(url=f'config/floorplan_{name}.json', accessed='2026-10-25', justification='The 0.25 m grid and 0.1 s step are inherited numerical choices. One rear-bedroom bed ignition at time zero with radius 0.3 m is a scenario choice. Furniture bounds use tools/random_layout_assets.json; surfaces and landmark kinds use the selected floorplan and props.json. Room boundaries align to the fire grid.')
    plan['fires'] = [dict(id='f1',room=fuel['room'],at=fuel['at'][:],intensity=1)]
    plan['emergency_lights'] = [dict(at=g.centre(r['rect']),note=r['id']) for r in rooms]
    reasons(plan,'Simulation construction choice inherited from floorplan.json: metre units, 2.8 m walls, 0.25/0.15 m wall thicknesses and 2.1 m door height. Furniture positions use measured conservative bounds and clear door approaches; light positions are room centres.')
    reasons(sc,'Copied from seed 934 full mission unless stated otherwise. The mission lasts at most 600 s with a 480 s search deadline; launch geometry is rotated to the only entrance. Survivor positions pass the conservative world validator; yaw is a chosen pose orientation.')
    reasons(fire,'Unchanged baseline fire.json parameter, or layout-specific rear-bedroom bed ignition at time zero with 0.3 m radius. Material and numerical assumptions retain the baseline sources.')
    validate(plan,fire,sc)
    for stem,obj in [('floorplan',plan),('fire',fire),('scenario',sc)]:
        suffix = '_full' if stem=='scenario' else ''
        (ROOT/f'config/{stem}_{name}{suffix}.json').write_text(json.dumps(obj,indent=2)+'\n')
    print(f'{name}: {len(rooms)} rooms, {width*11} m2, {len(doors)} doorways, {len(plan["landmarks"])} furniture groups, five people in five rooms; validated')

def build_long_v2():
    raw = [
        ('great','Living, kitchen and dining','kitchen',[0,0,8,5]),
        ('family','Family room','living',[0,5,8.5,7.5]),
        ('primary','Primary bedroom','bedroom',[8,0,12.5,5]),
        ('ensuite','Primary bathroom','bathroom',[12.5,0,15.25,3]),
        ('closet','Walk-in closet','closet',[15.25,0,18,3]),
        ('utility','Utility and mudroom','laundry',[12.5,3,18,5]),
        ('hall','Main hall','corridor',[8.5,5,18,7.5]),
        ('sidehall','Bedroom hall','corridor',[8.5,7.5,10.5,17.5]),
        ('bath','Family bathroom','bathroom',[10.5,7.5,13,10.5]),
        ('bed2','Bedroom 2','bedroom',[13,7.5,18,10.5]),
        ('bed3','Bedroom 3','bedroom',[10.5,10.5,18,14]),
        ('bed4','Bedroom 4','bedroom',[10.5,14,18,17.5]),
    ]
    rooms=[dict(id='lr_'+rid,name=label,kind=kind,rect=rect,
        rect_reason=f'{rect[2]-rect[0]:g} by {rect[3]-rect[1]:g} m room in a 230 m2 L-shaped house. Dimensions include walls.',
        floor='ceramic_tiles_checker' if kind in ('bathroom','laundry') else 'wood_oak_planks',
        walls='ceramic_tiles_large_white' if kind in ('bathroom','laundry') else 'paint_warm_grey_textured',ceiling='plaster_white') for rid,label,kind,rect in raw]
    links=[('front','outside','great','z',[4,0],1.6),
        ('family_hall','family','hall','x',[8.5,6.25],1.6),
        ('family','great','family','z',[6.5,5],1.8),
        ('primary','hall','primary','z',[10.75,5],1.2),
        ('ensuite','primary','ensuite','x',[12.5,2.25],1.0),
        ('closet','ensuite','closet','x',[15.25,2.25],1.0),
        ('utility','hall','utility','z',[15.5,5],1.2),
        ('turn','hall','sidehall','z',[9.5,7.5],1.6),
        ('bath','sidehall','bath','x',[10.5,9.75],1.0),
        ('bed2','hall','bed2','z',[15.5,7.5],1.2),
        ('bed3','sidehall','bed3','x',[10.5,12],1.2),
        ('bed4','sidehall','bed4','x',[10.5,15.75],1.2)]
    doors=[dict(id='d_'+rid,between=[a if a=='outside' else 'lr_'+a,'lr_'+b],wall=axis,at=at,width=width,state='removed',
        at_reason='Cased opening on the shared room boundary with a clear approach on both sides.',width_reason='Clear opening at least 1 m. Wider shared routes allow passing and turning.') for rid,a,b,axis,at,width in links]
    plan=dict(version=1,units='m',origin='South-west corner at Unity world (0,0,0). North is +z.',
        footprint=[0,0,18,17.5],footprint_reason='Bounding rectangle only. The 85 m2 north-west notch is outdoor lawn.',
        footprint_rects=[[0,0,18,7.5],[8.5,7.5,18,17.5]],
        footprint_rects_reason='Two disjoint rectangles tile the occupied 230 m2 L shape. Roof and yard use their union.',
        wall_height=2.8,exterior_wall_thickness=.25,interior_wall_thickness=.15,door_height=2.1,
        rooms=rooms,doors=doors,landmarks=[],fires=[],emergency_lights=[],
        clear_path_rule='Connected 0.9 m clear floor between all room doors and within 0.95 m of each posed body, with furniture and people as obstacles.')
    fire=json.loads((ROOT/'config/fire_long_ranch.json').read_text())
    sc=json.loads((ROOT/'config/scenario_long_ranch_full.json').read_text())
    furnishing={
        'great':['apartment_counter','stove_and_hood','tall_steel_fridge','dining_table_six_chairs','apartment_modern_sofa'],
        'family':['apartment_modern_sofa'],
        'primary':['apartment_bed','tall_wardrobe'],
        'ensuite':['clawfoot_bathtub','sink_with_mirror'],
        'closet':['tall_wardrobe'],
        'utility':['utility_washer','supply_boxes'],
        'bath':['sink_with_mirror','clawfoot_bathtub'],
        'bed2':['apartment_bed'],
        'bed3':['blue_double_bed'],
        'bed4':['blue_double_bed'],
    }
    for room in rooms:
        x0,z0,x1,z1=room['rect']
        for kind in furnishing.get(room['id'][3:],[]):
            w,d=fire['furniture'][kind]['size_m']
            candidates=[]
            for yaw in (0,90,180,270):
                hw,hd=(w/2,d/2) if yaw in (0,180) else (d/2,w/2)
                for i in range(1+max(0,int((x1-x0-2*hw-.60)/.2))):
                    x=x0+.30+hw+i*.2
                    for z in (z0+.30+hd,z1-.30-hd):
                        candidates.append(dict(room=room['id'],kind=kind,at=[round(x,4),round(z,4)],yaw=yaw))
                for i in range(1+max(0,int((z1-z0-2*hd-.60)/.2))):
                    z=z0+.30+hd+i*.2
                    for x in (x0+.30+hw,x1-.30-hw):
                        candidates.append(dict(room=room['id'],kind=kind,at=[round(x,4),round(z,4)],yaw=yaw))
            if room['id']=='lr_great' and kind=='apartment_counter':
                candidates.insert(0,dict(room=room['id'],kind=kind,at=[1.3424,3.08],yaw=90))
            previous=[l for l in plan['landmarks'] if l['room']==room['id']]
            lm=next((l for l in candidates if furniture_clear(l,room,plan,fire,previous)),None)
            if lm: plan['landmarks'].append(lm)
            else: plan.setdefault('generator_notes',[]).append(f"{room['id']}: {kind} omitted to preserve the checked 0.9 m paths.")
    sc.update(id='long_ranch',description='230 m2 L-shaped single-floor house with open living/kitchen/dining, family room, four bedrooms and a private suite. World data only.',
        search_area=plan['footprint'][:],dark_rooms=['lr_bed3','lr_ensuite'])
    sc['seed_reason']='World seed 2511 retained for fire and sensor randomness. Furniture uses deterministic clearance-checked wall sweeps and people use checked floor candidates. No layout ID reaches the brain.'
    sc['survivor_placement_reason']='Five fixed bodies in different rooms. Rotated posed-model bounds clear walls and furniture, and every body has a 0.9 m approach. No position reaches a brain.'
    sc['search_area_reason']='Bounds for world placement and evaluation, including the outdoor notch. Robots receive local sensor data only.'
    for i,r in enumerate(sc['robots']):
        r['spawn']=dict(at=[4+(-.024 if i else 0),-(2.402 if i else 1.2)],yaw=2.39 if i else 0)
    sc['smoke']['rooms']={'lr_great':.25,'lr_bed2':.3,'default':.15}
    # Generate safe world positions in five occupied rooms, with no hallway victims.
    from validate_building import inside, footprint
    from validate_ranches import room_paths
    chosen=[]
    body_fire=copy.deepcopy(fire)
    bounds=json.loads((ROOT/'handoff/houses_evidence/long_ranch_v2_body_bounds.json').read_text())['people']
    identities=[('s1','lying_on_side'),('s2','slumped_against_wall'),('s3','collapsed_at_desk'),('s4','curled_on_bed'),('s6','slumped_against_wall')]
    for rid in ('lr_great','lr_family','lr_primary','lr_bed2','lr_bed3','lr_bed4'):
        room=next(r for r in rooms if r['id']==rid)
        x0,z0,x1,z1=room['rect']
        furniture=[l for l in plan['landmarks'] if l['room']==rid]
        sid,pose=identities[len(chosen)]
        body_fire['furniture']['person_body']={'size_m':bounds[sid]['body_size_m']}
        candidates=[]
        for i in range(int((x1-x0-1.2)/.2)+1):
            for j in range(int((z1-z0-1.2)/.2)+1):
                p=[round(x0+.6+.2*i,4),round(z0+.6+.2*j,4)]
                body=dict(kind='person_body',at=p,yaw=90)
                box=footprint(body,body_fire)
                if not inside(box[:2],room['rect'],.25) or not inside(box[2:],room['rect'],.25): continue
                if any(g.overlap(box,footprint(l,fire)) for l in furniture): continue
                if any(inside(p,[a-.9,b-.9,c+.9,d+.9]) for a,b,c,d in [footprint(l,fire) for l in plan['landmarks']]): continue
                if any(g.math.dist(p,d['at'])<1.2 for d in doors): continue
                if any(g.math.dist(p,s['at'])<3 for s in chosen): continue
                if room_paths(room,[d for d in doors if rid in d['between']],furniture+[body],body_fire,targets=[body],require_all=True): continue
                candidates.append(p)
        if candidates:
            p=candidates[len(candidates)//2]
            chosen.append(dict(id=sid,room=rid,at=p,yaw=90,pose=pose,body_size_m=bounds[sid]['body_size_m'],body_size_m_reason='Transformed posed model bounds plus 0.05 m per side. See handoff/houses_evidence/long_ranch_v2_body_bounds.json for asset hashes and method.'))
        if len(chosen)==5: break
    assert len(chosen)==5, chosen
    sc['survivors']=chosen
    fuel=next(l for l in plan['landmarks'] if l['room']=='lr_bed4' and l['kind']=='blue_double_bed')
    fire['ignitions']=[dict(room='lr_bed4',at=fuel['at'][:],time_s=0,radius_m=.3)]
    fire['ignitions_reason']='One bedroom bed fire. Baseline fire parameters and the 320 s head start remain unchanged.'
    plan['fires']=[dict(id='f1',room='lr_bed4',at=fuel['at'][:],intensity=1)]
    plan['emergency_lights']=[dict(at=g.centre(r['rect']),note=r['id']) for r in rooms]
    for obj in (plan,sc,fire):
        reasons(obj,'World construction choice for the 230 m2 L-shaped long ranch. Existing mission, fire and sensor settings are inherited unchanged; room positions and furniture are checked by validate_ranches.py.')
    validate(plan,fire,sc)
    for stem,obj in [('floorplan',plan),('fire',fire),('scenario',sc)]:
        suffix='_full' if stem=='scenario' else ''
        (ROOT/f'config/{stem}_long_ranch{suffix}.json').write_text(json.dumps(obj,indent=2)+'\n')
    print(f'long_ranch v2: {len(rooms)} rooms, 230 m2, {len(plan["landmarks"])} furniture groups, five people; validated')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('house',choices=['family_ranch','long_ranch','all'],nargs='?',default='long_ranch')
    name=parser.parse_args().house
    for house in ('family_ranch','long_ranch') if name=='all' else (name,):
        build(house)
