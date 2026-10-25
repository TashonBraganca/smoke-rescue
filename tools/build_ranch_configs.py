#!/usr/bin/env python3
"""Write the WORLD-only ranches. Never imported by robot code.

The included initial_scenario_long_ranch snapshot preserves the checked people
while the entrance is revised; it is an explicit world-side input.
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
    if not family:
        checked = ROOT/'handoff/houses_evidence/initial_scenario_long_ranch.json'
        chosen = json.loads(checked.read_text())['survivors']
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

if __name__=='__main__':
    build('family_ranch')
    build('long_ranch')
