"""Ranch contract checks and rejection of geometry that could trap a robot."""
import copy
import json
from pathlib import Path
import pytest
from validate_ranches import validate

CONFIG = Path(__file__).resolve().parents[1] / 'config'

@pytest.fixture(params=['family_ranch','long_ranch'])
def house(request):
    name=request.param
    return tuple(json.loads((CONFIG/f).read_text()) for f in (f'floorplan_{name}.json',f'fire_{name}.json',f'scenario_{name}_full.json'))

def test_furnished_house_contract(house):
    validate(*house)

@pytest.mark.parametrize('damage',['overlap','disconnect','narrow','person','grid','reason','route'])
def test_rejects_unsafe_or_unexplained_house(house,damage):
    p,f,s=copy.deepcopy(house)
    if damage=='overlap': p['rooms'][1]['rect']=p['rooms'][2]['rect'][:]
    elif damage=='disconnect': p['doors']=[d for d in p['doors'] if 'outside' in d['between']]
    elif damage=='narrow': p['doors'][0]['width']=.89
    elif damage=='person': s['survivors'][0]['at']=p['landmarks'][0]['at'][:]
    elif damage=='grid': p['rooms'][0]['rect'][1]+=.01
    elif damage=='reason': del p['rooms'][0]['rect_reason']
    else: s['robots'][0]['route']=[[1,2]]
    with pytest.raises(AssertionError): validate(p,f,s)


def test_long_ranch_has_a_tiled_l_shape():
    p,f,s = (json.loads((CONFIG/n).read_text()) for n in ('floorplan_long_ranch.json','fire_long_ranch.json','scenario_long_ranch_full.json'))
    assert 10 <= len(p['rooms']) <= 12
    area = validate(p,f,s)
    assert 190 <= area <= 230
    bounds = p['footprint']
    assert area < (bounds[2]-bounds[0])*(bounds[3]-bounds[1])
    assert len(p['footprint_rects']) == 2


def test_validator_accepts_declared_notch_but_rejects_missing_room(house):
    p,f,s = copy.deepcopy(house)
    if 'footprint_rects' not in p:
        pytest.skip('Rectangular family ranch has no notch')
    validate(p,f,s)
    p['rooms'].pop()
    with pytest.raises(AssertionError):
        validate(p,f,s)


def test_service_room_can_use_clear_floor_beside_a_fixture():
    from validate_ranches import furniture_clear
    room=dict(id='bath',rect=[0,0,2.5,3])
    plan=dict(doors=[dict(id='entry',between=['hall','bath'],wall='z',at=[1.25,3],width=1,state='removed')])
    fire=dict(furniture=dict(sink=dict(size_m=[.95,.9844])))
    assert furniture_clear(dict(room='bath',kind='sink',at=[.795,.8122]),room,plan,fire,[])


def test_long_ranch_furnishes_its_named_spaces():
    plan=json.loads((CONFIG/'floorplan_long_ranch.json').read_text())
    kinds=lambda rid:{l['kind'] for l in plan['landmarks'] if l['room']==rid}
    assert {'apartment_counter','stove_and_hood','tall_steel_fridge','dining_table_six_chairs','apartment_modern_sofa'} <= kinds('lr_great')
    assert 'tall_wardrobe' in kinds('lr_closet')
    assert 'clawfoot_bathtub' in kinds('lr_ensuite')
    assert 'sink_with_mirror' in kinds('lr_bath')
    for rid in ('lr_primary','lr_bed2','lr_bed3','lr_bed4'):
        assert kinds(rid) & {'apartment_bed','blue_double_bed'}


def test_long_ranch_rejects_a_person_body_crossing_a_wall():
    p,f,s=(json.loads((CONFIG/n).read_text()) for n in ('floorplan_long_ranch.json','fire_long_ranch.json','scenario_long_ranch_full.json'))
    s['survivors'][0]['body_size_m']=[100,100]
    s['survivors'][0]['body_size_m_reason']='Deliberately impossible test body.'
    with pytest.raises(AssertionError): validate(p,f,s)


def test_validator_rejects_a_person_sealed_behind_a_bed():
    p,f,s=(json.loads((CONFIG/n).read_text()) for n in ('floorplan_long_ranch.json','fire_long_ranch.json','scenario_long_ranch_full.json'))
    room=next(r for r in p['rooms'] if r['id']=='lr_bed3')
    x0,z0,x1,z1=room['rect']
    f['furniture']['barrier_bed']=dict(f['furniture']['blue_double_bed'],size_m=[2,z1-z0-.64])
    p['landmarks']=[l for l in p['landmarks'] if l['room']!=room['id']]+[dict(room=room['id'],kind='barrier_bed',at=[(x0+x1)/2,(z0+z1)/2],yaw=0,at_reason='Test barrier across the bedroom.',yaw_reason='Test orientation.')]
    person=next(s for s in s['survivors'] if s['room']==room['id'])
    person['at']=[x1-1,(z0+z1)/2]
    with pytest.raises(AssertionError): validate(p,f,s)
