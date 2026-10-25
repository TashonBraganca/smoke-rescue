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
