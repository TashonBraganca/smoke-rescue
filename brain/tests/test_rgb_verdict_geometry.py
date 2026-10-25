"""A colour verdict must carry the geometry measured for that colour detection."""
from pathlib import Path
import pytest
from brain.agent import vision as V
from brain.agent.tools import ToolRegistry, AgentContext
from test_agent_vision import scene


def test_colour_fallback_does_not_keep_a_different_thermal_range(tmp_path,monkeypatch):
    f=scene('uniform')
    hit={'confidence':.9,'bbox_rgb':[60,30,30,70],'bearing_deg':2.,'range_m':4.2,'range_source':'depth'}
    monkeypatch.setattr(V,'detect_people_rgb',lambda frame:[hit])
    r=V.inspect_candidate(f,0.,3.,output_dir=tmp_path)
    assert r['verdict']=='person'
    assert r['candidate']['range_m']==4.2
    assert r['candidate']['bearing_deg']==2.
    assert r['candidate']['source']=='rgb_dnn'


def test_unranged_colour_person_does_not_borrow_a_warm_objects_range(tmp_path,monkeypatch):
    f=scene('uniform')
    hit={'confidence':.9,'bbox_rgb':[60,30,30,70],'bearing_deg':0.,'range_m':None,'range_source':None}
    monkeypatch.setattr(V,'detect_people_rgb',lambda frame:[hit])
    monkeypatch.setattr(V,'load_frame',lambda *a:f)
    reg=ToolRegistry(tmp_path,AgentContext('test_rgb',0.,0.,0.,0.),tmp_path)
    r=reg.call('inspect_candidate',{'bearing_deg':0.,'range_m':3.})
    assert r['verdict']=='person' and r['candidate']['range_m'] is None
    assert reg.call('confirm_survivor',{'x':r['x'],'y':r['y'],'evidence_id':r['id']}).get('refused')


def test_recorded_colour_person_outside_confirmation_range_stays_outside(tmp_path):
    rec=Path(__file__).resolve().parents[2]/'data/recordings/unity_house_onedoor_seed932_nolimit_run1/robot_B'
    if not rec.exists() or next(rec.glob('**/rgb/*.png'), None) is None: pytest.skip('local recording absent or its image frames were removed')
    f=V.load_frame(rec,82.)
    comp=next(c for c in V.scan_thermal(f)['people'] if c['bbox']==[0,176,14,8])
    reg=ToolRegistry(rec,AgentContext('recorded_rgb',82.,0.,0.,0.),tmp_path)
    r=reg.call('inspect_candidate',{'bearing_deg':comp['bearing_deg'],'range_m':comp['range_m'] or 3.})
    assert r['verdict']=='person'
    assert r['candidate']['range_m']==pytest.approx(5.096428,abs=.000001)
    guard=reg.call('confirm_survivor',{'x':r['x'],'y':r['y'],'evidence_id':r['id']})
    assert guard.get('refused') and 'confirm only within' in guard['reason']
