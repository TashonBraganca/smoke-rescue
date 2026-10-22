"""Colour detections need a body range, never a median of floor behind the box."""
from dataclasses import replace
from pathlib import Path
import math
import numpy as np
import pytest
from brain.agent import vision as V
from brain.agent.tools import AgentContext, ToolRegistry
from test_agent_vision import scene

ROOT = Path(__file__).resolve().parents[2]
REC = ROOT/'data/recordings/unity_house_onedoor_seed922_run1/robot_A'


def test_floor_only_box_has_no_depth_range():
    f = scene()
    import json
    mount = json.loads((ROOT/'config/sensors.json').read_text())['camera']['mount']
    pitch = math.radians(mount['pitch_down_deg'])
    rows = np.arange(f.rgb.shape[0])[:,None]
    down = math.sin(pitch)+(rows-f.K[1,2])/f.K[1,1]*math.cos(pitch)
    f.depth_m = np.broadcast_to(np.where(down>0,mount['height_m']/np.maximum(down,1e-6),np.nan),f.depth_m.shape).astype(np.float32).copy()
    hit = V._rgb_person(f,(50,30,110,118),.9)
    assert hit['range_m'] is None, 'floor pixels do not measure the person in a colour box'


def test_recorded_person_uses_lidar_when_depth_only_sees_floor(tmp_path, monkeypatch):
    if not REC.exists(): pytest.skip('local recording absent')
    f = V.load_frame(REC,37.4166666667)
    assert V._body_point(f,(251,257,343,372)) is None
    hit = V._rgb_person(f,(251,257,343,372),.92297)
    assert hit['range_source']=='lidar'
    assert 2.7 < hit['range_m'] < 3.2
    assert hit['range_evidence']['beams']
    # Use the robot's returned measurement through the real confirmation gate.
    monkeypatch.setattr(V,'detect_people_rgb',lambda frame:[hit])
    pose = AgentContext('robot_A',37.4166666667,-1.05747521,3.8536,-88.74788)
    reg = ToolRegistry(REC,pose,tmp_path)
    result = reg.call('inspect_candidate',{'bearing_deg':-3.098203,'range_m':4.5555})
    assert result['verdict']=='person'
    assert result['candidate']['range_m']==hit['range_m']
    assert reg.call('confirm_survivor',{'x':result['x'],'y':result['y'],'evidence_id':result['id']}).get('ok') is True
    # The old position is no longer accepted as the location this inspection measured.
    assert reg.call('confirm_survivor',{'x':-5.61,'y':3.71,'evidence_id':result['id']}).get('refused') is True


@pytest.mark.parametrize('control',['stale','behind_box','dropout','mixed_ranges'])
def test_lidar_without_consistent_body_support_leaves_range_unknown(control):
    if not REC.exists(): pytest.skip('local recording absent')
    from copy import copy
    f = copy(V.load_frame(REC,37.4166666667))
    scan = f.lidar
    ranges = scan.ranges_m.copy()
    if control=='stale': scan = replace(scan,t=0.)
    if control=='behind_box': ranges[:]=8.
    if control=='dropout': ranges[:]=0.
    if control=='mixed_ranges': ranges[::2]=8.
    f.lidar=replace(scan,ranges_m=ranges)
    hit=V._rgb_person(f,(251,257,343,372),.9)
    assert hit['range_m'] is None


@pytest.mark.parametrize('seed,robot,t,expected_range,can_confirm',[
    (924,'robot_B',56.56666666666667,4.56,True),
    (927,'robot_A',109.68333333333334,8.40,False),
])
def test_recorded_separate_person_is_not_rejected_with_a_warm_strip(seed,robot,t,expected_range,can_confirm,tmp_path):
    import json
    rec=ROOT/f'data/recordings/unity_house_onedoor_seed{seed}_run1'
    if not rec.exists(): pytest.skip('local recording absent')
    trace=[json.loads(l) for l in (rec/'agent_trace.jsonl').read_text().splitlines()]
    start=next(r for r in trace if r['kind']=='step_start' and r.get('robot')==robot and abs(r['t']-t)<1e-6)
    call=next(r for r in trace if r['kind']=='tool' and r['name']=='inspect_candidate' and r['step']==start['step'])
    pose=start['context']['pose']
    reg=ToolRegistry(rec/robot,AgentContext(robot,t,pose['x'],pose['y'],pose['heading_deg']),tmp_path)
    result=reg.call('inspect_candidate',call['arguments'])
    assert result['verdict']=='person'
    assert result['candidate']['source']=='rgb_dnn'
    assert result['candidate']['range_m']==pytest.approx(expected_range,abs=.02)
    rejected=result['cues']['rejected_thermal_object']
    assert rejected['verdict']=='not_person' and rejected['candidate']['bbox'][3]==1
    action=reg.call('confirm_survivor',{'x':result['x'],'y':result['y'],'evidence_id':result['id']})
    assert bool(action.get('ok'))==can_confirm
    if not can_confirm: assert 'confirm only within' in action['reason']


def test_lidar_filled_warm_patch_cannot_supply_vertical_body_shape(tmp_path,monkeypatch):
    rec=ROOT/'data/recordings/unity_house_onedoor_seed928_run1'
    if not rec.exists(): pytest.skip('local recording absent')
    frame=V.load_frame(rec/'robot_B',106.)
    monkeypatch.setattr(V,'detect_people_rgb',lambda frame:[])
    comp=next(c for c in V.scan_thermal(frame)['people'] if c['bbox']==[137,85,73,120])
    result=V.inspect_candidate(frame,comp['bearing_deg'],comp['range_m'] or 3.,output_dir=tmp_path)
    assert result['cues']['range_source']=='lidar'
    assert result['cues']['depth']['registered_fraction']<.6
    assert result['verdict']=='uncertain'
    assert not result['cues']['depth']['body_shape_supported']
