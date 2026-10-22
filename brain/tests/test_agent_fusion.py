"""Repeated partial views can confirm; replaying one frame and missing depth cannot."""
from copy import deepcopy
import math
from pathlib import Path
import numpy as np
import pytest
from brain.agent import vision as V
from brain.agent.tools import AgentContext, ToolRegistry
from test_agent_vision import scene

ROOT = Path(__file__).resolve().parents[2]


def partial(t):
    f = scene(t=t)
    # Crop the sensor, keeping the measured curved body and its temperature variation.
    f.thermal = f.thermal[:80].copy()
    return f


def result(t, tmp_path):
    return V.inspect_candidate(partial(t), 0., 3., output_dir=tmp_path)


def test_distinct_translated_partial_views_confirm_without_colour(tmp_path, monkeypatch):
    monkeypatch.setattr(V, 'detect_people_rgb', lambda f: [])
    history = V.InspectionHistory()
    a, b = result(0., tmp_path), result(1., tmp_path)
    assert a['verdict'] == b['verdict'] == 'uncertain'
    assert history.observe(a, position=(0., 3.), observer=(0., 0.))['verdict'] == 'uncertain'
    fused = history.observe(b, position=(0., 3.), observer=(.5, 0.))
    assert fused['verdict'] == 'person'
    assert len(fused['cues']['temporal']['evidence_ids']) == 2
    assert a['verdict'] == b['verdict'] == 'uncertain', 'the single-frame result remains inspectable'


@pytest.mark.parametrize('control', ['same_frame', 'same_view', 'old', 'different_object', 'lidar_only', 'ramp', 'depth_span', 'negative'])
def test_correlated_or_contradictory_evidence_cannot_accumulate(control, tmp_path, monkeypatch):
    monkeypatch.setattr(V, 'detect_people_rgb', lambda f: [])
    history = V.InspectionHistory()
    a, b = result(0., tmp_path), result(1., tmp_path)
    position, observer = (0., 3.), (.5, 0.)
    if control == 'same_frame': b = deepcopy(a)
    if control == 'same_view': observer = (0., 0.)
    if control == 'old': b['t'] = 120.
    if control == 'different_object': position = (2., 3.)
    if control == 'lidar_only':
        a['cues']['range_source'] = b['cues']['range_source'] = 'lidar'
    if control == 'ramp': b['cues']['temperature']['band_crossing_ramp'] = True
    if control == 'depth_span': b['cues']['depth']['single_surface'] = False
    if control == 'negative': b['verdict'] = 'not_person'
    history.observe(a, position=(0., 3.), observer=(0., 0.))
    assert history.observe(b, position=position, observer=observer)['verdict'] != 'person'


def test_clock_rewind_discards_previous_run(tmp_path, monkeypatch):
    monkeypatch.setattr(V, 'detect_people_rgb', lambda f: [])
    history=V.InspectionHistory()
    history.observe(result(10., tmp_path), position=(0.,3.), observer=(0.,0.))
    assert history.observe(result(0., tmp_path), position=(0.,3.), observer=(.5,0.))['verdict']=='uncertain'


def test_real_desk_person_two_partial_views_with_colour_disabled(tmp_path, monkeypatch):
    folder=ROOT/'data/recordings/unity_house_onedoor_seed925_run1/robot_A'
    if not folder.exists(): pytest.skip('local recording absent')
    monkeypatch.setattr(V, 'detect_people_rgb', lambda f: [])
    # This location comes from the first warm inspection, not a scene file.
    target=(-3.61,3.17)
    import json
    states=[json.loads(l) for l in (folder/'brain_state.jsonl').open()]
    history=V.InspectionHistory(); results=[]
    for t in (33.,34.):
        p=min(states,key=lambda p:abs(p['t']-t)); dx,dy=target[0]-p['x'],target[1]-p['y']
        b=(math.degrees(math.atan2(dx,dy))+math.degrees(p['yaw_rad'])+180)%360-180
        r=V.inspect_candidate(V.load_frame(folder,t),b,math.hypot(dx,dy),output_dir=tmp_path)
        assert r['verdict']=='uncertain'
        c=r['candidate']; a=-p['yaw_rad']+math.radians(c['bearing_deg'])
        xy=(p['x']+c['range_m']*math.sin(a),p['y']+c['range_m']*math.cos(a))
        results.append(history.observe(r,position=xy,observer=(p['x'],p['y'])))
    assert [r['verdict'] for r in results]==['uncertain','person']


def test_registry_remembers_evidence_between_decision_steps(tmp_path, monkeypatch):
    folder=tmp_path/'robot_A'; folder.mkdir()
    monkeypatch.setattr(V,'detect_people_rgb',lambda f: [])
    monkeypatch.setattr(V,'load_frame',lambda d,t: partial(t))
    first=ToolRegistry(folder,AgentContext('robot_A',0.,0.,0.,0.),tmp_path/'crops')
    assert first.call('inspect_candidate',{'bearing_deg':0.,'range_m':3.})['verdict']=='uncertain'
    # Half a metre of lateral translation and the corresponding bearing keep the same map location.
    second=ToolRegistry(folder,AgentContext('robot_A',1.,.5,0.,-math.degrees(math.atan2(.5,3))),tmp_path/'crops')
    r=second.call('inspect_candidate',{'bearing_deg':0.,'range_m':3.})
    assert r['verdict']=='person'
    assert second.call('confirm_survivor',{'x':r['x'],'y':r['y'],'evidence_id':r['id']}).get('refused') is not True


def test_recorded_side_sliver_with_last_pixel_box_stays_uncertain(tmp_path):
    folder=ROOT/'data/recordings/unity_house_onedoor_seed923_run1/robot_A'
    if not folder.exists(): pytest.skip('local recording absent')
    frame=V.load_frame(folder,40.)
    comp=next(c for c in V.scan_thermal(frame)['people'] if c['bbox']==[270,209,50,47])
    assert V.inspect_candidate(frame,comp['bearing_deg'],comp['range_m'],output_dir=tmp_path)['verdict']=='uncertain'
