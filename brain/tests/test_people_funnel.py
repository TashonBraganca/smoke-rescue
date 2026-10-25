"""Evaluator bookkeeping keeps missing logs and unmatched bodies distinct."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('people_funnel', Path(__file__).resolve().parents[2] / 'tools/eval/people_funnel.py')
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)


def test_live_recording_only_reads_metadata(tmp_path):
    rec = tmp_path / 'unity_house_onedoor_seed928'
    rec.mkdir()
    (rec / 'meta.json').write_text(json.dumps({'complete': False}))
    # Deliberately no other files. An incomplete recording must never reach readers.
    assert F.completed_recordings(tmp_path, [928]) == ([], ['928: incomplete, metadata only read'])
    (rec / 'meta.json').write_text(json.dumps({'complete': True}))
    assert F.completed_recordings(tmp_path, [928]) == ([rec], [])


def test_target_matching_uses_gate_placement_and_exact_tolerance():
    survivors = [{'id': 'body', 'position_house': [10., 0., 4.]}]
    xy = F.transform_xy([0., 4.], np.array([10., 0., 0.]))
    assert F.nearest(xy, survivors) == ('body', 0.)
    assert F.nearest(xy + [F.MATCH_M, 0.], survivors)[1] <= F.MATCH_M + 1e-10


def test_component_on_occluded_body_stays_ambiguous():
    result = {'candidate': {'bbox': [10,10,10,10], 'range_m': 3.}}
    vis = {'visible': False, 'range_m': 3., 'uv_thermal': np.array([[5.,5.],[25.,5.],[25.,25.],[5.,25.]])}
    assert F.owner(result, {'body': vis}) == (None, True)
    assert F.owner(result, {'body': vis | {'visible': True}}) == ('body', False)
    assert F.owner({'candidate': None}, {'body': vis}) == (None, False)


def test_departure_requires_measured_motion_and_preserves_heat_event():
    survivor = {'position_house': [0., 0., 4.]}
    states = {'A': [{'t': 0., 'x': 0., 'y': 0.}, {'t': 1., 'x': 0., 'y': -2.}]}
    decisions = {'A': [{'t': .5, 'kind': 'retreat', 'reason': 'measured heat'}]}
    result = F.departure(survivor, 'A', 0., states, decisions, np.zeros(3))
    assert result['left'] and result['t'] == 1. and result['reason'] == 'fire'
    assert not F.departure(survivor, 'A', 1., states, decisions, np.zeros(3))['left']


def test_camera_eval_attributes_the_returned_person_to_its_body(tmp_path,monkeypatch):
    from brain.agent import eval_vision as E,vision as V
    from test_agent_vision import scene
    frame=scene()
    comp={'bbox':[0,0,10,10],'bearing_deg':0.,'range_m':3.,'size_plausible':False,'mean_c':32.}
    result={'verdict':'person','candidate':{'source':'rgb_dnn','bearing_deg':0.,'range_m':3.},
            'reasons':['colour person'], 'cues':{'rgb_dnn':{'hits':[{'bbox_rgb':[69,30,22,60],'range_m':3.}]}}}
    monkeypatch.setattr(V,'scan_thermal',lambda f:{'people':[comp]})
    monkeypatch.setattr(V,'inspect_candidate',lambda *a,**k:result)
    vis={'visible':True,'line_of_sight':True,'range_m':3.,'uv_thermal':np.array([[60,20],[100,20],[100,100],[60,100]],float)}
    monkeypatch.setattr(E,'survivor_visibility',lambda *a,**k:vis)
    # The helper's calibrated projection is tested on real frames by the funnel.
    monkeypatch.setattr(E,'inspection_selection',lambda *a,**k:('body',False))
    rows=E.label_frame(frame,np.eye(4),[{'id':'body','position_house':[0.,0.,3.]}],{},'run','A',0.,tmp_path)
    positive=next(r for r in rows if r['label'])
    negative=next(r for r in rows if not r['label'])
    assert positive['inspect'] and not positive['raw']
    assert not negative['inspect'] and negative['returned_person']
    assert negative['selected_person']=='body'


def test_nonperson_time_counts_only_moves_the_brain_accepted():
    raw={'id':'ev','reasons':['flat wall']}
    inspection={'step':'A@0','arguments':{'bearing_deg':0.,'range_m':3.},'raw':raw,'fused':{'verdict':'not_person'},
                'robot':'A','t':0.,'label_person':None,'ambiguous_component':False,'map_position':[0.,3.]}
    trace=[{'kind':'step_start','step':'A@0','robot':'A','t':0.},
           {'kind':'tool','step':'A@0','name':'inspect_candidate','result_id':'ev','arguments':inspection['arguments']},
           {'kind':'tool','step':'A@0','name':'approach_candidate','arguments':{'evidence_id':'ev'}}]
    states={'A':[{'t':0.,'x':0.,'y':0.,'yaw_rad':0.},{'t':3.,'x':0.,'y':1.,'yaw_rad':0.}]}
    before={'t':-1.,'kind':'replan','reason':'Agent goal has no known free path'}
    accepted={'A':[before,{'t':0.,'kind':'agent_order','reason':'go_to: closer look'},
                   {'t':3.,'kind':'agent_goal_reached','reason':'arrived'}]}
    args=(trace,[inspection],accepted,states,{'A':np.zeros(3)},[{'id':'body','position_house':[10.,0.,10.]}])
    result=F.nonperson_time(*args)
    assert result['movement_seconds']['approach_candidate']==3.
    assert result['intervals'][0]['observed_travel_m']==1.
    accepted['A'][1]['reason']='Refused go_to: hot'
    assert not F.nonperson_time(*args)['intervals']


@pytest.mark.parametrize('seed,t',[(924,38.),(925,48.)])
def test_recorded_colour_fallback_is_not_counted_as_a_wall_false_positive(seed,t,tmp_path):
    rec=Path(__file__).resolve().parents[2]/f'data/recordings/unity_house_onedoor_seed{seed}_run1'
    if not rec.exists() or next(rec.glob('**/rgb/*.png'), None) is None: pytest.skip('local recording absent or its image frames were removed')
    frame=F.V.load_frame(rec/'robot_A',t)
    pose=F.gt_pose_at(F.load_trajectory(rec,'robot_A'),t)
    rows=F.E.label_frame(frame,pose,F.load_survivors(rec),F.load_floorplan(rec),rec.name,'robot_A',t,tmp_path)
    assert not any(not r['label'] and r['inspect'] for r in rows)
    assert any(not r['label'] and r.get('returned_person') and r.get('selected_person') for r in rows)


def test_a_failed_inspection_does_not_read_sensors_or_invent_a_verdict(tmp_path, monkeypatch):
    # Run 932 A at 93.783 s sent an extra evidence_id. The live tool
    # refused it before reading a frame; replay must keep the same boundary.
    def unexpected(*args, **kwargs):
        raise AssertionError('a refused inspection did not reach the camera')
    monkeypatch.setattr(F.V, 'load_frame', unexpected)
    trace = [
        {'kind':'step_start','step':'A@1','robot':'A','t':1.,'context':{'pose':{'x':0.,'y':0.,'heading_deg':0.}}},
        {'kind':'tool','step':'A@1','name':'inspect_candidate',
         'arguments':{'bearing_deg':0.,'range_m':3.,'evidence_id':'extra'},
         'result_id':None,'error':'bad arguments for inspect_candidate'},
    ]
    rows, confirms = F.replay_inspections(tmp_path, tmp_path, [], {}, {'A':np.zeros(3)}, {}, trace)
    assert rows == [] and confirms == []
    assert F.inspection_errors(trace) == [trace[1]]
