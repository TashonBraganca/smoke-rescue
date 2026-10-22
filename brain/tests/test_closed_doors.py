"""Closed door candidates need independent panel and depth evidence, not a flat wall."""
import json
import numpy as np
import pytest
from brain.agent import vision


def door_scene(kind='door', z=2.5, bearing=0):
    h,w=240,320; k=np.array([[240.,0,159.5],[0,240.,119.5],[0,0,1.]])
    rgb=np.full((h,w,3),70,np.uint8)
    depth=np.full((h,w),z-.06,np.float32)
    left,right,top,bottom=115,205,18,222
    if kind!='wall':
        rgb[top:bottom,left:right]=140
        depth[top:bottom,left:right]=z
    if kind in ('door','coplanar','no_frame'):
        rgb[128:134,187:201]=200
        depth[128:134,187:201]=z-.12
    if kind=='coplanar':
        depth[:]=z; depth[128:134,187:201]=z-.12
    if kind=='poster':
        depth[:]=z
    if kind=='dropout':
        depth[top:bottom,left:right]=0
    if kind=='window':
        rgb[:]=70;depth[:]=z-.06
        rgb[18:150,left:right]=140;depth[18:150,left:right]=z
    if kind=='clipped':
        rgb[:]=70;rgb[:bottom,left:right]=140
    if kind=='no_frame':
        rgb[:]=70
    return vision.Frame(t=0,rgb=rgb,depth_m=depth,thermal=np.full((h,w),2932,np.uint16),K=k,thermal_K=k,
                        rgb_to_thermal=np.eye(4),stamps=dict(rgb=0,depth=0,thermal=0))


@pytest.mark.parametrize('kind',['door','coplanar'])
def test_panel_plane_and_handle_are_a_candidate_not_a_passable_gap(kind):
    f=door_scene(kind)
    result=vision.measure_opening(f,0)
    assert not result['passable'] and result['clear_width_m'] is None
    door=result['door']
    assert door['verdict']=='closed_door_candidate'
    assert door['panel_width_m']==pytest.approx(.94,abs=.05)
    assert door['handles']
    assert door['opening_direction']=='unknown' and door['openable'] is None
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('kind',['wall','poster','dropout','window','clipped','no_frame'])
def test_wall_poster_dropout_window_and_clipping_do_not_certify_a_door(kind):
    door=vision.measure_opening(door_scene(kind),0)['door']
    assert door['verdict']=='unknown'
    assert door['openable'] is None


def test_bearing_and_nonplanarity_controls():
    f=door_scene()
    assert vision.detect_closed_door(f,35)['verdict']=='unknown'
    yy,xx=np.indices(f.depth_m.shape)
    f.depth_m[18:222,115:205] += (.25*np.sin(xx[18:222,115:205]/8)*np.cos(yy[18:222,115:205]/10)).astype(np.float32)
    assert vision.detect_closed_door(f,0)['verdict']=='unknown'


def test_detector_does_not_edit_sensor_arrays():
    f=door_scene();before=[a.copy() for a in (f.rgb,f.depth_m,f.thermal)]
    a=vision.detect_closed_door(f,0);b=vision.detect_closed_door(f,0)
    assert a==b
    for old,new in zip(before,(f.rgb,f.depth_m,f.thermal)):
        np.testing.assert_array_equal(old,new)


def test_open_door_requires_the_measured_candidate_ahead(tmp_path, monkeypatch):
    from brain.agent.tools import AgentContext, ToolRegistry
    for kind,allowed in [('wall',False),('poster',False),('door',True)]:
        monkeypatch.setattr(vision,'load_frame',lambda *args, kind=kind:door_scene(kind))
        tools=ToolRegistry(tmp_path,AgentContext('robot_A',0,0,0,0))
        measured=tools.call('measure_opening',{'bearing_deg':0})
        action=tools.call('open_door',{'evidence_id':measured['id']})
        assert bool(action.get('refused')) is not allowed
