"""Demand-driven sensor transfer through the real HTTP endpoints and real vision tools."""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from brain.agent import server, vision
from brain.agent.llm import Reply, ToolCall
from test_recording_transport import local, call, archive


def metadata(folder):
    from brain.agent.recording_transport import SMALL_FILES
    return {name:(folder/name).read_bytes() for name in SMALL_FILES if (folder/name).is_file()}


def start(local, monkeypatch, names, times=None):
    class Decide:
        model='test'
        def __init__(self):self.turn=0;self.measured=[]
        def chat(self,messages,tools):
            self.turn+=1
            if self.turn<=len(names):
                return Reply('',[ToolCall(names[self.turn-1],{})],0,self.model)
            self.measured=[json.loads(m['content']) for m in messages if m['role']=='tool']
            return Reply('',[ToolCall('finish',{'summary':'measured'})],0,self.model)
    model=Decide();monkeypatch.setattr(server,'_backend',model)
    assert call('PUT','/runs/test-run/robot_A/files',content=archive(metadata(local)),headers={'X-Recording-Source':str(local)}).status_code==200
    response=call('POST','/robot/robot_A/step',json={'transport':'pull-v1','robot_dir':'run://test-run/robot_A','t':max(times or [0.]),
        'frame_times':times or [0.], 'pose':{'x':0,'y':0,'heading_deg':0}})
    assert response.status_code==202,response.text
    return response.json()['poll_url'],model


def finish(url, local, *, supply=True):
    requested=[]
    for _ in range(200):
        response=call('GET',url);assert response.status_code==200,response.text
        state=response.json()
        if state['status'] in ('complete','failed'):
            return state,requested
        if state['status']=='needs_files':
            wanted=state['files'];requested.extend(wanted)
            if not supply:return state,requested
            payload={name:(local/name).read_bytes() for name in wanted}
            upload=call('PUT','/runs/test-run/robot_A/files',content=archive(payload))
            assert upload.status_code==200,upload.text
        time.sleep(.01)
    pytest.fail('pull decision did not finish')


def history(local):
    for sensor in ('rgb','depth','thermal'):
        original=cv2.imread(str(local/sensor/'0.png'),cv2.IMREAD_UNCHANGED)
        for stamp in (.5,1.):
            image=original.copy()
            if sensor=='thermal':image[20:70,30:int(75+20*stamp)]=5500
            assert cv2.imwrite(str(local/sensor/(str(stamp)+'.png')),image)
        (local/(sensor+'.txt')).write_text(''.join(f'{t} {sensor}/{t if t else "0"}.png\n' for t in (0,.5,1.)))


def test_finish_without_perception_requests_no_pngs(local,monkeypatch):
    url,_=start(local,monkeypatch,[])
    state,requested=finish(url,local)
    assert state['status']=='complete',state
    assert state['result']['summary']=='measured'
    assert requested==[]


def test_latest_scan_pulls_only_latest_original_pngs(local,monkeypatch):
    history(local)
    expected=vision.scan_thermal(vision.load_frame(str(local),1.))
    url,model=start(local,monkeypatch,['scan_thermal'],[0.,.5,1.])
    state,requested=finish(url,local)
    assert state['status']=='complete',state
    assert set(requested)=={'rgb/1.0.png','depth/1.0.png','thermal/1.0.png'}
    assert model.measured[0]['id']==expected['id']
    remote=server.RUN_DIR/'recordings/test-run/robot_A'
    for name in requested:assert (remote/name).read_bytes()==(local/name).read_bytes()


def test_fire_tracking_pulls_depth_and_thermal_history_without_rgb(local,monkeypatch):
    history(local)
    expected=vision.track_fire_spread([vision.load_frame(str(local),t) for t in (0.,.5,1.)])
    url,model=start(local,monkeypatch,['track_fire_spread'],[0.,.5,1.])
    state,requested=finish(url,local)
    assert state['status']=='complete',state
    assert len(requested)==6 and all(name.startswith(('depth/','thermal/')) for name in requested)
    for key in ('areas_m2','hot_pixels','growth_px_s','growth_m2_s','speed_m_s','direction_deg','hot_depth_rate_m_s'):
        assert model.measured[0][key]==expected[key]


def test_repeated_tools_share_acknowledged_frames(local,monkeypatch):
    url,_=start(local,monkeypatch,['scan_thermal','assess_visibility'])
    state,requested=finish(url,local)
    assert state['status']=='complete',state
    assert len(requested)==3


def test_unserved_pull_can_be_cancelled_and_does_not_become_measurement(local,monkeypatch):
    url,_=start(local,monkeypatch,['scan_thermal'])
    state,requested=finish(url,local,supply=False)
    assert state['status']=='needs_files' and len(requested)==3
    assert call('DELETE',url).status_code==200
    final,_=finish(url,local,supply=False)
    assert final['status']=='failed' and 'cancel' in final['error'].lower()


def test_pull_robot_must_match_uri(local,monkeypatch):
    assert call('PUT','/runs/test-run/robot_A/files',content=archive(metadata(local))).status_code==200
    response=call('POST','/robot/robot_B/step',json={'transport':'pull-v1','robot_dir':'run://test-run/robot_A','t':0})
    assert response.status_code==400


def test_people_request_pulls_only_rgb_and_depth(local,monkeypatch):
    monkeypatch.setattr(vision,'detect_people_rgb',lambda frame,**kw: [vision._rgb_person(frame,(60,25,100,95),.9)])
    expected=vision._rgb_person(vision.load_frame(str(local),0.),(60,25,100,95),.9)
    assert call('PUT','/runs/test-run/robot_A/files',content=archive(metadata(local))).status_code==200
    response=call('POST','/perceive/people',json={'transport':'pull-v1','robot_dir':'run://test-run/robot_A','t':0})
    assert response.status_code==202,response.text
    state,names=finish(response.json()['poll_url'],local)
    assert state['status']=='complete',state
    assert set(names)=={'rgb/0.png','depth/0.png'}
    assert state['result']['people']==[expected]


def test_job_snapshot_survives_a_new_index_upload(local,monkeypatch):
    url,_=start(local,monkeypatch,['scan_thermal'])
    state,_=finish(url,local,supply=False)
    assert state['status']=='needs_files'
    # A newer sensor index cannot change the pending decision's selected original files.
    bad={s+'.txt':f'9 {s}/9.png\n' for s in ('rgb','depth','thermal')}
    assert call('PUT','/runs/test-run/robot_A/files',content=archive(bad)).status_code==200
    state,names=finish(url,local)
    assert state['status']=='complete',state
    assert set(names)=={'rgb/0.png','depth/0.png','thermal/0.png'}


def test_history_selection_is_capped_exactly_like_the_existing_tool(local):
    from brain.agent.recording_transport import tool_frame_times
    times=[i/15 for i in range(45)]
    assert tool_frame_times(times[-1],times)==[times[int(i*4.5)] for i in range(10)]
    with pytest.raises(ValueError):tool_frame_times(0,[1])
    with pytest.raises(ValueError):tool_frame_times(float('nan'),[])


def test_two_robots_pull_their_own_files_concurrently(local,monkeypatch):
    from test_agent_lidar import recording
    other=local.parent/'robot_B';recording(other,hot=True)
    class ScanBoth:
        model='test'
        def chat(self,messages,tools):
            if messages[-1]['role']=='user':return Reply('',[ToolCall('scan_thermal',{})],0,self.model)
            return Reply('',[ToolCall('finish',{'summary':str(len(json.loads(messages[-1]['content'])['hotspots']))})],0,self.model)
    monkeypatch.setattr(server,'_backend',ScanBoth())
    jobs=[]
    for folder in (local,other):
        robot=folder.name
        assert call('PUT',f'/runs/test-run/{robot}/files',content=archive(metadata(folder)),headers={'X-Recording-Source':str(folder)}).status_code==200
        response=call('POST',f'/robot/{robot}/step',json={'transport':'pull-v1','robot_dir':f'run://test-run/{robot}','t':0,'pose':{'x':0,'y':0,'heading_deg':0}})
        assert response.status_code==202,response.text
        jobs.append((folder,response.json()['poll_url']))
    results={}
    from brain.agent.recording_transport import PULL_LIMITS
    # Two real OpenCV inferences share a lock. Poll for the transport's bounded
    # job lifetime instead of treating two seconds of CPU contention as failure.
    deadline=time.monotonic()+PULL_LIMITS['wait_s']
    while time.monotonic()<deadline:
        for folder,url in jobs:
            state=call('GET',url).json()
            if state['status']=='needs_files':
                assert call('PUT',f'/runs/test-run/{folder.name}/files',content=archive({name:(folder/name).read_bytes() for name in state['files']})).status_code==200
            if state['status']=='complete':results[folder.name]=state['result']['summary']
            assert state['status']!='failed',state
        if len(results)==2:break
        time.sleep(.01)
    assert results=={'robot_A':'0','robot_B':'1'}


def test_real_history_preserves_every_measurement_without_rgb_history():
    from brain.agent.recording_transport import thermal_depth_frame
    folder=Path(__file__).resolve().parents[2]/'data/recordings/unity_house_onedoor_seed924_run1/robot_B'
    if not folder.is_dir():pytest.skip('local seed 924 recording absent')
    times=[19.,19.5,20.]
    expected=vision.track_fire_spread([vision.load_frame(str(folder),t) for t in times])
    actual=vision.track_fire_spread([thermal_depth_frame(folder,t) for t in times])
    assert {k:v for k,v in actual.items() if k!='id'}=={k:v for k,v in expected.items() if k!='id'}


def test_depth_fusion_pull_and_remote_map_download(local,monkeypatch):
    assert call('PUT','/runs/test-run/robot_A/files',content=archive(metadata(local))).status_code==200
    response=call('POST','/map3d/robot_A/integrate',json={'transport':'pull-v1','robot_dir':'run://test-run/robot_A','t':0,
        'pose':{'x':0,'y':0,'heading_deg':0}})
    assert response.status_code==202,response.text
    try:
        state,names=finish(response.json()['poll_url'],local)
        assert state['status']=='complete',state
        assert set(names)=={'depth/0.png'} and state['result']['ok']
        exported=call('POST','/map3d/robot_A/export',json={'robot_dir':'run://test-run/robot_A','t':0})
        assert exported.status_code==200,exported.text
        downloaded=call('GET',exported.json()['download_url'])
        assert downloaded.status_code==200 and downloaded.content.startswith(b'ply\n')
    finally:
        for volume in server._maps3d.values():volume.close()
        server._maps3d.clear()


def test_snapshot_paths_do_not_reset_agent_memory_between_steps(local,monkeypatch):
    from brain.agent.tools import ToolRegistry
    url,_=start(local,monkeypatch,[])
    assert finish(url,local)[0]['status']=='complete'
    kept={'x':3,'y':4,'t':0,'dropped_because':'no path to it'}
    ToolRegistry._unfinished['robot_A']=[kept]
    url,_=start(local,monkeypatch,[])
    assert finish(url,local)[0]['status']=='complete'
    assert ToolRegistry._unfinished['robot_A']==[kept]


def test_missing_upload_fails_after_the_transport_deadline(local,monkeypatch):
    from brain.agent.recording_transport import PULL_LIMITS
    monkeypatch.setitem(PULL_LIMITS,'wait_s',.05)
    url,_=start(local,monkeypatch,['scan_thermal'])
    for _ in range(100):
        state=call('GET',url).json()
        if state['status']=='failed':break
        time.sleep(.01)
    assert state['status']=='failed' and 'timed out' in state['error']
