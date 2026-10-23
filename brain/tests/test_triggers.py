"""Trigger evaluation must distinguish missing situations from observed responses."""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('check_triggers',ROOT/'tools/check_triggers.py')
triggers=importlib.util.module_from_spec(spec)
spec.loader.exec_module(triggers)


def test_missing_logs_fail_every_fix(tmp_path):
    result=triggers.check(tmp_path,['12','14','36','43','55','66','19','23','79','57','59'])
    assert all(not row['pass'] for row in result.values())


def test_meeting_requires_both_elapsed_contact_loss_and_unshared_map(tmp_path):
    robot=tmp_path/'robot_A'; robot.mkdir()
    path=robot/'decisions.jsonl'
    for seconds,cells,expected in [(44,2000,False),(45,1499,False),(45,1500,True)]:
        path.write_text(json.dumps(dict(t=48,kind='rendezvous',numbers=dict(out_of_contact_s=seconds,unshared_cells=cells)))+'\n')
        assert triggers.check(tmp_path,['57'])['57']['pass'] is expected


def test_heat_retreat_needs_air_and_motion_not_just_event(tmp_path):
    robot=tmp_path/'robot_A';robot.mkdir()
    (robot/'decisions.jsonl').write_text(json.dumps(dict(t=2,kind='retreat'))+'\n')
    (robot/'air_temp.txt').write_text('# t temperature_c\n2 80\n')
    assert not triggers.check(tmp_path,['12'])['12']['pass']
    (robot/'brain_state.jsonl').write_text('\n'.join(json.dumps(dict(t=t,x=0,y=y)) for t,y in [(2,0),(3,.5)]))
    assert triggers.check(tmp_path,['12'])['12']['pass']


def test_two_streams_requires_overlap_and_the_same_measured_fire(tmp_path):
    def streams(t, second_target_x=0):
        contexts=[]
        for robot,x in [('robot_A',0),('robot_B',second_target_x)]:
            contexts.append(dict(kind='step_start',robot=robot,t=0,context=dict(pose=dict(x=x,y=0,heading_deg=0),
                spray_option=dict(bearing_deg=0,distance_m=3,assessment='two robots spraying together remove more than it releases'))))
            contexts.append(dict(kind='step_end',step=robot+'@0.000',orders=[dict(type='spray_water',bearing_deg=0)]))
        (tmp_path/'agent_trace.jsonl').write_text('\n'.join(map(json.dumps,contexts)))
        (tmp_path/'water.jsonl').write_text('\n'.join(json.dumps(dict(t=at,robot=robot,flow_lps=1,seconds=2)) for robot,at in [('robot_A',0),('robot_B',t)]))
    streams(3)
    assert not triggers.check(tmp_path,['59'],shared_gate=True)['59']['pass']
    streams(1,second_target_x=8)
    assert not triggers.check(tmp_path,['59'],shared_gate=True)['59']['pass']
    streams(1)
    assert triggers.check(tmp_path,['59'],shared_gate=True)['59']['pass']
    assert not triggers.check(tmp_path,['59'],shared_gate=False)['59']['pass'], 'unlinked map coordinates cannot be compared'


def test_scenarios_use_only_world_changes_and_real_fire_schema():
    for path in (ROOT/'config').glob('scenario_trigger_*.json'):
        scenario=json.loads(path.read_text())
        assert scenario['inherits']=='scenario_house_onedoor_seed925.json'
        assert 'robots' not in scenario and 'brain' not in scenario
        assert scenario['target_fixes']
        fire=json.loads((ROOT/'config'/scenario['fire']['config_file']).read_text())
        assert 'spread' in fire and isinstance(fire['ignitions'],list) and fire['ignitions']
        main_spread=json.loads((ROOT/'config/fire.json').read_text())['spread']
        growth=('growth_alpha_kw_s2','growth_alpha_kw_s2_reason')
        assert {k:v for k,v in fire['spread'].items() if k not in growth}=={k:v for k,v in main_spread.items() if k not in growth}
        # Trigger rigs may use a faster NFPA 72 Annex B class (alpha = 1055/t_g^2 for t_g 75, 150 or 300 s), never a made-up rate.
        assert any(abs(fire['spread']['growth_alpha_kw_s2']-1055/tg**2)<5e-4 for tg in (75,150,300))
        if '57' in scenario['target_fixes']:
            assert scenario['duration_s']>45
            assert min(i['time_s'] for i in fire['ignitions'])>scenario['duration_s']+scenario['fire']['head_start_s']


def test_successful_spray_can_start_after_the_arm_reaches(tmp_path):
    robot=tmp_path/'robot_A';robot.mkdir()
    (robot/'decisions.jsonl').write_text(json.dumps(dict(t=2,kind='agent_order',reason='spray_water at 0 deg, OpenCV evidence abc.'))+'\n')
    (tmp_path/'water.jsonl').write_text(json.dumps(dict(t=3.1,robot='robot_A',flow_lps=1,seconds=2))+'\n')
    assert triggers.check(tmp_path,['19'])['19']['pass']


def test_geometric_cutoff_does_not_claim_fire_cut_the_exit(tmp_path):
    robot=tmp_path/'robot_A';robot.mkdir()
    (robot/'decisions.jsonl').write_text(json.dumps(dict(t=2,kind='escape_closing',numbers=dict(route_points=0)))+'\n')
    assert not triggers.check(tmp_path,['36'])['36']['pass']
    (tmp_path/'agent_trace.jsonl').write_text(json.dumps(dict(kind='step_start',robot='robot_A',t=2,context=dict(event=dict(kind='escape_closing'),fire_blocks_exit=dict(distance_m=2,burning_m2=1))))+'\n')
    assert triggers.check(tmp_path,['36'])['36']['pass']
