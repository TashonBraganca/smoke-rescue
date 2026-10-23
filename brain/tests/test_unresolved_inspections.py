"""An inspected sighting remains visible to its robot until an accepted resolution."""
from pathlib import Path
import pytest
from brain.agent import tools as T
from brain.agent import vision as V

@pytest.fixture(autouse=True)
def isolated_queue(monkeypatch):
    for name in ('_unfinished', '_confirmed', '_pending', '_looks'):
        monkeypatch.setattr(T.ToolRegistry, name, {})


def reg(tmp_path,t=1.,robot='A',x=0.):
    return T.ToolRegistry(tmp_path,T.AgentContext(robot,t,x,0.,0.),tmp_path/'debug')


def fake_inspection(monkeypatch, verdict='uncertain', measured=True):
    monkeypatch.setattr(V,'load_frame',lambda *a:object())
    def inspect(frame,b,r,output_dir):
        return {'id':f'view-{b}-{r}','tool':'inspect_candidate','verdict':verdict,
                'candidate':{'bearing_deg':b,'range_m':r} if measured else None}
    monkeypatch.setattr(V,'inspect_candidate',inspect)


def test_inspection_without_an_approach_is_not_forgotten(tmp_path,monkeypatch):
    fake_inspection(monkeypatch)
    r=reg(tmp_path)
    r.call('inspect_candidate',{'bearing_deg':0.,'range_m':3.3})
    # No approach, prompt change, action order or new evidence is supplied.
    later=reg(tmp_path,1000.)
    [look]=later.unfinished()
    assert (look['x'],look['y'])==(0.,3.3)
    assert later.orders==[] and later.evidence=={}
    assert later.call('confirm_survivor',{'x':0.,'y':3.3,'evidence_id':'view-0.0-3.3'}).get('refused')


def test_many_inspected_sightings_are_kept_and_repeat_views_merge(tmp_path,monkeypatch):
    fake_inspection(monkeypatch)
    for i in range(7):
        reg(tmp_path,t=1.+i,x=3.*i).call('inspect_candidate',{'bearing_deg':0.,'range_m':3.3})
    assert len(reg(tmp_path,20.).unfinished())==7
    reg(tmp_path,21.).call('inspect_candidate',{'bearing_deg':0.,'range_m':3.3})
    assert len(reg(tmp_path,22.).unfinished())==7
    assert reg(tmp_path,22.,robot='B').unfinished()==[]


def test_only_an_accepted_resolution_removes_the_inspection(tmp_path,monkeypatch):
    fake_inspection(monkeypatch,'person')
    r=reg(tmp_path)
    result=r.call('inspect_candidate',{'bearing_deg':0.,'range_m':3.3})
    assert len(reg(tmp_path,1.5).unfinished()) == 1
    args={'x':result['x'],'y':result['y'],'evidence_id':result['id']}
    assert r.call('reject_candidate',args).get('refused')
    assert len(reg(tmp_path,2.).unfinished())==1
    assert r.call('confirm_survivor',args).get('ok')
    assert reg(tmp_path,3.).unfinished()==[]


def test_rejection_resolves_but_an_unknown_range_never_invents_a_spot(tmp_path,monkeypatch):
    fake_inspection(monkeypatch)
    r=reg(tmp_path)
    result=r.call('inspect_candidate',{'bearing_deg':0.,'range_m':3.3})
    fake_inspection(monkeypatch,'not_person')
    r=reg(tmp_path,2.)
    result=r.call('inspect_candidate',{'bearing_deg':0.,'range_m':3.3})
    assert r.call('reject_candidate',{'x':0.,'y':3.3,'evidence_id':result['id']}).get('ok')
    assert reg(tmp_path,3.).unfinished()==[]
    fake_inspection(monkeypatch,measured=False)
    reg(tmp_path,4.).call('inspect_candidate',{'bearing_deg':0.,'range_m':3.3})
    assert reg(tmp_path,5.).unfinished()==[]


def test_resolving_one_sighting_does_not_erase_a_distinct_neighbour(tmp_path,monkeypatch):
    fake_inspection(monkeypatch,'person')
    a=reg(tmp_path)
    first=a.call('inspect_candidate',{'bearing_deg':0.,'range_m':3.3})
    reg(tmp_path,2.,x=1.3).call('inspect_candidate',{'bearing_deg':0.,'range_m':3.3})
    assert len(reg(tmp_path,3.).unfinished()) == 2
    assert a.call('confirm_survivor',{'x':0.,'y':3.3,'evidence_id':first['id']}).get('ok')
    [remaining]=reg(tmp_path,4.).unfinished()
    assert remaining['x'] == 1.3
