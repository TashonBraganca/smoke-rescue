"""Exercise real gallery geometry against edited-layout safety and cache identity."""
import copy
import json
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/ryor')]
import layout
from placement import validate


def fixture():
    house = json.loads((ROOT / 'data/runs/ryor/gallery.json').read_text())['houses'][0]
    rooms = [{'id': r['id'], 'rect': r['rect'][:]} for r in house['placement_geometry']['rooms']]
    return house, rooms


def test_swap_rebuilds_real_geometry_and_preserves_outline():
    house, rooms = fixture()
    a, b = [next(r for r in rooms if r['id'] == rid) for rid in ('fr_bed1', 'fr_bed2')]
    a['rect'], b['rect'] = b['rect'], a['rect']
    edited = layout.build(house, rooms)
    assert edited['placement_geometry']['rooms'] != house['placement_geometry']['rooms']
    assert edited['furniture'] != house['furniture']
    assert edited['bounds'] == house['bounds'] and edited['door'] == house['door']
    assert all(d['width'] >= .9 for d in edited['placement_geometry']['doors'])
    placement = layout.generated_placement(edited)
    assert validate(edited, placement) == placement and len(placement['people']) >= 3
    baseline = [{'id': r['id'], 'rect': r['rect']} for r in house['placement_geometry']['rooms']]
    assert layout.key('family_ranch__r2__medium', rooms, placement, 'tag') != layout.key('family_ranch__r2__medium', baseline, placement, 'tag')
    assert layout.key('family_ranch__r2__medium', rooms, placement, 'tag') != layout.key('family_ranch__r2__medium', rooms, placement, 'new')


@pytest.mark.parametrize('change,message', [
    ('overlap', 'overlaps'), ('outside', 'outline'), ('resize', 'dimensions'),
    ('offgrid', '0.5 m'), ('nan', 'finite'), ('duplicate', 'once')])
def test_rejects_unsafe_or_ambiguous_layout(change, message):
    house, rooms = fixture()
    r = next(r for r in rooms if r['id'] == 'fr_bed1')
    if change in ('overlap', 'outside', 'offgrid'):
        dx = {'overlap': .5, 'outside': -10, 'offgrid': .25}[change]
        r['rect'][0] += dx; r['rect'][2] += dx
    if change == 'resize': r['rect'][2] += .5
    if change == 'nan': r['rect'][0] = float('nan')
    if change == 'duplicate': rooms[-1] = copy.deepcopy(rooms[-2])
    with pytest.raises(ValueError, match=message): layout.build(house, rooms)
