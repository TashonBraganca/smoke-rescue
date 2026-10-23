"""Offline attribution must follow order cancellation and separate travel from looks."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'tools/eval/room_ledger_audit.py'
spec = importlib.util.spec_from_file_location('room_ledger_audit', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def event(t, kind, reason='', **numbers):
    return dict(t=t, kind=kind, reason=reason, numbers=numbers)


def test_new_move_cancels_face_and_arrival_ends_candidate_time():
    activity = audit.Activity(face_budget_s=12, probe_arrival_m=.05)
    activity.apply(event(1, 'goal', "At the agent's goal; turn to face the point it named."))
    assert activity.cause(2, 'explore', [0, 0], True) == 'agent_face'
    activity.apply(event(3, 'agent_order', 'go_to: closer look at an uncertain warm region', hold_s=20))
    assert activity.cause(4, 'explore', [0, 0], True) == 'turn_then_drive'
    assert activity.order == 'candidate' and activity.order_until == 23
    activity.apply(event(5, 'agent_goal_reached'))
    assert activity.order_until < 5


def test_evidence_approach_is_a_heading_change_until_the_standpoint():
    activity = audit.Activity(face_budget_s=12, probe_arrival_m=.05)
    activity.apply(event(5, 'local_evidence_started', x_m=1, y_m=0))
    assert activity.cause(6, 'explore', [0, 0], True) == 'turn_then_drive'
    assert activity.cause(6, 'explore', [1, 0], True) == 'scan_sweep_evidence'
    activity.apply(event(7, 'doorway_probe_done'))
    assert activity.cause(8, 'explore', [1, 0], True) == 'turn_then_drive'


def test_new_survivor_approach_clears_previous_viewing_state():
    activity = audit.Activity(face_budget_s=12, probe_arrival_m=.05)
    activity.apply(event(8, 'goal', 'At survivor viewing distance; stop and face the person for confirmation.'))
    assert activity.cause(9, 'survivor', [0, 0], False) == 'survivor_facing'
    activity.apply(event(10, 'goal', 'Approach a repeated warm person-sized blob, then stop and face it.'))
    assert activity.cause(11, 'survivor', [0, 0], True) == 'turn_then_drive'
    assert activity.cause(11, 'retreat', [0, 0], True) == 'other_safety_or_recovery'


def test_incomplete_recording_is_rejected_before_other_files_are_read(tmp_path):
    (tmp_path / 'meta.json').write_text('{"complete": false}')
    with pytest.raises(ValueError, match='incomplete'):
        audit.audit(tmp_path)
