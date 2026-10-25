"""run_audit names the cause of a stall by time, not by file order (seed 934, 25 Oct: four agent_order rows stamped t=0.0)."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / 'tools/run_audit.py'
spec = importlib.util.spec_from_file_location('run_audit', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_a_row_stamped_zero_late_in_the_file_is_not_the_cause_of_an_early_stall():
    evs = [(10.0, 'goal', 'Walk into the next room'), (28.0, 'sweep_look', 'Hold a sector'),
           (0.0, 'agent_order', 'spray_water succeeded (target 11deg)')]
    assert audit.cause_before(evs, 30.0) == 'sweep_look: Hold a sector'
    assert audit.cause_before(evs, 5.0).startswith('agent_order')   # by time it really is the first row
