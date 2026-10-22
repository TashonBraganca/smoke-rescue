

import pytest as _pytest


@_pytest.fixture(autouse=True)
def _fresh_team_conversation():
    """Each test starts with no team conversation. Test helpers give every test the same recording folder parent, so the
    agent's new-recording check cannot tell tests apart, and one test's proposals or intent would reach the next."""
    from brain.agent.team import TeamState
    TeamState.reset()
    yield
