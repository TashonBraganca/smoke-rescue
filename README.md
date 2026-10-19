# Smoke Rescue

A multi-robot visual SLAM simulation. Quadruped rescue robots enter a smoke-filled house from different doors, each maps it with its own RGB-D camera, and their maps merge when they recognize the same place.

## Start here

- **Building it with a coding agent:** paste `MASTER_PROMPT.md` into a coding agent. `KICKOFF_PROMPT.md` has a two-session alternative.
- **What it is:** `docs/SIMULATION_SPEC.md`.
- **Data formats and coordinate frames:** `docs/INTERFACES.md`.
- **Order of work and acceptance tests:** `docs/BUILD_ORDER.md`.
- **Machine setup:** `docs/SETUP_MAC.md`.
- **Progress:** `STATUS.md`.
- **Ideas for later, not to build yet:** `docs/GAME_FEATURES_LATER.md`.

## For reviewers and testers

You don't need Unity. Pull a recording into `data/recordings/`, then:

```bash
cd brain
uv sync
uv run pytest
uv run python -m brain.pipeline ../data/recordings/<run_id>
uv run python -m brain.viewer ../data/runs/<run_id>
```

Check `data/runs/<run_id>/report.md` against the targets in `docs/SIMULATION_SPEC.md` section 7.
