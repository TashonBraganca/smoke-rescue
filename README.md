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

## Run the demo

```bash
tools/demo.sh                      # haze run: two quadrupeds, per-robot ORB-SLAM3, DINOv2 place recognition, merge snap
tools/demo.sh unity_clear          # same routes in clear air
tools/demo.sh tum_fr3_long_office_2halves   # the public TUM stand-in the brain was developed on
```

Each run folder in `data/runs/<run_id>/` holds `report.md` and `report.json` (ATE per robot, merge time and error,
false merges, map accuracy, coverage), `replay.rrd`, the merge candidate log `merge/candidates.jsonl`, and each robot's
SLAM output. Recordings in `data/recordings/<run_id>/` follow `docs/INTERFACES.md`; ground truth sits only in `oracle/`.

To make a new recording, open `unity/SmokeRescue` in Unity 6000.6, open `Assets/SmokeRescue/Scenes/House.unity`, and use
Tools > Smoke Rescue > Record. For the live showcase view use Tools > Smoke Rescue > Setup Showcase Camera, then Play.

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
