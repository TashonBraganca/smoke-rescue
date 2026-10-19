# Smoke Rescue brain

Run commands from `brain/` with Python 3.11 and `uv`. The ORB-SLAM3 image is `smoke-rescue-orbslam3`. The runner uses `docker` on PATH or `~/.orbstack/bin/docker`. Each invocation mounts one robot folder read-only and mounts its output folder read-write. Existing SLAM outputs cause an error so a failed process cannot reuse an old trajectory.

```sh
uv run python -m brain.slam_runner ../data/recordings/tum_fr3_long_office_2halves/robot_A ../data/runs/tum_fr3_long_office_2halves/robot_A
uv run python -m brain.slam_runner ../data/recordings/tum_fr3_long_office_2halves/robot_B ../data/runs/tum_fr3_long_office_2halves/robot_B
uv run python -m brain.dense_map ../data/recordings/tum_fr3_long_office_2halves/robot_A ../data/runs/tum_fr3_long_office_2halves/robot_A
uv run python -m brain.dense_map ../data/recordings/tum_fr3_long_office_2halves/robot_B ../data/runs/tum_fr3_long_office_2halves/robot_B
uv run python -m brain.evaluator ../data/recordings/tum_fr3_long_office_2halves ../data/runs/tum_fr3_long_office_2halves
uv run pytest -q
```

Set `UV_CACHE_DIR=/private/tmp/smoke-rescue-uv-cache` when the default uv cache is outside the writable sandbox. To rebuild the image from the repo root, run `BUILDX_CONFIG=/private/tmp/smoke-rescue-buildx ~/.orbstack/bin/docker build -f slam/Dockerfile -t smoke-rescue-orbslam3 .`. The Dockerfile applies a patch to the pinned upstream revision so shutdown waits for mapping and global bundle adjustment before pose export. The C++ runner converts decoded BGR images to RGB when camera.yaml requests RGB input.

The Python runner returns the two TUM trajectory paths. `slam_meta.json` records the image ID, command, input frame count, exported pose count, missing pose count, surviving keyframe count, and container wall time. Missing poses include initialization. Each missing frame contributes the interval until the next recorded RGB timestamp. A missing final frame contributes `1 / camera.fps`. This measures absent exported poses, not ORB-SLAM3's internal tracking-state duration.

ORB-SLAM3 exports camera poses relative to the first keyframe and keyframe poses relative to its atlas. The runner uses their common timestamps to compute a rigid conversion, applies it to keyframes, and rejects inconsistent exports. `KeyFrameTrajectory.raw.txt` preserves the original export. `keyframes.json` maps each surviving keyframe to its recording frame index, RGB path, depth path and timestamp. Its keyframe IDs are sequential exported IDs, not ORB-SLAM3's internal IDs.

The map builder reads intrinsics and depth factor from camera.yaml and uses estimated camera-to-world keyframe poses. It averages all valid depth points and RGB values in fixed 3 cm cells. `map.ply` is a point cloud, not a TSDF mesh. `map.json` records its sources and point count. `map.rrd` contains the final map and keyframe trajectory for inspection; a growing-map replay belongs to C11.

The evaluator reads truth only through `brain.oracle`. It uses evo's timestamp association with `max_diff=0.02`, then aligns translation with SE(3) and scale correction disabled. The report keeps the four fields in the frozen per-robot format. `ate_details.json` stores matched pose counts, timestamp differences and alignment transforms. `report.md` contains the measured results in prose, and `evaluation.rrd` stores the metrics. Unimplemented C10 metrics are omitted. The pure ATE function accepts a truth-as-estimate fixture; run-level evaluation rejects exact zero and non-finite ATE. Both evaluation and SLAM stop below 50 percent trajectory coverage. Evaluation stops below 100 timestamp matches.
