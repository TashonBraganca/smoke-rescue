# Headless ORB-SLAM3 RGB-D runner

Build the ARM64 image from the repository root. The image pins ORB-SLAM3 and Pangolin source commits, extracts `ORBvoc.txt`, and compiles only the ORB-SLAM3 library and the Smoke Rescue RGB-D runner.

```sh
docker build --platform linux/arm64 -f slam/Dockerfile -t smoke-rescue-orbslam3 .
```

Create a writable output folder, then mount one robot recording read-only. The robot folder must contain `camera.yaml`, `associations.txt`, `rgb/`, and `depth/`.

```sh
mkdir -p data/runs/tum_fr3_long_office_2halves/robot_A
docker run --rm --platform linux/arm64 \
  --mount type=bind,src="$(pwd)/data/recordings/tum_fr3_long_office_2halves/robot_A",dst=/recording,readonly \
  --mount type=bind,src="$(pwd)/data/runs/tum_fr3_long_office_2halves/robot_A",dst=/output \
  smoke-rescue-orbslam3
```

The runner passes `/recording/camera.yaml` to ORB-SLAM3, reads `/recording/associations.txt`, starts `ORB_SLAM3::System` with `bUseViewer=false`, and writes `CameraTrajectory.txt` and `KeyFrameTrajectory.txt` under `/output`.
