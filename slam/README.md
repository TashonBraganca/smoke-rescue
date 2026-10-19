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

RGB-D-inertial mode uses the same executable and image. From `brain/`, run `uv run python -m brain.slam_runner ../data/recordings/unity_clear_seed7/robot_A ../data/runs/<fresh-run>/robot_A --mode rgbd_inertial`. The default remains `rgbd`. The full pipeline accepts `--slam-mode rgbd_inertial`. Choose a fresh output directory for each experiment; the runner refuses to mix trajectories from different runs. The pipeline cache includes the mode, IMU contents and selected calibration for inertial runs.

The Python runner copies the recording's camera settings into `<output>/inertial.yaml` and appends IMU calibration from `config/sensors.json`. It mounts only one robot folder read-only and its output folder writable. Recorded `meta.json` calibration takes priority when it contains `camera.mount` or `camera_mount`, with `height_m`, `forward_offset_m` and `pitch_down_deg`; optional `imu` uses the sensor config's field names. The named `unity_clear_seed7` recording uses its measured old mount, height 0.50 m and forward offset 0.41 m. `unity_clear` and `unity_haze` fall back to the current sensor config mount. Other recordings must supply mount metadata, so a TUM recording cannot silently receive a Unity calibration. The checked-in `calibration/unity_clear_seed7.yaml` and `calibration/unity_mast.yaml` are reference settings for the two known mounts. Runtime generation reads the sensor config again.

`IMU.T_b_c1` maps a camera point into the IMU body frame. In Unity, `p_body = Rx(10 degrees) p_camera + [0, height, forward]`. Applying `S = diag(1,-1,1)` to both frames gives `R_body_camera = S Rx(10 degrees) S = Rx(-10 degrees)` and `t_body_camera = [0,-height,forward]`. The rotation rows are `[1,0,0]`, `[0,0.984807753,0.173648178]` and `[0,-0.173648178,0.984807753]`. A point 2 m ahead maps to `[0,-0.152703645,2.379615506]` m for seed7, or `[0,-1.252703645,2.219615506]` m for the newer mast. These are camera-to-body transforms, without inversion. The YAML matrix must have `dt: f`; the pinned converter reads matrix entries with `cv::Mat::at<float>`. Noise densities and random walks remain continuous-time values, since upstream scales them by the square root of the IMU frequency.

The runner uses `System::IMU_RGBD` and passes each frame's IMU batch to `TrackRGBD`. Acceleration remains specific force including gravity. It interpolates samples at exact image timestamps, keeps interior 200 Hz samples, and sends each boundary once because ORB-SLAM3 retains the previous endpoint in its queue. It rejects non-increasing clocks, gaps over 1.5 nominal IMU periods and images beyond IMU coverage. Seed7 has an image at zero but its first IMU sample is at 0.005 s. Inertial mode skips that image and initializes visually at 0.066667 s. It does not invent an IMU sample at zero. The diagnostic `tracking.csv` records frame time, ORB tracking state, batch count and time bounds, and IMU initialization. Metadata distinguishes missing exported poses from tracker states 3 and 4, which can still produce predicted exports. Viewer use stays disabled.

The pinned source establishes the settings keys in [Settings.cc](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4/src/Settings.cc), the float matrix requirement in [Converter.cc](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4/src/Converter.cc), and the queue endpoint behavior in [Tracking.cc](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4/src/Tracking.cc). The [RGB-D-inertial example](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4/Examples/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc) supplies accelerometer values, gyro values and seconds to `IMU::Point` in that order. `SaveTrajectoryTUM` and `SaveKeyFrameTrajectoryTUM` in [System.cc](https://github.com/UZ-SLAMLab/ORB_SLAM3/blob/4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4/src/System.cc) export camera poses for both modes. The EuRoC exporter has different inertial behavior and is not used here.
