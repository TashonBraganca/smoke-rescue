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

## OpenCV 5 image

The separate ARM64 image pins [OpenCV 5.0.0 to commit 40738fb16ceddb5fb3fea747585f7ce6abb0605b](https://github.com/opencv/opencv/tree/40738fb16ceddb5fb3fea747585f7ce6abb0605b). ORB-SLAM3 and Pangolin retain the baseline commits. The multistage build omits source and object files from the runtime image. It uses Release C++17 and builds `core imgproc imgcodecs features flann calib geometry stereo highgui objdetect`. The [calib module requires objdetect](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/modules/calib/CMakeLists.txt). CUDA, OpenCL, Python, tests, examples and OpenCV GUI backends are disabled. The runner remains headless.

```sh
BUILDX_CONFIG="$PWD/data/runs/cv5_compare/buildx" \
  docker build --platform linux/arm64 -f slam/Dockerfile.cv5 -t smoke-rescue-orbslam3-cv5 .
docker run --rm --network none smoke-rescue-orbslam3-cv5 --version
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=brain brain/.venv/bin/python -m brain.slam_runner \
  data/recordings/unity_clear_seed7/robot_A \
  data/runs/cv5_compare/smoke-rescue-orbslam3-cv5/robot_A/manual_fresh_run \
  --image smoke-rescue-orbslam3-cv5
```

Choose a fresh output directory every time. `--image` defaults to `smoke-rescue-orbslam3`. The original Dockerfile and baseline image remain unchanged. Buildx metadata stays in the comparison folder, and the Dockerfile-specific ignore file excludes recordings and the local source inspection cache.

[orbslam3_opencv5.patch](orbslam3_opencv5.patch) contains all ORB-SLAM3 changes. It updates package requirements, C++17, ARM64 compiler flags, feature headers, geometry/stereo includes and old constants. Three boolean increments become assignments to `true`, preserving their pre-C++17 behavior. The existing shutdown patch applies separately. No SLAM algorithm or parameter changes.

Source inspection found that OpenCV 5 retains `features2d` and `calib3d` compatibility headers, but `opencv.hpp` omits geometry. `undistortPoints`, including fisheye, needs `geometry.hpp`; `stereoRectify` needs `stereo.hpp`; `initUndistortRectifyMap` belongs to `imgproc`. `KeyPoint` and `copyMakeBorder` remain in core. The removed core/imgproc C API needs no active-call replacement in this pinned library. OpenCV's [SONAME convention](https://github.com/opencv/opencv/blob/40738fb16ceddb5fb3fea747585f7ce6abb0605b/cmake/OpenCVVersion.cmake) gives `.so.500` symlinks resolving to `.so.5.0.0`.

Run the comparison after the build finishes. It alternates images sequentially and preserves raw exports. ATE calls the unchanged `brain.evaluator.compute_ate` against the oracle, with SE(3) alignment, no scale correction and 0.02 s timestamp tolerance. Tracked frames count state 2, separately from exported poses. Maps include the initial map. Wall time includes startup, vocabulary loading, paced playback, shutdown and exports.

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=brain brain/.venv/bin/python slam/compare_cv5.py \
  --session fresh_session_name --repeats 2 \
  --recordings unity_clear_seed7 tum_fr3_long_office_2halves
```
