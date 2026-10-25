OpenCV 5 clips for the video edit (25 Oct 2026). Made by tools/opencv_clips.py and tools/visual_slam_clip.py from real recordings; every overlay is computed by the robot-side code.
- yolo_detections.mp4/.png: cv2.dnn YOLO11m person boxes, robot A, kitchen, 24-38 s (run unity_house_onedoor_seed934_full_1520_5of5)
- thermal_threshold.mp4/.png: OpenCV threshold + contours on the thermal camera, robot B facing the fire, 28-42 s (green 28-40 C body heat, red >= 200 C fire)
- rgbd.mp4/.png: colour and depth side by side, robot A, 24-38 s
- map3d_tsdf.mp4/.png: OpenCV 5 HashTSDF 3D map built live by robot A (1.04 M surface points), rotating view
- orb_matches_visual_slam.mp4, orb_matches.png: ORB matches between the two robots and the shared map snapping into place (best run 03:18)
All clips H.264, 15 fps.
