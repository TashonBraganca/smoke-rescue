import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from brain.dense_map import build_dense_map
from brain.pointcloud import read_cloud


def test_map_uses_yaml_depth_intrinsics_and_estimated_camera_to_world(tmp_path):
    robot = tmp_path / "robot_A"
    (robot / "rgb").mkdir(parents=True)
    (robot / "depth").mkdir()
    (robot / "camera.yaml").write_text(
        "Camera.fx: 2\nCamera.fy: 4\nCamera.cx: 0\nCamera.cy: 0\n"
        "Camera.width: 2\nCamera.height: 2\nCamera.fps: 1\nDepthMapFactor: 500\n"
    )
    (robot / "associations.txt").write_text("0 rgb/0.png 0 depth/0.png\n")
    cv2.imwrite(str(robot / "rgb/0.png"), np.full((2, 2, 3), [0, 0, 255], np.uint8))
    cv2.imwrite(str(robot / "depth/0.png"), np.array([[1000, 1000], [0, 1000]], np.uint16))
    rotation = Rotation.from_euler("y", 90, degrees=True)
    trajectory = tmp_path / "KeyFrameTrajectory.txt"
    np.savetxt(trajectory, [[0, 1, 2, 3, *rotation.as_quat()]])
    output = tmp_path / "map.ply"
    build_dense_map(robot, trajectory, output)
    cloud = read_cloud(output)
    expected = rotation.apply([[0, 0, 2], [1, 0, 2], [1, .5, 2]]) + [1, 2, 3]
    actual = cloud.points
    assert len(actual) == 3
    for point in expected:
        assert np.min(np.linalg.norm(actual - point, axis=1)) < 1e-6
    np.testing.assert_allclose(cloud.colors, [[1, 0, 0]] * 3, atol=1 / 255)
