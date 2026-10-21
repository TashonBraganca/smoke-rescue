"""Point cloud interchange, voxel means, and recorded RGB-D registration."""
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from brain import merge
from brain.io import RecordingReader, Frame

ROOT = Path(__file__).resolve().parents[2]


def test_ply_round_trip_preserves_xyz_rgb_and_normals(tmp_path):
    from brain.pointcloud import PointCloud, read_cloud, write_cloud
    points = np.array([[.1, .2, 1.], [-1., 2., 3.]])
    colors = np.array([[1., 0., 128 / 255], [0., 1., 0.]])
    normals = np.array([[0., 0., 1.], [1., 0., 0.]])
    path = tmp_path / 'cloud.ply'
    write_cloud(path, PointCloud(points, colors, normals))
    cloud = read_cloud(path)
    np.testing.assert_allclose(cloud.points, points, atol=1e-6)
    np.testing.assert_allclose(cloud.colors, colors, atol=1 / 255)
    np.testing.assert_allclose(cloud.normals, normals, atol=1e-6)
    header = path.read_bytes().split(b'end_header')[0]
    for name in ('x', 'y', 'z', 'red', 'green', 'blue', 'nx', 'ny', 'nz'):
        assert any(line.endswith(b' ' + name.encode()) for line in header.splitlines())


def test_voxel_averaging_uses_half_voxel_below_minimum():
    from brain.pointcloud import PointCloud, voxel_downsample
    points = np.array([[0., 0., 1.], [.01, 0., 1.], [.031, 0., 1.], [.035, 0., 1.]])
    colors = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]])
    result = voxel_downsample(PointCloud(points, colors), .03)
    np.testing.assert_allclose(result.points, [[.005, 0., 1.], [.033, 0., 1.]])
    np.testing.assert_allclose(result.colors, [[.5, .5, 0.], [.5, .5, 1.]])


def test_icp_recovers_rigid_motion_from_recorded_depth():
    from brain.pointcloud import register_rgbd
    reader = RecordingReader(ROOT / 'data/fixtures/tiny_2robots/robot_A')
    source = reader.read_frame(0)
    cam = reader.camera
    k = np.array([[cam.fx, 0., cam.cx], [0., cam.fy, cam.cy], [0., 0., 1.]])
    known = np.eye(4)
    known[:3, :3] = Rotation.from_euler('xyz', [.4, -.6, .3], degrees=True).as_matrix()
    known[:3, 3] = [.012, -.008, .009]
    xyz = cv2.depthTo3d(source.depth_m.astype(np.float64), k)[..., :3]
    valid = (source.depth_m > 0) & (source.depth_m < 6)
    p = xyz[valid] @ known[:3, :3].T + known[:3, 3]
    uv = np.rint(p[:, :2] / p[:, 2:] * [cam.fx, cam.fy] + [cam.cx, cam.cy]).astype(int)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
    p, uv, colors = p[inside], uv[inside], source.rgb[valid][inside]
    # Write the nearest return last when several rays hit the same pixel.
    order = np.argsort(-p[:, 2])
    depth = np.zeros_like(source.depth_m)
    rgb = np.zeros_like(source.rgb)
    depth[uv[order, 1], uv[order, 0]] = p[order, 2]
    rgb[uv[order, 1], uv[order, 0]] = colors[order]
    target = Frame(0., 0., rgb, np.rint(depth * cam.depth_factor).astype(np.uint16), depth)
    result = register_rgbd(source, target, cam, cam, np.eye(4))
    metres, degrees = merge.transform_delta(result.transformation, known)
    assert metres < .01
    assert degrees < .5
    assert result.fitness > .9
    assert result.inlier_rmse < .025
