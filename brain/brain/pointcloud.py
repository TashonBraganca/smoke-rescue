"""Point clouds on OpenCV 5, in place of Open3D.

PLY files are written and read by cv2.savePointCloud and cv2.loadPointCloud, depth images become points with
cv2.depthTo3d, and surface normals come from the depth image with cv2.RgbdNormals. The one step OpenCV 5's Python
bindings cannot do is ICP from a given estimate: cv2.Odometry ignores its Rt argument, and on a recorded depth frame
with a known 2 cm, 0.5 deg move it returned about no motion under every setting tried (2026-10-21). So the
point-to-plane ICP solver below is numpy, with a SciPy k-d tree for nearest points and cv2.Rodrigues for the update.
Fitness and inlier RMSE are defined as in Open3D (inliers over source points; RMS distance of the inliers), so the
acceptance thresholds in brain.merge keep their meaning.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree


@dataclass
class PointCloud:
    points: np.ndarray
    colors: np.ndarray | None = None    # 0 to 1, one row per point
    normals: np.ndarray | None = None


@dataclass
class Registration:
    transformation: np.ndarray          # 4 x 4, source to target
    fitness: float
    inlier_rmse: float


def _column(a: np.ndarray | None) -> np.ndarray:
    if a is None or len(a) == 0:
        return np.empty((0, 1, 3), np.float32)
    return np.ascontiguousarray(np.asarray(a, np.float32).reshape(-1, 1, 3))


def write_cloud(path: Path | str, cloud: PointCloud) -> None:
    """PLY with x y z, and red green blue and nx ny nz when present (cv2.savePointCloud)."""
    points = np.asarray(cloud.points)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("a point cloud needs at least one x y z point")
    for name, extra in (("colors", cloud.colors), ("normals", cloud.normals)):
        if extra is not None and len(extra) != len(points):
            raise ValueError(f"{name} must have one row per point")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.savePointCloud(str(path), _column(points), _column(cloud.normals), _column(cloud.colors))
    if not path.exists() or path.stat().st_size == 0:
        raise OSError(f"failed to write point cloud: {path}")


def read_cloud(path: Path | str) -> PointCloud:
    """Points, colours and normals from a PLY or OBJ file (cv2.loadPointCloud)."""
    vertices, normals, rgb = cv2.loadPointCloud(str(path))
    vertices = np.asarray(vertices, np.float64)
    if vertices.size % 3 or vertices.size < 3:
        raise ValueError(f"{path}: OpenCV could not read the points (it reads float x y z; files written with double "
                         "coordinates, as old Open3D maps were, must be rebuilt by brain.dense_map)")
    points = vertices.reshape(-1, 3)
    def rows(a):
        a = None if a is None else np.asarray(a, np.float64).reshape(-1, 3)
        return a if a is not None and len(a) == len(points) else None
    return PointCloud(points, rows(rgb), rows(normals))


def voxel_downsample(cloud: PointCloud, voxel_m: float) -> PointCloud:
    """Mean point, colour and (renormalised) normal per voxel; voxels start half a voxel below the minimum point, as in
    Open3D's voxel_down_sample, so results match it."""
    if voxel_m <= 0:
        raise ValueError("voxel size must be positive")
    points = np.asarray(cloud.points, np.float64)
    if not len(points):
        return PointCloud(points.reshape(0, 3))
    origin = points.min(axis=0) - voxel_m / 2
    keys = np.floor((points - origin) / voxel_m).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()

    def mean(values):
        if values is None:
            return None
        values = np.asarray(values, np.float64)
        return np.column_stack([np.bincount(inverse, weights=values[:, j], minlength=len(counts)) for j in range(3)]) / counts[:, None]

    normals = mean(cloud.normals)
    if normals is not None:
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return PointCloud(mean(points), mean(cloud.colors), normals)


def _K(cam) -> np.ndarray:
    return np.array([[cam.fx, 0., cam.cx], [0., cam.fy, cam.cy], [0., 0., 1.]], np.float32)


def rgbd_cloud(frame, cam, voxel_m: float, max_depth_m: float = 6.0, normal_window: int = 5) -> PointCloud:
    """Points (camera frame: x right, y down, z forward), colours and normals of one RGB-D frame, voxel-averaged.
    Normals are computed on the full depth image by cv2.RgbdNormals before downsampling."""
    depth = np.asarray(frame.depth_m, np.float32)
    xyzw = cv2.depthTo3d(depth, _K(cam)).astype(np.float32)          # 4 channels, as RgbdNormals.apply needs
    normals = cv2.RgbdNormals_create(depth.shape[0], depth.shape[1], cv2.CV_32F, _K(cam), normal_window, 50,
                                     cv2.RgbdNormals_RGBD_NORMALS_METHOD_FALS).apply(xyzw)[..., :3]
    xyz = xyzw[..., :3]
    valid = (depth > 0) & (depth <= max_depth_m) & np.isfinite(normals).all(axis=-1) & (np.abs(normals).sum(axis=-1) > 0)
    colors = np.asarray(frame.rgb, np.float64)[valid] / 255.0
    return voxel_downsample(PointCloud(xyz[valid].astype(np.float64), colors, normals[valid].astype(np.float64)), voxel_m)


def icp_point_to_plane(source: PointCloud, target: PointCloud, init: np.ndarray, max_distance_m: float,
                       max_iterations: int = 40, relative_change: float = 1e-6) -> Registration:
    """Point-to-plane ICP from init (target needs normals). Stops when fitness and inlier RMSE both change by less than
    relative_change, as Open3D's convergence criteria do."""
    if target.normals is None:
        raise ValueError("point-to-plane ICP needs target normals")
    src = np.asarray(source.points, np.float64)
    dst, nrm = np.asarray(target.points, np.float64), np.asarray(target.normals, np.float64)
    tree = cKDTree(dst)
    T = np.array(init, np.float64)

    def match(T):
        moved = src @ T[:3, :3].T + T[:3, 3]
        dist, idx = tree.query(moved, k=1, distance_upper_bound=max_distance_m)
        inlier = np.isfinite(dist)
        fitness = float(inlier.sum() / max(len(src), 1))
        rmse = float(np.sqrt(np.mean(dist[inlier] ** 2))) if inlier.any() else 0.0
        return moved, idx, inlier, fitness, rmse

    moved, idx, inlier, fitness, rmse = match(T)
    for _ in range(max_iterations):
        if inlier.sum() < 6:
            break
        p, q, n = moved[inlier], dst[idx[inlier]], nrm[idx[inlier]]
        A = np.hstack([np.cross(p, n), n])
        b = -np.einsum("ij,ij->i", p - q, n)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        step = np.eye(4)
        step[:3, :3] = cv2.Rodrigues(x[:3].reshape(3, 1))[0]
        step[:3, 3] = x[3:]
        T = step @ T
        moved, idx, inlier, new_fitness, new_rmse = match(T)
        done = (abs(new_fitness - fitness) < relative_change * max(fitness, 1e-12) and
                abs(new_rmse - rmse) < relative_change * max(rmse, 1e-12))
        fitness, rmse = new_fitness, new_rmse
        if done:
            break
    return Registration(T, fitness, rmse)


def register_rgbd(source_frame, target_frame, source_cam, target_cam, init: np.ndarray,
                  voxel_m: float = 0.03, max_distance_m: float | None = None) -> Registration:
    """Refine init (source camera to target camera) by point-to-plane ICP on the two frames' voxel clouds."""
    max_distance_m = voxel_m * 1.5 if max_distance_m is None else max_distance_m
    source = rgbd_cloud(source_frame, source_cam, voxel_m)
    target = rgbd_cloud(target_frame, target_cam, voxel_m)
    if len(source.points) < 6 or len(target.points) < 6:
        return Registration(np.array(init, np.float64), 0.0, float("nan"))
    return icp_point_to_plane(source, target, init, max_distance_m)
