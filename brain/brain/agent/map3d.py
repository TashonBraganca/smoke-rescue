"""Per-robot depth fusion in x right, y forward, z up coordinates.

All OpenCV volume operations run in a spawned process. A native fault loses that
robot's map and fails its request; the next integration starts an empty map.
Coverage records valid depth rays, including free columns before their returns.
It is rasterised at coverage_cell_m resolution and has that spatial uncertainty.
"""
from __future__ import annotations

import json
import math
import multiprocessing
from pathlib import Path
import threading

import cv2
import numpy as np

from brain.agent.vision import _index, _safe_file
from brain.io import read_camera

CONFIG_DIR = Path(__file__).resolve().parents[3] / 'config'


def camera_pose(pose, mount):
    """Camera optical coordinates to the robot's initial body frame."""
    x, y, heading = (float(pose[k]) for k in ('x', 'y', 'heading_deg'))
    if not np.isfinite([x, y, heading]).all():
        raise ValueError('pose must be finite')
    a, pitch = math.radians(heading), math.radians(mount['pitch_down_deg'])
    right = np.array([math.cos(a), -math.sin(a), 0.])
    forward = np.array([math.sin(a), math.cos(a), 0.])
    up = np.array([0., 0., 1.])
    result = np.eye(4, dtype=np.float32)
    result[:3, :3] = np.column_stack((right, -math.sin(pitch) * forward - math.cos(pitch) * up,
                                     math.cos(pitch) * forward - math.sin(pitch) * up))
    result[:3, 3] = np.array([x, y, mount['height_m']]) + mount['forward_offset_m'] * forward
    return result


class _VolumeMap:
    def __init__(self, settings, sensors):
        self.settings, self.sensors = settings, sensors
        self.volume = None
        self.folder = None
        self.last_t = -math.inf
        self.coverage = np.zeros((0, 0), np.uint8)
        self.coverage_origin = np.zeros(2, np.int64)

    def integrate(self, robot_dir, t, pose):
        t = float(t)
        if not math.isfinite(t):
            raise ValueError('t must be finite')
        transform = camera_pose(pose, self.sensors['camera']['mount'])
        folder = Path(robot_dir).resolve()
        cam = read_camera(_safe_file(folder, 'camera.yaml'))
        if cam.depth_factor <= 0 or min(cam.fx, cam.fy, cam.width, cam.height) <= 0:
            raise ValueError('invalid camera calibration')
        stamp, path = min(_index(folder, 'depth'), key=lambda row: (abs(row[0] - t), row[0]))
        if abs(stamp - t) > self.settings['frame_tolerance_s']:
            raise ValueError('no depth frame close to requested t')
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16 or raw.shape != (cam.height, cam.width):
            raise ValueError('depth must be a uint16 image matching camera.yaml')
        stride = self.settings['depth_stride']
        depth = np.ascontiguousarray(raw[::stride, ::stride], dtype=np.float32)
        depth_m = depth / cam.depth_factor
        valid = (depth_m >= self.sensors['depth']['min_m']) & (depth_m <= self.sensors['depth']['max_m'])
        depth[~valid] = 0
        k = np.array([[cam.fx / stride, 0., cam.cx / stride],
                      [0., cam.fy / stride, cam.cy / stride], [0., 0., 1.]], np.float32)
        calibration = (depth.shape, k.tobytes(), cam.depth_factor)
        if (self.volume is None or folder != self.folder or t < self.last_t
                or calibration != self.calibration):
            settings = cv2.VolumeSettings(cv2.VolumeType_HashTSDF)
            settings.setVoxelSize(self.settings['voxel_m'])
            settings.setTsdfTruncateDistance(self.settings['truncate_m'])
            settings.setMaxDepth(self.sensors['depth']['max_m'])
            settings.setDepthFactor(cam.depth_factor)
            settings.setMaxWeight(self.settings['max_weight'])
            settings.setCameraIntegrateIntrinsics(k)
            settings.setIntegrateWidth(depth.shape[1])
            settings.setIntegrateHeight(depth.shape[0])
            settings.setVolumePose(np.eye(4, dtype=np.float32))
            self.volume = cv2.Volume(cv2.VolumeType_HashTSDF, settings)
            self.volume.setEnableGrowth(True)
            self.coverage = np.zeros((0, 0), np.uint8)
            self.coverage_origin = np.zeros(2, np.int64)
        self.volume.integrate(depth, transform)
        self._observe(depth_m, valid, k, transform)
        self.folder, self.last_t, self.calibration = folder, t, calibration

    def _observe(self, depth_m, valid, k, transform):
        v, u = np.nonzero(valid)
        if not len(u):
            return
        z = depth_m[v, u]
        optical = np.column_stack(((u - k[0, 2]) * z / k[0, 0],
                                   (v - k[1, 2]) * z / k[1, 1], z))
        xy = (optical @ transform[:2, :3].T) + transform[:2, 3]
        cell = self.settings['coverage_cell_m']
        ends = np.unique(np.floor(xy / cell).astype(np.int64), axis=0)
        start = np.floor(transform[:2, 3] / cell).astype(np.int64)
        low = np.minimum(ends.min(axis=0), start)
        high = np.maximum(ends.max(axis=0), start) + 1
        if self.coverage.size:
            low = np.minimum(low, self.coverage_origin)
            high = np.maximum(high, self.coverage_origin + self.coverage.shape[::-1])
        shape = tuple((high - low)[::-1])
        if self.coverage.shape != shape or not np.array_equal(low, self.coverage_origin):
            expanded = np.zeros(shape, np.uint8)
            dx, dy = self.coverage_origin - low
            if self.coverage.size:
                h, w = self.coverage.shape
                expanded[dy:dy + h, dx:dx + w] = self.coverage
            self.coverage, self.coverage_origin = expanded, low
        a = tuple(map(int, start - low))
        for end in ends:
            cv2.line(self.coverage, a, tuple(map(int, end - low)), 1)

    def points_normals(self):
        if self.volume is None:
            return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
        points, normals = self.volume.fetchPointsNormals()
        if points is None:
            return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
        points = points.reshape(-1, points.shape[-1])[:, :3]
        normals = normals.reshape(-1, normals.shape[-1])[:, :3]
        valid = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1)
        return points[valid], normals[valid]

    def layers(self, cell_m, origin, size):
        """Return HxW arrays. Size is a square side or [width, height].

        Row zero starts at origin[1], the south edge. Seen means at least one
        observed coverage cell overlaps the query cell; it does not imply that
        the whole vertical column, or a traversable floor, has been observed.
        """
        cell_m = float(cell_m)
        origin = np.asarray(origin, dtype=float)
        dims = np.asarray([size, size] if np.isscalar(size) else size, dtype=float)
        if (not math.isfinite(cell_m) or cell_m <= 0 or origin.shape != (2,)
                or not np.isfinite(origin).all() or dims.shape != (2,)
                or not np.isfinite(dims).all() or np.any(dims <= 0) or np.any(dims != np.floor(dims))):
            raise ValueError('expected positive cell_m, finite origin and positive integer grid size')
        w, h = map(int, dims)
        top = np.zeros((h, w), dtype='<u2')
        seen = np.zeros((h, w), dtype=bool)
        points, _ = self.points_normals()
        points = points[points[:, 2] > self.settings['floor_exclude_m']]
        if len(points):
            xy = np.floor((points[:, :2] - origin) / cell_m).astype(np.int64)
            inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
            xy, heights = xy[inside], points[inside, 2]
            cm = np.clip(np.rint(heights * 100), 0, np.iinfo(np.uint16).max).astype('<u2')
            np.maximum.at(top, (xy[:, 1], xy[:, 0]), cm)
        if self.coverage.size:
            # Resample cell areas, including nonaligned grids and cells smaller than coverage resolution.
            c = self.settings['coverage_cell_m']
            low = self.coverage_origin * c
            for row in range(h):
                y0 = max(0, math.floor((origin[1] + row * cell_m - low[1]) / c))
                y1 = min(self.coverage.shape[0], math.ceil((origin[1] + (row + 1) * cell_m - low[1]) / c))
                if y1 <= y0:
                    continue
                strip = self.coverage[y0:y1].any(axis=0)
                prefix = np.r_[0, np.cumsum(strip)]
                x0 = np.clip(np.floor((origin[0] + np.arange(w) * cell_m - low[0]) / c).astype(int), 0, len(strip))
                x1 = np.clip(np.ceil((origin[0] + (np.arange(w) + 1) * cell_m - low[0]) / c).astype(int), 0, len(strip))
                seen[row] = prefix[x1] > prefix[x0]
        return top, seen

    def export(self, path):
        points, normals = self.points_normals()
        header = ('ply\nformat binary_little_endian 1.0\n'
                  'comment robot frame x right y forward z up, metres\n'
                  f'element vertex {len(points)}\n' + ''.join(
                      f'property float {name}\n' for name in ('x', 'y', 'z', 'nx', 'ny', 'nz')) + 'end_header\n')
        with Path(path).open('wb') as f:
            f.write(header.encode('ascii'))
            np.asarray(np.c_[points, normals], dtype='<f4').tofile(f)


def _worker(connection, settings, sensors):
    cv2.setNumThreads(settings['native_threads'])
    cv2.ocl.setUseOpenCL(False)
    volume = _VolumeMap(settings, sensors)
    try:
        while True:
            command, args = connection.recv()
            if command == 'close':
                break
            try:
                result = getattr(volume, command)(*args)
                connection.send(('ok', result))
            except (ValueError, KeyError, TypeError, OSError, cv2.error) as exc:
                connection.send(('error', str(exc)))
    except EOFError:
        pass
    finally:
        connection.close()


class RobotMap3D:
    """A serialized per-robot connection to a crash-isolated HashTSDF volume."""
    def __init__(self):
        self.settings = json.loads((CONFIG_DIR / 'map3d.json').read_text())
        self.sensors = json.loads((CONFIG_DIR / 'sensors.json').read_text())
        self._lock = threading.Lock()
        self._process = None
        self._connection = None

    def _start(self):
        context = multiprocessing.get_context('spawn')
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_worker, args=(child, self.settings, self.sensors), daemon=True)
        self._process.start()
        child.close()

    def _discard(self):
        if self._connection is not None:
            self._connection.close()
        if self._process is not None:
            if self._process.is_alive():
                self._process.kill()
            self._process.join()
        self._process, self._connection = None, None

    def _call(self, command, *args):
        with self._lock:
            if self._process is None:
                self._start()
            elif command == 'integrate' and not self._process.is_alive():
                self._discard()
                self._start()
            try:
                self._connection.send((command, args))
                if not self._connection.poll(self.settings['worker_timeout_s']):
                    raise RuntimeError('map3d worker timed out; map lost')
                status, result = self._connection.recv()
            except (BrokenPipeError, EOFError, OSError, RuntimeError) as exc:
                self._discard()
                raise RuntimeError(f'map3d worker failed; map lost: {exc}') from exc
            if status != 'ok':
                raise ValueError(result)
            return result

    def integrate(self, robot_dir, t, pose):
        return self._call('integrate', str(robot_dir), t, pose)

    def layers(self, cell_m, origin, size):
        return self._call('layers', cell_m, origin, size)

    def points_normals(self):
        return self._call('points_normals')

    def export(self, path):
        return self._call('export', str(path))

    def close(self):
        with self._lock:
            self._discard()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
