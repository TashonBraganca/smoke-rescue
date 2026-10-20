"""Recording readers shared by offline navigation analysis and live watching."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

CELL_M = .1
ORIGIN = np.array([-2., -2.])


def read_json_lines(path: Path) -> list[dict]:
    """Ignore an unfinished final line, but report malformed completed records."""
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith(b'\n') and line.strip()]


def poses_from_states(states: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    return (np.array([s['t'] for s in states]),
            np.array([[s['x'], s['y'], s['yaw_rad']] for s in states], dtype=float))


def interpolate_poses(times: np.ndarray, poses: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Interpolate unwrapped yaw; never silently extrapolate a trajectory."""
    if not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError('pose timestamps must increase')
    if np.any(query < times[0] - 1e-6) or np.any(query > times[-1] + 1e-6):
        raise ValueError('pose interpolation requested outside recorded interval')
    values = poses.copy()
    values[:, 2] = np.unwrap(values[:, 2])
    return np.column_stack([np.interp(query, times, values[:, i]) for i in range(3)])


def latest_map(robot_dir: Path) -> tuple[float, np.ndarray] | None:
    """Return the latest complete indexed image with grid row zero at the south."""
    path = robot_dir / 'brain_map.txt'
    if not path.exists():
        return None
    rows = path.read_bytes().splitlines(keepends=True)
    for line in reversed(rows):
        if not line.endswith(b'\n') or line.startswith(b'#') or not line.strip():
            continue
        parts = line.decode().split()
        if len(parts) != 5:
            raise ValueError(f'invalid map index row in {path}')
        image_path = robot_dir / parts[4]
        if not image_path.exists():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            if not np.isin(image, [0, 128, 255]).all():
                raise ValueError(f'invalid occupancy values in {image_path}')
            return float(parts[0]), np.flipud(image).copy()
    return None


def grid_cells(xy: np.ndarray) -> np.ndarray:
    return np.floor((np.asarray(xy) - ORIGIN) / CELL_M + 1e-8).astype(int)


def cell_centres(indices: np.ndarray) -> np.ndarray:
    return ORIGIN + (indices + .5) * CELL_M


def is_complete(state: dict) -> bool:
    return str(state.get('state', '')).lower() in {'complete', 'mission_complete', 'completed'}
