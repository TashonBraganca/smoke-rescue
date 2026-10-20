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


def searched_for(robot_dir: Path, map_t: float, layer: str = 'searched') -> np.ndarray | None:
    """A mask written beside the snapshot at map_t, south row first; None if absent. Layers: 'searched' (255 searched
    by a camera), 'indoor' (255 ceiling seen, 128 within the indoor radius of one, 64 walled in on the robot's own
    map, 0 not known to be indoors)."""
    path = robot_dir / 'brain_map' / f'{map_t:.6f}.{layer}.png'
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if image is None else np.flipud(image).copy()


def grid_cells(xy: np.ndarray) -> np.ndarray:
    return np.floor((np.asarray(xy) - ORIGIN) / CELL_M + 1e-8).astype(int)


def cell_centres(indices: np.ndarray) -> np.ndarray:
    return ORIGIN + (indices + .5) * CELL_M


def is_complete(state: dict) -> bool:
    return str(state.get('state', '')).lower() in {'complete', 'mission_complete', 'completed'}


def load_scenario(meta: dict) -> dict:
    """Resolve scenario inheritance for offline evaluation and display."""
    config = Path(__file__).resolve().parents[2] / 'config'
    name = meta.get('scenario')
    if not name:
        return {}
    path = Path(name)
    if not path.is_absolute():
        path = config / path.name
    seen = set()
    def read(path):
        if path in seen:
            raise ValueError('scenario inheritance cycle')
        seen.add(path)
        data = json.loads(path.read_text())
        parent = read(path.parent / data['inherits']) if 'inherits' in data else {}
        return parent | data
    return read(path)


def start_pose(scenario: dict, robot: str) -> np.ndarray | None:
    spawn = next((r['spawn'] for r in scenario.get('robots', []) if r['id'] == robot), None)
    return None if spawn is None else np.array([*spawn['at'], -np.radians(spawn['yaw'])], float)


def rotation(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def transform_xy(xy, transform):
    return np.asarray(xy) @ rotation(transform[2]).T + transform[:2]


def transform_poses(poses, transform):
    result = np.asarray(poses, dtype=float).copy()
    result[..., :2] = transform_xy(result[..., :2], transform)
    result[..., 2] += transform[2]
    return result


def recording_frame(states: list[dict], scenario: dict, robot: str):
    """Old logs start at the spawn; new logs start at their own origin.

    Missing scenario metadata identifies legacy recordings only when their first
    state is nonzero. An origin-only log needs a spawn to be placed on a plan.
    """
    start = start_pose(scenario, robot)
    if not states:
        return np.zeros(3), 'shared'
    first = states[0]
    pose = np.array([first['x'], first['y'], first['yaw_rad']])
    if start is not None:
        yaw_error = (pose[2] - start[2] + np.pi) % (2 * np.pi) - np.pi
        if first['t'] <= .1 and np.linalg.norm(pose[:2] - start[:2]) < .05 and abs(yaw_error) < np.radians(2):
            return np.zeros(3), 'shared'
        return start, 'local'
    if first['t'] <= .1 and np.linalg.norm(pose) < .05:
        raise ValueError(f'{robot}: local-frame recording requires scenario spawn metadata')
    return np.zeros(3), 'shared'


def placed_states(states, transform):
    result = []
    for state in states:
        s = dict(state)
        s['x'], s['y'], s['yaw_rad'] = transform_poses([s['x'], s['y'], s['yaw_rad']], transform).tolist()
        for key in ('path', 'goal'):
            if s.get(key):
                s[key] = transform_xy(s[key], transform).tolist()
        result.append(s)
    return result


def map_geometry(shape, frame: str):
    """Use image dimensions and configured metres; preserve legacy grid layout."""
    config = json.loads((Path(__file__).resolve().parents[2] / 'config/brain.json').read_text())
    cell = float(config['cell_m'])
    if frame == 'local':
        size = float(config['grid_size_m'])
        origin = np.array([-size / 2, -size / 2])
        if tuple(shape) != (round(size / cell), round(size / cell)):
            raise ValueError(f'local grid shape {shape} disagrees with grid_size_m/cell_m')
    else:
        origin = np.array([config.get('grid_origin_x_m', -2.), config.get('grid_origin_z_m', -2.)])
    return origin, cell


def sample_grid(grid, xy, origin, cell, fill=128):
    """Sample a south-first grid at arbitrary points, with fill (occupancy: 128 unknown) outside it."""
    cells = np.floor((np.asarray(xy) - origin) / cell + 1e-8).astype(int)
    valid = (cells[..., 0] >= 0) & (cells[..., 0] < grid.shape[1]) & (cells[..., 1] >= 0) & (cells[..., 1] < grid.shape[0])
    result = np.full(valid.shape, fill, np.uint8)
    result[valid] = grid[cells[..., 1][valid], cells[..., 0][valid]]
    return result


def frame_peer(event: dict, robot: str, robots: list[str]) -> str | None:
    for key in ('sender', 'peer', 'from', 'peer_id'):
        if event.get(key) in robots:
            return event[key]
    # Existing two-robot decisions identify the peer in the reason string.
    peers = [r for r in robots if r != robot]
    mentioned = [r for r in peers if r in event.get('reason', '')]
    return mentioned[0] if len(mentioned) == 1 else peers[0] if len(peers) == 1 else None
