"""Evaluate navigation recordings against hidden truth, entirely offline."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from brain.nav_io import searched_for
from brain.nav_io import (CELL_M, ORIGIN, interpolate_poses,
                          is_complete, latest_map, poses_from_states, read_json_lines)
from brain.nav_io import (load_scenario, start_pose, recording_frame, placed_states,
                          transform_xy, rotation, map_geometry, sample_grid, frame_peer)
from brain.oracle import load_collisions, load_floorplan, load_house_points, load_trajectory


def body_trajectory(recording: Path, robot: str) -> tuple[np.ndarray, np.ndarray]:
    trajectory = load_trajectory(recording, robot)
    forward = Rotation.from_quat(trajectory.quaternions_xyzw).apply([0., 0., 1.])[:, [0, 2]]
    lengths = np.linalg.norm(forward, axis=1)
    if np.any(lengths < .9):
        raise ValueError('camera forward direction is not approximately horizontal')
    forward /= lengths[:, None]
    xy = trajectory.positions[:, [0, 2]] - .25 * forward
    return trajectory.timestamps, np.c_[xy, np.unwrap(np.arctan2(-forward[:, 0], forward[:, 1]))]


def pose_convention_check(times: np.ndarray, poses: np.ndarray, robot: str, scenario: dict) -> dict:
    spawn = next((r['spawn'] for r in scenario.get('robots', []) if r['id'] == robot), None)
    result = dict(t=float(times[0]), body_xy=poses[0, :2].tolist(), yaw_deg=float(np.degrees(poses[0, 2])))
    if spawn is not None:
        error = float(np.linalg.norm(poses[0, :2] - spawn['at']))
        yaw_error = abs(float(np.degrees(wrap_angle(poses[0, 2] + np.radians(spawn['yaw'])))))
        result.update(spawn_xy=spawn['at'], spawn_yaw_deg=-spawn['yaw'], position_error_m=error,
                      yaw_error_deg=yaw_error, passed=bool(times[0] <= .1 and error < .05 and yaw_error < 2))
    return result


def wrap_angle(value):
    return (value + np.pi) % (2 * np.pi) - np.pi


def error_summary(values: np.ndarray) -> dict:
    if not len(values):
        return dict(median=None, p95=None, max=None, final=None)
    return dict(median=float(np.median(values)), p95=float(np.percentile(values, 95)),
                max=float(np.max(values)), final=float(values[-1]))


def localization_errors(states: list[dict], truth_t: np.ndarray, truth_pose: np.ndarray):
    times, poses = poses_from_states(states)
    keep = (times >= truth_t[0]) & (times <= truth_t[-1])
    times, poses = times[keep], poses[keep]
    truth = interpolate_poses(truth_t, truth_pose, times)
    position = np.linalg.norm(poses[:, :2] - truth[:, :2], axis=1)
    yaw = np.abs(np.degrees(wrap_angle(poses[:, 2] - truth[:, 2])))
    exceeded = times[position > .2]
    report = dict(position_m=error_summary(position), yaw_deg=error_summary(yaw),
                  first_over_0_2_m_t=float(exceeded[0]) if len(exceeded) else None,
                  samples=len(times), excluded_samples=int((~keep).sum()))
    return report, times, position, yaw


def read_lidar(path: Path):
    header, times, scans = None, [], []
    if not path.exists():
        return None
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            if line.startswith('# beams '):
                line = line[2:]
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0] == 'beams':
                header = {parts[i]: float(parts[i + 1]) for i in range(0, len(parts), 2)}
                continue
            if header is None:
                raise ValueError(f'LiDAR data precedes beam header in {path}')
            values = np.array([float(p) for p in parts])
            if len(values) != int(header['beams']) + 1 or not np.isfinite(values).all() or np.any(values[1:] < 0):
                raise ValueError(f'invalid LiDAR scan in {path}')
            times.append(values[0])
            scans.append(values[1:] * .001)
    if header is None or not scans:
        return None
    return header, np.array(times), np.array(scans)


def lidar_endpoints(poses: np.ndarray, ranges: np.ndarray, header: dict) -> np.ndarray:
    angles = header['angle_min_rad'] + np.arange(ranges.shape[1]) * header['increment_rad']
    yaw = poses[:, 2]
    origins = poses[:, :2] + header['mount_forward_m'] * np.c_[-np.sin(yaw), np.cos(yaw)]
    angle = yaw[:, None] + angles
    return origins[:, None, :] + ranges[:, :, None] * np.stack([-np.sin(angle), np.cos(angle)], axis=-1)


def endpoint_distances(poses: np.ndarray, ranges: np.ndarray, header: dict, tree: cKDTree) -> dict:
    valid = ranges > 0
    points = lidar_endpoints(poses, ranges, header)[valid]
    if not len(points):
        return dict(status='no_returns', passed=False, median_m=None, p90_m=None, within_5cm_share=None, endpoints=0)
    distances = tree.query(points)[0]
    median = float(np.median(distances))
    return dict(status='measured', passed=median <= .05, median_m=median,
                p90_m=float(np.percentile(distances, 90)), within_5cm_share=float(np.mean(distances <= .05)), endpoints=len(points))


def lidar_accuracy(robot_dir: Path, truth_t: np.ndarray, truth_pose: np.ndarray,
                   states: list[dict], points: np.ndarray) -> dict:
    data = read_lidar(robot_dir / 'lidar.txt')
    if data is None:
        return dict(status='missing_lidar', true_pose=dict(passed=False))
    header, times, ranges = data
    plane = points[np.abs(-points[:, 1] - header['mount_height_m']) <= .15][:, [0, 2]]
    if not len(plane):
        return dict(status='no_truth_points_at_sensor_height', true_pose=dict(passed=False))
    tree = cKDTree(plane)
    sample_t, sample_r = times[::10], ranges[::10]
    brain_t, brain_pose = poses_from_states(states)
    report = dict(status='measured', scans=len(times), no_return_share=float(np.mean(ranges == 0)),
                  truth_slice_points=len(plane), header=header)
    for name, t, pose in [('true_pose', truth_t, truth_pose), ('brain_pose', brain_t, brain_pose)]:
        keep = (sample_t >= t[0]) & (sample_t <= t[-1])
        if not np.any(keep):
            report[name] = dict(status='no_pose_overlap', passed=False)
            continue
        selected_pose = interpolate_poses(t, pose, sample_t[keep])
        report[name] = endpoint_distances(selected_pose, sample_r[keep], header, tree)
        report[name].update(scans=int(keep.sum()), excluded_scans=int((~keep).sum()))
    return report


def truth_occupancy(points: np.ndarray, shape: tuple, origin=ORIGIN, cell=CELL_M) -> np.ndarray:
    height = -points[:, 1]
    cells = np.floor((points[(height >= .15) & (height <= 1.9)][:, [0, 2]] - origin) / cell + 1e-8).astype(int)
    cells = cells[(cells[:, 0] >= 0) & (cells[:, 0] < shape[1]) & (cells[:, 1] >= 0) & (cells[:, 1] < shape[0])]
    result = np.zeros(shape, bool)
    result[cells[:, 1], cells[:, 0]] = True
    return result


def distance_to_cells(mask: np.ndarray, cell=CELL_M) -> np.ndarray:
    if not mask.any():
        return np.full(mask.shape, np.inf)
    return ndimage.distance_transform_edt(~mask) * cell


def phantom_clusters(mask: np.ndarray, origin=ORIGIN, cell=CELL_M) -> list[dict]:
    labels, count = ndimage.label(mask)
    if not count:
        return []
    sizes = np.bincount(labels.ravel())
    order = sorted(range(1, count + 1), key=lambda n: sizes[n], reverse=True)[:20]
    centres = ndimage.center_of_mass(mask, labels, order)
    return [dict(cells=int(sizes[n]), centre_xy=(origin + (np.array([c[1], c[0]]) + .5) * cell).tolist())
            for n, c in zip(order, centres)]


def door_status(grid: np.ndarray, door: dict, origin=ORIGIN, cell=CELL_M) -> str:
    """Restrict 4-connected traversal to the gap, so routes around walls cannot pass."""
    yy, xx = np.indices(grid.shape)
    x, y = origin[0] + (xx + .5) * cell, origin[1] + (yy + .5) * cell
    cx, cy = door['at']
    normal, tangent = (y - cy, x - cx) if door['wall'] == 'z' else (x - cx, y - cy)
    corridor = (np.abs(tangent) <= door['width'] / 2) & (np.abs(normal) <= .4 + cell + 1e-8)
    start = corridor & (normal <= -.4 + 1e-8)
    end = corridor & (normal >= .4 - 1e-8)
    if not start.any() or not end.any():
        return 'not_observed'
    free = corridor & (grid == 255)
    labels, _ = ndimage.label(free)
    starts = set(labels[start]) - {0}
    if starts.intersection(set(labels[end]) - {0}):
        return 'open'
    # If unknown cells could still supply the crossing, the evidence is incomplete.
    possible, _ = ndimage.label(corridor & (grid != 0))
    if (set(possible[start]) - {0}).intersection(set(possible[end]) - {0}):
        return 'not_observed'
    if np.all(grid[corridor] == 128):
        return 'not_observed'
    return 'closed'


def map_accuracy(grid: np.ndarray, points: np.ndarray, floorplan: dict, origin=ORIGIN, cell=CELL_M) -> tuple[dict, np.ndarray]:
    truth = truth_occupancy(points, grid.shape, origin, cell)
    occupied, observed = grid == 0, grid != 128
    truth_distance = distance_to_cells(truth, cell)
    brain_distance = distance_to_cells(occupied, cell)
    inside_observed = distance_to_cells(observed, cell) <= .3 + 1e-8
    eligible = truth & inside_observed
    phantom = occupied & (truth_distance > .3 + 1e-8)
    doors = dict(open=[], closed=[], not_observed=[])
    for door in floorplan.get('doors', []):
        if door.get('state') in {'open', 'half', 'removed'}:
            doors[door_status(grid, door, origin, cell)].append(door['id'])
    report = dict(precision=float(np.mean(truth_distance[occupied] <= .15)) if occupied.any() else None,
                  recall=float(np.mean(brain_distance[eligible] <= .15)) if eligible.any() else None,
                  occupied_cells=int(occupied.sum()), truth_cells=int(truth.sum()),
                  observed_truth_cells=int(eligible.sum()), phantom_cells=int(phantom.sum()),
                  phantom_clusters=phantom_clusters(phantom, origin, cell), doors=doors)
    return report, truth


STALL_REASONS = ('sweep_started', 'survivor_view_ready', 'survivor_confirmed')
STALL_EXPLAINED_MAX_S = 20.


def detect_stalls(states: list[dict], truth_t: np.ndarray, truth_pose: np.ndarray, events: list[dict]) -> list[dict]:
    """Merge overlapping ten-second windows whose true path stays within 15 cm.

    Require every sampled position to remain close to the window's start. A
    robot that drives a loop and returns to its start is therefore not a stall.
    """
    times, brain = poses_from_states(states)
    keep = (times >= truth_t[0]) & (times <= truth_t[-1])
    times, brain = times[keep], brain[keep]
    selected = [s for s, k in zip(states, keep) if k]
    true = interpolate_poses(truth_t, truth_pose, times)
    intervals = []
    for i, t in enumerate(times):
        j = int(np.searchsorted(times, t + 10))
        if j >= len(times) or any(is_complete(s) for s in selected[i:j + 1]):
            continue
        # Check the 15 Hz trajectory too, not only the brain's 2 Hz samples.
        dense = truth_pose[(truth_t >= t) & (truth_t <= times[j]), :2]
        if len(dense) and np.max(np.linalg.norm(dense - true[i, :2], axis=1)) < .15:
            if intervals and t <= intervals[-1][1]:
                intervals[-1][1] = float(times[j])
            else:
                intervals.append([float(t), float(times[j])])
    result = []
    for start, end in intervals:
        i = int(np.searchsorted(times, start))
        kinds = Counter(e['kind'] for e in events if start <= e['t'] <= end)
        # Standing still on purpose: a look-around sweep or a survivor view. Only a short stall can be explained this
        # way; one that outlasts the deliberate action by more than the 10 s window still counts.
        deliberate = sorted(k for k in STALL_REASONS if kinds.get(k))
        explained = bool(deliberate) and end - start <= STALL_EXPLAINED_MAX_S
        result.append(dict(start_t=start, end_t=end, duration_s=end - start,
                           true_xy=true[i, :2].tolist(), brain_xy=brain[i, :2].tolist(),
                           events=dict(kinds), explained=explained, reason=deliberate if explained else None))
    return result


def save_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f'could not write {path}')


def draw_error_plot(path: Path, times: np.ndarray, position: np.ndarray, yaw: np.ndarray) -> None:
    image = np.full((720, 1200, 3), 255, np.uint8)
    for row, values, title, colour in [(0, position, 'Position error (m)', (180, 50, 20)),
                                        (1, yaw, 'Yaw error (degrees)', (20, 80, 180))]:
        top, bottom = 45 + row * 345, 310 + row * 345
        cv2.putText(image, title, (85, top - 12), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1)
        cv2.rectangle(image, (85, top), (1150, bottom), (80, 80, 80), 1)
        if not len(times):
            continue
        maximum = max(float(values.max()) * 1.1, .25 if row == 0 else 1.)
        for fraction in np.linspace(0, 1, 6):
            y = int(bottom - fraction * (bottom - top))
            cv2.putText(image, f'{maximum * fraction:.2f}', (8, y + 5), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
            x = int(85 + fraction * 1065)
            cv2.putText(image, f'{times[0] + fraction * (times[-1] - times[0]):.1f}', (x - 12, bottom + 22), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
        x = 85 + (times - times[0]) / max(times[-1] - times[0], 1) * 1065
        y = bottom - values / maximum * (bottom - top)
        cv2.polylines(image, [np.c_[x, y].astype(np.int32)], False, colour, 2)
        if row == 0:
            threshold = int(bottom - .2 / maximum * (bottom - top))
            cv2.line(image, (85, threshold), (1150, threshold), (0, 0, 210), 1)
        cv2.putText(image, 'Simulation time (s)', (500, bottom + 42), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
    save_image(path, image)


def draw_overlay(path: Path, shape: tuple, truth: np.ndarray, layers: list[dict], floorplan: dict, origin=ORIGIN, cell=CELL_M) -> None:
    scale, margin = 50., 90
    height, width = int(shape[0] * cell * scale), int(shape[1] * cell * scale)
    base = np.full((*shape, 3), 250, np.uint8)
    for layer in layers:
        grid = layer['grid']
        if grid is not None:
            h, w = grid.shape
            patch = base[:h, :w]
            patch[grid == 255] = (215, 240, 215)
    base[truth] = (150, 150, 150)
    for layer in layers:
        grid = layer['grid']
        if grid is not None:
            base[:grid.shape[0], :grid.shape[1]][grid == 0] = (70, 70, 220)
    image = np.full((height + 2 * margin, width + 2 * margin, 3), 255, np.uint8)
    image[margin:margin + height, margin:margin + width] = cv2.resize(np.flipud(base), (width, height), interpolation=cv2.INTER_NEAREST)

    def pixel(xy):
        p = (np.asarray(xy) - origin) * scale
        return np.c_[margin + p[..., 0], margin + height - p[..., 1]].astype(np.int32).reshape(-1, 2)

    for layer in layers:
        for key, colour in [('true', (0, 0, 0)), ('brain', (220, 60, 20))]:
            points = pixel(layer[key][:, :2])
            if len(points) > 1:
                cv2.polylines(image, [points], False, colour, 2, cv2.LINE_AA)
            if len(points):
                cv2.putText(image, layer['robot'], tuple(points[-1] + [5, -5]), cv2.FONT_HERSHEY_SIMPLEX, .5, colour, 1)
        for stall in layer['stalls']:
            cv2.circle(image, tuple(pixel(stall['true_xy'])[0]), 14, (0, 140, 255), 3)
    for door in floorplan.get('doors', []):
        axis = np.array([1., 0.]) if door['wall'] == 'z' else np.array([0., 1.])
        centre = np.array(door['at'])
        ends = pixel([centre - axis * door['width'] / 2, centre + axis * door['width'] / 2])
        cv2.line(image, tuple(ends[0]), tuple(ends[1]), (150, 40, 140), 2)
    for axis, extent in [(0, shape[1]), (1, shape[0])]:
        for value in np.arange(origin[axis], origin[axis] + extent * cell + 1e-6, 2):
            x, y = pixel([value, origin[1]] if axis == 0 else [origin[0], value])[0]
            cv2.putText(image, f'{value:g}', (x - 12, y + 25) if axis == 0 else (x - 50, y + 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
    cv2.putText(image, 'x east (m)', (width // 2, height + margin + 60), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1)
    cv2.putText(image, 'y north (m)', (5, 32), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1)
    cv2.putText(image, 'Truth grey | occupied red | free green | true path black | brain path blue | stalls orange | doors purple', (margin, 60), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 1)
    save_image(path, image)


def frame_link_report(events, receiver, robots, scenario):
    event = next((e for e in events if e['kind'] == 'frame_linked'), None)
    if event is None:
        return dict(t=None, reported_transform=None, position_error_m=None, yaw_error_deg=None)
    sender = frame_peer(event, receiver, robots)
    n = event['numbers']
    reported = dict(dx_m=n['dx_m'], dy_m=n['dy_m'], dyaw_deg=n['dyaw_deg'])
    result = dict(t=event['t'], sender=sender, reported_transform=reported,
                  position_error_m=None, yaw_error_deg=None)
    a, b = start_pose(scenario, receiver), start_pose(scenario, sender)
    if a is not None and b is not None:
        xy = rotation(a[2]).T @ (b[:2] - a[:2])
        yaw = wrap_angle(b[2] - a[2])
        result.update(truth_transform=dict(dx_m=float(xy[0]), dy_m=float(xy[1]), dyaw_deg=float(np.degrees(yaw))),
                      position_error_m=float(np.linalg.norm(np.array([n['dx_m'], n['dy_m']]) - xy)),
                      yaw_error_deg=abs(float(np.degrees(wrap_angle(np.radians(n['dyaw_deg']) - yaw)))))
    return result


def indoor_report(indoor: np.ndarray, building: np.ndarray, searched: np.ndarray | None, cell: float, tolerance_m: float = .2) -> dict:
    """How much of the robot's indoor layer (and its searched floor) lies outside the true building. Room rectangles
    stop at the wall faces, so cells within tolerance_m of a room (door thresholds, wall cells) count as inside."""
    tolerance_cells = int(round(tolerance_m / cell))
    near = cv2.dilate(building.astype(np.uint8), np.ones((2 * tolerance_cells + 1,) * 2, np.uint8)) > 0
    marked = indoor > 0
    report = dict(indoor_cells=int(marked.sum()), indoor_cells_outside=int((marked & ~near).sum()),
                  tolerance_m=tolerance_cells * cell)
    report['indoor_outside_share'] = report['indoor_cells_outside'] / max(1, report['indoor_cells'])
    if searched is not None:
        report['searched_cells_outside'] = int(((searched == 255) & ~near).sum())
    return report


def floor_mask(floorplan: dict, scenario: dict, shape: tuple, origin, cell, occupied: np.ndarray) -> np.ndarray:
    """True floor a camera could search: cells inside the scenario's rooms that hold no wall or furniture.
    Evaluation only: the scenario's search_area picks which rooms belong to this run's building."""
    yy, xx = np.indices(shape)
    centres = origin + (np.stack([xx, yy], axis=-1) + .5) * cell
    area = scenario.get('search_area') if scenario else None
    mask = np.zeros(shape, bool)
    for room in floorplan.get('rooms', []):
        x0, y0, x1, y1 = room['rect']
        if area and not (x0 >= area[0] - 1e-6 and y0 >= area[1] - 1e-6 and x1 <= area[2] + 1e-6 and y1 <= area[3] + 1e-6):
            continue
        mask |= (centres[..., 0] >= x0) & (centres[..., 0] < x1) & (centres[..., 1] >= y0) & (centres[..., 1] < y1)
    return mask & ~occupied


def evaluate(recording: Path | str, out: Path | str | None = None) -> dict:
    recording = Path(recording).resolve()
    if not (recording / 'meta.json').exists():
        raise ValueError('recording is still running: meta.json is absent; use tools/watch_run.py')
    meta = json.loads((recording / 'meta.json').read_text())
    out = Path(out) if out is not None else Path(__file__).resolve().parents[2] / 'data/runs' / recording.name / 'nav_eval'
    out.mkdir(parents=True, exist_ok=True)
    points = load_house_points(recording)
    floorplan = load_floorplan(recording) or {}
    collisions = load_collisions(recording)
    scenario = load_scenario(meta)
    raw_states = {r: read_json_lines(recording / r / 'brain_state.jsonl') for r in meta['robots']}
    frames = {r: recording_frame(raw_states[r], scenario, r) for r in meta['robots']}
    snapshots = {r: latest_map(recording / r) for r in meta['robots']}
    origin, cell = ORIGIN.copy(), CELL_M
    common_shape = None
    if any(frame == 'local' for _, frame in frames.values()):
        cell = float(json.loads((Path(__file__).resolve().parents[2] / 'config/brain.json').read_text())['cell_m'])
        bounds = [points[:, [0, 2]]]
        for r, snapshot in snapshots.items():
            if snapshot is not None:
                _, grid = snapshot
                source_origin, source_cell = map_geometry(grid.shape, frames[r][1])
                iy, ix = np.nonzero(grid != 128)
                if len(ix):
                    bounds.append(transform_xy(source_origin + (np.c_[ix, iy] + .5) * source_cell, frames[r][0]))
        bounds = np.vstack(bounds)
        origin = np.floor((bounds.min(axis=0) - 2) / cell) * cell
        common_shape = tuple(np.ceil((bounds.max(axis=0) + 2 - origin) / cell).astype(int)[::-1])
    reports, layers = {}, []
    team_searched, team_floor = None, 0
    for robot in meta['robots']:
        robot_dir, output = recording / robot, out / robot
        output.mkdir(parents=True, exist_ok=True)
        states = placed_states(raw_states[robot], frames[robot][0])
        if not states:
            raise ValueError(f'{robot}: no completed brain states')
        truth_t, truth_pose = body_trajectory(recording, robot)
        localization, t, position, yaw = localization_errors(states, truth_t, truth_pose)
        convention = pose_convention_check(truth_t, truth_pose, robot, scenario)
        if scenario:
            convention['scenario_file'] = meta['scenario']
        convention['recorded_frame'] = frames[robot][1]
        initial = states[0]
        if initial['t'] <= .1 and truth_t[0] <= initial['t'] <= truth_t[-1]:
            true_initial = interpolate_poses(truth_t, truth_pose, np.array([initial['t']]))[0]
            convention['transformed_brain_position_error_m'] = float(np.linalg.norm(np.array([initial['x'], initial['y']]) - true_initial[:2]))
            convention['transformed_brain_yaw_error_deg'] = abs(float(np.degrees(wrap_angle(initial['yaw_rad'] - true_initial[2]))))
            convention['passed'] = convention.get('passed', True) and convention['transformed_brain_position_error_m'] < .05 and convention['transformed_brain_yaw_error_deg'] < 2
        print(f'{robot} frame check {json.dumps(convention)}; optical +z projected to house x/z, subtract 0.25 m forward; yaw=atan2(-forward_x, forward_z)')
        lidar = lidar_accuracy(robot_dir, truth_t, truth_pose, states, points)
        events = read_json_lines(robot_dir / 'decisions.jsonl')
        stalls = detect_stalls(states, truth_t, truth_pose, events)
        link = frame_link_report(events, robot, meta['robots'], scenario)
        snapshot = snapshots[robot]
        if snapshot is None:
            grid, map_report = None, dict(status='missing_map')
            shape = common_shape or (440, 440)
        else:
            map_t, grid = snapshot
            source_origin, source_cell = map_geometry(grid.shape, frames[robot][1])
            if common_shape is not None:
                yy, xx = np.indices(common_shape)
                xy = origin + (np.stack([xx, yy], axis=-1) + .5) * cell
                transform = frames[robot][0]
                own = (xy - transform[:2]) @ rotation(transform[2])
                grid = sample_grid(grid, own, source_origin, source_cell)
            shape = grid.shape
            map_report, _ = map_accuracy(grid, points, floorplan, origin, cell)
            map_report.update(status='measured', t=map_t, cell_m=cell, origin_xy=origin.tolist(), shape=list(shape))
            searched = searched_for(robot_dir, map_t)
            if searched is not None:
                if common_shape is not None:
                    searched = sample_grid(searched, own, source_origin, source_cell)
                floor = floor_mask(floorplan, scenario, shape, origin, cell, truth_occupancy(points, shape, origin, cell))
                covered = (searched == 255) & floor
                map_report['searched_floor_share'] = float(covered.sum() / max(1, floor.sum()))
                indoor = searched_for(robot_dir, map_t, 'indoor')
                if indoor is not None:
                    if common_shape is not None:
                        indoor = sample_grid(indoor, own, source_origin, source_cell, fill=0)
                    # Every room of the floor plan counts as inside: a robot with no outline may walk through a door into
                    # the next block of the building (run 12: robot A in office_link), which is indoors, not the yard.
                    building = floor_mask(floorplan, None, shape, origin, cell, np.zeros(shape, bool))
                    map_report['indoor'] = indoor_report(indoor, building, searched, cell)
                team_searched = covered if team_searched is None else team_searched | covered
                team_floor = int(floor.sum())
        report = dict(robot=robot, localization=localization, frame_check=convention, frame_link=link, lidar=lidar,
                      map=map_report, stalls=stalls, collision_seconds=None if collisions is None else collisions.get(robot),
                      collision_note='Writer supplies aggregate seconds only; contact locations and counts are unavailable.')
        reports[robot] = report
        (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        draw_error_plot(output / 'errors.png', t, position, yaw)
        _, brain_pose = poses_from_states(states)
        layer = dict(robot=robot, grid=grid, true=truth_pose, brain=brain_pose, stalls=stalls)
        layers.append(layer)
        draw_overlay(output / 'overlay.png', shape, truth_occupancy(points, shape, origin, cell), [layer], floorplan, origin, cell)
    if team_searched is not None:
        team = dict(searched_floor_share=float(team_searched.sum() / max(1, team_floor)), floor_cells=team_floor)
        (out / 'team.json').write_text(json.dumps(team, indent=2) + '\n')
        print(f"team searched floor share {team['searched_floor_share']:.3f} of {team_floor} floor cells")
    shape = common_shape or tuple(max((layer['grid'].shape[i] if layer['grid'] is not None else 440) for layer in layers) for i in (0, 1))
    draw_overlay(out / 'overlay.png', shape, truth_occupancy(points, shape, origin, cell), layers, floorplan, origin, cell)
    return reports


def print_table(reports: dict) -> None:
    print('robot        pos median/p95/max/final (m)   yaw median/max (deg)   lidar true/brain median (m)   map P/R   stalls   collision s')
    def number(value):
        return 'NA' if value is None else f'{value:.3f}'
    for robot, r in reports.items():
        p, y, lidar = r['localization']['position_m'], r['localization']['yaw_deg'], r['lidar']
        print(f"{robot:<12} {'/'.join(number(p[k]) for k in ['median', 'p95', 'max', 'final']):<29} "
              f"{number(y['median'])}/{number(y['max'])}   "
              f"{number(lidar.get('true_pose', {}).get('median_m'))}/{number(lidar.get('brain_pose', {}).get('median_m'))}   "
              f"{number(r['map'].get('precision'))}/{number(r['map'].get('recall'))}   {len(r['stalls'])} ({sum(not s['explained'] for s in r['stalls'])} unexplained)   {number(r['collision_seconds'])}")
        link = r['frame_link']
        print(f"{robot} frame_link: not linked" if link['t'] is None else
              f"{robot} frame_link: linked at {link['t']:.2f}s from {link['sender']} {link['reported_transform']}; error {number(link['position_error_m'])} m / {number(link['yaw_error_deg'])} deg")
        if lidar.get('true_pose', {}).get('median_m', 0) is not None and lidar.get('true_pose', {}).get('median_m', 0) > .05:
            print(f'LIDAR CONVENTION FAILURE for {robot}: true-pose median exceeds 0.05 m. Check sensor geometry and frame conventions before trusting navigation metrics.')
        elif not lidar['true_pose']['passed']:
            print(f"{robot} LiDAR gate NOT VERIFIED: {lidar.get('status')}, {lidar['true_pose'].get('status', '')}")
        if r['frame_check'].get('passed') is False:
            print(f'{robot} FRAME CHECK FAILED: initial body pose does not match the scenario spawn; inspect frame_check in report.json.')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    try:
        reports = evaluate(args.recording, args.out)
    except (ValueError, OSError, KeyError) as error:
        print(f'NAV EVALUATION INCOMPLETE: {error}')
        return 2
    print_table(reports)
    if any((r['lidar'].get('true_pose', {}).get('median_m') or 0) > .05
           or r['frame_check'].get('passed') is False for r in reports.values()):
        return 1
    if any(not r['lidar']['true_pose']['passed'] or r['map']['status'] != 'measured'
           or r['localization']['samples'] == 0 for r in reports.values()):
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
