#!/usr/bin/env python3
"""Watch appended robot logs without consulting hidden ground truth.

Run from brain/: uv run python ../tools/watch_run.py ../data/recordings/RUN
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'brain'))
from brain.nav_io import CELL_M, ORIGIN, interpolate_poses, is_complete, latest_map, poses_from_states

KINDS = ('collision_risk', 'collision_hold', 'stuck', 'goal_blacklisted', 'no_progress')


class JsonTail:
    """Keep an incomplete trailing record for the next poll."""
    def __init__(self, path):
        self.path, self.offset, self.pending, self.identity = path, 0, b'', None

    def read(self):
        if not self.path.exists():
            return []
        stat = self.path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity != self.identity or stat.st_size < self.offset:
            self.offset, self.pending, self.identity = 0, b'', identity
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            data = self.pending + stream.read()
            self.offset = stream.tell()
        lines = data.split(b'\n')
        self.pending = lines.pop()
        return [json.loads(line) for line in lines if line.strip()]


def is_pinned(states):
    if not states or is_complete(states[-1]):
        return False
    times, poses = poses_from_states(states)
    start = times[-1] - 15
    if times[0] > start:
        return False
    first = interpolate_poses(times, poses, np.array([start]))[0, :2]
    window = np.vstack([first, poses[times >= start, :2]])
    return bool(np.max(np.linalg.norm(window - first, axis=1)) < .15)


def save_snapshot(robot_dir, state, snap):
    latest = latest_map(robot_dir)
    if latest is None:
        return
    _, grid = latest
    image = cv2.cvtColor(np.flipud(grid), cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    def pixel(xy):
        p = (np.asarray(xy) - ORIGIN) / CELL_M * 3
        return np.c_[p[..., 0], image.shape[0] - 1 - p[..., 1]].astype(np.int32).reshape(-1, 2)
    path = state.get('path') or []
    if len(path) > 1:
        cv2.polylines(image, [pixel(path)], False, (240, 90, 20), 2)
    xy, yaw = np.array([state['x'], state['y']]), state['yaw_rad']
    arrow = pixel([xy, xy + .5 * np.array([-np.sin(yaw), np.cos(yaw)])])
    cv2.arrowedLine(image, tuple(arrow[0]), tuple(arrow[1]), (30, 30, 230), 2, tipLength=.3)
    snap.mkdir(parents=True, exist_ok=True)
    temporary = snap / f'.{robot_dir.name}_latest.tmp.png'
    if not cv2.imwrite(str(temporary), image):
        raise OSError(f'cannot write {temporary}')
    temporary.replace(snap / f'{robot_dir.name}_latest.png')


class RunWatcher:
    def __init__(self, recording, snap=None):
        self.recording = Path(recording)
        self.snap = Path(snap) if snap is not None else None
        self.readers, self.history, self.localization = {}, {}, {}

    def lines(self):
        result = []
        for robot in sorted(self.recording.glob('robot_*')):
            if not robot.is_dir():
                continue
            name = robot.name
            if name not in self.readers:
                self.readers[name] = (JsonTail(robot / 'brain_state.jsonl'), JsonTail(robot / 'decisions.jsonl'))
                self.history[name] = []
            state_reader, event_reader = self.readers[name]
            self.history[name].extend(state_reader.read())
            events = event_reader.read()
            counts = Counter(e['kind'] for e in events)
            for event in events:
                if event['kind'] == 'localization':
                    self.localization[name] = event.get('numbers', {})
            states = self.history[name]
            counters = ' '.join(f'{kind}={counts[kind]}' for kind in KINDS)
            spam = [kind for kind, count in counts.items() if count > 30]
            if states:
                state = states[-1]
                line = f"{name} t={state['t']:.1f} x={state['x']:.3f} y={state['y']:.3f} state={state['state']} {counters}"
                if is_pinned(states):
                    line += ' PINNED'
                if self.snap is not None:
                    save_snapshot(robot, state, self.snap)
                # Retain one sample before the fifteen-second window.
                cut = max(0, int(np.searchsorted([s['t'] for s in states], state['t'] - 15)) - 1)
                self.history[name] = states[cut:]
            else:
                line = f'{name} t=NA x=NA y=NA state=waiting {counters}'
            line += ' localization=' + json.dumps(self.localization.get(name), separators=(',', ':'))
            if spam:
                line += ' SPAM=' + ','.join(sorted(spam))
            result.append(line)
        return result


def file_signature(recording, snap=None):
    """Check file metadata only, excluding hidden truth and our own snapshots."""
    result = []
    snap = Path(snap).resolve() if snap is not None else None
    for root, dirs, files in os.walk(recording):
        dirs[:] = [d for d in dirs if d != 'oracle']
        for name in files:
            path = Path(root) / name
            if snap is not None and Path(root).resolve() == snap and (
                name.endswith('_latest.png') or name.endswith('_latest.tmp.png')
            ):
                continue
            try:
                stat = path.stat()
                result.append((str(path), stat.st_size, stat.st_mtime_ns))
            except FileNotFoundError:
                continue
    return tuple(sorted(result))


def watch(recording, every=120., snap=None, clock=time.monotonic, sleep=time.sleep):
    recording = Path(recording)
    if not recording.is_dir():
        raise ValueError(f'recording directory does not exist: {recording}')
    if every <= 0:
        raise ValueError('--every must be positive')
    watcher = RunWatcher(recording, snap)
    signature = file_signature(recording, snap)
    now = last_change = clock()
    next_print = now
    while True:
        if (recording / 'meta.json').exists():
            for line in watcher.lines():
                print(line, flush=True)
            print('RUN FINISHED', flush=True)
            return 0
        now = clock()
        current = file_signature(recording, snap)
        if current != signature:
            signature, last_change = current, now
        if now - last_change >= 600:
            print('NO PROGRESS', flush=True)
            return 2
        if now >= next_print:
            for line in watcher.lines():
                print(line, flush=True)
            next_print = now + every
        sleep(min(5., every, max(.01, next_print - now)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--every', type=float, default=120)
    parser.add_argument('--snap', type=Path)
    args = parser.parse_args(argv)
    try:
        return watch(args.recording, args.every, args.snap)
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError) as error:
        print(f'WATCH ERROR: {error}', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
