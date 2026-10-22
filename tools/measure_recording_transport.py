#!/usr/bin/env python3
"""Measure sensor-only transport batches from an existing recording. Never reads scene truth."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

SMALL_FILES = ('camera.yaml', 'thermal.yaml', 'rgb.txt', 'depth.txt', 'thermal.txt', 'lidar.txt', 'decisions.jsonl')
SENSORS = ('rgb', 'depth', 'thermal')


def snapshot_bytes(path, t):
    raw = path.read_bytes()
    if path.suffix == '.yaml':
        return raw
    kept = []
    for line in raw.splitlines(keepends=True):
        if not line.strip() or line.lstrip().startswith(b'#'):
            kept.append(line)
            continue
        stamp = json.loads(line).get('t', 0) if path.suffix == '.jsonl' else float(line.split()[0])
        if stamp <= t + 1e-9:
            kept.append(line)
    return b''.join(kept)


def plan_batch(folder, t, *, history=False, sent=None, hashes=None):
    """Reference for the Unity patch. Cursors belong to one server endpoint and robot."""
    sent = sent if sent is not None else set()
    hashes = hashes if hashes is not None else {}
    metadata = {name: snapshot_bytes(folder / name, t) for name in SMALL_FILES if (folder / name).is_file()}
    rows = {}
    for sensor in SENSORS:
        rows[sensor] = [(float(p[0]), p[1]) for line in metadata[sensor + '.txt'].decode().splitlines()
                        if line.strip() and not line.lstrip().startswith('#') and (p := line.split())]
        if not rows[sensor]:
            raise ValueError(f'no {sensor} frames at t={t}')
    times = [t] + ([s for s, _ in rows['thermal'][-45:]] if history else [])
    images = {min(rows[sensor], key=lambda row: (abs(row[0] - stamp), row[0]))[1]
              for sensor in SENSORS for stamp in times}
    files = {}
    for name in sorted(images - sent):
        # Apply the server's numeric sensor filename contract, with no recursive directory scan.
        import re
        if not re.fullmatch(r'(rgb|depth|thermal)/[0-9]+(?:\.[0-9]+)?\.png', name):
            raise ValueError(f'invalid sensor filename: {name}')
        path = folder / name
        if not path.resolve().is_relative_to(folder.resolve()) or path.is_symlink():
            raise ValueError('sensor path leaves the recording')
        files[name] = path.read_bytes()
    for name, data in metadata.items():
        digest = hashlib.sha256(data).digest()
        if hashes.get(name) != digest:
            files[name] = data
            hashes[name] = digest
    sent.update(images)
    return files


def measure(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_STORED) as out:
        for name, data in files.items():
            out.writestr(name, data)
    wire = len(stream.getvalue()) if files else 0
    return {'files': len(files), 'pngs': sum(name.endswith('.png') for name in files),
            'payload_bytes': sum(map(len, files.values())), 'zip_bytes': wire,
            'upload_seconds': {str(mbps): round(wire * 8 / (mbps * 1_000_000), 6) for mbps in (5, 20, 50)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--t', type=float, default=20.)
    args = parser.parse_args()
    if not args.recording.is_dir():
        parser.error(f'recording is absent: {args.recording}')
    result = {'recording': str(args.recording), 't': args.t,
              'method': 'ZIP_STORED reference, full metadata snapshots through t, last 45 thermal stamps for agent; seconds = zip_bytes * 8 / decimal bit rate; excludes HTTP/TLS/RTT, inference and intra-tick log order',
              'robots': {}}
    for folder in sorted(args.recording.glob('robot_*')):
        if not folder.is_dir():
            continue
        sent, hashes = set(), {}
        # The recorder scans once per simulation second. Reuse the same endpoint's acknowledgement cursor.
        for tick in range(max(0, int(args.t))):
            plan_batch(folder, float(tick), sent=sent, hashes=hashes)
        person = measure(plan_batch(folder, args.t, sent=sent, hashes=hashes))
        agent_after_person = measure(plan_batch(folder, args.t, history=True, sent=sent, hashes=hashes))
        result['robots'][folder.name] = {
            'person_cold': measure(plan_batch(folder, args.t)),
            'agent_cold': measure(plan_batch(folder, args.t, history=True)),
            'person_after_1hz_scans': person, 'agent_after_1hz_scans': agent_after_person}
    if not result['robots']:
        parser.error('recording contains no robot directories')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
