#!/usr/bin/env python3
"""Copy a recording's judge evidence into the static site without uploading it."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RUNS = ROOT / 'deploy' / 'aws' / 'site' / 'runs'
ID_PATTERN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,99}')


def data_path(value: Path, *, directory: bool = False) -> Path:
    path = value.resolve()
    if not path.is_relative_to(DATA.resolve()):
        raise ValueError(f'Source must be inside {DATA}: {value}')
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f'Source does not exist: {value}')
    return path


def read_json(path: Path):
    # Reject NaN and Infinity, which browsers cannot parse as JSON.
    def invalid_constant(value):
        raise ValueError(f'Non-JSON number in {path}: {value}')
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=invalid_constant)


def validate_compare(path: Path) -> None:
    rows = read_json(path)
    if not isinstance(rows, list) or not rows or any(
        not isinstance(row, dict) or not isinstance(row.get('recording'), str)
        for row in rows
    ):
        raise ValueError('compare.json must be the nonempty array written by brain.compare_runs.')


def publish(args: argparse.Namespace) -> Path:
    if not ID_PATTERN.fullmatch(args.run_id):
        raise ValueError('Run ID must start with a letter or digit and contain only letters, digits, _ or -, up to 100 characters.')
    recording = data_path(args.recording or DATA / 'recordings' / args.run_id, directory=True)
    trace = data_path(recording / 'agent_trace.jsonl')
    video = data_path(args.walkthrough or DATA / 'evidence' / f'walkthrough_{args.run_id}.mp4')
    if video.stat().st_size == 0:
        raise ValueError('Walkthrough video is empty.')
    for number, line in enumerate(trace.read_text(encoding='utf-8').splitlines(), 1):
        if line.strip():
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f'Trace line {number} must be an object.')
    baseline = data_path(args.baseline, directory=True) if args.baseline else None
    comparison = None if baseline else data_path(args.compare or DATA / 'runs' / args.run_id / 'compare.json')
    if comparison:
        validate_compare(comparison)
    site = ROOT / 'deploy' / 'aws' / 'site'
    if RUNS.is_symlink() or RUNS.resolve() != site.resolve() / 'runs':
        raise ValueError('Refusing a redirected runs directory.')
    RUNS.mkdir(parents=True, exist_ok=True)
    destination = RUNS / args.run_id
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ValueError('Run destination must be a regular directory.')
    catalog_path = RUNS / 'index.json'
    if catalog_path.is_symlink():
        raise ValueError('Refusing a redirected catalog.')
    catalog = read_json(catalog_path) if catalog_path.exists() else {'runs': []}
    if not isinstance(catalog, dict) or not isinstance(catalog.get('runs'), list) or any(
        not isinstance(item, dict) or not isinstance(item.get('id'), str) or not ID_PATTERN.fullmatch(item['id'])
        for item in catalog['runs']
    ):
        raise ValueError('Existing runs/index.json has an invalid catalog.')
    with tempfile.TemporaryDirectory(prefix='.publish-', dir=site) as temp:
        stage = Path(temp)
        shutil.copyfile(trace, stage / 'agent_trace.jsonl')
        shutil.copyfile(video, stage / 'walkthrough.mp4')
        if baseline:
            python = ROOT / 'brain' / '.venv' / 'bin' / 'python'
            if not python.is_file():
                raise ValueError('The --baseline option needs brain/.venv/bin/python and project dependencies.')
            subprocess.run([
                str(python), '-m', 'brain.compare_runs', str(baseline), str(recording),
                '--out', str(stage / 'compare.json'),
            ], cwd=ROOT / 'brain', check=True)
        else:
            shutil.copyfile(comparison, stage / 'compare.json')
        validate_compare(stage / 'compare.json')
        catalog['runs'] = [{'id': args.run_id, 'label': args.label or args.run_id}] + [
            item for item in catalog['runs'] if item['id'] != args.run_id
        ]
        (stage / 'index.json').write_text(json.dumps(catalog, indent=2) + '\n', encoding='utf-8')
        destination.mkdir(exist_ok=True)
        # Replace files rather than following any preexisting destination symlinks.
        for name in ('walkthrough.mp4', 'agent_trace.jsonl', 'compare.json'):
            os.replace(stage / name, destination / name)
        os.replace(stage / 'index.json', catalog_path)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    parser.add_argument('--recording', type=Path, help='Recording directory under data/.')
    parser.add_argument('--walkthrough', type=Path, help='MP4 file under data/.')
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument('--compare', type=Path, help='Existing brain.compare_runs JSON under data/.')
    sources.add_argument('--baseline', type=Path, help='Baseline recording under data/; generate the comparison.')
    parser.add_argument('--label', help='Run picker text, defaults to the run ID.')
    args = parser.parse_args()
    try:
        destination = publish(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Publish failed. {error}\n')
    print(f'Published {args.run_id} to {destination}')
    print(f'Updated {RUNS / "index.json"}. No files uploaded.')


if __name__ == '__main__':
    main()
