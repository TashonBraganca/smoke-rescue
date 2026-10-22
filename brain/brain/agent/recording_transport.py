"""ZIP batches of robot sensor files, stored under a server-owned run directory."""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import threading
import zipfile

from fastapi import HTTPException

# One batch covers 45 RGB-D/thermal frames. Bounds apply to both wire and expanded bytes.
MAX_BATCH_BYTES = 64 * 1024 * 1024
MAX_FILES = 2048
SMALL_FILES = frozenset(('camera.yaml', 'thermal.yaml', 'rgb.txt', 'depth.txt', 'thermal.txt',
                         'lidar.txt', 'decisions.jsonl'))
_IDENTIFIER = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}\Z')
_IMAGE = re.compile(r'(rgb|depth|thermal)/[0-9]+(?:\.[0-9]+)?\.png\Z')
_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()


def _lock(folder):
    with _locks_guard:
        return _locks.setdefault(str(folder), threading.RLock())


def _inside(root, path):
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('recording path leaves the server run folder')
    # Do not follow even an internal symlink when publishing or reading uploads.
    relative = path.relative_to(root)
    if root.is_symlink() or any((root.joinpath(*relative.parts[:i])).is_symlink()
                                for i in range(1, len(relative.parts) + 1)):
        raise ValueError('recording symlinks are refused')
    return path


def _identifier(value):
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError('run and robot ids must use letters, digits, underscores or hyphens, at most 96 characters')
    return value


def frame_source(folder):
    """Keep legacy evidence ids. The source is provenance only, never a path opened by the server."""
    marker = folder / '.recording-source.json'
    return json.loads(marker.read_text()) if marker.exists() else str(folder)


class RecordingStore:
    def __init__(self, root):
        self.root = Path(root)

    def folder(self, run, robot):
        return _inside(self.root, self.root / _identifier(run) / _identifier(robot))

    def resolve(self, uri):
        match = re.fullmatch(r'run://([^/]+)/([^/]+)', uri)
        if not match:
            raise ValueError('robot_dir must be run://{run_id}/{robot}')
        folder = self.folder(*match.groups())
        if not (folder / '.recording-source.json').is_file():
            raise HTTPException(404, 'run or robot not found; upload its sensor files first')
        return folder

    def upload(self, run, robot, stream, source=None):
        folder = self.folder(run, robot)
        if source is not None and (not source or len(source) > 4096 or '\x00' in source):
            raise ValueError('invalid recording source')
        with _lock(folder):
            self.root.mkdir(parents=True, exist_ok=True)
            # Validate the whole archive before publishing anything. extractall is deliberately unused.
            with tempfile.TemporaryDirectory(dir=self.root, prefix='.upload-') as staging:
                staging = Path(staging)
                try:
                    with zipfile.ZipFile(stream) as batch:
                        entries = batch.infolist()
                        if not entries or len(entries) > MAX_FILES:
                            raise ValueError(f'ZIP must contain 1 to {MAX_FILES} files')
                        if sum(e.file_size for e in entries) > MAX_BATCH_BYTES:
                            raise HTTPException(413, 'expanded ZIP exceeds 64 MiB')
                        names = set()
                        for entry in entries:
                            name = entry.filename
                            kind = stat.S_IFMT(entry.external_attr >> 16)
                            if (name in names or '\\' in name or entry.orig_filename != name
                                    or kind not in (0, stat.S_IFREG) or entry.flag_bits & 1
                                    or not (name in SMALL_FILES or _IMAGE.fullmatch(name))):
                                raise ValueError(f'file refused: {name}; only sensor PNGs, calibration, indexes, lidar and decisions are allowed; truth files are forbidden')
                            names.add(name)
                            target = _inside(self.root, folder / name)
                            output = staging / name
                            output.parent.mkdir(parents=True, exist_ok=True)
                            with batch.open(entry) as src, output.open('wb') as dst:
                                shutil.copyfileobj(src, dst)
                            # Images and calibration are immutable within a run. Safe retries send the same bytes.
                            if target.exists() and (name.endswith('.png') or name.endswith('.yaml')):
                                if target.read_bytes() != output.read_bytes():
                                    raise ValueError(f'immutable sensor file changed: {name}; start a new run')
                except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
                    raise ValueError(f'invalid ZIP batch: {exc}') from exc
                marker = folder / '.recording-source.json'
                if marker.exists():
                    previous = json.loads(marker.read_text())
                    if source is not None and source != previous:
                        raise ValueError('recording source changed; start a new run')
                    source = previous
                folder.mkdir(parents=True, exist_ok=True)
                # Publish PNGs before the rewritten indexes. Readers use this same robot lock.
                for name in sorted(names, key=lambda n: (not n.endswith('.png'), n)):
                    target = _inside(self.root, folder / name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(staging / name, target)
                temporary = staging / '.recording-source.json'
                temporary.write_text(json.dumps(source or f'run://{run}/{robot}'))
                os.replace(temporary, marker)
                from brain.agent import vision
                vision.load_frame.cache_clear()
        return {'robot_dir': f'run://{run}/{robot}', 'files': len(names),
                'bytes': sum(e.file_size for e in entries)}

    @contextmanager
    def request(self, req, sensors=('rgb', 'depth', 'thermal'), history=False, robot=None):
        uri = req.get('robot_dir', '')
        if not isinstance(uri, str):
            raise HTTPException(400, 'robot_dir must be a string')
        if not uri.startswith('run:'):
            yield req
            return
        try:
            folder = self.resolve(uri)
            if robot is not None and folder.name != robot:
                raise ValueError('request robot differs from run URI robot')
            with _lock(folder):
                from brain.agent import vision
                t = float(req['t'])
                times = [t] + ([float(x) for x in req.get('frame_times', [])] if history else [])
                if not all(math.isfinite(x) for x in times):
                    raise ValueError('frame times must be finite')
                calibration = ['camera.yaml'] + (['thermal.yaml'] if 'thermal' in sensors else [])
                for name in calibration:
                    if not _inside(self.root, folder / name).is_file():
                        raise ValueError(f'missing uploaded recording file: {name}')
                tolerance = vision._perception()['frame_tolerance_s']
                for sensor in sensors:
                    index = folder / (sensor + '.txt')
                    if not _inside(self.root, index).is_file():
                        raise ValueError(f'missing uploaded recording index: {sensor}.txt')
                    rows = vision._index(folder, sensor)
                    for stamp in times:
                        actual, path = min(rows, key=lambda r: (abs(r[0] - stamp), r[0]))
                        if abs(actual - stamp) > tolerance:
                            raise ValueError(f'missing {sensor} frame near t={stamp}; upload the requested history window')
                        if not _inside(self.root, path).is_file():
                            raise ValueError(f'missing uploaded frame: {path.relative_to(folder)} at t={actual}')
                remote = dict(req, robot_dir=str(folder), trace_path=str(folder.parent / 'agent_trace.jsonl'),
                              debug_dir=str(folder.parent / 'agent_debug'))
                yield remote
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc
