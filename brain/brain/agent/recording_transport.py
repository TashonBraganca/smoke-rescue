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

# Transport limits are engineering bounds, separate from all robot behavior settings.
PULL_LIMITS = {
    'active_jobs': 8, 'active_jobs_reason': 'Two robot decisions plus concurrent detector/place/depth requests, with a bounded server worker count.',
    'wait_s': 120., 'wait_s_reason': 'Matches the recorder HTTP timeout; absent uploads must fail instead of leaving a worker waiting forever.',
    'poll_wait_ms': 100, 'poll_wait_ms_reason': 'At most ten status polls per second per decision while inference runs; file demands return immediately.',
    'finished_kept': 64, 'finished_kept_reason': 'Keep recent results for polling retries while bounding retained job metadata.',
}


def tool_frame_times(t, frame_times=(), seconds=3.):
    """Use exactly the existing track_fire_spread tool's timestamp selection, including its ten-frame cap."""
    t, seconds = float(t), float(seconds)
    if not math.isfinite(t) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('finite time and positive history duration required')
    times = [float(s) for s in frame_times]
    if any(not math.isfinite(s) or s > t for s in times):
        raise ValueError('history times must be finite and cannot exceed decision time')
    times = [s for s in times if s >= t-seconds] or [t]
    if len(times) > 10:
        step = len(times)/10
        times = [times[int(i*step)] for i in range(10)]
    return times


def tool_files(folder, tool, t, frame_times=(), seconds=3.):
    """Sensor paths actually read by the current tools, without decoding or re-encoding their PNGs.

    Fire tracking measures depth-scaled area/speed as well as thermal flow. Dropping depth history would change
    those measurements. RGB history is never used by that tool and is never requested here.
    """
    from brain.agent import vision
    folder = Path(folder)
    sensors = {'track_fire_spread': ('depth', 'thermal'), 'depth': ('depth',),
               'rgb_depth': ('rgb', 'depth')}.get(tool, ('rgb', 'depth', 'thermal'))
    times = tool_frame_times(t, frame_times, seconds) if tool == 'track_fire_spread' else [float(t)]
    names = set()
    for sensor in sensors:
        rows = vision._index(folder, sensor)
        for stamp in times:
            actual, path = min(rows, key=lambda r: (abs(r[0]-stamp), r[0]))
            if abs(actual-stamp) > vision._perception()['frame_tolerance_s']:
                raise ValueError(f'no {sensor} frame near t={stamp}')
            name = str(path.relative_to(folder))
            if not _IMAGE.fullmatch(name):
                raise ValueError('index contains a non-sensor path')
            names.add(name)
    return sorted(names)


class PullCancelled(RuntimeError):
    pass


class PullJob:
    """A decision waits on named sensor files; Unity polls the demand and supplies the original bytes.

    Each job snapshots metadata so a later index upload cannot change the frame the tool will measure. PNGs are
    immutable in RecordingStore and are hard-linked after arrival. No callback URL, local client path or truth is opened.
    """
    def __init__(self, store, robot, req):
        import secrets
        import time
        self.store, self.robot = store, robot
        self.source = store.resolve(req.get('robot_dir', ''))
        if self.source.name != robot:
            raise ValueError('request robot differs from run URI robot')
        self.req = dict(req)
        t = float(req['t'])
        if not math.isfinite(t):
            raise ValueError('decision time must be finite')
        tool_frame_times(t, req.get('frame_times', ()))
        self.id = secrets.token_urlsafe(24)
        self.created = time.monotonic()
        self.condition = threading.Condition()
        self.pending = set()
        self.requested = set()
        self.status, self.result, self.error = 'running', None, None
        self.cancelled = False
        # Temporary snapshots are under the server-owned recording root and contain sensor files only.
        self.snapshot = Path(tempfile.mkdtemp(prefix='.pull-', dir=store.root))
        self.folder = self.snapshot / robot
        self.folder.mkdir()
        with _lock(self.source):
            for name in SMALL_FILES | {'.recording-source.json'}:
                path = _inside(store.root, self.source/name)
                if path.is_file():
                    (self.folder/name).write_bytes(path.read_bytes())
        self.resolved = dict(req, robot=robot, robot_dir=str(self.source),
                             trace_path=str(self.source.parent/'agent_trace.jsonl'),
                             debug_dir=str(self.source.parent/'agent_debug'))
        self.worker = None

    def require(self, tool, t, frame_times=(), seconds=3.):
        import time
        from brain.agent import vision
        names = tool_files(self.folder, tool, t, frame_times, seconds)
        # Keep snapshot metadata fixed. Uploaders hold the source lock, never this job's condition.
        with self.condition:
            while True:
                if self.cancelled:
                    raise PullCancelled('pull decision cancelled')
                remaining = PULL_LIMITS['wait_s'] - (time.monotonic()-self.created)
                if remaining <= 0:
                    raise PullCancelled('pull sensor upload timed out')
                missing = set()
                with _lock(self.source):
                    for name in names:
                        target = self.folder/name
                        if target.is_file():
                            continue
                        source = _inside(self.store.root, self.source/name)
                        if not source.is_file():
                            missing.add(name)
                            continue
                        target.parent.mkdir(parents=True, exist_ok=True)
                        os.link(source, target)
                if not missing:
                    self.pending.clear(); self.status = 'running'
                    return
                self.pending = missing; self.requested.update(missing); self.status = 'needs_files'
                self.condition.notify_all()
                self.condition.wait(remaining)

    def state(self):
        with self.condition:
            result = dict(job_id=self.id, status=self.status, files=sorted(self.pending),
                          requested_files=sorted(self.requested),
                          requested_png_bytes=sum((self.source/name).stat().st_size for name in self.requested
                                                  if (self.source/name).is_file()))
            if self.status == 'complete':
                result['result'] = self.result
            if self.status == 'failed':
                result['error'] = self.error
            return result

    def cancel(self):
        with self.condition:
            self.cancelled = True; self.status = 'failed'; self.error = 'pull decision cancelled'
            self.pending.clear(); self.condition.notify_all()

    def run(self, execute):
        token = ACTIVE_PULL.set(self)
        try:
            result = execute(self)
            with self.condition:
                if not self.cancelled:
                    self.result, self.status = result, 'complete'
        except Exception as exc:
            with self.condition:
                self.error, self.status = str(exc), 'failed'
        finally:
            ACTIVE_PULL.reset(token)
            with self.condition:
                self.pending.clear(); self.condition.notify_all()
            shutil.rmtree(self.snapshot)


class PullJobs:
    def __init__(self):
        self.jobs = {}
        self.gate = threading.RLock()

    def submit(self, store, robot, req, execute):
        with self.gate:
            finished = [key for key, job in self.jobs.items() if job.status in ('complete', 'failed')]
            for key in finished[:-PULL_LIMITS['finished_kept']]:
                # A cancelled worker can still be unwinding its tool/model call.
                if not self.jobs[key].worker.is_alive():
                    del self.jobs[key]
            active = [j for j in self.jobs.values() if j.worker.is_alive()]
            if len(active) >= PULL_LIMITS['active_jobs']:
                raise HTTPException(429, 'too many active pull requests')
            source = store.resolve(req.get('robot_dir', ''))
            if any(j.source == source and j.req.get('operation', 'step') == req.get('operation', 'step') for j in active):
                raise HTTPException(409, 'this robot already has an active pull operation')
            job = PullJob(store, robot, req)
            self.jobs[job.id] = job
            job.worker = threading.Thread(target=job.run, args=(execute,), daemon=True)
            job.worker.start()
            return dict(job_id=job.id, status='running', poll_url=f'/pull/{job.id}', poll_wait_ms=PULL_LIMITS['poll_wait_ms'])

    def get(self, key):
        with self.gate:
            if key not in self.jobs:
                raise HTTPException(404, 'unknown or expired pull request')
            return self.jobs[key]

    def notify(self, source):
        with self.gate:
            jobs = [j for j in self.jobs.values() if j.source == source]
        for job in jobs:
            with job.condition:
                job.condition.notify_all()


from contextvars import ContextVar
ACTIVE_PULL = ContextVar('smoke_rescue_pull', default=None)


def thermal_depth_frame(folder, t):
    """Only the two measured histories. No fake RGB image is constructed to satisfy the full Frame loader."""
    import hashlib
    import cv2
    import numpy as np
    from brain.agent import vision
    from brain.io import read_camera, load_opencv_yaml, required_number
    folder = Path(folder)
    cam = read_camera(vision._safe_file(folder, 'camera.yaml'))
    settings = load_opencv_yaml(vision._safe_file(folder, 'thermal.yaml'))
    k = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1.]])
    kt = np.array([[required_number(settings, 'Camera1.fx', 'Camera.fx', 'fx'), 0, required_number(settings, 'Camera1.cx', 'Camera.cx', 'cx')],
                   [0, required_number(settings, 'Camera1.fy', 'Camera.fy', 'fy'), required_number(settings, 'Camera1.cy', 'Camera.cy', 'cy')], [0, 0, 1.]])
    translation = settings.get('T_rgb_thermal.translation_m')
    if translation is None:
        translation = [required_number(settings, 'mount_right_offset_m'), 0., 0.]
    translation = np.asarray(translation, float)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError('invalid camera translation')
    transform = np.eye(4); transform[:3, 3] = -translation
    images, stamps = {}, {}
    for sensor in ('depth', 'thermal'):
        stamp, path = min(vision._index(folder, sensor), key=lambda r: (abs(r[0]-t), r[0]))
        if abs(stamp-t) > vision._perception()['frame_tolerance_s']:
            raise ValueError(f'no {sensor} frame near t={t}')
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16 or raw.ndim != 2:
            raise ValueError(f'invalid uint16 {sensor} PNG')
        images[sensor], stamps[sensor] = raw, stamp
    if images['depth'].shape != (cam.height, cam.width) or images['thermal'].shape != (int(required_number(settings, 'Camera.height', 'height')), int(required_number(settings, 'Camera.width', 'width'))):
        raise ValueError('sensor size differs from calibration')
    depth = images['depth'].astype(np.float32)/cam.depth_factor
    digest = hashlib.sha256()
    for a in (depth, images['thermal'], k, kt, transform):
        digest.update(a.tobytes())
    result = vision._result('load_thermal_depth', [stamps], dict(source=frame_source(folder), requested_t=t, content=digest.hexdigest()),
                            stamps=stamps, source=frame_source(folder))

    class ThermalDepthFrame(dict):
        pass

    frame = ThermalDepthFrame(result)
    frame.depth_m, frame.thermal = depth, images['thermal']
    frame.K, frame.thermal_K, frame.rgb_to_thermal = k, kt, transform
    frame._registered = vision.register_depth(depth, k, kt, transform, frame.thermal.shape)
    for a in (frame.depth_m, frame.thermal, k, kt, transform, frame._registered):
        a.flags.writeable = False
    return frame


def rgb_depth_frame(folder, t):
    """The colour detector measures RGB and depth only. Thermal pixels are neither substituted nor transmitted."""
    import cv2
    import numpy as np
    from brain.agent import vision
    from brain.io import read_camera
    folder = Path(folder)
    cam = read_camera(vision._safe_file(folder, 'camera.yaml'))
    images, stamps = {}, {}
    for sensor in ('rgb', 'depth'):
        stamp, path = min(vision._index(folder, sensor), key=lambda r: (abs(r[0]-t), r[0]))
        if abs(stamp-t) > vision._perception()['frame_tolerance_s']:
            raise ValueError(f'no {sensor} frame near t={t}')
        images[sensor] = cv2.imread(str(path), cv2.IMREAD_COLOR if sensor == 'rgb' else cv2.IMREAD_UNCHANGED)
        if images[sensor] is None:
            raise ValueError(f'invalid {sensor} PNG')
        stamps[sensor] = stamp
    if images['depth'].dtype != np.uint16 or images['depth'].shape != (cam.height, cam.width) or images['rgb'].shape != (cam.height, cam.width, 3):
        raise ValueError('RGB-D size or type differs from calibration')

    class RGBDepthFrame(dict):
        pass

    frame = RGBDepthFrame(stamps=stamps, source=frame_source(folder))
    frame.rgb = cv2.cvtColor(images['rgb'], cv2.COLOR_BGR2RGB)
    frame.depth_m = images['depth'].astype(np.float32)/cam.depth_factor
    frame.K = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1.]])
    for a in (frame.rgb, frame.depth_m, frame.K):
        a.flags.writeable = False
    return frame
