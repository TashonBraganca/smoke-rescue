"""Local forwarder to the AWS agent service. Unity's recorder talks to it exactly as it talks to brain.agent.server on
localhost; the forwarder uploads the robot's own sensor files that the remote call reads (PUT /runs ZIP batches, and the
pull-v1 protocol for agent steps), forwards the call with the X-Origin-Verify header and returns the remote answer.

    cd brain && SMOKE_REMOTE_URL=http://<alb-dns> .venv/bin/uvicorn brain.agent.remote_forward:app --port 8090
    Unity: SMOKE_AGENT_URL=http://localhost:8090 before opening the editor (Tools > Smoke Rescue > Record > ... agents)

Environment:
    SMOKE_REMOTE_URL          AWS service base URL (default: the us-east-1 ALB in deploy/aws/public_routes_25oct.txt)
    SMOKE_ORIGIN_HEADER_FILE  file holding "X-Origin-Verify: <value>" (same format as smoke_test.sh --header-file)
    SMOKE_ORIGIN_VERIFY       the header value itself; if neither is set the secret is read once with the aws cli
    SMOKE_ORIGIN_SECRET_ID    Secrets Manager id for that read (default smoke-rescue/origin-verify)
    SMOKE_FALLBACK_URL        optional local server (e.g. http://localhost:8080) used when AWS fails; unset means fail loudly
    SMOKE_REMOTE_RUN_TAG      optional suffix for the remote run id (a replay that wants fresh remote place/map state)
    SMOKE_FORWARD_LOG         optional JSONL file with one line per forwarded call

Failure behaviour: a transport error, timeout, 5xx, or an agent/commander step the remote model never answered is never turned into an empty answer. With a fallback URL
the original request goes to the fallback and the response carries X-Smoke-Served-By: fallback (and a WARNING log
line); without one the forwarder returns HTTP 502 with the reason, which Unity logs and treats as a failed call.
A 4xx from AWS is returned unchanged: it means a request or upload is wrong, and hiding it behind a fallback would hide
a forwarder bug.

Only allowlisted sensor files leave the machine: calibration, rgb/depth/thermal indexes (only the rows a call needs),
the PNGs a call reads, and decisions.jsonl for agent steps. Ground truth is never read here.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import zipfile

import httpx
from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response

log = logging.getLogger("smoke.remote_forward")

FORWARD = {
    'default_remote_url': 'http://smoke--LoadB-m7oFRuQ63UoE-1386562850.us-east-1.elb.amazonaws.com',
    'default_remote_url_reason': 'The stack smoke-rescue-agent ALB (us-east-1). CloudFront is unavailable until the account is verified (deploy/aws/public_routes_25oct.txt).',
    'connect_timeout_s': 5.0,
    'connect_timeout_s_reason': 'Measured /health from India to us-east-1 at 0.42 s with a new connection; ten times that means the ALB is unreachable, not slow.',
    'people_timeout_s': 30.0,
    'people_timeout_s_reason': 'YOLO11m on the 1 vCPU Fargate task measured 2.4 to 2.8 s per frame (25 Oct); 30 s covers a queue of both robots plus uploads, and a dead service still fails well inside Unity\'s 120 s.',
    'upload_timeout_s': 60.0,
    'upload_timeout_s_reason': 'One RGB-D-thermal frame is about 1 MB and uploaded in 1.7 s (25 Oct); 60 s allows a 45-frame pull batch on a slow uplink.',
    'step_deadline_s': 110.0,
    'step_deadline_s_reason': 'Unity\'s agent HttpClient times out at 120 s; the forwarder gives up first so its own error (or the fallback) reaches Unity instead of a client timeout.',
    'poll_interval_s': 0.4,
    'poll_interval_s_reason': 'The ALB WAF allows 2000 requests per 5 min per IP (6.7/s). Two robots polling at 2.5/s leave room for scans and uploads; the server suggests 0.1 s, which would exceed the limit.',
    'health_timeout_s': 5.0,
    'health_timeout_s_reason': 'Unity\'s StartRecording health check itself times out at 5 s.',
    'export_download_timeout_s': 300.0,
    'export_download_timeout_s_reason': 'map3d.ply reached 26 MB in the seed 934 recording; the download runs in the background so it never stalls Unity.',
}

_SENSORS = ('rgb', 'depth', 'thermal')
_IDENT_BAD = re.compile(r'[^A-Za-z0-9_-]')


def _tolerance():
    from brain.agent import vision
    return float(vision._perception()['frame_tolerance_s'])


# ----- local recording reading (the robot's own files only) -------------------------------------------------------
_index_cache: dict = {}
_index_lock = threading.Lock()


def local_index(robot_dir: Path, sensor: str):
    """(stamp, original line, relative path) rows of a local index, cached on size and mtime like vision._index."""
    path = Path(robot_dir) / f'{sensor}.txt'
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    with _index_lock:
        hit = _index_cache.get(str(path))
        if hit and hit[0] == key:
            return hit[1]
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        stamp, rel = line.split()
        rows.append((float(stamp), line.strip(), rel))
    with _index_lock:
        _index_cache[str(path)] = (key, rows)
    return rows


def nearest(rows, t, tolerance):
    """The row vision._index + min((abs, stamp)) would pick, or None beyond the tolerance (the remote then reports it)."""
    if not rows:
        return None
    row = min(rows, key=lambda r: (abs(r[0] - t), r[0]))
    return row if abs(row[0] - t) <= tolerance else None


def run_id_for(robot_dir: Path, tag: str = '') -> str:
    """A remote run id per recording folder instance: name plus the folder's creation time, so a re-recording into the
    same folder name (Unity moves the old one aside) gets a fresh id and never trips the immutable-PNG check."""
    parent = Path(robot_dir).resolve().parent
    st = parent.stat()
    born = int(getattr(st, 'st_birthtime', st.st_ctime))
    name = _IDENT_BAD.sub('_', parent.name).lstrip('_-')[:60] or 'run'
    out = f'{name}-{born}' + (f'-{_IDENT_BAD.sub("_", tag)[:24]}' if tag else '')
    return out[:96]


class RemoteRobot:
    def __init__(self, run_id, robot, robot_dir):
        self.run_id, self.robot, self.robot_dir = run_id, robot, Path(robot_dir)
        self.lock = threading.Lock()
        self.sent: set[str] = set()                      # immutable files already on the server (PNGs, yaml)
        self.rows: dict[str, dict[str, str]] = {s: {} for s in _SENSORS}   # sensor -> rel -> original index line

    @property
    def uri(self):
        return f'run://{self.run_id}/{self.robot}'


class Forwarder:
    def __init__(self, remote_url, origin_header, fallback_url=None, run_tag='', transport=None, log_path=None):
        self.remote = remote_url.rstrip('/')
        self.fallback = fallback_url.rstrip('/') if fallback_url else None
        self.run_tag = run_tag
        self.header = origin_header            # (name, value) or None
        timeout = httpx.Timeout(FORWARD['people_timeout_s'], connect=FORWARD['connect_timeout_s'])
        limits = httpx.Limits(max_connections=16, max_keepalive_connections=8)
        self.http = httpx.Client(timeout=timeout, limits=limits, transport=transport)
        self.local = httpx.Client(timeout=httpx.Timeout(120.0, connect=FORWARD['connect_timeout_s']))
        self.robots: dict[tuple, RemoteRobot] = {}
        self.robots_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        self.calls: list[dict] = []
        self.log_path = Path(log_path) if log_path else None
        self.exports = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ply')
        self.tolerance = _tolerance()

    # ----- helpers -------------------------------------------------------------------------------------------------
    def _headers(self, extra=None):
        h = dict(extra or {})
        if self.header:
            h[self.header[0]] = self.header[1]
        return h

    def robot_state(self, robot_dir, robot) -> RemoteRobot:
        robot_dir = Path(robot_dir).resolve()
        if robot_dir.name != robot:
            raise HTTPException(400, f'robot {robot} differs from its folder {robot_dir.name}')
        key = (str(robot_dir), robot)
        with self.robots_lock:
            state = self.robots.get(key)
            if state is None:
                state = self.robots[key] = RemoteRobot(run_id_for(robot_dir, self.run_tag), robot, robot_dir)
            return state

    def record(self, entry):
        with self.stats_lock:
            self.calls.append(entry)
        if self.log_path:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open('a') as f:
                    f.write(json.dumps(entry) + '\n')
            except OSError:
                log.warning('forward log not writable: %s', self.log_path)

    def upload(self, state: RemoteRobot, files: dict[str, bytes | Path]) -> int:
        """One ZIP batch. PNGs stored (already compressed), text deflated. Returns bytes on the wire."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as z:
            for name in sorted(files):
                data = files[name]
                data = data.read_bytes() if isinstance(data, Path) else data
                z.writestr(zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0)), data,
                           compress_type=zipfile.ZIP_STORED if name.endswith('.png') else zipfile.ZIP_DEFLATED)
        body = buf.getvalue()
        r = self.http.put(f'{self.remote}/runs/{state.run_id}/{state.robot}/files', content=body,
                          headers=self._headers({'Content-Type': 'application/zip',
                                                 'X-Recording-Source': str(state.robot_dir)}),
                          timeout=httpx.Timeout(FORWARD['upload_timeout_s'], connect=FORWARD['connect_timeout_s']))
        if r.status_code != 200:
            raise _Remote(r.status_code, r.content, r.headers.get('content-type'), f'upload refused: HTTP {r.status_code} {r.text[:300]}')
        return len(body)

    def ensure(self, state: RemoteRobot, index_times: dict[str, list[float]], png_times: dict[str, list[float]],
               decisions=False) -> int:
        """Upload what the remote call will read. Indexes are cumulative per robot (rows only ever added), so a
        concurrent call never loses a row it depends on, and the remote nearest-frame choice equals the local one: the
        local nearest row is always in the uploaded subset, and the minimum over a set is the minimum over any subset
        that contains it. Returns bytes uploaded (0 when nothing was missing)."""
        d = state.robot_dir
        with state.lock:
            files: dict[str, bytes | Path] = {}
            for name in ('camera.yaml', 'thermal.yaml'):
                if name not in state.sent and (d / name).is_file():
                    files[name] = d / name
            changed = set()
            for sensor in _SENSORS:
                times = set(index_times.get(sensor, ())) | set(png_times.get(sensor, ()))
                if not times:
                    continue
                rows = local_index(d, sensor)
                for t in sorted(times):
                    row = nearest(rows, t, self.tolerance)
                    if row is None:
                        continue
                    if row[2] not in state.rows[sensor]:
                        state.rows[sensor][row[2]] = row[1]
                        changed.add(sensor)
                    if t in png_times.get(sensor, ()) and row[2] not in state.sent:
                        files[row[2]] = d / row[2]
            # The first batch carries all three indexes: the remote request check reads every sensor's index.
            first = not any(n.endswith('.yaml') for n in state.sent)
            for sensor in _SENSORS:
                if (sensor in changed or first) and state.rows[sensor]:
                    files[f'{sensor}.txt'] = self._index_text(sensor, state.rows[sensor])
            if decisions and (d / 'decisions.jsonl').is_file():
                files['decisions.jsonl'] = d / 'decisions.jsonl'
            if not files:
                return 0
            n = self.upload(state, files)
            state.sent.update(k for k in files if k.endswith('.png') or k.endswith('.yaml'))
            return n

    @staticmethod
    def _index_text(sensor, rows: dict[str, str]) -> bytes:
        lines = sorted(rows.values(), key=lambda l: float(l.split()[0]))
        return ('# ' + sensor + ' index uploaded by remote_forward (rows this run needed)\n' + '\n'.join(lines) + '\n').encode()

    def post(self, path, body: dict, timeout=None):
        r = self.http.post(self.remote + path, json=body, headers=self._headers(),
                           timeout=None if timeout is None else httpx.Timeout(timeout, connect=FORWARD['connect_timeout_s']))
        return r

    # ----- routes --------------------------------------------------------------------------------------------------
    def perceive_people(self, req):
        robot_dir = Path(req['robot_dir'])
        state = self.robot_state(robot_dir, robot_dir.resolve().name)
        t = float(req['t'])
        # The local route loads the full frame (rgb, depth, thermal) through vision.load_frame; so does the run:// route.
        sent = self.ensure(state, {}, {s: [t] for s in _SENSORS})
        return self.post('/perceive/people', dict(req, robot_dir=state.uri)), sent

    def map3d_integrate(self, robot, req):
        state = self.robot_state(req['robot_dir'], robot)
        t = float(req['t'])
        sent = self.ensure(state, {}, {'depth': [t]})
        return self.post(f'/map3d/{robot}/integrate', dict(req, robot_dir=state.uri)), sent

    def place_keyframe(self, robot, req):
        state = self.robot_state(req['robot_dir'], robot)
        t = float(req['t'])
        sent = self.ensure(state, {}, {s: [t] for s in _SENSORS})
        return self.post(f'/place/{robot}/keyframe', dict(req, robot_dir=state.uri)), sent

    def place_match(self, robot, req):
        state = self.robot_state(req['robot_dir'], robot)
        sent = self.ensure(state, {}, {})
        return self.post(f'/place/{robot}/match', dict(req, robot_dir=state.uri)), sent

    def map3d_export(self, robot, req):
        """Unity sends only a local path. The remote writes the PLY in its run folder; the forwarder downloads it to the
        local path in the background (latest export wins), so the 26 MB transfer never stalls the simulation."""
        local_path = Path(req['path'])
        with self.robots_lock:
            state = next((s for (d, r), s in self.robots.items() if r == robot and Path(d) == local_path.resolve().parent), None)
        if state is None:
            raise HTTPException(409, f'no uploads for {robot} yet; export needs an earlier integrate')
        r = self.post(f'/map3d/{robot}/export', {'robot_dir': state.uri, 't': 0})
        if r.status_code == 200:
            url = r.json().get('download_url')
            if url:
                self.exports.submit(self._download, url, local_path)
        return r, 0

    def _download(self, url, local_path: Path):
        try:
            r = self.http.get(self.remote + url, headers=self._headers(),
                              timeout=httpx.Timeout(FORWARD['export_download_timeout_s'], connect=FORWARD['connect_timeout_s']))
            r.raise_for_status()
            tmp = local_path.parent / (local_path.name + '.part')
            tmp.write_bytes(r.content)
            os.replace(tmp, local_path)
            self.record({'route': 'map3d_export_download', 'bytes_down': len(r.content), 'ok': True, 'wall': time.time()})
        except (httpx.HTTPError, OSError) as exc:
            log.warning('map3d export download failed: %s', exc)
            self.record({'route': 'map3d_export_download', 'ok': False, 'error': str(exc), 'wall': time.time()})

    def robot_step(self, robot, req):
        """pull-v1: the remote agent asks for exactly the frames its tools read. The decision-time frame triple is pushed
        with the first batch because every step reads it; history frames (fire tracking) are uploaded when demanded."""
        state = self.robot_state(req['robot_dir'], robot)
        t = float(req['t'])
        history = [float(x) for x in req.get('frame_times', [])]
        deadline = time.monotonic() + FORWARD['step_deadline_s']
        sent = self.ensure(state, {s: [t] + history for s in _SENSORS}, {s: [t] for s in _SENSORS}, decisions=True)
        body = {k: v for k, v in req.items() if k not in ('trace_path', 'debug_dir')}
        body.update(robot_dir=state.uri, transport='pull-v1')
        r = self.post(f'/robot/{robot}/step', body, timeout=FORWARD['upload_timeout_s'])
        if r.status_code != 202:
            return r, sent
        job = r.json()
        polls = 0
        try:
            while True:
                if time.monotonic() > deadline:
                    raise _Remote(504, None, None, f'agent step {job["job_id"][:8]} exceeded {FORWARD["step_deadline_s"]} s')
                time.sleep(FORWARD['poll_interval_s'])
                p = self.http.get(f'{self.remote}/pull/{job["job_id"]}', headers=self._headers())
                polls += 1
                if p.status_code != 200:
                    return p, sent
                s = p.json()
                if s['status'] == 'needs_files':
                    files = {n: state.robot_dir / n for n in s['files'] if n not in state.sent}
                    if files:
                        with state.lock:
                            sent += self.upload(state, files)
                            state.sent.update(files)
                elif s['status'] == 'complete':
                    if model_failed(s['result']):
                        raise _Remote(502, None, None, 'remote agent step ended without a model answer (model_error or '
                                      'deadline in the remote trace); an empty 200 would keep Unity waiting on a dead agent')
                    out = httpx.Response(200, json=s['result'], headers={'X-Smoke-Pull-Polls': str(polls),
                                                                         'X-Smoke-Pull-Requested': str(len(s.get('requested_files', [])))})
                    return out, sent
                elif s['status'] == 'failed':
                    raise _Remote(502, None, None, f'remote agent step failed: {s.get("error")}')
        except BaseException:
            try:
                self.http.delete(f'{self.remote}/pull/{job["job_id"]}', headers=self._headers(), timeout=5)
            except httpx.HTTPError:
                pass
            raise

    def commander_step(self, req):
        # The local trace path is a Mac path; on the read-only container it would fail the step. The remote keeps its own.
        body = {k: v for k, v in req.items() if k != 'trace_path'}
        r = self.post('/commander/step', body, timeout=FORWARD['step_deadline_s'])
        if r.status_code == 200 and model_failed(r.json()):
            raise _Remote(502, None, None, 'remote commander step ended without a model answer')
        return r, 0


def model_failed(result: dict) -> bool:
    """A step the model never answered: the agent loop leaves summary None, with no orders or refusals, only after a
    model_error or its deadline. Deliberate skips (radio text only, repeat, no reports) carry their own flag.
    Measured 25 Oct: every AWS step looked like this (Azure 400) while returning HTTP 200."""
    skipped = any(result.get(k) for k in ('radio_text_only', 'repeat_of_recent_event', 'no_reports'))
    return (not skipped and result.get('summary') is None and not result.get('orders') and not result.get('refusals'))


class _Remote(Exception):
    def __init__(self, status, content, ctype, reason):
        super().__init__(reason)
        self.status, self.content, self.ctype, self.reason = status, content, ctype, reason


# ----- app -----------------------------------------------------------------------------------------------------------
def _origin_header():
    path = os.environ.get('SMOKE_ORIGIN_HEADER_FILE')
    if path:
        name, _, value = Path(path).read_text().strip().partition(':')
        return name.strip(), value.strip()
    if os.environ.get('SMOKE_ORIGIN_VERIFY'):
        return 'X-Origin-Verify', os.environ['SMOKE_ORIGIN_VERIFY'].strip()
    secret = os.environ.get('SMOKE_ORIGIN_SECRET_ID', 'smoke-rescue/origin-verify')
    try:
        value = subprocess.run(['aws', 'secretsmanager', 'get-secret-value', '--region', 'us-east-1', '--secret-id', secret,
                                '--query', 'SecretString', '--output', 'text'], capture_output=True, text=True,
                               timeout=30, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning('no origin header: %s; model and upload routes will return 403', type(exc).__name__)
        return None
    return ('X-Origin-Verify', value) if value else None


_fwd: Forwarder | None = None
_fwd_lock = threading.Lock()


def forwarder() -> Forwarder:
    global _fwd
    with _fwd_lock:
        if _fwd is None:
            _fwd = Forwarder(os.environ.get('SMOKE_REMOTE_URL', FORWARD['default_remote_url']), _origin_header(),
                             os.environ.get('SMOKE_FALLBACK_URL'), os.environ.get('SMOKE_REMOTE_RUN_TAG', ''),
                             log_path=os.environ.get('SMOKE_FORWARD_LOG'))
        return _fwd


app = FastAPI(title='Smoke Rescue AWS forwarder')


def _to_response(r: httpx.Response, served_by: str) -> Response:
    headers = {'X-Smoke-Served-By': served_by}
    for k in ('X-Smoke-Pull-Polls', 'X-Smoke-Pull-Requested'):
        if k in r.headers:
            headers[k] = r.headers[k]
    return Response(r.content, status_code=r.status_code, media_type=r.headers.get('content-type', 'application/json'),
                    headers=headers)


def _fallback(fwd: Forwarder, method, path, raw: bytes, ctype, reason) -> Response:
    if not fwd.fallback:
        log.error('AWS %s %s failed and no fallback is set: %s', method, path, reason)
        return Response(json.dumps({'detail': f'AWS call failed (no fallback configured): {reason}'}), status_code=502,
                        media_type='application/json', headers={'X-Smoke-Served-By': 'none'})
    log.warning('AWS %s %s failed (%s); serving it from the fallback %s', method, path, reason, fwd.fallback)
    try:
        r = fwd.local.request(method, fwd.fallback + path, content=raw, headers={'Content-Type': ctype} if ctype else {})
    except httpx.HTTPError as exc:
        return Response(json.dumps({'detail': f'AWS failed ({reason}) and the fallback failed ({exc})'}), status_code=502,
                        media_type='application/json', headers={'X-Smoke-Served-By': 'none'})
    return _to_response(r, 'fallback')


async def _forward(request: Request, route: str, fn, *args):
    """Run one forwarding function in a worker thread, measure it, fall back or fail loudly on transport errors/5xx."""
    fwd = forwarder()
    raw = await request.body()
    ctype = request.headers.get('content-type', 'application/json')
    start = time.perf_counter()
    entry = {'route': route, 'path': request.url.path, 'wall': time.time()}
    try:
        req = json.loads(raw) if raw else {}
        r, sent = await run_in_threadpool(fn, *args, req)
        entry.update(status=r.status_code, bytes_up=sent, served_by='aws', s=round(time.perf_counter() - start, 3))
        if r.status_code >= 500:
            raise _Remote(r.status_code, r.content, r.headers.get('content-type'), f'HTTP {r.status_code} {r.text[:300]}')
        fwd.record(entry)
        return _to_response(r, 'aws')
    except HTTPException:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, json.JSONDecodeError) or not raw:
            raise HTTPException(400, f'body must be JSON: {exc}') from exc
        raise HTTPException(400, f'bad request for {route}: {exc}') from exc
    except (httpx.HTTPError, _Remote, OSError) as exc:
        reason = exc.reason if isinstance(exc, _Remote) else f'{type(exc).__name__}: {exc}'
        if isinstance(exc, _Remote) and exc.status < 500 and exc.content is not None:
            entry.update(status=exc.status, served_by='aws', s=round(time.perf_counter() - start, 3), error=reason)
            fwd.record(entry)
            return Response(exc.content, status_code=exc.status, media_type=exc.ctype or 'application/json',
                            headers={'X-Smoke-Served-By': 'aws'})
        out = await run_in_threadpool(_fallback, fwd, request.method, request.url.path, raw, ctype, reason)
        entry.update(status=out.status_code, served_by=out.headers.get('X-Smoke-Served-By'), error=reason,
                     s=round(time.perf_counter() - start, 3))
        fwd.record(entry)
        return out


@app.get('/health')
def health():
    fwd = forwarder()
    info = {'remote': fwd.remote, 'fallback': fwd.fallback, 'origin_header': fwd.header is not None}
    try:
        r = fwd.http.get(fwd.remote + '/health', timeout=FORWARD['health_timeout_s'])
        if r.status_code == 200:
            return dict(r.json(), forwarder=dict(info, served_by='aws'))
        reason = f'HTTP {r.status_code}'
    except httpx.HTTPError as exc:
        reason = f'{type(exc).__name__}: {exc}'
    if fwd.fallback:
        try:
            r = fwd.local.get(fwd.fallback + '/health', timeout=FORWARD['health_timeout_s'])
            if r.status_code == 200:
                log.warning('AWS health failed (%s); fallback healthy', reason)
                return dict(r.json(), forwarder=dict(info, served_by='fallback', remote_error=reason))
        except httpx.HTTPError:
            pass
    raise HTTPException(503, f'AWS service unreachable: {reason}')


@app.get('/forwarder/stats')
def stats():
    fwd = forwarder()
    with fwd.stats_lock:
        calls = list(fwd.calls)
    by = {}
    for c in calls:
        b = by.setdefault(c['route'], {'n': 0, 'errors': 0, 'fallbacks': 0, 'bytes_up': 0, 's': []})
        b['n'] += 1
        b['errors'] += int(c.get('status', 200) >= 400 or c.get('ok') is False)
        b['fallbacks'] += int(c.get('served_by') == 'fallback')
        b['bytes_up'] += c.get('bytes_up', 0)
        if 's' in c:
            b['s'].append(c['s'])
    for b in by.values():
        s = sorted(b.pop('s'))
        b['median_s'] = s[len(s) // 2] if s else None
        b['max_s'] = s[-1] if s else None
    return {'remote': fwd.remote, 'routes': by}


@app.post('/perceive/people')
async def perceive_people(request: Request):
    return await _forward(request, 'perceive_people', forwarder().perceive_people)


@app.post('/robot/{robot}/step')
async def robot_step(robot: str, request: Request):
    return await _forward(request, 'robot_step', forwarder().robot_step, robot)


@app.post('/commander/step')
async def commander_step(request: Request):
    return await _forward(request, 'commander_step', forwarder().commander_step)


@app.post('/map3d/{robot}/integrate')
async def map3d_integrate(robot: str, request: Request):
    return await _forward(request, 'map3d_integrate', forwarder().map3d_integrate, robot)


@app.post('/map3d/{robot}/layers')
async def map3d_layers(robot: str, request: Request):
    fwd = forwarder()
    return await _forward(request, 'map3d_layers', lambda req: (fwd.post(f'/map3d/{robot}/layers', req), 0))


@app.post('/map3d/{robot}/export')
async def map3d_export(robot: str, request: Request):
    return await _forward(request, 'map3d_export', forwarder().map3d_export, robot)


@app.post('/place/{robot}/keyframe')
async def place_keyframe(robot: str, request: Request):
    return await _forward(request, 'place_keyframe', forwarder().place_keyframe, robot)


@app.post('/place/{robot}/match')
async def place_match(robot: str, request: Request):
    return await _forward(request, 'place_match', forwarder().place_match, robot)


def _passthrough(method, path):
    async def handler(request: Request):
        fwd = forwarder()
        raw = await request.body()
        ctype = request.headers.get('content-type', 'application/json')
        start = time.perf_counter()
        try:
            r = await run_in_threadpool(lambda: fwd.http.request(method, fwd.remote + request.url.path, content=raw,
                                                                 headers=fwd._headers({'Content-Type': ctype}),
                                                                 params=dict(request.query_params)))
            if r.status_code >= 500:
                raise _Remote(r.status_code, r.content, r.headers.get('content-type'), f'HTTP {r.status_code}')
            fwd.record({'route': path, 'path': request.url.path, 'status': r.status_code, 'served_by': 'aws',
                        's': round(time.perf_counter() - start, 3), 'wall': time.time()})
            return _to_response(r, 'aws')
        except (httpx.HTTPError, _Remote) as exc:
            reason = exc.reason if isinstance(exc, _Remote) else f'{type(exc).__name__}: {exc}'
            return await run_in_threadpool(_fallback, fwd, method, request.url.path, raw, ctype, reason)
    return handler


# Image-only vision tools, approvals and traces need no uploads: the body goes through unchanged.
app.add_api_route('/vision/{tool}', _passthrough('POST', 'vision'), methods=['POST'])
app.add_api_route('/approval', _passthrough('POST', 'approval'), methods=['POST'])
app.add_api_route('/approvals', _passthrough('GET', 'approvals'), methods=['GET'])
app.add_api_route('/approvals/{aid}', _passthrough('POST', 'approval_answer'), methods=['POST'])
app.add_api_route('/trace', _passthrough('GET', 'trace'), methods=['GET'])
