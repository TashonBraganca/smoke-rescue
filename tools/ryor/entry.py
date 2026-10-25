#!/usr/bin/env python3
"""Container entry point for one Run-your-own-rescue mission: run the simulator, stream the replay, validate, finish.

Env:  SETUP      setup key, e.g. hx_035feb0125__r2__medium (must exist in data/runs/ryor/setups/)
      MISSION_ID mission id (any [a-z0-9-]); "cache" writes a recorded replay
      BUCKET     S3 bucket; empty writes to LOCAL_OUT instead (precompute and tests)
      PREFIX     key prefix, e.g. ryor/missions/<id> or ryor/cache/<setup>
      LOCAL_OUT  local folder used when BUCKET is empty
      HARD_LIMIT_S  wall-clock limit, default 540; the simulator is killed and the mission marked failed after it
      KILL_AFTER_S  test only: kill the simulator after this many wall seconds (proves the fallback path)
Every failure prints one line "RYOR_FAIL <mission> <reason>" (a CloudWatch metric filter counts them) and exits 1.
"""
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from replay import Builder  # noqa: E402

SETUP, MISSION = os.environ["SETUP"], os.environ.get("MISSION_ID", "local")
BUCKET, PREFIX = os.environ.get("BUCKET", ""), os.environ.get("PREFIX", f"ryor/missions/{MISSION}").rstrip("/")
LOCAL = Path(os.environ.get("LOCAL_OUT", "/tmp/ryor_out"))
HARD_LIMIT_S = float(os.environ.get("HARD_LIMIT_S", "540"))
KILL_AFTER_S = float(os.environ.get("KILL_AFTER_S", "0") or 0)
DOTNET = os.environ.get("DOTNET", "dotnet")
ENGINE = os.environ.get("ENGINE_DLL", str(HERE / "engine/bin/RyorRunner.dll"))
ENGINE_TAG = os.environ.get("ENGINE_TAG", "local")
GIT = os.environ.get("ENGINE_GIT", "unknown")
PREPARE = os.environ.get('PREPARE') == '1'  # Prepare the exact deterministic fallback before enabling a custom run.
TERRA = os.environ.get('TERRA_ENABLED') == '1' and not PREPARE  # Recorded fallback always runs brain-only.

s3 = None
if BUCKET:
    import boto3
    s3 = boto3.client("s3")


def put(name, obj):
    body = json.dumps(obj, separators=(",", ":")).encode()
    if s3:
        s3.put_object(Bucket=BUCKET, Key=f"{PREFIX}/{name}", Body=body, ContentType="application/json", CacheControl="no-store")
    else:
        (LOCAL / name).parent.mkdir(parents=True, exist_ok=True)
        (LOCAL / name).write_bytes(body)


def log(msg):
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {MISSION} {msg}", flush=True)


status = {"v": 1, "mission": MISSION, "setup": SETUP, "state": "starting", "source": "recorded" if MISSION == "cache" else "live",
          "chunks": 0, "sim_t": 0.0, "updated": None, "error": None, "score": None, "engine": {"git": GIT, "image": ENGINE_TAG},
          "wall": {"started": time.time(), "first_frame_s": None, "total_s": None}, 'terra': TERRA}


def write_status(**kw):
    status.update(kw)
    status["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    put("status.json", status)


def fail(reason, proc=None):
    if proc and proc.poll() is None:
        proc.kill()
    print(f"RYOR_FAIL {MISSION} {SETUP} {reason}", flush=True)
    write_status(state="failed", error=reason)
    sys.exit(1)


def main():
    base = SETUP.split('__p', 1)[0]
    req_path = ROOT / "data/runs/ryor/setups" / f"{base}.json"
    if not req_path.exists():
        fail(f"unknown setup {SETUP}")
    gallery = json.loads((ROOT / "data/runs/ryor/gallery.json").read_text())
    req = json.loads(req_path.read_text())
    log('request and gallery loaded')
    run_dir = Path(os.environ.get("RUN_DIR") or f"/tmp/ryor_run/{MISSION}_{SETUP}")
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in ("trace.jsonl", "result.json", "running.marker"):
        (run_dir / f).unlink(missing_ok=True)
    req["runs"][0]["out"] = str(run_dir)
    req['runs'][0]['terra'] = TERRA
    if '__p' in SETUP:
        from placement import validate, setup_key
        if not s3:
            scene = json.loads(Path(os.environ['PLACEMENT_FILE']).read_text())
        else:
            scene = json.loads(s3.get_object(Bucket=BUCKET, Key='ryor/requests/' + SETUP + '.json')['Body'].read())
        house = next(h for h in gallery['houses'] if h['id'] == base.split('__')[0])
        placement = validate(house, scene['placement'])
        if setup_key(base, placement, ENGINE_TAG) != SETUP:
            fail('placement key does not match its content and engine')
        world = json.loads((ROOT / req['runs'][0]['world']).read_text())
        world['people'] = [{'id': 'p%d' % (i + 1), 'at': p} for i, p in enumerate(placement['people'])]
        fire = json.loads((ROOT / 'config' / world['fire']['config']).resolve().read_text())
        from validate_building import inside
        room = next(r for r in house['placement_geometry']['rooms'] if inside(placement['fire'], r['rect']))
        # Placement uses the gate frame; FireSim uses the house frame. This conversion stays world-side.
        a = math.radians(world['gate']['yaw_deg']); ca, sa = math.cos(a), math.sin(a)
        x, y = placement['fire']; gx, gy = world['gate']['origin']
        fire['ignitions'] = [dict(room=room['id'], at=[gx + ca * x + sa * y, gy - sa * x + ca * y], time_s=0, radius_m=.3)]
        # 0.3 m is the unchanged generator ignition radius. Difficulty supplies the head-start time.
        fire_path = run_dir / 'fire.json'; fire_path.write_text(json.dumps(fire))
        world['fire']['config'] = str(fire_path)
        world_path = run_dir / 'world.json'; world_path.write_text(json.dumps(world))
        req['runs'][0]['world'] = str(world_path)
        house['people_truth'] = [{'id': p['id'], 'x': p['at'][0], 'y': p['at'][1]} for p in world['people']]
        house['people'] = len(world['people'])
        status['placement'] = placement
    (run_dir / "request.json").write_text(json.dumps(req))
    write_status(state="starting")
    builder = Builder(gallery, SETUP, req["runs"][0]["duration_s"])
    log('replay floor grid ready')
    worker = None
    if TERRA:
        from terra import Worker
        worker = Worker(run_dir, MISSION, os.environ['RYOR_TABLE'], boto3.client('dynamodb'), boto3.client('secretsmanager'))
        log('asynchronous Terra worker ready')
    env = dict(os.environ, SMOKE_REPO=str(ROOT), DOTNET_NOLOGO="1", DOTNET_CLI_TELEMETRY_OPTOUT="1")
    t0 = time.time()
    proc = subprocess.Popen([DOTNET, ENGINE, str(run_dir / "request.json")], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    signal.signal(signal.SIGTERM, lambda *_: fail("stopped (SIGTERM, task stopped)", proc))
    log(f"simulator pid {proc.pid} started for {SETUP}")
    trace = run_dir / "trace.jsonl"
    while not trace.exists():
        if proc.poll() is not None:
            fail(f"simulator exited {proc.returncode} before writing a trace: {proc.stdout.read().decode()[-300:]}")
        if time.time() - t0 > 60:
            fail("no trace after 60 s", proc)
        time.sleep(0.2)
    write_status(state="running")
    buf, done = "", False
    with trace.open() as fh:
        while True:
            wall = time.time() - t0
            if KILL_AFTER_S and wall > KILL_AFTER_S and proc.poll() is None:
                fail(f"killed by test after {wall:.0f} s wall (KILL_AFTER_S)", proc)
            if wall > HARD_LIMIT_S and proc.poll() is None:
                fail(f"hard limit: {wall:.0f} s wall > {HARD_LIMIT_S:.0f} s at sim {builder.sim_t:.1f} s", proc)
            data = fh.read()
            if data:
                buf += data
                *lines, buf = buf.split("\n")
                for line in lines:
                    if not line:
                        continue
                    event = json.loads(line)
                    if worker and event.get('kind') == 'agent_observation':
                        worker.submit(event)
                    for chunk in builder.feed(event):
                        put(f"chunk_{chunk['chunk']:04d}.json", chunk)
                        if status["wall"]["first_frame_s"] is None:
                            status["wall"]["first_frame_s"] = round(time.time() - status["wall"]["started"], 1)
                        write_status(chunks=chunk["chunk"] + 1, sim_t=builder.sim_t)
                        log(f"chunk {chunk['chunk']} up to sim {builder.sim_t:.1f} s after {time.time() - t0:.0f} s wall")
                continue
            if done:
                break
            if proc.poll() is not None:
                done = True     # read once more: the last lines may have landed after the previous read
                continue
            time.sleep(0.3)
    out = proc.stdout.read().decode()
    if proc.returncode != 0:
        fail(f"simulator exit {proc.returncode}: {out[-300:]}")
    for chunk in builder.finish():
        put(f"chunk_{chunk['chunk']:04d}.json", chunk)
        write_status(chunks=chunk["chunk"] + 1, sim_t=builder.sim_t)
    bad = builder.validate()
    if bad:
        fail("validation: " + "; ".join(bad))
    status["wall"]["total_s"] = round(time.time() - status["wall"]["started"], 1)
    if worker:
        worker.closed = True
        audit = run_dir / 'terra_audit.jsonl'
        if audit.exists():
            # Public replay contains captions only. Full sensor-only observations stay in the private run bucket.
            put('terra_audit.json', [json.loads(line) for line in audit.read_text().splitlines()])
    write_status(state="done", score=builder.score(), validated=True)
    log(f"done: {json.dumps(status['score'])} wall {status['wall']}")


if __name__ == "__main__":
    main()
