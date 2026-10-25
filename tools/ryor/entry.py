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
          "wall": {"started": time.time(), "first_frame_s": None, "total_s": None}}


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
    req_path = ROOT / "data/runs/ryor/setups" / f"{SETUP}.json"
    if not req_path.exists():
        fail(f"unknown setup {SETUP}")
    gallery = json.loads((ROOT / "data/runs/ryor/gallery.json").read_text())
    req = json.loads(req_path.read_text())
    run_dir = Path(os.environ.get("RUN_DIR") or f"/tmp/ryor_run/{MISSION}_{SETUP}")
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in ("trace.jsonl", "result.json", "running.marker"):
        (run_dir / f).unlink(missing_ok=True)
    req["runs"][0]["out"] = str(run_dir)
    (run_dir / "request.json").write_text(json.dumps(req))
    write_status(state="starting")
    builder = Builder(gallery, SETUP, req["runs"][0]["duration_s"])
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
            data = fh.read()
            if data:
                buf += data
                *lines, buf = buf.split("\n")
                for line in lines:
                    if not line:
                        continue
                    for chunk in builder.feed(json.loads(line)):
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
            wall = time.time() - t0
            if KILL_AFTER_S and wall > KILL_AFTER_S:
                fail(f"killed by test after {wall:.0f} s wall (KILL_AFTER_S)", proc)
            if wall > HARD_LIMIT_S:
                fail(f"hard limit: {wall:.0f} s wall > {HARD_LIMIT_S:.0f} s at sim {builder.sim_t:.1f} s", proc)
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
    write_status(state="done", score=builder.score())
    log(f"done: {json.dumps(status['score'])} wall {status['wall']}")


if __name__ == "__main__":
    main()
