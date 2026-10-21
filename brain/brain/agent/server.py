"""HTTP service for the agents and the OpenCV 5 tools. Unity's recorder calls it; on AWS it runs on Fargate (Graviton).

    uv run uvicorn brain.agent.server:app --port 8080
    AGENT_BACKEND=ollama|bedrock|replay  AGENT_MODEL=qwen3:8b  AGENT_RUN_DIR=../data/runs/<run>  APPROVAL_MODE=simulated|human

Endpoints: GET /health; POST /robot/{robot}/step; POST /commander/step; POST /approval (ask), GET /approvals,
POST /approvals/{id} (a person answers); POST /vision/{tool} with a PNG body for the judge demo; GET /trace.
POST /map3d/{robot}/integrate, /layers, /export fuse each robot's own depth and estimated pose.
"""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager
import io
import itertools
import json
import os
import time
import threading
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from brain.agent.commander import Commander, simulated_operator
from brain.agent.llm import make_backend
from brain.agent.robot_agent import RobotAgent
from brain.agent.map3d import RobotMap3D

RUN_DIR = Path(os.environ.get("AGENT_RUN_DIR", "/tmp/smoke-rescue-agent"))
RUN_DIR.mkdir(parents=True, exist_ok=True)
TRACE = RUN_DIR / "agent_trace.jsonl"
APPROVAL_MODE = os.environ.get("APPROVAL_MODE", "simulated")

_maps3d: dict[str, RobotMap3D] = {}
_maps3d_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    try:
        yield
    finally:
        with _maps3d_lock:
            for volume in _maps3d.values():
                volume.close()
            _maps3d.clear()


app = FastAPI(title="Smoke Rescue agents", lifespan=lifespan)
_backend = None
_approvals: dict[str, dict] = {}
_ids = itertools.count(1)


def backend():
    global _backend
    if _backend is None:
        _backend = make_backend()
    return _backend


@app.get("/health")
def health():
    b = os.environ.get("AGENT_BACKEND", "ollama")
    return {"status": "ok", "backend": b, "model": os.environ.get("AGENT_MODEL", "qwen3:8b" if b == "ollama" else ""),
            "opencv": cv2.__version__, "run_dir": str(RUN_DIR), "approval_mode": APPROVAL_MODE}


@app.post("/robot/{robot}/step")
async def robot_step(robot: str, request: Request):
    req = await request.json()
    req["robot"] = robot
    # In a worker thread: the recorder sends both robots' steps at once, and a blocking call here would serialise them.
    return await run_in_threadpool(RobotAgent(backend(), Path(req.get("trace_path") or TRACE)).step, req)


@app.post("/commander/step")
async def commander_step(request: Request):
    req = await request.json()
    return await run_in_threadpool(Commander(backend(), Path(req.get("trace_path") or TRACE)).step, req)


@app.post("/approval")
async def ask_approval(request: Request):
    req = await request.json()
    aid = f"a{next(_ids)}"
    rec = {"id": aid, "t": req.get("t"), "robot": req.get("robot"), "question": req.get("question"),
           "evidence": req.get("evidence"), "asked_wall": time.time()}
    if APPROVAL_MODE == "simulated":
        rec.update(simulated_operator(req))
    else:
        rec.update({"approved": None, "by": None})
    _approvals[aid] = rec
    with TRACE.open("a") as f:
        f.write(json.dumps({"kind": "approval", **rec}, default=str) + "\n")
    return rec


@app.get("/approvals")
def list_approvals():
    return list(_approvals.values())


@app.post("/approvals/{aid}")
async def answer_approval(aid: str, request: Request):
    if aid not in _approvals:
        raise HTTPException(404, "no such approval")
    body = await request.json()
    _approvals[aid].update({"approved": bool(body.get("approved")), "by": "human", "reason": body.get("reason", "")})
    with TRACE.open("a") as f:
        f.write(json.dumps({"kind": "approval_answer", **_approvals[aid]}, default=str) + "\n")
    return _approvals[aid]


@app.post("/vision/{tool}")
async def vision_tool(tool: str, request: Request):
    """Judge demo on one uploaded image: assess_visibility on an RGB PNG, or a thermal scan on a 16-bit thermal PNG
    (kelvin x 10). Full tools need registered depth, so this endpoint runs the image-only parts."""
    data = await request.body()
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("multipart/form-data"):
        # Browsers and curl -F send multipart; take the first file part without needing python-multipart.
        from email.parser import BytesParser
        from email.policy import default
        msg = BytesParser(policy=default).parsebytes(b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + data)
        parts = [p for p in msg.iter_parts() if p.get_filename() or p.get_content_type().startswith("image/")]
        data = parts[0].get_payload(decode=True) if parts else b""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(400, "body must be a PNG or JPEG image")
    t0 = time.perf_counter()
    if tool == "assess_visibility":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.astype(np.uint8)
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(gray.std())
        dark = float(cv2.erode(img.min(axis=2) if img.ndim == 3 else gray, np.ones((15, 15), np.uint8)).mean() / 255)
        out = {"laplacian_var": round(lap, 1), "rms_contrast": round(contrast, 1), "dark_channel": round(dark, 3)}
    elif tool == "scan_thermal":
        if img.dtype != np.uint16:
            raise HTTPException(400, "scan_thermal needs a 16-bit thermal PNG (kelvin x 10)")
        c = img.astype(np.float32) / 10 - 273.15
        out = {}
        for name, mask in (("hot_regions", c >= 200), ("human_band_regions", (c >= 28) & (c <= 40))):
            n, _, stats, cents = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
            out[name] = [{"bbox": stats[i, :4].tolist(), "pixels": int(stats[i, 4]), "centre": cents[i].round(1).tolist()}
                         for i in range(1, n) if stats[i, 4] >= 8]
    else:
        raise HTTPException(404, f"unknown tool {tool}; use assess_visibility or scan_thermal")
    out.update({"tool": tool, "opencv": cv2.__version__, "ms": round((time.perf_counter() - t0) * 1000, 1)})
    return out


@app.post("/perceive/people")
async def perceive_people(request: Request):
    """The robots' live person detector: OpenCV 5 dnn running YOLO11m on the robot's own colour frame at t, with
    range from its own depth. Called by the recorder once per sim second per robot, with or without the LLM agents."""
    from brain.agent import vision
    req = await request.json()
    t0 = time.perf_counter()
    frame = await run_in_threadpool(vision.load_frame, req["robot_dir"], float(req["t"]))
    people = await run_in_threadpool(lambda: vision.detect_people_rgb(frame, min_conf=float(req.get("min_conf", 0.25))))
    return {"t": float((frame.get("stamps") or {}).get("rgb", req["t"])), "people": people, "opencv": cv2.__version__, "ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.get("/trace")
def trace(n: int = 200):
    if not TRACE.exists():
        return []
    lines = TRACE.read_text().splitlines()[-max(1, min(n, 2000)):]
    return [json.loads(l) for l in lines]


def _map3d_request(robot, operation, req):
    start = time.perf_counter()
    with _maps3d_lock:
        volume = _maps3d.get(robot)
        if volume is None:
            volume = _maps3d[robot] = RobotMap3D()
    try:
        if operation == 'integrate':
            volume.integrate(req['robot_dir'], req['t'], req['pose'])
            result = {'ok': True}
        elif operation == 'layers':
            top, seen = volume.layers(req['cell_m'], req['origin'], req['size'])
            result = {'top_cm': base64.b64encode(top.astype('<u2').tobytes(order='C')).decode('ascii'),
                      'seen': base64.b64encode(np.packbits(seen.ravel(), bitorder='little').tobytes()).decode('ascii')}
        else:
            volume.export(req['path'])
            result = {'ok': True}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    result['ms'] = (time.perf_counter() - start) * 1000
    return result


@app.post('/map3d/{robot}/integrate')
async def map3d_integrate(robot: str, request: Request):
    return await run_in_threadpool(_map3d_request, robot, 'integrate', await request.json())


@app.post('/map3d/{robot}/layers')
async def map3d_layers(robot: str, request: Request):
    """Row zero is south. Seen bits use little bit order with zero padding."""
    return await run_in_threadpool(_map3d_request, robot, 'layers', await request.json())


@app.post('/map3d/{robot}/export')
async def map3d_export(robot: str, request: Request):
    return await run_in_threadpool(_map3d_request, robot, 'export', await request.json())
