"""HTTP service for the agents and the OpenCV 5 tools. Unity's recorder calls it; on AWS it runs on Fargate (Graviton).

    uv run uvicorn brain.agent.server:app --port 8080
    AGENT_BACKEND=ollama|bedrock|replay  AGENT_MODEL=qwen3:8b  AGENT_RUN_DIR=../data/runs/<run>  APPROVAL_MODE=simulated|human

Endpoints: GET /health; POST /robot/{robot}/step; POST /commander/step; POST /approval (ask), GET /approvals,
POST /approvals/{id} (a person answers); POST /vision/{tool} with a PNG body for the judge demo; GET /trace.
"""

from __future__ import annotations

import io
import itertools
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request

from brain.agent.commander import Commander, simulated_operator
from brain.agent.llm import make_backend
from brain.agent.robot_agent import RobotAgent

RUN_DIR = Path(os.environ.get("AGENT_RUN_DIR", "/tmp/smoke-rescue-agent"))
RUN_DIR.mkdir(parents=True, exist_ok=True)
TRACE = RUN_DIR / "agent_trace.jsonl"
AREA = tuple(float(v) for v in os.environ.get("SEARCH_AREA", "0,0,40,28").split(","))
APPROVAL_MODE = os.environ.get("APPROVAL_MODE", "simulated")

app = FastAPI(title="Smoke Rescue agents")
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
    return RobotAgent(backend(), Path(req.get("trace_path") or TRACE)).step(req)


@app.post("/commander/step")
async def commander_step(request: Request):
    req = await request.json()
    return Commander(backend(), Path(req.get("trace_path") or TRACE), AREA).step(req)


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


@app.get("/trace")
def trace(n: int = 200):
    if not TRACE.exists():
        return []
    lines = TRACE.read_text().splitlines()[-max(1, min(n, 2000)):]
    return [json.loads(l) for l in lines]
