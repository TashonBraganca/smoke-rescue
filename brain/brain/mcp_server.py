"""Smoke Rescue as an MCP server: any robot or AI agent can call the perception the simulated robots use.

    cd brain
    .venv/bin/python -m brain.mcp_server                       # stdio (an MCP desktop client, a coding agent)
    .venv/bin/python -m brain.mcp_server --http --port 8096    # streamable HTTP at http://127.0.0.1:8096/mcp

Tools (images are a file path on this machine, base64 text, or a data: URL; PNG or JPEG):
    detect_people(image, depth?)         YOLO11m through OpenCV 5 cv2.dnn, with range from depth when given
    assess_visibility(image, depth?)     sharpness, contrast and dark-channel haze: is the colour camera usable?
    match_place(keyframe_a, keyframe_b)  ORB place keyframes and the planar rigid transform between two views
    describe_scene_heat(thermal, ...)    hot (>= 200 C) and human-band (28 to 40 C) regions in a 16-bit thermal frame

Backends: local (default) runs brain.agent.vision and brain.agent.place in this process, the code the robots run.
remote sends the same frame to the HTTP service at SMOKE_API_URL (default: the public AWS Lambda URL below) through
its public routes: /perceive/people, /vision/assess_visibility, /vision/scan_thermal. Choose with SMOKE_BACKEND or
the per-call backend argument. match_place has no public AWS route and always runs locally (the result says so).
Remote results are what the route returns; their field names differ from the local tools (see each tool's text).
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import functools
import inspect
import json
import math
import os
import sys
from typing import Any

import numpy as np

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

DEFAULT_API_URL = "https://ajfjm7evl33gu6ptprty6hfthq0ukygx.lambda-url.us-east-1.on.aws/api"
DEFAULT_API_URL_REASON = ("The public HTTPS front of the AWS service (deploy/aws/site_lambda_25oct.txt). It forwards "
                          "only GET /api/health, POST /api/vision/<tool> and POST /api/perceive/people, so no caller "
                          "can reach the model routes or upload recordings through it.")
REMOTE_TIMEOUT_S = 75.0
REMOTE_TIMEOUT_S_REASON = ("The Lambda function times out at 70 s (site_lambda.yaml); waiting 5 s longer lets its own "
                           "timeout error come back instead of a client-side one.")
DEFAULT_HTTP_PORT = 8096
DEFAULT_HTTP_PORT_REASON = ("Clear of the project's own services: 8080 and 8081 run the AWS forwarder and the agent "
                            "server on this machine, and 8090 is the forwarder's documented port (remote_forward.py).")
DEFAULT_MIN_CONF = 0.25
DEFAULT_MIN_CONF_REASON = ("The same default as POST /perceive/people, the robots' live detector. The brain confirms a "
                           "person only at brain.json person_rgb_min_confidence (0.5); callers wanting that pass 0.5.")

INSTRUCTIONS = """Perception from the Smoke Rescue robots (OpenCV 5). Send camera frames, get people, visibility, place
matches and heat regions back. Bearings are degrees, positive to the right of the camera axis. Ranges are horizontal
metres from the robot body centre and assume the project's camera mast (1.6 m high, pitched 10 deg down); a robot with a
different mount gets correct boxes and bearings but biased ranges. Images are a file path on the server machine, base64
text or a data: URL. Depth is a 16-bit PNG in depth_scale units per metre (default 1000, millimetres)."""

mcp = MCPServer("smoke-rescue", instructions=INSTRUCTIONS, version="0.1.0")


def api_url() -> str:
    return os.environ.get("SMOKE_API_URL", DEFAULT_API_URL).rstrip("/")


def _backend(choice: str | None) -> str:
    b = (choice or os.environ.get("SMOKE_BACKEND", "local")).strip().lower()
    if b not in ("local", "remote"):
        raise ValueError("backend must be 'local' or 'remote'")
    return b


def _client():
    """HTTP client for the remote backend. Tests replace it with a client bound to the FastAPI app in-process."""
    import httpx
    return httpx.Client(base_url=api_url(), timeout=REMOTE_TIMEOUT_S)


def _remote(path: str, **kwargs) -> dict:
    with _client() as c:
        r = c.post(path, **kwargs)
    if r.status_code != 200:
        hint = (" A deployment older than the inline-frame change to POST /perceive/people (25 Oct) answers 500 here."
                if path == "/perceive/people" and r.status_code == 500 else "")
        raise ValueError(f"remote {path} answered HTTP {r.status_code}: {r.text[:300]}.{hint}")
    out = r.json()
    out.update({"backend": "remote", "url": f"{api_url()}{path}"})
    return out


def _jsonable(value):
    """Plain JSON types (numpy scalars and arrays become numbers and lists; NaN becomes null)."""
    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serialisable: {type(o).__name__}")
    def clean(o):
        if isinstance(o, float) and not math.isfinite(o):
            return None
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        return o
    return clean(json.loads(json.dumps(value, default=default)))


def _errors(fn):
    """Bad input and remote failures reach the caller as a tool error with the reason; the SDK would otherwise hide
    the text of any exception that is not a ToolError."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc
    return wrapper


@contextlib.contextmanager
def _quiet():
    # stdio carries the protocol on stdout; anything a library prints must go to stderr instead.
    with contextlib.redirect_stdout(sys.stderr):
        yield


@mcp.tool()
@_errors
def detect_people(image: str, depth: str | None = None, depth_scale: float = 1000.0,
                  intrinsics: dict[str, float] | None = None, min_conf: float = DEFAULT_MIN_CONF,
                  backend: str | None = None) -> dict[str, Any]:
    """Find people in one colour camera frame with YOLO11m (ONNX) through OpenCV 5 cv2.dnn, the robots' live detector.

    image: colour PNG/JPEG (path, base64 or data: URL). depth: optional 16-bit depth PNG of the same size, depth_scale
    units per metre. intrinsics: optional {fx, fy, cx, cy} in pixels; default is the project camera scaled to the image.
    Returns people: [{confidence, bbox_rgb [x, y, w, h], bearing_deg (right positive), range_m (null without depth),
    range_source, floor_gap_m (about 0 for a person on the floor)}]. backend: 'local' or 'remote' (AWS /perceive/people).
    """
    from brain.agent import robot_frame
    b = _backend(backend)
    if b == "remote":
        body = {"rgb_b64": base64.b64encode(robot_frame.read_bytes(image, name="image")).decode("ascii"),
                "depth_scale": depth_scale, "min_conf": min_conf}
        if depth is not None:
            body["depth_b64"] = base64.b64encode(robot_frame.read_bytes(depth, name="depth")).decode("ascii")
        if intrinsics is not None:
            body["intrinsics"] = intrinsics
        return _remote("/perceive/people", json=body)
    from brain.agent import vision
    import cv2
    import time
    with _quiet():
        frame, meta = robot_frame.build_frame(rgb=image, depth=depth, intrinsics=intrinsics, depth_scale=depth_scale)
        t0 = time.perf_counter()
        people = vision.detect_people_rgb(frame, min_conf=float(min_conf))
    return _jsonable({"people": people, "count": len(people), "min_conf": float(min_conf), **meta,
                      "model": vision.PERSON_ONNX.name, "opencv": cv2.__version__, "backend": "local",
                      "ms": round((time.perf_counter() - t0) * 1000, 1)})


@mcp.tool()
@_errors
def assess_visibility(image: str, depth: str | None = None, depth_scale: float = 1000.0,
                      backend: str | None = None) -> dict[str, Any]:
    """Is the colour camera usable through smoke? Laplacian variance (sharpness), RMS grey contrast and dark-channel
    haze (a brightness proxy, not physical extinction), with the robots' verdict usable_rgb (variance >= 20, contrast
    >= 12, haze <= 0.8) and, given depth, the fraction of pixels with valid depth.

    backend 'remote' posts to AWS /vision/assess_visibility, which returns laplacian_var, rms_contrast and dark_channel
    (a mean, where the local haze is a median) and no verdict.
    """
    from brain.agent import robot_frame
    if _backend(backend) == "remote":
        return _remote("/vision/assess_visibility", content=robot_frame.read_bytes(image, name="image"),
                       headers={"content-type": "image/png"})
    from brain.agent import vision
    with _quiet():
        frame, meta = robot_frame.build_frame(rgb=image, depth=depth, depth_scale=depth_scale)
        out = vision.assess_visibility(frame)
    if depth is None:
        out["valid_depth_fraction"] = None   # no depth was sent; 0.0 would claim the depth camera saw nothing
    return _jsonable({**out, **meta, "backend": "local"})


def _keyframe(spec, name, settings):
    """A place keyframe from {rgb, depth, intrinsics?, depth_scale?} or from a radio payload (base64 text)."""
    from brain.agent import place, robot_frame
    if isinstance(spec, str):
        kf = place.decode_b64(spec)
        return kf, {"source": "payload", "features": len(kf)}
    if not isinstance(spec, dict) or "rgb" not in spec or "depth" not in spec:
        raise ValueError(f"{name} must be a radio payload (base64) or an object with rgb and depth images")
    frame, meta = robot_frame.build_frame(rgb=spec["rgb"], depth=spec["depth"], intrinsics=spec.get("intrinsics"),
                                          depth_scale=float(spec.get("depth_scale", 1000.0)))
    kf, n = place.make_keyframe(frame.rgb, frame.depth_m, frame.K, settings)
    if kf is None:
        raise ValueError(f"{name}: only {n} ORB features have usable depth; a keyframe needs "
                         f"{settings['place_min_features']}")
    return kf, {"source": "images", "features": len(kf), **meta}


@mcp.tool()
@_errors
def match_place(keyframe_a: dict[str, Any] | str, keyframe_b: dict[str, Any] | str,
                return_payloads: bool = False) -> dict[str, Any]:
    """Do two RGB-D views show the same place, and how are they placed relative to each other?

    Each keyframe is {rgb, depth, intrinsics?, depth_scale?} (images as for detect_people) or a radio payload, the base64
    text a robot sends its partner (b'VP' format, 38 bytes per feature). ORB features with depth become 3D points in the
    robot body frame; Lowe-ratio matches are fitted with the robots' seeded 2D rigid RANSAC (brain.agent.place).
    Returns matched, and when true the pose of keyframe_b in keyframe_a's body frame: dx_m (right), dy_m (forward),
    dyaw_deg (counter-clockwise positive), with inliers and rmse_m. Unmatched results give the reason. Always local:
    the public AWS routes have no stateless place match.
    """
    from brain.agent import place
    with _quiet():
        settings = place.load_settings()
        kf_a, info_a = _keyframe(keyframe_a, "keyframe_a", settings)
        kf_b, info_b = _keyframe(keyframe_b, "keyframe_b", settings)
        result = place.match_keyframes(kf_b, [(0, kf_a)], settings)
    result.pop("mine_id", None)
    out = {**result, "frame": "pose of keyframe_b in keyframe_a's body frame: x right, y forward, yaw ccw",
           "keyframe_a": info_a, "keyframe_b": info_b,
           "payload_bytes": {"a": place.HEADER_BYTES + len(kf_a) * place.RECORD_BYTES,
                             "b": place.HEADER_BYTES + len(kf_b) * place.RECORD_BYTES},
           "backend": "local"}
    if return_payloads:
        out["payload_b64"] = {"a": place.encode_b64(kf_a.desc, kf_a.xyh), "b": place.encode_b64(kf_b.desc, kf_b.xyh)}
    return _jsonable(out)


@mcp.tool()
@_errors
def describe_scene_heat(thermal: str, rgb: str | None = None, depth: str | None = None, depth_scale: float = 1000.0,
                        backend: str | None = None) -> dict[str, Any]:
    """Heat in one thermal frame: a 16-bit single-channel PNG holding kelvin x 10 (the project's FLIR Boson model).

    Local: the robots' scan_thermal. hotspots are connected regions at or above 200 C (fire); people are warm regions
    from 28 to 40 C, unconfirmed (a warm wall in a hot smoke layer also qualifies). With rgb and depth of the matching
    colour camera, regions get a range and size in metres and rgb_people adds colour person boxes. Also returns
    max_c, median_c and the fraction of valid pixels in each band.
    backend 'remote' posts to AWS /vision/scan_thermal: hot_regions and human_band_regions in thermal pixels only.
    """
    from brain.agent import robot_frame
    if _backend(backend) == "remote":
        return _remote("/vision/scan_thermal", content=robot_frame.read_bytes(thermal, name="thermal"),
                       headers={"content-type": "image/png"})
    from brain.agent import vision
    if depth is not None and rgb is None:
        raise ValueError("depth belongs to the colour camera; send rgb with it")
    with _quiet():
        frame, meta = robot_frame.build_frame(rgb=rgb, depth=depth, thermal=thermal, depth_scale=depth_scale)
        out = vision.scan_thermal(frame)
    defaults = inspect.signature(vision.scan_thermal).parameters
    lo, hi = defaults["human_band_c"].default
    hot_min = defaults["hot_min_c"].default
    valid = frame.thermal > 0
    c = frame.thermal[valid].astype(float) / 10 - 273.15
    summary = {"max_c": round(float(c.max()), 2) if c.size else None,
               "median_c": round(float(np.median(c)), 2) if c.size else None,
               "hot_fraction": round(float((c >= hot_min).mean()), 5) if c.size else None,
               "human_band_fraction": round(float(((c >= lo) & (c <= hi)).mean()), 5) if c.size else None,
               "valid_pixels": int(valid.sum())}
    if rgb is None:
        out["rgb_people"] = None     # no colour image was sent, so nothing was looked for
    return _jsonable({**out, **summary, **meta, "backend": "local"})


def main(argv=None):
    p = argparse.ArgumentParser(description="Smoke Rescue MCP server")
    p.add_argument("--http", action="store_true", help="streamable HTTP instead of stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_HTTP_PORT)
    a = p.parse_args(argv)
    if a.http:
        mcp.run("streamable-http", host=a.host, port=a.port)
    else:
        mcp.run("stdio")


if __name__ == "__main__":
    main()
