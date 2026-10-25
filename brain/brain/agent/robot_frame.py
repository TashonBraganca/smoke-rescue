"""Build a vision.Frame from images a robot sends over the network, rather than from a recording folder.

Used by the HTTP route POST /perceive/people (inline frame) and by brain/mcp_server.py, so both run the same OpenCV 5
code the simulated robots run. Images arrive as PNG or JPEG bytes:

    rgb      8-bit colour image (required for people and visibility)
    depth    16-bit single-channel PNG, depth_scale units per metre (camera.yaml RGBD.DepthMapFactor, 1000 here)
    thermal  16-bit single-channel PNG, kelvin x 10 (sensors.json thermal note)

Intrinsics default to config/sensors.json, scaled to the image size, when the caller gives none. Ranges, floor gaps and
place keyframes also assume this project's camera mast (sensors.json camera.mount: 1.6 m high, 10 deg down); a robot
with a different mount gets correct boxes and bearings but biased ranges. Nothing here reads ground truth.
"""
from __future__ import annotations

import base64
import binascii
import json
from pathlib import Path

import cv2
import numpy as np

SENSORS = Path(__file__).resolve().parents[3] / "config" / "sensors.json"
DEFAULT_DEPTH_SCALE = 1000.0
DEFAULT_DEPTH_SCALE_REASON = ("Depth PNG units per metre when the caller sends none: the recorder writes millimetres "
                              "(camera.yaml RGBD.DepthMapFactor 1000), as do most RGB-D drivers (RealSense, ROS 16UC1).")


def read_bytes(data, *, name: str) -> bytes:
    """Raw image bytes from bytes, a base64 string (optionally a data: URL), or a path to an image file."""
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    if not isinstance(data, str):
        raise ValueError(f"{name}: expected bytes, base64 text or a file path")
    text = data.strip()
    if text.startswith("data:"):
        text = text.split(",", 1)[-1]
    try:
        path = Path(text).expanduser() if len(text) < 4096 else None
        if path is not None and path.is_file():
            return path.read_bytes()
    except OSError:
        pass  # base64 text can look like an over-long file name; fall through to decoding it
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{name}: not an existing file path and not valid base64") from exc


def decode_image(data, *, name: str, flags=cv2.IMREAD_UNCHANGED) -> np.ndarray:
    """PNG or JPEG from anything read_bytes takes."""
    img = cv2.imdecode(np.frombuffer(read_bytes(data, name=name), np.uint8), flags)
    if img is None:
        raise ValueError(f"{name}: not a PNG or JPEG image")
    return img


def _sensors() -> dict:
    return json.loads(SENSORS.read_text())


def _scaled_K(block: dict, width: int, height: int) -> np.ndarray:
    sx, sy = width / float(block["width"]), height / float(block["height"])
    return np.array([[block["fx"] * sx, 0, (block["cx"] + .5) * sx - .5],
                     [0, block["fy"] * sy, (block["cy"] + .5) * sy - .5], [0, 0, 1.]])


def _caller_K(intrinsics) -> np.ndarray:
    try:
        fx, fy, cx, cy = (float(intrinsics[k]) for k in ("fx", "fy", "cx", "cy"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("intrinsics must hold numeric fx, fy, cx, cy in pixels") from exc
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])


def rgb_image(data) -> np.ndarray:
    """RGB-order uint8 HxWx3."""
    img = decode_image(data, name="rgb", flags=cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def depth_image(data, shape, depth_scale: float) -> np.ndarray:
    """float32 metres; zero stays zero (invalid), as in vision.load_frame."""
    d = decode_image(data, name="depth")
    if d.ndim != 2 or d.dtype != np.uint16:
        raise ValueError("depth must be a single-channel 16-bit PNG")
    if d.shape != tuple(shape):
        raise ValueError(f"depth is {d.shape[1]}x{d.shape[0]} but the colour image is {shape[1]}x{shape[0]}")
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale must be a positive number of depth units per metre")
    return d.astype(np.float32) / float(depth_scale)


def thermal_image(data) -> np.ndarray:
    t = decode_image(data, name="thermal")
    if t.ndim != 2 or t.dtype != np.uint16:
        raise ValueError("thermal must be a single-channel 16-bit PNG holding kelvin x 10")
    return t


def build_frame(*, rgb=None, depth=None, thermal=None, intrinsics=None, depth_scale=DEFAULT_DEPTH_SCALE, t=0.0):
    """A vision.Frame plus a note on where its calibration came from.

    Without rgb (a thermal-only call) the colour image is black at the thermal size; without depth every pixel is
    missing (NaN), so ranges come back unmeasured rather than invented. Without thermal it is zero (no reading)."""
    from brain.agent import vision
    sensors = _sensors()
    if rgb is None and thermal is None:
        raise ValueError("send at least an rgb or a thermal image")
    th = thermal_image(thermal) if thermal is not None else None
    if rgb is not None:
        colour = rgb_image(rgb)
    else:
        colour = np.zeros((*th.shape, 3), np.uint8)
    h, w = colour.shape[:2]
    if intrinsics is not None:
        K, source = _caller_K(intrinsics), "caller"
    elif rgb is not None:
        K, source = _scaled_K(sensors["camera"], w, h), "config/sensors.json camera, scaled to the image"
    else:
        K, source = _scaled_K(sensors["thermal"], w, h), "config/sensors.json thermal, scaled to the image"
    depth_m = depth_image(depth, (h, w), float(depth_scale)) if depth is not None else np.full((h, w), np.nan, np.float32)
    if th is not None:
        thermal_K = _scaled_K(sensors["thermal"], th.shape[1], th.shape[0])
        transform = np.eye(4)
        if rgb is not None:
            # Same convention as vision.load_frame: the thermal camera sits mount_right_offset_m to the right.
            transform[0, 3] = -float(sensors["thermal"]["mount_right_offset_m"])
    else:
        th, thermal_K, transform = np.zeros((h, w), np.uint16), K, np.eye(4)
    stamps = {"rgb": float(t), "depth": float(t), "thermal": float(t)}
    frame = vision.Frame(t=float(t), rgb=colour, depth_m=depth_m, thermal=th, K=K, thermal_K=thermal_K,
                         rgb_to_thermal=transform, stamps=stamps, source="network", requested_t=float(t))
    return frame, {"intrinsics_source": source, "rgb_size": [w, h], "depth": depth is not None,
                   "thermal": thermal is not None}


def frame_from_request(req: dict):
    """The JSON body of POST /perceive/people with an inline frame: rgb_b64, optional depth_b64, depth_scale,
    intrinsics {fx, fy, cx, cy} and t. File paths are refused here, so a remote caller cannot read server files."""
    def b64(key):
        value = req.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{key} must be base64 text")
        try:
            return base64.b64decode(value.split(",", 1)[-1] if value.startswith("data:") else value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"{key} is not valid base64") from exc
    return build_frame(rgb=b64("rgb_b64"), depth=b64("depth_b64"), intrinsics=req.get("intrinsics"),
                       depth_scale=float(req.get("depth_scale", DEFAULT_DEPTH_SCALE)), t=float(req.get("t", 0.0)))
