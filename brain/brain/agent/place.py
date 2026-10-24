"""Live visual place recognition from one robot's own camera frames.

A keyframe is a set of ORB features that have a registered depth. Each feature is stored with its 3D position in the
robot body frame (x right, y forward, height up, metres), the same frame OccupancyGrid.BackProject uses. A keyframe
is radioed to the partner as a compact binary payload (see encode_payload). The partner matches it against its own
keyframes and gets the planar rigid transform between the two body frames.

The service sees only one robot's recorded rgb, depth and camera.yaml plus the pose the caller passes. It never reads
ground truth. Each robot's database lives in a crash-isolated worker process, like RobotMap3D: a native fault loses
that robot's keyframes and fails its request, and the next keyframe call starts an empty database.

Body frame of a match result: (dx_m, dy_m, dyaw_deg) is the partner keyframe's body pose in the matched own
keyframe's body frame, so a partner point p maps to mine as R(dyaw) p + (dx, dy). Yaw is counter-clockwise positive.
"""
from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import json
import math
import multiprocessing
from pathlib import Path
import struct
import threading
import uuid

import cv2
import numpy as np

from brain.agent import vision

CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"

MAGIC = b"VP"
VERSION = 1
DESCRIPTOR_BYTES = 32
HEADER_BYTES = 5
RECORD_BYTES = DESCRIPTOR_BYTES + 6
_RECORD = np.dtype([("desc", "u1", (DESCRIPTOR_BYTES,)), ("xyh", "<i2", (3,))])
assert _RECORD.itemsize == RECORD_BYTES

REQUIRED_KEYS = (
    "place_features_per_keyframe", "place_detect_features", "place_orb_scale_factor", "place_orb_levels",
    "place_min_features", "place_max_depth_m",
    "place_ratio_test", "place_ransac_threshold_m", "place_height_tolerance_m", "place_min_inliers",
    "place_max_rmse_m", "place_keyframes_kept", "place_shortlist", "place_ransac_iterations",
    "place_ransac_seed", "place_ransac_min_baseline_m", "place_refine_iterations", "place_worker_timeout_s",
)


def load_settings() -> dict:
    """Place keys from config/perception.json plus the camera mount and depth range from config/sensors.json."""
    perception = json.loads((CONFIG_DIR / "perception.json").read_text())
    missing = [k for k in REQUIRED_KEYS if k not in perception]
    if missing:
        raise KeyError(f"config/perception.json is missing {missing}")
    sensors = json.loads((CONFIG_DIR / "sensors.json").read_text())
    settings = {k: perception[k] for k in REQUIRED_KEYS}
    settings["mount"] = dict(sensors["camera"]["mount"])
    settings["depth_min_m"] = float(sensors["depth"]["min_m"])
    settings["depth_max_m"] = float(sensors["depth"]["max_m"])
    return settings


@dataclass(frozen=True)
class Keyframe:
    """Descriptors (n, 32) uint8 and body-frame positions (n, 3) float64 metres: x right, y forward, height up."""
    desc: np.ndarray
    xyh: np.ndarray

    def __len__(self):
        return len(self.desc)


def encode_payload(desc, xyh_m) -> bytes:
    """b'VP', uint8 version, uint16 count, then count records of 32 descriptor bytes and int16 X, Y, H in cm (LE)."""
    desc = np.ascontiguousarray(desc, dtype=np.uint8)
    xyh_m = np.asarray(xyh_m, dtype=np.float64)
    if desc.ndim != 2 or desc.shape[1] != DESCRIPTOR_BYTES or xyh_m.shape != (len(desc), 3):
        raise ValueError("expected (n, 32) uint8 descriptors and (n, 3) positions")
    if len(desc) > 0xFFFF:
        raise ValueError("a payload holds at most 65535 features")
    if not np.isfinite(xyh_m).all():
        raise ValueError("positions must be finite")
    cm = np.rint(xyh_m * 100)
    if (np.abs(cm) > np.iinfo(np.int16).max).any():
        raise ValueError("a position is outside the int16 centimetre range")
    records = np.empty(len(desc), dtype=_RECORD)
    records["desc"] = desc
    records["xyh"] = cm.astype("<i2")
    return MAGIC + struct.pack("<BH", VERSION, len(desc)) + records.tobytes()


def decode_payload(data: bytes) -> Keyframe:
    data = bytes(data)
    if len(data) < HEADER_BYTES or data[:2] != MAGIC:
        raise ValueError("payload does not start with the VP magic")
    version, count = struct.unpack("<BH", data[2:HEADER_BYTES])
    if version != VERSION:
        raise ValueError(f"unsupported payload version {version}")
    if len(data) != HEADER_BYTES + count * RECORD_BYTES:
        raise ValueError(f"payload length {len(data)} does not match {count} records")
    records = np.frombuffer(data, dtype=_RECORD, count=count, offset=HEADER_BYTES)
    return Keyframe(records["desc"].copy(), records["xyh"].astype(np.float64) / 100.0)


def encode_b64(desc, xyh_m) -> str:
    return base64.b64encode(encode_payload(desc, xyh_m)).decode("ascii")


def decode_b64(payload_b64: str) -> Keyframe:
    try:
        raw = base64.b64decode(payload_b64, validate=True)
    except (binascii.Error, ValueError, TypeError) as exc:
        raise ValueError(f"payload is not valid base64: {exc}") from exc
    return decode_payload(raw)


def body_points(u, v, z, camera, mount):
    """Camera optical pixels and depth to body frame, as OccupancyGrid.BackProject does it.

    X = ox, Y = forward_offset + oz cos(p) - oy sin(p), H = height - oz sin(p) - oy cos(p), p = pitch down."""
    fx, fy, cx, cy = camera
    p = math.radians(mount["pitch_down_deg"])
    ox = (u - cx) / fx * z
    oy = (v - cy) / fy * z
    return np.column_stack((ox, mount["forward_offset_m"] + z * math.cos(p) - oy * math.sin(p),
                            mount["height_m"] - z * math.sin(p) - oy * math.cos(p)))


def extract_features(rgb, depth_m, K, settings):
    """ORB on the grey colour image. Keep features with valid depth within place_max_depth_m, strongest first.

    Returns descriptors (n, 32) uint8 and unquantised body-frame positions (n, 3), n <= place_features_per_keyframe."""
    gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=int(settings["place_detect_features"]),
                         scaleFactor=float(settings["place_orb_scale_factor"]), nlevels=int(settings["place_orb_levels"]))
    keypoints, desc = orb.detectAndCompute(gray, None)
    empty = np.zeros((0, DESCRIPTOR_BYTES), np.uint8), np.zeros((0, 3))
    if desc is None or not len(keypoints):
        return empty
    pts = np.array([k.pt for k in keypoints], dtype=np.float64)
    response = np.array([k.response for k in keypoints], dtype=np.float64)
    h, w = depth_m.shape
    u = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
    v = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
    z = depth_m[v, u].astype(np.float64)
    lo = settings["depth_min_m"]
    hi = min(settings["place_max_depth_m"], settings["depth_max_m"])
    usable = np.isfinite(z) & (z >= lo) & (z <= hi)
    index = np.flatnonzero(usable)
    # Stable sort on negative response keeps the order reproducible when responses tie.
    index = index[np.argsort(-response[index], kind="stable")][:int(settings["place_features_per_keyframe"])]
    if not len(index):
        return empty
    xyh = body_points(pts[index, 0], pts[index, 1], z[index], (K[0, 0], K[1, 1], K[0, 2], K[1, 2]), settings["mount"])
    return desc[index], xyh


def make_keyframe(rgb, depth_m, K, settings):
    """Features quantised through the payload codec, so own storage and a partner's copy hold identical values."""
    desc, xyh = extract_features(rgb, depth_m, K, settings)
    if len(desc) < int(settings["place_min_features"]):
        return None, len(desc)
    return decode_payload(encode_payload(desc, xyh)), len(desc)


def fit_rigid(p, q):
    """Least-squares rotation and translation mapping p to q, both (n, 2). Returns angle (rad) and translation."""
    pc, qc = p.mean(axis=0), q.mean(axis=0)
    a, b = p - pc, q - qc
    angle = math.atan2(float((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum()),
                       float((a * b).sum()))
    c, s = math.cos(angle), math.sin(angle)
    return angle, qc - np.array([c * pc[0] - s * pc[1], s * pc[0] + c * pc[1]])


def _apply(angle, t, p):
    c, s = math.cos(angle), math.sin(angle)
    return np.column_stack((c * p[:, 0] - s * p[:, 1] + t[0], s * p[:, 0] + c * p[:, 1] + t[1]))


def ransac_rigid(p, q, settings):
    """Seeded RANSAC for a 2D rigid transform p -> q. Two matches fix a hypothesis; a match is an inlier when its
    planar residual is within place_ransac_threshold_m. Returns (angle, t, inlier_mask, rmse) or None."""
    n = len(p)
    thr = settings["place_ransac_threshold_m"]
    iterations = int(settings["place_ransac_iterations"])
    if n < 2:
        return None
    if n * (n - 1) // 2 <= iterations:
        # Few matches: try every pair, so the result does not depend on the random draw.
        pairs = np.array([(a, b) for a in range(n) for b in range(a + 1, n)])
    else:
        pairs = np.random.default_rng(int(settings["place_ransac_seed"])).integers(0, n, size=(iterations, 2))
    i, j = pairs[:, 0], pairs[:, 1]
    dp, dq = p[j] - p[i], q[j] - q[i]
    lp, lq = np.hypot(dp[:, 0], dp[:, 1]), np.hypot(dq[:, 0], dq[:, 1])
    # A rigid transform keeps lengths: matches whose separations differ by more than two thresholds cannot both fit.
    keep = (i != j) & (lp >= settings["place_ransac_min_baseline_m"]) & (lq >= settings["place_ransac_min_baseline_m"]) \
        & (np.abs(lp - lq) <= 2 * thr)
    if not keep.any():
        return None
    i, j, dp, dq = i[keep], j[keep], dp[keep], dq[keep]
    angle = np.arctan2(dp[:, 0] * dq[:, 1] - dp[:, 1] * dq[:, 0], (dp * dq).sum(axis=1))
    c, s = np.cos(angle), np.sin(angle)
    mid_p, mid_q = (p[i] + p[j]) / 2, (q[i] + q[j]) / 2
    tx = mid_q[:, 0] - (c * mid_p[:, 0] - s * mid_p[:, 1])
    ty = mid_q[:, 1] - (s * mid_p[:, 0] + c * mid_p[:, 1])
    px, py = p[:, 0][None, :], p[:, 1][None, :]
    rx = c[:, None] * px - s[:, None] * py + tx[:, None] - q[:, 0][None, :]
    ry = s[:, None] * px + c[:, None] * py + ty[:, None] - q[:, 1][None, :]
    residual = np.hypot(rx, ry)
    inlier = residual <= thr
    # Most inliers wins; the summed inlier residual breaks ties so equal counts pick the tighter fit.
    score = inlier.sum(axis=1) - np.where(inlier, residual, 0).sum(axis=1) / (thr * n + 1)
    best = int(np.argmax(score))
    mask = inlier[best]
    a, t = float(angle[best]), np.array([tx[best], ty[best]])
    for _ in range(int(settings["place_refine_iterations"])):
        if mask.sum() < 2:
            break
        a, t = fit_rigid(p[mask], q[mask])
        new = np.hypot(*(_apply(a, t, p) - q).T) <= thr
        if np.array_equal(new, mask):
            break
        mask = new
    if mask.sum() < 2:
        return None
    a, t = fit_rigid(p[mask], q[mask])
    err = np.hypot(*(_apply(a, t, p) - q).T)
    mask = err <= thr
    if not mask.any():
        return None
    return a, t, mask, float(np.sqrt(np.mean(err[mask] ** 2)))


def good_matches(partner_desc, own_desc, ratio):
    """Index pairs (partner, own) that pass Lowe's ratio test, from knnMatch with k=2 on Hamming distance.

    When several partner features pick the same own feature (repeated stripes, bricks or tiles do this), only the
    closest one is kept, so a match set is one-to-one."""
    if len(partner_desc) < 1 or len(own_desc) < 2:
        return np.zeros((0, 2), int)
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(partner_desc, own_desc, k=2)
    kept = [(m[0].queryIdx, m[0].trainIdx, m[0].distance) for m in knn
            if len(m) == 2 and m[0].distance < ratio * m[1].distance]
    if not kept:
        return np.zeros((0, 2), int)
    kept = np.array(kept)
    order = np.lexsort((kept[:, 2], kept[:, 1]))
    kept = kept[order]
    first = np.r_[True, kept[1:, 1] != kept[:-1, 1]]
    return kept[first][:, :2].astype(int)


def match_keyframes(partner: Keyframe, own, settings, candidates=None) -> dict:
    """Match a partner keyframe against own keyframes, a list of (id, Keyframe). See the module docstring.

    candidates, when given, are the own keyframe ids the caller's pose prior allows (the brain's own views near where
    the partner stood, or its own earlier views of the place it stands in now); only those are compared."""
    if candidates is not None:
        allowed = {int(c) for c in candidates}
        own = [(i, kf) for i, kf in own if i in allowed]
        if not own:
            return {"matched": False, "reason": "no own keyframe near the pose prior"}
    if not own:
        return {"matched": False, "reason": "no own keyframes stored"}
    candidates = []
    for mine_id, kf in own:
        pairs = good_matches(partner.desc, kf.desc, settings["place_ratio_test"])
        candidates.append((len(pairs), mine_id, kf, pairs))
    candidates.sort(key=lambda c: (-c[0], c[1]))
    best = None
    seen = []
    for count, mine_id, kf, pairs in candidates[:int(settings["place_shortlist"])]:
        if count < 2:
            continue
        # The height of a point does not change under a planar transform, so a pair whose heights differ cannot be an inlier.
        dh = np.abs(partner.xyh[pairs[:, 0], 2] - kf.xyh[pairs[:, 1], 2])
        pairs = pairs[dh <= settings["place_height_tolerance_m"]]
        fit = ransac_rigid(partner.xyh[pairs[:, 0], :2], kf.xyh[pairs[:, 1], :2], settings) if len(pairs) >= 2 else None
        if fit is None:
            continue
        angle, t, mask, rmse = fit
        row = (int(mask.sum()), rmse, mine_id, angle, t)
        seen.append(row)
        if row[0] >= settings["place_min_inliers"] and rmse <= settings["place_max_rmse_m"]:
            if best is None or (row[0], -row[1]) > (best[0], -best[1]):
                best = row
    if best is None:
        top = max(seen, key=lambda r: (r[0], -r[1])) if seen else None
        why = "no shortlisted keyframe had two height-consistent matches" if top is None else \
            f"best candidate had {top[0]} inliers, rmse {top[1]:.3f} m"
        return {"matched": False, "reason": f"{why}; need {settings['place_min_inliers']} inliers "
                                            f"and rmse <= {settings['place_max_rmse_m']} m"}
    inliers, rmse, mine_id, angle, t = best
    yaw = math.degrees(angle)
    yaw = (yaw + 180.0) % 360.0 - 180.0
    return {"matched": True, "mine_id": int(mine_id), "dx_m": float(t[0]), "dy_m": float(t[1]),
            "dyaw_deg": float(yaw), "inliers": inliers, "rmse_m": float(rmse)}


class PlaceDatabase:
    """One robot's keyframes for one run. A new run (the recording folder's parent changes, or t goes backwards) clears it."""

    def __init__(self, settings=None):
        self.settings = settings if settings is not None else load_settings()
        self.keyframes: list[tuple[int, Keyframe]] = []
        self.next_id = 0
        self.database_id = uuid.uuid4().hex
        self.parent = None
        self.last_t = -math.inf

    def reset(self):
        self.keyframes, self.next_id = [], 0
        self.database_id = uuid.uuid4().hex

    def _check_run(self, robot_dir, t):
        t = float(t)
        if not math.isfinite(t):
            raise ValueError("t must be finite")
        parent = Path(robot_dir).resolve().parent
        if parent != self.parent or t < self.last_t:
            self.reset()
        self.parent, self.last_t = parent, t

    def keyframe(self, robot_dir, t) -> dict:
        self._check_run(robot_dir, t)
        frame = vision.load_frame(str(robot_dir), float(t))
        kf, usable = make_keyframe(frame.rgb, frame.depth_m, frame.K, self.settings)
        if kf is None:
            return {"ok": False, "database_id": self.database_id, "reason": f"{usable} usable features, need {self.settings['place_min_features']}"}
        ident = self.next_id
        self.next_id += 1
        self.keyframes.append((ident, kf))
        del self.keyframes[:max(0, len(self.keyframes) - int(self.settings["place_keyframes_kept"]))]
        return {"ok": True, "id": ident, "database_id": self.database_id, "t": float(frame["stamps"]["rgb"]), "features": len(kf),
                "payload_b64": encode_b64(kf.desc, kf.xyh)}

    def match(self, payload_b64, robot_dir=None, t=None, candidates=None) -> dict:
        if robot_dir is not None and t is not None:
            self._check_run(robot_dir, t)
        try:
            partner = decode_b64(payload_b64)
        except ValueError as exc:
            return {"matched": False, "database_id": self.database_id, "reason": f"invalid payload: {exc}"}
        result = match_keyframes(partner, self.keyframes, self.settings, candidates)
        result["database_id"] = self.database_id
        return result


def _worker(connection, settings):
    database = PlaceDatabase(settings)
    try:
        while True:
            command, args = connection.recv()
            if command == "close":
                break
            try:
                connection.send(("ok", getattr(database, command)(*args)))
            except (ValueError, KeyError, TypeError, OSError, cv2.error) as exc:
                connection.send(("error", str(exc)))
    except EOFError:
        pass
    finally:
        connection.close()


class RobotPlace:
    """A serialized per-robot connection to a crash-isolated PlaceDatabase."""

    def __init__(self, settings=None):
        self.settings = settings if settings is not None else load_settings()
        self._lock = threading.Lock()
        self._process = None
        self._connection = None

    def _start(self):
        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_worker, args=(child, self.settings), daemon=True)
        self._process.start()
        child.close()

    def _discard(self):
        if self._connection is not None:
            self._connection.close()
        if self._process is not None:
            if self._process.is_alive():
                self._process.kill()
            self._process.join()
        self._process, self._connection = None, None

    def _call(self, command, *args):
        with self._lock:
            if self._process is None:
                self._start()
            elif command == "keyframe" and not self._process.is_alive():
                self._discard()
                self._start()
            try:
                self._connection.send((command, args))
                if not self._connection.poll(self.settings["place_worker_timeout_s"]):
                    raise RuntimeError("place worker timed out; keyframes lost")
                status, result = self._connection.recv()
            except (BrokenPipeError, EOFError, OSError, RuntimeError) as exc:
                self._discard()
                raise RuntimeError(f"place worker failed; keyframes lost: {exc}") from exc
            if status != "ok":
                raise ValueError(result)
            return result

    def keyframe(self, robot_dir, t):
        return self._call("keyframe", str(robot_dir), float(t))

    def match(self, payload_b64, robot_dir=None, t=None, candidates=None):
        return self._call("match", payload_b64, None if robot_dir is None else str(robot_dir),
                          None if t is None else float(t), None if candidates is None else [int(c) for c in candidates])

    def close(self):
        with self._lock:
            self._discard()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


_places: dict[str, RobotPlace] = {}
_places_lock = threading.Lock()


def _robot(robot):
    with _places_lock:
        if robot not in _places:
            _places[robot] = RobotPlace()
        return _places[robot]


def keyframe(robot, robot_dir, t) -> dict:
    """Store an own RGB-D view. Success includes its image stamp, database_id, service id and binary radio payload."""
    return _robot(robot).keyframe(robot_dir, t)


def match(robot, payload_b64, robot_dir=None, t=None, candidates=None) -> dict:
    """Match a payload (a partner's, or this robot's own newest view for a loop closure) against this robot's keyframes,
    optionally only the candidate ids. robot_dir and t drive the run reset rule."""
    return _robot(robot).match(payload_b64, robot_dir, t, candidates)


def close_all():
    with _places_lock:
        for place in _places.values():
            place.close()
        _places.clear()
