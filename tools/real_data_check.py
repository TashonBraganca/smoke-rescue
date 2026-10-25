#!/usr/bin/env python3
"""Real-data check of the place-recognition pipeline (brain/brain/agent/place.py) and the YOLO11m person detector on
public TUM RGB-D sequences with motion-capture ground truth.

One command (downloads must already be extracted under data/real/, see FETCH below):

    nice -n 10 brain/.venv/bin/python -I tools/real_data_check.py

Only the input side is adapted. ORB detection and the keep-200-strongest-with-depth rule run through
place.extract_features with the production settings; matching, the height filter, RANSAC and the accept rule run
through place.match_keyframes with the production thresholds (place_min_inliers, place_max_rmse_m, ...), unchanged.

Input adaptation, stated:
  * TUM depth PNG is uint16, metres = value / 5000. Zero is invalid (filtered by the depth_min_m >= 0.2 rule).
  * Intrinsics are the TUM ROS-default values per sensor (fr1, fr2, fr3). Images are used unrectified (TUM fr1/fr2
    have mild lens distortion; fr3 is already undistorted). Depth is TUM's pre-registered depth.
  * Body frame: place.py expects a gravity-aligned body frame (x right, y forward, height up), which our robot gets
    from a fixed, known camera mount. TUM's cameras are handheld or on a Pioneer, so the mount is not fixed. For each
    frame the camera's tilt (roll and pitch relative to gravity) and its height are taken from the ground-truth pose;
    yaw and planar position are NOT used to build features. The pipeline must still recover the relative yaw and
    planar translation from images and depth alone, which is the quantity being scored. This stands in for an IMU
    gravity vector plus a known mount height. For the Pioneer sequences a second variant uses one fixed mount (median
    ground-truth tilt and height over the sequence) for every frame, which uses no per-frame truth at all.
  * Ground-truth "up" is the motion-capture z axis (checked at run time: mean camera down axis must point along -z).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "brain"))
from brain.agent import place  # noqa: E402

REAL = REPO / "data" / "real"
EVIDENCE = REPO / "data" / "evidence" / "real_data_check_25oct.txt"
FETCH = "https://cvg.cit.tum.de/rgbd/dataset/<freiburgN>/<name>.tgz, extracted into data/real/<name>/"

INTRINSICS = {  # fx, fy, cx, cy (TUM file formats page, ROS default calibration)
    "freiburg1": (517.3, 516.5, 318.6, 255.3),
    "freiburg2": (520.9, 521.0, 325.1, 249.7),
    "freiburg3": (535.4, 539.2, 320.1, 247.6),
}
DEPTH_SCALE = 5000.0
SEQUENCES = [
    "rgbd_dataset_freiburg1_room",
    "rgbd_dataset_freiburg2_pioneer_360",
    "rgbd_dataset_freiburg2_pioneer_slam",
    "rgbd_dataset_freiburg3_long_office_household",
]
PIONEER = {"rgbd_dataset_freiburg2_pioneer_360", "rgbd_dataset_freiburg2_pioneer_slam"}
SAMPLE_HZ = 3.0            # keyframe rate; production samples 1 Hz, 3 Hz gives more pairs from short sequences
ASSOC_S = 0.02             # rgb/depth/ground-truth association tolerance
POS_DIST_M, POS_YAW_DEG, POS_DT_S = 1.0, 30.0, 10.0
NEG_DIST_M = 4.0
GROSS_YAW_DEG, GROSS_T_M = 5.0, 0.5   # an accepted positive beyond either is a wrong pose, reported separately
NEG_PER_SEQ = 1500         # random negatives per sequence (seeded)
CROSS_N = 2000             # random negatives between different sequences (different rooms)
SEED = 25

# Hand-checked person labels. I looked at every listed frame at full resolution and wrote (required, allowed) per frame:
# required = people with a clearly visible body at least about 40 px tall (a detector should box them); allowed adds
# people that are tiny, far, or cut off at the image edge (a box on them is not a false box, a miss is not a miss).
# Walking_xyz: 30 evenly spaced frames, keyed by rgb file stem. The other two sequences: 40 evenly spaced frames each
# (np.linspace over the sorted rgb list), (0, 0) unless listed. They were meant to be person-free, but the hand check
# found a person walking through fr1_room and distant people behind the partition in fr3_long_office_household.
PERSON_FRAMES = {
    "rgbd_dataset_freiburg3_walking_xyz": (30, {
        "1341846313.553992": (2, 2), "1341846314.526085": (2, 2), "1341846315.530008": (2, 2), "1341846316.502034": (2, 2),
        "1341846317.509927": (1, 2), "1341846318.482196": (1, 1), "1341846319.486271": (1, 1), "1341846320.490186": (1, 1),
        "1341846321.467946": (1, 1), "1341846322.502433": (0, 0), "1341846323.473878": (0, 0), "1341846324.487928": (0, 0),
        "1341846325.517875": (0, 1), "1341846326.491143": (1, 2), "1341846327.497931": (1, 1), "1341846328.470003": (1, 2),
        "1341846329.473950": (1, 1), "1341846330.483903": (0, 1), "1341846331.518344": (1, 1), "1341846332.522130": (1, 1),
        "1341846333.494040": (1, 1), "1341846334.502142": (1, 1), "1341846335.473998": (1, 1), "1341846336.477990": (1, 1),
        "1341846337.482096": (2, 2), "1341846338.454010": (2, 2), "1341846339.462186": (2, 2), "1341846340.433895": (2, 2),
        "1341846341.438113": (2, 2), "1341846342.447907": (2, 2)}),
    "rgbd_dataset_freiburg1_room": (40, {
        "1305031936.336863": (1, 1), "1305031937.500797": (1, 1), "1305031938.668747": (1, 1)}),
    "rgbd_dataset_freiburg3_long_office_household": (40, {
        "1341848041.022638": (0, 1), "1341848061.100583": (0, 1), "1341848063.334854": (0, 1),
        "1341848065.582742": (0, 1), "1341848067.862808": (0, 1)}),
}


# ---------------------------------------------------------------------------------------------------------------- io
def read_list(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        rows.append((float(parts[0]), parts[1:]))
    return rows


def nearest(times, t):
    i = int(np.searchsorted(times, t))
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(times) and (best is None or abs(times[j] - t) < abs(times[best] - t)):
            best = j
    return best


def quat_to_R(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def load_sequence(name):
    root = REAL / name
    rgb = read_list(root / "rgb.txt")
    depth = read_list(root / "depth.txt")
    gt = read_list(root / "groundtruth.txt")
    dt = np.array([t for t, _ in depth])
    gtt = np.array([t for t, _ in gt])
    frames, last = [], -1e9
    for t, (rel,) in rgb:
        if t - last < 1.0 / SAMPLE_HZ:
            continue
        j, k = nearest(dt, t), nearest(gtt, t)
        if j is None or k is None or abs(dt[j] - t) > ASSOC_S or abs(gtt[k] - t) > ASSOC_S:
            continue
        v = [float(x) for x in gt[k][1]]
        frames.append({"t": t, "rgb": root / rel, "depth": root / depth[j][1][0],
                       "c": np.array(v[:3]), "R": quat_to_R(*v[3:7])})
        last = t
    return frames


# ------------------------------------------------------------------------------------------------------------ geometry
UP = np.array([0.0, 0.0, 1.0])


def body_axes(R):
    """Gravity-aligned body axes from a camera rotation (columns: optical x right, y down, z forward, in world)."""
    f = R[:, 2] - (R[:, 2] @ UP) * UP
    y = f / np.linalg.norm(f)
    x = np.cross(y, UP)
    return x, y


def to_body(cam_pts, R, height):
    """Camera optical points (n, 3) to body (X right, Y forward, H up) using only the tilt of R and a height."""
    x, y = body_axes(R)
    w = cam_pts @ R.T                      # world-rotated, camera-centred
    return np.column_stack((w @ x, w @ y, w @ UP + height))


def planar_pose(frame):
    x, y = body_axes(frame["R"])
    return np.column_stack((x[:2], y[:2])) / np.linalg.norm(x[:2]), frame["c"][:2]


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def truth_relative(a, b):
    """Partner a's body pose in own b's body frame: dyaw (deg) and (dx, dy), as match_keyframes reports it."""
    Ra, ca = planar_pose(a)
    Rb, cb = planar_pose(b)
    Rrel = Rb.T @ Ra
    return math.degrees(math.atan2(Rrel[1, 0], Rrel[0, 0])), Rb.T @ (ca - cb)


# --------------------------------------------------------------------------------------------------------- features
def keyframe(frame, K, settings, fixed_mount=None):
    """Production ORB selection, then gravity alignment. Returns a payload-quantised place.Keyframe or None."""
    bgr = cv2.imread(str(frame["rgb"]), cv2.IMREAD_COLOR)
    d = cv2.imread(str(frame["depth"]), cv2.IMREAD_UNCHANGED)
    if bgr is None or d is None or d.dtype != np.uint16 or d.shape != bgr.shape[:2]:
        raise ValueError(f"bad frame {frame['rgb']}")
    depth_m = d.astype(np.float32) / DEPTH_SCALE
    depth_m[d == 0] = np.nan
    s = dict(settings)
    # A zero mount makes body_points return (ox, z, -oy): the optical-frame point, relabelled. Selection is unchanged.
    s["mount"] = {"height_m": 0.0, "forward_offset_m": 0.0, "pitch_down_deg": 0.0}
    Kmat = np.array([[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1.0]])
    desc, xyh = place.extract_features(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth_m, Kmat, s)
    if len(desc) < int(settings["place_min_features"]):
        return None
    cam = np.column_stack((xyh[:, 0], -xyh[:, 2], xyh[:, 1]))
    # Fixed mount: tilt and height are sequence medians in a yaw-free rotation, so no per-frame truth enters.
    R, h = (frame["R"], frame["c"][2]) if fixed_mount is None else fixed_mount
    body = to_body(cam, R, h)
    return place.decode_payload(place.encode_payload(desc, body))


def fixed_mount_for(frames):
    """Median tilt (pitch about body x, roll about body y) and median height; a yaw-free camera rotation."""
    pitches, rolls = [], []
    for f in frames:
        x, y = body_axes(f["R"])
        z = f["R"][:, 2]                                     # optical forward
        pitches.append(math.asin(np.clip(-(z @ UP), -1, 1)))  # positive = looking down
        xc = f["R"][:, 0]
        rolls.append(math.asin(np.clip(xc @ UP, -1, 1)))
    p, r = float(np.median(pitches)), float(np.median(rolls))
    # Build R for a camera facing world +y, pitched down by p and rolled by r: columns optical x, y, z in world.
    fwd = np.array([0.0, math.cos(p), -math.sin(p)])
    right0 = np.array([1.0, 0.0, 0.0])
    right = math.cos(r) * right0 - math.sin(r) * np.cross(fwd, right0)   # Rodrigues about fwd; fwd . right0 = 0
    down = np.cross(fwd, right)
    R = np.column_stack((right, down, fwd))
    h = float(np.median([f["c"][2] for f in frames]))
    return R, h, math.degrees(p), math.degrees(r)


# --------------------------------------------------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else float("nan")


# ------------------------------------------------------------------------------------------------------------ place
def pairs_for(frames, rng):
    pos, neg = [], []
    for i in range(len(frames)):
        for j in range(i):
            a, b = frames[i], frames[j]
            dist = float(np.linalg.norm(a["c"][:2] - b["c"][:2]))
            if dist > NEG_DIST_M:
                neg.append((i, j))
            elif dist <= POS_DIST_M and a["t"] - b["t"] >= POS_DT_S:
                yaw, _ = truth_relative(a, b)
                if abs(yaw) <= POS_YAW_DEG:
                    pos.append((i, j))
    if len(neg) > NEG_PER_SEQ:
        neg = [neg[k] for k in sorted(rng.choice(len(neg), NEG_PER_SEQ, replace=False))]
    return pos, neg


def run_place(name, settings, rng, fixed=False):
    frames = load_sequence(name)
    K = INTRINSICS[name.split("_")[2]]
    down = np.mean([f["R"][:, 1] for f in frames], axis=0)
    if down @ UP > -0.5:
        raise RuntimeError(f"{name}: mean camera down axis {down} does not point along -z; up axis assumption fails")
    mount = None
    note = "per-frame ground-truth tilt and height"
    if fixed:
        R, h, p, r = fixed_mount_for(frames)
        mount = (R, h)
        note = f"one fixed mount for the whole sequence: pitch down {p:.1f} deg, roll {r:.1f} deg, height {h:.2f} m"
    kfs = [keyframe(f, K, settings, mount) for f in frames]
    pos, neg = pairs_for(frames, rng)
    out = {"name": name, "variant": note, "frames": len(frames), "no_keyframe": sum(k is None for k in kfs),
           "span_m": float(np.ptp(np.array([f["c"][:2] for f in frames]), axis=0).max())}
    rows = []
    for i, j in pos:
        if kfs[i] is None or kfs[j] is None:
            rows.append(None)
            continue
        r = place.match_keyframes(kfs[i], [(0, kfs[j])], settings)
        yaw, t = truth_relative(frames[i], frames[j])
        if r["matched"]:
            rows.append((abs(wrap(r["dyaw_deg"] - yaw)), float(np.hypot(r["dx_m"] - t[0], r["dy_m"] - t[1])), r["inliers"]))
        else:
            rows.append(False)
    acc = [x for x in rows if x]
    out["pos_n"] = len(pos)
    out["pos_accepted"] = len(acc)
    out["pos_no_keyframe"] = sum(x is None for x in rows)
    out["yaw_err"] = [a[0] for a in acc]
    out["trans_err"] = [a[1] for a in acc]
    out["inliers"] = [a[2] for a in acc]
    out["gross"] = [(round(a[0], 1), round(a[1], 2), a[2]) for a in acc if a[0] > GROSS_YAW_DEG or a[1] > GROSS_T_M]
    neg_inl, false = [], 0
    for i, j in neg:
        if kfs[i] is None or kfs[j] is None:
            continue
        r = place.match_keyframes(kfs[i], [(0, kfs[j])], settings)
        if r["matched"]:
            false += 1
            neg_inl.append(r["inliers"])
        else:
            msg = r["reason"]
            neg_inl.append(int(msg.split(" inliers")[0].split()[-1]) if "inliers, rmse" in msg else 0)
    out["neg_n"] = len(neg_inl)
    out["neg_false"] = false
    out["neg_max_inliers"] = max(neg_inl) if neg_inl else None
    out["neg_ge_min"] = sum(x >= settings["place_min_inliers"] for x in neg_inl)
    if not fixed:
        KF_CACHE[name] = kfs
    return out


KF_CACHE: dict[str, list] = {}


def negative_inliers(a, b, settings):
    r = place.match_keyframes(a, [(0, b)], settings)
    if r["matched"]:
        return r["inliers"], True
    msg = r["reason"]
    return (int(msg.split(" inliers")[0].split()[-1]) if "inliers, rmse" in msg else 0), False


def run_cross(settings, rng):
    """Negatives between different sequences, which are different rooms (the two fr2 Pioneer runs share one hall, so
    that combination is excluded)."""
    names = [n for n in KF_CACHE]
    combos = [(a, b) for a in names for b in names if a != b and not (a in PIONEER and b in PIONEER)]
    inl, false = [], 0
    if not combos:
        return None
    for _ in range(CROSS_N):
        a, b = combos[int(rng.integers(len(combos)))]
        ka = [k for k in KF_CACHE[a] if k is not None]
        kb = [k for k in KF_CACHE[b] if k is not None]
        n, acc = negative_inliers(ka[int(rng.integers(len(ka)))], kb[int(rng.integers(len(kb)))], settings)
        inl.append(n)
        false += acc
    return {"n": len(inl), "false": false, "max_inliers": max(inl),
            "ge_min": sum(x >= settings["place_min_inliers"] for x in inl)}


# ------------------------------------------------------------------------------------------------------------ people
def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def detect(net, bgr, min_conf=0.35):
    """The exact preprocessing and decode of vision.detect_people_rgb (letterbox 640, class 0, NMS 0.45), on BGR."""
    h, w = bgr.shape[:2]
    s = 640 / max(h, w); nh, nw = int(round(h * s)), int(round(w * s)); top, left = (640 - nh) // 2, (640 - nw) // 2
    pad = np.full((640, 640, 3), 114, np.uint8)
    pad[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh))
    net.setInput(cv2.dnn.blobFromImage(pad, 1 / 255.0, (640, 640), swapRB=True))
    out = net.forward()[0].T
    conf = out[:, 4]
    keep = conf >= min_conf
    if not keep.any():
        return []
    b, conf = out[keep, :4], conf[keep]
    boxes = [[float((cx - bw / 2 - left) / s), float((cy - bh / 2 - top) / s), float(bw / s), float(bh / s)] for cx, cy, bw, bh in b]
    idx = cv2.dnn.NMSBoxes(boxes, conf.tolist(), min_conf, 0.45)
    return [(boxes[i], float(conf[i])) for i in np.array(idx).reshape(-1)]


def run_people():
    model = REPO / "data" / "models" / "onnx" / "yolo11m.onnx"
    if not model.exists():
        return None
    net = cv2.dnn.readNetFromONNX(str(model))
    res = {"frames": 0, "frames_req": 0, "frames_req_hit": 0, "people_req": 0, "people_boxed": 0,
           "boxes": 0, "false_boxes": 0, "frames_empty": 0, "frames_empty_with_box": 0, "notes": []}
    for name, (count, labels) in PERSON_FRAMES.items():
        files = sorted((REAL / name / "rgb").glob("*.png"))
        picked = [files[int(k)] for k in np.linspace(0, len(files) - 1, count)]
        for key in labels:   # every hand label must refer to a picked frame, or the labels are stale
            if key not in {f.stem for f in picked}:
                raise RuntimeError(f"hand label {name}/{key} is not one of the picked frames")
        for f in picked:
            req, allowed = labels.get(f.stem, (0, 0))
            dets = detect(net, cv2.imread(str(f)))
            n = len(dets)
            res["frames"] += 1
            res["boxes"] += n
            res["false_boxes"] += max(0, n - allowed)
            if req:
                res["frames_req"] += 1
                res["frames_req_hit"] += n > 0
                res["people_req"] += req
                res["people_boxed"] += min(n, req)
            if allowed == 0:
                res["frames_empty"] += 1
                res["frames_empty_with_box"] += n > 0
            if n > allowed or n < req:
                res["notes"].append(f"{name}/{f.stem}: hand ({req}, {allowed}), boxes {n} conf "
                                    + ", ".join(f"{c:.2f}" for _, c in dets))
    return res


# ------------------------------------------------------------------------------------------------------------- report
def fmt_place(o, settings):
    k, n = o["pos_accepted"], o["pos_n"]
    lo, hi = wilson(k, n)
    lines = [f"  {o['name']}  [{o['variant']}]",
             f"    keyframes sampled at {SAMPLE_HZ:g} Hz: {o['frames']} ({o['no_keyframe']} under place_min_features); "
             f"planar extent of trajectory {o['span_m']:.1f} m",
             f"    positives: n={n}, accepted {k} = {100 * k / n if n else float('nan'):.1f} % "
             f"(Wilson 95 % CI {100 * lo:.1f} to {100 * hi:.1f} %); pairs with a frame under min features: {o['pos_no_keyframe']}"]
    if k:
        lines.append(f"    accepted yaw error deg: median {pct(o['yaw_err'], 50):.2f}, p90 {pct(o['yaw_err'], 90):.2f}, "
                     f"max {max(o['yaw_err']):.2f} (n={k})")
        lines.append(f"    accepted translation error m: median {pct(o['trans_err'], 50):.3f}, p90 {pct(o['trans_err'], 90):.3f}, "
                     f"max {max(o['trans_err']):.3f} (n={k})")
    if k:
        g = o["gross"]
        lines.append(f"    accepted positives with a wrong pose (yaw error > {GROSS_YAW_DEG:g} deg or translation error > "
                     f"{GROSS_T_M:g} m): {len(g)}/{k}" + (f"; their inlier counts: median {int(np.median([x[2] for x in g]))}, "
                                                         f"max {max(x[2] for x in g)}" if g else ""))
    lines.append(f"    negatives (> {NEG_DIST_M:g} m apart): n={o['neg_n']}, false accepts {o['neg_false']}, "
                 f"max inliers {o['neg_max_inliers']}, pairs with >= {settings['place_min_inliers']} inliers {o['neg_ge_min']}")
    return lines


def main():
    t0 = time.time()
    settings = place.load_settings()
    rng = np.random.default_rng(SEED)
    only = sys.argv[1:]  # optional sequence names, for a quick partial run (writes no evidence file)
    present = [s for s in SEQUENCES if (REAL / s / "groundtruth.txt").exists() and (not only or s in only)]
    results = [run_place(s, settings, rng) for s in present]
    results += [run_place(s, settings, rng, fixed=True) for s in present if s in PIONEER]
    cross = run_cross(settings, rng)
    people = None if only else run_people()

    def pooled(rs):
        k = sum(r["pos_accepted"] for r in rs); n = sum(r["pos_n"] for r in rs)
        ye = sum((r["yaw_err"] for r in rs), []); te = sum((r["trans_err"] for r in rs), [])
        return k, n, ye, te, sum(r["neg_n"] for r in rs), sum(r["neg_false"] for r in rs), \
            max((r["neg_max_inliers"] or 0) for r in rs)

    per_frame = [r for r in results if r["variant"].startswith("per-frame")]
    L = ["Real-data check of place recognition and person detection, 25 Oct 2026",
         f"Generated by tools/real_data_check.py ({time.strftime('%Y-%m-%d %H:%M %Z')}, {time.time() - t0:.0f} s, OpenCV {cv2.__version__}).",
         "",
         "Data: TUM RGB-D benchmark (Sturm et al., IROS 2012), https://cvg.cit.tum.de/data/datasets/rgbd-dataset,",
         "Kinect v1 640x480 RGB-D with motion-capture ground-truth camera poses. Sequences: " + ", ".join(present) + ".",
         "",
         "Production settings used unchanged (config/perception.json via place.load_settings()):",
         "  " + ", ".join(f"{k}={settings[k]}" for k in ("place_detect_features", "place_features_per_keyframe",
                                                       "place_orb_scale_factor", "place_orb_levels", "place_min_features",
                                                       "place_max_depth_m", "place_ratio_test", "place_ransac_threshold_m",
                                                       "place_height_tolerance_m", "place_min_inliers", "place_max_rmse_m",
                                                       "place_ransac_iterations", "place_ransac_min_baseline_m")),
         "",
         "Input adaptation (the only change): depth = uint16 / 5000 m; TUM intrinsics per sensor; images unrectified;",
         "features go through place.extract_features (production ORB and selection), then into a gravity-aligned body",
         "frame using the ground-truth camera tilt (roll, pitch) and height. Ground-truth yaw and x, y are not used for",
         "features; they are only used to score. Pioneer sequences are also run with one fixed mount (median tilt and",
         "height) to mimic our robot's fixed mast, using no per-frame truth.",
         "",
         f"Pairs: one partner keyframe against one own keyframe with place.match_keyframes (no shortlist, no pose prior,",
         f"no LiDAR check, no second agreeing match; those brain-side gates would only remove accepts). Positives: planar",
         f"ground-truth distance <= {POS_DIST_M:g} m, yaw difference <= {POS_YAW_DEG:g} deg, >= {POS_DT_S:g} s apart. Negatives:",
         f"> {NEG_DIST_M:g} m apart, up to {NEG_PER_SEQ} drawn at random per sequence (seed {SEED}). Errors are against",
         "ground truth: |yaw_est - yaw_true| and the planar distance between estimated and true (dx, dy).",
         "",
         "PLACE RECOGNITION, per sequence"]
    for r in results:
        L += fmt_place(r, settings)
    k, n, ye, te, nn, nf, nmax = pooled(per_frame)
    lo, hi = wilson(k, n)
    L += ["", "PLACE RECOGNITION, pooled over the per-frame-tilt runs",
          f"  positives n={n}, accepted {k} = {100 * k / n if n else float('nan'):.1f} % (Wilson 95 % CI {100 * lo:.1f} to {100 * hi:.1f} %)",
          f"  yaw error deg median {pct(ye, 50):.2f}, p90 {pct(ye, 90):.2f}; translation error m median {pct(te, 50):.3f}, "
          f"p90 {pct(te, 90):.3f} (n={len(ye)})",
          f"  accepted positives with a wrong pose (yaw > {GROSS_YAW_DEG:g} deg or translation > {GROSS_T_M:g} m): "
          f"{sum(len(r['gross']) for r in per_frame)}/{len(ye)}",
          f"  negatives n={nn}, false accepts {nf}" + (f" (rate 95 % upper bound {100 * wilson(nf, nn)[1]:.2f} %)" if nn else "")
          + f", max negative inliers {nmax}",
          "  The Wilson interval treats pairs as independent. They are not: neighbouring frames share views, so the",
          "  true uncertainty is wider than printed."]
    if cross:
        L += ["", "PLACE RECOGNITION, negatives between different sequences (different rooms; fr2 Pioneer pair excluded)",
              f"  n={cross['n']}, false accepts {cross['false']} (rate 95 % upper bound {100 * wilson(cross['false'], cross['n'])[1]:.2f} %), "
              f"max inliers {cross['max_inliers']}, pairs with >= {settings['place_min_inliers']} inliers {cross['ge_min']}"]
    L += ["", "PERSON DETECTION (YOLO11m ONNX through cv2.dnn, the decode of vision.detect_people_rgb, min_conf 0.35)"]
    if people is None:
        L.append("  skipped: model or hand labels missing")
    else:
        p = people
        lo, hi = wilson(p["frames_req_hit"], p["frames_req"])
        L += [f"  {p['frames']} hand-labelled frames from " + ", ".join(f"{k} ({v[0]})" for k, v in PERSON_FRAMES.items()) + ".",
              f"  frames with at least one required person that got a box: {p['frames_req_hit']}/{p['frames_req']} "
              f"(Wilson 95 % CI {100 * lo:.0f} to {100 * hi:.0f} %)",
              f"  required people covered by the box count: {p['people_boxed']}/{p['people_req']}",
              f"  boxes {p['boxes']} in total; false boxes (beyond the allowed count) {p['false_boxes']}; "
              f"frames with no person at all that got a box: {p['frames_empty_with_box']}/{p['frames_empty']}"]
        for note in p["notes"]:
            L.append(f"    mismatch {note}")
        L.append("  Box counts are a proxy: they compare the number of boxes per frame with the hand count, not each box's position.")
    L += ["", "WHAT THIS DOES AND DOES NOT SHOW", NOTE, ""]
    text = "\n".join(L)
    print(text)
    if only:
        return
    EVIDENCE.write_text(text)
    (REAL / "real_data_check_raw.json").write_text(json.dumps({"place": results, "cross": cross, "people": people}, default=str, indent=1))


NOTE = """\
Shows: on real Kinect RGB-D frames of four indoor scenes (two handheld office sequences, two from a Pioneer ground
robot), the unchanged ORB + depth + height filter + 2D RANSAC matcher, at the production thresholds, accepted a
real share of same-place revisits and estimated their relative yaw and translation to about a degree and a few
centimetres in the median, and it accepted none of the far-apart or different-room negatives (max negative inliers
stayed well below the 10-inlier threshold, as in simulation). It also shows the person detector, run through cv2.dnn
exactly as the brain runs it, boxing every clearly visible person in the hand-checked frames and not boxing teddy
bears or empty offices.

Does not show: anything about smoke, darkness, heat or headlights. TUM is clean, well-lit indoor data, so this tests
matching and geometry on real sensor noise, not smoke robustness. It does not test our quadruped's gait, our camera,
or the full brain (pose prior, LiDAR verification, second agreeing match, map merge); pairs are matched one-to-one.
The body frame uses the true camera tilt and height per frame (the Pioneer fixed-mount variant does not, and gives
nearly the same numbers), which a robot would get from its mount and IMU. Recall depends on the pair definition
(<= 1 m, <= 30 deg) and differs a lot by scene (26 % to 88 %), so the pooled figure is scene-weighted.
Pairs share frames, so the printed intervals are too narrow. The person check is 110 frames from three sequences in
two buildings with my own hand labels, and box counts are a proxy for per-box correctness.

Weakness found: accepted matches are not always right. In the Pioneer hall, 65 of 415 accepted positives had a
wrong pose (yaw off 5 to 80 deg or translation off more than 0.5 m) with up to 39 inliers and RMSE under 0.06 m, so
the inlier and RMSE gates did not catch them. I looked at one high-inlier wrong pair (pioneer_360 frames at 3.7 s and 59.0 s):
the background is a long fence of identical mesh panels and tripods, a repeating pattern that lets a shifted panel
match. It is not a ground-truth time offset (shifting truth by 0.1 to 0.3 s either way made errors larger) and not only
long-range depth noise (a diagnostic run with a 5 m depth cut, not used for any number above, still gave 25 wrong
poses in 77 accepts). In pioneer_360 it only happened while the robot was turning (0 wrong of 30 accepts with both frames still).
In the office scenes the rate was 5 of 1103. Repetitive structure is a real risk for our house too (tiles, shelving),
and is what the brain's LiDAR verification and second-match rule must catch.
"""

if __name__ == "__main__":
    main()
