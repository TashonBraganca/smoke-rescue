#!/usr/bin/env python3
"""Video segment of the live visual SLAM link in one recording: the two robots' camera frames with the ORB matches that
survived RANSAC, then the partner's map moving from the launch-pad guess onto the camera-measured link.

    brain/.venv/bin/python tools/visual_slam_clip.py data/recordings/<run> --out data/evidence/<name>.mp4 [--robot robot_A]

The matches are recomputed offline with the robot-side code (brain.agent.place) on the frames nearest to where each
robot stood for the place_recognised event that fed frame_link_refined; the link numbers drawn are the ones the robot
logged live. The only evaluator data used is the true launch headings, for the before and after error labels.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "brain")); sys.path.insert(0, str(ROOT / "tools"))
from brain.agent import place, vision  # noqa: E402
import run_audit as audit  # noqa: E402

W, H, FPS = 1280, 720, 15
BG, INK, DIM = (24, 22, 20), (236, 236, 236), (150, 150, 150)
COL = {"robot_A": (0, 169, 242), "robot_B": (230, 170, 40)}   # BGR: A amber, B blue
GOOD, BAD = (90, 220, 90), (60, 60, 200)


def events(rec, robot, kind):
    return [e for e in audit.rows(rec / robot / "decisions.jsonl") if e["kind"] == kind]


def poses(rec, robot):
    rows = [json.loads(l) for l in open(rec / robot / "brain_state.jsonl")]
    return np.array([[r["t"], r["x"], r["y"]] for r in rows])


def times_near(rec, robot, track, xy, until, radius):
    """Camera stamps within half a pose sample of a time the robot's own pose was within radius of xy."""
    d = np.hypot(track[:, 1] - xy[0], track[:, 2] - xy[1])
    near = track[(d <= radius) & (track[:, 0] <= until + 1e-6), 0]
    stamps = np.array([float(l.split()[0]) for l in open(rec / robot / "rgb.txt") if l[0].isdigit()])
    step = float(np.median(np.diff(track[:, 0])))
    return stamps[(np.abs(stamps[:, None] - near[None, :]) <= step / 2 + 1e-6).any(axis=1) & (stamps <= until + 1e-6)]


def features(robot_dir, t, settings):
    """place.extract_features, keeping the image keypoints so the matches can be drawn."""
    f = vision.load_frame(str(robot_dir), float(t))
    gray = cv2.cvtColor(np.ascontiguousarray(f.rgb), cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=int(settings["place_detect_features"]), scaleFactor=float(settings["place_orb_scale_factor"]),
                         nlevels=int(settings["place_orb_levels"]))
    kp, desc = orb.detectAndCompute(gray, None)
    pts = np.array([k.pt for k in kp]); resp = np.array([k.response for k in kp])
    h, w = f.depth_m.shape
    u = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1); v = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
    z = f.depth_m[v, u].astype(float)
    ok = np.isfinite(z) & (z >= settings["depth_min_m"]) & (z <= min(settings["place_max_depth_m"], settings["depth_max_m"]))
    idx = np.flatnonzero(ok); idx = idx[np.argsort(-resp[idx], kind="stable")][:int(settings["place_features_per_keyframe"])]
    K = f.K
    xyh = place.body_points(pts[idx, 0], pts[idx, 1], z[idx], (K[0, 0], K[1, 1], K[0, 2], K[1, 2]), settings["mount"])
    kf = place.decode_payload(place.encode_payload(desc[idx], xyh))
    return cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR), pts[idx], kf


def match(partner, own, settings):
    pairs = place.good_matches(partner.desc, own.desc, settings["place_ratio_test"])
    if len(pairs) < 2:
        return pairs, np.zeros(len(pairs), bool)
    dh = np.abs(partner.xyh[pairs[:, 0], 2] - own.xyh[pairs[:, 1], 2])
    pairs = pairs[dh <= settings["place_height_tolerance_m"]]
    fit = place.ransac_rigid(partner.xyh[pairs[:, 0], :2], own.xyh[pairs[:, 1], :2], settings) if len(pairs) >= 2 else None
    return pairs, (fit[2] if fit else np.zeros(len(pairs), bool))


def text(img, s, org, scale=.6, col=INK, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def last_map(rec, robot, t):
    rows = [l.split() for l in open(rec / robot / "brain_map.txt") if not l.startswith("#")]
    row = min(rows, key=lambda r: abs(float(r[0]) - t))
    return cv2.imread(str(rec / robot / row[4]), 0), float(row[0])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("rec", type=Path); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--robot", default="robot_A"); ap.add_argument("--map-t", type=float, default=150.0)
    a = ap.parse_args()
    settings = place.load_settings(); brain = json.load(open(ROOT / "config" / "brain.json"))
    me = a.robot; other = next(r for r in audit.ROBOTS if r != me)
    refined = events(a.rec, me, "frame_link_refined")[0]; tr = refined["t"]
    recog = [e for e in events(a.rec, me, "place_recognised") if e["t"] <= tr + 1e-6]
    best_ev = max(recog, key=lambda e: e["numbers"]["inliers"])
    n = best_ev["numbers"]
    # Frames where each robot stood (its own pose estimate) for that match; the strongest pair is drawn.
    mine_t = times_near(a.rec, me, poses(a.rec, me), (n["mine_x_m"], n["mine_y_m"]), best_ev["t"], .3)
    theirs_t = times_near(a.rec, other, poses(a.rec, other), (n["theirs_x_m"], n["theirs_y_m"]), best_ev["t"], .3)
    if not len(mine_t) or not len(theirs_t):
        raise SystemExit("no frames near the matched keyframe poses")
    # Robots turn on the spot near these points, so every frame near each one is tried (features computed once each).
    own_f = [(tm,) + features(a.rec / me, tm, settings)[:3] for tm in mine_t]
    th_f = [(tt,) + features(a.rec / other, tt, settings)[:3] for tt in theirs_t]
    best = None
    for tm, own_img, own_pts, own_kf in own_f:
        for tt, th_img, th_pts, th_kf in th_f:
            pairs, mask = match(th_kf, own_kf, settings)
            if best is None or mask.sum() > best[0]:
                best = (int(mask.sum()), tm, tt, own_img, own_pts, th_img, th_pts, pairs, mask, len(own_kf), len(th_kf))
    inl, tm, tt, own_img, own_pts, th_img, th_pts, pairs, mask, n_own, n_th = best
    print(f"frames {me} t={tm:.2f} s, {other} t={tt:.2f} s: {len(pairs)} ratio-test matches, {inl} RANSAC inliers "
          f"(live event: {int(n['inliers'])} inliers)")
    if inl < settings["place_min_inliers"]:
        raise SystemExit(f"offline match has {inl} inliers, below place_min_inliers; not drawing a match the robot would reject")

    heading = {r: math.degrees(audit.truth(a.rec, r)[0][3]) for r in audit.ROBOTS}
    true_link = -(heading[other] - heading[me])
    dyaw = ((refined["numbers"]["dyaw_deg"] + 180) % 360) - 180
    err_before, err_after = abs(true_link), abs(dyaw - true_link)

    # Camera panel: partner frame left, own frame right, both 600 px wide.
    pw = 600; ph = int(own_img.shape[0] * pw / own_img.shape[1])
    sl, sr = pw / th_img.shape[1], pw / own_img.shape[1]
    left = cv2.resize(th_img, (pw, ph)); right = cv2.resize(own_img, (pw, ph))
    x0, y0 = (W - 2 * pw - 20) // 2, 110

    # Map panel: partner map warped into own map pixels by a link (yaw deg, dx, dy). Grid centred on the frame origin.
    own_map, mt = last_map(a.rec, me, a.map_t); th_map, _ = last_map(a.rec, other, a.map_t)
    G = own_map.shape[0]; cell = brain["cell_m"]; half = G * cell / 2

    def warp(yaw_deg, dx, dy):
        th = math.radians(yaw_deg); c, s = math.cos(th), math.sin(th)
        # pixel -> metres: x = -half + (col + .5) cell, y = half - (row + .5) cell; then p_own = R p_other + t.
        P = np.array([[cell, 0, -half + .5 * cell], [0, -cell, half - .5 * cell], [0, 0, 1]])
        T = np.array([[c, -s, dx], [s, c, dy], [0, 0, 1]])
        M = np.linalg.inv(P) @ T @ P
        return cv2.warpAffine(th_map, M[:2], (G, G), flags=cv2.INTER_NEAREST, borderValue=128)

    occ_own = own_map < 64
    dist = cv2.distanceTransform((~occ_own).astype(np.uint8), cv2.DIST_L2, 3)

    def agreement(m):
        o = m < 64
        return float((dist[o] <= 1.0).mean()) if o.any() else 0.0

    link = (dyaw, refined["numbers"]["dx_m"], refined["numbers"]["dy_m"])
    a_prior, a_link = agreement(warp(0, 0, 0)), agreement(warp(*link))
    a_inverse = agreement(warp(-link[0], -link[1], -link[2]))
    print(f"wall agreement (partner walls within 1 cell of own walls): launch pads {a_prior:.3f}, measured link {a_link:.3f}, "
          f"inverse link {a_inverse:.3f}")
    if not a_link > a_prior:
        raise SystemExit("the measured link does not align the maps better than the launch pads: check the transform direction")

    ys, xs = np.where((own_map != 128) | (th_map != 128))
    pad = 15; r0, r1, c0, c1 = max(0, ys.min() - pad), min(G, ys.max() + pad), max(0, xs.min() - pad), min(G, xs.max() + pad)
    box_h = H - 170; scale = min(box_h / (r1 - r0), (W - 80) / (c1 - c0))

    def map_frame(f):
        k = min(1.0, max(0.0, f))
        m = warp(link[0] * k, link[1] * k, link[2] * k)
        img = np.full((r1 - r0, c1 - c0, 3), BG, np.uint8)
        free = (own_map[r0:r1, c0:c1] == 255) | (m[r0:r1, c0:c1] == 255)
        img[free] = (60, 58, 55)
        img[own_map[r0:r1, c0:c1] < 64] = COL[me]
        b = m[r0:r1, c0:c1] < 64
        img[b] = np.where(img[b] == np.array(COL[me]), (255, 255, 255), COL[other])
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        canvas = np.full((H, W, 3), BG, np.uint8)
        oy, ox = 120, (W - img.shape[1]) // 2
        canvas[oy:oy + img.shape[0], ox:ox + img.shape[1]] = img
        text(canvas, "VISUAL SLAM: the shared map snaps into place", (40, 50), .95, INK, 2)
        err = err_before + (err_after - err_before) * k
        text(canvas, f"{other[-1]}'s walls ({'blue'}) placed into {me[-1]}'s map ({'amber'}); white = walls that agree", (40, 85), .6, DIM)
        state = "launch-pad guess" if k == 0 else ("camera-measured link" if k >= 1 else "applying the link")
        text(canvas, f"{state}:  heading error {err:.2f} deg   wall agreement "
                     f"{(a_prior + (a_link - a_prior) * k) * 100:.0f} %", (40, H - 25), .7, GOOD if k >= 1 else INK, 2)
        text(canvas, f"maps at {mt:.0f} s; link measured live at {tr:.1f} s from {int(refined['numbers']['candidates'])} "
                     f"camera matches, each verified on LiDAR", (40, H - 55), .5, DIM)
        return canvas

    def match_frame(f):
        canvas = np.full((H, W, 3), BG, np.uint8)
        canvas[y0:y0 + ph, x0:x0 + pw] = left; canvas[y0:y0 + ph, x0 + pw + 20:x0 + 2 * pw + 20] = right
        text(canvas, "VISUAL SLAM: two robots recognise the same place", (40, 50), .95, INK, 2)
        text(canvas, f"OpenCV 5 ORB features with depth, Hamming ratio test, RANSAC on 3D positions", (40, 85), .6, DIM)
        text(canvas, f"robot {other[-1]} camera, t={tt:.1f} s", (x0, y0 + ph + 28), .6, COL[other], 2)
        text(canvas, f"robot {me[-1]} camera, t={tm:.1f} s", (x0 + pw + 20, y0 + ph + 28), .6, COL[me], 2)
        shown = int(len(pairs) * min(1.0, f))
        for k, ((pi, oi), good) in enumerate(zip(pairs, mask)):
            if k >= shown:
                break
            p = (int(x0 + th_pts[pi][0] * sl), int(y0 + th_pts[pi][1] * sl))
            q = (int(x0 + pw + 20 + own_pts[oi][0] * sr), int(y0 + own_pts[oi][1] * sr))
            cv2.line(canvas, p, q, GOOD if good else BAD, 2 if good else 1, cv2.LINE_AA)
            cv2.circle(canvas, p, 3, GOOD if good else BAD, -1); cv2.circle(canvas, q, 3, GOOD if good else BAD, -1)
        text(canvas, f"{n_th} and {n_own} features sent (under 8 KB each);  {len(pairs)} matches pass the ratio test,  "
                     f"{inl} agree on one rigid motion (green)", (40, H - 40), .6, INK, 1)
        return canvas

    a.out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(int(6 * FPS)):
        vw.write(match_frame(i / (3 * FPS)))
    for i in range(int(2 * FPS)):
        vw.write(map_frame(0))
    for i in range(int(3 * FPS)):
        vw.write(map_frame(i / (3 * FPS - 1)))
    for i in range(int(3 * FPS)):
        vw.write(map_frame(1))
    vw.release()
    cv2.imwrite(str(a.out.with_suffix("")) + "_matches.png", match_frame(1))
    cv2.imwrite(str(a.out.with_suffix("")) + "_before.png", map_frame(0))
    cv2.imwrite(str(a.out.with_suffix("")) + "_after.png", map_frame(1))
    print(f"heading error {err_before:.2f} deg before, {err_after:.3f} deg after; wrote {a.out}")


if __name__ == "__main__":
    main()
