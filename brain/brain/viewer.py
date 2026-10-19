"""C11 Rerun replay of a processed run: cameras, trajectories, growing maps, radio, matches, and the merge snap.

    cd brain && uv run python -m brain.viewer ../data/runs/<run_id> --recording ../data/recordings/<run_id> [--spawn]

World frame is robot A's map frame (its first camera, x right, y down, z forward). Robot B's map is drawn in its own
frame, sitting wrongly on top of A's origin, until the logged merge time; then it animates into place over one second.
The faint true-house outline comes from the oracle and is used for display only (SIMULATION_SPEC 2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from scipy.spatial.transform import Rotation, Slerp

from brain.io import RecordingReader
from brain.oracle import load_floorplan
from brain.trajectory import read_tum

COLOURS = {"robot_A": (242, 169, 0), "robot_B": (47, 163, 230), "robot_C": (229, 72, 77)}
TITLE = "Recorded run, processed by live SLAM pipeline"
SNAP_S = 1.0


def pose(row) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
    T[:3, 3] = row[1:4]
    return T


def log_transform(path: str, T: np.ndarray) -> None:
    rr.log(path, rr.Transform3D(translation=T[:3, 3], quaternion=Rotation.from_matrix(T[:3, :3]).as_quat()))


def keyframe_points(reader: RecordingReader, frame_index: int, step: int = 8) -> tuple[np.ndarray, np.ndarray]:
    f = reader.read_frame(frame_index)
    cam = reader.camera
    v, u = np.mgrid[0:cam.height:step, 0:cam.width:step]
    z = f.depth_m[v, u]
    ok = (z > 0.2) & (z < 6.0)
    pts = np.stack([(u[ok] - cam.cx) * z[ok] / cam.fx, (v[ok] - cam.cy) * z[ok] / cam.fy, z[ok]], 1)
    return pts, f.rgb[v, u][ok]


def house_outline(recording: Path, align_a: np.ndarray | None) -> list[np.ndarray]:
    plan = load_floorplan(recording)
    if not plan or align_a is None:
        return []
    to_a = np.linalg.inv(align_a)
    h = plan.get("wall_height", 2.8)
    strips = []
    for room in plan["rooms"]:
        x0, z0, x1, z1 = room["rect"]
        for y in (0.0, -h):                     # house frame: y down, floor at 0, ceiling at -h
            ring = np.array([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1], [x0, y, z0]], float)
            strips.append(ring @ to_a[:3, :3].T + to_a[:3, 3])
    return strips


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--recording", type=Path, required=True)
    ap.add_argument("--spawn", action="store_true")
    ap.add_argument("--rgb-every", type=int, default=2)
    ap.add_argument("--depth-every", type=int, default=6)
    args = ap.parse_args()
    run, rec = args.run.resolve(), args.recording.resolve()
    meta = json.loads((rec / "meta.json").read_text())
    robots = meta["robots"]
    a_name = robots[0]
    merge = json.loads((run / "merge" / "merge.json").read_text()) if (run / "merge" / "merge.json").exists() else {"merged": False}
    cands = [json.loads(l) for l in (run / "merge" / "candidates.jsonl").read_text().splitlines()] if (run / "merge" / "candidates.jsonl").exists() else []
    report = json.loads((run / "report.json").read_text()) if (run / "report.json").exists() else {}
    details = json.loads((run / "ate_details.json").read_text()) if (run / "ate_details.json").exists() else {}

    rr.init("smoke_rescue_replay", spawn=args.spawn)
    rr.save(str(run / "replay.rrd"))
    views = [rrb.Spatial2DView(origin=f"world/{r}/camera", name=f"{r} camera") for r in robots]
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
        rrb.Vertical(rrb.TextDocumentView(origin="title", name="Run"), rrb.Spatial3DView(origin="world", name="Merged map"), row_shares=[1, 9]),
        rrb.Vertical(*views, rrb.TimeSeriesView(origin="metrics", name="Metrics"), rrb.TextLogView(origin="log", name="Radio and merge log")),
        column_shares=[3, 2])))

    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    lines = [f"# {TITLE}", "", f"Run `{meta['run_id']}`, scenario `{meta.get('scenario')}`, smoke {meta.get('smoke_level')}.",
             "SLAM, place recognition and merging ran on the recorded sensor data; ground truth is used only for the grey outline and the error numbers."]
    for r, v in report.get("per_robot", {}).items():
        lines.append(f"- {r}: ATE {v['ate_rmse_m'] * 100:.1f} cm, {v['keyframes']} keyframes")
    for m in report.get("merges", []):
        lines.append(f"- merge at {m['t']:.1f} s: error {m['transform_error_m'] * 100:.1f} cm, {m['transform_error_deg']:.2f} deg, "
                     f"{m['accepted_matches']} accepted, {m['rejected_candidates']} rejected, false merges {report.get('false_merges')}")
    rr.log("title", rr.TextDocument("\n".join(lines), media_type=rr.MediaType.MARKDOWN), static=True)

    align_a = None
    if a_name in details:
        align_a = np.eye(4)
        align_a[:3, :3] = np.asarray(details[a_name]["alignment_rotation"])
        align_a[:3, 3] = np.asarray(details[a_name]["alignment_translation_m"])
    outline = house_outline(rec, align_a)
    if outline:
        rr.log("world/house_truth", rr.LineStrips3D(outline, colors=[(150, 150, 150, 90)], radii=0.008), static=True)

    # Robot B's frame: identity until the merge, then interpolate to T_A_B over SNAP_S seconds.
    T_A_B = np.asarray(merge["T_A_B"]) if merge.get("merged") else None
    t_merge = merge.get("t_merge")
    events = []   # (t, callable) merged in time order with per-frame logging
    for r in robots[1:]:
        rr.set_time("sim_time", duration=0.0)
        log_transform(f"world/{r}", np.eye(4))
        if T_A_B is not None and r == robots[1]:
            rots = Slerp([0, 1], Rotation.from_matrix([np.eye(3), T_A_B[:3, :3]]))
            for i in range(11):
                a = i / 10
                T = np.eye(4)
                T[:3, :3] = rots([a]).as_matrix()[0]
                T[:3, 3] = a * T_A_B[:3, 3]
                events.append((t_merge + a * SNAP_S, lambda T=T, r=r: log_transform(f"world/{r}", T)))
            events.append((t_merge, lambda: rr.log("log/merge", rr.TextLog(
                f"MERGE: {robots[1]} map snaps into {a_name}'s frame (two agreeing matches, "
                f"{merge['agreement_m'] * 100:.1f} cm / {merge['agreement_deg']:.2f} deg apart)", level=rr.TextLogLevel.WARN))))

    # Candidate links, in A's frame. Accepted stay green; rejected flash red for one second.
    kf_pos = {}
    for r in robots:
        traj = read_tum(run / r / "KeyFrameTrajectory.txt")
        for kf in json.loads((run / r / "keyframes.json").read_text()):
            i = int(np.argmin(np.abs(traj[:, 0] - kf["timestamp"])))
            kf_pos[(r, kf["keyframe_id"])] = traj[i, 1:4]
    acc_strips, n_acc, n_rej = [], 0, 0
    for c in sorted(cands, key=lambda c: c["t"]):
        a_kf = c["match_kf"] if c["to"] == a_name else c["query_kf"]
        b_kf = c["query_kf"] if c["to"] == a_name else c["match_kf"]
        b_name = c["from"] if c["to"] == a_name else c["to"]
        pa, pb = kf_pos.get((a_name, a_kf)), kf_pos.get((b_name, b_kf))
        if pa is None or pb is None:
            continue
        Tb = T_A_B if T_A_B is not None else np.eye(4)
        pb_a = Tb[:3, :3] @ pb + Tb[:3, 3]
        if c["accepted"]:
            n_acc += 1
            acc_strips.append(np.stack([pa, pb_a]))
            strips = list(acc_strips)
            events.append((c["t"], lambda s=strips, n=n_acc: (rr.log("world/merge/accepted", rr.LineStrips3D(s, colors=[(60, 200, 90)], radii=0.01)),
                                                               rr.log("metrics/accepted_matches", rr.Scalars(n)))))
        else:
            n_rej += 1
            seg = np.stack([pa, pb_a])
            events.append((c["t"], lambda s=seg, n=n_rej, why=c["reason"]: (rr.log("world/merge/rejected", rr.LineStrips3D([s], colors=[(230, 50, 50)], radii=0.012)),
                                                                             rr.log("metrics/rejected_candidates", rr.Scalars(n)))))
            events.append((c["t"] + 1.0, lambda: rr.log("world/merge/rejected", rr.Clear(recursive=False))))
        events.append((c["t"], lambda c=c: rr.log("log/merge", rr.TextLog(
            f"{c['msg_id']} {c['from']}->{c['to']} kf {c['query_kf']} vs {c['match_kf']} rank {c['rank']}: {c['reason']} "
            f"(inliers {c.get('inliers')}, fitness {c.get('fitness', 0):.2f})"))))

    # Survivors found by the robots' detector (in A's frame after the merge) and per-frame detection boxes.
    surv_path = run / "merge" / "survivors_merged.json"
    if surv_path.exists():
        found = json.loads(surv_path.read_text())
        for k in range(len(found)):
            shown = [f for f in found if f["first_seen_t"] <= found[k]["first_seen_t"]]
            events.append((found[k]["first_seen_t"], lambda sh=shown: rr.log("world/survivors", rr.Points3D(
                [f["position_a"] for f in sh], radii=0.18, colors=[(235, 60, 60)], labels=[f"survivor {f['id']}" for f in sh]))))
            events.append((found[k]["first_seen_t"], lambda f=found[k]: rr.log("log/merge", rr.TextLog(
                f"SURVIVOR {f['id']} found by {', '.join(f['seen_by'])} ({f['n_frames']} frames)", level=rr.TextLogLevel.WARN))))
    for r in robots:
        det_path = run / r / "detections.json"
        if det_path.exists():
            for d in json.loads(det_path.read_text()):
                x0, y0, x1, y1 = d["box"]
                events.append((d["t"], lambda r=r, x0=x0, y0=y0, x1=x1, y1=y1, c=d["conf"]: rr.log(
                    f"world/{r}/camera/rgb/detections", rr.Boxes2D(array=[[x0, y0, x1 - x0, y1 - y0]], array_format=rr.Box2DFormat.XYWH,
                                                                labels=[f"person {c:.2f}"], colors=[(235, 60, 60)]))))
                events.append((d["t"] + 0.3, lambda r=r: rr.log(f"world/{r}/camera/rgb/detections", rr.Clear(recursive=False))))

    for line in (rec / "radio.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        t_send, t_del, src, dst, msg_id, delivered, reason = line.split()
        events.append((float(t_send), lambda s=f"{msg_id} {src} -> {dst}: {'delivered' if delivered == '1' else 'not delivered'} ({reason})":
                       rr.log("log/radio", rr.TextLog(s))))
    if t_merge is not None and report.get("merges"):
        m = report["merges"][0]
        events.append((t_merge + SNAP_S, lambda: rr.log("metrics/merge_error_cm", rr.Scalars(m["transform_error_m"] * 100))))

    # Per-robot frames: camera pose, RGB, depth, trajectory and keyframe map chunks.
    for r in robots:
        reader = RecordingReader(rec / r)
        cam = reader.camera
        traj = read_tum(run / r / "CameraTrajectory.txt")
        kfs = {kf["frame_index"]: kf for kf in json.loads((run / r / "keyframes.json").read_text())}
        kf_traj = read_tum(run / r / "KeyFrameTrajectory.txt")
        colour = COLOURS.get(r, (200, 200, 200))
        rr.log(f"world/{r}/camera", rr.Pinhole(focal_length=[cam.fx, cam.fy], principal_point=[cam.cx, cam.cy],
                                                width=cam.width, height=cam.height, image_plane_distance=0.3), static=True)
        times = np.array([a.rgb_timestamp for a in reader.associations])
        n_kf = 0
        for i, t in enumerate(times):
            j = int(np.argmin(np.abs(traj[:, 0] - t)))
            if abs(traj[j, 0] - t) > 1e-3:
                continue
            rr.set_time("sim_time", duration=float(t))
            log_transform(f"world/{r}/camera", pose(traj[j]))
            if i % args.rgb_every == 0 or i in kfs:
                f = reader.read_frame(i)
                rr.log(f"world/{r}/camera/rgb", rr.Image(f.rgb).compress(jpeg_quality=75))
                if i % args.depth_every == 0:
                    rr.log(f"world/{r}/camera/depth", rr.DepthImage(f.depth[::2, ::2], meter=cam.depth_factor))
            if i % 15 == 0:
                rr.log(f"world/{r}/trajectory", rr.LineStrips3D([traj[: j + 1, 1:4]], colors=[colour], radii=0.01))
            if i in kfs:
                k = int(np.argmin(np.abs(kf_traj[:, 0] - kfs[i]["timestamp"])))
                pts, cols = keyframe_points(reader, i)
                T = pose(kf_traj[k])
                n_kf += 1
                rr.log(f"world/{r}/map/kf_{kfs[i]['keyframe_id']:05d}",
                       rr.Points3D(pts @ T[:3, :3].T + T[:3, 3], colors=(0.6 * cols + 0.4 * np.array(colour)).astype(np.uint8), radii=0.012))
                rr.log(f"metrics/keyframes_{r}", rr.Scalars(n_kf))
    for t, fn in sorted(events, key=lambda e: e[0]):
        rr.set_time("sim_time", duration=float(t))
        fn()
    (run / "replay_meta.json").write_text(json.dumps({"title": TITLE, "merge_time_s": t_merge, "snap_seconds": SNAP_S,
                                                      "accepted": n_acc, "rejected": n_rej}, indent=2) + "\n")
    print(f"wrote {run / 'replay.rrd'} (merge snap at {t_merge} s)")


if __name__ == "__main__":
    main()
