"""C9 map merging: radio-gated place recognition, geometric verification, pose-graph merge.

Robot-side module. It reads only the robots' own sensor data (recording folders), their own SLAM outputs
and descriptors, and radio.txt. It never reads ground truth.

For each delivered radio message, in delivery-time order (docs/INTERFACES.md 4.3):
  1. The message carries the sender's newest keyframe at t_send. The receiver may use it only from
     t_deliver, and only against keyframes it had itself mapped by then.
  2. Retrieve the top 5 receiver keyframes by descriptor cosine similarity.
  3. Verify each: ORB matches lifted to 3D with each image's depth, RANSAC rigid transform, then Open3D
     colored ICP on the two depth clouds. Accept when inliers > 60 and ICP fitness > 0.6.
  4. The first merge needs two independent accepted matches whose implied map-to-map transforms agree
     within 15 cm and 3 degrees (SIMULATION_SPEC 6.4). This rejects look-alike places.
  5. A pose graph over both robots' keyframes (odometry edges from SLAM, loop edges from every accepted
     match) is optimised; the result is T_A_B, which maps robot B's map frame into robot A's.

Every candidate, accepted or rejected, is logged with its reason and time.

    uv run python -m brain.merge <recording_dir> <run_dir>
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from brain.io import RecordingReader
from brain.trajectory import read_tum

INLIER_MIN = 60
FITNESS_MIN = 0.6
AGREE_M = 0.15
AGREE_DEG = 3.0
TOP_K = 5
RANSAC_THRESHOLD_M = 0.05
RANSAC_ITERS = 2000
ICP_VOXEL_M = 0.03
INDEPENDENT_MIN_S = 1.0     # two matches count as independent if their query keyframes are this far apart in time


# ----------------------------------------------------------------------------- geometry helpers

def pose_matrix(row: np.ndarray) -> np.ndarray:
    """TUM row (t tx ty tz qx qy qz qw) to a 4x4 camera-to-map matrix."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(row[4:8]).as_matrix()
    T[:3, 3] = row[1:4]
    return T


def transform_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Translation (m) and rotation (deg) between two 4x4 transforms."""
    d = np.linalg.inv(a) @ b
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(d[:3, :3]) - 1) / 2))))
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), angle


def kabsch(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rigid T with q ~ R p + t."""
    cp, cq = p.mean(0), q.mean(0)
    H = (p - cp).T @ (q - cq)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = cq - R @ cp
    return T


# ----------------------------------------------------------------------------- per-robot map

@dataclass
class RobotMap:
    name: str
    reader: RecordingReader
    keyframes: list[dict]
    poses: dict[int, np.ndarray]          # keyframe_id -> camera-to-map 4x4
    descriptors: np.ndarray               # (n, 512) float32, unit rows, same order as desc_ids
    desc_ids: list[int]
    desc_times: np.ndarray
    cache: dict = field(default_factory=dict)

    @classmethod
    def load(cls, recording: Path, run: Path, name: str) -> "RobotMap":
        reader = RecordingReader(recording / name)
        kfs = json.loads((run / name / "keyframes.json").read_text())
        traj = read_tum(run / name / "KeyFrameTrajectory.txt")
        poses = {}
        for kf in kfs:
            i = int(np.argmin(np.abs(traj[:, 0] - kf["timestamp"])))
            if abs(traj[i, 0] - kf["timestamp"]) < 1e-4:
                poses[kf["keyframe_id"]] = pose_matrix(traj[i])
        meta = json.loads((run / name / "descriptors.json").read_text())
        desc = np.load(run / name / "descriptors.npy").astype(np.float32)
        desc /= np.linalg.norm(desc, axis=1, keepdims=True) + 1e-12
        return cls(name, reader, kfs, poses, desc, list(meta["keyframe_ids"]), np.asarray(meta["timestamps"], float))

    def keyframe(self, kf_id: int) -> dict:
        return next(k for k in self.keyframes if k["keyframe_id"] == kf_id)

    def newest_before(self, t: float) -> int | None:
        idx = [i for i, ts in enumerate(self.desc_times) if ts <= t + 1e-9 and self.desc_ids[i] in self.poses]
        return self.desc_ids[idx[-1]] if idx else None

    def frame(self, kf_id: int):
        if kf_id not in self.cache:
            kf = self.keyframe(kf_id)
            self.cache[kf_id] = self.reader.read_frame(kf["frame_index"])
        return self.cache[kf_id]


# ----------------------------------------------------------------------------- verification

_orb = cv2.ORB_create(nfeatures=2500)
_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


def lift(kp, depth_m: np.ndarray, cam) -> tuple[np.ndarray, np.ndarray]:
    pts, ok = [], []
    for i, k in enumerate(kp):
        u, v = int(round(k.pt[0])), int(round(k.pt[1]))
        if 0 <= u < depth_m.shape[1] and 0 <= v < depth_m.shape[0]:
            z = float(depth_m[v, u])
            if 0.2 < z < 8.0:
                pts.append([(u - cam.cx) * z / cam.fx, (v - cam.cy) * z / cam.fy, z])
                ok.append(i)
    return np.asarray(pts, float).reshape(-1, 3), np.asarray(ok, int)


def cloud(frame, cam) -> o3d.geometry.PointCloud:
    rgb = o3d.geometry.Image(np.ascontiguousarray(frame.rgb))
    dep = o3d.geometry.Image(np.ascontiguousarray(frame.depth_m.astype(np.float32)))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, dep, depth_scale=1.0, depth_trunc=6.0, convert_rgb_to_intensity=False)
    intr = o3d.camera.PinholeCameraIntrinsic(cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy)
    pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr).voxel_down_sample(ICP_VOXEL_M)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=ICP_VOXEL_M * 3, max_nn=30))
    return pc


def verify(query: RobotMap, q_id: int, ref: RobotMap, r_id: int, rng: np.random.Generator) -> dict:
    """Estimate T (query camera -> reference camera). Returns a result dict with 'accepted' and 'reason'."""
    fq, fr = query.frame(q_id), ref.frame(r_id)
    gq = cv2.cvtColor(fq.rgb, cv2.COLOR_RGB2GRAY)
    gr = cv2.cvtColor(fr.rgb, cv2.COLOR_RGB2GRAY)
    kq, dq = _orb.detectAndCompute(gq, None)
    kr, dr = _orb.detectAndCompute(gr, None)
    res = {"inliers": 0, "fitness": 0.0, "accepted": False}
    if dq is None or dr is None or len(kq) < 20 or len(kr) < 20:
        res["reason"] = "too few features"
        return res
    pq, okq = lift(kq, fq.depth_m, query.reader.camera)
    pr, okr = lift(kr, fr.depth_m, ref.reader.camera)
    if len(okq) < 10 or len(okr) < 10:
        res["reason"] = "too few features with depth"
        return res
    knn = _matcher.knnMatch(dq[okq], dr[okr], k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
    res["matches"] = len(good)
    if len(good) < 12:
        res["reason"] = f"too few descriptor matches ({len(good)})"
        return res
    A = pq[[m.queryIdx for m in good]]
    B = pr[[m.trainIdx for m in good]]
    best, best_in = None, np.zeros(len(good), bool)
    for _ in range(RANSAC_ITERS):
        s = rng.choice(len(good), 3, replace=False)
        T = kabsch(A[s], B[s])
        err = np.linalg.norm((A @ T[:3, :3].T + T[:3, 3]) - B, axis=1)
        inl = err < RANSAC_THRESHOLD_M
        if inl.sum() > best_in.sum():
            best, best_in = T, inl
    res["inliers"] = int(best_in.sum())
    if best is None or best_in.sum() < 6:
        res["reason"] = f"RANSAC found no consistent transform ({int(best_in.sum())} inliers)"
        return res
    T0 = kabsch(A[best_in], B[best_in])
    src, dst = cloud(fq, query.reader.camera), cloud(fr, ref.reader.camera)
    try:
        icp = o3d.pipelines.registration.registration_colored_icp(
            src, dst, ICP_VOXEL_M * 1.5, T0,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=40))
        T, fitness, rmse = icp.transformation, icp.fitness, icp.inlier_rmse
    except RuntimeError:
        T, fitness, rmse = T0, 0.0, float("nan")
    res.update(fitness=float(fitness), icp_rmse=float(rmse), T_ref_query=np.asarray(T).tolist())
    if res["inliers"] <= INLIER_MIN:
        res["reason"] = f"inliers {res['inliers']} <= {INLIER_MIN}"
    elif fitness <= FITNESS_MIN:
        res["reason"] = f"ICP fitness {fitness:.2f} <= {FITNESS_MIN}"
    else:
        res["reason"] = "accepted"
        res["accepted"] = True
    return res


# ----------------------------------------------------------------------------- radio replay

def read_radio(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        t_send, t_deliver, src, dst, msg_id, delivered, reason = line.split()
        rows.append({"t_send": float(t_send), "t_deliver": float(t_deliver), "from": src, "to": dst,
                     "msg_id": msg_id, "delivered": delivered == "1", "reason": reason})
    return rows


def run(recording: Path, run_dir: Path, robots: tuple[str, str] = ("robot_A", "robot_B"), seed: int = 0) -> dict:
    maps = {r: RobotMap.load(recording, run_dir, r) for r in robots}
    a_name, b_name = robots
    rng = np.random.default_rng(seed)
    out = run_dir / "merge"
    out.mkdir(parents=True, exist_ok=True)
    log = []
    accepted = []          # dicts with T_A_B (b map -> a map)
    merge_event = None
    radio = sorted((m for m in read_radio(recording / "radio.txt") if m["delivered"]), key=lambda m: m["t_deliver"])

    for msg in radio:
        sender, receiver = maps[msg["from"]], maps[msg["to"]]
        q_id = sender.newest_before(msg["t_send"])
        if q_id is None:
            continue
        q_desc = sender.descriptors[sender.desc_ids.index(q_id)]
        # Receiver's database: only keyframes it had mapped by delivery time.
        avail = [i for i, t in enumerate(receiver.desc_times) if t <= msg["t_deliver"] + 1e-9 and receiver.desc_ids[i] in receiver.poses]
        if not avail:
            continue
        sims = receiver.descriptors[avail] @ q_desc
        order = np.argsort(-sims)[:TOP_K]
        for rank, j in enumerate(order):
            r_id = receiver.desc_ids[avail[j]]
            res = verify(sender, q_id, receiver, r_id, rng)
            entry = {"t": msg["t_deliver"], "msg_id": msg["msg_id"], "from": sender.name, "to": receiver.name,
                     "query_kf": q_id, "query_t": float(sender.desc_times[sender.desc_ids.index(q_id)]),
                     "match_kf": r_id, "match_t": float(receiver.desc_times[avail[j]]),
                     "rank": rank, "similarity": float(sims[j]), **{k: v for k, v in res.items() if k != "T_ref_query"}}
            if res["accepted"]:
                # Map-to-map: T_recvMap_sendMap = T_recvMap_recvCam * T_recvCam_sendCam * T_sendCam_sendMap
                T_rs = receiver.poses[r_id] @ np.asarray(res["T_ref_query"]) @ np.linalg.inv(sender.poses[q_id])
                T_A_B = T_rs if receiver.name == a_name else np.linalg.inv(T_rs)
                entry["T_A_B"] = T_A_B.tolist()
                entry["a_kf"] = r_id if receiver.name == a_name else q_id
                entry["b_kf"] = q_id if receiver.name == a_name else r_id
                accepted.append(entry)
                if merge_event is None:
                    for prev in accepted[:-1]:
                        dt_ = abs(prev["query_t"] - entry["query_t"]) if prev["from"] == entry["from"] else INDEPENDENT_MIN_S
                        if dt_ < INDEPENDENT_MIN_S or (prev["a_kf"], prev["b_kf"]) == (entry["a_kf"], entry["b_kf"]):
                            continue
                        dm, dd = transform_delta(np.asarray(prev["T_A_B"]), T_A_B)
                        if dm <= AGREE_M and dd <= AGREE_DEG:
                            merge_event = {"t": msg["t_deliver"], "pair": [a_name, b_name], "matches": [prev, entry],
                                           "agreement_m": dm, "agreement_deg": dd}
                            break
                    entry["status"] = "accepted, merged" if merge_event else "accepted, waiting for an agreeing second match"
            log.append(entry)

    result = {"merged": merge_event is not None, "pair": [a_name, b_name],
              "accepted": len(accepted), "rejected": sum(1 for e in log if not e["accepted"]), "candidates": len(log)}
    if merge_event:
        # Accepted matches that agree with the merge (within 0.5 m and 10 degrees of its first estimate) feed the pose graph;
        # the rest are logged as outliers by consistency and left out.
        T0 = np.asarray(merge_event["matches"][0]["T_A_B"])
        inliers = [e for e in accepted if all(x <= y for x, y in zip(transform_delta(T0, np.asarray(e["T_A_B"])), (0.5, 10.0)))]
        for e in accepted:
            e["used_in_pose_graph"] = e in inliers
        T_A_B = pose_graph(maps[a_name], maps[b_name], inliers, T0)
        merged_map(run_dir, a_name, b_name, T_A_B, out)
        result.update(t_merge=merge_event["t"], T_A_B=T_A_B.tolist(), T_A_B_first=T0.tolist(),
                      agreement_m=merge_event["agreement_m"], agreement_deg=merge_event["agreement_deg"],
                      pose_graph_edges=len(inliers))
    with (out / "candidates.jsonl").open("w") as f:
        for e in log:
            f.write(json.dumps(e) + "\n")
    (out / "merge.json").write_text(json.dumps(result, indent=2))
    return result


def pose_graph(a: RobotMap, b: RobotMap, matches: list[dict], T_init: np.ndarray) -> np.ndarray:
    """Optimise A and B keyframe poses in A's frame; returns the refined T_A_B (B map -> A map)."""
    reg = o3d.pipelines.registration
    graph = reg.PoseGraph()
    ids_a = sorted(a.poses, key=lambda k: a.keyframe(k)["timestamp"])
    ids_b = sorted(b.poses, key=lambda k: b.keyframe(k)["timestamp"])
    index = {}
    for k in ids_a:
        index[("a", k)] = len(graph.nodes)
        graph.nodes.append(reg.PoseGraphNode(a.poses[k]))
    for k in ids_b:
        index[("b", k)] = len(graph.nodes)
        graph.nodes.append(reg.PoseGraphNode(T_init @ b.poses[k]))
    info_odom = np.eye(6) * 1000.0
    for side, ids, m in (("a", ids_a, a), ("b", ids_b, b)):
        for k0, k1 in zip(ids, ids[1:]):
            rel = np.linalg.inv(m.poses[k1]) @ m.poses[k0]   # Open3D edge: T maps target-node frame... source->target
            graph.edges.append(reg.PoseGraphEdge(index[(side, k0)], index[(side, k1)], rel, info_odom, uncertain=False))
    for e in matches:
        # Loop edge between B keyframe (source) and A keyframe (target): relative pose implied by this match.
        Ta = a.poses[e["a_kf"]]
        Tb = np.asarray(e["T_A_B"]) @ b.poses[e["b_kf"]]
        rel = np.linalg.inv(Ta) @ Tb
        graph.edges.append(reg.PoseGraphEdge(index[("b", e["b_kf"])], index[("a", e["a_kf"])], rel, np.eye(6) * 100.0, uncertain=True))
    option = reg.GlobalOptimizationOption(max_correspondence_distance=0.05, edge_prune_threshold=0.25, reference_node=0)
    reg.global_optimization(graph, reg.GlobalOptimizationLevenbergMarquardt(), reg.GlobalOptimizationConvergenceCriteria(), option)
    # Refined T_A_B from B's first keyframe node: optimised pose times the inverse of its own-map pose.
    k0 = ids_b[0]
    return np.asarray(graph.nodes[index[("b", k0)]].pose) @ np.linalg.inv(b.poses[k0])


def merged_map(run_dir: Path, a_name: str, b_name: str, T_A_B: np.ndarray, out: Path) -> None:
    pa = o3d.io.read_point_cloud(str(run_dir / a_name / "map.ply"))
    pb = o3d.io.read_point_cloud(str(run_dir / b_name / "map.ply"))
    pb.transform(T_A_B)
    o3d.io.write_point_cloud(str(out / "merged_map.ply"), (pa + pb).voxel_down_sample(0.03))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recording", type=Path)
    ap.add_argument("run", type=Path)
    args = ap.parse_args()
    res = run(args.recording, args.run)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("T_")}, indent=2))


if __name__ == "__main__":
    main()
