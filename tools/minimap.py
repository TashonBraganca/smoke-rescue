"""Top-down 2D house map for the walkthrough video, in the style of a rescue robot's operator map.

Everything plotted comes from the robots' own outputs: SLAM paths (robot B only after its map merged into A's), obstacles
mapped from their depth images (grey squares), fire hotspots from their thermal cameras (red asterisks), people found
by their detector (green dots P1, P2, ...), and their air-temperature readings. The one piece of ground truth is the
faint house outline, drawn so a viewer can check the robots' map against the real house. To draw on it, robot A's map is
placed onto the house plan with a single rigid best-fit transform (rotation plus shift), so drift still shows as the
robots' map leaving the true walls.

    from minimap import MiniMap
    m = MiniMap(recording_dir, run_dir); img = m.render(t)
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

SIZE = 640
CELL = 0.2      # metres per obstacle square
EVERY = 6       # use every 6th frame (2.5 Hz) for mapping
INK = {"bg": (24, 22, 20), "grid": (38, 36, 34), "wall": (70, 70, 70), "label": (210, 210, 210), "obst": (190, 190, 190),
       "fire": (40, 40, 235), "person": (60, 200, 60), "A": (40, 170, 245), "B": (230, 170, 50), "warn": (40, 40, 220),
       "head": (40, 220, 230)}


def read_tum(path: Path) -> np.ndarray:
    rows = [l.split() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    return np.array(rows, dtype=np.float64)


def quat_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pose(row: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_R(row[4:8])
    T[:3, 3] = row[1:4]
    return T


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rigid transform (no scale) that best maps src points onto dst points."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = md - R @ ms
    return T


def yaml_values(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if ":" in line and not line.startswith(("%", "#")):
            k, v = line.split(":", 1)
            try:
                out[k.strip()] = float(v.strip())
            except ValueError:
                pass
    return out


class MiniMap:
    def __init__(self, recording: Path, run: Path):
        self.rec, self.run = Path(recording), Path(run)
        meta = json.loads((self.rec / "meta.json").read_text())
        self.robots = meta["robots"][:2]
        plan = json.loads((self.rec / "oracle" / "floorplan_used.json").read_text())
        self.plan = plan
        fp = plan["footprint"]
        self.x0, self.z0, self.x1, self.z1 = fp[0] - 2.0, fp[1] - 2.0, fp[2] + 2.0, fp[3] + 2.0
        self.scale = min((SIZE - 40) / (self.x1 - self.x0), (SIZE - 170) / (self.z1 - self.z0))
        merge = json.loads((self.run / "merge" / "merge.json").read_text()) if (self.run / "merge" / "merge.json").exists() else {}
        self.t_merge = float(merge["t_merge"]) if merge.get("merged") else float("inf")
        T_A_B = np.asarray(merge["T_A_B"]) if merge.get("merged") else None
        # Display transform: robot A's SLAM frame onto the house plan (one rigid fit). Robot B joins through T_A_B.
        trajA = read_tum(self.run / self.robots[0] / "CameraTrajectory.txt")
        gtA = read_tum(self.rec / "oracle" / f"{self.robots[0]}_groundtruth.txt")
        idx = [int(np.argmin(np.abs(gtA[:, 0] - t))) for t in trajA[:, 0]]
        # Fit on positions plus points 1 m along each camera's down and forward axes: the cameras all sit at one height,
        # so positions alone leave the tilt free and the floor leaked into the obstacle band.
        def anchors(rows):
            out = []
            for row in rows:
                T = pose(row)
                out += [T[:3, 3], T[:3, 3] + T[:3, 1], T[:3, 3] + T[:3, 2]]
            return np.array(out)
        self.T_disp = umeyama(anchors(trajA), anchors(gtA[idx]))
        self.frame_T = {self.robots[0]: self.T_disp}
        if T_A_B is not None and len(self.robots) > 1:
            self.frame_T[self.robots[1]] = self.T_disp @ T_A_B
        self.paths, self.points, self.fires, self.air = {}, {}, {}, {}
        for r in self.robots:
            self._load_robot(r)
        self.people = self._people()
        self.show_furniture = True
        self.furniture = []
        fire_cfg = Path(__file__).resolve().parents[1] / "config" / "fire.json"
        sizes = json.loads(fire_cfg.read_text()).get("furniture", {}) if fire_cfg.exists() else {}
        for l in plan.get("landmarks", []):
            if l["kind"] in sizes and not l["kind"].startswith(("hallway_art", "bedroom1_art", "large_rug")):
                w, d = sizes[l["kind"]]["size_m"]
                self.furniture.append(((l["at"][0], l["at"][1]), (w, d), l.get("yaw", 0), l["kind"]))

    # ------------------------------------------------------------------ data
    def _load_robot(self, r: str) -> None:
        rdir = self.rec / r
        traj = read_tum(self.run / r / "CameraTrajectory.txt")
        T = self.frame_T.get(r)
        if T is None:
            self.paths[r] = (np.zeros(0), np.zeros((0, 2)))
            self.points[r] = []
            self.fires[r] = []
            return
        world = np.array([(T @ pose(row))[:3, 3] for row in traj])
        self.paths[r] = (traj[:, 0], world[:, [0, 2]])
        cam = yaml_values(rdir / "camera.yaml")
        th = yaml_values(rdir / "thermal.yaml") if (rdir / "thermal.yaml").exists() else None
        assoc = [l.split() for l in (rdir / "associations.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
        times = traj[:, 0]
        pts, fires = [], []
        for k in range(0, len(assoc), EVERY):
            t, _, _, drel = assoc[k]
            t = float(t)
            j = int(np.argmin(np.abs(times - t)))
            if abs(times[j] - t) > 1e-3:
                continue
            Tw = T @ pose(traj[j])
            depth = cv2.imread(str(rdir / drel), cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue
            h, w = depth.shape
            vv, uu = np.mgrid[0:h:8, 0:w:8]
            z = depth[vv, uu].astype(np.float64) / 1000.0
            ok = (z > 0.3) & (z < 6.0)
            x = (uu[ok] - cam["Camera1.cx"]) * z[ok] / cam["Camera1.fx"]
            y = (vv[ok] - cam["Camera1.cy"]) * z[ok] / cam["Camera1.fy"]
            p = (Tw[:3, :3] @ np.stack([x, y, z[ok]]) ).T + Tw[:3, 3]
            height = -p[:, 1]                  # house frame has y pointing down
            keep = (height > 0.2) & (height < 1.9)
            cells = np.unique(np.floor(p[keep][:, [0, 2]] / CELL).astype(int), axis=0)
            floor = np.unique(np.floor(p[np.abs(height) < 0.12][:, [0, 2]] / CELL).astype(int), axis=0)
            pts.append((t, cells, floor))
            tp = rdir / "thermal" / f"{assoc[k][0]}.png"
            if th is not None and tp.exists():
                dk = cv2.imread(str(tp), cv2.IMREAD_UNCHANGED)
                hot = np.argwhere(dk[::2, ::2].astype(np.float32) / 10 - 273.15 > 250) * 2
                if len(hot):
                    vt, ut = hot[:, 0].astype(np.float64), hot[:, 1].astype(np.float64)
                    xn = (ut - th["Camera1.cx"]) / th["Camera1.fx"]
                    yn = (vt - th["Camera1.cy"]) / th["Camera1.fy"]
                    u = np.clip(np.round(cam["Camera1.cx"] + cam["Camera1.fx"] * xn).astype(int), 0, w - 1)
                    v = np.clip(np.round(cam["Camera1.cy"] + cam["Camera1.fy"] * yn).astype(int), 0, h - 1)
                    zf = depth[v, u].astype(np.float64) / 1000.0
                    okf = (zf > 0.3) & (zf < 6.0)
                    if okf.any():
                        pf = (Tw[:3, :3] @ np.stack([xn[okf] * zf[okf], yn[okf] * zf[okf], zf[okf]])).T + Tw[:3, 3]
                        fires.append((t, np.unique(np.floor(pf[:, [0, 2]] / 0.5).astype(int), axis=0)))
        self.points[r], self.fires[r] = pts, fires
        air = rdir / "air_temp.txt"
        self.air[r] = np.loadtxt(air) if air.exists() else np.zeros((0, 2))

    def _people(self) -> list[dict]:
        merged = self.run / "merge" / "survivors_merged.json"
        out = []
        if merged.exists():
            for k, s in enumerate(json.loads(merged.read_text())):
                p = self.T_disp @ np.r_[np.asarray(s["position_a"], float), 1.0]
                out.append({"label": f"P{k + 1}", "t": float(s["first_seen_t"]), "xz": p[[0, 2]], "by": s.get("seen_by", [])})
        return out

    # ------------------------------------------------------------------ drawing
    def px(self, xz) -> tuple[int, int]:
        x, z = float(xz[0]), float(xz[1])
        return int(20 + (x - self.x0) * self.scale), int(SIZE - 160 - (z - self.z0) * self.scale)

    def render(self, t: float) -> np.ndarray:
        img = np.full((SIZE, SIZE, 3), INK["bg"], np.uint8)
        cv2.putText(img, "2D HOUSE MAP", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(img, "grey: mapped surfaces   dim: seen free floor   brown: true furniture", (190, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
        for room in self.plan["rooms"]:
            r = room["rect"]
            cv2.rectangle(img, self.px((r[0], r[1])), self.px((r[2], r[3])), INK["wall"], 1, cv2.LINE_AA)
        hits, fire_cells, free_cells = {}, set(), set()
        for r in self.robots:
            for tt, cells, floor in self.points.get(r, []):
                if tt > t or (r != self.robots[0] and t < self.t_merge):
                    continue
                for c in map(tuple, cells):
                    hits[c] = hits.get(c, 0) + 1
                free_cells.update(map(tuple, floor))
            for tt, cells in self.fires.get(r, []):
                if tt <= t and not (r != self.robots[0] and t < self.t_merge):
                    fire_cells.update(map(tuple, cells))
        seen_cells = [c for c, n in hits.items() if n >= 2]      # seen in at least two frames: drops one-off noise
        # Seen free floor: dim fill, so empty space that was looked at differs from space never seen (black).
        fh = max(1, int(CELL * self.scale / 2))
        for cx, cz in free_cells:
            p = self.px(((cx + 0.5) * CELL, (cz + 0.5) * CELL))
            cv2.rectangle(img, (p[0] - fh, p[1] - fh), (p[0] + fh, p[1] + fh), (52, 48, 44), -1)
        for room in self.plan["rooms"]:
            r = room["rect"]
            cv2.rectangle(img, self.px((r[0], r[1])), self.px((r[2], r[3])), INK["wall"], 1, cv2.LINE_AA)
        if self.show_furniture:
            for (cx, cz), (w, d), yaw, name in self.furniture:
                c, sn = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
                corners = [(cx + c * a + sn * b, cz - sn * a + c * b) for a, b in ((-w/2, -d/2), (w/2, -d/2), (w/2, d/2), (-w/2, d/2))]
                cv2.polylines(img, [np.array([self.px(q) for q in corners], np.int32)], True, (110, 90, 60), 1, cv2.LINE_AA)
        half = max(1, int(CELL * self.scale / 2) - 2)
        for cx, cz in seen_cells:
            p = self.px(((cx + 0.5) * CELL, (cz + 0.5) * CELL))
            cv2.rectangle(img, (p[0] - half, p[1] - half), (p[0] + half, p[1] + half), INK["obst"], 1)
        fire_pts = []
        for fx, fz in fire_cells:
            p = self.px(((fx + 0.5) * 0.5, (fz + 0.5) * 0.5))
            fire_pts.append(((fx + 0.5) * 0.5, (fz + 0.5) * 0.5))
            for a in range(0, 180, 45):
                d = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]) * 4
                cv2.line(img, (int(p[0] - d[0]), int(p[1] - d[1])), (int(p[0] + d[0]), int(p[1] + d[1])), INK["fire"], 1, cv2.LINE_AA)
        robots_now = {}
        for r in self.robots:
            times, xz = self.paths[r]
            if len(times) == 0:
                continue
            if r != self.robots[0] and t < self.t_merge:
                continue
            m = times <= t
            if m.sum() < 2:
                continue
            pts = np.array([self.px(p) for p in xz[m]], np.int32)
            colour = INK["A"] if r == self.robots[0] else INK["B"]
            cv2.polylines(img, [pts], False, colour, 2, cv2.LINE_AA)
            cv2.circle(img, tuple(pts[0]), 5, (60, 230, 230), -1, cv2.LINE_AA)
            cv2.putText(img, "START", (pts[0][0] + 6, pts[0][1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 230, 230), 1, cv2.LINE_AA)
            cur = pts[-1]
            cv2.circle(img, tuple(cur), 8, colour, -1, cv2.LINE_AA)
            if len(pts) > 3:
                d = (pts[-1] - pts[-4]).astype(float)
                if np.linalg.norm(d) > 0:
                    d = d / np.linalg.norm(d) * 16
                    cv2.arrowedLine(img, tuple(cur), (int(cur[0] + d[0]), int(cur[1] + d[1])), colour, 2, cv2.LINE_AA, tipLength=0.4)
            cv2.putText(img, r.replace("robot_", "ROBOT "), (cur[0] + 10, cur[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
            robots_now[r] = xz[m][-1]
        if t < self.t_merge and len(self.robots) > 1:
            cv2.putText(img, "robot B not on this map until the maps merge", (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, INK["B"], 1, cv2.LINE_AA)
        else:
            cv2.putText(img, f"maps merged at {self.t_merge:.0f} s", (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, INK["B"], 1, cv2.LINE_AA)
        # A person found only by robot B appears once B's map has merged into A's.
        people = [p for p in self.people if p["t"] <= t and (t >= self.t_merge or self.robots[0] in p["by"])]
        for p in people:
            q = self.px(p["xz"])
            cv2.circle(img, q, 8, INK["person"], -1, cv2.LINE_AA)
            cv2.putText(img, p["label"], (q[0] + 9, q[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK["person"], 1, cv2.LINE_AA)

        # Bottom panel: environment readings and detections.
        y0 = SIZE - 128
        cv2.line(img, (0, y0), (SIZE, y0), (60, 60, 60), 1)
        cv2.putText(img, "ENVIRONMENT (SIMULATED SENSORS)", (16, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 230), 1, cv2.LINE_AA)
        cv2.putText(img, "DETECTION", (350, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 230), 1, cv2.LINE_AA)
        line = y0 + 50
        for r in self.robots:
            a = self.air.get(r, np.zeros((0, 2)))
            m = a[:, 0] <= t if len(a) else np.zeros(0, bool)
            temp = a[m][-1, 1] if m.any() else float("nan")
            colour = INK["A"] if r == self.robots[0] else INK["B"]
            warn = temp > 50
            reading = "n/a" if np.isnan(temp) else f"{temp:5.1f} C"
            cv2.putText(img, f"{r.replace('robot_', 'Robot ')} air: {reading}" + ("  HOT" if warn else ""), (16, line),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK["warn"] if warn else colour, 1, cv2.LINE_AA)
            line += 24
        hazard = 0
        for xz in robots_now.values():
            hazard += sum(1 for f in fire_pts if np.hypot(f[0] - xz[0], f[1] - xz[1]) < 2.5)
        fire_count = len(self.cluster(fire_pts))
        cv2.putText(img, f"People: {len(people)}   Fires: {fire_count}", (350, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(img, f"Fire cells within 2.5 m of a robot: {hazard}", (350, y0 + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    INK["warn"] if hazard else (230, 230, 230), 1, cv2.LINE_AA)
        if hazard:
            cv2.putText(img, "HAZARD: fire near a robot", (350, y0 + 98), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INK["warn"], 1, cv2.LINE_AA)
        return img

    @staticmethod
    def cluster(points, radius: float = 1.5) -> list:
        clusters = []
        for p in points:
            for c in clusters:
                if np.hypot(c[0] - p[0], c[1] - p[1]) < radius:
                    break
            else:
                clusters.append(p)
        return clusters
