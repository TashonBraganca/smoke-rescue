"""Shared-context panel for the walkthrough video: what each robot's brain believes, drawn on the real 2x2 building.

On the top-down capture (data/evidence/c1_topdown.png) it draws, for each robot and from its own logs only: its
estimated position and heading, the path it has walked, its current goal and planned path, the search sector the
commander assigned, survivors it confirmed, candidates its agent rejected, and areas its agent ordered it to avoid.
A line between the robots shows the radio: solid green when the last broadcast was delivered, dashed red when it
was queued, with the packet size. A footer counts partner maps received and notes when the maps first overlapped.

    from contextpanel import ContextPanel
    p = ContextPanel(recording_dir); img = p.render(t)      # 640 x 640, BGR
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

W, H = 640, 640
X0, X1, Z0, Z1 = -1.0, 41.0, -1.0, 29.0       # extent of the top-down capture (footprint plus 1 m)
COL = {0: (40, 170, 245), 1: (230, 170, 50)}  # BGR, matching the chat panel: A orange, B blue
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


class ContextPanel:
    def __init__(self, recording: Path, topdown: Path | None = None):
        rec = Path(recording)
        self.robots = json.loads((rec / "meta.json").read_text())["robots"][:2]
        img = cv2.imread(str(topdown or Path(__file__).resolve().parents[1] / "data" / "evidence" / "c1_topdown.png"))
        self.mw = W - 20
        self.mh = int(round(self.mw * img.shape[0] / img.shape[1]))
        self.base = (cv2.resize(img, (self.mw, self.mh), interpolation=cv2.INTER_AREA) * 0.55).astype(np.uint8)
        self.state = {r: _jsonl(rec / r / "brain_state.jsonl") for r in self.robots}
        self.events = {r: _jsonl(rec / r / "decisions.jsonl") for r in self.robots}
        self.radio = []
        rp = rec / "brain_radio.txt"
        if rp.exists():
            for line in rp.read_text().splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                t, frm, to, seq, size, full, status, _ = line.split()[:8]
                self.radio.append((float(t), frm, to, int(size), status))

    def px(self, x: float, y: float) -> tuple[int, int]:
        return (int(10 + (x - X0) / (X1 - X0) * self.mw), int(60 + (Z1 - y) / (Z1 - Z0) * self.mh))

    def render(self, t: float) -> np.ndarray:
        img = np.full((H, W, 3), (24, 22, 20), np.uint8)
        cv2.putText(img, "SHARED CONTEXT", (16, 30), FONT, 0.75, (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(img, "each robot's own estimate, 40 x 28 m building", (230, 30), FONT, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
        img[60:60 + self.mh, 10:10 + self.mw] = self.base
        footer = []
        poses = {}
        for k, r in enumerate(self.robots):
            c = COL[k]
            ev = [e for e in self.events[r] if e["t"] <= t]
            for e in ev:
                n = e.get("numbers") or {}
                if e["kind"] == "sector_assigned":
                    sector = (n.get("x0"), n.get("y0"), n.get("x1"), n.get("y1"))
            sectors = [e for e in ev if e["kind"] == "sector_assigned"]
            if sectors:
                n = sectors[-1]["numbers"]
                p0, p1 = self.px(n["x0"], n["y1"]), self.px(n["x1"], n["y0"])
                for x in range(p0[0], p1[0], 12):
                    cv2.line(img, (x, p0[1]), (min(x + 6, p1[0]), p0[1]), c, 1)
                    cv2.line(img, (x, p1[1]), (min(x + 6, p1[0]), p1[1]), c, 1)
                for y in range(p0[1], p1[1], 12):
                    cv2.line(img, (p0[0], y), (p0[0], min(y + 6, p1[1])), c, 1)
                    cv2.line(img, (p1[0], y), (p1[0], min(y + 6, p1[1])), c, 1)
            for e in ev:
                n = e.get("numbers") or {}
                if "x_m" not in n:
                    continue
                q = self.px(n["x_m"], n["y_m"])
                if e["kind"] == "survivor_confirmed":
                    cv2.drawMarker(img, q, (90, 230, 90), cv2.MARKER_CROSS, 18, 3)
                    cv2.circle(img, q, 11, c, 2, cv2.LINE_AA)
                elif e["kind"] == "candidate_rejected":
                    cv2.drawMarker(img, q, (150, 150, 150), cv2.MARKER_TILTED_CROSS, 12, 2)
                elif e["kind"] == "agent_order" and "radius_m" in n:
                    cv2.circle(img, q, max(3, int(n["radius_m"] / (X1 - X0) * self.mw)), (60, 60, 230), 1, cv2.LINE_AA)
            st = [s for s in self.state[r] if s["t"] <= t]
            if not st:
                continue
            trail = np.array([self.px(s["x"], s["y"]) for s in st], np.int32)
            if len(trail) > 1:
                cv2.polylines(img, [trail], False, c, 1, cv2.LINE_AA)
            s = st[-1]
            if s.get("path"):
                cv2.polylines(img, [np.array([self.px(*p) for p in s["path"]], np.int32)], False, (200, 200, 200), 1, cv2.LINE_AA)
            if s.get("goal"):
                cv2.drawMarker(img, self.px(*s["goal"]), c, cv2.MARKER_DIAMOND, 10, 2)
            p = self.px(s["x"], s["y"])
            poses[r] = p
            h = -s["yaw_rad"]                               # brain yaw is left-positive; draw right-positive heading
            tip = (int(p[0] + 14 * np.sin(h)), int(p[1] - 14 * np.cos(h)))
            cv2.circle(img, p, 7, c, -1, cv2.LINE_AA)
            cv2.line(img, p, tip, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, "AB"[k], (p[0] + 9, p[1] - 9), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            got = sum(1 for e in ev if e["kind"] == "peer_message_received")
            shared = next((e["t"] for e in ev if e["kind"] == "map_shared"), None)
            found = sum(1 for e in ev if e["kind"] == "survivor_confirmed")
            footer.append((c, f"{'AB'[k]}: {s.get('state', '')} | partner maps received {got}"
                              + (f" | maps overlap since {shared:.0f} s" if shared is not None else "") + f" | survivors {found}"))
        if len(poses) == 2:
            last = [x for x in self.radio if x[0] <= t]
            if last:
                lt, frm, to, size, status = last[-1]
                a, b = poses[self.robots[0]], poses[self.robots[1]]
                if status == "sent":
                    cv2.line(img, a, b, (90, 220, 90), 1, cv2.LINE_AA)
                    txt = f"radio up: {size} B map packet delivered"
                else:
                    n = 14
                    for i in range(0, n, 2):
                        u, v = i / n, (i + 1) / n
                        cv2.line(img, (int(a[0] + (b[0] - a[0]) * u), int(a[1] + (b[1] - a[1]) * u)),
                                 (int(a[0] + (b[0] - a[0]) * v), int(a[1] + (b[1] - a[1]) * v)), (60, 60, 230), 1, cv2.LINE_AA)
                    txt = f"radio down ({status.replace('queued_', '')}): newest map waits in queue"
                footer.insert(0, ((200, 200, 200), txt))
        y = 60 + self.mh + 26
        for c, line in footer[:4]:
            cv2.putText(img, line[:78], (16, y), FONT, 0.45, c, 1, cv2.LINE_AA)
            y += 24
        return img
