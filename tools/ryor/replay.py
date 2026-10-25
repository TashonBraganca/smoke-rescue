#!/usr/bin/env python3
"""Turn a headless trace into the replay the page plays: 0.5 s frames in 10 s chunks, a score and a validation verdict.

    python3 tools/ryor/replay.py <trace.jsonl> <setup key> <out dir>      (offline, writes status.json and chunk_*.json)

Used live by tools/ryor/entry.py, which feeds trace lines while the simulator is still writing them. Format:
tools/ryor/CONTRACT.txt section 3. Evaluator and viewer side: true robot poses, fire and the true people are used for
drawing and scoring only; the "seen" cells and the confirmations are the brains' own.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRAME_S = 0.5  # Match the headless pose cadence and keep phone replay payloads small.
CHUNK_S = 10.0  # Start playback before the full mission finishes.
MATCH_M = 1.5          # a confirmation within this of a true person counts as finding that person
MATCH_M_REASON = "The brain's confirmed spot is its own estimate; 1.5 m is about one body length plus a pose error of 0.5 m."
SAME_PERSON_M = 1.0    # Two robots' estimates within one body length are the same confirmation marker.
BOUNDS_MARGIN_M = 3.0  # Include the three launch pads queued outside the entrance.
EVENT_TEXT = {
    "map_shared": lambda e: f"{name(e['robot'])} linked its map with {name(e['reason'].rsplit(' ', 1)[-1].rstrip('.'))}.",
    "survivor_confirmed": lambda e: f"{name(e['robot'])} confirmed a person.",
    "survivor_from_peer": lambda e: f"{name(e['robot'])} heard about a person over the radio.",
    "room_marked_fire": lambda e: f"{name(e['robot'])} felt heat over the limit and marked the room on fire.",
    "mission_complete": lambda e: f"{name(e['robot'])} is walking out.",
    "room_searched": lambda e: f"{name(e['robot'])} finished searching a room.",
    "door_taken": lambda e: f"{name(e['robot'])} took a doorway; the others leave that room to it.",
    'agent_decision': lambda e: 'Terra for ' + name(e['robot']) + '. ' + e['reason'],
    'agent_unavailable': lambda e: e['reason'],
    'agent_order': lambda e: name(e['robot']) + ' reported an agent order. ' + e['reason'].replace('go_to: explore: ', ''),
}


def name(rid):
    return "Robot " + rid.split("_")[-1] if rid.startswith("robot_") else rid


def inside(poly, x, y):
    c = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i - 1], poly[i]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            c = not c
    return c


def floor_cells(house, grid):
    cm, ox, oy, n = grid["cell_m"], grid["origin"][0], grid["origin"][1], grid["size"]
    cells = set()
    for r in house["rooms_poly"]:
        xs, ys = [p[0] for p in r["poly"]], [p[1] for p in r["poly"]]
        for iy in range(max(0, int((min(ys) - oy) / cm)), min(n, int((max(ys) - oy) / cm) + 1)):
            for ix in range(max(0, int((min(xs) - ox) / cm)), min(n, int((max(xs) - ox) / cm) + 1)):
                if inside(r["poly"], ox + (ix + .5) * cm, oy + (iy + .5) * cm):
                    cells.add(iy * n + ix)
    return cells


class Builder:
    """Feed trace events in order; finished chunks come back from feed() and finish()."""

    def __init__(self, gallery, setup, mission_s):
        hid, r, diff = setup.split("__")[:3]
        self.house = next(h for h in gallery["houses"] if h["id"] == hid)
        self.grid, self.mission_s, self.setup = gallery["grid"], mission_s, setup
        self.robot_ids = ["robot_A", "robot_B", "robot_C"][:int(r[1:])]
        self.floor = floor_cells(self.house, self.grid)
        self.pose, self.entered, self.out_at = {}, set(), {}
        self.seen = {rid: [] for rid in self.robot_ids}
        self.union = set()
        self.events, self.found, self.confirms = [], [], []
        self.frames, self.chunk_no, self.sim_t = [], 0, 0.0
        self.frame_times = []
        self.bad = []

    def is_inside(self, x, y):
        return any(inside(r["poly"], x, y) for r in self.house["rooms_poly"])

    def feed(self, e):
        k = e.get("kind")
        if k == "pose":
            rid, x, y = e["robot"], e["x"], e["y"]
            ins = self.is_inside(x, y)
            if ins:
                self.entered.add(rid)
                self.out_at.pop(rid, None)
            elif rid in self.entered and rid not in self.out_at:
                self.out_at[rid] = e["t"]
            self.pose[rid] = {"id": rid, "x": round(x, 2), "y": round(y, 2), "yaw": round(e["yaw_deg"], 1), "state": e["state"],
                              "out": rid in self.out_at}
        elif k == "replay_map":
            self.seen[e["robot"]].extend(e["cells"])
            self.union.update(e["cells"])
        elif k == "survivor_confirmed":
            x, y = e["numbers"]["x_m"], e["numbers"]["y_m"]
            self.confirms.append((x, y, e["robot"], e["t"]))
            if all(math.dist((x, y), (f["x"], f["y"])) > SAME_PERSON_M for f in self.found):
                self.found.append({"id": f"c{len(self.found) + 1}", "x": round(x, 2), "y": round(y, 2), "by": e["robot"],
                                   "t": round(e["t"], 2), "new": True})
            self._event(e)
        elif k in EVENT_TEXT:
            self._event(e)
        elif k == "replay_fire":
            return self._frame(e["t"], e["flames"])
        return []

    def _event(self, e):
        try:
            text = EVENT_TEXT[e["kind"]](e)
        except Exception:
            text = e["kind"]
        self.events.append({"robot": e["robot"], "kind": e["kind"], "text": text, "t": round(e["t"], 2)})

    def _frame(self, t, flames):
        for rid in self.robot_ids:
            if rid not in self.pose:
                self.bad.append(f"no pose for {rid} at t={t}")
        b = self.house["bounds"]
        for p in self.pose.values():
            if not (b[0] - BOUNDS_MARGIN_M <= p["x"] <= b[2] + BOUNDS_MARGIN_M and b[1] - BOUNDS_MARGIN_M <= p["y"] <= b[3] + BOUNDS_MARGIN_M):
                self.bad.append(f"{p['id']} outside the house bounds at t={t}: ({p['x']}, {p['y']})")
        frame = {"t": round(t, 2), "robots": [dict(self.pose[r]) for r in self.robot_ids if r in self.pose],
                 "seen": {r: c for r, c in self.seen.items() if c}, "fire": flames,
                 "found": [{k: v for k, v in f.items() if k != "new"} for f in self.found if f.pop("new", False)],
                 "events": self.events}
        self.seen = {rid: [] for rid in self.robot_ids}
        self.events = []
        self.sim_t = t
        self.frame_times.append(round(t, 2))
        out = []
        while t >= (self.chunk_no + 1) * CHUNK_S - 1e-6 and self.frames:
            out.append(self._cut())
        self.frames.append(frame)
        return out

    def _cut(self):
        c = {"v": 1, "chunk": self.chunk_no, "t0": self.chunk_no * CHUNK_S, "frames": self.frames}
        self.frames, self.chunk_no = [], self.chunk_no + 1
        return c

    def finish(self):
        return [self._cut()] if self.frames else []

    def score(self):
        truth = self.house["people_truth"]
        # One marker can match one person. Two nearby people must never both score from one confirmation.
        edges = [[j for j, p in enumerate(truth) if math.dist((p['x'], p['y']), (f['x'], f['y'])) <= MATCH_M]
                 for f in self.found]
        matched = {}
        def assign(i, visited):
            for j in edges[i]:
                if j in visited:
                    continue
                visited.add(j)
                if j not in matched or assign(matched[j], visited):
                    matched[j] = i
                    return True
            return False
        for i in range(len(edges)):
            assign(i, set())
        found = len(matched)
        false = len(self.found) - found
        out = [r for r in self.robot_ids if r in self.out_at]
        all_out = len(out) == len(self.robot_ids)
        return {"people_found": found, "people_total": len(truth),
                "time_s": round(max(self.out_at.values()), 1) if all_out and out else round(self.sim_t, 1),
                "all_out": all_out, "coverage_pct": round(100 * len(self.union & self.floor) / max(1, len(self.floor)), 1),
                "robots_out": len(out), "robots": len(self.robot_ids), "false_confirmations": false,
                "first_person_s": min((c[3] for c in self.confirms), default=None)}

    def validate(self):
        """Contract section 5. Returns a list of reasons; empty means valid."""
        bad = list(dict.fromkeys(self.bad))[:5]
        want = int(round(self.mission_s / FRAME_S)) + 1
        if len(self.frame_times) != want:
            bad.append(f"{len(self.frame_times)} frames, expected {want}")
        gaps = [b for a, b in zip(self.frame_times, self.frame_times[1:]) if abs(b - a - FRAME_S) > 1e-3]
        if gaps:
            bad.append(f"{len(gaps)} frame gaps, first at t={gaps[0]}")
        if self.sim_t < self.mission_s - 1e-6:
            bad.append(f"sim time reached {self.sim_t} of {self.mission_s} s")
        return bad


def main():
    trace, setup, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    gallery = json.loads((ROOT / "data/runs/ryor/gallery.json").read_text())
    b = Builder(gallery, setup, gallery["mission_s"])
    out.mkdir(parents=True, exist_ok=True)
    chunks = []
    for line in trace.open():
        chunks += b.feed(json.loads(line))
    chunks += b.finish()
    for c in chunks:
        (out / f"chunk_{c['chunk']:04d}.json").write_text(json.dumps(c, separators=(",", ":")))
    bad = b.validate()
    status = {"v": 1, "setup": setup, "state": "failed" if bad else "done", "source": "recorded", "chunks": len(chunks),
              "sim_t": b.sim_t, "error": "; ".join(bad) or None, "score": b.score()}
    (out / "status.json").write_text(json.dumps(status, indent=1))
    print(json.dumps(status))


if __name__ == "__main__":
    main()
