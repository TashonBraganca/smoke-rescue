"""Side panel for the walkthrough video: the robots' radio traffic and decisions as a chat.

Left bubbles are robot A, right bubbles robot B, centred grey lines are link and team events. Sources, all robot-side
or the simulated radio: radio.txt (every broadcast and whether the radio delivered it), <robot>/decisions.jsonl (the
brain's decisions, including messages received from the other robot), <robot>/narration.jsonl (narrator sentences),
merge/merge.json and merge/survivors_merged.json (when the maps joined and who found which person).

    from chatpanel import ChatPanel
    c = ChatPanel(recording_dir, run_dir); img = c.render(t)
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

W, H = 640, 640
BG, INK, DIM = (24, 22, 20), (225, 225, 225), (140, 140, 140)
COL = {"A": (40, 170, 245), "B": (230, 170, 50), "sys": (150, 150, 150), "warn": (60, 60, 230), "good": (70, 200, 80)}
FONT = cv2.FONT_HERSHEY_SIMPLEX

DECISION_TEXT = {
    "goal": lambda e: f"Heading to new goal: {e.get('reason', '')}",
    "replan": lambda e: f"Replanning: {e.get('reason', '')}",
    "avoid_fire": lambda e: f"Avoiding fire: {e.get('reason', '')}",
    "retreat": lambda e: f"Too hot, retreating: {e.get('reason', '')}",
    "sensor_switch": lambda e: f"Camera: {e.get('reason', '')}",
    "survivor_candidate": lambda e: "Possible person on thermal, going to check",
    "survivor_confirmed": lambda e: "SURVIVOR CONFIRMED, position reported",
    "survivor_from_peer": lambda e: "Partner already found a person there, skipping",
    "goal_yielded": lambda e: f"Partner claimed that area, taking another: {e.get('reason', '')}",
    "map_shared": lambda e: "Our maps overlap now: I know which rooms my partner covered",
    "peer_message_received": None,       # shown as radio traffic instead
    "collision_risk": lambda e: f"Obstacle close: {e.get('reason', '')}",
    "mission_complete": lambda e: "Mission complete, heading out",
}


def wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + ([cur] if cur else [])


class ChatPanel:
    def __init__(self, recording: Path, run: Path):
        rec, run = Path(recording), Path(run)
        meta = json.loads((rec / "meta.json").read_text())
        robots = meta["robots"][:2]
        self.side = {robots[0]: "A", robots[1]: "B"} if len(robots) > 1 else {robots[0]: "A"}
        self.items: list[tuple[float, str, str, tuple]] = []   # (t, who, text, colour)

        # Radio link: only state changes and delivered traffic, so the chat stays readable.
        radio = rec / "radio.txt"
        if radio.exists():
            last_state, last_shown = {}, {}
            for line in radio.read_text().splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                t_send, _, frm, to, msg, ok, reason = line.split()[:7]
                t = float(t_send)
                key = (frm, to)
                state = "up" if ok == "1" else reason
                if last_state.get(key) != state:
                    if frm == robots[0]:
                        txt = "Radio link UP between A and B" if ok == "1" else f"Radio link down ({reason}): messages queue"
                        self.items.append((t, "sys", txt, COL["good"] if ok == "1" else COL["sys"]))
                    last_state[key] = state
                if ok == "1" and t - last_shown.get(key, -1e9) >= 10.0:
                    who = self.side.get(frm, "A")
                    self.items.append((t, who, f"-> {self.side.get(to, '?')}: sent place fingerprints ({msg})", COL[who]))
                    last_shown[key] = t

        for r, who in self.side.items():
            dec = run / r / "decisions.jsonl"
            if dec.exists():
                for line in dec.read_text().splitlines():
                    if not line.strip():
                        continue
                    e = json.loads(line)
                    fmt = DECISION_TEXT.get(e.get("kind"))
                    if e.get("kind") == "peer_message_received":
                        n = e.get("numbers", {})
                        self.items.append((float(e["t"]), who, f"<- received partner's map: {n.get('bytes', 0) / 1024:.1f} KB, "
                                           f"{n.get('age_s', 0):.1f} s old", COL[who]))
                    elif fmt is not None:
                        colour = COL["warn"] if e.get("kind") in ("retreat", "avoid_fire") else COL["good"] if "survivor" in e.get("kind", "") else COL[who]
                        self.items.append((float(e["t"]), who, fmt(e), colour))
            nar = run / r / "narration.jsonl"
            if nar.exists():
                for line in nar.read_text().splitlines():
                    if line.strip():
                        e = json.loads(line)
                        self.items.append((float(e["t"]), who, f"sees: {e['text']}", DIM))

        merge = run / "merge" / "merge.json"
        if merge.exists():
            m = json.loads(merge.read_text())
            if m.get("merged"):
                self.items.append((float(m["t_merge"]), "sys", "MAPS MERGED: both robots now share one map", COL["good"]))
        surv = run / "merge" / "survivors_merged.json"
        if surv.exists():
            for k, s in enumerate(json.loads(surv.read_text())):
                by = "+".join(self.side.get(b, "?") for b in s.get("seen_by", []))
                self.items.append((float(s["first_seen_t"]), "sys", f"Person P{k + 1} detected (seen by {by})", COL["good"]))
        self.items.sort(key=lambda x: x[0])

    def render(self, t: float) -> np.ndarray:
        img = np.full((H, W, 3), BG, np.uint8)
        cv2.putText(img, "TEAM RADIO", (16, 30), FONT, 0.75, (235, 235, 235), 2, cv2.LINE_AA)
        cv2.putText(img, "A (orange) left, B (blue) right, events centre", (190, 30), FONT, 0.42, DIM, 1, cv2.LINE_AA)
        shown = [it for it in self.items if it[0] <= t]
        y = H - 12
        for ti, who, text, colour in reversed(shown):
            lines = wrap(text, 46 if who != "sys" else 60)
            h = 20 * len(lines) + 22
            if y - h < 50:
                break
            y -= h
            if who == "sys":
                for n, line in enumerate(lines):
                    size = cv2.getTextSize(line, FONT, 0.45, 1)[0][0]
                    cv2.putText(img, line, ((W - size) // 2, y + 18 + 20 * n), FONT, 0.45, colour, 1, cv2.LINE_AA)
                cv2.putText(img, f"{ti:5.1f} s", (8, y + 18), FONT, 0.35, DIM, 1, cv2.LINE_AA)
            else:
                x0 = 12 if who == "A" else W - 12 - 420
                cv2.rectangle(img, (x0, y + 2), (x0 + 420, y + h - 4), (44, 41, 38), -1)
                cv2.rectangle(img, (x0, y + 2), (x0 + 4, y + h - 4), COL[who], -1)
                cv2.putText(img, f"{'A' if who == 'A' else 'B'}  {ti:5.1f} s", (x0 + 12, y + 18), FONT, 0.4, COL[who], 1, cv2.LINE_AA)
                for n, line in enumerate(lines):
                    cv2.putText(img, line, (x0 + 12, y + 36 + 20 * n), FONT, 0.45, colour if colour != COL[who] else INK, 1, cv2.LINE_AA)
            y -= 4
        return img
