"""Showcase video: one 1920x1080 frame per moment of a recording, laid out for people, not for debugging.

    brain/.venv/bin/python tools/showcase.py data/recordings/unity_haze_agents_v1 --run data/runs/unity_haze_agents_v1 \
        --speed 3 --out data/evidence/showcase_agents_v1.mp4

Layout: header with the sim clock; the two robots' cameras side by side, each with a thermal picture-in-picture and
its brain's live state; under them the robots' live sensor maps, or the SLAM map for older recordings, and the shared
context (each robot's own estimate on the real building plan, sectors, radio link); on the right the team radio:
agent decisions with the OpenCV tools they used, radio messages, survivors, map merges. Text is SF Pro, drawn with
PIL so it stays sharp. Reads only recorded outputs; nothing here changes what the robots did.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'brain'))
from brain.nav_io import (load_scenario, recording_frame, transform_xy, transform_poses,
                          map_geometry, rotation, sample_grid, frame_peer)
from make_walkthrough import frames, ironbow, narration, thermal_frames  # noqa: E402

W, H = 1920, 1080
BG, PANEL, EDGE = (14, 15, 18), (22, 24, 29), (40, 44, 52)
TEXT, DIM, FAINT = (236, 238, 242), (150, 156, 168), (98, 104, 116)
ROBOT = [(245, 165, 36), (46, 168, 230)]        # RGB: A amber, B cyan
GOOD, WARN, CV = (74, 222, 128), (239, 68, 68), (110, 231, 183)
SF = "/System/Library/Fonts/SFNS.ttf"


def font(size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(SF, size)
    try:
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f


F = {k: font(*v) for k, v in {"title": (30, "Bold"), "sub": (17, "Regular"), "clock": (34, "Semibold"),
                              "h": (18, "Semibold"), "body": (17, "Regular"), "small": (14, "Regular"),
                              "tiny": (12, "Medium"), "chip": (13, "Semibold"), "badge": (16, "Semibold")}.items()}


def panel(d: ImageDraw.ImageDraw, box, title: str | None = None, note: str | None = None):
    d.rounded_rectangle(box, 12, fill=PANEL, outline=EDGE, width=1)
    if title:
        d.text((box[0] + 16, box[1] + 12), title.upper(), font=F["h"], fill=TEXT)
        if note:
            tw = d.textlength(title.upper(), font=F["h"])
            d.text((box[0] + 28 + tw, box[1] + 15), note, font=F["small"], fill=FAINT)


def pill(d, xy, text, fill, fg=(14, 15, 18), f="chip"):
    w = d.textlength(text, font=F[f]) + 16
    d.rounded_rectangle((xy[0], xy[1], xy[0] + w, xy[1] + 22), 11, fill=fill)
    d.text((xy[0] + 8, xy[1] + 3), text, font=F[f], fill=fg)
    return xy[0] + w + 6


def wrap(d, text, f, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = f"{cur} {w}".strip()
        if d.textlength(test, font=F[f]) > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = test
    return lines + ([cur] if cur else [])


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


class LiveMaps:
    """Place recorded grids and sensors on the plan using known starts for display."""

    @staticmethod
    def available(rec, robots):
        return any((rec / r / 'brain_map.txt').exists() and
                   any((rec / r / 'brain_map').glob('*.png')) for r in robots)

    def __init__(self, rec, meta, robots, states):
        self.rec, self.robots, self.states = rec, robots, states
        scenario = load_scenario(meta)
        self.frames = {r: recording_frame(states[r], scenario, r) for r in robots}
        x0, y0, x1, y1 = scenario['search_area']
        starts = [transform_poses([[states[r][0]['x'], states[r][0]['y'], states[r][0]['yaw_rad']]], self.frames[r][0])[0, :2]
                  for r in robots if states[r]]
        extent = np.array([[x0, y0], [x1, y1], *starts])
        self.bounds = np.r_[extent.min(axis=0) - 1, extent.max(axis=0) + 1]
        self.maps, self.map_times, self.state_times, self.poses, self.lidar = {}, {}, {}, {}, {}
        self.decoded, self.base_key, self.base_image = {}, None, None
        for r in robots:
            rd = rec / r
            index = rd / 'brain_map.txt'
            rows = [line.split() for line in index.read_text().splitlines()
                    if line.strip() and not line.startswith('#')] if index.exists() else []
            self.maps[r] = sorted((float(row[0]), rd / row[4]) for row in rows)
            self.map_times[r] = np.array([row[0] for row in self.maps[r]])
            self.state_times[r] = np.array([s['t'] for s in states[r]])
            poses = np.array([[s['x'], s['y'], s['yaw_rad']] for s in states[r]], float).reshape(-1, 3)
            poses = transform_poses(poses, self.frames[r][0])
            poses[:, 2] = np.unwrap(poses[:, 2])
            self.poses[r] = poses
            path = rd / 'lidar.txt'
            if not path.exists():
                self.lidar[r] = None
                continue
            with path.open() as stream:
                comments = []
                for line in stream:
                    if line.startswith('#'):
                        comments.append(line)
                    elif line.strip():
                        break
                header = ''.join(comments)
            fields = dict(re.findall(r'(\w+)\s+([-+\d.eE]+)', header))
            # Parse the numeric table once, rather than splitting 15 MB every frame.
            table = np.loadtxt(path, dtype=np.float64, ndmin=2)
            n = int(fields['beams'])
            if table.shape[1] != n + 1:
                raise ValueError(f'{path}: expected {n} LiDAR ranges per sweep')
            angles = float(fields['angle_min_rad']) + np.arange(n) * float(fields['increment_rad'])
            self.lidar[r] = (table[:, 0], (table[:, 1:] * .001).astype(np.float32),
                             angles, float(fields['mount_forward_m']))

    def map_indices(self, t):
        return tuple(int(np.searchsorted(self.map_times[r], t, side='right')) - 1 for r in self.robots)

    def pose(self, r, t):
        times = self.state_times[r]
        if not len(times) or t < times[0]:
            return None
        # After logging stops, hold the last estimate rather than extrapolating.
        return np.array([np.interp(t, times, self.poses[r][:, i]) for i in range(3)])

    def sweep(self, r, t):
        scan = self.lidar[r]
        if scan is None:
            return None
        times, ranges, angles, forward = scan
        i = int(np.searchsorted(times, t, side='right')) - 1
        if i < 0 or not len(self.state_times[r]) or times[i] > self.state_times[r][-1]:
            return None
        pose = self.pose(r, times[i])
        if pose is None:
            return None
        x, y, yaw = pose
        origin = np.array([x - forward * np.sin(yaw), y + forward * np.cos(yaw)])
        valid = ranges[i] > 0
        a = yaw + angles[valid]
        hits = origin + np.column_stack((-np.sin(a), np.cos(a))) * ranges[i, valid, None]
        return origin, hits

    def pixels(self, xy, size):
        x0, y0, x1, y1 = self.bounds
        xy = np.asarray(xy)
        return np.rint((xy - [x0, y1]) * [size[0] / (x1 - x0), -size[1] / (y1 - y0)]).astype(np.int32)

    def base(self, t, size):
        indices = self.map_indices(t)
        key = indices, size
        if key == self.base_key:
            return self.base_image
        x0, y0, x1, y1 = self.bounds
        xs = x0 + (np.arange(size[0]) + .5) * (x1 - x0) / size[0]
        ys = y1 - (np.arange(size[1]) + .5) * (y1 - y0) / size[1]
        xx, yy = np.meshgrid(xs, ys)
        building = np.stack([xx, yy], axis=-1)
        occupied = np.zeros((size[1], size[0]), bool)
        free = np.zeros_like(occupied)
        tint = np.zeros((*occupied.shape, 3), float)
        count = np.zeros_like(occupied, dtype=np.uint8)
        for k, (r, i) in enumerate(zip(self.robots, indices)):
            if i < 0:
                continue
            path = self.maps[r][i][1]
            if path not in self.decoded:
                grid = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if grid is None or not np.isin(grid, [0, 128, 255]).all():
                    raise ValueError(f'Invalid brain map {path}')
                self.decoded[path] = grid
            grid = self.decoded[path]
            origin, cell = map_geometry(grid.shape, self.frames[r][1])
            transform = self.frames[r][0]
            own = (building - transform[:2]) @ rotation(transform[2])
            crop = sample_grid(np.flipud(grid), own, origin, cell)
            occupied |= crop == 0
            mask = crop == 255
            free |= mask
            tint[mask] += np.array(ROBOT[k]) * .22 + np.array(PANEL) * .78
            count[mask] += 1
        image = np.full((*occupied.shape, 3), PANEL, np.uint8)
        image[free] = (tint[free] / count[free, None]).astype(np.uint8)
        image[occupied] = TEXT
        self.base_key, self.base_image = key, image
        return image

    def render(self, t, available_size):
        x0, y0, x1, y1 = self.bounds
        scale = min(available_size[0] / (x1 - x0), available_size[1] / (y1 - y0))
        size = (round((x1 - x0) * scale), round((y1 - y0) * scale))
        img = self.base(t, size).copy()
        for k, r in enumerate(self.robots):
            colour = ROBOT[k]
            scan = self.sweep(r, t)
            if scan is not None:
                origin, hits = scan
                p = self.pixels(origin, size)
                ends = self.pixels(hits, size)
                segments = np.stack((np.broadcast_to(p, ends.shape), ends), axis=1)
                ray_colour = tuple(int(v * .3 + PANEL[i] * .7) for i, v in enumerate(colour))
                # Keep every return point, but space ray lines so the grid stays readable.
                cv2.polylines(img, list(segments[::4]), False, ray_colour, 1, cv2.LINE_AA)
                inside = (ends[:, 0] >= 0) & (ends[:, 0] < size[0]) & (ends[:, 1] >= 0) & (ends[:, 1] < size[1])
                img[ends[inside, 1], ends[inside, 0]] = colour
        labels = []
        for k, r in enumerate(self.robots):
            colour = ROBOT[k]
            pose = self.pose(r, t)
            if pose is None:
                continue
            si = int(np.searchsorted(self.state_times[r], t, side='right')) - 1
            state = self.states[r][si]
            if state.get('path'):
                cv2.polylines(img, [self.pixels(transform_xy(state['path'], self.frames[r][0]), size)], False, colour, 1, cv2.LINE_AA)
            if state.get('goal'):
                cv2.drawMarker(img, self.pixels(transform_xy(state['goal'], self.frames[r][0]), size), colour, cv2.MARKER_DIAMOND, 12, 2)
            p = self.pixels(pose[:2], size)
            tip = p + np.rint(17 * np.array([-np.sin(pose[2]), -np.cos(pose[2])])).astype(int)
            cv2.circle(img, p, 5, colour, -1, cv2.LINE_AA)
            cv2.arrowedLine(img, p, tip, TEXT, 2, cv2.LINE_AA, tipLength=.35)
            labels.append((tuple(p + [8, -15]), 'AB'[k], colour))
        result = Image.fromarray(img)
        draw = ImageDraw.Draw(result)
        draw.text((8, 4), 'N ↑', font=F['small'], fill=DIM)
        for position, label, colour in labels:
            draw.text(position, label, font=F['tiny'], fill=colour)
        return result


class Radio:
    """Team radio items: agent decisions with tools and orders, radio texts, survivors, merges, link changes."""

    def __init__(self, rec: Path, run: Path | None, robots: list[str]):
        self.items = []    # (t, side, kind, text, chips)
        side = {r: k for k, r in enumerate(robots)}
        tools_by_step = {}
        for e in jsonl(rec / "agent_trace.jsonl"):
            if e["kind"] == "tool" and e.get("name") not in ("finish",):
                tools_by_step.setdefault(e["step"], []).append(e)
            if e["kind"] == "step_end":
                who = e["step"].split("@")[0]
                t = float(e["step"].split("@")[1])
                used = [x for x in tools_by_step.get(e["step"], []) if x["name"] in
                        ("scan_thermal", "inspect_candidate", "assess_visibility", "track_fire_spread", "measure_opening")]
                orders = [o["type"] for o in e.get("orders", [])]
                if who == "commander":
                    if orders:
                        self.items.append((t, -1, "commander", "Commander: " + (e.get("summary") or "relayed a message"), []))
                    continue
                # Team messages come from the robot's own team_message_sent events below; listing the order here as
                # well showed every message twice.
                if not used and not orders:
                    continue
                chips = [("OpenCV " + x["name"] + (f" = {x['verdict']}" if x.get("verdict") else ""), CV) for x in used]
                chips += [(o.replace("_", " "), ROBOT[side.get(who, 0)]) for o in orders if o != "send_team_message"]
                chips += [("REFUSED", WARN)] if e.get("refusals") else []
                self.items.append((t, side.get(who, 0), "agent", e.get("summary") or "", chips))
        for r in robots:
            for e in jsonl(rec / r / "decisions.jsonl"):
                k, t = e["kind"], float(e["t"])
                if k == "team_message_sent":
                    self.items.append((t, side[r], "radio", "“" + e["reason"] + "”", [("radio out", DIM)]))
                elif k == "survivor_confirmed":
                    self.items.append((t, -1, "good", f"SURVIVOR CONFIRMED by {'AB'[side[r]]}", []))
                elif k == "candidate_rejected":
                    self.items.append((t, side[r], "agent", "OpenCV check: that warm shape is not a person. Ignoring it.", [("reject candidate", ROBOT[side[r]])]))
                elif k == "frame_linked":
                    peer = frame_peer(e, r, robots)
                    label = peer.removeprefix('robot_') if peer else 'its partner'
                    self.items.append((t, side[r], "frame_link", f"{'AB'[side[r]]} linked its map with {label} from matched LiDAR fingerprints", [("LiDAR frame linked", GOOD)]))
                elif k == "map_shared":
                    self.items.append((t, -1, "good", f"{'AB'[side[r]]} now holds its partner's map: no double search", []))
                elif k == "stuck":
                    self.items.append((t, side[r], "warn", "Legs not moving: backing off and replanning", []))
        last = {}
        rp = rec / "brain_radio.txt"
        if rp.exists():
            for line in rp.read_text().splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                t, frm, to, _, size, _, status, _ = line.split()[:8]
                state = "up" if status == "sent" else status.replace("queued_", "")
                if frm == robots[0] and last.get(frm) != state:
                    self.items.append((float(t), -1, "link", "Radio link up: maps flowing" if state == "up"
                                       else f"Radio link down ({state}): newest map waits", []))
                    last[frm] = state
        self.items.sort(key=lambda x: x[0])

    def draw(self, d, box, t):
        panel(d, box, "Team radio", "agents, OpenCV evidence, radio")
        x0, y_top, x1, y = box[0] + 14, box[1] + 48, box[2] - 14, box[3] - 12
        for ti, side, kind, text, chips in reversed([i for i in self.items if i[0] <= t]):
            width = x1 - x0 - 24
            lines = wrap(d, text, "body", width)[:4]
            h = 30 + 22 * len(lines) + (28 if chips else 0)
            if y - h < y_top:
                break
            y -= h
            if side < 0:
                colour = GOOD if kind == "good" else (FAINT if kind == "link" else DIM)
                d.text((x0, y + 4), f"{ti:5.1f} s", font=F["tiny"], fill=FAINT)
                for n, line in enumerate(lines):
                    d.text((x0 + 56, y + 2 + 22 * n), line, font=F["body"] if kind != "link" else F["small"], fill=colour)
                y -= 6
                continue
            c = ROBOT[side]
            d.rounded_rectangle((x0, y, x1, y + h - 6), 8, fill=(30, 33, 40))
            d.rounded_rectangle((x0, y, x0 + 4, y + h - 6), 2, fill=c)
            d.text((x0 + 14, y + 6), f"{'AB'[side]}  ·  {ti:5.1f} s", font=F["tiny"], fill=c)
            for n, line in enumerate(lines):
                d.text((x0 + 14, y + 24 + 22 * n), line, font=F["body"], fill=TEXT if kind != "radio" else (200, 214, 230))
            cx = x0 + 14
            for label, colour in chips[:4]:
                if cx > x1 - 120:
                    break
                cx = pill(d, (cx, y + 26 + 22 * len(lines)), label, colour)
            y -= 6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("recording", type=Path)
    ap.add_argument("--run", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--speed", type=float, default=3.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-t", type=float, help="stop at this simulation time in seconds")
    ap.add_argument("--title", default="Smoke Rescue")
    a = ap.parse_args()
    if a.speed <= 0 or a.fps <= 0 or (a.max_t is not None and a.max_t <= 0):
        ap.error('speed, fps and max-t must be positive')
    rec = a.recording
    meta = json.loads((rec / "meta.json").read_text())
    robots = meta["robots"][:2]
    out = a.out or Path("data/evidence") / f"showcase_{meta['run_id']}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    tracks = {r: frames(rec / r) for r in robots}
    thermals = {r: thermal_frames(rec / r) for r in robots}
    states = {r: jsonl(rec / r / "brain_state.jsonl") for r in robots}
    live_maps = LiveMaps(rec, meta, robots, states) if LiveMaps.available(rec, robots) else None
    torch = {r: [(e["t"], e["reason"]) for e in jsonl(rec / r / "decisions.jsonl") if e["kind"] == "agent_order" and "torch set to" in e["reason"]] for r in robots}
    captions = {r: narration(a.run, r) for r in robots} if a.run else {r: [] for r in robots}
    radio = Radio(rec, a.run, robots)
    from contextpanel import ContextPanel
    context = ContextPanel(rec)
    minimap = None
    if live_maps is None and a.run and (a.run / "merge").exists():
        from minimap import MiniMap
        minimap = MiniMap(rec, a.run)
    end = max(fr[-1][0] for fr in tracks.values())
    if a.max_t is not None:
        end = min(end, a.max_t)
    times = np.arange(0, end, a.speed / a.fps)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                           "-preset", "medium", str(out)], stdin=subprocess.PIPE)
    idx = {r: 0 for r in robots}
    cache = {}
    for t in times:
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        d.text((24, 16), a.title.upper(), font=F["title"], fill=TEXT)
        d.text((24 + d.textlength(a.title.upper(), font=F["title"]) + 18, 26),
               "two quadrupeds in a burning building  ·  OpenCV 5 perception  ·  LLM agents  ·  maps shared over a lossy radio",
               font=F["sub"], fill=DIM)
        clock = f"{int(t // 60):02d}:{t % 60:04.1f}"
        d.text((W - 300, 14), clock, font=F["clock"], fill=TEXT)
        pill(d, (W - 140, 22), f"{a.speed:g}x speed", (52, 58, 70), TEXT)
        # Cameras
        for k, r in enumerate(robots):
            box = (16 + k * 656, 72, 16 + k * 656 + 640, 72 + 480)
            fr = tracks[r]
            while idx[r] + 1 < len(fr) and fr[idx[r] + 1][0] <= t + 1e-4:
                idx[r] += 1
            ti, rgb_p, _ = fr[idx[r]]
            if cache.get(r, (None,))[0] != ti:
                rgb = cv2.cvtColor(cv2.imread(str(rgb_p)), cv2.COLOR_BGR2RGB)
                tp = thermals[r].get(round(ti, 4))
                th, hot = None, None
                if tp is not None and tp.exists():
                    dk = cv2.imread(str(tp), cv2.IMREAD_UNCHANGED)
                    th = cv2.cvtColor(cv2.resize(ironbow(dk), (200, 160), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                    hot = dk.max() / 10 - 273.15
                cache[r] = (ti, rgb, th, hot)
            _, rgb, th, hot = cache[r]
            cam = Image.fromarray(rgb)
            mask = Image.new("L", (640, 480), 0)
            ImageDraw.Draw(mask).rounded_rectangle((0, 0, 639, 479), 12, fill=255)
            im.paste(cam, box[:2], mask)
            c = ROBOT[k]
            d.rounded_rectangle((box[0] + 12, box[1] + 12, box[0] + 12 + 34, box[1] + 46), 8, fill=c)
            d.text((box[0] + 22, box[1] + 15), "AB"[k], font=F["clock"].font_variant(size=24), fill=BG)
            d.text((box[0] + 56, box[1] + 18), "front camera", font=F["badge"], fill=TEXT, stroke_width=2, stroke_fill=(0, 0, 0))
            if th is not None:
                tb = (box[2] - 214, box[3] - 174)
                d.rounded_rectangle((tb[0] - 2, tb[1] - 2, tb[0] + 202, tb[1] + 162), 8, fill=(0, 0, 0))
                im.paste(Image.fromarray(th), tb)
                d.text((tb[0] + 6, tb[1] + 4), f"THERMAL  max {hot:.0f}°C", font=F["tiny"], fill=TEXT, stroke_width=2, stroke_fill=(0, 0, 0))
            st = [s for s in states[r] if s["t"] <= t]
            x = box[0] + 12
            if st:
                x = pill(d, (x, box[3] - 34), st[-1].get("state", "").upper(), (36, 40, 48), TEXT)
                x = pill(d, (x, box[3] - 34), ("THERMAL VISION" if st[-1].get("mode") == "thermal" else "COLOUR VISION"), (36, 40, 48), TEXT)
            tor = [m for tt, m in torch[r] if tt <= t]
            if tor:
                mode = tor[-1].split("torch set to ")[1].split(",")[0]
                pill(d, (x, box[3] - 34), f"TORCH {mode.upper()}", (250, 204, 21))
            said = [txt for ct, txt in captions[r] if ct <= t < ct + 6]
            if said:
                lines = wrap(d, said[-1], "body", 600)[:2]
                y0 = box[3] - 76 - 22 * len(lines)
                d.rounded_rectangle((box[0] + 10, y0 - 6, box[2] - 224, y0 + 22 * len(lines) + 4), 8, fill=(0, 0, 0))
                for n, line in enumerate(lines):
                    d.text((box[0] + 20, y0 + 22 * n), line, font=F["body"], fill=TEXT)
        # Maps
        mbox = (16, 568, 16 + 640, H - 16)
        if live_maps is not None:
            panel(d, mbox, "Live maps")
            d.text((mbox[0] + 16, mbox[1] + 38),
                   "known start pose placement for display only", font=F['small'], fill=FAINT)
            m = live_maps.render(t, (mbox[2] - mbox[0] - 24, mbox[3] - mbox[1] - 112))
            im.paste(m, (mbox[0] + (mbox[2] - mbox[0] - m.width) // 2,
                         mbox[1] + 62 + (mbox[3] - mbox[1] - 112 - m.height) // 2))
        else:
            panel(d, mbox, "SLAM map", "what the robots built, merged when they meet")
        if live_maps is None and minimap is not None:
            m = minimap.render(t)[44:640 - 150, :, ::-1]
            mh, mw = m.shape[:2]
            s = min((mbox[2] - mbox[0] - 24) / mw, (mbox[3] - mbox[1] - 56) / mh)
            m = cv2.resize(np.ascontiguousarray(m), (int(mw * s), int(mh * s)), interpolation=cv2.INTER_AREA)
            im.paste(Image.fromarray(m), (mbox[0] + (mbox[2] - mbox[0] - m.shape[1]) // 2, mbox[1] + 46))
        elif live_maps is None:
            d.text((mbox[0] + 16, mbox[1] + 60), "SLAM pipeline not run for this recording", font=F["body"], fill=FAINT)
        if live_maps is not None:
            d.text((mbox[0] + 16, mbox[3] - 34), "placed on the plan with the known start pose for display;", font=F['tiny'], fill=DIM)
            d.text((mbox[0] + 16, mbox[3] - 18), "robots link frames from LiDAR fingerprints", font=F['tiny'], fill=DIM)
        cbox = (672, 568, 672 + 640, H - 16)
        panel(d, cbox, "Shared context", "known start pose placement for display only")
        c = context.render(t)[56:56 + context.mh + 4, 6:6 + context.mw + 8, ::-1]
        s = min((cbox[2] - cbox[0] - 24) / c.shape[1], (cbox[3] - cbox[1] - 132) / c.shape[0])
        c = cv2.resize(np.ascontiguousarray(c), (int(c.shape[1] * s), int(c.shape[0] * s)), interpolation=cv2.INTER_AREA)
        im.paste(Image.fromarray(c), (cbox[0] + (cbox[2] - cbox[0] - c.shape[1]) // 2, cbox[1] + 44))
        yl = cbox[1] + 50 + c.shape[0]
        for k, r in enumerate(robots):
            ev = [e for e in context.events[r] if e["t"] <= t]
            got = sum(1 for e in ev if e["kind"] == "peer_message_received")
            found = sum(1 for e in ev if e["kind"] == "survivor_confirmed")
            d.text((cbox[0] + 16 + k * 300, yl), f"{'AB'[k]}: partner maps {got}  ·  survivors {found}", font=F["small"], fill=ROBOT[k])
        d.text((cbox[0] + 16, cbox[3] - 34), "placed on the plan with the known start pose for display;", font=F['tiny'], fill=DIM)
        d.text((cbox[0] + 16, cbox[3] - 18), "robots link frames from LiDAR fingerprints", font=F['tiny'], fill=DIM)
        radio.draw(d, (1328, 72, W - 16, H - 16), t)
        ff.stdin.write(np.asarray(im).tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError('ffmpeg failed to encode the video')
    print(f"{out}: {len(times)} frames, {len(times) / a.fps:.0f} s at {a.speed:g}x")


if __name__ == "__main__":
    main()
