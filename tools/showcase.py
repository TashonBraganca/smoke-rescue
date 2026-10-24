"""Showcase video: one 1920x1080 frame per moment of a recording, laid out for people, not for debugging.

    brain/.venv/bin/python tools/showcase.py data/recordings/unity_house_onedoor_seed934_brain_final_1954 \
        --speed 3 --title "Smoke Rescue: seed 934" --out data/evidence/showcase_onedoor_seed934_final_v2.mp4

Layout: a header with the title, sim clock and speed; the two robots' front cameras large across the top, each with a
thermal picture-in-picture and its brain's live state; under them one floor plan of the searched house and the team
radio. The plan draws the true walls of that one house as a backdrop for viewers (the robots never see it) and on it
only what the robots recorded: one robot's team map (floor each robot searched, floor mapped but not yet searched,
walls lit where the map reaches them), fire cells from the brains' state, the people they confirmed, and each robot's
pose, path and camera view. Text is SF Pro, drawn with PIL so it stays sharp. Reads only recorded outputs; nothing
here changes what the robots did.
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
from brain.nav_eval import true_wall_segments
from brain.compare_runs import score_confirmations
from brain.team_map import latest_team_map, load_team_map
from make_walkthrough import frames, ironbow, narration, thermal_frames  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
W, H = 1920, 1080
BG, PANEL, EDGE = (11, 9, 16), (19, 17, 26), (40, 36, 52)
TEXT, DIM, FAINT = (238, 236, 242), (160, 155, 174), (108, 103, 124)
ROBOT = [(245, 165, 36), (46, 168, 230)]        # RGB: A amber, B cyan
GOOD, WARN, CV = (74, 222, 128), (239, 68, 68), (110, 231, 183)
WATER = (96, 140, 255)
FIRE = (255, 86, 48)
# The floor plan follows the owner's web UI: a near-black floor with a faint dot grid until the team map reaches it,
# walls as light rounded strokes once mapped (dim before), searched floor in a deep tint of the robot that searched it.
PLAN_BG, PLAN_DOTS = (15, 12, 21), (54, 48, 66)
WALL_DIM, WALL_LIT = (46, 42, 56), (188, 180, 174)
MAPPED = (40, 37, 50)
SEARCHED = {1: (112, 52, 30), 2: (26, 62, 120), 3: (86, 42, 118)}     # robot bits: A only, B only, both
NOZZLE = json.loads((ROOT / "config" / "actuators.json").read_text())["nozzle"]
# Rooms under the brain's walk-in minimum are map slivers between walls, not rooms a viewer would recognise.
WALK_IN_MIN_M2 = json.loads((ROOT / "config" / "brain.json").read_text())["room_walk_in_min_m2"]
SF = "/System/Library/Fonts/SFNS.ttf"
SF_MONO = "/System/Library/Fonts/SFNSMono.ttf"
FRAME_NOTE = "robots link frames from LiDAR fingerprints"
PLAN_NOTE = "Walls come from the true house plan, drawn for viewers. The robots never see it."

# Frame layout. The cameras keep their full width and lose only the top sixth (ceiling), so a person on the floor
# right in front of a robot stays in view.
HEADER = 64
CAM_W, CAM_H = 930, 581
CAM_X = (20, 970)
LOW_Y = HEADER + CAM_H + 16
PLAN_BOX = (20, LOW_Y, 950, H - 16)
RADIO_BOX = (970, LOW_Y, 1900, H - 16)
STATE_WORDS = {"explore": "Exploring", "retreat": "Backing off from heat", "survivor": "Checking a person",
               "complete": "Search complete", "return": "Heading out"}


def font(size: int, weight: str = "Regular", path: str = SF) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(path, size)
    try:
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f


F = {k: font(*v) for k, v in {"title": (30, "Bold"), "sub": (17, "Regular"), "clock": (34, "Semibold"),
                              "h": (18, "Semibold"), "body": (17, "Regular"), "small": (14, "Regular"),
                              "tiny": (12, "Medium"), "chip": (13, "Semibold"), "badge": (16, "Semibold"),
                              "panel": (19, "Semibold"), "row": (19, "Regular"), "quiet": (16, "Regular"), "rowb": (21, "Bold"),
                              "stat": (38, "Semibold"), "legend": (16, "Regular"), "label": (18, "Semibold"),
                              "letter": (24, "Bold"), "pill": (16, "Medium"), "dot": (16, "Bold"),
                              "mono": (15, "Medium", SF_MONO), "clockmono": (32, "Medium", SF_MONO),
                              "monosmall": (13, "Regular", SF_MONO)}.items()}


def mix(a, b, share):
    """share of colour a over colour b."""
    return tuple(int(round(x * share + y * (1 - share))) for x, y in zip(a, b))


def panel(d: ImageDraw.ImageDraw, box, title: str | None = None, note: str | None = None):
    d.rounded_rectangle(box, 14, fill=PANEL, outline=EDGE, width=1)
    if title:
        d.text((box[0] + 20, box[1] + 14), title.upper(), font=F["panel"], fill=TEXT)
        if note:
            tw = d.textlength(title.upper(), font=F["panel"])
            d.text((box[0] + 32 + tw, box[1] + 18), note, font=F["small"], fill=FAINT)


def confirmation_label(report, robot, t):
    rows = [c for c in report['confirmation_details']
            if c['t'] <= t and (robot is None or c['robot'] == robot)]
    people = {c['survivor'] for c in rows if c['survivor'] is not None}
    return f'confirmations {len(rows)}; distinct people {len(people)}'


def pill(d, xy, text, fill, fg=(14, 15, 18), f="chip"):
    w = d.textlength(text, font=F[f]) + 16
    d.rounded_rectangle((xy[0], xy[1], xy[0] + w, xy[1] + 22), 11, fill=fill)
    d.text((xy[0] + 8, xy[1] + 3), text, font=F[f], fill=fg)
    return xy[0] + w + 6


def chip(d, right, cy, text, colour):
    """A quiet tag: tinted ground, coloured text, right edge at `right`. Returns its left edge."""
    w = d.textlength(text, font=F["chip"]) + 18
    d.rounded_rectangle((right - w, cy - 11, right, cy + 11), 6, fill=mix(colour, PANEL, .17))
    d.text((right - w + 9, cy - 8), text, font=F["chip"], fill=mix(colour, (255, 255, 255), .8))
    return right - w


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


def clock(t: float, tenths: bool = False) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}" if tenths else f"{int(t // 60):02d}:{int(t % 60):02d}"


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def team_map_rgb(grid, at=None):
    """Colour only the exported robot belief. Wall brightness follows occupancy confidence.
    Free-floor colour follows camera search confidence; unseen floor stays dark."""
    image = np.full((grid.size, grid.size, 3), PANEL, np.uint8)
    tint = np.zeros_like(image, dtype=float)
    contributors = np.zeros((grid.size, grid.size), int)
    for i, source in enumerate(grid.sources):
        mask = (grid.source_mask & (1 << i)) != 0
        colour = ROBOT[1 if source.endswith('B') or source == 'B' else 0]
        tint[mask] += colour
        contributors[mask] += 1
    known = contributors > 0
    tint[known] /= contributors[known, None]
    certainty = np.abs(grid.occupancy - .5) * 2
    age = 0 if at is None else max(0, at - grid.at)
    search = grid.searched_confidence * .5 ** (age / 60)
    free = known & (grid.occupancy <= .5)
    strength = np.maximum(certainty * .18, search * .75)
    image[free] = (np.array(PANEL) + (tint[free] - PANEL) * strength[free, None]).astype(np.uint8)
    wall = known & (grid.occupancy > .5)
    image[wall] = (np.array(PANEL) + (np.array(TEXT) - PANEL) * certainty[wall, None]).astype(np.uint8)
    return image


class LiveMaps:
    """Place recorded grids and sensors on the plan using known starts for display."""

    @staticmethod
    def available(rec, robots):
        return any((rec / r / 'brain_map.txt').exists() and
                   any((rec / r / 'brain_map').glob('*.png')) for r in robots)

    def __init__(self, rec, meta, robots, states):
        self.rec, self.robots, self.states = rec, robots, states
        # Sprays as the recorder logged them: nozzle position and heading in house coordinates (heading 0 = north,
        # positive toward east), drawn as the nozzle's cone while the water flows.
        self.water = jsonl(rec / "water.jsonl")
        scenario = load_scenario(meta)
        # How the robots' frames relate, for the caption: one gate (launch pads) or linked by LiDAR fingerprints.
        global FRAME_NOTE
        if any(r.get('launch_pad') is not None for r in scenario.get('robots', [])):
            FRAME_NOTE = "robots share one frame from their launch pads at the door"
        self.frames = {r: recording_frame(states[r], scenario, r) for r in robots}
        x0, y0, x1, y1 = scenario['search_area']
        starts = [transform_poses([[states[r][0]['x'], states[r][0]['y'], states[r][0]['yaw_rad']]], self.frames[r][0])[0, :2]
                  for r in robots if states[r]]
        extent = np.array([[x0, y0], [x1, y1], *starts])
        self.bounds = np.r_[extent.min(axis=0) - 1, extent.max(axis=0) + 1]
        self.maps, self.map_times, self.state_times, self.poses, self.lidar = {}, {}, {}, {}, {}
        self.decoded, self.base_key, self.base_image = {}, None, None
        self.team_cache = {}
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
        receiver, belief = self.team_snapshot(t)
        key = indices, size, receiver, None if belief is None else belief.at, int(t * 2)
        if key == self.base_key:
            return self.base_image
        x0, y0, x1, y1 = self.bounds
        xs = x0 + (np.arange(size[0]) + .5) * (x1 - x0) / size[0]
        ys = y1 - (np.arange(size[1]) + .5) * (y1 - y0) / size[1]
        xx, yy = np.meshgrid(xs, ys)
        building = np.stack([xx, yy], axis=-1)
        if belief is not None:
            transform = self.frames[receiver][0]
            own = (building - transform[:2]) @ rotation(transform[2])
            cells = np.floor((own - belief.origin) / belief.cell_m).astype(int)
            valid = (cells[..., 0] >= 0) & (cells[..., 0] < belief.size) & (cells[..., 1] >= 0) & (cells[..., 1] < belief.size)
            image = np.full((size[1], size[0], 3), PANEL, np.uint8)
            colours = team_map_rgb(belief, t)
            image[valid] = colours[cells[..., 1][valid], cells[..., 0][valid]]
            self.base_key, self.base_image = key, image
            return image
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

    def team_snapshot(self, t):
        # One receiver's actual team state, rather than an evaluator's union of all robots' grids.
        # Refresh every half second so a watcher can pick up a file just finished by the recorder.
        key = int(t * 2)
        if key not in self.team_cache:
            found = [(r, latest_team_map(self.rec / r, t)) for r in self.robots]
            found = [(r, g) for r, g in found if g is not None]
            self.team_cache = {key: max(found, key=lambda item: item[1].at) if found else (None, None)}
        return self.team_cache[key]

    def belief_caption(self, t):
        receiver, grid = self.team_snapshot(t)
        if grid is None:
            return 'legacy binary grids; confidence was not recorded'
        return f"{receiver}'s team belief at {grid.at:.1f}s; bright = confident"

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
        for w in self.water:
            if not (w["t"] <= t < w["t"] + w["seconds"]):
                continue
            h, half, reach = np.radians(w["heading_deg"]), np.radians(NOZZLE["half_angle_deg"]), NOZZLE["range_m"]
            fan = [[w["x"], w["z"]]] + [[w["x"] + reach * np.sin(a), w["z"] + reach * np.cos(a)] for a in np.linspace(h - half, h + half, 7)]
            layer = img.copy()
            cv2.fillPoly(layer, [self.pixels(fan, size)], WATER, cv2.LINE_AA)
            img = cv2.addWeighted(layer, .45, img, .55, 0)
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



def robot_bits(grid, robots) -> np.ndarray:
    """Per cell, which robots searched it, as bit k for robots[k] (the export's own bit order follows its sources)."""
    bits = np.zeros(grid.searched_source_mask.shape, np.uint8)
    for i, source in enumerate(grid.sources):
        k = robots.index(source) if source in robots else 'AB'.find(source[-1:])
        if 0 <= k < 2:
            bits |= (((grid.searched_source_mask >> i) & 1) << k).astype(np.uint8)
    return bits


def tidy_bits(bits: np.ndarray) -> np.ndarray:
    """For drawing only: close one-cell holes in the searched area (adds about 5 % to its cells on seed 934, removes
    none) and give each searched cell the most common owner among its neighbours, so the fill reads as floor rather
    than noise. The square metres in the legend come from the raw export."""
    searched = (bits != 0).astype(np.uint8)
    closed = cv2.morphologyEx(searched, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    scores = np.stack([cv2.blur((bits == c).astype(np.float32), (3, 3)) for c in (1, 2, 3)])
    out = (np.argmax(scores, axis=0) + 1).astype(np.uint8)
    out[~closed] = 0
    return out


def glow(img: np.ndarray, centre, colour, sigma: float, strength: float):
    """Light a float image softly around a point, toward colour."""
    r = int(3 * sigma)
    x, y = int(round(centre[0])), int(round(centre[1]))
    h, w = img.shape[:2]
    xa, xb, ya, yb = max(0, x - r), min(w, x + r + 1), max(0, y - r), min(h, y + r + 1)
    if xa >= xb or ya >= yb:
        return
    yy, xx = np.mgrid[ya:yb, xa:xb]
    g = strength * np.exp(-((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2) / (2 * sigma ** 2))
    patch = img[ya:yb, xa:xb]
    patch += g[..., None] * (np.array(colour, np.float32) - patch)


def camera_hfov(robot_dir: Path) -> float:
    fx = width = None
    path = robot_dir / 'camera.yaml'
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip().startswith('Camera1.fx:'):
                fx = float(line.split(':')[1])
            elif line.strip().startswith('Camera.width:'):
                width = float(line.split(':')[1])
    return 2 * np.arctan(width / 2 / fx) if fx and width else np.radians(75)


class HousePlan:
    """One floor plan of the house the robots search, cropped to the scenario's search area.

    Backdrop (viewer only): the true walls of that house and a dot grid on its floor. On it, only recorded robot output:
    one receiver's team map, the brains' fire cells, confirmed people, poses, paths and camera views. Robot positions
    are in the brain frame and are placed with the known start pose, as LiveMaps does."""

    FADE_S = 1.5         # a new team-map snapshot blends in over this much sim time after it was written
    TRAIL_STEP_M = .3
    VIEW_M = 2.6

    def __init__(self, rec: Path, meta: dict, robots: list[str], states: dict, size, report: dict | None = None):
        self.rec, self.robots, self.states = rec, robots, states
        scenario = load_scenario(meta)
        self.frames = {r: recording_frame(states[r], scenario, r)[0] for r in robots}
        plan_path = rec / 'oracle' / 'floorplan_used.json'
        plan = json.loads((plan_path if plan_path.exists() else ROOT / 'config' / 'floorplan.json').read_text())
        area = scenario.get('search_area') or plan.get('footprint') or [0, 0, 20, 14]
        self.area = np.array(area, float)
        x0, y0, x1, y1 = self.area
        rooms = [r['rect'] for r in plan.get('rooms', []) if r['rect'][0] >= x0 - 1e-6 and r['rect'][1] >= y0 - 1e-6
                 and r['rect'][2] <= x1 + 1e-6 and r['rect'][3] <= y1 + 1e-6] or [list(area)]
        self.floor_m2 = float(sum((r[2] - r[0]) * (r[3] - r[1]) for r in rooms))
        self.bounds = np.array([x0 - .5, y0 - .9, x1 + .5, y1 + .5])
        bx0, by0, bx1, by1 = self.bounds
        self.scale = min(size[0] / (bx1 - bx0), size[1] / (by1 - by0))
        self.size = (int(round((bx1 - bx0) * self.scale)), int(round((by1 - by0) * self.scale)))
        w, h = self.size
        xs = bx0 + (np.arange(w) + .5) / self.scale
        ys = by1 - (np.arange(h) + .5) / self.scale
        self.xy = np.stack(np.meshgrid(xs, ys), axis=-1)
        self.floor = np.zeros((h, w), bool)
        for r in rooms:
            self.floor |= ((self.xy[..., 0] >= r[0]) & (self.xy[..., 0] < r[2]) &
                           (self.xy[..., 1] >= r[1]) & (self.xy[..., 1] < r[3]))
        # Static backdrop: floor dot grid; walls go on after the coverage fill so they stay on top.
        base = np.full((h, w, 3), PLAN_BG, np.uint8)
        for gx in np.arange(x0 + .25, x1, .5):
            for gy in np.arange(y0 + .25, y1, .5):
                cv2.circle(base, tuple(self.px([gx, gy])), 1, PLAN_DOTS, -1, cv2.LINE_AA)
        self.base = base.astype(np.float32)
        walls = true_wall_segments(dict(plan, rooms=[dict(rect=r) for r in rooms]))
        mask = np.zeros((h, w), np.uint8)
        thick = max(3, int(round(.17 * self.scale)))
        for a, b in walls:
            cv2.line(mask, tuple(self.px(a)), tuple(self.px(b)), 255, thick, cv2.LINE_AA)
        self.walls = mask.astype(np.float32)[..., None] / 255
        self.wall_reach = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(.35 * self.scale) + 1,) * 2)
        # Poses and paths, placed on the plan.
        self.times, self.poses, self.trails = {}, {}, {}
        for r in robots:
            st = states[r]
            self.times[r] = np.array([s['t'] for s in st], float)
            poses = np.array([[s['x'], s['y'], s['yaw_rad']] for s in st], float).reshape(-1, 3)
            poses = transform_poses(poses, self.frames[r])
            poses[:, 2] = np.unwrap(poses[:, 2])
            self.poses[r] = poses
            kept = []
            for t, p in zip(self.times[r], poses):
                if not kept or np.hypot(*(p[:2] - kept[-1][1:])) >= self.TRAIL_STEP_M:
                    kept.append((t, *p[:2]))
            self.trails[r] = np.array(kept, float).reshape(-1, 3)
        self.hfov = {r: camera_hfov(rec / r) for r in robots}
        self.water = jsonl(rec / "water.jsonl")
        self.rooms_done = []
        for k, r in enumerate(robots):
            for e in jsonl(rec / r / 'decisions.jsonl'):
                n = e.get('numbers') or {}
                if e['kind'] == 'room_searched' and n.get('floor_m2', 0) >= WALK_IN_MIN_M2 and 'x_m' in n:
                    self.rooms_done.append((float(e['t']), k, transform_xy([n['x_m'], n['y_m']], self.frames[r])))
        self.people = list((report or {}).get('confirmation_details', []))
        # The shared map: one receiver's actual team belief, held fixed so the picture never swaps between two
        # robots' different knowledge. The first robot with an export is the receiver.
        self.receiver, self.snapshots = None, []
        for r in robots:
            found = sorted((float(p.name.removesuffix('.team.bin.gz')), p)
                           for p in (rec / r / 'brain_map').glob('*.team.bin.gz'))
            if found:
                self.receiver, self.snapshots = r, found
                break
        self.stamps = np.array([s for s, _ in self.snapshots])
        self._grids, self._images, self._cells, self._fire = {}, {}, None, (None, None)
        self.stats = [self._stats(k) for k in range(len(self.snapshots))]

    def px(self, xy):
        bx0, _, _, by1 = self.bounds
        xy = np.asarray(xy, float)
        return np.rint((xy - [bx0, by1]) * [self.scale, -self.scale]).astype(np.int32)

    def _grid(self, k):
        if k not in self._grids:
            self._grids[k] = load_team_map(self.snapshots[k][1])
        return self._grids[k]

    def _stats(self, k):
        """Square metres of the house floor each robot searched, on the receiver's team map."""
        grid = self._grid(k)
        yy, xx = np.indices((grid.size, grid.size))
        house = transform_xy(grid.origin + (np.stack([xx, yy], axis=-1) + .5) * grid.cell_m, self.frames[self.receiver])
        x0, y0, x1, y1 = self.area
        inside = (house[..., 0] >= x0) & (house[..., 0] < x1) & (house[..., 1] >= y0) & (house[..., 1] < y1)
        bits = robot_bits(grid, self.robots)[inside]
        cell = grid.cell_m ** 2
        known = (grid.source_mask[inside] != 0) & (grid.occupancy[inside] <= .5)
        self._grids.pop(k, None)
        return dict(at=self.snapshots[k][0], a=float((bits == 1).sum() * cell), b=float((bits == 2).sum() * cell),
                    both=float((bits == 3).sum() * cell), union=float((bits != 0).sum() * cell),
                    mapped=float((known & (bits == 0)).sum() * cell))

    def snapshot_index(self, t):
        return int(np.searchsorted(self.stamps, t, side='right')) - 1 if len(self.stamps) else -1

    def _sample(self, grid):
        key = (grid.size, grid.cell_m, *grid.origin)
        if self._cells is None or self._cells[0] != key:
            T = self.frames[self.receiver]
            own = (self.xy - T[:2]) @ rotation(T[2])
            cells = np.floor((own - grid.origin) / grid.cell_m).astype(int)
            valid = ((cells >= 0) & (cells < grid.size)).all(axis=-1)
            self._cells = key, np.clip(cells, 0, grid.size - 1), valid
        return self._cells[1], self._cells[2]

    def snapshot_image(self, k):
        """The plan with snapshot k's coverage and lit walls; k < 0 is the house before any map."""
        if k in self._images:
            return self._images[k]
        img = self.base.copy()
        lit = np.zeros(self.floor.shape, np.float32)
        if k >= 0:
            grid = self._grid(k)
            cells, valid = self._sample(grid)
            iy, ix = cells[..., 1], cells[..., 0]
            bits = tidy_bits(robot_bits(grid, self.robots))[iy, ix] * valid
            known = (grid.source_mask[iy, ix] != 0) & valid
            free = known & (grid.occupancy[iy, ix] <= .5)
            conf = grid.searched_confidence[iy, ix]
            fill = np.zeros(img.shape, np.float32)
            alpha = np.zeros(self.floor.shape, np.float32)
            mapped = free & (bits == 0) & self.floor
            fill[mapped], alpha[mapped] = MAPPED, 1
            for b, colour in SEARCHED.items():
                m = (bits == b) & self.floor
                fill[m], alpha[m] = colour, .6 + .4 * conf[m]
            # Soften the 0.2 m cell steps so the fill reads as floor, not pixels.
            alpha_s = cv2.GaussianBlur(alpha, (0, 0), 1.6)
            fill_s = cv2.GaussianBlur(fill * alpha[..., None], (0, 0), 1.6)
            img = img * (1 - alpha_s[..., None]) + fill_s
            lit = cv2.GaussianBlur(cv2.dilate(known.astype(np.uint8), self.wall_reach).astype(np.float32), (0, 0), 1.5)
        img += (np.array(WALL_DIM, np.float32) - img) * self.walls
        img += (np.array(WALL_LIT, np.float32) - img) * self.walls * lit[..., None]
        self._images = {key: v for key, v in self._images.items() if key >= k - 1}
        self._images[k] = img
        self._grids = {key: v for key, v in self._grids.items() if key >= k}
        return img

    def pose(self, r, t):
        times = self.times[r]
        if not len(times) or t < times[0]:
            return None
        return np.array([np.interp(t, times, self.poses[r][:, i]) for i in range(3)])

    def fire_cells(self, t):
        """Fire cells the robots' brains hold at t (union of both), on a 0.4 m lattice for drawing."""
        idx = tuple(int(np.searchsorted(self.times[r], t, side='right')) - 1 for r in self.robots)
        if self._fire[0] == idx:
            return self._fire[1]
        pts = []
        for r, i in zip(self.robots, idx):
            if i >= 0 and self.states[r][i].get('fires'):
                pts.append(transform_xy(np.array(self.states[r][i]['fires'], float), self.frames[r]))
        cells = np.unique(np.floor(np.vstack(pts) / .4).astype(int), axis=0) if pts else np.zeros((0, 2), int)
        self._fire = idx, (cells + .5) * .4
        return self._fire[1]

    def stat(self, t):
        k = self.snapshot_index(t)
        return self.stats[k] if k >= 0 else dict(at=0, a=0, b=0, both=0, union=0, mapped=0)

    def render(self, t: float, link: str | None = None):
        """The plan at sim time t as an RGB array, plus label anchors [(xy, text, colour)]."""
        k = self.snapshot_index(t)
        img = self.snapshot_image(k).copy()
        if k >= 0:
            since = t - self.snapshots[k][0]
            if since < self.FADE_S:
                share = since / self.FADE_S
                img = self.snapshot_image(k - 1) * (1 - share) + img * share
        h, w = img.shape[:2]
        # Fire the robots hold: a lattice of ember dots over a soft glow.
        # Fire the robots hold: a lattice of ember dots over a soft glow, on the house floor only.
        fire = self.px(self.fire_cells(t)).reshape(-1, 2)
        fire = fire[(fire[:, 0] >= 0) & (fire[:, 0] < w) & (fire[:, 1] >= 0) & (fire[:, 1] < h)]
        fire = fire[self.floor[fire[:, 1], fire[:, 0]]]
        if len(fire):
            dots = np.zeros((h, w), np.uint8)
            for p in fire:
                cv2.circle(dots, tuple(p), 2, 255, -1, cv2.LINE_AA)
            halo = cv2.GaussianBlur(dots.astype(np.float32) / 255, (0, 0), 5) * 1.4
            img += (np.array(FIRE, np.float32) * .5 - img) * np.clip(halo, 0, .45)[..., None]
            img += (np.array(FIRE, np.float32) - img) * (dots.astype(np.float32) / 255 * .85)[..., None]
        for e in self.water:
            if e["t"] <= t < e["t"] + e["seconds"]:
                head, half, reach = np.radians(e["heading_deg"]), np.radians(NOZZLE["half_angle_deg"]), NOZZLE["range_m"]
                fan = [[e["x"], e["z"]]] + [[e["x"] + reach * np.sin(a), e["z"] + reach * np.cos(a)]
                                            for a in np.linspace(head - half, head + half, 9)]
                layer = img.copy()
                cv2.fillPoly(layer, [self.px(fan)], WATER, cv2.LINE_AA)
                img = img * .55 + layer * .45
        poses = {r: self.pose(r, t) for r in self.robots}
        # Camera view cones, so each camera picture above has a place on the plan.
        for k_, r in enumerate(self.robots):
            p = poses[r]
            if p is None:
                continue
            half = self.hfov[r] / 2
            angles = p[2] + np.linspace(-half, half, 13)
            cone = [p[:2]] + [p[:2] + self.VIEW_M * np.array([-np.sin(a), np.cos(a)]) for a in angles]
            layer = img.copy()
            cv2.fillPoly(layer, [self.px(cone)], ROBOT[k_], cv2.LINE_AA)
            img = img * .84 + layer * .16
        # Paths: dotted, older dots fading.
        for k_, r in enumerate(self.robots):
            trail = self.trails[r][self.trails[r][:, 0] <= t]
            for (ti, x, y), q in zip(trail, self.px(trail[:, 1:])):
                a = .3 + .7 * np.exp(-(t - ti) / 45)
                cv2.circle(img, tuple(q), 2, mix(ROBOT[k_], PLAN_BG, a), -1, cv2.LINE_AA)
        for ti, k_, xy in self.rooms_done:
            if ti <= t < ti + 2.4:
                u = (t - ti) / 2.4
                layer = img.copy()
                cv2.circle(layer, tuple(self.px(xy)), int(8 + 26 * u), mix(ROBOT[k_], (255, 255, 255), .7), 2, cv2.LINE_AA)
                img = img * u + layer * (1 - u)
        labels = []
        for c in self.people:
            if c['t'] > t:
                continue
            q = self.px([c['x'], c['y']])
            real = c.get('survivor') is not None
            colour = GOOD if real else FAINT
            age = t - c['t']
            if age < 3:
                layer = img.copy()
                cv2.circle(layer, tuple(q), int(10 + 34 * age / 3), colour, 3, cv2.LINE_AA)
                img = img * (age / 3) + layer * (1 - age / 3)
            glow(img, q, colour, 9, .5 if real else .2)
            cv2.circle(img, tuple(q), 8, colour, 2, cv2.LINE_AA)
            cv2.circle(img, tuple(q), 4, colour if real else PLAN_BG, -1, cv2.LINE_AA)
        if link and all(poses[r] is not None for r in self.robots[:2]) and len(self.robots) > 1:
            a, b = self.px(poses[self.robots[0]][:2]), self.px(poses[self.robots[1]][:2])
            n = max(2, int(np.hypot(*(b - a)) / 7))
            colour = (150, 210, 175) if link == 'up' else (170, 70, 70)
            for i in range(0, n, 1 if link == 'up' else 2):
                if link == 'up':
                    cv2.circle(img, tuple(np.rint(a + (b - a) * i / n).astype(int)), 1, colour, -1, cv2.LINE_AA)
                else:
                    cv2.line(img, tuple(np.rint(a + (b - a) * i / n).astype(int)),
                             tuple(np.rint(a + (b - a) * (i + 1) / n).astype(int)), colour, 1, cv2.LINE_AA)
        for k_, r in enumerate(self.robots):
            p = poses[r]
            if p is None:
                continue
            q = self.px(p[:2])
            pinned = np.clip(q, 10, [w - 11, h - 11])
            if np.any(pinned != q):
                # Still outside the house (on its launch pad): pinned to the plan's edge, hollow.
                glow(img, pinned, ROBOT[k_], 9, .35)
                cv2.circle(img, tuple(pinned), 7, ROBOT[k_], 2, cv2.LINE_AA)
                labels.append(((int(pinned[0]) + 12, int(pinned[1]) - 10), f"{'AB'[k_]} outside", ROBOT[k_]))
                continue
            glow(img, q, ROBOT[k_], 11, .6)
            cv2.circle(img, tuple(q), 8, (250, 246, 240), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(q), 6, ROBOT[k_], -1, cv2.LINE_AA)
            labels.append(((int(q[0]) + 11, int(q[1]) - 24), 'AB'[k_], ROBOT[k_]))
        return np.clip(img, 0, 255).astype(np.uint8), labels


def draw_plan_panel(im: Image.Image, d: ImageDraw.ImageDraw, plan: HousePlan, t: float, link: str | None, box=PLAN_BOX):
    panel(d, box)
    img, labels = plan.render(t, link)
    mx, my = box[0] + 12, box[1] + (box[3] - box[1] - img.shape[0]) // 2
    im.paste(Image.fromarray(img), (mx, my))
    for (x, y), text, colour in labels:
        d.text((mx + x, my + y), text, font=F["dot"], fill=colour, stroke_width=3, stroke_fill=PLAN_BG)
    # Scale bar, bottom left of the plan.
    five = int(round(5 * plan.scale))
    sx, sy = mx + 14, my + img.shape[0] - 14
    d.line((sx, sy, sx + five, sy), fill=DIM, width=2)
    for x in (sx, sx + five):
        d.line((x, sy - 4, x, sy + 4), fill=DIM, width=2)
    d.text((sx + five + 8, sy - 9), "5 m", font=F["small"], fill=DIM)
    # Legend column.
    x0, x1, y = mx + img.shape[1] + 24, box[2] - 20, box[1] + 16
    d.text((x0, y), "SHARED MAP", font=F["panel"], fill=TEXT)
    y += 30
    if plan.receiver is None:
        d.text((x0, y), "No team map in this recording.", font=F["small"], fill=DIM)
        y += 22
    else:
        me = plan.receiver.removeprefix('robot_')
        others = ' and '.join(r.removeprefix('robot_') for r in plan.robots if r != plan.receiver)
        for line in wrap(d, f"Robot {me}'s team map: its own search plus what {others} sent by radio.", "small", x1 - x0):
            d.text((x0, y), line, font=F["small"], fill=DIM)
            y += 18
    s = plan.stat(t)
    y += 10
    d.text((x0, y), f"{s['union']:.0f} m²", font=F["stat"], fill=TEXT)
    d.text((x0 + d.textlength(f"{s['union']:.0f} m²", font=F["stat"]) + 12, y + 17),
           f"of {plan.floor_m2:.0f} m² searched", font=F["legend"], fill=DIM)
    y += 52
    # Searched area over time, up to now.
    gw, gh = x1 - x0, 34
    d.line((x0, y + gh, x1, y + gh), fill=EDGE, width=1)
    end = max(plan.stamps[-1] if len(plan.stamps) else 1, 1)
    pts = [(x0 + gw * st['at'] / end, y + gh - gh * st['union'] / plan.floor_m2) for st in plan.stats if st['at'] <= t]
    if len(pts) > 1:
        d.polygon(pts + [(pts[-1][0], y + gh), (pts[0][0], y + gh)], fill=mix(SEARCHED[3], PANEL, .7))
        d.line(pts, fill=mix(SEARCHED[3], (255, 255, 255), .55), width=2)
        d.ellipse((pts[-1][0] - 3, pts[-1][1] - 3, pts[-1][0] + 3, pts[-1][1] + 3), fill=TEXT)
    y += gh + 16
    a, b = 'AB'[0], 'AB'[1]
    found = {c['survivor'] for c in plan.people if c['t'] <= t and c.get('survivor') is not None}
    claims = sum(1 for c in plan.people if c['t'] <= t)
    rows = [("fill", SEARCHED[1], f"Searched by {a}", f"{s['a']:.0f} m²"),
            ("fill", SEARCHED[2], f"Searched by {b}", f"{s['b']:.0f} m²"),
            ("fill", SEARCHED[3], "Searched by both", f"{s['both']:.0f} m²"),
            ("fill", MAPPED, "Mapped, not searched yet", f"{s['mapped']:.0f} m²"),
            ("fire", FIRE, "Fire on the robots' maps", ""),
            ("person", GOOD, "People found", f"{len(found)}" + (f" ({claims} calls)" if claims != len(found) else ""))]
    for kind, colour, label, value in rows:
        cy = y + 10
        if kind == "fill":
            d.rounded_rectangle((x0, cy - 8, x0 + 16, cy + 8), 4, fill=mix(colour, (255, 255, 255), .9),
                                outline=mix(colour, (255, 255, 255), .7))
        elif kind == "fire":
            for dx, dy in ((3, -3), (11, -3), (3, 5), (11, 5)):
                d.ellipse((x0 + dx - 2, cy + dy - 2, x0 + dx + 2, cy + dy + 2), fill=colour)
        else:
            d.ellipse((x0, cy - 8, x0 + 16, cy + 8), outline=colour, width=2)
            d.ellipse((x0 + 5, cy - 3, x0 + 11, cy + 3), fill=colour)
        d.text((x0 + 28, cy - 10), label, font=F["legend"], fill=TEXT)
        if value:
            d.text((x1 - d.textlength(value, font=F["legend"]), cy - 10), value, font=F["legend"], fill=DIM)
        y += 25
    lines = wrap(d, PLAN_NOTE, "tiny", x1 - x0)
    for n, line in enumerate(lines):
        d.text((x0, box[3] - 14 - 16 * (len(lines) - n)), line, font=F["tiny"], fill=FAINT)


class Radio:
    """Team radio items: agent decisions with tools and orders, radio texts, survivors, map merges, room work.
    Radio link changes are kept apart in `links` and shown as a quiet status, not as feed lines."""

    QUIET = ("room auction",)

    def __init__(self, rec: Path, run: Path | None, robots: list[str]):
        self.items = []    # (t, side, kind, text, chips)
        self.links = []    # (t, 'up' or the reason it is down)
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
            me = 'AB'[side[r]]; other = 'AB'[1 - side[r]] if len(robots) == 2 else 'its partner'
            last_place = last_loop = -1e9
            for e in jsonl(rec / r / "decisions.jsonl"):
                k, t = e["kind"], float(e["t"])
                n = e.get("numbers") or {}
                # Live visual SLAM (visual_slam_live): camera place matches between the robots, the shared-frame correction they
                # measure, and loop closures; the matches and loops are thinned so they do not flood the feed.
                if k == "place_recognised" and t - last_place >= 20:
                    last_place = t
                    self.items.append((t, side[r], "agent", f"{me}'s camera recognised a place {other} saw: {n.get('inliers', 0):.0f} OpenCV 5 "
                                       "ORB matches, checked on its LiDAR map", [("visual SLAM", CV)]))
                elif k == "frame_link_refined":
                    turn = ((n.get("dyaw_deg", 0) + 180) % 360) - 180
                    self.items.append((t, -1, "slam", f"VISUAL SLAM: {me} corrects the shared map by {turn:+.1f}° from camera matches", []))
                elif k == "visual_loop_closure" and t - last_loop >= 30:
                    last_loop = t
                    self.items.append((t, side[r], "agent", f"{me} recognised a place it saw {n.get('age_s', 0):.0f} s ago: "
                                       f"drift {100 * n.get('drift_m', 0):.0f} cm", [("loop closure", CV)]))
                if k == "team_message_sent":
                    self.items.append((t, side[r], "radio", "“" + e["reason"] + "”", [("radio out", DIM)]))
                elif k == "survivor_confirmed":
                    self.items.append((t, -1, "good", f"SURVIVOR CONFIRMED by {me}", []))
                elif k == "candidate_rejected":
                    self.items.append((t, side[r], "agent", "OpenCV check: that warm shape is not a person", [("reject candidate", ROBOT[side[r]])]))
                elif k == "frame_linked":
                    peer = frame_peer(e, r, robots)
                    label = peer.removeprefix('robot_') if peer else 'its partner'
                    self.items.append((t, side[r], "frame_link", f"{me} linked its map with {label} from matched LiDAR fingerprints", [("LiDAR frame linked", GOOD)]))
                elif k == "map_shared":
                    self.items.append((t, side[r], "agent", f"{me} now holds its partner's map: no double search", [("map shared", GOOD)]))
                elif k == "stuck":
                    self.items.append((t, side[r], "warn", "Legs not moving: backing off and replanning", [("stuck", WARN)]))
                # The onboard brain's own room decisions, so a run without LLM agents still shows its reasoning.
                elif k == "room_claimed":
                    self.items.append((t, side[r], "quiet", f"{me} claims the nearest unsearched room, {n['bid_s']:.0f} s away",
                                       [("room auction", ROBOT[side[r]])]))
                elif k == "room_searched" and n.get("floor_m2", 0) >= WALK_IN_MIN_M2:
                    self.items.append((t, side[r], "agent", f"{me} searched a {n['floor_m2']:.0f} m² room, {n['covered_share']:.0%} seen",
                                       [("room done", GOOD)]))
                elif k == "entrance_locked":
                    self.items.append((t, side[r], "agent", f"{me} is inside and keeps the way in clear as its exit",
                                       [("doorway marked", ROBOT[side[r]])]))
                elif k == "room_marked_fire":
                    self.items.append((t, side[r], "warn", f"{me} measured {n['air_c']:.0f} °C air: room marked on fire",
                                       [("room on fire", WARN)]))
                elif k == "room_fire_cleared":
                    self.items.append((t, side[r], "agent", f"{me} measures no fire left in that room: open again",
                                       [("fire out", GOOD)]))
                elif k == "retreat":
                    self.items.append((t, side[r], "warn", f"{me} backs off to cooler floor, {n['air_c']:.0f} °C air",
                                       [("too hot", WARN)]))
        for e in jsonl(rec / "water.jsonl"):
            who = side.get(e["robot"], 0)
            self.items.append((e["t"], who, "agent", f"{'AB'[who]} sprays water at the fire for {e['seconds']:.0f} s to open the way",
                               [("water spray", WATER)]))
        for e in jsonl(rec / "rescues.jsonl"):
            who = side.get(e["robot"], 0)
            if e["event"] == "gripped":
                self.items.append((e["t"], -1, "good", f"{'AB'[who]} lifts a trapped person and tows them to the way out", []))
            elif e["event"] == "released":
                where = "out of the building" if e.get("outside_building") else "inside the building (not yet out)"
                self.items.append((e["t"], -1, "good" if e.get("outside_building") else "warn", f"{'AB'[who]} sets the person down {where}", []))
        rp = rec / "brain_radio.txt"
        if rp.exists():
            for line in rp.read_text().splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                t, frm, to, _, size, _, status, _ = line.split()[:8]
                state = "up" if status == "sent" else status.replace("queued_", "")
                if frm == robots[0] and (not self.links or self.links[-1][1] != state):
                    self.links.append((float(t), state))
        self.items.sort(key=lambda x: x[0])

    def link(self, t):
        now = [s for lt, s in self.links if lt <= t]
        return now[-1] if now else None

    def draw(self, d, box, t):
        panel(d, box, "Team radio", "what the robots tell each other")
        state = self.link(t)
        if state is not None:
            text = "radio link up" if state == "up" else f"radio link down: {state.replace('_', ' ')}"
            colour = GOOD if state == "up" else WARN
            tw = d.textlength(text, font=F["small"])
            d.ellipse((box[2] - 34 - tw, box[1] + 22, box[2] - 26 - tw, box[1] + 30), fill=colour)
            d.text((box[2] - 20 - tw, box[1] + 17), text, font=F["small"], fill=DIM)
        x0, x1 = box[0] + 14, box[2] - 14
        top, y = box[1] + 52, box[3] - 10
        d.line((box[0] + 20, top - 4, box[2] - 20, top - 4), fill=EDGE, width=1)
        rows = []
        for ti, side, kind, text, chips in reversed([i for i in self.items if i[0] <= t]):
            loud = side < 0 and kind != "commander"
            quiet = kind == "quiet"
            if loud:
                lines = wrap(d, text, "rowb", x1 - x0 - 110)[:2]
                h = 22 + 26 * len(lines)
            else:
                tag_w = sum(d.textlength(label, font=F["chip"]) + 24 for label, _ in chips[:2])
                lines = wrap(d, text, "quiet" if quiet else "row", x1 - x0 - 120 - tag_w)[:2]
                h = (12 + 22 * len(lines) if quiet else 16 + 25 * len(lines))
            if y - h < top:
                break
            y -= h
            rows.append((y, h, ti, side, kind, lines, chips, loud, quiet))
        for y, h, ti, side, kind, lines, chips, loud, quiet in rows:
            # The oldest rows fade into the panel, so the feed reads as a stream rather than a cut-off list.
            keep = min(1., .3 + (y - top) / 90)
            fresh = max(0., 1 - (t - ti) / 2.5)
            fade = lambda colour: mix(colour, PANEL, keep)
            if loud:
                colour = GOOD if kind == "good" else CV if kind == "slam" else WARN
                d.rounded_rectangle((x0, y + 2, x1, y + h - 4), 9, fill=fade(mix(colour, PANEL, .13 + .12 * fresh)))
                d.rounded_rectangle((x0, y + 2, x0 + 5, y + h - 4), 2, fill=fade(colour))
                d.text((x0 + 20, y + h // 2 - 11), clock(ti), font=F["mono"], fill=fade(mix(colour, PANEL, .8)))
                for n, line in enumerate(lines):
                    d.text((x0 + 96, y + 9 + 26 * n), line, font=F["rowb"], fill=fade(colour))
                continue
            if fresh > 0:
                tint = ROBOT[side] if side >= 0 else DIM
                d.rounded_rectangle((x0, y + 1, x1, y + h - 1), 9, fill=mix(tint, PANEL, .12 * fresh))
            cy = y + h // 2 - (11 * (len(lines) - 1) if quiet else 12 * (len(lines) - 1))
            d.text((x0 + 6, cy - 9), clock(ti), font=F["mono"], fill=fade(FAINT))
            c = ROBOT[side] if side >= 0 else DIM
            letter = 'AB'[side] if side >= 0 else 'C'
            r = 9 if quiet else 12
            d.ellipse((x0 + 78 - r, cy - r, x0 + 78 + r, cy + r), fill=fade(mix(c, PANEL, .5 if quiet else 1)))
            d.text((x0 + 78 - d.textlength(letter, font=F["chip"]) / 2, cy - 9), letter, font=F["chip"], fill=BG)
            for n, line in enumerate(lines):
                if quiet:
                    d.text((x0 + 104, cy - 10 + 22 * n), line, font=F["quiet"], fill=fade(FAINT))
                else:
                    d.text((x0 + 104, cy - 12 + 25 * n), line, font=F["row"], fill=fade(TEXT))
            right = x1 - 10
            for label, c in chips[:2]:
                right = chip(d, right, cy, label, fade(mix(c, PANEL, .5) if quiet else c)) - 6

def camera_tile(k: int, rgb: np.ndarray, th, hot, state: dict | None, torch: str | None, said: str | None,
                flash: float) -> Image.Image:
    """One robot's camera at CAM_W x CAM_H with its overlays, as RGBA with rounded corners."""
    h, w = rgb.shape[:2]
    rows = int(round(w * CAM_H / CAM_W))
    crop = rgb[max(0, h - rows):]
    tile = Image.fromarray(cv2.resize(crop, (CAM_W, CAM_H), interpolation=cv2.INTER_CUBIC)).convert("RGBA")
    over = Image.new("RGBA", tile.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    c = ROBOT[k]
    # Name plate, top left.
    d.rounded_rectangle((16, 16, 56, 56), 10, fill=c + (255,))
    d.text((36 - d.textlength("AB"[k], font=F["letter"]) / 2, 20), "AB"[k], font=F["letter"], fill=BG + (255,))
    name = f"Robot {'AB'[k]}  front camera"
    d.rounded_rectangle((64, 20, 64 + d.textlength(name, font=F["label"]) + 24, 52), 16, fill=(8, 7, 12, 170))
    d.text((76, 25), name, font=F["label"], fill=TEXT + (255,))
    # Live state, bottom left.
    x = 16
    words = []
    if state:
        words.append(STATE_WORDS.get(state.get("state", ""), str(state.get("state", "")).replace("_", " ").capitalize()))
        words.append("Thermal vision" if state.get("mode") == "thermal" else "Colour vision")
    if torch:
        words.append(f"Torch {torch}")
    for word in words:
        tw = d.textlength(word, font=F["pill"])
        d.rounded_rectangle((x, CAM_H - 48, x + tw + 24, CAM_H - 16), 16, fill=(8, 7, 12, 175))
        d.text((x + 12, CAM_H - 43), word, font=F["pill"], fill=TEXT + (255,))
        x += tw + 32
    if said:
        lines = wrap(d, said, "body", CAM_W - 330)[:2]
        y0 = CAM_H - 62 - 24 * len(lines)
        d.rounded_rectangle((16, y0 - 8, CAM_W - 290, y0 + 24 * len(lines) + 4), 10, fill=(8, 7, 12, 190))
        for n, line in enumerate(lines):
            d.text((28, y0 + 24 * n), line, font=F["body"], fill=TEXT + (255,))
    if th is not None:
        tw_, th_ = th.shape[1], th.shape[0]
        tx, ty = CAM_W - tw_ - 16, CAM_H - th_ - 16
        d.rounded_rectangle((tx - 3, ty - 3, tx + tw_ + 3, ty + th_ + 3), 10, fill=(8, 7, 12, 230))
        label = f"Thermal  max {hot:.0f} °C"
        lw = d.textlength(label, font=F["chip"])
        d.rounded_rectangle((tx + tw_ - lw - 20, ty - 34, tx + tw_ + 3, ty - 8), 13, fill=(8, 7, 12, 200))
        d.text((tx + tw_ - lw - 9, ty - 29), label, font=F["chip"], fill=TEXT + (255,))
    if flash > 0:
        a = int(255 * min(1, flash))
        d.rounded_rectangle((2, 2, CAM_W - 3, CAM_H - 3), 14, outline=GOOD + (a,), width=5)
        text = "Survivor confirmed"
        tw = d.textlength(text, font=F["label"])
        d.rounded_rectangle((CAM_W - tw - 56, 20, CAM_W - 16, 54), 17, fill=GOOD + (a,))
        d.text((CAM_W - tw - 36, 25), text, font=F["label"], fill=BG + (a,))
    tile = Image.alpha_composite(tile, over)
    if th is not None:
        mask = Image.new("L", (th.shape[1], th.shape[0]), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, th.shape[1] - 1, th.shape[0] - 1), 8, fill=255)
        tile.paste(Image.fromarray(th), (CAM_W - th.shape[1] - 16, CAM_H - th.shape[0] - 16), mask)
    return tile


def draw_header(d, title: str, t: float, speed: float, agents: bool, vslam: bool = False):
    d.text((24, 14), title, font=F["title"], fill=TEXT)
    d.text((24 + d.textlength(title, font=F["title"]) + 20, 24),
           "Two quadruped robots search a burning house  ·  OpenCV 5 " + ("perception and live visual SLAM" if vslam else "perception") + "  ·  "
           + ("LLM agents" if agents else "onboard robot brains, no LLM") + "  ·  maps shared over a lossy radio",
           font=F["sub"], fill=DIM)
    badge = f"{speed:g}x speed"
    bw = d.textlength(badge, font=F["badge"]) + 24
    d.rounded_rectangle((W - 24 - bw, 18, W - 24, 46), 14, fill=(44, 40, 58))
    d.text((W - 12 - bw, 22), badge, font=F["badge"], fill=TEXT)
    text = clock(t, tenths=True)
    cw = d.textlength(text, font=F["clockmono"])
    d.text((W - 40 - bw - cw, 12), text, font=F["clockmono"], fill=TEXT)
    d.text((W - 52 - bw - cw - d.textlength("sim time", font=F["small"]), 26), "sim time", font=F["small"], fill=FAINT)


class Showcase:
    """Everything one recording needs for its frames; frame(t) composes the 1920x1080 picture at sim time t."""

    def __init__(self, rec: Path, title: str, speed: float, run: Path | None = None):
        self.rec, self.title, self.speed = rec, title, speed
        meta = json.loads((rec / "meta.json").read_text())
        self.meta = meta
        self.robots = robots = meta["robots"][:2]
        self.tracks = {r: frames(rec / r) for r in robots}
        self.track_times = {r: np.array([f[0] for f in self.tracks[r]]) for r in robots}
        self.thermals = {r: thermal_frames(rec / r) for r in robots}
        self.states = {r: jsonl(rec / r / "brain_state.jsonl") for r in robots}
        self.state_times = {r: np.array([s["t"] for s in self.states[r]]) for r in robots}
        self.torch = {r: [(e["t"], e["reason"]) for e in jsonl(rec / r / "decisions.jsonl")
                          if e["kind"] == "agent_order" and "torch set to" in e["reason"]] for r in robots}
        self.captions = {r: narration(run, r) for r in robots} if run else {r: [] for r in robots}
        self.radio = Radio(rec, run, robots)
        report = score_confirmations(rec)
        self.confirmed = {r: [c['t'] for c in report['confirmation_details'] if c['robot'] == r] for r in robots}
        self.plan = HousePlan(rec, meta, robots, self.states,
                              (PLAN_BOX[2] - PLAN_BOX[0] - 380, PLAN_BOX[3] - PLAN_BOX[1] - 16), report)
        self.agents = (rec / "agent_trace.jsonl").exists()
        # Live visual SLAM happened in this run if a robot refined the shared frame from camera place matches.
        self.vslam = any(e["kind"] == "frame_link_refined" for r in ("robot_A", "robot_B") for e in jsonl(rec / r / "decisions.jsonl"))
        self.end = max(fr[-1][0] for fr in self.tracks.values())
        self.cache = {}
        self.corner = Image.new("L", (CAM_W, CAM_H), 0)
        ImageDraw.Draw(self.corner).rounded_rectangle((0, 0, CAM_W - 1, CAM_H - 1), 14, fill=255)

    def camera(self, r, t):
        i = max(0, int(np.searchsorted(self.track_times[r], t + 1e-4, side='right')) - 1)
        ti, rgb_p, _ = self.tracks[r][i]
        if self.cache.get(r, (None,))[0] != ti:
            rgb = cv2.cvtColor(cv2.imread(str(rgb_p)), cv2.COLOR_BGR2RGB)
            tp = self.thermals[r].get(round(ti, 4))
            th, hot = None, None
            if tp is not None and tp.exists():
                dk = cv2.imread(str(tp), cv2.IMREAD_UNCHANGED)
                th = cv2.cvtColor(cv2.resize(ironbow(dk), (240, 192), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                hot = dk.max() / 10 - 273.15
            self.cache[r] = (ti, rgb, th, hot)
        return self.cache[r][1:]

    def frame(self, t: float) -> np.ndarray:
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        draw_header(d, self.title, t, self.speed, self.agents, self.vslam)
        for k, r in enumerate(self.robots):
            rgb, th, hot = self.camera(r, t)
            si = int(np.searchsorted(self.state_times[r], t, side='right')) - 1
            tor = [m for tt, m in self.torch[r] if tt <= t]
            said = [txt for ct, txt in self.captions[r] if ct <= t < ct + 6]
            since = [t - c for c in self.confirmed[r] if 0 <= t - c < 4]
            # A confirmation lights the confirming robot's camera for 3 s, then fades over 1 s.
            flash = min(1., 4 - min(since)) if since else 0.
            tile = camera_tile(k, rgb, th, hot, self.states[r][si] if si >= 0 else None,
                               tor[-1].split("torch set to ")[1].split(",")[0] if tor else None,
                               said[-1] if said else None, flash)
            im.paste(tile.convert("RGB"), (CAM_X[k], HEADER), self.corner)
            d.rounded_rectangle((CAM_X[k], HEADER, CAM_X[k] + CAM_W - 1, HEADER + CAM_H - 1), 14,
                                outline=mix(ROBOT[k], BG, .55), width=2)
        draw_plan_panel(im, d, self.plan, t, self.radio.link(t))
        self.radio.draw(d, RADIO_BOX, t)
        return np.asarray(im)


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
    show = Showcase(a.recording, a.title, a.speed, a.run)
    out = a.out or Path("data/evidence") / f"showcase_{show.meta['run_id']}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    end = show.end if a.max_t is None else min(show.end, a.max_t)
    times = np.arange(0, end, a.speed / a.fps)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                           "-preset", "medium", str(out)], stdin=subprocess.PIPE)
    for t in times:
        ff.stdin.write(show.frame(t).tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError('ffmpeg failed to encode the video')
    print(f"{out}: {len(times)} frames, {len(times) / a.fps:.0f} s at {a.speed:g}x")


if __name__ == "__main__":
    main()
