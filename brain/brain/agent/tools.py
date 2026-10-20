"""Tools the robot agents can call: OpenCV 5 perception on the robot's own frames, and physical or radio actions.

Perception tools run here and return evidence with an id. Action tools do not act directly: they are checked, then
returned to the simulator as orders. The checks are code, not prompt: an action that claims something about the world
must cite the id of a perception result that supports it, or it is refused and the refusal goes back to the model.

    reg = ToolRegistry(robot_dir, context)       # context: own-frame pose, heading, robot id, sim time
    result = reg.call("scan_thermal", {})
    reg.call("confirm_survivor", {"x": 9.6, "y": 5.0, "evidence_id": result["people"][0]["inspect_id"]})
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from brain.agent import vision


def _fn(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties or {}, "required": required or []}}}


NUM = {"type": "number"}
STR = {"type": "string"}
EVIDENCE = {"type": "string", "description": "id of the OpenCV result that supports this action"}

PERCEPTION = [
    _fn("scan_thermal", "OpenCV 5 thermal scan of the current view: fire hotspots (over 200 C) and person-sized warm "
        "regions, each with bearing (deg, + right), range (m) and building position."),
    _fn("inspect_candidate", "Verify one warm region with independent OpenCV cues (metric size, thermal structure, "
        "planarity, window depth, RGB contour). Returns verdict person / not_person / uncertain and an evidence id.",
        {"bearing_deg": NUM, "range_m": NUM}, ["bearing_deg", "range_m"]),
    _fn("assess_visibility", "Measure smoke and image quality of the colour camera (sharpness, contrast, haze, valid "
        "depth). Use before changing the torch."),
    _fn("track_fire_spread", "Measure fire growth and the hot front's speed and direction over the last seconds of "
        "thermal video.", {"seconds": NUM}),
    _fn("measure_opening", "Measure the clear width of the opening at a bearing from depth, and whether the robot fits.",
        {"bearing_deg": NUM}, ["bearing_deg"]),
]

ACTIONS = [
    _fn("explore_frontier", "Search one of the frontiers listed in the context next (unsearched floor the robot can reach). "
        "The robot's planner drives there, keeping clear of walls, and sweeps the area on arrival. Pick by more "
        "unsearched area, shorter path, and away from where your partner is searching.",
        {"frontier_id": {"type": "integer"}, "reason": STR}, ["frontier_id", "reason"]),
    _fn("go_to", "Drive to a building position (x, y in metres) by the robot's own planner.",
        {"x": NUM, "y": NUM, "reason": STR}, ["x", "y", "reason"]),
    _fn("approach_candidate", "Walk to a viewpoint 1.5 m short of an inspected warm region, on the line from the robot, "
        "computed from the inspection itself. Use when inspect_candidate says uncertain.", {"evidence_id": EVIDENCE},
        ["evidence_id"]),
    _fn("confirm_survivor", "Report a survivor at a building position. Needs an inspect_candidate result with verdict "
        "person.", {"x": NUM, "y": NUM, "evidence_id": EVIDENCE}, ["x", "y", "evidence_id"]),
    _fn("reject_candidate", "Mark a warm region as not a person so nobody inspects it again. Needs an "
        "inspect_candidate result with verdict not_person.", {"x": NUM, "y": NUM, "evidence_id": EVIDENCE},
        ["x", "y", "evidence_id"]),
    _fn("avoid_area", "Keep out of a circle around a fire for some seconds. Needs a scan_thermal hotspot or a "
        "track_fire_spread result.", {"x": NUM, "y": NUM, "radius_m": NUM, "seconds": NUM, "evidence_id": EVIDENCE},
        ["x", "y", "radius_m", "seconds", "evidence_id"]),
    _fn("set_torch", "Set the torch: wide, narrow (longer throw), low (dimmed, angled down, less smoke glare) or off. "
        "Needs an assess_visibility result.", {"mode": {"type": "string", "enum": ["wide", "narrow", "low", "off"]},
        "pitch_deg": NUM, "yaw_deg": NUM, "evidence_id": EVIDENCE}, ["mode", "evidence_id"]),
    _fn("open_door", "Use the arm to open the closed door directly ahead. Needs a measure_opening result showing the "
        "way is not passable.", {"evidence_id": EVIDENCE}, ["evidence_id"]),
    _fn("clear_debris", "Use the arm to push light debris ahead out of the way. Needs a measure_opening result showing "
        "the way is not passable.", {"evidence_id": EVIDENCE}, ["evidence_id"]),
    _fn("send_team_message", "Radio a short message to the partner robot and the commander. It may be delayed or "
        "lost if walls or distance block the link.", {"text": STR}, ["text"]),
    _fn("request_human_approval", "Ask the human incident commander before a risky move, citing evidence.",
        {"question": STR, "evidence_id": EVIDENCE}, ["question", "evidence_id"]),
    _fn("finish", "End this decision step with a one-sentence summary of what you decided and why.",
        {"summary": STR}, ["summary"]),
]

ALL_TOOLS = PERCEPTION + ACTIONS
ACTION_NAMES = {t["function"]["name"] for t in ACTIONS}


@dataclass
class AgentContext:
    robot: str
    t: float
    x: float                      # building frame, metres (x right, y = Unity z)
    y: float
    heading_deg: float            # right-positive, 0 faces +y
    event_kind: str = ""
    torch: str = "wide"
    frame_times: list[float] = field(default_factory=list)   # recent thermal stamps, oldest first
    frontiers: list[dict] = field(default_factory=list)      # search targets offered by the robot's own map


class ToolRegistry:
    # Close looks per spot, kept across steps (the server builds a new registry per step): (robot, cell) -> times.
    _looks: dict = {}
    MAX_LOOKS, LOOK_CELL_M, LOOK_MEMORY_S = 2, 1.5, 90.0

    def __init__(self, robot_dir: Path, ctx: AgentContext, debug_dir: Path | None = None):
        self.robot_dir = Path(robot_dir)
        self.ctx = ctx
        self.debug_dir = Path(debug_dir) if debug_dir else self.robot_dir.parent / "agent_debug"
        self.evidence: dict[str, dict] = {}
        self.orders: list[dict] = []
        self.refusals: list[dict] = []
        self.finished: str | None = None
        self._cache: dict = {}

    # ----- geometry --------------------------------------------------------------------------------------------
    def world(self, bearing_deg: float, range_m: float) -> tuple[float, float]:
        a = math.radians(self.ctx.heading_deg + bearing_deg)
        return self.ctx.x + range_m * math.sin(a), self.ctx.y + range_m * math.cos(a)

    def _frame(self):
        return vision.load_frame(self.robot_dir, self.ctx.t)

    # ----- dispatch --------------------------------------------------------------------------------------------
    def call(self, name: str, args: dict) -> dict:
        try:
            if name in ACTION_NAMES:
                return self._action(name, args or {})
            fn = getattr(self, f"_p_{name}", None)
            if fn is None:
                return {"error": f"unknown tool {name!r}"}
            key = (name, tuple(sorted((args or {}).items())))
            if key in self._cache:                   # same frame, same answer: say so instead of recomputing
                return {**self._cache[key], "note": "already run in this step; use this result"}
            result = fn(**(args or {}))
            self._cache[key] = result
            if isinstance(result, dict) and result.get("id"):
                self.evidence[result["id"]] = result
            return result
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        except (ValueError, FileNotFoundError, KeyError) as e:
            return {"error": f"{name} failed: {e}"}

    # ----- perception ------------------------------------------------------------------------------------------
    def _p_scan_thermal(self):
        r = vision.scan_thermal(self._frame())
        for item in r.get("hotspots", []) + r.get("people", []):
            if item.get("bearing_deg") is not None and item.get("range_m") is not None:
                item["x"], item["y"] = (round(v, 2) for v in self.world(item["bearing_deg"], item["range_m"]))
        return r

    def _p_inspect_candidate(self, bearing_deg: float, range_m: float):
        r = vision.inspect_candidate(self._frame(), float(bearing_deg), float(range_m), output_dir=self.debug_dir)
        r["x"], r["y"] = (round(v, 2) for v in self.world(float(bearing_deg), float(range_m)))
        return r

    def _p_assess_visibility(self):
        return vision.assess_visibility(self._frame())

    def _p_track_fire_spread(self, seconds: float = 3.0):
        times = [s for s in self.ctx.frame_times if s >= self.ctx.t - float(seconds)] or [self.ctx.t]
        if len(times) > 10:
            step = len(times) / 10
            times = [times[int(i * step)] for i in range(10)]
        frames = [vision.load_frame(self.robot_dir, s) for s in times]
        if len(frames) < 2:
            return {"error": "need at least two thermal frames"}
        return vision.track_fire_spread(frames)

    def _p_measure_opening(self, bearing_deg: float):
        return vision.measure_opening(self._frame(), float(bearing_deg))

    # ----- actions ---------------------------------------------------------------------------------------------
    def _need(self, args: dict, tools: tuple[str, ...], check=None, why: str = "") -> dict | None:
        ev = self.evidence.get(str(args.get("evidence_id", "")))
        if ev is None:
            return {"refused": True, "reason": "evidence_id does not name an OpenCV result from this step; call the "
                    f"matching perception tool first ({' or '.join(tools)})."}
        if ev.get("tool") not in tools:
            return {"refused": True, "reason": f"evidence {ev['id']} comes from {ev.get('tool')}, not {' or '.join(tools)}."}
        if check is not None and not check(ev):
            return {"refused": True, "reason": why}
        return None

    def _near(self, ev: dict, args: dict, tol: float = 1.0) -> bool:
        return ev.get("x") is None or math.hypot(ev["x"] - float(args["x"]), ev["y"] - float(args["y"])) <= tol

    def _action(self, name: str, args: dict) -> dict:
        refusal = None
        if name == "confirm_survivor":
            refusal = self._need(args, ("inspect_candidate",), lambda ev: ev.get("verdict") == "person" and self._near(ev, args),
                                 "the cited inspection is not a person verdict at that position")
        elif name == "reject_candidate":
            refusal = self._need(args, ("inspect_candidate",), lambda ev: ev.get("verdict") == "not_person" and self._near(ev, args),
                                 "the cited inspection is not a not_person verdict at that position")
        elif name == "avoid_area":
            refusal = self._need(args, ("scan_thermal", "track_fire_spread"),
                                 lambda ev: ev.get("tool") == "track_fire_spread" or bool(ev.get("hotspots")),
                                 "the cited scan found no hotspot")
        elif name == "set_torch":
            refusal = self._need(args, ("assess_visibility",))
        elif name in ("open_door", "clear_debris"):
            refusal = self._need(args, ("measure_opening",), lambda ev: ev.get("passable") is False,
                                 "the cited measurement says the way is passable; just drive")
        elif name == "explore_frontier":
            match = [f for f in self.ctx.frontiers if f.get("id") == args.get("frontier_id")]
            if not match:
                refusal = {"refused": True, "reason": "frontier_id is not one of the frontiers in this step's context; "
                           "choose an id from the list (coordinates are never invented)"}
            elif self.ctx.event_kind in ("avoid_fire", "retreat"):
                refusal = {"refused": True, "reason": "no exploring during a heat event: let the reflex layer back away"}
            else:
                args = {**args, "x": float(match[0]["x"]), "y": float(match[0]["y"])}
        elif name == "go_to":
            # No outline exists (no prior knowledge); the brain refuses targets not known to be indoors.
            if self.ctx.event_kind in ("avoid_fire", "retreat"):
                refusal = {"refused": True, "reason": "no go_to during a heat event: use track_fire_spread and avoid_area, "
                           "and let the reflex layer back away"}
        elif name == "approach_candidate":
            refusal = self._need(args, ("inspect_candidate",), lambda ev: ev.get("x") is not None,
                                 "the cited inspection has no position")
            if refusal is None:
                ev = self.evidence[args["evidence_id"]]
                key = (self.ctx.robot, round(ev["x"] / self.LOOK_CELL_M), round(ev["y"] / self.LOOK_CELL_M))
                looks = [s for s in self._looks.get(key, []) if self.ctx.t - s < self.LOOK_MEMORY_S]
                if len(looks) >= self.MAX_LOOKS:
                    refusal = {"refused": True, "reason": f"already looked closer {len(looks)} times at this spot; leave it, "
                               "tell your partner it is unconfirmed, and continue exploring"}
                    self.refusals.append({"tool": name, "args": args, **refusal})
                    return refusal
                self._looks[key] = looks + [self.ctx.t]
                dx, dy = ev["x"] - self.ctx.x, ev["y"] - self.ctx.y
                dist = max(1e-6, (dx * dx + dy * dy) ** 0.5)
                k = max(0.0, dist - 1.5) / dist
                args = {**args, "x": round(self.ctx.x + dx * k, 2), "y": round(self.ctx.y + dy * k, 2),
                        "reason": f"closer look at an uncertain warm region ({ev['x']}, {ev['y']})"}
                name = "go_to"
        elif name == "request_human_approval":
            refusal = self._need(args, tuple(f["function"]["name"] for f in PERCEPTION))
        elif name == "finish":
            self.finished = str(args.get("summary", ""))
            return {"ok": True}
        if refusal:
            self.refusals.append({"tool": name, "args": args, **refusal})
            return refusal
        order = {"type": name, **args, "t": self.ctx.t, "robot": self.ctx.robot}
        if "evidence_id" in args:
            order["evidence_tool"] = self.evidence[args["evidence_id"]]["tool"]
            if name == "request_human_approval":
                order["evidence"] = {k: v for k, v in self.evidence[args["evidence_id"]].items() if k not in ("intrinsics", "rgb_to_thermal")}
        self.orders.append(order)
        return {"ok": True, "queued": name}
