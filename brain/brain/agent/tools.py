"""Tools the robot agents can call: OpenCV 5 perception on the robot's own frames, and physical or radio actions.

Perception tools run here and return evidence with an id. Action tools do not act directly: they are checked, then
returned to the simulator as orders. The checks are code, not prompt: an action that claims something about the world
must cite the id of a perception result that supports it, or it is refused and the refusal goes back to the model.

    reg = ToolRegistry(robot_dir, context)       # context: own-frame pose, heading, robot id, sim time
    result = reg.call("scan_thermal", {})
    reg.call("confirm_survivor", {"x": 9.6, "y": 5.0, "evidence_id": result["people"][0]["inspect_id"]})
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path as _Path

# Thresholds come from config/brain.json, each with its reason there.
_CONFIG = json.loads((_Path(__file__).resolve().parents[3] / "config" / "brain.json").read_text())
CONFIRM_MAX_RANGE_M = float(_CONFIG["person_confirm_max_range_m"])
# The brain's own survivor standoff: inside the arm's reach (actuators.json rescue.reach_m), so a closer look can end
# in a lift (agent run 14 stopped 1.5 m short, at the edge of the 1.6 m reach).
APPROACH_STANDOFF_M = float(_CONFIG["survivor_standoff_m"])
# The same-person distance the brain uses everywhere (merging sightings, confirmations, carried-out reports).
SURVIVOR_MERGE_M = float(_CONFIG["survivor_merge_m"])
_ACTUATORS = json.loads((_Path(__file__).resolve().parents[3] / "config" / "actuators.json").read_text())
_NOZZLE, _RESCUE = _ACTUATORS["nozzle"], _ACTUATORS["rescue"]
# A walk to a person (seen or confirmed) ends inside the arm's measured reach (actuators.json rescue.approach_m).
RESCUE_APPROACH_M = float(_RESCUE["approach_m"])
# Carrying people out can be switched off (actuators.json rescue.enabled): the lift tool, the rescue list and the prompt's
# rescue text then never reach the model, and a walk to a person stops at the camera's viewing distance instead.
RESCUE_ENABLED = bool(_RESCUE.get("enabled", True))
_PERCEPTION = json.loads((_Path(__file__).resolve().parents[3] / "config" / "perception.json").read_text())
_CAMERA = json.loads((_Path(__file__).resolve().parents[3] / "config" / "sensors.json").read_text())["camera"]
# Half the colour camera's horizontal view (atan of half the width over fx: 37.6 deg). A person further off the heading
# than this is outside the picture, so an inspection there cannot see them (run 20: looks at 106 and 148 deg).
CAMERA_HALF_HFOV_DEG = math.degrees(math.atan(_CAMERA["width"] / 2 / _CAMERA["fx"]))
# What this robot's water can do against a fire, from its own engineering assumptions (config, each with a reason).
BURNING_KW_PER_M2 = float(_PERCEPTION["burning_kw_per_m2"])
ONE_ROBOT_COOLS_KW = float(_NOZZLE["flow_lps"]) * float(_NOZZLE["cooling_kj_per_l"]) * float(_NOZZLE["application_efficiency"])


def spray_option(req: dict) -> dict | None:
    """The fire blocking this robot's way out (or, failing that, its way to unsearched floor), judged against its water.

    The brain measures the burning floor near where fire cuts the route (burning_m2, its own thermal map, measured not
    forecast). Code compares the heat it releases with what this robot's stream removes, so the agent can choose between
    spraying alone, asking its partner to spray with it, and going around; the agent decides. Held-out seed 921: robot A,
    cut off by a burning wood pile, scanned at -91 deg (outside the camera's view), saw nothing and held for 25 s before it
    turned, scanned again and sprayed.
    """
    which = "exit" if req.get("fire_blocks_exit") else "frontier" if req.get("fire_blocks_frontier") else None
    block = req.get("fire_blocks_exit") or req.get("fire_blocks_frontier")
    if which is None or block.get("burning_m2") is None:
        return None
    area, water_s = float(block["burning_m2"]), float(req.get("water_l") or 0.0) / float(_NOZZLE["flow_lps"])
    kw = area * BURNING_KW_PER_M2
    if area <= 0:
        assessment = "no floor measured burning near the block: the margin there is heat or a forecast, not flames"
    elif water_s <= 0:
        assessment = "no water left"
    elif kw <= ONE_ROBOT_COOLS_KW:
        assessment = "this robot's stream removes more heat than the fire releases"
    elif kw <= 2 * ONE_ROBOT_COOLS_KW:
        assessment = "more than one stream removes; two robots spraying together remove more than it releases"
    else:
        assessment = "more than two robots' streams remove"
    bearing = float(block.get("bearing_deg", 0.0))
    return {"blocks": which, "bearing_deg": round(bearing, 1), "distance_m": block.get("distance_m"),
            "in_camera_view": abs(bearing) <= CAMERA_HALF_HFOV_DEG, "within_nozzle_reach": float(block.get("distance_m") or 99) <= float(_NOZZLE["range_m"]),
            "burning_m2": round(area, 2), "est_heat_release_kw": round(kw), "one_robot_stream_removes_kw": round(ONE_ROBOT_COOLS_KW),
            "water_s_left": round(water_s, 1), "assessment": assessment}
from pathlib import Path

from brain.agent import vision
from brain.agent.team import TeamState


def _fn(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties or {}, "required": required or []}}}


NUM = {"type": "number"}
STR = {"type": "string"}
EVIDENCE = {"type": "string", "description": "id of the OpenCV result that supports this action"}

PERCEPTION = [
    _fn("check_indoors", "Check whether a direction is indoors using this robot's depth frame. Returns indoors / "
        "outdoors / unknown, ceiling height, walls, void rays and an evidence id.",
        {"bearing_deg": NUM}, ["bearing_deg"]),
    _fn("scan_thermal", "OpenCV 5 thermal scan of the current view: fire hotspots (over 200 C) and person-sized warm "
        "regions, each with bearing (deg, + right), range (m) and building position."),
    _fn("inspect_candidate", "Verify one warm region with independent OpenCV cues (metric size, thermal structure, "
        "planarity, window depth, RGB contour). Returns verdict person / not_person / uncertain and an evidence id.",
        {"bearing_deg": NUM, "range_m": NUM}, ["bearing_deg", "range_m"]),
    _fn("assess_visibility", "Measure smoke and image quality of the colour camera (sharpness, contrast, haze, valid "
        "depth)."),
    _fn("track_fire_spread", "Measure fire growth and the hot front's speed and direction over the last seconds of "
        "thermal video.", {"seconds": NUM}),
    _fn("measure_opening", "Measure the clear width of the opening at a bearing from depth, and whether the robot fits.",
        {"bearing_deg": NUM}, ["bearing_deg"]),
]

ACTIONS = [
    _fn("explore_frontier", "Search one of the frontiers listed in the context next (unsearched floor the robot can reach). "
        "The robot's planner drives there, keeping clear of walls, and sweeps the area on arrival.",
        {"frontier_id": {"type": "integer"}, "reason": STR, "evidence_id": EVIDENCE}, ["frontier_id", "reason"]),
    _fn("go_to", "Drive to a building position (x, y in metres) by the robot's own planner.",
        {"x": NUM, "y": NUM, "reason": STR}, ["x", "y", "reason"]),
    _fn("approach_candidate", f"Walk to a viewpoint {APPROACH_STANDOFF_M:.1f} m short of an inspected warm region, on the line from the robot, "
        f"computed from the inspection itself; for a person the inspection saw, or one this robot confirmed there, the walk "
        f"stops {RESCUE_APPROACH_M:.1f} m short, inside the arm's reach. You get an agent_goal_reached event on arrival.", {"evidence_id": EVIDENCE},
        ["evidence_id"]),
    _fn("turn_to_face", "Turn in place to face a building position. You get agent_goal_reached when facing it.",
        {"x": NUM, "y": NUM, "reason": STR}, ["x", "y", "reason"]),
    _fn("confirm_survivor", "Report a survivor at a building position. Needs an inspect_candidate result with verdict "
        "person.", {"x": NUM, "y": NUM, "evidence_id": EVIDENCE}, ["x", "y", "evidence_id"]),
    _fn("reject_candidate", "Mark a warm region as not a person so nobody inspects it again. Needs an "
        "inspect_candidate result with verdict not_person.", {"x": NUM, "y": NUM, "evidence_id": EVIDENCE},
        ["x", "y", "evidence_id"]),
    _fn("withdraw_survivor", "Withdraw a survivor confirmed earlier at a position, when a fresh look shows nobody there. Needs "
        "an inspect_candidate result from this step aimed at that position, inside the camera's view and within the "
        f"confirmation range ({CONFIRM_MAX_RANGE_M:.1f} m), with depth reaching the spot (nothing nearer blocking it), that "
        "found no person: verdict not_person, or no warm body and no whole colour person detection.",
        {"x": NUM, "y": NUM, "evidence_id": EVIDENCE}, ["x", "y", "evidence_id"]),
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
    # Held-out seed 914 run 5: this description named only a fresh person verdict taken within the reach; a lying person
    # that close is below the colour camera's view, so both agents re-inspected confirmed people for 70 s and never lifted
    # anyone. It now states both ways the check below accepts.
    _fn("extract_survivor", "Strap the person directly ahead onto the rescue sled and take them out to the nearest known "
        f"way out, slowly. Cite the evidence that confirmed a person in survivors_to_rescue while that person is within the "
        f"rescue reach ({float(_RESCUE['reach_m']):.1f} m) and within {float(_RESCUE['grip_cone_deg']):.0f} deg of the "
        "heading; or cite an inspect_candidate result with verdict person taken within that reach.",
        {"evidence_id": EVIDENCE}, ["evidence_id"]),
    _fn("spray_water", "Spray water from the robot's nozzle toward a fire to knock it down. Needs a scan_thermal result "
        "with a hot spot near that bearing and within the nozzle's range. Water is limited (context water_l); the "
        "robot holds still while spraying.", {"bearing_deg": NUM, "seconds": NUM, "evidence_id": EVIDENCE},
        ["bearing_deg", "seconds", "evidence_id"]),
    _fn("send_team_message", "Radio a short message to the partner robot and the commander. It may be delayed or "
        "lost if walls or distance block the link. kind: finding (what you sensed) or outcome (how an agreed task ended).",
        {"text": STR, "kind": {"type": "string", "enum": ["finding", "outcome"]}}, ["text"]),
    _fn("propose_plan", "Radio a proposed division of work to the partner: what you will do and what you ask of it, with why. "
        "It stays open for the partner's answer for a limited time (team.json proposal_ttl_s). target (optional) is a position in "
        "your map that both parts refer to; only with your maps linked, so it is a position in your partner's map too.",
        {"my_part": STR, "your_part": STR, "why": STR, "target": {"type": "object", "properties": {"x": NUM, "y": NUM}, "required": ["x", "y"]}},
        ["my_part", "your_part", "why"]),
    _fn("answer_proposal", "Accept or reject a plan your partner proposed (team.awaiting_your_answer), with the reason.",
        {"proposal_id": STR, "accept": {"type": "boolean"}, "reason": STR}, ["proposal_id", "accept", "reason"]),
    _fn("request_human_approval", "Ask the human incident commander before a risky move, citing evidence.",
        {"question": STR, "evidence_id": EVIDENCE}, ["question", "evidence_id"]),
    _fn("finish", "End this decision step with a one-sentence summary of what you decided and why.",
        {"summary": STR}, ["summary"]),
]

ALL_TOOLS = PERCEPTION + ACTIONS


def tool_specs() -> list:
    """The tools offered to the model this step: all of them, less the lift when rescue is switched off."""
    return [t for t in ALL_TOOLS if RESCUE_ENABLED or t["function"]["name"] != "extract_survivor"]
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
    water_l: float = 0.0                                      # water left in the robot's tank
    carrying_person: bool = False                             # a person is already on the sled
    survivors: list = field(default_factory=list)             # people confirmed so far, [x, y] in this robot's frame
    maps_linked: bool = False                                 # this robot's frame is linked with its partner's


class ToolRegistry:
    # Close looks per spot, kept across steps (the server builds a new registry per step): (robot, cell) -> times.
    _looks: dict = {}
    # Where each robot has lifted someone, from its own extract orders, so a lifted person is not offered again.
    _lifted: dict = {}
    # People each robot confirmed, with the confirming inspection: robot -> [{"id", "x", "y"}]. A lying person within
    # the arm's reach is below the colour camera's view (1.6 m up, pitched 10 deg down, 60 deg vertical view: floor visible
    # from about 1.9 m), so a lift cites the inspection that confirmed them (run 25: re-inspection at 1.34 m, uncertain).
    _confirmed: dict = {}
    # The uncertain candidate each robot is walking to look at closer: robot -> {"x", "y", "t", "evidence_id"}. Until it is
    # confirmed or rejected, or the approach fails, explore_frontier is refused (hidden-corner run 3: both agents ordered
    # an approach and dropped it on the next area_searched event, and nobody was confirmed).
    _pending: dict = {}
    # Events that mean the approach failed: the robot could not get there, or must leave for heat.
    APPROACH_FAILED = ("goal_blacklisted", "no_progress", "collision_hold", "stuck", "avoid_fire", "retreat", "escape_closing")
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

    def pending(self) -> dict | None:
        """This robot's unresolved approach, with distance and bearing from where it stands now; None when there is
        none, it has timed out (pending_candidate_s), or this step's event says the approach failed."""
        p = self._pending.get(self.ctx.robot)
        if p is None:
            return None
        if self.ctx.event_kind in self.APPROACH_FAILED or self.ctx.t - p["t"] > float(_PERCEPTION["pending_candidate_s"]):
            self._pending.pop(self.ctx.robot, None)
            return None
        dx, dy = p["x"] - self.ctx.x, p["y"] - self.ctx.y
        bearing = (math.degrees(math.atan2(dx, dy)) - self.ctx.heading_deg + 180) % 360 - 180
        return {"x": p["x"], "y": p["y"], "distance_m": round(math.hypot(dx, dy), 2), "bearing_deg": round(bearing, 1),
                "since_s": round(self.ctx.t - p["t"], 1), "evidence_id": p["evidence_id"]}

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
        # Where the matched warm body was measured, not where the agent asked to look: the scan's bearing and range are
        # rough, and confirmations inherited them (held-out seed 914 run 4: asked at -2.9 deg, 3.05 m; the body measured
        # at 3.2 deg, 3.52 m; the same person was then confirmed twice, 1.4 m apart).
        cand = r.get("candidate") or {}
        at = (cand["bearing_deg"], cand["range_m"]) if cand.get("bearing_deg") is not None and cand.get("range_m") is not None \
            else (float(bearing_deg), float(range_m))
        r["x"], r["y"] = (round(v, 2) for v in self.world(float(at[0]), float(at[1])))
        r["requested"] = {"bearing_deg": float(bearing_deg), "range_m": float(range_m)}
        return r

    def _p_check_indoors(self, bearing_deg: float):
        return vision.check_indoors(self._frame(), float(bearing_deg), output_dir=self.debug_dir)

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

    def _withdraw_refusal(self, args: dict) -> dict | None:
        """A confirmation is withdrawn only on a fresh, unblocked, close look at the spot that finds nobody (run 3: robot B
        confirmed a person from a colour sliver at the image edge and spent over a minute trying to reach them)."""
        x, y = float(args["x"]), float(args["y"])
        if not any(math.hypot(float(p[0]) - x, float(p[1]) - y) <= SURVIVOR_MERGE_M for p in self.ctx.survivors):
            return {"refused": True, "reason": "no confirmed survivor at that position"}
        refusal = self._need(args, ("inspect_candidate",))
        if refusal is not None:
            return refusal
        ev = self.evidence[str(args["evidence_id"])]
        bearing, rng = float(ev["requested"]["bearing_deg"]), float(ev["requested"]["range_m"])
        aim = self.world(bearing, rng)
        if math.hypot(aim[0] - x, aim[1] - y) > 1.0:
            return {"refused": True, "reason": "the cited inspection was not aimed within 1 m of that position"}
        if abs(bearing) > CAMERA_HALF_HFOV_DEG - 3:
            return {"refused": True, "reason": f"bearing {bearing:.0f} deg is at or outside the camera's view; turn_to_face the "
                    "position, then inspect it"}
        if rng > CONFIRM_MAX_RANGE_M:
            return {"refused": True, "reason": f"looked from {rng:.1f} m; withdraw only on a look from within {CONFIRM_MAX_RANGE_M:.1f} m"}
        dnn = (ev.get("cues") or {}).get("rgb_dnn") or {}
        whole = len(dnn.get("hits", [])) - int(dnn.get("clipped_by_side", 0))
        verdict = ev.get("verdict")
        if not (verdict == "not_person" or (verdict == "uncertain" and ev.get("candidate") is None and whole == 0)):
            return {"refused": True, "reason": f"the cited inspection does not show the spot empty (verdict {verdict}, "
                    f"warm body {'found' if ev.get('candidate') else 'none'}, whole colour detections {whole})"}
        frame = self._frame()
        u = frame.K[0, 2] + frame.K[0, 0] * math.tan(math.radians(bearing))
        col = frame.depth_m[:, max(0, int(u) - 3):int(u) + 4]
        z = col[(col > 0) & (col < 1e3)]
        reach = float(sorted(z.tolist())[int(.9 * (len(z) - 1))]) if len(z) >= 10 else 0.0
        if reach < rng * math.cos(math.radians(bearing)) - .5:
            return {"refused": True, "reason": f"depth along that bearing ends at {reach:.1f} m, before the spot: something nearer "
                    "blocks the view, so the spot was not seen"}
        return None

    def _confirmed_near(self, ev: dict) -> bool:
        """This robot confirmed a person within SURVIVOR_MERGE_M of the inspected spot."""
        if ev.get("x") is None:
            return False
        return any(math.hypot(c["x"] - float(ev["x"]), c["y"] - float(ev["y"])) <= SURVIVOR_MERGE_M
                   for c in self._confirmed.get(self.ctx.robot, []))

    @staticmethod
    def _inspection_range(ev: dict) -> float | None:
        """Measured range of the inspected person, else the requested range."""
        candidate = ev.get("candidate") or {}
        r = candidate.get("range_m")
        if r is None:
            r = (ev.get("requested") or {}).get("range_m")
        return None if r is None else float(r)

    def _near(self, ev: dict, args: dict, tol: float = 1.0) -> bool:
        return ev.get("x") is None or math.hypot(ev["x"] - float(args["x"]), ev["y"] - float(args["y"])) <= tol

    def _set_down_near(self, x: float, y: float) -> tuple[float, float, float] | None:
        """Where this robot set down a person it carried out, if (x, y) is within SURVIVOR_MERGE_M of it, else None. From the
        robot's own decision log (person_carried_out, own frame), which the recorder writes out before every agent call.
        Held-out run 2: robot B's agent confirmed the person B had set down in the yard as a new survivor."""
        path = self.robot_dir / "decisions.jsonl"
        if not path.exists():
            return None
        for line in path.read_text().splitlines():
            if '"person_carried_out"' not in line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            n = event.get("numbers") or {}
            if "person_x_m" not in n or float(event.get("t", 0)) > self.ctx.t + 1e-6:
                continue
            if math.hypot(float(n["person_x_m"]) - x, float(n["person_y_m"]) - y) <= SURVIVOR_MERGE_M:
                return float(n["person_x_m"]), float(n["person_y_m"]), float(event["t"])
        return None

    def _action(self, name: str, args: dict) -> dict:
        refusal = None
        if name == "confirm_survivor":
            refusal = self._need(args, ("inspect_candidate",), lambda ev: ev.get("verdict") == "person" and self._near(ev, args),
                                 "the cited inspection is not a person verdict at that position")
            if refusal is None and (self.evidence[str(args["evidence_id"])].get("candidate") or {}).get("range_m") is None:
                # Held-out seed 920, robot B at 0 s: a person 11.6 m away through two doorways, no depth that far, and the
                # model asked to inspect at 1.3 m; with nothing measured the spot was its guess, and the confirmation
                # landed 5.6 m from anyone. A position needs a measured range to the body.
                refusal = {"refused": True, "reason": "the inspection measured no range to the person (no depth or LiDAR "
                           "return on the body), so the position is only the requested guess: approach_candidate citing it "
                           "to get closer along that bearing, then inspect again"}
            if refusal is None:
                looked = self._inspection_range(self.evidence[str(args["evidence_id"])])
                if looked is not None and looked > CONFIRM_MAX_RANGE_M:
                    refusal = {"refused": True, "reason": f"the inspection looked from {looked:.1f} m; confirm only within "
                               f"{CONFIRM_MAX_RANGE_M:.1f} m: approach_candidate citing it, then inspect again"}
            if refusal is None:
                spot = self._set_down_near(float(args["x"]), float(args["y"]))
                if spot is not None:
                    refusal = {"refused": True, "reason": f"this robot set down a person it carried out at ({spot[0]:.1f}, "
                               f"{spot[1]:.1f}) at {spot[2]:.0f} s; that is the person already rescued, not a new survivor. "
                               "Search the building for others."}
        elif name == "reject_candidate":
            refusal = self._need(args, ("inspect_candidate",), lambda ev: ev.get("verdict") == "not_person" and self._near(ev, args),
                                 "the cited inspection is not a not_person verdict at that position")
        elif name == "withdraw_survivor":
            refusal = self._withdraw_refusal(args)
        elif name == "avoid_area":
            refusal = self._need(args, ("scan_thermal", "track_fire_spread"),
                                 lambda ev: ev.get("tool") == "track_fire_spread" or bool(ev.get("hotspots")),
                                 "the cited scan found no hotspot")
        elif name == "extract_survivor" and not RESCUE_ENABLED:
            refusal = {"refused": True, "reason": "carrying people out is switched off for this mission; report the person instead"}
        elif name == "extract_survivor":
            reach = float(_RESCUE["reach_m"])
            known = next((c for c in self._confirmed.get(self.ctx.robot, []) if c["id"] == str(args.get("evidence_id"))), None)
            if known is not None and str(args.get("evidence_id")) not in self.evidence:
                # The inspection that confirmed this person: the robot must now stand within reach of them and face them.
                dx, dy = known["x"] - self.ctx.x, known["y"] - self.ctx.y
                bearing = (math.degrees(math.atan2(dx, dy)) - self.ctx.heading_deg + 180) % 360 - 180
                refusal = None
                if math.hypot(dx, dy) > reach:
                    refusal = {"refused": True, "reason": f"the confirmed person is {math.hypot(dx, dy):.1f} m away, beyond the "
                               f"arm's {reach:.1f} m reach; go_to their approach point first"}
                elif abs(bearing) > float(_RESCUE["grip_cone_deg"]):
                    refusal = {"refused": True, "reason": f"the confirmed person is {bearing:.0f} deg off the heading; "
                               "turn_to_face them first"}
                else:
                    self.evidence[str(args["evidence_id"])] = {"id": known["id"], "tool": "inspect_candidate",
                                                               "verdict": "person", "x": known["x"], "y": known["y"]}
            else:
                refusal = self._need(args, ("inspect_candidate",), lambda ev: ev.get("verdict") == "person" and
                                     self._inspection_range(ev) is not None and self._inspection_range(ev) <= reach,
                                     f"the cited inspection is not a person verdict within {reach:.1f} m; approach_candidate "
                                     "first, or cite the inspection that confirmed them once within reach")
            if refusal is None and self.ctx.carrying_person:
                refusal = {"refused": True, "reason": "already towing a person; take them out first"}
            if refusal is None:
                ev = self.evidence[str(args["evidence_id"])]
                if ev.get("x") is not None:
                    self._lifted.setdefault(self.ctx.robot, []).append((float(ev["x"]), float(ev["y"])))
                    # The arm straps on the person at this position only (review: it used to take whoever was nearest).
                    args = {**args, "target_x": round(float(ev["x"]), 2), "target_y": round(float(ev["y"]), 2)}
        elif name == "spray_water":
            aim, reach = float(_NOZZLE["aim_tolerance_deg"]), float(_NOZZLE["range_m"])
            bearing = float(args.get("bearing_deg", 0))
            refusal = self._need(args, ("scan_thermal",), lambda ev: any(
                abs(float(h.get("bearing_deg", 1e9)) - bearing) <= aim and h.get("range_m") is not None and float(h["range_m"]) <= reach
                for h in ev.get("hotspots") or []),
                f"the cited scan has no hot spot within {aim:.0f} deg of bearing {bearing:.0f} and {reach:.0f} m")
            if refusal is None and self.ctx.water_l <= 0:
                refusal = {"refused": True, "reason": "the water tank is empty"}
            if refusal is None:
                args = {**args, "seconds": round(min(max(0.0, float(args.get("seconds", 0))), float(_NOZZLE["max_spray_s"])), 2)}
        elif name == "set_torch":
            refusal = self._need(args, ("assess_visibility",))
        elif name in ("open_door", "clear_debris"):
            refusal = self._need(args, ("measure_opening",), lambda ev: ev.get("passable") is False,
                                 "the cited measurement says the way is passable; just drive")
        elif name == "explore_frontier":
            match = [f for f in self.ctx.frontiers if f.get("id") == args.get("frontier_id")]
            pending = self.pending()
            if pending is not None:
                refusal = {"refused": True, "reason": f"you are walking to look closer at an unresolved warm candidate "
                           f"{pending['distance_m']:.1f} m away at bearing {pending['bearing_deg']:.0f} deg; finish that first: "
                           "once there, inspect_candidate at it, then confirm_survivor or reject_candidate (or finish this "
                           "step without a new order to keep walking)"}
            elif not match:
                refusal = {"refused": True, "reason": "frontier_id is not one of the frontiers in this step's context; "
                           "choose an id from the list (coordinates are never invented)"}
            elif self.ctx.event_kind in ("avoid_fire", "retreat"):
                refusal = {"refused": True, "reason": "no exploring during a heat event: let the reflex layer back away"}
            else:
                frontier = match[0]
                if frontier.get("proven_indoors", True) is False:
                    dx, dy = float(frontier["x"]) - self.ctx.x, float(frontier["y"]) - self.ctx.y
                    bearing = (math.degrees(math.atan2(dx, dy)) - self.ctx.heading_deg + 180) % 360 - 180
                    refusal = self._need(args, ("check_indoors",), lambda ev:
                        ev.get("verdict") == "indoors" and
                        abs((float(ev["bearing_deg"]) - bearing + 180) % 360 - 180) <= float(ev["bearing_tolerance_deg"]),
                        "the cited indoor check must say indoors within its bearing tolerance of this frontier")
                    if refusal is None:
                        args = {**args, "unproven": True}
                if refusal is None:
                    args = {**args, "x": float(frontier["x"]), "y": float(frontier["y"])}
        elif name == "go_to":
            # No outline exists (no prior knowledge); the brain refuses targets not known to be indoors.
            if self.ctx.event_kind in ("avoid_fire", "retreat"):
                refusal = {"refused": True, "reason": "no go_to during a heat event: use track_fire_spread and avoid_area, "
                           "and let the reflex layer back away"}
        elif name == "approach_candidate":
            refusal = self._need(args, ("inspect_candidate",), lambda ev: ev.get("x") is not None,
                                 "the cited inspection has no position")
            if refusal is None and (self.evidence[args["evidence_id"]].get("verdict") == "person" or self._confirmed_near(self.evidence[args["evidence_id"]])):
                # Walking to a person the cited inspection saw is the way to the arm's reach, not another uncertain look, so
                # the close-look limit does not apply and nothing is left pending (held-out seed 914 run 4: robot B confirmed
                # a woman 3 m away, confirm_survivor said "approach_candidate, then extract", and approach_candidate refused
                # after two earlier looks at that spot; B never came within reach in the 145 s left). Run 5: robot A cited a
                # newer uncertain look at the person it had confirmed and was refused the same way.
                ev = self.evidence[args["evidence_id"]]
                dx, dy = ev["x"] - self.ctx.x, ev["y"] - self.ctx.y
                dist = max(1e-6, (dx * dx + dy * dy) ** 0.5)
                k = max(0.0, dist - (RESCUE_APPROACH_M if RESCUE_ENABLED else APPROACH_STANDOFF_M)) / dist
                args = {**args, "x": round(self.ctx.x + dx * k, 2), "y": round(self.ctx.y + dy * k, 2),
                        "face_x": ev["x"], "face_y": ev["y"],
                        "reason": f"reach the person seen at ({ev['x']}, {ev['y']}) to lift them"}
                name = "go_to"
            elif refusal is None:
                ev = self.evidence[args["evidence_id"]]
                key = (self.ctx.robot, round(ev["x"] / self.LOOK_CELL_M), round(ev["y"] / self.LOOK_CELL_M))
                looks = [s for s in self._looks.get(key, []) if self.ctx.t - s < self.LOOK_MEMORY_S]
                if len(looks) >= self.MAX_LOOKS:
                    self._pending.pop(self.ctx.robot, None)
                    refusal = {"refused": True, "reason": f"already looked closer {len(looks)} times at this spot; leave it, "
                               "tell your partner it is unconfirmed, and continue exploring"}
                    self.refusals.append({"tool": name, "args": args, **refusal})
                    return refusal
                self._looks[key] = looks + [self.ctx.t]
                dx, dy = ev["x"] - self.ctx.x, ev["y"] - self.ctx.y
                dist = max(1e-6, (dx * dx + dy * dy) ** 0.5)
                k = max(0.0, dist - APPROACH_STANDOFF_M) / dist
                args = {**args, "x": round(self.ctx.x + dx * k, 2), "y": round(self.ctx.y + dy * k, 2),
                        "face_x": ev["x"], "face_y": ev["y"],
                        "reason": f"closer look at an uncertain warm region ({ev['x']}, {ev['y']})"}
                name = "go_to"
                self._pending[self.ctx.robot] = {"x": float(ev["x"]), "y": float(ev["y"]), "t": self.ctx.t,
                                                 "evidence_id": str(args["evidence_id"])}
        elif name == "turn_to_face":
            # A go_to where the robot stands, with a point to face: the brain arrives at once, turns, then reports.
            if self.ctx.event_kind in ("avoid_fire", "retreat"):
                refusal = {"refused": True, "reason": "no turning toward a target during a heat event; let the reflex layer back away"}
            else:
                args = {"x": round(self.ctx.x, 2), "y": round(self.ctx.y, 2), "face_x": float(args["x"]), "face_y": float(args["y"]),
                        "reason": str(args.get("reason") or "turn to face")}
                name = "go_to"
        elif name == "send_team_message":
            # The radio carries the robot's own state next to the model's words, written by code from what the tools
            # accepted, so a listener can tell a claim from a fact (held-out seed 917, robot B: confirm_survivor was
            # refused and the next call radioed "Confirmed survivor"). Filtering words cannot catch a paraphrase.
            mine = self._confirmed.get(self.ctx.robot, [])
            state = ("confirmed by this robot so far, own map: " + ", ".join(f"({c['x']:.1f}, {c['y']:.1f})" for c in mine)
                     if mine else "this robot has confirmed nobody")
            refused = [r for r in self.refusals if r.get("tool") == "confirm_survivor"]
            if refused:
                state += "; a confirmation was refused this step: " + str(refused[-1].get("reason", ""))[:90]
            args = {**args, "text": f"{str(args.get('text', '')).strip()} [robot state: {state}]"}
            kind = args.get("kind") if args.get("kind") in ("finding", "outcome") else "finding"
            args = {**args, **TeamState.of(self.ctx.robot).message(kind, args["text"], self.ctx.t)}
        elif name == "propose_plan" and args.get("target") is not None and not self.ctx.maps_linked:
            refusal = {"refused": True, "reason": "your maps are not linked, so a position in your map means nothing to your "
                       "partner: describe the place in words instead"}
        elif name == "propose_plan":
            target = args.get("target")
            msg = TeamState.of(self.ctx.robot).propose(str(args.get("my_part", "")), str(args.get("your_part", "")),
                                                       str(args.get("why", "")), self.ctx.t,
                                                       target={"x": round(float(target["x"]), 2), "y": round(float(target["y"]), 2)} if target else None)
            self.orders.append({"type": "send_team_message", **msg, "robot": self.ctx.robot})
            return {"ok": True, "queued": "proposal", "proposal_id": msg["proposal_id"], "expires_t": msg["expires_t"]}
        elif name == "answer_proposal":
            team = TeamState.of(self.ctx.robot)
            why = team.answer_refusal(str(args.get("proposal_id")), self.ctx.t)
            if why is not None:
                refusal = {"refused": True, "reason": why}
            else:
                msg = team.answer(str(args["proposal_id"]), bool(args.get("accept")), str(args.get("reason", "")), self.ctx.t)
                self.orders.append({"type": "send_team_message", **msg, "robot": self.ctx.robot})
                return {"ok": True, "queued": msg["kind"], "reply_to": msg["reply_to"]}
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
        if name in ("confirm_survivor", "reject_candidate"):
            p = self._pending.get(self.ctx.robot)
            if p is not None and math.hypot(p["x"] - float(args["x"]), p["y"] - float(args["y"])) <= self.LOOK_CELL_M:
                self._pending.pop(self.ctx.robot, None)
        if name == "withdraw_survivor":
            self._confirmed[self.ctx.robot] = [c for c in self._confirmed.get(self.ctx.robot, [])
                                               if math.hypot(c["x"] - float(args["x"]), c["y"] - float(args["y"])) > SURVIVOR_MERGE_M]
        if name == "confirm_survivor":
            self._confirmed.setdefault(self.ctx.robot, []).append(
                {"id": str(args["evidence_id"]), "x": float(args["x"]), "y": float(args["y"])})
            # Run 15: robot A confirmed a person from 2.7 m and searched on; the next step to get them out was only in the
            # prompt. The result says it, from the same inspection.
            looked, reach = self._inspection_range(self.evidence[str(args["evidence_id"])]), float(_RESCUE["reach_m"])
            if not RESCUE_ENABLED:
                return {"ok": True, "queued": name}
            if self.ctx.carrying_person:
                hint = "you are already towing a person; report this one to your partner and the commander"
            elif looked is not None and looked <= reach:
                hint = f"within the arm's {reach:.1f} m reach: extract_survivor citing {args['evidence_id']}"
            else:
                hint = f"beyond the arm's {reach:.1f} m reach: approach_candidate citing {args['evidence_id']}, then extract on arrival"
            return {"ok": True, "queued": name, "next": hint}
        return {"ok": True, "queued": name}


def survivors_to_rescue(robot: str, pose: dict, survivors: list, standoff_m: float = RESCUE_APPROACH_M,
                        lifted_within_m: float = 1.5) -> list[dict]:
    """Confirmed people this robot has not lifted, nearest first, with the bearing (degrees, right positive) and
    distance from the robot and an approach point standoff_m short of them on the line from the robot. Run 19: robot A
    confirmed a person, a heat alarm interrupted the approach, and nothing brought the agent back to them."""
    lifted = ToolRegistry._lifted.get(robot, [])
    out = []
    for p in survivors or []:
        x, y = float(p[0]), float(p[1])
        if any(math.hypot(x - lx, y - ly) <= lifted_within_m for lx, ly in lifted):
            continue
        dx, dy = x - float(pose["x"]), y - float(pose["y"])
        dist = math.hypot(dx, dy)
        bearing = (math.degrees(math.atan2(dx, dy)) - float(pose["heading_deg"]) + 180) % 360 - 180
        k = max(0.0, dist - standoff_m) / max(dist, 1e-6)
        ev = next((c["id"] for c in ToolRegistry._confirmed.get(robot, []) if math.hypot(c["x"] - x, c["y"] - y) <= lifted_within_m), None)
        out.append({"x": round(x, 2), "y": round(y, 2), "bearing_deg": round(bearing, 1), "distance_m": round(dist, 2),
                    "in_view": abs(bearing) <= CAMERA_HALF_HFOV_DEG, "confirmed_by": ev,
                    "approach": {"x": round(float(pose["x"]) + dx * k, 2), "y": round(float(pose["y"]) + dy * k, 2)}})
    return sorted(out, key=lambda s: s["distance_m"])
