using System;
using System.Collections.Generic;
using UnityEngine;

namespace SmokeRescue.Brain
{
    /// <summary>
    /// Orders from the robot's LLM agent and the commander. The agent decides from OpenCV evidence; the reflex brain
    /// still owns motion, collision and heat safety, so an order can be overruled by a hotter fire or a blocked path.
    /// Every order is logged as an "agent_order" decision with the evidence id that justified it.
    /// </summary>
    public sealed partial class RobotBrain
    {
        Vector2? agentGoal;
        double agentGoalUntil;
        string agentGoalReason;

        bool AgentGoalActive(double t) => agentGoal.HasValue && t <= agentGoalUntil && !returning && !retreating;

        /// <summary>Drive to target; with face set, turn in place on arrival until the robot faces that point, and only
        /// then tell the agent it arrived (the camera sees about 37 degrees either side, the arm grips within 60).</summary>
        public void AgentGoTo(Vector2 target, double t, double holdS, string reason, Vector2? face = null) => GoTo(target, t, holdS, reason, face, false);

        void GoTo(Vector2 target, double t, double holdS, string reason, Vector2? face, bool exploring)
        {
            // Turning where the robot stands (turn_to_face) goes nowhere new, so it needs no indoor evidence (Codex review:
            // near an entrance the robot's own cell may not be known to be inside yet).
            bool turnInPlace = face.HasValue && Vector2.Distance(target, motion.Pose.position) <= c.goal_tolerance_m;
            if (!turnInPlace && !InSearchArea(target)) { Emit(t, "agent_order", "Refused go_to: no ceiling has been seen over that point, so it is not known to be inside.", "x_m", target.x, "y_m", target.y); return; }
            string busy = AgentMoveRefusal(t);
            if (busy != null) { Emit(t, "agent_order", "Refused go_to: " + busy + ".", "x_m", target.x, "y_m", target.y); return; }
            agentGoal = target; agentGoalUntil = t + holdS; agentGoalReason = reason ?? ""; agentGoalIsFrontier = false;
            agentFace = face; facing = null; agentHoldUntil = double.NegativeInfinity;
            if (active == null && !KeepEvidenceLook(exploring)) ClearPath();
            Emit(t, "agent_order", "go_to: " + agentGoalReason, "x_m", target.x, "y_m", target.y, "hold_s", holdS);
        }

        bool agentGoalIsFrontier;
        Vector2? agentFace, facing; double facingUntil; string facingReason;
        double agentHoldUntil = double.NegativeInfinity;

        /// <summary>Arrival at an agent go_to: start facing the named point, or tell the agent now.</summary>
        void AgentArrived(double t, string how)
        {
            if (agentFace.HasValue)
            {
                facing = agentFace; facingUntil = t + c.sweep_max_s; facingReason = agentGoalReason;
                Emit(t, "goal", "At the agent's goal; turn to face the point it named.", "x_m", agentFace.Value.x, "y_m", agentFace.Value.y);
            }
            else
            {
                Emit(t, "agent_goal_reached", how + agentGoalReason, "x_m", agentGoal.Value.x, "y_m", agentGoal.Value.y);
                agentHoldUntil = t + c.agent_arrival_hold_s;
            }
            agentFace = null;
        }

        /// <summary>Turning in place toward the point an agent go_to named; null when not facing. Run 20: robot A stood
        /// within the arm's reach of a confirmed person three times, but looked at them 106 to 148 degrees off its heading,
        /// outside the camera's view, so every inspection came back uncertain and the lift was refused.</summary>
        BrainCommand? FaceStep(double t, float dt)
        {
            if (!facing.HasValue) return null;
            if (returning || retreating) { facing = null; return null; }
            float error = BrainMath.Angle(motion.Pose.yawRad, BrainMath.Heading(facing.Value - motion.Pose.position));
            float tolerance = c.survivor_face_tolerance_deg;
            if (c.motion_face_visible)
            {
                var cam = sensors.camera; var thermal = sensors.thermal;
                float rgbHalf = Mathf.Atan(Mathf.Min(cam.cx + .5f, cam.width - .5f - cam.cx) / cam.fx) * Mathf.Rad2Deg;
                float thermalHalf = Mathf.Atan(Mathf.Min(thermal.cx + .5f, thermal.width - .5f - thermal.cx) / thermal.fx) * Mathf.Rad2Deg;
                tolerance = Mathf.Max(tolerance, .5f * Mathf.Min(rgbHalf, thermalHalf));
            }
            if (Mathf.Abs(error) > tolerance && t < facingUntil)
                return Command(0, Mathf.Clamp(error * c.heading_gain, -c.sweep_yaw_dps, c.sweep_yaw_dps), dt, true);
            Emit(t, "agent_goal_reached", "Arrived where the agent asked and turned to face the point it named: " + facingReason,
                "x_m", facing.Value.x, "y_m", facing.Value.y, "bearing_error_deg", error);
            facing = null; agentHoldUntil = t + c.agent_arrival_hold_s;
            return null;
        }

        /// <summary>The frontiers the agent chooses from: reachable, worth searching, not claimed by a partner, inside
        /// the search area or explicitly unproven for an agent indoor check. Positions are walkable cells; path_m is the planned distance
        /// including heat and wall costs; unsearched_m2 is the floor no camera on the team has seen near it.</summary>
        public IReadOnlyList<BrainFrontier> FrontierOptions(double t, int max)
        {
            if (max <= 0) return Array.Empty<BrainFrontier>();
            planner.Rooms.Update(t, motion.Pose.position, InSearchAreaOwn);
            // Keep the indoor boundary when clustering, so a nearby proven standpoint cannot hide an unproven
            // doorway in the same cluster. Wide slots compete across both sets before either set is truncated.
            int claimed = 0, blacklisted = 0, visited = 0, outside = 0;
            bool Excluded(Vector2 p)
            {
                if (ClaimBlocks(p, t)) { claimed++; return true; }
                if (GoalBlacklisted(p, t)) { blacklisted++; return true; }
                if (VisitedGoal(p, t)) { visited++; return true; }
                if (!InSearchArea(p)) { outside++; return true; }
                return false;
            }
            var proven = planner.Frontiers(motion.Pose.position, Excluded);
            // The same fallback as the brain's own choice: when only a partner's claim or searched floor hides the indoor
            // frontiers, offer them, so the agent is not left with yard options alone (one-door seed 922, robot B).
            if (proven.Count == 0 && peers.Count > 0)
                proven = planner.Frontiers(motion.Pose.position, p => GoalBlacklisted(p, t) || VisitedGoal(p, t) || !InSearchArea(p), true);
            // Still nothing proven: the way in is behind the partner (one-door seed 923, robot B): offer walking in after it.
            if (proven.Count == 0 && peers.Count > 0 && FollowPartnerGoal(out var followRoute) is Vector2 follow)
            {
                float cost = 0; for (int i = 1; i < followRoute.Count; i++) cost += Vector2.Distance(followRoute[i - 1], followRoute[i]);
                proven.Add(new GridPlanner.FrontierChoice { position = follow, claimCentre = follow, path = followRoute, cost = cost, score = cost, information = 0, towardPartner = true });
            }
            var unproven = planner.Frontiers(motion.Pose.position,
                p => ClaimBlocks(p, t) || GoalBlacklisted(p, t) || VisitedGoal(p, t) || InSearchArea(p), false, false);
            // With proven indoor floor to search, an unproven frontier on ground this robot's own LiDAR saw as open, with no
            // mapped floor beyond it, is yard, not a way in: it is not offered (one-door seed 922, robot B: 6 of its 8
            // options were such yard frontiers, larger on paper than the 2 proven ones; its agent turned to check them for
            // 14 s). Without a proven option they stay, so a robot outside can still find its way in.
            if (proven.Count > 0)
                unproven.RemoveAll(f => grid.OpenGroundAt(f.position) && grid.UnsearchedFloorBeyond(f.position, c.frontier_beyond_m) < c.frontier_wide_min_m2);
            // Merge the already sorted sets. Equal scores prefer proven evidence, as does the local-slot fill below.
            var all = new List<GridPlanner.FrontierChoice>(); int pi = 0, ui = 0;
            while (pi < proven.Count || ui < unproven.Count)
                if (ui >= unproven.Count || (pi < proven.Count && proven[pi].score <= unproven[ui].score)) all.Add(proven[pi++]);
                else all.Add(unproven[ui++]);
            planner.Rooms.Rank(all, motion.Pose.position);
            var picked = new List<int>(); int wide = 0, wideUnproven = 0;
            if (c.room_ledger && all.Count > 0) picked.Add(0);
            if (c.frontier_wide_options > 0 && all.Count > max)
            {
                var label = grid.UnsearchedRegions(); var nearestOf = new Dictionary<int, int>();
                for (int i = 0; i < all.Count; i++)
                    foreach (int l in grid.RegionsNear(label, all[i].position)) if (!nearestOf.ContainsKey(l)) nearestOf[l] = i;
                var regions = new List<(int index, float m2)>();
                foreach (int i in new HashSet<int>(nearestOf.Values))
                {
                    // Ranking must include unproven mapped floor. The reported area keeps its indoor-only meaning,
                    // and provenIndoors stays false until the robot has evidence. The tool still requires check_indoors.
                    float m2 = grid.UnsearchedFloorBeyond(all[i].position, c.frontier_beyond_m, false);
                    if (m2 >= c.frontier_wide_min_m2) regions.Add((i, m2));
                }
                regions.Sort((a, b) => { int k = b.m2.CompareTo(a.m2); return k != 0 ? k : a.index.CompareTo(b.index); });
                for (int k = 0; k < regions.Count && picked.Count < max && wide < Math.Min(max, c.frontier_wide_options); k++)
                {
                    int i = regions[k].index; if (picked.Contains(i)) continue; picked.Add(i); wide++;
                    if (!InSearchArea(all[i].position)) wideUnproven++;
                }
            }
            for (int pass = 0; pass < 2; pass++)
                for (int i = 0; i < all.Count && picked.Count < max; i++)
                    if (InSearchArea(all[i].position) == (pass == 0) && !picked.Contains(i)) picked.Add(i);
            picked.Sort();
            var list = new List<BrainFrontier>();
            foreach (int index in picked)
            {
                var f = all[index];
                list.Add(new BrainFrontier { id = list.Count, position = f.position, pathM = f.cost, towardPartner = f.towardPartner,
                    unsearchedM2 = f.information * grid.CellM * grid.CellM, provenIndoors = InSearchArea(f.position),
                    mappedUnsearchedM2 = grid.UnsearchedFloorBeyond(f.position, c.frontier_beyond_m),
                    roomId = c.room_ledger ? planner.Rooms.RoomForFrontier(f.position) : -1 });
            }
            Emit(t, "frontier_options", $"{proven.Count} proven and {unproven.Count} unproven frontiers; boundary cells dropped from the proven set as claimed by a partner, blacklisted, visited, or not known to be indoors.",
                "frontiers", proven.Count, "cells_claimed", claimed, "cells_blacklisted", blacklisted, "cells_visited", visited, "cells_not_indoors", outside,
                "unproven_frontiers", unproven.Count, "wide_options", wide, "wide_unproven_options", wideUnproven);
            return list;
        }

        /// <summary>Estimated room labels for agent context, in this robot's frame. IDs are local to this robot.
        /// A split or merge can change IDs; visit time stays attached to sensed pose cells.</summary>
        public IReadOnlyList<RoomLedger.Room> RoomOptions(double t)
        {
            planner.Rooms.Update(t, motion.Pose.position, InSearchAreaOwn);
            return c.room_ledger ? planner.Rooms.Rooms : Array.Empty<RoomLedger.Room>();
        }

        /// <summary>For the agent: the nearest unsearched floor this robot's map shows but cannot reach, with its bearing
        /// (degrees, right positive, from the robot's heading), distance, area and a reachable approach point, so the model
        /// can look at what blocks it (measure_opening) and clear debris or open a door. Null when there is none.
        /// Run 13: robot B gave up 2 m short of the hallway debris without looking at it.</summary>
        public BrainBlockedFloor? BlockedUnsearched()
        {
            var found = planner.NearestBlocked(motion.Pose.position, InSearchArea);
            if (!found.HasValue) return null;
            Vector2 d = found.Value.point - motion.Pose.position;
            float bearingRight = -BrainMath.Angle(motion.Pose.yawRad, Mathf.Atan2(-d.x, d.y));
            return new BrainBlockedFloor { point = found.Value.point, approach = found.Value.approach, bearingDegRight = bearingRight,
                distanceM = d.magnitude, areaM2 = found.Value.cells * grid.CellM * grid.CellM };
        }

        /// <summary>
        /// For the agent, when no frontier is reachable: is fire what cuts them off? The frontiers are planned again as if
        /// there were no fire (water is what would clear it; the robot never drives this plan); if one is reachable that
        /// way, its route first enters the fire margin at the point returned, which is where to aim the water. Null when the fire margin is not the cause. Agent
        /// run 16: a burning wood pile closed the 2 m hallway, every frontier vanished, and the agent, told only of floor
        /// it could not reach, measured for a door instead of using the hose it carried.
        /// </summary>
        /// <summary>
        /// For the agent, when no route to a known way out is open at the current fire clearance: the way out planned as
        /// if there were no fire margin (never driven), and where that route first enters the margin, which is where water
        /// would open it. Null when a way out is open or none exists even past the fire. Held-out seed 914 run 5: robot A
        /// held at the coolest floor while its air rose from 40 to 185 C; its agent heard only where fire blocks frontiers.
        /// </summary>
        public BrainFireBlock? FireBlockedExit(double t)
        {
            if (PlanOut().Count > 0) return null;
            float saved = grid.FireClearance; List<Vector2> route;
            try { grid.FireClearance = -1; route = PlanOut(); }
            finally { grid.FireClearance = saved; }
            float margin = c.fire_clearance_m + grid.CellM * .707107f, routeM = 0;
            for (int i = 1; i < route.Count; i++) routeM += Vector2.Distance(route[i - 1], route[i]);
            foreach (var p in route)
            {
                if (grid.FireDistance(grid.Cell(p)) > margin) continue;
                Vector2 d = p - motion.Pose.position;
                return new BrainFireBlock { blockAt = p, frontier = route[route.Count - 1], bearingDegRight = -BrainMath.Angle(motion.Pose.yawRad, Mathf.Atan2(-d.x, d.y)),
                    distanceM = d.magnitude, frontierPathM = routeM, unsearchedM2 = 0, burningM2 = grid.BurningAreaNear(p, c.burning_radius_m) };
            }
            return null;
        }
        /// <summary>The nearest floor this robot measured burning within fire_near_m (the nozzle's throw), whatever the air at
        /// the robot: the fire water can reach from here, whether or not it blocks a route. Seed 929, robot B: retreating in its
        /// own living room, its air rose from 30 to 211 C; the spray option appeared only for a fire blocking a way out or a
        /// frontier, so its agent was never offered the water it carried (12 L, none used). Held-out runs 1 and 2 (seed 913):
        /// with the offer also gated on the robot's air reaching air_warm_c, a fire in reach in cool air was never offered;
        /// the air gate is gone (issue list 23 Oct, item 8).</summary>
        public BrainFireBlock? FireNearRobot(double t) => NearestFire(c.fire_near_m);
        /// <summary>The nearest floor this robot's own thermal map holds burning (heat at or above fire_c, never a forecast and
        /// never the simulator's fire), within reachM, or anywhere in the map when reachM is infinite: bearing (right positive),
        /// straight-line distance and the burning area around it. Null when the map holds no burning floor.</summary>
        public BrainFireBlock? NearestFire(float reachM = float.PositiveInfinity)
        {
            Vector2 here = motion.Pose.position; int centre = grid.Cell(here); if (centre < 0) return null;
            int r = float.IsPositiveInfinity(reachM) ? grid.Size : Mathf.CeilToInt(reachM / grid.CellM), cx = centre % grid.Size, cy = centre / grid.Size, best = -1;
            float bestM = float.PositiveInfinity;
            for (int y = Math.Max(0, cy - r); y <= Math.Min(grid.Size - 1, cy + r); y++)
                for (int x = Math.Max(0, cx - r); x <= Math.Min(grid.Size - 1, cx + r); x++)
                {
                    int i = y * grid.Size + x; if (grid.Heat(i) < c.fire_c) continue;
                    float m = Vector2.Distance(grid.Centre(i), here);
                    if (m <= reachM && m < bestM) { bestM = m; best = i; }
                }
            if (best < 0) return null;
            Vector2 p = grid.Centre(best), d = p - here;
            return new BrainFireBlock { blockAt = p, frontier = p, bearingDegRight = -BrainMath.Angle(motion.Pose.yawRad, Mathf.Atan2(-d.x, d.y)),
                distanceM = d.magnitude, frontierPathM = 0, unsearchedM2 = 0, burningM2 = grid.BurningAreaNear(p, c.burning_radius_m) };
        }
        /// <summary>The air at this robot's own sensor for the agent, null when no reading is recent. Held-out run 2 (seed 913),
        /// robot A: the agent saw only heat_budget, which read 1.0 at 122 s and 0 at 128 s while the air jumped from 44 to
        /// 375 C as the hot layer came down past the 0.6 m sensor; it radioed "no fire seen" at 378 C. The budget drains at
        /// heat_drain_per_degree_s per degree above heat_drain_start_c, so heatSecondsLeft = budget / that rate.</summary>
        public BrainAirState? AirState(double t)
        {
            if (t - lastAir > c.frame_max_age_s) return null;
            float drain = airC > c.heat_drain_start_c ? (airC - c.heat_drain_start_c) * c.heat_drain_per_degree_s : 0;
            return new BrainAirState { airC = airC, trendCPerMin = AirSlopePerMin(t), heatBudget = budget,
                heatSecondsLeft = drain > 0 ? budget / drain : (float?)null };
        }
        public BrainFireBlock? FireBlockedFrontier(double t)
        {
            float saved = grid.FireClearance;
            List<GridPlanner.FrontierChoice> all;
            try
            {
                grid.FireClearance = -1;   // fire distance is never below zero: every fire cell counts as clear
                // Blacklisted goals stay in: when fire cuts a route the brain blacklists that frontier (fire_cut_blacklist_s), and
                // filtering it here hid exactly the fire the agent could spray. Seed 934 (23 Oct), robot A: 6 fire-cut blacklists
                // from 119 s, fire_blocks_frontier never sent, 0 sprays with 12 L aboard.
                all = planner.Frontiers(motion.Pose.position, p => ClaimBlocks(p, t) || VisitedGoal(p, t) || !InSearchArea(p));
            }
            finally { grid.FireClearance = saved; }
            float margin = c.fire_clearance_m + grid.CellM * .707107f;
            foreach (var f in all)
                foreach (var p in f.path)
                {
                    if (grid.FireDistance(grid.Cell(p)) > margin) continue;
                    Vector2 d = p - motion.Pose.position;
                    return new BrainFireBlock { blockAt = p, frontier = f.position, bearingDegRight = -BrainMath.Angle(motion.Pose.yawRad, Mathf.Atan2(-d.x, d.y)),
                        distanceM = d.magnitude, frontierPathM = f.cost, unsearchedM2 = f.information * grid.CellM * grid.CellM,
                        burningM2 = grid.BurningAreaNear(p, c.burning_radius_m) };
                }
            return null;
        }

        /// <summary>Agent order: search this frontier next. It is driven like a go_to, and on arrival the robot sweeps
        /// and marks it visited exactly as for a frontier it chose itself.</summary>
        public void AgentExplore(Vector2 target, double t, double holdS, string reason, string indoorEvidence = null)
        {
            string busy = AgentMoveRefusal(t);
            if (busy != null) { Emit(t, "agent_order", "Refused explore_frontier: " + busy + ".", "x_m", target.x, "y_m", target.y); return; }
            if (!string.IsNullOrEmpty(indoorEvidence) && !InSearchArea(target))
            {
                // Floor the robot cannot prove is indoors, chosen by the agent on an OpenCV indoor check of its own frame.
                agentGoal = target; agentGoalUntil = t + holdS; agentGoalReason = "explore (unproven, OpenCV indoor check " + indoorEvidence + "): " + (reason ?? "");
                agentGoalIsFrontier = true; agentFace = null; facing = null; agentHoldUntil = double.NegativeInfinity;
                if (active == null && !KeepEvidenceLook(true)) ClearPath();
                Emit(t, "agent_order", "explore_frontier on floor not yet proven indoors, citing OpenCV evidence " + indoorEvidence + ": " + (reason ?? ""),
                    "x_m", target.x, "y_m", target.y, "hold_s", holdS);
                return;
            }
            GoTo(target, t, holdS, "explore: " + (reason ?? ""), null, true);
            agentGoalIsFrontier = agentGoal.HasValue;
        }

        /// <summary>An exploration order arriving during a bounded evidence look leaves the look's path alone: the order is
        /// kept as the agent goal and driven once the look ends (at most sweep_max_s). One-door seed 928: the agents re-sent
        /// their exploration order every 4 to 7 s; each one cleared the look's path and the look ended as "no longer safe" at
        /// 20 C (8 of 18 looks, every one at an exploration order; 3 finished), and neither robot left the foyer and kitchen in 120 s.
        /// Approaches, faces and other go_to orders still take over at once.</summary>
        bool KeepEvidenceLook(bool exploring) => exploring && c.frontier_doorway_looks && doorwayProbeGoal.HasValue;

        public void AgentConfirmSurvivor(Vector2 p, double t, string evidence)
        {
            if (AtOwnSetDown(p))
            {
                Emit(t, "agent_order", "Refused confirm_survivor: this robot set down a person it carried out there; that is the person already rescued.",
                    "x_m", p.x, "y_m", p.y);
                return;
            }
            foreach (var known in survivors) if (Vector2.Distance(known, p) <= c.survivor_merge_m) return;
            withdrawn.RemoveAll(w => Vector2.Distance(w, p) <= c.survivor_merge_m);
            survivors.Add(p);
            confirmed.Add(new BrainSurvivor { position = p, confirmedAt = t, sourceRobot = id });
            foreach (var track in tracks) if (Vector2.Distance(track.position, p) <= c.survivor_merge_m) track.confirmed = true;
            if (active != null && Vector2.Distance(active.position, p) <= c.survivor_merge_m) { active = null; dwell = 0; inspecting = false; ClearPath(); }
            Emit(t, "survivor_confirmed", "Agent confirmed a person with OpenCV evidence " + evidence + ".", "x_m", p.x, "y_m", p.y);
        }

        /// <summary>Spots this robot's agent withdrew a confirmation at; a partner's report of a survivor there is ignored
        /// until this robot confirms someone there again. (The radio format carries no withdrawals, so a partner keeps its
        /// copy until its own agent looks.)</summary>
        readonly List<Vector2> withdrawn = new List<Vector2>();
        /// <summary>Agent order: withdraw a confirmation. The tool layer allows it only on a fresh look aimed at the spot,
        /// from within confirmation range, with depth reaching it, that found nobody (held-out seed 914 run 3: robot B
        /// confirmed a person from a colour sliver at the image edge and spent over a minute trying to reach them). The track
        /// is marked rejected from here, so the same view does not raise it again.</summary>
        public void AgentWithdrawSurvivor(Vector2 p, double t, string evidence)
        {
            int before = survivors.Count; ForgetSurvivorNear(p); int removed = before - survivors.Count;
            withdrawn.Add(p);
            foreach (var track in tracks)
            {
                if (Vector2.Distance(track.position, p) > c.survivor_merge_m) continue;
                track.confirmed = false; track.candidate = false; track.retryAfter = double.PositiveInfinity;
                MarkRejected(track, t);
                if (active == track) { active = null; dwell = 0; inspecting = false; ClearPath(); }
            }
            Emit(t, "survivor_withdrawn", "Agent withdrew a confirmation: a fresh close look found nobody there, OpenCV evidence " + evidence + ".",
                "x_m", p.x, "y_m", p.y, "removed", removed);
        }

        void MarkRejected(Track track, double t)
        {
            track.rejected = true; track.rejectedFrom = motion.Pose.position; track.rejectedAt = t;
            track.rejectedUntil = double.PositiveInfinity; track.peerRejected = false;
        }

        public void AgentRejectCandidate(Vector2 p, double t, string evidence)
        {
            agentHoldUntil = double.NegativeInfinity;
            int n = 0;
            foreach (var track in tracks)
            {
                if (track.confirmed || Vector2.Distance(track.position, p) > 1.0f) continue;
                track.candidate = false; track.retryAfter = double.PositiveInfinity; MarkRejected(track, t); n++;
                if (active == track) { active = null; dwell = 0; inspecting = false; ClearPath(); }
            }
            NoteRejectedSpot(p, t);
            Emit(t, "candidate_rejected", "Agent rejected a warm region as not a person, OpenCV evidence " + evidence + ".",
                "x_m", p.x, "y_m", p.y, "tracks", n);
        }

        /// <summary>Keep out of a circle for some seconds by writing hot cells into the heat layer, so the existing
        /// fire margin and heat cost keep paths away; the heat decays like observed heat.</summary>
        public void AgentAvoidArea(Vector2 centre, float radiusM, double t, double seconds, string evidence)
        {
            // A keep-out area must never contain the robot itself: it was written as fire, so every plan started inside
            // fire and the robot froze (agent run 9, robot B, 1.24 m inside a 2 m circle). Shrink it to leave the robot
            // outside with its fire clearance, or refuse it.
            float room = Vector2.Distance(centre, motion.Pose.position) - c.fire_clearance_m - c.robot_radius_m;
            if (room < c.agent_avoid_min_radius_m)
            {
                Emit(t, "agent_order", "avoid_area refused: the robot is inside or at the edge of it; it backs off on heat by itself.",
                    "x_m", centre.x, "y_m", centre.y, "radius_m", radiusM);
                return;
            }
            // A cut-off robot has no route out to keep clear, so the check below passed every circle: one-door seed 925, robot A,
            // cut off at 122 s, had its agent draw 2.0 m of "fire" beside the front door at 134 s while it looked for a way
            // out. The reflex keeps clear of measured heat by itself; a drawn circle can only hide the way out.
            if (cutOff)
            {
                Emit(t, "agent_order", "avoid_area refused: the robot is cut off from its way out, and a keep-out area could only hide a way out; the reflex keeps clear of real heat by itself.",
                    "x_m", centre.x, "y_m", centre.y, "radius_m", radiusM);
                return;
            }
            // Nor may it cut the robot's own way out (2x2 building run: robot A's agent drew 2.5 m around its own frontier
            // target, the route home was gone, and the robot held still from 88 s). Stay clear of the planned route out.
            float routeRoom = float.PositiveInfinity;
            foreach (var p in PlanOut()) routeRoom = Mathf.Min(routeRoom, Vector2.Distance(centre, p) - c.fire_clearance_m - c.robot_radius_m);
            if (routeRoom < c.agent_avoid_min_radius_m)
            {
                Emit(t, "agent_order", "avoid_area refused: it would cut the robot's way out; the reflex keeps clear of real heat by itself.",
                    "x_m", centre.x, "y_m", centre.y, "radius_m", radiusM);
                return;
            }
            room = Mathf.Min(room, routeRoom);
            radiusM = Mathf.Clamp(Mathf.Min(radiusM, room), c.agent_avoid_min_radius_m, c.agent_avoid_max_radius_m);
            float temperature = c.fire_c + c.heat_decay_c_per_s * (float)Math.Max(1.0, seconds) + 50f;
            int cells = 0;
            for (float dx = -radiusM; dx <= radiusM; dx += grid.CellM)
                for (float dy = -radiusM; dy <= radiusM; dy += grid.CellM)
                {
                    if (dx * dx + dy * dy > radiusM * radiusM) continue;
                    int i = grid.Cell(centre + new Vector2(dx, dy));
                    if (i < 0) continue;
                    grid.ObservePeerHeat(i, temperature, t, t); grid.HoldHeat(i, t + Math.Max(1.0, seconds)); cells++;
                }
            lastPlan = double.NegativeInfinity;   // replan through the new margin at once
            Emit(t, "agent_order", "avoid_area around spreading fire, OpenCV evidence " + evidence + ".",
                "x_m", centre.x, "y_m", centre.y, "radius_m", radiusM, "seconds", seconds, "cells", cells);
        }

        /// <summary>Logs agent decisions that act outside the brain (torch, arm, radio, approvals) next to its own.</summary>
        public void LogAgentAction(double t, string reason, params object[] values) => Emit(t, "agent_order", reason, values);

        /// <summary>Records an agent-layer event (decision summary, radio text, approval) in this robot's decision log.</summary>
        public void Log(double t, string kind, string reason, params object[] values) => Emit(t, kind, reason, values);
    }
}
