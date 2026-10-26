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

        bool GoTo(Vector2 target, double t, double holdS, string reason, Vector2? face, bool exploring, bool ownReflex = false)
        {
            // Turning where the robot stands (turn_to_face) goes nowhere new, so it needs no indoor evidence (Codex review:
            // near an entrance the robot's own cell may not be known to be inside yet).
            bool turnInPlace = face.HasValue && Vector2.Distance(target, motion.Pose.position) <= c.goal_tolerance_m;
            if (!turnInPlace && !InSearchArea(target)) { Emit(t, "agent_order", "Refused go_to: no ceiling has been seen over that point, so it is not known to be inside.", "x_m", target.x, "y_m", target.y); return false; }
            // code_spray_ignores_leave_refusal: the refusals below protect an agent from ordering a robot that is leaving; a code spray
            // approach is the robot's own reflex and the robot is not returning.
            string busy = ownReflex && c.code_spray_ignores_leave_refusal && !returning ? null : AgentMoveRefusal(t);
            if (busy != null) { Emit(t, "agent_order", "Refused go_to: " + busy + ".", "x_m", target.x, "y_m", target.y); return false; }
            agentGoal = target; agentGoalUntil = t + holdS; agentGoalReason = reason ?? ""; agentGoalIsFrontier = false;
            agentFace = face; facing = null; agentHoldUntil = double.NegativeInfinity;
            if (active == null && !KeepEvidenceLook(exploring)) ClearPath();
            Emit(t, "agent_order", "go_to: " + agentGoalReason, "x_m", target.x, "y_m", target.y, "hold_s", holdS);
            return true;
        }

        /// <summary>code_spray_retry_s: spray spots the robot could not reach, with the time they may be tried again.</summary>
        readonly List<(Vector2 spot, double until)> unreachableSpraySpots = new List<(Vector2, double)>();
        Vector2? spraySpotGoal;   // the spot of the code spray approach now driving as the agent goal
        /// <summary>code_spray_max_approaches: approaches per spray spot since the last spray; a spot walked to that often without
        /// a spray goes on the unreachable list (three-bed ranch live, 26 Oct: robot B walked to the same spot outside the house
        /// 35 times without spraying and never entered).</summary>
        readonly List<(Vector2 spot, int count)> sprayApproaches = new List<(Vector2, int)>();
        void NoteSpraySpotFailed(Vector2 spot, double t)
        {
            if (c.code_spray_retry_s <= 0) return;
            unreachableSpraySpots.Add((spot, t + c.code_spray_retry_s));
            Emit(t, "code_spray_spot_unreachable", "No route to the spray spot: not tried again for code_spray_retry_s.", "x_m", spot.x, "y_m", spot.y, "retry_s", c.code_spray_retry_s);
        }
        bool SpraySpotUnreachable(Vector2 spot, double t)
        {
            unreachableSpraySpots.RemoveAll(u => t >= u.until);
            return unreachableSpraySpots.Exists(u => Vector2.Distance(u.spot, spot) <= 1f);
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
                    // The block is the burning cell itself, not the path point where its margin starts: measured from that
                    // point, sprays fired from up to 4.6 m and landed short (owner, 25 Oct).
                    var fire = NearestFire(p, margin + grid.CellM);
                    Vector2 d = fire - motion.Pose.position;
                    return new BrainFireBlock { blockAt = fire, frontier = f.position, bearingDegRight = -BrainMath.Angle(motion.Pose.yawRad, Mathf.Atan2(-d.x, d.y)),
                        distanceM = d.magnitude, frontierPathM = f.cost, unsearchedM2 = f.information * grid.CellM * grid.CellM,
                        burningM2 = grid.BurningAreaNear(fire, c.burning_radius_m) };
                }
            return null;
        }

        /// <summary>code_spray_goal_room (owner, 24 Oct: "use the water to turn off the fire" instead of walking into a burning
        /// room): the measured fire nearest this robot inside the room its route ends in, or null when the route ends in no
        /// room or this robot measured no fire there. Seed 934 (24 Oct 18:53): robot A walked into the burning living room at
        /// 16 s, its air reached 131 C at 21.7 s, and it spent most of its heat budget.</summary>
        /// <summary>walk_out_past_fire: the first measured fire within the fire margin along the way out, planned as if fire
        /// did not close cells; null when even that plan finds no way out (a wall, not fire, closes it).</summary>
        BrainFireBlock? FireOnWayOut()
        {
            float saved = grid.FireClearance; bool predicted = grid.FirePrediction; List<Vector2> exit;
            try { grid.FireClearance = -1; grid.FirePrediction = false; exit = PlanOut(); }
            finally { grid.FireClearance = saved; grid.FirePrediction = predicted; }
            float margin = c.fire_clearance_m + grid.CellM * .707107f; var at = motion.Pose.position;
            foreach (var p in exit)
            {
                if (grid.FireDistance(grid.Cell(p)) > margin) continue;
                var fire = NearestFire(p, margin + grid.CellM);
                Vector2 v = fire - at;
                return new BrainFireBlock { blockAt = fire, frontier = p, bearingDegRight = -BrainMath.Angle(motion.Pose.yawRad, Mathf.Atan2(-v.x, v.y)),
                    distanceM = v.magnitude, frontierPathM = v.magnitude, unsearchedM2 = 0, burningM2 = grid.BurningAreaNear(fire, c.burning_radius_m) };
            }
            return null;
        }

        /// <summary>The measured burning cell nearest p within radius (p itself when only forecast heat closes it).</summary>
        Vector2 NearestFire(Vector2 p, float radius)
        {
            int r = Mathf.CeilToInt(radius / grid.CellM), centre = grid.Cell(p); if (centre < 0) return p;
            int cx = centre % grid.Size, cy = centre / grid.Size; float best = float.PositiveInfinity; Vector2 found = p;
            for (int dy = -r; dy <= r; dy++) for (int dx = -r; dx <= r; dx++)
            {
                int x = cx + dx, y = cy + dy; if (x < 0 || y < 0 || x >= grid.Size || y >= grid.Size) continue;
                int i = y * grid.Size + x; if (!grid.MeasuredFire(i)) continue;
                float d = Vector2.Distance(grid.Centre(i), p); if (d < best && d <= radius) { best = d; found = grid.Centre(i); }
            }
            return found;
        }

        /// <summary>code_spray_route_fire: the measured fire nearest this robot within the fire margin of the route being walked.</summary>
        BrainFireBlock? FireOnRoute()
        {
            if (path.Count == 0) return null;
            float margin = c.fire_clearance_m + grid.CellM * .707107f; var at = motion.Pose.position;
            for (int i = Math.Max(0, cursor); i < path.Count; i++)
            {
                int cell = grid.Cell(path[i]); if (cell < 0 || grid.FireDistance(cell) > margin) continue;
                var fire = NearestFire(path[i], margin + grid.CellM);
                int fc = grid.Cell(fire); if (fc < 0 || !grid.MeasuredFire(fc)) continue;
                Vector2 v = fire - at;
                return new BrainFireBlock { blockAt = fire, frontier = path[path.Count - 1], bearingDegRight = -BrainMath.Angle(motion.Pose.yawRad, Mathf.Atan2(-v.x, v.y)),
                    distanceM = v.magnitude, frontierPathM = Vector2.Distance(at, path[path.Count - 1]), unsearchedM2 = 0, burningM2 = grid.BurningAreaNear(fire, c.burning_radius_m) };
            }
            return null;
        }
        /// <summary>code_spray_route_fire: the measured fire nearest this robot on floor no camera has searched yet, in the search area.</summary>
        BrainFireBlock? FireOnUnsearchedFloor()
        {
            Vector2 at = motion.Pose.position; float best = float.PositiveInfinity; Vector2 fire = default;
            for (int i = 0; i < grid.Count; i++)
            {
                if (!grid.MeasuredFire(i) || !grid.Unsearched(i)) continue;
                var p = grid.Centre(i); float d = Vector2.Distance(p, at);
                if (d < best && InSearchArea(p)) { best = d; fire = p; }
            }
            if (float.IsPositiveInfinity(best)) return null;
            Vector2 v = fire - at;
            return new BrainFireBlock { blockAt = fire, frontier = fire, bearingDegRight = -BrainMath.Angle(motion.Pose.yawRad, Mathf.Atan2(-v.x, v.y)),
                distanceM = v.magnitude, frontierPathM = v.magnitude, unsearchedM2 = 0, burningM2 = grid.BurningAreaNear(fire, c.burning_radius_m) };
        }

        BrainFireBlock? FireInGoalRoom()
        {
            if (!c.room_ledger || path.Count == 0) return null;
            var ledger = planner.Rooms; var end = path[path.Count - 1]; int room = ledger.RoomNear(end);
            if (room < 0) return null;
            Vector2 at = motion.Pose.position; float best = float.PositiveInfinity; Vector2 fire = default;
            for (int i = 0; i < grid.Count; i++)
            {
                if (!grid.MeasuredFire(i)) continue;
                var p = grid.Centre(i); float d = Vector2.Distance(p, at);
                if (d < best && ledger.RoomNear(p) == room) { best = d; fire = p; }
            }
            if (float.IsPositiveInfinity(best)) return null;
            Vector2 v = fire - at;
            return new BrainFireBlock { blockAt = fire, frontier = end, bearingDegRight = -BrainMath.Angle(motion.Pose.yawRad, Mathf.Atan2(-v.x, v.y)),
                distanceM = v.magnitude, frontierPathM = Vector2.Distance(at, end), unsearchedM2 = 0, burningM2 = grid.BurningAreaNear(fire, c.burning_radius_m) };
        }

        readonly List<(Vector2 centre, float radius)> fireRooms = new List<(Vector2 centre, float radius)>();

        /// <summary>heat_marks_room (owner, 24 Oct: "marks that room and area in map as with fire"): where this robot's air passed
        /// heat_retreat_c, the room it stands in is marked on fire on its map: a circle over the room (its half-diagonal around
        /// its centre, at least out to where the robot stood). No goal in it (same side of the walls) is taken while this robot
        /// measures fire there; the mark lifts once that fire is out, sprayed or burned out. Seed 934 (24 Oct 18:53): robot A
        /// walked into the burning living room at 16 s and its air reached 131 C.</summary>
        void MarkFireRoom(double t)
        {
            if (!c.room_ledger) return;
            var ledger = planner.Rooms; var at = motion.Pose.position; int id = ledger.RoomNear(at);
            RoomLedger.Room room = null;
            if (id >= 0) foreach (var r in ledger.Rooms) if (r.id == id) { room = r; break; }
            Vector2 centre = room?.centre ?? at;
            float radius = Mathf.Max(room != null ? .5f * Mathf.Sqrt(room.widthM * room.widthM + room.lengthM * room.lengthM) : c.burning_radius_m,
                Vector2.Distance(centre, at));
            fireRooms.Add((centre, radius));
            Emit(t, "room_marked_fire", c.heat_spray_through ? "The air here passed the heat limit: this room is marked on fire on the map; spray its fire and search on."
                : "The air here passed the retreat limit: this room is marked on fire on the map; no entry until the fire there is out.",
                "x_m", centre.x, "y_m", centre.y, "radius_m", radius, "air_c", airC);
        }
        /// <summary>Lifts each fire mark once this robot measures no burning floor within it or burning_radius_m of its edge.</summary>
        void UpdateFireRooms(double t)
        {
            for (int k = fireRooms.Count - 1; k >= 0; k--)
            {
                var m = fireRooms[k];
                if (grid.BurningAreaNear(m.centre, m.radius + c.burning_radius_m) > 0) continue;
                fireRooms.RemoveAt(k);
                Emit(t, "room_fire_cleared", "No fire measured in the room marked on fire any more: it is open again.", "x_m", m.centre.x, "y_m", m.centre.y);
            }
        }
        bool InFireRoom(Vector2 p) => !c.heat_spray_through && fireRooms.Exists(m => Vector2.Distance(p, m.centre) <= m.radius && SameSideOfWalls(p, m.centre));
        bool hotAir;
        /// <summary>heat_spray_through (owner, 24 Oct: "remove the heat budget ... let it just spray the fire, assume that the fire
        /// is gone off, and just walk through, figure out if there is a person; not there: retreat"): hot air never holds a robot.
        /// Where the air at it first passes heat_retreat_c, the room is marked on fire and its fire sprayed at once if one stream
        /// wins; the robot then searches on and leaves the room when it has nothing left, as for any room. Fire cells keep their
        /// margin, so it never walks onto burning floor. Seed 934 (24 Oct 19:54): the heat budget held robot A still 46 % of the
        /// run and kept it out of the living room, where a person was missed.</summary>
        void HotAirSprayThrough(double t)
        {
            bool hot = airC > c.heat_retreat_c && t - lastAir <= c.frame_max_age_s;
            if (hot && !hotAir)
            {
                Emit(t, "hot_air", "The air here passed the heat limit: mark the room on fire, spray its fire and search on.", "air_c", airC);
                if (c.heat_marks_room) MarkFireRoom(t);
                nextCodeSpray = Math.Min(nextCodeSpray, t);
            }
            hotAir = hot;
        }
        /// <summary>The measured fire nearest this robot inside a room marked on fire (or burning_radius_m beyond its edge).</summary>
        BrainFireBlock? FireInMarkedRoom()
        {
            Vector2 at = motion.Pose.position; float best = float.PositiveInfinity; Vector2 fire = default;
            for (int i = 0; i < grid.Count; i++)
            {
                if (!grid.MeasuredFire(i)) continue;
                var p = grid.Centre(i); float d = Vector2.Distance(p, at);
                if (d < best && fireRooms.Exists(m => Vector2.Distance(p, m.centre) <= m.radius + c.burning_radius_m)) { best = d; fire = p; }
            }
            if (float.IsPositiveInfinity(best)) return null;
            Vector2 v = fire - at;
            return new BrainFireBlock { blockAt = fire, frontier = fire, bearingDegRight = -BrainMath.Angle(motion.Pose.yawRad, Mathf.Atan2(-v.x, v.y)),
                distanceM = v.magnitude, frontierPathM = v.magnitude, unsearchedM2 = 0, burningM2 = grid.BurningAreaNear(fire, c.burning_radius_m) };
        }

        /// <summary>The nozzle and burning-floor numbers code_sprays_fire judges a fire by: config/actuators.json nozzle and
        /// config/perception.json, read by the host (the same values the agent's spray advice uses, never the simulator's).</summary>
        public struct NozzleSpec
        {
            public float flowLps, coolingKjPerL, efficiency, halfAngleDeg, rangeM, maxSprayS, burningKwPerM2, burningSurfaceC, knockdownC, fuelKjPerM2K;
        }
        /// <summary>Set by the host when the robot carries a nozzle; null leaves code_sprays_fire idle.</summary>
        public NozzleSpec? Nozzle { get; set; }
        double nextCodeSpray = double.NegativeInfinity;

        /// <summary>Seconds of one stream that knock down areaM2 of burning floor distanceM away, or null when the stream
        /// cannot outpace the fire: the stream spreads over the nozzle's sector, only the burning share of its footprint lands
        /// on fire, that share first takes up the fire's heat release, and the rest cools the burning surface from
        /// burningSurfaceC to knockdownC. The same model as the agent's spray advice (brain/brain/agent/tools.py knock_down_s).</summary>
        public static double? KnockDownS(NozzleSpec n, float areaM2, float distanceM)
        {
            if (areaM2 <= 0) return 0;
            float d = Mathf.Max(distanceM, 1f);
            float footprint = Mathf.Max(areaM2, n.halfAngleDeg * Mathf.Deg2Rad * d * d);
            float net = n.flowLps * n.coolingKjPerL * n.efficiency * Mathf.Min(1f, areaM2 / footprint) - areaM2 * n.burningKwPerM2;
            if (net <= 0) return null;
            return areaM2 * n.fuelKjPerM2K * (n.burningSurfaceC - n.knockdownC) / net;
        }

        /// <summary>code_sprays_fire (owner, 24 Oct: "the nearest fire block in the way, put it out, walk through and scan the
        /// room"): fire this robot measured burning (heat at or above fire_c, never a forecast) on the route to unsearched floor,
        /// that one stream knocks down with the water aboard, is sprayed by the robot itself; from beyond the nozzle's throw,
        /// or where the stream is too spread to win, it first walks to the spray spot fire_clearance_m + 0.4 m short of the
        /// fire, facing it. Seed 934 (24 Oct 04:53): 64 spray offers to the agents, two 2 s sprays, the living room fire never
        /// put out. True with the bearing and seconds the host passes to the nozzle.</summary>
        public bool CodeSpray(double t, float waterL, out float bearingDegRight, out float seconds)
        {
            bearingDegRight = 0; seconds = 0; lastWaterL = waterL;
            // Walking out with the way blocked (wayOutBlockedAt), the spray opens it: the only returning case allowed.
            bool exitBlocked = c.walk_out_past_fire && t - wayOutBlockedAt <= c.no_frontier_scan_s * 2;
            if (!c.code_sprays_fire || !Nozzle.HasValue || waterL <= 1e-6f || t < nextCodeSpray || (returning && !exitBlocked) || active != null) return false;
            // A person the colour detector just saw comes first: confirm her, then spray (seed 934, 24 Oct 19:54: robot A's
            // camera saw the living room person at 11.0 s and it left to spray at 11.0 s; she was never confirmed).
            if (c.person_before_spray && t < glanceUntil) return false;
            // One look per replan_s, as often as the planner refreshes frontiers: the host asks after every 60 Hz brain step, and
            // a full frontier search each time took 190 s of every 10 simulated seconds (seed 934, 24 Oct 18:13, 316 s wall per 10 s).
            nextCodeSpray = t + c.replan_s;
            // Fire in the room this robot is walking into: put it out from the doorway before going in (code_spray_goal_room).
            var block = exitBlocked ? FireOnWayOut() : null;
            string why = "Measured fire blocks the way out";
            bool wayOut = block.HasValue;
            if (!block.HasValue && c.code_spray_route_fire && !retreating)
            {
                // code_spray_route_fire: only fire that cuts the way home, the route being walked, or burns on unsearched floor.
                block = FireOnWayOut(); why = "Measured fire cuts the way out";
                if (block.HasValue && block.Value.distanceM > c.fire_near_m) block = null;   // within the nozzle's throw: no walk to a far exit fire mid-search
                wayOut = block.HasValue;
                if (!block.HasValue) { block = FireOnRoute(); why = "Measured fire burns beside the route this robot is walking"; }
                if (!block.HasValue) { block = FireOnUnsearchedFloor(); why = "Measured fire burns on unsearched floor"; }
            }
            if (!block.HasValue)
            {
                block = c.code_spray_goal_room && !c.code_spray_route_fire && !retreating ? FireInGoalRoom() : null;
                why = block.HasValue ? "Measured fire burns in the room this robot is walking into" : "Measured fire blocks the way to unsearched floor";
            }
            bool fromHereOnly = false;
            if (!block.HasValue && fireRooms.Count > 0 && (block = FireInMarkedRoom()).HasValue)
            {
                // The room marked on fire: spray it only from where the robot stands, never walk into its hot air.
                why = "Measured fire burns in the room marked on fire"; fromHereOnly = !c.heat_spray_through;
            }
            if (!block.HasValue)
            {
                // Otherwise only fire that blocks the way: while unsearched floor is reachable without passing fire, search it
                // first. Seed 934 (24 Oct 18:30): both robots left the room they had claimed to spray living room fire, 1 room
                // entered by 50 s.
                if (planner.Frontiers(motion.Pose.position, p => ClaimBlocks(p, t) || VisitedGoal(p, t) || GoalBlacklisted(p, t) || !InSearchArea(p)).Count > 0)
                    return false;
                block = FireBlockedFrontier(t);
            }
            if (!block.HasValue || !(block.Value.burningM2 > 0)) return false;
            // code_spray_reserve_l: only a spray on the way out may use the water kept for it.
            if (!wayOut) waterL = Mathf.Max(0, waterL - c.code_spray_reserve_l);
            if (waterL <= 1e-6f) return false;
            var n = Nozzle.Value; var b = block.Value; float waterS = waterL / n.flowLps;
            // Only from close: the owner saw a spray from 3.4 m land short of the fire (25 Oct); farther away, walk to the spray spot.
            double? here = b.distanceM <= Mathf.Min(n.rangeM, c.code_spray_reach_m) ? KnockDownS(n, b.burningM2, b.distanceM) : null;
            if (here.HasValue && here.Value <= waterS)
            {
                bearingDegRight = b.bearingDegRight;
                seconds = Mathf.Min(waterS, n.maxSprayS, (float)here.Value + c.code_spray_margin_s);
                nextCodeSpray = t + seconds + c.code_spray_repeat_s;
                sprayApproaches.Clear();
                Emit(t, "code_spray", why + " and one stream knocks it down: spray it.",
                    "x_m", b.blockAt.x, "y_m", b.blockAt.y, "burning_m2", b.burningM2, "distance_m", b.distanceM, "bearing_deg", b.bearingDegRight,
                    "seconds", seconds, "knock_down_s", here.Value, "water_l", waterL);
                return true;
            }
            if (fromHereOnly) return false;
            // Too far or too spread from here: walk to the spray spot on this robot's side, facing the fire, if one stream wins there.
            float standoff = c.fire_clearance_m + .4f;
            double? there = KnockDownS(n, b.burningM2, standoff);
            if (there.HasValue && there.Value <= waterS && !AgentGoalActive(t) && b.distanceM > standoff + c.goal_tolerance_m)
            {
                Vector2 from = motion.Pose.position, dir = (b.blockAt - from).normalized, spot = b.blockAt - dir * standoff;
                nextCodeSpray = t + c.replan_s;
                if (SpraySpotUnreachable(spot, t)) return false;
                // A spray spot outside the search area only for fire on the way out; never leave the house to chase a fire.
                if (c.code_spray_spot_inside && !exitBlocked && !InSearchArea(spot)) return false;
                int k = sprayApproaches.FindIndex(a => Vector2.Distance(a.spot, spot) <= 1f);
                int n2 = k < 0 ? 1 : sprayApproaches[k].count + 1;
                if (k < 0) sprayApproaches.Add((spot, 1)); else sprayApproaches[k] = (sprayApproaches[k].spot, n2);
                if (c.code_spray_max_approaches > 0 && n2 > c.code_spray_max_approaches)
                {
                    NoteSpraySpotFailed(spot, t); if (k >= 0) sprayApproaches.RemoveAt(k); return false;
                }
                if (!GoTo(spot, t, c.code_spray_repeat_s * 2, "walk to the spray spot of the fire blocking the way", b.blockAt, false, true))
                {
                    NoteSpraySpotFailed(spot, t); return false;
                }
                spraySpotGoal = spot;
                Emit(t, "code_spray_approach", why + "; one stream knocks it down from the spray spot: walk there facing it.",
                    "x_m", b.blockAt.x, "y_m", b.blockAt.y, "burning_m2", b.burningM2, "distance_m", b.distanceM, "standoff_m", standoff);
            }
            return false;
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
            LogSurvivorRoom(t, p);
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
        /// <summary>Never earlier than this brain's own clock: seed 934 (25 Oct, brain only) logged four "spray_water succeeded" lines at
        /// t = 0.0 because the host passed a stale agent time, and run_audit then blamed every early stall on them.</summary>
        public void LogAgentAction(double t, string reason, params object[] values) =>
            Emit(double.IsNaN(lastStep) ? t : Math.Max(t, lastStep), "agent_order", reason, values);

        /// <summary>Records an agent-layer event (decision summary, radio text, approval) in this robot's decision log.</summary>
        public void Log(double t, string kind, string reason, params object[] values) => Emit(t, kind, reason, values);
    }
}
