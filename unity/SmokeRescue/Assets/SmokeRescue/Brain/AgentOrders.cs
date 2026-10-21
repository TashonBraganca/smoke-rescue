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
        public void AgentGoTo(Vector2 target, double t, double holdS, string reason, Vector2? face = null)
        {
            // Turning where the robot stands (turn_to_face) goes nowhere new, so it needs no indoor evidence (Codex review:
            // near an entrance the robot's own cell may not be known to be inside yet).
            bool turnInPlace = face.HasValue && Vector2.Distance(target, motion.Pose.position) <= c.goal_tolerance_m;
            if (!turnInPlace && !InSearchArea(target)) { Emit(t, "agent_order", "Refused go_to: no ceiling has been seen over that point, so it is not known to be inside.", "x_m", target.x, "y_m", target.y); return; }
            string busy = AgentMoveRefusal(t);
            if (busy != null) { Emit(t, "agent_order", "Refused go_to: " + busy + ".", "x_m", target.x, "y_m", target.y); return; }
            agentGoal = target; agentGoalUntil = t + holdS; agentGoalReason = reason ?? ""; agentGoalIsFrontier = false;
            agentFace = face; facing = null; agentHoldUntil = double.NegativeInfinity;
            if (active == null) ClearPath();
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
            if (Mathf.Abs(error) > c.survivor_face_tolerance_deg && t < facingUntil)
                return Command(0, Mathf.Clamp(error * c.heading_gain, -c.sweep_yaw_dps, c.sweep_yaw_dps), dt, true);
            Emit(t, "agent_goal_reached", "Arrived where the agent asked and turned to face the point it named: " + facingReason,
                "x_m", facing.Value.x, "y_m", facing.Value.y, "bearing_error_deg", error);
            facing = null; agentHoldUntil = t + c.agent_arrival_hold_s;
            return null;
        }

        /// <summary>The frontiers the agent chooses from: reachable, worth searching, not claimed by a partner, inside
        /// the search area. Positions are the walkable cell the robot would drive to; path_m is the planned distance
        /// including heat and wall costs; unsearched_m2 is the floor no camera on the team has seen near it.</summary>
        public IReadOnlyList<BrainFrontier> FrontierOptions(double t, int max)
        {
            // Proven frontiers first; then floor the robot cannot yet prove is indoors, which only the agent may choose
            // (with an OpenCV indoor check of its own frame). Unentered rooms have unmapped walls and were never proven,
            // so they were never searched (hidden-corner run: robot A walked out with 30 % of the floor unsearched).
            var proven = FrontierOptions(t, max, true);
            if (proven.Count >= max) return proven;
            var list = new List<BrainFrontier>(proven);
            foreach (var f in FrontierOptions(t, max, false))
            {
                if (list.Count >= max) break;
                if (f.provenIndoors) continue;
                var copy = f; copy.id = list.Count; list.Add(copy);
            }
            return list;
        }

        IReadOnlyList<BrainFrontier> FrontierOptions(double t, int max, bool provenOnly)
        {
            // Frontier cells dropped by each filter, so a short list can be explained from the log (agent run 9: robot A
            // was offered one 0.5 m2 frontier at 20 s with most of its LiDAR-mapped floor unsearched).
            int claimed = 0, blacklisted = 0, visited = 0, outside = 0;
            bool Excluded(Vector2 p)
            {
                if (ClaimBlocks(p, t)) { claimed++; return true; }
                if (GoalBlacklisted(p, t)) { blacklisted++; return true; }
                if (VisitedGoal(p, t)) { visited++; return true; }
                if (provenOnly && !InSearchArea(p)) { outside++; return true; }
                return false;
            }
            var list = new List<BrainFrontier>(); var all = planner.Frontiers(motion.Pose.position, Excluded);
            foreach (var f in all)
            {
                if (list.Count >= max) break;
                list.Add(new BrainFrontier { id = list.Count, position = f.position, pathM = f.cost,
                    unsearchedM2 = f.information * grid.CellM * grid.CellM, provenIndoors = InSearchArea(f.position) });
            }
            if (!provenOnly) return list;
            Emit(t, "frontier_options", $"{all.Count} frontiers; boundary cells dropped as claimed by a partner, blacklisted, visited, or not known to be indoors.",
                "frontiers", all.Count, "cells_claimed", claimed, "cells_blacklisted", blacklisted, "cells_visited", visited, "cells_not_indoors", outside);
            return list;
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
        public BrainFireBlock? FireBlockedFrontier(double t)
        {
            float saved = grid.FireClearance;
            List<GridPlanner.FrontierChoice> all;
            try
            {
                grid.FireClearance = -1;   // fire distance is never below zero: every fire cell counts as clear
                all = planner.Frontiers(motion.Pose.position, p => ClaimBlocks(p, t) || GoalBlacklisted(p, t) || VisitedGoal(p, t) || !InSearchArea(p));
            }
            finally { grid.FireClearance = saved; }
            float margin = c.fire_clearance_m + grid.CellM * .707107f;
            foreach (var f in all)
                foreach (var p in f.path)
                {
                    if (grid.FireDistance(grid.Cell(p)) > margin) continue;
                    Vector2 d = p - motion.Pose.position;
                    return new BrainFireBlock { blockAt = p, frontier = f.position, bearingDegRight = -BrainMath.Angle(motion.Pose.yawRad, Mathf.Atan2(-d.x, d.y)),
                        distanceM = d.magnitude, frontierPathM = f.cost, unsearchedM2 = f.information * grid.CellM * grid.CellM };
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
                if (active == null) ClearPath();
                Emit(t, "agent_order", "explore_frontier on floor not yet proven indoors, citing OpenCV evidence " + indoorEvidence + ": " + (reason ?? ""),
                    "x_m", target.x, "y_m", target.y, "hold_s", holdS);
                return;
            }
            AgentGoTo(target, t, holdS, "explore: " + (reason ?? ""));
            agentGoalIsFrontier = agentGoal.HasValue;
        }

        public void AgentConfirmSurvivor(Vector2 p, double t, string evidence)
        {
            if (AtOwnSetDown(p))
            {
                Emit(t, "agent_order", "Refused confirm_survivor: this robot set down a person it carried out there; that is the person already rescued.",
                    "x_m", p.x, "y_m", p.y);
                return;
            }
            foreach (var known in survivors) if (Vector2.Distance(known, p) <= c.survivor_merge_m) return;
            survivors.Add(p);
            confirmed.Add(new BrainSurvivor { position = p, confirmedAt = t, sourceRobot = id });
            foreach (var track in tracks) if (Vector2.Distance(track.position, p) <= c.survivor_merge_m) track.confirmed = true;
            if (active != null && Vector2.Distance(active.position, p) <= c.survivor_merge_m) { active = null; dwell = 0; inspecting = false; ClearPath(); }
            Emit(t, "survivor_confirmed", "Agent confirmed a person with OpenCV evidence " + evidence + ".", "x_m", p.x, "y_m", p.y);
        }

        public void AgentRejectCandidate(Vector2 p, double t, string evidence)
        {
            agentHoldUntil = double.NegativeInfinity;
            int n = 0;
            foreach (var track in tracks)
            {
                if (track.confirmed || Vector2.Distance(track.position, p) > 1.0f) continue;
                track.candidate = false; track.retryAfter = double.PositiveInfinity; n++;
                if (active == track) { active = null; dwell = 0; inspecting = false; ClearPath(); }
            }
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
                    grid.ObservePeerHeat(i, temperature, t, t); cells++;
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
