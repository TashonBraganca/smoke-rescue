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
        Rect? sector;

        bool AgentGoalActive(double t) => agentGoal.HasValue && t <= agentGoalUntil && !returning && !retreating;

        public void AgentGoTo(Vector2 target, double t, double holdS, string reason)
        {
            if (!InSearchArea(target)) { Emit(t, "agent_order", "Refused go_to outside the search area.", "x_m", target.x, "y_m", target.y); return; }
            agentGoal = target; agentGoalUntil = t + holdS; agentGoalReason = reason ?? ""; agentGoalIsFrontier = false;
            if (active == null) ClearPath();
            Emit(t, "agent_order", "go_to: " + agentGoalReason, "x_m", target.x, "y_m", target.y, "hold_s", holdS);
        }

        bool agentGoalIsFrontier;

        /// <summary>The frontiers the agent chooses from: reachable, worth searching, not claimed by a partner, inside
        /// the search area. Positions are the walkable cell the robot would drive to; path_m is the planned distance
        /// including heat and wall costs; unsearched_m2 is the floor no camera on the team has seen near it.</summary>
        public IReadOnlyList<BrainFrontier> FrontierOptions(double t, int max)
        {
            System.Func<Vector2, bool> excluded = p => ClaimBlocks(p, t) || GoalBlacklisted(p, t) || VisitedGoal(p, t) || !InSearchArea(p);
            var list = new List<BrainFrontier>();
            foreach (var f in planner.Frontiers(motion.Pose.position, excluded))
            {
                if (list.Count >= max) break;
                list.Add(new BrainFrontier { id = list.Count, position = f.position, pathM = f.cost,
                    unsearchedM2 = f.information * grid.CellM * grid.CellM, inSector = !sector.HasValue || sector.Value.Contains(f.position) });
            }
            return list;
        }

        /// <summary>Agent order: search this frontier next. It is driven like a go_to, and on arrival the robot sweeps
        /// and marks it visited exactly as for a frontier it chose itself.</summary>
        public void AgentExplore(Vector2 target, double t, double holdS, string reason)
        {
            AgentGoTo(target, t, holdS, "explore: " + (reason ?? ""));
            agentGoalIsFrontier = agentGoal.HasValue;
        }

        public void AgentConfirmSurvivor(Vector2 p, double t, string evidence)
        {
            foreach (var known in survivors) if (Vector2.Distance(known, p) <= c.survivor_merge_m) return;
            survivors.Add(p);
            confirmed.Add(new BrainSurvivor { position = p, confirmedAt = t, sourceRobot = id });
            foreach (var track in tracks) if (Vector2.Distance(track.position, p) <= c.survivor_merge_m) track.confirmed = true;
            if (active != null && Vector2.Distance(active.position, p) <= c.survivor_merge_m) { active = null; dwell = 0; inspecting = false; ClearPath(); }
            Emit(t, "survivor_confirmed", "Agent confirmed a person with OpenCV evidence " + evidence + ".", "x_m", p.x, "y_m", p.y);
        }

        public void AgentRejectCandidate(Vector2 p, double t, string evidence)
        {
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
            radiusM = Mathf.Clamp(radiusM, 0.3f, 4f);
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

        public void AssignSector(Rect area, double t, string from, string reason)
        {
            sector = area;
            if (active == null && !AgentGoalActive(t)) ClearPath();
            Emit(t, "sector_assigned", (from ?? "commander") + ": " + (reason ?? ""), "x0", area.xMin, "y0", area.yMin, "x1", area.xMax, "y1", area.yMax);
        }

        /// <summary>Logs agent decisions that act outside the brain (torch, arm, radio, approvals) next to its own.</summary>
        public void LogAgentAction(double t, string reason, params object[] values) => Emit(t, "agent_order", reason, values);

        /// <summary>Records an agent-layer event (decision summary, radio text, approval) in this robot's decision log.</summary>
        public void Log(double t, string kind, string reason, params object[] values) => Emit(t, kind, reason, values);
    }
}
