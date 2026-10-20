using System;
using System.Collections.Generic;
using UnityEngine;

namespace SmokeRescue.Brain
{
    /// <summary>
    /// Search behaviour on top of frontier exploration, from the robot's own map only, so it works for any floor plan
    /// and any start pose:
    ///  - Sweep: on reaching a frontier in an area not swept recently, turn left, right, then back to the arrival
    ///    heading, holding briefly at each, so the cameras see the whole area and thermal can raise candidates.
    ///  - Visited: a reached frontier is not chosen again for a while. Spots the cameras cannot resolve (under beds,
    ///    behind furniture) otherwise pull the robot back to the same place, which looked like spinning in one room.
    ///  - Progress watchdog: if the map stops growing while exploring, frontiers near the robot are set aside and it
    ///    moves on.
    ///  - Area report: after a sweep, the share of floor around the robot that it has seen, as evidence of coverage.
    /// Survivor approach and heat retreat run before this in Step, so they always win over a sweep.
    /// </summary>
    public sealed partial class RobotBrain
    {
        struct Visited { public Vector2 position; public float radius; public double until; }
        readonly List<Visited> visited = new List<Visited>();
        readonly List<(Vector2 position, double t)> sweeps = new List<(Vector2, double)>();
        bool sweeping;
        float sweepBase;
        int sweepIndex;
        double sweepHoldUntil = double.NaN;
        static readonly float[] SweepOffsetsDeg = { 60f, -60f, 0f };   // left, right, back to the arrival heading
        int knownCells, knownAtMark;
        double progressMark = double.NaN, nextCount;

        bool VisitedGoal(Vector2 p, double t)
        {
            visited.RemoveAll(v => t >= v.until);
            return visited.Exists(v => Vector2.Distance(p, v.position) <= v.radius);
        }

        void OnFrontierReached(Vector2 at, double t)
        {
            visited.Add(new Visited { position = at, radius = c.visited_radius_m, until = t + c.visited_hold_s });
            sweeps.RemoveAll(s => t - s.t > c.sweep_repeat_s);
            if (sweeps.Exists(s => Vector2.Distance(s.position, motion.Pose.position) < c.sweep_spacing_m)) return;
            sweeps.Add((motion.Pose.position, t));
            sweeping = true; sweepIndex = 0; sweepBase = motion.Pose.yawRad; sweepHoldUntil = double.NaN; sweepStarted = t;
            Emit(t, "sweep_started", "New area: look left, right and ahead before choosing where to go next.",
                "x_m", motion.Pose.position.x, "y_m", motion.Pose.position.y);
        }

        double sweepStarted;

        void SkipSweepLook(double t, string why)
        {
            sweepHoldUntil = double.NaN;
            if (++sweepIndex >= SweepOffsetsDeg.Length) EndSweep(t, why);
        }

        void EndSweep(double t, string note)
        {
            sweeping = false;
            float coverage = AreaCoverage(motion.Pose.position, c.area_report_radius_m);
            Emit(t, "area_searched", $"Swept this area ({note}); {coverage * 100f:0}% of the floor within {c.area_report_radius_m:0.#} m has been seen.",
                "x_m", motion.Pose.position.x, "y_m", motion.Pose.position.y, "coverage", coverage);
            lastPlan = double.NegativeInfinity;
        }

        BrainCommand SweepStep(double t, float dt)
        {
            if (t - sweepStarted > c.sweep_max_s) { EndSweep(t, "time limit"); return Command(0, 0, dt, true); }
            float target = sweepBase + SweepOffsetsDeg[sweepIndex] * Mathf.Deg2Rad;
            float error = BrainMath.Angle(motion.Pose.yawRad, target);
            if (Mathf.Abs(error) > c.sweep_tolerance_deg)
                return Command(0, Mathf.Clamp(error * c.heading_gain, -c.sweep_yaw_dps, c.sweep_yaw_dps), dt, true);
            if (double.IsNaN(sweepHoldUntil)) sweepHoldUntil = t + c.sweep_hold_s;
            if (t < sweepHoldUntil) return Command(0, 0, dt, true);
            sweepHoldUntil = double.NaN;
            if (++sweepIndex < SweepOffsetsDeg.Length) return Command(0, 0, dt, true);
            EndSweep(t, "all looks done");
            return Command(0, 0, dt, true);
        }

        /// <summary>Share of cells within the radius, reachable from the robot without crossing a mapped wall, that
        /// the robot has observed. Cells behind walls are not counted, so a small room can reach 100 percent.</summary>
        float AreaCoverage(Vector2 centre, float radius)
        {
            int start = grid.Cell(centre);
            if (start < 0) return 0f;
            int seen = 0, total = 0;
            var queue = new Queue<int>(); var mark = new HashSet<int> { start }; queue.Enqueue(start);
            while (queue.Count > 0)
            {
                int i = queue.Dequeue(); total++;
                if (grid.OwnObserved(i)) seen++;
                int x = i % grid.Size, y = i / grid.Size;
                for (int k = 0; k < 4; k++)
                {
                    int nx = x + (k == 0 ? 1 : k == 1 ? -1 : 0), ny = y + (k == 2 ? 1 : k == 3 ? -1 : 0);
                    if (!grid.Inside(nx, ny)) continue;
                    int n = ny * grid.Size + nx;
                    if (mark.Contains(n) || grid.Occupancy(n) == 255 || Vector2.Distance(grid.Centre(n), centre) > radius || !InSearchArea(grid.Centre(n))) continue;
                    mark.Add(n); queue.Enqueue(n);
                }
            }
            return total > 0 ? seen / (float)total : 0f;
        }

        /// <summary>Once a second while exploring: if fewer than progress_min_cells new cells were mapped in the
        /// window, set aside frontiers near the robot and move on.</summary>
        void CheckProgress(double t)
        {
            if (t < nextCount) return;
            nextCount = t + 1.0;
            int n = 0;
            for (int i = 0; i < grid.Count; i++) if (grid.OwnObserved(i)) n++;
            knownCells = n;
            bool exploring = active == null && !retreating && !returning && !sweeping && !AgentGoalActive(t);
            if (!exploring || double.IsNaN(progressMark)) { progressMark = t; knownAtMark = n; return; }
            if (n - knownAtMark >= c.progress_min_cells) { progressMark = t; knownAtMark = n; return; }
            if (t - progressMark < c.progress_window_s) return;
            visited.Add(new Visited { position = motion.Pose.position, radius = c.progress_skip_radius_m, until = t + c.visited_hold_s });
            Emit(t, "no_progress", $"Map grew by only {n - knownAtMark} cells in {t - progressMark:0} s; leave this spot for the next unexplored area.",
                "x_m", motion.Pose.position.x, "y_m", motion.Pose.position.y, "new_cells", n - knownAtMark);
            progressMark = t; knownAtMark = n;
            ClearPath();
        }
    }
}
