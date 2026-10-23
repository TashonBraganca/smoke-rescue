using System;
using System.Collections.Generic;
using UnityEngine;

namespace SmokeRescue.Brain
{
    /// <summary>
    /// Search behaviour on top of frontier exploration, from the robot's own map only, so it works for any floor plan
    /// and any start pose:
    ///  - Sweep: on reaching a frontier in an area not swept recently, check the left, right and arrival-heading
    ///    camera sectors. Turn and hold only where unsearched floor remains visible within search range.
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
        int sweepIndex, sweepCheckedIndex = -1;
        double sweepHoldUntil = double.NaN;
        static readonly float[] SweepOffsetsDeg = { 60f, -60f, 0f };   // left, right, back to the arrival heading
        int knownCells, knownAtMark;
        double progressMark = double.NaN, nextCount;
        Vector2 progressPose; float travelSinceMark;

        bool VisitedGoal(Vector2 p, double t)
        {
            visited.RemoveAll(v => t >= v.until);
            return visited.Exists(v => Vector2.Distance(p, v.position) <= v.radius && SameSideOfWalls(p, v.position));
        }

        void OnFrontierReached(Vector2 at, double t)
        {
            visited.Add(new Visited { position = at, radius = c.visited_radius_m, until = t + c.visited_hold_s });
            sweeps.RemoveAll(s => t - s.t > c.sweep_repeat_s);
            if (sweeps.Exists(s => Vector2.Distance(s.position, motion.Pose.position) < c.sweep_spacing_m)) return;
            // sweep_once_per_room: one sweep on first entering a room of the ledger, none at later frontiers in it. Headless seed
            // 5001: sweeps took 9 to 15 s of turning and 13 to 19 s of standing per robot in 240 s.
            if (c.room_ledger && c.sweep_once_per_room)
            {
                int room = planner.Rooms.RoomAt(motion.Pose.position);
                if (room < 0 || !sweptRooms.Add(room)) return;
            }
            sweeps.Add((motion.Pose.position, t));
            sweeping = true; sweepIndex = 0; sweepCheckedIndex = -1; sweepBase = motion.Pose.yawRad; sweepHoldUntil = double.NaN; sweepStarted = t;
            Emit(t, "sweep_started", "New area: inspect the left, right and forward camera sectors that still contain unsearched floor.",
                "x_m", motion.Pose.position.x, "y_m", motion.Pose.position.y);
        }

        double sweepStarted;
        readonly HashSet<int> sweptRooms = new HashSet<int>();
        /// <summary>A look lasts long enough for the colour person detector to see the same place person_rgb_sightings times,
        /// whatever the phase of its scans. 0.6 s held the camera for less than one scan period (1 s) while a candidate needs two
        /// sightings, so a look could never make one (23 Oct memo, section 1.4; 6 of 30 people in 922 to 928 were seen but never
        /// became a candidate).</summary>
        float LookHoldS => Mathf.Max(c.sweep_hold_s, c.person_rgb_sightings * c.person_rgb_scan_period_s);

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
            while (sweeping && sweepCheckedIndex != sweepIndex)
            {
                float heading = sweepBase + SweepOffsetsDeg[sweepIndex] * Mathf.Deg2Rad;
                var forward = new Vector2(-Mathf.Sin(heading), Mathf.Cos(heading));
                Vector2 eye = motion.Pose.position + forward * sensors.camera.mount.forward_offset_m;
                var camera = sensors.camera;
                float left = Mathf.Atan((camera.cx + .5f) / camera.fx);
                float right = Mathf.Atan((camera.width - .5f - camera.cx) / camera.fx);
                float minForward = 0;
                if (c.motion_sweep_visible_floor && c.search_low_m > 0)
                {
                    float bottom = camera.mount.pitch_down_deg * Mathf.Deg2Rad + Mathf.Atan((camera.height - .5f - camera.cy) / camera.fy);
                    minForward = bottom > 0 ? Mathf.Max(0, camera.mount.height_m - c.search_low_m) / Mathf.Tan(bottom) : float.PositiveInfinity;
                }
                if (!grid.HasUnsearchedFloorInView(eye, heading, left, right, c.search_max_range_m, minForward))
                {
                    Emit(t, "sweep_look_skipped", "This camera sector contains only searched floor or mapped opaque obstacles within search range.",
                        "look_index", sweepIndex, "offset_deg", SweepOffsetsDeg[sweepIndex]);
                    SkipSweepLook(t, "remaining sectors already searched or walled off");
                    continue;
                }
                sweepCheckedIndex = sweepIndex;
            }
            if (!sweeping) return Command(0, 0, dt, true);
            float target = sweepBase + SweepOffsetsDeg[sweepIndex] * Mathf.Deg2Rad;
            float error = BrainMath.Angle(motion.Pose.yawRad, target);
            if (Mathf.Abs(error) > c.sweep_tolerance_deg)
                return Command(0, Mathf.Clamp(error * c.heading_gain, -c.sweep_yaw_dps, c.sweep_yaw_dps), dt, true);
            if (double.IsNaN(sweepHoldUntil))
            {
                sweepHoldUntil = t + LookHoldS;
                Emit(t, "sweep_look", "Hold a camera sector that contains unsearched floor.",
                    "look_index", sweepIndex, "offset_deg", SweepOffsetsDeg[sweepIndex]);
            }
            if (t < sweepHoldUntil) return Command(0, 0, dt, true);
            sweepHoldUntil = double.NaN;
            if (++sweepIndex < SweepOffsetsDeg.Length) return Command(0, 0, dt, true);
            EndSweep(t, "all looks done");
            return Command(0, 0, dt, true);
        }

        Vector2? doorwayProbeGoal;
        Vector2 doorwayProbeForward;
        double doorwayProbeStarted, doorwayProbeHoldUntil = double.NaN;
        readonly List<(Vector2 position, double t)> doorwayProbes = new List<(Vector2, double)>();
        readonly List<(Vector2 position, Vector2 forward, double t)> localEvidenceLooks = new List<(Vector2, Vector2, double)>();

        bool TryStartLocalEvidenceLook(double t)
        {
            if (!c.frontier_local_evidence || !c.search_requires_ceiling || !c.ceiling_sight_lines || nearFire ||
                AgentMoveRefusal(t) != null || !InSearchArea(motion.Pose.position)) return false;
            localEvidenceLooks.RemoveAll(p => t - p.t > c.sweep_repeat_s);
            // Resume the interrupted frontier after one attempt here. Selecting a different point in the same
            // camera view must not start another look before the robot has walked to a new standpoint.
            if (localEvidenceLooks.Exists(p => Vector2.Distance(p.position, motion.Pose.position) <= c.visited_radius_m)) return false;
            var exits = new List<Vector2>(Exits());
            bool Excluded(Vector2 p) => ClaimBlocks(p, t) || GoalBlacklisted(p, t) || VisitedGoal(p, t) ||
                exits.Exists(e => Vector2.Distance(e, p) <= c.threshold_exit_clear_m);
            float halfFov = Mathf.Atan(Mathf.Min(sensors.camera.cx + .5f, sensors.camera.width - .5f - sensors.camera.cx) / sensors.camera.fx) * Mathf.Rad2Deg;
            bool Tried(Vector2 at, Vector2 toward) => localEvidenceLooks.Exists(q =>
                Vector2.Distance(q.position, at) <= c.visited_radius_m && Vector2.Distance(q.position, toward) <= c.search_max_range_m &&
                Vector2.Angle(q.forward, toward - q.position) <= halfFov && grid.EvidenceSightLineClear(q.position, toward));
            var look = planner.LocalEvidenceLook(motion.Pose.position, InSearchArea, Excluded, Tried);
            if (look == null || look.path.Count == 0 || !planner.PathSafe(look.path)) return false;
            var at = look.path[look.path.Count - 1];
            if (exits.Exists(e => Vector2.Distance(e, at) <= c.threshold_exit_clear_m)) return false;
            SetPath(look.path); lastPlan = t; doorwayProbeGoal = goal;
            doorwayProbeForward = (look.toward - at).normalized;
            doorwayProbeStarted = t; doorwayProbeHoldUntil = double.NaN;
            localEvidenceLooks.Add((at, doorwayProbeForward, t));
            Emit(t, "local_evidence_started", "Local mapped floor is still unproven indoors: look from proven floor before leaving this search horizon.",
                "x_m", at.x, "y_m", at.y, "toward_x_m", look.toward.x, "toward_y_m", look.toward.y,
                "path_cost", look.cost, "mapped_unsearched_m2", look.mappedCells * grid.CellM * grid.CellM);
            return true;
        }

        /// <summary>Take one bounded look at a nearby mapped wall opening before walking past it. The camera goes
        /// only to the wall line; the body stays its calibrated camera offset behind it. A doorway is never labelled
        /// indoors just because it exists. Depth frames from the look still have to supply valid ceiling evidence.</summary>
        /// <summary>An opening within threshold_exit_clear_m of a known way out is the entrance, whose far side is the yard.
        /// A robot staying inside never goes back out through it, so a ceiling look through it proves nothing it can use
        /// (one-door seed 928, robot A at 9 s turned back to its own front door and looked out at the yard for 9.8 s).</summary>
        bool NearWayOut(Vector2 p)
        {
            foreach (var e in Exits()) if (Vector2.Distance(e, p) <= c.threshold_exit_clear_m) return true;
            return false;
        }
        bool TryStartDoorwayProbe(double t)
        {
            if (!c.search_requires_ceiling || !c.ceiling_sight_lines || c.frontier_peek_m <= 0 || nearFire ||
                AgentMoveRefusal(t) != null || !InSearchArea(motion.Pose.position)) return false;
            doorwayProbes.RemoveAll(p => t - p.t > c.sweep_repeat_s);
            grid.PrepareNavigation();
            int start = grid.Cell(motion.Pose.position), reach = Mathf.CeilToInt(c.frontier_peek_m / grid.CellM);
            var candidates = new List<(Vector2 openingCentre, Vector2 forward, Vector2 target, float distance)>();
            for (int dy = -reach; dy <= reach; dy++) for (int dx = -reach; dx <= reach; dx++)
            {
                int x = start % grid.Size + dx, y = start / grid.Size + dy;
                if (!grid.Inside(x, y)) continue;
                Vector2 camera = grid.Centre(y * grid.Size + x);
                if (Vector2.Distance(camera, motion.Pose.position) > c.frontier_peek_m ||
                    doorwayProbes.Exists(p => Vector2.Distance(p.position, camera) <= 2 * c.robot_radius_m)) continue;
                for (int r = 0; r < c.enclosure_rays; r++)
                {
                    float a = r * 2 * Mathf.PI / c.enclosure_rays; var forward = new Vector2(Mathf.Cos(a), Mathf.Sin(a));
                    Vector2 beyond = camera + forward * c.frontier_peek_m; int cell = grid.Cell(beyond);
                    if (grid.OwnOccupancy(cell) != 1 || !grid.Unsearched(cell) ||
                        !grid.DoorwayFromInside(camera, forward, out var openingCentre) || !grid.OwnFreeLine(camera, beyond) || InSearchArea(beyond) ||
                        doorwayProbes.Exists(p => Vector2.Distance(p.position, openingCentre) <= 2 * c.robot_radius_m) ||
                        (c.stay_inside && NearWayOut(openingCentre))) continue;
                    Vector2 target = camera - forward * sensors.camera.mount.forward_offset_m;
                    float distance = Vector2.Distance(motion.Pose.position, target);
                    if (distance > c.frontier_peek_m || !grid.Standable(grid.Cell(target)) ||
                        ClaimBlocks(target, t) || GoalBlacklisted(target, t) || VisitedGoal(target, t)) continue;
                    candidates.Add((openingCentre, forward, target, distance));
                }
            }
            candidates.Sort((a, b) => a.distance.CompareTo(b.distance));
            foreach (var candidate in candidates)
            {
                var route = planner.Plan(motion.Pose.position, candidate.target);
                if (route.Count == 0 || Vector2.Distance(route[route.Count - 1], candidate.target) > c.goal_tolerance_m) continue;
                float walked = 0; Vector2 previous = motion.Pose.position; bool ownFloor = true;
                foreach (var p in route)
                {
                    walked += Vector2.Distance(previous, p); previous = p;
                    if (grid.OwnOccupancy(grid.Cell(p)) != 1) ownFloor = false;
                }
                if (!ownFloor || walked > c.frontier_peek_m || !planner.PathSafe(route)) continue;
                SetPath(route); lastPlan = t; doorwayProbeGoal = goal;
                doorwayProbeForward = candidate.forward; doorwayProbeStarted = t; doorwayProbeHoldUntil = double.NaN;
                doorwayProbes.Add((candidate.openingCentre, t));
                Emit(t, "doorway_probe_started", "Move the camera to a mapped opening for one ceiling-evidence look; the opening alone proves no indoor floor.",
                    "x_m", goal.Value.x, "y_m", goal.Value.y, "path_m", walked, "max_m", c.frontier_peek_m);
                return true;
            }
            return false;
        }

        void EndDoorwayProbe(double t, string why)
        {
            Emit(t, "doorway_probe_done", why, "elapsed_s", t - doorwayProbeStarted,
                "x_m", motion.Pose.position.x, "y_m", motion.Pose.position.y);
            doorwayProbeGoal = null; doorwayProbeHoldUntil = double.NaN; ClearPath();
            progressMark = t; knownAtMark = knownCells;
        }

        BrainCommand? DoorwayProbeStep(double t, float dt)
        {
            if (!doorwayProbeGoal.HasValue) return null;
            if (t - doorwayProbeStarted >= c.sweep_max_s)
            { EndDoorwayProbe(t, "Evidence look reached its time limit."); return Command(0, 0, dt, true); }
            // Ordinary frontier arrival can stop short of the jamb plane. Place the camera within half a map cell
            // for this evidence look, while keeping the same body clearance and swept-path checks.
            float arriveM = Mathf.Min(c.goal_tolerance_m, grid.CellM * .5f);
            if (Vector2.Distance(motion.Pose.position, doorwayProbeGoal.Value) > arriveM)
            {
                if (path.Count == 0 || !planner.PathSafe(path, cursor))
                { EndDoorwayProbe(t, "The evidence approach is no longer safe."); return Command(0, 0, dt, true); }
                return FollowOrHold(t, dt, arriveM);
            }
            float error = BrainMath.Angle(motion.Pose.yawRad, BrainMath.Heading(doorwayProbeForward));
            if (Mathf.Abs(error) > c.sweep_tolerance_deg)
                return Command(0, Mathf.Clamp(error * c.heading_gain, -c.sweep_yaw_dps, c.sweep_yaw_dps), dt, true);
            if (double.IsNaN(doorwayProbeHoldUntil)) doorwayProbeHoldUntil = t + LookHoldS;
            if (t >= doorwayProbeHoldUntil) EndDoorwayProbe(t, "One doorway look completed; only sensed ceiling evidence can admit the floor ahead.");
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
            Vector2 at = motion.Pose.position;
            if (!exploring || double.IsNaN(progressMark)) { progressMark = t; knownAtMark = n; progressPose = at; travelSinceMark = 0; return; }
            travelSinceMark += Vector2.Distance(at, progressPose); progressPose = at;
            if (n - knownAtMark >= c.progress_min_cells) { progressMark = t; knownAtMark = n; travelSinceMark = 0; return; }
            if (t - progressMark < c.progress_window_s) return;
            // Standing (turning in place, held between agent orders) is not circling: the frontiers near a robot that has
            // not moved are not what failed (held-out seed 914 run 4, robot A: a 2.5 m circle set aside the open door of an
            // unsearched bedroom while A turned in place to face warm shapes).
            if (travelSinceMark < c.progress_min_travel_m) { progressMark = t; knownAtMark = n; travelSinceMark = 0; return; }
            visited.Add(new Visited { position = motion.Pose.position, radius = c.progress_skip_radius_m, until = t + c.visited_hold_s });
            Emit(t, "no_progress", $"Map grew by only {n - knownAtMark} cells in {t - progressMark:0} s; leave this spot for the next unexplored area.",
                "x_m", motion.Pose.position.x, "y_m", motion.Pose.position.y, "new_cells", n - knownAtMark);
            progressMark = t; knownAtMark = n; travelSinceMark = 0;
            ClearPath();
        }
    }
}
