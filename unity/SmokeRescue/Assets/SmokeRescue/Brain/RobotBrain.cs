using System;
using System.Collections.Generic;
using SmokeRescue.Sensors;
using UnityEngine;

namespace SmokeRescue.Brain
{
    /// <summary>One robot's deterministic sensor-to-command state machine. Call from one simulation thread.
    /// Deliver all due motion samples before Step. Image readbacks may be delayed up to frame_max_age_s.</summary>
    public sealed partial class RobotBrain
    {
        sealed class Track
        {
            public Vector2 position;
            public double lastSeen, retryAfter;
            public readonly Queue<double> observations = new Queue<double>();
            public bool candidate, confirmed, rgb;
        }
        readonly struct AirSample
        { public readonly double t; public readonly float c; public AirSample(double time, float value) { t = time; c = value; } }
        readonly SensorConfig sensors;
        readonly BrainConfig c;
        readonly string id;
        readonly int scanSign;
        readonly DeadReckoning motion;
        readonly OccupancyGrid grid;
        readonly GridPlanner planner;
        readonly ScanMatcher matcher;
        readonly CameraHealth health;
        readonly List<RgbdFrame> rgbQueue = new List<RgbdFrame>(), rgbHistory = new List<RgbdFrame>();
        readonly List<ThermalFrame> thermalQueue = new List<ThermalFrame>();
        readonly List<AirSample> airQueue = new List<AirSample>();
        readonly List<BrainEvent> events = new List<BrainEvent>();
        readonly List<Track> tracks = new List<Track>();
        readonly List<Vector2> survivors = new List<Vector2>();
        List<Vector2> path = new List<Vector2>();
        Track active;
        Vector2? goal;
        int cursor;
        double lastStep = double.NaN, lastRgb = double.NegativeInfinity, lastThermal = double.NegativeInfinity;
        double lastAir = double.NegativeInfinity, lastPlan = double.NegativeInfinity, scanSince = double.NaN;
        double nextSurvivorSearch;
        float airC, budget = 1, dwell, commandedSpeed;
        bool retreating, returning, completed, nearFire, waitingSensors, inspecting, holdPending;
        string returnReason, motionPolicy;
        string State => completed ? "complete" : retreating ? "retreat" : returning ? "return" : recoveryPhase != 0 ? "recover" : active != null ? "survivor" : "explore";

        public RobotBrain(SensorConfig cfg, BrainConfig brainCfg, string robotId, int seed, EntryPose? entryPose = null)
        {
            sensors = cfg ?? throw new ArgumentNullException(nameof(cfg)); c = brainCfg ?? throw new ArgumentNullException(nameof(brainCfg));
            c.Validate(); ValidateCalibration(cfg);
            if (string.IsNullOrWhiteSpace(robotId)) throw new ArgumentException("Robot ID is required.", nameof(robotId));
            id = robotId; scanSign = (seed & 1) == 0 ? 1 : -1; LinkDiagnostics = c.link_diagnostics;
            var entry = entryPose.HasValue ? entryPose.Value.pose : new BrainPose();
            home = entry.position;
            airC = c.ambient_c; motion = new DeadReckoning(c.motion_stale_s, c.frame_max_age_s + c.frame_pair_tolerance_s, entry); grid = new OccupancyGrid(c);
            if (grid.Cell(home) < 0) throw new ArgumentException("Entry door is outside the configured building grid.");
            planner = new GridPlanner(grid, c) { Home = home }; health = new CameraHealth(c); matcher = new ScanMatcher(grid, c);
            grid.MarkFootprint(home);
            observedSequence = new long[SharedSize * SharedSize];
        }
        public void OnOdometry(OdometrySample sample) => motion.Add(sample);
        public void OnImu(ImuSample sample) => motion.Add(sample);
        public void OnRgbd(RgbdFrame frame)
        {
            if (frame == null) throw new ArgumentNullException(nameof(frame)); BrainMath.Time(frame.Timestamp);
            if (frame.Width != sensors.camera.width || frame.Height != sensors.camera.height || frame.Rgb == null || frame.Depth == null ||
                frame.Rgb.Length != frame.Width * frame.Height * 3 || frame.Depth.Length != frame.Width * frame.Height)
                throw new ArgumentException("RGB-D dimensions or buffers do not match calibration.");
            if (frame.Timestamp <= lastRgb) return;
            if (rgbQueue.Exists(f => f.Timestamp == frame.Timestamp)) return;
            rgbQueue.Add(frame); rgbQueue.Sort((a, b) => a.Timestamp.CompareTo(b.Timestamp));
        }
        public void OnThermal(ThermalFrame frame)
        {
            if (frame == null) throw new ArgumentNullException(nameof(frame)); BrainMath.Time(frame.Timestamp);
            if (frame.Width != sensors.thermal.width || frame.Height != sensors.thermal.height || frame.DeciKelvin == null || frame.DeciKelvin.Length != frame.Width * frame.Height)
                throw new ArgumentException("Thermal dimensions or buffer do not match calibration.");
            if (frame.Timestamp <= lastThermal) return;
            if (thermalQueue.Exists(f => f.Timestamp == frame.Timestamp)) return;
            thermalQueue.Add(frame); thermalQueue.Sort((a, b) => a.Timestamp.CompareTo(b.Timestamp));
        }
        public void OnLidar(LidarScan scan)
        {
            if (scan == null) throw new ArgumentNullException(nameof(scan)); BrainMath.Time(scan.Timestamp);
            if (scan.RangesM == null || scan.RangesM.Length < 8 || !(scan.AngleIncrementRad > 0) || !(scan.MaxM > scan.MinM))
                throw new ArgumentException("LiDAR sweep is malformed.");
            if (scan.Timestamp <= lastLidar || lidarQueue.Exists(s => s.Timestamp == scan.Timestamp)) return;
            lidarQueue.Add(scan); lidarQueue.Sort((a, b) => a.Timestamp.CompareTo(b.Timestamp));
        }
        public void OnAirTemperature(double t, float temperatureC)
        {
            BrainMath.Time(t); if (!BrainMath.Finite(temperatureC)) throw new ArgumentException("Nonfinite air temperature.");
            if (t <= lastAir || airQueue.Exists(a => a.t == t)) return;
            airQueue.Add(new AirSample(t, temperatureC)); airQueue.Sort((a, b) => a.t.CompareTo(b.t));
        }
        public BrainCommand Step(double t, float dt)
        {
            BrainMath.Time(t);
            if (t < lastPeerReceived || !BrainMath.Finite(dt) || dt <= 0 || (!double.IsNaN(lastStep) && t <= lastStep))
                throw new ArgumentException("Step requires increasing simulation timestamps and positive dt.");
            float elapsed = double.IsNaN(lastStep) ? 0 : (float)(t - lastStep);
            if (!double.IsNaN(lastStep) && Math.Abs(elapsed - dt) > Math.Max(1e-5, dt * .001))
                throw new ArgumentException("dt must equal elapsed simulation time.");
            double previous = double.IsNaN(lastStep) ? t : lastStep; lastStep = t;
            motion.Advance(t); grid.DecayTo(t); ConsumeLidar(t); ConsumeImages(t); UpdateHeatBudget(previous, t);
            grid.ClearBody(motion.Pose.position, c.robot_radius_m - grid.CellM * .5f);
            InvalidatePeerGoal(t); CheckMapOverlap(t);
            grid.PrepareNavigation();
            int cell = grid.Cell(motion.Pose.position);
            bool fireNow = grid.FireDistance(cell) < c.heat_cost_radius_m;
            if (fireNow && !nearFire)
                Emit(t, "avoid_fire", "Observed or rising heat imposes a forbidden margin and a graded path cost.",
                    "fire_distance_m", grid.FireDistance(cell), "clearance_m", c.fire_clearance_m);
            nearFire = fireNow;

            if (!completed && !returning && t >= c.mission_time_limit_s) BeginReturn(t, "Exploration time limit reached.");
            CheckWayOut(t);
            CheckEmergencyExit(t);
            if (!retreating && !emergencyExit && (airC > c.heat_retreat_c || budget < c.heat_retreat_fraction))
            {
                ResetStuckRecovery(); retreating = true; sweeping = false; active = null; dwell = 0; ClearPath();
                Emit(t, "retreat", "Air temperature or remaining heat budget requires a cool reachable goal.", "air_c", airC, "heat_budget", budget);
            }
            else if (retreating && airC < c.heat_recover_below_c && t - lastAir <= c.frame_max_age_s && budget >= c.heat_resume_fraction)
            {
                ResetStuckRecovery(); retreating = false; ClearPath();
                Emit(t, "goal", "Cool air and a recovered heat budget permit travel again.", "air_c", airC, "heat_budget", budget);
            }
            CheckReenter(t);
            if (completed && !retreating) return Command(0, 0, dt, true);
            bool sensorsHealthy = t - lastRgb <= c.frame_max_age_s && motion.Healthy(t) &&
                t - lastAir <= c.frame_max_age_s && (!health.Thermal || t - lastThermal <= c.frame_max_age_s);
            if (!sensorsHealthy)
            {
                if (!waitingSensors)
                    Emit(t, "replan", "Hold until geometry, motion, air temperature and the active camera are recent.", "rgb_age_s", Age(t, lastRgb));
                waitingSensors = true; dwell = 0; ResetStuckRecovery(); return Command(0, 0, dt, true);
            }
            if (waitingSensors) { waitingSensors = false; ClearPath(); Emit(t, "replan", "Sensor streams recovered; rebuild the local path."); }

            if (recoveryPhase != 0) return RecoverStuck(t, dt);
            if (DetectStuck(t, previous, elapsed)) return Command(0, 0, dt, true);

            if (holdPending)
            {
                // The swept-path limit has held a forward command at zero for collision_escape_s: back up, turn and
                // re-route. No contact strip is marked, because the obstacle is already in the map.
                holdPending = false; sweeping = false;
                BeginRecovery(t, "collision_hold", "Forward motion blocked by a mapped obstacle in the body's path; back up and choose another route.", goal, false);
                return Command(0, 0, dt, true);
            }
            if (retreating)
            {
                if (NeedPlan(t))
                {
                    bool blocked = path.Count > 0 && !planner.PathSafe(path, cursor);
                    var escape = goal.HasValue && !blocked ? planner.Plan(motion.Pose.position, goal.Value) : planner.CoolestPath(motion.Pose.position);
                    SetPath(escape); lastPlan = t;
                    Emit(t, "replan", escape.Count > 0 ? "Retreat along the coolest reachable known route." : "No safe cool route is known; hold position.", "path_points", path.Count);
                    if (goal.HasValue) EmitGoal(t, "Coolest reachable known cell, with a preference for the start.");
                }
                return FollowOrHold(t, dt);
            }
            if (returning)
            {
                // The start is usually a doorway, which is narrower than the planning clearance, so the return path ends at
                // the nearest walkable cell short of it (house reflex check: robot B stopped 0.3 m from its start and replanned
                // the same 3-point path for 45 s). Within return_arrive_m of the start counts as back.
                float toExit = float.PositiveInfinity; foreach (var e in Exits()) toExit = Mathf.Min(toExit, Vector2.Distance(motion.Pose.position, e));
                if (toExit <= Mathf.Max(c.goal_tolerance_m, c.return_arrive_m))
                {
                    completed = true; ClearPath();
                    // Left early to stay safe, or carried someone out: it may go back in once that is safe (CheckReenter).
                    canReenter = returnReason == WayOutClosing || returnReason == EmergencyExit || returnReason == CarryingOut;
                    if (returnReason == CarryingOut)
                    {
                        carryingSpeed = 0;
                        Emit(t, "person_carried_out", "Reached a way out with the person on the sled.", "distance_to_exit_m", toExit);
                    }
                    Emit(t, "mission_complete", returnReason + " Reached a known way out.", "distance_to_start_m", Vector2.Distance(motion.Pose.position, home),
                        "distance_to_exit_m", toExit);
                    return Command(0, 0, dt, true);
                }
                if (NeedPlan(t))
                {
                    SetPath(PlanOut()); lastPlan = t;
                    if (path.Count == 0 && returnReason == WayOutClosing)
                    {
                        // Fire cut the way home while the robot was on it (agent run 12: robot A held still for the rest
                        // of the run). Same as finding it cut: retreat to the coolest floor and search on while it is cool.
                        LoseWayOut(t);
                        return Command(0, 0, dt, true);
                    }
                    Emit(t, "replan", path.Count > 0 ? "Return to the start through observed free space." : "The known return route is blocked; hold and keep sensing.", "path_points", path.Count);
                }
                return FollowOrHold(t, dt);
            }

            if (active != null && (active.confirmed || t - active.lastSeen > c.survivor_lost_s))
            {
                Emit(t, "goal", "Survivor inspection ended or recent evidence was lost; resume exploration.");
                active = null; dwell = 0; ClearPath();
            }
            if (active == null && t >= nextSurvivorSearch)
            {
                float best = float.PositiveInfinity;
                foreach (var track in tracks)
                {
                    if (!track.candidate || track.confirmed || track.retryAfter > t || GoalBlacklisted(track.position, t) || t - track.lastSeen > FreshS(track)) continue;
                    // A warm shape outside the ceilinged floor is beyond a window or door: not a person inside.
                    if (!InSearchArea(track.position)) continue;
                    float range = Vector2.Distance(track.position, motion.Pose.position);
                    if (range < best) { best = range; active = track; }
                }
                if (active != null)
                {
                    // The no-frontier scan must be a continuous look around; an inspection interrupts it.
                    ClearPath(); dwell = 0; inspecting = false; scanSince = double.NaN;
                    Emit(t, "goal", "Inspect the nearest recent survivor candidate at a safe standoff.",
                        "person_x_m", active.position.x, "person_y_m", active.position.y, "standoff_m", c.survivor_standoff_m);
                }
            }
            if (active != null)
            {
                float range = Vector2.Distance(active.position, motion.Pose.position);
                float error = BrainMath.Angle(motion.Pose.yawRad, BrainMath.Heading(active.position - motion.Pose.position));
                if (range >= c.survivor_stop_min_m && range <= c.survivor_stop_max_m)
                {
                    if (!inspecting)
                    {
                        inspecting = true; SetPath(new List<Vector2>());
                        Emit(t, "goal", "At survivor viewing distance; stop and face the person for confirmation.", "range_m", range);
                    }
                    bool steady = Mathf.Abs(error) <= c.survivor_face_tolerance_deg && motion.Speed <= c.survivor_stop_speed_mps &&
                        Mathf.Abs(motion.YawRate) * Mathf.Rad2Deg <= c.survivor_stop_yaw_dps && t - active.lastSeen <= FreshS(active);
                    dwell = steady ? dwell + elapsed : 0;
                    if (dwell + 1e-6f >= c.survivor_dwell_s && AgentConfirms)
                    {
                        // The agent decides who is a person, from OpenCV evidence; the reflex layer only brings the
                        // cameras to a steady close view, then moves on until the agent confirms or rejects.
                        Emit(t, "survivor_view_ready", "Stopped facing a warm person-sized shape; the agent decides whether it is a person.",
                            "x_m", active.position.x, "y_m", active.position.y, "range_m", range);
                        active.retryAfter = t + c.stuck_blacklist_s; active = null; dwell = 0; inspecting = false; ClearPath();
                        nextSurvivorSearch = t + c.replan_s;
                        return Command(0, 0, dt, true);
                    }
                    if (dwell + 1e-6f >= c.survivor_dwell_s)
                    {
                        active.confirmed = true; survivors.Add(active.position);
                        confirmed.Add(new BrainSurvivor { position = active.position, confirmedAt = t, sourceRobot = id });
                        Emit(t, "survivor_confirmed", "Stopped facing a recently observed person for the required dwell.",
                            "x_m", active.position.x, "y_m", active.position.y, "range_m", range, "dwell_s", dwell);
                        active = null; dwell = 0; ClearPath(); nextSurvivorSearch = t + c.replan_s;
                    }
                    // A deadband permits a measured stationary dwell rather than continuous tiny turns.
                    return Command(0, Mathf.Abs(error) <= c.survivor_face_tolerance_deg ? 0 : error * c.heading_gain, dt, true);
                }
                dwell = 0; inspecting = false;
                if (NeedPlan(t))
                {
                    SetPath(planner.ApproachPath(motion.Pose.position, active.position)); lastPlan = t;
                    Emit(t, "replan", "Find a known free approach at the survivor standoff distance.", "path_points", path.Count, "standoff_m", c.survivor_standoff_m);
                    if (goal.HasValue) EmitGoal(t, "Approach a repeated warm person-sized blob, then stop and face it.");
                    else
                    {
                        active.retryAfter = t + c.survivor_lost_s; active = null; nextSurvivorSearch = t + c.replan_s;
                        Emit(t, "goal", "No safe known survivor approach exists yet; explore for another approach.");
                    }
                }
                if (active != null) return FollowOrHold(t, dt);
            }
            if (goal.HasValue && Vector2.Distance(goal.Value, motion.Pose.position) <= c.goal_tolerance_m)
            {
                bool wasFrontier = frontierGoal; var reached = goal.Value;
                ClearPath();
                if (wasFrontier) OnFrontierReached(reached, t);
            }
            if (sweeping && !returning) return SweepStep(t, dt);
            CheckProgress(t);
            if (agentGoal.HasValue && Vector2.Distance(agentGoal.Value, motion.Pose.position) <= c.goal_tolerance_m)
            {
                Emit(t, "goal", "Reached the agent's goal.", "x_m", agentGoal.Value.x, "y_m", agentGoal.Value.y);
                var reached = agentGoal.Value; bool frontier = agentGoalIsFrontier; agentGoal = null; agentGoalIsFrontier = false;
                if (frontier) { ClearPath(); OnFrontierReached(reached, t); if (sweeping) return SweepStep(t, dt); }
            }
            if (NeedPlan(t))
            {
                bool blocked = path.Count > 0 && !planner.PathSafe(path, cursor);
                if (blocked) Emit(t, nearFire ? "avoid_fire" : "replan", "A mapped obstacle or fire margin blocks the previous path; discard it.");
                if (goal.HasValue && !blocked)
                {
                    // Keep what the goal is: an agent go_to refreshed as a frontier sent a claim with radius 0, which the
                    // partner rejected (slow-fire run, 7 s: the recording stopped on "Invalid peer claim").
                    SetPath(planner.Plan(motion.Pose.position, goal.Value), frontierGoal); lastPlan = t;
                    Emit(t, "replan", "Periodic refresh through current free space and heat costs.", "path_points", path.Count);
                }
                if ((path.Count == 0 || blocked) && AgentGoalActive(t))
                {
                    // An agent order outranks frontier choice until it is reached, expires or proves unreachable.
                    var agentPath = planner.Plan(motion.Pose.position, agentGoal.Value); lastPlan = t;
                    if (agentPath.Count > 0)
                    {
                        SetPath(agentPath, agentGoalIsFrontier); scanSince = double.NaN;
                        if (agentGoalIsFrontier) { frontierCentre = agentGoal.Value; frontierRadius = c.peer_claim_radius_m; }
                        Emit(t, "goal", "Agent order: " + agentGoalReason, "x_m", agentGoal.Value.x, "y_m", agentGoal.Value.y, "path_points", path.Count);
                    }
                    else { Emit(t, "replan", "Agent goal has no known free path; back to exploring.", "x_m", agentGoal.Value.x, "y_m", agentGoal.Value.y); agentGoal = null; }
                }
                if (path.Count == 0 || blocked)
                {
                    System.Func<Vector2, bool> excluded = p => ClaimBlocks(p, t) || GoalBlacklisted(p, t) || VisitedGoal(p, t) || !InSearchArea(p);
                    var frontier = planner.Frontier(motion.Pose.position, excluded);
                    lastPlan = t;
                    if (frontier != null)
                    {
                        SetPath(frontier.path, true); frontierCentre = frontier.claimCentre; frontierRadius = frontier.claimRadius;
                        scanSince = double.NaN;
                        Emit(t, "goal", "Reachable frontier minimizes travel and heat cost minus nearby unknown-cell gain.",
                            "x_m", frontier.position.x, "y_m", frontier.position.y, "path_cost", frontier.cost, "information_cells", frontier.information, "score", frontier.score);
                        Emit(t, "replan", "Build a path to the selected frontier.", "path_points", path.Count);
                    }
                    else
                    {
                        ClearPath(); lastPlan = t; if (double.IsNaN(scanSince))
                        {
                            scanSince = t; Emit(t, "goal", "No reachable frontier is visible; scan unseen directions before returning.");
                            // With an agent, the model hears of it during the scan and may order a way on (a door, debris).
                            Emit(t, "no_frontier", "No reachable unsearched floor in this robot's map; returning after the scan unless ordered on.",
                                "scan_s", c.no_frontier_scan_s);
                        }
                        if (t - scanSince >= c.no_frontier_scan_s) BeginReturn(t, "No reachable frontier remains after a full scan.");
                    }
                }
            }
            if (!goal.HasValue && !returning) return Command(0, scanSign * c.scan_yaw_dps, dt, true);
            return FollowOrHold(t, dt);
        }

        readonly List<LidarScan> lidarQueue = new List<LidarScan>();
        double lastLidar = double.NegativeInfinity, nextLocalizationReport;
        int matched, unmatched; float fitSum; Vector3 correctionSum;
        /// <summary>LiDAR first, so the pose it corrects is the one the camera frames are then projected from.</summary>
        void ConsumeLidar(double t)
        {
            while (lidarQueue.Count > 0 && lidarQueue[0].Timestamp <= t)
            {
                var scan = lidarQueue[0]; lidarQueue.RemoveAt(0);
                if (scan.Timestamp <= lastLidar || t - scan.Timestamp > c.frame_max_age_s) continue;
                var prior = motion.At(scan.Timestamp);
                var match = matcher.Match(scan, prior);
                if (match.applied)
                {
                    // Heading follows the gyro and takes only scan_yaw_gain of each match's turn: the gyro alone stays within
                    // 0.55 deg over 200 s and 3669 deg of turning (run 13, robot A, against truth), while full scan-matched turns
                    // pulled the heading toward a map that had itself turned, up to 2.4 deg. Position takes the full match.
                    var corrected = new BrainPose { position = match.pose.position,
                        yawRad = prior.yawRad + c.scan_yaw_gain * BrainMath.Angle(prior.yawRad, match.pose.yawRad) * Mathf.Deg2Rad };
                    motion.ApplyCorrection(ScanMatcher.Delta(prior, corrected));
                    matched++; fitSum += match.fit; correctionSum += new Vector3(match.dx, match.dy, match.dyawRad * Mathf.Rad2Deg);
                }
                else unmatched++;
                var integrated = motion.At(scan.Timestamp);
                grid.IntegrateLidar(scan, integrated);
                KeepKeyframe(scan, integrated, t);
                lastLidar = scan.Timestamp;
            }
            if (t >= nextLocalizationReport && matched + unmatched > 0)
            {
                Emit(t, "localization", "LiDAR sweeps matched against the robot's own map; odometry corrected by the summed shift.",
                    "matched", matched, "skipped", unmatched, "mean_fit", matched > 0 ? fitSum / matched : 0,
                    "dx_m", correctionSum.x, "dy_m", correctionSum.y, "dyaw_deg", correctionSum.z);
                matched = unmatched = 0; fitSum = 0; correctionSum = Vector3.zero; nextLocalizationReport = t + c.localization_report_s;
            }
        }
        void ConsumeImages(double t)
        {
            while (rgbQueue.Count > 0 && rgbQueue[0].Timestamp <= t)
            {
                var f = rgbQueue[0]; rgbQueue.RemoveAt(0);
                if (f.Timestamp <= lastRgb || t - f.Timestamp > c.frame_max_age_s) continue;
                var pose = motion.At(f.Timestamp); grid.IntegrateDepth(f, sensors, pose);
                if (health.Observe(f, sensors))
                    Emit(f.Timestamp, "sensor_switch", health.Thermal ? "Sustained poor colour health; use thermal perception and reduce speed." : "Sustained colour recovery; use RGB mode again.",
                        "score", health.Score, "corners", health.Corners, "contrast_std", health.Contrast, "valid_depth_fraction", health.ValidDepth, "threshold", c.health_threshold,
                        "required_duration_s", health.Thermal ? c.switch_to_thermal_s : c.switch_to_rgb_s);
                lastRgb = f.Timestamp; rgbHistory.Add(f);
            }
            while (thermalQueue.Count > 0 && thermalQueue[0].Timestamp <= t)
            {
                var thermal = thermalQueue[0]; RgbdFrame match = null; double difference = double.PositiveInfinity;
                foreach (var rgb in rgbHistory)
                {
                    double delta = Math.Abs(rgb.Timestamp - thermal.Timestamp);
                    if (delta < difference) { match = rgb; difference = delta; }
                }
                if (match == null || difference > c.frame_pair_tolerance_s)
                {
                    if (t - thermal.Timestamp <= c.frame_max_age_s) break;
                    thermalQueue.RemoveAt(0); continue;
                }
                thermalQueue.RemoveAt(0);
                if (thermal.Timestamp <= lastThermal || t - thermal.Timestamp > c.frame_max_age_s) continue;
                var pose = motion.At(match.Timestamp);
                ThermalPerception.ProjectHeat(thermal, match, sensors, c, pose, grid);
                ObservePeople(thermal.Timestamp, ThermalPerception.Detect(thermal, match, sensors, c, pose));
                lastThermal = thermal.Timestamp;
            }
            rgbHistory.RemoveAll(f => t - f.Timestamp > c.frame_max_age_s + c.frame_pair_tolerance_s);
        }
        void ObservePeople(double t, List<PersonDetection> detections)
        {
            foreach (var detection in detections)
            {
                bool known = false; foreach (var person in survivors) if (Vector2.Distance(person, detection.position) <= c.survivor_merge_m) { known = true; break; }
                if (known) continue;
                Track nearest = null; float best = c.survivor_merge_m;
                foreach (var track in tracks)
                {
                    float d = Vector2.Distance(track.position, detection.position);
                    if (!track.confirmed && d <= best) { best = d; nearest = track; }
                }
                if (nearest == null) { nearest = new Track { position = detection.position, lastSeen = double.NegativeInfinity }; tracks.Add(nearest); }
                if (t <= nearest.lastSeen) continue; // At most one vote per thermal frame.
                nearest.position = detection.position; nearest.lastSeen = t;
                nearest.observations.Enqueue(t);
                while (nearest.observations.Count > 0 && t - nearest.observations.Peek() > c.candidate_window_s) nearest.observations.Dequeue();
                if (!nearest.candidate && nearest.observations.Count >= c.candidate_frames)
                {
                    nearest.candidate = true;
                    Emit(t, "survivor_candidate", "A human-temperature, person-sized depth blob persisted across independent frames.",
                        "x_m", nearest.position.x, "y_m", nearest.position.y, "area_m2", detection.areaM2, "frames", nearest.observations.Count);
                }
            }
        }
        /// <summary>Colour sightings arrive once per scan period, thermal blobs every frame, so freshness depends on the source.</summary>
        double FreshS(Track track) => track.rgb ? c.person_rgb_fresh_s : c.survivor_fresh_s;

        /// <summary>A person seen by the colour camera's OpenCV 5 dnn detector (YOLO11m, run by the recorder's vision
        /// service on this robot's own frame at t): bearing in degrees, positive to the right; range is the median depth
        /// of the box. Thermal saturates inside a fire room's hot gas layer and found no person in 161 visible views
        /// (warm_eval); the colour camera still sees them. Repeated sightings make a candidate like thermal blobs do.</summary>
        public void OnPersonSighting(double t, float bearingDegRight, float rangeM, float confidence)
        {
            BrainMath.Time(t);
            if (!BrainMath.Finite(bearingDegRight) || !BrainMath.Finite(rangeM) || !BrainMath.Finite(confidence)) throw new ArgumentException("Nonfinite person sighting.");
            if (confidence < c.person_rgb_min_confidence || !(rangeM > 0) || rangeM > c.person_rgb_max_range_m) return;
            var pose = motion.Covers(t) ? motion.At(t) : motion.Pose;
            Vector2 camera = pose.Transform(0, sensors.camera.mount.forward_offset_m);
            float heading = pose.yawRad - bearingDegRight * Mathf.Deg2Rad;
            var position = camera + new Vector2(-Mathf.Sin(heading), Mathf.Cos(heading)) * rangeM;
            foreach (var person in survivors) if (Vector2.Distance(person, position) <= c.survivor_merge_m) return;
            Track nearest = null; float best = c.survivor_merge_m;
            foreach (var track in tracks)
            {
                float d = Vector2.Distance(track.position, position);
                if (!track.confirmed && d <= best) { best = d; nearest = track; }
            }
            if (nearest == null) { nearest = new Track { position = position, lastSeen = double.NegativeInfinity }; tracks.Add(nearest); }
            if (t <= nearest.lastSeen) return;
            nearest.rgb = true; nearest.position = position; nearest.lastSeen = t; nearest.observations.Enqueue(t);
            while (nearest.observations.Count > 0 && t - nearest.observations.Peek() > c.person_rgb_window_s) nearest.observations.Dequeue();
            if (!nearest.candidate && nearest.observations.Count >= c.person_rgb_sightings)
            {
                nearest.candidate = true;
                Emit(t, "survivor_candidate", "OpenCV dnn person detector (YOLO11m) saw a person in the colour image in repeated scans.",
                    "x_m", position.x, "y_m", position.y, "confidence", confidence, "range_m", rangeM, "frames", nearest.observations.Count, "rgb", 1);
            }
        }

        void UpdateHeatBudget(double from, double t)
        {
            while (airQueue.Count > 0 && airQueue[0].t <= t)
            {
                var sample = airQueue[0]; airQueue.RemoveAt(0);
                if (sample.t < from) { if (sample.t > lastAir) { airC = sample.c; lastAir = sample.t; } continue; }
                IntegrateBudget((float)(sample.t - from), from); from = sample.t;
                airC = sample.c; lastAir = sample.t;
            }
            IntegrateBudget((float)(t - from), from);
        }
        void IntegrateBudget(float duration, double t)
        {
            if (airC > c.heat_drain_start_c) budget -= (airC - c.heat_drain_start_c) * c.heat_drain_per_degree_s * duration;
            else if (airC < c.heat_recover_below_c && t - lastAir <= c.frame_max_age_s) budget += c.heat_recover_per_s * duration;
            budget = Mathf.Clamp01(budget);
        }
        bool NeedPlan(double t) => t - lastPlan >= c.replan_s || (path.Count > 0 && !planner.PathSafe(path, cursor));
        sealed class StuckPlace
        { public Vector2 position; public int repeats; public double lastAt; }
        sealed class BlockedGoal
        { public Vector2 position; public double until; }
        readonly List<StuckPlace> stuckPlaces = new List<StuckPlace>();
        readonly List<BlockedGoal> blockedGoals = new List<BlockedGoal>();
        double stalledFor, recoverySince;
        int recoveryPhase, recoveryTurnSign;
        BrainPose contactPose;
        float recoveryYaw;
        Vector2? recoveryGoal;
        bool recoveryFrontier;

        void ResetStuckRecovery()
        { stalledFor = 0; recoveryPhase = 0; recoveryGoal = null; }
        bool GoalBlacklisted(Vector2 point, double t)
        {
            blockedGoals.RemoveAll(g => t >= g.until);
            return blockedGoals.Exists(g => Vector2.Distance(point, g.position) <= c.stuck_repeat_radius_m);
        }
        void BlacklistGoal(Vector2 point, double t, int repeats)
        {
            blockedGoals.Add(new BlockedGoal { position = point, until = t + c.stuck_blacklist_s });
            Emit(t, "goal_blacklisted", "Repeated stalls near one location exclude this goal area until the retry time.",
                "x_m", point.x, "y_m", point.y, "repeats", repeats, "until_s", t + c.stuck_blacklist_s);
        }
        bool DetectStuck(double t, double previous, float elapsed)
        {
            // commandedSpeed is the command held over the interval just integrated, after limits.
            if (commandedSpeed <= c.stuck_command_mps ||
                !motion.ForwardStayedNearZero(previous, t, c.stuck_measured_mps))
            { stalledFor = 0; return false; }
            stalledFor += elapsed;
            if (stalledFor + 1e-6 < c.stuck_window_s) return false;
            BeginRecovery(t, "stuck", "Forward commands produced negligible forward odometry for the detection window.", goal, true);
            return true;
        }
        /// <summary>markContact: the body stalled against something the map does not show (low debris under the LiDAR
        /// plane, a leaf in the camera's blind zone), so mark a contact strip ahead. A stop for a mapped obstacle must
        /// not: repeated strips used to ring the robot in phantom walls until no route existed.</summary>
        void BeginRecovery(double t, string kind, string reason, Vector2? target, bool markContact)
        {
            contactPose = motion.Pose; recoveryGoal = target; recoveryFrontier = frontierGoal;
            ObstacleAhead(c.near_cone_deg, out float side);
            recoveryTurnSign = Mathf.Abs(side) < 1e-5f ? scanSign : side > 0 ? -1 : 1;
            stuckPlaces.RemoveAll(p => t - p.lastAt >= c.stuck_blacklist_s);
            var place = stuckPlaces.Find(p => Vector2.Distance(p.position, contactPose.position) <= c.stuck_repeat_radius_m);
            if (place == null) { place = new StuckPlace { position = contactPose.position }; stuckPlaces.Add(place); }
            place.repeats++; place.lastAt = t;
            int marked = markContact ? grid.MarkStuckStrip(contactPose) : 0;
            Emit(t, kind, reason,
                "commanded_speed_mps", commandedSpeed, "measured_speed_mps", motion.ForwardSpeed,
                "duration_s", stalledFor, "marked_cells", marked, "repeats", place.repeats,
                "x_m", contactPose.position.x, "y_m", contactPose.position.y);
            if (place.repeats >= c.stuck_repeat_count && !returning && !retreating)
            {
                if (recoveryGoal.HasValue) BlacklistGoal(recoveryGoal.Value, t, place.repeats);
                if (active != null)
                {
                    active.retryAfter = t + c.stuck_blacklist_s;
                    BlacklistGoal(active.position, t, place.repeats); active = null;
                }
                recoveryGoal = null;
            }
            ClearPath(); dwell = 0; inspecting = false; scanSince = double.NaN;
            stalledFor = 0; recoveryPhase = 1; recoverySince = t;
        }
        BrainCommand RecoverStuck(double t, float dt)
        {
            if (recoveryPhase == 1)
            {
                float backed = Mathf.Max(0, Vector2.Dot(contactPose.position - motion.Pose.position, contactPose.Forward));
                float remaining = c.stuck_reverse_m - backed;
                float speed = Mathf.Min(c.stuck_reverse_mps, c.limits.max_reverse_mps, Mathf.Max(0, remaining) / dt);
                // A rear camera is not assumed. Reverse only through the mapped free footprint.
                float nextSpeed = Mathf.MoveTowards(commandedSpeed, -speed, c.limits.max_accel_mps2 * dt);
                if (remaining > grid.CellM * .05f && speed > 0 && -nextSpeed * dt <= remaining &&
                    t - recoverySince < c.stuck_reverse_timeout_s &&
                    planner.ReverseSafe(motion.Pose.position, motion.Pose.position - motion.Pose.Forward * remaining))
                    return Command(-speed, 0, dt, false);
                recoveryPhase = 2; recoverySince = t; recoveryYaw = motion.Pose.yawRad;
                return Command(0, 0, dt, true);
            }
            float turned = recoveryTurnSign * Mathf.DeltaAngle(recoveryYaw * Mathf.Rad2Deg, motion.Pose.yawRad * Mathf.Rad2Deg);
            if (turned < c.stuck_turn_deg && t - recoverySince < c.stuck_turn_timeout_s)
            {
                float yaw = Mathf.Min(c.scan_yaw_dps, (c.stuck_turn_deg - turned) / dt);
                return Command(0, recoveryTurnSign * yaw, dt, true);
            }
            var target = recoveryGoal; bool frontier = recoveryFrontier;
            ResetStuckRecovery(); ClearPath();
            if (target.HasValue && (returning || retreating || !GoalBlacklisted(target.Value, t)))
            {
                SetPath(planner.Plan(motion.Pose.position, target.Value), frontier); lastPlan = t;
                // A failed route must allow frontier selection on the next step.
                if (path.Count == 0) lastPlan = double.NegativeInfinity;
            }
            Emit(t, "replan", "Stuck recovery ended; rebuild the route around the inferred contact or select another frontier.",
                "path_points", path.Count);
            return Command(0, 0, dt, true);
        }
        void SetPath(List<Vector2> value, bool isFrontier = false) { frontierGoal = isFrontier && value.Count > 0; path = value; cursor = 0; goal = path.Count > 0 ? path[path.Count - 1] : (Vector2?)null; }
        void ClearPath() { frontierGoal = false; path.Clear(); cursor = 0; goal = null; lastPlan = double.NegativeInfinity; }
        double nextWayOutCheck; bool hadWayOut;
        /// <summary>
        /// Keep a way out. Every escape_check_s while searching, plan the route home; if fire comes within
        /// escape_fire_margin_m of it (beyond escape_ignore_m from the robot, where the avoid margins already act), the
        /// route is about to close: return now. Agent run 11: robot A searched beside a spreading fire for 26 s with
        /// its paths cut again and again, retreated only when the air reached 70 C, and by then its 15 m route home
        /// was gone (95 s above 60 C, 383 C peak).
        /// </summary>
        void CheckWayOut(double t)
        {
            if (completed || returning || retreating || t < nextWayOutCheck) return;
            nextWayOutCheck = t + c.escape_check_s;
            var route = PlanOut();
            if (route.Count == 0)
            {
                if (hadWayOut) LoseWayOut(t);
                return;
            }
            hadWayOut = true;
            float closest = float.PositiveInfinity; Vector2 at = motion.Pose.position;
            foreach (var p in route)
                if (Vector2.Distance(p, at) > c.escape_ignore_m) closest = Mathf.Min(closest, grid.FireDistance(grid.Cell(p)));
            if (closest < c.escape_fire_margin_m)
            {
                Emit(t, "escape_closing", "Fire is closing on the way out; leave while it is open.", "fire_distance_m", closest, "margin_m", c.escape_fire_margin_m);
                BeginReturn(t, WayOutClosing);
            }
        }
        const string WayOutClosing = "The way out was closing.", EmergencyExit = "Emergency exit.", CarryingOut = "Carrying a person out.";
        float carryingSpeed;
        /// <summary>A person is strapped onto the sled: take them to the nearest known way out at carry speed, then the
        /// robot may go back in (CheckReenter). The heat reflexes stay active.</summary>
        public void BeginCarry(double t, float speedMps, string who)
        {
            if (!(speedMps > 0)) throw new ArgumentOutOfRangeException(nameof(speedMps));
            carryingSpeed = speedMps; canReenter = false; completed = false; retreating = false; emergencyExit = false;
            Emit(t, "carrying_out", "Person strapped onto the sled; taking them to the nearest known way out.", "speed_mps", speedMps);
            BeginReturn(t, CarryingOut);
        }
        bool canReenter; double nextReenterCheck;
        /// <summary>
        /// A robot that left early to stay safe goes back in once its heat budget has recovered and a frontier can be
        /// reached by a path that stays escape_fire_margin_m clear of fire (beyond escape_ignore_m), so the way back out
        /// is open too. Slow-fire run: robot A left at 74 s and stood outside its door for the last two minutes.
        /// </summary>
        void CheckReenter(double t)
        {
            if (!completed || !canReenter || t < nextReenterCheck) return;
            nextReenterCheck = t + c.escape_check_s;
            if (budget < c.heat_resume_fraction || t >= c.mission_time_limit_s) return;
            grid.FireClearance = c.fire_clearance_m;
            System.Func<Vector2, bool> excluded = p => ClaimBlocks(p, t) || GoalBlacklisted(p, t) || VisitedGoal(p, t) || !InSearchArea(p);
            var frontier = planner.Frontier(motion.Pose.position, excluded);
            if (frontier == null) return;
            float closest = float.PositiveInfinity;
            foreach (var p in frontier.path)
                if (Vector2.Distance(p, motion.Pose.position) > c.escape_ignore_m) closest = Mathf.Min(closest, grid.FireDistance(grid.Cell(p)));
            if (closest < c.escape_fire_margin_m) return;
            completed = false; canReenter = false; emergencyExit = false; returning = false; hadWayOut = false; ClearPath();
            Emit(t, "reenter", "Heat budget recovered and a frontier is reachable with the fire clear of the way: go back in.",
                "fire_distance_m", closest, "information_cells", frontier.information, "heat_budget", budget);
        }
        bool emergencyExit; double nextEmergencyCheck;
        /// <summary>
        /// Last resort: the heat budget is nearly spent and no exit can be reached with the normal fire clearance. Plan the
        /// exit with emergency_fire_clearance_m (past the fire, never through a burning cell) and leave. 2x2 building run:
        /// robot A retreated beside its entry door with its heat budget at 0 while fire margins closed every route out.
        /// </summary>
        void CheckEmergencyExit(double t)
        {
            if (emergencyExit || completed || budget >= c.heat_emergency_fraction || t < nextEmergencyCheck) return;
            nextEmergencyCheck = t + c.escape_check_s;
            // Leave by the normal route when there is one: the retreat's "coolest reachable cell" can be a hot spot next
            // to the robot (2x2 attempt 4: robot A sat 0.2 m from its retreat goal with the budget at 0 and a route out open).
            var exit = PlanOut(); float clearance = c.fire_clearance_m;
            if (exit.Count == 0)
            {
                grid.FireClearance = clearance = c.emergency_fire_clearance_m;
                exit = PlanOut();
                if (exit.Count == 0) { grid.FireClearance = c.fire_clearance_m; return; }
            }
            emergencyExit = true; retreating = false;
            Emit(t, "emergency_exit", "Heat budget nearly spent: leave by the nearest way out" +
                (clearance < c.fire_clearance_m ? ", past the fire at reduced clearance." : "."),
                "heat_budget", budget, "clearance_m", clearance, "route_points", exit.Count);
            BeginReturn(t, EmergencyExit);
        }
        /// <summary>Known ways out: this robot's own entry and, once a partner's map is linked, the partner's entry (the
        /// origin of its frame), which it reported by radio. Slow-fire run: robot B's way home was cut by fire 5.5 m from
        /// the door robot A had entered and left by, and it held still for 50 s.</summary>
        IEnumerable<Vector2> Exits()
        {
            yield return home;
            foreach (var partner in peers.Keys)
                if (PartnerToOwn(partner, Vector2.zero, out var entry)) yield return entry;
        }
        /// <summary>The shortest known route to any way out (fewest waypoints), or an empty path.</summary>
        List<Vector2> PlanOut()
        {
            List<Vector2> best = new List<Vector2>();
            foreach (var exit in Exits())
            {
                var route = planner.Plan(motion.Pose.position, exit);
                if (route.Count > 0 && (best.Count == 0 || route.Count < best.Count)) best = route;
            }
            return best;
        }
        void LoseWayOut(double t)
        {
            hadWayOut = false; returning = false; ResetStuckRecovery(); retreating = true; sweeping = false; active = null; dwell = 0; ClearPath();
            // A route that reappears next to the fire would start a return that is cut again at once; wait a while.
            nextWayOutCheck = t + c.escape_recheck_s;
            Emit(t, "escape_closing", "The known way out is closed; retreat to the coolest reachable floor and search on while it is cool.", "route_points", 0);
        }
        void BeginReturn(double t, string reason)
        {
            ResetStuckRecovery(); returning = true; active = null; dwell = 0; returnReason = reason; ClearPath();
            Emit(t, "goal", reason + " Return to the estimated start.", "x_m", home.x, "y_m", home.y);
        }
        BrainCommand FollowOrHold(double t, float dt)
        {
            if (path.Count == 0 || !goal.HasValue) return Command(0, 0, dt, true);
            Vector2 position = motion.Pose.position;
            while (cursor + 1 < path.Count && Vector2.Distance(position, path[cursor]) < c.goal_tolerance_m) cursor++;
            if (Vector2.Distance(position, goal.Value) < c.goal_tolerance_m) return Command(0, 0, dt, true);
            Vector2 target = path[cursor], previous = position; float remain = c.lookahead_m;
            for (int i = cursor; i < path.Count; i++)
            {
                float distance = Vector2.Distance(previous, path[i]);
                if (distance >= remain) { target = Vector2.MoveTowards(previous, path[i], remain); break; }
                target = path[i]; remain -= distance; previous = path[i];
            }
            if (!planner.SegmentSafe(position, target)) target = path[cursor];
            if (!planner.SegmentSafe(position, target))
            {
                // The planned cells are checked by PathSafe; when they are still safe, only the straight line from the exact
                // position grazes a margin cell on the way to the next waypoint, about one cell away: steer to it. Clearing the
                // path here rebuilt the same path 185 times in 20 s (2x2 building run, robot B, cause not reproduced).
                int here = grid.Cell(position);
                if (!planner.PathSafe(path, cursor))
                {
                    Emit(t, "replan", "Pursuit chord crosses an inflated obstacle or heat margin; stop and refresh.",
                        "fire_distance_m", grid.FireDistance(here), "obstacle_distance_m", grid.ObstacleDistance(here));
                    ClearPath(); return Command(0, 0, dt, true);
                }
                target = path[cursor];
            }
            float error = BrainMath.Angle(motion.Pose.yawRad, BrainMath.Heading(target - position));
            float near = ObstacleAhead(c.near_cone_deg, out _);
            float speed = near < c.near_obstacle_m ? c.near_speed_mps : c.open_speed_mps;
            string policy = near < c.near_obstacle_m ? "near" : "open";
            if (health.Thermal) { speed = Mathf.Min(speed, c.thermal_speed_mps); policy = "thermal"; }
            speed *= Mathf.Clamp01(Vector2.Distance(position, goal.Value) / c.goal_slow_distance_m);
            float yaw = 2 * speed * Mathf.Sin(error * Mathf.Deg2Rad) / Mathf.Max(grid.CellM, Vector2.Distance(position, target)) * Mathf.Rad2Deg;
            if (Mathf.Abs(error) > c.turn_in_place_deg)
            {
                speed = 0; policy = "turn";
                // Brake before rotating, so the acceleration ramp cannot make a turn drift into a wall.
                yaw = commandedSpeed > c.survivor_stop_speed_mps ? 0 : error * c.heading_gain;
            }
            float nextSpeed = Mathf.MoveTowards(commandedSpeed, speed, c.limits.max_accel_mps2 * dt);
            float nextYaw = Mathf.Clamp(yaw, -c.limits.max_yaw_rate_dps, c.limits.max_yaw_rate_dps);
            var midPose = motion.Pose; midPose.yawRad += nextYaw * Mathf.Deg2Rad * dt / 2;
            // Check the speed after acceleration limiting, including motion while braking.
            if (nextSpeed > 0 && !planner.SegmentSafe(position, position + midPose.Forward * nextSpeed * dt))
            {
                Emit(t, "collision_risk", "The next commanded translation leaves known safe free space.");
                return Command(0, yaw, dt, true);
            }
            if (policy != motionPolicy)
            {
                motionPolicy = policy;
                Emit(t, "replan", "Local pursuit selects the " + policy + " motion policy from image health, clearance and heading.",
                    "speed_target_mps", speed, "yaw_target_dps", Mathf.Clamp(yaw, -c.limits.max_yaw_rate_dps, c.limits.max_yaw_rate_dps),
                    "occupied_ahead_m", float.IsPositiveInfinity(near) ? -1 : near, "heading_error_deg", error);
            }
            return Command(speed, yaw, dt, false);
        }
        double blockedSince = double.NaN, lastBlocked = double.NegativeInfinity;

        /// <summary>Distance the body can drive straight ahead before its circle (robot radius plus half a cell, since
        /// a mapped cell's surface can sit half a cell from its centre) touches an occupied cell. Cells beside or
        /// behind the body never limit forward motion, so door jambs and a wall at the robot's side do not stop it.</summary>
        float SweptClearance()
        {
            float best = float.PositiveInfinity, r = c.robot_radius_m + grid.CellM * .5f;
            Vector2 position = motion.Pose.position, forward = motion.Pose.Forward;
            float reach = r + c.collision_margin_m + c.limits.max_forward_mps * c.limits.max_forward_mps / (2 * c.limits.max_accel_mps2) + grid.CellM;
            int cell = grid.Cell(position); if (cell < 0) return 0;
            int n = Mathf.CeilToInt(reach / grid.CellM), col = cell % grid.Size, row = cell / grid.Size;
            for (int dy = -n; dy <= n; dy++) for (int dx = -n; dx <= n; dx++)
            {
                if (!grid.Inside(col + dx, row + dy)) continue;
                int i = (row + dy) * grid.Size + col + dx; if (grid.Occupancy(i) != 255) continue;
                Vector2 delta = grid.Centre(i) - position;
                float along = Vector2.Dot(delta, forward), lateral = forward.x * delta.y - forward.y * delta.x;
                if (along <= 0 || Mathf.Abs(lateral) >= r) continue;
                best = Mathf.Min(best, along - Mathf.Sqrt(r * r - lateral * lateral));
            }
            return best;
        }

        /// <summary>Highest forward speed that can still stop collision_margin_m short of the swept clearance at the
        /// acceleration limit. Turning is never limited: the body is round, so turning in place cannot hit anything.</summary>
        float SafeForward(float requested)
        {
            double t = lastStep;
            float room = SweptClearance() - c.collision_margin_m;
            float safe = room <= 0 ? 0 : Mathf.Sqrt(2 * c.limits.max_accel_mps2 * room);
            if (requested <= safe + 1e-4f) { if (t - lastBlocked > 1.0) blockedSince = double.NaN; return requested; }
            if (double.IsNaN(blockedSince))
            {
                blockedSince = t;
                Emit(t, "collision_risk", "A mapped obstacle in the body's path limits forward speed to what can stop in time.",
                    "clearance_m", float.IsPositiveInfinity(room) ? -1 : room + c.collision_margin_m, "requested_mps", requested, "allowed_mps", safe);
            }
            lastBlocked = t;
            if (safe < c.stuck_command_mps && t - blockedSince >= c.collision_escape_s && recoveryPhase == 0) { holdPending = true; blockedSince = double.NaN; }
            return safe;
        }

        float ObstacleAhead(float halfAngle, out float side)
        {
            float best = float.PositiveInfinity; side = 0; Vector2 position = motion.Pose.position, forward = motion.Pose.Forward;
            // Only local cells can affect the speed or stop decision.
            float radius = Mathf.Max(c.near_obstacle_m, c.collision_stop_m + c.limits.max_forward_mps * c.limits.max_forward_mps / (2 * c.limits.max_accel_mps2));
            int cell = grid.Cell(position); if (cell < 0) return 0;
            int r = Mathf.CeilToInt(radius / grid.CellM), col = cell % grid.Size, row = cell / grid.Size;
            for (int dy = -r; dy <= r; dy++) for (int dx = -r; dx <= r; dx++)
            {
                if (!grid.Inside(col + dx, row + dy)) continue;
                int i = (row + dy) * grid.Size + col + dx; if (grid.Occupancy(i) != 255) continue;
                Vector2 delta = grid.Centre(i) - position; float distance = delta.magnitude;
                if (distance >= best || distance > radius || Vector2.Angle(forward, delta) > halfAngle) continue;
                best = distance; side = forward.x * delta.y - forward.y * delta.x;
            }
            return best;
        }
        BrainCommand Command(float speed, float yaw, float dt, bool stop)
        {
            float allowed = Mathf.Min(c.limits.max_forward_mps, health.Thermal ? c.thermal_speed_mps : c.open_speed_mps);
            if (carryingSpeed > 0) allowed = Mathf.Min(allowed, carryingSpeed);
            float limited = Mathf.Clamp(speed, -c.limits.max_reverse_mps, allowed);
            if (!stop && limited > 0) limited = SafeForward(limited);
            // A safety stop bypasses the nominal acceleration ramp. The simulator must honour emergency zero.
            commandedSpeed = stop ? 0 : Mathf.MoveTowards(commandedSpeed, limited, c.limits.max_accel_mps2 * dt);
            commandedSpeed = Mathf.Clamp(commandedSpeed, -c.limits.max_reverse_mps, allowed);
            return new BrainCommand(commandedSpeed, Mathf.Clamp(yaw, -c.limits.max_yaw_rate_dps, c.limits.max_yaw_rate_dps));
        }
        void EmitGoal(double t, string reason)
        { if (goal.HasValue) Emit(t, "goal", reason, "x_m", goal.Value.x, "y_m", goal.Value.y); }
        void Emit(double t, string kind, string reason, params object[] values)
        {
            var numbers = new SortedDictionary<string, double>(StringComparer.Ordinal);
            for (int i = 0; i < values.Length; i += 2) numbers.Add((string)values[i], Convert.ToDouble(values[i + 1], System.Globalization.CultureInfo.InvariantCulture));
            events.Add(new BrainEvent { t = t, robot = id, kind = kind, reason = reason, numbers = numbers });
        }
        static double Age(double t, double observed) => double.IsNegativeInfinity(observed) ? -1 : Math.Max(0, t - observed);
        public IReadOnlyList<BrainEvent> DrainEvents()
        {
            // Stable by timestamp, then insertion order, including delayed camera decisions.
            var indexed = new List<(int index, BrainEvent value)>();
            for (int i = 0; i < events.Count; i++) indexed.Add((i, events[i]));
            indexed.Sort((a, b) => { int order = a.value.t.CompareTo(b.value.t); return order == 0 ? a.index.CompareTo(b.index) : order; });
            var result = new List<BrainEvent>(); foreach (var item in indexed) result.Add(item.value); events.Clear();
            return result.AsReadOnly();
        }
        /// <summary>Where this robot may search. With search_requires_ceiling, only floor its own camera has seen a
        /// ceiling over (OccupancyGrid.Indoors): no building outline, floor plan or operator map is given to any robot.</summary>
        public bool InSearchArea(Vector2 p) => !c.search_requires_ceiling || grid.Indoors(p);
        /// <summary>The indoor layer (OccupancyGrid.CopyIndoors), written beside each map snapshot.</summary>
        public byte[] IndoorMask() => grid.CopyIndoors();
        /// <summary>Set when an agent layer is running: survivors are then confirmed only by the agent's OpenCV-backed
        /// decision, never by the reflex dwell alone.</summary>
        public bool AgentConfirms { get; set; }

        public BrainSnapshot Snapshot() => new BrainSnapshot
        {
            pose = motion.Pose, gridSize = grid.Size, cellM = grid.CellM, gridOrigin = grid.Origin,
            occupancy = grid.CopyOccupancy(), heat = grid.CopyHeat(), path = path.ToArray(), goal = goal,
            mode = health.Thermal ? "thermal" : "rgb", state = State, heatBudget = budget,
            confirmedSurvivors = survivors.ToArray(), fireHotspots = grid.FireHotspots(),
            peerExplored = grid.CopyPeerExplored(), searched = grid.CopySearched(), peers = CopyPeers(), survivors = CopySurvivors()
        };
        static void ValidateCalibration(SensorConfig s)
        {
            if (s.camera?.mount == null || s.depth == null || s.thermal == null || s.camera.width < 3 || s.camera.height < 3 ||
                s.thermal.width < 3 || s.thermal.height < 3 || !(s.camera.fx > 0) || !(s.camera.fy > 0) || !(s.thermal.fx > 0) || !(s.thermal.fy > 0) ||
                !(s.depth.min_m > 0) || !(s.depth.max_m > s.depth.min_m) || !BrainMath.Finite(s.camera.mount.height_m + s.camera.mount.forward_offset_m + s.camera.mount.pitch_down_deg))
                throw new ArgumentException("Brain requires calibrated RGB-D, thermal intrinsics and a camera mount.");
        }
    }
}
