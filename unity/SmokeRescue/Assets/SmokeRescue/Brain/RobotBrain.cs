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
            public bool candidate, confirmed, rgb, onWall, rejected;
            public Vector2 rejectedFrom;   // where the robot stood when the agent rejected it
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
            airC = c.ambient_c; motion = new DeadReckoning(c.motion_stale_s, c.frame_max_age_s + c.frame_pair_tolerance_s, entry); grid = new OccupancyGrid(c) { SeeOverMaxM = cfg.camera.mount.height_m - c.see_over_clearance_m }; grid.TeammateAt = OnTeammate;
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

            if (!completed && !returning && t >= c.mission_time_limit_s) BeginReturn(t, MissionTimeUp);
            RefreshSensedExits(t);
            CheckWayOut(t);
            CheckEmergencyExit(t);
            CheckHeatBuilding(t);
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
            NoteBodyTrack();
            CheckReenter(t);
            if (doorwayProbeGoal.HasValue && (returning || retreating || completed || nearFire || active != null ||
                AgentGoalActive(t) || recoveryPhase != 0 || t - doorwayProbeStarted >= c.sweep_max_s))
                EndDoorwayProbe(t, "Evidence look cancelled by safety, a survivor, an agent order, or its time limit.");
            if (routeLook.HasValue && (retreating || returning || completed || nearFire))
                EndRouteLook(t, "Cancelled: a heat reflex or a return to the way out takes precedence over the route look.", true);
            if (completed && !retreating) return RelocateStep(t, dt) ?? Command(0, 0, dt, true);
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
                bool clearing = !float.IsNaN(carryClearFrom.x);
                bool arrived = toExit <= Mathf.Max(c.goal_tolerance_m, c.return_arrive_m) || clearing;
                if (arrived && returnReason == CarryingOut && carryTowM > 0 && float.IsNaN(stepAwayFrom.x))
                {
                    // The sled rides carryTowM behind the robot: keep walking straight out until the person lies a body
                    // radius plus a robot's width past the way out (hidden-corner run 1: the robot stopped 0.5 m short of its
                    // start, set the person down 5 cm inside the front doorway, then drove into them 13 times trying to go
                    // back in, outside for 156 of 200 s).
                    if (!clearing) { carryClearFrom = motion.Pose.position; carryClearM = toExit + carryTowM + c.carried_person_radius_m + 2 * c.robot_radius_m; }
                    bool more = Vector2.Distance(motion.Pose.position, carryClearFrom) < carryClearM;
                    var ahead = motion.Pose.position + motion.Pose.Forward * Mathf.Max(grid.CellM, carryingSpeed * dt);
                    if (more && planner.SegmentSafe(motion.Pose.position, ahead)) return Command(carryingSpeed, 0, dt, false);
                    // Set them down here. The robot knows where: on its own trail, carryTowM behind it. A person lying on the
                    // floor is below the LiDAR plane and too close for the camera, so the map would never show them; it
                    // marks them itself, then steps clear of that mark so its next plan does not start inside it.
                    Vector2 body = motion.Pose.position - motion.Pose.Forward * carryTowM;
                    setDownAt = body;   // marked once the robot has stepped clear, or its own spot would be inside the mark
                    NoteSetDown(body);
                    Emit(t, "person_carried_out", "Set the person down past the way out and marked where they lie.",
                        "distance_to_exit_m", toExit, "person_x_m", body.x, "person_y_m", body.y);
                    stepAwayFrom = motion.Pose.position; stepAwaySpeed = carryingSpeed; carryingSpeed = 0;
                    stepAwayM = Mathf.Max(0, c.carried_person_radius_m + c.robot_radius_m + grid.CellM * 1.5f - carryTowM);
                    carriedOutReported = true;
                }
                if (!float.IsNaN(stepAwayFrom.x))
                {
                    var ahead = motion.Pose.position + motion.Pose.Forward * Mathf.Max(grid.CellM, stepAwaySpeed * dt);
                    if (Vector2.Distance(motion.Pose.position, stepAwayFrom) < stepAwayM && planner.SegmentSafe(motion.Pose.position, ahead))
                        return Command(stepAwaySpeed, 0, dt, false);
                    grid.PlaceObstacle(setDownAt, c.carried_person_radius_m);
                    arrived = true;
                }
                if (arrived)
                {
                    completed = true; completedAt = t; relocateGoal = null; relocateFrom = null; ClearPath(); carryClearFrom = stepAwayFrom = new Vector2(float.NaN, float.NaN);
                    // Left early to stay safe, or carried someone out: it may go back in once that is safe (CheckReenter).
                    // NoFrontierLeft too: the exclusions that emptied the list (fire-cut blacklist, visited circles, a
                    // partner's claim) expire, and CheckReenter still needs a reachable frontier with the fire clear of the way
                    // (Codex review of held-out seed 914: "no frontier" ended the search for good).
                    canReenter = returnReason == WayOutClosing || returnReason == EmergencyExit || returnReason == CarryingOut || returnReason == HeatLeave || returnReason == NoFrontierLeft;
                    if (returnReason == HeatLeave || returnReason == EmergencyExit) { burned = true; heatExitAt = t; }
                    if (returnReason == CarryingOut && !carriedOutReported)
                    {
                        carryingSpeed = 0;
                        Emit(t, "person_carried_out", "Reached a way out with the person on the sled.", "distance_to_exit_m", toExit);
                    }
                    carriedOutReported = false;
                    Emit(t, "mission_complete", returnReason + " Reached a known way out.", "distance_to_start_m", Vector2.Distance(motion.Pose.position, home),
                        "distance_to_exit_m", toExit);
                    return Command(0, 0, dt, true);
                }
                if (NeedPlan(t))
                {
                    var fresh = PlanOut(); lastPlan = t;
                    // A margin that shuts for a moment does not stop an emergency exit: keep the route while nothing on it burns.
                    if (!(fresh.Count == 0 && EmergencyRoute && path.Count > 0 && EmergencyPathOk(path, cursor))) SetPath(fresh);
                    if (path.Count == 0 && returnReason == EmergencyExit && grid.FireClearance > c.emergency_fire_clearance_m)
                    {
                        // The exit route at normal clearance was cut on the way out (2x2 attempt 5: robot A held still with its
                        // budget spent): leave at the emergency clearance instead, still never through a burning cell.
                        grid.FireClearance = c.emergency_fire_clearance_m; grid.FirePrediction = false; SetPath(PlanOut());
                        Emit(t, "emergency_exit", "The way out was cut on the way: leave past the fire at reduced clearance.",
                            "clearance_m", c.emergency_fire_clearance_m, "route_points", path.Count);
                    }
                    if (path.Count == 0 && EmergencyRoute && grid.FireClearance > c.last_resort_fire_clearance_m)
                    {
                        grid.FireClearance = c.last_resort_fire_clearance_m; SetPath(PlanOut());
                        Emit(t, "emergency_exit", "The way out is shut at the emergency clearance: last-resort route with only the body clear of the fire, never through it.",
                            "clearance_m", c.last_resort_fire_clearance_m, "route_points", path.Count, "air_c", airC);
                    }
                    if (path.Count == 0 && EmergencyRoute)
                    {
                        // No way out is open at all: standing still in rising heat is the worst choice. Go to the coolest
                        // reachable floor; the next replan tries the way out again.
                        var cool = planner.CoolestPath(motion.Pose.position);
                        if (cool.Count > 0)
                        {
                            SetPath(cool);
                            Emit(t, "replan", "No way out is open: move to the coolest reachable floor and keep trying the way out.", "path_points", path.Count);
                            return FollowOrHold(t, dt);
                        }
                    }
                    if (path.Count == 0 && returnReason == CarryingOut && grid.FireClearance > c.emergency_fire_clearance_m && HeatBuilding(t, out _))
                    {
                        // Carrying a person, the way out blocked and the air heating: leave past the fire at reduced
                        // clearance, still as a carry so the person is set down clear of the way out.
                        grid.FireClearance = c.emergency_fire_clearance_m; SetPath(PlanOut());
                        Emit(t, "emergency_exit", "Carrying a person, the way out blocked and the air heating: leave past the fire at reduced clearance.",
                            "clearance_m", c.emergency_fire_clearance_m, "route_points", path.Count, "air_c", airC);
                    }
                    if (path.Count == 0 && returnReason != EmergencyExit && returnReason != CarryingOut && returnReason != WayOutClosing &&
                        returnReason != HeatLeave && returnReason != MissionTimeUp && HeatBuilding(t, out _) &&
                        TryEmergencyExit(t, "The way out is blocked and the air is heating: leave by the nearest way out"))
                        return Command(0, 0, dt, true);
                    // A deadline return cut on the way is the same: retreat and search on, never hold (held-out seed 919,
                    // robot B held 20 s in cool air when fire cut its route home after a deadline return began).
                    if (path.Count == 0 && (returnReason == WayOutClosing || returnReason == HeatLeave || returnReason == MissionTimeUp))
                    {
                        // Fire cut the way home while the robot was on it (agent run 12: robot A held still for the rest
                        // of the run). Same as finding it cut: retreat to the coolest floor and search on while it is cool.
                        LoseWayOut(t);
                        return Command(0, 0, dt, true);
                    }
                    if (path.Count > 0) Emit(t, "replan", "Return to the start through observed free space.", "path_points", path.Count);
                    else
                    {
                        // Say why, so a blocked return can be diagnosed from the log alone (held-out seed 914, robot B).
                        int at = grid.Cell(motion.Pose.position), home0 = grid.Cell(home);
                        Emit(t, "replan", "The known return route is blocked; hold and keep sensing.", "path_points", 0,
                            "start_walkable", grid.Walkable(at) ? 1 : 0, "home_walkable", grid.Walkable(home0) ? 1 : 0,
                            "fire_clearance_m", grid.FireClearance, "fire_prediction", grid.FirePrediction ? 1 : 0,
                            "start_fire_m", grid.FireDistance(at), "start_obstacle_m", grid.ObstacleDistance(at));
                    }
                }
                return FollowOrHold(t, dt);
            }

            // A route look turns in place and keeps the path; a survivor candidate is taken up when it ends.
            var routeLookCommand = RouteLookStep(t, dt);
            if (routeLookCommand.HasValue) return routeLookCommand.Value;

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
                    if (!track.candidate || track.confirmed || track.onWall || track.retryAfter > t || GoalBlacklisted(track.position, t) || t - track.lastSeen > FreshS(track)) continue;
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
            var doorwayLook = DoorwayProbeStep(t, dt);
            if (doorwayLook.HasValue) return doorwayLook.Value;
            if (!sweeping && !AgentGoalActive(t) && !facing.HasValue && t >= agentHoldUntil && NeedPlan(t) && TryStartDoorwayProbe(t))
                return DoorwayProbeStep(t, dt).Value;
            if (goal.HasValue && Vector2.Distance(goal.Value, motion.Pose.position) <= c.goal_tolerance_m)
            {
                bool wasFrontier = frontierGoal; var reached = goal.Value;
                ClearPath();
                if (wasFrontier) OnFrontierReached(reached, t);
            }
            var meet = RendezvousStep(t, dt);
            if (meet.HasValue) return meet.Value;
            if (sweeping && !returning) return SweepStep(t, dt);
            CheckProgress(t);
            var face = FaceStep(t, dt);
            if (face.HasValue) return face.Value;
            // Sent here for a reason: wait for the agent's next order instead of driving off to a frontier.
            if (t < agentHoldUntil && !returning && !retreating) return Command(0, 0, dt, true);
            if (agentGoal.HasValue && Vector2.Distance(agentGoal.Value, motion.Pose.position) <= c.goal_tolerance_m)
            {
                Emit(t, "goal", "Reached the agent's goal.", "x_m", agentGoal.Value.x, "y_m", agentGoal.Value.y);
                // The agent asked to be here for a reason (a closer look, a person to lift, a blocked opening to measure):
                // tell it, or the next look waits for an unrelated event (agent run 14: robot A stood 1.4 m from the person
                // it had confirmed, within the arm's reach, and walked on because nothing asked the agent to act).
                if (!agentGoalIsFrontier) AgentArrived(t, "Arrived where the agent asked: ");
                var reached = agentGoal.Value; bool frontier = agentGoalIsFrontier; agentGoal = null; agentGoalIsFrontier = false;
                if (frontier) { ClearPath(); OnFrontierReached(reached, t); if (sweeping) return SweepStep(t, dt); }
                // Turn or hold from this tick on, before any frontier is chosen (Codex review).
                var turn = FaceStep(t, dt); if (turn.HasValue) return turn.Value;
                if (t < agentHoldUntil) return Command(0, 0, dt, true);
            }
            if (NeedPlan(t))
            {
                bool blocked = path.Count > 0 && !planner.PathSafe(path, cursor);
                if (blocked) Emit(t, nearFire ? "avoid_fire" : "replan", "A mapped obstacle or fire margin blocks the previous path; discard it.");
                // Held-out run seed 914, robot A at 87 to 94 s: the same frontier by a burning doorway was chosen, its path cut
                // by the fire margin 1.5 s later, chosen again as the flames flickered, three times in 4 s: no_frontier kept
                // restarting, the scan never ended and the agent was woken each time. Leave such a frontier for a while.
                // The whole frontier is left, not one standpoint: its standable look-out cells vary from plan to plan (in the
                // test the next one chosen lay 1.0 m from the cut one), and every look-out for the same unsearched floor lies
                // within frontier_peek_m of it, so frontier_peek_m plus the stall radius (or the claim, if larger) is excluded.
                if (blocked && frontierGoal && goal.HasValue && FireCuts(path, cursor))
                    BlacklistGoal(frontierCentre, t, 0, c.fire_cut_blacklist_s, "The fire margin cut the path to this frontier; leave it until the retry time.",
                        Mathf.Max(frontierRadius, c.frontier_peek_m + c.stuck_repeat_radius_m));
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
                    if (agentPath.Count > 0 && !agentGoalIsFrontier &&
                        Vector2.Distance(agentPath[agentPath.Count - 1], motion.Pose.position) <= c.goal_tolerance_m)
                    {
                        // The goal itself is not standable (inside a wall or obstacle margin) and the robot already stands
                        // at the nearest point it can reach: that is arrival. Replanning to the same spot every tick held
                        // robot A still for 13 to 20 s three times in run 19 (700 to 900 identical goal events each).
                        Emit(t, "goal", "Reached the agent's goal as closely as the map allows.", "x_m", agentGoal.Value.x, "y_m", agentGoal.Value.y,
                            "offset_m", Vector2.Distance(agentGoal.Value, motion.Pose.position));
                        AgentArrived(t, "Arrived where the agent asked (as closely as the map allows): ");
                        agentGoal = null; ClearPath();
                    }
                    else if (agentPath.Count > 0)
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
                    // Only the partner stands between this robot and unsearched floor (its claim or the floor its camera
                    // searched): follow through, without claiming, rather than end the search. One-door seed 922, robot B:
                    // behind A at the one door, it yielded the door frontier 120 times in 2 s, found nothing else, and
                    // declared its mission complete at 16.6 s. Beyond the partner's floor its own frontiers take over.
                    var through = frontier == null && peers.Count > 0
                        ? planner.Frontier(motion.Pose.position, p => GoalBlacklisted(p, t) || VisitedGoal(p, t) || !InSearchArea(p), true) : null;
                    List<Vector2> followRoute = null;
                    var follow = frontier == null && through == null && peers.Count > 0 ? FollowPartnerGoal(out followRoute) : null;
                    if (follow.HasValue)
                    {
                        SetPath(followRoute, false); scanSince = double.NaN; noFrontierAt = double.NaN;
                        Emit(t, "goal", "Nothing left to search here and the partner went in: follow it in, so this robot's own sensors map onward.",
                            "x_m", follow.Value.x, "y_m", follow.Value.y);
                    }
                    else if (through != null)
                    {
                        SetPath(through.path, false); scanSince = double.NaN; noFrontierAt = double.NaN;
                        Emit(t, "goal", "Only the partner's claimed or searched floor leads on: follow through it to unsearched floor beyond.",
                            "x_m", through.position.x, "y_m", through.position.y, "path_cost", through.cost);
                    }
                    else if (frontier != null)
                    {
                        SetPath(frontier.path, true); frontierCentre = frontier.claimCentre; frontierRadius = frontier.claimRadius;
                        scanSince = double.NaN; noFrontierAt = double.NaN;
                        Emit(t, "goal", "Reachable frontier minimizes travel and heat cost minus nearby unknown-cell gain.",
                            "x_m", frontier.position.x, "y_m", frontier.position.y, "path_cost", frontier.cost, "information_cells", frontier.information, "score", frontier.score);
                        Emit(t, "replan", "Build a path to the selected frontier.", "path_points", path.Count);
                    }
                    else if (ThresholdLook(t, out var lookRoute, out var beyond))
                    {
                        SetPath(lookRoute, false); scanSince = double.NaN; noFrontierAt = double.NaN;
                        Emit(t, "goal", "Unsearched floor lies only past floor not yet proven indoors: stand at the edge of proven floor and look in for a ceiling.",
                            "x_m", goal.Value.x, "y_m", goal.Value.y, "beyond_x_m", beyond.x, "beyond_y_m", beyond.y);
                    }
                    else
                    {
                        ClearPath(); lastPlan = t; if (double.IsNaN(scanSince))
                        {
                            scanSince = t; noFrontierAt = t; Emit(t, "goal", "No reachable frontier is visible; scan unseen directions before returning.");
                            // With an agent, the model hears of it during the scan and may order a way on (a door, debris).
                            Emit(t, "no_frontier", "No reachable unsearched floor in this robot's map; returning after the scan unless ordered on.",
                                "scan_s", c.no_frontier_scan_s);
                        }
                        // Cut off, a return would only find the way out closed: stay, keep checking for a way out, and leave the
                        // agent free to act (held-out seed 914 run 4, robot B).
                        if (t - scanSince >= c.no_frontier_scan_s && !cutOff) BeginReturn(t, NoFrontierLeft);
                    }
                }
            }
            if (!goal.HasValue && !returning) return Command(0, scanSign * c.scan_yaw_dps, dt, true);
            return FollowOrHold(t, dt);
        }

        readonly List<LidarScan> lidarQueue = new List<LidarScan>();
        double lastLidar = double.NegativeInfinity, nextLocalizationReport;
        LidarScan latestScan;
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
                lastLidar = scan.Timestamp; latestScan = scan;
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
                var scanForRange = c.thermal_lidar_range_fallback && latestScan != null &&
                    Math.Abs(latestScan.Timestamp - thermal.Timestamp) <= c.frame_max_age_s ? latestScan : null;
                ObservePeople(thermal.Timestamp, ThermalPerception.Detect(thermal, match, sensors, c, pose, scanForRange));
                lastThermal = thermal.Timestamp;
            }
            rgbHistory.RemoveAll(f => t - f.Timestamp > c.frame_max_age_s + c.frame_pair_tolerance_s);
        }
        void ObservePeople(double t, List<PersonDetection> detections)
        {
            foreach (var detection in detections)
            {
                bool known = false; foreach (var person in survivors) if (Vector2.Distance(person, detection.position) <= c.survivor_merge_m) { known = true; break; }
                if (known || AtOwnSetDown(detection.position)) continue;
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
                    // A rejected track stays quiet until the robot sees it from a new viewpoint; from the same place a
                    // second look adds nothing (held-out seed 914 run 2: the next thermal frames re-armed the rejected
                    // 38 C fridge in the same second, three times).
                    if (nearest.rejected && !NewView(nearest)) continue;
                    bool again = nearest.rejected; nearest.rejected = false;
                    nearest.candidate = true;
                    // A thermal-only warm shape on a long mapped wall is almost always wall heated by the fire (15 runs: 150 of
                    // 151 such candidates); flagged for the agent and not approached by the robot on its own.
                    nearest.onWall = !nearest.rgb && grid.OccupiedExtent(nearest.position, c.candidate_wall_disc_m, c.candidate_wall_window_m) >= c.candidate_wall_extent_m;
                    Emit(t, "survivor_candidate", nearest.onWall ? "A human-temperature, person-sized depth blob on a long mapped wall persisted across independent frames (thermal only)."
                        : "A human-temperature, person-sized depth blob persisted across independent frames.",
                        "x_m", nearest.position.x, "y_m", nearest.position.y, "area_m2", detection.areaM2, "frames", nearest.observations.Count, "on_wall", nearest.onWall ? 1 : 0,
                        "previously_rejected", again ? 1 : 0);
                }
            }
        }
        /// <summary>Seen from reject_new_view_deg round, or from half the distance, since the agent rejected it.</summary>
        bool NewView(Track track)
        {
            Vector2 then = track.position - track.rejectedFrom, now = track.position - motion.Pose.position;
            return Vector2.Angle(then, now) >= c.reject_new_view_deg || now.magnitude <= .5f * then.magnitude;
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
            if (AtOwnSetDown(position)) return;
            Track nearest = null; float best = c.survivor_merge_m;
            foreach (var track in tracks)
            {
                float d = Vector2.Distance(track.position, position);
                if (!track.confirmed && d <= best) { best = d; nearest = track; }
            }
            if (nearest == null) { nearest = new Track { position = position, lastSeen = double.NegativeInfinity }; tracks.Add(nearest); }
            if (t <= nearest.lastSeen) return;
            // The colour detector is independent evidence: it lifts an earlier thermal rejection.
            nearest.rgb = true; nearest.onWall = false; nearest.rejected = false; nearest.position = position; nearest.lastSeen = t; nearest.observations.Enqueue(t);
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
                airHistory.Add(sample);
                if (sample.t < from) { if (sample.t > lastAir) { airC = sample.c; lastAir = sample.t; } continue; }
                IntegrateBudget((float)(sample.t - from), from); from = sample.t;
                airC = sample.c; lastAir = sample.t;
            }
            IntegrateBudget((float)(t - from), from);
            double keep = t - Math.Max(c.air_rise_window_s, c.air_rise_baseline_s) - 1;
            int old = 0; while (old < airHistory.Count && airHistory[old].t < keep) old++;
            if (old > 0) airHistory.RemoveRange(0, old);
        }
        void IntegrateBudget(float duration, double t)
        {
            if (airC > c.heat_drain_start_c) budget -= (airC - c.heat_drain_start_c) * c.heat_drain_per_degree_s * duration;
            else if (airC < c.heat_recover_below_c && t - lastAir <= c.frame_max_age_s) budget += c.heat_recover_per_s * duration;
            budget = Mathf.Clamp01(budget);
        }
        bool NeedPlan(double t) => t - lastPlan >= c.replan_s || (path.Count > 0 && !(EmergencyRoute ? EmergencyPathOk(path, cursor) : planner.PathSafe(path, cursor)));
        bool EmergencyRoute => returning && returnReason == EmergencyExit;
        /// <summary>On an emergency exit only a burning or occupied cell on the rest of the route breaks it; the fire margin
        /// is already reduced and flickers with the flames (held-out seed 914, robot B, 148 to 163 s: the route flipped between
        /// planned and blocked every second and the robot stood still in the foyer while its air went from 26 to 52 C).</summary>
        bool EmergencyPathOk(List<Vector2> route, int from)
        {
            for (int i = Math.Max(0, from); i < route.Count; i++)
            {
                // Not through an obstacle, and the body (last_resort_fire_clearance_m) kept off burning cells.
                int cell = grid.Cell(route[i]);
                if (cell < 0 || grid.Occupancy(cell) == 255 || grid.FireDistance(cell) <= c.last_resort_fire_clearance_m) return false;
            }
            return true;
        }
        sealed class StuckPlace
        { public Vector2 position; public int repeats; public double lastAt; }
        sealed class BlockedGoal
        { public Vector2 position; public double until; public float radius; }
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
            return blockedGoals.Exists(g => Vector2.Distance(point, g.position) <= (g.radius > 0 ? g.radius : c.stuck_repeat_radius_m) &&
                                            SameSideOfWalls(point, g.position));
        }
        /// <summary>No mapped wall cell lies on the straight line from a to b. Exclusion circles (blacklisted and visited
        /// goals, up to 2.5 m) reached through walls and hid unrelated frontiers in the next room (#27, Codex review).</summary>
        /// <summary>Where this robot's body has stood: a pose every cell of travel, the most recent body_track_poses.</summary>
        readonly List<BrainPose> bodyTrack = new List<BrainPose>();
        void NoteBodyTrack()
        {
            var pose = motion.Pose;
            if (bodyTrack.Count > 0 && Vector2.Distance(bodyTrack[bodyTrack.Count - 1].position, pose.position) < grid.CellM) return;
            bodyTrack.Add(pose); if (bodyTrack.Count > c.body_track_poses) bodyTrack.RemoveAt(0);
        }
        /// <summary>p lies inside this robot's body (the capsule, body_half_length_m and body_radius_m) at a pose it has held.</summary>
        bool OwnBodyWasAt(Vector2 p)
        {
            float half = Mathf.Max(0, c.body_half_length_m - c.body_radius_m), r = c.body_radius_m + grid.CellM * .5f;
            foreach (var pose in bodyTrack)
            {
                Vector2 d = p - pose.position, f = pose.Forward;
                float along = Vector2.Dot(d, f), lateral = f.x * d.y - f.y * d.x, beyond = Mathf.Max(0, Mathf.Abs(along) - half);
                if (beyond * beyond + lateral * lateral <= r * r) return true;
            }
            return false;
        }
        /// <summary>Where to walk to follow a partner in: on the planned route to the partner's last known position, stopping
        /// follow_partner_standoff_m short of it, when the partner stands on floor this robot's map shows indoors and the
        /// route exists; else null. One-door seed 923, robot B: A went in through the one door and B, at the pad, had no
        /// frontier at all (its map held only its own view through the door and A's searched floor), so it scanned and
        /// held at the gate. Walking in after A gives its own sensors the floor beyond.</summary>
        internal Vector2? FollowPartnerGoal(out List<Vector2> route)
        {
            route = null; Vector2? best = null; float bestD = float.PositiveInfinity;
            foreach (var peer in peers.Values)
            {
                if (peer.message == null || !TryPartnerLink(peer.message.Sender, out _)) continue;
                var at = peer.pose.position;
                if (!InSearchArea(at) || Vector2.Distance(at, motion.Pose.position) <= c.follow_partner_standoff_m + c.goal_tolerance_m) continue;
                var path = planner.Plan(motion.Pose.position, at);
                if (path.Count == 0) continue;
                float walked = 0; int cut = path.Count - 1;
                for (int i = path.Count - 1; i > 0; i--) { walked += Vector2.Distance(path[i - 1], path[i]); if (walked >= c.follow_partner_standoff_m) { cut = i - 1; break; } }
                var goalAt = path[cut]; float d = Vector2.Distance(goalAt, motion.Pose.position);
                if (d < bestD) { bestD = d; best = goalAt; route = path.GetRange(0, cut + 1); }
            }
            return best;
        }
        bool SameSideOfWalls(Vector2 a, Vector2 b)
        {
            int steps = Mathf.CeilToInt(Vector2.Distance(a, b) / (grid.CellM * .5f));
            for (int i = 1; i < steps; i++)
                if (grid.OwnOccupancy(grid.Cell(Vector2.Lerp(a, b, i / (float)steps))) == 255) return false;
            return true;
        }
        void BlacklistGoal(Vector2 point, double t, int repeats, double forS = double.NaN, string why = null, float radius = 0)
        {
            double until = t + (double.IsNaN(forS) ? c.stuck_blacklist_s : forS);
            blockedGoals.Add(new BlockedGoal { position = point, until = until, radius = radius });
            Emit(t, "goal_blacklisted", why ?? "Repeated stalls near one location exclude this goal area until the retry time.",
                "x_m", point.x, "y_m", point.y, "repeats", repeats, "until_s", until, "radius_m", radius > 0 ? radius : c.stuck_repeat_radius_m);
        }
        /// <summary>True when the rest of the path passes within the fire clearance of a dangerous cell (the fire margin,
        /// not a wall, is what made it unsafe).</summary>
        bool FireCuts(List<Vector2> route, int from)
        {
            float margin = grid.FireClearance + grid.CellM * .707107f;
            for (int i = Math.Max(0, from); i < route.Count; i++) if (grid.FireDistance(grid.Cell(route[i])) <= margin) return true;
            return false;
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
                wayOutPoints.Clear(); wayOutRates.Clear();
                if (hadWayOut) LoseWayOut(t);
                return;
            }
            hadWayOut = true; cutOff = false;
            float closest = float.PositiveInfinity; Vector2 at = motion.Pose.position;
            foreach (var p in route)
                if (Vector2.Distance(p, at) > c.escape_ignore_m) closest = Mathf.Min(closest, grid.FireDistance(grid.Cell(p)));
            // How fast is the fire closing on the way out? The fire distance at the previous check's route points, then and
            // now: fixed points, so the robot walking nearer a still fire (a new route passing closer) does not read as a
            // closing front (Codex review, run 23 robot A at 14 s).
            float routeM = 0; for (int i = 1; i < route.Count; i++) routeM += Vector2.Distance(route[i - 1], route[i]);
            float walkS = routeM / c.open_speed_mps, closeS = float.PositiveInfinity;
            // Out by the deadline: leave while the walk out, with the same escape_time_factor allowance the fire test uses,
            // still fits before mission_time_limit_s. It used to start back only at the deadline itself.
            if (routeM > c.return_arrive_m && t + walkS * c.escape_time_factor + c.escape_check_s >= c.mission_time_limit_s)
            {
                BeginReturn(t, MissionTimeUp);
                return;
            }
            if (wayOutPoints.Count > 0)
            {
                float now = float.PositiveInfinity; foreach (int cell in wayOutPoints) now = Mathf.Min(now, grid.FireDistance(cell));
                // No fire on the map reads as 1e4 m (the distance transform's empty value): no front to track.
                if (now <= c.grid_size_m && wayOutThen <= c.grid_size_m && t > wayOutThenT)
                    wayOutRates.Add((wayOutThen - now) / (float)(t - wayOutThenT));
                while (wayOutRates.Count > 2) wayOutRates.RemoveAt(0);
                // The smaller of the last two intervals' rates: an advancing front closes in both, while fire seen for the
                // first time nearer the route (run 22, robot A at 10 s) moves the distance in one jump.
                if (wayOutRates.Count == 2 && 2 * c.escape_check_s >= c.escape_rate_window_s - 1e-3f)
                {
                    float rate = Mathf.Min(wayOutRates[0], wayOutRates[1]);
                    if (rate > 0) closeS = Mathf.Max(0, closest - c.fire_clearance_m) / rate;
                }
            }
            wayOutPoints.Clear();
            foreach (var p in route) if (Vector2.Distance(p, at) > c.escape_ignore_m) wayOutPoints.Add(grid.Cell(p));
            wayOutThen = float.PositiveInfinity; foreach (int cell in wayOutPoints) wayOutThen = Mathf.Min(wayOutThen, grid.FireDistance(cell));
            wayOutThenT = t;
            if (closest < c.escape_fire_margin_m || closeS < walkS * c.escape_time_factor)
            {
                Emit(t, "escape_closing", closest < c.escape_fire_margin_m ? "Fire is closing on the way out; leave while it is open."
                    : "Fire is closing on the way out faster than the robot could walk it later; leave while it is open.",
                    "fire_distance_m", closest, "margin_m", c.escape_fire_margin_m, "closes_in_s", float.IsInfinity(closeS) ? -1 : closeS, "walk_out_s", walkS);
                BeginReturn(t, WayOutClosing);
            }
            else CheckRouteWatch(t, route, routeM);
        }
        readonly List<int> wayOutPoints = new List<int>(); readonly List<float> wayOutRates = new List<float>();
        float wayOutThen; double wayOutThenT;

        Vector2? routeLook;
        double routeLookSince, routeLookFacedAt = double.NaN, routeLookEnded = double.NaN;
        /// <summary>
        /// Route watch. The way-out check knows only fire this robot's thermal camera has seen, and a robot searching rooms
        /// that face away from its route home never looks at that route. Held-out run 1: robot A's map held no fire cells
        /// from 96 s to 116 s while the fire spread along its route; at 120 s it turned, 84 fire cells appeared 1.3 m from the
        /// route and the way out closed 0.2 s later (8.6 s above 60 C). So, from the way-out check while searching: when more
        /// than route_watch_max_stale_share of the route home beyond escape_ignore_m has had no thermal look for
        /// route_watch_stale_s, and the last look ended route_watch_min_interval_s ago, turn in place to face the nearest
        /// stale route point, hold for route_watch_dwell_s, then carry on along the same path. The way-out check keeps
        /// running during the look, so fire seen then ends the search as usual.
        /// </summary>
        void CheckRouteWatch(double t, List<Vector2> route, float routeM)
        {
            // The first look comes no sooner than route_watch_min_interval_s after the first check.
            if (double.IsNaN(routeLookEnded)) routeLookEnded = t;
            // Returning, retreating and carrying (a return) are already excluded by the way-out check itself.
            if (!c.route_watch || budget < c.heat_retreat_fraction || airC > c.heat_retreat_c || routeLook.HasValue || t - routeLookEnded < c.route_watch_min_interval_s || active != null || sweeping ||
                nearFire || recoveryPhase != 0 || waitingSensors || facing.HasValue || t < agentHoldUntil) return;
            Vector2 at = motion.Pose.position, nearest = at; int counted = 0, stale = 0; float nearestM = float.PositiveInfinity;
            foreach (var p in route)
            {
                float d = Vector2.Distance(p, at);
                if (d <= c.escape_ignore_m) continue;
                counted++;
                if (t - grid.HeatLookTime(grid.Cell(p)) <= c.route_watch_stale_s) continue;
                stale++;
                if (d < nearestM) { nearestM = d; nearest = p; }
            }
            if (counted == 0 || stale <= c.route_watch_max_stale_share * counted) return;
            routeLook = nearest; routeLookSince = t; routeLookFacedAt = double.NaN;
            Emit(t, "route_watch", "Most of the route home has had no thermal look recently; turn and look along it before searching on.",
                "stale_share", stale / (float)counted, "x_m", nearest.x, "y_m", nearest.y, "route_m", routeM, "distance_m", nearestM);
        }
        /// <summary>Turning in place toward the route look point, then holding still; null when no look is running. Kept
        /// apart from the agent's facing (FaceStep), so neither is ever taken for the other.</summary>
        BrainCommand? RouteLookStep(double t, float dt)
        {
            if (!routeLook.HasValue) return null;
            float error = BrainMath.Angle(motion.Pose.yawRad, BrainMath.Heading(routeLook.Value - motion.Pose.position));
            bool faced = Mathf.Abs(error) <= c.survivor_face_tolerance_deg;
            if (faced && double.IsNaN(routeLookFacedAt))
            {
                // Check the way out on this view now and then every escape_check_s while holding, so one look can give the
                // closing-rate rule the two intervals with a fresh view of the route that it needs.
                routeLookFacedAt = t; nextWayOutCheck = Math.Min(nextWayOutCheck, t);
            }
            bool held = !double.IsNaN(routeLookFacedAt) && t - routeLookFacedAt >= c.route_watch_dwell_s;
            if (held || t - routeLookSince >= c.route_watch_max_s)
            {
                EndRouteLook(t, held ? "Looked along the route home; carry on along the same path." :
                    "The route look reached its time limit; carry on along the same path.", false);
                return null;
            }
            return Command(0, faced ? 0 : Mathf.Clamp(error * c.heading_gain, -c.sweep_yaw_dps, c.sweep_yaw_dps), dt, true);
        }
        void EndRouteLook(double t, string reason, bool cancelled)
        {
            Emit(t, "route_watch_done", reason, "duration_s", t - routeLookSince, "faced", double.IsNaN(routeLookFacedAt) ? 0 : 1,
                "cancelled", cancelled ? 1 : 0);
            routeLook = null; routeLookFacedAt = double.NaN; routeLookEnded = t;
            progressMark = double.NaN;   // restart the progress window: standing to look maps nothing new
        }
        /// <summary>Forget the way-out history and any pending agent arrival: on a return, a retreat or a lost route the
        /// next readings belong to another route and another situation (Codex review).</summary>
        void ResetWayOutAndArrival()
        {
            wayOutPoints.Clear(); wayOutRates.Clear();
            facing = null; agentFace = null; agentHoldUntil = double.NegativeInfinity;
        }
        const string WayOutClosing = "The way out was closing.", EmergencyExit = "Emergency exit.", CarryingOut = "Carrying a person out.";
        float carryingSpeed, carryTowM, carryClearM, stepAwayM, stepAwaySpeed; bool carriedOutReported;
        Vector2 carryClearFrom = new Vector2(float.NaN, float.NaN), stepAwayFrom = new Vector2(float.NaN, float.NaN), setDownAt;
        readonly List<Vector2> setDownPoints = new List<Vector2>();
        /// <summary>Where this robot set down each person it carried out (own frame). Someone seen within survivor_merge_m
        /// (the same-person distance used everywhere) of one is that person, not a new survivor. Held-out run 2: robot B's
        /// agent confirmed the person B itself had set down in the yard as a new survivor at 99.5 s (the sighting lay 0.09 m
        /// from the set-down point). A set-down point lies at least carried_person_radius_m plus a robot's width past the
        /// way out, outside, so a person lying just inside the doorway stays farther than survivor_merge_m from it.</summary>
        internal bool AtOwnSetDown(Vector2 p)
        {
            foreach (var s in setDownPoints) if (Vector2.Distance(s, p) <= c.survivor_merge_m) return true;
            return false;
        }
        void NoteSetDown(Vector2 p)
        {
            setDownPoints.Add(p);
            // A track already open there (the person seen while being set down) must not become a candidate either.
            foreach (var track in tracks)
            {
                if (track.confirmed || Vector2.Distance(track.position, p) > c.survivor_merge_m) continue;
                track.candidate = false; track.retryAfter = double.PositiveInfinity;
                if (active == track) { active = null; dwell = 0; inspecting = false; }
            }
        }
        /// <summary>A person is strapped onto the sled: take them to the nearest known way out at carry speed, then the
        /// robot may go back in (CheckReenter). The heat reflexes stay active.</summary>
        public void BeginCarry(double t, float speedMps, string who, float towOffsetM = 0)
        {
            if (!(speedMps > 0)) throw new ArgumentOutOfRangeException(nameof(speedMps));
            if (!(towOffsetM >= 0)) throw new ArgumentOutOfRangeException(nameof(towOffsetM));
            carryingSpeed = speedMps; carryTowM = towOffsetM; carryClearFrom = new Vector2(float.NaN, float.NaN);
            MarkCarried(t); canReenter = false; completed = false; retreating = false; emergencyExit = false;
            Emit(t, "carrying_out", "Person strapped onto the sled; taking them to the nearest known way out.", "speed_mps", speedMps);
            BeginReturn(t, CarryingOut);
        }
        bool canReenter; double nextReenterCheck;
        /// <summary>Outside after leaving for heat, fire or a rescue, and allowed back in once it is safe (CheckReenter).</summary>
        public bool MayReenter => completed && canReenter;
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
            // Left because heat was building: the hot layer inside cannot be seen from outside and does not clear in
            // seconds (held-out seed 914 run 1: robot A re-entered 0.1 s after its emergency exit, into the hallway it had
            // just left with its air at 34 C and rising). Wait reenter_after_heat_s in cool, steady air first.
            if (burned && (t - heatExitAt < c.reenter_after_heat_s || HeatBuilding(t, out _))) return;
            grid.FireClearance = c.fire_clearance_m; grid.FirePrediction = true;
            System.Func<Vector2, bool> excluded = p => ClaimBlocks(p, t) || GoalBlacklisted(p, t) || VisitedGoal(p, t) || !InSearchArea(p);
            // Every frontier, in the planner's own order, not only the first: the nearest can lie by the fire while a farther
            // one is clear (Codex review, 22 Oct: robot B stood at its door for 119 s of held-out seed 916 while the only
            // frontier checked was the one beside the burning foyer).
            GridPlanner.FrontierChoice frontier = null; float closest = float.PositiveInfinity;
            // Behind a partner at one door, its claim and searched floor can be the only way in: go back in through them
            // (one-door seed 922, robot B walked the outside wall at 31 s while A searched inside).
            var candidates = planner.Frontiers(motion.Pose.position, excluded);
            if (candidates.Count == 0 && peers.Count > 0)
                candidates = planner.Frontiers(motion.Pose.position, p => GoalBlacklisted(p, t) || VisitedGoal(p, t) || !InSearchArea(p), true);
            foreach (var candidate in candidates)
            {
                float near = ReentryFireDistance(candidate.path);
                if (near < c.escape_fire_margin_m) continue;
                frontier = candidate; closest = near; break;
            }
            if (frontier == null) return;
            // The leave rule's closing test too (CheckWayOut): held-out seed 914 run 4, robot A left at 50 s because the fire
            // was closing on its way out and went back in the second it got out (75 s), along a route the front was still
            // advancing on. Go back in only where the robot would not have to leave again at once.
            if (!ReentryFrontSteady(t, frontier.path, closest)) return;
            reenterCells.Clear(); reenterRates.Clear(); reenterThenT = double.NaN;
            completed = false; canReenter = false; emergencyExit = false; returning = false; hadWayOut = false; ClearPath();
            noFrontierAt = double.NaN; cutOff = false; relocateGoal = null; relocateFrom = null;
            // The spot it goes back in from is outside, where it stands: a way out proven on foot. Held-out seed 920, robot A
            // walked 5.7 m along the wall, went back in, and turned round at once because its way out was still planned to
            // its first door, by the fire (174.5 s, fire 1.3 m from that route).
            if (Vector2.Distance(motion.Pose.position, home) > c.survivor_merge_m && !walkedIn.Exists(e => Vector2.Distance(e, motion.Pose.position) <= c.survivor_merge_m))
                walkedIn.Add(motion.Pose.position);
            Emit(t, "reenter", "Heat budget recovered and a frontier is reachable with the fire clear of the way: go back in.",
                "fire_distance_m", closest, "information_cells", frontier.information, "heat_budget", budget);
        }
        readonly List<int> reenterCells = new List<int>(); readonly List<float> reenterRates = new List<float>();
        float reenterThen = float.PositiveInfinity; double reenterThenT = double.NaN;
        /// <summary>The leave rule applied to the way in: the fire distance at the previous check's route cells, now and
        /// then, gives a closing rate per interval; once two intervals spanning escape_rate_window_s are measured, the way in
        /// is steady when the smaller rate is not positive, or when the front would take at least escape_time_factor times
        /// the walk along the route to reach fire_clearance_m of it.</summary>
        bool ReentryFrontSteady(double t, List<Vector2> route, float closest)
        {
            Vector2 at = motion.Pose.position;
            if (reenterCells.Count > 0 && t > reenterThenT)
            {
                float now = float.PositiveInfinity; foreach (int cell in reenterCells) now = Mathf.Min(now, grid.FireDistance(cell));
                // No fire on the map reads as 1e4 m (the distance transform's empty value): nothing closing.
                reenterRates.Add(now <= c.grid_size_m && reenterThen <= c.grid_size_m ? (reenterThen - now) / (float)(t - reenterThenT) : 0);
                while (reenterRates.Count > 2) reenterRates.RemoveAt(0);
            }
            reenterCells.Clear();
            foreach (var p in route) if (Vector2.Distance(p, at) > c.escape_ignore_m) reenterCells.Add(grid.Cell(p));
            reenterThen = float.PositiveInfinity; foreach (int cell in reenterCells) reenterThen = Mathf.Min(reenterThen, grid.FireDistance(cell));
            reenterThenT = t;
            if (reenterRates.Count < 2 || 2 * c.escape_check_s < c.escape_rate_window_s - 1e-3f) return false;
            float rate = Mathf.Min(reenterRates[0], reenterRates[1]);
            if (rate <= 0) return true;
            float routeM = 0; for (int i = 1; i < route.Count; i++) routeM += Vector2.Distance(route[i - 1], route[i]);
            return Mathf.Max(0, closest - c.fire_clearance_m) / rate >= routeM / c.open_speed_mps * c.escape_time_factor;
        }
        /// <summary>The fire distance along a way back in, over every point of it. The leave rule skips the stretch within
        /// escape_ignore_m of the robot, but from outside that stretch is the door, the part the leave rule measures once the
        /// robot is in: held-out seed 920, robot A went back in at 113.5 s (4.8 m clear beyond the door) and turned round at
        /// 115.6 s with the fire 1.97 m from its way out.</summary>
        float ReentryFireDistance(List<Vector2> path)
        {
            float near = float.PositiveInfinity;
            foreach (var p in path) near = Mathf.Min(near, grid.FireDistance(grid.Cell(p)));
            return near;
        }
        double completedAt; Vector2? relocateGoal, relocateFrom;
        /// <summary>
        /// Outside after leaving for safety, allowed back in, and every way in still blocked after exterior_dwell_s: drive to
        /// unexplored floor along the building's wall (outside the search area, within facade_band_m of a mapped wall) to
        /// look for another way in, then wait again while the re-entry check runs. DARPA SubT: CERBERUS backtracks after a
        /// set time without motion, and GBPlanner2 keeps exploring when home is unreachable. Held-out seeds 916/917:
        /// robots stood at their own blocked door for 106 to 124 s. Never open yard: the wall keeps the robot's LiDAR
        /// localisation anchored (held-out run 2: robot B lost its position searching an open yard).
        /// </summary>
        Vector2? meetGoal; double meetSince = double.NaN, meetArrived = double.NaN, nextMeet;
        /// <summary>Back into radio contact. The radio reaches 12 m through at most one wall, so in the house the robots were out
        /// of contact for 168 to 184 s of 200 (held-out seeds 919, 920) and each searched blind to the other's map. Out of
        /// contact for rendezvous_after_s, with at least rendezvous_min_unshared_cells of own map the partner has not
        /// acknowledged, the robot walks back to where it last heard the partner: the link reached between there and the
        /// partner, and the partner, out of contact over the same interval, does the same, so no meeting has to be agreed
        /// while apart (the robots cannot talk then). Messages queue while out of contact and carry everything since the
        /// partner's last acknowledgement, so one heard message syncs the maps. Contact ends the meeting; after
        /// rendezvous_hold_s at the spot without contact the robot searches on. The return must still leave time to get out.
        /// Uses only this robot's own memory: no map of the house, no partner position it has not heard.</summary>
        BrainCommand? RendezvousStep(double t, float dt)
        {
            if (c.rendezvous_after_s <= 0 || !contactAt.HasValue || peers.Count == 0) return null;
            if (meetGoal.HasValue)
            {
                if (lastPeerReceived > meetSince)
                {
                    Emit(t, "rendezvous_done", "Radio contact with the partner again: maps exchanged; search on.", "out_of_contact_s", t - meetSince);
                    meetGoal = null; ClearPath(); nextMeet = t + c.rendezvous_after_s; return null;
                }
                if (Vector2.Distance(motion.Pose.position, meetGoal.Value) <= c.return_arrive_m || (path.Count == 0 && !double.IsNaN(meetArrived)))
                {
                    if (double.IsNaN(meetArrived)) { meetArrived = t; ClearPath(); }
                    if (t - meetArrived < c.rendezvous_hold_s) return Command(0, 0, dt, true);
                    Emit(t, "rendezvous_done", "No contact where the partner was last heard; search on.", "waited_s", t - meetArrived);
                    meetGoal = null; nextMeet = t + c.rendezvous_after_s; return null;
                }
                if (path.Count == 0 || !planner.PathSafe(path, cursor))
                {
                    var again = planner.Plan(motion.Pose.position, meetGoal.Value);
                    if (again.Count == 0)
                    {
                        Emit(t, "rendezvous_done", "The way back to where the partner was last heard is closed; search on.");
                        meetGoal = null; nextMeet = t + c.rendezvous_after_s; return null;
                    }
                    SetPath(again); lastPlan = t;
                }
                return FollowOrHold(t, dt);
            }
            if (t - lastPeerReceived < c.rendezvous_after_s || t < nextMeet || sweeping || AgentGoalActive(t) || active != null ||
                facing.HasValue || t < agentHoldUntil) return null;
            int unshared = UnsharedCells();
            if (unshared < c.rendezvous_min_unshared_cells) return null;
            var route = planner.Plan(motion.Pose.position, contactAt.Value);
            float routeM = 0; for (int i = 1; i < route.Count; i++) routeM += Vector2.Distance(route[i - 1], route[i]);
            float outM = Vector2.Distance(contactAt.Value, home);
            if (route.Count == 0 || t + (routeM + outM) / c.open_speed_mps * c.escape_time_factor + c.rendezvous_hold_s >= c.mission_time_limit_s)
            { nextMeet = t + c.escape_check_s; return null; }
            meetGoal = contactAt; meetSince = lastPeerReceived; meetArrived = double.NaN; frontierGoal = false;
            SetPath(route); lastPlan = t;
            Emit(t, "rendezvous", "Out of radio contact with much of the map unshared: walk back to where the partner was last heard; it does the same.",
                "x_m", contactAt.Value.x, "y_m", contactAt.Value.y, "out_of_contact_s", t - lastPeerReceived, "unshared_cells", unshared, "route_m", routeM);
            return FollowOrHold(t, dt);
        }

        BrainCommand? RelocateStep(double t, float dt)
        {
            if (!canReenter || c.exterior_dwell_s <= 0 || t >= c.mission_time_limit_s) return null;
            if (relocateGoal.HasValue)
            {
                if (Vector2.Distance(motion.Pose.position, relocateGoal.Value) <= c.goal_tolerance_m || path.Count == 0 || !planner.PathSafe(path, cursor))
                {
                    visited.Add(new Visited { position = relocateGoal.Value, radius = c.visited_radius_m, until = t + c.visited_hold_s });
                    Emit(t, "relocate_done", "At the wall: one re-entry check here, then on along the wall.", "x_m", motion.Pose.position.x, "y_m", motion.Pose.position.y);
                    // One re-entry check (escape_check_s) at each stop, not another full dwell: held-out seed 920, robot A
                    // moved 0.6 m in its first relocation and would then have waited 15 s again before each next step.
                    relocateGoal = null; ClearPath(); completedAt = t - c.exterior_dwell_s + c.escape_check_s;
                    return null;
                }
                return FollowOrHold(t, dt);
            }
            if (t - completedAt < c.exterior_dwell_s) return null;
            grid.FireClearance = c.fire_clearance_m; grid.FirePrediction = true;
            // Onward only: a goal farther from where the robot first waited than it is now, so it keeps one direction along
            // the wall rather than stepping back and forth beside the blocked door (seed 920, robot A: the nearest edge was
            // 0.6 m from where it stood).
            var from = relocateFrom ?? motion.Pose.position; float here = Vector2.Distance(motion.Pose.position, from);
            var options = planner.Frontiers(motion.Pose.position, p => InSearchArea(p) || VisitedGoal(p, t) || GoalBlacklisted(p, t) ||
                grid.ObstacleDistance(grid.Cell(p)) > c.facade_band_m || Vector2.Distance(p, from) <= here + c.visited_radius_m);
            if (options.Count == 0) { completedAt = t; return null; }
            var f = options[0];
            SetPath(f.path); relocateGoal = f.position; relocateFrom = from; lastPlan = t;
            Emit(t, "relocate", "Every way in is still blocked after waiting outside: follow the building's wall to look for another way in.",
                "x_m", f.position.x, "y_m", f.position.y, "waited_s", t - completedAt);
            return FollowOrHold(t, dt);
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
            TryEmergencyExit(t, "Heat budget nearly spent: leave by the nearest way out");
        }
        /// <summary>Plan the nearest way out at the normal fire clearance, else at emergency_fire_clearance_m (past the fire,
        /// never through a burning cell), and leave by it; false if there is none. The normal route comes first: the
        /// retreat's "coolest reachable cell" can be a hot spot next to the robot (2x2 attempt 4: robot A sat 0.2 m from its
        /// retreat goal with the budget at 0 and a route out open).</summary>
        bool TryEmergencyExit(double t, string why)
        {
            var exit = PlanOut(); float clearance = c.fire_clearance_m;
            if (exit.Count == 0)
            {
                // Leaving now: only measured fire closes a cell (OccupancyGrid.FirePrediction).
                grid.FireClearance = clearance = c.emergency_fire_clearance_m; grid.FirePrediction = false;
                exit = PlanOut();
            }
            if (exit.Count == 0)
            {
                grid.FireClearance = clearance = c.last_resort_fire_clearance_m;
                exit = PlanOut();
                if (exit.Count == 0) { grid.FireClearance = c.fire_clearance_m; grid.FirePrediction = true; return false; }
            }
            emergencyExit = true; retreating = false; cutOff = false;
            Emit(t, "emergency_exit", why + (clearance < c.fire_clearance_m ? ", past the fire at reduced clearance." : "."),
                "heat_budget", budget, "clearance_m", clearance, "route_points", exit.Count, "air_c", airC);
            BeginReturn(t, EmergencyExit);
            return true;
        }
        // The robot's own thermocouple readings, oldest first, kept for HeatBuilding.
        readonly List<AirSample> airHistory = new List<AirSample>();
        // Set when the brain finds no reachable frontier (it would return after its scan unless an agent orders it on);
        // cleared when it finds one again. cutOff: the way out was lost and the robot retreats to cool floor.
        double noFrontierAt = double.NaN, nextCutOffCheck; bool cutOff;
        const string NoFrontierLeft = "No reachable frontier remains after a full scan.";
        // Left once because heat was building (a heat return or an emergency exit), and when.
        bool burned; double heatExitAt = double.NegativeInfinity;
        const string HeatLeave = "No reachable frontier was left and the air was heating.";
        const string MissionTimeUp = "Mission time is nearly up: leave now to be out by the deadline.";
        /// <summary>
        /// The hot gas layer is reaching the robot: its own air reading is at least air_warm_c, or it rises at
        /// air_rise_c_per_min or faster (least squares over air_rise_window_s) and is at least air_rise_min_delta_c above
        /// its lowest reading of the last air_rise_baseline_s. Used only when the robot is already in trouble (no reachable
        /// frontier left, or cut off from its way out): on its own it would fire in 15 of 33 recorded robot runs that
        /// stayed safe, because robots walk from the yard into halls already at 25 to 45 C (air_rise check, 2026-10-21).
        /// No fresh reading: no decision.
        /// </summary>
        bool HeatBuilding(double t, out float risePerMin)
        {
            risePerMin = AirSlopePerMin(t);
            if (t - lastAir > c.frame_max_age_s) return false;
            if (airC >= c.air_warm_c) return true;
            if (risePerMin < c.air_rise_c_per_min) return false;
            float low = float.PositiveInfinity;
            foreach (var sample in airHistory) if (sample.t >= t - c.air_rise_baseline_s && sample.c < low) low = sample.c;
            return airC >= low + c.air_rise_min_delta_c;
        }
        float AirSlopePerMin(double t)
        {
            double from = t - c.air_rise_window_s, n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0, first = double.NaN;
            foreach (var sample in airHistory)
            {
                if (sample.t < from || sample.t > t) continue;
                if (double.IsNaN(first)) first = sample.t;
                double x = sample.t - t; n++; sx += x; sy += sample.c; sxx += x * x; sxy += x * sample.c;
            }
            // Most of the window must be covered: two seconds of readings after a gap are not a trend.
            if (n < 5 || t - first < .8 * c.air_rise_window_s) return 0;
            double den = n * sxx - sx * sx;
            return den <= 1e-9 ? 0 : (float)((n * sxy - sx * sy) / den * 60);
        }
        /// <summary>
        /// Held-out run 2, robot A: at 89.8 s the brain found no reachable frontier (fire blocked them) and would have
        /// returned after its scan; the agent ordered it on, then on again at 106.9 s. From 95 s its air rose 24 to 39 C
        /// (about 45 C/min) while it inspected warm shapes beside the fire; its way out closed at 114 s, it retreated
        /// "to search on while it is cool", the hot gas reached it at 123 s and it took its emergency exit only at 126 s
        /// (299 C, budget 0.08): 77 s above 60 C, peak 381 C. Now: with no reachable frontier left, heat building ends
        /// the scan or any agent order to stay, and the robot leaves while its way out is open (it may come back in
        /// later). A robot cut off from its way out leaves at once by the emergency route when heat builds, instead of
        /// retreating; in cool, steady air it still retreats and searches on.
        /// </summary>
        void CheckHeatBuilding(double t)
        {
            if (completed || emergencyExit) return;
            // Cut off (the known way out closed, no emergency route yet), retreating or searching on in cooler air: a
            // normal return would only find the way out closed again. Held-out seed 914 run 4, robot B: with its air in the
            // 30 to 35 C band (heat building from air_warm_c, yet below heat_recover_below_c, so each retreat ended at once)
            // it started 360 returns in 45 s. Retry the emergency exit on the check interval instead.
            if (cutOff)
            {
                if (t < nextCutOffCheck) return;
                nextCutOffCheck = t + c.escape_check_s;
                if (HeatBuilding(t, out float cutRise) && !TryEmergencyExit(t, "Cut off from the way out and the air is heating: leave now by the nearest way out")
                    && !retreating && airC >= c.heat_recover_below_c)
                {
                    // No route out even past the fire and the air is heating: never stand in it on an agent's order (one-door
                    // seed 923, robot A turned on the spot for 30 s on its agent's orders while its air rose from 30 to 70 C).
                    bool agentOrdered = AgentGoalActive(t) || t < agentHoldUntil || facing.HasValue;
                    agentGoal = null; agentGoalIsFrontier = false; agentHoldUntil = double.NegativeInfinity; facing = null; agentFace = null;
                    ResetStuckRecovery(); retreating = true; sweeping = false; active = null; dwell = 0; ClearPath();
                    Emit(t, "stay_overridden", (agentOrdered ? "The agent's order is overridden: " : "") +
                        "cut off with the air heating and no route out even past the fire; retreat to the coolest reachable floor.",
                        "air_c", airC, "rise_c_per_min", cutRise);
                }
                return;
            }
            if (!returning && !retreating && burned && HeatBuilding(t, out float again))
            {
                // Back inside after leaving for heat: the building is known to be heating, so leave at the first sign.
                agentGoal = null; agentGoalIsFrontier = false; agentHoldUntil = double.NegativeInfinity; facing = null; agentFace = null;
                Emit(t, "heat_again", "Back inside after leaving for heat, and the air is heating again: leave while the way out is open.",
                    "air_c", airC, "rise_c_per_min", again);
                BeginReturn(t, HeatLeave);
                return;
            }
            if (returning || retreating || double.IsNaN(noFrontierAt)) return;
            bool ordered = AgentGoalActive(t) || t < agentHoldUntil || facing.HasValue;
            if ((!ordered && double.IsNaN(scanSince)) || !HeatBuilding(t, out float rise)) return;
            agentGoal = null; agentGoalIsFrontier = false; agentHoldUntil = double.NegativeInfinity; facing = null; agentFace = null;
            Emit(t, "stay_overridden", (ordered ? "The agent's order to stay is overridden: " : "") +
                "no reachable frontier is left and the air is heating; leave while the way out is open.",
                "air_c", airC, "rise_c_per_min", rise, "no_frontier_s", t - noFrontierAt);
            BeginReturn(t, HeatLeave);
        }
        /// <summary>Why an agent order to move must be refused now, or null.</summary>
        string AgentMoveRefusal(double t)
        {
            if (returning) return "the robot is leaving (" + returnReason + ")";
            // A cut-off robot cannot leave, so "the robot is leaving" is not a reason to refuse its agent (run 4: robot B's
            // moves toward a confirmed person were refused this way while its way out was closed).
            if (!cutOff && !double.IsNaN(noFrontierAt) && HeatBuilding(t, out float rise))
                return $"no reachable frontier is left and the air is heating ({airC:0} C, {rise:0} C/min); the robot is leaving";
            return null;
        }
        /// <summary>Known ways out: this robot's own entry and, once a partner's map is linked, the partner's entry (the
        /// origin of its frame), which it reported by radio. Slow-fire run: robot B's way home was cut by fire 5.5 m from
        /// the door robot A had entered and left by, and it held still for 50 s.</summary>
        IEnumerable<Vector2> Exits()
        {
            yield return home;
            foreach (var partner in peers.Keys)
                if (PartnerToOwn(partner, Vector2.zero, out var entry)) yield return entry;
            // And ways out this robot found with its own LiDAR (OccupancyGrid.SensedExits). The planner reaches the nearest
            // one, as DARPA SubT's GBPlanner2 plans home over every traversed edge rather than one fixed door: a robot
            // whose entry is cut need not leave if another sensed way out is open (held-out seed 914 run 5: robot A held
            // at 185 C with the front door 5.5 m away, its LiDAR seeing open ground through it).
            foreach (var exit in sensedExits) yield return exit;
            foreach (var exit in walkedIn) yield return exit;
        }
        readonly List<Vector2> walkedIn = new List<Vector2>();
        /// <summary>A route to the last proven-indoor floor on the way to an unproven frontier, when no proven one is left.
        /// Floor seen through an interior doorway is mapped free but not proven indoors until the camera sees its ceiling,
        /// which it does only about 4.7 m away; from the far side of a room it never does. One-door seed 924: both robots
        /// mapped the hallway and the bedrooms behind the north wall through its doorways, never stood in one, and robot A
        /// began its return at 90 s with two of five people in those rooms unsearched. Standing at the threshold facing in
        /// (the route ends heading through it) lets the camera prove or disprove the room. Never from near a known way out,
        /// where the floor beyond is the yard; each frontier and each standpoint is tried once.</summary>
        bool ThresholdLook(double t, out List<Vector2> route, out Vector2 beyond)
        {
            route = null; beyond = default;
            if (!c.search_requires_ceiling) return false;
            var options = planner.Frontiers(motion.Pose.position, p => InSearchArea(p) || GoalBlacklisted(p, t) || VisitedGoal(p, t) ||
                lookedAt.Exists(q => Vector2.Distance(q, p) <= c.stuck_repeat_radius_m));
            for (int k = 0; k < options.Count && k < c.threshold_look_options; k++)
            {
                var f = options[k]; lookedAt.Add(f.position);
                int i = f.path.FindIndex(p => !InSearchArea(p));
                if (i <= 0) continue;   // standing on unproven floor already: nothing to look in from
                var at = f.path[i - 1]; bool nearExit = false;
                foreach (var e in Exits()) nearExit |= Vector2.Distance(e, at) <= c.threshold_exit_clear_m;
                if (nearExit || lookedFrom.Exists(q => Vector2.Distance(q, at) <= c.stuck_repeat_radius_m)) continue;
                lookedFrom.Add(at); route = f.path.GetRange(0, i); beyond = f.path[i];
                return true;
            }
            return false;
        }
        readonly List<Vector2> lookedAt = new List<Vector2>(), lookedFrom = new List<Vector2>();
        List<Vector2> sensedExits = new List<Vector2>(); double nextExitScan;
        void RefreshSensedExits(double t)
        {
            if (t < nextExitScan) return;
            nextExitScan = t + c.escape_check_s;
            var now = grid.SensedExits();
            foreach (var exit in now)
                if (!sensedExits.Exists(e => Vector2.Distance(e, exit) <= c.survivor_merge_m) && Vector2.Distance(exit, home) > c.survivor_merge_m)
                    Emit(t, "exit_found", "The LiDAR sees open ground past this walkable floor in sweep after sweep: a way out.", "x_m", exit.x, "y_m", exit.y);
            sensedExits = now;
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
            if (HeatBuilding(t, out _) && TryEmergencyExit(t, "The known way out is closed and the air is heating: leave now by the nearest way out"))
            {
                Emit(t, "escape_closing", "The known way out is closed and the air is heating: emergency exit instead of a retreat.", "route_points", 0, "air_c", airC);
                return;
            }
            cutOff = true; nextCutOffCheck = t + c.escape_check_s;
            hadWayOut = false; returning = false; ResetStuckRecovery(); ResetWayOutAndArrival(); retreating = true; sweeping = false; active = null; dwell = 0; ClearPath();
            // A route that reappears next to the fire would start a return that is cut again at once; wait a while.
            nextWayOutCheck = t + c.escape_recheck_s;
            Emit(t, "escape_closing", "The known way out is closed; retreat to the coolest reachable floor and search on while it is cool.", "route_points", 0);
        }
        void BeginReturn(double t, string reason)
        {
            ResetStuckRecovery(); ResetWayOutAndArrival(); returning = true; active = null; dwell = 0; returnReason = reason; ClearPath();
            Emit(t, "goal", reason + " Return to the estimated start.", "x_m", home.x, "y_m", home.y);
        }
        /// <summary>Target speed and policy from the nearest obstacle ahead. Near walls the robot halves its speed, except when
        /// it is leaving because of heat or a closing way out: held-out seed 920, robot B walked its 6.5 m way out through a
        /// heating foyer at 0.32 to 0.38 m/s, every doorway "near", and spent 11.8 s in air over 60 C (peak 76.5 C); at
        /// open speed it would have been out by about 154 s, under 55 C. Every command still passes the stopping-distance
        /// cap (SafeForward) and the thermal-mode cap in Command.</summary>
        float PursuitSpeed(float near, out string policy)
        {
            bool fleeing = returning && (emergencyExit || returnReason == WayOutClosing || returnReason == HeatLeave);
            if (near >= c.near_obstacle_m) { policy = "open"; return c.open_speed_mps; }
            if (fleeing) { policy = "escape"; return c.open_speed_mps; }
            policy = "near"; return c.near_speed_mps;
        }
        bool swingBlocked;
        BrainCommand FollowOrHold(double t, float dt, float? arriveM = null)
        {
            if (path.Count == 0 || !goal.HasValue) return Command(0, 0, dt, true);
            Vector2 position = motion.Pose.position;
            while (cursor + 1 < path.Count && Vector2.Distance(position, path[cursor]) < c.goal_tolerance_m) cursor++;
            if (Vector2.Distance(position, goal.Value) < (arriveM ?? c.goal_tolerance_m)) return Command(0, 0, dt, true);
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
            float speed = PursuitSpeed(near, out string policy);
            if (health.Thermal) { speed = Mathf.Min(speed, c.thermal_speed_mps); policy = "thermal"; }
            speed *= Mathf.Clamp01(Vector2.Distance(position, goal.Value) / c.goal_slow_distance_m);
            float yaw = 2 * speed * Mathf.Sin(error * Mathf.Deg2Rad) / Mathf.Max(grid.CellM, Vector2.Distance(position, target)) * Mathf.Rad2Deg;
            if (Mathf.Abs(error) > c.turn_in_place_deg)
            {
                speed = 0; policy = "turn";
                // Brake before rotating, so the acceleration ramp cannot make a turn drift into a wall.
                yaw = commandedSpeed > c.survivor_stop_speed_mps ? 0 : error * c.heading_gain;
                // A turn that would swing the nose or tail into a mapped wall backs straight off first, through the
                // mapped free footprint, then turns (seed 920, robot B at the foyer-kitchen opening).
                if (yaw != 0 && !SwingClear(error))
                {
                    if (!swingBlocked) { swingBlocked = true; Emit(t, "body_swing_blocked", "Turning here would swing the body into a mapped wall; back off first.", "heading_error_deg", error); }
                    float back = Mathf.Min(c.stuck_reverse_mps, c.limits.max_reverse_mps);
                    if (back > 0 && planner.ReverseSafe(position, position - motion.Pose.Forward * c.body_half_length_m))
                        return Command(-back, 0, dt, false);
                }
                else swingBlocked = false;
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
            // The nose: the body is a capsule (body_half_length_m, body_radius_m), its front cap centred this far ahead.
            float nose = Mathf.Max(0, c.body_half_length_m - c.body_radius_m), rn = c.body_radius_m + grid.CellM * .5f;
            Vector2 position = motion.Pose.position, forward = motion.Pose.Forward;
            float reach = Mathf.Max(r, nose + rn) + c.collision_margin_m + c.limits.max_forward_mps * c.limits.max_forward_mps / (2 * c.limits.max_accel_mps2) + grid.CellM;
            int cell = grid.Cell(position); if (cell < 0) return 0;
            int n = Mathf.CeilToInt(reach / grid.CellM), col = cell % grid.Size, row = cell / grid.Size;
            for (int dy = -n; dy <= n; dy++) for (int dx = -n; dx <= n; dx++)
            {
                if (!grid.Inside(col + dx, row + dy)) continue;
                int i = (row + dy) * grid.Size + col + dx; if (grid.Occupancy(i) != 255) continue;
                Vector2 delta = grid.Centre(i) - position;
                float along = Vector2.Dot(delta, forward), lateral = forward.x * delta.y - forward.y * delta.x;
                if (along - nose > 0 && Mathf.Abs(lateral) < rn) best = Mathf.Min(best, along - nose - Mathf.Sqrt(rn * rn - lateral * lateral));
                if (along <= 0 || Mathf.Abs(lateral) >= r) continue;
                best = Mathf.Min(best, along - Mathf.Sqrt(r * r - lateral * lateral));
            }
            return best;
        }

        /// <summary>Whether the body's two end caps stay clear of mapped walls while it turns in place from its heading by up
        /// to swingDeg (signed). The body is a capsule 2 x body_half_length_m long, not round: held-out seed 920, robot B
        /// stalled five times turning at the foyer-kitchen opening, its nose 0.375 m from its centre against the wall
        /// edge, where the 0.25 m planning circle was clear. A pose already touching is not judged (turning may free it).</summary>
        bool SwingClear(float swingDeg)
        {
            float nose = Mathf.Max(0, c.body_half_length_m - c.body_radius_m), h = grid.CellM * .5f, r2 = c.body_radius_m * c.body_radius_m;
            Vector2 at = motion.Pose.position; int cell = grid.Cell(at); if (cell < 0) return true;
            // Mapped wall cells within reach of the ends, tested as squares: an end cap touches a cell when the nearest point
            // of the square lies within body_radius_m of the cap's centre.
            var walls = new List<Vector2>(); int n = Mathf.CeilToInt((c.body_half_length_m + grid.CellM) / grid.CellM), col = cell % grid.Size, row = cell / grid.Size;
            for (int dy = -n; dy <= n; dy++) for (int dx = -n; dx <= n; dx++)
                if (grid.Inside(col + dx, row + dy) && grid.Occupancy((row + dy) * grid.Size + col + dx) == 255) walls.Add(grid.Centre((row + dy) * grid.Size + col + dx));
            bool Clear(float yaw)
            {
                var f = new Vector2(-Mathf.Sin(yaw), Mathf.Cos(yaw));            // BrainPose.Forward
                foreach (var end in new[] { at + f * nose, at - f * nose })
                    foreach (var w in walls)
                    {
                        float ex = Mathf.Max(Mathf.Abs(end.x - w.x) - h, 0), ey = Mathf.Max(Mathf.Abs(end.y - w.y) - h, 0);
                        if (ex * ex + ey * ey < r2) return false;
                    }
                return true;
            }
            float yaw0 = motion.Pose.yawRad;
            if (!Clear(yaw0)) return true;
            int steps = Mathf.Max(1, Mathf.CeilToInt(Mathf.Abs(swingDeg) / 5f));
            for (int i = 1; i <= steps; i++) if (!Clear(yaw0 + swingDeg * Mathf.Deg2Rad * i / steps)) return false;
            return true;
        }

        /// <summary>Highest forward speed that can still stop collision_margin_m short of the swept clearance at the
        /// acceleration limit, counting the capsule's nose. Turning is checked separately (SwingClear).</summary>
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
        /// <summary>The live 3D map's top-height layer (centimetres per cell, this brain's grid), from the agent service.</summary>
        public void OnTopHeights(ushort[] cm) => grid.SetTopHeights(cm);
        public bool InSearchArea(Vector2 p) => !c.search_requires_ceiling || grid.Indoors(p);
        /// <summary>Seconds until the robot's operating limit (mission_time_limit_s), for the agent.</summary>
        public double MissionTimeLeft(double t) => Math.Max(0, c.mission_time_limit_s - t);
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
            peerExplored = grid.CopyPeerExplored(), searched = grid.CopySearched(), ownSearched = grid.CopyOwnSearched(), peers = CopyPeers(), survivors = CopySurvivors()
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
