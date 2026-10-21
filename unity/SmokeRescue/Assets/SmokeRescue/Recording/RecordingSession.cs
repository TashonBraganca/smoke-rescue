using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using SmokeRescue.Brain;
using SmokeRescue.Oracle;
using SmokeRescue.Radio;
using SmokeRescue.Robots;
using SmokeRescue.Sensors;
using SmokeRescue.World;
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;
using Debug = UnityEngine.Debug;

namespace SmokeRescue.Recording
{
    /// <summary>
    /// Recording mode on a fixed simulation clock (SIMULATION_SPEC 5, BUILD_ORDER C5). The clock runs at 600 Hz:
    /// robots step every tick, IMU samples every 3rd tick (200 Hz), odometry every 12th (50 Hz), cameras every
    /// 40th (15 Hz), radio every broadcast period. Rendering is synchronous, so the result does not depend on
    /// wall-clock speed: two runs with the same seed write identical text files.
    /// Output layout: docs/INTERFACES.md section 3. The house-frame conversion happens here, once.
    /// </summary>
    public static partial class RecordingSession
    {
        public const int ClockHz = 600;

        public sealed class Result
        {
            public string dir;
            public double simSeconds, wallSeconds;
            public Dictionary<string, int> frames = new Dictionary<string, int>();
            public int radioLines, truthPoints;
        }

        partial class RobotRig
        {
            public Robot robot;
            public RgbdCamera rgbd;
            public ThermalCamera thermal;
            public AirTemperatureSensor air;
            public LidarSensor lidar;
            public LidarScan lastScan;
            public StringBuilder lidarTxt, mapTxt = new StringBuilder("# brain occupancy snapshots, 8-bit PNG, 0 occupied, 128 unknown, 255 free; row 0 is the grid's south edge\n# t x y yaw_rad file\n");
            public RobotBrain brain;
            public BrainCommand command;
            public bool missionDone;
            public StringBuilder decisions = new StringBuilder();
            public StringBuilder brainState = new StringBuilder();
            public StringBuilder thermalTxt = Header("thermal images, 16-bit PNG, kelvin x 10, 0 = invalid");
            public StringBuilder airTxt = new StringBuilder("# body air temperature, thermocouple with lag and noise\n# t temperature_c\n");
            public ImuSensor imu;
            public OdometrySensor odom;
            public string dir;
            public StringBuilder rgb = Header("color images"), depth = Header("depth images, 16-bit PNG, millimetres, 0 = invalid");
            public StringBuilder assoc = new StringBuilder("# associated RGB-D frames\n# Smoke Rescue shared simulation clock\n# t_rgb rgb_path t_depth depth_path\n");
            public StringBuilder imuTxt = new StringBuilder("# IMU, body frame (x right, y down, z forward), m/s^2 and rad/s, noise included\n# t ax ay az gx gy gz\n");
            public StringBuilder odomTxt = new StringBuilder("# leg odometry, body frame, m/s and rad/s, scale error and slip included\n# t vx vy wz\n");
            public int frames;
            static StringBuilder Header(string what) => new StringBuilder($"# {what}\n# Smoke Rescue shared simulation clock\n# timestamp filename\n");
        }

        const int YieldEveryS = 5;
        public static Result Record(string scenarioFile, string runId, float maxSeconds = float.PositiveInfinity, bool noise = true, Action<string> progress = null)
        {
            Result last = null;
            foreach (var step in RecordSteps(scenarioFile, runId, maxSeconds, noise, progress)) if (step != null) last = step;
            return last;
        }

        /// <summary>
        /// The recording as steps: null every YieldEveryS simulated seconds, then the result. The editor menu runs one step per
        /// editor update, so the editor completes a frame every simulated second. Run as one call, a 200 s recording was a
        /// single 20-minute editor frame, and macOS frees each render's Metal objects only when a frame ends: the editor's
        /// footprint grew about 1 GB a minute (13 GB to 21 GB in 10 minutes of run 15) until the system killed it
        /// (OS_REASON_JETSAM at 11:17:42 in run 14, with about 40 other processes). The simulation is the same either way.
        /// </summary>
        public static IEnumerable<Result> RecordSteps(string scenarioFile, string runId, float maxSeconds = float.PositiveInfinity, bool noise = true, Action<string> progress = null)
        {
            var wall = Stopwatch.StartNew();
            var plan = FloorPlan.Load();
            var scenario = Scenario.Load(scenarioFile);
            var sensorCfg = SensorConfig.Load();
            PeoplePeriodS = BrainConfig.Load().person_rgb_scan_period_s;
            string root = Path.GetFullPath(Path.Combine(FloorPlan.ConfigDir, "..", "data", "recordings", runId));
            if (Directory.Exists(root))
            {
                // Move the old run aside first: a recursive delete of thousands of files can race with macOS
                // (Spotlight, .DS_Store) and throw "Directory not empty", which aborted a full recording.
                string trash = root + ".old_" + DateTime.Now.Ticks;
                Directory.Move(root, trash);
                try { Directory.Delete(trash, true); } catch (IOException e) { Debug.LogWarning($"[SmokeRescue] could not delete {trash}: {e.Message}"); }
            }
            Directory.CreateDirectory(root);

            var atmosphere = Atmosphere.Build(plan, scenario);
            var oldRobots = GameObject.Find("Robots");
            if (oldRobots != null) UnityEngine.Object.DestroyImmediate(oldRobots);
            var robotsRoot = new GameObject("Robots");
            var rigs = new List<RobotRig>();
            for (int i = 0; i < scenario.robots.Count; i++)
            {
                var spec = scenario.robots[i];
                var rig = new RobotRig { robot = Robot.Create(spec, scenario.look_around, plan, robotsRoot.transform) };
                var camGo = new GameObject("SensorCamera");
                camGo.transform.SetParent(rig.robot.CameraMount, false);
                camGo.AddComponent<Camera>();
                camGo.AddComponent<HDAdditionalCameraData>();
                rig.rgbd = camGo.AddComponent<RgbdCamera>();
                rig.rgbd.Initialize(sensorCfg, scenario.seed * 1000 + i * 10 + 1);
                rig.rgbd.NoiseEnabled = noise;
                rig.rgbd.LocalDensity = atmosphere.Field.Density;
                if (sensorCfg.thermal != null)
                {
                    var thGo = new GameObject("ThermalCamera");
                    thGo.transform.SetParent(rig.robot.CameraMount, false);
                    thGo.transform.localPosition = new Vector3(sensorCfg.thermal.mount_right_offset_m, 0f, 0f);
                    thGo.AddComponent<Camera>();
                    thGo.AddComponent<HDAdditionalCameraData>();
                    rig.thermal = thGo.AddComponent<ThermalCamera>();
                    rig.thermal.Initialize(sensorCfg.thermal, scenario.seed * 1000 + i * 10 + 4);
                    rig.thermal.NoiseEnabled = noise;
                }
                if (sensorCfg.air_temperature != null)
                    rig.air = new AirTemperatureSensor(sensorCfg.air_temperature, scenario.seed * 1000 + i * 10 + 5) { NoiseEnabled = noise };
                if (sensorCfg.lidar != null)
                {
                    rig.lidar = new LidarSensor(sensorCfg.lidar, scenario.seed * 1000 + i * 10 + 7)
                    {
                        NoiseEnabled = noise, LocalDensity = atmosphere.Field.Density, Self = rig.robot.transform,
                        Mask = LayerMask.GetMask("Walls", "Props", "Robots"),
                    };
                    var ci = CultureInfo.InvariantCulture; var l = sensorCfg.lidar;
                    rig.lidarTxt = new StringBuilder("# planar LiDAR, one sweep per line, ranges in millimetres, 0 = no return\n" +
                        "# body frame: beam i at angle_min_rad + i * increment_rad, 0 = forward, positive = left (counter-clockwise seen from above)\n" +
                        $"# beams {l.beams} angle_min_rad {rig.lidar.AngleMinRad.ToString("F6", ci)} increment_rad {rig.lidar.AngleIncrementRad.ToString("F6", ci)} " +
                        $"mount_height_m {l.mount.height_m.ToString("F3", ci)} mount_forward_m {l.mount.forward_offset_m.ToString("F3", ci)} min_m {l.min_m.ToString("F3", ci)} max_m {l.max_m.ToString("F3", ci)}\n" +
                        "# t r0 r1 ...\n");
                }
                rig.imu = new ImuSensor(sensorCfg, scenario.seed * 1000 + i * 10 + 2) { NoiseEnabled = noise };
                rig.odom = new OdometrySensor(sensorCfg, scenario.seed * 1000 + i * 10 + 3) { NoiseEnabled = noise };
                rig.imu.Reset(rig.robot.Body.position, rig.robot.Body.rotation);
                rig.odom.Reset(rig.robot.transform.position, rig.robot.transform.rotation);
                if (!string.IsNullOrEmpty(spec.torch) && spec.torch != "wide") rig.robot.SetTorch(spec.torch, 0f, 0f);
                rig.dir = Path.Combine(root, spec.id);
                if (spec.control == "autonomous")
                {
                    // The robot drives itself: its brain sees only this robot's sensors (fed below at their own rates)
                    // and returns forward speed and yaw rate. Brain yaw is left-positive; Unity yaw is right-positive.
                    // No prior knowledge: each brain starts at the origin of its own frame, facing its own +y. Frames are
                    // linked only from matched LiDAR fingerprints (FrameLink); nobody tells a robot where it is.
                    var brainCfg = BrainConfig.Load();
                    rig.brain = brainCfg.frames_shared_at_start
                        ? new RobotBrain(sensorCfg, brainCfg, spec.id, scenario.seed * 1000 + i * 10 + 6, new EntryPose(spec.spawn.at[0], spec.spawn.at[1], spec.spawn.yaw))
                        : new RobotBrain(sensorCfg, brainCfg, spec.id, scenario.seed * 1000 + i * 10 + 6);
                    rig.brain.AgentConfirms = AgentUrl != null;
                    var r0 = rig;
                    rig.robot.Driver.Command = () => (r0.command.forwardMps, -r0.command.yawRateDeg, r0.missionDone);
                }
                Directory.CreateDirectory(Path.Combine(rig.dir, "rgb"));
                Directory.CreateDirectory(Path.Combine(rig.dir, "depth"));
                if (rig.thermal != null) Directory.CreateDirectory(Path.Combine(rig.dir, "thermal"));
                rigs.Add(rig);
            }

            var oracle = new OracleWriter(root);
            var radio = new RadioModel(scenario.radio.range_m, scenario.radio.max_walls);
            var radioLog = new RadioLog();
            // Live shared context between brains. The recorder alone decides delivery with the RadioModel. While a
            // link is down only the newest state waits: it carries everything since the receiver's last ACK.
            var waiting = new Dictionary<(RobotRig from, RobotRig to), (byte[] bytes, BrainMessage m)>();
            var inFlight = new List<(double at, RobotRig to, byte[] bytes)>();
            var brainRadio = new StringBuilder("# brain state radio, one line per broadcast and receiver with a message to carry\n" +
                                               "# t_send from to sequence bytes full status t_deliver\n");
            var pending = new Queue<Task>();
            var world = AgentSetup(plan, scenario, rigs);
            // Water from a robot's nozzle reaches the fire simulation at the moment it starts to flow.
            foreach (var rig in rigs)
            {
                var id = rig.robot.RobotId; var r0 = rig;
                rig.actions.GripCompleted += who =>
                {
                    r0.brain?.BeginCarry(SimNow, world.rescue.carry_speed_mps, who);
                    world.rescueLog.Append(new JObject { ["t"] = Math.Round(SimNow, 3), ["robot"] = id, ["event"] = "gripped", ["person"] = who }.ToString(Formatting.None)).Append('\n');
                };
                rig.actions.SprayStarted += (nozzleAt, headingRad, flow, seconds) =>
                {
                    atmosphere.FireSpread?.ScheduleWater(atmosphere.FireSpread.FireTime((float)SimNow), new Vector2(nozzleAt.x, nozzleAt.z),
                        headingRad, world.nozzle.range_m, world.nozzle.half_angle_deg, flow, seconds);
                    world.waterLog.Append(new JObject { ["t"] = Math.Round(SimNow, 3), ["robot"] = id, ["x"] = Math.Round(nozzleAt.x, 2),
                        ["z"] = Math.Round(nozzleAt.z, 2), ["heading_deg"] = Math.Round(headingRad * Mathf.Rad2Deg, 1),
                        ["flow_lps"] = flow, ["seconds"] = Math.Round(seconds, 2), ["fire"] = atmosphere.FireSpread != null }.ToString(Formatting.None)).Append('\n');
                };
            }
            int camEvery = Mathf.RoundToInt(ClockHz / sensorCfg.camera.rate_hz);
            int radioEvery = Mathf.RoundToInt(ClockHz * scenario.radio.broadcast_period_s);
            int brainEvery = Mathf.RoundToInt(ClockHz / 60f), stateEvery = Mathf.RoundToInt(ClockHz / 2f);
            int airEvery = Mathf.RoundToInt(ClockHz / (sensorCfg.air_temperature?.rate_hz ?? 10f));
            const float dt = 1f / ClockHz;
            float duration = Mathf.Min(scenario.duration_s, maxSeconds);
            double finishedAt = double.NaN;
            var result = new Result { dir = root };

            Physics.SyncTransforms();
            var perf = new StagePerf();
            bool truthExported = false;
            int rewindsBefore = FireSimHeat.Rewinds;
            perf.Counter("fire_rewinds", () => FireSimHeat.Rewinds - rewindsBefore);
            for (long k = 0; ; k++)
            {
                double t = k / (double)ClockHz;
                if (t > duration + 1e-9) break;
                // Let the editor finish a frame (see RecordSteps). Every 5 simulated seconds: about 0.5 GB of render objects
                // build up in between, and each editor frame costs a scene repaint (yielding every second, run 16 took 3 times as
                // long per simulated second).
                if (k > 0 && k % (ClockHz * YieldEveryS) == 0) yield return null;
                SimNow = t;
                // Every reader of the smoke field sees this tick's time. It was set only at camera frames, so the LiDAR
                // asked the fire simulation for a time older than the thermocouple had, and each such request reloaded
                // the fire and replayed it from t = 0 (perf.txt: LiDAR smoke lookups 15 to 32 s per 10 sim s, growing).
                atmosphere.Field.Time = (float)t;
                if (k > 0 && k % (ClockHz * CheckpointS) == 0) Checkpoint(t, false);
                if (k > 0 && k % (ClockHz * 10) == 0)
                {
                    perf.Line(t, ("rgbd_noise", RgbdNoise.ProcessTime), ("lidar_sync", LidarSensor.SyncTime),
                        ("lidar_rays", LidarSensor.RayTime), ("lidar_smoke", LidarSensor.SmokeTime));
                    File.WriteAllText(Path.Combine(root, "perf.txt"), perf.ToString());
                }
                using (perf.Time("physics"))
                {
                    if (k > 0)
                        foreach (var rig in rigs) rig.robot.Step(dt);
                    if (k > 0) AgentWorldStep(world, rigs, dt);
                }
                bool allDone = rigs.All(r => r.robot.Driver.Finished);
                if (allDone && double.IsNaN(finishedAt)) finishedAt = t;
                if (allDone && t - finishedAt >= 1.0) break;   // one second of standing still at the end

                if (k > 0)
                    foreach (var rig in rigs)
                    {
                        string ts = Fmt(t);
                        var s = rig.imu.Step(t, rig.robot.Body.position, rig.robot.Body.rotation, dt);
                        if (s.HasValue) rig.brain?.OnImu(s.Value);
                        if (s.HasValue)
                            rig.imuTxt.Append(ts).Append(' ').Append(V(s.Value.SpecificForce)).Append(' ').Append(V(s.Value.AngularRate)).Append('\n');
                        var o = rig.odom.Step(t, rig.robot.transform.position, rig.robot.transform.rotation, dt);
                        if (o.HasValue) rig.brain?.OnOdometry(o.Value);
                        if (o.HasValue)
                            rig.odomTxt.Append(ts).Append(' ').Append(F(o.Value.Vx)).Append(' ').Append(F(o.Value.Vy)).Append(' ').Append(F(o.Value.Wz)).Append('\n');
                        LidarScan scan;
                        using (perf.Time("lidar")) scan = rig.lidar?.Step(t, rig.robot.transform.position, rig.robot.transform.rotation);
                        if (scan != null)
                        {
                            rig.lastScan = scan; rig.brain?.OnLidar(scan);
                            rig.lidarTxt.Append(ts);
                            foreach (float r in scan.RangesM) rig.lidarTxt.Append(' ').Append(Mathf.RoundToInt(r * 1000f));
                            rig.lidarTxt.Append('\n');
                        }
                    }

                if (k % airEvery == 0)
                    foreach (var rig in rigs)
                    {
                        var body = rig.robot.transform.position + Vector3.up * (sensorCfg.air_temperature?.height_m ?? 0.6f);
                        float trueAir = atmosphere.Thermal.AirTemperatureC(body, (float)t);
                        float hottest = HottestSurfaceNear(atmosphere, body, (float)t);
                        oracle.Heat(rig.robot.RobotId, t, trueAir, hottest);
                        var reading = rig.air?.Step(t, trueAir);
                        if (reading.HasValue) rig.brain?.OnAirTemperature(t, reading.Value);
                        if (reading.HasValue) rig.airTxt.Append(Fmt(t)).Append(' ').Append(reading.Value.ToString("F2", CultureInfo.InvariantCulture)).Append('\n');
                    }

                if (k % brainEvery == 0 && inFlight.Count > 0)
                {
                    // Packets whose radio delay has elapsed reach their receiver before its next Step, oldest first.
                    // The receiver hears it at this tick (the brain contract), decoded from the frozen wire bytes.
                    foreach (var p in inFlight.Where(p => p.at <= t).OrderBy(p => p.at).ToList())
                        p.to.brain.OnPeerMessage(BrainMessage.FromBytes(p.bytes), t);
                    inFlight.RemoveAll(p => p.at <= t);
                }

                if (k % brainEvery == 0)
                    foreach (var rig in rigs)
                    {
                        if (rig.brain == null) continue;
                        if (rig.actions != null && rig.actions.IsRunning) continue;   // the arm is working; legs hold
                        // After an arm action the brain resumes with the time that really passed (run 17: the first agent arm
                        // action stopped the recording with "dt must equal elapsed simulation time").
                        float brainDt = double.IsNaN(rig.lastBrainStep) ? brainEvery / (float)ClockHz : (float)(t - rig.lastBrainStep);
                        rig.lastBrainStep = t;
                        using (perf.Time("brain_step")) rig.command = rig.brain.Step(t, brainDt);
                        foreach (var e in rig.brain.DrainEvents())
                        {
                            rig.decisions.Append(JsonConvert.SerializeObject(e)).Append('\n');
                            if (e.kind == "mission_complete") rig.missionDone = true;
                            if (e.kind == "reenter") rig.missionDone = false;   // went back in: its agent decides again
                            if (e.kind == "person_carried_out" && rig.actions?.Carrying != null)
                            {
                                var who = rig.actions.Release(); var at = GameObject.Find(who)?.transform.position ?? Vector3.zero;
                                bool inside = plan.rooms.Any(r => at.x >= r.rect[0] && at.x <= r.rect[2] && at.z >= r.rect[1] && at.z <= r.rect[3]);
                                world.rescueLog.Append(new JObject { ["t"] = Math.Round(t, 3), ["robot"] = rig.robot.RobotId, ["event"] = "released", ["person"] = who,
                                    ["x"] = Math.Round(at.x, 2), ["z"] = Math.Round(at.z, 2), ["outside_building"] = !inside }.ToString(Formatting.None)).Append('\n');
                            }
                            NoteAgentTrigger(rig, e);
                        }
                        if (k % stateEvery == 0) rig.brainState.Append(BrainStateLine(t, rig.brain.Snapshot())).Append('\n');
                        if (k % (ClockHz * MapEveryS) == 0) using (perf.Time("brain_map")) WriteBrainMap(rig, t, pending);
                    }

                if (k % brainEvery == 0 && AgentUrl != null) using (perf.Time("agents")) AgentDecisions(world, rigs, t, pending, sensorCfg, scenario, root);
                if (k % radioEvery == 0) AgentRadio(world, rigs, radio, t);

                if (k % radioEvery == 0)
                    foreach (var from in rigs)
                    {
                        var links = rigs.Where(r => r != from).Select(r =>
                            (to: r, link: radio.Evaluate(from.robot.transform.position, r.robot.transform.position))).ToList();
                        radioLog.Tick(t, from.robot.RobotId, links.Select(x => (x.to.robot.RobotId, x.link)));
                        var message = from.brain?.BuildMessage(t);   // after this tick's Step; null when nothing is new
                        var frozen = message?.ToBytes();              // the sender may not change a packet once sent
                        foreach (var (to, link) in links)
                        {
                            if (to.brain == null) continue;
                            if (message != null) waiting[(from, to)] = (frozen, message);
                            if (!waiting.TryGetValue((from, to), out var w)) continue;
                            var m = w.m;
                            string status = link.delivered ? "sent" : "queued_" + link.reason;
                            if (link.delivered) { inFlight.Add((t + link.delay, to, w.bytes)); waiting.Remove((from, to)); }
                            brainRadio.Append(Fmt(t)).Append(' ').Append(from.robot.RobotId).Append(' ').Append(to.robot.RobotId)
                                .Append(' ').Append(m.Sequence).Append(' ').Append(w.bytes.Length).Append(' ').Append(m.FullSnapshot ? 1 : 0)
                                .Append(' ').Append(status).Append(' ').Append(link.delivered ? Fmt(t + link.delay) : "-1").Append('\n');
                        }
                    }

                if (k % camEvery != 0) continue;
                atmosphere.Apply((float)t);
                string stamp = Fmt(t);
                foreach (var rig in rigs)
                {
                    rig.rgbd.NearFire = atmosphere.NearFire(rig.robot.transform.position);
                    var cam = rig.rgbd.transform;
                    RgbdFrame frame;
                    using (perf.Time("rgbd_capture")) frame = rig.rgbd.CaptureSync(t);
                    using (perf.Time("brain_rgbd")) rig.brain?.OnRgbd(frame);
                    if (rig.frames == 0) WriteOrientationProbe(root, rig.robot.RobotId, rig.rgbd, sensorCfg);
                    oracle.CameraPose(rig.robot.RobotId, t, cam.position, cam.rotation);
                    string rgbRel = $"rgb/{stamp}.png", depthRel = $"depth/{stamp}.png";
                    rig.rgb.Append(stamp).Append(' ').Append(rgbRel).Append('\n');
                    rig.depth.Append(stamp).Append(' ').Append(depthRel).Append('\n');
                    rig.assoc.Append(stamp).Append(' ').Append(rgbRel).Append(' ').Append(stamp).Append(' ').Append(depthRel).Append('\n');
                    string rgbPath = Path.Combine(rig.dir, rgbRel), depthPath = Path.Combine(rig.dir, depthRel);
                    pending.Enqueue(SensorPng.EncodeRgbAsync(frame.Rgb, frame.Width, frame.Height).ContinueWith(x => File.WriteAllBytes(rgbPath, x.Result)));
                    pending.Enqueue(SensorPng.EncodeDepthAsync(frame.Depth, frame.Width, frame.Height).ContinueWith(x => File.WriteAllBytes(depthPath, x.Result)));
                    if (rig.thermal != null)
                    {
                        ThermalFrame th;
                        using (perf.Time("thermal_capture")) th = rig.thermal.CaptureSync(t);
                        using (perf.Time("brain_thermal")) rig.brain?.OnThermal(th);
                        rig.thermalTimes.Add(t);
                        string thRel = $"thermal/{stamp}.png", thPath = Path.Combine(rig.dir, thRel);
                        rig.thermalTxt.Append(stamp).Append(' ').Append(thRel).Append('\n');
                        pending.Enqueue(SensorPng.EncodeDepthAsync(th.DeciKelvin, th.Width, th.Height).ContinueWith(x => File.WriteAllBytes(thPath, x.Result)));
                    }
                    rig.frames++;
                }
                using (perf.Time("people")) PeopleScan(rigs, t, pending, sensorCfg);
                using (perf.Time("pending_io")) while (pending.Count > 48) pending.Dequeue().Wait();
                if (rigs[0].frames % 150 == 0)
                    progress?.Invoke($"t={t:0.0}s frames={rigs[0].frames} wall={wall.Elapsed.TotalSeconds:0}s");
            }
            while (pending.Count > 0) pending.Dequeue().Wait();
            result.simSeconds = rigs.Count > 0 ? (rigs[0].frames - 1) / (double)sensorCfg.camera.rate_hz : 0;

            Checkpoint(result.simSeconds, true);
            result.wallSeconds = wall.Elapsed.TotalSeconds;
            Debug.Log($"[SmokeRescue] Recorded {runId}: {string.Join(", ", result.frames.Select(kv => $"{kv.Key} {kv.Value} frames"))}, " +
                      $"sim {result.simSeconds:0.0} s in {result.wallSeconds:0} s wall, {result.radioLines} radio lines, {result.truthPoints} truth points -> {root}");
            yield return result;
            // Everything a scorer needs, written at the end and every checkpoint_s during the run, so a run that stops
            // early (run 12: the editor quit at 163 s with no crash report and nothing could be scored) is still scored.
            void Checkpoint(double tNow, bool final)
            {
                foreach (var rig in rigs)
                {
                    File.WriteAllText(Path.Combine(rig.dir, "rgb.txt"), rig.rgb.ToString());
                    File.WriteAllText(Path.Combine(rig.dir, "depth.txt"), rig.depth.ToString());
                    File.WriteAllText(Path.Combine(rig.dir, "associations.txt"), rig.assoc.ToString());
                    File.WriteAllText(Path.Combine(rig.dir, "imu.txt"), rig.imuTxt.ToString());
                    File.WriteAllText(Path.Combine(rig.dir, "odom.txt"), rig.odomTxt.ToString());
                    if (rig.lidarTxt != null) File.WriteAllText(Path.Combine(rig.dir, "lidar.txt"), rig.lidarTxt.ToString());
                    if (rig.brain != null) File.WriteAllText(Path.Combine(rig.dir, "brain_map.txt"), rig.mapTxt.ToString());
                    File.WriteAllText(Path.Combine(rig.dir, "camera.yaml"), CameraYaml(sensorCfg));
                    if (rig.thermal != null)
                    {
                        File.WriteAllText(Path.Combine(rig.dir, "thermal.txt"), rig.thermalTxt.ToString());
                        File.WriteAllText(Path.Combine(rig.dir, "thermal.yaml"), ThermalYaml(sensorCfg));
                    }
                    if (rig.air != null) File.WriteAllText(Path.Combine(rig.dir, "air_temp.txt"), rig.airTxt.ToString());
                    if (rig.brain != null)
                    {
                        File.WriteAllText(Path.Combine(rig.dir, "decisions.jsonl"), rig.decisions.ToString());
                        File.WriteAllText(Path.Combine(rig.dir, "brain_state.jsonl"), rig.brainState.ToString());
                        oracle.Collisions(rig.robot.RobotId, rig.robot.Driver.CollisionSteps / (double)ClockHz);
                    }
                    result.frames[rig.robot.RobotId] = rig.frames;
                }
                string radioText = radioLog.ToString();
                File.WriteAllText(Path.Combine(root, "radio.txt"), radioText);
                WriteAgentLogs(world, rigs, root);
                if (rigs.Count(r => r.brain != null) > 1) File.WriteAllText(Path.Combine(root, "brain_radio.txt"), brainRadio.ToString());
                result.radioLines = radioText.Split('\n').Count(l => l.Length > 0 && l[0] != '#');

                if (atmosphere.FireSpread != null)
                {
                    var sim = atmosphere.FireSpread.Sim;
                    oracle.FireEvents(new
                    {
                        head_start_s = atmosphere.FireSpread.HeadStart,
                        note = "times are on the robots' clock (fire time minus head start)",
                        collapses = sim.Collapses.Select(c => new { t = c.Time - atmosphere.FireSpread.HeadStart, c.Room,
                            rect = new[] { c.CellRect.xMin, c.CellRect.yMin, c.CellRect.xMax, c.CellRect.yMax } }).ToArray(),
                        flashover = sim.RoomStates.Where(r => r.FlashoverTime >= 0).Select(r => new { r.Room, t = r.FlashoverTime - atmosphere.FireSpread.HeadStart }).ToArray(),
                    });
                }
                oracle.Finish(scenario, Path.Combine(FloorPlan.ConfigDir, "floorplan.json"));
                // Ceilings stay in: the cameras look 20 degrees above the horizon and map them too.
                if (!truthExported) result.truthPoints = oracle.ExportHouseTruth(new[] { GameObject.Find(HouseBuilder.RootName), GameObject.Find("Furniture") }, layerToSkip: -1);

                var meta = new Dictionary<string, object>
                {
                    ["run_id"] = runId,
                    ["complete"] = final, ["sim_seconds_written"] = Math.Round(tNow, 3),
                    ["scenario"] = scenarioFile,
                    ["seed"] = scenario.seed,
                    ["unity_version"] = Application.unityVersion,
                    ["sim_rate_hz"] = sensorCfg.camera.rate_hz,
                    ["robots"] = rigs.Select(r => r.robot.RobotId).ToArray(),
                    ["smoke_level"] = scenario.smoke.global_haze,
                    ["smoke_rooms"] = scenario.smoke.rooms,
                    ["recorded_offline"] = true,
                    ["clock_hz"] = ClockHz,
                    ["noise"] = noise,
                    ["thermal"] = sensorCfg.thermal != null ? "thermal/<t>.png 16-bit kelvin x 10, 320x256, 15 Hz, see thermal.yaml" : null,
                    ["air_temperature"] = sensorCfg.air_temperature != null ? "air_temp.txt, 10 Hz, C" : null,
                    ["lidar"] = sensorCfg.lidar != null ? $"lidar.txt, planar 360 degree, {sensorCfg.lidar.beams} beams, {sensorCfg.lidar.rate_hz} Hz, {sensorCfg.lidar.mount.height_m} m up" : null,
                    ["fire"] = atmosphere.FireSpread != null ? $"spreading fire (config/fire.json), head start {atmosphere.FireSpread.HeadStart} s" : "static fires",
                    ["camera_mount"] = sensorCfg.camera.mount,
                    ["exposure"] = $"per-sensor deterministic auto exposure, target grey {RgbdCamera.TargetGrey * 255:0}, EV100 {RgbdCamera.ExposureMinEv} to {RgbdCamera.ExposureMaxEv}",
                };
                File.WriteAllText(Path.Combine(root, "meta.json"), JsonConvert.SerializeObject(meta, Formatting.Indented));
                truthExported = true;
            }
        }

        /// <summary>
        /// Ground-truth depth by physics raycast at a grid of pixels in the first frame, written to oracle/ so the
        /// brain can check the PNGs on disk are not flipped or shifted (an image-orientation gate, not a metric).
        /// </summary>
        static void WriteOrientationProbe(string root, string robotId, RgbdCamera sensor, SensorConfig cfg)
        {
            var cam = sensor.Camera;
            var rows = new List<object>();
            for (int v = 30; v < cfg.camera.height; v += 60)
                for (int u = 40; u < cfg.camera.width; u += 80)
                {
                    var ray = cam.ScreenPointToRay(new Vector3(u + 0.5f, cfg.camera.height - 0.5f - v, 0f));
                    if (!Physics.Raycast(ray, out var hit, cfg.depth.max_m * 1.5f, ~LayerMask.GetMask("Robots"))) continue;
                    float z = Vector3.Dot(hit.point - cam.transform.position, cam.transform.forward);
                    rows.Add(new { u, v, z_m = Math.Round(z, 4), collider = hit.collider.name });
                }
            File.WriteAllText(Path.Combine(root, "oracle", $"{robotId}_orientation_probe.json"), JsonConvert.SerializeObject(rows, Formatting.Indented));
        }

        static double SimNow;
        const int MapEveryS = 5;
        const int CheckpointS = 30;   // sim seconds between scoring checkpoints (Checkpoint in Record)
        /// <summary>The brain's own occupancy grid every 5 s, for the watcher and the evaluator (not read by any robot).
        /// EncodeArrayToPNG puts array row 0 at the bottom of the image, which is the grid's south edge, so north is up.</summary>
        static void WriteBrainMap(RobotRig rig, double t, Queue<Task> pending)
        {
            var s = rig.brain.Snapshot();
            var pixels = new byte[s.occupancy.Length];
            for (int i = 0; i < pixels.Length; i++) pixels[i] = s.occupancy[i] == 255 ? (byte)0 : s.occupancy[i] == 1 ? (byte)255 : (byte)128;
            string rel = $"brain_map/{Fmt(t)}.png", path = Path.Combine(rig.dir, rel); int size = s.gridSize;
            Directory.CreateDirectory(Path.Combine(rig.dir, "brain_map"));
            pending.Enqueue(Task.Run(() => File.WriteAllBytes(path, ImageConversion.EncodeArrayToPNG(pixels,
                UnityEngine.Experimental.Rendering.GraphicsFormat.R8_UNorm, (uint)size, (uint)size))));
            var searched = new byte[s.searched.Length];
            for (int i = 0; i < searched.Length; i++) searched[i] = s.searched[i] != 0 ? (byte)255 : (byte)0;
            string searchedPath = Path.Combine(rig.dir, $"brain_map/{Fmt(t)}.searched.png");
            var indoor = rig.brain.IndoorMask();
            // 255 ceiling seen, 128 near a ceiling, 64 walled in on the robot's own map (OccupancyGrid.Enclosed), 0 not known indoors.
            for (int i = 0; i < indoor.Length; i++) indoor[i] = indoor[i] == 2 ? (byte)255 : indoor[i] == 1 ? (byte)128 : indoor[i] == 3 ? (byte)64 : (byte)0;
            string indoorPath = Path.Combine(rig.dir, $"brain_map/{Fmt(t)}.indoor.png");
            pending.Enqueue(Task.Run(() => File.WriteAllBytes(indoorPath, ImageConversion.EncodeArrayToPNG(indoor,
                UnityEngine.Experimental.Rendering.GraphicsFormat.R8_UNorm, (uint)size, (uint)size))));
            pending.Enqueue(Task.Run(() => File.WriteAllBytes(searchedPath, ImageConversion.EncodeArrayToPNG(searched,
                UnityEngine.Experimental.Rendering.GraphicsFormat.R8_UNorm, (uint)size, (uint)size))));
            rig.mapTxt.Append(Fmt(t)).Append(' ').Append(F(s.pose.position.x)).Append(' ').Append(F(s.pose.position.y)).Append(' ')
                .Append(F(s.pose.yawRad)).Append(' ').Append(rel).Append('\n');
            // Also flush the brain logs, so a run without agents can be watched live and a crash keeps them.
            File.WriteAllText(Path.Combine(rig.dir, "brain_map.txt"), rig.mapTxt.ToString());
            File.WriteAllText(Path.Combine(rig.dir, "decisions.jsonl"), rig.decisions.ToString());
            File.WriteAllText(Path.Combine(rig.dir, "brain_state.jsonl"), rig.brainState.ToString());
        }

        static string CameraYaml(SensorConfig c)
        {
            var k = c.camera;
            var ci = CultureInfo.InvariantCulture;
            return string.Join("\n", new[]
            {
                "%YAML:1.0", "---", "File.version: \"1.0\"", "Camera.type: \"PinHole\"", "",
                $"Camera1.fx: {k.fx.ToString("0.000", ci)}", $"Camera1.fy: {k.fy.ToString("0.000", ci)}",
                $"Camera1.cx: {k.cx.ToString("0.0", ci)}", $"Camera1.cy: {k.cy.ToString("0.0", ci)}",
                "Camera1.k1: 0.0", "Camera1.k2: 0.0", "Camera1.p1: 0.0", "Camera1.p2: 0.0", "Camera1.k3: 0.0",
                $"Camera.width: {k.width}", $"Camera.height: {k.height}", $"Camera.fps: {Mathf.RoundToInt(k.rate_hz)}", "Camera.RGB: 1", "",
                "Stereo.ThDepth: 40.0", "Stereo.b: 0.07732", "RGBD.DepthMapFactor: 1000.0", "",
                "ORBextractor.nFeatures: 1250", "ORBextractor.scaleFactor: 1.2", "ORBextractor.nLevels: 8",
                "ORBextractor.iniThFAST: 20", "ORBextractor.minThFAST: 7", "",
                "Viewer.KeyFrameSize: 0.05", "Viewer.KeyFrameLineWidth: 1.0", "Viewer.GraphLineWidth: 0.9", "Viewer.PointSize: 2.0",
                "Viewer.CameraSize: 0.08", "Viewer.CameraLineWidth: 3.0", "Viewer.ViewpointX: 0.0", "Viewer.ViewpointY: -0.7",
                "Viewer.ViewpointZ: -1.8", "Viewer.ViewpointF: 500.0", "",
            });
        }

        /// <summary>The brain's own view, twice a second, for the map and replay: pose, goal, path, mode, state, heat
        /// budget, survivors and fire hotspots, all in the brain's own frame.</summary>
        static string BrainStateLine(double t, BrainSnapshot s) => JsonConvert.SerializeObject(new
        {
            t, x = s.pose.position.x, y = s.pose.position.y, yaw_rad = s.pose.yawRad,
            goal = s.goal.HasValue ? new[] { s.goal.Value.x, s.goal.Value.y } : null,
            path = s.path?.Select(p => new[] { (float)System.Math.Round(p.x, 2), (float)System.Math.Round(p.y, 2) }).ToArray(),
            s.mode, s.state, heat_budget = s.heatBudget,
            survivors = s.confirmedSurvivors?.Select(p => new[] { p.x, p.y }).ToArray(),
            fires = s.fireHotspots?.Select(p => new[] { p.x, p.y }).ToArray(),
        });

        /// <summary>Hottest floor surface within 1 m of a point (for scoring how close a robot came to fire).</summary>
        static float HottestSurfaceNear(Atmosphere atm, Vector3 p, float t)
        {
            float best = float.MinValue;
            for (float dx = -1f; dx <= 1.001f; dx += 0.25f)
            for (float dz = -1f; dz <= 1.001f; dz += 0.25f)
            {
                if (dx * dx + dz * dz > 1.0001f) continue;
                best = Mathf.Max(best, atm.Thermal.Source.Sample(new Vector2(p.x + dx, p.z + dz), t).x);
            }
            return best;
        }

        static string ThermalYaml(SensorConfig c)
        {
            var k = c.thermal;
            var ci = CultureInfo.InvariantCulture;
            return string.Join("\n", new[]
            {
                "%YAML:1.0", "---", "# Long-wave infrared camera, pinhole, no distortion; PNG value = kelvin x 10",
                $"Camera1.fx: {k.fx.ToString("0.000", ci)}", $"Camera1.fy: {k.fy.ToString("0.000", ci)}",
                $"Camera1.cx: {k.cx.ToString("0.0", ci)}", $"Camera1.cy: {k.cy.ToString("0.0", ci)}",
                $"Camera.width: {k.width}", $"Camera.height: {k.height}", $"Camera.fps: {Mathf.RoundToInt(k.rate_hz)}",
                $"T_rgb_thermal.translation_m: [{k.mount_right_offset_m.ToString("0.000", ci)}, 0.0, 0.0]",
                $"NETD_K: {k.netd_k.ToString("0.000", ci)}", $"Range_max_C: {k.range_max_c.ToString("0", ci)}", "",
            });
        }

        static string Fmt(double t) => t.ToString("F6", CultureInfo.InvariantCulture);
        static string F(float v) => v.ToString("F6", CultureInfo.InvariantCulture);
        static string V(Vector3 v) => $"{F(v.x)} {F(v.y)} {F(v.z)}";
    }
}
