using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
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
    public static class RecordingSession
    {
        public const int ClockHz = 600;

        public sealed class Result
        {
            public string dir;
            public double simSeconds, wallSeconds;
            public Dictionary<string, int> frames = new Dictionary<string, int>();
            public int radioLines, truthPoints;
        }

        class RobotRig
        {
            public Robot robot;
            public RgbdCamera rgbd;
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

        public static Result Record(string scenarioFile, string runId, float maxSeconds = float.PositiveInfinity, bool noise = true, Action<string> progress = null)
        {
            var wall = Stopwatch.StartNew();
            var plan = FloorPlan.Load();
            var scenario = Scenario.Load(scenarioFile);
            var sensorCfg = SensorConfig.Load();
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
                rig.imu = new ImuSensor(sensorCfg, scenario.seed * 1000 + i * 10 + 2) { NoiseEnabled = noise };
                rig.odom = new OdometrySensor(sensorCfg, scenario.seed * 1000 + i * 10 + 3) { NoiseEnabled = noise };
                rig.imu.Reset(rig.robot.Body.position, rig.robot.Body.rotation);
                rig.odom.Reset(rig.robot.transform.position, rig.robot.transform.rotation);
                rig.dir = Path.Combine(root, spec.id);
                Directory.CreateDirectory(Path.Combine(rig.dir, "rgb"));
                Directory.CreateDirectory(Path.Combine(rig.dir, "depth"));
                rigs.Add(rig);
            }

            var oracle = new OracleWriter(root);
            var radio = new RadioModel(scenario.radio.range_m, scenario.radio.max_walls);
            var radioLog = new RadioLog();
            var pending = new Queue<Task>();
            int camEvery = Mathf.RoundToInt(ClockHz / sensorCfg.camera.rate_hz);
            int radioEvery = Mathf.RoundToInt(ClockHz * scenario.radio.broadcast_period_s);
            const float dt = 1f / ClockHz;
            float duration = Mathf.Min(scenario.duration_s, maxSeconds);
            double finishedAt = double.NaN;
            var result = new Result { dir = root };

            Physics.SyncTransforms();
            for (long k = 0; ; k++)
            {
                double t = k / (double)ClockHz;
                if (t > duration + 1e-9) break;
                if (k > 0)
                    foreach (var rig in rigs) rig.robot.Step(dt);
                bool allDone = rigs.All(r => r.robot.Driver.Finished);
                if (allDone && double.IsNaN(finishedAt)) finishedAt = t;
                if (allDone && t - finishedAt >= 1.0) break;   // one second of standing still at the end

                if (k > 0)
                    foreach (var rig in rigs)
                    {
                        string ts = Fmt(t);
                        var s = rig.imu.Step(t, rig.robot.Body.position, rig.robot.Body.rotation, dt);
                        if (s.HasValue)
                            rig.imuTxt.Append(ts).Append(' ').Append(V(s.Value.SpecificForce)).Append(' ').Append(V(s.Value.AngularRate)).Append('\n');
                        var o = rig.odom.Step(t, rig.robot.transform.position, rig.robot.transform.rotation, dt);
                        if (o.HasValue)
                            rig.odomTxt.Append(ts).Append(' ').Append(F(o.Value.Vx)).Append(' ').Append(F(o.Value.Vy)).Append(' ').Append(F(o.Value.Wz)).Append('\n');
                    }

                if (k % radioEvery == 0)
                    foreach (var from in rigs)
                        radioLog.Tick(t, from.robot.RobotId, rigs.Where(r => r != from).Select(r =>
                            (r.robot.RobotId, radio.Evaluate(from.robot.transform.position, r.robot.transform.position))));

                if (k % camEvery != 0) continue;
                atmosphere.Apply((float)t);
                string stamp = Fmt(t);
                foreach (var rig in rigs)
                {
                    rig.rgbd.NearFire = atmosphere.NearFire(rig.robot.transform.position);
                    var cam = rig.rgbd.transform;
                    var frame = rig.rgbd.CaptureSync(t);
                    if (rig.frames == 0) WriteOrientationProbe(root, rig.robot.RobotId, rig.rgbd, sensorCfg);
                    oracle.CameraPose(rig.robot.RobotId, t, cam.position, cam.rotation);
                    string rgbRel = $"rgb/{stamp}.png", depthRel = $"depth/{stamp}.png";
                    rig.rgb.Append(stamp).Append(' ').Append(rgbRel).Append('\n');
                    rig.depth.Append(stamp).Append(' ').Append(depthRel).Append('\n');
                    rig.assoc.Append(stamp).Append(' ').Append(rgbRel).Append(' ').Append(stamp).Append(' ').Append(depthRel).Append('\n');
                    string rgbPath = Path.Combine(rig.dir, rgbRel), depthPath = Path.Combine(rig.dir, depthRel);
                    pending.Enqueue(SensorPng.EncodeRgbAsync(frame.Rgb, frame.Width, frame.Height).ContinueWith(x => File.WriteAllBytes(rgbPath, x.Result)));
                    pending.Enqueue(SensorPng.EncodeDepthAsync(frame.Depth, frame.Width, frame.Height).ContinueWith(x => File.WriteAllBytes(depthPath, x.Result)));
                    rig.frames++;
                }
                while (pending.Count > 48) pending.Dequeue().Wait();
                if (rigs[0].frames % 150 == 0)
                    progress?.Invoke($"t={t:0.0}s frames={rigs[0].frames} wall={wall.Elapsed.TotalSeconds:0}s");
            }
            while (pending.Count > 0) pending.Dequeue().Wait();
            result.simSeconds = rigs.Count > 0 ? (rigs[0].frames - 1) / (double)sensorCfg.camera.rate_hz : 0;

            foreach (var rig in rigs)
            {
                File.WriteAllText(Path.Combine(rig.dir, "rgb.txt"), rig.rgb.ToString());
                File.WriteAllText(Path.Combine(rig.dir, "depth.txt"), rig.depth.ToString());
                File.WriteAllText(Path.Combine(rig.dir, "associations.txt"), rig.assoc.ToString());
                File.WriteAllText(Path.Combine(rig.dir, "imu.txt"), rig.imuTxt.ToString());
                File.WriteAllText(Path.Combine(rig.dir, "odom.txt"), rig.odomTxt.ToString());
                File.WriteAllText(Path.Combine(rig.dir, "camera.yaml"), CameraYaml(sensorCfg));
                result.frames[rig.robot.RobotId] = rig.frames;
            }
            string radioText = radioLog.ToString();
            File.WriteAllText(Path.Combine(root, "radio.txt"), radioText);
            result.radioLines = radioText.Split('\n').Count(l => l.Length > 0 && l[0] != '#');

            oracle.Finish(scenario, Path.Combine(FloorPlan.ConfigDir, "floorplan.json"));
            // Ceilings stay in: the cameras look 20 degrees above the horizon and map them too.
            result.truthPoints = oracle.ExportHouseTruth(new[] { GameObject.Find(HouseBuilder.RootName), GameObject.Find("Furniture") }, layerToSkip: -1);

            var meta = new Dictionary<string, object>
            {
                ["run_id"] = runId,
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
                ["exposure"] = $"per-sensor deterministic auto exposure, target grey {RgbdCamera.TargetGrey * 255:0}, EV100 {RgbdCamera.ExposureMinEv} to {RgbdCamera.ExposureMaxEv}",
            };
            File.WriteAllText(Path.Combine(root, "meta.json"), JsonConvert.SerializeObject(meta, Formatting.Indented));
            result.wallSeconds = wall.Elapsed.TotalSeconds;
            Debug.Log($"[SmokeRescue] Recorded {runId}: {string.Join(", ", result.frames.Select(kv => $"{kv.Key} {kv.Value} frames"))}, " +
                      $"sim {result.simSeconds:0.0} s in {result.wallSeconds:0} s wall, {result.radioLines} radio lines, {result.truthPoints} truth points -> {root}");
            return result;
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

        static string Fmt(double t) => t.ToString("F6", CultureInfo.InvariantCulture);
        static string F(float v) => v.ToString("F6", CultureInfo.InvariantCulture);
        static string V(Vector3 v) => $"{F(v.x)} {F(v.y)} {F(v.z)}";
    }
}
