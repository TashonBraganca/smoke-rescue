using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SmokeRescue.Brain;
using SmokeRescue.Sensors;
using SmokeRescue.World;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    /// <summary>Headless simulator (evaluator side): the real RobotBrain code for every robot, stepped against a plain world file
    /// from tools/headless_world.py, with ray-cast LiDAR and depth, true odometry and gyro, a 2 s radio with the scenario's range
    /// and wall rule, and an idealised colour person detector (a person inside the camera view, past the camera's blind floor,
    /// within person_rgb_max_range_m and with a clear line of sight is reported once per scan period). No rendering, smoke,
    /// fire, thermal camera or agents: it measures search, navigation and team behaviour, not perception. The brains see only
    /// their own sensor data; the world file is never passed to them.</summary>
    public sealed class HeadlessSim
    {
        sealed class Prism { public Vector2[] poly; public float height; }
        sealed class Body { public string id; public RobotBrain brain; public Vector2 p; public float yaw, v, w; public double start, nextLidar, nextSight; public float moved, air = 20, peakAir = 20, hotS, contactS; public AirTemperatureSensor sensor; }
        FireSimHeat heat; Vector2 gateOrigin; float gateYaw;
        /// <summary>House (x, z) of a gate-frame point.</summary>
        Vector2 House(Vector2 g) { Vector2 f = new Vector2(Mathf.Sin(gateYaw), Mathf.Cos(gateYaw)), r = new Vector2(Mathf.Cos(gateYaw), -Mathf.Sin(gateYaw)); return gateOrigin + r * g.x + f * g.y; }
        /// <summary>Air at the thermocouple, by the recorder's formula (ThermalField.AirTemperatureC, reproduced because that class
        /// builds a texture, which a worker thread cannot).</summary>
        float TrueAir(Vector2 g, float t)
        {
            float y = cal.air_temperature?.height_m ?? .6f; var s = heat.Sample(House(g), t);
            float ambient = heat.AmbientC, gas = s.y, layer = s.z, lower = ambient + .25f * Mathf.Max(0f, gas - ambient);
            float air = y >= layer ? gas : Mathf.Lerp(lower, gas, Mathf.Clamp01(1f - (layer - y) / .5f) * .5f);
            return Mathf.Max(air, ambient + .35f * Mathf.Max(0f, s.x - ambient));
        }

        float[][] segs; List<Prism> prisms; float ceiling; List<(string id, Vector2 at)> people; List<Vector2[]> roofs;
        readonly System.Diagnostics.Stopwatch renderClock = new System.Diagnostics.Stopwatch(), brainClock = new System.Diagnostics.Stopwatch();
        SensorConfig cal; BrainConfig cfg;

        static float Cross(Vector2 a, Vector2 b) => a.x * b.y - a.y * b.x;
        // Ray p + t d against segment a-b: the t of the hit, or +inf.
        static float Hit(Vector2 p, Vector2 d, Vector2 a, Vector2 b)
        {
            Vector2 e = b - a; float den = Cross(d, e); if (Mathf.Abs(den) < 1e-9f) return float.PositiveInfinity;
            float t = Cross(a - p, e) / den, u = Cross(a - p, d) / den;
            return t > 1e-5f && u >= 0 && u <= 1 ? t : float.PositiveInfinity;
        }
        static bool Inside(Vector2[] poly, Vector2 q)
        {
            bool inside = false;
            for (int i = 0, j = poly.Length - 1; i < poly.Length; j = i++)
                if ((poly[i].y > q.y) != (poly[j].y > q.y) && q.x < (poly[j].x - poly[i].x) * (q.y - poly[i].y) / (poly[j].y - poly[i].y) + poly[i].x) inside = !inside;
            return inside;
        }
        static Vector2 Fwd(float yaw) => new Vector2(-Mathf.Sin(yaw), Mathf.Cos(yaw));

        /// <summary>Every frontier on a robot's map at the end, with each exclusion the brain applies, for diagnosis.</summary>
        static JArray FrontierDump(RobotBrain b, double t)
        {
            const System.Reflection.BindingFlags F = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
            var planner = (GridPlanner)typeof(RobotBrain).GetField("planner", F).GetValue(b);
            bool Call(string name, Vector2 p) => (bool)typeof(RobotBrain).GetMethod(name, F, null, new[] { typeof(Vector2), typeof(double) }, null).Invoke(b, new object[] { p, t });
            var pos = b.Snapshot().pose.position; var list = new JArray();
            foreach (var f in planner.Frontiers(pos))
                list.Add(new JObject { ["x"] = f.position.x, ["y"] = f.position.y, ["cost"] = f.cost, ["info"] = f.information, ["room"] = f.roomId, ["tier"] = f.roomTier,
                    ["walked_m"] = f.path.Count == 0 ? 0 : f.path.Zip(f.path.Skip(1), (p, q) => Vector2.Distance(p, q)).Sum() + Vector2.Distance(pos, f.path[0]),
                    ["claim"] = Call("ClaimBlocks", f.position), ["blacklisted"] = Call("GoalBlacklisted", f.position), ["visited"] = Call("VisitedGoal", f.position),
                    ["in_search_area"] = b.InSearchArea(f.position) });
            foreach (var probe in Probes)
            {
                var path = planner.Plan(pos, probe);
                list.Add(new JObject { ["probe_x"] = probe.x, ["probe_y"] = probe.y, ["path_points"] = path.Count,
                    ["path_end_m"] = path.Count > 0 ? Vector2.Distance(path[path.Count - 1], probe) : -1, ["in_search_area"] = b.InSearchArea(probe) });
            }
            return list;
        }
        public static List<Vector2> Probes = new List<Vector2>();

        /// <summary>Horizontal ray at height h: nearest wall, lintel or furniture face that spans h, and other robots' bodies.</summary>
        float Cast(Vector2 p, Vector2 d, float h, Body self, List<Body> bodies)
        {
            float best = float.PositiveInfinity;
            foreach (var s in segs) if (s[4] <= h && h <= s[5]) best = Mathf.Min(best, Hit(p, d, new Vector2(s[0], s[1]), new Vector2(s[2], s[3])));
            foreach (var q in prisms) if (q.height >= h) for (int i = 0; i < 4; i++) best = Mathf.Min(best, Hit(p, d, q.poly[i], q.poly[(i + 1) % 4]));
            if (bodies != null)
                foreach (var o in bodies)
                {
                    if (o == self || o.brain == null) continue;
                    foreach (var centre in Circles(o))
                    {
                        Vector2 m = centre - p; float b = Vector2.Dot(m, d), c2 = m.sqrMagnitude - cfg.body_radius_m * cfg.body_radius_m, disc = b * b - c2;
                        if (disc >= 0 && b - Mathf.Sqrt(disc) > 0) best = Mathf.Min(best, b - Mathf.Sqrt(disc));
                    }
                }
            return best;
        }
        /// <summary>robot_collision (request option): robots are solid. Each body is three circles of body_radius_m along its
        /// axis, at the centre and body_half_length_m - body_radius_m fore and aft (a 0.75 m by 0.36 m outline), for the LiDAR
        /// and for contact. Without it a body is one circle the LiDAR sees and the other robot walks through. Headless house
        /// 5015 (23 Oct): both robots ended on the same point, which two real bodies cannot do.</summary>
        bool solidBodies;
        Vector2[] Circles(Body o) => Circles(o.p, o.yaw);
        Vector2[] Circles(Vector2 p, float yaw)
        {
            if (!solidBodies) return new[] { p };
            var f = Fwd(yaw) * Mathf.Max(0, cfg.body_half_length_m - cfg.body_radius_m);
            return new[] { p - f, p, p + f };
        }
        /// <summary>Smallest clearance between this pose's body and any other robot's body (negative: overlapping).</summary>
        float BodyGap(Body self, Vector2 p, float yaw, List<Body> bodies)
        {
            float gap = float.PositiveInfinity;
            foreach (var o in bodies)
            {
                if (o == self || o.brain == null) continue;
                foreach (var a in Circles(p, yaw)) foreach (var b in Circles(o)) gap = Mathf.Min(gap, Vector2.Distance(a, b) - 2 * cfg.body_radius_m);
            }
            return gap;
        }

        LidarScan Lidar(double t, Body r, List<Body> bodies)
        {
            int n = 720; float inc = 2 * Mathf.PI / n, min = -Mathf.PI; var ranges = new float[n];
            for (int i = 0; i < n; i++)
            {
                float range = Cast(r.p, Fwd(r.yaw + min + i * inc), .45f, r, bodies);
                ranges[i] = range <= 12f ? range : float.PositiveInfinity;
            }
            return new LidarScan(t, min, inc, .15f, 12f, 0f, .45f, ranges);
        }

        /// <summary>Optical-z depth in mm at the pixels the occupancy grid reads (every depth_stride-th); others stay 0.</summary>
        RgbdFrame Depth(double t, Body r)
        {
            var cam = cal.camera; int W = cam.width, H = cam.height;
            var depth = new ushort[W * H]; var rgb = new byte[W * H * 3];
            for (int y = 0; y < H; y++) for (int x = 0; x < W; x++)
            { byte b = ((x / 16 + y / 16) % 2 == 0) ? (byte)200 : (byte)40; int k = (y * W + x) * 3; rgb[k] = rgb[k + 1] = rgb[k + 2] = b; }
            float th = cam.mount.pitch_down_deg * Mathf.Deg2Rad, ch = cam.mount.height_m;
            Vector2 f = Fwd(r.yaw), right = new Vector2(f.y, -f.x), p = r.p + f * cam.mount.forward_offset_m;
            int stride = cfg.depth_stride;
            // Rows are independent and the world is read-only: render them on all cores.
            System.Threading.Tasks.Parallel.For(0, (H + stride - 1) / stride, row => { int v = row * stride; for (int u = 0; u < W; u += stride)
            {
                float xn = (u - cam.cx) / cam.fx, yn = (v - cam.cy) / cam.fy;
                Vector2 hd = right * xn + f * (Mathf.Cos(th) - yn * Mathf.Sin(th));
                float vz = -yn * Mathf.Cos(th) - Mathf.Sin(th), best = float.PositiveInfinity;
                // The ceiling exists only over the house's rooms; outside, a ray going up meets sky and returns nothing.
                if (vz < -1e-6f) best = -ch / vz; else if (vz > 1e-6f) { float tc = (ceiling - ch) / vz; var q = p + hd * tc; foreach (var poly in roofs) if (Inside(poly, q)) { best = tc; break; } }
                foreach (var s in segs)
                {
                    float tt = Hit(p, hd, new Vector2(s[0], s[1]), new Vector2(s[2], s[3]));
                    if (tt < best) { float h = ch + vz * tt; if (h >= s[4] && h <= s[5]) best = tt; }
                }
                foreach (var q in prisms)
                {
                    for (int i = 0; i < 4; i++)
                    {
                        float tt = Hit(p, hd, q.poly[i], q.poly[(i + 1) % 4]);
                        if (tt < best) { float h = ch + vz * tt; if (h >= 0 && h <= q.height) best = tt; }
                    }
                    if (vz < -1e-6f) { float tt = (q.height - ch) / vz; if (tt > 0 && tt < best && Inside(q.poly, p + hd * tt)) best = tt; }
                }
                if (best >= cal.depth.min_m && best <= cal.depth.max_m) depth[v * W + u] = (ushort)Mathf.RoundToInt(best * 1000);
            } });
            return new RgbdFrame(t, W, H, rgb, depth);
        }

        bool Collides(Vector2 q)
        {
            float rad = cfg.body_radius_m;
            foreach (var s in segs)
            {
                if (s[4] > .6f) continue;
                Vector2 a = new Vector2(s[0], s[1]), b = new Vector2(s[2], s[3]), e = b - a;
                float k = Mathf.Clamp01(Vector2.Dot(q - a, e) / Mathf.Max(1e-9f, e.sqrMagnitude));
                if (Vector2.Distance(q, a + e * k) < rad) return true;
            }
            foreach (var pr in prisms)
            {
                if (pr.height < .1f) continue;
                if (Inside(pr.poly, q)) return true;
                for (int i = 0; i < 4; i++)
                {
                    Vector2 a = pr.poly[i], e = pr.poly[(i + 1) % 4] - a;
                    float k = Mathf.Clamp01(Vector2.Dot(q - a, e) / Mathf.Max(1e-9f, e.sqrMagnitude));
                    if (Vector2.Distance(q, a + e * k) < rad) return true;
                }
            }
            return false;
        }

        int WallsBetween(Vector2 a, Vector2 b)
        {
            Vector2 d = b - a; float len = d.magnitude; if (len < 1e-6f) return 0; d /= len; int n = 0;
            foreach (var s in segs) if (s[5] - s[4] > 1.5f && Hit(a, d, new Vector2(s[0], s[1]), new Vector2(s[2], s[3])) < len) n++;
            return n;
        }

        [Test, Explicit("Headless simulator: run by hand with data/runs/headless/request.json")]
        public void RunHeadlessRequest()
        {
            string root = Path.GetFullPath(Path.Combine(FloorPlan.ConfigDir, ".."));
            foreach (var line in RunRequest(root, Path.Combine(root, "data/runs/headless/request.json"))) TestContext.WriteLine(line);
        }

        /// <summary>Entry point shared with the standalone build (tools/headless): runs every run in a request file.</summary>
        public static string[] RunRequest(string root, string requestPath)
        {
            var req = JObject.Parse(File.ReadAllText(requestPath));
            // Runs share nothing (each gets its own simulator and brains), so they go on separate cores.
            var runs = ((JArray)req["runs"]).Cast<JObject>().ToList(); var lines = new string[runs.Count];
            System.Threading.Tasks.Parallel.For(0, runs.Count, i => lines[i] = new HeadlessSim().Run(root, runs[i]));
            return lines;
        }

        string Run(string root, JObject run)
        {
            var world = JObject.Parse(File.ReadAllText(Path.Combine(root, (string)run["world"])));
            string outDir = Path.Combine(root, (string)run["out"]); Directory.CreateDirectory(outDir);
            // One copy per folder: the editor once replayed queued duplicate requests, and two copies wrote one trace.
            string marker = Path.Combine(outDir, "running.marker");
            if (File.Exists(marker) || File.Exists(Path.Combine(outDir, "result.json"))) return run["out"] + ": skipped, already running or done";
            File.WriteAllText(marker, DateTime.Now.ToString("o"));
            ceiling = (float)world["ceiling_m"];
            segs = world["segments"].Select(s => s.Select(v => (float)v).ToArray()).ToArray();
            prisms = world["prisms"].Select(q => new Prism { height = (float)q["height_m"],
                poly = q["poly"].Select(c => new Vector2((float)c[0], (float)c[1])).ToArray() }).ToList();
            roofs = world["rooms"].Select(q => q["poly"].Select(c => new Vector2((float)c[0], (float)c[1])).ToArray()).ToList();
            people = world["people"].Select(q => ((string)q["id"], new Vector2((float)q["at"][0], (float)q["at"][1]))).ToList();
            solidBodies = (bool?)run["robot_collision"] ?? false;
            if (run["probes"] != null) Probes = run["probes"].Select(q => new Vector2((float)q[0], (float)q[1])).ToList();
            cal = SensorConfig.Load(); cfg = BrainConfig.Load(); cfg.frames_shared_at_start = true;
            // As the recorder does for a scenario with time_limit false (RecordingSession: mission_time_limit_s = 86400).
            if (!((bool?)run["time_limit"] ?? false)) cfg.mission_time_limit_s = 86400f;
            var overrides = (JObject)run["settings"] ?? new JObject();
            foreach (var kv in overrides)
            {
                var field = typeof(BrainConfig).GetField(kv.Key) ?? throw new ArgumentException("Unknown brain setting " + kv.Key);
                field.SetValue(cfg, kv.Value.ToObject(field.FieldType));
            }
            // As the recorder: the brain steps at 60 Hz, cameras at 15 Hz, LiDAR at 10 Hz (RecordingSession brainEvery = ClockHz / 60).
            double duration = (double?)run["duration_s"] ?? 300, dt = 1.0 / 60, radioPeriod = (double?)world["radio"]["broadcast_period_s"] ?? 2;
            float radioRange = (float?)world["radio"]["range_m"] ?? 12; int maxWalls = (int?)world["radio"]["max_walls"] ?? 1;
            // Fire (optional): the scenario's FireSim with its obstacles as fuel, as Atmosphere builds it; air only, no thermal camera.
            if (((bool?)run["fire"] ?? false) && (bool)world["fire"]["spread"])
            {
                var plan = FloorPlan.Load(Path.Combine(root, "config", (string)world["fire"]["floorplan"]));
                plan.landmarks.AddRange(world["fire"]["obstacles"].ToObject<List<FloorPlan.Landmark>>());
                float head = (float?)run["head_start_s"] ?? (float)world["fire"]["head_start_s"];
                heat = new FireSimHeat(plan, (int)world["fire"]["seed"], head, Path.Combine(root, "config", (string)world["fire"]["config"]));
                gateOrigin = new Vector2((float)world["gate"]["origin"][0], (float)world["gate"]["origin"][1]);
                gateYaw = (float)world["gate"]["yaw_deg"] * Mathf.Deg2Rad;
            }
            var robots = new List<Body>(); int i0 = 0;
            foreach (var r in world["robots"])
            {
                var pad = r["pad"].Select(v => (float)v).ToArray();
                robots.Add(new Body { id = (string)r["id"], p = new Vector2((float)r["at"][0], (float)r["at"][1]),
                    yaw = -(float)r["heading_deg"] * Mathf.Deg2Rad, start = i0 * ((double?)run["stagger_s"] ?? 0),
                    brain = new RobotBrain(cal, cfg, (string)r["id"], 1000 + i0 * 10 + 6, new EntryPose(pad[0], pad[1], pad[2])),
                    sensor = cal.air_temperature != null ? new AirTemperatureSensor(cal.air_temperature, 2000 + i0) : null });
                i0++;
            }
            var log = new StreamWriter(Path.Combine(outDir, "trace.jsonl"));
            var pending = new List<(Body to, byte[] bytes, double at, string from)>(); double nextRadio = radioPeriod;
            var sighted = new Dictionary<string, double>(); var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            int steps = (int)Math.Round(duration / dt);
            for (int k = 0; k <= steps; k++)
            {
                double t = k * dt;
                foreach (var m in pending.Where(m => m.at <= t + 1e-9).ToList()) { m.to.brain.OnPeerPacket(m.bytes, Math.Min(m.at, t)); /* a due time within rounding of this step must not lie after it */ pending.Remove(m); }
                foreach (var r in robots)
                {
                    if (t < r.start) continue;
                    r.brain.OnOdometry(new OdometrySample(t, r.v, 0, r.w));
                    r.brain.OnImu(new ImuSample(t, Vector3.zero, new Vector3(0, -r.w, 0)));
                    if (heat == null || r.sensor == null) r.brain.OnAirTemperature(t, 20);
                    else
                    {
                        r.air = TrueAir(r.p, (float)t); r.peakAir = Mathf.Max(r.peakAir, r.air); if (r.air > 60) r.hotS += (float)dt;
                        var reading = r.sensor.Step(t, r.air); if (reading.HasValue) r.brain.OnAirTemperature(t, reading.Value);
                    }
                    renderClock.Start(); if (k % 4 == 0) r.brain.OnRgbd(Depth(t, r));
                    if (t >= r.nextLidar) { r.brain.OnLidar(Lidar(t, r, robots)); r.nextLidar = t + .1; }
                    renderClock.Stop();
                    if (t >= r.nextSight)
                    {
                        r.nextSight = t + cfg.person_rgb_scan_period_s;
                        foreach (var (pid, at) in people)
                        {
                            Vector2 d = at - r.p; float range = d.magnitude;
                            float bearing = Vector2.SignedAngle(d, Fwd(r.yaw));   // positive to the right
                            float half = Mathf.Atan(cal.camera.width / 2f / cal.camera.fx) * Mathf.Rad2Deg;
                            float below = Mathf.Atan2(cal.camera.mount.height_m - .3f, range) * Mathf.Rad2Deg;
                            float lowest = cal.camera.mount.pitch_down_deg + Mathf.Atan(cal.camera.height / 2f / cal.camera.fy) * Mathf.Rad2Deg;
                            if (Mathf.Abs(bearing) > half || range > cfg.person_rgb_max_range_m || below > lowest) continue;
                            if (Cast(r.p, d / range, .3f, null, null) < range - .3f) continue;   // a wall or furniture in the way
                            r.brain.OnPersonSighting(t, bearing, range, .8f);
                            if (!sighted.ContainsKey(pid + "|" + r.id)) sighted[pid + "|" + r.id] = t;
                            log.WriteLine(new JObject { ["t"] = t, ["kind"] = "sighting", ["robot"] = r.id, ["person"] = pid, ["range_m"] = range, ["bearing_deg"] = bearing }.ToString(Newtonsoft.Json.Formatting.None));
                        }
                    }
                    brainClock.Start(); var cmd = r.brain.Step(t, (float)dt); brainClock.Stop();
                    float v = Mathf.Clamp(cmd.forwardMps, -cfg.limits.max_reverse_mps, cfg.limits.max_forward_mps);
                    float w = Mathf.Clamp(cmd.yawRateDeg, -cfg.limits.max_yaw_rate_dps, cfg.limits.max_yaw_rate_dps) * Mathf.Deg2Rad;
                    float yaw = r.yaw + w * (float)dt; Vector2 next = r.p + Fwd(yaw) * v * (float)dt;
                    if (Collides(next) && !Collides(r.p)) { next = r.p; v = 0; }
                    // A step that would press into another robot is refused (turn and move apart are tried in that order);
                    // moving out of an overlap is always allowed.
                    if (solidBodies)
                    {
                        float now = BodyGap(r, r.p, r.yaw, robots);
                        if (BodyGap(r, next, yaw, robots) < 0 && BodyGap(r, next, yaw, robots) < now)
                        {
                            r.contactS += (float)dt;
                            if (BodyGap(r, r.p, yaw, robots) >= Mathf.Min(0, now)) { next = r.p; v = 0; }
                            else { next = r.p; yaw = r.yaw; v = 0; w = 0; }
                        }
                    }
                    r.moved += Vector2.Distance(r.p, next); r.p = next; r.yaw = yaw; r.v = v; r.w = w;
                    foreach (var e in r.brain.DrainEvents())
                    {
                        var j = new JObject { ["t"] = e.t, ["kind"] = e.kind, ["robot"] = r.id, ["reason"] = e.reason.Length > 160 ? e.reason.Substring(0, 160) : e.reason };
                        if (e.numbers != null) j["numbers"] = JObject.FromObject(e.numbers);
                        log.WriteLine(j.ToString(Newtonsoft.Json.Formatting.None));
                    }
                    if (k % 30 == 0)
                    {
                        var s = r.brain.Snapshot();
                        log.WriteLine(new JObject { ["t"] = t, ["kind"] = "pose", ["robot"] = r.id, ["x"] = r.p.x, ["y"] = r.p.y, ["yaw_deg"] = -r.yaw * Mathf.Rad2Deg,
                            ["bx"] = s.pose.position.x, ["by"] = s.pose.position.y, ["v"] = v, ["w_dps"] = w * Mathf.Rad2Deg, ["state"] = s.state, ["mode"] = s.mode,
                            ["gx"] = s.goal?.x, ["gy"] = s.goal?.y, ["air_c"] = r.air }.ToString(Newtonsoft.Json.Formatting.None));
                    }
                }
                if (t + 1e-9 >= nextRadio)
                {
                    nextRadio += radioPeriod;
                    foreach (var from in robots)
                    {
                        if (t < from.start) continue;
                        var m = from.brain.BuildMessage(t); if (m == null) continue; var bytes = m.ToBytes();
                        foreach (var to in robots)
                        {
                            if (to == from || t < to.start) continue;
                            bool ok = Vector2.Distance(from.p, to.p) <= radioRange && WallsBetween(from.p, to.p) <= maxWalls;
                            log.WriteLine(new JObject { ["t"] = t, ["kind"] = "radio", ["from"] = from.id, ["to"] = to.id, ["ok"] = ok, ["bytes"] = bytes.Length,
                                ["room_claim"] = m.RoomClaim == null ? null : new JObject { ["x"] = m.RoomClaim.anchor.x, ["y"] = m.RoomClaim.anchor.y, ["bid_s"] = m.RoomClaim.bidS },
                                ["claim"] = m.Claim == null ? null : new JObject { ["x"] = m.Claim.centre.x, ["y"] = m.Claim.centre.y } }.ToString(Newtonsoft.Json.Formatting.None));
                            if (ok) pending.Add((to, bytes, t + .05, from.id));
                        }
                    }
                }
            }
            log.Close();
            var result = new JObject { ["world"] = (string)run["world"], ["duration_s"] = duration, ["settings"] = overrides, ["wall_s"] = stopwatch.Elapsed.TotalSeconds, ["render_s"] = renderClock.Elapsed.TotalSeconds, ["brain_s"] = brainClock.Elapsed.TotalSeconds,
                ["first_sighting_s"] = JObject.FromObject(sighted) };
            var maps = new JArray();
            foreach (var r in robots)
            {
                var s = r.brain.Snapshot();
                maps.Add(new JObject { ["robot"] = r.id, ["grid_size"] = s.gridSize, ["cell_m"] = s.cellM, ["origin"] = new JArray(s.gridOrigin.x, s.gridOrigin.y),
                    ["own_searched"] = Convert.ToBase64String(s.ownSearched), ["searched"] = Convert.ToBase64String(s.searched),
                    ["occupancy"] = Convert.ToBase64String(s.occupancy), ["walked_m"] = r.moved, ["final_state"] = s.state, ["peak_air_c"] = r.peakAir, ["air_over_60_s"] = r.hotS, ["robot_contact_s"] = r.contactS,
                    ["indoor"] = Convert.ToBase64String(r.brain.IndoorMask()),
                    ["frontiers"] = FrontierDump(r.brain, duration),
                    ["rooms"] = new JArray(((GridPlanner)typeof(RobotBrain).GetField("planner", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(r.brain)).Rooms.Rooms
                        .Select(q => new JObject { ["id"] = q.id, ["x"] = q.anchor.x, ["y"] = q.anchor.y, ["floor_m2"] = q.floorM2, ["indoor_m2"] = q.indoorFloorM2,
                            ["covered"] = q.coveredShare, ["corridor"] = q.corridor, ["promoted"] = q.promoted, ["wall_share"] = q.wallShare,
                            ["entered"] = q.entered, ["openings"] = q.openings, ["width_m"] = q.widthM, ["length_m"] = q.lengthM })) });
            }
            result["maps"] = maps;
            File.WriteAllText(Path.Combine(outDir, "result.json"), result.ToString()); File.Delete(marker);
            return $"{run["out"]}: {duration} s simulated in {stopwatch.Elapsed.TotalSeconds:0} s";
        }
    }
}
