using System;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using SmokeRescue.Brain;
using SmokeRescue.Sensors;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    /// <summary>
    /// Route watch (RobotBrain.CheckRouteWatch). Held-out run 1: robot A searched rooms facing away from its route home,
    /// its map held no fire from 96 s to 116 s while the fire spread along that route, and the way out closed 0.2 s after
    /// it turned at 120 s. A searching robot whose route home has gone unseen by its thermal camera turns to look along it.
    /// Every scenario here: the robot enters at the origin, is moved 3 m along +y at 0.5 m/s, then stands, and its thermal
    /// camera sees bare 20 C floor ahead of it the whole time. Heading 0 faces away from the way out (its back to the route
    /// home); heading 180 faces the way out while being moved the same way.
    /// </summary>
    public sealed class RouteWatchTests
    {
        const int W = 160, H = 120;
        // The BrainTests view (75 x 60 degrees) at twice the pixels, so floor rows lie under 0.1 m apart out to 4 m.
        const float F = 104;
        static readonly ushort[] Floor = FloorDepth();
        static readonly byte[] Rgb = Checker();
        static readonly ushort[] Cool = Enumerable.Repeat((ushort)2932, W * H).ToArray();   // 20 C everywhere

        static BrainConfig Settings(bool routeWatch = true)
        {
            var c = new BrainConfig
            {
                limits = new BrainConfig.MotionLimits
                { max_forward_mps = .8f, max_reverse_mps = .3f, max_yaw_rate_dps = 60, max_accel_mps2 = .6f }
            };
            c.grid_size_m = 16; c.robot_radius_m = .25f;
            // The body is moved open loop, so the brain's forward commands meet zero or reverse odometry: no stall recovery
            // and no progress watchdog, either of which would clear the path these tests watch.
            c.stuck_window_s = 1000; c.stuck_blacklist_s = 2000; c.progress_window_s = 1000;
            c.route_watch = routeWatch; c.route_watch_stale_s = 3; c.route_watch_max_stale_share = .5f;
            c.route_watch_min_interval_s = 5; c.route_watch_dwell_s = .5f; c.route_watch_max_s = 8;
            return c;
        }
        static SensorConfig Calibration()
        {
            return new SensorConfig
            {
                camera = new SensorConfig.CameraConfig
                {
                    width = W, height = H, fx = F, fy = F, cx = (W - 1) / 2f, cy = (H - 1) / 2f, rate_hz = 15,
                    mount = new SensorConfig.Mount { height_m = 1.6f, forward_offset_m = .25f, pitch_down_deg = 10 }
                },
                depth = new SensorConfig.DepthConfig { min_m = .2f, max_m = 8 },
                thermal = new SensorConfig.ThermalConfig
                { width = W, height = H, fx = F, fy = F, cx = (W - 1) / 2f, cy = (H - 1) / 2f, rate_hz = 15 }
            };
        }
        /// <summary>Optical depth of flat floor under the 1.6 m, 10 degree mount: row v meets the floor at z = h / (sin p +
        /// (v - cy) / fy cos p). Rows at or above the horizon, or beyond the 8 m depth range, have no return. The image does
        /// not depend on where the robot stands or faces, so one array serves every frame.</summary>
        static ushort[] FloorDepth()
        {
            var depth = new ushort[W * H]; float pitch = 10 * Mathf.Deg2Rad;
            for (int v = 0; v < H; v++)
            {
                float k = Mathf.Sin(pitch) + (v - (H - 1) / 2f) / F * Mathf.Cos(pitch);
                if (k <= 0 || 1.6f / k > 8) continue;
                for (int u = 0; u < W; u++) depth[v * W + u] = (ushort)Mathf.RoundToInt(1.6f / k * 1000);
            }
            return depth;
        }
        static byte[] Checker()
        {
            var rgb = new byte[W * H * 3];
            for (int y = 0; y < H; y++) for (int x = 0; x < W; x++)
            {
                byte b = (x / 16 + y / 16) % 2 == 0 ? (byte)255 : (byte)0;
                for (int ch = 0; ch < 3; ch++) rgb[(y * W + x) * 3 + ch] = b;
            }
            return rgb;
        }
        static OccupancyGrid Map(RobotBrain b) => (OccupancyGrid)typeof(RobotBrain)
            .GetField("grid", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(b);
        static void Room(RobotBrain b, float left, float right, float bottom, float top)
        {
            var g = Map(b);
            for (int i = 0; i < g.Count; i++)
            {
                var p = g.Centre(i);
                if (p.x >= left && p.x <= right && p.y >= bottom && p.y <= top) g.ObserveFree(i);
            }
        }
        static void Motion(RobotBrain b, double t, float v = 0, float ccw = 0)
        {
            b.OnOdometry(new OdometrySample(t, v, 0, 99)); // Odometry yaw must be ignored.
            b.OnImu(new ImuSample(t, Vector3.zero, new Vector3(0, -ccw, 0)));
        }
        static float HeadingError(BrainPose pose, Vector2 point) => Mathf.DeltaAngle(pose.yawRad * Mathf.Rad2Deg,
            Mathf.Atan2(-(point.x - pose.position.x), point.y - pose.position.y) * Mathf.Rad2Deg);

        sealed class Tick { public double t; public BrainPose pose; public Vector2? goal; public Vector2[] path; }
        sealed class Run { public RobotBrain brain; public readonly List<BrainEvent> events = new List<BrainEvent>(); public readonly List<Tick> ticks = new List<Tick>(); }

        /// <summary>The walk described on the class, at 20 Hz. The only unknown floor is the strip beyond y = 7.5 (past the
        /// 4 m the camera counts as searched from where the robot stops), so the frontier the robot is heading for lies
        /// straight ahead, away from its way out. With followYaw, the brain's own yaw commands turn the body from its first
        /// route_watch event on (the body stays where it stands).</summary>
        static Run Walk(BrainConfig c, float headingDeg, double seconds, Func<List<BrainEvent>, float> airC = null, bool followYaw = false, Action<RobotBrain, double> beforeStep = null)
        {
            var b = new RobotBrain(Calibration(), c, "A", 0, new EntryPose(.05f, .05f, headingDeg));
            Room(b, -8, 8, -8, 7.5f);
            var run = new Run { brain = b }; var command = new BrainCommand();
            float forward = headingDeg == 0 ? .5f : -.5f;   // facing the way out it is moved backwards, along +y all the same
            for (int tick = 0; tick / 20.0 <= seconds; tick++)
            {
                double t = tick / 20.0;
                bool turning = followYaw && run.events.Any(e => e.kind == "route_watch");
                Motion(b, t, t < 6 ? forward : 0, turning ? command.yawRateDeg * Mathf.Deg2Rad : 0);
                b.OnRgbd(new RgbdFrame(t, W, H, Rgb, Floor)); b.OnThermal(new ThermalFrame(t, W, H, Cool));
                b.OnAirTemperature(t, airC == null ? 20 : airC(run.events));
                beforeStep?.Invoke(b, t);
                command = b.Step(t, .05f);
                run.events.AddRange(b.DrainEvents());
                var s = b.Snapshot();
                run.ticks.Add(new Tick { t = t, pose = s.pose, goal = s.goal, path = s.path });
            }
            return run;
        }
        /// <summary>Route cells home (x = 0.05, from the door at y = 0.05) beyond escape_ignore_m of where the robot stands.</summary>
        static List<int> RouteCells(Run run, BrainConfig c)
        {
            var g = Map(run.brain); float stand = run.ticks.Last().pose.position.y; var cells = new List<int>();
            for (int i = 0; .05f + .1f * i <= stand - c.escape_ignore_m - .05f; i++) cells.Add(g.Cell(new Vector2(.05f, .05f + .1f * i)));
            return cells;
        }

        [Test]
        public void ARobotSearchingWithItsBackToTheRouteHomeTurnsToLookAlongIt()
        {
            // Fails without the route watch (no route_watch event), without the turn (the look times out unfaced and the
            // heading stays 180 degrees off), or with a look point off the route (x, distance checks).
            var c = Settings(); var run = Walk(c, 0, 15, followYaw: true);
            var look = run.events.FirstOrDefault(e => e.kind == "route_watch");
            Assert.That(look, Is.Not.Null, "a route look started");
            Assert.That(look.t, Is.GreaterThan(c.route_watch_stale_s), "after searching with its back to the route for longer than route_watch_stale_s");
            Assert.That(look.numbers["stale_share"], Is.GreaterThan(c.route_watch_max_stale_share));
            Assert.That(look.numbers["route_m"], Is.EqualTo(3).Within(.2), "the route home from where it stands");
            var point = new Vector2((float)look.numbers["x_m"], (float)look.numbers["y_m"]);
            Assert.That(point.x, Is.EqualTo(.05f).Within(.06f), "the look point is on the route home");
            Assert.That(look.numbers["distance_m"], Is.GreaterThan(c.escape_ignore_m).And.LessThan(c.escape_ignore_m + .3f),
                "the stale route point nearest the robot beyond escape_ignore_m");
            var done = run.events.FirstOrDefault(e => e.kind == "route_watch_done" && e.t > look.t);
            Assert.That(done, Is.Not.Null, "the look ended");
            Assert.That(done.numbers["faced"], Is.EqualTo(1)); Assert.That(done.numbers["cancelled"], Is.EqualTo(0));
            Assert.That(done.numbers["duration_s"], Is.LessThan(c.route_watch_max_s), "ended by facing and holding, not by the time limit");
            var pose = run.ticks.First(k => Math.Abs(k.t - done.t) < 1e-9).pose;
            Assert.That(Mathf.Abs(HeadingError(pose, point)), Is.LessThanOrEqualTo(c.survivor_face_tolerance_deg + 2f), "facing the route when the look ends");
            // The look really looked: the thermal camera has now seen most of the route home (WithRouteWatchOff shows the
            // same walk without a look leaves every one of these cells unseen).
            var g = Map(run.brain); var route = RouteCells(run, c);
            Assert.That(route.Count(i => g.HeatLookTime(i) >= look.t), Is.GreaterThan(route.Count / 2));
        }

        [Test]
        public void ARobotWhoseThermalCameraRecentlyLookedAlongTheRouteDoesNotTurnBack()
        {
            // Control: the same walk facing the way out. Fails if look times are not recorded for cool floor (every route cell
            // would read as never seen) or if the stale test ignores them.
            var c = Settings(); var run = Walk(c, 180, 15);
            Assert.That(run.events.Count(e => e.kind == "route_watch"), Is.Zero);
            // The control exercises fresh looks: most route cells beyond escape_ignore_m were looked at within route_watch_stale_s.
            var g = Map(run.brain); var route = RouteCells(run, c); double end = run.ticks.Last().t;
            Assert.That(route.Count, Is.GreaterThanOrEqualTo(8));
            Assert.That(route.Count(i => end - g.HeatLookTime(i) <= c.route_watch_stale_s), Is.GreaterThan(route.Count * (1 - c.route_watch_max_stale_share)));
            Assert.That(run.brain.Snapshot().heat.Max(), Is.LessThanOrEqualTo(c.ambient_c), "cool floor adds look times, never heat");
        }

        [Test]
        public void WithRouteWatchOffNoLookStarts()
        {
            // Fails if the route_watch flag is ignored: this is the walk that starts a look in the first test.
            var c = Settings(false); var run = Walk(c, 0, 15, followYaw: true);
            Assert.That(run.events.Count(e => e.kind == "route_watch"), Is.Zero);
            // The walk keeps its back to the route home: its thermal camera never looked at any of it.
            var g = Map(run.brain); var route = RouteCells(run, c);
            Assert.That(route.Count, Is.GreaterThanOrEqualTo(8));
            Assert.That(route.All(i => double.IsNegativeInfinity(g.HeatLookTime(i))), Is.True);
        }

        [Test]
        public void AHeatRetreatCancelsALookAndNoLookStartsWhileRetreating()
        {
            // The air turns hot on the step after a look starts. Fails if the retreat does not cancel the look (no
            // route_watch_done at the retreat) or if a look can start while retreating.
            var c = Settings();
            var run = Walk(c, 0, 12, events => events.Any(e => e.kind == "route_watch") ? 80f : 20f);
            var look = run.events.FirstOrDefault(e => e.kind == "route_watch");
            Assert.That(look, Is.Not.Null, "precondition: the look started");
            var retreat = run.events.First(e => e.kind == "retreat");
            Assert.That(retreat.t, Is.GreaterThan(look.t));
            var done = run.events.Where(e => e.kind == "route_watch_done").ToList();
            Assert.That(done, Has.Count.EqualTo(1));
            Assert.That(done[0].t, Is.EqualTo(retreat.t), "cancelled on the step the retreat began");
            Assert.That(done[0].numbers["cancelled"], Is.EqualTo(1));
            Assert.That(run.events.Count(e => e.kind == "route_watch"), Is.EqualTo(1), "no look starts while retreating");
            Assert.That(run.brain.Snapshot().state, Is.EqualTo("retreat"));
            // Hot from the start: the robot retreats all along and never starts a look.
            var hot = Walk(c, 0, 12, _ => 80f);
            Assert.That(hot.events.Any(e => e.kind == "retreat"), Is.True);
            Assert.That(hot.events.Count(e => e.kind == "route_watch"), Is.Zero);
        }

        [Test]
        public void AfterTheLookTheRobotResumesItsPreviousGoal()
        {
            // Fails if the look clears the path (the goal goes null during the look), if it leaves a different goal behind,
            // or if the robot never turns back to its path once the look ends.
            var c = Settings(); var run = Walk(c, 0, 15, followYaw: true);
            var look = run.events.FirstOrDefault(e => e.kind == "route_watch");
            Assert.That(look, Is.Not.Null, "precondition: the look started");
            var done = run.events.First(e => e.kind == "route_watch_done" && e.t > look.t);
            var before = run.ticks.Last(k => k.t < look.t).goal;
            Assert.That(before.HasValue, Is.True, "heading for a frontier when the look started");
            bool Same(Tick k) => k.goal.HasValue && Vector2.Distance(k.goal.Value, before.Value) < 1e-4f;
            foreach (var k in run.ticks.Where(k => k.t >= look.t && k.t <= done.t))
            {
                Assert.That(Same(k), Is.True, $"goal kept during the look at {k.t:0.00} s");
                Assert.That(k.path.Length, Is.GreaterThan(0), $"path kept during the look at {k.t:0.00} s");
            }
            Vector2 Ahead(Tick k) { foreach (var p in k.path) if (Vector2.Distance(p, k.pose.position) >= c.lookahead_m) return p; return k.goal.Value; }
            var end = run.ticks.First(k => Math.Abs(k.t - done.t) < 1e-9);
            Assert.That(Mathf.Abs(HeadingError(end.pose, Ahead(end))), Is.GreaterThan(90f), "precondition: the look turned it away from its path");
            var after = run.ticks.Where(k => k.t > done.t && k.t <= done.t + 4).ToList();
            Assert.That(after.All(Same), Is.True, "the same goal after the look");
            Assert.That(after.Any(k => Mathf.Abs(HeadingError(k.pose, Ahead(k))) < c.turn_in_place_deg + 5f), Is.True,
                "turned back to its path within 4 s");
        }
        static void Field(RobotBrain b, string name, object value) => typeof(RobotBrain)
            .GetField(name, System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).SetValue(b, value);

        static void NoLookDuring(Action<RobotBrain, double> change)
        {
            Assert.That(Walk(Settings(), 0, 12).events.Any(e => e.kind == "route_watch"), Is.True, "same walk starts a look when searching safely");
            foreach (bool enabled in new[] { true, false })
                Assert.That(Walk(Settings(enabled), 0, 12, beforeStep: change).events.Any(e => e.kind == "route_watch"), Is.False,
                    "the safety case keeps the feature-off result");
        }

        [Test]
        public void CarryingAPersonDoesNotStartARouteLook() => NoLookDuring((b, t) =>
        {
            if (t == 6) b.BeginCarry(t, .3f, "sensor-confirmed person");
        });

        [Test]
        public void ReturningDoesNotStartARouteLook() => NoLookDuring((b, t) =>
        {
            if (t == 6) typeof(RobotBrain).GetMethod("BeginReturn", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
                .Invoke(b, new object[] { t, "Return test" });
        });

        [Test]
        public void LowHeatBudgetDoesNotStartARouteLook() => NoLookDuring((b, t) =>
        {
            if (t >= 6) Field(b, "budget", .2f);
        });

        [Test]
        public void UnknownRouteHomeDoesNotStartARouteLook() => NoLookDuring((b, t) =>
        {
            if (t != 6) return;
            var g = Map(b);
            for (int i = 0; i < g.Count; i++)
            {
                var p = g.Centre(i);
                if (p.y < 1 || p.y > 1.5f) continue;
                for (int hit = 0; hit < 20; hit++) g.ObserveOccupied(i);
            }
        });
    }
}
