using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NUnit.Framework;
using SmokeRescue.Brain;
using SmokeRescue.Sensors;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    public sealed partial class BrainTests
    {
        static BrainConfig Settings() => new BrainConfig
        {
            limits = new BrainConfig.MotionLimits
            { max_forward_mps = .8f, max_reverse_mps = .3f, max_yaw_rate_dps = 60, max_accel_mps2 = .6f }
        };
        static BrainConfig HouseSettings()
        {
            var c = Settings(); c.fixed_grid_origin = true; return c;
        }
        static SensorConfig Calibration()
        {
            return new SensorConfig
            {
                camera = new SensorConfig.CameraConfig
                {
                    width = 80, height = 60, fx = 52, fy = 52, cx = 39.5f, cy = 29.5f, rate_hz = 15,
                    mount = new SensorConfig.Mount { height_m = 1.6f, forward_offset_m = .25f, pitch_down_deg = 10 }
                },
                depth = new SensorConfig.DepthConfig { min_m = .2f, max_m = 8 },
                thermal = new SensorConfig.ThermalConfig
                { width = 80, height = 60, fx = 52, fy = 52, cx = 39.5f, cy = 29.5f, rate_hz = 15 }
            };
        }
        static RgbdFrame Image(double t, bool textured, ushort mm = 2000)
        {
            var rgb = new byte[80 * 60 * 3];
            var depth = Enumerable.Repeat(mm, 80 * 60).ToArray();
            for (int y = 0; y < 60; y++) for (int x = 0; x < 80; x++)
            {
                byte b = textured && ((x / 8 + y / 8) % 2 == 0) ? (byte)255 : (byte)0;
                for (int c = 0; c < 3; c++) rgb[(y * 80 + x) * 3 + c] = b;
            }
            return new RgbdFrame(t, 80, 60, rgb, depth);
        }
        static ThermalFrame Person(double t, float c = 34)
        {
            var pixels = Enumerable.Repeat((ushort)2932, 80 * 60).ToArray();
            // 20 x 30 pixels at 2 m is 0.89 m2 with this calibration.
            for (int y = 15; y < 45; y++) for (int x = 30; x < 50; x++) pixels[y * 80 + x] = (ushort)Math.Round((c + 273.15f) * 10);
            pixels[2] = pixels[3] = 3072;
            return new ThermalFrame(t, 80, 60, pixels);
        }
        static OccupancyGrid OpenGrid(BrainConfig c)
        {
            var g = new OccupancyGrid(c);
            for (int y = 0; y < g.Size; y++) for (int x = 0; x < g.Size; x++) g.ObserveFree(x, y);
            return g;
        }
        static void Motion(RobotBrain b, double t, float v = 0, float ccw = 0)
        {
            b.OnOdometry(new OdometrySample(t, v, 0, 99)); // Odometry yaw must be ignored.
            b.OnImu(new ImuSample(t, Vector3.zero, new Vector3(0, -ccw, 0)));
        }

        static RobotBrain BlockedBrain(out BrainConfig c)
        {
            c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f;
            var b = new RobotBrain(Calibration(), c, "A", 0, new EntryPose(.05f, .05f, 0));
            Room(b, -4, 4, -4, 4);
            var route = new GridPlanner(Map(b), c).Plan(new Vector2(.05f, .05f), new Vector2(.05f, 3.05f));
            typeof(RobotBrain).GetMethod("SetPath", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
                .Invoke(b, new object[] { route, true });
            return b;
        }
        static BrainCommand DriveTick(RobotBrain b, int tick, float v = 0, float yawDps = 0)
        {
            double t = tick / 20.0;
            Motion(b, t, v, yawDps * Mathf.Deg2Rad);
            b.OnRgbd(Image(t, true, 0)); b.OnAirTemperature(t, 20);
            return b.Step(t, .05f);
        }

        [Test]
        public void StalledForwardCommandEmitsOnceReversesAndReplansAroundOwnObstacle()
        {
            var b = BlockedBrain(out var c); var before = b.Snapshot().path;
            var all = new List<BrainEvent>(); var command = new BrainCommand();
            int tick = 0; bool detected = false, reversed = false, recovered = false;
            Vector2 contact = Vector2.zero;
            for (; tick < 200; tick++)
            {
                command = DriveTick(b, tick, command.forwardMps < 0 ? command.forwardMps : 0, command.yawRateDeg);
                var events = b.DrainEvents(); all.AddRange(events);
                Assert.That(command.forwardMps, Is.GreaterThanOrEqualTo(-c.limits.max_reverse_mps));
                if (events.Any(e => e.kind == "stuck"))
                {
                    Assert.That(detected, Is.False, "One event per recovery episode."); detected = true;
                    var e = events.Single(x => x.kind == "stuck"); contact = b.Snapshot().pose.position;
                    Assert.That(e.numbers["commanded_speed_mps"], Is.GreaterThan(.15));
                    Assert.That(e.numbers["measured_speed_mps"], Is.Zero);
                    Assert.That(e.numbers["duration_s"], Is.InRange(1.2, 1.26));
                    Assert.That(Map(b).OwnOccupancy(Map(b).Cell(contact + new Vector2(0, .5f))), Is.EqualTo(255));
                    Assert.That(new GridPlanner(Map(b), c).PathSafe(before), Is.False);
                }
                reversed |= command.forwardMps < 0;
                if (detected && b.Snapshot().state != "recover") { recovered = true; break; }
            }
            Assert.That(detected && reversed && recovered, Is.True);
            Assert.That(all.Count(e => e.kind == "stuck"), Is.EqualTo(1));
            Assert.That(Vector2.Distance(contact, b.Snapshot().pose.position), Is.InRange(.15f, .21f));
            var after = b.Snapshot().path;
            Assert.That(after, Is.Not.Empty); Assert.That(after.SequenceEqual(before), Is.False);
            Assert.That(new GridPlanner(Map(b), c).PathSafe(after), Is.True);
        }

        [Test]
        public void MovingOdometryNeverTriggersStuck()
        {
            var b = BlockedBrain(out _); var all = new List<BrainEvent>(); var command = new BrainCommand();
            for (int tick = 0; tick < 100; tick++)
            {
                command = DriveTick(b, tick, command.forwardMps, command.yawRateDeg); all.AddRange(b.DrainEvents());
                Assert.That(command.forwardMps, Is.GreaterThanOrEqualTo(0));
            }
            Assert.That(b.Snapshot().pose.position.y, Is.GreaterThan(1));
            Assert.That(all.Any(e => e.kind == "stuck"), Is.False);
        }

        [Test]
        public void RepeatedStallsNearOnePlaceBlacklistTheGoalAndSelectAnotherFrontier()
        {
            var b = BlockedBrain(out var c); var initialGoal = b.Snapshot().goal.Value;
            var all = new List<BrainEvent>(); var command = new BrainCommand(); bool changed = false;
            for (int tick = 0; tick < 1000; tick++)
            {
                // A depth stream that keeps missing the contact can clear the inferred strip.
                var g = Map(b);
                for (int i = 0; i < g.Count; i++) if (g.OwnOccupancy(i) == 255)
                    for (int ray = 0; ray < 4; ray++) g.ObserveFree(i);
                command = DriveTick(b, tick, command.forwardMps < 0 ? command.forwardMps : 0, command.yawRateDeg);
                all.AddRange(b.DrainEvents());
                var blacklist = all.FirstOrDefault(e => e.kind == "goal_blacklisted");
                if (blacklist != null && b.Snapshot().goal.HasValue)
                {
                    Assert.That(blacklist.numbers["repeats"], Is.EqualTo(3));
                    Assert.That(blacklist.numbers["until_s"] - blacklist.t, Is.EqualTo(30).Within(.001));
                    Assert.That(Vector2.Distance(initialGoal, b.Snapshot().goal.Value), Is.GreaterThan(.75));
                    var excluded = typeof(RobotBrain).GetMethod("GoalBlacklisted", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                    double until = blacklist.numbers["until_s"];
                    Assert.That(excluded.Invoke(b, new object[] { initialGoal, until - .01 }), Is.EqualTo(true));
                    Assert.That(excluded.Invoke(b, new object[] { initialGoal, until }), Is.EqualTo(false));
                    changed = true; break;
                }
            }
            Assert.That(changed, Is.True);
        }

        [Test]
        public void StuckRecoveryCommandsEventsAndMapsAreDeterministic()
        {
            var a = BlockedBrain(out _); var b = BlockedBrain(out _);
            var command = new BrainCommand(); int stuck = 0;
            for (int tick = 0; tick < 180; tick++)
            {
                var ac = DriveTick(a, tick, command.forwardMps < 0 ? command.forwardMps : 0, command.yawRateDeg);
                var bc = DriveTick(b, tick, command.forwardMps < 0 ? command.forwardMps : 0, command.yawRateDeg);
                Assert.That(ac.forwardMps, Is.EqualTo(bc.forwardMps)); Assert.That(ac.yawRateDeg, Is.EqualTo(bc.yawRateDeg));
                var ae = a.DrainEvents(); var be = b.DrainEvents(); Assert.That(ae.Count, Is.EqualTo(be.Count));
                for (int i = 0; i < ae.Count; i++)
                {
                    Assert.That(ae[i].t, Is.EqualTo(be[i].t)); Assert.That(ae[i].kind, Is.EqualTo(be[i].kind));
                    Assert.That(ae[i].reason, Is.EqualTo(be[i].reason)); CollectionAssert.AreEqual(ae[i].numbers, be[i].numbers);
                    if (ae[i].kind == "stuck") stuck++;
                }
                CollectionAssert.AreEqual(a.Snapshot().path, b.Snapshot().path); command = ac;
            }
            Assert.That(stuck, Is.GreaterThan(0));
            CollectionAssert.AreEqual(a.Snapshot().occupancy, b.Snapshot().occupancy);
        }

        [Test]
        public void StuckEvidenceResetsAcrossMotionAndSensorGaps()
        {
            var b = BlockedBrain(out _); var all = new List<BrainEvent>();
            // Only the end of the scripted route counts as searchable floor: a ceiling "seen" over x -0.2..0.2, y 3.5..4.1.
            ((BrainConfig)typeof(RobotBrain).GetField("c", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(b)).search_requires_ceiling = true;
            var grid = Map(b); var roof = (byte[])typeof(OccupancyGrid).GetField("roofHits", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(grid);
            for (float x = -.15f; x <= .15f; x += .1f) for (float y = 3.55f; y <= 4.05f; y += .1f) roof[grid.Cell(new Vector2(x, y))] = 255;
            typeof(OccupancyGrid).GetField("roofDirty", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).SetValue(grid, true);
            ((BrainConfig)typeof(RobotBrain).GetField("c", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(b)).ceiling_radius_m = 0f;
            for (int tick = 0; tick <= 16; tick++) { DriveTick(b, tick); all.AddRange(b.DrainEvents()); }
            // The first interval moves; the last one is zero. Looking only at the latest sample is wrong.
            Motion(b, .825, .1f); DriveTick(b, 17); all.AddRange(b.DrainEvents());
            for (int tick = 18; tick <= 32; tick++) { DriveTick(b, tick); all.AddRange(b.DrainEvents()); }
            Assert.That(all.Any(e => e.kind == "stuck"), Is.False);
            b.Step(2, .4f); all.AddRange(b.DrainEvents());
            for (int tick = 41; tick <= 60; tick++) { DriveTick(b, tick); all.AddRange(b.DrainEvents()); }
            Assert.That(all.Any(e => e.kind == "stuck"), Is.False);
            for (int tick = 61; tick <= 85; tick++) { DriveTick(b, tick); all.AddRange(b.DrainEvents()); }
            Assert.That(all.Count(e => e.kind == "stuck"), Is.EqualTo(1));
        }

        [Test]
        public void FreshDepthClearsAnInferredContactButOlderDepthDoesNot()
        {
            var b = BlockedBrain(out _); int tick = 0;
            for (; tick < 80; tick++)
            { DriveTick(b, tick); if (b.DrainEvents().Any(e => e.kind == "stuck")) break; }
            Assert.That(tick, Is.LessThan(80));
            var g = Map(b); var s = Calibration(); s.camera.mount.pitch_down_deg = 0; s.camera.mount.forward_offset_m = 0;
            int cell = g.Cell(new Vector2(.05f, .55f)); Assert.That(g.OwnOccupancy(cell), Is.EqualTo(255));
            var pose = new BrainPose { position = new Vector2(.05f, .05f) };
            g.IntegrateDepth(Image(tick / 20.0 - .1, true, 3000), s, pose);
            Assert.That(g.OwnOccupancy(cell), Is.EqualTo(255));
            for (int ray = 1; ray <= 3; ray++) g.IntegrateDepth(Image(tick / 20.0 + ray * .05, true, 3000), s, pose);
            Assert.That(g.OwnOccupancy(cell), Is.EqualTo(1));
        }

        [Test]
        public void RecoveryTimesOutWhenReverseAndTurningAlsoCannotMove()
        {
            var b = BlockedBrain(out _); var all = new List<BrainEvent>();
            double contactAt = -1; bool reverse = false, turn = false, ended = false;
            for (int tick = 0; tick < 200; tick++)
            {
                var cmd = DriveTick(b, tick); var events = b.DrainEvents(); all.AddRange(events);
                var contact = events.FirstOrDefault(e => e.kind == "stuck");
                if (contact != null) contactAt = contact.t;
                reverse |= cmd.forwardMps < 0; turn |= contactAt >= 0 && cmd.yawRateDeg != 0;
                if (contactAt >= 0 && b.Snapshot().state != "recover")
                {
                    Assert.That(tick / 20.0 - contactAt, Is.InRange(5.0, 5.2)); ended = true; break;
                }
            }
            Assert.That(reverse && turn && ended, Is.True);
            Assert.That(all.Count(e => e.kind == "stuck"), Is.EqualTo(1));
        }

        [Test]
        public void NoReverseLimitUsesATurnAndDoesNotBackIntoUnknownSpace()
        {
            foreach (bool noReverse in new[] { true, false })
            {
                var c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f;
                if (noReverse) c.limits.max_reverse_mps = 0;
                var b = new RobotBrain(Calibration(), c, "A", 0, new EntryPose(.05f, .05f, 0));
                // The current footprint is known, but the next rear footprint would enter unknown cells.
                Room(b, -.4f, .4f, -.25f, 4);
                var route = new GridPlanner(Map(b), c).Plan(new Vector2(.05f, .05f), new Vector2(.05f, 3.05f));
                typeof(RobotBrain).GetMethod("SetPath", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
                    .Invoke(b, new object[] { route, true });
                bool detected = false, turned = false;
                for (int tick = 0; tick < 100; tick++)
                {
                    var cmd = DriveTick(b, tick); detected |= b.DrainEvents().Any(e => e.kind == "stuck");
                    Assert.That(cmd.forwardMps, Is.GreaterThanOrEqualTo(0));
                    if (detected && cmd.yawRateDeg != 0) { turned = true; break; }
                }
                Assert.That(detected && turned, Is.True);
            }
        }

        [Test]
        public void StuckConfigurationRejectsInvalidThresholdsAndRecoveryBounds()
        {
            foreach (Action<BrainConfig> invalid in new Action<BrainConfig>[]
            {
                c => c.stuck_window_s = 0, c => c.stuck_command_mps = 0,
                c => c.stuck_measured_mps = c.stuck_command_mps, c => c.stuck_measured_mps = -1,
                c => c.stuck_strip_depth_m = 0, c => c.stuck_log_odds = c.occupied_threshold - .01f,
                c => c.stuck_log_odds = c.log_odds_cap + .01f, c => c.stuck_reverse_m = 0,
                c => c.stuck_reverse_mps = 0, c => c.stuck_reverse_timeout_s = 0,
                c => c.stuck_turn_deg = 91, c => c.stuck_turn_timeout_s = 0,
                c => c.stuck_repeat_radius_m = 0, c => c.stuck_repeat_count = 1,
                c => c.stuck_blacklist_s = 0, c => c.obstacle_cost_margin_m = 0,
                c => c.obstacle_cost_weight = -1, c => c.stuck_window_s = float.NaN
            })
            {
                var c = Settings(); invalid(c); Assert.Throws<ArgumentException>(() => c.Validate());
            }
            var limited = Settings(); limited.limits.max_reverse_mps = 0;
            Assert.DoesNotThrow(() => limited.Validate());
        }

        static OccupancyGrid DoorGrid(BrainConfig c, float width)
        {
            var g = OpenGrid(c);
            for (int row = 0; row < g.Size; row++)
            {
                int i = row * g.Size + g.Size / 2;
                if (g.Centre(i).y < 0 || g.Centre(i).y >= width) g.ObserveOccupied(i);
            }
            return g;
        }
        [Test]
        public void SoftClearanceMovesDoorCrossingAwayFromTheJambWithoutClosingNarrowDoors()
        {
            var c = Settings(); c.grid_size_m = 8; c.robot_radius_m = .25f;
            // Turn off the soft cost for an independently measured shortest-path control.
            c.obstacle_cost_weight = 0;
            var g = DoorGrid(c, 1.6f); var planner = new GridPlanner(g, c);
            var from = new Vector2(2.05f, .35f); var to = new Vector2(-2.05f, .35f);
            var shortest = planner.Plan(from, to); c.obstacle_cost_weight = 2;
            var preferred = planner.Plan(from, to);
            Assert.That(shortest, Is.Not.Empty); Assert.That(preferred, Is.Not.Empty);
            float before = shortest.Where(p => Mathf.Abs(p.x) < .16f).Min(p => p.y);
            float after = preferred.Where(p => Mathf.Abs(p.x) < .16f).Min(p => p.y);
            Assert.That(after - before, Is.GreaterThanOrEqualTo(.099f));
            Assert.That(planner.PathSafe(preferred), Is.True);
            var narrow = DoorGrid(c, .8f);
            Assert.That(new GridPlanner(narrow, c).Plan(new Vector2(2.05f, .45f), new Vector2(-2.05f, .45f)), Is.Not.Empty);
            TestContext.WriteLine($"Door crossing y changed from {before:F2} m to {after:F2} m; 0.8 m door remains reachable.");
        }

        [Test]
        public void DeadReckoningUsesGyroYawAndBodyTranslation()
        {
            var c = Settings(); c.motion_stale_s = 2;
            var b = new RobotBrain(Calibration(), c, "A", 7);
            Motion(b, 0, 1); b.Step(0, .02f);
            Motion(b, 1, 1); b.Step(1, 1);
            Assert.That(b.Snapshot().pose.position.y, Is.EqualTo(1).Within(.001));
            Motion(b, 2, 0, Mathf.PI / 2); b.Step(2, 1);
            Assert.That(b.Snapshot().pose.yawRad, Is.EqualTo(Mathf.PI / 2).Within(.001));
            Motion(b, 3, 1); b.Step(3, 1);
            Assert.That(b.Snapshot().pose.position.x, Is.EqualTo(-1).Within(.001));
            Assert.That(b.Snapshot().pose.position.y, Is.EqualTo(1).Within(.001));
        }

        [Test]
        public void MixedRateMotionIntegratesAConstantQuarterCircle()
        {
            var b = new RobotBrain(Calibration(), Settings(), "A", 0); int odomTick = 0, imuTick = 0;
            for (int tick = 0; tick <= 60; tick++)
            {
                double t = tick / 60.0;
                while (odomTick / 50.0 <= t)
                { b.OnOdometry(new OdometrySample(odomTick++ / 50.0, 1, 0, 99)); }
                while (imuTick / 200.0 <= t)
                { b.OnImu(new ImuSample(imuTick++ / 200.0, Vector3.zero, new Vector3(0, -Mathf.PI / 2, 0))); }
                b.Step(t, 1f / 60);
            }
            Assert.That(b.Snapshot().pose.yawRad, Is.EqualTo(Mathf.PI / 2).Within(.001));
            Assert.That(b.Snapshot().pose.position.x, Is.EqualTo(-2 / Mathf.PI).Within(.002));
            Assert.That(b.Snapshot().pose.position.y, Is.EqualTo(2 / Mathf.PI).Within(.002));
        }

        [Test]
        public void WallAndFloorProjectionRespectMountAndTopRowConvention()
        {
            var c = Settings(); var s = Calibration(); var g = new OccupancyGrid(c);
            float pitch = s.camera.mount.pitch_down_deg * Mathf.Deg2Rad;
            var depth = new ushort[80 * 60];
            // Vertical wall at brain y = 2 m; lower rays first hit the floor.
            for (int v = 0; v < 60; v++) for (int u = 0; u < 80; u++)
            {
                float down = (v - s.camera.cy) / s.camera.fy;
                float zWall = (2 - s.camera.mount.forward_offset_m) / (Mathf.Cos(pitch) - down * Mathf.Sin(pitch));
                float zFloor = s.camera.mount.height_m / (Mathf.Sin(pitch) + down * Mathf.Cos(pitch));
                float z = zFloor > 0 ? Mathf.Min(zWall, zFloor) : zWall;
                depth[v * 80 + u] = (ushort)Math.Round(z * 1000);
            }
            g.IntegrateDepth(new RgbdFrame(0, 80, 60, new byte[80 * 60 * 3], depth), s, new BrainPose());
            Assert.That(Enumerable.Range(0, 3).Any(i => g.OccupancyAt(new Vector2(0, 1.95f + i * .05f)) == 255), Is.True);
            Assert.That(g.OccupancyAt(new Vector2(0, 1)), Is.EqualTo(1));
            var floor = new ushort[80 * 60];
            for (int v = 0; v < 60; v++) for (int u = 0; u < 80; u++)
            {
                float denominator = Mathf.Sin(pitch) + (v - s.camera.cy) / s.camera.fy * Mathf.Cos(pitch);
                if (denominator > .2f) floor[v * 80 + u] = (ushort)Math.Round(s.camera.mount.height_m / denominator * 1000);
            }
            var floorGrid = new OccupancyGrid(c);
            floorGrid.IntegrateDepth(new RgbdFrame(0, 80, 60, new byte[80 * 60 * 3], floor), s, new BrainPose());
            Assert.That(floorGrid.CopyOccupancy().Count(v => v == 255), Is.Zero);
        }

        [Test]
        public void AStarGoesAroundWallsAndKeepsFireClearance()
        {
            var c = Settings(); c.grid_size_m = 12;
            var g = OpenGrid(c);
            for (float y = -2; y <= 2; y += .05f) g.ObserveOccupied(g.Cell(new Vector2(0, y)));
            g.ObserveHeat(g.Cell(new Vector2(1, 3)), 600, 0);
            var p = new GridPlanner(g, c).Plan(new Vector2(-3, 0), new Vector2(3, 0));
            Assert.That(p, Is.Not.Empty);
            Assert.That(p.Any(v => Mathf.Abs(v.y) > 2.4f), Is.True);
            Assert.That(p.All(v => Vector2.Distance(v, new Vector2(1, 3)) >= 1.2f), Is.True);
            Assert.That(new GridPlanner(g, c).Plan(new Vector2(-3, 0), new Vector2(1, 3)), Is.Empty);
        }

        [Test]
        public void FrontierSelectionIgnoresFrontiersCutOffByFire()
        {
            var c = Settings(); c.grid_size_m = 12; var g = new OccupancyGrid(c);
            for (float y = -4; y <= 4; y += .05f) for (float x = -3; x <= 3; x += .05f) g.ObserveFree(g.Cell(new Vector2(x, y)));
            for (float x = -3; x <= 3; x += .1f) g.ObserveHeat(g.Cell(new Vector2(x, 1)), 600, 0);
            var planner = new GridPlanner(g, c);
            var choice = planner.SelectFrontier(new Vector2(0, -2));
            Assert.That(choice, Is.Not.Null);
            Assert.That(choice.Value.y, Is.LessThan(0));
            Assert.That(planner.Plan(new Vector2(0, -2), new Vector2(0, 3)), Is.Empty);
        }

        [Test]
        public void FireCuttingOffTheOnlyFrontierIsReportedWhereWaterWouldOpenIt()
        {
            // Agent run 16: a burning wood pile closed a 2 m hallway; every frontier vanished and the agent was never told
            // fire was the cause, so it did not use its hose. A corridor 2 m wide, unmapped beyond y = 4, fire across it at
            // y = 2: no frontier with the fire margin, and the fire-blocked check points straight ahead to where the route
            // enters the margin. Control: a wall across the corridor instead is not a fire block.
            BrainFireBlock? Check(bool fire, out int frontiers)
            {
                var b = new RobotBrain(Calibration(), Settings(), "A", 0); var g = Map(b);
                for (float y = -1; y <= 4; y += .05f)
                {
                    for (float x = -.95f; x <= .95f; x += .05f) g.ObserveFree(g.Cell(new Vector2(x, y)));
                    g.ObserveOccupied(g.Cell(new Vector2(-1.05f, y))); g.ObserveOccupied(g.Cell(new Vector2(1.05f, y)));
                }
                for (float x = -1; x <= 1; x += .05f) g.ObserveOccupied(g.Cell(new Vector2(x, -1.05f)));
                for (float x = -.95f; x <= .95f; x += .1f)
                    if (fire) g.ObserveHeat(g.Cell(new Vector2(x, 2)), 600, 0); else g.ObserveOccupied(g.Cell(new Vector2(x, 2)));
                frontiers = b.FrontierOptions(0, 10).Count;
                return b.FireBlockedFrontier(0);
            }
            var blocked = Check(true, out int withFire);
            Assert.That(withFire, Is.Zero, "the fire margin cuts the corridor");
            Assert.That(blocked.HasValue, Is.True);
            // Where the hose must aim: inside the corridor, between the robot and the fire (not sideways or behind). With
            // frontiers that look past tight strips, the best frontier beyond the fire may sit beside a corridor wall, so
            // the route can meet the margin up to the corridor's half width off the centre line.
            Assert.That(blocked.Value.blockAt.y, Is.InRange(0f, 2f), "ahead, before the fire");
            Assert.That(Mathf.Abs(blocked.Value.blockAt.x), Is.LessThan(1f), "inside the corridor");
            Assert.That(Mathf.Abs(blocked.Value.bearingDegRight), Is.LessThan(Mathf.Atan2(1f, blocked.Value.distanceM) * Mathf.Rad2Deg), "within the corridor's angle from here");
            Assert.That(blocked.Value.distanceM, Is.InRange(.3f, 1.2f), "where the route meets the 1.2 m margin before the fire at 2 m");
            var walled = Check(false, out int withWall);
            Assert.That(withWall, Is.Zero, "control: the wall cuts it too");
            Assert.That(walled.HasValue, Is.False, "control: water does not open a wall");
        }

        [Test]
        public void ThermalBlobsRejectFireAndTwoPixelNoise()
        {
            var s = Calibration(); var c = Settings();
            var detections = ThermalPerception.Detect(Person(0), Image(0, true), s, c, new BrainPose());
            Assert.That(detections.Count, Is.EqualTo(1));
            Assert.That(detections[0].areaM2, Is.InRange(.25f, 2.5f));
            Assert.That(detections[0].position.y, Is.GreaterThan(1.8f));
            Assert.That(ThermalPerception.Detect(Person(0, 600), Image(0, true), s, c, new BrainPose()), Is.Empty);
        }

        [Test]
        public void ThermalSizesAWarmBlobWithLidarRangeWhenSmokeRemovesDepth()
        {
            // Hidden-corner run 2: dense smoke removed nearly all depth returns, so warm blobs could not be sized and
            // thermal confirmed nobody. With no depth at all, the LiDAR range at the blob's bearing sizes it. Controls:
            // no LiDAR, no detection; LiDAR reading a wall 7 m behind (as over a person lying below the scan plane), no
            // detection either.
            var s = Calibration(); var c = Settings();
            LidarScan Ring(float r) => new LidarScan(0, -Mathf.PI, 2 * Mathf.PI / 720, .15f, 12f, 0f, .45f, Enumerable.Repeat(r, 720).ToArray());
            var smoky = Image(0, true, 0);
            Assert.That(ThermalPerception.Detect(Person(0), smoky, s, c, new BrainPose()), Is.Empty, "control: no depth, no LiDAR");
            var found = ThermalPerception.Detect(Person(0), smoky, s, c, new BrainPose(), Ring(2f));
            Assert.That(found.Count, Is.EqualTo(1), "LiDAR range sizes the blob");
            Assert.That(found[0].position.y, Is.InRange(1.6f, 2.6f), "at about the LiDAR's 2 m");
            Assert.That(ThermalPerception.Detect(Person(0), smoky, s, c, new BrainPose(), Ring(7f)), Is.Empty, "control: a wall behind is too large a blob");
        }

        [Test]
        public void CameraSwitchRequiresContinuousHysteresisDurations()
        {
            var b = new RobotBrain(Calibration(), Settings(), "A", 0);
            for (int i = 0; i <= 7; i++) { double t = i / 15.0; b.OnRgbd(Image(t, false)); b.Step(t, 1f / 15); }
            Assert.That(b.Snapshot().mode, Is.EqualTo("rgb"));
            b.OnRgbd(Image(8 / 15.0, false)); b.Step(8 / 15.0, 1f / 15);
            Assert.That(b.Snapshot().mode, Is.EqualTo("thermal"));
            for (int i = 9; i <= 38; i++) { double t = i / 15.0; b.OnRgbd(Image(t, true)); b.Step(t, 1f / 15); }
            Assert.That(b.Snapshot().mode, Is.EqualTo("thermal"));
            b.OnRgbd(Image(39 / 15.0, true)); b.Step(39 / 15.0, 1f / 15);
            Assert.That(b.Snapshot().mode, Is.EqualTo("rgb"));
            var switches = b.DrainEvents().Where(e => e.kind == "sensor_switch").ToArray();
            Assert.That(switches.Length, Is.EqualTo(2));
            Assert.That(switches[1].numbers["corners"], Is.GreaterThan(0));
        }

        [Test]
        public void AirTemperatureRampTriggersRetreat()
        {
            var b = new RobotBrain(Calibration(), Settings(), "A", 0);
            for (int i = 0; i <= 50; i++)
            { double t = i * .1; b.OnAirTemperature(t, 20 + i * 1.1f); b.Step(t, .1f); }
            Assert.That(b.DrainEvents().Any(e => e.kind == "retreat" && e.numbers["air_c"] > 70), Is.True);
            Assert.That(b.Snapshot().state, Is.EqualTo("retreat"));
        }

        [Test]
        public void ARobotInsideAFireMarginCanAlwaysPlanItsWayOut()
        {
            // Agent run 9: robot B ended up inside a keep-out area its own agent marked as fire and froze, since every
            // plan started inside fire. Leaving must always be possible, by steps that never go nearer the fire.
            var c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f; var g = OpenGrid(c);
            for (float x = -.6f; x <= .6f; x += .1f) for (float y = -.2f; y <= 1.2f; y += .1f) g.ObserveHeat(g.Cell(new Vector2(x, y)), 500, 0);
            var planner = new GridPlanner(g, c); var from = new Vector2(.05f, -.05f);
            float ToFire(Vector2 q) => new Vector2(Mathf.Max(0, Mathf.Abs(q.x) - .6f), Mathf.Max(0, Mathf.Max(-.2f - q.y, q.y - 1.2f))).magnitude;
            Assert.That(ToFire(from), Is.Zero, "the robot stands in the hot area");
            foreach (var goal in new[] { new Vector2(.05f, -4f), new Vector2(.05f, 4.5f) })
            {
                var path = planner.Plan(from, goal);
                Assert.That(path, Is.Not.Empty, $"a way out towards {goal}");
                Assert.That(planner.PathSafe(path), Is.True, "and the path survives the safety check on the next step");
                // Once a distance from the fire is reached (up to the clearance), the route never comes back nearer.
                float reached = 0;
                foreach (var q in path)
                {
                    Assert.That(ToFire(q), Is.GreaterThanOrEqualTo(Mathf.Min(reached, c.fire_clearance_m) - .1f), $"{q} goes back towards the fire");
                    reached = Mathf.Max(reached, ToFire(q));
                }
            }
        }

        [Test]
        public void AnAgentKeepOutAreaNeverContainsTheRobot()
        {
            var b = new RobotBrain(Calibration(), Settings(), "A", 0); Room(b, -3, 3, -3, 3);
            Motion(b, 0); b.OnAirTemperature(0, 20); b.Step(0, .05f); b.DrainEvents();
            b.AgentAvoidArea(new Vector2(0, 1.2f), 2f, 0, 30, "ev");
            var e = b.DrainEvents().Single(x => x.kind == "agent_order");
            Assert.That(e.reason, Does.Contain("refused"), "robot 1.2 m from the centre of a 2 m circle");
            b.AgentAvoidArea(new Vector2(0, 4f), 3f, 0, 30, "ev");
            var shrunk = b.DrainEvents().Single(x => x.kind == "agent_order");
            Assert.That(shrunk.numbers["radius_m"], Is.LessThan(4f - Settings().fire_clearance_m - Settings().robot_radius_m + 1e-3), "shrunk to leave the robot outside");
        }

        [Test]
        public void ColourPersonSightingsBecomeACandidateWhereThePersonIs()
        {
            // Thermal saturates in fire rooms (warm_eval: 0 of 161 visible survivors); the colour dnn detector still sees
            // them. A person 2.4 m ahead and 10 degrees to the right, seen twice, must become one candidate there.
            float Spot(float confidence, out int candidates)
            {
                var b = new RobotBrain(Calibration(), Settings(), "A", 0);
                var all = new List<BrainEvent>();
                for (int tick = 0; tick <= 40; tick++)
                {
                    double t = tick / 20.0; Motion(b, t); b.OnAirTemperature(t, 20);
                    if (tick % 20 == 0) b.OnPersonSighting(t, 10f, 2.4f, confidence);
                    b.Step(t, .05f); all.AddRange(b.DrainEvents());
                }
                var events = all.Where(e => e.kind == "survivor_candidate").ToList(); candidates = events.Count;
                if (candidates == 0) return float.NaN;
                // Camera 0.25 m ahead of the body; bearing 10 degrees right.
                var expected = new Vector2(0, .25f) + new Vector2(Mathf.Sin(10 * Mathf.Deg2Rad), Mathf.Cos(10 * Mathf.Deg2Rad)) * 2.4f;
                return Vector2.Distance(new Vector2((float)events[0].numbers["x_m"], (float)events[0].numbers["y_m"]), expected);
            }
            float error = Spot(.8f, out int confident);
            Assert.That(confident, Is.EqualTo(1));
            Assert.That(error, Is.LessThan(.05f), "candidate where the person is");
            Spot(.3f, out int weak);
            Assert.That(weak, Is.Zero, "control: sightings under person_rgb_min_confidence are ignored");
        }

        [Test]
        public void WithAnAgentTheReflexOnlyBringsTheCamerasCloseAndNeverConfirms()
        {
            var s = Calibration(); var b = new RobotBrain(s, Settings(), "A", 4) { AgentConfirms = true };
            var position = Vector2.zero; float yaw = 0, v = 0, w = 0; var all = new List<BrainEvent>();
            for (int tick = 0; tick < 900; tick++)
            {
                double t = tick / 60.0;
                position += new Vector2(-Mathf.Sin(yaw), Mathf.Cos(yaw)) * v / 60; yaw += w * Mathf.Deg2Rad / 60;
                Motion(b, t, v, w * Mathf.Deg2Rad);
                if (tick % 6 == 0) b.OnAirTemperature(t, 20);
                if (tick % 4 == 0) { var pair = SceneImages(t, position, yaw, s); b.OnRgbd(pair.Item1); b.OnThermal(pair.Item2); }
                var command = b.Step(t, 1f / 60); v = command.forwardMps; w = command.yawRateDeg;
                all.AddRange(b.DrainEvents());
            }
            var ready = all.Where(e => e.kind == "survivor_view_ready").ToList();
            Assert.That(ready.Count, Is.EqualTo(1), "one hand-off per warm shape until the agent decides or the retry time passes");
            Assert.That(ready[0].numbers["range_m"], Is.InRange(1.0, 1.5));
            Assert.That(all.Any(e => e.kind == "survivor_confirmed"), Is.False, "control for the test below: only the agent confirms");
        }

        [Test]
        public void SurvivorApproachStopsFacesAndConfirmsOnlyOnce()
        {
            var s = Calibration(); var c = Settings(); var b = new RobotBrain(s, c, "A", 4);
            // A synthetic moving robot observes a stationary person in an otherwise open room.
            var position = Vector2.zero; float yaw = 0, v = 0, w = 0;
            bool candidate = false, confirmed = false; double stoppedSince = -1;
            var all = new List<BrainEvent>();
            for (int tick = 0; tick < 900; tick++)
            {
                double t = tick / 60.0;
                position += new Vector2(-Mathf.Sin(yaw), Mathf.Cos(yaw)) * v / 60;
                yaw += w * Mathf.Deg2Rad / 60;
                Motion(b, t, v, w * Mathf.Deg2Rad);
                if (tick % 6 == 0) b.OnAirTemperature(t, 20);
                if (tick % 4 == 0)
                {
                    var pair = SceneImages(t, position, yaw, s);
                    b.OnRgbd(pair.Item1); b.OnThermal(pair.Item2);
                }
                var command = b.Step(t, 1f / 60); v = command.forwardMps; w = command.yawRateDeg;
                var events = b.DrainEvents(); all.AddRange(events);
                candidate |= events.Any(e => e.kind == "survivor_candidate");
                if (!confirmed && v == 0 && Mathf.Abs(w) < 1 && candidate && Vector2.Distance(position, new Vector2(0, 3)) <= 1.5f)
                { if (stoppedSince < 0) stoppedSince = t; }
                else if (!confirmed) stoppedSince = -1;
                if (events.Any(e => e.kind == "survivor_confirmed"))
                {
                    Assert.That(Vector2.Distance(position, new Vector2(0, 3)), Is.InRange(1f, 1.5f));
                    Assert.That(Mathf.Abs(Mathf.DeltaAngle(yaw * Mathf.Rad2Deg, Mathf.Atan2(position.x, 3 - position.y) * Mathf.Rad2Deg)), Is.LessThan(10));
                    Assert.That(t - stoppedSince, Is.GreaterThanOrEqualTo(2.95));
                    Assert.That(v, Is.Zero); confirmed = true;
                }
            }
            Assert.That(candidate, Is.True); Assert.That(confirmed, Is.True);
            Assert.That(all.Count(e => e.kind == "survivor_confirmed"), Is.EqualTo(1));
        }

        static Tuple<RgbdFrame, ThermalFrame> SceneImages(double t, Vector2 robot, float yaw, SensorConfig s)
        {
            var image = Image(t, true, 6000); var temperatures = Enumerable.Repeat((ushort)2932, 4800).ToArray();
            // Project an upright 0.55 x 1.6 m person in the brain frame by inverting the mount.
            float cp = Mathf.Cos(10 * Mathf.Deg2Rad), sp = Mathf.Sin(10 * Mathf.Deg2Rad);
            var delta = new Vector2(0, 3) - robot;
            float right = delta.x * Mathf.Cos(yaw) + delta.y * Mathf.Sin(yaw);
            float forward = -delta.x * Mathf.Sin(yaw) + delta.y * Mathf.Cos(yaw) - .25f;
            for (int row = 0; row < 60; row++) for (int col = 0; col < 80; col++)
            {
                float down = (row - 29.5f) / 52, rayForward = cp - down * sp;
                if (forward <= .2f || rayForward <= 0) continue;
                float z = forward / rayForward;
                float x = (col - 39.5f) / 52 * z, height = 1.6f - z * (sp + down * cp);
                if (Mathf.Abs(x - right) < .275f && height >= 0 && height <= 1.6f)
                { image.Depth[row * 80 + col] = (ushort)Math.Round(z * 1000); temperatures[row * 80 + col] = 3072; }
            }
            return Tuple.Create(image, new ThermalFrame(t, 80, 60, temperatures));
        }

        [Test]
        public void IdenticalSensorStreamsProduceIdenticalCommandsAndEvents()
        {
            var a = new RobotBrain(Calibration(), Settings(), "A", 17);
            var b = new RobotBrain(Calibration(), Settings(), "A", 17);
            for (int i = 0; i < 80; i++)
            {
                double t = i / 15.0;
                foreach (var brain in new[] { a, b })
                { Motion(brain, t, .1f, .05f); brain.OnRgbd(Image(t, i > 20)); brain.OnThermal(Person(t)); brain.OnAirTemperature(t, 20 + i); }
                var ac = a.Step(t, 1f / 15); var bc = b.Step(t, 1f / 15);
                Assert.That(ac.forwardMps, Is.EqualTo(bc.forwardMps)); Assert.That(ac.yawRateDeg, Is.EqualTo(bc.yawRateDeg));
                var ae = a.DrainEvents(); var be = b.DrainEvents(); Assert.That(ae.Count, Is.EqualTo(be.Count));
                for (int j = 0; j < ae.Count; j++)
                {
                    Assert.That(ae[j].t, Is.EqualTo(be[j].t)); Assert.That(ae[j].robot, Is.EqualTo(be[j].robot));
                    Assert.That(ae[j].kind, Is.EqualTo(be[j].kind)); Assert.That(ae[j].reason, Is.EqualTo(be[j].reason));
                    CollectionAssert.AreEqual(ae[j].numbers, be[j].numbers);
                }
            }
            CollectionAssert.AreEqual(a.Snapshot().occupancy, b.Snapshot().occupancy);
            CollectionAssert.AreEqual(a.Snapshot().heat, b.Snapshot().heat);
        }

        [Test]
        public void RisingHeatIsForbiddenBeforeItCrossesTheFireThreshold()
        {
            var c = Settings(); c.grid_size_m = 12; var g = OpenGrid(c);
            int cell = g.Cell(new Vector2(0, 2));
            g.ObserveHeat(cell, 100, 0); g.DecayTo(.5); g.ObserveHeat(cell, 180, .5);
            Assert.That(g.Heat(cell), Is.LessThan(200));
            Assert.That(new GridPlanner(g, c).Plan(new Vector2(0, 0), new Vector2(0, 2)), Is.Empty);
        }

        [Test]
        public void ANewDepthHitBlocksPreviouslyFreeSpaceAndInvalidatesAPath()
        {
            var c = Settings(); c.grid_size_m = 12; var g = OpenGrid(c);
            var planner = new GridPlanner(g, c); var path = planner.Plan(new Vector2(0, 0), new Vector2(0, 3));
            Assert.That(path, Is.Not.Empty);
            int i = g.Cell(new Vector2(0, 1.5f));
            for (int observation = 0; observation < 30; observation++) g.ObserveFree(i);
            g.ObserveOccupied(i);
            Assert.That(g.Occupancy(i), Is.EqualTo(255));
            Assert.That(planner.PathSafe(path), Is.False);
        }

        [Test]
        public void DelayedDepthIsProjectedAtCapturePose()
        {
            var s = Calibration(); var c = Settings(); c.motion_stale_s = 2; var b = new RobotBrain(s, c, "A", 0);
            Motion(b, 0, 1); b.Step(0, .1f);
            Motion(b, .1, 1); b.Step(.1, .1f);
            b.OnRgbd(Image(0, true)); Motion(b, .2, 1); b.Step(.2, .1f);
            var snapshot = b.Snapshot();
            // Optical centre at 2 m and the pitched mount place the wall around y = 2.2 m at t=0.
            bool hit = false;
            for (int row = 0; row < snapshot.gridSize; row++) for (int col = 0; col < snapshot.gridSize; col++)
            {
                Vector2 p = snapshot.gridOrigin + new Vector2(col + .5f, row + .5f) * snapshot.cellM;
                if (Mathf.Abs(p.x) < .1f && p.y > 2.1f && p.y < 2.3f && snapshot.occupancy[row * snapshot.gridSize + col] == 255) hit = true;
            }
            Assert.That(hit, Is.True);
            Assert.That(snapshot.pose.position.y, Is.EqualTo(.2f).Within(.001));
        }

        [Test]
        public void CameraDropoutResetsThePoorHealthTimer()
        {
            var b = new RobotBrain(Calibration(), Settings(), "A", 0);
            for (int i = 0; i < 7; i++) { double t = i / 15.0; b.OnRgbd(Image(t, false)); b.Step(t, 1f / 15); }
            b.OnRgbd(Image(1, false)); b.Step(1, .6f);
            Assert.That(b.Snapshot().mode, Is.EqualTo("rgb"));
            for (int i = 1; i <= 8; i++) { double t = 1 + i / 15.0; b.OnRgbd(Image(t, false)); b.Step(t, 1f / 15); }
            Assert.That(b.Snapshot().mode, Is.EqualTo("thermal"));
        }

        [Test]
        public void StaleSensorsStopMotionAndCommandsRespectMotionLimits()
        {
            var c = Settings(); c.limits.max_forward_mps = .2f; c.limits.max_yaw_rate_dps = 12;
            var b = new RobotBrain(Calibration(), c, "A", 0);
            float previous = 0;
            for (int i = 0; i < 30; i++)
            {
                double t = i / 15.0; Motion(b, t, previous); b.OnAirTemperature(t, 20); b.OnRgbd(Image(t, true));
                var command = b.Step(t, 1f / 15);
                Assert.That(command.forwardMps, Is.InRange(0, .2f));
                Assert.That(Mathf.Abs(command.yawRateDeg), Is.LessThanOrEqualTo(12));
                if (command.forwardMps > 0) Assert.That(command.forwardMps - previous, Is.LessThanOrEqualTo(c.limits.max_accel_mps2 / 15 + .00001f));
                previous = command.forwardMps;
            }
            var stop = b.Step(3, 3 - 29f / 15);
            Assert.That(stop.forwardMps, Is.Zero); Assert.That(stop.yawRateDeg, Is.Zero);
        }

        [Test]
        public void ARobotLeavesWhileFireIsClosingOnItsWayOut()
        {
            // Agent run 11: robot A searched beside a spreading fire until the air reached 70 C, and its route home was
            // gone by then. Fire 1.6 m beside the way out, 3 m behind the robot, must end the search; control: 3.5 m.
            string StateWithFireAt(Vector2 fire, out int closing)
            {
                var c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f;
                var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -5, 5, -2, 6);
                closing = 0;
                for (int i = 0; i < 90; i++)
                {
                    double t = i / 15.0; Motion(b, t, t < 3 ? 1f : 0); b.OnRgbd(Image(t, true)); b.OnAirTemperature(t, 20);
                    if (t >= 3.2) { var g = Map(b); for (float dx = -.3f; dx <= .3f; dx += .1f) for (float dy = -.3f; dy <= .3f; dy += .1f) g.ObserveHeat(g.Cell(fire + new Vector2(dx, dy)), 500, t); }
                    b.Step(t, 1f / 15); closing += b.DrainEvents().Count(e => e.kind == "escape_closing");
                }
                return b.Snapshot().state;
            }
            Assert.That(StateWithFireAt(new Vector2(1.9f, .8f), out int near), Is.EqualTo("return"));
            Assert.That(near, Is.EqualTo(1));
            Assert.That(StateWithFireAt(new Vector2(4.2f, .8f), out int far), Is.Not.EqualTo("return"), "control: fire 3.5 m from the way out");
            Assert.That(far, Is.Zero);
        }

        [Test]
        public void ARobotLeavesEarlyWhenFireClosesOnItsWayOutFasterThanItCouldWalkIt()
        {
            // Run 21: the fixed 2 m margin fired 2.4 s before robot B's hallway route closed, and it was trapped for 137 s.
            // A fire front moving toward the route at 0.3 m/s must start the return while it is still beyond 2 m, from the
            // closing rate the robot measures on its own map. Control: escape_time_factor 0 leaves only the 2 m margin.
            float FireDistanceAtLeave(float factor, bool jump = false)
            {
                var c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f; c.escape_time_factor = factor;
                var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -5, 5, -2, 6);
                for (int i = 0; i < 225; i++)
                {
                    double t = i / 15.0; Motion(b, t, t < 3 ? 1f : 0); b.OnRgbd(Image(t, true)); b.OnAirTemperature(t, 20);
                    if (t >= 3.2)
                    {
                        var g = Map(b); float x = jump ? (t < 6 ? 4.8f : 3.8f) : 5.0f - .3f * (float)(t - 3.2);
                        for (float dx = -.3f; dx <= .3f; dx += .1f) for (float dy = -.3f; dy <= .3f; dy += .1f) g.ObserveHeat(g.Cell(new Vector2(x + dx, .8f + dy)), 500, t);
                    }
                    b.Step(t, 1f / 15);
                    var leave = b.DrainEvents().FirstOrDefault(e => e.kind == "escape_closing");
                    if (leave != null) return (float)leave.numbers["fire_distance_m"];
                }
                return -1;
            }
            float early = FireDistanceAtLeave(1.5f), late = FireDistanceAtLeave(0f);
            Assert.That(FireDistanceAtLeave(1.5f, jump: true), Is.EqualTo(-1f), "control: a nearer fire seen once (one 1 m jump), then static");
            Assert.That(early, Is.GreaterThan(2.3f), "left on the closing rate, well before the 2 m margin");
            Assert.That(late, Is.InRange(0f, 2.0f), "control: without the rate only the 2 m margin acts");
        }

        [Test]
        public void WalkingTowardAStillFireIsNotAClosingFront()
        {
            // Codex review: comparing fire distance on routes that change as the robot moves made a robot walking nearer a
            // still fire read the shrinking distance as a closing front. The rate is now measured at fixed route points.
            // The robot walks 4 m up a room past a fire that never moves (closest approach 2.5 m beyond the 2 m margin) and
            // must not leave. The control is the moving-front test above, which must still leave early.
            var c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f;
            var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -5, 5, -2, 6);
            int closing = 0;
            for (int i = 0; i < 180; i++)
            {
                double t = i / 15.0; Motion(b, t, t < 8 ? .5f : 0); b.OnRgbd(Image(t, true)); b.OnAirTemperature(t, 20);
                var g = Map(b);
                for (float dx = -.3f; dx <= .3f; dx += .1f) for (float dy = -.3f; dy <= .3f; dy += .1f) g.ObserveHeat(g.Cell(new Vector2(2.8f + dx, 4.5f + dy)), 500, t);
                b.Step(t, 1f / 15); closing += b.DrainEvents().Count(e => e.kind == "escape_closing");
            }
            Assert.That(closing, Is.Zero);
        }

        [Test]
        public void AReturnCutByFireBecomesARetreatNotAStandstill()
        {
            // Agent run 12: robot A started home as fire closed on its route, the route was cut a moment later, and it
            // held still for the rest of the run. It must retreat to cool floor (and search on) instead.
            var c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f;
            var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -5, 5, -2, 6);
            var events = new List<BrainEvent>();
            for (int i = 0; i < 120; i++)
            {
                double t = i / 15.0; Motion(b, t, t < 3 ? 1f : 0); b.OnRgbd(Image(t, true)); b.OnAirTemperature(t, 20);
                var g = Map(b);
                if (t >= 3.2) for (float dx = -.3f; dx <= .3f; dx += .1f) for (float dy = -.3f; dy <= .3f; dy += .1f) g.ObserveHeat(g.Cell(new Vector2(1.9f + dx, .8f + dy)), 500, t);
                if (events.Any(e => e.kind == "goal" && e.reason.StartsWith("The way out was closing")))
                    for (float x = -5; x <= 5; x += .1f) g.ObserveHeat(g.Cell(new Vector2(x, 1.5f)), 500, t);   // then fire cuts the room
                b.Step(t, 1f / 15); events.AddRange(b.DrainEvents());
            }
            Assert.That(events.Count(e => e.kind == "escape_closing"), Is.EqualTo(2), "closing, then lost");
            Assert.That(b.Snapshot().state, Is.Not.EqualTo("return"), "no standstill on a cut route");
        }

        [Test]
        public void AFrontierWhoseRouteTheFireCutIsLeftAloneWhileTheFlamesFlicker()
        {
            // Held-out run seed 914, robot A at 87 to 94 s: the only frontier, past a burning doorway, was chosen, its path cut
            // by the fire margin 1.5 s later and chosen again as the flames flickered, three times in 4 s; no_frontier kept
            // restarting and the scan never ended. Now the cut frontier is left for fire_cut_blacklist_s. Control: with that
            // time 0 the same frontier is chosen again as soon as the fire reads cool.
            string lastLog = "";
            (int chosenAfterCut, int blacklisted, int noFrontier) Run(float blacklistS)
            {
                var c = Settings(); c.grid_size_m = 12; c.fire_cut_blacklist_s = blacklistS;
                var b = new RobotBrain(Calibration(), c, "A", 0); var g = Map(b);
                for (int i = 0; i < g.Count; i++)
                {
                    var p = g.Centre(i);
                    bool pocket = Mathf.Abs(p.x) < .3f && p.y > 3f && p.y < 3.6f;                  // the only unsearched floor
                    bool wall = (Mathf.Abs(Mathf.Abs(p.x) - .7f) < .06f && p.y > .6f && p.y < 4.4f) || (Mathf.Abs(p.x) < .75f && Mathf.Abs(p.y - 4.4f) < .06f);
                    if (wall) g.ObserveOccupied(i); else if (!pocket) g.ObserveFree(i);          // at the end of a 1.3 m corridor
                }
                var events = new List<BrainEvent>(); double cutAt = double.NaN; bool cooled = false;
                for (int k = 0; k <= 15 * 14; k++)
                {
                    double t = k / 15.0;
                    Motion(b, t); b.OnRgbd(Image(t, true, 0)); b.OnAirTemperature(t, 20);
                    bool chosen = events.Any(e => e.kind == "goal" && e.reason.StartsWith("Reachable frontier"));
                    if (chosen && double.IsNaN(cutAt)) cutAt = t;
                    if (!double.IsNaN(cutAt) && t < cutAt + 2.5)          // fire on the route, then it reads cool (flicker)
                        for (float x = -.3f; x <= .3f; x += .1f) g.ObserveHeat(g.Cell(new Vector2(x, 1.6f)), 500, t);
                    else if (!double.IsNaN(cutAt) && t >= cutAt + 5 && !cooled)
                    { cooled = true; for (float x = -.3f; x <= .3f; x += .1f) g.ObserveHeat(g.Cell(new Vector2(x, 1.6f)), 20, t); }
                    b.Step(t, 1f / 15); events.AddRange(b.DrainEvents());
                }
                Assert.That(double.IsNaN(cutAt), Is.False, "the frontier was chosen");
                lastLog = string.Join("; ", events.Where(e => e.kind == "goal_blacklisted" || (e.kind == "goal" && e.reason.StartsWith("Reachable frontier")) || e.kind == "no_frontier")
                    .Select(e => $"{e.t:F1} {e.kind} {(e.numbers != null && e.numbers.ContainsKey("x_m") ? $"({e.numbers["x_m"]:F2},{e.numbers["y_m"]:F2})" : "")}"));
                return (events.Count(e => e.t > cutAt + 5 && e.kind == "goal" && e.reason.StartsWith("Reachable frontier")),
                        events.Count(e => e.kind == "goal_blacklisted" && e.reason.StartsWith("The fire margin cut")),
                        events.Count(e => e.kind == "no_frontier"));
            }
            var left = Run(20);
            Assert.That(left.blacklisted, Is.EqualTo(1), "the cut frontier is left alone once: " + lastLog);
            Assert.That(left.chosenAfterCut, Is.Zero, "and not chosen again while the flames flicker: " + lastLog);
            Assert.That(left.noFrontier, Is.EqualTo(1), "one no_frontier, one scan");
            var control = Run(0);
            Assert.That(control.chosenAfterCut, Is.GreaterThan(0), "control: with no exclusion it is chosen again once the fire reads cool");
        }

        [Test]
        public void CutOffInWarmAirTheRobotTakesALastResortRouteWhenTheEmergencyOneIsShut()
        {
            // Held-out seed 914, robot B: on its emergency exit the route was shut at the 0.5 m emergency clearance and it
            // stood still for 15 s while its air rose from 26 to 52 C. Fire cuts the room leaving a 0.9 m gap: shut at 0.5 m,
            // open with only the body clear (0.25 m). With heat building it now leaves through the gap. Control: with the
            // last-resort clearance equal to the emergency one there is no route and it retreats, as before.
            (int exits, int retreats, string state) Run(float lastResort)
            {
                var c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f; c.last_resort_fire_clearance_m = lastResort;
                var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -5, 5, -2, 6);
                var events = new List<BrainEvent>(); double cutAt = double.NaN;
                for (int i = 0; i < 150; i++)
                {
                    double t = i / 15.0;
                    if (double.IsNaN(cutAt) && events.Any(e => e.kind == "goal" && e.reason.StartsWith("The way out was closing"))) cutAt = t;
                    Motion(b, t, t < 3 ? 1f : 0); b.OnRgbd(Image(t, true)); b.OnAirTemperature(t, double.IsNaN(cutAt) ? 20 : 35);
                    var g = Map(b);
                    if (t >= 3.2) for (float dx = -.3f; dx <= .3f; dx += .1f) for (float dy = -.3f; dy <= .3f; dy += .1f) g.ObserveHeat(g.Cell(new Vector2(1.9f + dx, .8f + dy)), 500, t);
                    if (!double.IsNaN(cutAt))
                        for (float x = -5; x <= 5; x += .05f) if (Mathf.Abs(x) >= .45f) g.ObserveHeat(g.Cell(new Vector2(x, 1.5f)), 500, t);
                    b.Step(t, 1f / 15); events.AddRange(b.DrainEvents());
                }
                Assert.That(double.IsNaN(cutAt), Is.False, "the way out started closing");
                return (events.Count(e => e.kind == "emergency_exit"), events.Count(e => e.kind == "escape_closing" && e.reason.StartsWith("The known way out is closed; retreat")),
                    b.Snapshot().state);
            }
            var through = Run(.25f);
            Assert.That((through.exits, through.retreats, through.state), Is.EqualTo((1, 0, "return")), "leaves through the gap with only the body clear");
            var control = Run(.5f);
            Assert.That(control.exits, Is.Zero, "control: no route at 0.5 m");
            Assert.That(control.retreats, Is.EqualTo(1), "control: a retreat, as before");
        }

        [Test]
        public void CutOffInWarmAirWithNoRouteOutTheRobotNeitherFlipsNorRefusesMoves()
        {
            // Held-out seed 914 run 4, robot B: way out closed by fire, no emergency route, air in the 30 to 35 C band (heat
            // "building" from air_warm_c, yet cool enough to end a retreat below heat_recover_below_c). Every two ticks it
            // started a return (stay_overridden), found the way out closed (escape_closing, retreat), ended the retreat as
            // cool, and started again: 360 returns in 45 s, and every agent move was refused as "the robot is leaving" while
            // it could not leave. Now a cut-off robot never starts a normal return; it retries the emergency exit every
            // escape_check_s and agent moves stay allowed.
            var c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f; c.last_resort_fire_clearance_m = c.emergency_fire_clearance_m;
            var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -5, 5, -2, 6);
            var events = new List<BrainEvent>(); double cutAt = double.NaN;
            for (int i = 0; i < 15 * 40; i++)
            {
                double t = i / 15.0;
                if (double.IsNaN(cutAt) && events.Any(e => e.kind == "goal" && e.reason.StartsWith("The way out was closing"))) cutAt = t;
                Motion(b, t, t < 3 ? 1f : 0); b.OnRgbd(Image(t, true)); b.OnAirTemperature(t, double.IsNaN(cutAt) ? 20 : 32);
                var g = Map(b);
                if (t >= 3.2) for (float dx = -.3f; dx <= .3f; dx += .1f) for (float dy = -.3f; dy <= .3f; dy += .1f) g.ObserveHeat(g.Cell(new Vector2(1.9f + dx, .8f + dy)), 500, t);
                if (!double.IsNaN(cutAt))
                    for (float x = -5; x <= 5; x += .05f) if (Mathf.Abs(x) >= .45f) g.ObserveHeat(g.Cell(new Vector2(x, 1.5f)), 500, t);
                b.Step(t, 1f / 15); events.AddRange(b.DrainEvents());
            }
            Assert.That(double.IsNaN(cutAt), Is.False, "the way out started closing");
            Assert.That(events.Count(e => e.kind == "escape_closing" && e.reason.StartsWith("The known way out is closed")), Is.LessThanOrEqualTo(2), "cut off once, not every two ticks");
            Assert.That(events.Count(e => e.kind == "stay_overridden"), Is.LessThanOrEqualTo(1), "no return started toward a closed way out");
            var refusal = typeof(RobotBrain).GetMethod("AgentMoveRefusal", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            Assert.That(refusal.Invoke(b, new object[] { 40.0 }), Is.Null, "a robot that cannot leave may still be moved by its agent");
        }

        [Test]
        public void ARobotCutOffIsToldWhereTheFireBlocksItsWayOut()
        {
            // Held-out seed 914 run 5, robot A: fire cut the hallway to its only known way out, its emergency route shut even at
            // the last-resort clearance, and it held at the coolest floor while its air rose from 40 to 185 C. Its agent was
            // told where fire blocks routes to unsearched floor, never where it blocks the way out, though the robot carries
            // a hose. Now the brain plans the way out ignoring the fire margin and reports where that route first enters it.
            // Control: with the way out open there is nothing to report.
            var c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f;
            var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -5, 5, -2, 6);
            for (int i = 0; i <= 45; i++) { double t = i / 15.0; Motion(b, t, 1f); b.OnRgbd(Image(t, true)); b.OnAirTemperature(t, 20); b.Step(t, 1f / 15); b.DrainEvents(); }
            var g = Map(b); var prepare = typeof(OccupancyGrid).GetMethod("PrepareNavigation", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            prepare.Invoke(g, null);
            Assert.That(b.FireBlockedExit(3.0).HasValue, Is.False, "control: the way out is open");
            for (float x = -5; x <= 5; x += .05f) for (float y = 1.4f; y <= 1.6f; y += .05f) g.ObserveHeat(g.Cell(new Vector2(x, y)), 500, 3.0);
            prepare.Invoke(g, null);
            var block = b.FireBlockedExit(3.0);
            Assert.That(block.HasValue, Is.True, "the fire line cuts the only way out");
            Assert.That(Mathf.Abs(block.Value.bearingDegRight), Is.GreaterThan(135f), "behind the robot, toward its entry");
            Assert.That(block.Value.distanceM, Is.LessThan(3f));
        }

        [Test]
        public void OnAnEmergencyExitOnlyMeasuredFireShutsTheDoorNotAFastRisingReading()
        {
            // Held-out seed 914, robot B: in the hot gas filling the foyer its thermal camera read the floor of the only door
            // out at 60 to 187 C and rising fast; the 1 s fire forecast marked the doorway about to ignite, the route home
            // opened and shut every second and the robot stood still for 50 s (185 C peak). On an emergency exit only
            // measured fire closes a cell. Controls: the same doorway shuts a normal plan (forecast on), and a doorway
            // measured at fire_c stays shut on the emergency exit too.
            var c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f;
            int Route(float doorC, bool emergency)
            {
                var g = OpenGrid(c);
                for (float x = -6; x <= 6; x += .05f) if (Mathf.Abs(x) > .55f) for (float y = 1.9f; y <= 2.1f; y += .05f) g.ObserveOccupied(g.Cell(new Vector2(x, y)));   // wall across the grid, 1.1 m door
                for (float x = -.4f; x <= .4f; x += .1f) for (float y = 1.7f; y <= 2.3f; y += .1f)
                {
                    int cell = g.Cell(new Vector2(x, y));
                    g.ObserveHeat(cell, doorC - 35, 10.0); g.ObserveHeat(cell, doorC, 10.2);   // rising 175 C/s
                }
                g.DecayTo(10.2);
                var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
                if (emergency)
                {
                    typeof(OccupancyGrid).GetProperty("FirePrediction", flags).SetValue(g, false);
                    typeof(OccupancyGrid).GetProperty("FireClearance", flags).SetValue(g, c.emergency_fire_clearance_m);
                }
                return new GridPlanner(g, c).Plan(new Vector2(0, 3.5f), Vector2.zero).Count;
            }
            Assert.That(Route(185, false), Is.Zero, "control: a normal plan treats the fast-rising doorway as fire");
            Assert.That(Route(185, true), Is.GreaterThan(0), "on an emergency exit a 185 C reading does not shut the door");
            Assert.That(Route(260, true), Is.Zero, "control: a doorway measured above fire_c stays shut");
        }

        [Test]
        public void AWarmPatchOnALongWallIsFlaggedAndNotApproachedButABodyIs()
        {
            // Held-out seed 914: walls heated by the burning hallway made a stream of thermal-only "person" candidates on the
            // wall lines, and the robots drove over to each; over 15 runs 150 of 151 such candidates were walls. The map
            // extent under the spot tells a wall (runs on) from a body (a compact blob). Controls: a compact 1 m blob is
            // not flagged; the same 4 m wall line is.
            var c = Settings(); c.grid_size_m = 12; var g = OpenGrid(c);
            for (float x = -2; x <= 2; x += .05f) g.ObserveOccupied(g.Cell(new Vector2(x, 3)));          // a 4 m wall
            for (float x = 2.6f; x <= 3.4f; x += .1f) for (float y = -1.3f; y <= -.7f; y += .1f) g.ObserveOccupied(g.Cell(new Vector2(x, y)));   // a body-sized blob
            Assert.That(g.OccupiedExtent(new Vector2(0, 3.05f), .3f, 3f), Is.GreaterThanOrEqualTo(c.candidate_wall_extent_m), "on the wall");
            Assert.That(g.OccupiedExtent(new Vector2(3, -1), .3f, 3f), Is.LessThan(c.candidate_wall_extent_m), "control: a compact blob");
            Assert.That(g.OccupiedExtent(new Vector2(0, 1.5f), .3f, 3f), Is.Zero, "nothing occupied near open floor");
        }

        [Test]
        public void AfterLeavingForHeatTheRobotWaitsBeforeGoingBackAndLeavesAgainAtTheFirstSign()
        {
            // Held-out seed 914 run 1: robot A re-entered 0.1 s after its emergency exit, into the hallway it had just left
            // with its air at 34 C and rising. After a heat exit it now waits reenter_after_heat_s in cool air; once back in,
            // heat building makes it leave at once. Control: with no wait it goes straight back in.
            (double reenterAt, bool leftAgain) Run(float wait)
            {
                var c = Settings(); c.grid_size_m = 12; c.reenter_after_heat_s = wait;
                var b = new RobotBrain(Calibration(), c, "A", 0); var g = Map(b);
                for (int i = 0; i < g.Count; i++) { var p = g.Centre(i); if (!(Mathf.Abs(p.x) < .6f && p.y > 3f && p.y < 4f)) g.ObserveFree(i); }   // one unsearched pocket
                var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Static;
                string heatLeave = (string)typeof(RobotBrain).GetField("HeatLeave", flags).GetValue(null);
                var begin = typeof(RobotBrain).GetMethod("BeginReturn", flags);
                double reenterAt = double.NaN; bool left = false;
                for (int k = 0; k <= 15 * 40; k++)
                {
                    double t = k / 15.0;
                    if (k == 8) begin.Invoke(b, new object[] { t, heatLeave });   // left for heat, standing at its start
                    float air = !double.IsNaN(reenterAt) && t > reenterAt + 1 ? 35 : 20;   // warm air once back in
                    Motion(b, t); b.OnRgbd(Image(t, true, 0)); b.OnAirTemperature(t, air); b.Step(t, 1f / 15);
                    foreach (var e in b.DrainEvents())
                    {
                        if (e.kind == "reenter" && double.IsNaN(reenterAt)) reenterAt = e.t;
                        if (e.kind == "heat_again") left = true;
                    }
                }
                return (reenterAt, left);
            }
            var waits = Run(30);
            Assert.That(waits.reenterAt, Is.GreaterThanOrEqualTo(30.5).Or.NaN, "no re-entry within 30 s of the heat exit");
            var control = Run(0);
            Assert.That(control.reenterAt, Is.LessThan(5), "control: with no wait it goes straight back in");
            Assert.That(control.leftAgain, Is.True, "once back in, warm air makes it leave again");
        }

        [Test]
        public void AnAgentCannotKeepARobotInOnceNoFrontierIsLeftAndTheAirIsHeating()
        {
            // Held-out run 2, robot A: no reachable frontier at 89.8 s; its agent ordered it on twice while its air rose from
            // 24 to 39 C, its way out then closed and it spent 77 s above 60 C. With no frontier left, heat building ends
            // the agent's order to stay and the robot leaves; a new order is then refused with the reason. Control: the
            // same order in steady 21 C air keeps the robot in.
            (List<BrainEvent> events, string state, string next) Run(bool heating)
            {
                var c = Settings(); c.grid_size_m = 12;
                var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -6, 6, -6, 6);   // all searched: nothing left to explore
                var all = new List<BrainEvent>(); var command = new BrainCommand(); double orderedAt = double.NaN;
                for (int tick = 0; tick <= 240; tick++)
                {
                    double t = tick / 20.0;
                    float air = heating && t > orderedAt + 2 ? Mathf.Min(45, 21 + 1.2f * (float)(t - orderedAt - 2)) : 21;   // 72 C/min
                    Motion(b, t, command.forwardMps, command.yawRateDeg * Mathf.Deg2Rad);
                    b.OnRgbd(Image(t, true, 0)); b.OnAirTemperature(t, air); command = b.Step(t, .05f);
                    all.AddRange(b.DrainEvents());
                    if (double.IsNaN(orderedAt) && all.Any(e => e.kind == "no_frontier"))
                    { b.AgentGoTo(new Vector2(4.5f, 4.5f), t, 60, "closer look at a warm shape"); orderedAt = t; all.AddRange(b.DrainEvents()); }
                }
                b.AgentGoTo(new Vector2(-2, -2), 12.05, 30, "another look");
                var next = b.DrainEvents().Last(e => e.kind == "agent_order").reason;
                Assert.That(orderedAt, Is.LessThan(2), "no frontier was found and the agent's order went in");
                return (all, b.Snapshot().state, next);
            }
            var hot = Run(true);
            Assert.That(hot.events.Count(e => e.kind == "stay_overridden"), Is.EqualTo(1), "the order to stay is overridden once");
            Assert.That(hot.state, Is.EqualTo("return").Or.EqualTo("complete"), "it leaves");
            Assert.That(hot.next, Does.StartWith("Refused go_to: the robot is leaving"));
            var cool = Run(false);
            Assert.That(cool.events.Any(e => e.kind == "stay_overridden"), Is.False, "control: steady air");
            Assert.That(cool.state, Is.EqualTo("explore"), "control: the agent keeps it in");
            Assert.That(cool.next, Does.StartWith("go_to: another look"));
        }

        [Test]
        public void CutOffWithTheAirHeatingTheRobotLeavesByTheEmergencyRouteAtOnce()
        {
            // Held-out run 2, robot A: its way out closed at 114 s with the air at 39 C and rising; it retreated "to search on
            // while it is cool", the hot gas reached it 9 s later, and it left only with its heat budget spent (299 C).
            // Cut off with heat building, the robot now takes the emergency route (past the fire at reduced clearance, never
            // through it) at once. Controls: in steady 20 C air the cut is a retreat, as before; when the air then warms
            // during the retreat, it takes the emergency route at that point.
            (int exits, int retreats, string state) Run(float airAtCut, float airLater)
            {
                var c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f;
                var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -5, 5, -2, 6);
                var events = new List<BrainEvent>(); double cutAt = double.NaN;
                for (int i = 0; i < 150; i++)
                {
                    double t = i / 15.0;
                    if (double.IsNaN(cutAt) && events.Any(e => e.kind == "goal" && e.reason.StartsWith("The way out was closing"))) cutAt = t;
                    float air = double.IsNaN(cutAt) ? 20 : t < cutAt + 3 ? airAtCut : airLater;
                    Motion(b, t, t < 3 ? 1f : 0); b.OnRgbd(Image(t, true)); b.OnAirTemperature(t, air);
                    var g = Map(b);
                    if (t >= 3.2) for (float dx = -.3f; dx <= .3f; dx += .1f) for (float dy = -.3f; dy <= .3f; dy += .1f) g.ObserveHeat(g.Cell(new Vector2(1.9f + dx, .8f + dy)), 500, t);
                    if (!double.IsNaN(cutAt))   // fire cuts the room, leaving a 1.4 m gap: shut at the normal clearance, open at the emergency one
                        for (float x = -5; x <= 5; x += .1f) if (Mathf.Abs(x) >= .7f) g.ObserveHeat(g.Cell(new Vector2(x, 1.5f)), 500, t);
                    b.Step(t, 1f / 15); events.AddRange(b.DrainEvents());
                }
                Assert.That(double.IsNaN(cutAt), Is.False, "the way out started closing");
                return (events.Count(e => e.kind == "emergency_exit"), events.Count(e => e.kind == "escape_closing" && e.reason.StartsWith("The known way out is closed; retreat")),
                    b.Snapshot().state);
            }
            var warm = Run(35, 35);
            Assert.That((warm.exits, warm.retreats, warm.state), Is.EqualTo((1, 0, "return")), "warm air when cut off: emergency route at once");
            var cool = Run(20, 20);
            Assert.That((cool.exits, cool.retreats), Is.EqualTo((0, 1)), "control: steady cool air, a retreat as before");
            Assert.That(cool.state, Is.Not.EqualTo("return"));
            var later = Run(20, 35);
            Assert.That((later.exits, later.retreats, later.state), Is.EqualTo((1, 1, "return")), "control: retreat, then the emergency route when the air warms");
        }

        [Test]
        public void ArrivingWhereTheAgentAskedTellsTheAgentWhy()
        {
            // Agent run 14: robot A reached its closer look 1.4 m from a person it had confirmed, within the arm's reach,
            // and walked on, because no event asked the agent to act there. Arrival at a go_to raises agent_goal_reached
            // with the agent's reason. Control: reaching a frontier the agent picked is routine and raises none.
            List<BrainEvent> Drive(bool frontier)
            {
                var b = new RobotBrain(Calibration(), Settings(), "A", 0); Room(b, -4, 4, -4, 4);
                if (frontier) b.AgentExplore(new Vector2(0, 1.5f), 0, 30, "frontier 2");
                else b.AgentGoTo(new Vector2(0, 1.5f), 0, 30, "closer look at a person to lift");
                var all = new List<BrainEvent>(); var command = new BrainCommand();
                for (int tick = 0; tick < 300; tick++)
                {
                    command = DriveTick(b, tick, command.forwardMps, command.yawRateDeg); all.AddRange(b.DrainEvents());
                }
                return all;
            }
            var goTo = Drive(false);
            Assert.That(goTo.Count(e => e.reason == "Reached the agent's goal."), Is.EqualTo(1), "the robot got there");
            var arrived = goTo.Where(e => e.kind == "agent_goal_reached").ToList();
            Assert.That(arrived, Has.Count.EqualTo(1));
            Assert.That(arrived[0].reason, Does.EndWith("closer look at a person to lift"));
            var explore = Drive(true);
            Assert.That(explore.Count(e => e.reason == "Reached the agent's goal."), Is.EqualTo(1), "control: the robot got there too");
            Assert.That(explore.Count(e => e.kind == "agent_goal_reached"), Is.Zero);
        }

        [Test]
        public void AnAgentGoToWithAFacePointTurnsToFaceItBeforeReportingArrival()
        {
            // Run 20: robot A stood within the arm's reach of a confirmed person but looked at them 106 to 148 degrees off
            // its heading, outside the camera's view, so every inspection was uncertain and the lift was refused. A go_to
            // with a face point turns in place on arrival and reports arrival only when facing it. Control: without a face
            // point arrival is reported at once and the heading does not change.
            (List<BrainEvent> events, float yawErrorDeg, float yawDeg) Drive(Vector2? face)
            {
                var b = new RobotBrain(Calibration(), Settings(), "A", 0); Room(b, -4, 4, -4, 4);
                b.AgentGoTo(new Vector2(0, 1.0f), 0, 30, "person to lift", face);
                var all = new List<BrainEvent>(); var command = new BrainCommand();
                for (int tick = 0; tick < 300; tick++)
                {
                    command = DriveTick(b, tick, command.forwardMps, command.yawRateDeg); var events = b.DrainEvents(); all.AddRange(events);
                    if (events.Any(e => e.kind == "agent_goal_reached")) break;   // the heading at the moment arrival is reported
                }
                var pose = b.Snapshot().pose;
                float error = face.HasValue ? Mathf.DeltaAngle(pose.yawRad * Mathf.Rad2Deg, Mathf.Atan2(-(face.Value.x - pose.position.x), face.Value.y - pose.position.y) * Mathf.Rad2Deg) : 0;
                return (all, error, pose.yawRad * Mathf.Rad2Deg);
            }
            var faced = Drive(new Vector2(2.5f, 0.5f));   // behind and to the right of the robot on arrival
            var arrived = faced.events.Where(e => e.kind == "agent_goal_reached").ToList();
            Assert.That(arrived, Has.Count.EqualTo(1));
            Assert.That(arrived[0].reason, Does.StartWith("Arrived where the agent asked and turned to face"));
            Assert.That(Mathf.Abs(faced.yawErrorDeg), Is.LessThanOrEqualTo(12f), "facing the named point");
            int reachedAt = faced.events.FindIndex(e => e.reason == "Reached the agent's goal."), reportedAt = faced.events.IndexOf(arrived[0]);
            Assert.That(reachedAt, Is.GreaterThanOrEqualTo(0).And.LessThan(reportedAt), "arrival is reported after the turn");
            var plain = Drive(null);
            Assert.That(plain.events.Count(e => e.kind == "agent_goal_reached" && e.reason.StartsWith("Arrived where the agent asked: ")), Is.EqualTo(1), "control");
            Assert.That(Mathf.Abs(Mathf.DeltaAngle(plain.yawDeg, 0)), Is.LessThan(20f), "control: no turn toward anything");
        }

        static void WalledRoom(OccupancyGrid g, float half, System.Func<Vector2, bool> mapped = null, System.Func<Vector2, bool> unknown = null)
        {
            for (int i = 0; i < g.Count; i++)
            {
                var p = g.Centre(i);
                bool inside = Mathf.Abs(p.x) <= half && Mathf.Abs(p.y) <= half;
                bool wall = !inside && Mathf.Abs(p.x) <= half + .1f && Mathf.Abs(p.y) <= half + .1f;
                if (wall) { if (mapped == null || mapped(p)) g.ObserveOccupied(i); }
                else if (inside && (unknown == null || !unknown(p))) g.ObserveFree(i);
            }
        }

        [Test]
        public void FloorBehindALowDeskIsAFrontierByTheLive3DMapButNotBehindATallCabinet()
        {
            // The live 3D map (OpenCV 5 TSDF) gives each cell's top height; the camera at 1.6 m can look over a 0.75 m desk
            // but not over a 2 m cabinet. A corner closed off completely by furniture (no gap): with the desk's height
            // known, the floor behind it is a frontier to look at; with a cabinet's height it is not; with no height
            // known (no 3D map yet) it is not either.
            bool Offered(ushort? topCm)
            {
                var b = new RobotBrain(Calibration(), Settings(), "A", 0); var g = Map(b);
                bool Pocket(Vector2 p) => p.x > 2.05f && p.y > 2.05f;
                WalledRoom(g, 3f, null, Pocket);
                var heights = new ushort[g.Count];
                for (int i = 0; i < g.Count; i++)
                {
                    var p = g.Centre(i);
                    bool bar = (Mathf.Abs(p.y - 2.0f) <= .05f && p.x >= 1.95f && p.x <= 3.05f) || (Mathf.Abs(p.x - 2.0f) <= .05f && p.y >= 1.95f && p.y <= 3.05f);
                    if (bar) { g.ObserveOccupied(i); if (topCm.HasValue) heights[i] = topCm.Value; }
                }
                b.OnTopHeights(heights);
                return b.FrontierOptions(0, 6).Any(f => Vector2.Distance(f.position, new Vector2(1.6f, 1.6f)) <= 1.0f);
            }
            Assert.That(Offered(75), Is.True, "a 0.75 m desk: look over it");
            Assert.That(Offered(200), Is.False, "control: a 2 m cabinet blocks the view");
            Assert.That(Offered(null), Is.False, "control: no height known");
        }

        [Test]
        public void UnsearchedFloorBehindFurnitureThroughAGapTooTightToStandInIsAFrontier()
        {
            // Hidden-corner run: a woman behind a desk across a kitchen corner; no standable cell touched her floor, so no
            // frontier was offered and the robot finished. A standable cell that looks through a strip too tight for the
            // body counts. Controls: the same pocket already searched gives no frontier there, and frontier_peek_m 0
            // (the old rule) gives none either.
            List<BrainFrontier> Options(bool pocketSearched, float peek)
            {
                var settings = Settings(); settings.frontier_peek_m = peek;
                var b = new RobotBrain(Calibration(), settings, "A", 0); var g = Map(b);
                bool Pocket(Vector2 p) => p.x > 2.05f && p.y > 2.05f;
                WalledRoom(g, 3f, null, pocketSearched ? null : (System.Func<Vector2, bool>)Pocket);
                for (int i = 0; i < g.Count; i++)
                {
                    var p = g.Centre(i);
                    bool bar = (Mathf.Abs(p.y - 2.0f) <= .05f && p.x >= 1.95f && p.x <= 2.6f) || (Mathf.Abs(p.x - 2.0f) <= .05f && p.y >= 1.95f && p.y <= 3.05f);
                    if (bar) g.ObserveOccupied(i);   // an L of furniture; a 0.4 m gap to the east wall, narrower than the body
                }
                return b.FrontierOptions(0, 6).ToList();
            }
            var open = Options(false, 1.2f);
            Assert.That(open.Any(f => Vector2.Distance(f.position, new Vector2(2.75f, 1.7f)) <= 1.0f), Is.True,
                "a frontier at the gap: " + string.Join(", ", open.Select(f => f.position.ToString())));
            Assert.That(Options(true, 1.2f).Any(f => Vector2.Distance(f.position, new Vector2(2.75f, 1.7f)) <= 1.0f), Is.False, "control: pocket searched");
            Assert.That(Options(false, 0f).Any(f => Vector2.Distance(f.position, new Vector2(2.75f, 1.7f)) <= 1.0f), Is.False, "control: no peek, old rule");
        }

        [Test]
        public void ACarriedPersonIsSetDownClearOfTheWayOutAndTheRobotPlansBackInAroundThem()
        {
            // Hidden-corner run 1: the robot stopped 0.5 m short of its start, the sled 0.9 m behind it set the person down
            // 5 cm inside the front doorway, and the robot drove into them 13 times trying to go back in (outside for 156
            // of 200 s). Now: the person ends at least a body radius plus a robot's width past the start; the robot marks
            // where it set them down, steps clear, and its plan back in goes around them. Control: tow offset 0 completes
            // on arrival as before.
            var c = Settings();
            (Vector2 person, Vector2 robot, RobotBrain brain)? Carry(float tow)
            {
                var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -6, 6, -6, 6);
                b.AgentGoTo(new Vector2(0, 2.0f), 0, 30, "person");
                var command = new BrainCommand(); bool carrying = false; Vector2? person = null;
                for (int tick = 0; tick < 1600; tick++)
                {
                    command = DriveTick(b, tick, command.forwardMps, command.yawRateDeg);
                    var events = b.DrainEvents();
                    if (!carrying && events.Any(e => e.kind == "agent_goal_reached")) { b.BeginCarry(tick / 20.0, .25f, "p", tow); carrying = true; }
                    var released = events.FirstOrDefault(e => e.kind == "person_carried_out");
                    if (released != null)
                        person = released.numbers.ContainsKey("person_x_m") ? new Vector2((float)released.numbers["person_x_m"], (float)released.numbers["person_y_m"]) : b.Snapshot().pose.position;
                    if (events.Any(e => e.kind == "mission_complete")) return (person ?? Vector2.zero, b.Snapshot().pose.position, b);
                }
                return null;
            }
            var done = Carry(.9f);
            Assert.That(done.HasValue, Is.True, "carried out and finished");
            var (body, at, brain) = done.Value;
            Assert.That(-body.y, Is.GreaterThanOrEqualTo(c.carried_person_radius_m + 2 * c.robot_radius_m - .05f), "the person lies clear of the way out");
            Assert.That(Vector2.Distance(at, body), Is.GreaterThanOrEqualTo(c.carried_person_radius_m + c.robot_radius_m), "the robot stepped clear of them");
            var planner = (GridPlanner)typeof(RobotBrain).GetField("planner", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(brain);
            var back = planner.Plan(at, new Vector2(0, 2f));
            Assert.That(back, Is.Not.Empty, "a way back in exists");
            float closest = float.PositiveInfinity;
            for (int k = 1; k < back.Count; k++) for (float f = 0; f <= 1; f += .05f) closest = Mathf.Min(closest, Vector2.Distance(Vector2.Lerp(back[k - 1], back[k], f), body));
            Assert.That(closest, Is.GreaterThanOrEqualTo(c.carried_person_radius_m + c.robot_radius_m - .05f), "the way back in goes around the person");
            var none = Carry(0f);
            Assert.That(none.HasValue, Is.True);
            Assert.That(-none.Value.robot.y, Is.LessThan(.6f), "control: no tow, completes on arrival");
        }

        [Test]
        public void APersonThisRobotCarriedOutIsNotANewSurvivor()
        {
            // Held-out run 2: robot B carried a person out and set them down in the yard; at 98.9 s its colour detector raised
            // a candidate 0.09 m from that spot and its agent confirmed it as a new survivor (the run's one false
            // confirmation). Sightings within survivor_merge_m of a set-down point make no candidate, and an agent
            // confirmation there is refused with its reason. Control: the same sightings and confirmation at a spot inside,
            // 2 m and more from the set-down point, still count.
            var c = Settings();
            var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -6, 6, -6, 6);
            b.AgentGoTo(new Vector2(0, 2.0f), 0, 30, "person");
            var command = new BrainCommand(); bool carrying = false; Vector2? body = null; double end = -1;
            for (int tick = 0; tick < 1600 && end < 0; tick++)
            {
                command = DriveTick(b, tick, command.forwardMps, command.yawRateDeg);
                var events = b.DrainEvents();
                if (!carrying && events.Any(e => e.kind == "agent_goal_reached")) { b.BeginCarry(tick / 20.0, .25f, "p", .9f); carrying = true; }
                var released = events.FirstOrDefault(e => e.kind == "person_carried_out" && e.numbers.ContainsKey("person_x_m"));
                if (released != null) body = new Vector2((float)released.numbers["person_x_m"], (float)released.numbers["person_y_m"]);
                if (events.Any(e => e.kind == "mission_complete")) end = tick / 20.0;
            }
            Assert.That(body.HasValue && end >= 0, Is.True, "carried out and set down");
            int Candidates(Vector2 at, double t0)
            {
                var pose = b.Snapshot().pose; Vector2 camera = pose.Transform(0, .25f), d = at - camera;
                float bearing = (pose.yawRad - Mathf.Atan2(-d.x, d.y)) * Mathf.Rad2Deg;
                Assert.That(d.magnitude, Is.LessThan(c.person_rgb_max_range_m), "within the detector's range");
                b.OnPersonSighting(t0, bearing, d.magnitude, .9f); b.OnPersonSighting(t0 + .5, bearing, d.magnitude, .9f);
                return b.DrainEvents().Count(e => e.kind == "survivor_candidate");
            }
            var spot = body.Value + new Vector2(.3f, .2f);                          // 0.36 m from where they lie
            Assert.That(Candidates(spot, end + 1), Is.Zero, "no candidate where this robot set someone down");
            b.AgentConfirmSurvivor(spot, end + 2, "ev_inspect");
            Assert.That(b.DrainEvents().Any(e => e.kind == "agent_order" && e.reason.StartsWith("Refused confirm_survivor")), Is.True,
                "the confirmation is refused with its reason");
            Assert.That(b.Snapshot().confirmedSurvivors, Is.Empty, "nobody new is confirmed");
            var inside = new Vector2(1.5f, .5f);
            Assert.That(Vector2.Distance(inside, body.Value), Is.GreaterThan(2f));
            Assert.That(Candidates(inside, end + 3), Is.EqualTo(1), "control: a person elsewhere still makes a candidate");
            b.AgentConfirmSurvivor(inside, end + 4, "ev_inspect");
            Assert.That(b.Snapshot().confirmedSurvivors, Has.Length.EqualTo(1), "control: and is confirmed");
        }

        [Test]
        public void AfterArrivingWhereTheAgentAskedTheRobotWaitsForItsNextOrder()
        {
            // Run 21: robot A arrived facing a confirmed person within 9 deg and drove toward a frontier in the same tick;
            // when the agent answered 1.1 s later the person was 61 deg off, outside the camera. The robot now holds
            // agent_arrival_hold_s for the next order. Controls: a new go_to during the hold is driven at once, and with
            // the hold set to 0 the robot leaves for a frontier.
            float Moved(float holdS, bool newOrder)
            {
                var settings = Settings(); settings.agent_arrival_hold_s = holdS;
                var b = new RobotBrain(Calibration(), settings, "A", 0); Room(b, -4, 4, -4, 4);
                var g = Map(b);
                for (float y = 1.0f; y <= 3.9f; y += .1f) g.ObserveFree(g.Cell(new Vector2(0, y)));   // a frontier to tempt it
                b.AgentGoTo(new Vector2(0, -1.0f), 0, 30, "closer look");
                var command = new BrainCommand(); int tick = 0; Vector2 at = default; bool arrived = false;
                for (; tick < 400 && !arrived; tick++)
                {
                    command = DriveTick(b, tick, command.forwardMps, command.yawRateDeg);
                    arrived = b.DrainEvents().Any(e => e.kind == "agent_goal_reached");
                }
                Assert.That(arrived, "reached the goal");
                at = b.Snapshot().pose.position;
                if (newOrder) b.AgentGoTo(new Vector2(-1.5f, -1.0f), tick / 20.0, 30, "next look");
                for (int k = 0; k < 80; k++, tick++) { command = DriveTick(b, tick, command.forwardMps, command.yawRateDeg); b.DrainEvents(); }   // 4 s
                return Vector2.Distance(at, b.Snapshot().pose.position);
            }
            Assert.That(Moved(5f, false), Is.LessThan(.05f), "holds for the agent");
            Assert.That(Moved(5f, true), Is.GreaterThan(.3f), "control: a new order is driven at once");
            Assert.That(Moved(0f, false), Is.GreaterThan(.3f), "control: without the hold it drives off");
        }

        [Test]
        public void AnAgentGoalOffItsCellCentreEndsAtTheCellAndDoesNotReplanEveryTick()
        {
            // Run 19: an agent goal sat 0.2 to 0.25 m from where path following stops (within goal_tolerance_m of the
            // goal cell's centre, not of the exact point). The path was cleared on arrival at the cell, the agent goal was
            // not yet "reached", and it was replanned every tick: robot A stood still 13 to 20 s three times (700 to 900
            // identical goal events each). Reaching the planned end cell is arrival. Control: a goal on its cell centre.
            List<BrainEvent> Drive(float offset)
            {
                var b = new RobotBrain(Calibration(), Settings(), "A", 0); Room(b, -4, 4, -4, 4);
                var g = Map(b); var centre = g.Centre(g.Cell(new Vector2(0, 1.5f)));
                b.AgentGoTo(centre + new Vector2(0, offset), 0, 30, "closer look");
                var all = new List<BrainEvent>(); var command = new BrainCommand();
                for (int tick = 0; tick < 300; tick++) { command = DriveTick(b, tick, command.forwardMps, command.yawRateDeg); all.AddRange(b.DrainEvents()); }
                return all;
            }
            var off = Drive(.049f);   // just inside the goal cell, beyond its centre along the approach
            Assert.That(off.Count(e => e.kind == "agent_goal_reached"), Is.EqualTo(1), "goal off its cell centre");
            Assert.That(off.Count(e => e.kind == "goal" && e.reason.StartsWith("Agent order")), Is.LessThan(20), "no replanning loop");
            var centred = Drive(0);
            Assert.That(centred.Count(e => e.kind == "agent_goal_reached"), Is.EqualTo(1), "control: goal on its cell centre");
        }

        [Test]
        public void AnAgentGoToNeverSendsAnEmptyClaimAndABadPacketIsDropped()
        {
            // Slow-fire run: an agent go_to was refreshed as a frontier and sent a claim with radius 0; the partner threw
            // "Invalid peer claim" and the whole recording stopped at 7 s.
            var a = new RobotBrain(Calibration(), Settings(), "A", 0); Room(a, -4, 4, -4, 4);
            a.AgentGoTo(new Vector2(0, 3), 0, 30, "closer look");
            BrainMessage last = null;
            for (int i = 0; i < 60; i++)
            {
                double t = i / 15.0; Motion(a, t); a.OnRgbd(Image(t, true)); a.OnAirTemperature(t, 20); a.Step(t, 1f / 15); a.DrainEvents();
                var m = a.BuildMessage(t);
                if (m == null) continue;
                last = m;
                Assert.That(m.Claim == null || m.Claim.radiusM > 0, Is.True, $"claim radius at {t:F2} s");
            }
            Assert.That(last, Is.Not.Null);
            var bad = Wire(last); bad.Claim = new BrainClaim { centre = Vector2.zero, radiusM = 0, expiresAt = bad.SentAt + 1 };
            var b = new RobotBrain(Calibration(), Settings(), "B", 0);
            Assert.DoesNotThrow(() => b.OnPeerMessage(bad, bad.SentAt + .1));
            Assert.That(b.DrainEvents().Count(e => e.kind == "peer_message_rejected"), Is.EqualTo(1));
        }

        [Test]
        public void AReturnUsesALinkedPartnersEntryWhenItsOwnIsCutOff()
        {
            // Slow-fire run: robot B's way home was cut by fire 5.5 m from the door robot A had used, and it held still.
            // With A's map linked, A's entry (the origin of A's frame) is a known way out too. Control: no link. (The test
            // camera sees corridor walls 1 m to each side, so the entry is placed ahead in the corridor.)
            string ReturnPlan(bool linked)
            {
                var c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f; c.mission_time_limit_s = 3.5f; c.frames_shared_at_start = false;
                var b = new RobotBrain(Calibration(), c, "B", 0); Room(b, -5, 5, -2, 6);
                var partner = new RobotBrain(Calibration(), c, "A", 0); Motion(partner, 0); partner.OnAirTemperature(0, 20); partner.Step(0, .05f);
                var events = new List<BrainEvent>();
                for (int i = 0; i < 75; i++)
                {
                    double t = i / 15.0; Motion(b, t, t < 3 ? 1f : 0); b.OnRgbd(Image(t, true)); b.OnAirTemperature(t, 20);
                    if (i == 1) b.OnPeerMessage(Wire(partner.BuildMessage(0)), t);
                    if (i == 1 && linked)
                    {
                        var links = (System.Collections.IDictionary)typeof(RobotBrain).GetField("links", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(b);
                        links["A"] = new BrainPose { position = new Vector2(0, 4), yawRad = 0 };   // A's entry lies 1 m ahead of B's position
                    }
                    var g = Map(b);
                    if (t >= 3.2) for (float x = -5; x <= 5; x += .1f) g.ObserveHeat(g.Cell(new Vector2(x, 1.2f)), 500, t);   // fire across the room
                    b.Step(t, 1f / 15); events.AddRange(b.DrainEvents());
                }
                var plans = events.Where(e => e.kind == "replan" && (e.reason.StartsWith("Return to the start") || e.reason.StartsWith("The known return route"))).ToList();
                Assert.That(plans, Is.Not.Empty, "the time limit started a return");
                // A deadline return cut by fire now loses the way out and retreats rather than holding (#43).
                if (events.Any(e => e.kind == "escape_closing" && e.reason.StartsWith("The known way out is closed"))) return "lost the way out";
                return plans.Last().reason;
            }
            Assert.That(ReturnPlan(true), Does.StartWith("Return to the start"), "a route to the partner's entry");
            Assert.That(ReturnPlan(false), Is.EqualTo("lost the way out"), "control: own entry only, cut, so the way out is lost");
        }

        [Test]
        public void UnsearchedFloorBehindAnObstacleIsReportedWithItsBearing()
        {
            // Run 13: robot B gave up 2 m short of the hallway debris without looking at it. The brain reports the nearest
            // unsearched floor it cannot reach, so the agent can measure and clear what blocks it. Here the LiDAR sees
            // through a 0.3 m gap in a line of debris that the body does not fit through. Control: the line is closed.
            var c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f;
            BrainBlockedFloor? Report(bool gap)
            {
                var walls = Box(-3, -3, 3, 4).Concat(gap
                    ? new[] { (new Vector2(-3, 1.6f), new Vector2(-.15f, 1.6f)), (new Vector2(.15f, 1.6f), new Vector2(3, 1.6f)) }
                    : new[] { (new Vector2(-3, 1.6f), new Vector2(3, 1.6f)) }).ToArray();
                var b = new RobotBrain(Calibration(), c, "A", 0);
                Motion(b, 0); b.OnAirTemperature(0, 20); b.OnLidar(SegmentScan(0, Vector2.zero, 0, walls)); b.Step(0, .05f);
                return b.BlockedUnsearched();
            }
            var found = Report(true);
            Assert.That(found.HasValue, Is.True, "free floor mapped through the gap, never searched");
            Assert.That(found.Value.bearingDegRight, Is.InRange(-20f, 20f), "straight ahead");
            Assert.That(found.Value.distanceM, Is.InRange(1.6f, 2.4f));
            Assert.That(found.Value.approach.y, Is.InRange(.8f, 1.4f), "the approach is on this side, before the line");
            Assert.That(Report(false).HasValue, Is.False, "control: nothing seen beyond a closed line");
        }

        [Test]
        public void ARobotInAHeatMarginFollowsItsPathWithoutALoop()
        {
            // 2x2 building run: robot B cleared and rebuilt the same path 185 times in 20 s. Not reproduced here (this passes
            // without the fix too); it guards that a robot beside a heat margin keeps following its path.
            var c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f;
            var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -4, 4, -4, 4);
            int loops = 0; float moved = 0;
            for (int i = 0; i < 90; i++)
            {
                double t = i / 15.0; Motion(b, t, t < .5 ? .1f : 0); b.OnRgbd(Image(t, true)); b.OnAirTemperature(t, 20);
                var g = Map(b);
                for (float x = -1.6f; x <= -.4f; x += .1f) for (float y = -.6f; y <= .6f; y += .1f) g.ObserveHeat(g.Cell(new Vector2(x, y)), 500, t);
                var cmd = b.Step(t, 1f / 15); moved = Mathf.Max(moved, Mathf.Abs(cmd.forwardMps) + Mathf.Abs(cmd.yawRateDeg));
                loops += b.DrainEvents().Count(e => e.reason.StartsWith("Pursuit chord crosses"));
            }
            Assert.That(loops, Is.LessThan(3), "no clear-and-rebuild loop");
            Assert.That(moved, Is.GreaterThan(0));
        }

        [Test]
        public void AnAgentKeepOutAreaNeverCutsTheWayOut()
        {
            // 2x2 building run: robot A's agent drew 2.5 m around its own target, the route home was cut, and the robot held
            // still. Across the route home: refused. To one side: shrunk to stay clear of the route.
            string Order(Vector2 centre, out float radius)
            {
                var c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f;
                var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -5, 5, -2, 6);
                for (int i = 0; i < 46; i++) { double t = i / 15.0; Motion(b, t, t < 3 ? 1f : 0); b.OnRgbd(Image(t, true)); b.OnAirTemperature(t, 20); b.Step(t, 1f / 15); }
                b.DrainEvents();
                b.AgentAvoidArea(centre, 2f, 3.1, 30, "ev");
                var e = b.DrainEvents().Single(x => x.kind == "agent_order");
                radius = (float)e.numbers["radius_m"]; return e.reason;
            }
            Assert.That(Order(new Vector2(0, -.2f), out _), Does.Contain("way out"), "behind the robot, over its start");
            // 2.2 m beside the route home, 3.1 m from the robot: the robot-only rule would allow 1.66 m, the route rule 0.75 m.
            Assert.That(Order(new Vector2(-2.2f, .8f), out float r), Does.StartWith("avoid_area around"), "beside the route: accepted");
            Assert.That(r, Is.LessThan(1f), "and shrunk clear of the route home");
        }

        [Test]
        public void AnExhaustedRobotTakesAnEmergencyExitPastTheFire()
        {
            // 2x2 building run: robot A sat beside its entry door with its heat budget at 0, every route out closed by the
            // 1.2 m fire margin. Fire on both sides of a 1.4 m gap between the robot and its start: with the budget nearly
            // spent it must leave through the gap. Control: emergency clearance as large as the normal one, it cannot.
            string Final(float emergencyClearance, out int exits)
            {
                var c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f; c.emergency_fire_clearance_m = emergencyClearance;
                c.last_resort_fire_clearance_m = emergencyClearance;   // no smaller clearance either, so the control has no route at all
                var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -3, 3, -2, 6);
                exits = 0;
                for (int i = 0; i < 150; i++)
                {
                    double t = i / 15.0; Motion(b, t, t < 3 ? 1f : 0); b.OnRgbd(Image(t, true)); b.OnAirTemperature(t, t < 3.2 ? 20 : 150);
                    var g = Map(b);
                    if (t >= 3.2)
                        for (float y = 1.3f; y <= 1.7f; y += .1f)
                        {
                            for (float x = -3f; x <= -.7f; x += .1f) g.ObserveHeat(g.Cell(new Vector2(x, y)), 500, t);
                            for (float x = .7f; x <= 3f; x += .1f) g.ObserveHeat(g.Cell(new Vector2(x, y)), 500, t);
                        }
                    b.Step(t, 1f / 15); exits += b.DrainEvents().Count(e => e.kind == "emergency_exit");
                }
                return b.Snapshot().state;
            }
            Assert.That(Final(.5f, out int taken), Is.EqualTo("return"), "leaving by the emergency route");
            Assert.That(taken, Is.EqualTo(1));
            Assert.That(Final(1.2f, out int none), Is.Not.EqualTo("return"), "control: no route at the normal clearance");
            Assert.That(none, Is.Zero);
            // Hot air but no fire on the way: with the budget spent it leaves by the normal route instead of sitting in the
            // heat at the nearest "coolest" cell (2x2 attempt 4).
            var c2 = Settings(); c2.grid_size_m = 12; c2.robot_radius_m = .25f;
            var hot = new RobotBrain(Calibration(), c2, "A", 0); Room(hot, -3, 3, -2, 6); int left = 0;
            for (int i = 0; i < 150; i++)
            {
                double t = i / 15.0; Motion(hot, t, t < 3 ? 1f : 0); hot.OnRgbd(Image(t, true)); hot.OnAirTemperature(t, t < 3.2 ? 20 : 150);
                hot.Step(t, 1f / 15); left += hot.DrainEvents().Count(e => e.kind == "emergency_exit");
            }
            Assert.That(left, Is.EqualTo(1));
            Assert.That(hot.Snapshot().state, Is.EqualTo("return"));
            // Then fire closes that route to a 1.4 m gap on the way out (2x2 attempt 5): it re-plans at the emergency
            // clearance instead of holding still.
            var cut = new RobotBrain(Calibration(), c2, "A", 0); Room(cut, -3, 3, -2, 6); var cutEvents = new List<BrainEvent>();
            for (int i = 0; i < 180; i++)
            {
                double t = i / 15.0; Motion(cut, t, t < 3 ? 1f : 0); cut.OnRgbd(Image(t, true)); cut.OnAirTemperature(t, t < 3.2 ? 20 : 150);
                if (cutEvents.Any(e => e.kind == "emergency_exit"))
                {
                    var g = Map(cut);
                    for (float y = 1.3f; y <= 1.7f; y += .1f)
                    {
                        for (float x = -3f; x <= -.7f; x += .1f) g.ObserveHeat(g.Cell(new Vector2(x, y)), 500, t);
                        for (float x = .7f; x <= 3f; x += .1f) g.ObserveHeat(g.Cell(new Vector2(x, y)), 500, t);
                    }
                }
                cut.Step(t, 1f / 15); cutEvents.AddRange(cut.DrainEvents());
            }
            // It never holds still: the route is kept while no cell on it burns and the body stays clear of flames, or
            // re-planned at a reduced clearance (held-out seed 914 replaced "re-plan" with "keep the route" when it is safe).
            Assert.That(cutEvents.Any(e => e.kind == "replan" && e.reason.StartsWith("The known return route is blocked")), Is.False, "no hold on the way out");
            Assert.That(cut.Snapshot().path.Length, Is.GreaterThan(0), "a route out is still being followed");
            Assert.That(cutEvents.Last(e => e.kind == "emergency_exit").numbers["route_points"], Is.GreaterThan(0));
        }

        [Test]
        public void ARobotThatLeftEarlyGoesBackInWhenTheWayIsClear()
        {
            // Slow-fire run: robot A left at 74 s because fire neared its route and stood outside for the last two minutes.
            // Left early and the way in is clear: it re-enters. Controls: a normal finish stays finished; still-hot air
            // (heat budget not recovered) keeps it out.
            string After(bool leftEarly, bool hotAir, out int reentries)
            {
                var c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f;
                var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -5, 5, -2, 6);
                var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
                typeof(RobotBrain).GetField("completed", flags).SetValue(b, true);
                typeof(RobotBrain).GetField("canReenter", flags).SetValue(b, leftEarly);
                if (hotAir) typeof(RobotBrain).GetField("budget", flags).SetValue(b, .2f);   // spent, and the air keeps it low
                reentries = 0;
                // 8 s: re-entry waits for the fire's closing rate along the way in, measured over two checks (escape_rate_window_s).
                for (int i = 0; i < 120; i++)
                {
                    double t = i / 15.0; Motion(b, t); b.OnRgbd(Image(t, true)); b.OnAirTemperature(t, hotAir ? 60 : 20);
                    b.Step(t, 1f / 15); reentries += b.DrainEvents().Count(e => e.kind == "reenter");
                }
                return b.Snapshot().state;
            }
            Assert.That(After(true, false, out int back), Is.Not.EqualTo("complete"), "goes back in");
            Assert.That(back, Is.EqualTo(1));
            Assert.That(After(false, false, out int none), Is.EqualTo("complete"), "control: a normal finish stays finished");
            Assert.That(none, Is.Zero);
            Assert.That(After(true, true, out int hot), Is.Not.EqualTo("explore"), "control: hot air, budget not recovered");
            Assert.That(hot, Is.Zero);
        }

        [Test]
        public void ARobotWaitingOutsideAtABlockedEntryFollowsTheWallToLookForAnotherWayIn()
        {
            // Held-out seeds 916/917: robots that left for safety stood at their own door for 106 to 124 s while every way in
            // through it stayed unsafe. DARPA SubT's CERBERUS backtracks after a set time without motion; GBPlanner2 keeps
            // exploring when home is unreachable. Here a robot outside a building, its entry inside cut by fire, waits
            // exterior_dwell_s, then drives to unexplored floor along the building's wall (within facade_band_m of mapped
            // wall, outside the search area), never into open yard. Control: with the dwell switched off it stays.
            (int relocations, Vector2 end, float maxFromWall) Run(float dwell)
            {
                var c = Settings(); c.grid_size_m = 30; c.robot_radius_m = .25f; c.search_requires_ceiling = true;
                c.exterior_dwell_s = dwell; c.facade_band_m = 2f; c.mission_time_limit_s = 1000;
                var b = new RobotBrain(Calibration(), c, "A", 0); var g = Map(b);
                var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
                var indoors = (bool[])typeof(OccupancyGrid).GetField("sightIndoors", flags).GetValue(g);
                for (int i = 0; i < g.Count; i++)
                {
                    var p = g.Centre(i);
                    bool facade = Mathf.Abs(p.y - 2.05f) < .1f && p.x > -4 && p.x < 12 && !(Mathf.Abs(p.x) < .5f);
                    bool side = (Mathf.Abs(p.x + 4) < .1f || Mathf.Abs(p.x - 12) < .1f) && p.y > 2 && p.y < 9;
                    if (facade || side || (Mathf.Abs(p.y - 9) < .1f && p.x > -4 && p.x < 12)) g.ObserveOccupied(i);
                    else if (p.y > 2.15f && p.y < 8.9f && p.x > -3.9f && p.x < 6f) { g.ObserveFree(i); indoors[i] = true; }   // searched part of the inside
                    // The yard the robot saw along the wall, with small unseen pockets against the wall every 1.5 m: each is
                    // an unexplored edge, as on held-out seed 920 where the first one was 0.6 m from robot A.
                    else if (p.y > -1.5f && p.y < 1.95f && p.x > -3f && p.x < 11f &&
                             !(p.y > 1.75f && Mathf.Abs(Mathf.Repeat(p.x + .75f, 1.5f) - .75f) < .15f && Mathf.Abs(p.x) > 1f)) g.ObserveFree(i);
                }
                typeof(RobotBrain).GetField("completed", flags).SetValue(b, true);
                typeof(RobotBrain).GetField("canReenter", flags).SetValue(b, true);
                int n = 0; var command = new BrainCommand(); float maxFromWall = 0;
                for (int tick = 0; tick < 20 * 60; tick++)
                {
                    for (float x = -1.5f; x <= 1.5f; x += .1f) g.ObserveHeat(g.Cell(new Vector2(x, 3.2f)), 500, tick / 20.0);   // fire just inside the door
                    command = DriveTick(b, tick, command.forwardMps, command.yawRateDeg);
                    n += b.DrainEvents().Count(e => e.kind == "relocate");
                    maxFromWall = Mathf.Max(maxFromWall, 2.05f - b.Snapshot().pose.position.y);
                }
                return (n, b.Snapshot().pose.position, maxFromWall);
            }
            var moved = Run(15f);
            Assert.That(moved.relocations, Is.GreaterThan(0), "it sets off along the wall");
            // Either way along the wall: the robot cannot know where another door is. Held-out seed 920: the first step was
            // 0.6 m and each next one waited a full dwell; in the 45 s after a 15 s dwell it now walks on past 7 m; hopping with a full dwell each time
            // reaches about 4.5 m (three hops).
            Assert.That(Mathf.Abs(moved.end.x), Is.GreaterThan(7f), "on along the facade: " + moved.end);
            Assert.That(moved.maxFromWall, Is.LessThan(2.6f), "never out into the yard");
            var stays = Run(0f);
            Assert.That(stays.relocations, Is.Zero, "control: no dwell budget, it waits at its door");
            Assert.That(stays.end.magnitude, Is.LessThan(.3f));
        }

        [Test]
        public void TheLidarFindsAWayOutThroughAnOpeningInHeavySmoke()
        {
            // A robot knew only its own entry as a way out (held-out seed 914 run 5: robot A held at 185 C with the front
            // door 5.5 m away). Its own LiDAR sees open ground through an exterior opening sweep after sweep, while smoke
            // drops pulses at random. A 6 x 6 m room with a 1.2 m opening in its south wall, sweeps from moving poses inside,
            // half of all returning pulses dropped (sensors.json's worst case). The floor just past the opening is a way out;
            // nothing inside the room is. Control: the same room and smoke with the wall closed has no way out.
            int openInside = 0;
            List<Vector2> Exits(bool opening, float share = -1)
            {
                var c = Settings(); c.grid_size_m = 20; c.robot_radius_m = .25f; if (share > 0) c.exit_open_share = share;
                var g = new OccupancyGrid(c); var rng = new System.Random(7);
                var walls = new List<(Vector2 a, Vector2 b)> { (new Vector2(-3, -3), new Vector2(-3, 3)), (new Vector2(3, -3), new Vector2(3, 3)), (new Vector2(-3, 3), new Vector2(3, 3)) };
                if (opening) { walls.Add((new Vector2(-3, -3), new Vector2(-.6f, -3))); walls.Add((new Vector2(.6f, -3), new Vector2(3, -3))); }
                else walls.Add((new Vector2(-3, -3), new Vector2(3, -3)));
                float Hit(Vector2 o, Vector2 d)
                {
                    float best = float.PositiveInfinity;
                    foreach (var (a, b) in walls)
                    {
                        Vector2 e = b - a; float den = d.x * e.y - d.y * e.x; if (Mathf.Abs(den) < 1e-6f) continue;
                        Vector2 w = a - o; float t = (w.x * e.y - w.y * e.x) / den, u = (w.x * d.y - w.y * d.x) / den;
                        if (t > 0 && u >= 0 && u <= 1) best = Mathf.Min(best, t);
                    }
                    return best;
                }
                const int beams = 720; float inc = 2 * Mathf.PI / beams;
                for (int k = 0; k < 60; k++)
                {
                    var pose = new BrainPose { position = new Vector2(1.2f * Mathf.Sin(k * .37f), .9f * Mathf.Cos(k * .23f)), yawRad = k * .11f };
                    var ranges = new float[beams];
                    for (int i = 0; i < beams; i++)
                    {
                        float ang = pose.yawRad - Mathf.PI + i * inc; float r = Hit(pose.position, new Vector2(-Mathf.Sin(ang), Mathf.Cos(ang)));
                        ranges[i] = r > 12f || rng.NextDouble() < .5 ? 0f : r;
                    }
                    g.IntegrateLidar(new LidarScan(k * .1, -Mathf.PI, inc, .15f, 12f, 0f, .45f, ranges), pose);
                }
                // The camera saw the floor inside and, through the opening, just past it.
                for (int i = 0; i < g.Count; i++)
                {
                    var q = g.Centre(i);
                    if ((Mathf.Abs(q.x) < 2.9f && Mathf.Abs(q.y) < 2.9f) || (opening && Mathf.Abs(q.x) < .5f && q.y < -3.1f && q.y > -4.6f)) g.ObserveFree(i);
                }
                var openBeyond = typeof(OccupancyGrid).GetMethod("OpenBeyond", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                openInside = 0;
                for (int i = 0; i < g.Count; i++) { var q = g.Centre(i); if (Mathf.Abs(q.x) < 2.4f && Mathf.Abs(q.y) < 2.4f && (bool)openBeyond.Invoke(g, new object[] { i })) openInside++; }
                return g.SensedExits();
            }
            var open = Exits(true);
            Assert.That(open.Any(e => Mathf.Abs(e.x) < 1f && e.y < -3f && e.y > -5f), Is.True, "the floor past the opening is a way out: " + string.Join(";", open));
            Assert.That(open.All(e => e.y < -2.5f), Is.True, "nothing more than half a metre inside the wall line: " + string.Join(";", open));
            Assert.That(Exits(false), Is.Empty, "control: a closed room in the same smoke has no way out");
            Assert.That(openInside, Is.Zero, "the statistic alone, before any indoor test: no cell of the closed room reads open beyond");
            // A sweep counts a cell open only if every beam crossing it came back empty; a few metres out about four beams
            // cross each 10 cm cell, so half-pulse smoke empties all of them about 6 % of the time. Below that share the
            // smoke alone reads open: the threshold, not luck, holds the control.
            Exits(false, .03f);
            Assert.That(openInside, Is.GreaterThan(50), "at a 0.03 open share the smoke alone reads open");
        }

        [Test]
        public void ADeadlineReturnCutByFireRetreatsInsteadOfHolding()
        {
            // Held-out seed 919, robot B: a deadline return began, fire then cut the route home, and the robot held "the
            // known return route is blocked" for 20 s in cool air. Like a closing-route return, it now loses the way out
            // and retreats to search on. Control: the same cut on a long mission (no deadline return) is unaffected.
            int Holds(float limit)
            {
                var c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f; c.mission_time_limit_s = limit;
                var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -5, 5, -2, 6);
                var events = new List<BrainEvent>();
                for (int i = 0; i < 15 * 20; i++)
                {
                    double t = i / 15.0; Motion(b, t, t < 3 ? 1f : 0); b.OnRgbd(Image(t, true)); b.OnAirTemperature(t, 20);
                    if (t >= 5) { var g = Map(b); for (float x = -5; x <= 5; x += .05f) g.ObserveHeat(g.Cell(new Vector2(x, 1.5f)), 500, t); }
                    b.Step(t, 1f / 15); events.AddRange(b.DrainEvents());
                }
                return events.Count(e => e.kind == "replan" && e.reason.StartsWith("The known return route is blocked"));
            }
            Assert.That(Holds(12f), Is.Zero, "a deadline return cut by fire does not hold");
        }

        [Test]
        public void ARobotLeavesInTimeToBeOutByTheMissionDeadline()
        {
            // The brain started back only at mission_time_limit_s itself, so it was still inside when the mission ended
            // (and the recorder's 200 s did not match the brain's 240 s). 3 m in, a 20 s mission: the walk out with its
            // allowance is about 3 / 0.6 * 1.5 = 7.5 s, so it leaves around 10 s, not at 20 s. Control: a long mission.
            double LeftAt(float limit)
            {
                var c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f; c.mission_time_limit_s = limit;
                var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -5, 5, -2, 6);
                for (int i = 0; i < 15 * 20; i++)
                {
                    double t = i / 15.0; Motion(b, t, t < 3 ? 1f : 0); b.OnRgbd(Image(t, true)); b.OnAirTemperature(t, 20);
                    b.Step(t, 1f / 15);
                    if (b.DrainEvents().Any(e => e.kind == "goal" && e.reason.StartsWith("Mission time is nearly up"))) return t;
                }
                return double.NaN;
            }
            Assert.That(LeftAt(20), Is.InRange(5.0, 16.0), "leaves with time to walk out");
            Assert.That(double.IsNaN(LeftAt(1000)), Is.True, "control: a long mission keeps searching");
        }

        [Test]
        public void ARobotOutsideGoesBackInByAFrontierClearOfTheFireEvenWhenTheNearestIsNot()
        {
            // Codex review of held-out seed 916: re-entry tested only the planner's first frontier, so a robot whose nearest
            // frontier lay by the fire stayed out although other frontiers were clear. A fire line 1.8 m beyond the nearest
            // edge of the searched floor; the sides, 6 m away, are clear. Control: with the fire along every edge it stays out.
            int Reentries(bool fireEverywhere)
            {
                var c = Settings(); c.grid_size_m = 16; c.robot_radius_m = .25f;
                var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -6, 6, -2.5f, 7);
                var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
                typeof(RobotBrain).GetField("completed", flags).SetValue(b, true);
                typeof(RobotBrain).GetField("canReenter", flags).SetValue(b, true);
                int n = 0;
                for (int i = 0; i < 150; i++)
                {
                    double t = i / 15.0; Motion(b, t); b.OnRgbd(Image(t, true)); b.OnAirTemperature(t, 20);
                    var g = Map(b);
                    for (float x = -7.5f; x <= 7.5f; x += .1f) g.ObserveHeat(g.Cell(new Vector2(x, -4.3f)), 500, t);
                    if (fireEverywhere)
                        for (float y = -7.5f; y <= 7.5f; y += .1f) { g.ObserveHeat(g.Cell(new Vector2(-7.5f, y)), 500, t); g.ObserveHeat(g.Cell(new Vector2(7.5f, y)), 500, t); g.ObserveHeat(g.Cell(new Vector2(y, 7.9f)), 500, t); }   // inside the 16 m grid
                    b.Step(t, 1f / 15); n += b.DrainEvents().Count(e => e.kind == "reenter");
                }
                return n;
            }
            Assert.That(Reentries(false), Is.EqualTo(1), "back in by a clear frontier");
            Assert.That(Reentries(true), Is.Zero, "control: every frontier by the fire");
        }

        [Test]
        public void GoingBackInCountsTheFireBesideTheDoor()
        {
            // Held-out seed 920, robot A: from outside, re-entry skipped the route within escape_ignore_m (2 m) of the robot,
            // which was the door, and went in with the fire 1.97 m from it; the leave rule turned it round 2 s later. A way
            // in from the robot straight through a door 1.2 m ahead, fire 1.9 m beside the door: under the margin. Control:
            // fire 2.6 m beside it.
            var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
            float Near(float fireX)
            {
                var c = Settings(); c.grid_size_m = 12;
                var b = new RobotBrain(Calibration(), c, "A", 0); var g = Map(b);
                g.ObserveHeat(g.Cell(new Vector2(fireX, 1.2f)), 500, 0);
                typeof(OccupancyGrid).GetMethod("PrepareNavigation", flags).Invoke(g, null);
                var path = new List<Vector2>(); for (float y = 0; y <= 6f; y += .1f) path.Add(new Vector2(0, y));
                return (float)typeof(RobotBrain).GetMethod("ReentryFireDistance", flags).Invoke(b, new object[] { path });
            }
            Assert.That(Near(1.9f), Is.LessThan(Settings().escape_fire_margin_m), "fire beside the door blocks the way in");
            Assert.That(Near(2.6f), Is.GreaterThanOrEqualTo(Settings().escape_fire_margin_m), "control: fire clear of the door");
        }

        [Test]
        public void ARobotThatWalksBackInFromANewSpotCanLeaveThere()
        {
            // Held-out seed 920, robot A: after following the wall it went back in 5.7 m from its first door and turned round
            // at once, because its way out was still planned to that first door, 1.3 m from the fire. Here its first door is
            // 2.2 m behind it with a fire line 1.8 m beyond that; it goes back in where it stands. The spot it walked in from
            // is a way out, so it searches on; without it (the first door its only way out) the leave rule turns it round.
            var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
            (int reentries, int turned) Run(bool remember)
            {
                var c = Settings(); c.grid_size_m = 16; c.robot_radius_m = .25f;
                var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -6, 6, -2.5f, 7);
                typeof(RobotBrain).GetField("completed", flags).SetValue(b, true);
                typeof(RobotBrain).GetField("canReenter", flags).SetValue(b, true);
                typeof(RobotBrain).GetField("home", flags).SetValue(b, new Vector2(0, -2.2f));
                var walked = (List<Vector2>)typeof(RobotBrain).GetField("walkedIn", flags).GetValue(b);
                int n = 0, turned = 0;
                for (int i = 0; i < 150; i++)
                {
                    double t = i / 15.0; Motion(b, t); b.OnRgbd(Image(t, true)); b.OnAirTemperature(t, 20);
                    var g = Map(b);
                    for (float x = -7.5f; x <= 7.5f; x += .1f) g.ObserveHeat(g.Cell(new Vector2(x, -4.0f)), 500, t);
                    b.Step(t, 1f / 15);
                    if (!remember) walked.Clear();
                    foreach (var e in b.DrainEvents())
                    {
                        if (e.kind == "reenter") n++;
                        if (n > 0 && e.kind == "escape_closing") turned++;
                    }
                }
                return (n, turned);
            }
            var kept = Run(true);
            Assert.That(kept.reentries, Is.EqualTo(1));
            Assert.That(kept.turned, Is.Zero, "it may leave where it walked in, so it searches on");
            var lost = Run(false);
            Assert.That(lost.reentries, Is.EqualTo(1));
            Assert.That(lost.turned, Is.GreaterThan(0), "control: only the first door, by the fire, turns it round");
        }

        [Test]
        public void ARobotGoesBackInOnlyWhereTheFireIsNotClosingOnTheWayIn()
        {
            // Held-out seed 914 run 4, robot A: it left at 50 s because the fire was closing on its way out and re-entered in
            // the same second it got out (75 s), along the route the front was still advancing on. Re-entry now applies the
            // leave rule's closing test: the fire distance at fixed route cells over two checks spanning escape_rate_window_s
            // must not close faster than the robot could walk the route. A still fire 3 m off the route passes after the
            // measurement; a front advancing 0.5 m/s from the same place does not.
            var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
            bool[] Checks(float speed)
            {
                var c = Settings(); c.grid_size_m = 12;
                var b = new RobotBrain(Calibration(), c, "A", 0); var g = Map(b);
                var route = new List<Vector2>(); for (float y = 0; y <= 6f; y += .1f) route.Add(new Vector2(0, y));
                var steady = typeof(RobotBrain).GetMethod("ReentryFrontSteady", flags);
                var prepare = typeof(OccupancyGrid).GetMethod("PrepareNavigation", flags);
                var results = new List<bool>();
                for (int k = 0; k <= 3; k++)
                {
                    double t = k * c.escape_check_s; float x = 4f - speed * (float)t;
                    for (float y = 3.5f; y <= 4.5f; y += .1f) g.ObserveHeat(g.Cell(new Vector2(x, y)), 500, t);
                    prepare.Invoke(g, null);
                    results.Add((bool)steady.Invoke(b, new object[] { t, route, x }));
                }
                return results.ToArray();
            }
            Assert.That(Checks(0), Is.EqualTo(new[] { false, false, true, true }), "a still fire: in once two intervals are measured");
            var closing = Checks(.5f);
            Assert.That(closing[2] || closing[3], Is.False, "a front advancing 0.5 m/s keeps the robot out");
            // Wired into CheckReenter: a robot that left early goes back in only once the rate is measured.
            var cfg = Settings(); cfg.grid_size_m = 12; cfg.robot_radius_m = .25f;
            var brain = new RobotBrain(Calibration(), cfg, "A", 0); Room(brain, -5, 5, -2, 6);
            typeof(RobotBrain).GetField("completed", flags).SetValue(brain, true);
            typeof(RobotBrain).GetField("canReenter", flags).SetValue(brain, true);
            double back = double.NaN;
            for (int i = 0; i < 150 && double.IsNaN(back); i++)
            {
                double t = i / 15.0; Motion(brain, t); brain.OnRgbd(Image(t, true)); brain.OnAirTemperature(t, 20);
                brain.Step(t, 1f / 15); if (brain.DrainEvents().Any(e => e.kind == "reenter")) back = t;
            }
            Assert.That(back, Is.GreaterThanOrEqualTo(cfg.escape_rate_window_s - 1e-3), "re-entry waits for two measured intervals");
        }

        [Test]
        public void AStandingRobotIsNotCirclingSoTheProgressWatchdogLeavesTheNearbyDoorsAlone()
        {
            // Held-out seed 914 run 4, robot A: turning in place on its agent's orders to face warm shapes, it mapped nothing
            // new for 15 s; the progress watchdog, meant for a robot circling, set aside a 2.5 m circle around the doorway
            // it stood in, which held the open door of an unsearched bedroom 2.3 m away. Now only a robot that travelled
            // progress_min_travel_m in the window counts as circling. Control: the same 25 s spent driving back and forth
            // over the same floor (several metres travelled, nothing new seen) still trips the watchdog.
            int NoProgress(float speed)
            {
                var c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f;
                var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -5, 5, -5, 5);
                var g = Map(b); for (int i = 0; i < g.Count; i++) { var p = g.Centre(i); if (p.x > 4.5f) g.ObserveFree(i); }
                int count = 0; var kinds = new Dictionary<string, int>();
                for (int k = 0; k <= 15 * 25; k++)
                {
                    double t = k / 15.0;
                    Motion(b, t, ((int)(t / 2)) % 2 == 0 ? speed : -speed); b.OnRgbd(Image(t, true)); b.OnAirTemperature(t, 20);
                    b.Step(t, 1f / 15);
                    foreach (var e in b.DrainEvents()) { if (e.kind == "no_progress") count++; kinds[e.kind] = kinds.TryGetValue(e.kind, out int n) ? n + 1 : 1; }
                }
                TestContext.WriteLine($"speed {speed}: " + string.Join(", ", kinds.Select(kv => kv.Key + "=" + kv.Value)) + " state " + b.Snapshot().state);
                return count;
            }
            Assert.That(NoProgress(0), Is.Zero, "standing still is not circling");
            Assert.That(NoProgress(.3f), Is.GreaterThan(0), "control: driving back and forth over seen floor is");
        }

        [Test]
        public void AFireSeenOutForTwoSecondsIsNoLongerFireButFlickerAndPartnerHeatStay()
        {
            // A 500 C reading stayed "fire" for over 30 minutes (0.2 C/s decay), so water could never open a route in the
            // robot's own map. Seen cool with no hot reading for heat_cool_after_s, a cell takes the cool value. Controls:
            // a flickering flame (hot every 0.3 s) stays fire; heat from a partner or a keep-out order stays.
            var c = Settings(); c.grid_size_m = 6;
            var g = OpenGrid(c); int i = g.Cell(new Vector2(1, 1));
            g.ObserveHeat(i, 500, 0);
            for (double t = .1; t <= 1.5; t += .1) g.ObserveHeat(i, 30, t);
            Assert.That(g.Dangerous(i), Is.True, "1.5 s after the last hot reading: still fire");
            for (double t = 1.6; t <= 2.5; t += .1) g.ObserveHeat(i, 30, t);
            Assert.That(g.Dangerous(i), Is.False, "2 s and more of cool readings: out");
            var f = OpenGrid(c); int j = f.Cell(new Vector2(1, 1));
            for (int k = 0; k < 20; k++) f.ObserveHeat(j, k % 3 == 0 ? 500 : 30, k * .1);
            Assert.That(f.Dangerous(j), Is.True, "flicker: hot every 0.3 s");
            var p = OpenGrid(c); int q = p.Cell(new Vector2(1, 1));
            typeof(OccupancyGrid).GetMethod("ObservePeerHeat", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
                .Invoke(p, new object[] { q, 500f, 0.0, 0.0 });
            for (double t = .1; t <= 3; t += .1) p.ObserveHeat(q, 30, t);
            Assert.That(p.Dangerous(q), Is.True, "a partner's or an agent's fire is not erased by this robot's cool readings");
        }

        [Test]
        public void ACarriedPersonGoesOutSlowlyAndTheRobotMayComeBack()
        {
            // With a person on the sled the robot heads for the nearest known way out at carry speed, and once there it
            // reports them carried out and may go back in. Control: without carrying, the speed is not capped.
            var c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f;
            var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -5, 5, -2, 6);
            float fastest = 0; var events = new List<BrainEvent>();
            for (int i = 0; i < 90; i++)
            {
                double t = i / 15.0; Motion(b, t, t < 3 ? 1f : 0); b.OnRgbd(Image(t, true)); b.OnAirTemperature(t, 20);
                if (i == 47) b.BeginCarry(t, .25f, "Survivor_s1");
                var cmd = b.Step(t, 1f / 15); events.AddRange(b.DrainEvents());
                if (i > 47) fastest = Mathf.Max(fastest, cmd.forwardMps);
            }
            Assert.That(events.Any(e => e.kind == "carrying_out"), Is.True);
            Assert.That(b.Snapshot().state, Is.EqualTo("return"), "heading for a way out");
            // The robot turns toward its way out first, so check the cap where it applies: the speed it settles at when
            // full speed is commanded (control: the same robot before carrying settles faster).
            float Settle(RobotBrain brain)
            {
                var command = typeof(RobotBrain).GetMethod("Command", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                float v = 0; for (int k = 0; k < 120; k++) v = ((BrainCommand)command.Invoke(brain, new object[] { 2f, 0f, 1f / 15, false })).forwardMps;
                return v;
            }
            var free = new RobotBrain(Calibration(), c, "B", 0); Room(free, -5, 5, -2, 6);
            Assert.That(Settle(b), Is.LessThanOrEqualTo(.25f + 1e-4f), "carry speed");
            Assert.That(Settle(free), Is.GreaterThan(.25f), "control: not carrying");
            var home = new RobotBrain(Calibration(), c, "A", 0); Room(home, -5, 5, -2, 6); var done = new List<BrainEvent>();
            // 8 s: going back in waits for the fire's closing rate along the way in (two checks, escape_rate_window_s).
            for (int i = 0; i < 120; i++)
            {
                double t = i / 15.0; Motion(home, t); home.OnRgbd(Image(t, true)); home.OnAirTemperature(t, 20);
                if (i == 2) home.BeginCarry(t, .25f, "Survivor_s1");
                home.Step(t, 1f / 15); done.AddRange(home.DrainEvents());
            }
            Assert.That(done.Count(e => e.kind == "person_carried_out"), Is.EqualTo(1), "at the way out: carried out");
            Assert.That(done.Any(e => e.kind == "reenter"), Is.True, "and it may go back in");
        }

        [Test]
        public void AReturnEndsWithinHalfAMetreOfADoorwayStart()
        {
            // House reflex check: robot B stopped 0.3 m from its doorway start, where the return path ended, and replanned
            // the same path for 45 s. Standing 0.3 m from the start must complete the mission; control: 0.18 m does not.
            int Completions(float arrive)
            {
                var c = Settings(); c.mission_time_limit_s = 1.2f; c.return_arrive_m = arrive; var b = new RobotBrain(Calibration(), c, "A", 0);
                int done = 0;
                for (int i = 0; i < 60; i++)
                {
                    double t = i / 15.0; Motion(b, t, t < 1 ? .3f : 0); b.OnRgbd(Image(t, true)); b.OnAirTemperature(t, 20);
                    b.Step(t, 1f / 15); done += b.DrainEvents().Count(e => e.kind == "mission_complete");
                }
                Assert.That(Vector2.Distance(b.Snapshot().pose.position, Vector2.zero), Is.InRange(.25f, .35f), "the robot stands 0.3 m from its start");
                return done;
            }
            Assert.That(Completions(.5f), Is.EqualTo(1));
            Assert.That(Completions(0f), Is.Zero, "control: the 0.18 m waypoint tolerance alone never completes");
        }

        [Test]
        public void NoReachableFrontierIsReportedOncePerScanSoAnAgentCanAct()
        {
            var c = Settings(); c.search_requires_ceiling = true;
            var b = new RobotBrain(Calibration(), c, "A", 0); var all = new List<BrainEvent>();
            for (int i = 0; i < 30; i++) { double t = i / 15.0; Motion(b, t); b.OnRgbd(Image(t, true)); b.OnAirTemperature(t, 20); b.Step(t, 1f / 15); all.AddRange(b.DrainEvents()); }
            Assert.That(all.Count(e => e.kind == "no_frontier"), Is.EqualTo(1), "open floor with no walls or ceiling: nothing known to be indoors to search; events: " +
                string.Join(",", all.Select(e => e.kind).Distinct()));
            Assert.That(all.Count(e => e.kind == "goal" && e.reason.StartsWith("No reachable frontier is visible")), Is.EqualTo(1), "one scan");
        }

        [Test]
        public void TimeLimitReturnsAndEmitsMissionCompleteOnlyOnce()
        {
            var c = Settings(); c.mission_time_limit_s = .1f; var b = new RobotBrain(Calibration(), c, "A", 0);
            var all = new List<BrainEvent>();
            for (int i = 0; i < 12; i++)
            {
                double t = i / 15.0; Motion(b, t); b.OnRgbd(Image(t, true)); b.OnAirTemperature(t, 20);
                b.Step(t, 1f / 15); all.AddRange(b.DrainEvents());
            }
            Assert.That(all.Count(e => e.kind == "mission_complete"), Is.EqualTo(1));
            Assert.That(b.Snapshot().state, Is.EqualTo("complete"));
        }

        /// <summary>A 360 degree sweep of an empty square room centred on the origin, seen from p with heading yaw.</summary>
        static LidarScan RoomScan(double t, Vector2 p, float yaw, float half = 2f, int beams = 360)
        {
            var ranges = new float[beams]; float inc = 2 * Mathf.PI / beams, min = -Mathf.PI;
            for (int i = 0; i < beams; i++)
            {
                float a = yaw + min + i * inc; var d = new Vector2(-Mathf.Sin(a), Mathf.Cos(a)); float r = float.PositiveInfinity;
                if (d.x > 1e-6f) r = Mathf.Min(r, (half - p.x) / d.x); if (d.x < -1e-6f) r = Mathf.Min(r, (-half - p.x) / d.x);
                if (d.y > 1e-6f) r = Mathf.Min(r, (half - p.y) / d.y); if (d.y < -1e-6f) r = Mathf.Min(r, (-half - p.y) / d.y);
                ranges[i] = r;
            }
            return new LidarScan(t, min, inc, .15f, 12f, 0f, .45f, ranges);
        }
        static float SafeForward(RobotBrain b, float requested) => (float)typeof(RobotBrain)
            .GetMethod("SafeForward", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).Invoke(b, new object[] { requested });

        [Test]
        public void LidarMapsWallsAndFloorButDoesNotCountAsSearched()
        {
            var b = new RobotBrain(Calibration(), Settings(), "A", 0);
            Motion(b, 0); b.OnAirTemperature(0, 20); b.OnLidar(RoomScan(0, Vector2.zero, 0)); b.Step(0, 1f / 60);
            var g = Map(b); int wall = 0, inside = 0;
            for (int i = 0; i < g.Count; i++)
            {
                if (g.LidarOccupancy(i) != 255) continue;
                var p = g.Centre(i); float edge = Mathf.Max(Mathf.Abs(p.x), Mathf.Abs(p.y));
                if (Mathf.Abs(edge - 2f) <= .1f) wall++; else inside++;
            }
            Assert.That(wall, Is.GreaterThan(100)); Assert.That(inside, Is.Zero);
            int floor = g.Cell(new Vector2(1.05f, -.95f));
            Assert.That(g.Occupancy(floor), Is.EqualTo(1));
            Assert.That(g.OwnObserved(floor), Is.False, "Only a camera search makes floor searched.");
            Assert.That(g.Unsearched(floor), Is.True);
            Assert.That(new GridPlanner(g, Settings()).SelectFrontier(Vector2.zero).HasValue, Is.True);
            Room(b, -2, 2, -2, 2);
            Assert.That(new GridPlanner(g, Settings()).SelectFrontier(Vector2.zero).HasValue, Is.False);
        }

        [Test]
        public void LongUnsearchedEdgeOffersSeveralFrontiersAndClaimsStayLocal()
        {
            // LiDAR maps an 8 m room; cameras have searched only its west half. The 8 m edge between them must be
            // offered as several targets, and a partner's claim on one must leave the others available.
            var b = new RobotBrain(Calibration(), Settings(), "B", 0, new EntryPose(-3, 0, 90));
            Motion(b, 0); b.OnAirTemperature(0, 20); b.OnLidar(RoomScan(0, new Vector2(-3, 0), 0, 4f, 720)); b.Step(0, 1f / 60);
            Room(b, -4, 0, -4, 4);
            var options = b.FrontierOptions(0, 10);
            Assert.That(options.Count, Is.GreaterThanOrEqualTo(3), "an 8 m edge is more than one target");
            var a = new RobotBrain(Calibration(), Settings(), "A", 0); Room(a, -4, 0, -4, 4);
            var m = a.BuildMessage(0); var near = options[0].position;
            m.Claim = new BrainClaim { centre = near, radiusM = 2.5f, expiresAt = 15 };
            b.OnPeerMessage(Wire(m), .1);
            var after = b.FrontierOptions(.1, 10);
            Assert.That(after.Any(), Is.True, "frontiers away from the claim stay available");
            float exclusion = 2.5f + Settings().peer_claim_radius_m;   // the partner's claim plus the claim we would make
            Assert.That(after.All(f => Vector2.Distance(f.position, near) > exclusion), Is.True, "no frontier whose own claim would overlap the partner's");
            // No pick-then-yield loop: a step with the partner's claim live must not yield the goal it just chose.
            Motion(b, .2); b.OnAirTemperature(.2, 20); b.OnLidar(RoomScan(.2, new Vector2(-3, 0), 0, 4f, 720)); b.Step(.2, .2f);
            for (int i = 1; i <= 6; i++) { double t = .2 + i / 60.0; Motion(b, t); b.OnAirTemperature(t, 20); b.Step(t, 1f / 60); }
            Assert.That(b.DrainEvents().Count(e => e.kind == "goal_yielded"), Is.Zero);
        }

        /// <summary>A 360 beam sweep of wall segments (building frame) from a true pose.</summary>
        static LidarScan SegmentScan(double t, Vector2 p, float yaw, (Vector2 a, Vector2 b)[] walls, int beams = 720)
        {
            var ranges = new float[beams]; float inc = 2 * Mathf.PI / beams, min = -Mathf.PI;
            for (int i = 0; i < beams; i++)
            {
                float ang = yaw + min + i * inc; var d = new Vector2(-Mathf.Sin(ang), Mathf.Cos(ang)); float best = 0;
                foreach (var w in walls)
                {
                    Vector2 e = w.b - w.a; float den = d.x * e.y - d.y * e.x; if (Mathf.Abs(den) < 1e-6f) continue;
                    Vector2 q = w.a - p; float r = (q.x * e.y - q.y * e.x) / den, u = (q.x * d.y - q.y * d.x) / den;
                    if (r > .15f && r < 12f && u >= 0 && u <= 1 && (best == 0 || r < best)) best = r;
                }
                ranges[i] = best;
            }
            return new LidarScan(t, min, inc, .15f, 12f, 0f, .45f, ranges);
        }
        static (Vector2, Vector2)[] Box(float x0, float y0, float x1, float y1) => new[]
        {
            (new Vector2(x0, y0), new Vector2(x1, y0)), (new Vector2(x1, y0), new Vector2(x1, y1)),
            (new Vector2(x1, y1), new Vector2(x0, y1)), (new Vector2(x0, y1), new Vector2(x0, y0)),
        };
        // A rectangle with off-centre pillars: only one turn and offset fits it, as in a furnished room.
        static readonly (Vector2, Vector2)[] RoomOne = Box(-3, -2f, 5.5f, 4f).Concat(Box(2.2f, 1.6f, 2.9f, 2.3f)).Concat(Box(-2.4f, -1.6f, -1.9f, -.4f)).ToArray();
        static readonly (Vector2, Vector2)[] RoomTwo = Box(-5, -1.5f, 2.5f, 3f).Concat(Box(-1.2f, -.6f, -.3f, .1f)).ToArray();

        /// <summary>Two robots, each in its own frame, walking past each other: returns the brains after radio exchanges.</summary>
        static (RobotBrain a, RobotBrain b, List<BrainEvent> eventsA) LinkWalk((Vector2, Vector2)[] roomA, (Vector2, Vector2)[] roomB)
        {
            var c = Settings(); c.frames_shared_at_start = false;
            var a = new RobotBrain(Calibration(), c, "A", 0) { LinkDiagnostics = true }; var b = new RobotBrain(Calibration(), c, "B", 0);
            Vector2 startB = new Vector2(.5f, 2.5f); float yawB = Mathf.PI;          // B faces A's -y
            var eventsA = new List<BrainEvent>();
            for (int tick = 0; tick <= 120; tick++)
            {
                double t = tick / 20.0; float d = Mathf.Min(.4f * (float)t, 2.4f), v = .4f * (float)t < 2.4f ? .4f : 0;
                var truthA = new Vector2(0, d); var truthB = startB + new Vector2(0, -d);
                Motion(a, t, v); Motion(b, t, v); a.OnAirTemperature(t, 20); b.OnAirTemperature(t, 20);
                if (tick % 2 == 0) { a.OnLidar(SegmentScan(t, truthA, 0, roomA)); b.OnLidar(SegmentScan(t, truthB, yawB, roomB)); }
                a.Step(t, .05f); b.Step(t, .05f);
                if (tick % 40 == 0 && tick > 0)
                {
                    var ma = a.BuildMessage(t); var mb = b.BuildMessage(t);
                    if (mb != null) a.OnPeerMessage(Wire(mb), t);
                    if (ma != null) b.OnPeerMessage(Wire(ma), t);
                }
                eventsA.AddRange(a.DrainEvents()); b.DrainEvents();
            }
            return (a, b, eventsA);
        }

        [Test]
        public void LidarFingerprintsLinkTwoRobotFramesWithoutAnyOperatorPose()
        {
            var (a, _, events) = LinkWalk(RoomOne, RoomOne);
            Assert.That(events.Count(e => e.kind == "frame_linked"), Is.EqualTo(1), string.Join(" | ", events.Where(e => e.kind == "place_recognised" || e.kind == "place_rejected").Select(e => e.kind + ": " + e.reason + " " + string.Join(" ", e.numbers.Select(kv => $"{kv.Key}={kv.Value:F2}")))));
            Assert.That(a.TryPartnerLink("B", out var link), Is.True);
            // Truth: B's frame origin is its start (0.5, 2.5) in A's frame, turned 180 degrees.
            // Each map stores walls in 10 cm cells, so each robot can sit up to half a cell off truth per axis: two maps
            // can disagree by about 0.14 m before any matching error.
            Assert.That(Vector2.Distance(link.position, new Vector2(.5f, 2.5f)), Is.LessThan(.15f), $"link at {link.position}");
            Assert.That(Mathf.Abs(Mathf.DeltaAngle(link.yawRad * Mathf.Rad2Deg, 180f)), Is.LessThan(2f), $"link yaw {link.yawRad * Mathf.Rad2Deg}");
            var peer = a.Snapshot().peers.Single();
            Assert.That(peer.linked, Is.True);
            Assert.That(Vector2.Distance(peer.pose.position, new Vector2(.5f, .1f)), Is.LessThan(.15f), "B's last pose converted into A's frame");
        }

        [Test, Explicit("timing probe, run by hand")]
        public void StepCostProbe()
        {
            string Run(System.Action<BrainConfig> tweak)
            {
                var c = BrainConfig.Load(); tweak(c);
                var b = new RobotBrain(Calibration(), c, "A", 0); var sw = System.Diagnostics.Stopwatch.StartNew();
                for (int tick = 0; tick < 120; tick++)
                {
                    double t = tick / 60.0; Motion(b, t, .3f); b.OnAirTemperature(t, 20);
                    if (tick % 6 == 0) b.OnLidar(SegmentScan(t, new Vector2(0, .3f * (float)t), 0, RoomOne));
                    if (tick % 4 == 0) b.OnRgbd(Image(t, true, 3000));
                    b.Step(t, 1f / 60);
                }
                return $"{sw.Elapsed.TotalMilliseconds / 120:F1} ms/step";
            }
            Assert.Fail("all " + Run(_ => { }) + " | no ceiling rule " + Run(c => c.search_requires_ceiling = false) +
                " | shared frames " + Run(c => c.frames_shared_at_start = true) + " | 44 m grid " + Run(c => c.grid_size_m = 44) +
                " | no scan matching " + Run(c => { c.scan_window_m = 0; c.scan_window_deg = 0; }));
        }

        [Test]
        public void OpenGroundAlongOneWallNeverLinks()
        {
            // Check run 3 found this: two robots outside, each seeing one long facade and empty space, matched each
            // other's start (fingerprint distance 0.08). Empty sweeps and a single wall must never link maps.
            var facade = new[] { (new Vector2(-20, 3), new Vector2(20, 3)) };
            var (a, _, events) = LinkWalk(facade, facade);
            Assert.That(events.Any(e => e.kind == "frame_linked"), Is.False);
            Assert.That(events.Any(e => e.kind == "place_recognised"), Is.False, "not even a candidate");
        }

        [Test]
        public void DifferentRoomsNeverLinkAndHoldBackThePartnersMap()
        {
            var (a, _, events) = LinkWalk(RoomOne, RoomTwo);
            Assert.That(events.Any(e => e.kind == "frame_linked"), Is.False, "control: no shared place, no link");
            Assert.That(a.TryPartnerLink("B", out _), Is.False);
            var g = Map(a); int peerCells = 0;
            for (int i = 0; i < g.Count; i++) if (g.PeerExplored(i)) peerCells++;
            Assert.That(peerCells, Is.Zero, "an unlinked partner's map is never drawn into this frame");
            Assert.That(a.Snapshot().peers.Single().linked, Is.False);
        }

        [Test]
        public void ScanMatchingStopsOdometryDriftFromGrowing()
        {
            // Odometry reads 20 % fast while the robot drives 0.3 m/s north for 6 s in a 4 m room. Without matching the
            // error grows with distance (0.06 m per second). With matching it stays bounded by the map's quantisation:
            // a wall is stored in the 10 cm cell its surface falls in, so the pose can sit up to half a cell off truth
            // in each axis, but it does not grow. Control: a zero search window, so no correction is possible.
            var log = new System.Text.StringBuilder();
            (float mid, float end) Error(BrainConfig c)
            {
                c.localization_report_s = 1f; float mid = 0;
                var b = new RobotBrain(Calibration(), c, "A", 0, new EntryPose(0, -1, 0)); Vector2 truth = Vector2.zero;
                for (int tick = 0; tick <= 120; tick++)
                {
                    double t = tick / 20.0; truth = new Vector2(0, -1 + .3f * (float)t);
                    Motion(b, t, .36f); b.OnAirTemperature(t, 20);
                    if (tick % 2 == 0) b.OnLidar(RoomScan(t, truth, 0));
                    b.Step(t, .05f);
                    if (tick == 60) mid = Vector2.Distance(b.Snapshot().pose.position, truth);
                    foreach (var e in b.DrainEvents().Where(e => e.kind == "localization"))
                        log.Append($"[t {e.t:F0} matched {e.numbers["matched"]} fit {e.numbers["mean_fit"]:F2} dy {e.numbers["dy_m"]:F3}] ");
                }
                log.Append("|| ");
                return (mid, Vector2.Distance(b.Snapshot().pose.position, truth));
            }
            var matching = Settings(); var control = Settings(); control.scan_window_m = 0; control.scan_window_deg = 0;
            var with = Error(matching); var without = Error(control);
            string report = $"error at 3 s / 6 s: matched {with.mid:F3} / {with.end:F3} m, control {without.mid:F3} / {without.end:F3} m. {log}";
            TestContext.WriteLine(report);
            Assert.That(without.end, Is.GreaterThan(.3f), report);
            Assert.That(with.end, Is.LessThan(.08f), report);
            Assert.That(with.end - with.mid, Is.LessThan(.02f), "Matched error must not keep growing. " + report);
        }

        [Test]
        public void PlanningStartsFromASpotTighterThanThePlanningClearance()
        {
            var c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f; var g = OpenGrid(c);
            for (float x = -3; x <= 3; x += .05f) g.ObserveOccupied(g.Cell(new Vector2(x, .32f)));   // a wall 0.27 m ahead
            var planner = new GridPlanner(g, c); var from = new Vector2(.05f, .05f);
            var path = planner.Plan(from, new Vector2(.05f, -2.5f));
            Assert.That(path, Is.Not.Empty, "A robot closer to a wall than the planning clearance must still be able to leave.");
            Assert.That(Vector2.Distance(path[0], from), Is.LessThan(.15f));
            Assert.That(planner.PathSafe(path), Is.True);
            Assert.That(planner.Plan(new Vector2(.05f, .35f), new Vector2(.05f, -2.5f)), Is.Empty, "Control: no lead out of a wall cell.");
        }

        [Test]
        public void SweptPathLimitsForwardSpeedOnlyForObstaclesAhead()
        {
            var c = Settings(); c.robot_radius_m = .25f;
            var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -3, 3, -3, 3);
            Motion(b, 0); b.OnAirTemperature(0, 20); b.Step(0, 1f / 60); var g = Map(b);
            for (float y = -2; y <= 2; y += .05f) { g.ObserveOccupied(g.Cell(new Vector2(.36f, y))); g.ObserveOccupied(g.Cell(new Vector2(-.36f, y))); }
            for (float x = -1; x <= 1; x += .05f) g.ObserveOccupied(g.Cell(new Vector2(x, -.3f)));
            Assert.That(SafeForward(b, .6f), Is.EqualTo(.6f), "Door jambs beside the body and a wall behind it do not limit forward speed.");
            for (float x = -.3f; x <= .3f; x += .05f) g.ObserveOccupied(g.Cell(new Vector2(x, .55f)));
            float limited = SafeForward(b, .6f);
            Assert.That(limited, Is.GreaterThan(0f).And.LessThan(.6f), "A wall 0.55 m ahead allows only a speed that can stop in time.");
            for (float x = -.3f; x <= .3f; x += .05f) g.ObserveOccupied(g.Cell(new Vector2(x, .25f)));
            Assert.That(SafeForward(b, .6f), Is.Zero);
            Assert.That(b.DrainEvents().Count(e => e.kind == "collision_risk"), Is.EqualTo(1), "One event per blocked episode.");
        }

        [Test]
        public void TheBurningAreaCountsMeasuredFireNearTheBlockNotForecastOrFarFire()
        {
            // The agent judges water against the fire that blocks its way. 1 m2 measured burning at the block counts; a
            // forecast (rising but below fire_c) and fire 5 m away do not (controls).
            var c = Settings(); c.grid_size_m = 12; var b = new RobotBrain(Calibration(), c, "A", 0); var g = Map(b);
            for (float x = -.45f; x < .5f; x += .1f) for (float y = 1.55f; y < 2.5f; y += .1f) g.ObserveHeat(g.Cell(new Vector2(x, y)), 600, 1);
            for (float x = -.45f; x < .5f; x += .1f) g.ObserveHeat(g.Cell(new Vector2(x, 2.75f)), c.fire_c - 20, 1);
            for (float x = -.45f; x < .5f; x += .1f) g.ObserveHeat(g.Cell(new Vector2(x, 7f)), 600, 1);
            Assert.That(g.BurningAreaNear(new Vector2(0, 2f), 2f), Is.EqualTo(1f).Within(.05f));
        }

        [Test]
        public void AnExclusionCircleDoesNotReachThroughAWall()
        {
            // Codex review #27: a blacklisted goal (up to 2.5 m) or a visited spot hid frontiers in the next room. A 2.5 m
            // circle at the origin, a mapped wall at x = 1: the point beyond it (1.5, 0) is not excluded; the point on the
            // same side (-1.5, 0) still is (control).
            var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
            var c = Settings(); c.grid_size_m = 12; var b = new RobotBrain(Calibration(), c, "A", 0); var g = Map(b);
            for (float y = -2f; y <= 2f; y += .05f) g.ObserveOccupied(g.Cell(new Vector2(1.02f, y)));
            typeof(RobotBrain).GetMethod("BlacklistGoal", flags).Invoke(b, new object[] { Vector2.zero, 0.0, 0, 100.0, "test", 2.5f });
            bool Black(float x) => (bool)typeof(RobotBrain).GetMethod("GoalBlacklisted", flags).Invoke(b, new object[] { new Vector2(x, 0), 1.0 });
            Assert.That(Black(1.5f), Is.False, "beyond the wall: not hidden");
            Assert.That(Black(-1.5f), Is.True, "control: same side, hidden");
        }

        [Test]
        public void YardFrontiersOnOpenGroundGiveWayToProvenIndoorOptions()
        {
            // One-door seed 922, robot B in the yard: 6 of 8 options were yard frontiers (no indoor floor beyond), larger on
            // paper than the 2 proven ones. Here a robot at a doorway has a searched-indoors room behind it and yard its LiDAR
            // saw as open ground in front: yard options are dropped. Control: with no proven indoor option, they stay.
            var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
            (int yard, int total) Options(bool roomIndoors)
            {
                var c = Settings(); c.grid_size_m = 30; c.robot_radius_m = .25f; c.search_requires_ceiling = true; c.frontier_wide_options = 0;
                var b = new RobotBrain(Calibration(), c, "A", 0); var g = Map(b);
                var indoors = (bool[])typeof(OccupancyGrid).GetField("sightIndoors", flags).GetValue(g);
                var open = (bool[])typeof(OccupancyGrid).GetField("openGround", flags).GetValue(g);
                for (int i = 0; i < g.Count; i++)
                {
                    var p = g.Centre(i);
                    bool facade = Mathf.Abs(p.y - 2.05f) < .1f && p.x > -4 && p.x < 12 && !(Mathf.Abs(p.x) < .5f);
                    if (facade) g.ObserveOccupied(i);
                    else if (Mathf.Abs(p.y - 2.05f) < .1f && Mathf.Abs(p.x) < .5f) { g.ObserveFree(i); if (roomIndoors) indoors[i] = true; }   // the doorway
                    else if (p.y > 2.15f && p.y < 5f && p.x > -3.9f && p.x < 3.9f) { g.ObserveFree(i); if (roomIndoors) indoors[i] = true; }
                    else if (p.y > -4f && p.y < 1.95f && p.x > -4f && p.x < 4f) { g.ObserveFree(i); open[i] = true; }
                }
                typeof(OccupancyGrid).GetField("dirty", flags).SetValue(g, true);
                typeof(OccupancyGrid).GetField("roofDirty", flags).SetValue(g, true);   // the indoor layer is derived from sightIndoors
                Motion(b, 0); b.OnAirTemperature(0, 20); b.Step(0, .05f);
                var list = b.FrontierOptions(0, 8);
                return (list.Count(f => f.position.y < 1.95f), list.Count);
            }
            var withRoom = Options(true);
            Assert.That(withRoom.total - withRoom.yard, Is.GreaterThan(0), "proven indoor options exist");
            Assert.That(withRoom.yard, Is.Zero, "no yard frontiers beside proven indoor ones");
            var noRoom = Options(false);
            Assert.That(noRoom.yard, Is.GreaterThan(0), "control: with nothing proven, the yard options stay");
        }

        [Test]
        public void APartnersWallWhereThisRobotWalkedIsItsOwnBody()
        {
            // One-door seed 922: B's LiDAR saw A walk through the one door and B's map reported walls there; at 4 s, the first
            // exchange after that, both robots lost every route and frontier. A walked from its pad 2 m ahead; the partner
            // reports walls on that path (0, 1), 0.6 m beside it, and 1 m past where it stopped. The one on its path is its
            // own body and is dropped; the others stay (controls).
            var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
            var c = Settings(); c.grid_size_m = 12; c.frames_shared_at_start = true;
            var b = new RobotBrain(Calibration(), c, "B", 0); var a = new RobotBrain(Calibration(), c, "A", 0);
            Motion(a, 0); a.OnAirTemperature(0, 20); a.Step(0, .05f); Motion(b, 0); b.OnAirTemperature(0, 20); b.Step(0, .05f);
            var track = (List<BrainPose>)typeof(RobotBrain).GetField("bodyTrack", flags).GetValue(b);
            for (float y = 0; y <= 2f; y += .1f) track.Add(new BrainPose { position = new Vector2(0, y), yawRad = 0 });
            var gb = Map(b);
            Assert.That(gb.OwnOccupancy(gb.Cell(new Vector2(0, 1f))), Is.Zero, "precondition: B's own map knows nothing at (0, 1)");
            var ga = Map(a);
            foreach (var w in new[] { new Vector2(0, 1f), new Vector2(.6f, 1f), new Vector2(0, 3f) }) ga.ObserveOccupied(ga.Cell(w));
            Motion(a, .1); a.OnAirTemperature(.1, 20); a.Step(.1, .1f);
            b.OnPeerMessage(Wire(a.BuildMessage(.1)), .1);
            Assert.That(gb.Occupancy(gb.Cell(new Vector2(0, 1f))), Is.Not.EqualTo(255), "where B's body stood is no wall");
            Assert.That(gb.Occupancy(gb.Cell(new Vector2(.6f, 1f))), Is.EqualTo(255), "control: beside its path stays");
            Assert.That(gb.Occupancy(gb.Cell(new Vector2(0, 3f))), Is.EqualTo(255), "control: past where it went stays");
        }

        [Test]
        public void ATeammateSeenStandingInTheDoorwayIsNotLeftAsAWall()
        {
            // One-door seed 923: A's own LiDAR saw B standing on A's pad in the doorway; out of view afterwards, that wall
            // stayed, A's only way out stayed shut and A counted itself cut off for the rest of the run. B's radio pose there
            // frees it; a real wall 0.6 m beside B stays (control).
            var c = Settings(); c.grid_size_m = 12; c.frames_shared_at_start = true;
            var a = new RobotBrain(Calibration(), c, "A", 0); var b = new RobotBrain(Calibration(), c, "B", 0);
            Motion(a, 0); a.OnAirTemperature(0, 20); a.Step(0, .05f);
            for (int i = 0; i <= 30; i++) { double t = i / 15.0; Motion(b, t, i < 30 ? .5f : 0); b.OnAirTemperature(t, 20); b.Step(t, i == 0 ? .05f : 1f / 15); }
            Vector2 bAt = b.Snapshot().pose.position;
            var g = Map(a);
            for (int k = 0; k < 4; k++) { g.ObserveOccupied(g.Cell(bAt)); g.ObserveOccupied(g.Cell(bAt + new Vector2(.6f, 0))); }
            Assert.That(g.OwnOccupancy(g.Cell(bAt)), Is.EqualTo(255), "precondition: A mapped B's body as a wall");
            a.OnPeerMessage(Wire(b.BuildMessage(2)), 2);
            Assert.That(g.OwnOccupancy(g.Cell(bAt)), Is.Not.EqualTo(255), "where B stands is floor");
            Assert.That(g.OwnOccupancy(g.Cell(bAt + new Vector2(.6f, 0))), Is.EqualTo(255), "control: a wall 0.6 m beside B stays");
        }

        [Test]
        public void CutOffInHeatingAirTheRobotRetreatsInsteadOfObeyingAnOrderToStay()
        {
            // One-door seed 923: A, counted cut off, turned on the spot for 30 s on its agent's orders while its air rose from 30
            // to 70 C. Cut off, heating, no route out even past the fire: the order is overridden and it retreats. Control: the
            // same in cool air keeps the order.
            var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
            RobotBrain lastBrain = null;
            (bool retreating, int overridden) Run(float air)
            {
                var c = Settings(); c.grid_size_m = 12; var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -4, 4, -2, 6);
                // A genuine cut-off: the way out is walled in (as B's body walled A's in 923), so no route exists even past fire.
                typeof(RobotBrain).GetField("home", flags).SetValue(b, new Vector2(3f, 4.5f));
                var g = Map(b);
                var events = new List<BrainEvent>();
                for (int i = 0; i < 15 * 12; i++)
                {
                    double t = i / 15.0; Motion(b, t); b.OnRgbd(Image(t, true));
                    if (i == 15)   // the route it had closes, as B's body closed A's doorway in 923
                        for (float ang = 0; ang < 360; ang += 3)
                            for (int k = 0; k < 4; k++) g.ObserveOccupied(g.Cell(new Vector2(3f, 4.5f) + 0.6f * new Vector2(Mathf.Cos(ang * Mathf.Deg2Rad), Mathf.Sin(ang * Mathf.Deg2Rad))));
                    b.OnAirTemperature(t, t < 2 ? 20 : 20 + (air - 20) * (float)Math.Min(1, (t - 2) / 6));
                    if (i == 30) b.AgentGoTo(new Vector2(0, 1), t, 60, "look again", new Vector2(0, 3));
                    b.Step(t, 1f / 15); events.AddRange(b.DrainEvents());
                }
                lastBrain = b;
                return ((bool)typeof(RobotBrain).GetField("retreating", flags).GetValue(b), events.Count(e => e.kind == "stay_overridden"));
            }
            var hot = Run(60);
            Assert.That((bool)typeof(RobotBrain).GetField("cutOff", flags).GetValue(lastBrain), Is.True, "precondition: the brain found itself cut off");
            Assert.That(hot.overridden, Is.GreaterThan(0), "the order to stay is overridden");
            Assert.That(hot.retreating, Is.True);
            Assert.That(Run(22).overridden, Is.Zero, "control: in cool air the order stands");
        }

        [Test]
        public void WithNothingLeftAtTheDoorARobotWalksInAfterItsPartner()
        {
            // One-door seed 923, robot B: A went in and B, at the pad, had no frontier at all, so it scanned and held at the
            // gate. Here B's map shows indoor floor up to 6 m ahead and A is reported 5 m in: B's goal lies on the way to A,
            // 2 m short. Controls: a partner standing outside the indoor floor, and no partner, give no goal.
            var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
            Vector2? Goal(Vector2? partnerAt, bool indoorFloor)
            {
                var c = Settings(); c.grid_size_m = 16; c.robot_radius_m = .25f; c.frames_shared_at_start = true; c.search_requires_ceiling = true;
                var b = new RobotBrain(Calibration(), c, "B", 0); var g = Map(b); Room(b, -2, 2, -1, 7);
                var indoors = (bool[])typeof(OccupancyGrid).GetField("sightIndoors", flags).GetValue(g);
                if (indoorFloor) for (int i = 0; i < g.Count; i++) { var p = g.Centre(i); if (Mathf.Abs(p.x) < 2 && p.y > 1 && p.y < 7) indoors[i] = true; }
                typeof(OccupancyGrid).GetField("roofDirty", flags).SetValue(g, true);
                Motion(b, 0); b.OnAirTemperature(0, 20); b.Step(0, .05f);
                if (partnerAt.HasValue)
                {
                    var a = new RobotBrain(Calibration(), c, "A", 0);
                    for (int i = 0; i <= 150; i++) { double t = i / 15.0; Motion(a, t, i < 150 ? .5f : 0); a.OnAirTemperature(t, 20); a.Step(t, i == 0 ? .05f : 1f / 15); }
                    b.OnPeerMessage(Wire(a.BuildMessage(10)), 10);
                }
                var args = new object[] { null };
                return (Vector2?)typeof(RobotBrain).GetMethod("FollowPartnerGoal", flags).Invoke(b, args);
            }
            var follow = Goal(new Vector2(0, 5), true);
            Assert.That(follow.HasValue, Is.True, "a way in behind the partner");
            Assert.That(follow.Value.y, Is.InRange(2.3f, 3.9f), "about 2 m short of A (A drove 5 m): " + follow);
            Assert.That(Goal(new Vector2(0, 5), false).HasValue, Is.False, "control: the partner is not on floor proven indoors");
            Assert.That(Goal(null, true).HasValue, Is.False, "control: no partner");
        }

        [Test]
        public void BehindItsPartnerAtOneDoorARobotFollowsThroughInsteadOfStopping()
        {
            // One-door seed 922, robot B: at the one door behind A, every frontier it saw lay in A's claim or A's searched
            // floor, so it yielded 120 times, scanned and ended its mission at 16.6 s. Here both robots stand at the same pad
            // seeing the same corridor; A searches it and claims the way on. B follows through rather than reporting no
            // frontier. Control: alone (no partner), the same view gives B its own frontier anyway.
            (int follows, int noFrontier, int ownGoals) Run(bool partner)
            {
                var c = Settings(); c.grid_size_m = 16; c.robot_radius_m = .25f; c.frames_shared_at_start = true; c.mission_time_limit_s = 1000;
                var b = new RobotBrain(Calibration(), c, "B", 0);
                var a = new RobotBrain(Calibration(), c, "A", 0);
                var events = new List<BrainEvent>();
                for (int i = 0; i < 15 * 6; i++)
                {
                    double t = i / 15.0;
                    Motion(a, t); a.OnRgbd(Image(t, true)); a.OnAirTemperature(t, 20); a.Step(t, 1f / 15);
                    if (partner && i % 15 == 0) { var m = a.BuildMessage(t); if (m != null) b.OnPeerMessage(Wire(m), t); }
                    Motion(b, t); b.OnRgbd(Image(t, true)); b.OnAirTemperature(t, 20); b.Step(t, 1f / 15); events.AddRange(b.DrainEvents());
                }
                return (events.Count(e => e.kind == "goal" && e.reason.StartsWith("Only the partner's")), events.Count(e => e.kind == "no_frontier"),
                        events.Count(e => e.kind == "goal" && e.reason.StartsWith("Reachable frontier")));
            }
            var behind = Run(true);
            Assert.That(behind.noFrontier, Is.Zero, "no 'nothing left' behind a partner at one door");
            Assert.That(behind.follows + behind.ownGoals, Is.GreaterThan(0), "it heads on through");
            var alone = Run(false);
            Assert.That(alone.ownGoals, Is.GreaterThan(0), "control: the view itself has a way on");
            Assert.That(alone.follows, Is.Zero, "control: alone there is no partner to follow");
        }

        [Test]
        public void OutOfContactTooLongTheRobotWalksBackToWhereItLastHeardItsPartner()
        {
            // Held-out seeds 919 and 920: out of radio contact 168 to 184 s of 200. Robot B hears A at the start, walks 4 m on
            // mapping new floor, then hears nothing: after rendezvous_after_s it walks back toward where it last heard A, and
            // a message from A ends the meeting. Control: in range of A (its full snapshot arrives every 10 s, the heartbeat), it
            // never walks back.
            (int meets, int done, Vector2? goal) Run(bool silent)
            {
                var c = Settings(); c.grid_size_m = 16; c.robot_radius_m = .25f; c.frames_shared_at_start = true;
                c.rendezvous_after_s = 14; c.rendezvous_min_unshared_cells = 20; c.rendezvous_hold_s = 4; c.mission_time_limit_s = 1000;
                var b = new RobotBrain(Calibration(), c, "B", 0); Room(b, -5, 5, -2, 10);
                var partner = new RobotBrain(Calibration(), c, "A", 0); Motion(partner, 0); partner.OnAirTemperature(0, 20); partner.Step(0, .05f);
                var events = new List<BrainEvent>(); Vector2? goal = null; double partnerT = 0;
                for (int i = 0; i < 15 * 24; i++)
                {
                    double t = i / 15.0; Motion(b, t, t > .5 && t < 8.5 ? .5f : 0); b.OnRgbd(Image(t, true)); b.OnAirTemperature(t, 20);
                    bool hear = i == 3 || (!silent && i % 30 == 0 && i > 0) || (silent && i == 15 * 22);
                    if (hear)
                    {
                        Motion(partner, t); partner.OnAirTemperature(t, 20); partner.Step(t, (float)(t - partnerT)); partnerT = t;
                        // A partner with nothing new sends nothing; its full snapshot every full_snapshot_s (10 s) is the heartbeat.
                        var sent = partner.BuildMessage(t); if (sent != null) b.OnPeerMessage(Wire(sent), t);
                    }
                    b.Step(t, 1f / 15); var e = b.DrainEvents(); events.AddRange(e);
                    if (e.Any(x => x.kind == "rendezvous")) goal = b.Snapshot().goal;
                    if (i % 30 == 0) b.BuildMessage(t);
                }
                return (events.Count(e => e.kind == "rendezvous"), events.Count(e => e.kind == "rendezvous_done"), goal);
            }
            var silent = Run(true);
            Assert.That(silent.meets, Is.EqualTo(1), "walks back once");
            Assert.That(silent.goal.HasValue && silent.goal.Value.magnitude < 1f, Is.True, "toward where it last heard A (near the start): " + silent.goal);
            Assert.That(silent.done, Is.EqualTo(1), "hearing A again ends it");
            Assert.That(Run(false).meets, Is.Zero, "control: in contact, no walking back");
        }

        [Test]
        public void LeavingBecauseOfHeatDoesNotCrawlPastWalls()
        {
            // Held-out seed 920, robot B: leaving a heating house, it walked at the 0.3 m/s near-wall speed through every
            // doorway and spent 11.8 s in air over 60 C. Leaving because of heat, a wall 0.8 m ahead keeps open speed (the
            // stopping-distance cap still applies in Command); control: searching, the same wall halves it.
            var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
            float Speed(bool leaving)
            {
                var c = Settings(); var b = new RobotBrain(Calibration(), c, "A", 0);
                if (leaving)
                {
                    typeof(RobotBrain).GetField("returning", flags).SetValue(b, true);
                    typeof(RobotBrain).GetField("returnReason", flags).SetValue(b, "No reachable frontier was left and the air was heating.");
                }
                var args = new object[] { .8f, null };
                return (float)typeof(RobotBrain).GetMethod("PursuitSpeed", flags).Invoke(b, args);
            }
            var cfg = Settings();
            Assert.That(Speed(true), Is.EqualTo(cfg.open_speed_mps), "leaving because of heat");
            Assert.That(Speed(false), Is.EqualTo(cfg.near_speed_mps), "control: searching near a wall");
        }

        [Test]
        public void TheBodyIsTheCapsuleUnityCollidesNotARoundRobot()
        {
            // Held-out seed 920, robot B: five stalls turning at the foyer-kitchen opening with its face 0.2 to 0.4 m from the
            // wall edge. Unity's body is a capsule 0.75 m long (radius 0.18 m); the brain treated it as a 0.25 m circle, so it
            // drove its nose to the wall and swung it into corners. Forward: a wall 0.40 m ahead leaves no forward room (the
            // nose reaches 0.375 m); control 0.60 m does. Turning: a wall cell 0.2 to 0.3 m to the front right (inside the 0.375 m the ends sweep) blocks
            // a 90 degree turn toward it;
            // control: a 90 degree turn the other way is clear.
            var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
            RobotBrain Fresh(out OccupancyGrid g)
            {
                var c = Settings(); c.robot_radius_m = .25f; c.body_half_length_m = .375f; c.body_radius_m = .18f;
                var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -3, 3, -3, 3);
                Motion(b, 0); b.OnAirTemperature(0, 20); b.Step(0, 1f / 60); g = Map(b); return b;
            }
            float Room0(float wallY)
            {
                var b = Fresh(out var g);
                for (float x = -.3f; x <= .3f; x += .05f) g.ObserveOccupied(g.Cell(new Vector2(x, wallY)));
                return (float)typeof(RobotBrain).GetMethod("SweptClearance", flags).Invoke(b, null) - Settings().collision_margin_m;
            }
            Assert.That(Room0(.40f), Is.LessThanOrEqualTo(0f), "the nose is at the wall 0.40 m ahead");
            Assert.That(Room0(.60f), Is.GreaterThan(0f), "control: room to move 0.60 m from a wall");
            bool Swing(float deg)
            {
                var b = Fresh(out var g); g.ObserveOccupied(g.Cell(new Vector2(.26f, .26f)));
                typeof(OccupancyGrid).GetMethod("PrepareNavigation", flags).Invoke(g, null);
                return (bool)typeof(RobotBrain).GetMethod("SwingClear", flags).Invoke(b, new object[] { deg });
            }
            float toward = -90f;   // from facing +y to facing +x: the brain's heading is atan2(-x, y)
            Assert.That(Swing(toward), Is.False, "the nose would swing into the corner");
            Assert.That(Swing(-toward), Is.True, "control: turning away from it");
        }

        [Test]
        public void HeldCollisionLimitBecomesOneRecoveryWithoutPhantomWalls()
        {
            var c = Settings(); c.robot_radius_m = .25f;
            var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -3, 3, -3, 3);
            var all = new List<BrainEvent>(); int tick = 0;
            DriveTick(b, tick++); var g = Map(b);
            for (float x = -.3f; x <= .3f; x += .05f) g.ObserveOccupied(g.Cell(new Vector2(x, .35f)));
            int occupiedBefore = Enumerable.Range(0, g.Count).Count(i => g.OwnOccupancy(i) == 255);
            for (; tick < 100; tick++)
            {
                // The follower asks for forward speed on every step it is not recovering.
                DriveTick(b, tick); if (b.Snapshot().state != "recover") SafeForward(b, .6f); all.AddRange(b.DrainEvents());
            }
            Assert.That(all.Count(e => e.kind == "collision_risk"), Is.EqualTo(1));
            var hold = all.Where(e => e.kind == "collision_hold").ToList();
            Assert.That(hold.Count, Is.EqualTo(1));
            Assert.That(hold[0].numbers["marked_cells"], Is.Zero, "A mapped obstacle must not add a contact strip.");
            Assert.That(Enumerable.Range(0, g.Count).Count(i => g.OwnOccupancy(i) == 255), Is.EqualTo(occupiedBefore));
        }

        [Test]
        public void PeerWallsNeverOverrideCellsThisRobotHasSeenFree()
        {
            var a = new RobotBrain(Calibration(), Settings(), "A", 0);
            var b = new RobotBrain(Calibration(), Settings(), "B", 0);
            Room(a, -3, 3, -3, 3);
            var seen = new Vector2(.05f, 2.05f); var unseen = new Vector2(2.55f, 2.55f);
            Map(a).ObserveOccupied(Map(a).Cell(seen)); Map(a).ObserveOccupied(Map(a).Cell(unseen));
            var g = Map(b); g.ObserveFree(g.Cell(seen));
            b.OnPeerMessage(Wire(a.BuildMessage(0)), .1); g.DecayTo(30);
            Assert.That(g.OccupancyAt(seen), Is.EqualTo(1), "Own free evidence wins, however old.");
            Assert.That(g.OccupancyAt(unseen), Is.EqualTo(255), "Control: the peer's wall fills a cell this robot never saw.");
        }

        [Test]
        public void SnapshotAndDrainedEventsDoNotExposeMutableBrainBuffers()
        {
            var b = new RobotBrain(Calibration(), Settings(), "A", 0);
            Motion(b, 0); b.OnAirTemperature(0, 20); b.OnRgbd(Image(0, true)); b.Step(0, 1f / 60);
            var snapshot = b.Snapshot(); var expected = b.Snapshot();
            Array.Clear(snapshot.occupancy, 0, snapshot.occupancy.Length);
            Array.Clear(snapshot.heat, 0, snapshot.heat.Length);
            CollectionAssert.AreEqual(expected.occupancy, b.Snapshot().occupancy);
            CollectionAssert.AreEqual(expected.heat, b.Snapshot().heat);
            b.DrainEvents(); Assert.That(b.DrainEvents(), Is.Empty);
        }

        [Test]
        public void OpenGroundTheLidarSawDoesNotWinAWideSlotForAnUnprovenFrontier()
        {
            // Held-out seed 917, robot B: with wide slots ranked across unproven frontiers too, all three went to frontiers by
            // the front door whose "large area beyond" was the yard; B spent 40 s turning to check them while its proven list
            // flickered empty. Its own LiDAR had marked that yard as open ground at its outdoor spawn. For ranking, unproven
            // floor the LiDAR saw as open ground does not count; mapped floor behind an interior opening still does.
            var c = Settings(); c.grid_size_m = 12;
            var g = new OccupancyGrid(c);   // nothing searched by a camera; the LiDAR maps the floor below
            var lidar = (float[])typeof(OccupancyGrid).GetField("lidarOdds", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(g);
            var open = (bool[])typeof(OccupancyGrid).GetField("openGround", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(g);
            var yard = new List<int>();
            for (int i = 0; i < g.Count; i++) { var p = g.Centre(i); if (p.y > 0 && p.y < 4 && Mathf.Abs(p.x) < 3) { lidar[i] = -c.log_odds_cap; yard.Add(i); } }
            typeof(OccupancyGrid).GetField("dirty", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).SetValue(g, true);
            float mapped = g.UnsearchedFloorBeyond(new Vector2(0, .5f), 12f, false);
            Assert.That(mapped, Is.GreaterThan(10f), "control: mapped unsearched floor counts while nothing marks it open ground");
            foreach (int i in yard) open[i] = true;
            Assert.That(g.UnsearchedFloorBeyond(new Vector2(0, .5f), 12f, false), Is.LessThan(1f), "the same floor seen as open ground does not");
        }

        [Test]
        public void TheShippedViewingWindowLiesWhereTheCameraCanSeeALyingPerson()
        {
            // Since 20 Oct the camera sits on a 1.6 m mast (it was 0.5 m). The 1.0 to 1.5 m viewing window set for the old mount
            // put a lying person below the bottom of the image, so every reflex stop and every closer look was blind (held-out
            // seed 915: the person in the study was seen at 4.7 m, too far to confirm, then from 1.6 m, below the view). The
            // shipped window must start past the floor the camera cannot see and end inside the confirm range.
            var c = BrainConfig.Load(); var cam = SensorConfig.Load().camera;
            float bottom = cam.mount.pitch_down_deg * Mathf.Deg2Rad + Mathf.Atan((cam.height - cam.cy) / cam.fy);
            float blind = cam.mount.forward_offset_m + cam.mount.height_m / Mathf.Tan(bottom);
            Assert.That(blind, Is.InRange(2.0f, 2.3f), "the 1.6 m mast sees the floor from about 2.16 m");
            Assert.That(c.survivor_stop_min_m, Is.GreaterThan(blind), "no viewing stop inside the blind floor");
            Assert.That(c.survivor_standoff_m, Is.GreaterThan(blind));
            Assert.That(c.survivor_stop_max_m, Is.LessThan(c.search_max_range_m), "the stop stays inside the confirm range");
        }

        [Test]
        public void ConfigurationLoadsThresholdsAndSensorMotionLimits()
        {
            var c = BrainConfig.Load(); var sensorPath = Path.Combine(SensorConfig.ConfigDir, "sensors.json");
            var sensor = JsonUtility.FromJson<SensorLimitDocument>(File.ReadAllText(sensorPath));
            Assert.That(c.limits.max_forward_mps, Is.EqualTo(sensor.limits.max_forward_mps));
            Assert.That(c.limits.max_yaw_rate_dps, Is.EqualTo(sensor.limits.max_yaw_rate_dps));
            Assert.That(c.cell_m, Is.EqualTo(.1f)); Assert.That(c.grid_size_m, Is.EqualTo(50));
            // Each robot's grid is centred on where it entered: no building coordinates.
            Assert.That(c.fixed_grid_origin, Is.False); Assert.That(c.frames_shared_at_start, Is.False);
        }
        [Serializable] public sealed class SensorLimitDocument { public BrainConfig.MotionLimits limits; }

        // These cases fail if peer state is ignored, replayed, or projected in a robot-local frame.
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
        static BrainMessage Wire(BrainMessage m)
        {
            var bytes = m.ToBytes();
            Assert.That(m.EstimatedBytes, Is.EqualTo(bytes.Length));
            return BrainMessage.FromBytes(bytes);
        }

        [Test]
        public void EntryPoseProjectsFirstWallIntoBuildingCells()
        {
            var s = Calibration(); s.camera.mount.pitch_down_deg = 0; s.camera.mount.forward_offset_m = 0;
            var b = new RobotBrain(s, Settings(), "B", 0, new EntryPose(3, -2, 90));
            b.OnRgbd(Image(0, true)); b.Step(0, .02f);
            Assert.That(b.Snapshot().pose.yawRad, Is.EqualTo(-Mathf.PI / 2).Within(.0001));
            Assert.That(Map(b).OccupancyAt(new Vector2(5.01f, -2.01f)), Is.EqualTo(255));
            Assert.That(Map(b).OccupancyAt(new Vector2(4, -2.01f)), Is.EqualTo(1));
            var before = b.Snapshot(); b.ApplyCorrection(new BrainPose());
            CollectionAssert.AreEqual(before.occupancy, b.Snapshot().occupancy);
            Assert.That(b.Snapshot().pose.position, Is.EqualTo(before.pose.position));
            b.ApplyCorrection(new BrainPose { position = new Vector2(1, 2), yawRad = .1f });
            CollectionAssert.AreEqual(before.occupancy, b.Snapshot().occupancy);
        }

        [Test]
        public void ReturnUsesTheEnteredDoorInBuildingCoordinates()
        {
            var c = Settings(); c.mission_time_limit_s = .1f;
            var b = new RobotBrain(Calibration(), c, "B", 0, new EntryPose(3, 2, 90));
            foreach (double t in new[] { 0.0, .2 })
            { Motion(b, t); b.OnAirTemperature(t, 20); b.OnRgbd(Image(t, true)); b.Step(t, .2f); }
            Assert.That(b.Snapshot().state, Is.EqualTo("complete"));
            Assert.That(b.DrainEvents().Count(e => e.kind == "mission_complete"), Is.EqualTo(1));
        }

        [Test]
        public void PeerRoomsAreTraversableButNeverBecomeFrontierGoals()
        {
            var a = new RobotBrain(Calibration(), Settings(), "A", 0, new EntryPose(-2, 0, 90));
            var b = new RobotBrain(Calibration(), Settings(), "B", 0, new EntryPose(2, 0, 270));
            Room(a, -4, 0, -3, 3); Room(b, -1, 4, -3, 3);
            b.OnPeerMessage(Wire(a.BuildMessage(0)), .1);
            var g = Map(b); var planner = new GridPlanner(g, Settings());
            Assert.That(planner.Plan(new Vector2(2, 0), new Vector2(-3, 0)), Is.Not.Empty);
            var frontier = planner.SelectFrontier(new Vector2(2, 0));
            Assert.That(frontier.HasValue, Is.True);
            Assert.That(g.PeerExplored(g.Cell(frontier.Value)), Is.False);
            a.OnPeerMessage(Wire(b.BuildMessage(.1)), .2);
            Assert.That(a.DrainEvents().Count(e => e.kind == "map_shared"), Is.EqualTo(1));
            Assert.That(b.DrainEvents().Count(e => e.kind == "map_shared"), Is.EqualTo(1));
            b.OnPeerMessage(Wire(a.BuildMessage(2)), 2.1);
            Assert.That(b.DrainEvents().Count(e => e.kind == "map_shared"), Is.Zero);
        }

        [Test]
        public void CeilingPointsOnAWallFaceDoNotMakeTheYardIndoors()
        {
            // Run 6: points from the top of an exterior wall face landed just outside the wall cell and counted as
            // ceiling over the yard. Only ceiling evidence clear of walls may make floor indoors.
            var c = Settings(); c.grid_size_m = 12; var g = OpenGrid(c);
            for (float y = -3; y <= 3; y += .05f) g.ObserveOccupied(g.Cell(new Vector2(.05f, y)));   // a wall at x = 0..0.1
            var roof = (byte[])typeof(OccupancyGrid).GetField("roofHits", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(g);
            for (float y = -1; y <= 1; y += .1f) roof[g.Cell(new Vector2(-.15f, y))] = 255;              // wall-face points, 0.15 m out
            g.ObserveFree(g.Cell(new Vector2(-.15f, 0)));
            Assert.That(g.Indoors(new Vector2(-.9f, 0)), Is.False, "yard 0.9 m outside the wall");
            for (float y = -1; y <= 1; y += .1f) roof[g.Cell(new Vector2(-1.35f, y))] = 255;             // a real ceiling, clear of walls
            g.ObserveFree(g.Cell(new Vector2(-1.35f, 0)));
            typeof(OccupancyGrid).GetField("roofDirty", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).SetValue(g, true);
            Assert.That(g.Indoors(new Vector2(-.9f, 0)), Is.True, "control: the same spot under real ceiling evidence");
            // An unmapped stretch of wall between that ceiling and the yard must not let "indoors" through.
            var u = new OccupancyGrid(c);
            for (float y = -2; y <= 2; y += .1f) for (float x = -3; x <= -.25f; x += .1f) u.ObserveFree(u.Cell(new Vector2(x, y)));   // room floor
            for (float y = -2; y <= 2; y += .1f) for (float x = .05f; x <= 2; x += .1f) u.ObserveFree(u.Cell(new Vector2(x, y)));     // yard floor
            var uroof = (byte[])typeof(OccupancyGrid).GetField("roofHits", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(u);
            for (float y = -1; y <= 1; y += .1f) uroof[u.Cell(new Vector2(-.35f, y))] = 255;
            Assert.That(u.Indoors(new Vector2(-.6f, 0)), Is.True, "control: floor beside the ceiling");
            Assert.That(u.Indoors(new Vector2(.15f, 0)), Is.False, "yard 0.5 m away, behind 0.3 m of wall not yet seen");
        }

        /// <summary>A room seen from the test camera: ceiling at 2.8 m, floor, and a far wall wallM ahead; each pixel's depth
        /// is its ray's first hit (optical z, mm).</summary>
        static RgbdFrame RoomImage(double t, float wallM) => RoomImage(t, wallM, 2.8f);
        static RgbdFrame RoomImage(double t, float wallM, float ceilingM)
        {
            var frame = Image(t, true, 0); float p = 10 * Mathf.Deg2Rad, sp = Mathf.Sin(p), cp = Mathf.Cos(p);
            for (int v = 0; v < 60; v++) for (int u = 0; u < 80; u++)
            {
                float dn = (v - 29.5f) / 52f, up = -(sp + dn * cp), forward = cp - dn * sp, z = float.PositiveInfinity;
                if (up > 1e-4f) z = Mathf.Min(z, (ceilingM - 1.6f) / up);        // ceiling
                if (up < -1e-4f) z = Mathf.Min(z, 1.6f / -up);                   // floor
                if (forward > 1e-4f) z = Mathf.Min(z, (wallM - .25f) / forward); // far wall
                frame.Depth[v * 80 + u] = (ushort)Mathf.Clamp(Mathf.RoundToInt(z * 1000), 0, 65535);
            }
            return frame;
        }

        [Test]
        public void FloorBehindATableIsMappedButNotSearched()
        {
            // Held-out seed 914 run 1: a person slumped by the kitchen island lay on floor the depth rays had crossed high on
            // their way to the wall, so it counted as searched though no depth image ever showed it. A 0.75 m table spans the
            // view 2.6 to 3.2 m ahead; the floor behind it at 3.6 m is crossed only by rays 0.6 m up. Control: search_low_m 0
            // (the old rule) counts it searched.
            (bool front, bool shadowSearched, bool shadowFree, bool table) Look(float low)
            {
                var c = Settings(); c.search_low_m = low;
                var b = new RobotBrain(Calibration(), c, "A", 0);
                Motion(b, 0); b.OnAirTemperature(0, 20); b.Step(0, 1f / 60);
                var frame = Image(.1, true, 0); float pitch = 10 * Mathf.Deg2Rad, sp = Mathf.Sin(pitch), cp = Mathf.Cos(pitch);
                for (int v = 0; v < 60; v++) for (int u = 0; u < 80; u++)
                {
                    float dn = (v - 29.5f) / 52f, up = -(sp + dn * cp), forward = cp - dn * sp, z = float.PositiveInfinity;
                    if (up > 1e-4f) z = Mathf.Min(z, 1.2f / up);                              // ceiling at 2.8 m
                    if (up < -1e-4f) z = Mathf.Min(z, 1.6f / -up);                            // floor
                    if (forward > 1e-4f) z = Mathf.Min(z, (7f - .25f) / forward);              // far wall
                    if (forward > 1e-4f)
                    {
                        float front = (2.6f - .25f) / forward, h = 1.6f + front * up;          // table front face
                        if (h >= 0 && h <= .75f) z = Mathf.Min(z, front);
                    }
                    if (up < -1e-4f)
                    {
                        float top = .85f / -up, y = .25f + top * forward;                     // table top
                        if (y >= 2.6f && y <= 3.2f) z = Mathf.Min(z, top);
                    }
                    frame.Depth[v * 80 + u] = (ushort)Mathf.Clamp(Mathf.RoundToInt(z * 1000), 0, 65535);
                }
                Motion(b, .1); b.OnAirTemperature(.1, 20); b.OnRgbd(frame); b.Step(.1, .1f);
                var g = Map(b);
                int front0 = g.Cell(new Vector2(0, 2.3f)), shadow = g.Cell(new Vector2(0, 3.6f)), table = g.Cell(new Vector2(0, 2.65f));
                return (g.OwnObserved(front0), g.OwnObserved(shadow), g.Occupancy(shadow) == 1, g.Occupancy(table) == 255);
            }
            var now = Look(.3f);
            Assert.That(now.front, Is.True, "floor in front of the table is seen");
            Assert.That(now.table, Is.True, "the table is mapped as an obstacle");
            Assert.That(now.shadowSearched, Is.False, "floor behind the table is not searched");
            Assert.That(now.shadowFree, Is.True, "but it is still mapped free for driving");
            var old = Look(0);
            Assert.That(old.shadowSearched, Is.True, "control: the old rule counted it searched");
        }

        [Test]
        public void FloorSeenFromBeyondTheSearchRangeIsMappedButNotSearched()
        {
            // Hidden-corner runs: the depth camera marked floor searched out to 8 m, so a glance through a doorway counted a
            // room as done. Floor 2 m ahead counts as searched; floor 6 m ahead is mapped free but stays a search target.
            // Control: with a 10 m search range it counts as searched again.
            (bool near, bool farSearched, bool farFree) Look(float range)
            {
                var c = Settings(); c.search_max_range_m = range;
                var b = new RobotBrain(Calibration(), c, "A", 0);
                Motion(b, 0); b.OnAirTemperature(0, 20); b.Step(0, 1f / 60);
                Motion(b, .1); b.OnAirTemperature(.1, 20); b.OnRgbd(RoomImage(.1, 9f)); b.Step(.1, .1f);
                var g = Map(b);
                int near = g.Cell(new Vector2(0, 2f)), far = g.Cell(new Vector2(0, 6f));
                return (g.OwnObserved(near), g.OwnObserved(far), g.Occupancy(far) == 1);
            }
            var limited = Look(4f);
            Assert.That(limited.near, Is.True, "2 m ahead is searched");
            Assert.That(limited.farSearched, Is.False, "6 m ahead is not searched");
            Assert.That(limited.farFree, Is.True, "but it is mapped free");
            Assert.That(Look(10f).farSearched, Is.True, "control: within a 10 m range it counts");
        }

        [Test]
        public void AHighCeilingCountsAsCeiling()
        {
            // Review: ceiling returns counted only between 2.5 and 3.5 m, so a higher lobby or hall never looked indoors.
            // The cap is now the camera's own reach. Physically the camera (1.6 m up, top edge 20 deg above horizontal,
            // depth to 8 m) sees ceilings up to about 4.3 m, so a 4.0 m ceiling (rejected by the old cap) now counts.
            // Control: with the ceiling at 2.0 m (below the lower bound) nothing counts.
            bool Seen(float ceilingM)
            {
                var c = Settings(); c.search_requires_ceiling = true; c.ceiling_min_hits = 1;
                var a = new RobotBrain(Calibration(), c, "A", 0);
                Motion(a, 0); a.OnAirTemperature(0, 20); a.OnLidar(RoomScan(0, Vector2.zero, 0, 9f, 720)); a.Step(0, 1f / 60);
                Motion(a, .1); a.OnAirTemperature(.1, 20); a.OnRgbd(RoomImage(.1, 9f, ceilingM)); a.OnLidar(RoomScan(.1, Vector2.zero, 0, 9f, 720)); a.Step(.1, .1f);
                var g = Map(a);
                for (float y = 1; y <= 8.5f; y += .25f) if (g.UnderCeiling(new Vector2(0, y))) return true;
                return false;
            }
            Assert.That(Seen(4f), Is.True, "a 4.0 m ceiling");
            Assert.That(Seen(2.8f), Is.True, "a house ceiling still counts");
            Assert.That(Seen(2.0f), Is.False, "control: below the lower bound");
        }

        [Test]
        public void ARoomMappedOnlyByLidarIsSearchableButTheYardIsNot()
        {
            // House reflex check: rooms never became indoors, because the camera sees ceiling points only 4.5 m and
            // more away, beyond the far wall of a small room. A room walled in on the robot's own LiDAR map must be
            // searchable; ground in front of one facade must not, ceiling or not.
            var c = Settings(); c.search_requires_ceiling = true;
            var room = new RobotBrain(Calibration(), c, "A", 0);
            Motion(room, 0); room.OnAirTemperature(0, 20); room.OnLidar(SegmentScan(0, Vector2.zero, 0, Box(-2, -2, 2.5f, 3))); room.Step(0, 1f / 60);
            var g = Map(room);
            Assert.That(g.Indoors(new Vector2(1, 1.5f)), Is.True, "inside a 4.5 x 5 m room seen only by the LiDAR");
            Assert.That(room.FrontierOptions(0, 10), Is.Not.Empty, "its unsearched floor is offered");
            var yard = new RobotBrain(Calibration(), c, "B", 0);
            var facade = new[] { (new Vector2(-8, 3), new Vector2(8, 3)) };
            Motion(yard, 0); yard.OnAirTemperature(0, 20); yard.OnLidar(SegmentScan(0, Vector2.zero, 0, facade)); yard.Step(0, 1f / 60);
            var h = Map(yard);
            Assert.That(h.Indoors(new Vector2(0, 2)), Is.False, "1 m in front of a facade");
            // Nothing the robot will search by itself: anything offered there is marked unproven, for the agent to check
            // with OpenCV first (and the brain refuses an explore there without that evidence).
            Assert.That(yard.FrontierOptions(0, 10).All(f => !f.provenIndoors), Is.True, "nothing proven in front of a facade");
        }

        [Test]
        public void FloorNotProvenIndoorsIsExploredOnlyWithAnOpenCvIndoorCheck()
        {
            // Hidden-corner run: rooms not yet entered had unmapped walls, were never proven indoors, never offered, and
            // the robot walked out with 30 % unsearched. Such frontiers are now offered marked unproven; the brain drives
            // there only for an agent order citing OpenCV evidence. Control: the same order without evidence is refused.
            var c = Settings(); c.search_requires_ceiling = true;
            string Order(string evidence)
            {
                var b = new RobotBrain(Calibration(), c, "A", 0);
                Motion(b, 0); b.OnAirTemperature(0, 20); b.OnLidar(RoomScan(0, Vector2.zero, 0, 9f, 720)); b.Step(0, 1f / 60); b.DrainEvents();
                var unproven = b.FrontierOptions(0, 10).Where(f => !f.provenIndoors).ToList();
                Assert.That(unproven, Is.Not.Empty, "unproven floor is offered");
                b.AgentExplore(unproven[0].position, 0, 30, "check", evidence);
                return b.DrainEvents().Last(e => e.kind == "agent_order").reason;
            }
            Assert.That(Order("ev_indoors"), Does.StartWith("explore_frontier on floor not yet proven indoors, citing OpenCV evidence ev_indoors"));
            Assert.That(Order(null), Does.StartWith("Refused go_to"), "control: no evidence, refused");
        }

        /// <summary>A map built only from LiDAR sweeps of wall segments, taken at each true position (yaw 0).</summary>
        static OccupancyGrid LidarMap(BrainConfig c, (Vector2, Vector2)[] walls, params Vector2[] from)
        {
            var g = new OccupancyGrid(c);
            foreach (var p in from) g.IntegrateLidar(SegmentScan(0, p, 0, walls), new BrainPose { position = p });
            return g;
        }

        [Test]
        public void TheYardBesideAFacadeIsNotWalledInWhateverStandsThere()
        {
            // Held-out run 2: robot B set the person it carried out down 2.5 m in front of the 40 m south wall. On its own
            // map the wall closed the north half of every ray fan from the strip between, the person closed a few more,
            // and rays into the empty yard ended in unmapped ground as unknown: 787 yard cells were walled in, and B
            // searched along the outside of the wall for two minutes. Wide LiDAR spans with no return are open ground and
            // the rays run on across it. Control: with the open-ground span wider than a full turn the same map is walled
            // in again, so the test exercises the rule.
            var facade = new[] { (new Vector2(-20, 3), new Vector2(20, 3)) };
            var scene = facade.Concat(Box(-.3f, .8f, .3f, 2.5f)).ToArray();          // a person 1.7 m long, 0.5 m from the wall
            var stops = new[] { new Vector2(1.2f, .5f), new Vector2(-1.2f, .5f), new Vector2(0, -1.5f) };
            (bool cell, int yard) Walled(float spanDeg)
            {
                var c = Settings(); c.grid_size_m = 40; c.lidar_open_span_deg = spanDeg;
                var g = LidarMap(c, scene, stops);
                var p = new Vector2(.55f, 1.8f);                                       // 0.25 m beside the person, 1.2 m from the wall
                Assert.That(g.OccupancyAt(p), Is.EqualTo(1), "the probe is mapped free floor");
                var layer = g.CopyIndoors(); int yard = 0;
                for (int i = 0; i < g.Count; i++) if (layer[i] == 3 && g.Centre(i).y < 2.9f) yard++;
                return (g.Indoors(p), yard);
            }
            var fixedRule = Walled(30);
            var control = Walled(400);
            Assert.That(control.cell, Is.True, "control: without open ground the strip beside the person is walled in");
            Assert.That(fixedRule.cell, Is.False, "the strip beside the person is yard");
            Assert.That(fixedRule.yard, Is.LessThan(control.yard / 10 + 1), $"walled-in yard cells: {fixedRule.yard} with open ground, {control.yard} without");
        }

        [Test]
        public void ARoomStaysWalledInWhenItsLidarLooksOutOfAnOpenDoor()
        {
            // Open ground is marked through an outside door the robot stands close to (0.8 m: the 1 m door spans 64
            // degrees). A cell 2 m inside sees the door in 3 of 36 directions and must stay walled in.
            var room = new[]
            {
                (new Vector2(-3.2f, -2), new Vector2(.8f, -2)), (new Vector2(-3.2f, 2), new Vector2(.8f, 2)), (new Vector2(-3.2f, -2), new Vector2(-3.2f, 2)),
                (new Vector2(.8f, -2), new Vector2(.8f, -.5f)), (new Vector2(.8f, .5f), new Vector2(.8f, 2)),
            };
            var c = Settings(); c.grid_size_m = 30;
            var g = LidarMap(c, room, Vector2.zero);
            var open = (bool[])typeof(OccupancyGrid).GetField("openGround", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(g);
            Assert.That(open[g.Cell(new Vector2(5, 0))], Is.True, "the yard seen through the door is open ground");
            Assert.That(open[g.Cell(new Vector2(-1.2f, 1.2f))], Is.False, "no open ground inside the walls away from the door");
            Assert.That(g.Indoors(new Vector2(-1.2f, 0)), Is.True, "the room 2 m inside the door is walled in");
        }

        [Test]
        public void SweepsSkipSearchedWallSectorsInACorridorButKeepAllThreeLooksInAnOpenRoom()
        {
            // In the 2 m corridor, the side sectors hit walls or searched floor before they can see the unsearched
            // strip ahead. Old code spends a full +60, -60, 0 sweep here, so its angular travel fails the first assertion.
            (float turned, int looks, double seconds) Sweep(bool corridor, bool sideOpening = false, bool allSearched = false)
            {
                var c = Settings(); c.grid_size_m = 14;
                var b = new RobotBrain(Calibration(), c, "A", 0); var g = Map(b);
                var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
                var lidar = (float[])typeof(OccupancyGrid).GetField("lidarOdds", flags).GetValue(g);
                for (int i = 0; i < g.Count; i++)
                {
                    var p = g.Centre(i); float halfWidth = corridor ? 1f : 5f;
                    if (Mathf.Abs(p.y) >= 6) continue;
                    if (Mathf.Abs(p.x) >= halfWidth && Mathf.Abs(p.x) < halfWidth + .2f)
                    {
                        if (!(sideOpening && p.x < 0 && Mathf.Abs(p.y) < .7f)) g.ObserveOccupied(i);
                    }
                    else if (Mathf.Abs(p.x) < halfWidth)
                    {
                        if (allSearched || (corridor && p.y < 2.7f)) g.ObserveFree(i);
                        else lidar[i] = -c.log_odds_cap;
                    }
                }
                typeof(OccupancyGrid).GetField("dirty", flags).SetValue(g, true);
                typeof(RobotBrain).GetMethod("OnFrontierReached", flags).Invoke(b, new object[] { Vector2.zero, 0.0 });
                BrainCommand cmd = default; float turned = 0; int looks = 0; double seconds = 0;
                for (int k = 0; k <= 16 * 15; k++)
                {
                    double t = k / 15.0; Motion(b, t, cmd.forwardMps, cmd.yawRateDeg * Mathf.Deg2Rad);
                    b.OnAirTemperature(t, 20); b.OnRgbd(Image(t, true, 0)); cmd = b.Step(t, 1f / 15);
                    turned += Mathf.Abs(cmd.yawRateDeg) / 15;
                    var events = b.DrainEvents(); looks += events.Count(e => e.kind == "sweep_look");
                    if (events.Any(e => e.kind == "area_searched")) { seconds = t; break; }
                }
                return (turned, looks, seconds);
            }
            var hall = Sweep(true);
            Assert.That(hall.turned, Is.LessThan(6), "searched walls do not justify either side turn");
            Assert.That(hall.looks, Is.EqualTo(1), "only the forward sector contains unsearched floor");
            Assert.That(hall.seconds, Is.LessThan(1.5), "one short dwell replaces the corridor sweep");
            TestContext.WriteLine($"corridor: {hall.looks} look, {hall.turned:0.0} degrees, {hall.seconds:0.00} s");
            var room = Sweep(false);
            TestContext.WriteLine($"open room: {room.looks} looks, {room.turned:0.0} degrees, {room.seconds:0.00} s");
            Assert.That(room.looks, Is.EqualTo(3), "all three camera sectors in an open unsearched room are kept");
            Assert.That(room.turned, Is.GreaterThan(200), "the room still receives the complete sweep");
            Assert.That(room.seconds, Is.LessThan(12), "all three holds finish before the sweep time limit");
            var door = Sweep(true, sideOpening: true);
            Assert.That(door.looks, Is.EqualTo(2), "an opening in the side wall keeps that side look as well as the forward look");
            Assert.That(Sweep(false, allSearched: true).looks, Is.Zero, "a searched room needs no look");
        }

        // Known floor and a wall with an opening, as if mapped by LiDAR. Only the approach side is
        // camera-searched. A ceiling observation retained behind the camera is the robot's earlier indoor evidence.
        static RobotBrain DoorwayBrain(float bodyY, float yawDeg = 0)
        {
            var c = Settings(); c.grid_size_m = 24; c.search_requires_ceiling = true;
            float angle = yawDeg * Mathf.Deg2Rad, co = Mathf.Cos(angle), si = Mathf.Sin(angle);
            var b = new RobotBrain(Calibration(), c, "A", 0, new EntryPose(-si * bodyY, co * bodyY, -yawDeg)); var g = Map(b);
            var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
            var lidar = (float[])typeof(OccupancyGrid).GetField("lidarOdds", flags).GetValue(g);
            var roof = (byte[])typeof(OccupancyGrid).GetField("roofHits", flags).GetValue(g);
            for (int i = 0; i < g.Count; i++)
            {
                var world = g.Centre(i); var p = new Vector2(co * world.x + si * world.y, -si * world.x + co * world.y);
                if (Mathf.Abs(p.x) >= 8 || p.y <= -3 || p.y >= 9) continue;
                if (Mathf.Abs(p.y) < .15f && Mathf.Abs(p.x) >= .6f) g.ObserveOccupied(i);
                else if (p.y < -.3f) g.ObserveFree(i);
                else lidar[i] = -c.log_odds_cap;
            }
            roof[g.Cell(new Vector2(si * 1.1f, -co * 1.1f))] = 255;
            typeof(OccupancyGrid).GetField("roofDirty", flags).SetValue(g, true);
            typeof(OccupancyGrid).GetField("dirty", flags).SetValue(g, true);
            return b;
        }

        [Test]
        public void ACeilingSeenFromAnUnprovenDoorwayProvesTheFloorUnderItsSightLine()
        {
            // The old cameraIndoors gate returns here, even though the camera is at an opening with proven floor behind
            // it and its own camera sees a real ceiling beyond it. It leaves the probe at y=2 outside the search area.
            foreach (float yaw in new[] { 0f, 37f, 90f, 193f })
            {
                var b = DoorwayBrain(-.25f, yaw); var g = Map(b); var sensor = Calibration(); var pose = b.Snapshot().pose;
                Assert.That(g.Indoors(pose.Transform(0, .25f)), Is.False, "the camera at the opening has no indoor proof yet");
                Assert.That(g.Indoors(pose.Transform(0, -.85f)), Is.True, "earlier ceiling evidence behind it");
                g.IntegrateDepth(RoomImage(.1, 7, 2.8f), sensor, pose);
                g.IntegrateDepth(RoomImage(.2, 7, 2.8f), sensor, pose);
                Assert.That(g.UnderCeiling(pose.Transform(0, 2.25f)), Is.True, $"fresh ceiling sight lines through an opening rotated {yaw} degrees");
                Assert.That(g.UnderCeiling(pose.Transform(0, -2.25f)), Is.False, "no flood into floor behind the camera");
                var outside = DoorwayBrain(-.25f, yaw); var h = Map(outside);
                var backwards = outside.Snapshot().pose;
                backwards.position += backwards.Forward * .5f; backwards.yawRad += Mathf.PI;
                h.IntegrateDepth(RoomImage(.1, 7, 2.8f), sensor, backwards);
                h.IntegrateDepth(RoomImage(.2, 7, 2.8f), sensor, backwards);
                Assert.That(h.UnderCeiling(pose.Transform(0, .55f)), Is.False,
                    "looking back through an exterior door at an indoor ceiling does not prove the ground outside");
            }
        }

        [Test]
        public void AnExteriorDoorGetsOnlyABoundedEvidenceMoveAndOpenGroundStaysUnproven()
        {
            // Old code never starts an evidence move. The control then checks that moving to the opening gives no
            // indoor label without overhead evidence, even when the RGB-D camera sees a tall facade ahead.
            var b = DoorwayBrain(-1.1f); var g = Map(b); BrainCommand cmd = default;
            var logged = new List<BrainEvent>(); float farthest = 0;
            for (int k = 0; k <= 16 * 15; k++)
            {
                double t = k / 15.0;
                Motion(b, t, cmd.forwardMps, cmd.yawRateDeg * Mathf.Deg2Rad);
                b.OnAirTemperature(t, 20); b.OnRgbd(Image(t, true, 0)); cmd = b.Step(t, 1f / 15);
                farthest = Mathf.Max(farthest, Vector2.Distance(new Vector2(0, -1.1f), b.Snapshot().pose.position));
                logged.AddRange(b.DrainEvents());
                if (logged.Any(e => e.kind == "doorway_probe_done")) break;
            }
            Assert.That(logged.Any(e => e.kind == "doorway_probe_started"), Is.True, "a mapped opening gets a reflex evidence move");
            Assert.That(logged.Any(e => e.kind == "doorway_probe_done"), Is.True, "the probe ends within the sweep time limit");
            Assert.That(farthest, Is.LessThanOrEqualTo(1.2f), "motion is bounded near the doorway");
            Assert.That(b.Snapshot().pose.position.y, Is.LessThanOrEqualTo(.15f), "the probe ends on the inside of the wall line");
            var pose = b.Snapshot().pose;
            g.IntegrateDepth(RoomImage(17, 7, 60), Calibration(), pose);
            Assert.That(g.Indoors(new Vector2(0, 2)), Is.False, "open ground beyond an exterior door stays unproven");
        }

        [Test]
        public void AReflexDoorwayLookGathersCeilingEvidenceBeforeItExploresBeyondTheWall()
        {
            // Simulate depth dropout until the camera reaches the mapped wall opening. The robot must move there
            // itself, face through it, and prove indoor floor from its own fresh depth frames. Old code never probes.
            var b = DoorwayBrain(-1.1f); var g = Map(b); BrainCommand cmd = default;
            var logged = new List<BrainEvent>(); bool suppliedCeiling = false;
            for (int k = 0; k <= 16 * 15; k++)
            {
                double t = k / 15.0;
                Motion(b, t, cmd.forwardMps, cmd.yawRateDeg * Mathf.Deg2Rad);
                b.OnAirTemperature(t, 20);
                var pose = b.Snapshot().pose;
                bool atOpening = Mathf.Abs(pose.Transform(0, .25f).y) < .15f && Mathf.Abs(pose.yawRad) < .15f;
                b.OnRgbd(atOpening ? RoomImage(t, 7, 2.8f) : Image(t, true, 0)); suppliedCeiling |= atOpening;
                cmd = b.Step(t, 1f / 15); logged.AddRange(b.DrainEvents());
                if (logged.Any(e => e.kind == "doorway_probe_done")) break;
            }
            Assert.That(logged.Any(e => e.kind == "doorway_probe_started"), Is.True, "the reflex starts the evidence approach");
            Assert.That(suppliedCeiling, Is.True, "it brings the camera to the opening");
            Assert.That(logged.Any(e => e.kind == "doorway_probe_done" && e.reason.StartsWith("One doorway look")), Is.True, "the look finishes before its timeout");
            Assert.That(g.UnderCeiling(new Vector2(0, 2)), Is.True, "the new sensor view proves floor beyond the wall");
            Assert.That(b.Snapshot().pose.position.y, Is.LessThan(0), "the evidence move keeps the body on the indoor side");
        }

        [Test]
        public void HeatStillCancelsADoorwayEvidenceMove()
        {
            var b = DoorwayBrain(-1.1f);
            Motion(b, 0); b.OnAirTemperature(0, 20); b.OnRgbd(Image(0, true, 0)); b.Step(0, 1f / 15);
            Assert.That(b.DrainEvents().Any(e => e.kind == "doorway_probe_started"), Is.True, "control: a probe was active");
            Motion(b, .1); b.OnAirTemperature(.1, 100); b.OnRgbd(Image(.1, true, 0)); b.Step(.1, .1f);
            var events = b.DrainEvents();
            Assert.That(events.Any(e => e.kind == "doorway_probe_done"), Is.True, "heat cancels the probe");
            Assert.That(b.Snapshot().state == "retreat" || b.Snapshot().state == "return", Is.True, "the heat reflex owns motion");
        }

        [Test]
        public void TheTopOfAWallFaceDrawsNoCeilingSightLines()
        {
            // Held-out run 2: from the yard, 98 % of robot B's ceiling-height depth points (2.5 to 2.9 m) lay within 0.4 m of
            // the south wall, on the top of its outside face, and sight lines to them made the floor along the facade
            // indoors. Sight lines now need ceiling evidence 0.4 m clear of walls, as ceiling seeds do. The camera is made to
            // count as indoors (ceiling evidence put under it), so only that margin decides. Control: a real ceiling at
            // the same wall distance still draws them.
            int Marked(float ceilingM)
            {
                var c = Settings(); c.search_requires_ceiling = true; c.ceiling_min_hits = 1;
                var a = new RobotBrain(Calibration(), c, "A", 0);
                Motion(a, 0); a.OnAirTemperature(0, 20); a.OnLidar(RoomScan(0, Vector2.zero, 0, 9f, 720)); a.Step(0, 1f / 60);
                var g = Map(a); var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
                var roof = (byte[])typeof(OccupancyGrid).GetField("roofHits", flags).GetValue(g);
                for (float y = -.5f; y <= .5f; y += .1f) roof[g.Cell(new Vector2(0, y))] = 255;
                typeof(OccupancyGrid).GetField("roofDirty", flags).SetValue(g, true);
                Assert.That(g.UnderCeiling(new Vector2(0, .25f)), Is.True, "the camera stands indoors");
                Motion(a, .1); a.OnAirTemperature(.1, 20); a.OnRgbd(RoomImage(.1, 5f, ceilingM)); a.OnLidar(RoomScan(.1, Vector2.zero, 0, 9f, 720)); a.Step(.1, .1f);
                var sight = (bool[])typeof(OccupancyGrid).GetField("sightIndoors", flags).GetValue(g);
                int marked = 0;
                for (float x = -1; x <= 1; x += .1f) for (float y = 1.5f; y <= 4.5f; y += .1f) if (sight[g.Cell(new Vector2(x, y))]) marked++;
                return marked;
            }
            Assert.That(Marked(60f), Is.Zero, "a wall face 5 m ahead with no ceiling over the floor: no sight lines");
            Assert.That(Marked(2.8f), Is.GreaterThan(0), "control: a ceiling clear of the wall still draws them");
        }

        [Test]
        public void ARoomWithAnOpenOutsideDoorIsWalledInButNotTheGroundOutside()
        {
            // Mapped open ground outside a 1 m door: the room's middle is walled in, floor 0.5 m outside the door is not.
            var c = Settings(); c.grid_size_m = 30; var g = OpenGrid(c);
            for (float x = -2; x <= 2; x += .05f) { g.ObserveOccupied(g.Cell(new Vector2(x, -2))); g.ObserveOccupied(g.Cell(new Vector2(x, 2))); }
            for (float y = -2; y <= 2; y += .05f) { g.ObserveOccupied(g.Cell(new Vector2(-2, y))); if (Mathf.Abs(y) > .5f) g.ObserveOccupied(g.Cell(new Vector2(2, y))); }
            Assert.That(g.Indoors(new Vector2(0, 0)), Is.True, "middle of the room, 2 m from the door");
            Assert.That(g.Indoors(new Vector2(2.5f, 0)), Is.False, "0.5 m outside the door");
            Assert.That(g.Indoors(new Vector2(8, 8)), Is.False, "open ground");
        }

        [Test]
        public void FloorOutsideAnOpenDoorIsNotIndoors()
        {
            // Run 8: through an open door the yard just outside sat within 1.5 m of the room's ceiling evidence.
            var c = Settings(); c.grid_size_m = 12; c.ceiling_radius_m = .6f; c.ceiling_wall_margin_m = .4f; var g = OpenGrid(c);
            for (float y = -3; y <= 3; y += .05f) if (Mathf.Abs(y) > .5f) for (float x = -.1f; x <= .14f; x += .05f) g.ObserveOccupied(g.Cell(new Vector2(x, y)));   // 0.25 m wall, 1 m door
            var roof = (byte[])typeof(OccupancyGrid).GetField("roofHits", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(g);
            for (float y = -2.5f; y <= 2.5f; y += .1f) for (float x = -3; x <= -.15f; x += .1f) roof[g.Cell(new Vector2(x, y))] = 255;   // the room's ceiling ends at the inner wall face
            Assert.That(g.Indoors(new Vector2(-.2f, 1.5f)), Is.True, "control: floor against the inner wall");
            Assert.That(g.Indoors(new Vector2(.6f, 0)), Is.False, "yard 0.45 m outside the door");
        }

        [Test]
        public void SearchNeedsACeilingSeenByTheRobotsOwnCamera()
        {
            // No outline: with search_requires_ceiling a frontier must lie near floor where the camera saw a ceiling.
            // The test camera is 80 x 60 sampled every 4th pixel, about one ceiling return per cell (640 x 480 in runs).
            var c = Settings(); c.search_requires_ceiling = true; c.ceiling_min_hits = 1;
            var a = new RobotBrain(Calibration(), c, "A", 0);
            Motion(a, 0); a.OnAirTemperature(0, 20); a.OnLidar(RoomScan(0, Vector2.zero, 0, 9f, 720)); a.Step(0, 1f / 60);
            a.DrainEvents();
            Assert.That(a.FrontierOptions(0, 10).All(f => !f.provenIndoors), Is.True, "an 18 m hall mapped only by LiDAR: not walled in within 10 m and no ceiling seen, nothing proven to search yet");
            // The log says why the list is empty: frontier cells dropped as not known to be indoors, none for other reasons.
            var why = a.DrainEvents().Single(e => e.kind == "frontier_options").numbers;
            Assert.That(why["frontiers"], Is.Zero);
            Assert.That(why["cells_not_indoors"], Is.GreaterThan(0));
            Assert.That(why["cells_claimed"] + why["cells_blacklisted"] + why["cells_visited"], Is.Zero);
            Motion(a, .1); a.OnAirTemperature(.1, 20); a.OnRgbd(RoomImage(.1, 7f)); a.OnLidar(RoomScan(.1, Vector2.zero, 0, 9f, 720)); a.Step(.1, .1f);
            var g = Map(a);
            Assert.That(g.UnderCeiling(new Vector2(0, 4.5f)), Is.True, "ceiling seen overhead");
            Assert.That(g.UnderCeiling(new Vector2(0, -4f)), Is.False, "behind the robot no ceiling was seen");
            // Beyond the far wall the camera mapped at 7 m: within 1.5 m of ceiling evidence in a straight line, but not
            // reachable without crossing the wall (run 5, the yard outside a wall).
            Assert.That(g.UnderCeiling(new Vector2(0, 7.6f)), Is.False, "ceiling distance does not pass a wall");
            var options = a.FrontierOptions(.1, 10);
            Assert.That(options, Is.Not.Empty);
            Assert.That(options.Where(f => f.provenIndoors).All(f => g.Indoors(f.position)), Is.True);
            Assert.That(options.Any(f => f.provenIndoors), Is.True, "floor under the seen ceiling is proven");
            // The agent has no outline either: the brain refuses a go_to where no ceiling was seen, and accepts one under it.
            a.DrainEvents();
            a.AgentGoTo(new Vector2(0, -4f), .1, 30, "behind");
            a.AgentGoTo(new Vector2(0, 4.5f), .1, 30, "ahead");
            var orders = a.DrainEvents().Where(e => e.kind == "agent_order").Select(e => e.reason).ToList();
            Assert.That(orders, Has.Count.EqualTo(2));
            Assert.That(orders[0], Does.StartWith("Refused go_to"));
            Assert.That(orders[1], Does.StartWith("go_to: ahead"));
        }

        [Test]
        public void ADoorwayFrontierReportsTheUnsearchedFloorMappedBeyondIt()
        {
            // Held-out seed 914 run 1: the doorway onto a LiDAR-mapped hallway showed unsearched_m2 1.5, counted only beside
            // the frontier, and the agent chose a kitchen corner at 8.5 m2. A searched room with a doorway north onto an
            // 8 x 1.8 m corridor mapped by LiDAR only; the doorway frontier reports the corridor. A yard behind a south door,
            // with no ceiling seen over it, adds nothing.
            var c = Settings(); c.grid_size_m = 16; c.search_requires_ceiling = true; c.ceiling_min_hits = 1;
            var b = new RobotBrain(Calibration(), c, "A", 0); var g = Map(b);
            var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
            var lidar = (float[])typeof(OccupancyGrid).GetField("lidarOdds", flags).GetValue(g);
            var roof = (byte[])typeof(OccupancyGrid).GetField("roofHits", flags).GetValue(g);
            for (int i = 0; i < g.Count; i++)
            {
                var p = g.Centre(i); float ax = Mathf.Abs(p.x), ay = Mathf.Abs(p.y);
                bool room = ax < 2f && ay < 2f, wall = !room && ax < 2.2f && ay < 2.2f, door = wall && ax < .5f;
                bool corridor = ax < 4f && p.y >= 2.2f && p.y < 4f, yard = ax < 5f && p.y <= -2.2f && p.y > -7f;
                if (wall && !door) g.ObserveOccupied(i);
                else if (room) g.ObserveFree(i);
                else if (door || corridor || yard) lidar[i] = -c.log_odds_cap;
                if ((room && ax <= 1.5f && ay <= 1.5f) || (corridor && p.y > 2.6f && p.y < 3.6f)) roof[i] = 255;   // ceiling seen over the room and the corridor
            }
            typeof(OccupancyGrid).GetField("roofDirty", flags).SetValue(g, true);
            typeof(OccupancyGrid).GetField("dirty", flags).SetValue(g, true);
            float corridorM2 = g.UnsearchedFloorBeyond(new Vector2(0, 1.9f), 12f), yardM2 = g.UnsearchedFloorBeyond(new Vector2(0, -1.9f), 12f);
            Assert.That(corridorM2, Is.GreaterThan(.8f * 8f * 1.8f), "most of the 14.4 m2 corridor");
            Assert.That(corridorM2, Is.LessThan(8f * 1.8f + 1f), "and not much more");
            Assert.That(yardM2, Is.LessThan(.5f), "the yard has no ceiling seen over it");
            Assert.That(g.UnsearchedFloorBeyond(new Vector2(0, 1.9f), 1f), Is.LessThan(corridorM2 / 3), "bounded by the radius");
            // Run 3: frontiers stand on searched floor with the unsearched floor starting up to frontier_peek_m away.
            Assert.That(g.UnsearchedFloorBeyond(new Vector2(0, 1f), 12f), Is.GreaterThan(.8f * 8f * 1.8f), "from 1 m back from the doorway too");
            // Codex review: the seed disc must not reach through a wall. 0.9 m along the room's east wall, inside, the corridor
            // and the yard lie behind walls, not through the door.
            Assert.That(g.UnsearchedFloorBeyond(new Vector2(1.6f, 1.6f), 12f), Is.LessThan(1f), "nothing counted through the wall");
        }

        [Test]
        public void ARejectedWarmShapeIsNotRaisedAgainUntilSeenFromANewViewpoint()
        {
            // Held-out seed 914 run 2: robot B rejected the 38 C fridge and the same track was raised again in the same
            // second (the next thermal frames re-armed it), three times. A rejected track stays quiet until the robot sees
            // the spot from reject_new_view_deg round or from half the distance. Control: 0 deg re-raises at once.
            (int sameView, int newView) Run(float deg)
            {
                var c = Settings(); c.reject_new_view_deg = deg;
                var b = new RobotBrain(Calibration(), c, "A", 0);
                var observe = typeof(RobotBrain).GetMethod("ObservePeople", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                var spot = new Vector2(2, 2);
                double t = 0;
                void Tick(float v, bool see)
                {
                    t += 1 / 15.0; Motion(b, t, v); b.OnAirTemperature(t, 20); b.Step(t, 1f / 15);
                    if (see) observe.Invoke(b, new object[] { t, new List<PersonDetection> { new PersonDetection(spot, 1.9f) } });
                }
                for (int k = 0; k < 5; k++) Tick(0, true);
                Assert.That(b.DrainEvents().Count(e => e.kind == "survivor_candidate"), Is.EqualTo(1), "raised once");
                b.AgentRejectCandidate(spot, t, "ev"); b.DrainEvents();
                for (int k = 0; k < 15; k++) Tick(0, true);
                int same = b.DrainEvents().Count(e => e.kind == "survivor_candidate");
                for (int k = 0; k < 60; k++) Tick(1f, false);   // 4 m on: the spot is now 90 degrees round from the rejecting view
                for (int k = 0; k < 5; k++) Tick(0, true);
                var again = b.DrainEvents().Where(e => e.kind == "survivor_candidate").ToList();
                Assert.That(again.All(e => e.numbers["previously_rejected"] == 1), Is.True, "the agent is told it rejected this spot before");
                return (same, again.Count);
            }
            Assert.That(Run(45), Is.EqualTo((0, 1)), "quiet from the same view, raised again from a new one");
            Assert.That(Run(0).sameView, Is.GreaterThanOrEqualTo(1), "control: the old behaviour re-raises at once");
        }

        [Test]
        public void AWithdrawnConfirmationLeavesTheListAndAPartnerCannotPutItBack()
        {
            // Held-out seed 914 run 3: a false confirmation kept robot B in the living room for over a minute. Withdrawn, it
            // leaves the to-rescue list, a partner's radioed copy does not restore it, and a fresh confirmation here does.
            var b = new RobotBrain(Calibration(), Settings(), "A", 0);
            Motion(b, 0); b.OnAirTemperature(0, 20); b.Step(0, 1f / 60);
            var spot = new Vector2(0, 3);
            b.AgentConfirmSurvivor(spot, .1, "ev1");
            Assert.That(b.Snapshot().confirmedSurvivors, Has.Length.EqualTo(1));
            b.DrainEvents();
            b.AgentWithdrawSurvivor(spot, .2, "ev2");
            Assert.That(b.Snapshot().confirmedSurvivors, Is.Empty, "off the list");
            Assert.That(b.DrainEvents().Single(e => e.kind == "survivor_withdrawn").numbers["removed"], Is.EqualTo(1));
            var receive = typeof(RobotBrain).GetMethod("ReceiveSurvivor", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            receive.Invoke(b, new object[] { new BrainSurvivor { position = spot + new Vector2(.3f, 0), sourceRobot = "B" }, "B", .3 });
            Assert.That(b.Snapshot().confirmedSurvivors, Is.Empty, "a partner's copy does not bring it back");
            receive.Invoke(b, new object[] { new BrainSurvivor { position = new Vector2(4, 4), sourceRobot = "B" }, "B", .4 });
            Assert.That(b.Snapshot().confirmedSurvivors, Has.Length.EqualTo(1), "a partner's survivor elsewhere still counts");
            b.AgentConfirmSurvivor(spot, .5, "ev3");
            Assert.That(b.Snapshot().confirmedSurvivors, Has.Length.EqualTo(2), "a fresh confirmation here counts again");
        }

        [Test]
        public void TheAgentIsOfferedTheDoorwayOntoTheLargestUnsearchedAreaNotOnlyNearbyCorners()
        {
            // Held-out seed 914 runs 3 and 4: 102 of 162 options were in the robot's current room; in the foyer all six were
            // corners with under 2.5 m2 beyond them while the doors to three unsearched rooms were not offered. A searched
            // 12 x 12 m room with twelve small unsearched pockets 2 to 3.3 m from the robot and a doorway 6 m away onto a
            // 12 x 1.8 m corridor the LiDAR mapped. Control: with no wide slots the eight nearest options are all pockets.
            (bool doorway, int pockets) Offer(int wide)
            {
                // The shipped gain weight (config/brain.json 0.01, a 1.5 m bonus at the cap); the C# default 0.04 gives a 6 m
                // bonus that ranks the doorway first on its own and hides what the robots actually run with.
                var c = Settings(); c.grid_size_m = 26; c.frontier_wide_options = wide; c.frontier_gain_weight = .01f;
                var b = new RobotBrain(Calibration(), c, "A", 0); var g = Map(b);
                var lidar = (float[])typeof(OccupancyGrid).GetField("lidarOdds", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(g);
                var pockets = new List<Vector2>();
                for (int k = 0; k < 12; k++) { float a = (k * 30 + 15) * Mathf.Deg2Rad; pockets.Add(new Vector2(Mathf.Sin(a), Mathf.Cos(a)) * (2f + .12f * k)); }
                for (int i = 0; i < g.Count; i++)
                {
                    var p = g.Centre(i); float ax = Mathf.Abs(p.x), ay = Mathf.Abs(p.y);
                    bool room = ax < 6f && ay < 6f, wall = !room && ax < 6.2f && ay < 6.2f, door = wall && p.y > 0 && ax < .5f;
                    bool corridor = ax < 6f && p.y >= 6.2f && p.y < 8f;
                    bool pocket = pockets.Exists(q => Mathf.Abs(p.x - q.x) < .3f && Mathf.Abs(p.y - q.y) < .3f);
                    if (wall && !door) g.ObserveOccupied(i);
                    else if (room && !pocket) g.ObserveFree(i);
                    else if (room || door || corridor) lidar[i] = -c.log_odds_cap;   // mapped free by LiDAR, not searched
                }
                typeof(OccupancyGrid).GetField("dirty", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).SetValue(g, true);
                Motion(b, 0);
                var options = b.FrontierOptions(0, 8);
                TestContext.WriteLine($"wide {wide}: " + string.Join("; ", options.Select(f => $"({f.position.x:0.0},{f.position.y:0.0}) cost {f.pathM:0.00} beyond {f.mappedUnsearchedM2:0.0} proven {f.provenIndoors}")));
                return (options.Any(f => Mathf.Abs(f.position.x) < 1f && f.position.y > 4.5f && f.mappedUnsearchedM2 > 10f),
                        options.Count(f => pockets.Exists(q => Vector2.Distance(q, f.position) < .9f)));
            }
            var wide = Offer(3);
            Assert.That(wide.doorway, Is.True, "the doorway onto the corridor is offered");
            Assert.That(wide.pockets, Is.GreaterThanOrEqualTo(5), "nearby pockets are still offered too");
            var nearest = Offer(0);
            Assert.That(nearest.doorway, Is.False, "control: nearest first offers only pockets");
        }

        [Test]
        public void WideSlotsIncludeAnUnprovenDoorwayEvenWhenProvenPocketsFillTheList()
        {
            // Regression: the old proven-only wide pass omits this doorway with wide slots enabled.
            // Held-out seed 914 runs 3 and 4: 102 of 162 options were in the robot's current room; in the foyer all six were
            // corners with under 2.5 m2 beyond them while the doors to three unsearched rooms were not offered. A searched
            // 12 x 12 m room with twelve small unsearched pockets 2 to 3.3 m from the robot and a doorway 6 m away onto a
            // 12 x 1.8 m corridor the LiDAR mapped. Control: with no wide slots the eight nearest options are all pockets.
            (bool doorway, int pockets, double unprovenWide) Offer(int wide)
            {
                // The shipped gain weight (config/brain.json 0.01, a 1.5 m bonus at the cap); the C# default 0.04 gives a 6 m
                // bonus that ranks the doorway first on its own and hides what the robots actually run with.
                var c = Settings(); c.grid_size_m = 26; c.frontier_wide_options = wide; c.frontier_gain_weight = .01f;
                c.search_requires_ceiling = true; c.enclosure_min_closed_share = 1f;
                var b = new RobotBrain(Calibration(), c, "A", 0); var g = Map(b);
                var lidar = (float[])typeof(OccupancyGrid).GetField("lidarOdds", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(g);
                var indoor = (bool[])typeof(OccupancyGrid).GetField("sightIndoors", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(g);
                var pockets = new List<Vector2>();
                for (int k = 0; k < 12; k++) { float a = (k * 30 + 15) * Mathf.Deg2Rad; pockets.Add(new Vector2(Mathf.Sin(a), Mathf.Cos(a)) * (2f + .12f * k)); }
                for (int i = 0; i < g.Count; i++)
                {
                    var p = g.Centre(i); float ax = Mathf.Abs(p.x), ay = Mathf.Abs(p.y);
                    bool room = ax < 6f && ay < 6f, wall = !room && ax < 6.2f && ay < 6.2f, door = wall && p.y > 0 && ax < .5f;
                    bool corridor = ax < 6f && p.y >= 6.2f && p.y < 8f;
                    bool pocket = pockets.Exists(q => Mathf.Abs(p.x - q.x) < .3f && Mathf.Abs(p.y - q.y) < .3f);
                    indoor[i] = room && !(ax < 1f && p.y > 4.5f);
                    if (wall && !door) g.ObserveOccupied(i);
                    else if (room && !pocket) g.ObserveFree(i);
                    else if (room || door || corridor) lidar[i] = -c.log_odds_cap;   // mapped free by LiDAR, not searched
                }
                typeof(OccupancyGrid).GetField("dirty", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).SetValue(g, true);
                Motion(b, 0);
                var options = b.FrontierOptions(0, 8);
                var counts = b.DrainEvents().Single(e => e.kind == "frontier_options").numbers;
                counts.TryGetValue("wide_unproven_options", out double unprovenWide);
                TestContext.WriteLine($"wide {wide}: " + string.Join("; ", options.Select(f => $"({f.position.x:0.0},{f.position.y:0.0}) cost {f.pathM:0.00} beyond {f.mappedUnsearchedM2:0.0} proven {f.provenIndoors}")));
                return (options.Any(f => Mathf.Abs(f.position.x) < 1f && f.position.y > 4.5f && !f.provenIndoors),
                        options.Count(f => pockets.Exists(q => Vector2.Distance(q, f.position) < .9f)), unprovenWide);
            }
            var wide = Offer(3);
            Assert.That(wide.doorway, Is.True, "the doorway onto the corridor is offered");
            Assert.That(wide.unprovenWide, Is.EqualTo(1), "unproven wide slots are logged");
            Assert.That(wide.pockets, Is.GreaterThanOrEqualTo(5), "nearby pockets are still offered too");
            var nearest = Offer(0);
            Assert.That(nearest.doorway, Is.False, "control: nearest first offers only pockets");
        }

        [Test]
        public void ARobotThatLeftForLackOfFrontiersGoesBackInWhenATemporaryExclusionExpires()
        {
            // Codex review of held-out seed 914: "no reachable frontier" ended the search for good, although the exclusions
            // that emptied the list (a fire-cut blacklist, visited circles, a partner's claim) expire. The one unsearched pocket
            // is blacklisted for 20 s; the robot scans, leaves, and goes back in once the blacklist lapses.
            var c = Settings(); c.grid_size_m = 12;
            var b = new RobotBrain(Calibration(), c, "A", 0); var g = Map(b);
            for (int i = 0; i < g.Count; i++) { var p = g.Centre(i); if (!(Mathf.Abs(p.x) < .6f && p.y > 3f && p.y < 4f)) g.ObserveFree(i); }
            var flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;
            var blacklist = typeof(RobotBrain).GetMethod("BlacklistGoal", flags);
            blacklist.Invoke(b, new object[] { new Vector2(0, 3.5f), 0.0, 0, 20.0, "test: fire cut", 1.5f });
            var kinds = new List<(string kind, double t)>();
            for (int k = 0; k <= 15 * 45; k++)
            {
                double t = k / 15.0;
                Motion(b, t); b.OnRgbd(Image(t, true, 0)); b.OnAirTemperature(t, 20); b.Step(t, 1f / 15);
                foreach (var e in b.DrainEvents()) kinds.Add((e.kind, e.t));
            }
            double done = kinds.FirstOrDefault(e => e.kind == "mission_complete").t;
            Assert.That(kinds.Any(e => e.kind == "no_frontier"), Is.True, "the pocket was excluded at first");
            Assert.That(kinds.Any(e => e.kind == "mission_complete"), Is.True, "so the robot left");
            Assert.That(kinds.Any(e => e.kind == "reenter" && e.t > done && e.t >= 20), Is.True, "and went back in once the blacklist lapsed");
        }

        [Test]
        public void FloorBetweenAnIndoorCameraAndTheCeilingItSeesIsIndoors()
        {
            // Agent run 14: the camera sees ceiling only about 4.7 m away, so the 0.6 m ceiling radius left most of a small
            // room outside the search area. Floor on a known-free line from an indoor camera to ceiling it saw is indoors.
            // An 18 m hall mapped by LiDAR is not walled in within 10 m, so only the ceiling rules can make (0, 2.5) indoors.
            string Run(bool sightLines, bool cameraIndoors)
            {
                var c = Settings(); c.search_requires_ceiling = true; c.ceiling_min_hits = 1; c.ceiling_sight_lines = sightLines;
                var a = new RobotBrain(Calibration(), c, "A", 0);
                Motion(a, 0); a.OnAirTemperature(0, 20); a.OnLidar(RoomScan(0, Vector2.zero, 0, 9f, 720)); a.Step(0, 1f / 60);
                var g = Map(a);
                if (cameraIndoors)
                {
                    // Ceiling seen earlier over where the robot now stands.
                    var roof = (byte[])typeof(OccupancyGrid).GetField("roofHits", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(g);
                    for (float y = -.5f; y <= .5f; y += .1f) roof[g.Cell(new Vector2(0, y))] = 255;
                    typeof(OccupancyGrid).GetField("roofDirty", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).SetValue(g, true);
                }
                bool standsIndoors = g.UnderCeiling(new Vector2(0, .25f));
                Motion(a, .1); a.OnAirTemperature(.1, 20); a.OnRgbd(RoomImage(.1, 7f)); a.OnLidar(RoomScan(.1, Vector2.zero, 0, 9f, 720)); a.Step(.1, .1f);
                // Sight lines must stop at the far wall the camera mapped at 7 m. (Floor past it can still be walled in
                // between that wall and the LiDAR's at 9 m, so Indoors there is not the probe; the sight-line layer is.)
                var sight = (bool[])typeof(OccupancyGrid).GetField("sightIndoors", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance).GetValue(g);
                bool pastWall = false;
                for (float x = -4; x <= 4; x += .1f) for (float y = 7.15f; y <= 9; y += .1f) pastWall |= sight[g.Cell(new Vector2(x, y))];
                return $"{standsIndoors} {g.Indoors(new Vector2(0, 2.5f))} {pastWall} {g.Indoors(new Vector2(0, -3f))}";
            }
            // Camera indoors: the floor 2.5 m ahead, under the line to the ceiling seen 4.7 m on, is indoors. No sight line
            // passes the far wall, and the floor behind the robot (-3 m) is not indoors: the lines never point back.
            Assert.That(Run(true, true), Is.EqualTo("True True False False"));
            // Controls: without sight lines the same floor stays outside the search area, and a camera that is not known
            // to stand indoors (in the yard, looking through a door) marks nothing.
            Assert.That(Run(false, true), Is.EqualTo("True False False False"));
            Assert.That(Run(true, false), Is.EqualTo("False False False False"));
        }

        [Test]
        public void LowerIdKeepsAContestedGoalAndClaimsExpire()
        {
            var a = new RobotBrain(Calibration(), Settings(), "robot_2", 0);
            var b = new RobotBrain(Calibration(), Settings(), "robot_10", 0);
            foreach (var brain in new[] { a, b })
            {
                Room(brain, -3, 3, -3, 3); Motion(brain, 0); brain.OnRgbd(Image(0, true, 0));
                brain.OnAirTemperature(0, 20); brain.Step(0, .02f);
                Assert.That(brain.Snapshot().goal.HasValue, Is.True);
            }
            var am = a.BuildMessage(0); var bm = b.BuildMessage(0);
            a.OnPeerMessage(Wire(bm), .1); b.OnPeerMessage(Wire(am), .1);
            Assert.That(a.Snapshot().goal.HasValue, Is.True);
            Assert.That(b.Snapshot().goal.HasValue, Is.False);
            Assert.That(b.DrainEvents().Count(e => e.kind == "goal_yielded"), Is.EqualTo(1));
            b.Step(16, 16);
            Assert.That(b.Snapshot().peers.Single().claim, Is.Null);
        }

        [Test]
        public void APersonCarriedOutLeavesTheToRescueListHereAndOnThePartner()
        {
            // Hidden-corner run: robot B went back twice for s1 after robot A had carried them out, because only the
            // confirmation travelled by radio. The carrier drops the person and broadcasts it (message version 3); the
            // partner drops them too and ignores the stale confirmation. Control: without the carried-out entry the
            // partner keeps the person.
            var a = new RobotBrain(Calibration(), Settings(), "A", 0);
            a.AgentConfirmSurvivor(new Vector2(0, 1.2f), 0, "ev1");
            Assert.That(a.Snapshot().confirmedSurvivors, Has.Length.EqualTo(1));
            a.BeginCarry(1, .25f, "p", .9f);
            Assert.That(a.Snapshot().confirmedSurvivors, Is.Empty, "off the carrier's list");
            Motion(a, 2); a.OnAirTemperature(2, 20); a.Step(2, 1f);
            var sent = Wire(a.BuildMessage(2));
            Assert.That(sent.CarriedOut, Has.Length.EqualTo(1), "carried out, round trip through the radio format");
            Assert.That(sent.Survivors, Is.Empty);

            List<BrainEvent> Receive(bool withCarried, out int remaining)
            {
                var b = new RobotBrain(Calibration(), Settings(), "B", 0);
                var sender = new RobotBrain(Calibration(), Settings(), "A", 0);
                var m = sender.BuildMessage(0);
                m.Survivors = new[] { new BrainSurvivor { position = new Vector2(0, 3), confirmedAt = 0, sourceRobot = "A" } };
                b.OnPeerMessage(Wire(m), .2);
                m.Sequence++; m.SentAt = .3;
                if (withCarried) m.CarriedOut = new[] { new BrainCarried { position = new Vector2(0, 3), at = .25, sourceRobot = "A" } };
                b.OnPeerMessage(Wire(m), .4);
                remaining = b.Snapshot().confirmedSurvivors.Length;
                return b.DrainEvents().ToList();
            }
            var events = Receive(true, out int left);
            Assert.That(left, Is.Zero, "the partner drops the person");
            Assert.That(events.Count(e => e.kind == "survivor_carried_by_peer"), Is.EqualTo(1));
            Receive(false, out int kept);
            Assert.That(kept, Is.EqualTo(1), "control: no carried-out entry, the person stays");
        }

        [Test]
        public void TwoAgreeingCameraPlaceMatchesLinkTheFramesAndOneOrAConflictDoesNot()
        {
            var two = CameraScene();
            Assert.That(two.brain.Snapshot().survivors, Is.Empty, "partner content stays withheld before linking");
            Assert.That(two.brain.Snapshot().peerExplored.All(x => x == 0), Is.True);
            two.Match(0); two.Match(1);
            Assert.That(two.brain.TryPartnerLink("A", out var link), Is.True, "two map-verified camera candidates agree");
            Assert.That(Vector2.Distance(link.position, two.truth.position), Is.LessThan(.01f));
            var shared = two.brain.Snapshot();
            Assert.That(Vector2.Distance(shared.survivors.Single().position, two.truth.position), Is.LessThan(.01f));
            Assert.That(shared.peerExplored.Any(x => x != 0), Is.True, "the verified camera link applies the waiting partner map");
            Assert.That(Mathf.Abs(Mathf.DeltaAngle(link.yawRad * Mathf.Rad2Deg, two.truth.yawRad * Mathf.Rad2Deg)), Is.LessThan(.1f));
            Assert.That(two.brain.DrainEvents().Single(e => e.kind == "frame_linked").reason, Does.Contain("camera"));
            var one = CameraScene(); one.Match(0);
            Assert.That(one.brain.TryPartnerLink("A", out _), Is.False, "control: one match waits");
            var conflict = CameraScene(); conflict.Match(0); conflict.Match(1, 1f);
            Assert.That(conflict.brain.TryPartnerLink("A", out _), Is.False, "a camera pose displaced by 1 m cannot link");
            var off = CameraScene(enabled: false); off.Match(0); off.Match(1);
            Assert.That(off.brain.TryPartnerLink("A", out _), Is.False, "control: camera linking disabled");
        }

        [Test]
        public void CameraKeyframesSurviveTheRadioFormat()
        {
            var m = new RobotBrain(Calibration(), Settings(), "A", 0).BuildMessage(0);
            Assert.That(Wire(m).VisualKeyframes, Is.Empty, "control: no camera keyframes added");
            var payload = Enumerable.Range(0, 1900).Select(i => (byte)(i * 7)).ToArray();
            m.VisualKeyframes = new[] { new BrainVisualKeyframe { id = 7, pose = new BrainPose { position = new Vector2(1.5f, -2), yawRad = .4f }, payload = payload } };
            var back = Wire(m).VisualKeyframes.Single();
            Assert.That(back.id, Is.EqualTo(7));
            Assert.That(back.pose.position, Is.EqualTo(new Vector2(1.5f, -2)));
            Assert.That(back.pose.yawRad, Is.EqualTo(.4f));
            CollectionAssert.AreEqual(payload, back.payload);
        }

        [Test]
        public void PeerSurvivorCancelsInspectionAndLogsOnlyOnce()
        {
            var b = new RobotBrain(Calibration(), Settings(), "B", 0);
            for (int i = 0; i < 3; i++)
            {
                double t = i / 15.0; Motion(b, t); b.OnAirTemperature(t, 20);
                var pair = SceneImages(t, Vector2.zero, 0, Calibration());
                b.OnRgbd(pair.Item1); b.OnThermal(pair.Item2); b.Step(t, 1f / 15);
            }
            Assert.That(b.Snapshot().state, Is.EqualTo("survivor")); b.DrainEvents();
            var sender = new RobotBrain(Calibration(), Settings(), "A", 0);
            var m = sender.BuildMessage(0);
            m.Survivors = new[] { new BrainSurvivor { position = new Vector2(0, 3), confirmedAt = 0, sourceRobot = "A" } };
            b.OnPeerMessage(Wire(m), .2); b.OnPeerMessage(Wire(m), .21);
            m.Sequence++; m.SentAt = .22; b.OnPeerMessage(Wire(m), .3);
            var all = new List<BrainEvent>(b.DrainEvents());
            for (int i = 0; i <= 60; i++)
            {
                double t = .4 + i / 15.0; Motion(b, t);
                var pair = SceneImages(t, Vector2.zero, 0, Calibration());
                b.OnRgbd(pair.Item1); b.OnThermal(pair.Item2); b.OnAirTemperature(t, 20);
                b.Step(t, i == 0 ? (float)(t - 2 / 15.0) : 1f / 15); all.AddRange(b.DrainEvents());
                Assert.That(b.Snapshot().state, Is.Not.EqualTo("survivor"));
            }
            Assert.That(all.Count(e => e.kind == "survivor_from_peer"), Is.EqualTo(1));
            Assert.That(all.Any(e => e.kind == "survivor_candidate" || e.kind == "survivor_confirmed"), Is.False);
            Assert.That(b.Snapshot().survivors.Single().source, Is.EqualTo("peer"));
        }

        [Test]
        public void PeerObstacleBlocksUnseenRoomsAndFreshDepthOverridesIt()
        {
            var a = new RobotBrain(Calibration(), Settings(), "A", 0);
            var b = new RobotBrain(Calibration(), Settings(), "B", 0);
            Room(a, -3, 3, -3, 3);
            var point = new Vector2(.05f, 2.05f); Map(a).ObserveOccupied(Map(a).Cell(point));
            b.OnPeerMessage(Wire(a.BuildMessage(0)), .1);
            var g = Map(b); var planner = new GridPlanner(g, Settings());
            Assert.That(g.OccupancyAt(point), Is.EqualTo(255));
            Assert.That(planner.Plan(Vector2.zero, point), Is.Empty);
            g.ObserveFree(g.Cell(point));
            Assert.That(g.OccupancyAt(point), Is.EqualTo(1));
            Assert.That(g.PeerExplored(g.Cell(point)), Is.True);
        }

        [Test]
        public void PeerHotspotBlocksPlanningAndOldHeatArrivesAlreadyDecayed()
        {
            var a = new RobotBrain(Calibration(), Settings(), "A", 0);
            var b = new RobotBrain(Calibration(), Settings(), "B", 0);
            Room(a, -4, 4, -4, 4); var point = new Vector2(.05f, 2.05f);
            Map(a).ObserveHeat(Map(a).Cell(point), 600, 0);
            var m = Wire(a.BuildMessage(0)); b.OnPeerMessage(m, 2);
            Assert.That(Map(b).Heat(Map(b).Cell(point)), Is.EqualTo(599.6f).Within(.01));
            Assert.That(new GridPlanner(Map(b), Settings()).Plan(Vector2.zero, point), Is.Empty);
            var old = new RobotBrain(Calibration(), Settings(), "C", 0); old.OnPeerMessage(m, 3000);
            Assert.That(Map(old).Heat(Map(old).Cell(point)), Is.EqualTo(20).Within(.01));
        }

        [Test]
        public void LostDeltasRecoverFromAcknowledgementAndFromFullSnapshot()
        {
            var a = new RobotBrain(Calibration(), Settings(), "A", 0);
            var b = new RobotBrain(Calibration(), Settings(), "B", 0);
            b.OnPeerMessage(Wire(a.BuildMessage(0)), .1);
            a.OnPeerMessage(Wire(b.BuildMessage(.1)), .2);
            Room(a, -4, -2, -2, 2); var dropped = a.BuildMessage(2);
            Assert.That(dropped.FullSnapshot, Is.False);
            Room(a, 2, 4, -2, 2);
            b.OnPeerMessage(Wire(a.BuildMessage(4)), 4.1);
            Assert.That(Map(b).PeerExplored(Map(b).Cell(new Vector2(-3, 0))), Is.True);
            b.OnPeerMessage(Wire(dropped), 4.2);
            Assert.That(b.Snapshot().peers.Single().sequence, Is.GreaterThan(dropped.Sequence));
            var late = new RobotBrain(Calibration(), Settings(), "C", 0);
            var full = Wire(a.BuildMessage(10)); Assert.That(full.FullSnapshot, Is.True);
            late.OnPeerMessage(full, 10.1);
            Assert.That(Map(late).PeerExplored(Map(late).Cell(new Vector2(-3, 0))), Is.True);
            Assert.That(Map(late).PeerExplored(Map(late).Cell(new Vector2(3, 0))), Is.True);
            Assert.That(b.DrainEvents().Count(e => e.kind == "peer_message_received"), Is.EqualTo(2));
        }

        [Test]
        public void NewPeerDoesNotAcknowledgeAnUnseenDeltaBaseline()
        {
            var a = new RobotBrain(Calibration(), Settings(), "A", 0);
            var b = new RobotBrain(Calibration(), Settings(), "B", 0);
            var late = new RobotBrain(Calibration(), Settings(), "C", 0);
            Room(a, -4, -2, -2, 2);
            b.OnPeerMessage(Wire(a.BuildMessage(0)), .1);
            a.OnPeerMessage(Wire(b.BuildMessage(.1)), .2);
            Room(a, 2, 4, -2, 2);
            var delta = Wire(a.BuildMessage(2)); Assert.That(delta.BaseSequence, Is.EqualTo(1));
            late.OnPeerMessage(delta, 2.1);
            Assert.That(Map(late).PeerExplored(Map(late).Cell(new Vector2(-3, 0))), Is.False);
            var ack = Wire(late.BuildMessage(2.1));
            Assert.That(ack.Acknowledgements.Single(x => x.sender == "A").sequence, Is.Zero);
            a.OnPeerMessage(ack, 2.2);
            late.OnPeerMessage(Wire(a.BuildMessage(4)), 4.1);
            Assert.That(Map(late).PeerExplored(Map(late).Cell(new Vector2(-3, 0))), Is.True);
        }

        [Test]
        public void MessageByteLimitsUseActualBinaryRoundTripsAtHouseScale()
        {
            var a = new RobotBrain(Calibration(), HouseSettings(), "robot_A", 0);
            Room(a, 0, 40, 0, 28); var g = Map(a);
            Assert.That(g.Cell(new Vector2(40, 28)), Is.GreaterThanOrEqualTo(0));
            for (int i = 0; i < g.Count; i++)
            {
                var p = g.Centre(i);
                if ((p.x > 0 && p.x < .2f || p.x > 39.8f && p.x < 40) && p.y > 0 && p.y < 28) g.ObserveOccupied(i);
                if (p.x > 10 && p.x < 12 && p.y > 8 && p.y < 10) g.ObserveHeat(i, 600, 0);
            }
            var full = Wire(a.BuildMessage(0));
            Assert.That(full.GridSize, Is.EqualTo(220)); Assert.That(full.Origin, Is.EqualTo(new Vector2(-2, -2)));
            // The receiver stands on open floor: a partner's reported pose clears this robot's walls under its body
            // (MarkTeammateFree), and the default (0, 0) lies inside the wall strip at x 0 to 0.2, where no robot can stand.
            var receiver = new RobotBrain(Calibration(), HouseSettings(), "robot_B", 0, new EntryPose(20, 14, 0));
            receiver.OnPeerMessage(full, 0);
            a.OnPeerMessage(Wire(receiver.BuildMessage(0)), .1);
            var delta = Wire(a.BuildMessage(2));
            Assert.That(delta.BaseSequence, Is.EqualTo(full.Sequence));
            foreach (var m in new[] { full, delta })
            {
                m.Goal = new Vector2(2, 1);
                m.Claim = new BrainClaim { centre = m.Goal.Value, radiusM = 1, expiresAt = m.SentAt + 15 };
                m.Survivors = Enumerable.Range(0, 4).Select(i => new BrainSurvivor
                    { position = new Vector2(i, 3), confirmedAt = 0, sourceRobot = "robot_A" }).ToArray();
                m.Acknowledgements = new[] { new BrainAck { sender = "robot_B", sequence = 1 }, new BrainAck { sender = "robot_C", sequence = 2 } };
            }
            int baselineFull = full.EstimatedBytes, baselineDelta = delta.EstimatedBytes;
            foreach (var m in new[] { full, delta })
            {
                // The deployed packet carries one 360-beam sweep and one 96-feature ORB payload.
                m.Keyframes = Enumerable.Range(0, Settings().visual_lidar_keyframes_per_message).Select(i => new BrainKeyframe { id = i, pose = new BrainPose(),
                    angleMinRad = -Mathf.PI, incrementRad = 2 * Mathf.PI / 360, minM = .15f, maxM = 12,
                    rangesCm = Enumerable.Repeat((ushort)200, 360).ToArray() }).ToArray();
                m.VisualKeyframes = new[] { new BrainVisualKeyframe { id = 1, pose = new BrainPose(), payload = VisualPayload() } };
            }
            full = Wire(full); delta = Wire(delta);
            TestContext.WriteLine($"Without keyframes full = {baselineFull}, delta = {baselineDelta}; visual payload = {VisualPayload().Length} bytes. House full = " + full.EstimatedBytes + " bytes; delta = " + delta.EstimatedBytes + " bytes.");
            Assert.That(full.EstimatedBytes, Is.LessThan(40000)); Assert.That(delta.EstimatedBytes, Is.LessThan(8000));
            CollectionAssert.AreEqual(full.Occupied, delta.Occupied);
            var idle = new RobotBrain(Calibration(), Settings(), "idle", 0);
            idle.BuildMessage(0); Assert.That(idle.BuildMessage(2), Is.Null);
            Assert.That(idle.BuildMessage(10).FullSnapshot, Is.True);
        }

        [Test]
        public void DenseFireSnapshotFitsTheEntireHouseWithoutDroppingHotspots()
        {
            var a = new RobotBrain(Calibration(), HouseSettings(), "robot_A", 0);
            Room(a, 0, 40, 0, 28); var g = Map(a);
            for (int i = 0; i < g.Count; i++)
            {
                var p = g.Centre(i);
                if (p.x > 10 && p.x < 30 && p.y > 7 && p.y < 21) g.ObserveHeat(i, 600 + i % 50, 0);
            }
            var full = Wire(a.BuildMessage(0));
            int withoutCamera = full.EstimatedBytes;
            full.Keyframes = new[] { new BrainKeyframe { id = 0, angleMinRad = -Mathf.PI, incrementRad = 2 * Mathf.PI / 360,
                minM = .15f, maxM = 12, rangesCm = Enumerable.Repeat((ushort)200, 360).ToArray() } };
            full.VisualKeyframes = new[] { new BrainVisualKeyframe { id = 0, payload = VisualPayload() } };
            full = Wire(full);
            TestContext.WriteLine("Dense fire without camera = " + withoutCamera + " bytes.");
            TestContext.WriteLine("Dense fire full = " + full.EstimatedBytes + " bytes for " + full.Hotspots.Length + " hot cells.");
            Assert.That(full.Hotspots.Length, Is.EqualTo(7000));
            Assert.That(full.EstimatedBytes, Is.LessThan(40000));
        }

        [Test]
        public void DenseHeatWirePrecisionIsBoundedAndReencodingIsStable()
        {
            var a = new RobotBrain(Calibration(), Settings(), "A", 0);
            var m = a.BuildMessage(1);
            m.Hotspots = Enumerable.Range(0, 1000).Select(i => new BrainHotspot
                { cell = i, temperatureC = 345.16f, observedAt = .13 }).ToArray();
            var copy = Wire(m);
            Assert.That(copy.Hotspots.Length, Is.EqualTo(1000));
            foreach (var h in copy.Hotspots)
            {
                Assert.That(h.temperatureC, Is.EqualTo(345.16f).Within(.051));
                Assert.That(h.observedAt, Is.InRange(.03, .13));
            }
            CollectionAssert.AreEqual(copy.ToBytes(), Wire(copy).ToBytes());
        }

        [Test]
        public void NextFullSnapshotRecoversAnObservationDroppedDuringAnOutage()
        {
            var a = new RobotBrain(Calibration(), Settings(), "A", 0);
            var b = new RobotBrain(Calibration(), Settings(), "B", 0);
            b.OnPeerMessage(Wire(a.BuildMessage(0)), .1);
            a.OnPeerMessage(Wire(b.BuildMessage(.1)), .2);
            Room(a, 2, 4, -2, 2); a.BuildMessage(2); // Link drops this observation.
            var cell = Map(b).Cell(new Vector2(3, 0));
            Assert.That(Map(b).PeerExplored(cell), Is.False);
            var newest = Wire(a.BuildMessage(10)); Assert.That(newest.FullSnapshot, Is.True);
            b.OnPeerMessage(newest, 10.1);
            Assert.That(Map(b).PeerExplored(cell), Is.True);
        }

        [Test]
        public void MessageExchangeIsDeterministicAndSnapshotsAreDetached()
        {
            var first = new[] { new RobotBrain(Calibration(), Settings(), "A", 5), new RobotBrain(Calibration(), Settings(), "B", 6) };
            var second = new[] { new RobotBrain(Calibration(), Settings(), "A", 5), new RobotBrain(Calibration(), Settings(), "B", 6) };
            for (int tick = 0; tick <= 60; tick++)
            {
                double t = tick / 15.0;
                for (int i = 0; i < 2; i++)
                {
                    foreach (var group in new[] { first, second })
                    { Motion(group[i], t); group[i].OnRgbd(Image(t, true)); group[i].OnAirTemperature(t, 20); }
                    var x = first[i].Step(t, 1f / 15); var y = second[i].Step(t, 1f / 15);
                    Assert.That(x.forwardMps, Is.EqualTo(y.forwardMps)); Assert.That(x.yawRateDeg, Is.EqualTo(y.yawRateDeg));
                }
                if (tick % 30 == 0)
                {
                    var x = first.Select(b => b.BuildMessage(t)).ToArray(); var y = second.Select(b => b.BuildMessage(t)).ToArray();
                    for (int i = 0; i < 2; i++)
                    {
                        Assert.That(x[i] == null, Is.EqualTo(y[i] == null)); if (x[i] == null) continue;
                        CollectionAssert.AreEqual(x[i].ToBytes(), y[i].ToBytes());
                        first[1-i].OnPeerMessage(Wire(x[i]), t); second[1-i].OnPeerMessage(Wire(y[i]), t);
                    }
                }
            }
            for (int i = 0; i < 2; i++)
            {
                CollectionAssert.AreEqual(first[i].Snapshot().occupancy, second[i].Snapshot().occupancy);
                var x = first[i].DrainEvents(); var y = second[i].DrainEvents(); Assert.That(x.Count, Is.EqualTo(y.Count));
                for (int j = 0; j < x.Count; j++)
                { Assert.That(x[j].kind, Is.EqualTo(y[j].kind)); Assert.That(x[j].reason, Is.EqualTo(y[j].reason)); CollectionAssert.AreEqual(x[j].numbers, y[j].numbers); }
                var snapshot = first[i].Snapshot(); Array.Clear(snapshot.peerExplored, 0, snapshot.peerExplored.Length);
                Assert.That(first[i].Snapshot().peerExplored.Any(v => v != 0), Is.True);
            }
        }

        [Test]
        public void BrainAssemblyHasOnlySensorDependency()
        {
            string path = Path.Combine(Application.dataPath, "SmokeRescue", "Brain", "SmokeRescue.Brain.asmdef");
            var def = JsonUtility.FromJson<AssemblyDefinition>(File.ReadAllText(path));
            CollectionAssert.AreEqual(new[] { "SmokeRescue.Sensors" }, def.references);
            Assert.That(def.noEngineReferences, Is.False);
        }
        [Serializable] public sealed class AssemblyDefinition { public string[] references; public bool noEngineReferences; }
    }
}
