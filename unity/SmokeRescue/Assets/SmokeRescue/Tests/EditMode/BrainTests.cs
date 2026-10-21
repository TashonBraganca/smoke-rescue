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
    public sealed class BrainTests
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
        public void ACarriedPersonEndsPastTheWayOutNotOnTheThreshold()
        {
            // Hidden-corner run: the robot stopped 0.5 m short of its start (outside the front door) and the sled, 0.9 m
            // behind it, set the person down 5 cm inside the doorway. With a tow offset the robot walks straight on until
            // the sled is past its start. Control: tow offset 0 completes on arrival as before.
            float FinalDistancePastStart(float tow)
            {
                var b = new RobotBrain(Calibration(), Settings(), "A", 0); Room(b, -4, 4, -4, 4);
                b.AgentGoTo(new Vector2(0, 2.0f), 0, 30, "person");
                var command = new BrainCommand(); int tick = 0; bool carrying = false;
                for (; tick < 1200; tick++)
                {
                    command = DriveTick(b, tick, command.forwardMps, command.yawRateDeg);
                    var events = b.DrainEvents();
                    if (!carrying && events.Any(e => e.kind == "agent_goal_reached")) { b.BeginCarry(tick / 20.0, .25f, "p", tow); carrying = true; }
                    if (events.Any(e => e.kind == "person_carried_out")) break;
                }
                Assert.That(tick, Is.LessThan(1200), "carried out");
                var pose = b.Snapshot().pose;
                // Distance beyond the start along the way the robot left (toward -y): where the sled ends is tow behind it.
                return -pose.position.y;
            }
            Assert.That(FinalDistancePastStart(.9f), Is.GreaterThanOrEqualTo(.9f - .05f), "the sled is past the start");
            Assert.That(FinalDistancePastStart(0f), Is.LessThan(.6f), "control: no tow, completes on arrival");
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
                return plans.Last().reason;
            }
            Assert.That(ReturnPlan(true), Does.StartWith("Return to the start"), "a route to the partner's entry");
            Assert.That(ReturnPlan(false), Does.StartWith("The known return route is blocked"), "control: own entry only");
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
            Assert.That(cutEvents.Count(e => e.kind == "emergency_exit"), Is.EqualTo(2), "normal route first, then the reduced clearance");
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
                for (int i = 0; i < 60; i++)
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
            for (int i = 0; i < 30; i++)
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
            var receiver = new RobotBrain(Calibration(), HouseSettings(), "robot_B", 0);
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
            full = Wire(full); delta = Wire(delta);
            TestContext.WriteLine("House full = " + full.EstimatedBytes + " bytes; delta = " + delta.EstimatedBytes + " bytes.");
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
