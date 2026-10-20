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
            b.SearchArea = Rect.MinMaxRect(-.2f, 3.5f, .2f, 4.1f);
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
            Assert.That(after.Any(f => Vector2.Distance(f.position, near) > 2.5f), Is.True, "frontiers outside the claim stay available");
            Assert.That(after.All(f => Vector2.Distance(f.position, near) > 2.5f), Is.True, "frontiers inside the claim are skipped");
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
            Assert.That(c.cell_m, Is.EqualTo(.1f)); Assert.That(c.grid_size_m, Is.EqualTo(44));
            Assert.That(c.fixed_grid_origin, Is.True); Assert.That(c.grid_origin_x_m, Is.EqualTo(-2));
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
        public void FrontiersOutsideTheOperatorSearchAreaAreNeverGoals()
        {
            var a = new RobotBrain(Calibration(), Settings(), "A", 0, new EntryPose(0, -1, 0));
            Room(a, -3, 3, -3, 3);
            var planner = new GridPlanner(Map(a), Settings());
            var from = new Vector2(-2, -2);
            // Control: with no search area the nearest frontier lies outside the building (here x < 0 or y < 0).
            var open = planner.SelectFrontier(from, p => !a.InSearchArea(p));
            Assert.That(open.HasValue && (open.Value.x < 0 || open.Value.y < 0), Is.True, $"control frontier {open}");
            a.SearchArea = Rect.MinMaxRect(0, 0, 10, 10);
            var inside = planner.SelectFrontier(from, p => !a.InSearchArea(p));
            Assert.That(inside.HasValue, Is.True);
            Assert.That(inside.Value.x, Is.GreaterThanOrEqualTo(0)); Assert.That(inside.Value.y, Is.GreaterThanOrEqualTo(0));
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
