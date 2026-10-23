using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using SmokeRescue.Brain;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    public sealed partial class BrainTests
    {
        const BindingFlags MotionPrivate = BindingFlags.Instance | BindingFlags.NonPublic;

        [Test]
        public void MotionKeepsTheNewAgentRouteWhenTheOldPathWasBlocked()
        {
            bool Kept(bool enabled)
            {
                var c = Settings(); c.grid_size_m = 12; c.motion_keep_agent_route = enabled;
                var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -5, 5, -5, 5);
                var target = new Vector2(-3.05f, .05f);
                b.AgentGoTo(target, 0, 30, "look from the other side");
                var old = new List<Vector2> { new Vector2(.05f, .05f), new Vector2(.05f, 3.05f) };
                Map(b).ObserveOccupied(Map(b).Cell(old[1]));
                typeof(RobotBrain).GetMethod("SetPath", MotionPrivate).Invoke(b, new object[] { old, true });
                DriveTick(b, 0);
                var events = b.DrainEvents();
                Assert.That(events.Any(e => e.kind == "goal" && e.reason.StartsWith("Agent order:")), Is.True,
                    "the agent gets a valid replacement route before the old blocked flag is tested again");
                return b.Snapshot().goal.HasValue && Vector2.Distance(b.Snapshot().goal.Value, target) < .15f;
            }
            Assert.That(Kept(false), Is.False, "control discards the newly built agent route in the same tick");
            Assert.That(Kept(true), Is.True, "the replacement route must survive frontier fallback");
        }

        [Test]
        public void MotionSweepDoesNotTurnForFloorBelowTheCameraView()
        {
            int Starts(bool enabled, bool distant)
            {
                var c = Settings(); c.grid_size_m = 12; c.motion_sweep_visible_floor = enabled;
                var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -5, 5, -5, 5);
                var g = Map(b); var searched = (bool[])typeof(OccupancyGrid).GetField("observed", MotionPrivate).GetValue(g);
                for (int i = 0; i < g.Count; i++)
                {
                    var p = g.Centre(i);
                    // The near patch is inside the horizontal view but below the bottom camera ray.
                    searched[i] = distant ? !(p.y > 2.8f && p.y < 3.2f && Mathf.Abs(p.x) < .2f)
                        : p.magnitude > .9f;
                }
                typeof(RobotBrain).GetMethod("OnFrontierReached", MotionPrivate).Invoke(b, new object[] { Vector2.zero, 0.0 });
                typeof(RobotBrain).GetMethod("SweepStep", MotionPrivate).Invoke(b, new object[] { 0.0, .05f });
                return (bool)typeof(RobotBrain).GetField("sweeping", MotionPrivate).GetValue(b) ? 1 : 0;
            }
            Assert.That(Starts(false, false), Is.EqualTo(1), "old eligibility spends a sweep on its blind floor");
            Assert.That(Starts(true, false), Is.Zero, "rotation cannot reveal floor below the camera");
            Assert.That(Starts(true, true), Is.EqualTo(1), "a distant unsearched patch still earns a look");
        }

        [Test]
        public void MotionInspectionTimeoutReleasesAnUnsettledViewWithoutConfirmingIt()
        {
            bool TimedOut(bool enabled, bool steady)
            {
                var c = Settings(); c.grid_size_m = 12; c.motion_bounded_inspection = enabled;
                c.survivor_stop_min_m = 3; c.survivor_stop_max_m = 4; c.survivor_standoff_m = 3.3f;
                c.person_rgb_strong_confidence = .7f; c.survivor_lost_in_view = true;
                var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -5, 5, -5, 5);
                var events = new List<BrainEvent>();
                for (int tick = 0; tick < 150; tick++)
                {
                    double t = tick / 20.0;
                    b.OnPersonSighting(t, steady ? 0 : 30, 3.5f, .9f);
                    DriveTick(b, tick); // A pinned body supplies no yaw, so an off-axis view cannot settle.
                    events.AddRange(b.DrainEvents());
                }
                if (!steady) Assert.That(b.Snapshot().survivors.Count, Is.Zero, "a timeout supplies no confirmation evidence");
                else Assert.That(events.Any(e => e.kind == "survivor_confirmed"), Is.True, "a normal steady dwell still confirms");
                return events.Any(e => e.kind == "survivor_view_timeout");
            }
            Assert.That(TimedOut(false, false), Is.False, "control remains inspecting");
            Assert.That(TimedOut(true, false), Is.True, "an unsettled view must release the robot after its viewing budget");
            Assert.That(TimedOut(true, true), Is.False, "successful inspection keeps its full dwell");
        }

        [Test]
        public void MotionInspectionTimeoutLetsAnInProgressDwellFinish()
        {
            var c = Settings(); c.grid_size_m = 12; c.motion_bounded_inspection = true;
            c.survivor_stop_min_m = 3; c.survivor_stop_max_m = 4; c.survivor_standoff_m = 3.3f;
            c.person_rgb_strong_confidence = .7f; c.survivor_lost_in_view = true;
            var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -5, 5, -5, 5);
            var events = new List<BrainEvent>();
            for (int tick = 0; tick < 180; tick++)
            {
                b.OnPersonSighting(tick / 20.0, 0, 3.5f, .9f);
                DriveTick(b, tick, yawDps: tick < 90 ? (tick % 2 == 0 ? 8 : -8) : 0);
                events.AddRange(b.DrainEvents());
            }
            Assert.That(events.Any(e => e.kind == "survivor_confirmed"), Is.True,
                "the viewing budget must not discard a steady fresh dwell already in progress");
            Assert.That(events.Any(e => e.kind == "survivor_view_timeout"), Is.False);
        }

        [Test]
        public void MotionAgentFaceAcceptsTheCentralCameraViewButStillTurnsForAnOffscreenPoint()
        {
            BrainCommand? Face(bool enabled, float bearing)
            {
                var c = Settings(); c.grid_size_m = 12; c.motion_face_visible = enabled;
                var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -5, 5, -5, 5);
                b.AgentGoTo(Vector2.zero, 0, 30, "inspect", new Vector2(Mathf.Sin(bearing * Mathf.Deg2Rad), Mathf.Cos(bearing * Mathf.Deg2Rad)) * 3.5f);
                typeof(RobotBrain).GetMethod("AgentArrived", MotionPrivate).Invoke(b, new object[] { 0.0, "arrived " });
                return (BrainCommand?)typeof(RobotBrain).GetMethod("FaceStep", MotionPrivate).Invoke(b, new object[] { 0.0, .05f });
            }
            Assert.That(Face(false, 15).HasValue, Is.True, "old code insists on ten degrees even for an in-view point");
            Assert.That(Face(true, 15).HasValue, Is.False, "a point in the central half of both camera views is ready to inspect");
            var away = Face(true, 60);
            Assert.That(away.HasValue && away.Value.yawRateDeg != 0, Is.True, "a point outside the camera still needs a turn");
        }
    }
}
