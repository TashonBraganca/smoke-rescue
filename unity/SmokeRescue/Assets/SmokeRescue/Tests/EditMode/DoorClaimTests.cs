using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using SmokeRescue.Brain;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    public sealed partial class BrainTests
    {
        /// <summary>A brain at (0, -3) in a 5 x 8 m space split by a wall along y = 0 with a 0.9 m doorway at x = 0 (the room
        /// beyond is 4 m deep, the space before 3 m). The body is walked in a straight line by pose corrections, and the
        /// doorway rules run every 0.05 s.</summary>
        static (RobotBrain b, OccupancyGrid g, List<BrainEvent> events) DoorWalk(bool claims, params float[] ys)
        {
            var c = Settings(); c.grid_size_m = 12; c.door_claims = claims; c.door_home_clear_m = 1;
            var b = new RobotBrain(Calibration(), c, "A", 0, new EntryPose(0, -3, 0)); Room(b, -2.5f, 2.5f, -3.5f, 4.5f);
            var g = Map(b);
            for (int i = 0; i < g.Count; i++) { var p = g.Centre(i); if (Mathf.Abs(p.y) < .06f && Mathf.Abs(p.x) <= 2.5f && Mathf.Abs(p.x) > .45f) g.ObserveOccupied(i); }
            var motion = (DeadReckoning)typeof(RobotBrain).GetField("motion", Private).GetValue(b);
            var update = typeof(RobotBrain).GetMethod("UpdateDoors", Private);
            var events = new List<BrainEvent>(); double t = 0;
            float y = -3;
            foreach (float target in ys)
            {
                float yaw = target > y ? 0 : Mathf.PI;
                float turn = yaw - motion.Pose.yawRad; var now = motion.Pose.position;   // turn in place: keep the position
                motion.ApplyCorrection(new BrainPose { position = now - new BrainPose { position = Vector2.zero, yawRad = turn }.Transform(now.x, now.y), yawRad = turn });
                while (Mathf.Abs(target - y) > .01f)
                {
                    float step = Mathf.Clamp(target - y, -.05f, .05f); y += step; t += .05;
                    motion.ApplyCorrection(new BrainPose { position = new Vector2(0, step), yawRad = 0 });
                    update.Invoke(b, new object[] { t }); events.AddRange(b.DrainEvents());
                }
            }
            update.Invoke(b, new object[] { t + 1 }); events.AddRange(b.DrainEvents());
            return (b, g, events);
        }

        /// <summary>door_claims. Walking through a 0.9 m doorway into a room takes it (radioed to the partner); walking back out
        /// clears it, and from then on no route passes it. Control: switched off, nothing is taken and the route stays.</summary>
        [Test]
        public void WalkingIntoARoomTakesItsDoorwayAndWalkingOutClearsIt()
        {
            var inside = DoorWalk(true, 2.5f);
            Assert.That(inside.events.Count(e => e.kind == "door_taken"), Is.EqualTo(1), "taken on entry");
            var sent = inside.b.BuildMessage(100).Doors;
            Assert.That(sent.Length, Is.EqualTo(1)); Assert.That(sent[0].state, Is.EqualTo(1));
            Assert.That(sent[0].position.magnitude, Is.LessThan(.3f), "the doorway point is in the gap");
            Assert.That(sent[0].inward.y, Is.GreaterThan(.9f), "pointing into the room");

            var outAgain = DoorWalk(true, 2.5f, -2.5f);
            Assert.That(outAgain.events.Count(e => e.kind == "door_cleared"), Is.EqualTo(1), "cleared on the way out");
            var planner = new GridPlanner(outAgain.g, Settings());
            Assert.That(planner.Plan(new Vector2(0, -2.5f), new Vector2(0, 2.5f)), Is.Empty, "a cleared room is not entered again");

            var off = DoorWalk(false, 2.5f, -2.5f);
            Assert.That(off.events.Any(e => e.kind.StartsWith("door_")), Is.False, "control: off");
            Assert.That(new GridPlanner(off.g, Settings()).Plan(new Vector2(0, -2.5f), new Vector2(0, 2.5f)), Is.Not.Empty, "control: the route stays");
        }

        /// <summary>A narrow corridor is no doorway: walking 3 m down a 1.0 m corridor takes nothing.</summary>
        [Test]
        public void ANarrowCorridorIsNoDoorway()
        {
            var c = Settings(); c.grid_size_m = 12; c.door_claims = true; c.door_home_clear_m = 1;
            var b = new RobotBrain(Calibration(), c, "A", 0, new EntryPose(0, -3, 0)); Room(b, -.5f, .5f, -3.5f, 4.5f);
            var g = Map(b);
            for (int i = 0; i < g.Count; i++) { var p = g.Centre(i); if (Mathf.Abs(Mathf.Abs(p.x) - .55f) < .06f) g.ObserveOccupied(i); }
            var motion = (DeadReckoning)typeof(RobotBrain).GetField("motion", Private).GetValue(b);
            var update = typeof(RobotBrain).GetMethod("UpdateDoors", Private);
            var events = new List<BrainEvent>();
            for (int k = 1; k <= 80; k++)
            {
                motion.ApplyCorrection(new BrainPose { position = new Vector2(0, .05f), yawRad = 0 });
                update.Invoke(b, new object[] { k * .05 }); events.AddRange(b.DrainEvents());
            }
            Assert.That(events.Any(e => e.kind == "door_taken"), Is.False);
        }

        /// <summary>Radio version 9 carries the doorways exactly.</summary>
        [Test]
        public void DoorsTravelInTheRadioMessage()
        {
            var b = new RobotBrain(Calibration(), Settings(), "A", 0);
            var m = b.BuildMessage(1);
            m.Doors = new[] { new BrainDoor { position = new Vector2(1.5f, -2), inward = new Vector2(0, 1), owner = "robot_A", state = 2, since = .5 } };
            var back = Wire(m);
            Assert.That(back.Doors.Length, Is.EqualTo(1));
            Assert.That(back.Doors[0].position, Is.EqualTo(new Vector2(1.5f, -2))); Assert.That(back.Doors[0].owner, Is.EqualTo("robot_A"));
            Assert.That(back.Doors[0].state, Is.EqualTo(2)); Assert.That(back.Doors[0].since, Is.EqualTo(.5));
        }
    }
}
