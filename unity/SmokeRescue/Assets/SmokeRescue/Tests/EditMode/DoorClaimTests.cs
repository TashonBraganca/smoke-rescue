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
        /// clears it, and from then on no search goal lies behind it. Control: switched off, nothing is taken and the route stays.</summary>
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
            var blocks = typeof(RobotBrain).GetMethod("ClaimBlocks", Private);
            Assert.That(blocks.Invoke(outAgain.b, new object[] { new Vector2(0, 2.5f), 300.0 }), Is.True, "no search goal in a cleared room");
            Assert.That(blocks.Invoke(outAgain.b, new object[] { new Vector2(0, -2.5f), 300.0 }), Is.False, "control: the floor outside it");

            var off = DoorWalk(false, 2.5f, -2.5f);
            Assert.That(off.events.Any(e => e.kind.StartsWith("door_")), Is.False, "control: off");
            Assert.That(typeof(RobotBrain).GetMethod("ClaimBlocks", Private).Invoke(off.b, new object[] { new Vector2(0, 2.5f), 300.0 }), Is.False, "control: off, the room stays a goal");
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
        /// <summary>A doorway crossed at 45 deg from a 1.2 m hallway is still taken: the gap is measured as the narrowest cut
        /// through the robot, not across its heading (headless houses 5001-5022: 28 of 77 room entries were claimed).</summary>
        [Test]
        public void ADoorwayCrossedAtAnAngleFromAHallwayIsTaken()
        {
            var c = Settings(); c.grid_size_m = 12; c.door_claims = true; c.door_home_clear_m = .5f;
            var b = new RobotBrain(Calibration(), c, "A", 0, new EntryPose(-.6f, -.6f, 0)); var g = Map(b);
            Room(b, -4, 4, -1.2f, 4.5f);
            for (int i = 0; i < g.Count; i++)
            {
                var q = g.Centre(i);
                if ((Mathf.Abs(q.y) < .06f && Mathf.Abs(q.x) > .45f) || Mathf.Abs(q.y + 1.25f) < .06f) g.ObserveOccupied(i);
            }
            var motion = (DeadReckoning)typeof(RobotBrain).GetField("motion", Private).GetValue(b);
            var update = typeof(RobotBrain).GetMethod("UpdateDoors", Private);
            float yaw = -45 * Mathf.Deg2Rad; var now = motion.Pose.position;
            motion.ApplyCorrection(new BrainPose { position = now - new BrainPose { position = Vector2.zero, yawRad = yaw }.Transform(now.x, now.y), yawRad = yaw });
            var step = new Vector2(.05f, .05f).normalized * .05f; var events = new List<BrainEvent>();
            for (int k = 1; k <= 70; k++)
            {
                motion.ApplyCorrection(new BrainPose { position = step, yawRad = 0 });
                update.Invoke(b, new object[] { k * .05 }); events.AddRange(b.DrainEvents());
            }
            Assert.That(motion.Pose.position.y, Is.GreaterThan(1.2f), "walked through into the room");
            Assert.That(events.Count(e => e.kind == "door_taken"), Is.EqualTo(1));
        }

        /// <summary>door_enter. A robot 3 m from a 0.9 m doorway lists it from its map and plans a route that ends inside the room
        /// beyond. Controls: switched off, nothing is listed; a partner already heading there (radio) wins it.</summary>
        [Test]
        public void ADoorwayInTheMapBecomesAGoalInsideItsRoom()
        {
            (RobotBrain b, List<BrainEvent> ev, List<Vector2> route, bool ok) Run(bool enter, bool partnerFirst)
            {
                var c = Settings(); c.grid_size_m = 12; c.door_claims = true; c.door_enter = enter; c.door_home_clear_m = 1;
                var b = new RobotBrain(Calibration(), c, "A", 0, new EntryPose(0, -3, 0)); Room(b, -2.5f, 2.5f, -3.5f, 4.5f);
                var g = Map(b);
                for (int i = 0; i < g.Count; i++) { var q = g.Centre(i); if (Mathf.Abs(q.y) < .06f && Mathf.Abs(q.x) <= 2.5f && Mathf.Abs(q.x) > .45f) g.ObserveOccupied(i); }
                typeof(RobotBrain).GetMethod("UpdateDoors", Private).Invoke(b, new object[] { 1.0 });
                if (partnerFirst)
                    typeof(RobotBrain).GetMethod("ReceiveDoors", Private).Invoke(b, new object[] { new[] { new BrainDoor { position = Vector2.zero, inward = Vector2.up, owner = "robot_B", state = 3, since = .5 } }, 1.0 });
                var args = new object[] { 1.1, null };
                bool ok = (bool)typeof(RobotBrain).GetMethod("EnterDoorway", Private).Invoke(b, args);
                return (b, b.DrainEvents().ToList(), (List<Vector2>)args[1], ok);
            }
            var on = Run(true, false);
            Assert.That(on.ev.Count(e => e.kind == "door_seen"), Is.EqualTo(1), "one doorway listed");
            Assert.That(on.ok, Is.True, "a route to it");
            Assert.That(on.route[on.route.Count - 1].y, Is.GreaterThan(.8f), "the route ends inside the room, past the crossing distance");
            Assert.That(on.b.BuildMessage(2).Doors.Count(d => d.state == 3), Is.EqualTo(1), "radioed as heading");
            var off = Run(false, false);
            Assert.That(off.ev.Any(e => e.kind == "door_seen"), Is.False, "control: off");
            Assert.That(off.ok, Is.False, "control: off, no route");
            var partner = Run(true, true);
            Assert.That(partner.ok, Is.False, "control: the partner was heading there first");
        }
    }
}
