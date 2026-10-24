using System.Linq;
using NUnit.Framework;
using SmokeRescue.Brain;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    public sealed partial class BrainTests
    {
        /// <summary>A robot facing +y ordered to a point 3 m to its left, whose body cannot turn left (each left yaw command
        /// is refused, as the partner's touching tail refused it in headless house 5027) and can turn right. Returns the
        /// most it turned, how far it moved and whether it raised turn_stall.</summary>
        static (float turnedDeg, float movedM, bool stalled) LeftBlocked(bool flip)
        {
            var c = Settings(); c.grid_size_m = 12; c.turn_stall_flip = flip;
            var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -5, 5, -5, 5);
            b.AgentGoTo(new Vector2(-3, 0), 0, 60, "a point to the left");
            var command = new BrainCommand(); float turned = 0; bool stalled = false;
            for (int tick = 0; tick < 240; tick++)
            {
                float ccw = command.yawRateDeg > 0 ? 0 : command.yawRateDeg * Mathf.Deg2Rad;   // left refused, right allowed
                command = DriveTick(b, tick, command.forwardMps, ccw * Mathf.Rad2Deg);
                stalled |= b.DrainEvents().Any(e => e.kind == "turn_stall");
                turned = Mathf.Max(turned, Mathf.Abs(Mathf.DeltaAngle(0, b.Snapshot().pose.yawRad * Mathf.Rad2Deg)));
            }
            return (turned, b.Snapshot().pose.position.magnitude, stalled);
        }

        /// <summary>turn_stall_flip. Headless house 5027: robot A commanded the short left turn every tick from 30 to 240 s while
        /// the touching partner refused it; 0 of 5 found. With the flip it turns right round and drives to the goal (5027: 4 of
        /// 5, house searched 22 -> 90 %). Control: without it the robot never turns and never moves.</summary>
        [Test]
        public void ATurnTheBodyCannotMakeIsTakenTheOtherWayRound()
        {
            var off = LeftBlocked(false);
            Assert.That(off.turnedDeg, Is.LessThan(5), "control: the refused left turn is commanded forever");
            Assert.That(off.movedM, Is.LessThan(.1f));
            var on = LeftBlocked(true);
            Assert.That(on.stalled, Is.True);
            Assert.That(on.turnedDeg, Is.GreaterThan(150), "turns right round past the back");
            Assert.That(on.movedM, Is.GreaterThan(1f), "then drives toward the goal");
        }
    }
}
