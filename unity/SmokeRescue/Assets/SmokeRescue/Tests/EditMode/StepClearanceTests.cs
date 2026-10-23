using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using SmokeRescue.Brain;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    public sealed partial class BrainTests
    {
        /// <summary>The pursuit (FollowOrHold) driven for 2.9 s, under collision_escape_s, from house 5021's bedroom door: a wall
        /// column with a 0.9 m gap of free cells, the body two columns short of the wall and 0.02 mm short of the next cell's
        /// boundary (robot B at 86.6 s: (0.59998, 8.9955), the jamb cell at (0.75, 8.65), heading -55 deg), routed to a goal
        /// 0.5 m past the door. The body moves as commanded. Returns how far it travelled, its least distance to any wall
        /// cell's square, and whether a safe-chord hold was raised.</summary>
        static (float moved, float closest, bool held) DoorStep(bool exact)
        {
            var c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f; c.step_exact_clearance = exact;
            var b = new RobotBrain(Calibration(), c, "A", 0, new EntryPose(0, 0, 0));
            Room(b, -4, 4, -4, 4);
            var g = Map(b);
            int jamb = g.Cell(new Vector2(1f, -1f)), jx = jamb % g.Size, jy = jamb / g.Size;
            var walls = new List<Vector2>();
            for (int row = 0; row < g.Size; row++)
                if (row <= jy || row >= jy + 10) { g.ObserveOccupied(row * g.Size + jx); walls.Add(g.Centre(row * g.Size + jx)); }
            Vector2 j = g.Centre(jamb);
            var at = new Vector2(j.x - 1.5f * g.CellM - 2e-5f, j.y + 3 * g.CellM + .0455f);
            var motion = (DeadReckoning)typeof(RobotBrain).GetField("motion", Private).GetValue(b);
            void MoveTo(Vector2 p, float yaw)
            {
                var now = motion.Pose; float turn = yaw - now.yawRad;
                var rotated = new BrainPose { position = Vector2.zero, yawRad = turn }.Transform(now.position.x, now.position.y);
                motion.ApplyCorrection(new BrainPose { position = p - rotated, yawRad = turn });
            }
            MoveTo(at, -.9666f);
            Assert.That(Vector2.Distance(motion.Pose.position, at), Is.LessThan(1e-4f));
            typeof(OccupancyGrid).GetMethod("PrepareNavigation", Private).Invoke(g, null);
            var walkable = typeof(OccupancyGrid).GetMethod("Walkable", Private);
            Assert.That(walkable.Invoke(g, new object[] { g.Cell(at) }), Is.True, "the body's own cell is walkable");
            Assert.That(walkable.Invoke(g, new object[] { g.Cell(at + new Vector2(1e-3f, 0)) }), Is.False, "the cell 1 mm ahead is inside the planning margin");
            var route = new GridPlanner(g, c).Plan(at, j + new Vector2(.5f, .5f));
            Assert.That(route, Is.Not.Empty);
            typeof(RobotBrain).GetMethod("SetPath", Private).Invoke(b, new object[] { route, true });
            var follow = typeof(RobotBrain).GetMethod("FollowOrHold", Private);
            float dt = 1f / 60, closest = float.PositiveInfinity; bool held = false;
            for (int k = 0; k < 2.9f * 60; k++)
            {
                var command = (BrainCommand)follow.Invoke(b, new object[] { k * (double)dt, dt, null });
                held |= b.DrainEvents().Any(e => e.kind == "collision_risk" && e.reason.Contains("safe chord"));
                var pose = motion.Pose;
                MoveTo(pose.position + pose.Forward * command.forwardMps * dt, pose.yawRad + command.yawRateDeg * Mathf.Deg2Rad * dt);
                foreach (var w in walls)
                {
                    var p = motion.Pose.position; float ex = Mathf.Max(Mathf.Abs(p.x - w.x) - g.CellM / 2, 0), ey = Mathf.Max(Mathf.Abs(p.y - w.y) - g.CellM / 2, 0);
                    closest = Mathf.Min(closest, Mathf.Sqrt(ex * ex + ey * ey));
                }
            }
            return (Vector2.Distance(motion.Pose.position, at), closest, held);
        }

        /// <summary>step_exact_clearance. Headless house 5021 (23 Oct final), robot B: 57 s at a bedroom door, every one of its
        /// safe-chord holds with the body more than 0.32 m from the walls. With the step judged by its cell the robot does not
        /// move and raises the hold; judged by its point it drives on through the door, its centre never within robot_radius_m
        /// of a wall cell's square.</summary>
        [Test]
        public void AStepJudgedByItsPointLeavesACellBoundaryAtADoorTheCellRuleHeldItOn()
        {
            var cellRule = DoorStep(false);
            Assert.That(cellRule.held, Is.True, "control: the cell rule raises the safe-chord hold");
            Assert.That(cellRule.moved, Is.LessThan(.02f), "control: the cell rule holds the body still");
            var pointRule = DoorStep(true);
            TestContext.WriteLine($"cell rule moved {cellRule.moved:0.000} m; point rule moved {pointRule.moved:0.000} m, closest {pointRule.closest:0.000} m");
            Assert.That(pointRule.held, Is.False);
            Assert.That(pointRule.moved, Is.GreaterThan(.3f), "drives on toward the door");
            Assert.That(pointRule.closest, Is.GreaterThan(.25f), "the centre keeps robot_radius_m from every wall cell");
        }
    }
}
