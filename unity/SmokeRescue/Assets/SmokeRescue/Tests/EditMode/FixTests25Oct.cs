using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using SmokeRescue.Brain;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    /// <summary>Fixes from the 25 Oct root-cause analysis of data/recordings/unity_house_onedoor_seed934_full_1520_5of5. Each test
    /// names the time and count that motivated it; the matching config keys carry the same evidence in their *_reason.</summary>
    public sealed partial class BrainTests
    {
        [Test]
        public void AnAgentActionResultTakesTheBrainsClockNotAStaleHostTime()
        {
            // F8. Brain-only seed 934: four "spray_water succeeded" rows were written with t = 0.0 at 371 to 467 s.
            var c = Settings(); c.grid_size_m = 12;
            var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -4, 4, -4, 4);
            for (int i = 0; i <= 60; i++) DriveTick(b, i);
            b.DrainEvents();
            b.LogAgentAction(0.0, "spray_water succeeded (target 11deg)");
            var e = b.DrainEvents().Single(x => x.kind == "agent_order");
            Assert.That(e.t, Is.GreaterThanOrEqualTo(2.95), "stamped with the brain's clock (3.0 s), not 0.0");
        }

        const System.Reflection.BindingFlags NP = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;

        [Test]
        public void WalkingALongRouteOverSeenFloorIsNotCircling()
        {
            // F1. Seed 934 (25 Oct), robot A: no_progress at 71.0, 118.0 and 148.0 s while the route to its goal shortened by about 4 m
            // per 15 s over floor it had already mapped; each firing cleared the path and sent it back into the kitchen. Control: with
            // progress_route_shrink off the same walk trips the watchdog, as before.
            int NoProgress(bool shrink)
            {
                var c = Settings(); c.grid_size_m = 30; c.robot_radius_m = .25f; c.progress_route_shrink = shrink;
                var b = new RobotBrain(Calibration(), c, "A", 0, new EntryPose(.05f, -1.5f, 0)); Room(b, -4, 4, -3, 14);
                var route = new GridPlanner(Map(b), c).Plan(new Vector2(.05f, -1.5f), new Vector2(.05f, 13.05f));
                Assert.That(route.Count, Is.GreaterThan(120), "a 14.5 m route exists");
                typeof(RobotBrain).GetMethod("SetPath", NP).Invoke(b, new object[] { route, true });
                int count = 0;
                for (int k = 0; k <= 15 * 25; k++)
                {
                    double t = k / 15.0; Motion(b, t, .3f); b.OnRgbd(Image(t, true, 0)); b.OnAirTemperature(t, 20); b.Step(t, 1f / 15);
                    count += b.DrainEvents().Count(e => e.kind == "no_progress");
                }
                return count;
            }
            Assert.That(NoProgress(true), Is.Zero, "7.5 m closer to the same goal in 25 s is progress");
            Assert.That(NoProgress(false), Is.GreaterThan(0), "control: without the route measure the watchdog fires");
        }

        [Test]
        public void AFrontierChosenEarlierIsKeptUnlessTheNewBestClearlyBeatsIt()
        {
            // F1. Seed 934 (25 Oct): robot A's goal flipped 7 times between a near kitchen frontier and a far hall frontier from 55.8 to
            // 134.0 s; (-8.45, 8.95) was chosen 8 times in 7 s at 451 to 457 s.
            GridPlanner.FrontierChoice F(float x, int tier, float score) => new GridPlanner.FrontierChoice
            { position = new Vector2(x, 0), roomTier = tier, roomScore = score, path = new List<Vector2>() };
            GridPlanner.FrontierChoice Pick(bool commit, double now, List<GridPlanner.FrontierChoice> list)
            {
                var c = Settings(); c.grid_size_m = 12; c.goal_commitment = commit;
                var b = new RobotBrain(Calibration(), c, "A", 0);
                typeof(RobotBrain).GetField("committedAt", NP).SetValue(b, (Vector2?)new Vector2(5, 0));
                typeof(RobotBrain).GetField("committedSince", NP).SetValue(b, 0.0);
                return (GridPlanner.FrontierChoice)typeof(RobotBrain).GetMethod("CommittedFrontier", NP).Invoke(b, new object[] { list, now });
            }
            var near = F(-5, 4, 50f); var kept = F(5, 4, 57f);
            Assert.That(Pick(true, 10, new List<GridPlanner.FrontierChoice> { near, kept }), Is.SameAs(kept), "7 s ahead is inside the 12 s noise: keep");
            Assert.That(Pick(false, 10, new List<GridPlanner.FrontierChoice> { near, kept }), Is.SameAs(near), "control: switch off, best wins");
            Assert.That(Pick(true, 10, new List<GridPlanner.FrontierChoice> { F(-5, 4, 40f), kept }).position.x, Is.EqualTo(-5), "17 s ahead beats it");
            Assert.That(Pick(true, 10, new List<GridPlanner.FrontierChoice> { F(-5, 2, 90f), kept }).position.x, Is.EqualTo(-5), "a better room tier beats it");
            Assert.That(Pick(true, 10, new List<GridPlanner.FrontierChoice> { near }), Is.SameAs(near), "the kept frontier is no longer admitted");
            Assert.That(Pick(true, 100, new List<GridPlanner.FrontierChoice> { near, kept }), Is.SameAs(near), "a commitment expires after goal_commit_max_s");
        }
    }
}
