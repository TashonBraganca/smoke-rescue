using System;
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

        [Test]
        public void AVisitedCircleHidingTheOnlyFrontierIsDroppedOnceItIsOldEnough()
        {
            // F2. Seed 934 (25 Oct), robot B: the bathroom door frontier reached at about 200 s stayed hidden by visited_hold_s (90 s);
            // B reported no frontier at 233.9 s, held still 245.9 to 294.0 s, and the frontier came back at 294.9 s.
            bool Retry(bool on, double circleAge, out int circlesLeft)
            {
                var c = Settings(); c.grid_size_m = 12; c.robot_radius_m = .25f; c.visited_retry = on;
                var b = new RobotBrain(Calibration(), c, "A", 0, new EntryPose(.05f, .05f, 0)); Room(b, -4, 4, -4, 4);
                var planner = (GridPlanner)typeof(RobotBrain).GetField("planner", NP).GetValue(b);
                Assert.That(planner.Frontier(new Vector2(.05f, .05f)), Is.Not.Null, "fixture: the open room has a frontier");
                var type = typeof(RobotBrain).GetNestedType("Visited", System.Reflection.BindingFlags.NonPublic);
                var list = (System.Collections.IList)typeof(RobotBrain).GetField("visited", NP).GetValue(b);
                double now = 200;
                object v = System.Activator.CreateInstance(type);
                type.GetField("position").SetValue(v, new Vector2(0, 0)); type.GetField("radius").SetValue(v, 50f);
                type.GetField("until").SetValue(v, now - circleAge + c.visited_hold_s);
                list.Add(v);
                var hidden = typeof(RobotBrain).GetMethod("VisitedGoal", NP);
                Assert.That(planner.Frontier(new Vector2(.05f, .05f), p => (bool)hidden.Invoke(b, new object[] { p, now })), Is.Null, "the circle hides every frontier");
                bool result = (bool)typeof(RobotBrain).GetMethod("RetryVisited", NP).Invoke(b, new object[] { now });
                circlesLeft = list.Count; return result;
            }
            Assert.That(Retry(true, 46, out int left), Is.True, "a 46 s old circle over the only frontier is dropped"); Assert.That(left, Is.Zero);
            Assert.That(Retry(true, 5, out left), Is.False, "a circle 5 s old stays: the spot was just seen"); Assert.That(left, Is.EqualTo(1));
            Assert.That(Retry(false, 46, out left), Is.False, "control: switch off"); Assert.That(left, Is.EqualTo(1));
        }

        static readonly RobotBrain.NozzleSpec TestNozzle = new RobotBrain.NozzleSpec { flowLps = 1, coolingKjPerL = 2590, efficiency = .5f, halfAngleDeg = 12, rangeM = 5,
            maxSprayS = 6, burningKwPerM2 = 324, burningSurfaceC = 600, knockdownC = 300, fuelKjPerM2K = 6.5f };
        // The burning corridor of TheRobotSpraysAMeasuredFireBlockingItsWayByItself: fire across the only way on at y = 2, walled behind.
        static RobotBrain BurningCorridor(Action<BrainConfig> tweak)
        {
            var s = Settings(); s.code_sprays_fire = true; s.code_spray_reach_m = 1f; tweak(s);
            var b = new RobotBrain(Calibration(), s, "A", 0); var g = Map(b);
            for (float y = -1; y <= 4; y += .05f)
            {
                for (float x = -.95f; x <= .95f; x += .05f) g.ObserveFree(g.Cell(new Vector2(x, y)));
                g.ObserveOccupied(g.Cell(new Vector2(-1.05f, y))); g.ObserveOccupied(g.Cell(new Vector2(1.05f, y)));
            }
            for (float x = -1; x <= 1; x += .05f) g.ObserveOccupied(g.Cell(new Vector2(x, -1.05f)));
            for (float x = -.95f; x <= .95f; x += .1f) g.ObserveHeat(g.Cell(new Vector2(x, 2)), 600, 0);
            b.Nozzle = TestNozzle; return b;
        }

        static void Hot(RobotBrain b, double now = 0)
        {
            typeof(RobotBrain).GetField("noFrontierAt", NP).SetValue(b, 0.0);
            typeof(RobotBrain).GetField("airC", NP).SetValue(b, 45f); typeof(RobotBrain).GetField("lastAir", NP).SetValue(b, now);
        }

        [Test]
        public void ASpraySpotTheRobotCannotWalkToIsNotOrderedAgainEverySecond()
        {
            // F4. Seed 934 (25 Oct), robot B in the bathroom 396 to 464 s: 112 code_spray_approach events, 49 go_to orders to the door
            // cell, every one clearing its path, 22 "no known free path", 41 refusals, still 404 to 457 s. The corridor fire is 2 m
            // ahead with a 1 m reach, so the robot must walk to its spray spot; here every order is refused (the heat refusal meant for
            // agents, kept on by the switch off), the robot's own air reads 45 C. Control: retry 0 orders it every replan_s as before.
            (int refusals, int noted) Run(float retry)
            {
                var b = BurningCorridor(c => { c.code_spray_ignores_leave_refusal = false; c.code_spray_retry_s = retry; });
                Hot(b);
                int refusals = 0, noted = 0;
                for (int t = 0; t <= 61; t++)
                {
                    Hot(b, t); b.CodeSpray(t, 12, out _, out _);
                    foreach (var e in b.DrainEvents())
                    {
                        if (e.kind == "agent_order" && e.reason.StartsWith("Refused go_to")) refusals++;
                        if (e.kind == "code_spray_spot_unreachable") noted++;
                    }
                }
                return (refusals, noted);
            }
            var withMemory = Run(30);
            Assert.That(withMemory.refusals, Is.InRange(1, 3), "a refused spot is retried after 30 s, not every second");
            Assert.That(withMemory.noted, Is.InRange(1, 3));
            Assert.That(Run(0).refusals, Is.GreaterThan(30), "control: before the fix it was ordered about once a second");
        }

        [Test]
        public void ASpraySpotWithNoFreePathIsRememberedAsUnreachable()
        {
            // F4. Seed 934 (25 Oct), robot B: 22 "Agent goal has no known free path; back to exploring." for the same spray spot.
            var b = BurningCorridor(c => { c.code_spray_retry_s = 30; });
            var spot = new Vector2(0, 3.4f);   // beyond the fire across the only way on: no route inside the fire margin
            typeof(RobotBrain).GetField("agentGoal", NP).SetValue(b, (Vector2?)spot);
            typeof(RobotBrain).GetField("agentGoalUntil", NP).SetValue(b, 60.0);
            typeof(RobotBrain).GetField("spraySpotGoal", NP).SetValue(b, (Vector2?)spot);
            var events = new List<BrainEvent>();
            for (int i = 0; i < 60; i++) { DriveTick(b, i); events.AddRange(b.DrainEvents()); }
            Assert.That(events.Any(e => e.kind == "replan" && e.reason.StartsWith("Agent goal has no known free path")), Is.True, "fixture: the spot has no path");
            Assert.That(events.Count(e => e.kind == "code_spray_spot_unreachable"), Is.EqualTo(1));
        }

        [Test]
        public void ACodeSprayApproachIsNotRefusedBecauseTheRobotIsSaidToBeLeaving()
        {
            // F4. Seed 934 (25 Oct), robot B: 41 "Refused go_to: no reachable frontier is left and the air is heating ...; the robot is
            // leaving" for spray approaches (411 to 464 s) although B was not returning and could not leave.
            bool Ordered(bool ignore)
            {
                var b = BurningCorridor(c => { c.code_spray_ignores_leave_refusal = ignore; });
                Hot(b);
                b.CodeSpray(0, 12, out _, out _);
                var events = b.DrainEvents();
                Assert.That(events.Any(e => e.kind == "agent_order"), Is.True, "an order or a refusal is logged");
                return events.Any(e => e.kind == "agent_order" && e.reason.StartsWith("go_to: walk to the spray spot"));
            }
            Assert.That(Ordered(true), Is.True, "the reflex walks to its spray spot");
            Assert.That(Ordered(false), Is.False, "control: with the switch off it is refused as 'the robot is leaving'");
        }

        [Test]
        public void AStallBesideAConfirmedPersonBlacklistsTheGoalAtOnce()
        {
            // F3. Seed 934 (25 Oct), robot A: stalls at 238.3, 243.3 and 248.6 s at (8.8, 13.1), 0.80 to 0.92 m from person s1's body
            // (a lying body reaches 0.85 m, below the LiDAR plane), 1.7 m from the believed position; the goal was blacklisted only at
            // the third stall, 13 s later (stuck_repeat_count 3). Control: the same stall with the person 6 m away needs three.
            int FirstBlacklistRepeats(Vector2 person, bool on)
            {
                var b = BlockedBrain(out var c); c.stall_near_person_blacklists = on;
                ((List<Vector2>)typeof(RobotBrain).GetField("survivors", NP).GetValue(b)).Add(person);
                var command = new BrainCommand();
                for (int tick = 0; tick < 1000; tick++)
                {
                    command = DriveTick(b, tick, command.forwardMps < 0 ? command.forwardMps : 0, command.yawRateDeg);
                    var e = b.DrainEvents().FirstOrDefault(x => x.kind == "goal_blacklisted");
                    if (e != null) return (int)e.numbers["repeats"];
                }
                return -1;
            }
            Assert.That(FirstBlacklistRepeats(new Vector2(1.5f, 1.2f), true), Is.EqualTo(1), "a stall beside a confirmed person blacklists at the first stall");
            Assert.That(FirstBlacklistRepeats(new Vector2(1.5f, 1.2f), false), Is.EqualTo(3), "control: switch off, three stalls");
            Assert.That(FirstBlacklistRepeats(new Vector2(0, 5.5f), true), Is.EqualTo(3), "control: a person 5.5 m away changes nothing");
        }

        [Test]
        public void OnlyThePersonWhoSealsTheFloorGetsTheSmallKeepOutCore()
        {
            // F3. Seed 934 (25 Oct): robot A squeezed past the kitchen person s4 at 50.2 s; the flag stayed on for every person, and at
            // 238 to 249 s A drove into the lying body of s1 (core 0.45 m shrunk to 0.23 m).
            RobotBrain Brain(bool perPerson)
            {
                var s = Settings(); s.grid_size_m = 16; s.survivor_squeeze = true; s.squeeze_per_person = perPerson; s.robot_radius_m = .25f; s.search_requires_ceiling = false;
                var b = new RobotBrain(Calibration(), s, "A", 0); var g = Map(b);
                for (float y = -3; y <= 0; y += .05f) for (float x = -4; x <= 5; x += .05f) g.ObserveFree(g.Cell(new Vector2(x, y)));
                for (float y = 0; y <= 5; y += .05f)
                {
                    for (float x = -.9f; x <= .9f; x += .05f) g.ObserveFree(g.Cell(new Vector2(x, y)));
                    g.ObserveOccupied(g.Cell(new Vector2(-1f, y))); g.ObserveOccupied(g.Cell(new Vector2(1f, y)));
                }
                var searched = (bool[])typeof(OccupancyGrid).GetField("observed", NP).GetValue(g);
                for (int i = 0; i < g.Count; i++) searched[i] = g.Centre(i).y < 1.5f;   // the corridor beyond the person is unsearched
                var people = (List<Vector2>)typeof(RobotBrain).GetField("survivors", NP).GetValue(b);
                people.Add(new Vector2(3, -1.5f));    // in the open room
                people.Add(new Vector2(0, 1.5f));     // lying across the 1.8 m corridor: seals it with the full core
                typeof(RobotBrain).GetField("keptOut", NP).SetValue(b, -1);
                typeof(RobotBrain).GetMethod("ApplyKeepOut", NP).Invoke(b, null);
                return b;
            }
            bool Keep(RobotBrain b, Vector2 at) => ((bool[])typeof(OccupancyGrid).GetField("keepOut", NP).GetValue(Map(b)))[Map(b).Cell(at)];
            foreach (bool per in new[] { true, false })
            {
                var b = Brain(per);
                Assert.That(Keep(b, new Vector2(0, 1.8f)), Is.True, "fixture: the full core 0.3 m from the corridor person is an obstacle");
                Assert.That((bool)typeof(RobotBrain).GetMethod("SqueezePastPerson", NP).Invoke(b, new object[] { 0.0 }), Is.True, "fixture: floor is cut off, squeeze");
                typeof(RobotBrain).GetMethod("ApplyKeepOut", NP).Invoke(b, null);
                Assert.That(Keep(b, new Vector2(0, 1.8f)), Is.False, "the sealing person's core shrinks");
                Assert.That(Keep(b, new Vector2(3.3f, -1.5f)), Is.EqualTo(per), per ? "the person in the open room keeps the full core" : "control: without squeeze_per_person it shrinks too");
            }
        }

        // A 4 x 6 m room, all of it searched floor, with fire at (fireX, fireY) and this robot at the origin walking a route north along x = 0.
        static RobotBrain RoomWithFire(Action<BrainConfig> tweak, float fireX, float fireY, bool unsearched = false, float half = .1f)
        {
            var s = Settings(); s.code_sprays_fire = true; s.code_spray_goal_room = true; s.room_ledger = true; tweak(s);
            var b = new RobotBrain(Calibration(), s, "A", 0); var g = Map(b);
            for (float y = -1; y <= 5; y += .05f)
            {
                for (float x = -2f; x <= 2f; x += .05f) g.ObserveFree(g.Cell(new Vector2(x, y)));
                g.ObserveOccupied(g.Cell(new Vector2(-2.1f, y))); g.ObserveOccupied(g.Cell(new Vector2(2.1f, y)));
            }
            for (float x = -2.1f; x <= 2.1f; x += .05f) { g.ObserveOccupied(g.Cell(new Vector2(x, -1.1f))); g.ObserveOccupied(g.Cell(new Vector2(x, 5.1f))); }
            var searched = (bool[])typeof(OccupancyGrid).GetField("observed", NP).GetValue(g);
            for (int i = 0; i < g.Count; i++) searched[i] = !unsearched || g.Centre(i).y < 0;
            for (float dx = -half; dx <= half + 1e-3f; dx += .1f) for (float dy = -half; dy <= half + 1e-3f; dy += .1f) g.ObserveHeat(g.Cell(new Vector2(fireX + dx, fireY + dy)), 600, 0);
            b.Nozzle = TestNozzle;
            ((GridPlanner)typeof(RobotBrain).GetField("planner", NP).GetValue(b)).Rooms.Update(0, Vector2.zero);
            var route = (List<Vector2>)typeof(RobotBrain).GetField("path", NP).GetValue(b);
            route.Clear(); for (float y = 0; y <= 4.5f; y += .1f) route.Add(new Vector2(0, y));   // a planned route is dense
            return b;
        }

        [Test]
        public void WaterGoesToFireOnTheRouteNotToAnyFireInTheGoalRoomAndKeepsAReserve()
        {
            // F5. Seed 934 (25 Oct), robot A: all 12 L at 371.1, 385.1 and 403.6 s on 2.5, 1.5 and 0.6 m2 of fire about 3 m from its route,
            // in the same merged ledger room as its goal; A had no water when the exit closed. Fire 1.8 m beside the route, in a room that
            // is already searched: not sprayed with code_spray_route_fire (control: the old goal-room trigger sprays it).
            Assert.That(RoomWithFire(c => c.code_spray_route_fire = false, 1.8f, .5f).CodeSpray(0, 12, out _, out _), Is.True, "control: the goal-room trigger sprays it");
            Assert.That(RoomWithFire(c => c.code_spray_route_fire = true, 1.8f, .5f).CodeSpray(0, 12, out _, out _), Is.False, "fire off the route on searched floor: keep the water");
            // The same fire on the route (0.6 m from it): sprayed.
            Assert.That(RoomWithFire(c => c.code_spray_route_fire = true, .6f, 2.4f).CodeSpray(0, 12, out float bearing, out _), Is.True, "fire within the margin of the route: spray it");
            Assert.That(Mathf.Abs(bearing), Is.LessThan(60f));
            // Fire on unsearched floor, off the route: sprayed. Control: the same fire on searched floor was not.
            Assert.That(RoomWithFire(c => c.code_spray_route_fire = true, 1.8f, .5f, true).CodeSpray(0, 12, out _, out _), Is.True, "fire burning on unsearched floor: spray it");
            // The reserve: 7 L aboard and a 6 L reserve leave 1 s for any spray that is not on the way home; this 1.7 m2 fire needs about 4.5 s.
            Assert.That(RoomWithFire(c => { c.code_spray_route_fire = true; c.code_spray_reserve_l = 6; }, .6f, 2.4f, false, .65f).CodeSpray(0, 7, out _, out _), Is.False, "7 L with a 6 L reserve: not for this fire");
            Assert.That(RoomWithFire(c => { c.code_spray_route_fire = true; c.code_spray_reserve_l = 0; }, .6f, 2.4f, false, .65f).CodeSpray(0, 7, out _, out _), Is.True, "control: no reserve, 7 L is enough");
            Assert.That(RoomWithFire(c => { c.code_spray_route_fire = true; c.code_spray_reserve_l = 6; }, .6f, 2.4f, false, .65f).CodeSpray(0, 12, out _, out _), Is.True, "12 L with the reserve still leaves enough");
        }
    }
}
