using System.Collections.Generic;
using NUnit.Framework;
using SmokeRescue.Brain;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    public sealed partial class BrainTests
    {
        /// <summary>A synthetic layout, not the house: two 4 x 4 m rooms side by side along x, joined by an opening of the
        /// given width in their shared wall, plus a 0.5 x 0.6 m free strip walled off on its own. Walls are occupied cells.</summary>
        static void TwoRoomsWithOpening(OccupancyGrid g, float opening)
        {
            for (int i = 0; i < g.Count; i++)
            {
                var p = g.Centre(i);
                if (p.x < -4.2f || p.x > 4.2f || p.y < -2.2f || p.y > 3.4f) continue;
                bool roomA = p.x > -4 && p.x < -.1f && Mathf.Abs(p.y) < 2, roomB = p.x > .1f && p.x < 4 && Mathf.Abs(p.y) < 2;
                bool door = Mathf.Abs(p.x) <= .1f && Mathf.Abs(p.y) < opening / 2;
                bool strip = p.x > 1 && p.x < 1.5f && p.y > 2.5f && p.y < 3.1f;
                if (roomA || roomB || door || strip) g.ObserveFree(i); else g.ObserveOccupied(i);
            }
        }

        [Test]
        public void ADoorIsANarrowingBetweenTwoWiderSpacesWhateverItsWidthAndSlithersAreNoRooms()
        {
            // Seed 934: openings of 1.2 to 1.6 m (at or above the fixed 1.2 m door) merged foyer, living room and hallway into
            // one room; 126 of robot A's 133 rooms were under 1 m2.
            int Rooms(bool relative, float opening, out int strip, bool slivers = false)
            {
                var c = AuctionSettings(); c.room_ledger_relative_doors = relative; c.room_ledger_drop_slivers = slivers;
                var b = new RobotBrain(Calibration(), c, "robot_A", 0); TwoRoomsWithOpening(Map(b), opening);
                var ledger = Planner(b).Rooms; ledger.Update(0, new Vector2(-2, 0), p => true);
                strip = ledger.RoomAt(new Vector2(1.25f, 2.8f));
                int left = ledger.RoomAt(new Vector2(-2, 0)), right = ledger.RoomAt(new Vector2(2, 0));
                Assert.That(left, Is.GreaterThanOrEqualTo(0)); Assert.That(right, Is.GreaterThanOrEqualTo(0));
                return left == right ? 1 : 2;
            }
            Assert.That(Rooms(false, 1.6f, out int oldStrip), Is.EqualTo(1), "control: the fixed 1.2 m door merges rooms joined by 1.6 m");
            Assert.That(oldStrip, Is.GreaterThanOrEqualTo(0), "control: the fixed rule gives a sliver its own room");
            Assert.That(Rooms(true, 1.6f, out int strip), Is.EqualTo(2), "relative doors: a 1.6 m opening between two 4 m rooms is a door");
            Assert.That(Rooms(true, .9f, out _), Is.EqualTo(2), "and so is an ordinary 0.9 m door");
            Assert.That(strip, Is.LessThan(0), "a 0.3 m2 sliver is no room");
            Assert.That(Rooms(true, 3.8f, out _), Is.EqualTo(1), "an opening nearly as wide as the rooms joins them into one space");
            Assert.That(Rooms(false, 1.6f, out int fixedStrip, slivers: true), Is.EqualTo(1), "drop_slivers alone keeps the fixed door rule");
            Assert.That(fixedStrip, Is.LessThan(0), "but the sliver is no room");
        }

        [Test]
        public void InARoomTheRobotWorksFromTheLeftSideAcrossToTheRightAsSeenFromWhereItCameIn()
        {
            // Owner (23 Oct): enter a room, start from its left side, cover it, and leave on the right.
            List<int> Order(bool sweep)
            {
                var c = AuctionSettings(); c.room_sweep = sweep;
                var b = new RobotBrain(Calibration(), c, "robot_A", 0); Room(b, -5, 5, -5, 5);
                var g = Map(b); var searched = (bool[])typeof(OccupancyGrid).GetField("observed", Private).GetValue(g);
                System.Array.Clear(searched, 0, searched.Length);   // nothing camera-searched yet: the room is being finished
                var ledger = Planner(b).Rooms; ledger.Update(0, new Vector2(0, -4), p => true);   // came in at the south side
                ledger.Update(1, new Vector2(0, -3), p => true);
                var right = new GridPlanner.FrontierChoice { position = new Vector2(3, 2), cost = 5, score = 5 };
                var left = new GridPlanner.FrontierChoice { position = new Vector2(-3, 2), cost = 5, score = 5 };
                var choices = new List<GridPlanner.FrontierChoice> { right, left };
                ledger.Rank(choices, new Vector2(0, -3));
                Assert.That(choices[0].roomTier, Is.EqualTo(0), "control: both lie in the room being finished");
                return choices.ConvertAll(f => f == left ? 0 : 1);
            }
            Assert.That(Order(false), Is.EqualTo(new List<int> { 1, 0 }), "control: without the sweep, equal frontiers keep their order");
            Assert.That(Order(true), Is.EqualTo(new List<int> { 0, 1 }), "with it, the left side comes first");
        }

        [Test]
        public void ALowBedIsNoWallForAWarmShapeButATallWallIs()
        {
            // Seed 934: the kitchen person curled on a bed was flagged on-wall, since the 2 m bed made a long occupied structure.
            var c = AuctionSettings(); c.candidate_wall_tall_only = true;
            var b = new RobotBrain(Calibration(), c, "robot_A", 0); Room(b, -6, 6, -6, 6); var g = Map(b);
            var top = new ushort[g.Count];
            for (int i = 0; i < g.Count; i++)
            {
                var p = g.Centre(i);
                if (p.x >= -1.1f && p.x <= 1.1f && p.y >= 2 && p.y <= 2.4f) { g.ObserveOccupied(i); top[i] = 50; }   // a bed 2.2 m long, 0.5 m high
                if (p.x >= -1.1f && p.x <= 1.1f && p.y >= -2.4f && p.y <= -2) { g.ObserveOccupied(i); top[i] = 270; } // a wall, 2.7 m high
            }
            g.SetTopHeights(top); g.SeeOverMaxM = 1.0f;
            float Extent(Vector2 p, bool tall) => g.OccupiedExtent(p, c.candidate_wall_disc_m, c.candidate_wall_window_m, tall);
            Assert.That(Extent(new Vector2(0, 2.5f), false), Is.GreaterThanOrEqualTo(c.candidate_wall_extent_m), "control: counting every occupied cell, the bed reads as a wall");
            Assert.That(Extent(new Vector2(0, 2.5f), true), Is.LessThan(c.candidate_wall_extent_m), "low furniture the camera sees over is no wall");
            Assert.That(Extent(new Vector2(0, -1.9f), true), Is.GreaterThanOrEqualTo(c.candidate_wall_extent_m), "a tall wall still is");
        }

        [Test]
        public void ARoomSearchedFromItsDoorwayIsStillWalkedIntoOnce()
        {
            // Headless (23 Oct): 17 of 55 rooms were never entered; floor seen from the door counted as searched.
            var c = AuctionSettings(); c.room_walk_in = true; c.room_walk_in_max_m = 12;
            var b = new RobotBrain(Calibration(), c, "robot_A", 0); var g = Map(b);
            CorridorWithRooms(g, .8f, -4, 4);
            var searched = (bool[])typeof(OccupancyGrid).GetField("observed", Private).GetValue(g);
            for (int i = 0; i < g.Count; i++) { var p = g.Centre(i); if (p.y > .9f) searched[i] = true; }   // both rooms searched from the corridor
            var ledger = Planner(b).Rooms; ledger.Update(0, Vector2.zero, p => true);
            var walk = typeof(RobotBrain).GetMethod("WalkIntoRoom", Private);
            var args = new object[] { 1.0, null, -1 };
            Assert.That((bool)walk.Invoke(b, args), Is.True, "a searched room it never stood in is walked into");
            var route = (List<Vector2>)args[1];
            Assert.That(ledger.RoomAt(route[route.Count - 1]), Is.EqualTo((int)args[2]), "the walk ends inside that room");
            int first = (int)args[2];
            var again = new object[] { 2.0, null, -1 };
            bool second = (bool)walk.Invoke(b, again);
            Assert.That(!second || (int)again[2] != first, Is.True, "never the same room twice");
        }
    }
}
