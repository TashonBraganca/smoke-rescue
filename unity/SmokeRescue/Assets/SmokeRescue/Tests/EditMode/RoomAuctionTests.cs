using System.Linq;
using NUnit.Framework;
using SmokeRescue.Brain;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    public sealed partial class BrainTests
    {
        const System.Reflection.BindingFlags Private = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance;

        static BrainConfig AuctionSettings()
        {
            var c = Settings(); c.grid_size_m = 16; c.room_ledger = true; c.room_auction = true;
            c.frontier_peek_m = 0; c.search_requires_ceiling = false; c.robot_radius_m = .25f;
            return c;
        }

        /// <summary>A synthetic layout, not the house: a corridor 12 m long along x, 2 * half wide, with 3.6 m square rooms
        /// above it at the given x centres, each through a 0.9 m door. Walls are occupied cells; the rest is unknown. No floor is camera-searched yet.</summary>
        static void CorridorWithRooms(OccupancyGrid g, float half, params float[] roomX)
        {
            float wall = half + .1f, top = wall + 3.6f;
            for (int i = 0; i < g.Count; i++)
            {
                var p = g.Centre(i);
                if (Mathf.Abs(p.x) > 6.1f || p.y < -half - .1f || p.y > top + .1f) continue;
                bool corridor = Mathf.Abs(p.x) < 6 && Mathf.Abs(p.y) < half;
                bool inRoom = false, door = false;
                foreach (float x in roomX)
                {
                    if (Mathf.Abs(p.x - x) < 1.75f && p.y >= wall && p.y < top) inRoom = true;
                    if (Mathf.Abs(p.x - x) < .45f && p.y >= half && p.y < wall) door = true;
                }
                if (corridor || inRoom || door) g.ObserveFree(i);
                else if (p.y < top + .1f) g.ObserveOccupied(i);
            }
            // Mapped by the LiDAR, not yet searched by the camera: every room still has floor to search.
            var searched = (bool[])typeof(OccupancyGrid).GetField("observed", Private).GetValue(g);
            System.Array.Clear(searched, 0, searched.Length);
        }

        static GridPlanner Planner(RobotBrain b) => (GridPlanner)typeof(RobotBrain).GetField("planner", Private).GetValue(b);

        [Test]
        public void ACorridorIsALongNarrowRegionWithSeveralRoomsOffIt()
        {
            (RoomLedger ledger, OccupancyGrid g) Build(float half, params float[] rooms)
            {
                var c = AuctionSettings(); var g = new OccupancyGrid(c); CorridorWithRooms(g, half, rooms);
                var ledger = new RoomLedger(g, c); ledger.Update(0, new Vector2(0, 0)); return (ledger, g);
            }
            var (three, _) = Build(.8f, -4, 0, 4);
            var hall = three.Get(three.RoomAt(Vector2.zero));
            Assert.That(hall.widthM, Is.EqualTo(1.6f).Within(.21f), "width is measured from the robot's own map");
            Assert.That(hall.openings, Is.EqualTo(3));
            Assert.That(hall.corridor, Is.True, "12 x 1.6 m with three rooms off it");
            foreach (float x in new[] { -4f, 0f, 4f })
            {
                var room = three.Get(three.RoomAt(new Vector2(x, 2.7f)));
                Assert.That(room.id, Is.Not.EqualTo(hall.id), "each room is its own region behind its door");
                Assert.That(room.corridor, Is.False, "a 3.6 m square room with one door is a room");
                Assert.That(three.RoomAt(room.anchor), Is.EqualTo(room.id), "the anchor lies inside its own room");
            }
            var (two, _) = Build(.8f, -4, 4);
            Assert.That(two.Get(two.RoomAt(Vector2.zero)).corridor, Is.False, "control: two openings make a passage, not a hub");
            var (wide, _) = Build(1.8f, -4, 0, 4);
            Assert.That(wide.Get(wide.RoomAt(Vector2.zero)).corridor, Is.False, "control: 3.6 m is wider than any code corridor");
        }

        [Test]
        public void TheRoomAuctionGivesEachRoomToTheLowerBidForAnyNumberOfRobotsAndClaimsLapse()
        {
            var c = AuctionSettings();
            var b = new RobotBrain(Calibration(), c, "robot_2", 0);
            CorridorWithRooms(Map(b), .8f, -4, 0, 4);
            var planner = Planner(b); planner.Rooms.Update(0, Vector2.zero);
            int west = planner.Rooms.RoomAt(new Vector2(-4, 2.7f)), middle = planner.Rooms.RoomAt(new Vector2(0, 2.7f)),
                east = planner.Rooms.RoomAt(new Vector2(4, 2.7f)), hall = planner.Rooms.RoomAt(Vector2.zero);
            bool Offered(int room) => planner.Frontiers(Vector2.zero).Any(f => f.roomId == room);
            float MyBid(int room) => planner.Frontiers(Vector2.zero).Where(f => f.roomId == room).Min(f => f.cost) / c.open_speed_mps;
            Assert.That(Offered(west) && Offered(middle) && Offered(east) && Offered(hall), Is.True, "control: every room is offered before any claim; rooms " + string.Join(",", new[] { west, middle, east, hall }) + " offered " + string.Join(",", planner.Frontiers(Vector2.zero).Select(f => f.roomId + "@" + f.position)));
            float westBid = MyBid(west); long sequence = 0;
            void Claim(string robot, Vector2 anchor, float bid, double sent)
            {
                var m = new RobotBrain(Calibration(), c, robot, 0).BuildMessage(sent); m.Sequence = ++sequence;
                m.RoomClaim = new BrainRoomClaim { anchor = anchor, radiusM = 2, bidS = bid, since = sent, expiresAt = sent + c.peer_claim_s };
                b.OnPeerMessage(Wire(m), sent + .1);
            }
            Claim("robot_1", new Vector2(-4, 2.7f), westBid + c.room_auction_margin_s + 5, 0);
            Assert.That(Offered(west), Is.True, "a partner bidding more than this robot plus the margin does not take the room");
            Claim("robot_1", new Vector2(-4, 2.7f), westBid - c.room_auction_margin_s - .5f, .2);
            Assert.That(Offered(west), Is.False, "a lower bid takes the room");
            Claim("robot_3", new Vector2(4, 2.7f), westBid + 1, .4);
            Assert.That(Offered(east), Is.False, "within the margin the robot that took the room first keeps it");
            Assert.That(Offered(west), Is.False, "a third robot's claim does not undo the first one's");
            Assert.That(Offered(middle), Is.True, "an unclaimed room stays open with three robots");
            Claim("robot_3", new Vector2(0, 0), 0, .6);
            Assert.That(Offered(hall), Is.True, "a corridor is never taken, whatever the bid");
            Claim("robot_3", new Vector2(4, 2.7f), 0, c.peer_claim_s + 1);
            Assert.That(Offered(west), Is.True, "robot_1 went quiet: its claim lapsed peer_claim_s after it was sent");
            Assert.That(Offered(east), Is.False, "robot_3's fresh claim still holds");
            c.room_auction = false;
            Assert.That(Offered(east), Is.True, "control: with the auction off the room is offered");
        }

        [Test]
        public void AHolderRadiosItsRoomAndYieldsItOnceToALowerBid()
        {
            var c = AuctionSettings();
            var b = new RobotBrain(Calibration(), c, "robot_2", 0);
            CorridorWithRooms(Map(b), .8f, -4, 0, 4);
            var planner = Planner(b); planner.Rooms.Update(0, Vector2.zero);
            int west = planner.Rooms.RoomAt(new Vector2(-4, 2.7f));
            void Hold(double since)
            {
                typeof(RobotBrain).GetField("claimedRoom", Private).SetValue(b, west);
                typeof(RobotBrain).GetField("claimedSince", Private).SetValue(b, since);
                typeof(RobotBrain).GetField("claimedBid", Private).SetValue(b, 8f);
                typeof(RobotBrain).GetField("frontierGoal", Private).SetValue(b, true);
                typeof(RobotBrain).GetField("frontierRadius", Private).SetValue(b, 2.5f);
                typeof(RobotBrain).GetField("frontierCentre", Private).SetValue(b, new Vector2(-4, 2.7f));
                typeof(RobotBrain).GetField("goal", Private).SetValue(b, (Vector2?)new Vector2(-4, 2.7f));
            }
            Hold(0);
            var sent = Wire(b.BuildMessage(0));
            Assert.That(sent.RoomClaim, Is.Not.Null, "the held room travels on the radio");
            Assert.That(planner.Rooms.RoomAt(sent.RoomClaim.anchor), Is.EqualTo(west));
            Assert.That(sent.RoomClaim.bidS, Is.EqualTo(8f));
            long sequence = 0;
            void Partner(float bid, double t)
            {
                var m = new RobotBrain(Calibration(), c, "robot_1", 0).BuildMessage(t); m.Sequence = ++sequence;
                m.RoomClaim = new BrainRoomClaim { anchor = new Vector2(-4, 2.7f), radiusM = 2, bidS = bid, since = t, expiresAt = t + c.peer_claim_s };
                b.OnPeerMessage(Wire(m), t + .1);
            }
            b.DrainEvents();
            Partner(7, 1);
            Assert.That(b.DrainEvents().Count(e => e.kind == "room_yielded"), Is.Zero, "control: a bid within the margin taken later does not win");
            Partner(1, 2);
            Assert.That(b.DrainEvents().Count(e => e.kind == "room_yielded"), Is.EqualTo(1), "a lower bid takes the room");
            Assert.That(typeof(RobotBrain).GetField("claimedRoom", Private).GetValue(b), Is.EqualTo(-1));
            Partner(1, 3);
            Assert.That(b.DrainEvents().Count(e => e.kind == "room_yielded"), Is.Zero, "yielded once, not on every message");
            var bad = new RobotBrain(Calibration(), c, "robot_1", 0).BuildMessage(4); bad.Sequence = ++sequence;
            bad.RoomClaim = new BrainRoomClaim { anchor = Vector2.zero, radiusM = 2, bidS = -1, since = 4, expiresAt = 4 + c.peer_claim_s };
            b.OnPeerPacket(bad.ToBytes(), 4.1);
            Assert.That(b.DrainEvents().Any(e => e.kind == "peer_message_rejected"), Is.True, "a negative bid is dropped at the radio boundary");
        }
    }
}
