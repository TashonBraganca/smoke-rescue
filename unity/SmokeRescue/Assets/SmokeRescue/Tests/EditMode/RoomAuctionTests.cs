using System;
using System.Collections.Generic;
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
        public void StandingInACorridorItsFrontiersComeBeforePocketsInEnteredRoomsButAfterUnenteredRooms()
        {
            // Headless seed 5001: the corridor ranked last whenever a robot stood in it, so both robots went back to pockets in
            // rooms they had entered and never walked far enough up the corridor to see the kitchen door.
            var c = AuctionSettings(); c.room_auction = false; var g = new OccupancyGrid(c); CorridorWithRooms(g, .8f, -4, 0, 4);
            var planner = new GridPlanner(g, c); double t = 0;
            planner.Rooms.Update(t, new Vector2(-4, 2.7f));                       // walked into the west room only
            planner.Rooms.Update(t += 1, new Vector2(-3.5f, 0));                  // and back out into the corridor
            int hall = planner.Rooms.RoomAt(Vector2.zero), west = planner.Rooms.RoomAt(new Vector2(-4, 2.7f));
            Assert.That(planner.Rooms.Get(hall).corridor, Is.True, "control: the layout's corridor is detected");
            var first = planner.Frontiers(new Vector2(-3.5f, 0))[0];
            Assert.That(first.roomId, Is.Not.EqualTo(hall).And.Not.EqualTo(west), "a room not yet entered comes first");
            planner.Rooms.Update(t += 1, new Vector2(0, 2.7f)); planner.Rooms.Update(t += 1, new Vector2(4, 2.7f));
            planner.Rooms.Update(t += 1, new Vector2(-3.5f, 0));                  // every room entered, standing in the corridor
            var ranked = planner.Frontiers(new Vector2(-3.5f, 0));
            Assert.That(ranked.Any(f => f.roomId == west), Is.True, "control: the entered west room still has unsearched floor");
            Assert.That(ranked[0].roomId, Is.EqualTo(hall), "the corridor, which reveals doors, comes before pockets in entered rooms; " +
                string.Join(", ", ranked.Take(6).Select(f => f.roomId + "/" + f.roomTier)));
        }

        [Test]
        public void CorridorFirstWalksTheCorridorBeforeEnteringRoomsButStillFinishesTheRoomItIsIn()
        {
            // Headless seed 5001: the kitchen door 17 m down the corridor was never seen from its lower half. With corridor first,
            // a robot in the corridor walks it before taking a room; inside a room it still finishes that room.
            var c = AuctionSettings(); c.room_auction = false; c.room_ledger_corridor_first = true;
            var g = new OccupancyGrid(c); CorridorWithRooms(g, .8f, -4, 0, 4);
            var planner = new GridPlanner(g, c);
            planner.Rooms.Update(0, new Vector2(-3.5f, 0));                       // standing in the corridor, no room entered
            int hall = planner.Rooms.RoomAt(Vector2.zero);
            Assert.That(planner.Frontiers(new Vector2(-3.5f, 0))[0].roomId, Is.EqualTo(hall), "corridor before rooms not yet entered");
            c.room_ledger_corridor_first = false;
            Assert.That(planner.Frontiers(new Vector2(-3.5f, 0))[0].roomId, Is.Not.EqualTo(hall), "control: off, a room not yet entered comes first");
            c.room_ledger_corridor_first = true;
            planner.Rooms.Update(1, new Vector2(4, 2.7f)); planner.Rooms.Update(2, new Vector2(4, 2.7f));   // now inside the east room
            int east = planner.Rooms.RoomAt(new Vector2(4, 2.7f));
            Assert.That(planner.Frontiers(new Vector2(4, 2.7f))[0].roomId, Is.EqualTo(east), "inside an unfinished room, it finishes that room first");
        }

        [Test]
        public void WithTheTurnCostAFrontierAheadBeatsASlightlyNearerOneBehind()
        {
            // Headless seed 5001: turning in place toward newly chosen goals was 25 to 27 s per robot in 240 s.
            var c = AuctionSettings(); c.room_ledger = false; c.frontier_gain_weight = 0; var g = new OccupancyGrid(c);
            var searched = (bool[])typeof(OccupancyGrid).GetField("observed", Private).GetValue(g);
            for (int i = 0; i < g.Count; i++)
            {
                var p = g.Centre(i);
                if (Mathf.Abs(p.x) > 1.7f || p.y < -3.2f || p.y > 4.2f) continue;
                // A walled room: unknown floor around it would itself be frontier beside the robot.
                if (Mathf.Abs(p.x) > 1.5f || p.y < -3f || p.y > 4f) { g.ObserveOccupied(i); continue; }
                g.ObserveFree(i);
                // Searched floor everywhere except a patch 2.6 m ahead and a patch 2.0 m behind (robot at the origin facing +y).
                searched[i] = !((p.y > 2.6f && p.y < 3.6f) || (p.y < -2.0f && p.y > -2.9f));
            }
            var planner = new GridPlanner(g, c) { HeadingRad = 0 };
            c.frontier_turn_cost = false;
            Assert.That(planner.Frontiers(Vector2.zero)[0].position.y, Is.LessThan(0), "control: travel alone picks the nearer patch behind");
            c.frontier_turn_cost = true;
            Assert.That(planner.Frontiers(Vector2.zero)[0].position.y, Is.GreaterThan(0), "with the turn cost the patch ahead wins");
            planner.HeadingRad = Mathf.PI;
            Assert.That(planner.Frontiers(Vector2.zero)[0].position.y, Is.LessThan(0), "facing the other way, the patch behind is now ahead");
        }

        [Test]
        public void AnEnclosedRoomWithSomeProvenFloorBecomesSearchAreaButAnOpenYardNever()
        {
            // Headless seeds 5001 to 5004: the far side of a room stayed unproven, was never offered, and robots left after
            // the near corner. Controls: an open-edged region (a yard) is never promoted, and the switch off promotes nothing.
            (RoomLedger ledger, BrainConfig c) Build(bool walled, bool on)
            {
                var c = AuctionSettings(); c.room_indoor_promotion = on; var g = new OccupancyGrid(c);
                for (int i = 0; i < g.Count; i++)
                {
                    var p = g.Centre(i);
                    bool inside = Mathf.Abs(p.x) < 2.5f && p.y > 0 && p.y < 5;
                    bool edge = Mathf.Abs(p.x) < 2.7f && p.y > -.2f && p.y < 5.2f && !inside;
                    if (inside) g.ObserveFree(i); else if (edge && walled) g.ObserveOccupied(i);
                }
                var searched = (bool[])typeof(OccupancyGrid).GetField("observed", Private).GetValue(g);
                System.Array.Clear(searched, 0, searched.Length);
                var ledger = new RoomLedger(g, c);
                ledger.Update(0, new Vector2(0, .5f), p => p.y < 1.5f);   // only a strip by the door is proven indoors
                return (ledger, c);
            }
            var room = Build(true, true).ledger;
            Assert.That(room.Promoted(new Vector2(0, 4.5f)), Is.True, "the far side of an enclosed, partly proven room is search area");
            Assert.That(room.Get(room.RoomAt(new Vector2(0, 4.5f))).coveredShare, Is.LessThan(.1f), "coverage is counted on all its floor");
            Assert.That(Build(false, true).ledger.Promoted(new Vector2(0, 4.5f)), Is.False, "an open-edged region (a yard) is never promoted");
            Assert.That(Build(true, false).ledger.Promoted(new Vector2(0, 4.5f)), Is.False, "control: off promotes nothing");
        }

        [Test]
        public void ARoomBidIsTheWalkedLengthOverOpenSpeedNotThePlannerCost()
        {
            // The planner cost adds clearance and heat penalties: on 223 frontiers of the headless houses (exp6, 23 Oct) it ran
            // 1.42 times the walked length (median; 1.19 to 2.07 between the 10th and 90th percentiles, at most 3.56). Bidding it
            // lost rooms to a partner that was really farther away.
            // A 1.0 m corridor with the shipped clearance band (brain.json obstacle_cost_margin_m 0.6, weight 3; the code defaults,
            // 0.15 m, leave every walkable cell outside the band): its whole length lies inside the band, as a narrow hall does.
            var c = AuctionSettings(); c.obstacle_cost_margin_m = .6f; c.obstacle_cost_weight = 3f;
            var g = new OccupancyGrid(c); CorridorWithRooms(g, .5f, -4, 0, 4);
            var planner = new GridPlanner(g, c); planner.Rooms.Update(0, new Vector2(-5.5f, 0));
            var bids = new Dictionary<int, float>();
            planner.RoomTaken = (room, bid) => { bids[room] = bid; return false; };
            var from = new Vector2(-5.5f, 0); var options = planner.Frontiers(from);
            int east = planner.Rooms.RoomAt(new Vector2(4, 2.7f));
            var nearest = options.Where(f => f.roomId == east).OrderBy(f => GridPlanner.Length(f.path, from)).First();
            Assert.That(nearest.cost, Is.GreaterThan(GridPlanner.Length(nearest.path, from) * 1.1f), "control: the planner cost exceeds the walked length here");
            Assert.That(bids[east], Is.EqualTo(options.Where(f => f.roomId == east).Min(f => GridPlanner.Length(f.path, from)) / c.open_speed_mps).Within(1e-3),
                "the bid is the shortest walked length to the room over open_speed_mps");
        }

        [Test]
        public void APersonLyingInACorridorNoLongerSealsItWhenFloorBeyondIsCutOff()
        {
            // Headless seed 5001: a confirmed person in the 1.7 m hallway sealed it (0.45 m core plus the walking clearance) and
            // four rooms beyond went unsearched. Controls: the full core seals the corridor; the rule fires only when floor is cut
            // off, and only once.
            var c = AuctionSettings(); c.survivor_squeeze = true;
            var b = new RobotBrain(Calibration(), c, "robot_A", 0); var g = Map(b);
            for (int i = 0; i < g.Count; i++)
            {
                var p = g.Centre(i);
                if (Mathf.Abs(p.y) > 6.1f || Mathf.Abs(p.x) > 1.1f) continue;
                if (Mathf.Abs(p.x) > .9f || Mathf.Abs(p.y) > 6f) g.ObserveOccupied(i); else g.ObserveFree(i);   // a 1.8 m corridor
            }
            var searched = (bool[])typeof(OccupancyGrid).GetField("observed", Private).GetValue(g);
            for (int i = 0; i < g.Count; i++) searched[i] = g.Centre(i).y < 1.5f;      // floor beyond the person is unsearched
            var person = new Vector2(0, 2); var planner = Planner(b);
            ((List<Vector2>)typeof(RobotBrain).GetField("survivors", Private).GetValue(b)).Add(person);
            bool Reaches(float core)
            {
                g.SetKeepOut(new[] { person }, core, c.survivor_keepout_m + c.robot_radius_m);
                var path = planner.Plan(new Vector2(0, -3), new Vector2(0, 5));
                return path.Count > 0 && Vector2.Distance(path[path.Count - 1], new Vector2(0, 5)) < .2f;
            }
            Assert.That(Reaches(c.survivor_keepout_core_m), Is.False, "control: the full core seals a 1.8 m corridor");
            Assert.That(Reaches(c.survivor_squeeze_core_m), Is.True, "at the body's half-width the robot can pass along the side");
            g.SetKeepOut(new[] { person }, c.survivor_keepout_core_m, c.survivor_keepout_m + c.robot_radius_m);
            typeof(RobotBrain).GetField("motion", Private).GetValue(b);
            var squeeze = typeof(RobotBrain).GetMethod("SqueezePastPerson", Private);
            Assert.That((bool)squeeze.Invoke(b, new object[] { 1.0 }), Is.True, "floor beyond the person is cut off: squeeze");
            Assert.That((bool)squeeze.Invoke(b, new object[] { 2.0 }), Is.False, "once only");
            c.survivor_squeeze = false; typeof(RobotBrain).GetField("squeezePast", Private).SetValue(b, false);
            Assert.That((bool)squeeze.Invoke(b, new object[] { 3.0 }), Is.False, "control: switched off");
        }

        [Test]
        public void WithCorridorFirstOneRobotHoldsTheHallAndAFollowThroughIgnoresRoomClaims()
        {
            // Fire crews split at a hallway: one works the hall, the others take the rooms (Miami-Dade 2024; NFPA skill drill
            // 12-2). Headless seed 5001: with unclaimable halls both robots walked the hall and the living room was left at 46 %.
            // Headless seed 5002: room claims filtered robot B's follow-through goals and it waited outside 42 s.
            var c = AuctionSettings(); c.room_ledger_corridor_first = true;
            var b = new RobotBrain(Calibration(), c, "robot_2", 0);
            CorridorWithRooms(Map(b), .8f, -4, 0, 4);
            var planner = Planner(b); planner.Rooms.Update(0, Vector2.zero);
            int hall = planner.Rooms.RoomAt(Vector2.zero), west = planner.Rooms.RoomAt(new Vector2(-4, 2.7f));
            long sequence = 0;
            void Claim(Vector2 anchor, double sent)
            {
                var m = new RobotBrain(Calibration(), c, "robot_1", 0).BuildMessage(sent); m.Sequence = ++sequence;
                m.RoomClaim = new BrainRoomClaim { anchor = anchor, radiusM = 2, bidS = 0, since = sent, expiresAt = sent + c.peer_claim_s };
                b.OnPeerMessage(Wire(m), sent + .1);
            }
            Assert.That(planner.Frontiers(Vector2.zero).Any(f => f.roomId == hall), Is.True, "control: the hall is offered before any claim");
            Claim(Vector2.zero, 0);
            Assert.That(planner.Frontiers(Vector2.zero).Any(f => f.roomId == hall), Is.False, "with corridor first a partner can hold the hall");
            Assert.That(planner.Frontiers(Vector2.zero).Any(f => f.roomId == west), Is.True, "rooms off the held hall stay open");
            Claim(new Vector2(-4, 2.7f), .2);
            Assert.That(planner.Frontiers(Vector2.zero).Any(f => f.roomId == west), Is.False, "control: the partner's room is not offered to the search");
            Assert.That(planner.Frontiers(Vector2.zero, null, true).Any(f => f.roomId == west), Is.True, "a follow-through still sees it");
            // Headless seed 5002: threshold looks for floor not yet proven indoors were filtered by the partner's room, and robot B
            // scanned outside for 42 s.
            Assert.That(planner.Frontiers(Vector2.zero, null, false, false).Any(f => f.roomId == west), Is.True, "a caller not dividing search floor still sees it");
        }

        [Test]
        public void ACandidateOutOfViewDuringTheApproachIsNotLostButOneInViewAndUnseenIs()
        {
            // Headless house 5007: all five people seen, one confirmed; approaches ended "evidence was lost" while the person's
            // spot was out of view behind furniture or a door frame. A collapsed person does not move.
            var c = AuctionSettings(); c.survivor_lost_in_view = true;
            var b = new RobotBrain(Calibration(), c, "robot_A", 0); Room(b, -6, 6, -6, 6);   // open mapped floor, robot at origin facing +y
            var trackType = typeof(RobotBrain).GetNestedType("Track", Private);
            object Track(Vector2 at) { var k = Activator.CreateInstance(trackType, true); trackType.GetField("position").SetValue(k, at);
                trackType.GetField("lastSeen").SetValue(k, 0.0); trackType.GetField("candidate").SetValue(k, true); return k; }
            var lost = typeof(RobotBrain).GetMethod("SurvivorLost", Private);
            bool Lost(object track, double t) { typeof(RobotBrain).GetField("active", Private).SetValue(b, track); return (bool)lost.Invoke(b, new object[] { t }); }
            void Reset() { typeof(RobotBrain).GetField("activeSince", Private).SetValue(b, 0.0); typeof(RobotBrain).GetField("lostClockFrom", Private).SetValue(b, double.NaN); }
            Reset(); var behind = Track(new Vector2(0, -3));
            Assert.That(Lost(behind, 1) || Lost(behind, 5), Is.False, "a spot behind the robot is not looked at, so it is not lost");
            Reset(); var ahead = Track(new Vector2(0, 3));
            Assert.That(Lost(ahead, 1), Is.False, "in view, unseen for 1 s: not yet lost");
            Assert.That(Lost(ahead, 4.5), Is.True, "in view and unseen for more than survivor_lost_s: lost");
            Reset();
            Assert.That(Lost(behind, 100), Is.True, "the approach is capped (2 x detection range at near speed)");
            c.survivor_lost_in_view = false; Reset();
            Assert.That(Lost(behind, 5), Is.True, "control: off, an unseen spot is lost after survivor_lost_s wherever it is");
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
