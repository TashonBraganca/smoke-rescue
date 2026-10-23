using System;
using UnityEngine;

namespace SmokeRescue.Brain
{
    /// <summary>Rooms split between any number of robots by code (owner decision, 23 Oct; seed 932: the agents split the house
    /// in words and nobody searched the east). Each robot holds at most one room and radios it with its bid, the seconds of
    /// travel to the room's nearest frontier. A room belongs to the lowest bid. A challenger must beat the holder by more than
    /// room_auction_margin_s; within the margin the robot that took the room first keeps it, and on equal times the lower ID.
    /// Every robot applies the same rule to the same radioed claims, so they agree without an auctioneer. A claim lapses
    /// peer_claim_s after it was sent, so a robot that goes quiet loses its room. Corridors are never claimed: a claim on a
    /// corridor would lock partners out of every room off it. With room_ledger_corridor_first a corridor is claimable: one robot
    /// works the hall while partners take the rooms off it, as fire crews split at a hallway (Miami-Dade Fire Rescue, Searching
    /// Techniques, 2024; Jones and Bartlett / NFPA skill drill 12-2, both in prior_art_23oct.txt). Rooms off the hall keep their
    /// own claims, so nobody is locked out of them; a claimed hall only stops a partner choosing frontiers in the hall itself.
    /// Headless seed 5001: with corridor first and unclaimable halls, both robots walked the hall and the living room was left
    /// at 46 %.</summary>
    public sealed partial class RobotBrain
    {
        int claimedRoom = -1;
        double claimedSince;
        float claimedBid;

        bool AuctionOn => c.room_ledger && c.room_auction;
        double AuctionNow => Math.Max(double.IsNaN(lastStep) ? 0 : lastStep, lastPeerReceived);

        /// <summary>This robot's own room at a partner's anchor: the anchor cell, or the nearest labelled cell within
        /// three cells (the two maps place one wall up to a cell or two apart). -1 when this robot has not mapped it.</summary>
        int RoomForAnchor(Vector2 anchor)
        {
            int room = planner.Rooms.RoomAt(anchor); if (room >= 0) return room;
            float best = float.PositiveInfinity;
            for (int dy = -3; dy <= 3; dy++) for (int dx = -3; dx <= 3; dx++)
            {
                var p = anchor + new Vector2(dx, dy) * grid.CellM; int r = planner.Rooms.RoomAt(p);
                float d = Vector2.Distance(p, anchor);
                if (r >= 0 && d < best) { best = d; room = r; }
            }
            return room;
        }

        /// <summary>Does a live partner claim win this room against this robot's bid? mySince is when this robot took the room,
        /// or now for a room it does not hold.</summary>
        bool PeerWinsRoom(int room, float myBid, double mySince, double t)
        {
            var own = planner.Rooms.Get(room);
            if (!AuctionOn || own == null || (own.corridor && !c.room_ledger_corridor_first)) return false;
            foreach (var pair in peers)
            {
                var claim = pair.Value.room;
                if (claim == null || claim.expiresAt <= t || RoomForAnchor(claim.anchor) != room) continue;
                if (claim.bidS + c.room_auction_margin_s < myBid) return true;
                if (myBid + c.room_auction_margin_s < claim.bidS) continue;
                if (claim.since < mySince || (claim.since == mySince && CompareIds(pair.Key, id) < 0)) return true;
            }
            return false;
        }

        /// <summary>Planner filter: a frontier in a room a partner won is not offered, to the brain or to the agent.</summary>
        bool RoomTaken(int room, float myBid)
        {
            double t = AuctionNow;
            return PeerWinsRoom(room, myBid, room == claimedRoom ? claimedSince : t, t);
        }

        /// <summary>The robot set off for a frontier: it holds that frontier's room, unless the room is a corridor.</summary>
        void ClaimRoom(double t, Vector2 frontier, float pathLengthM)
        {
            if (!AuctionOn) return;
            int room = planner.Rooms.RoomForFrontier(frontier); var r = planner.Rooms.Get(room);
            if (r == null || (r.corridor && !c.room_ledger_corridor_first)) { claimedRoom = -1; return; }
            claimedBid = pathLengthM / c.open_speed_mps;
            if (room == claimedRoom) return;
            claimedRoom = room; claimedSince = t;
            Emit(t, "room_claimed", "This robot holds the room of its frontier; partners hear its bid.", "room_id", room,
                "bid_s", claimedBid, "anchor_x_m", r.anchor.x, "anchor_y_m", r.anchor.y, "radius_m", r.radiusM);
        }

        /// <summary>Metres from the robot along a path.</summary>
        float PathLength(System.Collections.Generic.List<Vector2> route)
        {
            float m = 0; var at = motion.Pose.position;
            foreach (var p in route) { m += Vector2.Distance(at, p); at = p; }
            return m;
        }

        /// <summary>Seconds of travel left along the current path, from the waypoint being followed. Summing the whole path made
        /// the bid grow as the robot walked: 9 m into a 10 m route it bid 19 m, lost the room to a partner 6 m away, and
        /// re-claimed it on the next plan (review of 23 Oct).</summary>
        float RemainingBid()
        {
            float m = 0; var at = motion.Pose.position;
            for (int i = Mathf.Clamp(cursor, 0, path.Count); i < path.Count; i++) { m += Vector2.Distance(at, path[i]); at = path[i]; }
            return m / c.open_speed_mps;
        }

        BrainRoomClaim RoomClaimToSend(double t, bool searching)
        {
            var r = claimedRoom >= 0 ? planner.Rooms.Get(claimedRoom) : null;
            if (!AuctionOn || !searching || r == null || (r.corridor && !c.room_ledger_corridor_first)) { claimedRoom = -1; return null; }
            if (path.Count > 0) claimedBid = RemainingBid();
            return new BrainRoomClaim { anchor = r.anchor, radiusM = r.radiusM, bidS = claimedBid, since = claimedSince, expiresAt = t + c.peer_claim_s };
        }

        /// <summary>A partner's better bid takes this robot's room: give it up once and plan elsewhere.</summary>
        void YieldLostRoom(double t)
        {
            if (!AuctionOn || claimedRoom < 0 || !frontierGoal || !goal.HasValue) return;
            if (!PeerWinsRoom(claimedRoom, path.Count > 0 ? RemainingBid() : claimedBid, claimedSince, t)) return;
            Emit(t, "room_yielded", "A partner bid less time to this room, or took it first within the margin; it keeps the room.",
                "room_id", claimedRoom, "bid_s", claimedBid);
            if (AgentGoalActive(t) && agentGoalIsFrontier) agentGoal = null;
            claimedRoom = -1; ClearPath();
        }
    }
}
