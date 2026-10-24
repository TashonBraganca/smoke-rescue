using System;
using System.Collections.Generic;
using UnityEngine;

namespace SmokeRescue.Brain
{
    /// <summary>
    /// room_memory (owner, 24 Oct: "enter every room, mark it, never go back; remember where to go"). The robot keeps a record
    /// of every room it has mapped (RoomLedger.RoomRecord, keyed by the room's seat in the shared launch frame). Between rooms
    /// it walks to the seat of the next room on a tour (the room that starts the shortest walk through all rooms not yet
    /// entered), so it steps inside; it finishes the room it stands in first (tier 0 of the ledger), and it keeps its chosen
    /// room unless that room is entered or taken, cannot be reached, or another tour is shorter by room_tour_switch_s.
    /// Headless 24 Oct: entering any unentered room between goals without finishing the current one or keeping a target
    /// zigzagged (rooms 57/69 -> 51/69); greedy nearest-first left 12 of 69 rooms unentered, 6 of them deferred until 240 s.
    /// </summary>
    public sealed partial class RobotBrain
    {
        RoomLedger.RoomRecord roomTarget;
        readonly List<double> roomTargetSwitches = new List<double>();

        /// <summary>Called every step: a record whose seat this robot reached is entered, with its doorway.</summary>
        void UpdateRoomMemory(double t)
        {
            if (!c.room_memory || !c.room_ledger) return;
            var rec = planner.Rooms.MarkEnteredNear(motion.Pose.position, t);
            if (rec != null)
                Emit(t, "room_entered", "Stepped inside a room in this robot's memory (onto its deep floor): entered.", "room_key", rec.key,
                    "x_m", rec.seat.x, "y_m", rec.seat.y, "door_x_m", rec.door?.x ?? float.NaN, "door_y_m", rec.door?.y ?? float.NaN);
            // A room walk is only re-chosen when the path runs out, so a partner that got there first or heads there from nearer
            // would otherwise be followed all the way in.
            if (roomTarget != null && roomTarget.label >= 0 && goal.HasValue && active == null && !returning && !retreating && !AgentGoalActive(t) &&
                planner.Rooms.RoomNear(goal.Value) == roomTarget.label &&
                (roomTarget.partnerEntered || PartnerInRoom(roomTarget) || PartnerHeadsFor(roomTarget, motion.Pose.position)))
            {
                Emit(t, "room_target_dropped", "The partner is in, or heading from nearer for, the room this robot was walking into: choose another.",
                    "room_key", roomTarget.key);
                roomTarget = null; ClearPath();
            }
        }

        /// <summary>The next room to walk into, as a route to its seat; false while the room this robot stands in is being
        /// finished, or when no remembered room is waiting.</summary>
        bool NextRoom(double t, out List<Vector2> route)
        {
            route = null;
            if (!c.room_memory || !c.room_ledger) return false;
            var at = motion.Pose.position; var ledger = planner.Rooms;
            // Finish the room this robot stands in first, but only while it still offers an admitted frontier: with none left
            // (its floor seen, the rest unprovable) the robot would otherwise scan in place until the dwell budget ran out.
            if (ledger.Finishing(at) && CurrentRoomHasFrontier(t)) return false;
            if (roomTarget != null && (roomTarget.entered || roomTarget.partnerEntered || roomTarget.done || roomTarget.label < 0)) roomTarget = null;
            var open = new List<RoomLedger.RoomRecord>();
            foreach (var r in ledger.Records)
            {
                if (r.label < 0 || r.entered || r.partnerEntered || r.done || r.corridor || r.floorM2 < c.room_walk_in_min_m2) continue;
                if (r.retryAt > t || !InSearchArea(r.seat) || GoalBlacklisted(r.seat, t) || ClaimBlocks(r.seat, t) || PartnerInRoom(r) || PartnerHeadsFor(r, at)) continue;
                open.Add(r);
            }
            if (open.Count == 0) { roomTarget = null; return false; }
            open.Sort((a, b) => Vector2.Distance(a.seat, at).CompareTo(Vector2.Distance(b.seat, at)));
            if (roomTarget != null && !open.Contains(roomTarget)) roomTarget = null;
            var reach = new List<(RoomLedger.RoomRecord r, List<Vector2> p, float s)>();
            int plans = 0;
            foreach (var r in open)
            {
                if (plans++ >= c.door_enter_max_plans && r != roomTarget) continue;
                var p = planner.Plan(at, r.seat);
                // No route: set this seat aside for stuck_blacklist_s, or the same unreachable nearest seats take every plan
                // slot at each call and rooms further away are never considered.
                if (p.Count > 0) reach.Add((r, p, PathLength(p) / c.open_speed_mps)); else r.retryAt = t + c.stuck_blacklist_s;
            }
            if (reach.Count == 0) { roomTarget = null; return false; }
            var seats = reach.ConvertAll(x => x.r.seat);
            (RoomLedger.RoomRecord r, List<Vector2> p, float s) best = default; float bestCost = float.PositiveInfinity, keepCost = float.PositiveInfinity;
            foreach (var x in reach)
            {
                float cost = x.s + SeatTour(x.r.seat, seats) / c.open_speed_mps + PartnerSide(x.r.seat, at) / c.open_speed_mps;
                if (cost < bestCost) { bestCost = cost; best = x; }
                if (x.r == roomTarget) keepCost = cost;
            }
            roomTargetSwitches.RemoveAll(s => t - s > c.room_tour_breaker_s);
            bool breaker = roomTargetSwitches.Count >= c.room_tour_breaker_switches;
            if (roomTarget != null && !float.IsPositiveInfinity(keepCost) && (breaker || bestCost + c.room_tour_switch_s >= keepCost))
                best = reach.Find(x => x.r == roomTarget);
            if (best.r != roomTarget)
            {
                if (roomTarget != null) roomTargetSwitches.Add(t);
                roomTarget = best.r;
                Emit(t, "room_target", "Next room on the tour of rooms this robot remembers but has not entered: walk to its centre.",
                    "room_key", best.r.key, "x_m", best.r.seat.x, "y_m", best.r.seat.y, "path_s", best.s, "tour_s", bestCost, "open_rooms", reach.Count);
            }
            route = best.p;
            return true;
        }

        /// <summary>A linked partner stands in the record's room now.</summary>
        bool PartnerInRoom(RoomLedger.RoomRecord r)
        {
            foreach (var (pid, p, goal) in planner.Rooms.PartnersNow) if (planner.Rooms.RoomNear(p) == r.label) return true;
            return false;
        }

        /// <summary>A linked partner radioed a goal in the record's room and stands nearer its seat (equal: the lower ID keeps it),
        /// so two robots never walk into one room together and never both give it up. The walk to a seat is not a frontier
        /// claim (SetPath(route, false)), so the radio carries it only as the goal.</summary>
        bool PartnerHeadsFor(RoomLedger.RoomRecord r, Vector2 at)
        {
            float mine = Vector2.Distance(at, r.seat);
            foreach (var (pid, p, goal) in planner.Rooms.PartnersNow)
            {
                if (!goal.HasValue || planner.Rooms.RoomNear(goal.Value) != r.label) continue;
                float theirs = Vector2.Distance(p, r.seat);
                if (theirs < mine || (theirs == mine && CompareIds(pid, id) < 0)) return true;
            }
            return false;
        }

        /// <summary>The room this robot stands in still offers an admitted frontier (tier 0 of the ledger's ranking).</summary>
        bool CurrentRoomHasFrontier(double t) =>
            planner.Frontiers(motion.Pose.position, p => ClaimBlocks(p, t) || GoalBlacklisted(p, t) || VisitedGoal(p, t) || !InSearchArea(p)).Exists(f => f.roomTier == 0);

        /// <summary>Straight-line length of a nearest-neighbour walk from first through every other seat.</summary>
        static float SeatTour(Vector2 first, List<Vector2> seats)
        {
            var left = seats.FindAll(s => s != first); var here = first; float total = 0;
            while (left.Count > 0)
            {
                int k = 0; float m = Vector2.Distance(here, left[0]);
                for (int i = 1; i < left.Count; i++) { float q = Vector2.Distance(here, left[i]); if (q < m) { m = q; k = i; } }
                total += m; here = left[k]; left.RemoveAt(k);
            }
            return total;
        }

        /// <summary>The rooms this robot remembers, for logs and the agent: seen, entered and done counts.</summary>
        public (int seen, int entered, int done) RoomMemoryCounts()
        {
            int s = 0, e = 0, d = 0;
            foreach (var r in planner.Rooms.Records) { if (r.label < 0) continue; s++; if (r.entered || r.partnerEntered) e++; if (r.done) d++; }
            return (s, e, d);
        }
    }
}
