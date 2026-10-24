using System;
using System.Collections.Generic;
using UnityEngine;

namespace SmokeRescue.Brain
{
    /// <summary>
    /// door_claims (owner, 24 Oct): a robot that walks through a doorway into a room records the doorway point and radios
    /// "taken" by it; when it walks back out it radios "cleared". The partner takes no search goal behind a doorway the other
    /// robot has taken, and nobody takes one behind a cleared doorway, so no room is searched twice. A doorway is found in the robot's own
    /// LiDAR map: a free gap door_min_width_m to door_max_width_m wide across the heading, with wider floor
    /// (door_side_extra_m more) on both sides, which a narrow corridor does not have. A "taken" doorway lapses door_claim_lease_s
    /// after it was last heard, so a stuck owner cannot lock a room. Seed 934 (24 Oct 04:53 run): robot A walked into the
    /// kitchen robot B was searching and stayed 28.7 s; 68 room claims were sent and none stopped it (Codex cx-rooms report).
    /// </summary>
    public sealed partial class RobotBrain
    {
        internal const byte DoorSeen = 0, DoorTaken = 1, DoorCleared = 2, DoorHeading = 3;
        /// <summary>Order of states when two reports of one doorway meet: seen, then heading, then taken, then cleared.</summary>
        static int DoorRank(byte state) => state == DoorSeen ? 0 : state == DoorHeading ? 1 : state == DoorTaken ? 2 : 3;
        Door doorTarget; double nextDoorScan = double.NegativeInfinity; bool doorSweep;
        readonly List<Vector2> trail = new List<Vector2>();
        sealed class Door { public Vector2 position, inward; public string owner; public byte state; public double since, heardAt; }
        readonly List<Door> doors = new List<Door>();
        Vector2? doorwayAt; Vector2 doorwayAxis; double nextDoorCheck = double.NegativeInfinity;

        /// <summary>Free distance from p along dir to the first occupied cell, out to maxM.</summary>
        float FreeRun(Vector2 p, Vector2 dir, float maxM)
        {
            for (float s = 0; s <= maxM; s += grid.CellM * .5f)
            {
                int cell = grid.Cell(p + dir * s);
                if (cell < 0 || grid.OwnOccupancy(cell) == 255) return s;
            }
            return maxM;
        }

        /// <summary>The narrowest free gap through p over axes within 60 deg of the heading (a doorway crossed at an angle reads
        /// wider across the heading), when it is door_min_width_m to door_max_width_m wide, the floor door_cross_m along its axis
        /// opens up by door_side_extra_m on at least one side, and is no narrower than the gap on the other (a door off a narrow
        /// hallway; walking along a corridor never opens up). Headless houses 5001-5022: the across-the-heading test with wider
        /// floor required on both sides claimed 28 of 77 room entries.</summary>
        bool InDoorway(Vector2 p, Vector2 heading, out Vector2 middle, out Vector2 axis)
        {
            middle = p; axis = heading; float best = float.PositiveInfinity, bestL = 0, bestR = 0; Vector2 bestSide = Vector2.zero;
            float h = Mathf.Atan2(heading.y, heading.x);
            for (int k = -6; k <= 6; k++)
            {
                float a = h + k * 10 * Mathf.Deg2Rad; var ax = new Vector2(Mathf.Cos(a), Mathf.Sin(a)); var side = new Vector2(-ax.y, ax.x);
                float l = FreeRun(p, side, c.door_max_width_m), r = FreeRun(p, -side, c.door_max_width_m);
                if (l + r < best) { best = l + r; bestL = l; bestR = r; bestSide = side; axis = ax; }
            }
            if (best < c.door_min_width_m || best > c.door_max_width_m) return false;
            float reach = c.door_max_width_m + c.door_side_extra_m, wide = 0, narrow = float.PositiveInfinity;
            foreach (float k in new[] { 1f, -1f })
            {
                var q = p + axis * (k * c.door_cross_m);
                float wq = FreeRun(q, bestSide, reach) + FreeRun(q, -bestSide, reach);
                wide = Mathf.Max(wide, wq); narrow = Mathf.Min(narrow, wq);
            }
            if (wide < best + c.door_side_extra_m || narrow < best - grid.CellM) return false;
            // A door frame is thin along the axis (a wall's thickness); a hallway's walls run along it. Standing in a 1.2 m
            // hallway beside a doorway, the cut across the hallway passed every other test.
            foreach (var jamb in new[] { p + bestSide * (bestL - grid.CellM * .5f), p - bestSide * (bestR - grid.CellM * .5f) })
                foreach (float k in new[] { 1f, -1f })
                {
                    int cell = grid.Cell(jamb + axis * (k * c.door_frame_depth_m));
                    if (cell < 0 || grid.OwnOccupancy(cell) == 255) return false;
                }
            middle = p + bestSide * ((bestL - bestR) / 2); return true;
        }

        void UpdateDoors(double t)
        {
            if (!c.door_claims || t < nextDoorCheck) return;
            nextDoorCheck = t + .1;
            var at = motion.Pose.position; var forward = motion.Pose.Forward;
            if (trail.Count == 0 || Vector2.Distance(trail[trail.Count - 1], at) >= c.door_trail_step_m) trail.Add(at);
            ScanDoorways(t);
            if (doorTarget != null && (doorTarget.owner != id || doorTarget.state != DoorHeading))
            {
                // The partner won this doorway (it was heading there first) or it was taken: drop the route to it.
                if (doorTarget.owner != id && doorTarget.state == DoorHeading)
                {
                    Emit(t, "door_yielded", "The partner was heading to this doorway first: leave the room to it and pick another.",
                        "x_m", doorTarget.position.x, "y_m", doorTarget.position.y);
                    ClearPath();
                }
                doorTarget = null;
            }
            if (InDoorway(at, forward, out var middle, out var axis))
            {
                if (!doorwayAt.HasValue || Vector2.Distance(doorwayAt.Value, middle) > c.door_match_m)
                { doorwayAt = middle; doorwayAxis = Vector2.Dot(axis, forward) >= 0 ? axis : -axis; }
            }
            else if (doorwayAt.HasValue)
            {
                float along = Vector2.Dot(at - doorwayAt.Value, doorwayAxis);
                if (Mathf.Abs(along) >= c.door_cross_m) { Crossed(t, doorwayAt.Value, doorwayAxis * Mathf.Sign(along)); doorwayAt = null; }
                else if (Vector2.Distance(at, doorwayAt.Value) > c.door_cross_m * 2) doorwayAt = null;
            }
            // The owner clears its room once it has been out of it for door_left_s, by any doorway. Headless 5002: robot A took
            // the kitchen and left by another doorway; the room stayed taken, robot B never searched it, nor the bedroom beyond.
            foreach (var d in doors)
            {
                if (d.owner != id || d.state != DoorTaken) continue;
                if (InsideDoorRoom(d, at)) { d.heardAt = t; continue; }
                if (t - d.heardAt >= c.door_left_s)
                {
                    d.state = DoorCleared; d.since = t;
                    Emit(t, "door_cleared", "Left the room this robot searched (by another way): the room is cleared; nobody goes in again.",
                        "x_m", d.position.x, "y_m", d.position.y);
                }
            }
        }

        void Crossed(double t, Vector2 door, Vector2 direction)
        {
            var known = doors.Find(d => Vector2.Distance(d.position, door) <= c.door_match_m);
            // A doorway seen from the map, or the one this robot was heading for, is taken like an unknown one.
            if (known != null && (known.state == DoorSeen || (known.state == DoorHeading && known.owner == id)) && Vector2.Dot(direction, known.inward) > 0)
            {
                doors.Remove(known); if (doorTarget == known) doorTarget = null; known = null;
            }
            if (known != null)
            {
                if (known.owner == id && known.state == DoorTaken && Vector2.Dot(direction, known.inward) < 0)
                {
                    known.state = DoorCleared; known.since = t; known.heardAt = t;
                    Emit(t, "door_cleared", "Walked back out through the doorway of a room this robot searched: the room is cleared; nobody goes in again.",
                        "x_m", door.x, "y_m", door.y);
                }
                return;
            }
            // Not a room: coming in from outside (the entrance), the start area, or a corridor on the far side.
            if (!InSearchArea(door - direction * c.door_cross_m) || Vector2.Distance(door, home) <= c.door_home_clear_m) return;
            int room = planner.Rooms.RoomAt(door + direction * c.door_cross_m);
            var info = room >= 0 ? planner.Rooms.Get(room) : null;
            if (info != null && info.corridor) return;
            // A room bigger than one robot can search in the time left stays open to both (headless 5021: robot A took a 134 m2
            // bedroom, robot B left it, and its person s1 was not found).
            if (info != null && info.indoorFloorM2 > c.door_claim_max_m2) return;
            doors.Add(new Door { position = door, inward = direction.normalized, owner = id, state = DoorTaken, since = t, heardAt = t });
            Emit(t, "door_taken", "Walked through a doorway into a room: this robot takes it; the partner does not go in.",
                "x_m", door.x, "y_m", door.y, "inward_x", direction.x, "inward_y", direction.y);
        }

        /// <summary>Whether a search goal at p lies in a room this robot must not search: behind a doorway the partner has taken
        /// (heard within door_claim_lease_s) or behind any cleared doorway. Routes may still pass through such a room to reach
        /// others; only goals inside it are dropped. Walls across the doorways cut thoroughfare rooms off in headless house 5002
        /// (coverage 86 -> 71 %, standing 35 -> 98 s).</summary>
        internal bool DoorBlocksGoal(Vector2 p, double t)
        {
            if (!c.door_claims) return false;
            foreach (var d in doors)
            {
                bool block = d.state == DoorCleared || (d.owner != id && d.state == DoorTaken && t - d.heardAt <= c.door_claim_lease_s);
                if (block && InsideDoorRoom(d, p)) return true;
            }
            return false;
        }

        /// <summary>Whether point p lies in the room behind doorway d: the same room of this robot's own ledger as the floor just
        /// inside the doorway; without a ledger room, on the inward side within door_room_reach_m. A half-plane alone counted
        /// half the house as behind every doorway.</summary>
        bool InsideDoorRoom(Door d, Vector2 at)
        {
            int here = planner.Rooms.RoomAt(at), room = planner.Rooms.RoomAt(d.position + d.inward * c.door_cross_m);
            if (here >= 0 && room >= 0) return here == room;
            // Without ledger rooms: on the inward side, within reach, and in a clear straight line of known free floor from just
            // inside the doorway. Headless 5021: without the line test a study through the wall from a cleared bedroom counted as
            // behind its doorway, and its person (s5) was never found.
            return Vector2.Dot(at - d.position, d.inward) > 0 && Vector2.Distance(at, d.position) <= c.door_room_reach_m &&
                   grid.OwnFreeLine(d.position + d.inward * grid.CellM * 3, at);
        }

        /// <summary>door_enter: doorways in this robot's own map within door_scan_m, found with the same narrow-cut test as the
        /// doorway underfoot, are listed as seen; the room is taken to lie on the far side from the robot. Headless houses
        /// 5001-5022: 14 of 63 rooms were never entered, and in 5022 robot A confirmed four people from the hallway without
        /// stepping into a room, because floor seen through a doorway leaves no frontier inside.</summary>
        void ScanDoorways(double t)
        {
            if (!c.door_enter || t < nextDoorScan) return;
            nextDoorScan = t + c.door_scan_period_s;
            var at = motion.Pose.position; float r = c.door_scan_m, step = c.door_scan_step_m;
            for (float dx = -r; dx <= r; dx += step)
                for (float dy = -r; dy <= r; dy += step)
                {
                    if (dx * dx + dy * dy > r * r) continue;
                    var p = at + new Vector2(dx, dy); int cell = grid.Cell(p);
                    if (cell < 0 || grid.OwnOccupancy(cell) != 1) continue;
                    if (!InDoorway(p, Vector2.right, out var middle, out var axis) && !InDoorway(p, Vector2.up, out middle, out axis)) continue;
                    // One doorway is listed once: its narrowest cut shifts along the wall as more of the wall is mapped (headless houses
                    // 5001-5021, first 15 s: 40 % of listings repeated a doorway 0.8 to 1.2 m from its first one).
                    if (Vector2.Distance(middle, home) <= c.door_home_clear_m || doors.Exists(d => Vector2.Distance(d.position, middle) <= c.door_seen_merge_m)) continue;
                    var inward = Vector2.Dot(axis, middle - at) >= 0 ? axis : -axis;
                    doors.Add(new Door { position = middle, inward = inward, state = DoorSeen, since = t, heardAt = t });
                    Emit(t, "door_seen", "A doorway in this robot's own map: a room to walk into.", "x_m", middle.x, "y_m", middle.y,
                        "inward_x", inward.x, "inward_y", inward.y);
                }
        }

        bool BeenNear(Vector2 p) => trail.Exists(q => Vector2.Distance(q, p) <= c.door_been_m);

        /// <summary>Stay in the room this robot took until door_done_share of it is seen or door_room_max_s has passed.</summary>
        bool StayInTakenRoom(double t)
        {
            var at = motion.Pose.position;
            foreach (var d in doors)
            {
                if (d.owner != id || d.state != DoorTaken || !InsideDoorRoom(d, at)) continue;
                if (t - d.since >= c.door_room_max_s) return false;
                var room = planner.Rooms.Get(planner.Rooms.RoomAt(at));
                return room == null || room.coveredShare < c.door_done_share;
            }
            return false;
        }

        /// <summary>The nearest listed doorway this robot has not been through, that no robot has taken or cleared and the
        /// partner is not heading for: a route to door_enter_depth_m inside it (or just past the doorway when the floor
        /// deeper in is not yet known). Radioed as heading, so the partner picks another doorway; the first to head for one
        /// keeps it.</summary>
        bool EnterDoorway(double t, out List<Vector2> route)
        {
            route = null; Door chosen = null; float best = float.PositiveInfinity;
            if (!c.door_enter || StayInTakenRoom(t)) return false;
            var at = motion.Pose.position;
            var candidates = doors.FindAll(d =>
                (d.state == DoorSeen || (d.state == DoorHeading && (d.owner == id || t - d.heardAt > c.door_heading_lease_s))) &&
                !BeenNear(d.position + d.inward * c.door_enter_depth_m) && !DoorBlocksGoal(d.position + d.inward * c.door_enter_depth_m, t) &&
                !doors.Exists(o => (o.state == DoorTaken || o.state == DoorCleared) && Vector2.Distance(o.position, d.position) <= c.door_seen_merge_m) &&
                !GoalBlacklisted(d.position, t));
            candidates.Sort((a, b) => Vector2.Distance(a.position, at).CompareTo(Vector2.Distance(b.position, at)));
            int plans = 0;
            foreach (var d in candidates)
            {
                if (Vector2.Distance(d.position, at) >= best || plans >= c.door_enter_max_plans) break;
                foreach (float depth in new[] { c.door_enter_depth_m, c.door_cross_m + grid.CellM })
                {
                    plans++;
                    var p = planner.Plan(at, d.position + d.inward * depth);
                    if (p.Count == 0) continue;
                    float m = PathLength(p);
                    if (m < best) { best = m; route = p; chosen = d; }
                    break;
                }
            }
            if (chosen == null) return false;
            if (doorTarget != null && doorTarget != chosen && doorTarget.owner == id && doorTarget.state == DoorHeading) { doorTarget.state = DoorSeen; doorTarget.owner = null; }
            if (doorTarget != chosen) { chosen.since = t; Emit(t, "door_heading", "Nearest doorway not yet entered: walk into its room; the partner hears this robot is heading there.",
                "x_m", chosen.position.x, "y_m", chosen.position.y, "path_m", best); }
            doorTarget = chosen; chosen.state = DoorHeading; chosen.owner = id; chosen.heardAt = t; doorSweep = true;
            return true;
        }

        internal BrainDoor[] DoorsToSend()
        {
            var own = doors.FindAll(d => d.owner == id && d.state != DoorSeen);
            if (own.Count > BrainMessage.MaxDoors) { own.Sort((a, b) => b.since.CompareTo(a.since)); own = own.GetRange(0, BrainMessage.MaxDoors); }
            return own.ConvertAll(d => new BrainDoor { position = d.position, inward = d.inward, owner = d.owner, state = d.state, since = d.since }).ToArray();
        }

        /// <summary>The partner's doorways, already converted into this frame.</summary>
        internal void ReceiveDoors(IEnumerable<BrainDoor> received, double t)
        {
            foreach (var r in received)
            {
                var known = doors.Find(d => Vector2.Distance(d.position, r.position) <= c.door_match_m);
                if (known == null) { doors.Add(new Door { position = r.position, inward = r.inward, owner = r.owner, state = r.state, since = r.since, heardAt = t }); continue; }
                if (known.owner == id && known.state == DoorCleared) continue;   // this robot's own cleared room stays cleared
                // Both heading for one doorway: the first to choose it keeps it (equal times: the lower id).
                if (known.owner == id && known.state == DoorHeading && r.state == DoorHeading &&
                    !(r.since < known.since || (r.since == known.since && string.CompareOrdinal(r.owner, id) < 0))) continue;
                if (DoorRank(r.state) >= DoorRank(known.state) || known.owner == r.owner)
                { known.owner = r.owner; known.state = DoorRank(r.state) >= DoorRank(known.state) ? r.state : known.state; known.since = r.since; known.inward = r.inward; }
                known.heardAt = t;
            }
        }

        /// <summary>Taken and cleared doorways this robot knows, for logs and the agent.</summary>
        public int DoorCount(byte state) => doors.FindAll(d => d.state == state).Count;
    }
}
