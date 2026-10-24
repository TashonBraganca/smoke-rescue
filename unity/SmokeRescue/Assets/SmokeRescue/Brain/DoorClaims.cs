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
        internal const byte DoorTaken = 1, DoorCleared = 2;
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

        internal BrainDoor[] DoorsToSend()
        {
            var own = doors.FindAll(d => d.owner == id);
            if (own.Count > BrainMessage.MaxDoors) own = own.GetRange(own.Count - BrainMessage.MaxDoors, BrainMessage.MaxDoors);
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
                if (r.state >= known.state || known.owner == r.owner) { known.owner = r.owner; known.state = Math.Max(known.state, r.state); known.since = r.since; known.inward = r.inward; }
                known.heardAt = t;
            }
        }

        /// <summary>Taken and cleared doorways this robot knows, for logs and the agent.</summary>
        public int DoorCount(byte state) => doors.FindAll(d => d.state == state).Count;
    }
}
