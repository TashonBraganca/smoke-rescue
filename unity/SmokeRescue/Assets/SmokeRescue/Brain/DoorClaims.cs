using System;
using System.Collections.Generic;
using UnityEngine;

namespace SmokeRescue.Brain
{
    /// <summary>
    /// door_claims (owner, 24 Oct): a robot that walks through a doorway into a room records the doorway point and radios
    /// "taken" by it; when it walks back out it radios "cleared". The partner never routes through a doorway the other robot
    /// has taken, and nobody routes through a cleared one, so no room is searched twice. A doorway is found in the robot's own
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
        Vector2? doorwayAt; Vector2 doorwayAxis; double nextDoorCheck = double.NegativeInfinity, nextDoorBlocks = double.NegativeInfinity;
        readonly List<(Vector2 a, Vector2 b)> blockedDoors = new List<(Vector2, Vector2)>();

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

        /// <summary>The gap across the heading at p: its width and its middle; with the widths 0.8 m (door_cross_m) ahead and behind.</summary>
        bool InDoorway(Vector2 p, Vector2 axis, out Vector2 middle)
        {
            middle = p; var side = new Vector2(-axis.y, axis.x);
            float l = FreeRun(p, side, c.door_max_width_m), r = FreeRun(p, -side, c.door_max_width_m), w = l + r;
            if (w < c.door_min_width_m || w > c.door_max_width_m) return false;
            float reach = c.door_max_width_m + c.door_side_extra_m;
            foreach (float k in new[] { 1f, -1f })
            {
                var q = p + axis * (k * c.door_cross_m);
                if (FreeRun(q, side, reach) + FreeRun(q, -side, reach) < w + c.door_side_extra_m) return false;
            }
            middle = p + side * ((l - r) / 2); return true;
        }

        void UpdateDoors(double t)
        {
            if (!c.door_claims || t < nextDoorCheck) return;
            nextDoorCheck = t + .1;
            var at = motion.Pose.position; var forward = motion.Pose.Forward;
            if (InDoorway(at, forward, out var middle))
            {
                if (!doorwayAt.HasValue) { doorwayAt = middle; doorwayAxis = forward; }
            }
            else if (doorwayAt.HasValue)
            {
                float along = Vector2.Dot(at - doorwayAt.Value, doorwayAxis);
                if (Mathf.Abs(along) >= c.door_cross_m) { Crossed(t, doorwayAt.Value, doorwayAxis * Mathf.Sign(along)); doorwayAt = null; }
                else if (Vector2.Distance(at, doorwayAt.Value) > c.door_cross_m * 2) doorwayAt = null;
            }
            if (t >= nextDoorBlocks) { nextDoorBlocks = t + .5; ApplyDoorBlocks(t); }
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
            doors.Add(new Door { position = door, inward = direction.normalized, owner = id, state = DoorTaken, since = t, heardAt = t });
            Emit(t, "door_taken", "Walked through a doorway into a room: this robot takes it; the partner does not go in.",
                "x_m", door.x, "y_m", door.y, "inward_x", direction.x, "inward_y", direction.y);
        }

        /// <summary>Doorways this robot must not pass, as short walls across them: the partner's taken doorways (heard within
        /// door_claim_lease_s) and every cleared one. A robot standing on the inward side of a doorway may always walk out.</summary>
        void ApplyDoorBlocks(double t)
        {
            var at = motion.Pose.position; var walls = new List<(Vector2, Vector2)>();
            foreach (var d in doors)
            {
                bool block = d.state == DoorCleared || (d.owner != id && d.state == DoorTaken && t - d.heardAt <= c.door_claim_lease_s);
                if (!block || Vector2.Dot(at - d.position, d.inward) > 0) continue;
                var side = new Vector2(-d.inward.y, d.inward.x) * (c.door_max_width_m * .6f);
                walls.Add((d.position - side, d.position + side));
            }
            if (walls.Count == blockedDoors.Count && walls.TrueForAll(w => blockedDoors.Contains(w))) return;
            blockedDoors.Clear(); blockedDoors.AddRange(walls);
            grid.SetDoorBlocks(blockedDoors);
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
