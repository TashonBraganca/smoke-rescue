using System;
using System.Collections.Generic;
using UnityEngine;

namespace SmokeRescue.Brain
{
    /// <summary>Room estimates from this robot's sensed free floor. Unknown cells bound the distance transform.
    /// Eroded cores split at narrow openings, then a multi-source flood assigns the remaining free floor.
    /// Labels are planning hints. They never supply occupancy, indoor evidence, or camera coverage.</summary>
    public sealed class RoomLedger
    {
        public sealed class Room
        {
            public int id;
            public Vector2 centre;
            public float coveredShare, indoorFloorM2;
            public double dwellS;
            public bool entered;
            /// <summary>Twice the widest clearance inside the room, floor area over that width, and the number of other rooms it
            /// touches across at least one body width. A corridor is long, narrow and opens onto several rooms.</summary>
            public float widthM, lengthM;
            public int openings;
            public bool corridor;
            /// <summary>The room's own cell nearest its centroid (the centroid of an L-shaped room can lie in a wall), and the
            /// farthest room cell from it. A radio room claim names the room by these.</summary>
            public Vector2 anchor;
            public float radiusM;
        }

        readonly OccupancyGrid grid;
        readonly BrainConfig config;
        double[] dwellByCell;
        bool[] enteredCells;
        readonly List<Room> rooms = new List<Room>();
        readonly Dictionary<int, Room> byId = new Dictionary<int, Room>();
        int[] labels;
        int nextId, lastCell = -1;
        double lastT = double.NaN, nextSegmentation = double.NegativeInfinity;
        public IReadOnlyList<Room> Rooms => rooms;
        public double LastSegmentationAt { get; private set; } = double.NegativeInfinity;

        public RoomLedger(OccupancyGrid map, BrainConfig settings)
        {
            grid = map ?? throw new ArgumentNullException(nameof(map));
            config = settings ?? throw new ArgumentNullException(nameof(settings));
        }

        /// <summary>Credit elapsed time to the previous sensed pose cell. Repeated calls at one timestamp are safe.
        /// Per-cell visit history survives room splits and merges without marking every child as entered.</summary>
        public void Update(double t, Vector2 at, Func<Vector2, bool> indoors = null)
        {
            if (!config.room_ledger) return;
            BrainMath.Time(t);
            if (labels == null)
            {
                labels = new int[grid.Count];
                for (int i = 0; i < labels.Length; i++) labels[i] = -1;
                dwellByCell = new double[grid.Count]; enteredCells = new bool[grid.Count];
            }
            if (!double.IsNaN(lastT) && t < lastT) throw new ArgumentException("Room ledger timestamps must not decrease.");
            if (!double.IsNaN(lastT) && t == lastT) return;
            if (lastCell >= 0 && !double.IsNaN(lastT))
            {
                double elapsed = t - lastT; dwellByCell[lastCell] += elapsed;
                if (byId.TryGetValue(labels[lastCell], out var previous)) previous.dwellS += elapsed;
            }
            int cell = grid.Cell(at);
            if (cell >= 0 && grid.OwnOccupancy(cell) == 1) enteredCells[cell] = true;
            if (t >= nextSegmentation)
            {
                Rebuild(indoors ?? (p => !config.search_requires_ceiling || grid.Indoors(p)));
                LastSegmentationAt = t; nextSegmentation = t + config.room_ledger_period_s;
            }
            if (cell >= 0 && grid.OwnOccupancy(cell) == 1 && byId.TryGetValue(labels[cell], out var current)) current.entered = true;
            lastCell = cell >= 0 && grid.OwnOccupancy(cell) == 1 ? cell : -1; lastT = t;
        }

        public Room Get(int id) => byId.TryGetValue(id, out var room) ? room : null;

        public int RoomAt(Vector2 at)
        {
            int cell = grid.Cell(at);
            return labels != null && cell >= 0 && grid.OwnOccupancy(cell) == 1 ? labels[cell] : -1;
        }

        void Neighbours(int cell, Action<int> visit)
        {
            int x = cell % grid.Size, y = cell / grid.Size;
            if (x > 0) visit(cell - 1);
            if (x + 1 < grid.Size) visit(cell + 1);
            if (y > 0) visit(cell - grid.Size);
            if (y + 1 < grid.Size) visit(cell + grid.Size);
        }

        void Rebuild(Func<Vector2, bool> indoors)
        {
            var clearance = grid.KnownFreeClearance();
            var free = new bool[grid.Count]; var core = new bool[grid.Count]; var next = new int[grid.Count];
            for (int i = 0; i < grid.Count; i++)
            {
                free[i] = grid.OwnOccupancy(i) == 1;
                core[i] = free[i] && clearance[i] > config.room_ledger_door_width_m / 2;
                next[i] = -1;
            }
            int count = 0; var queue = new Queue<int>();
            for (int i = 0; i < grid.Count; i++)
            {
                if (!core[i] || next[i] >= 0) continue;
                int label = count++; next[i] = label; queue.Enqueue(i);
                while (queue.Count > 0)
                {
                    int cell = queue.Dequeue();
                    Neighbours(cell, n => { if (core[n] && next[n] < 0) { next[n] = label; queue.Enqueue(n); } });
                }
            }
            // Seed every core cell together. Equal-distance growth keeps row-major order.
            for (int i = 0; i < grid.Count; i++) if (next[i] >= 0) queue.Enqueue(i);
            while (queue.Count > 0)
            {
                int cell = queue.Dequeue();
                Neighbours(cell, n => { if (free[n] && next[n] < 0) { next[n] = next[cell]; queue.Enqueue(n); } });
            }
            // A sensed strip narrower than a door still needs a label, even if no room core exists there.
            for (int i = 0; i < grid.Count; i++)
            {
                if (!free[i] || next[i] >= 0) continue;
                int label = count++; next[i] = label; queue.Enqueue(i);
                while (queue.Count > 0)
                {
                    int cell = queue.Dequeue();
                    Neighbours(cell, n => { if (free[n] && next[n] < 0) { next[n] = label; queue.Enqueue(n); } });
                }
            }
            var overlap = new Dictionary<(int label, int id), int>();
            for (int i = 0; i < grid.Count; i++)
            {
                if (next[i] < 0 || labels[i] < 0) continue;
                var key = (next[i], labels[i]); overlap[key] = overlap.TryGetValue(key, out int n) ? n + 1 : 1;
            }
            var matches = new List<(int label, int id, int cells)>();
            foreach (var pair in overlap) matches.Add((pair.Key.label, pair.Key.id, pair.Value));
            matches.Sort((a, b) => { int k = b.cells.CompareTo(a.cells); if (k != 0) return k;
                k = a.id.CompareTo(b.id); return k != 0 ? k : a.label.CompareTo(b.label); });
            var ids = new int[count]; for (int i = 0; i < count; i++) ids[i] = -1;
            var used = new HashSet<int>();
            foreach (var match in matches)
                if (ids[match.label] < 0 && used.Add(match.id)) ids[match.label] = match.id;
            rooms.Clear(); byId.Clear();
            for (int i = 0; i < count; i++)
            {
                if (ids[i] < 0) ids[i] = nextId++;
                var room = new Room { id = ids[i] }; rooms.Add(room); byId.Add(room.id, room);
            }
            var floor = new int[count]; var indoorCount = new int[count]; var searched = new int[count];
            for (int i = 0; i < grid.Count; i++)
            {
                int label = next[i]; if (label < 0) continue;
                var room = rooms[label]; room.centre += grid.Centre(i); floor[label]++;
                room.entered |= enteredCells[i]; room.dwellS += dwellByCell[i];
                if (indoors(grid.Centre(i)))
                {
                    indoorCount[label]++;
                    if (grid.OwnObserved(i) || grid.PeerExplored(i)) searched[label]++;
                }
                next[i] = ids[label];
            }
            for (int i = 0; i < count; i++)
            {
                rooms[i].centre /= floor[i]; rooms[i].indoorFloorM2 = indoorCount[i] * grid.CellM * grid.CellM;
                rooms[i].coveredShare = indoorCount[i] > 0 ? searched[i] / (float)indoorCount[i] : 0;
            }
            labels = next;
            Shape(clearance, free);
        }

        /// <summary>Width, length, openings, corridor flag, anchor and radius of every room, from labels and clearance.</summary>
        void Shape(float[] clearance, bool[] free)
        {
            var widest = new Dictionary<int, float>(); var area = new Dictionary<int, int>();
            var contact = new Dictionary<(int, int), int>(); var nearest = new Dictionary<int, float>();
            foreach (var room in rooms) { widest[room.id] = 0; area[room.id] = 0; nearest[room.id] = float.PositiveInfinity; }
            for (int i = 0; i < grid.Count; i++)
            {
                int id = labels[i]; if (id < 0 || !free[i]) continue;
                widest[id] = Math.Max(widest[id], clearance[i]); area[id]++;
                var room = byId[id]; float d = Vector2.Distance(grid.Centre(i), room.centre);
                if (d < nearest[id]) { nearest[id] = d; room.anchor = grid.Centre(i); }
                int x = i % grid.Size, y = i / grid.Size;
                // Right and up neighbours only, so each touching pair of cells is counted once.
                if (x + 1 < grid.Size) Touch(contact, id, i + 1, free);
                if (y + 1 < grid.Size) Touch(contact, id, i + grid.Size, free);
            }
            foreach (var room in rooms) { room.openings = 0; room.radiusM = 0; }
            foreach (var pair in contact)
                if (pair.Value * grid.CellM >= 2 * config.robot_radius_m) { byId[pair.Key.Item1].openings++; byId[pair.Key.Item2].openings++; }
            for (int i = 0; i < grid.Count; i++)
            {
                int id = labels[i]; if (id < 0 || !free[i]) continue;
                var room = byId[id]; room.radiusM = Math.Max(room.radiusM, Vector2.Distance(grid.Centre(i), room.anchor));
            }
            foreach (var room in rooms)
            {
                // KnownFreeClearance is measured from the cell centre to the nearest cell that is not known free, less half a
                // cell, so twice the widest clearance is the free width across the room.
                room.widthM = 2 * (widest[room.id] + grid.CellM / 2);
                room.lengthM = area[room.id] * grid.CellM * grid.CellM / Math.Max(room.widthM, grid.CellM);
                room.corridor = room.widthM <= config.corridor_max_width_m && room.lengthM >= config.corridor_min_elongation * room.widthM &&
                    room.openings >= config.corridor_min_openings;
            }
        }

        void Touch(Dictionary<(int, int), int> contact, int id, int n, bool[] free)
        {
            if (!free[n] || labels[n] < 0 || labels[n] == id) return;
            var key = id < labels[n] ? (id, labels[n]) : (labels[n], id);
            contact[key] = contact.TryGetValue(key, out int k) ? k + 1 : 1;
        }

        /// <summary>A frontier just short of a doorway can serve known unsearched floor on the other side.
        /// Attribute it only across a clear line of this robot's sensed free cells, within the existing peek range.</summary>
        public int RoomForFrontier(Vector2 at)
        {
            int own = RoomAt(at);
            if (labels == null || config.frontier_peek_m <= 0) return own;
            int start = grid.Cell(at); if (start < 0) return -1;
            int reach = Mathf.CeilToInt(config.frontier_peek_m / grid.CellM);
            int sx = start % grid.Size, sy = start / grid.Size;
            var counts = new Dictionary<int, int>();
            for (int y = Math.Max(0, sy - reach); y <= Math.Min(grid.Size - 1, sy + reach); y++)
                for (int x = Math.Max(0, sx - reach); x <= Math.Min(grid.Size - 1, sx + reach); x++)
                {
                    int i = y * grid.Size + x, id = labels[i]; var point = grid.Centre(i);
                    if (id < 0 || grid.OwnOccupancy(i) != 1 || !grid.Unsearched(i) ||
                        Vector2.Distance(at, point) > config.frontier_peek_m || !grid.OwnFreeLine(at, point)) continue;
                    counts[id] = counts.TryGetValue(id, out int n) ? n + 1 : 1;
                }
            int best = own, most = counts.TryGetValue(own, out int local) ? local : 0;
            foreach (var pair in counts)
                if (pair.Value >= config.frontier_min_information_cells &&
                    (pair.Value > most || (pair.Value == most && pair.Key < best))) { best = pair.Key; most = pair.Value; }
            return best;
        }

        /// <summary>Complete the current room while its coverage and time budgets permit. After that, unentered rooms
        /// precede entered rooms. Within a group, missing coverage earns up to one dwell budget against travel seconds.
        /// Only candidates that the planner already admitted can move in this ordering.</summary>
        public void Rank(List<GridPlanner.FrontierChoice> choices, Vector2 at)
        {
            if (!config.room_ledger || rooms.Count == 0) return;
            int current = RoomAt(at);
            bool finishing = byId.TryGetValue(current, out var here) && here.indoorFloorM2 > 0 &&
                here.coveredShare < config.room_ledger_covered_share && here.dwellS < config.room_ledger_dwell_s;
            var order = new Dictionary<GridPlanner.FrontierChoice, int>();
            for (int i = 0; i < choices.Count; i++)
            {
                var f = choices[i]; order[f] = i; f.roomId = RoomForFrontier(f.position);
                bool known = byId.TryGetValue(f.roomId, out var room) && room.indoorFloorM2 > 0;
                f.roomTier = !known ? 3 : finishing ? (f.roomId == current ? 0 : 1) :
                    !room.entered ? 0 : f.roomId == current ? 2 : 1;
                f.roomScore = f.cost / config.open_speed_mps + (known ? room.coveredShare * config.room_ledger_dwell_s : 0);
            }
            choices.Sort((a, b) =>
            {
                int k = a.roomTier.CompareTo(b.roomTier); if (k != 0) return k;
                k = a.roomScore.CompareTo(b.roomScore); if (k != 0) return k;
                k = a.score.CompareTo(b.score); return k != 0 ? k : order[a].CompareTo(order[b]);
            });
        }
    }
}
