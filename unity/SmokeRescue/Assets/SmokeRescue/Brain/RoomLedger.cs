using System;
using System.Collections.Generic;
using System.Linq;
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
            /// <summary>Enclosed by mapped walls and partly proven indoors: all its floor counts as search area.</summary>
            public bool promoted;
            public float floorM2, wallShare;
            /// <summary>The room's own cell nearest its centroid (the centroid of an L-shaped room can lie in a wall), and the
            /// farthest room cell from it. A radio room claim names the room by these.</summary>
            public Vector2 anchor;
            public float radiusM;
            /// <summary>partner_rooms: a partner's radioed body stood on this room's floor (partnerEntered), or stands there now
            /// (partnerInside, set at each ranking from poses no older than peer_claim_s).</summary>
            public bool partnerEntered, partnerInside;
        }

        readonly OccupancyGrid grid;
        readonly BrainConfig config;
        double[] dwellByCell;
        bool[] enteredCells, partnerCells;
        /// <summary>Partner bodies now, in this robot's frame, with the partner's ID and its radioed goal (null when it sent
        /// none); the brain refreshes it every step.</summary>
        public readonly List<(string id, Vector2 at, Vector2? goal)> PartnersNow = new List<(string, Vector2, Vector2?)>();
        /// <summary>This robot's ID, for the tie-break when two robots stand in one room.</summary>
        public string SelfId = "";
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
                dwellByCell = new double[grid.Count]; enteredCells = new bool[grid.Count]; partnerCells = new bool[grid.Count];
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
                RecordProgress(t);
            }
            if (cell >= 0 && grid.OwnOccupancy(cell) == 1 && byId.TryGetValue(labels[cell], out var current))
            {
                current.entered = true;
                if (!entryById.ContainsKey(current.id)) entryById[current.id] = at;   // where this robot first stood in the room
            }
            lastCell = cell >= 0 && grid.OwnOccupancy(cell) == 1 ? cell : -1; lastT = t;
        }

        readonly Dictionary<int, Vector2> entryById = new Dictionary<int, Vector2>();
        public Room Get(int id) => byId.TryGetValue(id, out var room) ? room : null;
        /// <summary>A partner's radioed pose, converted into this frame: the floor under its body is floor it stood on.</summary>
        public void MarkPartner(Vector2 at, float radius)
        {
            if (partnerCells == null) { partnerCells = new bool[grid.Count]; }
            int cell = grid.Cell(at); if (cell < 0) return;
            int r = Mathf.CeilToInt(radius / grid.CellM), cx = cell % grid.Size, cy = cell / grid.Size;
            for (int y = Math.Max(0, cy - r); y <= Math.Min(grid.Size - 1, cy + r); y++)
                for (int x = Math.Max(0, cx - r); x <= Math.Min(grid.Size - 1, cx + r); x++)
                    if (Vector2.Distance(grid.Centre(y * grid.Size + x), at) <= radius) partnerCells[y * grid.Size + x] = true;
        }
        /// <summary>The room at p, or the nearest labelled cell within three cells (two maps place one wall a cell or two apart).</summary>
        /// <summary>partner_rooms applies only to a room one robot finishes within room_ledger_dwell_s (its floor over the strip a
        /// camera sweeps per second, as DwellBudget): about 48 m2. A larger labelled space is open plan or several rooms the
        /// ledger has not split, and is shared. Headless 5001 (24 Oct): a 9 m radius space left to robot A alone held it from
        /// 30 to 108 s.</summary>
        public bool OneRobotRoom(Room r) => r.indoorFloorM2 > 0 && DwellBudget(r) <= config.room_ledger_dwell_s;
        public int RoomNear(Vector2 p)
        {
            int room = RoomAt(p); if (room >= 0) return room;
            float best = float.PositiveInfinity;
            for (int dy = -3; dy <= 3; dy++) for (int dx = -3; dx <= 3; dx++)
            {
                var q = p + new Vector2(dx, dy) * grid.CellM; int r = RoomAt(q); float d = Vector2.Distance(p, q);
                if (r >= 0 && d < best) { best = d; room = r; }
            }
            return room;
        }
        /// <summary>Is p on the floor of a promoted room (see Shape)?</summary>
        public bool Promoted(Vector2 p) { var r = Get(RoomAt(p)); return r != null && r.promoted; }
        /// <summary>Time to finish a room: the dwell budget, or longer for a big room, its floor area over the strip a camera
        /// sweeps per second (search_max_range_m wide at near_speed_mps). Headless seed 5001: a 102 m2 bedroom got the same 40 s
        /// as a 7 m2 closet.</summary>
        public double DwellBudget(Room r) => Math.Max(config.room_ledger_dwell_s, r.indoorFloorM2 / (config.search_max_range_m * config.near_speed_mps));

        /// <summary>room_progress_dwell: searched floor per room at each segmentation, over the last room_progress_window_s.</summary>
        readonly Dictionary<int, List<(double t, float m2)>> searchedHistory = new Dictionary<int, List<(double, float)>>();
        public void RecordProgress(double t)
        {
            foreach (var r in rooms)
            {
                if (!searchedHistory.TryGetValue(r.id, out var h)) searchedHistory[r.id] = h = new List<(double, float)>();
                h.Add((t, r.coveredShare * r.indoorFloorM2));
                h.RemoveAll(e => t - e.t > config.room_progress_window_s);
            }
        }
        /// <summary>Is the room still giving new floor? Searched floor grew over the last room_progress_window_s at
        /// room_progress_min_rate_fraction or more of the camera's sweep rate (search_max_range_m wide at near_speed_mps).
        /// Headless house 5003 (24 Oct): robot A left the kitchen when its 40 s dwell budget ran out with 548 unseen cells
        /// beside its frontier, and walked back 49 s later. A room is left when it stops yielding floor, not on a clock.</summary>
        public bool StillGaining(Room r)
        {
            if (!config.room_progress_dwell || r == null || !searchedHistory.TryGetValue(r.id, out var h) || h.Count < 2) return false;
            double span = h[h.Count - 1].t - h[0].t; if (span < config.room_progress_window_s / 2) return false;
            double rate = (h[h.Count - 1].m2 - h[0].m2) / span;
            return rate >= config.room_progress_min_rate_fraction * config.search_max_range_m * config.near_speed_mps;
        }

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
            if (config.room_ledger_relative_doors) count = RelativeCores(free, clearance, next);
            else
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
            // A sensed strip narrower than a door still needs a label, even if no room core exists there. With relative doors a
            // strip smaller than room_ledger_min_room_m2 is no room at all (seed 934: 126 of robot A's 133 rooms were under
            // 1 m2, median one cell, each with a new id, ranked, claimed and shown to the agent).
            var strip = new List<int>();
            for (int i = 0; i < grid.Count; i++)
            {
                if (!free[i] || next[i] >= 0) continue;
                int label = count++; next[i] = label; queue.Enqueue(i); strip.Clear();
                while (queue.Count > 0)
                {
                    int cell = queue.Dequeue(); strip.Add(cell);
                    Neighbours(cell, n => { if (free[n] && next[n] < 0) { next[n] = label; queue.Enqueue(n); } });
                }
                if ((config.room_ledger_relative_doors || config.room_ledger_drop_slivers) && strip.Count * grid.CellM * grid.CellM < config.room_ledger_min_room_m2)
                { foreach (int cell in strip) next[cell] = -2; count--; }
            }
            for (int i = 0; i < grid.Count; i++) if (next[i] == -2) next[i] = -1;
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
                if (partnerCells != null && partnerCells[i]) room.partnerEntered = true;
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
            Shape(clearance, free, indoors);
        }

        /// <summary>Room cores by hierarchical erosion (distance-transform room segmentation; Bormann et al., ICRA 2016, "Room
        /// segmentation: survey, implementation, and analysis"): free floor is cut at rising clearance thresholds, and a region
        /// becomes two rooms only where it splits into two cores of at least room_ledger_min_core_m2. A doorway is then any
        /// narrowing between two wider spaces, whatever its width; a corridor that thins away without splitting stays one room.
        /// Seed 934: the foyer, living room and hallway openings were 1.2 to 1.6 m wide, at or above the fixed 1.2 m door, so the
        /// three spaces were one room and neither robot took a side door.</summary>
        int RelativeCores(bool[] free, float[] clearance, int[] next)
        {
            float step = Mathf.Max(grid.CellM, config.room_ledger_split_step_m);
            int minCore = Mathf.Max(1, Mathf.CeilToInt(config.room_ledger_min_core_m2 / (grid.CellM * grid.CellM)));
            var cores = new List<List<int>>();
            var start = Components(free, clearance, config.room_ledger_split_min_m, null);
            var work = new Stack<(List<int> cells, float level)>();
            foreach (var comp in start) if (comp.Count >= minCore) work.Push((comp, config.room_ledger_split_min_m));
            while (work.Count > 0)
            {
                var (cells, level) = work.Pop(); bool split = false;
                for (float r = level + step; r <= config.room_ledger_split_max_m + 1e-4f; r += step)
                {
                    var parts = Components(free, clearance, r, cells).FindAll(part => part.Count >= minCore);
                    if (parts.Count == 0) break;
                    if (parts.Count >= 2) { foreach (var part in parts) work.Push((part, r)); split = true; break; }
                }
                if (!split) cores.Add(cells);
            }
            // Stable order (smallest first cell first), so labels and ids do not depend on stack order.
            cores.Sort((a, b) => a.Min().CompareTo(b.Min()));
            for (int k = 0; k < cores.Count; k++) foreach (int cell in cores[k]) next[cell] = k;
            return cores.Count;
        }
        /// <summary>4-connected components of free cells whose clearance exceeds level, within subset when given.</summary>
        List<List<int>> Components(bool[] free, float[] clearance, float level, List<int> subset)
        {
            var result = new List<List<int>>(); var seen = new HashSet<int>(); var queue = new Queue<int>();
            HashSet<int> inside = subset == null ? null : new HashSet<int>(subset);
            IEnumerable<int> cells = subset ?? Enumerable.Range(0, grid.Count);
            foreach (int i in cells)
            {
                if (!free[i] || clearance[i] <= level || !seen.Add(i)) continue;
                var comp = new List<int> { i }; queue.Enqueue(i);
                while (queue.Count > 0)
                {
                    int cell = queue.Dequeue();
                    Neighbours(cell, n => { if (free[n] && clearance[n] > level && (inside == null || inside.Contains(n)) && seen.Add(n)) { comp.Add(n); queue.Enqueue(n); } });
                }
                result.Add(comp);
            }
            return result;
        }

        /// <summary>Width, length, openings, corridor flag, anchor and radius of every room, from labels and clearance.</summary>
        void Shape(float[] clearance, bool[] free, Func<Vector2, bool> indoors)
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
            // Room-level indoor proof (room_indoor_promotion). A robot entering a room proves the floor near it indoors from
            // ceiling sight lines, while the LiDAR maps the whole room at once; the far side stayed unproven, was never offered as
            // a frontier, and the robot left after the near corner (headless seeds 5001 to 5004). A room whose own floor is partly
            // proven and whose edge is mostly mapped wall (enclosure_min_closed_share, the walled-in test's share) is one enclosed
            // space, so all its floor is search area. Edges to other rooms count neither way; a yard's edge is mostly unknown.
            var wall = new Dictionary<int, int>(); var open = new Dictionary<int, int>(); var proven = new Dictionary<int, int>();
            foreach (var room in rooms) { wall[room.id] = 0; open[room.id] = 0; proven[room.id] = 0; }
            for (int i = 0; i < grid.Count; i++)
            {
                int id = labels[i]; if (id < 0 || !free[i]) continue;
                if (indoors(grid.Centre(i))) proven[id]++;
                int x = i % grid.Size, y = i / grid.Size;
                for (int k = 0; k < 4; k++)
                {
                    int nx = x + (k == 0 ? 1 : k == 1 ? -1 : 0), ny = y + (k == 2 ? 1 : k == 3 ? -1 : 0);
                    if (nx < 0 || ny < 0 || nx >= grid.Size || ny >= grid.Size) { open[id]++; continue; }
                    int o = grid.OwnOccupancy(ny * grid.Size + nx);
                    if (o == 255) wall[id]++; else if (o == 0) open[id]++;
                }
            }
            foreach (var room in rooms)
            {
                room.floorM2 = area[room.id] * grid.CellM * grid.CellM;
                room.wallShare = wall[room.id] + open[room.id] > 0 ? wall[room.id] / (float)(wall[room.id] + open[room.id]) : 0;
                room.promoted = config.room_indoor_promotion && proven[room.id] >= config.frontier_min_information_cells &&
                    room.wallShare >= config.enclosure_min_closed_share;
            }
            if (config.room_indoor_promotion)
            {
                // A promoted room's coverage is measured on all its floor, not only the proven part.
                var searchedAll = new Dictionary<int, int>(); foreach (var room in rooms) searchedAll[room.id] = 0;
                for (int i = 0; i < grid.Count; i++)
                {
                    int id = labels[i]; if (id < 0 || !free[i] || !byId[id].promoted) continue;
                    if (grid.OwnObserved(i) || grid.PeerExplored(i)) searchedAll[id]++;
                }
                foreach (var room in rooms)
                    if (room.promoted) { room.indoorFloorM2 = room.floorM2; room.coveredShare = area[room.id] > 0 ? searchedAll[room.id] / (float)area[room.id] : 0; }
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

        /// <summary>Enter every room, sweep it, mark it done. Tiers, best first: (0) the room this robot stands in while it is
        /// unfinished and within its dwell budget; (1) rooms not yet entered; (2) corridor frontiers, since walking a corridor
        /// reveals the doors along it; (3) leftover floor in rooms already entered; (4) rooms already covered past
        /// room_ledger_covered_share; (5) floor the ledger has no room for. Within a tier, travel seconds plus covered share
        /// times the dwell budget. A corridor is never the room being finished. Headless seed 5001: ranked as an ordinary
        /// entered room, the corridor came last whenever a robot stood in it, so both robots kept returning to pockets in the
        /// living room and bedroom and never walked far enough up the corridor to see the kitchen door.
        /// Only candidates that the planner already admitted can move in this ordering.</summary>
        public void Rank(List<GridPlanner.FrontierChoice> choices, Vector2 at)
        {
            if (!config.room_ledger || rooms.Count == 0) return;
            int current = RoomAt(at);
            bool finishing = byId.TryGetValue(current, out var here) && here.indoorFloorM2 > 0 && !here.corridor &&
                here.coveredShare < config.room_ledger_covered_share && (here.dwellS < DwellBudget(here) || StillGaining(here));
            var order = new Dictionary<GridPlanner.FrontierChoice, int>();
            // partner_rooms: the room a partner is searching now: its body stands there and its goal, when it radioed one, lies
            // there too. A partner passing through (its goal in another room) holds nothing, and a corridor is the way to
            // every room off it, never one robot's room. Headless seed 934 with one door per room (24 Oct): with the hall
            // left to robot A whenever A stood in it, robot B's only options after the kitchen were the kitchen again, and it
            // went back in from 140 to 202 s. When both search one room, the higher ID yields it.
            var partnerNow = new HashSet<int>();
            foreach (var r in rooms) r.partnerInside = false;
            if (config.partner_rooms)
                foreach (var (pid, p, goal) in PartnersNow)
                {
                    int r = RoomNear(p); if (r < 0 || byId[r].corridor || !OneRobotRoom(byId[r])) continue;
                    if (goal.HasValue && RoomNear(goal.Value) != r) continue;
                    if (r == current && string.CompareOrdinal(SelfId, pid) < 0) continue;
                    partnerNow.Add(r); byId[r].partnerInside = true;
                }
            for (int i = 0; i < choices.Count; i++)
            {
                var f = choices[i]; order[f] = i; f.roomId = RoomForFrontier(f.position);
                bool known = byId.TryGetValue(f.roomId, out var room) && room.indoorFloorM2 > 0;
                // room_ledger_corridor_first: walk the corridor before the rooms off it, which reveals every door along it
                // (UK Operational Guidance, Breathing Apparatus, Appendix 5, directional search; Miami-Dade threshold search at each
                // door). Headless seed 5001: the kitchen door 17 m down the corridor was never seen from its lower half, so the
                // kitchen was never in either robot's ledger.
                int corridorTier = config.room_ledger_corridor_first ? 1 : 2, unenteredTier = config.room_ledger_corridor_first ? 2 : 1;
                // An unentered room stays unentered however covered it looks from the door: walking in sees behind the furniture.
                // Headless 24 Oct, houses 5001-5004: ranking covered rooms done before entry (all of them, or only those enclosed
                // by mapped walls) reached 23 and 24 of 31 rooms against 26, mean floor searched 69.2 and 74.0 % against 75.5 %.
                f.roomTier = !known ? 5 : finishing && f.roomId == current ? 0 : room.corridor ? corridorTier : !room.entered ? unenteredTier :
                    room.coveredShare >= config.room_ledger_covered_share ? 4 : 3;
                // frontier_beyond_room: a frontier in an entered room with far more unsearched cells beside it than the room has
                // left (its uncovered share of its floor) looks past the room through an opening onto space no robot has seen:
                // a new room, ranked as one. Headless house 5003 (24 Oct): robot A's frontier at the study door, 657 unseen
                // cells beyond it, stood in the kitchen's floor; the kitchen was covered, so the door ranked last for 20 s, and
                // A had left for another bedroom at 90 s and come back at 139 s. The study, the laundry and the bathroom past it
                // were reached at 192 s or never.
                bool beyond = false;
                if (config.frontier_beyond_room && known && !room.corridor && (f.roomTier == 3 || f.roomTier == 4))
                {
                    float leftCells = (1 - room.coveredShare) * room.indoorFloorM2 / (grid.CellM * grid.CellM);
                    if (f.information > leftCells + config.frontier_min_information_cells) { f.roomTier = unenteredTier; beyond = true; }
                }
                // partner_rooms (owner, 24 Oct: "it knows the coordinates of the other; it won't even go near that room"). A room a
                // partner stands in now is left to it: last tier, kept only as a fallback. A room a partner already walked
                // through is entered, never "unentered" again. Seed 934 (24 Oct 04:53): A walked into the kitchen four times
                // between 107 and 117 s while B stood in it, and B into rooms A had searched.
                if (config.partner_rooms && known)
                {
                    if (partnerNow.Contains(f.roomId)) f.roomTier = 4;
                    else if (room.partnerEntered && !room.corridor && f.roomTier == unenteredTier && OneRobotRoom(room))
                        f.roomTier = room.coveredShare >= config.room_ledger_covered_share ? 4 : 3;
                }
                // A frontier looking past its room carries the coverage of the space beyond, none: with its room's share it
                // lost to an unentered room 9 m away (headless seed 934 one door, 24 Oct, robot A 1.6 m from bedroom_1's door).
                f.roomScore = f.cost / config.open_speed_mps + (known && !beyond ? room.coveredShare * config.room_ledger_dwell_s : 0);
                // partner_territory (owner, 24 Oct: "one goes on the right, one goes on the left"): a frontier nearer a partner,
                // or the goal it radioed, than this robot is the partner's side and costs a room's dwell more, so two robots at
                // one fork take different ways; still a fallback when nothing else is left (Burgard et al., IEEE T-RO 2005,
                // coordinated multi-robot exploration: discount targets near another robot's target). Seed 934 (24 Oct
                // 04:53): both robots entered the house together; headless houses (tour session, 24 Oct): the same first
                // room in 3 of 8 houses.
                if (config.partner_territory)
                    foreach (var (pid, p, goal) in PartnersNow)
                    {
                        float mine = Vector2.Distance(f.position, at);
                        bool theirs = Vector2.Distance(f.position, p) < mine || (goal.HasValue && Vector2.Distance(f.position, goal.Value) < mine);
                        if (theirs) { f.roomScore += config.room_ledger_dwell_s; break; }
                    }
                // room_sweep (owner, 23 Oct): finish a room from the left side across to the right, as seen from where the robot
                // came in, then leave on the right. Frontiers in the room being finished cost extra the further right of the line
                // from the entry point to the room centre they lie, up to room_sweep_s; travel time still counts within a band.
                if (config.room_sweep && f.roomTier == 0 && entryById.TryGetValue(current, out var entry) && Vector2.Distance(here.centre, entry) > grid.CellM)
                {
                    float left = Vector2.SignedAngle(here.centre - entry, f.position - entry);   // positive to the left
                    f.roomScore += (90 - Mathf.Clamp(left, -90, 90)) / 180f * config.room_sweep_s;
                }
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
