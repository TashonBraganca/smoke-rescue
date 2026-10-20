using System;
using System.Collections.Generic;
using SmokeRescue.Sensors;
using UnityEngine;

namespace SmokeRescue.Brain
{
    /// <summary>Sensor-only log-odds map and maximum observed surface temperature layer.
    /// Two geometry layers: the depth camera (odds) and the planar LiDAR (lidarOdds). Navigation uses both: a cell is
    /// occupied if either layer says so, because each sees obstacles the other misses (the LiDAR plane passes over low
    /// debris; the camera on its mast cannot see the floor within about 2 m). "Observed" stays camera-only: it means
    /// searched, since only the cameras can see a person, so frontiers and the shared explored mask follow the cameras
    /// while the LiDAR makes the near field safe to drive.</summary>
    public sealed class OccupancyGrid
    {
        readonly BrainConfig c;
        readonly float[] odds, lidarOdds, heat, rise;
        readonly double[] heatTime;
        readonly bool[] hits, frees, observed;
        readonly byte[] roofHits;
        float[] roofDistance; bool roofDirty = true;
        int[] enclosureGeneration; bool[] enclosureValue; int generation;
        readonly double[] localTime, ownHeatTime;
        readonly float[] ownHeat;
        readonly SortedDictionary<string, byte[]> peerLayers = new SortedDictionary<string, byte[]>(StringComparer.Ordinal);
        readonly byte[] peer;
        double localObservationTime;
        float[] obstacleDistance, fireDistance;
        bool dirty = true;
        double time;
        public int Size { get; }
        public int Count => Size * Size;
        public float CellM => c.cell_m;
        public Vector2 Origin { get; }
        public OccupancyGrid(BrainConfig config)
        {
            c = config; Size = Mathf.CeilToInt(c.grid_size_m / c.cell_m);
            Origin = c.fixed_grid_origin ? new Vector2(c.grid_origin_x_m, c.grid_origin_z_m) :
                new Vector2(-Size * c.cell_m / 2, -Size * c.cell_m / 2);
            odds = new float[Count]; lidarOdds = new float[Count]; heat = new float[Count]; rise = new float[Count]; heatTime = new double[Count];
            hits = new bool[Count]; frees = new bool[Count]; observed = new bool[Count];
            localTime = new double[Count]; peer = new byte[Count]; roofHits = new byte[Count];
            ownHeat = new float[Count]; ownHeatTime = new double[Count];
            for (int i = 0; i < Count; i++) { heat[i] = c.ambient_c; heatTime[i] = double.NegativeInfinity; ownHeat[i] = c.ambient_c; ownHeatTime[i] = double.NegativeInfinity; }
        }
        public int Cell(Vector2 p)
        {
            int x = Mathf.FloorToInt((p.x - Origin.x) / CellM), y = Mathf.FloorToInt((p.y - Origin.y) / CellM);
            return Inside(x, y) ? y * Size + x : -1;
        }
        public Vector2 Centre(int index) => Origin + new Vector2(index % Size + .5f, index / Size + .5f) * CellM;
        public bool Inside(int x, int y) => x >= 0 && y >= 0 && x < Size && y < Size;
        public byte OwnOccupancy(int index)
        {
            if (index < 0 || index >= Count) return 0;
            if (odds[index] >= c.occupied_threshold || lidarOdds[index] >= c.occupied_threshold) return 255;
            return odds[index] <= c.free_threshold || lidarOdds[index] <= c.free_threshold ? (byte)1 : (byte)0;
        }
        public byte LidarOccupancy(int i) => i < 0 || i >= Count ? (byte)0 :
            lidarOdds[i] >= c.occupied_threshold ? (byte)255 : lidarOdds[i] <= c.free_threshold ? (byte)1 : (byte)0;
        /// <summary>Seen by this robot's own cameras, which is what "searched" means.</summary>
        public bool OwnObserved(int i) => i >= 0 && i < Count && observed[i];
        public bool PeerExplored(int i) => i >= 0 && i < Count && peer[i] != 0;
        /// <summary>No camera on the team has looked at this cell yet and it is not a known obstacle: a search target.</summary>
        public bool Unsearched(int i) => i >= 0 && i < Count && !observed[i] && peer[i] == 0 && OwnOccupancy(i) != 255;
        /// <summary>Own evidence first. A peer's map only fills cells this robot has no evidence for: the peer's walls
        /// arrive at 20 cm resolution and carry its own drift, and letting them override cells this robot has seen
        /// free closed 0.9 m doors, whose drivable gap is about 0.26 m.</summary>
        public byte Occupancy(int i)
        {
            if (i < 0 || i >= Count) return 0;
            byte own = OwnOccupancy(i);
            return own != 0 ? own : peer[i];
        }
        public byte OccupancyAt(Vector2 p) => Occupancy(Cell(p));
        public float Heat(int i) => heat[i];
        public void ObserveFree(int x, int y) { if (Inside(x, y)) ObserveFree(y * Size + x); }
        public void ObserveFree(int i)
        {
            if (i < 0 || i >= Count) return;
            odds[i] = Mathf.Max(-c.log_odds_cap, odds[i] + c.free_log_odds);
            observed[i] = true; localTime[i] = localObservationTime; dirty = roofDirty = true;
        }
        public void ObserveOccupied(int i)
        {
            if (i < 0 || i >= Count) return;
            odds[i] = Mathf.Min(c.log_odds_cap, Mathf.Max(0, odds[i]) + c.occupied_log_odds);
            observed[i] = true; localTime[i] = localObservationTime; dirty = roofDirty = true;
        }
        // Keep the inferred contact beyond the current inflated footprint, so replanning can start.
        internal int MarkStuckStrip(BrainPose pose)
        {
            float start = c.robot_radius_m + 2 * CellM, end = start + c.stuck_strip_depth_m;
            int centre = Cell(pose.position); if (centre < 0) return 0;
            int radius = Mathf.CeilToInt((end + c.robot_radius_m) / CellM), marked = 0;
            for (int dy = -radius; dy <= radius; dy++) for (int dx = -radius; dx <= radius; dx++)
            {
                int x = centre % Size + dx, y = centre / Size + dy; if (!Inside(x, y)) continue;
                int i = y * Size + x; var delta = Centre(i) - pose.position;
                float forward = Vector2.Dot(delta, pose.Forward);
                float side = pose.Forward.x * delta.y - pose.Forward.y * delta.x;
                if (forward < start || forward > end || Mathf.Abs(side) > c.robot_radius_m) continue;
                odds[i] = Mathf.Max(odds[i], c.stuck_log_odds);
                observed[i] = true; localTime[i] = time; marked++;
            }
            dirty = roofDirty = true; return marked;
        }
        /// <summary>toOwn maps the sender's frame into this one (FrameLink); each own cell looks up the sender cell under it.</summary>
        internal void ReplacePeerLayer(string sender, BrainMessage m, byte[] explored, BrainPose toOwn)
        {
            var layer = new byte[Count]; float cos = Mathf.Cos(-toOwn.yawRad), sin = Mathf.Sin(-toOwn.yawRad);
            for (int i = 0; i < Count; i++)
            {
                Vector2 d = Centre(i) - toOwn.position;
                int cell = m.Cell(new Vector2(cos * d.x - sin * d.y, sin * d.x + cos * d.y)); if (cell < 0) continue;
                if (BrainMessage.Bit(m.Occupied, cell)) layer[i] = 255;
                else if (BrainMessage.Bit(explored, cell)) layer[i] = 1;
            }
            peerLayers[sender] = layer;
            Array.Clear(peer, 0, peer.Length);
            foreach (var pair in peerLayers)
                for (int i = 0; i < Count; i++) if (pair.Value[i] > peer[i]) peer[i] = pair.Value[i];
            dirty = roofDirty = true;
        }
        internal bool OwnOverlap(string sender)
        {
            if (!peerLayers.TryGetValue(sender, out var layer)) return false;
            for (int i = 0; i < Count; i++) if (observed[i] && layer[i] != 0) return true;
            return false;
        }
        public byte[] CopySearched()
        {
            var copy = new byte[Count]; for (int i = 0; i < Count; i++) if (observed[i] || peer[i] != 0) copy[i] = 1;
            return copy;
        }
        public byte[] CopyPeerExplored()
        {
            var copy = new byte[Count]; for (int i = 0; i < Count; i++) if (peer[i] != 0) copy[i] = 1;
            return copy;
        }
        internal float LocalHeat(int i) => ownHeat[i];
        internal double LocalHeatTime(int i) => ownHeatTime[i];
        internal void ObservePeerHeat(int i, float temperatureC, double observedAt, double receivedAt)
        {
            if (i < 0 || i >= Count || observedAt < heatTime[i]) return;
            float decayed = Mathf.Max(c.ambient_c, temperatureC - c.heat_decay_c_per_s * (float)(receivedAt - observedAt));
            bool before = Dangerous(i);
            heat[i] = Mathf.Max(heat[i], decayed); heatTime[i] = observedAt;
            // A packet contains one historical sample, so it cannot establish a local temperature slope.
            rise[i] = 0; dirty |= before != Dangerous(i);
        }
        public void ObserveHeat(int i, float temperatureC, double t)
        {
            if (i < 0 || i >= Count || !BrainMath.Finite(temperatureC) || t < heatTime[i]) return;
            float ownDecayed = double.IsNegativeInfinity(ownHeatTime[i]) ? c.ambient_c :
                Mathf.Max(c.ambient_c, ownHeat[i] - c.heat_decay_c_per_s * (float)(t - ownHeatTime[i]));
            if (t >= ownHeatTime[i] && temperatureC >= ownDecayed) { ownHeat[i] = temperatureC; ownHeatTime[i] = t; }
            bool wasDangerous = Dangerous(i);
            if (t > heatTime[i] && t - heatTime[i] <= c.heat_rise_memory_s)
                rise[i] = Mathf.Max(0, (temperatureC - heat[i]) / (float)(t - heatTime[i]));
            else if (t > heatTime[i]) rise[i] = 0;
            heat[i] = Mathf.Max(heat[i], temperatureC); heatTime[i] = t; dirty |= wasDangerous != Dangerous(i);
        }
        public void DecayTo(double t)
        {
            float dt = (float)Math.Max(0, t - time); bool changed = false;
            for (int i = 0; i < Count; i++)
            {
                bool before = Dangerous(i);
                if (heat[i] > c.ambient_c) { heat[i] = Mathf.Max(c.ambient_c, heat[i] - c.heat_decay_c_per_s * dt); }
                if (t - heatTime[i] > c.heat_rise_memory_s) rise[i] = 0;
                if (before != Dangerous(i)) changed = true;
            }
            time = t; localObservationTime = t; dirty |= changed; roofDirty |= changed;
        }
        public bool Dangerous(int i) => heat[i] >= c.fire_c ||
            (time - heatTime[i] <= c.heat_rise_memory_s && heat[i] + rise[i] * c.heat_prediction_s >= c.fire_c);
        public void MarkFootprint(Vector2 position)
        {
            int cell = Cell(position); if (cell < 0) return;
            int radius = Mathf.CeilToInt(c.robot_radius_m / CellM);
            for (int y = -radius; y <= radius; y++) for (int x = -radius; x <= radius; x++)
            {
                int col = cell % Size + x, row = cell / Size + y;
                if (Inside(col, row) && Vector2.Distance(Centre(row * Size + col), position) <= c.robot_radius_m)
                    ObserveFree(row * Size + col);
            }
        }
        /// <summary>The robot stands here, so nothing solid can occupy its body. Cells inside the body radius that read
        /// occupied (depth noise, smoke, an inferred contact strip after the robot moved) are reset to free; without this
        /// a phantom cell under the robot blocks every heading and the robot never moves again.</summary>
        public void ClearBody(Vector2 position, float radius)
        {
            int cell = Cell(position); if (cell < 0) return;
            int r = Mathf.CeilToInt(radius / CellM);
            for (int y = -r; y <= r; y++) for (int x = -r; x <= r; x++)
            {
                int col = cell % Size + x, row = cell / Size + y; if (!Inside(col, row)) continue;
                int i = row * Size + col;
                if ((odds[i] <= c.free_log_odds && lidarOdds[i] <= c.free_log_odds) || Vector2.Distance(Centre(i), position) > radius) continue;
                odds[i] = Mathf.Min(odds[i], c.free_log_odds); lidarOdds[i] = Mathf.Min(lidarOdds[i], c.free_log_odds);
                observed[i] = true; localTime[i] = localObservationTime; dirty = roofDirty = true;
            }
        }
        public void IntegrateDepth(RgbdFrame frame, SensorConfig s, BrainPose pose)
        {
            localObservationTime = frame.Timestamp;
            Array.Clear(hits, 0, Count); Array.Clear(frees, 0, Count);
            Vector2 camera = pose.Transform(0, s.camera.mount.forward_offset_m);
            int start = Cell(camera); if (start < 0) { localObservationTime = time; return; }
            for (int v = 0; v < frame.Height; v += c.depth_stride) for (int u = 0; u < frame.Width; u += c.depth_stride)
            {
                float z = frame.Depth[v * frame.Width + u] * .001f;
                if (z < s.depth.min_m || z > s.depth.max_m) continue;
                var p = BackProject(u, v, z, s);
                if (p.z >= c.ceiling_min_height_m && p.z <= c.ceiling_max_height_m)
                {
                    // A ceiling seen overhead: this floor is indoors. Walls are excluded when the roof is used.
                    int roof = Cell(pose.Transform(p.x, p.y));
                    if (roof >= 0 && roofHits[roof] < 255) { roofHits[roof]++; roofDirty = true; }
                }
                if (p.z > c.obstacle_max_height_m) continue;
                Vector2 end = pose.Transform(p.x, p.y);
                bool obstacle = p.z >= c.obstacle_min_height_m;
                Ray(start, end, obstacle);
            }
            // Aggregate a frame before updating odds. A floor ray cannot erase a wall hit in this frame.
            for (int i = 0; i < Count; i++)
                if (hits[i]) ObserveOccupied(i); else if (frees[i] && frame.Timestamp >= localTime[i]) ObserveFree(i);
            localObservationTime = time;
        }
        /// <summary>One LiDAR sweep at its capture pose: free along each beam, occupied at its end. A beam with no
        /// return carries no evidence, because smoke can swallow a pulse in front of a wall.</summary>
        public void IntegrateLidar(LidarScan scan, BrainPose pose)
        {
            Array.Clear(hits, 0, Count); Array.Clear(frees, 0, Count);
            Vector2 origin = pose.Transform(0, scan.MountForwardM);
            int start = Cell(origin); if (start < 0) return;
            for (int i = 0; i < scan.RangesM.Length; i++)
            {
                float r = scan.RangesM[i];
                if (!(r >= scan.MinM) || r > scan.MaxM) continue;
                float a = pose.yawRad + scan.Angle(i);
                Ray(start, origin + new Vector2(-Mathf.Sin(a), Mathf.Cos(a)) * r, true);
            }
            // As for depth, a grazing beam through a wall cell cannot erase a hit from the same sweep.
            for (int i = 0; i < Count; i++)
            {
                if (hits[i]) { lidarOdds[i] = Mathf.Min(c.log_odds_cap, Mathf.Max(0, lidarOdds[i]) + c.occupied_log_odds); dirty = roofDirty = true; }
                else if (frees[i]) { lidarOdds[i] = Mathf.Max(-c.log_odds_cap, lidarOdds[i] + c.free_log_odds); dirty = roofDirty = true; }
            }
        }
        /// <summary>Distance in metres from each cell of a square window to the nearest LiDAR-occupied cell, for scan
        /// matching. Window cells are indexed [y * side + x] from (x0, y0); cells outside the grid count as empty.</summary>
        internal float[] LidarDistanceWindow(int x0, int y0, int side)
        {
            var f = new float[side]; var d = new float[side]; var sites = new int[side]; var cuts = new float[side + 1];
            var temp = new float[side * side]; var output = new float[side * side];
            for (int y = 0; y < side; y++)
            {
                for (int x = 0; x < side; x++)
                {
                    int gx = x0 + x, gy = y0 + y;
                    f[x] = Inside(gx, gy) && lidarOdds[gy * Size + gx] >= c.occupied_threshold ? 0 : 1e10f;
                }
                DistanceLine(f, d, sites, cuts);
                for (int x = 0; x < side; x++) temp[y * side + x] = d[x];
            }
            for (int x = 0; x < side; x++)
            {
                for (int y = 0; y < side; y++) f[y] = temp[y * side + x];
                DistanceLine(f, d, sites, cuts);
                for (int y = 0; y < side; y++) output[y * side + x] = Mathf.Sqrt(d[y]) * CellM;
            }
            return output;
        }
        /// <summary>Returns body x right, y forward, z height above floor. Depth is optical z.</summary>
        public static Vector3 BackProject(float u, float v, float z, SensorConfig s)
        {
            var k = s.camera; float pitch = k.mount.pitch_down_deg * Mathf.Deg2Rad;
            float down = (v - k.cy) / k.fy * z;
            return new Vector3((u - k.cx) / k.fx * z,
                k.mount.forward_offset_m + z * Mathf.Cos(pitch) - down * Mathf.Sin(pitch),
                k.mount.height_m - z * Mathf.Sin(pitch) - down * Mathf.Cos(pitch));
        }
        void Ray(int start, Vector2 end, bool obstacle)
        {
            int x = start % Size, y = start / Size;
            int ex = Mathf.FloorToInt((end.x - Origin.x) / CellM), ey = Mathf.FloorToInt((end.y - Origin.y) / CellM);
            int dx = Math.Abs(ex - x), dy = Math.Abs(ey - y), sx = x < ex ? 1 : -1, sy = y < ey ? 1 : -1, error = dx - dy;
            // Bounded even if a projected ray ends outside the map.
            for (int n = 0; n <= Size * 2 && Inside(x, y); n++)
            {
                if (x == ex && y == ey) { if (obstacle) hits[y * Size + x] = true; else frees[y * Size + x] = true; break; }
                frees[y * Size + x] = true;
                int twice = 2 * error;
                if (twice > -dy) { error -= dy; x += sx; }
                if (twice < dx) { error += dx; y += sy; }
            }
        }
        internal void PrepareNavigation()
        {
            if (!dirty) return;
            if (obstacleDistance == null) { obstacleDistance = new float[Count]; fireDistance = new float[Count]; }
            Distances(i => Occupancy(i) == 255, obstacleDistance);
            Distances(Dangerous, fireDistance); dirty = false;
        }
        internal bool Walkable(int i)
        {
            if (i < 0 || i >= Count || Occupancy(i) != 1) return false;
            // Include half a cell diagonal so clearance applies to the whole hot/occupied cell.
            float padding = CellM * .707107f;
            return obstacleDistance[i] > c.robot_radius_m + padding && fireDistance[i] > c.fire_clearance_m + padding;
        }
        /// <summary>Free and wide enough for the body itself, without the planning margin. Used only to step out of a
        /// spot that is tighter than the planning clearance; see GridPlanner.Lead.</summary>
        internal bool Passable(int i) => i >= 0 && i < Count && Occupancy(i) == 1 &&
            obstacleDistance[i] > c.robot_radius_m * .8f && fireDistance[i] > CellM;
        /// <summary>Distance from ceiling evidence, walked through known free floor (4-neighbour steps), out to
        /// ceiling_radius_m. A straight-line distance reached through exterior walls and counted a strip of yard as
        /// indoors (check run 5: robot B left by its door and searched along the outside of the west wall).</summary>
        void RoofDistances()
        {
            if (roofDistance == null) roofDistance = new float[Count];
            for (int i = 0; i < Count; i++) roofDistance[i] = float.PositiveInfinity;
            PrepareNavigation();
            var queue = new Queue<int>();
            // Ceiling evidence counts only away from walls: the top of a wall face is in the ceiling height band too,
            // and depth noise put some of its points outside the wall cell, which made a strip of yard look roofed
            // (run 6: robot B's search target at 0.55 m outside the west wall).
            for (int i = 0; i < Count; i++)
                if (roofHits[i] >= c.ceiling_min_hits && OwnOccupancy(i) != 255 && obstacleDistance[i] >= c.ceiling_wall_margin_m)
                { roofDistance[i] = 0; queue.Enqueue(i); }
            while (queue.Count > 0)
            {
                int cell = queue.Dequeue(); float next = roofDistance[cell] + CellM;
                if (next > c.ceiling_radius_m) continue;
                int x = cell % Size, y = cell / Size;
                for (int k = 0; k < 4; k++)
                {
                    int nx = x + (k == 0 ? 1 : k == 1 ? -1 : 0), ny = y + (k == 2 ? 1 : k == 3 ? -1 : 0);
                    if (!Inside(nx, ny)) continue;
                    int n = ny * Size + nx;
                    // Only across floor known to be free: an unmapped stretch of wall is not a way out (run 7, the
                    // yard 0.5 m outside the west wall reached through wall cells the robot had not seen yet).
                    if (roofDistance[n] <= next || OwnOccupancy(n) != 1) continue;
                    roofDistance[n] = next; queue.Enqueue(n);
                }
            }
        }
        internal float ObstacleDistance(int i) => i < 0 || i >= Count ? 0 : obstacleDistance[i];
        /// <summary>Indoors: within ceiling_radius_m of floor where this robot's own camera has seen a ceiling
        /// (ceiling_min_hits returns in the ceiling height band on a cell that is not a wall). No outline or floor plan
        /// is involved: the yard has no ceiling, and a warm shape seen through a window lies outside.</summary>
        /// <summary>The indoor layer for map snapshots: 2 where this robot's camera saw a ceiling away from walls, 1 within
        /// ceiling_radius_m of that across known free floor, 0 elsewhere.</summary>
        public byte[] CopyIndoors()
        {
            if (roofDirty || roofDistance == null) { RoofDistances(); roofDirty = false; generation++; }
            var copy = new byte[Count];
            for (int i = 0; i < Count; i++)
                copy[i] = roofDistance[i] == 0 ? (byte)2 : roofDistance[i] <= c.ceiling_radius_m ? (byte)1 : OwnOccupancy(i) == 1 && Enclosed(i) ? (byte)3 : (byte)0;
            return copy;
        }
        public bool Indoors(Vector2 p) => UnderCeiling(p) || (OwnOccupancy(Cell(p)) == 1 && Enclosed(Cell(p)));
        /// <summary>The ceiling half of Indoors: within ceiling_radius_m of wall-clear ceiling evidence, walked across known
        /// free floor.</summary>
        public bool UnderCeiling(Vector2 p)
        {
            int i = Cell(p); if (i < 0) return false;
            if (roofDirty || roofDistance == null) { RoofDistances(); roofDirty = false; generation++; }
            return roofDistance[i] <= c.ceiling_radius_m;
        }

        /// <summary>
        /// Free floor walled in on this robot's own map: of enclosure_rays rays cast from the cell, at least
        /// enclosure_min_closed_share reach a mapped wall within enclosure_range_m and at most enclosure_max_open_share run
        /// that far over floor mapped free (rays into unmapped space count as neither). A room is walled in even when
        /// its ceiling was never seen; the yard is not, since the LiDAR maps open ground there. The ceiling rule alone
        /// left rooms outside the search area: the camera's top edge is about 12 degrees above horizontal, so ceiling
        /// points are seen only 4.5 m and more away, beyond the far wall of most rooms (house reflex check: robot B
        /// gave up at 27 s with 27 percent searched; agent run 9, robot A offered one 0.5 m2 frontier at 20 s).
        /// </summary>
        bool Enclosed(int i)
        {
            if (enclosureGeneration == null) { enclosureGeneration = new int[Count]; enclosureValue = new bool[Count]; for (int k = 0; k < Count; k++) enclosureGeneration[k] = -1; }
            if (enclosureGeneration[i] == generation) return enclosureValue[i];
            Vector2 p = Centre(i); float step = CellM * .5f; int steps = Mathf.CeilToInt(c.enclosure_range_m / step), closed = 0, open = 0;
            for (int r = 0; r < c.enclosure_rays; r++)
            {
                float a = r * 2 * Mathf.PI / c.enclosure_rays; var dir = new Vector2(Mathf.Cos(a), Mathf.Sin(a));
                bool ended = false;
                for (int s = 1; s <= steps && !ended; s++)
                {
                    int cell = Cell(p + dir * (s * step));
                    byte occupancy = cell < 0 ? (byte)1 : OwnOccupancy(cell);
                    if (cell < 0) { open++; ended = true; }
                    else if (occupancy == 255) { closed++; ended = true; }
                    else if (occupancy == 0) ended = true;
                }
                if (!ended) open++;
            }
            bool result = closed >= c.enclosure_min_closed_share * c.enclosure_rays && open <= c.enclosure_max_open_share * c.enclosure_rays;
            enclosureGeneration[i] = generation; enclosureValue[i] = result;
            return result;
        }
        internal float Cost(int i)
        {
            float strength = Mathf.Clamp01((c.heat_cost_radius_m - fireDistance[i]) / (c.heat_cost_radius_m - c.fire_clearance_m));
            float warm = Mathf.Clamp01((heat[i] - c.heat_project_min_c) / (c.fire_c - c.heat_project_min_c));
            float clearance = obstacleDistance[i] - c.robot_radius_m - CellM * .707107f;
            float edge = Mathf.Clamp01(1 - clearance / c.obstacle_cost_margin_m);
            return 1 + c.heat_cost_weight * (strength * strength + warm * warm) + c.obstacle_cost_weight * edge * edge;
        }
        internal float FireDistance(int i) => i < 0 ? 0 : fireDistance[i];
        public byte[] CopyOccupancy() { var result = new byte[Count]; for (int i = 0; i < Count; i++) result[i] = Occupancy(i); return result; }
        public float[] CopyHeat() => (float[])heat.Clone();
        public Vector2[] FireHotspots()
        {
            var list = new List<Vector2>(); for (int i = 0; i < Count; i++) if (Dangerous(i)) list.Add(Centre(i)); return list.ToArray();
        }
        // Exact squared Euclidean distance transform, two one-dimensional passes, O(grid cells).
        // Scratch buffers reused across calls: the map changes 25 times a second per robot, and fresh arrays each time
        // (about 3 MB) kept the garbage collector busy enough to slow the whole recording.
        float[] dtTemp, dtF, dtD, dtCuts; int[] dtSites;
        void Distances(Func<int, bool> source, float[] output)
        {
            if (dtTemp == null) { dtTemp = new float[Count]; dtF = new float[Size]; dtD = new float[Size]; dtSites = new int[Size]; dtCuts = new float[Size + 1]; }
            var temp = dtTemp; var f = dtF; var d = dtD; var sites = dtSites; var cuts = dtCuts;
            for (int y = 0; y < Size; y++)
            {
                for (int x = 0; x < Size; x++) f[x] = source(y * Size + x) ? 0 : 1e10f;
                DistanceLine(f, d, sites, cuts);
                for (int x = 0; x < Size; x++) temp[y * Size + x] = d[x];
            }
            for (int x = 0; x < Size; x++)
            {
                for (int y = 0; y < Size; y++) f[y] = temp[y * Size + x];
                DistanceLine(f, d, sites, cuts);
                for (int y = 0; y < Size; y++) output[y * Size + x] = Mathf.Sqrt(d[y]) * CellM;
            }
        }
        static void DistanceLine(float[] f, float[] d, int[] sites, float[] cuts)
        {
            int k = 0; sites[0] = 0; cuts[0] = float.NegativeInfinity; cuts[1] = float.PositiveInfinity;
            for (int q = 1; q < f.Length; q++)
            {
                float intersection;
                while (true)
                {
                    int site = sites[k];
                    intersection = (float)(((double)f[q] + q * q - ((double)f[site] + site * site)) / (2 * (q - site)));
                    if (intersection > cuts[k] || k == 0) break;
                    k--;
                }
                sites[++k] = q; cuts[k] = intersection; cuts[k + 1] = float.PositiveInfinity;
            }
            k = 0;
            for (int q = 0; q < f.Length; q++)
            { while (cuts[k + 1] < q) k++; float delta = q - sites[k]; d[q] = delta * delta + f[sites[k]]; }
        }
    }
}
