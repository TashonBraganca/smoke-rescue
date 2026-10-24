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
        // When this robot's own thermal camera last looked at each cell, at any temperature (RobotBrain route watch).
        readonly double[] heatLookTime;
        readonly bool[] hits, frees, observed, lows;
        readonly byte[] roofHits;
        // Floor on a clear sight line from an indoor camera to ceiling it saw (see MarkCeilingSightLines).
        readonly bool[] sightIndoors;
        // Ground the LiDAR looked across inside a wide span of beams with no return: nothing there within its range.
        readonly bool[] openGround;
        readonly List<int> frameRoof = new List<int>(); readonly int[] roofFrame; int depthFrames;
        float[] roofDistance; bool roofDirty = true;
        int[] enclosureGeneration; bool[] enclosureValue; int generation;
        readonly double[] localTime, ownHeatTime;
        readonly float[] ownHeat;
        // Heat from partners and agent keep-out orders, kept apart so this robot's own cooler readings never erase it.
        readonly float[] externalHeat; readonly double[] externalTime;
        readonly SortedDictionary<string, byte[]> peerLayers = new SortedDictionary<string, byte[]>(StringComparer.Ordinal);
        readonly byte[] peer;
        double localObservationTime;
        float[] obstacleDistance, fireDistance;
        bool dirty = true;
        double time;
        internal TeamMap TeamBelief { get; set; }
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
            riseBase = new float[Count]; riseBaseTime = new double[Count]; for (int k = 0; k < Count; k++) riseBaseTime[k] = double.NegativeInfinity;
            hits = new bool[Count]; frees = new bool[Count]; observed = new bool[Count]; lows = new bool[Count];
            openScans = new ushort[Count]; returnScans = new ushort[Count]; openMark = new bool[Count];
            localTime = new double[Count]; peer = new byte[Count]; roofHits = new byte[Count];
            sightIndoors = new bool[Count]; openGround = new bool[Count]; roofFrame = new int[Count];
            ownHeat = new float[Count]; ownHeatTime = new double[Count]; externalHeat = new float[Count]; externalTime = new double[Count];
            heatLookTime = new double[Count];
            for (int i = 0; i < Count; i++)
            {
                heat[i] = c.ambient_c; heatTime[i] = double.NegativeInfinity; ownHeat[i] = c.ambient_c; ownHeatTime[i] = double.NegativeInfinity;
                heatLookTime[i] = double.NegativeInfinity;
                externalHeat[i] = c.ambient_c; externalTime[i] = double.NegativeInfinity;
            }
        }
        public int Cell(Vector2 p)
        {
            int x = Mathf.FloorToInt((p.x - Origin.x) / CellM), y = Mathf.FloorToInt((p.y - Origin.y) / CellM);
            return Inside(x, y) ? y * Size + x : -1;
        }
        public Vector2 Centre(int index) => Origin + new Vector2(index % Size + .5f, index / Size + .5f) * CellM;
        public bool Inside(int x, int y) => x >= 0 && y >= 0 && x < Size && y < Size;
        bool[] placed;
        /// <summary>An obstacle this robot put down itself (a person it carried out): blocked for the rest of the run, whatever
        /// the sensors later say, because a person lying on the floor is below the LiDAR plane and the camera does not see
        /// the floor within about 2 m.</summary>
        public void PlaceObstacle(Vector2 centre, float radiusM)
        {
            if (placed == null) placed = new bool[Count];
            for (int i = 0; i < Count; i++) if (Vector2.Distance(Centre(i), centre) <= radiusM) placed[i] = true;
            dirty = true;
        }
        public byte OwnOccupancy(int index)
        {
            if (index < 0 || index >= Count) return 0;
            if (placed != null && placed[index]) return 255;
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
            if (c.team_map_planner && (placed == null || !placed[i]) && TeamBelief != null && TeamBelief.TryOccupancy(Centre(i), time, out float belief))
                return belief >= c.occupied_threshold ? (byte)255 : belief <= c.free_threshold ? (byte)1 : (byte)0;
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
        /// <summary>A teammate's body is not a wall. This robot's own sensors map a partner standing in view as an
        /// obstacle, and floor out of view keeps that wall (one-door seed 923: robot A's LiDAR saw B standing on A's pad in
        /// the one doorway; A walked on into the living room, its only way out stayed walled for the rest of the run, it
        /// counted itself cut off, and its heat response stayed off while the air rose to 70 C). Own wall cells within
        /// radius of the straight line the partner reported moving along (from one radio pose to the next) are set free;
        /// real walls beside its path stay. Returns how many cells changed.</summary>
        public int MarkTeammateFree(Vector2 from, Vector2 to, float radius)
        {
            float r = radius + CellM * .5f; int changed = 0;
            Vector2 lo = Vector2.Min(from, to) - Vector2.one * r, hi = Vector2.Max(from, to) + Vector2.one * r;
            int a = Cell(lo), b = Cell(hi); if (a < 0 || b < 0) return 0;
            for (int y = a / Size; y <= b / Size; y++) for (int x = a % Size; x <= b % Size; x++)
            {
                int i = y * Size + x; Vector2 p = Centre(i), d = to - from;
                float u = d.sqrMagnitude < 1e-9f ? 0 : Mathf.Clamp01(Vector2.Dot(p - from, d) / d.sqrMagnitude);
                if (Vector2.Distance(p, from + d * u) > r) continue;
                if (odds[i] >= c.occupied_threshold || lidarOdds[i] >= c.occupied_threshold)
                {
                    odds[i] = Mathf.Min(odds[i], c.free_threshold - .1f); lidarOdds[i] = Mathf.Min(lidarOdds[i], c.free_threshold - .1f);
                    changed++;
                }
            }
            if (changed > 0) dirty = roofDirty = true;
            return changed;
        }
        /// <summary>Free floor mapped by the depth camera but too far away to count as searched.</summary>
        void MapFree(int i)
        {
            if (i < 0 || i >= Count) return;
            odds[i] = Mathf.Max(-c.log_odds_cap, odds[i] + c.free_log_odds);
            localTime[i] = localObservationTime; dirty = roofDirty = true;
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
        /// <param name="ownBody">True where this robot's own body has stood. A partner's LiDAR sees this robot as an
        /// obstacle, so its map reports walls wherever this robot walked in its view; floor this robot's body covered is
        /// not a wall (one-door seed 922: B saw A walk through the one door and reported the doorway as wall; at 4 s both
        /// robots lost every route and frontier).</param>
        internal void ReplacePeerLayer(string sender, BrainMessage m, byte[] explored, BrainPose toOwn, Func<Vector2, bool> ownBody = null)
        {
            var layer = new byte[Count]; float cos = Mathf.Cos(-toOwn.yawRad), sin = Mathf.Sin(-toOwn.yawRad);
            for (int i = 0; i < Count; i++)
            {
                Vector2 d = Centre(i) - toOwn.position;
                int cell = m.Cell(new Vector2(cos * d.x - sin * d.y, sin * d.x + cos * d.y)); if (cell < 0) continue;
                if (BrainMessage.Bit(m.Occupied, cell) && (ownBody == null || !ownBody(Centre(i)))) layer[i] = 255;
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
        public byte[] CopyOwnSearched()
        {
            var copy = new byte[Count]; for (int i = 0; i < Count; i++) if (observed[i]) copy[i] = 1;
            return copy;
        }
        public byte[] CopyPeerExplored()
        {
            var copy = new byte[Count]; for (int i = 0; i < Count; i++) if (peer[i] != 0) copy[i] = 1;
            return copy;
        }
        internal float LocalHeat(int i) => ownHeat[i];
        internal double LocalHeatTime(int i) => ownHeatTime[i];
        /// <summary>When this robot's own thermal camera last looked at the cell, whatever it read (cool floor too);
        /// negative infinity if never. Partner heat and agent keep-out areas do not count: they are not a look of its own.</summary>
        public double HeatLookTime(int i) => i < 0 || i >= Count ? double.NegativeInfinity : heatLookTime[i];
        internal void MarkHeatLook(int i, double t) { if (i >= 0 && i < Count && t > heatLookTime[i]) heatLookTime[i] = t; }
        float ExternalHeatAt(int i, double t) => double.IsNegativeInfinity(externalTime[i]) ? c.ambient_c :
            Mathf.Max(c.ambient_c, externalHeat[i] - c.heat_decay_c_per_s * (float)(t - externalTime[i]));
        /// <summary>An agent's keep-out (avoid_area) is written as heat and must last its full time: this robot's cool
        /// readings never clear a held cell before until (a fire spreading toward cool floor is why it was ordered).</summary>
        internal void HoldHeat(int i, double until) { if (i < 0 || i >= Count) return; heatHeldUntil ??= new double[Count]; heatHeldUntil[i] = Math.Max(heatHeldUntil[i], until); }
        double[] heatHeldUntil;
        internal void ObservePeerHeat(int i, float temperatureC, double observedAt, double receivedAt)
        {
            if (i < 0 || i >= Count || observedAt < heatTime[i]) return;
            float decayed = Mathf.Max(c.ambient_c, temperatureC - c.heat_decay_c_per_s * (float)(receivedAt - observedAt));
            bool before = Dangerous(i);
            externalHeat[i] = Mathf.Max(ExternalHeatAt(i, receivedAt), decayed); externalTime[i] = receivedAt;
            heat[i] = Mathf.Max(heat[i], decayed); heatTime[i] = observedAt;
            // A packet contains one historical sample, so it cannot establish a local temperature slope.
            rise[i] = 0; dirty |= before != Dangerous(i);
        }
        public void ObserveHeat(int i, float temperatureC, double t)
        {
            if (i < 0 || i >= Count || !BrainMath.Finite(temperatureC) || t < heatTime[i]) return;
            float ownDecayed = double.IsNegativeInfinity(ownHeatTime[i]) ? c.ambient_c :
                Mathf.Max(c.ambient_c, ownHeat[i] - c.heat_decay_c_per_s * (float)(t - ownHeatTime[i]));
            // Cooled: seen below its decayed own heat, or seen cool (heat_project_min_c) where the map holds heat that came
            // only from a partner, with no hot reading of its own for heat_cool_after_s. Tested before the "warmer than
            // remembered" update, which a cool look at a partner-reported cell would otherwise take (ambient own heat).
            bool cooled = (temperatureC < ownDecayed || (temperatureC <= c.heat_project_min_c && heat[i] > c.heat_project_min_c))
                          && t - ownHeatTime[i] >= c.heat_cool_after_s && !(heatHeldUntil != null && heatHeldUntil[i] > t);
            if (!cooled && t >= ownHeatTime[i] && temperatureC >= ownDecayed) { ownHeat[i] = temperatureC; ownHeatTime[i] = t; }
            else if (cooled)
            {
                // Seen cooler, with no hot reading for heat_cool_after_s: the fire there is out (water, burnt out). Flames
                // flicker, but a burning cell keeps refreshing its hot reading, so it stays hot. Without this a 500 C
                // reading stayed "fire" for over 30 minutes (0.2 C/s decay) and sprayed water never opened a route.
                bool before = Dangerous(i);
                ownHeat[i] = temperatureC; ownHeatTime[i] = t;
                // This robot's own newer look beats a partner's older report: keeping max(cool, partner heat) held a
                // partner's 500 C sample at 499 C against a fresh local 20 C (Codex HeatCheck, 23 Oct). A partner packet
                // observed before t is already ignored by ObservePeerHeat's time check.
                externalHeat[i] = temperatureC; externalTime[i] = t;
                heat[i] = temperatureC; heatTime[i] = t; rise[i] = 0;
                dirty |= before != Dangerous(i);
                return;
            }
            bool wasDangerous = Dangerous(i);
            if (c.heat_rise_window_s > 0) WindowRise(i, temperatureC, t);
            else if (t > heatTime[i] && t - heatTime[i] <= c.heat_rise_memory_s)
                rise[i] = Mathf.Max(0, (temperatureC - heat[i]) / (float)(t - heatTime[i]));
            else if (t > heatTime[i]) rise[i] = 0;
            heat[i] = Mathf.Max(heat[i], temperatureC); heatTime[i] = t; dirty |= wasDangerous != Dangerous(i);
        }
        /// <summary>heat_rise_window_s: the rise rate is the slope from a reading at least heat_rise_window_s old, not from the
        /// previous frame. Between two frames 67 ms apart a 15 to 25 C flicker of hot-gas radiance read as 210 to 285 C/s, and
        /// heat + rise * heat_prediction_s marked 89 to 95 C floor as fire. Seed 934 (24 Oct 04:53 run), robot B at 212.5 to
        /// 215 s: 0 cells measured at fire_c, 441 forecast as fire from its own frames (replayed); with partner reports, 28 of
        /// its routes were cut by fire margins while its air stayed under 55 C, and it never reached the laundry.
        /// A sustained rise still forecasts fire before it is measured.</summary>
        void WindowRise(int i, float temperatureC, double t)
        {
            if (t - riseBaseTime[i] > c.heat_rise_memory_s) { riseBase[i] = temperatureC; riseBaseTime[i] = t; rise[i] = 0; return; }
            if (t - riseBaseTime[i] < c.heat_rise_window_s) return;
            rise[i] = Mathf.Max(0, (temperatureC - riseBase[i]) / (float)(t - riseBaseTime[i]));
            riseBase[i] = temperatureC; riseBaseTime[i] = t;
        }
        float[] riseBase; double[] riseBaseTime;
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
        /// <summary>Cells measured at or above fire_c (not forecast) within radius of p, as floor area in square metres.</summary>
        public float BurningAreaNear(Vector2 p, float radius)
        {
            int n = 0, r = Mathf.CeilToInt(radius / CellM), centre = Cell(p); if (centre < 0) return 0;
            int cx = centre % Size, cy = centre / Size;
            for (int dy = -r; dy <= r; dy++) for (int dx = -r; dx <= r; dx++)
            {
                if (!Inside(cx + dx, cy + dy)) continue;
                int i = (cy + dy) * Size + cx + dx;
                if (heat[i] >= c.fire_c && Vector2.Distance(Centre(i), p) <= radius) n++;
            }
            return n * CellM * CellM;
        }
        /// <summary>This robot's own LiDAR saw open ground here (wide no-return spans, MarkOpenGround).</summary>
        public bool OpenGroundAt(Vector2 p) { int i = Cell(p); return i >= 0 && openGround[i]; }
        internal float HeatRise(int i) => rise[i];
        /// <summary>Measured at or above fire_c, never a forecast.</summary>
        internal bool MeasuredFire(int i) => heat[i] >= c.fire_c;
        /// <summary>A fire cell within the route cell's clearance, measured between cell centres as FireDistance does.</summary>
        internal int FireCellNear(int cell, float radius, bool measuredOnly)
        {
            if (cell < 0 || cell >= Count) return -1;
            int reach = Mathf.CeilToInt(radius / CellM), cx = cell % Size, cy = cell / Size;
            var at = Centre(cell);
            for (int y = Math.Max(0, cy - reach); y <= Math.Min(Size - 1, cy + reach); y++)
                for (int x = Math.Max(0, cx - reach); x <= Math.Min(Size - 1, cx + reach); x++)
                {
                    int i = y * Size + x;
                    if ((measuredOnly ? heat[i] >= c.fire_c : Dangerous(i)) && Vector2.Distance(at, Centre(i)) <= radius) return i;
                }
            return -1;
        }
        public bool Dangerous(int i) => heat[i] >= c.fire_c ||
            (firePrediction && time - heatTime[i] <= c.heat_rise_memory_s && heat[i] + rise[i] * c.heat_prediction_s >= c.fire_c);
        /// <summary>Whether a cell forecast to reach fire_c within heat_prediction_s counts as fire. True while searching; false
        /// on an emergency exit (RobotBrain.TryEmergencyExit). Held-out seed 914, robot B: standing in the hot gas filling the
        /// foyer, its thermal camera read the floor of the only doorway out at 60 to 187 C and rising fast (gas radiance along
        /// the ray, below fire_c everywhere); the forecast marked the doorway about to ignite, so the route home opened and
        /// shut every second and the robot stood still for 50 s, 185 C peak. Measured fire still closes a cell.</summary>
        internal bool FirePrediction
        {
            get => firePrediction;
            set { if (firePrediction == value) return; firePrediction = value; dirty = roofDirty = true; }
        }
        bool firePrediction = true;
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
            Array.Clear(hits, 0, Count); Array.Clear(frees, 0, Count); Array.Clear(lows, 0, Count);
            float eyeHeight = s.camera.mount.height_m;
            Vector2 camera = pose.Transform(0, s.camera.mount.forward_offset_m);
            int start = Cell(camera); if (start < 0) { localObservationTime = time; return; }
            depthFrames++; frameRoof.Clear();
            for (int v = 0; v < frame.Height; v += c.depth_stride) for (int u = 0; u < frame.Width; u += c.depth_stride)
            {
                float z = frame.Depth[v * frame.Width + u] * .001f;
                if (z < s.depth.min_m || z > s.depth.max_m) continue;
                var p = BackProject(u, v, z, s);
                // No upper height band: a lobby, gym or warehouse ceiling is higher than a house's. The camera's own reach
                // (mount height plus the depth sensor's maximum range) is the only upper limit; wall faces cannot fake a
                // ceiling because cells near mapped walls are excluded (ceiling_wall_margin_m).
                if (p.z >= c.ceiling_min_height_m && p.z <= s.camera.mount.height_m + s.depth.max_m)
                {
                    // A ceiling seen overhead: this floor is indoors. Walls are excluded when the roof is used.
                    int roof = Cell(pose.Transform(p.x, p.y));
                    if (roof >= 0 && roofHits[roof] < 255) { roofHits[roof]++; roofDirty = true; }
                    if (roof >= 0 && roofFrame[roof] != depthFrames) { roofFrame[roof] = depthFrames; frameRoof.Add(roof); }
                }
                if (p.z > c.obstacle_max_height_m) continue;
                Vector2 end = pose.Transform(p.x, p.y);
                bool obstacle = p.z >= c.obstacle_min_height_m;
                if (obstacle && TeammateAt != null && TeammateAt(end, frame.Timestamp)) continue;
                Ray(start, end, obstacle);
                // Searched floor is where the ray runs within search_low_m of it: a person lying there would have blocked
                // it. Higher up, a ray over a table to the wall behind says nothing about the floor below (held-out seed
                // 914 run 1: a person slumped by the kitchen island lay on floor marked searched and was never in view).
                if (c.search_low_m > 0 && p.z < c.search_low_m && eyeHeight > c.search_low_m)
                    LowRay(Cell(camera + (end - camera) * ((eyeHeight - c.search_low_m) / (eyeHeight - p.z))), end);
            }
            // Aggregate a frame before updating odds. A floor ray cannot erase a wall hit in this frame.
            // Free floor further than search_max_range_m from the camera is mapped but not searched: a person lying there
            // could not have been found from here (hidden-corner runs: a glance through a doorway from 8 m counted a whole
            // room as searched and the robot never went in, while a confirmation needs an inspection within 4 m).
            Vector2 eye = Centre(start); float reach = c.search_max_range_m;
            TeamBelief?.BeginObservation(frame.Timestamp);
            for (int i = 0; i < Count; i++)
                if (hits[i]) { ObserveOccupied(i); TeamBelief?.Observe(Centre(i), true, false); }
                else if (frees[i] && frame.Timestamp >= localTime[i])
                {
                    bool searchedFloor = Vector2.Distance(Centre(i), eye) <= reach && (c.search_low_m <= 0 || lows[i]);
                    if (searchedFloor) ObserveFree(i); else MapFree(i);
                    TeamBelief?.Observe(Centre(i), false, searchedFloor);
                }
            TeamBelief?.EndObservation();
            MarkCeilingSightLines(start);
            localObservationTime = time;
        }
        /// <summary>One LiDAR sweep at its capture pose: free along each beam, occupied at its end. A single beam with no
        /// return carries no evidence, because smoke can swallow a pulse in front of a wall; a wide span of them is open
        /// ground (MarkOpenGround).</summary>
        /// <summary>Where a partner robot's body is predicted at a time (RobotBrain.OnTeammate): returns hitting it are not
        /// walls and are left out of the map, beam and all, so nothing in the map changes for them. One-door seed 924:
        /// robot A's LiDAR drew robot B, walking in behind it, as a line of wall through the 0.9 m front door, and A lost
        /// its way out at 12 s; clearing B's radioed track afterwards (MarkTeammateFree) missed its flank, because the pose
        /// is up to 2 s old and carries B's placement error.</summary>
        public Func<Vector2, double, bool> TeammateAt;
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
                Vector2 end = origin + new Vector2(-Mathf.Sin(a), Mathf.Cos(a)) * r;
                if (TeammateAt != null && TeammateAt(end, scan.Timestamp)) continue;
                Ray(start, end, true);
            }
            // As for depth, a grazing beam through a wall cell cannot erase a hit from the same sweep.
            TeamBelief?.BeginObservation(scan.Timestamp, 1);
            for (int i = 0; i < Count; i++)
            {
                if (hits[i]) { lidarOdds[i] = Mathf.Min(c.log_odds_cap, Mathf.Max(0, lidarOdds[i]) + c.occupied_log_odds); dirty = roofDirty = true; TeamBelief?.Observe(Centre(i), true, false); }
                else if (frees[i]) { lidarOdds[i] = Mathf.Max(-c.log_odds_cap, lidarOdds[i] + c.free_log_odds); dirty = roofDirty = true; TeamBelief?.Observe(Centre(i), false, false); }
            }
            TeamBelief?.EndObservation();
            MarkOpenGround(scan, origin, pose.yawRad);
            CountOpenBeams(scan, origin, pose.yawRad);
        }
        // Per cell, the sweeps in which a beam crossed it and came back with nothing (open) or crossed it and hit something
        // beyond (returned). See OpenBeyond.
        readonly ushort[] openScans, returnScans; readonly bool[] openMark;
        void CountOpenBeams(LidarScan scan, Vector2 origin, float yawRad)
        {
            for (int i = 0; i < Count; i++) if (frees[i] && returnScans[i] < ushort.MaxValue) returnScans[i]++;
            float step = CellM * .5f; int steps = Mathf.FloorToInt(scan.MaxM / step); var touched = new List<int>();
            for (int b = 0; b < scan.RangesM.Length; b++)
            {
                if (!NoReturn(scan, b)) continue;
                float angle = yawRad + scan.Angle(b); var dir = new Vector2(-Mathf.Sin(angle), Mathf.Cos(angle));
                for (int s = 1; s <= steps; s++)
                {
                    int cell = Cell(origin + dir * (s * step));
                    if (cell < 0 || lidarOdds[cell] >= c.occupied_threshold) break;
                    if (!openMark[cell]) { openMark[cell] = true; touched.Add(cell); }
                }
            }
            foreach (int cell in touched) { openMark[cell] = false; if (!frees[cell] && openScans[cell] < ushort.MaxValue) openScans[cell]++; }
        }
        /// <summary>
        /// Open beyond: in at least exit_min_scans sweeps a beam crossed this cell, and in at least exit_open_share of them
        /// it came back with nothing within the LiDAR's range. Smoke drops each pulse on its own (sensors.json caps it at
        /// half), so a cell whose beams return off a wall reads open at most about that often, while floor past an
        /// exterior opening reads open nearly always (held-out seed 914 run 5, robot A in the foyer: 77.7 % of beams
        /// through the front door returned nothing, 4.2 % of beams at the wall beside it).
        /// </summary>
        internal bool OpenBeyond(int i)
        {
            int n = openScans[i] + returnScans[i];
            return n >= c.exit_min_scans && openScans[i] >= c.exit_open_share * n;
        }
        /// <summary>
        /// Ways out this robot found with its own sensors: connected patches of floor it has mapped walkable, that its own
        /// evidence does not put indoors, and that its LiDAR sees open beyond (OpenBeyond). A door to the outside shows as
        /// such a patch just past the doorway; a room or an enclosed courtyard within the LiDAR's range returns its beams
        /// and does not. One point per patch (the cell seen open most purely), largest patches first, at most exit_max_count.
        /// A hall wider than the LiDAR's range would also read open: a known limit.
        /// </summary>
        public List<Vector2> SensedExits()
        {
            PrepareNavigation();
            var label = new int[Count]; var found = new List<(Vector2 point, int size)>(); var queue = new Queue<int>();
            for (int s = 0; s < Count; s++)
            {
                if (label[s] != 0 || !ExitCell(s)) continue;
                label[s] = 1; queue.Enqueue(s); int size = 0, best = s;
                while (queue.Count > 0)
                {
                    int cell = queue.Dequeue(); size++;
                    // The point to head for: the cell seen open most purely (floor outside reads open in nearly every sweep,
                    // the threshold just inside a doorway slightly less), then the most sweeps.
                    float share = OpenShare(cell), bestShare = OpenShare(best);
                    if (share > bestShare + 1e-4f || (Mathf.Abs(share - bestShare) <= 1e-4f && openScans[cell] + returnScans[cell] > openScans[best] + returnScans[best])) best = cell;
                    int x = cell % Size, y = cell / Size;
                    for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++)
                    {
                        if ((dx == 0 && dy == 0) || !Inside(x + dx, y + dy)) continue;
                        int n = (y + dy) * Size + x + dx;
                        if (label[n] == 0 && ExitCell(n)) { label[n] = 1; queue.Enqueue(n); }
                    }
                }
                found.Add((Centre(best), size));
            }
            found.Sort((a, b) => b.size.CompareTo(a.size));
            var points = new List<Vector2>();
            for (int k = 0; k < found.Count && k < c.exit_max_count; k++) points.Add(found[k].point);
            return points;
        }
        float OpenShare(int i) { int n = openScans[i] + returnScans[i]; return n == 0 ? 0 : openScans[i] / (float)n; }
        bool ExitCell(int i) => OpenBeyond(i) && OwnOccupancy(i) == 1 && Walkable(i) && !Indoors(Centre(i));
        static bool NoReturn(LidarScan scan, int i) { float r = scan.RangesM[i]; return !(r >= scan.MinM) || r > scan.MaxM; }
        /// <summary>
        /// Open ground from beams with no return. A pulse lost in smoke also reads 0, but each pulse is lost on its own
        /// (at most half of them, sensors.json smoke_dropout.max), so lidar_open_span_deg of neighbouring beams all empty
        /// is not smoke: nothing stands within the LiDAR's range that way. The cells those beams cross are open ground, up
        /// to the first cell the LiDAR has mapped as a wall. The enclosure test read such rays as unknown, so one long
        /// facade plus the person a robot had just carried out made the yard "walled in" (held-out run 2: robot B searched
        /// along the outside of the south wall for two minutes and lost its localisation there).
        /// </summary>
        void MarkOpenGround(LidarScan scan, Vector2 origin, float yawRad)
        {
            int n = scan.RangesM.Length;
            if (n == 0 || !(scan.AngleIncrementRad > 0)) return;
            int need = Mathf.Max(1, Mathf.CeilToInt(c.lidar_open_span_deg * Mathf.Deg2Rad / scan.AngleIncrementRad - 1e-3f));
            if (need > n) return;
            // A full circle wraps: start after a beam with a return so a span is not cut in two at the seam.
            int first = 0;
            if (n * scan.AngleIncrementRad >= 2 * Mathf.PI - scan.AngleIncrementRad * .5f)
                for (int i = 0; i < n; i++) if (!NoReturn(scan, i)) { first = i; break; }
            int runStart = 0, runLength = 0;
            for (int k = 0; k <= n; k++)
            {
                if (k < n && NoReturn(scan, (first + k) % n)) { if (runLength++ == 0) runStart = k; continue; }
                if (runLength >= need)
                    for (int j = runStart; j < runStart + runLength; j++) OpenBeam(origin, yawRad + scan.Angle((first + j) % n), scan.MaxM);
                runLength = 0;
            }
        }
        void OpenBeam(Vector2 from, float angle, float rangeM)
        {
            var dir = new Vector2(-Mathf.Sin(angle), Mathf.Cos(angle)); float step = CellM * .5f;
            int steps = Mathf.FloorToInt(rangeM / step);
            for (int s = 0; s <= steps; s++)
            {
                int cell = Cell(from + dir * (s * step));
                if (cell < 0 || lidarOdds[cell] >= c.occupied_threshold) return;
                if (!openGround[cell]) { openGround[cell] = true; roofDirty = true; }
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
        /// <summary>The cells a depth ray crosses within search_low_m of the floor, from where it comes down to that height
        /// to its end.</summary>
        void LowRay(int from, Vector2 end)
        {
            if (from < 0) return;
            int x = from % Size, y = from / Size;
            int ex = Mathf.FloorToInt((end.x - Origin.x) / CellM), ey = Mathf.FloorToInt((end.y - Origin.y) / CellM);
            int dx = Math.Abs(ex - x), dy = Math.Abs(ey - y), sx = x < ex ? 1 : -1, sy = y < ey ? 1 : -1, error = dx - dy;
            for (int n = 0; n <= Size * 2 && Inside(x, y); n++)
            {
                lows[y * Size + x] = true;
                if (x == ex && y == ey) break;
                int twice = 2 * error;
                if (twice > -dy) { error -= dy; x += sx; }
                if (twice < dx) { error += dx; y += sy; }
            }
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
        /// <summary>Discs the planner treats as obstacles: confirmed people, so a robot never walks into a casualty (one-door
        /// seed 927: robot A, routing past the person it had just confirmed, pressed into her body's collision volume, which
        /// the depth map had recorded smaller, and stayed pinned for over a minute).</summary>
        /// <summary>Around each confirmed person: a hard core (an obstacle) where the body surely is, and a soft ring out to
        /// radius that costs keepout_cost_weight at the core edge, falling to nothing at the rim. One-door seed 931: a person
        /// confirmed 2.4 m inside the one door, with the whole 1.05 m disc an obstacle, closed the entrance to both robots.</summary>
        public void SetKeepOut(IReadOnlyList<Vector2> centres, float coreRadius, float radius)
        {
            keepOut ??= new bool[Count]; System.Array.Clear(keepOut, 0, Count);
            keepSoft ??= new float[Count]; System.Array.Clear(keepSoft, 0, Count);
            foreach (var p in centres)
            {
                int c0 = Cell(p); if (c0 < 0) continue;
                int r = Mathf.CeilToInt(radius / CellM), cx = c0 % Size, cy = c0 / Size;
                for (int dy = -r; dy <= r; dy++) for (int dx = -r; dx <= r; dx++)
                {
                    if (!Inside(cx + dx, cy + dy)) continue;
                    int i = (cy + dy) * Size + cx + dx; float d = Vector2.Distance(Centre(i), p);
                    if (d <= coreRadius) keepOut[i] = true;
                    else if (d <= radius) keepSoft[i] = Mathf.Max(keepSoft[i], 1 - (d - coreRadius) / Mathf.Max(1e-3f, radius - coreRadius));
                }
            }
            dirty = true;
        }
        bool[] keepOut; float[] keepSoft;
        /// <summary>The entrance lock: floor within radius of where this robot first stood on floor it had proven indoors.
        /// Owner's rule (23 Oct): a robot that is in never goes back to, or near, the way it came in (seed 932: robot B's
        /// retreat ended 1 m inside the front door, looking out at the yard). Without a passage (inside null) the whole disc is
        /// an obstacle. With one (entrance_lock_passage), only the part not inside is an obstacle, so no route leaves by the way
        /// in; the indoor part costs entrance_lock_cost_weight per cell and is never a goal (Standable), so routes cross it only
        /// when no other way leads on. Headless house 5015 (23 Oct): the way in opened onto the hallway beside the living room
        /// door, the whole-disc lock walled both robots into the living room, and 8 of 9 rooms went unsearched. Called again every 0.5 s
        /// as the map grows, since floor unmapped at the lock may later prove indoors.</summary>
        public void SetEntranceLock(Vector2 entry, float radius, Func<Vector2, bool> inside = null)
        {
            int c0 = Cell(entry); if (c0 < 0) return;
            if (entranceLock == null || entranceEntry != entry || entranceRadius != radius)
            {
                entranceLock = new bool[Count]; entranceHard = new bool[Count]; var cells = new List<int>();
                int r = Mathf.CeilToInt(radius / CellM), cx = c0 % Size, cy = c0 / Size;
                for (int dy = -r; dy <= r; dy++) for (int dx = -r; dx <= r; dx++)
                {
                    if (!Inside(cx + dx, cy + dy)) continue;
                    int i = (cy + dy) * Size + cx + dx;
                    if (Vector2.Distance(Centre(i), entry) <= radius) { entranceLock[i] = true; cells.Add(i); }
                }
                entranceCells = cells.ToArray(); entranceEntry = entry; entranceRadius = radius; dirty = true;
            }
            // Only mapped free floor not proven indoors is shut: unmapped cells are not walkable anyway, and counting them as
            // walls would narrow the clearance of the passage beside them. A cell is looked at again on the next refresh.
            bool changed = false;
            foreach (int i in entranceCells)
            {
                bool hard = inside == null || (Occupancy(i) == 1 && !inside(Centre(i)));
                if (hard != entranceHard[i]) { entranceHard[i] = hard; changed = true; }
            }
            if (changed) dirty = true;
        }
        public bool EntranceLocked(Vector2 p) { int i = Cell(p); return entranceLock != null && i >= 0 && entranceLock[i]; }
        bool[] entranceLock, entranceHard; int[] entranceCells; Vector2 entranceEntry; float entranceRadius;
        /// <summary>Walkable and allowed as a goal: not on the entrance lock's passage (SetEntranceLock).</summary>
        internal bool Standable(int i) => Walkable(i) && (entranceLock == null || !entranceLock[i]);
        internal void PrepareNavigation()
        {
            if (!dirty) return;
            if (obstacleDistance == null) { obstacleDistance = new float[Count]; fireDistance = new float[Count]; }
            Distances(ObstacleCell, obstacleDistance);
            Distances(Dangerous, fireDistance); dirty = false;
        }
        /// <summary>A cell routes keep their clearance from: occupied, inside a confirmed person's keep-out, or on the hard part
        /// of the entrance lock.</summary>
        bool ObstacleCell(int i) => Occupancy(i) == 255 || (keepOut != null && keepOut[i]) || (entranceHard != null && entranceHard[i]);
        /// <summary>Distance from the point p to the nearest obstacle cell (ObstacleCell), measured to the cell's square, out to
        /// maxM (returned when nothing is nearer). ObstacleDistance is a cell centre's distance to obstacle cell centres, so a point
        /// near the edge of a cell can be farther from every obstacle than its cell's value says.</summary>
        internal float PointClearance(Vector2 p, float maxM)
        {
            int cell = Cell(p); if (cell < 0) return 0;
            int n = Mathf.CeilToInt(maxM / CellM) + 1, col = cell % Size, row = cell / Size; float h = CellM * .5f, best = maxM;
            for (int dy = -n; dy <= n; dy++) for (int dx = -n; dx <= n; dx++)
            {
                if (!Inside(col + dx, row + dy) || !ObstacleCell((row + dy) * Size + col + dx)) continue;
                var q = Centre((row + dy) * Size + col + dx);
                float ex = Mathf.Max(Mathf.Abs(p.x - q.x) - h, 0), ey = Mathf.Max(Mathf.Abs(p.y - q.y) - h, 0);
                best = Mathf.Min(best, Mathf.Sqrt(ex * ex + ey * ey));
            }
            return best;
        }
        /// <summary>Distance planned routes keep from fire: fire_clearance_m, lowered to emergency_fire_clearance_m only for an
        /// emergency exit (RobotBrain.CheckEmergencyExit).</summary>
        internal float FireClearance { get => fireClearance ?? c.fire_clearance_m; set => fireClearance = value; }
        float? fireClearance;
        internal bool Walkable(int i)
        {
            if (i < 0 || i >= Count || Occupancy(i) != 1) return false;
            // Include half a cell diagonal so clearance applies to the whole hot/occupied cell.
            float padding = CellM * .707107f;
            return obstacleDistance[i] > c.robot_radius_m + padding && fireDistance[i] > FireClearance + padding;
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
        /// <summary>Longest extent (bounding-box diagonal, metres) of the occupied structure touching a discM disc around p,
        /// followed through 8-connected occupied cells within a windowM half-width box. A wall runs on; a body lying on the
        /// floor is a compact blob (see candidate_wall_extent_m).</summary>
        public float OccupiedExtent(Vector2 p, float discM, float windowM, bool tallOnly = false)
        {
            int centre = Cell(p); if (centre < 0) return 0;
            int cx = centre % Size, cy = centre / Size, w = Mathf.CeilToInt(windowM / CellM), r = Mathf.CeilToInt(discM / CellM);
            var seen = new HashSet<int>(); var queue = new Queue<int>(); float best = 0;
            for (int dy = -r; dy <= r; dy++) for (int dx = -r; dx <= r; dx++)
            {
                int x = cx + dx, y = cy + dy; if (!Inside(x, y) || dx * dx + dy * dy > r * r) continue;
                int i = y * Size + x; if (OwnOccupancy(i) != 255 || (tallOnly && SeeOver(i)) || seen.Contains(i)) continue;
                // One component at a time: its bounding box is its extent.
                int minX = x, maxX = x, minY = y, maxY = y; seen.Add(i); queue.Enqueue(i);
                while (queue.Count > 0)
                {
                    int k = queue.Dequeue(), kx = k % Size, ky = k / Size;
                    minX = Math.Min(minX, kx); maxX = Math.Max(maxX, kx); minY = Math.Min(minY, ky); maxY = Math.Max(maxY, ky);
                    for (int ny = ky - 1; ny <= ky + 1; ny++) for (int nx = kx - 1; nx <= kx + 1; nx++)
                    {
                        if (!Inside(nx, ny) || Math.Abs(nx - cx) > w || Math.Abs(ny - cy) > w) continue;
                        int n = ny * Size + nx; if (seen.Contains(n) || OwnOccupancy(n) != 255 || (tallOnly && SeeOver(n))) continue;
                        seen.Add(n); queue.Enqueue(n);
                    }
                }
                best = Mathf.Max(best, Mathf.Sqrt((maxX - minX) * (maxX - minX) + (maxY - minY) * (maxY - minY)) * CellM);
            }
            return best;
        }
        internal float ObstacleDistance(int i) => i < 0 || i >= Count ? 0 : obstacleDistance[i];
        /// <summary>Indoors: within ceiling_radius_m of floor where this robot's own camera has seen a ceiling
        /// (ceiling_min_hits returns in the ceiling height band on a cell that is not a wall). No outline or floor plan
        /// is involved: the yard has no ceiling, and a warm shape seen through a window lies outside.</summary>
        /// <summary>The indoor layer for map snapshots: 2 where this robot's camera saw a ceiling away from walls, 1 within
        /// ceiling_radius_m of that across known free floor or on a ceiling sight line, 3 walled in, 0 elsewhere.</summary>
        public byte[] CopyIndoors()
        {
            if (roofDirty || roofDistance == null) { RoofDistances(); roofDirty = false; generation++; }
            var copy = new byte[Count];
            for (int i = 0; i < Count; i++)
                copy[i] = roofDistance[i] == 0 ? (byte)2 : roofDistance[i] <= c.ceiling_radius_m || sightIndoors[i] ? (byte)1 : OwnOccupancy(i) == 1 && Enclosed(i) ? (byte)3 : (byte)0;
            return copy;
        }
        /// <summary>
        /// A camera that stands indoors and sees a ceiling looks up at it across the floor in between. When that floor is
        /// known free all the way (no wall, nothing unmapped), it lies under the same roof, so it is indoors. This needs no
        /// outline: the camera must already be indoors or stand in a mapped wall opening with proven indoor floor
        /// behind it. A camera in the yard without that anchor marks nothing. A wall or unmapped floor stops the line. The ceiling radius alone left most
        /// of small rooms outside the search area, because the camera sees ceiling only about 4.7 m away (agent run 14:
        /// 530 of robot A's frontier cells inside the house were hidden as not indoors, 454 of them in the study, where a
        /// person lay). Sight-line floor does not seed the ceiling radius. Only the current ceiling view can extend it.
        /// </summary>
        void MarkCeilingSightLines(int camera)
        {
            if (!c.ceiling_sight_lines || frameRoof.Count == 0) return;
            bool cameraIndoors = sightIndoors[camera] || UnderCeiling(Centre(camera)) || (OwnOccupancy(camera) == 1 && Enclosed(camera));
            PrepareNavigation();
            Vector2 from = Centre(camera); float step = CellM * .5f;
            foreach (int roof in frameRoof)
            {
                if (roofHits[roof] < c.ceiling_min_hits) continue;
                // Only ceiling evidence clear of walls, as for ceiling seeds (RoofDistances): the top of a wall face is in
                // the ceiling height band too. Seen from the yard it drew sight lines along the outside of the facade
                // (held-out run 2: 98 % of robot B's ceiling-height points at 50 to 56 s lay within 0.4 m of a wall).
                if (OwnOccupancy(roof) == 255 || obstacleDistance[roof] < c.ceiling_wall_margin_m) continue;
                Vector2 to = Centre(roof);
                // At an unproven doorway the camera can use ceiling ahead only when its own wall jambs and a clear
                // line back to proven indoor floor anchor the view. Looking back at an indoor roof from the yard
                // does not qualify: there is no indoor anchor behind that outward camera-to-roof direction.
                if (!cameraIndoors && !DoorwayFromInside(from, (to - from).normalized)) continue;
                int steps = Mathf.CeilToInt(Vector2.Distance(from, to) / step);
                for (int k = 0; k <= steps; k++)
                {
                    int cell = Cell(Vector2.Lerp(from, to, steps == 0 ? 0 : k / (float)steps));
                    if (cell < 0 || OwnOccupancy(cell) != 1) break;
                    if (!sightIndoors[cell]) { sightIndoors[cell] = true; roofDirty = true; }
                }
                // The first accepted line proves the camera's own floor too. Reuse that fact for the rest of this
                // frame instead of rebuilding all ceiling distances for each new roof endpoint.
                cameraIndoors |= sightIndoors[camera];
            }
        }
        /// <summary>Known-free line, including both ends. An unknown cell cannot support an evidence move.</summary>
        internal bool OwnFreeLine(Vector2 from, Vector2 to)
        {
            int steps = Mathf.Max(1, Mathf.CeilToInt(Vector2.Distance(from, to) / (CellM * .5f)));
            for (int k = 0; k <= steps; k++) if (OwnOccupancy(Cell(Vector2.Lerp(from, to, k / (float)steps))) != 1) return false;
            return true;
        }

        /// <summary>A local opening in this robot's mapped wall line, seen from proven indoor floor. Each jamb must
        /// continue along the wall for ceiling_wall_margin_m. All floor to both jambs and back to the indoor anchor
        /// must be mapped free. The same frontier_peek_m bounds each ray and the later evidence move. This establishes
        /// an observation position, never indoor proof by itself. Only a fresh ceiling sight line supplies that proof.</summary>
        internal bool DoorwayFromInside(Vector2 at, Vector2 forward) => DoorwayFromInside(at, forward, out _);
        internal bool DoorwayFromInside(Vector2 at, Vector2 forward, out Vector2 openingCentre)
        {
            openingCentre = at;
            if (OwnOccupancy(Cell(at)) != 1 || forward.sqrMagnitude == 0 || c.frontier_peek_m <= 0) return false;
            float step = CellM * .5f;
            bool Jamb(Vector2 side, out float distance)
            {
                distance = 0;
                for (float d = step; d <= c.frontier_peek_m; d += step)
                {
                    byte own = OwnOccupancy(Cell(at + side * d));
                    if (own == 1) continue;
                    if (own != 255 || d < c.robot_radius_m) return false;
                    for (float wall = step; wall <= c.ceiling_wall_margin_m; wall += step)
                        if (OwnOccupancy(Cell(at + side * (d + wall))) != 255) return false;
                    distance = d; return true;
                }
                return false;
            }
            var across = new Vector2(-forward.y, forward.x);
            if (!Jamb(across, out float left) || !Jamb(-across, out float right)) return false;
            openingCentre = at + across * ((left - right) * .5f);
            for (float d = step; d <= c.frontier_peek_m; d += step)
            {
                Vector2 back = at - forward * d; int cell = Cell(back);
                if (OwnOccupancy(cell) != 1) return false;
                if (UnderCeiling(back) || Enclosed(cell)) return true;
            }
            return false;
        }

        ushort[] topCm;
        /// <summary>Heights above which an obstacle blocks the camera's view (camera mount height less
        /// see_over_clearance_m); set by the brain from its calibration.</summary>
        public float SeeOverMaxM { get; set; }
        /// <summary>The live 3D map's top-surface height per cell, in centimetres (0: no surface), from the OpenCV 5 TSDF
        /// volume the robot builds from its own depth frames (agent service /map3d). Same grid as this map.</summary>
        public void SetTopHeights(ushort[] cm)
        {
            if (cm == null || cm.Length != Count) throw new ArgumentException("Top-height layer must match the grid.");
            topCm = cm;
        }
        public float TopHeightM(int i) => topCm == null || i < 0 || i >= Count ? 0f : topCm[i] * .01f;
        /// <summary>An obstacle low enough for the camera to look over (a desk, a bed), by the live 3D map. Unknown
        /// height is not low.</summary>
        public bool SeeOver(int i) => topCm != null && i >= 0 && i < Count && topCm[i] > 0 && topCm[i] * .01f <= SeeOverMaxM;
        public bool Indoors(Vector2 p) => UnderCeiling(p) || (OwnOccupancy(Cell(p)) == 1 && Enclosed(Cell(p)));
        /// <summary>The ceiling half of Indoors: within ceiling_radius_m of wall-clear ceiling evidence, walked across known
        /// free floor.</summary>
        /// <summary>
        /// Floor beyond a frontier as far as this robot's map knows: cells mapped free by its own sensors that no camera on
        /// the team has searched, joined to p through such cells and within maxM of it. The fill starts from such cells within
        /// frontier_peek_m of p, the reach the planner gives a frontier's information (held-out seed 914 run 3: every
        /// frontier read 0 when it started only from cells beside p, because past furniture the unsearched floor begins up
        /// to that far away). With search_requires_ceiling only floor proven indoors counts (the fill still crosses the
        /// rest), so the yard past a front door adds nothing.
        /// Set indoorsOnly false only for wide-slot ranking, so an unproven region can compete for an agent option.
        /// This does not supply indoor evidence or change the reported indoor area.
        /// The frontier's own unsearched_m2 counts only cells beside the frontier, so a doorway onto a whole unsearched
        /// hallway read 1.5 m2 (held-out seed 914 run 1, robot B, which chose the kitchen at 8.5 m2 instead).
        /// </summary>
        public float UnsearchedFloorBeyond(Vector2 p, float maxM, bool indoorsOnly = true)
        {
            int start = Cell(p); if (start < 0) return 0;
            var mark = new HashSet<int>(); var queue = new Queue<int>(); int count = 0;
            bool Open(int n) => OwnOccupancy(n) == 1 && Unsearched(n) && Vector2.Distance(Centre(n), p) <= maxM;
            int sx = start % Size, sy = start / Size, reach = Mathf.Max(2, Mathf.CeilToInt(c.frontier_peek_m / CellM));
            for (int dy = -reach; dy <= reach; dy++) for (int dx = -reach; dx <= reach; dx++)
                if (dx * dx + dy * dy <= reach * reach && Inside(sx + dx, sy + dy)) { int n = (sy + dy) * Size + sx + dx; if (Open(n) && ClearLine(start, n) && mark.Add(n)) queue.Enqueue(n); }
            while (queue.Count > 0)
            {
                int cell = queue.Dequeue();
                // Ranking unproven floor (indoorsOnly false), floor this robot's LiDAR saw as open ground is the yard, not a
                // room behind a door (held-out seed 917, robot B: all three wide slots went to the yard by its front door).
                if (indoorsOnly ? !c.search_requires_ceiling || Indoors(Centre(cell)) : !openGround[cell]) count++;
                int x = cell % Size, y = cell / Size;
                for (int k = 0; k < 4; k++)
                {
                    int nx = x + (k == 0 ? 1 : k == 1 ? -1 : 0), ny = y + (k == 2 ? 1 : k == 3 ? -1 : 0);
                    if (!Inside(nx, ny)) continue;
                    int n = ny * Size + nx;
                    if (Open(n) && mark.Add(n)) queue.Enqueue(n);
                }
            }
            return count * CellM * CellM;
        }
        /// <summary>No cell this robot's own map holds as occupied lies on the straight line between two cells, so floor seen
        /// past a frontier is not counted through a wall (Codex review: a 1.2 m seed disc reached the room behind a thin wall).</summary>
        bool ClearLine(int from, int to, bool seeOver = false)
        {
            int x = from % Size, y = from / Size, ex = to % Size, ey = to / Size;
            int dx = Math.Abs(ex - x), dy = Math.Abs(ey - y), sx = x < ex ? 1 : -1, sy = y < ey ? 1 : -1, error = dx - dy;
            while (true)
            {
                int cell = y * Size + x;
                if (OwnOccupancy(cell) == 255 && (!seeOver || !SeeOver(cell))) return false;
                if (x == ex && y == ey) return true;
                int twice = 2 * error;
                if (twice > -dy) { error -= dy; x += sx; }
                if (twice < dx) { error += dx; y += sy; }
            }
        }
        /// <summary>A camera may look across unmapped floor. This checks only for mapped obstructions and supplies
        /// neither navigation clearance nor indoor evidence.</summary>
        internal bool EvidenceSightLineClear(Vector2 from, Vector2 to)
        {
            int a = Cell(from), b = Cell(to);
            return a >= 0 && b >= 0 && ClearLine(a, b);
        }
        /// <summary>Whether a camera sector can still search floor. Unknown floor keeps a look eligible; only known
        /// opaque obstacles hide it. A low obstacle with measured top height does not hide the whole view behind it.
        /// This is a prediction for choosing looks, never an observation or a searched-cell update.</summary>
        internal bool HasUnsearchedFloorInView(Vector2 eye, float heading, float leftRad, float rightRad, float rangeM, float minForwardM = 0)
        {
            var forward = new Vector2(-Mathf.Sin(heading), Mathf.Cos(heading));
            int start = Cell(eye); if (start < 0) return true;
            int reach = Mathf.CeilToInt(rangeM / CellM), sx = start % Size, sy = start / Size;
            for (int dy = -reach; dy <= reach; dy++) for (int dx = -reach; dx <= reach; dx++)
            {
                int x = sx + dx, y = sy + dy;
                Vector2 d = Origin + new Vector2(x + .5f, y + .5f) * CellM - eye;
                if (d.sqrMagnitude > rangeM * rangeM || Vector2.Dot(d, forward) + CellM * .707107f < minForwardM) continue;
                float bearing = BrainMath.Angle(heading, BrainMath.Heading(d));
                if (bearing < -rightRad * Mathf.Rad2Deg || bearing > leftRad * Mathf.Rad2Deg) continue;
                // Floor outside this finite map is unknown, so it cannot justify skipping a look.
                if (!Inside(x, y)) return true;
                int cell = y * Size + x;
                if (Unsearched(cell) && ClearLine(start, cell, true)) return true;
            }
            return false;
        }

        /// <summary>Connected regions of floor this robot mapped free that no camera on the team has searched (4-connected):
        /// a label per cell, -1 elsewhere. Labels only; no indoor test, which would cost rays per cell.</summary>
        public int[] UnsearchedRegions()
        {
            var label = new int[Count]; for (int i = 0; i < Count; i++) label[i] = -1;
            var queue = new Queue<int>(); int next = 0;
            for (int s = 0; s < Count; s++)
            {
                if (label[s] >= 0 || OwnOccupancy(s) != 1 || !Unsearched(s)) continue;
                label[s] = next; queue.Enqueue(s);
                while (queue.Count > 0)
                {
                    int cell = queue.Dequeue(), x = cell % Size, y = cell / Size;
                    for (int k = 0; k < 4; k++)
                    {
                        int nx = x + (k == 0 ? 1 : k == 1 ? -1 : 0), ny = y + (k == 2 ? 1 : k == 3 ? -1 : 0);
                        if (!Inside(nx, ny)) continue;
                        int n = ny * Size + nx;
                        if (label[n] < 0 && OwnOccupancy(n) == 1 && Unsearched(n)) { label[n] = next; queue.Enqueue(n); }
                    }
                }
                next++;
            }
            return label;
        }
        /// <summary>The labels (UnsearchedRegions) within frontier_peek_m of p: the regions a frontier at p opens onto.</summary>
        public HashSet<int> RegionsNear(int[] label, Vector2 p)
        {
            var found = new HashSet<int>(); int start = Cell(p); if (start < 0) return found;
            int sx = start % Size, sy = start / Size, reach = Mathf.Max(2, Mathf.CeilToInt(c.frontier_peek_m / CellM));
            for (int dy = -reach; dy <= reach; dy++) for (int dx = -reach; dx <= reach; dx++)
                if (dx * dx + dy * dy <= reach * reach && Inside(sx + dx, sy + dy)) { int n = (sy + dy) * Size + sx + dx; if (label[n] >= 0 && ClearLine(start, n)) found.Add(label[n]); }
            return found;
        }
        /// <summary>Geometry-only indoor connectivity to an earlier sensor map. A yard route cannot join two buildings.
        /// Fire is deliberately ignored here: it can block travel without changing which interior was explored.
        /// Four neighbours prevent a diagonal contact through wall corners from claiming a connection.</summary>
        public bool[] InteriorConnectedTo(bool[] anchors)
        {
            if (anchors == null || anchors.Length != Count) throw new ArgumentException("anchor mask must match the grid");
            PrepareNavigation();
            var connected = new bool[Count]; var queue = new Queue<int>();
            bool IndoorFloor(int i) => Occupancy(i) == 1 && UnderCeiling(Centre(i)) &&
                obstacleDistance[i] > c.robot_radius_m + CellM * .707107f;
            for (int i = 0; i < Count; i++)
                if (anchors[i] && IndoorFloor(i)) { connected[i] = true; queue.Enqueue(i); }
            while (queue.Count > 0)
            {
                int i = queue.Dequeue(), x = i % Size, y = i / Size;
                for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++)
                {
                    if (Math.Abs(dx) + Math.Abs(dy) != 1 || !Inside(x + dx, y + dy)) continue;
                    int n = (y + dy) * Size + x + dx;
                    if (!connected[n] && IndoorFloor(n)) { connected[n] = true; queue.Enqueue(n); }
                }
            }
            return connected;
        }

        public bool UnderCeiling(Vector2 p)
        {
            int i = Cell(p); if (i < 0) return false;
            if (roofDirty || roofDistance == null) { RoofDistances(); roofDirty = false; generation++; }
            return roofDistance[i] <= c.ceiling_radius_m || sightIndoors[i];
        }

        /// <summary>
        /// Free floor walled in on this robot's own map: of enclosure_rays rays cast from the cell, at least
        /// enclosure_min_closed_share reach a mapped wall within enclosure_range_m and at most enclosure_max_open_share run
        /// that far over floor mapped free or LiDAR open ground (rays into other unmapped space count as neither). A room
        /// is walled in even when its ceiling was never seen; the yard is not, since its LiDAR beams come back empty in
        /// wide spans (MarkOpenGround), whatever stands beside the facade. The ceiling rule alone
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
                    // Unmapped counts as neither, except where the LiDAR looked across it and saw nothing: the ray goes on.
                    else if (occupancy == 0 && !openGround[cell]) ended = true;
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
            float soft = keepSoft != null ? keepSoft[i] : 0;
            float passage = entranceLock != null && entranceLock[i] ? c.entrance_lock_cost_weight : 0;
            return 1 + c.heat_cost_weight * (strength * strength + warm * warm) + c.obstacle_cost_weight * edge * edge + c.keepout_cost_weight * soft * soft + passage;
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
        /// <summary>Distance from sensed free-cell centres to the boundary of known free floor. Unknown floor is a
        /// boundary too. Subtract half a cell because the transform measures between cell centres.</summary>
        public float[] KnownFreeClearance()
        {
            var result = new float[Count];
            Distances(i => OwnOccupancy(i) != 1 || i % Size == 0 || i % Size == Size - 1 || i / Size == 0 || i / Size == Size - 1, result);
            for (int i = 0; i < Count; i++) result[i] = Mathf.Max(0, result[i] - CellM / 2);
            return result;
        }

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
