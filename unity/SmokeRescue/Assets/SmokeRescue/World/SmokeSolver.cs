using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using UnityEngine;

namespace SmokeRescue.World
{
    /// <summary>
    /// Fixed-step two-layer soot transport on FireSim's grid, in kg, metres and seconds.
    /// Stam advection and implicit diffusion feed an aperture-weighted MAC pressure projection.
    /// Hot carrier volume sets the zone interface. Gas densities and entrainment are reduced-model
    /// closures, not a gas energy equation or an implementation of CFAST. See config/smoke.json.
    /// One lock serializes parallel sensor requests. No Unity clock, physics or random state is used.
    /// Determinism covers identical inputs on the same runtime and machine.
    /// </summary>
    public sealed class SmokeSolver
    {
        #pragma warning disable 0649
        sealed class Configuration
        {
            public int version, diffusion_iterations, pressure_iterations, snapshot_interval_steps, snapshot_count;
            public double step_s, soot_yield_kg_kg, extinction_m2_kg;
            public double upper_gas_density_kg_m3, lower_gas_density_kg_m3, plume_entrainment_m3_kj;
            public double smoke_diffusion_m2_s, kinematic_viscosity_m2_s, min_lower_depth_m;
            public double flame_mixing_per_s, interface_mixing_per_s, vent_exchange_m_s, max_advection_speed_m_s;
        }
        #pragma warning restore 0649

        public readonly struct CellState
        {
            public string Room { get; }
            public double UpperMassKg { get; }
            public double LowerMassKg { get; }
            public double UpperDepthM { get; }
            public double InterfaceHeightM { get; }
            public double UpperConcentrationKgM3 { get; }
            public double LowerConcentrationKgM3 { get; }
            public double VelocityX { get; }
            public double VelocityZ { get; }
            internal CellState(string room, double upper, double lower, double volume, double area,
                double ceiling, double vx, double vz)
            {
                Room = room; UpperMassKg = upper; LowerMassKg = lower;
                UpperDepthM = volume / area; InterfaceHeightM = ceiling - UpperDepthM;
                UpperConcentrationKgM3 = volume > 0 ? upper / volume : 0;
                LowerConcentrationKgM3 = lower / (area * ceiling - volume);
                VelocityX = vx; VelocityZ = vz;
            }
        }

        public readonly struct RoomState
        {
            public float UpperDepthM { get; }
            public float UpperDensity { get; }
            public float LowerDensity { get; }
            internal RoomState(double upper, double lower, double volume, double area, double ceiling, double extinction)
            {
                UpperDepthM = area > 0 ? (float)(volume / area) : 0;
                UpperDensity = volume > 0 ? (float)(3 * extinction * upper / volume) : 0;
                double lowerVolume = area * ceiling - volume;
                LowerDensity = lowerVolume > 0 ? (float)(3 * extinction * lower / lowerVolume) : 0;
            }
        }

        readonly struct Face
        {
            public readonly int a, b;
            public readonly bool alongX;
            public readonly double aperture;
            public readonly string opening;
            public Face(int a, int b, bool alongX, double aperture, string opening)
            { this.a = a; this.b = b; this.alongX = alongX; this.aperture = aperture; this.opening = opening; }
        }
        readonly struct Emission
        {
            public readonly int cell;
            public readonly double kw;
            public Emission(int cell, double kw) { this.cell = cell; this.kw = kw; }
        }
        sealed class Snapshot
        {
            public double[] upper, lower, volume, velocity;
            public double injected, vented;
        }
        sealed class Spray
        {
            public double time, flow, duration;
            public Vector2 nozzle;
            public float heading, range, halfAngle;
            public void Apply(FireSim fire) => fire.ScheduleWater(time, nozzle, heading, range, halfAngle, flow, duration);
        }

        readonly object gate = new object();
        readonly Configuration config;
        readonly FloorPlan plan;
        readonly string firePath;
        readonly int seed;
        readonly double h, area, ceiling, originX, originZ, maxUpperVolume;
        readonly string[] room;
        readonly int[] adjacent, faceAt, component;
        readonly double[] weight;
        readonly Face[] faces;
        readonly bool[] componentVents;
        readonly int[] componentSize;
        readonly double[] upper, lower, volume, velocity, vx, vz, nextX, nextZ;
        readonly double[] pressure, divergence, target, rhs, original, scratch, sumBefore, sumAfter;
        readonly Dictionary<string, int> roomNumbers = new Dictionary<string, int>(StringComparer.Ordinal);
        readonly int[] roomNumber;
        readonly double[] roomUpper, roomLower, roomVolume, roomArea;
        long roomCacheTick = -1;
        readonly List<Spray> sprays = new List<Spray>();
        readonly SortedList<long, Snapshot> snapshots = new SortedList<long, Snapshot>();
        readonly Dictionary<long, Emission[]> emissions = new Dictionary<long, Emission[]>();
        FireSim fire;
        long tick;
        double injected, vented;
        int lastReplaySteps;

        public int Width { get; }
        public int Height { get; }
        public float CellSize => (float)h;
        public float HeadStart { get; }
        public double Time { get { lock (gate) return tick * config.step_s; } }
        public double TotalInjectedKg { get { lock (gate) return injected; } }
        public double TotalVentedKg { get { lock (gate) return vented; } }
        public int LastReplaySteps { get { lock (gate) return lastReplaySteps; } }
        public double TotalMassKg
        {
            get { lock (gate) { double total = 0; for (int i = 0; i < room.Length; i++) total += upper[i] + lower[i]; return total; } }
        }

        public static SmokeSolver Load(FloorPlan plan, string firePath, string smokePath, int seed, float headStart = 0)
        {
            if (plan == null) throw new ArgumentNullException(nameof(plan));
            if (!Finite(headStart) || headStart < 0) throw new ArgumentOutOfRangeException(nameof(headStart));
            var cfg = JsonConvert.DeserializeObject<Configuration>(File.ReadAllText(smokePath));
            Validate(cfg, plan.wall_height);
            // Freeze geometry and opening states. Caller-owned plans may later be used by scene builders.
            var copy = JsonConvert.DeserializeObject<FloorPlan>(JsonConvert.SerializeObject(plan));
            return new SmokeSolver(copy, firePath, cfg, seed, headStart);
        }

        SmokeSolver(FloorPlan plan, string firePath, Configuration config, int seed, float headStart)
        {
            this.plan = plan; this.firePath = firePath; this.config = config; this.seed = seed; HeadStart = headStart;
            fire = FireSim.Load(plan, firePath, seed);
            if (Math.Abs(fire.StepSeconds - config.step_s) > 1e-12)
                throw new InvalidDataException("Smoke and fire step_s must match.");
            Width = fire.Width; Height = fire.Height; h = fire.CellSize; area = h * h;
            ceiling = plan.wall_height; originX = fire.Origin.x; originZ = fire.Origin.y;
            maxUpperVolume = area * (ceiling - config.min_lower_depth_m);
            int n = checked(Width * Height);
            room = new string[n]; adjacent = new int[n * 4]; faceAt = new int[n * 4]; weight = new double[n * 4];
            component = new int[n];
            for (int i = 0; i < n; i++) { room[i] = fire.GetCell(i % Width, i / Width).Room; component[i] = -1; }
            for (int r = 0; r < plan.rooms.Count; r++) roomNumbers.Add(plan.rooms[r].id, r);
            roomNumber = new int[n]; roomUpper = new double[plan.rooms.Count]; roomLower = new double[plan.rooms.Count];
            roomVolume = new double[plan.rooms.Count]; roomArea = new double[plan.rooms.Count];
            for (int i = 0; i < n; i++)
            {
                roomNumber[i] = room[i] == null ? -1 : roomNumbers[room[i]];
                if (roomNumber[i] >= 0) roomArea[roomNumber[i]] += area;
            }
            for (int i = 0; i < adjacent.Length; i++) { adjacent[i] = -1; faceAt[i] = -1; }
            var list = new List<Face>();
            for (int z = 0; z < Height; z++) for (int x = 0; x <= Width; x++)
                AddFace(list, x > 0 ? z * Width + x - 1 : -1, x < Width ? z * Width + x : -1,
                    true, originX + x * h, originZ + z * h);
            for (int z = 0; z <= Height; z++) for (int x = 0; x < Width; x++)
                AddFace(list, z > 0 ? (z - 1) * Width + x : -1, z < Height ? z * Width + x : -1,
                    false, originZ + z * h, originX + x * h);
            faces = list.ToArray();
            var sizes = new List<int>(); var vents = new List<bool>();
            var queue = new Queue<int>();
            for (int i = 0; i < n; i++)
            {
                if (room[i] == null || component[i] >= 0) continue;
                int c = sizes.Count, count = 0; bool vent = false; component[i] = c; queue.Enqueue(i);
                while (queue.Count > 0)
                {
                    int cell = queue.Dequeue(); count++;
                    for (int d = 0; d < 4; d++)
                    {
                        int e = cell * 4 + d, j = adjacent[e];
                        if (weight[e] <= 0) continue;
                        if (j < 0) { vent = true; continue; }
                        if (component[j] >= 0) continue;
                        component[j] = c; queue.Enqueue(j);
                    }
                }
                sizes.Add(count); vents.Add(vent);
            }
            componentSize = sizes.ToArray(); componentVents = vents.ToArray();
            upper = new double[n]; lower = new double[n]; volume = new double[n]; velocity = new double[faces.Length];
            vx = new double[n]; vz = new double[n]; nextX = new double[n]; nextZ = new double[n];
            pressure = new double[n]; divergence = new double[n]; target = new double[n]; rhs = new double[n];
            original = new double[n]; scratch = new double[n]; sumBefore = new double[sizes.Count]; sumAfter = new double[sizes.Count];
            SaveSnapshot();
        }

        void AddFace(List<Face> result, int a, int b, bool alongX, double line, double lo)
        {
            if (a >= 0 && room[a] == null) a = -1;
            if (b >= 0 && room[b] == null) b = -1;
            if (a < 0 && b < 0) return;
            string ra = a < 0 ? "outside" : room[a], rb = b < 0 ? "outside" : room[b];
            double aperture = ra == rb ? 1 : 0; string id = null;
            if (ra != rb)
            {
                OpeningOverlap(plan.doors, ra, rb, alongX, line, lo, ref aperture, ref id);
                OpeningOverlap(plan.windows, ra, rb, alongX, line, lo, ref aperture, ref id);
            }
            aperture = Math.Min(1, aperture);
            if (aperture <= 0) return;
            int face = result.Count; result.Add(new Face(a, b, alongX, aperture, id));
            int da = alongX ? 1 : 3, db = alongX ? 0 : 2;
            if (a >= 0) { adjacent[a * 4 + da] = b; faceAt[a * 4 + da] = face; weight[a * 4 + da] = aperture; }
            if (b >= 0) { adjacent[b * 4 + db] = a; faceAt[b * 4 + db] = face; weight[b * 4 + db] = aperture; }
        }

        void OpeningOverlap(List<FloorPlan.Door> openings, string a, string b, bool faceAlongX,
            double line, double lo, ref double aperture, ref string id)
        {
            if (openings == null) return;
            foreach (var d in openings)
            {
                if (d.between == null || d.between.Length != 2 || d.at == null || d.at.Length != 2 ||
                    !Finite(d.width) || d.width <= 0 || (d.wall != "x" && d.wall != "z"))
                    throw new InvalidDataException("Invalid smoke opening geometry.");
                double fraction;
                switch (d.state)
                {
                    case "closed": fraction = 0; break;
                    case "half": fraction = .5; break;
                    case "open": case "removed": fraction = 1; break;
                    default: throw new InvalidDataException("Unknown smoke opening state: " + d.state);
                }
                if (d.RunsAlongX == faceAlongX ||
                    !((d.between[0] == a && d.between[1] == b) || (d.between[0] == b && d.between[1] == a))) continue;
                double at = faceAlongX ? d.at[0] : d.at[1], centre = faceAlongX ? d.at[1] : d.at[0];
                if (Math.Abs(at - line) > 1e-6) continue;
                double half = d.width * fraction / 2;
                double overlap = Math.Max(0, Math.Min(lo + h, centre + half) - Math.Max(lo, centre - half));
                if (overlap <= 0) continue;
                aperture += overlap / h; id = d.id;
            }
        }

        /// <summary>Total clear face width for an opening, including fractional edge cells.</summary>
        public double OpenWidthM(string id)
        {
            double width = 0; foreach (var f in faces) if (f.opening == id) width += f.aperture * h; return width;
        }

        /// <summary>Accepts absolute fire time. Backward queries restore a checkpoint or rebuild from zero.</summary>
        public void AdvanceTo(double t) { lock (gate) Advance(t); }
        void Advance(double t)
        {
            if (!Finite(t) || t < 0 || t / config.step_s > int.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(t));
            long goal = (long)Math.Floor(t / config.step_s + 1e-9);
            bool rewind = goal < tick; lastReplaySteps = 0;
            if (rewind)
            {
                int at = snapshots.Count - 1;
                while (at >= 0 && snapshots.Keys[at] > goal) at--;
                if (at < 0) Reset();
                else Restore(snapshots.Keys[at], snapshots.Values[at]);
            }
            while (tick < goal)
            {
                Step(HeatAt(tick + 1)); tick++;
                roomCacheTick = -1;
                if (rewind) lastReplaySteps++;
                if (tick % config.snapshot_interval_steps == 0 && !snapshots.ContainsKey(tick)) SaveSnapshot();
            }
        }

        Emission[] HeatAt(long step)
        {
            if (emissions.TryGetValue(step, out var saved)) return saved;
            // During retained replays FireSim remains at the latest queried time. Its sparse heat
            // history supplies the earlier steps without rewinding that independent fire instance.
            if (Math.Abs(fire.Time - (step - 1) * config.step_s) > 1e-7)
                throw new InvalidOperationException("Missing retained fire heat history.");
            fire.AdvanceTo(step * config.step_s);
            var list = new List<Emission>();
            for (int i = 0; i < room.Length; i++)
            {
                if (room[i] == null) continue;
                double kw = fire.GetCell(i % Width, i / Width).HeatReleaseKw;
                if (kw > 0) list.Add(new Emission(i, kw));
            }
            var result = list.ToArray(); emissions.Add(step, result); return result;
        }

        void Step(Emission[] heat)
        {
            Array.Clear(target, 0, target.Length);
            foreach (var e in heat)
            {
                double soot = e.kw / fire.HeatOfCombustionKjKg * config.soot_yield_kg_kg * config.step_s;
                double hotGas = e.kw * config.plume_entrainment_m3_kj *
                    config.lower_gas_density_kg_m3 / config.upper_gas_density_kg_m3;
                upper[e.cell] += soot; injected += soot;
                volume[e.cell] += hotGas * config.step_s + soot / config.upper_gas_density_kg_m3;
                // A depth-averaged ceiling source. Closed components receive a distributed return
                // into the lower zone, so their Neumann pressure equation has a compatible RHS.
                target[e.cell] = hotGas / (area * ceiling);
            }
            Array.Clear(sumBefore, 0, sumBefore.Length);
            for (int i = 0; i < room.Length; i++) if (component[i] >= 0) sumBefore[component[i]] += target[i];
            for (int i = 0; i < room.Length; i++)
                if (component[i] >= 0 && !componentVents[component[i]])
                    target[i] -= sumBefore[component[i]] / componentSize[component[i]];

            ReconstructVelocity();
            Diffuse(vx, config.kinematic_viscosity_m2_s); Diffuse(vz, config.kinematic_viscosity_m2_s);
            Project(); ReconstructVelocity();
            Advect(vx, nextX); Advect(vz, nextZ);
            Array.Copy(nextX, vx, vx.Length); Array.Copy(nextZ, vz, vz.Length);
            Project(); ReconstructVelocity();
            Transport(upper, true); Transport(volume, true); Transport(lower, false);
            MixAndSpill(heat);
            Vent();
        }

        void ReconstructVelocity()
        {
            for (int i = 0; i < room.Length; i++)
            {
                int e = i * 4;
                vx[i] = Clamp((FaceVelocity(e) + FaceVelocity(e + 1)) * .5, config.max_advection_speed_m_s);
                vz[i] = Clamp((FaceVelocity(e + 2) + FaceVelocity(e + 3)) * .5, config.max_advection_speed_m_s);
            }
        }
        double FaceVelocity(int e) => faceAt[e] >= 0 ? velocity[faceAt[e]] : 0;

        void Project()
        {
            for (int f = 0; f < faces.Length; f++)
            {
                var face = faces[f]; var v = face.alongX ? vx : vz;
                velocity[f] = face.a < 0 ? v[face.b] : face.b < 0 ? v[face.a] : (v[face.a] + v[face.b]) * .5;
            }
            Array.Clear(pressure, 0, pressure.Length);
            for (int i = 0; i < room.Length; i++)
            {
                int e = i * 4;
                divergence[i] = (weight[e + 1] * FaceVelocity(e + 1) - weight[e] * FaceVelocity(e) +
                    weight[e + 3] * FaceVelocity(e + 3) - weight[e + 2] * FaceVelocity(e + 2)) / h;
            }
            for (int iteration = 0; iteration < config.pressure_iterations; iteration++)
                for (int i = 0; i < room.Length; i++)
                {
                    if (room[i] == null) continue;
                    double sum = 0, diagonal = 0;
                    for (int d = 0; d < 4; d++)
                    {
                        int e = i * 4 + d, j = adjacent[e]; double k = weight[e];
                        diagonal += k;
                        if (j >= 0) sum += k * pressure[j];
                        // Exterior pressure is zero. Solid faces have zero weight.
                    }
                    pressure[i] = diagonal > 0 ? (sum - (divergence[i] - target[i]) * area) / diagonal : 0;
                }
            for (int f = 0; f < faces.Length; f++)
            {
                var face = faces[f];
                velocity[f] -= ((face.b < 0 ? 0 : pressure[face.b]) - (face.a < 0 ? 0 : pressure[face.a])) / h;
            }
        }

        void Diffuse(double[] values, double coefficient)
        {
            if (coefficient == 0) return;
            Array.Copy(values, rhs, values.Length);
            double a = config.step_s * coefficient / area;
            for (int iteration = 0; iteration < config.diffusion_iterations; iteration++)
                for (int i = 0; i < values.Length; i++)
                {
                    if (room[i] == null) continue;
                    double sum = 0, diagonal = 0;
                    for (int d = 0; d < 4; d++)
                    {
                        int e = i * 4 + d, j = adjacent[e];
                        if (j < 0) continue;
                        double k = weight[e]; diagonal += k; sum += k * values[j];
                    }
                    values[i] = (rhs[i] + a * sum) / (1 + a * diagonal);
                }
        }

        void Advect(double[] values, double[] result)
        {
            for (int i = 0; i < values.Length; i++)
            {
                if (room[i] == null) { result[i] = 0; continue; }
                double x = i % Width + .5 - config.step_s * vx[i] / h;
                double z = i / Width + .5 - config.step_s * vz[i] / h;
                if (Trace(i, x, z) <= 0) { result[i] = values[i]; continue; }
                int ix = (int)Math.Floor(x - .5), iz = (int)Math.Floor(z - .5);
                double tx = x - .5 - ix, tz = z - .5 - iz, sum = 0, total = 0;
                for (int dz = 0; dz <= 1; dz++) for (int dx = 0; dx <= 1; dx++)
                {
                    int nx = ix + dx, nz = iz + dz;
                    if (nx < 0 || nx >= Width || nz < 0 || nz >= Height) continue;
                    int j = nz * Width + nx;
                    if (room[j] == null) continue;
                    double w = (dx == 0 ? 1 - tx : tx) * (dz == 0 ? 1 - tz : tz) * Trace(i, nx + .5, nz + .5);
                    sum += w * values[j]; total += w;
                }
                // Blocked interpolation weight stays in the departure cell. It never samples across a wall.
                result[i] = sum + Math.Max(0, 1 - total) * values[i];
            }
        }

        double Trace(int start, double endX, double endZ)
        {
            int x = start % Width, z = start / Width;
            int ex = (int)Math.Floor(endX), ez = (int)Math.Floor(endZ);
            if (ex < 0 || ex >= Width || ez < 0 || ez >= Height) return 0;
            double dx = endX - (x + .5), dz = endZ - (z + .5), path = 1;
            int sx = dx > 0 ? 1 : -1, sz = dz > 0 ? 1 : -1;
            double stepX = dx == 0 ? double.PositiveInfinity : 1 / Math.Abs(dx);
            double stepZ = dz == 0 ? double.PositiveInfinity : 1 / Math.Abs(dz);
            double crossX = .5 * stepX, crossZ = .5 * stepZ;
            while (x != ex || z != ez)
            {
                int i = z * Width + x;
                bool moveX = crossX <= crossZ, moveZ = crossZ <= crossX;
                // At a corner both incident faces must be open. A diagonal cannot cut a wall corner.
                if (moveX && moveZ && (weight[i * 4 + (sx > 0 ? 1 : 0)] == 0 || weight[i * 4 + (sz > 0 ? 3 : 2)] == 0)) return 0;
                if (moveX)
                {
                    int e = i * 4 + (sx > 0 ? 1 : 0);
                    if (adjacent[e] < 0 || weight[e] <= 0) return 0;
                    path = Math.Min(path, weight[e]); x += sx; crossX += stepX;
                }
                if (moveZ)
                {
                    i = z * Width + x;
                    int e = i * 4 + (sz > 0 ? 3 : 2);
                    if (adjacent[e] < 0 || weight[e] <= 0) return 0;
                    path = Math.Min(path, weight[e]); z += sz; crossZ += stepZ;
                }
            }
            return path;
        }

        void Transport(double[] values, bool advect)
        {
            Array.Copy(values, original, values.Length);
            Array.Clear(sumBefore, 0, sumBefore.Length); Array.Clear(sumAfter, 0, sumAfter.Length);
            for (int i = 0; i < values.Length; i++) if (component[i] >= 0) sumBefore[component[i]] += values[i];
            if (advect) { Advect(values, scratch); Array.Copy(scratch, values, values.Length); }
            Diffuse(values, config.smoke_diffusion_m2_s);
            for (int i = 0; i < values.Length; i++)
                if (component[i] >= 0) { values[i] = Math.Max(0, values[i]); sumAfter[component[i]] += values[i]; }
            // Semi-Lagrangian interpolation is not conservative. Restore each connected component's
            // own total before explicit vent losses. No correction can move soot through a sealed wall.
            for (int i = 0; i < values.Length; i++)
            {
                int c = component[i]; if (c < 0) continue;
                values[i] = sumAfter[c] > 0 ? values[i] * sumBefore[c] / sumAfter[c] : original[i];
            }
        }

        void MixAndSpill(Emission[] heat)
        {
            Array.Clear(scratch, 0, scratch.Length);
            foreach (var e in heat) scratch[e.cell] = config.flame_mixing_per_s;
            for (int i = 0; i < room.Length; i++)
            {
                if (room[i] == null) continue;
                if (volume[i] > maxUpperVolume)
                {
                    double overflow = upper[i] * (1 - maxUpperVolume / volume[i]);
                    upper[i] -= overflow; lower[i] += overflow; volume[i] = maxUpperVolume;
                }
                if (volume[i] <= 0) continue;
                double lv = area * ceiling - volume[i];
                double rate = scratch[i] + (volume[i] >= maxUpperVolume * (1 - 1e-9) ? config.interface_mixing_per_s : 0);
                double difference = upper[i] / volume[i] - lower[i] / lv;
                double transfer = difference * volume[i] * lv / (volume[i] + lv) * (1 - Math.Exp(-rate * config.step_s));
                upper[i] -= transfer; lower[i] += transfer;
            }
        }

        void Vent()
        {
            // Scalar advection above has reflecting exterior sampling. This boundary flux is its
            // conservative open-boundary replacement, with zero-soot outside inflow.
            foreach (var f in faces)
            {
                if (f.a >= 0 && f.b >= 0) continue;
                int i = f.a >= 0 ? f.a : f.b;
                int d = f.alongX ? (f.a >= 0 ? 1 : 0) : (f.a >= 0 ? 3 : 2);
                double outward = FaceVelocity(i * 4 + d) * (f.a >= 0 ? 1 : -1);
                double fraction = 1 - Math.Exp(-(config.vent_exchange_m_s + Math.Max(0, outward)) *
                    f.aperture * h / area * config.step_s);
                double lostUpper = upper[i] * fraction, lostLower = lower[i] * fraction;
                upper[i] -= lostUpper; lower[i] -= lostLower; volume[i] *= 1 - fraction;
                vented += lostUpper + lostLower;
            }
        }

        public CellState GetCell(int x, int z)
        {
            if (x < 0 || x >= Width || z < 0 || z >= Height) throw new ArgumentOutOfRangeException();
            lock (gate)
            {
                int i = z * Width + x;
                return new CellState(room[i], upper[i], lower[i], volume[i], area, ceiling, vx[i], vz[i]);
            }
        }

        /// <summary>Robot-clock seconds, including configured fire head start. Density equals 3 * extinction per metre.</summary>
        public float Density(Vector3 p, float t)
        {
            if (!Finite(t) || !Finite(p.x) || !Finite(p.y) || !Finite(p.z)) throw new ArgumentOutOfRangeException();
            if (p.y < 0 || p.y > ceiling) return 0;
            int x = (int)Math.Floor((p.x - originX) / h), z = (int)Math.Floor((p.z - originZ) / h);
            if (x < 0 || x >= Width || z < 0 || z >= Height || room[z * Width + x] == null) return 0;
            lock (gate)
            {
                Advance(RobotTime(t));
                int i = z * Width + x;
                double concentration = p.y >= ceiling - volume[i] / area && volume[i] > 0 ?
                    upper[i] / volume[i] : lower[i] / (area * ceiling - volume[i]);
                return (float)(3 * config.extinction_m2_kg * concentration);
            }
        }

        double RobotTime(float t)
        {
            if (!Finite(t)) throw new ArgumentOutOfRangeException(nameof(t));
            double ft = Math.Max(0, (double)t + HeadStart), nearest = Math.Round(ft / config.step_s);
            // Float sensor clocks can represent a tick just below its exact double value.
            if (Math.Abs(ft - nearest * config.step_s) <= 1e-7 * Math.Max(1, ft)) ft = nearest * config.step_s;
            return ft;
        }

        /// <summary>Mass/volume room averages for the renderer. Point sampling retains the cell interfaces.</summary>
        public RoomState GetRoomState(string id, float t)
        {
            lock (gate)
            {
                Advance(RobotTime(t));
                if (!roomNumbers.TryGetValue(id, out int r)) throw new ArgumentException("Unknown smoke room: " + id);
                if (roomCacheTick != tick)
                {
                    Array.Clear(roomUpper, 0, roomUpper.Length); Array.Clear(roomLower, 0, roomLower.Length);
                    Array.Clear(roomVolume, 0, roomVolume.Length);
                    for (int i = 0; i < room.Length; i++)
                    {
                        int number = roomNumber[i]; if (number < 0) continue;
                        roomUpper[number] += upper[i]; roomLower[number] += lower[i]; roomVolume[number] += volume[i];
                    }
                    roomCacheTick = tick;
                }
                return new RoomState(roomUpper[r], roomLower[r], roomVolume[r], roomArea[r], ceiling, config.extinction_m2_kg);
            }
        }

        void SaveSnapshot()
        {
            snapshots.Add(tick, new Snapshot { upper = (double[])upper.Clone(), lower = (double[])lower.Clone(),
                volume = (double[])volume.Clone(), velocity = (double[])velocity.Clone(), injected = injected, vented = vented });
            if (snapshots.Count <= config.snapshot_count) return;
            long old = snapshots.Keys[0]; snapshots.RemoveAt(0);
            long retained = snapshots.Keys[0];
            for (long t = old + 1; t <= retained; t++) emissions.Remove(t);
        }
        void Restore(long time, Snapshot s)
        {
            tick = time; injected = s.injected; vented = s.vented; roomCacheTick = -1;
            Array.Copy(s.upper, upper, upper.Length); Array.Copy(s.lower, lower, lower.Length);
            Array.Copy(s.volume, volume, volume.Length); Array.Copy(s.velocity, velocity, velocity.Length);
            ReconstructVelocity();
        }
        void Reset(FireSim replacement = null)
        {
            fire = replacement ?? FireSim.Load(plan, firePath, seed);
            if (replacement == null) foreach (var spray in sprays) spray.Apply(fire);
            tick = 0; injected = 0; vented = 0; roomCacheTick = -1;
            Array.Clear(upper, 0, upper.Length); Array.Clear(lower, 0, lower.Length); Array.Clear(volume, 0, volume.Length);
            Array.Clear(velocity, 0, velocity.Length); ReconstructVelocity();
            emissions.Clear(); snapshots.Clear(); SaveSnapshot();
        }

        /// <summary>Fire-time spray log, shared by FireSimHeat. A new command invalidates cached future smoke.</summary>
        public void ScheduleWater(double time, Vector2 nozzle, float heading, float range, float halfAngle, double flow, double duration)
        {
            lock (gate)
            {
                var spray = new Spray { time = time, nozzle = nozzle, heading = heading, range = range,
                    halfAngle = halfAngle, flow = flow, duration = duration };
                if (time >= fire.Time)
                {
                    // Normal live commands affect only steps not simulated yet. Preserve their history.
                    spray.Apply(fire); sprays.Add(spray); return;
                }
                // A preview may have simulated beyond a newly scheduled command. Rebuild that future.
                var replacement = FireSim.Load(plan, firePath, seed);
                foreach (var old in sprays) old.Apply(replacement);
                spray.Apply(replacement); // Validate before changing retained history.
                sprays.Add(spray); Reset(replacement);
            }
        }

        /// <summary>Canonical state bytes for recordings and deterministic regression checks. Cache layout is excluded.</summary>
        public byte[] StateBytes()
        {
            lock (gate)
            using (var stream = new MemoryStream())
            using (var writer = new BinaryWriter(stream))
            {
                writer.Write(tick); writer.Write(injected); writer.Write(vented);
                for (int i = 0; i < room.Length; i++) { writer.Write(upper[i]); writer.Write(lower[i]); writer.Write(volume[i]); }
                foreach (double v in velocity) writer.Write(v);
                writer.Flush(); return stream.ToArray();
            }
        }

        static bool Finite(double x) => !double.IsNaN(x) && !double.IsInfinity(x);
        static double Clamp(double value, double limit) => Math.Max(-limit, Math.Min(limit, value));
        static void Nonnegative(double x, string name)
        { if (!Finite(x) || x < 0) throw new InvalidDataException("Invalid smoke setting: " + name); }
        static void Positive(double x, string name)
        { if (!Finite(x) || x <= 0) throw new InvalidDataException("Invalid smoke setting: " + name); }
        static void Validate(Configuration c, double ceiling)
        {
            if (c == null || c.version != 1) throw new InvalidDataException("Unknown smoke configuration version.");
            Positive(c.step_s, nameof(c.step_s)); Positive(c.soot_yield_kg_kg, nameof(c.soot_yield_kg_kg));
            Positive(c.extinction_m2_kg, nameof(c.extinction_m2_kg)); Positive(c.upper_gas_density_kg_m3, nameof(c.upper_gas_density_kg_m3));
            Positive(c.lower_gas_density_kg_m3, nameof(c.lower_gas_density_kg_m3));
            Nonnegative(c.plume_entrainment_m3_kj, nameof(c.plume_entrainment_m3_kj));
            Nonnegative(c.smoke_diffusion_m2_s, nameof(c.smoke_diffusion_m2_s));
            Nonnegative(c.kinematic_viscosity_m2_s, nameof(c.kinematic_viscosity_m2_s));
            Nonnegative(c.flame_mixing_per_s, nameof(c.flame_mixing_per_s));
            Nonnegative(c.interface_mixing_per_s, nameof(c.interface_mixing_per_s));
            Nonnegative(c.vent_exchange_m_s, nameof(c.vent_exchange_m_s));
            Positive(c.min_lower_depth_m, nameof(c.min_lower_depth_m)); Positive(c.max_advection_speed_m_s, nameof(c.max_advection_speed_m_s));
            if (c.min_lower_depth_m >= ceiling || c.diffusion_iterations < 1 || c.pressure_iterations < 1 ||
                c.snapshot_interval_steps < 1 || c.snapshot_count < 2)
                throw new InvalidDataException("Invalid smoke layer depth, iteration count or checkpoint settings.");
        }
    }
}
