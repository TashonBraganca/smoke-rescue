using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using Newtonsoft.Json;
using UnityEngine;

namespace SmokeRescue.World
{
    /// <summary>
    /// Deterministic fire logic, owned by World. Never expose it to robot-side code.
    /// Distances use Unity world x/z metres and y above the floor; temperatures are Celsius.
    /// This is an estimated grid/room model, not CFD, CFAST, or a structural solver.
    /// Calls on one instance must be serialized. No scene, physics, Unity clock, or global RNG is used.
    /// Reproducibility covers the same config, plan, runtime and machine. Floating-point results
    /// across different CPU/runtime implementations are not promised to be bit-identical.
    /// </summary>
    public sealed class FireSim
    {
        public readonly struct CellState
        {
            public string Room { get; }
            public float InitialFuelKg { get; }
            public float FuelRemainingKg { get; }
            public float SurfaceTemperatureC { get; }
            public float FlameIntensity { get; }
            /// <summary>Mean retained heat release during the last fixed step, in kW, after water cooling.</summary>
            public double HeatReleaseKw { get; }
            public float BurntFraction { get; }
            public bool Collapsed { get; }
            internal CellState(string room, Cell c)
            {
                Room = room; InitialFuelKg = (float)c.initialFuel; FuelRemainingKg = (float)c.fuel;
                SurfaceTemperatureC = (float)c.temperature; FlameIntensity = (float)c.flame;
                HeatReleaseKw = c.heatReleaseKw;
                BurntFraction = c.initialFuel > 0 ? (float)Clamp(1 - c.fuel / c.initialFuel, 0, 1) : 0;
                Collapsed = c.collapsed;
            }
        }

        public readonly struct RoomState
        {
            public string Room { get; }
            public float UpperTemperatureC { get; }
            public float LowerTemperatureC { get; }
            public float LayerHeight { get; }
            public float SmokeDensity { get; }
            public double FlashoverTime { get; }
            public double HotExposureSeconds { get; }
            internal RoomState(string id, Gas g, double ambient, double lowerFraction)
            {
                Room = id; UpperTemperatureC = (float)g.temperature;
                LowerTemperatureC = (float)(ambient + lowerFraction * (g.temperature - ambient));
                LayerHeight = (float)g.height; SmokeDensity = (float)g.smoke;
                FlashoverTime = g.flashoverTime; HotExposureSeconds = g.hotSeconds;
            }
        }

        public readonly struct CollapseEvent
        {
            public double Time { get; }
            public Rect CellRect { get; }
            public string Room { get; }
            public double HotExposureSeconds { get; }
            internal CollapseEvent(double time, Rect rect, string room, double exposure)
            { Time = time; CellRect = rect; Room = room; HotExposureSeconds = exposure; }
        }

        public readonly struct BurningRegion
        {
            public Vector3 Centre { get; }
            public float Radius { get; }
            public float Intensity { get; }
            internal BurningRegion(Vector3 centre, float radius, float intensity)
            { Centre = centre; Radius = radius; Intensity = intensity; }
        }

        // Config DTOs use fields to match the JSON. Extra source/justification fields are descriptive.
        #pragma warning disable 0649 // Json.NET fills these configuration fields.
        sealed class Configuration
        {
            public int version;
            public double cell_size_m, step_s, ambient_c, flashover_threshold_c;
            public List<Ignition> ignitions;
            public Dictionary<string, Material> materials;
            public Dictionary<string, string> surfaces;
            public Dictionary<string, Furniture> furniture;
            public Spread spread;
            public GasConfig gas;
            public WaterConfig water;
            public CollapseConfig collapse;
            public QueryConfig query;
        }
        sealed class Ignition { public string room; public double[] at; public double time_s, radius_m; }
        sealed class Material { public double fuel_kg_m2, ignition_c; }
        sealed class Furniture { public string material; public double[] size_m; }
        sealed class Spread
        {
            // Zero retains the uncapped model for older standalone diagnostic configs.
            public double growth_alpha_kw_s2;
            public double neighbour_heat_per_s, surface_cooling_per_s, gas_surface_heat_per_s;
            public double flashover_surface_heat_c_s, fuel_burn_kg_m2_s, flame_ramp_s, flame_relaxation_s;
            public double flame_min_c, flame_max_c, max_temperature_c, fuel_variation_fraction, heat_of_combustion_kj_kg;
        }
        sealed class GasConfig
        {
            public double effective_heat_capacity_kj_m2_k, heat_loss_kw_m2_k, convective_fraction;
            public double door_exchange_m3_s_m, outside_exchange_m3_s_m, gas_heat_capacity_kj_m3_k;
            public double entrainment_m3_s_kw, min_layer_height_m, initial_layer_depth_m;
            public double lower_temperature_fraction, smoke_production_per_cell_s, smoke_reference_volume_m3;
            public double smoke_decay_per_s, lower_smoke_fraction;
        }
        sealed class WaterConfig
        {
            public double latent_heat_kj_l, sensible_heat_kj_l, application_efficiency, wet_seconds_per_litre;
        }
        sealed class CollapseConfig
        {
            public double temperature_c, duration_s, interval_s, door_clear_width_m;
            public int max_events;
            public string[] eligible_ceilings;
        }
        sealed class QueryConfig { public double flame_heating_radius_m, flame_heating_c, plume_height_scale_m; }

        #pragma warning restore 0649

        internal sealed class Cell
        {
            public int room = -1;
            public int origin = -1;
            public double initialFuel, fuel, temperature, flame, burnIntensity, wetUntil, heatReleaseKw, ignition = double.MaxValue;
            public bool burning, collapsed, protectedApproach;
        }
        internal sealed class Gas
        {
            public string id;
            public double area, temperature, height, smoke, hotSeconds, flashoverTime = -1;
            public long hotTicks;
            public bool canCollapse, hasFuel;
        }
        sealed class Opening
        {
            public int a, b;
            public bool alongX;
            public double x, z, width;
        }

        sealed class Spray
        {
            public double start, end, flow;
            public int[] targets;
        }
        readonly List<Spray> sprays = new List<Spray>();
        readonly Rect[] roomRects;
        readonly double[] waterLitres;
        /// <summary>Actual flame and surface energy absorbed, excluding unused cooling capacity, in kJ.</summary>
        public double TotalWaterEnergyRemovedKj { get; private set; }

        readonly Configuration config;
        readonly int seed;
        readonly double originX, originZ, ceiling, doorHead;
        readonly Cell[] cells;
        readonly Gas[] rooms;
        readonly Dictionary<string, int> roomIndices = new Dictionary<string, int>(StringComparer.Ordinal);
        readonly List<Opening> openings = new List<Opening>();
        readonly List<CollapseEvent> collapses = new List<CollapseEvent>();
        readonly ReadOnlyCollection<CollapseEvent> collapseView;
        readonly int[][] neighbours;
        readonly double[] nextTemperature, nextFlame, heatKw, smokeProduction, gasHeatFlow, smokeFlow, volumeFlow;
        readonly double[] nextBurnIntensity, potentialMass, originPotentialKw, originScale, originStart;
        readonly int[] nextOrigin;
        // Attribute room temperature excess to origins using the same linear gas balance.
        // A gas-ignited cell inherits the largest contribution; it does not create another growth budget.
        readonly double[,] originGasTemperature, originGasFlow, originRoomHeatKw;
        readonly bool[] ignitionApplied, hotPreviously;
        // Walls are stored as separate segments with the door gaps already removed, for preview only.
        readonly List<double[]> walls = new List<double[]>();
        long tick;
        double lastRequestedTime, lastCollapseTime = double.NegativeInfinity;

        public int Width { get; }
        public int Height { get; }
        public float CellSize => (float)config.cell_size_m;
        public Vector2 Origin => new Vector2((float)originX, (float)originZ);
        public double Time => tick * config.step_s;
        public double StepSeconds => config.step_s;
        public double HeatOfCombustionKjKg => config.spread.heat_of_combustion_kj_kg;
        public IReadOnlyList<CollapseEvent> Collapses => collapseView;
        // A snapshot prevents a caller from modifying live room state.
        public IReadOnlyList<RoomState> RoomStates
        {
            get
            {
                var result = new RoomState[rooms.Length];
                for (int i = 0; i < rooms.Length; i++) result[i] = RoomSnapshot(i);
                return Array.AsReadOnly(result);
            }
        }

        public static FireSim Load(FloorPlan plan, string firePath, int seed)
        {
            if (plan == null) throw new ArgumentNullException(nameof(plan));
            if (firePath == null) throw new ArgumentNullException(nameof(firePath));
            var cfg = JsonConvert.DeserializeObject<Configuration>(File.ReadAllText(firePath));
            ValidateConfig(cfg);
            return new FireSim(plan, cfg, seed);
        }

        FireSim(FloorPlan plan, Configuration cfg, int seed)
        {
            config = cfg; this.seed = seed;
            ValidatePlan(plan);
            originX = plan.footprint[0]; originZ = plan.footprint[1]; ceiling = plan.wall_height; doorHead = plan.door_height;
            Width = GridCount(plan.footprint[2] - originX); Height = GridCount(plan.footprint[3] - originZ);
            cells = new Cell[checked(Width * Height)];
            nextTemperature = new double[cells.Length]; nextFlame = new double[cells.Length];
            nextBurnIntensity = new double[cells.Length]; potentialMass = new double[cells.Length];
            nextOrigin = new int[cells.Length];
            originPotentialKw = new double[cfg.ignitions.Count]; originScale = new double[cfg.ignitions.Count];
            originStart = new double[cfg.ignitions.Count];
            for (int n = 0; n < originStart.Length; n++) originStart[n] = -1;
            rooms = new Gas[plan.rooms.Count];
            originGasTemperature = new double[cfg.ignitions.Count, rooms.Length];
            originGasFlow = new double[cfg.ignitions.Count, rooms.Length];
            originRoomHeatKw = new double[cfg.ignitions.Count, rooms.Length];
            roomRects = new Rect[rooms.Length];
            waterLitres = new double[cells.Length];
            for (int r = 0; r < rooms.Length; r++)
            {
                var room = plan.rooms[r];
                roomRects[r] = room.Rect;
                roomIndices.Add(room.id, r);
                rooms[r] = new Gas { id = room.id, area = room.Rect.width * room.Rect.height,
                    temperature = cfg.ambient_c, height = ceiling - cfg.gas.initial_layer_depth_m,
                    canCollapse = Array.IndexOf(cfg.collapse.eligible_ceilings, room.ceiling) >= 0 };
                GetSurface(room.floor); GetSurface(room.walls); GetSurface(room.ceiling);
            }
            foreach (var d in plan.doors)
            {
                int a = RoomIndex(d.between[0]), b = RoomIndex(d.between[1]);
                if (a < 0 && b < 0) throw new InvalidDataException("Door has no interior room: " + d.id);
                openings.Add(new Opening { a = a, b = b, alongX = d.RunsAlongX, x = d.at[0], z = d.at[1],
                    width = d.width * (d.state == "closed" ? 0 : d.state == "half" ? .5 : 1) });
            }
            for (int z = 0; z < Height; z++) for (int x = 0; x < Width; x++)
            {
                int i = Index(x, z); Vector2 p = CellCentre(x, z);
                var cell = new Cell { temperature = cfg.ambient_c };
                for (int r = 0; r < rooms.Length; r++)
                    if (plan.rooms[r].Rect.Contains(p)) { cell.room = r; break; }
                cells[i] = cell;
                if (cell.room < 0) continue;
                var room = plan.rooms[cell.room];
                AddFuel(cell, GetSurface(room.floor), cfg.cell_size_m * cfg.cell_size_m);
                AddFuel(cell, GetSurface(room.ceiling), cfg.cell_size_m * cfg.cell_size_m);
                // Assign the actual vertical finish area to cells along each room edge. No fuel
                // from a neighbouring room is counted. Door openings omit this finish band.
                double edgeArea = cfg.cell_size_m * ceiling;
                if (x == 0 || !room.Rect.Contains(CellCentre(x - 1, z))) AddWallFuel(cell, room, p, true, room.rect[0], edgeArea);
                if (x == Width - 1 || !room.Rect.Contains(CellCentre(x + 1, z))) AddWallFuel(cell, room, p, true, room.rect[2], edgeArea);
                if (z == 0 || !room.Rect.Contains(CellCentre(x, z - 1))) AddWallFuel(cell, room, p, false, room.rect[1], edgeArea);
                if (z == Height - 1 || !room.Rect.Contains(CellCentre(x, z + 1))) AddWallFuel(cell, room, p, false, room.rect[3], edgeArea);
                cell.protectedApproach = IsDoorApproach(cell.room, CellRect(x, z));
            }
            AddFurniture(plan);
            for (int i = 0; i < cells.Length; i++)
            {
                // Stable integer mixing, independent of System.Random and UnityEngine.Random.
                double variation = 1 + cfg.spread.fuel_variation_fraction * (2 * UnitNoise(seed, i) - 1);
                cells[i].initialFuel *= variation; cells[i].fuel = cells[i].initialFuel;
                if (cells[i].room >= 0 && cells[i].initialFuel > 0) rooms[cells[i].room].hasFuel = true;
            }
            neighbours = new int[cells.Length][];
            for (int z = 0; z < Height; z++) for (int x = 0; x < Width; x++)
            {
                var adjacent = new List<int>(4);
                AddNeighbour(adjacent, x, z, x - 1, z); AddNeighbour(adjacent, x, z, x + 1, z);
                AddNeighbour(adjacent, x, z, x, z - 1); AddNeighbour(adjacent, x, z, x, z + 1);
                neighbours[Index(x, z)] = adjacent.ToArray();
            }
            heatKw = new double[rooms.Length]; smokeProduction = new double[rooms.Length];
            gasHeatFlow = new double[rooms.Length]; smokeFlow = new double[rooms.Length]; volumeFlow = new double[rooms.Length];
            ignitionApplied = new bool[cfg.ignitions.Count]; hotPreviously = new bool[rooms.Length];
            foreach (var ignition in cfg.ignitions)
            {
                int r = RoomIndex(ignition.room);
                int i = PointIndex(new Vector2((float)ignition.at[0], (float)ignition.at[1]));
                if (r < 0 || i < 0 || cells[i].room != r || cells[i].fuel <= 0)
                    throw new InvalidDataException("Ignition must name a fuel-bearing cell in its room.");
            }
            BuildPreviewWalls(plan);
            collapseView = collapses.AsReadOnly();
            ApplyIgnitions();
        }

        /// <summary>
        /// Advances whole fixed steps from zero, never integrates a fractional remainder.
        /// Time is floor(t/step)*step; tick-boundary roundoff within 1e-9 ticks snaps up.
        /// Equal t is a no-op. Backward requests and nonfinite/negative t throw.
        /// </summary>
        public void AdvanceTo(double t)
        {
            if (!Finite(t) || t < 0 || t < lastRequestedTime || t / config.step_s > int.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(t), "Time must be finite and monotone within the supported tick range.");
            long target = (long)Math.Floor(t / config.step_s + 1e-9);
            while (tick < target)
            {
                Step(); tick++; ApplyIgnitions();
            }
            lastRequestedTime = t;
        }

        /// <summary>
        /// Schedules a fixed cone in world x/z metres. Heading 0 points north, positive yaw turns east.
        /// Each tick integrates its overlap with [t, t + durationS), sharing flow equally among visible
        /// cell centres, including cold cells. Exact room edges and current door widths block sight.
        /// All inputs must be finite and nonnegative. Normalize signed yaw to [0, 2*pi) before scheduling.
        /// Schedules are ordered by start time, with insertion order breaking ties. No past scheduling.
        /// </summary>
        public void ScheduleWater(double t, Vector2 nozzleXZ, float headingRad, float rangeM,
            float halfAngleDeg, double flowLps, double durationS)
        {
            WaterInput(t, nameof(t)); WaterInput(rangeM, nameof(rangeM));
            WaterInput(halfAngleDeg, nameof(halfAngleDeg)); WaterInput(flowLps, nameof(flowLps));
            WaterInput(durationS, nameof(durationS));
            if (t < lastRequestedTime || !Finite(t + durationS)) throw new ArgumentOutOfRangeException(nameof(t));
            WaterInput(nozzleXZ.x, nameof(nozzleXZ)); WaterInput(nozzleXZ.y, nameof(nozzleXZ));
            WaterInput(headingRad, nameof(headingRad));
            if (halfAngleDeg > 180) throw new ArgumentOutOfRangeException(nameof(halfAngleDeg));
            // Reject overflow before a tick can change any state.
            double energyPerLitre = config.water.latent_heat_kj_l + config.water.sensible_heat_kj_l;
            if (!Finite(flowLps * durationS * energyPerLitre) ||
                !Finite(t + durationS + flowLps * durationS * config.water.wet_seconds_per_litre))
                throw new ArgumentOutOfRangeException(nameof(flowLps));
            var targets = new List<int>();
            int nozzle = PointIndex(nozzleXZ);
            if (nozzle >= 0 && cells[nozzle].room >= 0)
            {
                double hx = Math.Sin(headingRad), hz = Math.Cos(headingRad);
                double cosine = Math.Cos(halfAngleDeg * Math.PI / 180);
                for (int i = 0; i < cells.Length; i++)
                {
                    if (cells[i].room < 0) continue;
                    Vector2 p = CellCentre(i % Width, i / Width);
                    double dx = p.x - nozzleXZ.x, dz = p.y - nozzleXZ.y;
                    double distance = Math.Sqrt(dx * dx + dz * dz);
                    if (distance > rangeM || (distance > 0 && (dx * hx + dz * hz) / distance < cosine)) continue;
                    if (WaterVisible(nozzleXZ, p)) targets.Add(i);
                }
            }
            var spray = new Spray { start = t, end = t + durationS, flow = flowLps, targets = targets.ToArray() };
            int at = sprays.Count;
            while (at > 0 && sprays[at - 1].start > t) at--;
            sprays.Insert(at, spray);
        }

        static void WaterInput(double value, string name)
        { if (!Finite(value) || value < 0) throw new ArgumentOutOfRangeException(name); }

        bool WaterVisible(Vector2 p, Vector2 q)
        {
            // The preview rounds apertures to grid cells. Test actual room edges instead so half-open
            // doors retain their exact clear width and a diagonal ray cannot cut through a wall corner.
            for (int r = 0; r < roomRects.Length; r++)
            {
                var rect = roomRects[r];
                if (WaterWall(p, q, r, true, rect.yMin, rect.xMin, rect.xMax) ||
                    WaterWall(p, q, r, true, rect.yMax, rect.xMin, rect.xMax) ||
                    WaterWall(p, q, r, false, rect.xMin, rect.yMin, rect.yMax) ||
                    WaterWall(p, q, r, false, rect.xMax, rect.yMin, rect.yMax)) return false;
            }
            return true;
        }

        bool WaterWall(Vector2 p, Vector2 q, int room, bool alongX, double line, double lo, double hi)
        {
            double a = alongX ? p.y : p.x, b = alongX ? q.y : q.x;
            if (a == b)
            {
                // A ray lying on a wall is blocked, including a nozzle placed on that wall.
                if (a != line) return false;
                double u = alongX ? p.x : p.y, v = alongX ? q.x : q.y;
                return Math.Max(u, v) >= lo && Math.Min(u, v) <= hi;
            }
            double t = (line - a) / (b - a);
            if (t < 0 || t > 1) return false;
            double along = alongX ? p.x + t * (q.x - p.x) : p.y + t * (q.y - p.y);
            if (along < lo || along > hi) return false;
            foreach (var d in openings)
                if ((d.a == room || d.b == room) && d.alongX == alongX &&
                    (alongX ? d.z : d.x) == line && Math.Abs(along - (alongX ? d.x : d.z)) < d.width / 2)
                    return false;
            return true;
        }

        void ApplyWater(double dt)
        {
            Array.Clear(waterLitres, 0, waterLitres.Length);
            foreach (var spray in sprays)
            {
                if (spray.start >= Time + dt) break;
                double overlap = Math.Max(0, Math.Min(Time + dt, spray.end) - Math.Max(Time, spray.start));
                if (overlap <= 0 || spray.flow == 0 || spray.targets.Length == 0) continue;
                double litres = spray.flow * overlap / spray.targets.Length;
                foreach (int i in spray.targets)
                {
                    waterLitres[i] += litres;
                    // Retained water buys an additive dry-out time after the last wetted part of a tick.
                    cells[i].wetUntil = Math.Max(cells[i].wetUntil, Math.Min(Time + dt, spray.end)) +
                        litres * config.water.wet_seconds_per_litre;
                }
            }
        }

        void ApplyIgnitions()
        {
            for (int n = 0; n < ignitionApplied.Length; n++)
            {
                var fire = config.ignitions[n];
                if (ignitionApplied[n] || Time + 1e-10 < fire.time_s) continue;
                int room = RoomIndex(fire.room);
                int centre = PointIndex(new Vector2((float)fire.at[0], (float)fire.at[1]));
                for (int z = 0; z < Height; z++) for (int x = 0; x < Width; x++)
                {
                    int i = Index(x, z); var c = cells[i]; var p = CellCentre(x, z);
                    double dx = p.x - fire.at[0], dz = p.y - fire.at[1];
                    if (c.burning || c.wetUntil > Time || c.room != room || c.fuel <= 0 || (i != centre && dx * dx + dz * dz > fire.radius_m * fire.radius_m)) continue;
                    c.origin = n; originStart[n] = Time;
                    c.burning = true; c.temperature = Math.Max(c.temperature, c.ignition);
                    c.burnIntensity = Math.Max(c.burnIntensity, .01); c.flame = c.burnIntensity;
                }
                ignitionApplied[n] = true;
            }
        }

        void Step()
        {
            double dt = config.step_s, area = config.cell_size_m * config.cell_size_m;
            var s = config.spread; var g = config.gas;
            ApplyWater(dt);
            Array.Clear(heatKw, 0, heatKw.Length); Array.Clear(smokeProduction, 0, smokeProduction.Length);
            Array.Clear(originPotentialKw, 0, originPotentialKw.Length);
            Array.Clear(originRoomHeatKw, 0, originRoomHeatKw.Length);
            // Every neighbour read comes from the previous tick, never a partly updated row.
            for (int i = 0; i < cells.Length; i++)
            {
                var c = cells[i]; nextTemperature[i] = c.temperature; nextFlame[i] = 0; c.heatReleaseKw = 0;
                nextBurnIntensity[i] = 0; potentialMass[i] = 0;
                nextOrigin[i] = c.origin;
                if (c.room < 0) continue;
                var room = rooms[c.room];
                double warming = 0, strongest = 0;
                // Fuel reheated while wet may still be above ignition when it dries,
                // even if the heat source has since cooled. Keep that source's lineage.
                int source = c.origin >= 0 ? c.origin : GasOrigin(c.room);
                // One-sided radiative preheating proxy, not energy-conserving solid conduction.
                foreach (int j in neighbours[i])
                {
                    double heat = Math.Max(0, cells[j].temperature - c.temperature) * s.neighbour_heat_per_s;
                    warming += heat;
                    if (cells[j].origin >= 0 && heat > strongest) { strongest = heat; source = cells[j].origin; }
                }
                double gasWarming = Math.Max(0, room.temperature - c.temperature) * s.gas_surface_heat_per_s;
                if (room.temperature >= config.flashover_threshold_c)
                    gasWarming += s.flashover_surface_heat_c_s * Math.Max(0, (room.temperature - c.temperature) / room.temperature);
                warming += gasWarming;
                if (gasWarming > strongest) source = GasOrigin(c.room);
                double temperature = c.temperature + dt * (warming - s.surface_cooling_per_s * (c.temperature - config.ambient_c));
                nextTemperature[i] = temperature;
                if (c.fuel > 0 && (c.burning || (c.wetUntil <= Time && temperature >= c.ignition && source >= 0)))
                {
                    if (!c.burning) nextOrigin[i] = source;
                    double intensity = Math.Min(1, c.burnIntensity + dt / s.flame_ramp_s);
                    intensity = Math.Min(intensity, c.fuel / (s.fuel_burn_kg_m2_s * area));
                    nextBurnIntensity[i] = intensity;
                    potentialMass[i] = Math.Min(c.fuel, dt * s.fuel_burn_kg_m2_s * area * intensity);
                    originPotentialKw[nextOrigin[i]] += potentialMass[i] * s.heat_of_combustion_kj_kg / dt;
                }
            }
            for (int n = 0; n < originScale.Length; n++)
            {
                double age = originStart[n] < 0 ? 0 : Math.Max(0, Time - originStart[n]);
                // Exact mean of alpha*t^2 over this tick. The factor 1/3 is its integral,
                // not a fitted coefficient. NFPA 72 (2002), Annex B, equations B.4/B.5.
                double budget = s.growth_alpha_kw_s2 * (age * age + age * dt + dt * dt / 3);
                originScale[n] = s.growth_alpha_kw_s2 == 0 || originPotentialKw[n] <= budget
                    ? 1 : budget / originPotentialKw[n];
            }
            for (int i = 0; i < cells.Length; i++)
            {
                var c = cells[i];
                if (c.room < 0) continue;
                double temperature = nextTemperature[i];
                double waterEnergy = waterLitres[i] * (config.water.latent_heat_kj_l + config.water.sensible_heat_kj_l) *
                    config.water.application_efficiency;
                double remainingWaterEnergy = waterEnergy;
                bool flameAbsorbed = false;
                if (potentialMass[i] > 0)
                {
                    c.origin = nextOrigin[i];
                    double consumed = potentialMass[i] * originScale[c.origin];
                    double flame = consumed / (dt * s.fuel_burn_kg_m2_s * area);
                    c.fuel -= consumed; c.burning = c.fuel > 1e-10;
                    if (!c.burning) { c.fuel = 0; flame = 0; }
                    double release = consumed * s.heat_of_combustion_kj_kg;
                    double absorbed = Math.Min(release, remainingWaterEnergy);
                    remainingWaterEnergy -= absorbed;
                    flameAbsorbed = waterEnergy > 0 && absorbed >= release;
                    double retained = release > 0 ? (release - absorbed) / release : 0;
                    nextFlame[i] = flame * retained;
                    double flameC = s.flame_min_c + (s.flame_max_c - s.flame_min_c) * flame;
                    temperature += (flameC - temperature) * Math.Min(1, dt / s.flame_relaxation_s) * flame * retained;
                    c.heatReleaseKw = (release - absorbed) / dt;
                    heatKw[c.room] += c.heatReleaseKw;
                    originRoomHeatKw[c.origin, c.room] += c.heatReleaseKw;
                    smokeProduction[c.room] += g.smoke_production_per_cell_s * g.smoke_reference_volume_m3 *
                        consumed * retained / (dt * s.fuel_burn_kg_m2_s * area);
                }
                if (remainingWaterEnergy > 0)
                {
                    // There is no separate solid capacity in the original temperature proxy. Reuse its
                    // effective lining/gas capacity per floor area rather than invent a material constant.
                    double capacity = g.effective_heat_capacity_kj_m2_k * area;
                    double cooling = Math.Min(remainingWaterEnergy, Math.Max(0, temperature - config.ambient_c) * capacity);
                    temperature -= cooling / capacity;
                    remainingWaterEnergy -= cooling;
                }
                TotalWaterEnergyRemovedKj += waterEnergy - remainingWaterEnergy;
                // Extinction is a state transition, not a visual fade. A subsequent ignition
                // needs dry fuel and neighbour/room heating in the first pass above.
                // Grimwood (2005), p.6; NISTIR 6191 (1998), chapter 2.
                if (flameAbsorbed && temperature < c.ignition) { c.burning = false; nextFlame[i] = 0; }
                c.burnIntensity = c.burning ? nextBurnIntensity[i] : 0;
                nextTemperature[i] = Clamp(temperature, config.ambient_c, s.max_temperature_c);
            }
            for (int i = 0; i < cells.Length; i++) { cells[i].temperature = nextTemperature[i]; cells[i].flame = nextFlame[i]; }
            StepGas(dt);
            TryCollapse((tick + 1) * dt);
        }

        void StepGas(double dt)
        {
            var g = config.gas;
            Array.Clear(gasHeatFlow, 0, rooms.Length); Array.Clear(smokeFlow, 0, rooms.Length); Array.Clear(volumeFlow, 0, rooms.Length);
            Array.Clear(originGasFlow, 0, originGasFlow.Length);
            // Symmetric exchange uses old layer states. No room can heat another through a wall.
            foreach (var d in openings)
            {
                if (d.width <= 0) continue;
                if (d.a >= 0 && d.b >= 0)
                {
                    var a = rooms[d.a]; var b = rooms[d.b];
                    double spill = Clamp((doorHead - Math.Min(a.height, b.height)) / doorHead, 0, 1);
                    double v = d.width * g.door_exchange_m3_s_m * spill;
                    double heat = v * g.gas_heat_capacity_kj_m3_k * (a.temperature - b.temperature);
                    double smoke = v * (a.smoke - b.smoke);
                    double volume = v * (b.height - a.height) / ceiling;
                    gasHeatFlow[d.a] -= heat; gasHeatFlow[d.b] += heat;
                    for (int n = 0; n < originStart.Length; n++)
                    {
                        double transfer = v * g.gas_heat_capacity_kj_m3_k * (originGasTemperature[n, d.a] - originGasTemperature[n, d.b]);
                        originGasFlow[n, d.a] -= transfer; originGasFlow[n, d.b] += transfer;
                    }
                    smokeFlow[d.a] -= smoke; smokeFlow[d.b] += smoke;
                    volumeFlow[d.a] -= volume; volumeFlow[d.b] += volume;
                }
                else
                {
                    int r = d.a >= 0 ? d.a : d.b; var room = rooms[r];
                    double spill = Clamp((doorHead - room.height) / doorHead, 0, 1);
                    double v = d.width * g.outside_exchange_m3_s_m * spill;
                    gasHeatFlow[r] -= v * g.gas_heat_capacity_kj_m3_k * (room.temperature - config.ambient_c);
                    for (int n = 0; n < originStart.Length; n++)
                        originGasFlow[n, r] -= v * g.gas_heat_capacity_kj_m3_k * originGasTemperature[n, r];
                    smokeFlow[r] -= v * room.smoke;
                    volumeFlow[r] -= v * (ceiling - room.height) / ceiling;
                }
            }
            for (int i = 0; i < rooms.Length; i++)
            {
                var room = rooms[i];
                double volume = room.area * (ceiling - room.height);
                double newVolume = Clamp(volume + dt * (g.entrainment_m3_s_kw * heatKw[i] + volumeFlow[i]),
                    room.area * g.initial_layer_depth_m, room.area * (ceiling - g.min_layer_height_m));
                room.temperature = Clamp(room.temperature + dt *
                    (g.convective_fraction * heatKw[i] + gasHeatFlow[i] - g.heat_loss_kw_m2_k * room.area *
                     (room.temperature - config.ambient_c)) / (g.effective_heat_capacity_kj_m2_k * room.area),
                    config.ambient_c, config.spread.max_temperature_c);
                double attributed = 0;
                for (int n = 0; n < originStart.Length; n++)
                {
                    originGasTemperature[n, i] = Math.Max(0, originGasTemperature[n, i] + dt *
                        (g.convective_fraction * originRoomHeatKw[n, i] + originGasFlow[n, i] -
                         g.heat_loss_kw_m2_k * room.area * originGasTemperature[n, i]) /
                        (g.effective_heat_capacity_kj_m2_k * room.area));
                    attributed += originGasTemperature[n, i];
                }
                // Apply the scalar temperature clamp to all source contributions equally.
                if (attributed > 0)
                    for (int n = 0; n < originStart.Length; n++)
                        originGasTemperature[n, i] *= (room.temperature - config.ambient_c) / attributed;
                room.smoke = Clamp((room.smoke * volume + dt * (smokeProduction[i] + smokeFlow[i] -
                    g.smoke_decay_per_s * room.smoke * volume)) / newVolume, 0, 1);
                room.height = ceiling - newVolume / room.area;
                if (room.hasFuel && room.temperature >= config.flashover_threshold_c && room.flashoverTime < 0)
                    room.flashoverTime = (tick + 1) * dt;
                // Left-endpoint accounting: do not count the interval in which a room first
                // crosses the threshold. The previous tick must also have been hot.
                if (room.temperature >= config.collapse.temperature_c)
                {
                    if (hotPreviously[i]) room.hotTicks++;
                    room.hotSeconds = room.hotTicks * dt;
                    hotPreviously[i] = true;
                }
                else { room.hotTicks = 0; room.hotSeconds = 0; hotPreviously[i] = false; }
            }
        }

        int GasOrigin(int room)
        {
            int source = -1; double hottest = 0;
            for (int n = 0; n < originStart.Length; n++)
                if (originGasTemperature[n, room] > hottest) { hottest = originGasTemperature[n, room]; source = n; }
            return source;
        }

        void TryCollapse(double time)
        {
            var rule = config.collapse;
            if (collapses.Count >= rule.max_events || time - lastCollapseTime + 1e-9 < rule.interval_s) return;
            int choice = -1; double best = double.NegativeInfinity;
            for (int i = 0; i < cells.Length; i++)
            {
                var c = cells[i];
                if (c.room < 0 || c.collapsed || c.protectedApproach) continue;
                var room = rooms[c.room];
                if (!room.canCollapse || room.hotSeconds + 1e-9 < rule.duration_s || room.temperature < rule.temperature_c) continue;
                double score = c.temperature + UnitNoise(seed, i) * .01;
                if (score > best) { choice = i; best = score; }
            }
            if (choice < 0) return;
            var selected = cells[choice]; selected.collapsed = true;
            collapses.Add(new CollapseEvent(time, CellRect(choice % Width, choice / Width), rooms[selected.room].id,
                rooms[selected.room].hotSeconds));
            lastCollapseTime = time;
        }

        public CellState GetCell(int x, int z)
        {
            if (x < 0 || z < 0 || x >= Width || z >= Height) throw new ArgumentOutOfRangeException();
            var c = cells[Index(x, z)];
            return new CellState(c.room >= 0 ? rooms[c.room].id : null, c);
        }
        public Vector2 CellCentre(int x, int z) => new Vector2((float)(originX + (x + .5) * config.cell_size_m),
            (float)(originZ + (z + .5) * config.cell_size_m));
        Rect CellRect(int x, int z) => new Rect((float)(originX + x * config.cell_size_m),
            (float)(originZ + z * config.cell_size_m), CellSize, CellSize);
        public RoomState GetRoomState(string room) => RoomSnapshot(RoomIndex(room));
        RoomState RoomSnapshot(int i)
        {
            if (i < 0) throw new ArgumentException("Outside has no room state.");
            return new RoomState(rooms[i].id, rooms[i], config.ambient_c, config.gas.lower_temperature_fraction);
        }
        public float SurfaceTemperatureC(Vector2 xz)
        { int i = PointIndex(xz); return i < 0 ? (float)config.ambient_c : (float)cells[i].temperature; }
        public float FlameIntensity(Vector2 xz)
        { int i = PointIndex(xz); return i < 0 ? 0 : (float)cells[i].flame; }
        public float BurntFraction(Vector2 xz)
        { int i = PointIndex(xz); return i < 0 || cells[i].initialFuel <= 0 ? 0 : (float)Clamp(1 - cells[i].fuel / cells[i].initialFuel, 0, 1); }

        /// <summary>
        /// Uniform upper/lower gas plus the strongest nearby same-room flame contribution.
        /// Horizontal falloff is [1/(1+(r/cellSize)^2)] * [1-r^2/radius^2], zero beyond radius.
        /// The vertical factor is 0.5+0.5*y/(y+scale); density does not include this plume term.
        /// This bounded temperature proxy is not a radiation heat-flux calculation.
        /// </summary>
        public float AirTemperatureC(Vector3 p)
        {
            int i = PointIndex(new Vector2(p.x, p.z));
            if (!Finite(p.y) || p.y < 0 || p.y > ceiling || i < 0 || cells[i].room < 0) return (float)config.ambient_c;
            int r = cells[i].room; var room = rooms[r];
            double baseC = p.y >= room.height ? room.temperature :
                config.ambient_c + config.gas.lower_temperature_fraction * (room.temperature - config.ambient_c);
            double nearby = 0, radius = config.query.flame_heating_radius_m;
            int cx = i % Width, cz = i / Width, reach = (int)Math.Ceiling(radius / config.cell_size_m);
            for (int z = Math.Max(0, cz - reach); z <= Math.Min(Height - 1, cz + reach); z++)
                for (int x = Math.Max(0, cx - reach); x <= Math.Min(Width - 1, cx + reach); x++)
                {
                    var c = cells[Index(x, z)]; if (c.room != r || c.flame <= 0) continue;
                    var centre = CellCentre(x, z); double dx = p.x - centre.x, dz = p.z - centre.y, r2 = dx * dx + dz * dz;
                    if (r2 >= radius * radius) continue;
                    double falloff = (1 - r2 / (radius * radius)) / (1 + r2 / (config.cell_size_m * config.cell_size_m));
                    double plume = .5 + .5 * p.y / (p.y + config.query.plume_height_scale_m);
                    nearby = Math.Max(nearby, config.query.flame_heating_c * c.flame * falloff * plume);
                }
            return (float)Clamp(baseC + nearby, config.ambient_c, config.spread.max_temperature_c);
        }
        public float SmokeDensity(Vector3 p)
        {
            int i = PointIndex(new Vector2(p.x, p.z));
            if (!Finite(p.y) || p.y < 0 || p.y > ceiling || i < 0 || cells[i].room < 0) return 0;
            var room = rooms[cells[i].room];
            return (float)(room.smoke * (p.y >= room.height ? 1 : config.gas.lower_smoke_fraction));
        }
        /// <summary>One region per burning cell in stable z-major order; callers may merge these for VFX.</summary>
        public IReadOnlyList<BurningRegion> ActiveFlames()
        {
            var result = new List<BurningRegion>();
            for (int i = 0; i < cells.Length; i++)
                if (cells[i].flame > 0)
                {
                    var p = CellCentre(i % Width, i / Width);
                    result.Add(new BurningRegion(new Vector3(p.x, 0, p.y), CellSize * .5f, (float)cells[i].flame));
                }
            return result.AsReadOnly();
        }

        /// <summary>Physical state hash, including double precision values and latches hidden by query floats.</summary>
        public string StateHash()
        {
            using (var stream = new MemoryStream())
            using (var writer = new BinaryWriter(stream))
            {
                writer.Write(tick); writer.Write(seed); writer.Write(Width); writer.Write(Height); writer.Write(lastCollapseTime);
                foreach (var c in cells)
                {
                    writer.Write(c.room); writer.Write(c.initialFuel); writer.Write(c.fuel); writer.Write(c.temperature);
                    writer.Write(c.flame); writer.Write(c.wetUntil); writer.Write(c.ignition); writer.Write(c.burning); writer.Write(c.collapsed); writer.Write(c.protectedApproach);
                    writer.Write(c.burnIntensity); writer.Write(c.origin);
                }
                foreach (double start in originStart) writer.Write(start);
                foreach (double temperature in originGasTemperature) writer.Write(temperature);
                for (int i = 0; i < rooms.Length; i++)
                {
                    var g = rooms[i]; writer.Write(g.id); writer.Write(g.temperature); writer.Write(g.height);
                    writer.Write(g.smoke); writer.Write(g.hotSeconds); writer.Write(g.hotTicks); writer.Write(g.flashoverTime); writer.Write(g.hasFuel);
                    writer.Write(hotPreviously[i]);
                }
                writer.Write(TotalWaterEnergyRemovedKj);
                writer.Write(sprays.Count);
                foreach (var spray in sprays)
                {
                    writer.Write(spray.start); writer.Write(spray.end); writer.Write(spray.flow);
                    writer.Write(spray.targets.Length);
                    foreach (int i in spray.targets) writer.Write(i);
                }
                foreach (bool applied in ignitionApplied) writer.Write(applied);
                foreach (var e in collapses)
                {
                    writer.Write(e.Time); writer.Write(e.Room); writer.Write(e.CellRect.x); writer.Write(e.CellRect.y);
                    writer.Write(e.HotExposureSeconds);
                }
                writer.Flush();
                using (var sha = SHA256.Create()) return BitConverter.ToString(sha.ComputeHash(stream.ToArray())).Replace("-", "").ToLowerInvariant();
            }
        }

        /// <summary>Snapshot schema v1 is documented in tools/fire_preview.py. This method does no file IO.</summary>
        public string ToJson()
        {
            var grid = new List<object>(cells.Length);
            for (int i = 0; i < cells.Length; i++)
            {
                var c = GetCell(i % Width, i / Width);
                grid.Add(new { room = c.Room, fuel_kg = c.FuelRemainingKg, initial_fuel_kg = c.InitialFuelKg,
                    temperature_c = c.SurfaceTemperatureC, flame = c.FlameIntensity, burnt = c.BurntFraction, collapsed = c.Collapsed });
            }
            var gas = new List<object>();
            foreach (var r in RoomStates) gas.Add(new { room = r.Room, upper_c = r.UpperTemperatureC, lower_c = r.LowerTemperatureC,
                layer_height_m = r.LayerHeight, smoke = r.SmokeDensity, flashover_time_s = r.FlashoverTime, hot_exposure_s = r.HotExposureSeconds });
            var events = new List<object>();
            foreach (var e in collapses) events.Add(new { time_s = e.Time, room = e.Room,
                rect = new[] { e.CellRect.xMin, e.CellRect.yMin, e.CellRect.xMax, e.CellRect.yMax }, hot_exposure_s = e.HotExposureSeconds });
            return JsonConvert.SerializeObject(new { version = 1, time_s = Time, seed, width = Width, height = Height,
                cell_size_m = CellSize, origin = new[] { originX, originZ }, wall_height_m = ceiling,
                ambient_c = config.ambient_c, max_temperature_c = config.spread.max_temperature_c,
                state_hash = StateHash(), cells = grid, rooms = gas, collapses = events, walls },
                new JsonSerializerSettings { Culture = CultureInfo.InvariantCulture, Formatting = Formatting.None });
        }
        /// <summary>Fresh default house, seed 7. An Editor menu may write this string to a chosen path.</summary>
        public static string DumpJson(double t) => DumpJson(FloorPlan.Load(), Path.Combine(FloorPlan.ConfigDir, "fire.json"), 7, t);
        public static string DumpJson(FloorPlan plan, string firePath, int seed, double t)
        { var sim = Load(plan, firePath, seed); sim.AdvanceTo(t); return sim.ToJson(); }

        void AddWallFuel(Cell c, FloorPlan.Room room, Vector2 p, bool wallX, double at, double area)
        {
            foreach (var d in openings)
            {
                if (d.a != c.room && d.b != c.room) continue;
                if (d.alongX == wallX || Math.Abs((wallX ? d.x : d.z) - at) > 1e-6) continue;
                double along = wallX ? p.y : p.x, centre = wallX ? d.z : d.x;
                if (Math.Abs(along - centre) < d.width / 2) return;
            }
            AddFuel(c, GetSurface(room.walls), area);
        }
        void AddFurniture(FloorPlan plan)
        {
            foreach (var landmark in plan.landmarks)
            {
                if (!config.furniture.TryGetValue(landmark.kind, out var definition))
                    throw new InvalidDataException("No fire furniture definition for " + landmark.kind);
                int r = RoomIndex(landmark.room); var mat = config.materials[definition.material];
                double angle = landmark.yaw * Math.PI / 180, cos = Math.Cos(angle), sin = Math.Sin(angle);
                double hx = definition.size_m[0] / 2, hz = definition.size_m[1] / 2;
                for (int z = 0; z < Height; z++) for (int x = 0; x < Width; x++)
                {
                    var c = cells[Index(x, z)]; if (c.room != r) continue;
                    var p = CellCentre(x, z); double dx = p.x - landmark.at[0], dz = p.y - landmark.at[1];
                    // Inverse of Unity's positive yaw about +y.
                    double localX = dx * cos - dz * sin, localZ = dx * sin + dz * cos;
                    if (Math.Abs(localX) <= hx + 1e-7 && Math.Abs(localZ) <= hz + 1e-7)
                        AddFuel(c, mat, config.cell_size_m * config.cell_size_m);
                }
            }
        }
        void AddFuel(Cell cell, Material material, double area)
        {
            cell.initialFuel += material.fuel_kg_m2 * area;
            if (material.fuel_kg_m2 > 0) cell.ignition = Math.Min(cell.ignition, material.ignition_c);
        }
        Material GetSurface(string name)
        {
            if (name == null || !config.surfaces.TryGetValue(name, out var material))
                throw new InvalidDataException("No fire material mapping for " + name);
            return config.materials[material];
        }
        void AddNeighbour(List<int> result, int x, int z, int nx, int nz)
        {
            if (nx < 0 || nz < 0 || nx >= Width || nz >= Height) return;
            int i = Index(x, z), j = Index(nx, nz), a = cells[i].room, b = cells[j].room;
            if (a < 0 || b < 0) return;
            if (a == b || IsOpenEdge(a, b, CellCentre(x, z), CellCentre(nx, nz))) result.Add(j);
        }
        bool IsOpenEdge(int a, int b, Vector2 p, Vector2 q)
        {
            foreach (var d in openings)
            {
                if (d.width <= 0 || !((d.a == a && d.b == b) || (d.a == b && d.b == a))) continue;
                double acrossP = d.alongX ? p.y : p.x, acrossQ = d.alongX ? q.y : q.x;
                double line = d.alongX ? d.z : d.x;
                if (Math.Abs(acrossP - acrossQ) < 1e-8 || line < Math.Min(acrossP, acrossQ) || line > Math.Max(acrossP, acrossQ)) continue;
                double along = d.alongX ? p.x : p.y, centre = d.alongX ? d.x : d.z;
                if (Math.Abs(along - centre) < d.width / 2 - 1e-7) return true;
            }
            return false;
        }
        bool IsDoorApproach(int room, Rect cell)
        {
            foreach (var d in openings)
            {
                if (d.a != room && d.b != room) continue;
                double lo = d.alongX ? cell.xMin : cell.yMin, hi = d.alongX ? cell.xMax : cell.yMax;
                double centre = d.alongX ? d.x : d.z, half = config.collapse.door_clear_width_m / 2;
                if (lo < centre + half && hi > centre - half) return true;
            }
            return false;
        }
        void BuildPreviewWalls(FloorPlan plan)
        {
            // A set removes duplicate shared-wall segments. Integer grid coordinates avoid float keys.
            var seen = new HashSet<string>(StringComparer.Ordinal);
            foreach (var r in plan.rooms)
            {
                int x0 = GridOffset(r.rect[0], originX), x1 = GridOffset(r.rect[2], originX);
                int z0 = GridOffset(r.rect[1], originZ), z1 = GridOffset(r.rect[3], originZ);
                for (int x = x0; x < x1; x++) { PreviewEdge(seen, x, z0, true); PreviewEdge(seen, x, z1, true); }
                for (int z = z0; z < z1; z++) { PreviewEdge(seen, x0, z, false); PreviewEdge(seen, x1, z, false); }
            }
        }
        void PreviewEdge(HashSet<string> seen, int x, int z, bool alongX)
        {
            string key = x + "/" + z + "/" + alongX;
            if (!seen.Add(key)) return;
            double px = originX + x * config.cell_size_m, pz = originZ + z * config.cell_size_m;
            double mid = (alongX ? px : pz) + config.cell_size_m / 2;
            foreach (var d in openings)
                if (d.alongX == alongX && Math.Abs((alongX ? d.z - pz : d.x - px)) < 1e-6 &&
                    Math.Abs(mid - (alongX ? d.x : d.z)) < d.width / 2) return;
            walls.Add(new[] { px, pz, px + (alongX ? config.cell_size_m : 0), pz + (alongX ? 0 : config.cell_size_m) });
        }
        int RoomIndex(string id)
        {
            if (id == "outside") return -1;
            if (id == null || !roomIndices.TryGetValue(id, out int i)) throw new InvalidDataException("Unknown room " + id);
            return i;
        }
        int Index(int x, int z) => z * Width + x;
        int PointIndex(Vector2 p)
        {
            if (!Finite(p.x) || !Finite(p.y)) return -1;
            double gx = (p.x - originX) / config.cell_size_m, gz = (p.y - originZ) / config.cell_size_m;
            if (gx < 0 || gz < 0 || gx >= Width || gz >= Height) return -1;
            return Index((int)Math.Floor(gx), (int)Math.Floor(gz));
        }
        int GridCount(double length)
        {
            double n = length / config.cell_size_m;
            if (!Finite(n) || n <= 0 || n > 100000 || Math.Abs(n - Math.Round(n)) > 1e-4)
                throw new InvalidDataException("Footprint and room edges must align to the fire grid.");
            return checked((int)Math.Round(n));
        }
        int GridOffset(double value, double origin)
        {
            double n = (value - origin) / config.cell_size_m;
            if (Math.Abs(n - Math.Round(n)) > 1e-4) throw new InvalidDataException("Room edge does not align to grid.");
            return checked((int)Math.Round(n));
        }
        static double UnitNoise(int seed, int index)
        {
            unchecked
            {
                uint n = (uint)seed ^ ((uint)index + 0x9e3779b9u);
                n ^= n >> 16; n *= 0x7feb352du; n ^= n >> 15; n *= 0x846ca68bu; n ^= n >> 16;
                return n / (double)uint.MaxValue;
            }
        }
        static double Clamp(double v, double min, double max) => Math.Max(min, Math.Min(max, v));
        static bool Finite(double x) => !double.IsNaN(x) && !double.IsInfinity(x);
        static void Require(bool condition, string message)
        { if (!condition) throw new InvalidDataException("fire.json: " + message); }
        static void Positive(double value, string name) => Require(Finite(value) && value > 0, name + " must be finite and positive");
        static void Nonnegative(double value, string name) => Require(Finite(value) && value >= 0, name + " must be finite and nonnegative");
        static void Fraction(double value, string name) => Require(Finite(value) && value >= 0 && value <= 1, name + " must be in [0,1]");
        static void ValidateConfig(Configuration c)
        {
            Require(c != null && c.version == 1, "unsupported or missing version");
            Positive(c.cell_size_m, "cell_size_m"); Positive(c.step_s, "step_s");
            Require(c.step_s <= .1 && c.cell_size_m >= .05, "step must be <= 0.1 s and cell size >= 0.05 m");
            Require(Finite(c.ambient_c) && c.ambient_c >= -40 && c.ambient_c <= 60, "invalid ambient_c");
            Require(c.materials != null && c.surfaces != null && c.furniture != null && c.ignitions != null && c.ignitions.Count > 0,
                "materials, surfaces, furniture and ignitions are required");
            Require(c.spread != null && c.gas != null && c.collapse != null && c.query != null, "missing model parameters");
            Require(c.water != null, "missing water parameters");
            Positive(c.water.latent_heat_kj_l, "water.latent_heat_kj_l");
            Nonnegative(c.water.sensible_heat_kj_l, "water.sensible_heat_kj_l");
            Positive(c.water.wet_seconds_per_litre, "water.wet_seconds_per_litre");
            Fraction(c.water.application_efficiency, "water.application_efficiency");
            Require(Finite(c.water.latent_heat_kj_l + c.water.sensible_heat_kj_l), "water energy overflow");
            var s = c.spread; var g = c.gas; var r = c.collapse; var q = c.query;
            Nonnegative(s.growth_alpha_kw_s2, "spread.growth_alpha_kw_s2");
            Nonnegative(s.neighbour_heat_per_s, "neighbour_heat_per_s"); Nonnegative(s.surface_cooling_per_s, "surface_cooling_per_s");
            Nonnegative(s.gas_surface_heat_per_s, "gas_surface_heat_per_s"); Nonnegative(s.flashover_surface_heat_c_s, "flashover_surface_heat_c_s");
            Positive(s.fuel_burn_kg_m2_s, "fuel_burn_kg_m2_s"); Positive(s.flame_ramp_s, "flame_ramp_s"); Positive(s.flame_relaxation_s, "flame_relaxation_s");
            Require(Finite(s.flame_min_c) && s.flame_min_c >= 600 && s.flame_max_c >= s.flame_min_c && s.max_temperature_c >= s.flame_max_c && s.max_temperature_c <= 1100, "invalid flame temperature limits");
            Require(Finite(s.fuel_variation_fraction) && s.fuel_variation_fraction >= 0 && s.fuel_variation_fraction < 1, "invalid fuel variation");
            Positive(s.heat_of_combustion_kj_kg, "heat_of_combustion_kj_kg");
            Require(Finite(c.flashover_threshold_c) && c.flashover_threshold_c > c.ambient_c && c.flashover_threshold_c <= s.max_temperature_c, "invalid flashover threshold");
            Positive(g.effective_heat_capacity_kj_m2_k, "effective_heat_capacity_kj_m2_k"); Nonnegative(g.heat_loss_kw_m2_k, "heat_loss_kw_m2_k");
            Fraction(g.convective_fraction, "convective_fraction"); Positive(g.gas_heat_capacity_kj_m3_k, "gas_heat_capacity_kj_m3_k");
            Nonnegative(g.door_exchange_m3_s_m, "door_exchange_m3_s_m"); Nonnegative(g.outside_exchange_m3_s_m, "outside_exchange_m3_s_m");
            Nonnegative(g.entrainment_m3_s_kw, "entrainment_m3_s_kw"); Positive(g.min_layer_height_m, "min_layer_height_m"); Positive(g.initial_layer_depth_m, "initial_layer_depth_m");
            Fraction(g.lower_temperature_fraction, "lower_temperature_fraction"); Fraction(g.lower_smoke_fraction, "lower_smoke_fraction");
            Nonnegative(g.smoke_production_per_cell_s, "smoke_production_per_cell_s"); Positive(g.smoke_reference_volume_m3, "smoke_reference_volume_m3"); Nonnegative(g.smoke_decay_per_s, "smoke_decay_per_s");
            Require(Finite(r.temperature_c) && r.temperature_c > c.ambient_c && r.temperature_c <= s.max_temperature_c, "invalid collapse temperature");
            Positive(r.duration_s, "collapse.duration_s"); Nonnegative(r.interval_s, "collapse.interval_s"); Positive(r.door_clear_width_m, "collapse.door_clear_width_m");
            Require(r.max_events >= 0 && r.max_events <= 1000 && r.eligible_ceilings != null, "invalid collapse limits");
            Positive(q.flame_heating_radius_m, "flame_heating_radius_m"); Nonnegative(q.flame_heating_c, "flame_heating_c"); Positive(q.plume_height_scale_m, "plume_height_scale_m");
            Require(c.step_s * (4 * s.neighbour_heat_per_s + s.surface_cooling_per_s + s.gas_surface_heat_per_s) <= 1, "unstable surface exchange step");
            foreach (var m in c.materials.Values)
            {
                Require(m != null, "null material"); Nonnegative(m.fuel_kg_m2, "material fuel");
                Require(Finite(m.ignition_c) && m.ignition_c > c.ambient_c && m.ignition_c <= s.max_temperature_c, "invalid material ignition temperature");
            }
            foreach (string m in c.surfaces.Values) Require(m != null && c.materials.ContainsKey(m), "unknown surface material");
            foreach (var f in c.furniture.Values)
            {
                Require(f != null && f.material != null && c.materials.ContainsKey(f.material) && f.size_m != null && f.size_m.Length == 2, "invalid furniture");
                Positive(f.size_m[0], "furniture width"); Positive(f.size_m[1], "furniture depth");
            }
            foreach (var ignition in c.ignitions)
            {
                Require(ignition != null && ignition.room != null && ignition.at != null && ignition.at.Length == 2 && Finite(ignition.at[0]) && Finite(ignition.at[1]), "invalid ignition position");
                Nonnegative(ignition.time_s, "ignition time"); Nonnegative(ignition.radius_m, "ignition radius");
            }
        }
        void ValidatePlan(FloorPlan plan)
        {
            Require(plan.footprint != null && plan.footprint.Length == 4 && plan.rooms != null && plan.rooms.Count > 0 && plan.doors != null && plan.landmarks != null, "incomplete floor plan");
            foreach (float v in plan.footprint) Require(Finite(v), "nonfinite footprint");
            Require(Finite(plan.door_height) && plan.door_height > 0 && plan.door_height <= plan.wall_height, "invalid door height");
            Require(Finite(plan.wall_height) && plan.wall_height > config.gas.min_layer_height_m + config.gas.initial_layer_depth_m, "wall height is too low for layers");
            int w = GridCount(plan.footprint[2] - plan.footprint[0]), h = GridCount(plan.footprint[3] - plan.footprint[1]);
            Require((long)w * h <= 1000000, "grid exceeds one million cells");
            var ids = new HashSet<string>(StringComparer.Ordinal);
            for (int i = 0; i < plan.rooms.Count; i++)
            {
                var r = plan.rooms[i]; Require(r != null && !string.IsNullOrEmpty(r.id) && r.id != "outside" && ids.Add(r.id) && r.rect != null && r.rect.Length == 4, "invalid or duplicate room");
                foreach (float v in r.rect) Require(Finite(v), "nonfinite room rectangle");
                Require(r.rect[2] > r.rect[0] && r.rect[3] > r.rect[1] && r.rect[0] >= plan.footprint[0] && r.rect[1] >= plan.footprint[1] && r.rect[2] <= plan.footprint[2] && r.rect[3] <= plan.footprint[3], "room outside footprint or empty");
                GridOffset(r.rect[0], plan.footprint[0]); GridOffset(r.rect[2], plan.footprint[0]); GridOffset(r.rect[1], plan.footprint[1]); GridOffset(r.rect[3], plan.footprint[1]);
                for (int j = 0; j < i; j++) Require(!r.Rect.Overlaps(plan.rooms[j].Rect), "overlapping rooms");
            }
            foreach (var d in plan.doors)
            {
                Require(d != null && d.between != null && d.between.Length == 2 && d.between[0] != d.between[1] &&
                    (ids.Contains(d.between[0]) || d.between[0] == "outside") && (ids.Contains(d.between[1]) || d.between[1] == "outside") &&
                    d.at != null && d.at.Length == 2 && Finite(d.at[0]) && Finite(d.at[1]) && Finite(d.width) && d.width > 0 &&
                    (d.wall == "x" || d.wall == "z") && (d.state == "open" || d.state == "half" || d.state == "removed" || d.state == "closed"), "invalid door");
                // A named connection must cut both named boundaries. A misplaced door must
                // not exchange room gas while its grid edge still behaves as a solid wall.
                foreach (string side in d.between)
                {
                    float[] rect = side == "outside" ? plan.footprint : plan.FindRoom(side).rect;
                    double line = d.RunsAlongX ? d.at[1] : d.at[0];
                    double loAcross = d.RunsAlongX ? rect[1] : rect[0], hiAcross = d.RunsAlongX ? rect[3] : rect[2];
                    double along = d.RunsAlongX ? d.at[0] : d.at[1];
                    double lo = d.RunsAlongX ? rect[0] : rect[1], hi = d.RunsAlongX ? rect[2] : rect[3];
                    Require((Math.Abs(line - loAcross) < 1e-4 || Math.Abs(line - hiAcross) < 1e-4) &&
                        along - d.width / 2 >= lo - 1e-4 && along + d.width / 2 <= hi + 1e-4,
                        "door does not cut its named room boundary");
                }
            }
            foreach (var l in plan.landmarks)
                Require(l != null && l.room != null && ids.Contains(l.room) && l.kind != null && l.at != null && l.at.Length == 2 && Finite(l.at[0]) && Finite(l.at[1]) && Finite(l.yaw), "invalid landmark");
        }
    }
}
