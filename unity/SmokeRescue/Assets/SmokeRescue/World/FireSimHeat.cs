using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace SmokeRescue.World
{
    /// <summary>
    /// The spreading fire (FireSim) as a heat source for thermal cameras, air sensors, flames, smoke and burn damage.
    /// FireSim reports one flame per 0.25 m cell; for visuals the burning cells are grouped into 1.25 m blocks, and each
    /// block keeps the slot it got when it first ignited, so a flame never jumps to another place as the fire changes.
    /// The fire runs head_start_s ahead of the robots' clock: they enter a house that is already burning.
    /// </summary>
    public sealed class FireSimHeat : ThermalField.ISource
    {
        public const float BlockM = 1.25f;
        public FireSim Sim { get; private set; }
        public float HeadStart { get; }
        public float AmbientC => 20f;
        readonly FloorPlan plan;
        readonly Dictionary<Vector2Int, int> slotOf = new Dictionary<Vector2Int, int>();
        readonly List<Vector2Int> slots = new List<Vector2Int>();
        readonly List<(Vector3, float, float, float)> flames = new List<(Vector3, float, float, float)>();
        float flamesT = float.NaN;
        const float GasAbsorptionPerDensity = 0.8f;   // LWIR absorption per metre at smoke density 1 (estimate)

        readonly struct Spray
        {
            public readonly double time, flow, duration;
            public readonly Vector2 nozzle;
            public readonly float heading, range, halfAngle;
            public Spray(double time, Vector2 nozzle, float heading, float range, float halfAngle, double flow, double duration)
            { this.time = time; this.nozzle = nozzle; this.heading = heading; this.range = range;
              this.halfAngle = halfAngle; this.flow = flow; this.duration = duration; }
            public void Schedule(FireSim sim) => sim.ScheduleWater(time, nozzle, heading, range, halfAngle, flow, duration);
        }
        readonly List<Spray> sprays = new List<Spray>();
        // The smoke solver owns a separate deterministic fire and replays these same water commands.
        internal SmokeSolver FluidSmoke { get; set; }

        /// <summary>Uses absolute fire time, in seconds, as FireSim.ScheduleWater does. Convert robot time
        /// with FireTime first. The wrapper owns this log so previews can replay water after a rewind.</summary>
        public void ScheduleWater(double t, Vector2 nozzleXZ, float headingRad, float rangeM,
            float halfAngleDeg, double flowLps, double durationS)
        {
            lock (advanceGate)
            {
                var spray = new Spray(t, nozzleXZ, headingRad, rangeM, halfAngleDeg, flowLps, durationS);
                spray.Schedule(Sim); // Validate before retaining a command for future replays.
                FluidSmoke?.ScheduleWater(t, nozzleXZ, headingRad, rangeM, halfAngleDeg, flowLps, durationS);
                sprays.Add(spray);
                flamesT = float.NaN;
            }
        }

        readonly int seed;
        readonly string firePath;
        double lastFireTime = -1;

        public FireSimHeat(FloorPlan plan, int seed, float headStart, string firePath = null)
        {
            this.plan = plan;
            this.seed = seed;
            HeadStart = headStart;
            this.firePath = firePath ?? Path.Combine(FloorPlan.ConfigDir, "fire.json");
            Sim = FireSim.Load(plan, this.firePath, seed);
        }

        public double FireTime(float t) => Mathf.Max(0f, t + HeadStart);

        /// <summary>FireSim only moves forward. Going back in time (previews, re-renders) restarts it from the seed and
        /// replays to t, which gives the identical state because the simulation is deterministic.</summary>
        readonly object advanceGate = new object();
        /// <summary>Times the simulation was reloaded and replayed from t = 0 to serve an earlier time; each costs a
        /// replay of the whole fire so far. Read by the recorder's perf log; it should stay 0 during a recording.</summary>
        public static int Rewinds;
        void Advance(float t)
        {
            // Depth noise samples smoke from parallel image rows: the first caller at a new time advances the fire
            // under the lock, and the rest see the new time only once that is finished.
            double ft = FireTime(t);
            if (ft == System.Threading.Volatile.Read(ref lastFireTime)) return;
            lock (advanceGate)
            {
                if (ft == lastFireTime) return;
                if (ft < lastFireTime)    // FireSim rejects any request earlier than the previous one
                {
                    Rewinds++;
                    Sim = FireSim.Load(plan, this.firePath, seed);
                    foreach (var spray in sprays) spray.Schedule(Sim);
                    flamesT = float.NaN;
                }
                Sim.AdvanceTo(ft);
                System.Threading.Volatile.Write(ref lastFireTime, ft);
            }
        }

        public Vector4 Sample(Vector2 xz, float t)
        {
            Advance(t);
            float surf = Sim.SurfaceTemperatureC(xz);
            string room = RoomAt(xz);
            if (room == null) return new Vector4(surf, AmbientC, plan.wall_height, 0f);
            var r = Sim.GetRoomState(room);
            return new Vector4(surf, r.UpperTemperatureC, r.LayerHeight, GasAbsorptionPerDensity * r.SmokeDensity);
        }

        public string RoomAt(Vector2 xz)
        {
            foreach (var r in plan.rooms) if (r.Rect.Contains(xz)) return r.id;
            return null;
        }

        /// <summary>One entry per block slot (inactive blocks have radius 0): centre on the floor, radius, flame height, C.</summary>
        public IReadOnlyList<(Vector3 centre, float radius, float height, float tempC)> Flames(float t)
        {
            if (t == flamesT) return flames;
            Advance(t);
            flamesT = t;
            var sum = new Dictionary<Vector2Int, (Vector2 xz, float flame, float temp, int n)>();
            foreach (var f in Sim.ActiveFlames())
            {
                var key = new Vector2Int(Mathf.FloorToInt(f.Centre.x / BlockM), Mathf.FloorToInt(f.Centre.z / BlockM));
                sum.TryGetValue(key, out var acc);
                var xz = new Vector2(f.Centre.x, f.Centre.z);
                acc.xz += xz * f.Intensity; acc.flame += f.Intensity; acc.n++;
                acc.temp = Mathf.Max(acc.temp, Sim.SurfaceTemperatureC(xz));
                sum[key] = acc;
            }
            foreach (var key in sum.Keys)
                if (!slotOf.ContainsKey(key)) { slotOf[key] = slots.Count; slots.Add(key); }
            flames.Clear();
            float cellArea = Sim.CellSize * Sim.CellSize;
            foreach (var key in slots)
            {
                if (!sum.TryGetValue(key, out var acc) || acc.flame <= 0f) { flames.Add((Vector3.zero, 0f, 0f, AmbientC)); continue; }
                var c = acc.xz / acc.flame;
                float burningArea = acc.n * cellArea;
                float meanFlame = acc.flame / acc.n;
                float radius = Mathf.Clamp(Mathf.Sqrt(burningArea / Mathf.PI), 0.15f, BlockM * 0.7f);
                float height = Mathf.Lerp(0.4f, 2.2f, meanFlame) * Mathf.Clamp01(0.4f + burningArea / (BlockM * BlockM));
                // Keep the flame body inside its room: billboards near a wall otherwise poke through it.
                foreach (var room in plan.rooms)
                {
                    var rr = room.Rect;
                    if (!rr.Contains(c)) continue;
                    float m = Mathf.Min(radius + 0.15f, Mathf.Min(rr.width, rr.height) / 2f - 0.05f);
                    c = new Vector2(Mathf.Clamp(c.x, rr.xMin + m, rr.xMax - m), Mathf.Clamp(c.y, rr.yMin + m, rr.yMax - m));
                    break;
                }
                flames.Add((new Vector3(c.x, 0f, c.y), radius, height, Mathf.Max(acc.temp, 600f)));
            }
            return flames;
        }

        public float BurntFraction(Vector2 xz, float t) { Advance(t); return Sim.BurntFraction(xz); }
        public float SmokeDensity(Vector3 p, float t) { Advance(t); return Sim.SmokeDensity(p); }
        public FireSim.RoomState Room(string id, float t) { Advance(t); return Sim.GetRoomState(id); }
    }
}
