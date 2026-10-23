using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace SmokeRescue.World
{
    /// <summary>config/scenario_*.json. Handles "inherits": the child's keys replace the parent's, and
    /// "extra_robots" are appended to the parent's robots.</summary>
    [Serializable]
    public class Scenario
    {
        public string id;
        public string description;
        /// <summary>World-side hinge overrides applied by DoorRuntime after the house is built.</summary>
        [UnityEngine.SerializeField] public Dictionary<string, string> door_states = new Dictionary<string, string>();   // read from JSON; the attribute only silences Unity's serializer check
        public int seed;
        public float duration_s;
        // false: no mission deadline for the robots; the recording still ends at duration_s (owner toggle, 22 Oct).
        public bool time_limit = true;
        public float sim_rate_hz = 15f;
        public float[] search_area;
        public List<string> dark_rooms = new List<string>();   // rooms whose emergency and room lights are out   // operator knowledge: building outline x_min, z_min, x_max, z_max, seen from outside
        public Smoke smoke = new Smoke();
        public List<Robot> robots = new List<Robot>();
        public LookAround look_around = new LookAround();
        public List<Survivor> survivors = new List<Survivor>();
        public Radio radio = new Radio();
        public FireSpread fire = new FireSpread();
        /// <summary>Obstacles for this run only (burning wood, debris, rubble), placed like landmarks.</summary>
        public List<FloorPlan.Landmark> obstacles = new List<FloorPlan.Landmark>();
        /// <summary>"fixed" (positions as listed) or "random": each survivor keeps its id and pose but is put at a
        /// seeded random spot on free floor in any room, so robots cannot be tuned to known positions.</summary>
        public string survivor_placement = "fixed";

        /// <summary>Whole-house spreading fire (FireSim, config/fire.json). Off: the static fires in floorplan.json.</summary>
        [Serializable]
        public class FireSpread
        {
            public bool spread;
            public float head_start_s = 60f;   // fire time when the robots enter
            // World setup only. Both heat and fluid smoke must run this same fire configuration.
            public string config_file = "fire.json";
            public string ConfigPath
            {
                get
                {
                    if (string.IsNullOrEmpty(config_file) || Path.GetFileName(config_file) != config_file ||
                        config_file.Contains("\\") || !config_file.StartsWith("fire") || !config_file.EndsWith(".json"))
                        throw new InvalidDataException("fire.config_file must be a fire*.json basename in config");
                    return Path.Combine(FloorPlan.ConfigDir, config_file);
                }
            }
        }

        [Serializable]
        public class Smoke
        {
            public string model = "rooms";
            public float global_haze;
            [SerializeField] public Dictionary<string, float> rooms = new Dictionary<string, float>();
            public float ramp_per_minute;

            /// <summary>Smoke density for a room id at sim time t, clamped to [0, 1].</summary>
            public float Density(string roomId, float t)
            {
                float d = rooms != null && roomId != null && rooms.TryGetValue(roomId, out var v) ? v
                    : rooms != null && rooms.TryGetValue("default", out var dv) ? dv : global_haze;
                return Mathf.Clamp01(d + ramp_per_minute * t / 60f);
            }
        }

        [Serializable]
        public class Robot
        {
            public string id;
            public string color;
            public Spawn spawn;
            // Gate frame pose the team set this robot on: x right and y ahead of the launch line (m), heading (deg). The
            // gate frame is defined by the pads, so the brains share it without knowing the house's coordinates. null
            // when robots launch separately (each brain then starts in its own frame).
            public float[] launch_pad;
            public string control;
            public float cruise_speed = 0.5f;
            public string torch = "wide";                      // starting torch mode: wide, narrow, low, off
            public float doorway_speed = 0.3f;
            public List<float[]> route = new List<float[]>();
            public List<float[]> look_around_at = new List<float[]>();

            public Color Colour => ColorUtility.TryParseHtmlString(color, out var c) ? c : Color.white;
        }

        [Serializable]
        public class Spawn
        {
            public float[] at;
            public float yaw;
        }

        [Serializable]
        public class LookAround
        {
            public float sweep_deg = 60f;
            public float speed_dps = 25f;
        }

        [Serializable]
        public class Survivor
        {
            public string id;
            public string room;
            public float[] at;
            public float yaw;
            public string pose;
        }

        [Serializable]
        public class Radio
        {
            public float range_m = 12f;
            public int max_walls = 1;
            public float broadcast_period_s = 2f;
            public int budget_bytes_per_s = 2048;
        }

        public static Scenario Load(string fileName)
        {
            string path = Path.IsPathRooted(fileName) ? fileName : Path.Combine(FloorPlan.ConfigDir, fileName);
            var merged = LoadMerged(path);
            var scenario = merged.ToObject<Scenario>();
            if (scenario.robots.Count == 0) throw new InvalidDataException($"scenario has no robots: {path}");
            if (scenario.survivor_placement == "random")
            {
                // Placed once per scenario file and seed in an editor session, then reused: the editor loads a scenario to
                // build the scene and again inside the recorder, and with PlacementBlocked the second load saw the people
                // the first had placed and chose other spots, so oracle/survivors.json disagreed with the scene (held-out
                // seed 914 runs 3 and 4).
                string key = Path.GetFullPath(path) + "|" + scenario.seed;
                if (RandomPlacements.TryGetValue(key, out var placed) && placed.Count == scenario.survivors.Count)
                    for (int i = 0; i < placed.Count; i++)
                    { scenario.survivors[i].room = placed[i].room; scenario.survivors[i].at = (float[])placed[i].at.Clone(); scenario.survivors[i].yaw = placed[i].yaw; }
                else
                {
                    scenario.PlaceSurvivorsRandomly(FloorPlan.Load());
                    RandomPlacements[key] = scenario.survivors.Select(s => (s.room, (float[])s.at.Clone(), s.yaw)).ToList();
                }
            }
            return scenario;
        }
        static readonly Dictionary<string, List<(string room, float[] at, float yaw)>> RandomPlacements =
            new Dictionary<string, List<(string room, float[] at, float yaw)>>();
        /// <summary>Forget random placements (tests; a changed house).</summary>
        public static void ForgetRandomPlacements() => RandomPlacements.Clear();

        /// <summary>Seeded random survivor spots: any room, at least 0.6 m from its walls, 0.9 m from landmarks and
        /// obstacles, 1.2 m from doors and spawn points, and 3 m from each other.</summary>
        bool InSearchArea(Rect room) => search_area == null || search_area.Length < 4 ||
            (room.xMin >= search_area[0] - 1e-3f && room.yMin >= search_area[1] - 1e-3f &&
             room.xMax <= search_area[2] + 1e-3f && room.yMax <= search_area[3] + 1e-3f);
        /// <summary>Set by the editor before a recording: true where a body lying at this floor point would overlap scene
        /// furniture. The floor plan only lists landmarks; filler props (an ottoman, armchairs) are placed by the house
        /// builder, and held-out seed 914 put s6 inside the foyer ottoman with only head and legs outside it.</summary>
        public static System.Func<Vector2, bool> PlacementBlocked;
        /// <summary>SplitMix64 of the seed: System.Random seeded with nearby integers (seed * 7919 + 101) gave correlated
        /// first draws, and the first person landed in the laundry in all nine one-door runs 922 to 932.</summary>
        static int MixSeed(int s)
        {
            ulong z = unchecked((ulong)s + 0x9E3779B97F4A7C15UL);
            z = unchecked((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL); z = unchecked((z ^ (z >> 27)) * 0x94D049BB133111EBUL);
            return (int)((z ^ (z >> 31)) & 0x7FFFFFFF);
        }
        void PlaceSurvivorsRandomly(FloorPlan plan)
        {
            var rng = new System.Random(MixSeed(seed));
            var taken = new List<Vector2>();
            var avoid = new List<(Vector2 p, float r)>();
            foreach (var l in plan.landmarks) avoid.Add((new Vector2(l.at[0], l.at[1]), 0.9f));
            foreach (var o in obstacles) avoid.Add((new Vector2(o.at[0], o.at[1]), 0.9f));
            foreach (var d in plan.doors) avoid.Add((new Vector2(d.at[0], d.at[1]), 1.2f));
            foreach (var r in robots) avoid.Add((new Vector2(r.spawn.at[0], r.spawn.at[1]), 1.2f));
            // A scripted robot follows its route blind (sensor tests, not search runs): nobody lies within 1.2 m of it.
            var routeSegments = robots.Where(r => r.route != null && r.route.Count > 0)
                .SelectMany(r => new[] { r.spawn.at }.Concat(r.route).Zip(r.route, (p0, p1) => (new Vector2(p0[0], p0[1]), new Vector2(p1[0], p1[1])))).ToList();
            bool OnRoute(Vector2 q) => routeSegments.Any(seg =>
            {
                Vector2 d = seg.Item2 - seg.Item1; float u = d.sqrMagnitude > 0 ? Mathf.Clamp01(Vector2.Dot(q - seg.Item1, d) / d.sqrMagnitude) : 0;
                return Vector2.Distance(q, seg.Item1 + u * d) < 1.2f;
            });
            // Only rooms inside the scenario's search area (held-out run 1: four of five people landed in the apartment,
            // office and workshop of the larger building, beyond the house the robots search). Every room, the hallway
            // included, with a chance in proportion to its floor area: a point is drawn over the whole area and kept only
            // where it lies 0.6 m inside a room. Drawing the room first and the point second gave a 6.6 m2 bathroom the same
            // pull as a 36.6 m2 living room, and the margins then rejected most bathroom points (no one in the bathroom or
            // closet in runs 922 to 932, someone in the laundry in all of them).
            var rooms = plan.rooms.Where(r => InSearchArea(r.Rect) && r.Rect.width >= 1.4f && r.Rect.height >= 1.4f).ToList();
            float xMin = rooms.Min(r => r.Rect.xMin), xMax = rooms.Max(r => r.Rect.xMax), yMin = rooms.Min(r => r.Rect.yMin), yMax = rooms.Max(r => r.Rect.yMax);
            foreach (var s in survivors)
            {
                for (int attempt = 0; attempt < 20000; attempt++)
                {
                    var p = new Vector2(xMin + (float)rng.NextDouble() * (xMax - xMin), yMin + (float)rng.NextDouble() * (yMax - yMin));
                    var room = rooms.FirstOrDefault(r => p.x >= r.Rect.xMin + 0.6f && p.x <= r.Rect.xMax - 0.6f && p.y >= r.Rect.yMin + 0.6f && p.y <= r.Rect.yMax - 0.6f);
                    if (room == null) continue;
                    if (avoid.Any(a => Vector2.Distance(a.p, p) < a.r) || taken.Any(q => Vector2.Distance(q, p) < 3f) || OnRoute(p)) continue;
                    if (PlacementBlocked != null && PlacementBlocked(p)) continue;
                    s.at = new[] { p.x, p.y };
                    s.room = room.id;
                    s.yaw = (float)(rng.NextDouble() * 360.0);
                    taken.Add(p);
                    break;
                }
            }
        }

        static JObject LoadMerged(string path)
        {
            var obj = JObject.Parse(File.ReadAllText(path));
            var parentName = obj.Value<string>("inherits");
            if (string.IsNullOrEmpty(parentName)) return obj;
            var parent = LoadMerged(Path.Combine(Path.GetDirectoryName(path), parentName));
            foreach (var prop in obj.Properties())
            {
                if (prop.Name == "inherits" || prop.Name == "extra_robots") continue;
                parent[prop.Name] = prop.Value.DeepClone();
            }
            if (obj["extra_robots"] is JArray extra)
            {
                var robots = (JArray)parent["robots"];
                foreach (var r in extra) robots.Add(r.DeepClone());
            }
            parent.Remove("inherits");
            return parent;
        }
    }
}
