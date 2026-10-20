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
        public Dictionary<string, string> door_states = new Dictionary<string, string>();
        public int seed;
        public float duration_s;
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
        }

        [Serializable]
        public class Smoke
        {
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
            if (scenario.survivor_placement == "random") scenario.PlaceSurvivorsRandomly(FloorPlan.Load());
            return scenario;
        }

        /// <summary>Seeded random survivor spots: any room, at least 0.6 m from its walls, 0.9 m from landmarks and
        /// obstacles, 1.2 m from doors and spawn points, and 3 m from each other.</summary>
        void PlaceSurvivorsRandomly(FloorPlan plan)
        {
            var rng = new System.Random(seed * 7919 + 101);
            var taken = new List<Vector2>();
            var avoid = new List<(Vector2 p, float r)>();
            foreach (var l in plan.landmarks) avoid.Add((new Vector2(l.at[0], l.at[1]), 0.9f));
            foreach (var o in obstacles) avoid.Add((new Vector2(o.at[0], o.at[1]), 0.9f));
            foreach (var d in plan.doors) avoid.Add((new Vector2(d.at[0], d.at[1]), 1.2f));
            foreach (var r in robots) avoid.Add((new Vector2(r.spawn.at[0], r.spawn.at[1]), 1.2f));
            var rooms = plan.rooms.Where(r => r.id != "hallway").ToList();
            foreach (var s in survivors)
            {
                for (int attempt = 0; attempt < 2000; attempt++)
                {
                    var room = rooms[rng.Next(rooms.Count)];
                    var rect = room.Rect;
                    if (rect.width < 1.4f || rect.height < 1.4f) continue;
                    var p = new Vector2(rect.xMin + 0.6f + (float)rng.NextDouble() * (rect.width - 1.2f),
                                        rect.yMin + 0.6f + (float)rng.NextDouble() * (rect.height - 1.2f));
                    if (avoid.Any(a => Vector2.Distance(a.p, p) < a.r) || taken.Any(q => Vector2.Distance(q, p) < 3f)) continue;
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
