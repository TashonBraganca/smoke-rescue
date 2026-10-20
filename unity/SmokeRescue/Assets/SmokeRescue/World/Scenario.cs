using System;
using System.Collections.Generic;
using System.IO;
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
        public int seed;
        public float duration_s;
        public float sim_rate_hz = 15f;
        public Smoke smoke = new Smoke();
        public List<Robot> robots = new List<Robot>();
        public LookAround look_around = new LookAround();
        public List<Survivor> survivors = new List<Survivor>();
        public Radio radio = new Radio();
        public FireSpread fire = new FireSpread();

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
            return scenario;
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
