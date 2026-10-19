using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using UnityEngine;

namespace SmokeRescue.World
{
    /// <summary>config/floorplan.json, the single source of truth for the house. Field names match the JSON.</summary>
    [Serializable]
    public class FloorPlan
    {
        public int version;
        public string units;
        public float[] footprint;                  // x_min, z_min, x_max, z_max
        public float wall_height;
        public float exterior_wall_thickness;
        public float interior_wall_thickness;
        public float door_height;
        public List<Room> rooms = new List<Room>();
        public List<Door> doors = new List<Door>();
        public List<Landmark> landmarks = new List<Landmark>();
        public List<Fire> fires = new List<Fire>();
        public List<EmergencyLight> emergency_lights = new List<EmergencyLight>();

        [Serializable]
        public class Room
        {
            public string id;
            public string name;
            public float[] rect;                   // x_min, z_min, x_max, z_max
            public string floor;
            public string walls;
            public string ceiling;
            public Rect Rect => Rect.MinMaxRect(rect[0], rect[1], rect[2], rect[3]);
        }

        [Serializable]
        public class Door
        {
            public string id;
            public string[] between;
            public string wall;                    // "z": wall runs along x at fixed z. "x": wall runs along z at fixed x.
            public float[] at;                     // door centre x, z
            public float width;
            public string state;                   // open, half, closed, removed
            public string note;
            public bool RunsAlongX => wall == "z";
        }

        [Serializable]
        public class Landmark
        {
            public string room;
            public string kind;
            public float[] at;
            public float yaw;
        }

        [Serializable]
        public class Fire
        {
            public string id;
            public string room;
            public float[] at;
            public float intensity;
        }

        [Serializable]
        public class EmergencyLight
        {
            public float[] at;
            public string note;
        }

        public Room FindRoom(string id) => rooms.Find(r => r.id == id);

        public static string ConfigDir =>
            Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", "..", "config"));

        public static FloorPlan Load(string path = null)
        {
            path ??= Path.Combine(ConfigDir, "floorplan.json");
            var plan = JsonConvert.DeserializeObject<FloorPlan>(File.ReadAllText(path));
            if (plan == null || plan.rooms.Count == 0)
                throw new InvalidDataException($"floorplan has no rooms: {path}");
            return plan;
        }
    }
}
