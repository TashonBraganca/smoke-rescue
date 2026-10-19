using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using SmokeRescue.World;
using UnityEditor;
using UnityEngine;

namespace SmokeRescue.Editor
{
    /// <summary>
    /// Places every landmark from floorplan.json using the models listed in config/props.json.
    /// Models come from assets_cache/ (git-ignored) and are copied into Assets/SmokeRescue/Art/Models/,
    /// where glTFast imports them. Each placed part gets a BoxCollider from its render bounds on the Props layer.
    /// </summary>
    public static class FurnitureBuilder
    {
        public const string RootName = "Furniture";
        public const string ModelsDir = "Assets/SmokeRescue/Art/Models";
        static string CacheRoot => Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", "..", "assets_cache"));

        static readonly HashSet<string> Walkable = new HashSet<string> { "large_rug_patterned" };

        [MenuItem("Tools/Smoke Rescue/Place Furniture")]
        public static void PlaceMenu() => Place(FloorPlan.Load());

        public static GameObject Place(FloorPlan plan)
        {
            var old = GameObject.Find(RootName);
            if (old != null) Object.DestroyImmediate(old);
            var cfg = JObject.Parse(File.ReadAllText(Path.Combine(FloorPlan.ConfigDir, "props.json")));
            var root = new GameObject(RootName);
            int props = LayerMask.NameToLayer(HouseBuilder.PropsLayer);
            var missing = new List<string>();

            foreach (var lm in plan.landmarks)
            {
                var parts = cfg["landmarks"]?[lm.kind] as JArray;
                if (parts == null) { missing.Add($"{lm.kind}: not in props.json"); continue; }
                var group = new GameObject($"{lm.room}_{lm.kind}");
                group.transform.SetParent(root.transform, false);
                group.transform.SetPositionAndRotation(new Vector3(lm.at[0], 0f, lm.at[1]), Quaternion.Euler(0f, lm.yaw, 0f));
                float lastTop = 0f;
                int k = 0;
                foreach (JObject part in parts)
                {
                    var prefab = LoadModel(part, out string label);
                    if (prefab == null) { missing.Add($"{lm.kind}: {label}"); continue; }
                    var go = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
                    go.name = $"{k++}_{label}";
                    go.transform.SetParent(group.transform, false);
                    var off = part["offset"] as JArray;
                    float ox = off != null ? off[0].Value<float>() : 0f;
                    float oy = off == null ? 0f : off[1].Type == JTokenType.String ? lastTop : off[1].Value<float>();
                    float oz = off != null ? off[2].Value<float>() : 0f;
                    go.transform.localPosition = new Vector3(ox, oy, oz);
                    go.transform.localRotation = Quaternion.Euler(0f, part.Value<float?>("yaw") ?? 0f, 0f);
                    if (part["fit_width"] != null) FitWidth(go, part.Value<float>("fit_width"));
                    var b = RenderBounds(go);
                    lastTop = b.max.y;
                    foreach (var t in go.GetComponentsInChildren<Transform>()) t.gameObject.layer = props;
                    if (!Walkable.Contains(lm.kind)) AddBoxCollider(go, b);
                }
            }
            PlaceFiller(plan, cfg, root, props, missing);
            foreach (var t in root.GetComponentsInChildren<Transform>())
                GameObjectUtility.SetStaticEditorFlags(t.gameObject, StaticEditorFlags.ContributeGI | StaticEditorFlags.OccluderStatic | StaticEditorFlags.OccludeeStatic | StaticEditorFlags.BatchingStatic);
            Debug.Log($"[SmokeRescue] Furniture placed: {root.GetComponentsInChildren<MeshRenderer>().Length} renderers for {plan.landmarks.Count} landmarks" +
                      (missing.Count > 0 ? $"\nMissing:\n{string.Join("\n", missing)}" : ""));
            if (missing.Count > 0) Debug.LogWarning($"[SmokeRescue] {missing.Count} furniture parts missing");
            return root;
        }

        /// <summary>The quadruped body model named by props.json "robot_body", imported like furniture. Null if absent.</summary>
        public static GameObject RobotBodyPrefab()
        {
            var cfg = JObject.Parse(File.ReadAllText(Path.Combine(FloorPlan.ConfigDir, "props.json")));
            return cfg["robot_body"] is JObject part ? LoadModel(part, out _) : null;
        }

        public static GameObject LoadPart(JObject part) => LoadModel(part, out _);

        /// <summary>
        /// Filler furniture from props.json "filler": each item along a wall, facing into the room, at the first
        /// position (scanning the walls in 0.4 m steps) that overlaps no other prop, keeps 1.2 m from every door and
        /// keeps the 0.9 m clear path from each of the room's doors to its centre (floorplan clear_path_rule).
        /// </summary>
        static void PlaceFiller(FloorPlan plan, JObject cfg, GameObject root, int propsLayer, List<string> missing)
        {
            if (!(cfg["filler"] is JObject filler)) return;
            var parent = new GameObject("Filler");
            parent.transform.SetParent(root.transform, false);
            int mask = LayerMask.GetMask(HouseBuilder.PropsLayer, HouseBuilder.WallsLayer);
            // Every scripted route segment in every scenario (spawn included) must stay clear of filler.
            var routes = new List<(Vector2, Vector2)>();
            foreach (var file in Directory.GetFiles(FloorPlan.ConfigDir, "scenario_*.json"))
                foreach (var rb in Scenario.Load(Path.GetFileName(file)).robots)
                {
                    var prev = new Vector2(rb.spawn.at[0], rb.spawn.at[1]);
                    foreach (var w in rb.route) { var q = new Vector2(w[0], w[1]); routes.Add((prev, q)); prev = q; }
                }
            foreach (var room in plan.rooms)
            {
                if (!(filler[room.id] is JArray items)) continue;
                var r = room.Rect;
                var doors = plan.doors.Where(d => d.between.Contains(room.id)).ToList();
                var centre = r.center;
                foreach (var item in items)
                {
                    var prefab = LoadModel(new JObject { ["polyhaven"] = item.Value<string>() }, out string label);
                    if (prefab == null) { missing.Add($"filler {room.id}: {label}"); continue; }
                    var go = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
                    go.name = $"{room.id}_{label}";
                    go.transform.SetParent(parent.transform, false);
                    foreach (var t in go.GetComponentsInChildren<Transform>()) t.gameObject.layer = propsLayer;
                    bool placed = false;
                    // Walls: south, east, north, west; the item faces into the room.
                    var walls = new (Vector2 a, Vector2 b, float yaw)[]
                    {
                        (new Vector2(r.xMin, r.yMin), new Vector2(r.xMax, r.yMin), 0f),
                        (new Vector2(r.xMax, r.yMin), new Vector2(r.xMax, r.yMax), 270f),
                        (new Vector2(r.xMax, r.yMax), new Vector2(r.xMin, r.yMax), 180f),
                        (new Vector2(r.xMin, r.yMax), new Vector2(r.xMin, r.yMin), 90f),
                    };
                    foreach (var (a, b, yaw) in walls)
                    {
                        go.transform.SetPositionAndRotation(Vector3.zero, Quaternion.Euler(0f, yaw, 0f));
                        var size = RenderBounds(go).size;
                        float depth = yaw % 180f == 0f ? size.z : size.x, width = yaw % 180f == 0f ? size.x : size.z;
                        Vector2 dir = (b - a).normalized, inward = new Vector2(-dir.y, dir.x);
                        float len = Vector2.Distance(a, b);
                        for (float s = 0.6f + width / 2f; s < len - 0.6f - width / 2f && !placed; s += 0.4f)
                        {
                            Vector2 p = a + dir * s + inward * (0.12f + depth / 2f + 0.03f);
                            if (doors.Any(d => Vector2.Distance(new Vector2(d.at[0], d.at[1]), p) < 1.2f + width / 2f)) continue;
                            if (doors.Any(d => DistToSegment(p, new Vector2(d.at[0], d.at[1]), centre) < 0.45f + Mathf.Max(width, depth) / 2f)) continue;
                            if (routes.Any(seg => DistToSegment(p, seg.Item1, seg.Item2) < 0.7f + Mathf.Max(width, depth) / 2f)) continue;
                            go.transform.SetPositionAndRotation(new Vector3(p.x, 0f, p.y), Quaternion.Euler(0f, yaw, 0f));
                            Physics.SyncTransforms();
                            var bb = RenderBounds(go);
                            if (Physics.OverlapBox(bb.center, bb.extents * 0.98f + new Vector3(0.05f, 0f, 0.05f), Quaternion.identity, mask).Length > 0) continue;
                            AddBoxCollider(go, bb);
                            Physics.SyncTransforms();
                            placed = true;
                        }
                        if (placed) break;
                    }
                    if (!placed) { missing.Add($"filler {room.id}: no free spot for {label}"); Object.DestroyImmediate(go); }
                }
            }
        }

        static float DistToSegment(Vector2 p, Vector2 a, Vector2 b)
        {
            var ab = b - a;
            float t = ab.sqrMagnitude > 1e-6f ? Mathf.Clamp01(Vector2.Dot(p - a, ab) / ab.sqrMagnitude) : 0f;
            return Vector2.Distance(p, a + ab * t);
        }

        static GameObject LoadModel(JObject part, out string label)
        {
            string rel;
            if (part["downloaded"] != null) { rel = part.Value<string>("downloaded"); label = part.Value<string>("credit") ?? Path.GetFileName(Path.GetDirectoryName(rel)); }
            else if (part["polyhaven"] != null) { rel = $"models/{part.Value<string>("polyhaven")}.glb"; label = part.Value<string>("polyhaven"); }
            else if (part["blender"] != null)
            {
                string name = part.Value<string>("variant") ?? part.Value<string>("blender");
                rel = $"models/blender_{name}.glb"; label = name;
            }
            else { label = "unknown part"; return null; }

            string src = Path.Combine(CacheRoot, rel);
            if (!File.Exists(src)) { label += $" (missing {rel})"; return null; }
            // Unique, stable asset name: downloaded models are all called model.glb.
            string assetName = rel.Replace('/', '_').Replace("candidates_", "");
            string dst = $"{ModelsDir}/{assetName}";
            Directory.CreateDirectory(ModelsDir);
            if (!File.Exists(dst) || new FileInfo(dst).Length != new FileInfo(src).Length)
            {
                File.Copy(src, dst, true);
                AssetDatabase.ImportAsset(dst, ImportAssetOptions.ForceUpdate);
            }
            return AssetDatabase.LoadAssetAtPath<GameObject>(dst);
        }

        public static Bounds RenderBounds(GameObject go)
        {
            var rs = go.GetComponentsInChildren<Renderer>();
            if (rs.Length == 0) return new Bounds(go.transform.position, Vector3.zero);
            var b = rs[0].bounds;
            foreach (var r in rs.Skip(1)) b.Encapsulate(r.bounds);
            return b;
        }

        static void FitWidth(GameObject go, float width)
        {
            var b = RenderBounds(go);
            float w = Mathf.Max(b.size.x, b.size.z);
            if (w > 1e-3f) go.transform.localScale *= width / w;
        }

        static void AddBoxCollider(GameObject go, Bounds worldBounds)
        {
            var col = go.AddComponent<BoxCollider>();
            // Bounds are world axis-aligned; convert centre and size into the part's local frame (yaw-only rotations).
            col.center = go.transform.InverseTransformPoint(worldBounds.center);
            Vector3 s = go.transform.InverseTransformVector(worldBounds.size);
            col.size = new Vector3(Mathf.Abs(s.x), Mathf.Abs(s.y), Mathf.Abs(s.z));
        }
    }
}
