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
