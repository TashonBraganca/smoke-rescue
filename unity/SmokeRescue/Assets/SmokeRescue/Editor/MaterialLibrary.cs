using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json.Linq;
using SmokeRescue.World;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

namespace SmokeRescue.Editor
{
    /// <summary>
    /// Turns config/materials.json plus the textures built by tools/fetch_assets.py into HDRP Lit materials.
    /// Textures are copied from assets_cache/textures/&lt;name&gt;/ into Assets/SmokeRescue/Art/Textures/&lt;name&gt;/.
    /// UVs from WorldMesh are in metres, so tiling is 1 / tile_m.
    /// </summary>
    public static class MaterialLibrary
    {
        public const string TexturesDir = "Assets/SmokeRescue/Art/Textures";
        public const string MaterialsDir = "Assets/SmokeRescue/Art/Materials";

        static string CacheDir => Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", "..", "assets_cache", "textures"));

        static readonly Dictionary<string, Material> Cache = new Dictionary<string, Material>();

        public static Material Get(string name)
        {
            if (Cache.TryGetValue(name, out var m) && m != null) return m;
            m = AssetDatabase.LoadAssetAtPath<Material>($"{MaterialsDir}/{name}.mat");
            if (m == null) m = Create(name);
            Cache[name] = m;
            return m;
        }

        public static Material GreyBox(string name)
        {
            const string path = MaterialsDir + "/_greybox.mat";
            var m = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (m != null) return m;
            Directory.CreateDirectory(MaterialsDir);
            m = new Material(Shader.Find("HDRP/Lit"));
            m.SetColor("_BaseColor", new Color(0.62f, 0.62f, 0.6f));
            HDMaterial.ValidateMaterial(m);
            AssetDatabase.CreateAsset(m, path);
            return m;
        }

        [MenuItem("Tools/Smoke Rescue/Import All Materials")]
        public static void ImportAll()
        {
            var cfg = JObject.Parse(File.ReadAllText(Path.Combine(FloorPlan.ConfigDir, "materials.json")));
            int n = 0;
            foreach (var kv in (JObject)cfg["materials"])
            {
                Cache.Remove(kv.Key);
                AssetDatabase.DeleteAsset($"{MaterialsDir}/{kv.Key}.mat");
                Get(kv.Key);
                n++;
            }
            AssetDatabase.SaveAssets();
            Debug.Log($"[SmokeRescue] Imported {n} materials");
        }

        static Material Create(string name)
        {
            var cfg = JObject.Parse(File.ReadAllText(Path.Combine(FloorPlan.ConfigDir, "materials.json")));
            var spec = cfg["materials"]?[name] as JObject;
            if (spec == null) throw new KeyNotFoundException($"material '{name}' is not in config/materials.json");
            string src = Path.Combine(CacheDir, name);
            if (!Directory.Exists(src)) throw new DirectoryNotFoundException($"run tools/fetch_assets.py first: {src} missing");

            string dst = $"{TexturesDir}/{name}";
            Directory.CreateDirectory(dst);
            foreach (var f in new[] { "albedo.png", "normal.png", "mask.png" })
                File.Copy(Path.Combine(src, f), Path.Combine(dst, f), true);
            AssetDatabase.ImportAsset($"{dst}/albedo.png");
            Configure($"{dst}/normal.png", normal: true, srgb: false);
            Configure($"{dst}/mask.png", normal: false, srgb: false);

            var mat = new Material(Shader.Find("HDRP/Lit")) { name = name };
            float tile = spec.Value<float?>("tile_m") ?? 1f;
            var scale = new Vector2(1f / tile, 1f / tile);
            mat.SetTexture("_BaseColorMap", AssetDatabase.LoadAssetAtPath<Texture2D>($"{dst}/albedo.png"));
            mat.SetTextureScale("_BaseColorMap", scale);
            mat.SetTexture("_NormalMap", AssetDatabase.LoadAssetAtPath<Texture2D>($"{dst}/normal.png"));
            mat.SetFloat("_NormalScale", 1f);
            mat.SetTexture("_MaskMap", AssetDatabase.LoadAssetAtPath<Texture2D>($"{dst}/mask.png"));
            mat.SetFloat("_MetallicRemapMax", spec.Value<float?>("metallic") ?? 0f);
            mat.SetFloat("_Metallic", spec.Value<float?>("metallic") ?? 0f);
            HDMaterial.ValidateMaterial(mat);
            Directory.CreateDirectory(MaterialsDir);
            AssetDatabase.CreateAsset(mat, $"{MaterialsDir}/{name}.mat");
            return mat;
        }

        static void Configure(string path, bool normal, bool srgb)
        {
            AssetDatabase.ImportAsset(path);
            var imp = (TextureImporter)AssetImporter.GetAtPath(path);
            imp.textureType = normal ? TextureImporterType.NormalMap : TextureImporterType.Default;
            imp.sRGBTexture = srgb;
            imp.SaveAndReimport();
        }
    }
}
