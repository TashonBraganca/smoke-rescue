using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

namespace SmokeRescue.World
{
    /// <summary>
    /// Charring from the spreading fire: every furniture renderer darkens toward charcoal as the fuel under it burns
    /// (FireSim burnt fraction), and black char marks fade in on the floor of every block that has burned. Colours are
    /// applied through property blocks, so the saved materials and scene are never changed.
    /// </summary>
    public sealed class BurnDamage
    {
        static readonly int BaseColor = Shader.PropertyToID("_BaseColor");
        static readonly int GltfBaseColor = Shader.PropertyToID("baseColorFactor");
        static readonly Color Char = new Color(0.035f, 0.03f, 0.028f);

        sealed class Part
        {
            public Renderer Renderer;
            public Vector2 Xz;
            public Color[] Original;     // per material slot
            public int[] Property;       // which colour property each slot uses, or -1
            public float Applied = -1f;
        }

        readonly FireSimHeat fire;
        readonly List<Part> parts = new List<Part>();
        readonly Dictionary<Vector2Int, DecalProjector> scorch = new Dictionary<Vector2Int, DecalProjector>();
        readonly Transform root;
        readonly Material scorchMaterial;

        public BurnDamage(FireSimHeat fire, Transform parent, Texture2D sootTexture)
        {
            this.fire = fire;
            root = new GameObject("BurnDamage").transform;
            root.SetParent(parent, false);
            scorchMaterial = new Material(Shader.Find("HDRP/Decal")) { name = "Scorch" };
            scorchMaterial.SetTexture("_BaseColorMap", sootTexture);
            scorchMaterial.SetColor("_BaseColor", new Color(0.02f, 0.018f, 0.016f, 1f));
            HDMaterial.ValidateMaterial(scorchMaterial);

            var furniture = GameObject.Find("Furniture");
            if (furniture == null) return;
            foreach (var r in furniture.GetComponentsInChildren<Renderer>())
            {
                var mats = r.sharedMaterials;
                var part = new Part { Renderer = r, Xz = new Vector2(r.bounds.center.x, r.bounds.center.z),
                                      Original = new Color[mats.Length], Property = new int[mats.Length] };
                for (int i = 0; i < mats.Length; i++)
                {
                    var m = mats[i];
                    part.Property[i] = m == null ? -1 : m.HasProperty(BaseColor) ? BaseColor : m.HasProperty(GltfBaseColor) ? GltfBaseColor : -1;
                    part.Original[i] = part.Property[i] < 0 ? Color.white : m.GetColor(part.Property[i]);
                }
                parts.Add(part);
            }
        }

        public void Apply(float t, IReadOnlyList<(Vector3 centre, float radius, float height, float tempC)> flames)
        {
            var block = new MaterialPropertyBlock();
            foreach (var p in parts)
            {
                float burnt = fire.BurntFraction(p.Xz, t);
                float k = Mathf.Pow(Mathf.Clamp01(burnt * 1.4f), 0.7f);   // char shows well before the fuel is gone
                if (Mathf.Abs(k - p.Applied) < 0.02f) continue;
                p.Applied = k;
                for (int i = 0; i < p.Original.Length; i++)
                {
                    if (p.Property[i] < 0) continue;
                    p.Renderer.GetPropertyBlock(block, i);
                    var c = Color.Lerp(p.Original[i], Char, k);
                    c.a = p.Original[i].a;
                    block.SetColor(p.Property[i], c);
                    p.Renderer.SetPropertyBlock(block, i);
                }
            }
            // Floor scorch: one 1.6 m decal per block that has ever burned, opacity from its burnt fraction.
            foreach (var f in flames)
            {
                if (f.radius <= 0f) continue;
                var key = new Vector2Int(Mathf.FloorToInt(f.centre.x / FireSimHeat.BlockM), Mathf.FloorToInt(f.centre.z / FireSimHeat.BlockM));
                if (scorch.ContainsKey(key)) continue;
                var go = new GameObject($"Scorch_{key.x}_{key.y}");
                go.transform.SetParent(root, false);
                go.transform.SetPositionAndRotation(new Vector3(f.centre.x, 0.3f, f.centre.z), Quaternion.Euler(90f, (key.x * 37 + key.y * 61) % 360, 0f));
                var d = go.AddComponent<DecalProjector>();
                d.material = scorchMaterial;
                d.size = new Vector3(1.8f, 1.8f, 0.6f);
                d.fadeFactor = 0f;
                scorch[key] = d;
            }
            foreach (var kv in scorch)
            {
                var c = new Vector2((kv.Key.x + 0.5f) * FireSimHeat.BlockM, (kv.Key.y + 0.5f) * FireSimHeat.BlockM);
                kv.Value.fadeFactor = Mathf.Clamp01(fire.BurntFraction(c, t) * 1.5f + 0.15f);
            }
        }
    }
}
