using System.Collections.Generic;
using UnityEngine;

namespace SmokeRescue.World
{
    /// <summary>Gives every renderer under this object its own temperature for thermal cameras (people, robot motors,
    /// flame bodies). Property blocks are not saved with scenes, so this re-applies on enable.</summary>
    [ExecuteAlways]
    public sealed class ThermalBody : MonoBehaviour
    {
        public float TemperatureC = 34f;
        static readonly int Prop = Shader.PropertyToID("_ThermalBaseC");
        void OnEnable() => Apply();
        void OnValidate() => Apply();
        public void Apply()
        {
            var block = new MaterialPropertyBlock();
            foreach (var r in GetComponentsInChildren<Renderer>(true))
            {
                r.GetPropertyBlock(block);
                block.SetFloat(Prop, TemperatureC);
                r.SetPropertyBlock(block);
            }
        }
        public static ThermalBody Add(GameObject go, float tempC)
        {
            var b = go.GetComponent<ThermalBody>() ?? go.AddComponent<ThermalBody>();
            b.TemperatureC = tempC;
            b.Apply();
            return b;
        }
    }

    /// <summary>Flames as thermal cameras see them: hot bodies of gas on the ThermalOnly layer, which colour cameras
    /// skip (their visible look comes from the particle fire). Resized from the heat source every frame.</summary>
    public sealed class FlameBodies
    {
        public const string Layer = "ThermalOnly";
        readonly Transform root;
        readonly List<GameObject> bodies = new List<GameObject>();
        readonly Material material;

        public FlameBodies(Transform parent)
        {
            root = new GameObject("FlameBodies").transform;
            root.SetParent(parent, false);
            material = new Material(Shader.Find("HDRP/Unlit")) { name = "FlameBodyThermalOnly" };
        }

        const int Lobes = 6;

        /// <summary>Each flame is a hot core plus five cooler lobes that sway and flicker with sim time, so the thermal
        /// image shows a turbulent plume rather than a solid shape.</summary>
        public void Apply(float t, IReadOnlyList<(Vector3 centre, float radius, float height, float tempC)> flames)
        {
            int layer = LayerMask.NameToLayer(Layer);
            if (layer < 0) throw new System.InvalidOperationException("Layer ThermalOnly is missing: run Tools > Smoke Rescue > Setup Layers.");
            int needed = flames.Count * Lobes;
            while (bodies.Count < needed)
            {
                var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                Object.DestroyImmediate(go.GetComponent<Collider>());
                go.name = $"FlameLobe{bodies.Count}";
                go.layer = layer;
                go.transform.SetParent(root, false);
                var r = go.GetComponent<Renderer>();
                r.sharedMaterial = material;
                r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
                go.AddComponent<ThermalBody>();
                bodies.Add(go);
            }
            for (int i = 0; i < bodies.Count; i++)
            {
                int f = i / Lobes, k = i % Lobes;
                bool on = f < flames.Count && flames[f].radius > 0f;
                bodies[i].SetActive(on);
                if (!on) continue;
                var (c, rad, h, temp) = flames[f];
                float seed = f * 31.7f + k * 7.3f;
                float sway = 0.35f * rad;
                float lift = k == 0 ? 0.35f : 0.25f + 0.6f * Mathf.PerlinNoise(seed, t * 1.7f);
                var offset = new Vector3(sway * (Mathf.PerlinNoise(seed + 3f, t * 2.3f) - 0.5f) * 2f, h * lift,
                                         sway * (Mathf.PerlinNoise(seed + 5f, t * 2.1f) - 0.5f) * 2f);
                float w = rad * (k == 0 ? 1.1f : 0.55f + 0.4f * Mathf.PerlinNoise(seed + 9f, t * 3.1f));
                float tall = h * (k == 0 ? 0.55f : 0.35f + 0.35f * Mathf.PerlinNoise(seed + 11f, t * 2.7f));
                bodies[i].transform.position = c + offset;
                bodies[i].transform.localScale = new Vector3(2f * w, tall, 2f * w);
                float lobeTemp = k == 0 ? temp : temp * (0.6f + 0.25f * Mathf.PerlinNoise(seed + 13f, t * 4f));
                var tb = bodies[i].GetComponent<ThermalBody>();
                if (Mathf.Abs(tb.TemperatureC - lobeTemp) > 0.5f) { tb.TemperatureC = lobeTemp; tb.Apply(); }
            }
        }
    }
}
