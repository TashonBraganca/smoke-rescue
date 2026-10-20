using System;
using System.Collections.Generic;
using UnityEngine;

namespace SmokeRescue.World
{
    /// <summary>
    /// The world's heat, as the thermal camera and air-temperature sensors see it. A top-down float texture over the
    /// house footprint, published as shader globals for the thermal override shader:
    ///   R surface temperature C, G upper-layer gas temperature C, B layer height above floor m, A gas absorption per m.
    /// The state comes from a heat source (today the static fires in floorplan.json; the fire-spread simulation
    /// replaces it). World side only: robots read it through their sensors, never directly.
    /// </summary>
    public sealed class ThermalField
    {
        public interface ISource
        {
            float AmbientC { get; }
            /// <summary>Surface temperature at floor position xz, upper gas temperature, layer height, gas absorption.</summary>
            Vector4 Sample(Vector2 xz, float t);
            /// <summary>Flames as (centre, radius, height, temperature C) for thermal-only flame bodies.</summary>
            IReadOnlyList<(Vector3 centre, float radius, float height, float tempC)> Flames(float t);
        }

        public const float CellM = 0.25f;
        public ISource Source { get; }
        public Texture2D Texture { get; }
        readonly Rect area;
        readonly Color[] pixels;
        float lastT = float.NaN;

        public ThermalField(FloorPlan plan, ISource source)
        {
            Source = source;
            area = Rect.MinMaxRect(plan.footprint[0] - 1f, plan.footprint[1] - 1f, plan.footprint[2] + 1f, plan.footprint[3] + 1f);
            int w = Mathf.CeilToInt(area.width / CellM), h = Mathf.CeilToInt(area.height / CellM);
            Texture = new Texture2D(w, h, TextureFormat.RGBAFloat, false, true)
            { name = "ThermalField", filterMode = FilterMode.Bilinear, wrapMode = TextureWrapMode.Clamp };
            pixels = new Color[w * h];
        }

        /// <summary>Rebuild the field for sim time t and publish it to shaders. Cheap enough to call once per frame.</summary>
        public void Apply(float t)
        {
            if (t == lastT) { Publish(); return; }
            lastT = t;
            int w = Texture.width, h = Texture.height;
            for (int j = 0; j < h; j++)
            for (int i = 0; i < w; i++)
            {
                var xz = new Vector2(area.xMin + (i + 0.5f) * CellM, area.yMin + (j + 0.5f) * CellM);
                var s = Source.Sample(xz, t);
                pixels[j * w + i] = new Color(s.x, s.y, s.z, s.w);
            }
            Texture.SetPixels(pixels);
            Texture.Apply(false);
            Publish();
        }

        void Publish()
        {
            Shader.SetGlobalTexture("_ThermalField", Texture);
            Shader.SetGlobalVector("_ThermalFieldRect", new Vector4(area.xMin, area.yMin, 1f / area.width, 1f / area.height));
            Shader.SetGlobalFloat("_ThermalAmbientC", Source.AmbientC);
        }

        /// <summary>Air temperature at a point: hot upper layer above the layer height, warm lower layer, plus plume heat
        /// near flames. What a body-mounted thermocouple would settle to.</summary>
        public float AirTemperatureC(Vector3 p, float t)
        {
            var s = Source.Sample(new Vector2(p.x, p.z), t);
            float ambient = Source.AmbientC, gas = s.y, layer = s.z;
            float lower = ambient + 0.25f * Mathf.Max(0f, gas - ambient);
            float air = p.y >= layer ? gas : Mathf.Lerp(lower, gas, Mathf.Clamp01(1f - (layer - p.y) / 0.5f) * 0.5f);
            // Radiant and plume heating from the floor-level surface temperature right here (flames, burning furniture).
            return Mathf.Max(air, ambient + 0.35f * Mathf.Max(0f, s.x - ambient));
        }
    }

    /// <summary>Heat from the static fires in floorplan.json: hot surfaces around each fire, a hot gas layer in the fire's
    /// room growing over time, and a weaker layer in rooms that share a door with it. A stand-in until the fire-spread
    /// simulation exists; the numbers are within the ranges of residential room fires before flashover.</summary>
    public sealed class StaticFireHeat : ThermalField.ISource
    {
        public float AmbientC => 20f;
        readonly FloorPlan plan;
        readonly List<(Vector2 at, float intensity, Rect room)> fires = new List<(Vector2, float, Rect)>();

        public StaticFireHeat(FloorPlan plan)
        {
            this.plan = plan;
            foreach (var f in plan.fires)
                fires.Add((new Vector2(f.at[0], f.at[1]), f.intensity, plan.FindRoom(f.room).Rect));
        }

        public Vector4 Sample(Vector2 xz, float t)
        {
            float surf = AmbientC, gas = AmbientC, layer = plan.wall_height, absorb = 0f;
            float growth = Mathf.Clamp01(0.35f + t / 240f);    // fires grow over the first four minutes
            foreach (var (at, intensity, room) in fires)
            {
                float d = Vector2.Distance(xz, at);
                surf = Mathf.Max(surf, AmbientC + 650f * intensity * growth * Mathf.Exp(-(d * d) / (0.9f * 0.9f)));
                bool inRoom = room.Contains(xz);
                bool near = !inRoom && Inflate(room, 2.5f).Contains(xz);
                if (inRoom || near)
                {
                    float k = inRoom ? 1f : 0.35f;
                    gas = Mathf.Max(gas, AmbientC + 380f * intensity * growth * k);
                    layer = Mathf.Min(layer, plan.wall_height - (inRoom ? 1.3f : 0.6f) * growth);
                    absorb = Mathf.Max(absorb, (inRoom ? 0.45f : 0.2f) * growth);
                }
            }
            return new Vector4(surf, gas, layer, absorb);
        }

        public IReadOnlyList<(Vector3 centre, float radius, float height, float tempC)> Flames(float t)
        {
            float growth = Mathf.Clamp01(0.35f + t / 240f);
            var list = new List<(Vector3, float, float, float)>();
            foreach (var (at, intensity, _) in fires)
                list.Add((new Vector3(at.x, 0f, at.y), 0.35f + 0.4f * intensity * growth, 0.6f + 1.2f * intensity * growth, 750f + 250f * intensity));
            return list;
        }

        static Rect Inflate(Rect r, float m) => Rect.MinMaxRect(r.xMin - m, r.yMin - m, r.xMax + m, r.yMax + m);
    }
}
