using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

namespace SmokeRescue.World
{
    /// <summary>Smoke density (0 to 1) at a Unity world point: the room's scenario density, 0 outside the house.</summary>
    public sealed class SmokeField
    {
        readonly FloorPlan plan;
        readonly Scenario.Smoke smoke;
        public float Time { get; set; }

        public SmokeField(FloorPlan plan, Scenario.Smoke smoke) { this.plan = plan; this.smoke = smoke; }

        public string RoomAt(Vector3 p)
        {
            foreach (var r in plan.rooms)
                if (p.x >= r.rect[0] && p.x <= r.rect[2] && p.z >= r.rect[1] && p.z <= r.rect[3]) return r.id;
            return null;
        }

        public float Density(Vector3 p)
        {
            var room = RoomAt(p);
            return room == null || p.y < 0f || p.y > plan.wall_height ? 0f : smoke.Density(room, Time);
        }

        /// <summary>Density to HDRP mean free path (metres of fog before light is attenuated to 1/e).
        /// 0.15 gives about 20 m (light haze), 0.35 about 8.6 m, 0.6 about 5 m, 1.0 gives 3 m.</summary>
        public static float MeanFreePath(float density) => Mathf.Clamp(3f / Mathf.Max(density, 1e-3f), 1f, 400f);
    }

    /// <summary>
    /// Lighting and smoke for a scenario: no sun, a dark night sky, emergency lights, flickering fire light
    /// clusters, global volumetric haze and one Local Volumetric Fog per room. Apply(t) advances flicker and
    /// density ramp deterministically from sim time, so recordings repeat exactly.
    /// </summary>
    public class Atmosphere : MonoBehaviour
    {
        public const string RootName = "Atmosphere";
        /// <summary>Fixed camera exposure (EV100). Interior lit by fires and 900 lm headlights.</summary>
        public static float FixedExposureEv = 3.0f;

        FloorPlan plan;
        Scenario scenario;
        SmokeField field;
        Fog fog;
        readonly List<(Light light, float baseIntensity, float seed)> fireLights = new List<(Light, float, float)>();
        readonly List<(LocalVolumetricFog volume, string room)> roomFogs = new List<(LocalVolumetricFog, string)>();

        public SmokeField Field => field;

        public static Atmosphere Build(FloorPlan plan, Scenario scenario)
        {
            var old = GameObject.Find(RootName);
            if (old != null) DestroyImmediate(old);
            foreach (var l in FindObjectsByType<Light>(FindObjectsInactive.Exclude))
                if (l.type == LightType.Directional) l.enabled = false;   // night, roof on: no sun

            var go = new GameObject(RootName);
            var a = go.AddComponent<Atmosphere>();
            a.plan = plan;
            a.scenario = scenario;
            a.field = new SmokeField(plan, scenario.smoke);

            var volume = go.AddComponent<Volume>();
            volume.isGlobal = true;
            volume.priority = 100f;
            var profile = ScriptableObject.CreateInstance<VolumeProfile>();
            profile.name = "SmokeRescueAtmosphere";
            var env = profile.Add<VisualEnvironment>(true);
            env.skyType.Override(0);                         // no sky: black night outside, no sky ambient
            env.skyAmbientMode.Override(SkyAmbientMode.Static);
            var exposure = profile.Add<Exposure>(true);
            exposure.mode.Override(ExposureMode.Fixed);
            exposure.fixedExposure.Override(FixedExposureEv);
            a.fog = profile.Add<Fog>(true);
            float haze = scenario.smoke.global_haze;
            a.fog.enabled.Override(haze > 0f);
            a.fog.enableVolumetricFog.Override(true);
            a.fog.meanFreePath.Override(SmokeField.MeanFreePath(haze));
            a.fog.baseHeight.Override(plan.wall_height);     // uniform density below the ceiling
            a.fog.maximumHeight.Override(plan.wall_height + 2f);
            a.fog.albedo.Override(new Color(0.78f, 0.76f, 0.72f));
            a.fog.anisotropy.Override(0.6f);                 // forward scattering: visible headlight shafts
            a.fog.colorMode.Override(FogColorMode.ConstantColor);
            a.fog.color.Override(new Color(0.02f, 0.02f, 0.02f));
            a.fog.depthExtent.Override(24f);
            volume.sharedProfile = profile;

            // One local fog volume per room, so rooms near fires can be thicker than the global haze.
            foreach (var r in plan.rooms)
            {
                var rgo = new GameObject($"Smoke_{r.id}");
                rgo.transform.SetParent(go.transform, false);
                rgo.transform.position = new Vector3((r.rect[0] + r.rect[2]) / 2f, plan.wall_height / 2f, (r.rect[1] + r.rect[3]) / 2f);
                var lvf = rgo.AddComponent<LocalVolumetricFog>();
                lvf.parameters.size = new Vector3(r.rect[2] - r.rect[0], plan.wall_height, r.rect[3] - r.rect[1]);
                lvf.parameters.albedo = new Color(0.72f, 0.7f, 0.66f);
                lvf.parameters.positiveFade = lvf.parameters.negativeFade = new Vector3(0.05f, 0.1f, 0.05f);
                a.roomFogs.Add((lvf, r.id));
            }

            foreach (var e in plan.emergency_lights)
            {
                var l = AddPointLight(go, "EmergencyLight", new Vector3(e.at[0], plan.wall_height - 0.25f, e.at[1]),
                    new Color(1f, 0.92f, 0.82f), 350f, 9f);
                l.shadows = LightShadows.Soft;
            }

            // Dim battery emergency lighting in every room, so walls and furniture beyond the headlight carry
            // world-fixed texture. Without it the house was black outside the headlight and SLAM lost tracking.
            foreach (var r in plan.rooms)
            {
                var c = new Vector3((r.rect[0] + r.rect[2]) / 2f, plan.wall_height - 0.2f, (r.rect[1] + r.rect[3]) / 2f);
                var l = AddPointLight(go, $"RoomLight_{r.id}", c, new Color(1f, 0.86f, 0.7f), 300f, 9f);
                l.shadows = LightShadows.None;
            }

            var rng = new System.Random(scenario.seed);
            foreach (var f in plan.fires)
            {
                var fgo = new GameObject($"Fire_{f.id}");
                fgo.transform.SetParent(go.transform, false);
                fgo.transform.position = new Vector3(f.at[0], 0f, f.at[1]);
                for (int i = 0; i < 3; i++)
                {
                    var off = new Vector3((float)rng.NextDouble() * 0.4f - 0.2f, 0.3f + 0.25f * i, (float)rng.NextDouble() * 0.4f - 0.2f);
                    float lm = 2500f * f.intensity;
                    var l = AddPointLight(fgo, $"FireLight{i}", fgo.transform.position + off, new Color(1f, 0.48f, 0.16f), lm, 7f);
                    l.shadows = i == 0 ? LightShadows.Soft : LightShadows.None;
                    a.fireLights.Add((l, lm, (float)rng.NextDouble() * 100f));
                }
                FlameVisual(fgo, f.intensity);
            }
            a.Apply(0f);
            return a;
        }

        /// <summary>Advance flicker and smoke ramp to sim time t (seconds).</summary>
        public void Apply(float t)
        {
            field.Time = t;
            foreach (var (light, baseIntensity, seed) in fireLights)
            {
                // Two octaves of Perlin noise: slow swell plus fast flicker, 70 to 130 percent.
                float n = 0.65f * Mathf.PerlinNoise(seed, t * 1.3f) + 0.35f * Mathf.PerlinNoise(seed + 17f, t * 9f);
                light.intensity = baseIntensity * (0.7f + 0.6f * n);
            }
            foreach (var (volume, room) in roomFogs)
            {
                float d = scenario.smoke.Density(room, t);
                volume.gameObject.SetActive(d > 1e-3f);
                volume.parameters.meanFreePath = SmokeField.MeanFreePath(d);
            }
            if (fog != null && scenario.smoke.ramp_per_minute > 0f)
                fog.meanFreePath.value = SmokeField.MeanFreePath(Mathf.Clamp01(scenario.smoke.global_haze + scenario.smoke.ramp_per_minute * t / 60f));
        }

        /// <summary>Normalised fire proximity at a point (1 at a fire, 0 beyond 4 m), for sensor exposure flicker.</summary>
        public float NearFire(Vector3 p)
        {
            float best = 0f;
            foreach (var f in plan.fires)
                best = Mathf.Max(best, f.intensity * Mathf.Clamp01(1f - Vector2.Distance(new Vector2(p.x, p.z), new Vector2(f.at[0], f.at[1])) / 4f));
            return best;
        }

        static Light AddPointLight(GameObject parent, string name, Vector3 pos, Color colour, float lumen, float range)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent.transform, false);
            go.transform.position = pos;
            var l = go.AddComponent<Light>();
            l.type = LightType.Point;
            l.color = colour;
            l.lightUnit = LightUnit.Lumen;
            l.intensity = lumen;
            l.range = range;
            return l;
        }

        /// <summary>Glowing flame tongues: emissive stretched spheres. Stands in until a VFX Graph fire is added.</summary>
        static void FlameVisual(GameObject parent, float intensity)
        {
            var mat = new Material(Shader.Find("HDRP/Unlit"));
            mat.SetColor("_UnlitColor", new Color(1f, 0.42f, 0.08f));
            mat.SetColor("_EmissiveColor", new Color(1f, 0.38f, 0.06f) * 40f * intensity);
            HDMaterial.ValidateMaterial(mat);
            var rng = new System.Random(parent.name.GetHashCode());
            for (int i = 0; i < 7; i++)
            {
                var s = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                DestroyImmediate(s.GetComponent<Collider>());
                s.name = "Flame";
                s.transform.SetParent(parent.transform, false);
                float h = 0.25f + (float)rng.NextDouble() * 0.45f * (0.5f + intensity);
                s.transform.localPosition = new Vector3((float)rng.NextDouble() * 0.5f - 0.25f, h / 2f, (float)rng.NextDouble() * 0.5f - 0.25f);
                s.transform.localScale = new Vector3(0.12f, h, 0.12f);
                var r = s.GetComponent<MeshRenderer>();
                r.sharedMaterial = mat;
                r.shadowCastingMode = ShadowCastingMode.Off;
            }
        }
    }
}
