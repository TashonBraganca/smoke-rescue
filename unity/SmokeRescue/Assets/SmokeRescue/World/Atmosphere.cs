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
        public static float FixedExposureEv = 3.0f;   // recorded in meta.json; kept for reference
        public const float ExposureLimitMinEv = 0f;
        public const float ExposureLimitMaxEv = 9f;

        FloorPlan plan;
        Scenario scenario;
        SmokeField field;
        Fog fog;
        readonly List<(Light light, float baseIntensity, float seed)> fireLights = new List<(Light, float, float)>();
        readonly List<(LocalVolumetricFog volume, string room)> roomFogs = new List<(LocalVolumetricFog, string)>();
        readonly List<(Transform t, Vector3 scale, Vector3 pos, float seed)> flames = new List<(Transform, Vector3, Vector3, float)>();
        readonly List<ParticleSystem> particles = new List<ParticleSystem>();
        float particleTime;
        static Texture3D smokeNoise;
        static Texture2D softDot;
        static Texture2D sootTexture;

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
            // Exposure comes from each camera's own aperture, shutter and ISO. The robot sensors run their own
            // deterministic exposure loop on those (RgbdCamera.AutoExposure). HDRP's automatic exposure did not adapt for
            // cameras rendered on demand (54 to 66 percent of haze pixels clipped), and a single fixed EV blew out
            // frames half a metre from a lit wall while leaving dark rooms black.
            var exposure = profile.Add<Exposure>(true);
            exposure.mode.Override(ExposureMode.UsePhysicalCamera);
            exposure.compensation.Override(0f);
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
                // Billowing smoke: a 3D noise mask, tiled about every 2.5 m, drifted from sim time in Apply.
                lvf.parameters.maskMode = LocalVolumetricFogMaskMode.Texture;
                lvf.parameters.volumeMask = SmokeNoise();
                lvf.parameters.textureTiling = new Vector3((r.rect[2] - r.rect[0]) / 2.5f, plan.wall_height / 2.5f, (r.rect[3] - r.rect[1]) / 2.5f);
                lvf.parameters.textureScrollingSpeed = Vector3.zero;
                a.roomFogs.Add((lvf, r.id));

                // Hot smoke pools under the ceiling: a denser layer in the top 0.8 m of every smoky room.
                var lgo = new GameObject($"SmokeLayer_{r.id}");
                lgo.transform.SetParent(go.transform, false);
                lgo.transform.position = new Vector3((r.rect[0] + r.rect[2]) / 2f, plan.wall_height - 0.4f, (r.rect[1] + r.rect[3]) / 2f);
                var layer = lgo.AddComponent<LocalVolumetricFog>();
                layer.parameters.size = new Vector3(r.rect[2] - r.rect[0], 0.8f, r.rect[3] - r.rect[1]);
                layer.parameters.albedo = new Color(0.45f, 0.43f, 0.41f);
                layer.parameters.positiveFade = new Vector3(0.02f, 0.05f, 0.02f);
                layer.parameters.negativeFade = new Vector3(0.02f, 0.6f, 0.02f);   // soft lower edge
                layer.parameters.maskMode = LocalVolumetricFogMaskMode.Texture;
                layer.parameters.volumeMask = SmokeNoise();
                layer.parameters.textureTiling = new Vector3((r.rect[2] - r.rect[0]) / 1.8f, 0.6f, (r.rect[3] - r.rect[1]) / 1.8f);
                a.roomFogs.Add((layer, r.id + "#layer"));
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
                a.flames.AddRange(FlameVisual(fgo, f.intensity));
                a.particles.AddRange(FireParticles(fgo, f.intensity, scenario.seed * 31 + a.particles.Count));
                SootDecals(fgo, f, plan);
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
            foreach (var (volume, key) in roomFogs)
            {
                bool isLayer = key.EndsWith("#layer");
                string room = isLayer ? key.Substring(0, key.Length - 6) : key;
                float d = scenario.smoke.Density(room, t);
                if (isLayer) d = Mathf.Clamp01(d * 2.5f + (d > 0f ? 0.1f : 0f));
                volume.gameObject.SetActive(d > 1e-3f);
                volume.parameters.meanFreePath = SmokeField.MeanFreePath(d);
                volume.parameters.textureOffset = new Vector3(0.011f * t, -0.017f * t, 0.007f * t);   // slow rising drift
            }
            // Particles advance only by sim time, from a fixed seed, so recordings repeat exactly.
            float pdt = t - particleTime;
            if (pdt < 0f) { foreach (var ps in particles) ps.Simulate(t, true, true, true); }
            else if (pdt > 0f) { foreach (var ps in particles) ps.Simulate(pdt, true, false, true); }
            particleTime = t;
            foreach (var (tr, scale, pos, seed) in flames)
            {
                float h = 0.6f + 0.8f * Mathf.PerlinNoise(seed, t * 3.1f);
                float w = 0.8f + 0.35f * Mathf.PerlinNoise(seed + 5f, t * 4.3f);
                tr.localScale = new Vector3(scale.x * w, scale.y * h, scale.z * w);
                tr.localPosition = pos + new Vector3(0.05f * (Mathf.PerlinNoise(seed + 9f, t * 2f) - 0.5f), (h - 1f) * scale.y * 0.5f, 0.05f * (Mathf.PerlinNoise(seed + 13f, t * 2f) - 0.5f));
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

        /// <summary>Flame tongues: emissive stretched spheres whose height and sway flicker from sim time in Apply,
        /// plus a glowing ember bed. Stands in until a VFX Graph fire is authored.</summary>
        static List<(Transform, Vector3, Vector3, float)> FlameVisual(GameObject parent, float intensity)
        {
            var tongues = new List<(Transform, Vector3, Vector3, float)>();
            Material Emissive(Color c, float power)
            {
                var m = new Material(Shader.Find("HDRP/Unlit"));
                m.SetColor("_UnlitColor", c);
                m.SetColor("_EmissiveColor", c * power * intensity);
                HDMaterial.ValidateMaterial(m);
                return m;
            }
            var outer = Emissive(new Color(1f, 0.33f, 0.05f), 30f);
            var inner = Emissive(new Color(1f, 0.75f, 0.3f), 60f);
            var embers = Emissive(new Color(0.9f, 0.18f, 0.02f), 12f);
            var rng = new System.Random(parent.name.GetHashCode());
            var bed = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            DestroyImmediate(bed.GetComponent<Collider>());
            bed.name = "EmberBed";
            bed.transform.SetParent(parent.transform, false);
            bed.transform.localScale = new Vector3(0.7f, 0.02f, 0.7f);
            bed.transform.localPosition = new Vector3(0f, 0.02f, 0f);
            bed.GetComponent<MeshRenderer>().sharedMaterial = embers;
            for (int i = 0; i < 5; i++)
            {
                var sph = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                DestroyImmediate(sph.GetComponent<Collider>());
                sph.name = "Flame";
                sph.transform.SetParent(parent.transform, false);
                bool core = i < 3;
                float h = (core ? 0.12f : 0.18f) + (float)rng.NextDouble() * 0.2f * (0.5f + intensity);
                var scale = new Vector3(core ? 0.07f : 0.1f, h, core ? 0.07f : 0.1f);
                var pos = new Vector3((float)rng.NextDouble() * 0.5f - 0.25f, h / 2f, (float)rng.NextDouble() * 0.5f - 0.25f);
                sph.transform.localScale = scale;
                sph.transform.localPosition = pos;
                var r = sph.GetComponent<MeshRenderer>();
                r.sharedMaterial = core ? inner : outer;
                r.shadowCastingMode = ShadowCastingMode.Off;
                tongues.Add((sph.transform, scale, pos, (float)rng.NextDouble() * 100f));
            }
            return tongues;
        }

        /// <summary>
        /// Flames, embers and smoke for one fire as particle systems with fixed seeds, simulated manually from sim time.
        /// Flames and embers use an additive shader that honours particle colour; smoke is alpha-blended and dark.
        /// </summary>
        static List<ParticleSystem> FireParticles(GameObject parent, float intensity, int seed)
        {
            var list = new List<ParticleSystem>();
            ParticleSystem Make(string name, Shader shader, float rate, Vector2 life, Vector2 speed, Vector2 size, float radius,
                                Gradient colour, AnimationCurve sizeCurve, float noise, float gravity, int maxN)
            {
                var go = new GameObject(name);
                go.transform.SetParent(parent.transform, false);
                go.transform.localRotation = Quaternion.Euler(-90f, 0f, 0f);   // emit upward
                var ps = go.AddComponent<ParticleSystem>();
                ps.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
                ps.useAutoRandomSeed = false;
                ps.randomSeed = (uint)(seed + list.Count * 101);
                var main = ps.main;
                main.playOnAwake = false;
                main.loop = true;
                main.duration = 5f;
                main.startLifetime = new ParticleSystem.MinMaxCurve(life.x, life.y);
                main.startSpeed = new ParticleSystem.MinMaxCurve(speed.x, speed.y);
                main.startSize = new ParticleSystem.MinMaxCurve(size.x, size.y);
                main.startRotation = new ParticleSystem.MinMaxCurve(0f, 2f * Mathf.PI);
                main.simulationSpace = ParticleSystemSimulationSpace.World;
                main.gravityModifier = gravity;
                main.maxParticles = maxN;
                var em = ps.emission; em.rateOverTime = rate * Mathf.Max(0.3f, intensity);
                var sh = ps.shape; sh.shapeType = ParticleSystemShapeType.Cone; sh.angle = 12f; sh.radius = radius;
                var col = ps.colorOverLifetime; col.enabled = true; col.color = colour;
                var sz = ps.sizeOverLifetime; sz.enabled = true; sz.size = new ParticleSystem.MinMaxCurve(1f, sizeCurve);
                var nz = ps.noise; nz.enabled = noise > 0f; nz.strength = noise; nz.frequency = 1.4f; nz.scrollSpeed = 0.6f; nz.quality = ParticleSystemNoiseQuality.Medium;
                var rot = ps.rotationOverLifetime; rot.enabled = true; rot.z = new ParticleSystem.MinMaxCurve(-1f, 1f);
                var r = go.GetComponent<ParticleSystemRenderer>();
                r.renderMode = ParticleSystemRenderMode.Billboard;
                var m = new Material(shader);
                m.SetTexture("_MainTex", SoftDot());
                if (shader.name.Contains("Additive")) m.SetFloat("_Intensity", 6f + 10f * intensity);
                r.sharedMaterial = m;
                r.shadowCastingMode = ShadowCastingMode.Off;
                list.Add(ps);
                return ps;
            }
            var add = Shader.Find("SmokeRescue/ParticleAdditive");
            var smoke = Shader.Find("SmokeRescue/ParticleSmoke");
            if (add == null || smoke == null) { Debug.LogWarning("[SmokeRescue] particle shaders not found"); return list; }
            Gradient G(params (float t, Color c)[] keys)
            {
                var g = new Gradient();
                var ck = new GradientColorKey[keys.Length]; var ak = new GradientAlphaKey[keys.Length];
                for (int i = 0; i < keys.Length; i++) { ck[i] = new GradientColorKey(keys[i].c, keys[i].t); ak[i] = new GradientAlphaKey(keys[i].c.a, keys[i].t); }
                g.SetKeys(ck, ak);
                return g;
            }
            var shrink = new AnimationCurve(new Keyframe(0f, 0.6f), new Keyframe(0.25f, 1f), new Keyframe(1f, 0.15f));
            var grow = new AnimationCurve(new Keyframe(0f, 0.3f), new Keyframe(1f, 1f));
            Make("FlameCore", add, 70f, new Vector2(0.35f, 0.6f), new Vector2(0.5f, 0.9f), new Vector2(0.18f, 0.32f), 0.12f,
                G((0f, new Color(1f, 0.95f, 0.7f, 0f)), (0.1f, new Color(1f, 0.85f, 0.45f, 1f)), (0.6f, new Color(1f, 0.5f, 0.12f, 0.6f)), (1f, new Color(0.8f, 0.15f, 0.02f, 0f))),
                shrink, 0.25f, -0.15f, 300);
            Make("FlameOuter", add, 45f, new Vector2(0.6f, 1.1f), new Vector2(0.6f, 1.2f), new Vector2(0.3f, 0.55f), 0.28f,
                G((0f, new Color(1f, 0.6f, 0.15f, 0f)), (0.15f, new Color(1f, 0.45f, 0.08f, 0.8f)), (0.7f, new Color(0.85f, 0.18f, 0.03f, 0.4f)), (1f, new Color(0.3f, 0.05f, 0.01f, 0f))),
                shrink, 0.45f, -0.2f, 300);
            Make("Embers", add, 14f, new Vector2(1.5f, 3f), new Vector2(0.6f, 1.6f), new Vector2(0.015f, 0.035f), 0.3f,
                G((0f, new Color(1f, 0.7f, 0.3f, 1f)), (0.7f, new Color(1f, 0.35f, 0.05f, 0.8f)), (1f, new Color(0.6f, 0.1f, 0f, 0f))),
                AnimationCurve.Constant(0f, 1f, 1f), 1.2f, -0.05f, 120);
            var smk = Make("SmokePlume", smoke, 5f, new Vector2(5f, 8f), new Vector2(0.25f, 0.45f), new Vector2(0.5f, 0.8f), 0.3f,
                G((0f, new Color(0.15f, 0.14f, 0.13f, 0f)), (0.15f, new Color(0.12f, 0.11f, 0.1f, 0.55f)), (0.8f, new Color(0.18f, 0.17f, 0.16f, 0.35f)), (1f, new Color(0.2f, 0.2f, 0.2f, 0f))),
                grow, 0.35f, 0f, 80);
            smk.transform.localPosition = new Vector3(0f, 0.6f, 0f);
            var smm = smk.main; smm.startSize = new ParticleSystem.MinMaxCurve(0.6f, 1.0f);
            var sr = smk.GetComponent<ParticleSystemRenderer>().sharedMaterial; sr.SetColor("_Tint", new Color(0.09f, 0.085f, 0.08f, 0.85f));
            var sizeMul = smk.sizeOverLifetime; sizeMul.sizeMultiplier = 3.2f;
            return list;
        }

        /// <summary>Soft round sprite with a grainy edge, for flame, ember and smoke particles.</summary>
        static Texture2D SoftDot()
        {
            if (softDot != null) return softDot;
            const int n = 128;
            var px = new Color[n * n];
            for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++)
            {
                float u = x / (n - 1f) * 2f - 1f, v = y / (n - 1f) * 2f - 1f;
                float r = Mathf.Sqrt(u * u + v * v);
                float g = 0.75f + 0.25f * Mathf.PerlinNoise(x * 0.09f, y * 0.09f);
                float a = Mathf.Clamp01(1f - r) ;
                a = a * a * (3f - 2f * a) * g;
                px[y * n + x] = new Color(1f, 1f, 1f, a);
            }
            softDot = new Texture2D(n, n, TextureFormat.RGBA32, true) { wrapMode = TextureWrapMode.Clamp, name = "SoftDot" };
            softDot.SetPixels(px);
            softDot.Apply();
            return softDot;
        }

        /// <summary>Scorch marks around a fire: a decal on the floor, one on the ceiling above, and one on each wall
        /// closer than 1.5 m (SIMULATION_SPEC 3.3).</summary>
        static void SootDecals(GameObject parent, FloorPlan.Fire f, FloorPlan plan)
        {
            var mat = new Material(Shader.Find("HDRP/Decal"));
            mat.SetTexture("_BaseColorMap", Soot());
            mat.SetColor("_BaseColor", new Color(0.04f, 0.035f, 0.03f, 1f));
            mat.SetFloat("_AffectAlbedo", 1f);
            mat.SetFloat("_AffectNormal", 0f);
            mat.SetFloat("_AffectMetal", 0f);
            mat.SetFloat("_AffectAO", 0f);
            mat.SetFloat("_AffectSmoothness", 1f);
            mat.SetFloat("_DecalBlend", 0.95f);
            mat.enableInstancing = true;
            HDMaterial.ValidateMaterial(mat);
            void Decal(string name, Vector3 pos, Quaternion rot, Vector3 size)
            {
                var go = new GameObject(name);
                go.transform.SetParent(parent.transform, false);
                go.transform.SetPositionAndRotation(pos, rot);
                var d = go.AddComponent<DecalProjector>();
                d.material = mat;
                d.size = size;
                d.pivot = Vector3.zero;
            }
            var c = new Vector3(f.at[0], 0f, f.at[1]);
            float s = 1.6f + 1.2f * f.intensity;
            Decal("SootFloor", c + Vector3.up * 0.05f, Quaternion.Euler(90f, 0f, 0f), new Vector3(s, s, 0.3f));
            Decal("SootCeiling", c + Vector3.up * (plan.wall_height - 0.05f), Quaternion.Euler(-90f, 0f, 0f), new Vector3(s * 1.4f, s * 1.4f, 0.3f));
            var room = plan.FindRoom(f.room);
            if (room == null) return;
            var r = room.Rect;
            (float dist, Vector3 wallPoint, Quaternion facing)[] walls =
            {
                (c.x - r.xMin, new Vector3(r.xMin, 1.2f, c.z), Quaternion.Euler(0f, -90f, 0f)),
                (r.xMax - c.x, new Vector3(r.xMax, 1.2f, c.z), Quaternion.Euler(0f, 90f, 0f)),
                (c.z - r.yMin, new Vector3(c.x, 1.2f, r.yMin), Quaternion.Euler(0f, 180f, 0f)),
                (r.yMax - c.z, new Vector3(c.x, 1.2f, r.yMax), Quaternion.identity),
            };
            foreach (var (dist, p, rot) in walls)
                if (dist < 1.5f) Decal("SootWall", p, rot, new Vector3(s, 2.4f, 0.6f));
        }

        /// <summary>Tileable fractal value noise in 32^3, generated once.</summary>
        static Texture3D SmokeNoise()
        {
            if (smokeNoise != null) return smokeNoise;
            const int n = 32;
            var rng = new System.Random(4242);
            var lattice = new float[8 * 8 * 8];
            for (int i = 0; i < lattice.Length; i++) lattice[i] = (float)rng.NextDouble();
            float L(int x, int y, int z) => lattice[((z & 7) * 8 + (y & 7)) * 8 + (x & 7)];
            float Smooth(float t) => t * t * (3f - 2f * t);
            float Value(float x, float y, float z)
            {
                int xi = Mathf.FloorToInt(x), yi = Mathf.FloorToInt(y), zi = Mathf.FloorToInt(z);
                float tx = Smooth(x - xi), ty = Smooth(y - yi), tz = Smooth(z - zi);
                float c00 = Mathf.Lerp(L(xi, yi, zi), L(xi + 1, yi, zi), tx), c10 = Mathf.Lerp(L(xi, yi + 1, zi), L(xi + 1, yi + 1, zi), tx);
                float c01 = Mathf.Lerp(L(xi, yi, zi + 1), L(xi + 1, yi, zi + 1), tx), c11 = Mathf.Lerp(L(xi, yi + 1, zi + 1), L(xi + 1, yi + 1, zi + 1), tx);
                return Mathf.Lerp(Mathf.Lerp(c00, c10, ty), Mathf.Lerp(c01, c11, ty), tz);
            }
            var cols = new Color[n * n * n];
            for (int z = 0; z < n; z++)
            for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++)
            {
                float fx = x * 8f / n, fy = y * 8f / n, fz = z * 8f / n;
                float v = 0.55f * Value(fx, fy, fz) + 0.3f * Value(fx * 2f, fy * 2f, fz * 2f) + 0.15f * Value(fx * 4f, fy * 4f, fz * 4f);
                v = Mathf.Clamp01(0.25f + 1.1f * v);
                cols[(z * n + y) * n + x] = new Color(v, v, v, v);
            }
            smokeNoise = new Texture3D(n, n, n, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Repeat, name = "SmokeNoise" };
            smokeNoise.SetPixels(cols);
            smokeNoise.Apply();
            return smokeNoise;
        }

        /// <summary>Radial soot stain with ragged edges: dark centre, noisy falloff to transparent.</summary>
        static Texture2D Soot()
        {
            if (sootTexture != null) return sootTexture;
            const int n = 256;
            var rng = new System.Random(77);
            float[] offs = new float[16];
            for (int i = 0; i < offs.Length; i++) offs[i] = (float)rng.NextDouble();
            var px = new Color[n * n];
            for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++)
            {
                float u = x / (n - 1f) * 2f - 1f, v = y / (n - 1f) * 2f - 1f;
                float r = Mathf.Sqrt(u * u + v * v);
                float ang = Mathf.Atan2(v, u);
                // Irregular, not star-shaped: the edge wobbles with low-frequency noise around the circle.
                float edge = 0.7f + 0.22f * (Mathf.PerlinNoise(Mathf.Cos(ang) * 1.3f + offs[0] * 10f, Mathf.Sin(ang) * 1.3f + offs[1] * 10f) - 0.5f);
                float grain = 0.6f * Mathf.PerlinNoise(x * 0.04f + offs[2] * 50f, y * 0.04f + offs[3] * 50f) + 0.4f * Mathf.PerlinNoise(x * 0.15f, y * 0.15f);
                float a = Mathf.Clamp01((edge - r) / 0.45f) * (0.55f + 0.45f * grain);
                px[y * n + x] = new Color(0.05f, 0.045f, 0.04f, a);
            }
            sootTexture = new Texture2D(n, n, TextureFormat.RGBA32, true) { wrapMode = TextureWrapMode.Clamp, name = "Soot" };
            sootTexture.SetPixels(px);
            sootTexture.Apply();
            return sootTexture;
        }
    }
}
