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
        /// <summary>When set (spreading fire), smoke density comes from the fire simulation instead of the scenario.</summary>
        public System.Func<Vector3, float, float> Override { get; set; }

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
            if (room == null || p.y < 0f || p.y > plan.wall_height) return 0f;
            float d = smoke.Density(room, Time);
            return Override == null ? d : Mathf.Max(d, Mathf.Clamp01(Override(p, Time)));
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
        readonly List<(LocalVolumetricFog volume, string room)> roomFogs = new List<(LocalVolumetricFog, string)>();
        static Texture3D smokeNoise;
        static Texture2D softDot;
        static Texture2D sootTexture;

        public SmokeField Field => field;
        /// <summary>World heat for thermal cameras and air-temperature sensors.</summary>
        public ThermalField Thermal => thermal;
        ThermalField thermal;
        FireVisuals fireVisuals;
        FlameBodies flameBodies;
        FireSimHeat fireSim;
        BurnDamage burn;
        /// <summary>The spreading fire, or null when the scenario uses static fires.</summary>
        public FireSimHeat FireSpread => fireSim;

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

            bool Dark(float x, float z) => scenario.dark_rooms != null && plan.rooms.Exists(r => scenario.dark_rooms.Contains(r.id)
                && x >= r.rect[0] && x <= r.rect[2] && z >= r.rect[1] && z <= r.rect[3]);
            foreach (var e in plan.emergency_lights)
            {
                if (Dark(e.at[0], e.at[1])) continue;     // power and battery lighting failed in this room
                var l = AddPointLight(go, "EmergencyLight", new Vector3(e.at[0], plan.wall_height - 0.25f, e.at[1]),
                    new Color(1f, 0.92f, 0.82f), 350f, 9f);
                l.shadows = LightShadows.Soft;
            }

            // Dim battery emergency lighting in every room, so walls and furniture beyond the headlight carry
            // world-fixed texture. Without it the house was black outside the headlight and SLAM lost tracking.
            foreach (var r in plan.rooms)
            {
                if (scenario.dark_rooms != null && scenario.dark_rooms.Contains(r.id)) continue;
                var c = new Vector3((r.rect[0] + r.rect[2]) / 2f, plan.wall_height - 0.2f, (r.rect[1] + r.rect[3]) / 2f);
                var l = AddPointLight(go, $"RoomLight_{r.id}", c, new Color(1f, 0.86f, 0.7f), 300f, 9f);
                l.shadows = LightShadows.None;
            }

            // Fire: one heat source drives what colour cameras see (flames, smoke, light), what thermal cameras see
            // (heat field, flame bodies) and what the air-temperature sensors read.
            if (scenario.fire != null && scenario.fire.spread)
            {
                // The fire sees this run's obstacles as fuel too (burning wood piles in the hallway).
                var firePlan = FloorPlan.Load();     // a fresh copy, so the shared plan keeps only the house
                if (scenario.obstacles != null) firePlan.landmarks.AddRange(scenario.obstacles);
                var spread = new FireSimHeat(firePlan, scenario.seed, scenario.fire.head_start_s);
                a.fireSim = spread;
                a.thermal = new ThermalField(plan, spread);
                a.field.Override = spread.SmokeDensity;
                a.burn = new BurnDamage(spread, go.transform, Soot());
            }
            else a.thermal = new ThermalField(plan, new StaticFireHeat(plan));
            a.fireVisuals = new FireVisuals(go.transform, plan.wall_height, scenario.seed);
            a.flameBodies = new FlameBodies(go.transform);
            foreach (var f in plan.fires)
            {
                var fgo = new GameObject($"Soot_{f.id}");
                fgo.transform.SetParent(go.transform, false);
                fgo.transform.position = new Vector3(f.at[0], 0f, f.at[1]);
                SootDecals(fgo, f, plan);
            }
            a.Apply(0f);
            return a;
        }

        /// <summary>Advance fire, heat, smoke drift and smoke ramp to sim time t (seconds).</summary>
        public void Apply(float t)
        {
            field.Time = t;
            foreach (var (volume, key) in roomFogs)
            {
                bool isLayer = key.EndsWith("#layer");
                string room = isLayer ? key.Substring(0, key.Length - 6) : key;
                float d = scenario.smoke.Density(room, t);
                if (isLayer) d = Mathf.Clamp01(d * 2.5f + (d > 0f ? 0.1f : 0f));
                if (fireSim != null)
                {
                    // Smoke from the fire simulation: a dense hot layer from the ceiling down to the layer height,
                    // thinner smoke below it. Layer volumes resize as the layer sinks.
                    var rs = fireSim.Room(room, t);
                    if (isLayer)
                    {
                        float depth = Mathf.Max(0.3f, plan.wall_height - rs.LayerHeight);
                        var lp = volume.transform.position;
                        volume.transform.position = new Vector3(lp.x, plan.wall_height - depth / 2f, lp.z);
                        var size = volume.parameters.size;
                        volume.parameters.size = new Vector3(size.x, depth, size.z);
                        volume.parameters.albedo = new Color(0.2f, 0.19f, 0.18f);
                        d = Mathf.Max(d, Mathf.Clamp01(rs.SmokeDensity * 1.2f));
                    }
                    else d = Mathf.Max(d, Mathf.Clamp01(fireSim.SmokeDensity(new Vector3(volume.transform.position.x, 0.5f, volume.transform.position.z), t)));
                }
                volume.gameObject.SetActive(d > 1e-3f);
                volume.parameters.meanFreePath = SmokeField.MeanFreePath(d);
                volume.parameters.textureOffset = new Vector3(0.011f * t, -0.017f * t, 0.007f * t);   // slow rising drift
            }
            if (thermal != null)
            {
                thermal.Apply(t);
                var burning = thermal.Source.Flames(t);
                fireVisuals.Apply(t, burning);
                flameBodies.Apply(t, burning);
                burn?.Apply(t, burning);
            }
            if (fog != null && scenario.smoke.ramp_per_minute > 0f)
                fog.meanFreePath.value = SmokeField.MeanFreePath(Mathf.Clamp01(scenario.smoke.global_haze + scenario.smoke.ramp_per_minute * t / 60f));
        }

        /// <summary>Normalised fire proximity at a point (1 at a fire, 0 beyond 4 m), for sensor exposure flicker.</summary>
        public float NearFire(Vector3 p)
        {
            float best = 0f;
            var xz = new Vector2(p.x, p.z);
            foreach (var (c, radius, height, _) in thermal.Source.Flames(field.Time))
                if (radius > 0f)
                    best = Mathf.Max(best, Mathf.Clamp01(height / 1.5f) * Mathf.Clamp01(1f - (Vector2.Distance(xz, new Vector2(c.x, c.z)) - radius) / 4f));
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

        /// <summary>Soft round sprite with a grainy edge, for flame, ember and smoke particles.</summary>
        internal static Texture2D SoftDot()
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
        internal static Texture2D Soot()
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
