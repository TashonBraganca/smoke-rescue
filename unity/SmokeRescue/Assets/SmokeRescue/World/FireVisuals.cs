using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace SmokeRescue.World
{
    /// <summary>
    /// What a burning region looks like: animated flame flipbooks (Unity Labs, CC0) sized to the region, rising embers,
    /// thick black smoke that climbs to the ceiling and spreads under it, and flickering fire light that scatters in the
    /// smoke. Regions come from the heat source (static fires today, the spreading fire simulation next); each region keeps
    /// its emitters for the whole run and only grows or shrinks, so the look is continuous. Particles advance only by
    /// sim time from fixed seeds, so recordings repeat exactly.
    /// </summary>
    public sealed class FireVisuals
    {
        sealed class Region
        {
            public Transform Root;
            public ParticleSystem Body, Tongues, Embers, Smoke;
            public Light[] Lights;
            public float LightSeed;
        }

        readonly Transform root;
        readonly float ceiling;
        readonly int seed;
        readonly List<Region> regions = new List<Region>();
        readonly Transform ceilingPlane;
        float simTime;
        bool warmed;
        static Material flameBody, flameTongue, ember, smoke;

        public FireVisuals(Transform parent, float wallHeight, int seed)
        {
            root = new GameObject("FireVisuals").transform;
            root.SetParent(parent, false);
            ceiling = wallHeight;
            this.seed = seed;
            ceilingPlane = new GameObject("SmokeCeilingPlane").transform;
            ceilingPlane.SetParent(root, false);
            ceilingPlane.position = new Vector3(0f, wallHeight - 0.08f, 0f);
            ceilingPlane.rotation = Quaternion.Euler(180f, 0f, 0f);   // collision plane normal points down
            Materials();
        }

        /// <summary>Advance to sim time t with the current burning regions (centre on the floor, radius, flame height, C).</summary>
        public const int MaxLights = 8, MaxShadowLights = 8;   // every fire light casts shadows, so walls stop it

        public void Apply(float t, IReadOnlyList<(Vector3 centre, float radius, float height, float tempC)> flames)
        {
            while (regions.Count < flames.Count) regions.Add(MakeRegion(regions.Count));
            // Light budget: HDRP slows down with many shadowed point lights, so only the biggest fires get a light, and
            // only the biggest few cast shadows.
            var rank = new List<int>();
            for (int i = 0; i < flames.Count; i++) if (flames[i].radius > 0f) rank.Add(i);
            rank.Sort((a, b) => (flames[b].radius * flames[b].height).CompareTo(flames[a].radius * flames[a].height));
            var lightOrder = new Dictionary<int, int>();
            for (int k = 0; k < rank.Count && k < MaxLights; k++) lightOrder[rank[k]] = k;

            for (int i = 0; i < regions.Count; i++)
            {
                var r = regions[i];
                bool on = i < flames.Count && flames[i].radius > 0f;
                float radius = on ? flames[i].radius : 0f, height = on ? flames[i].height : 0f;
                if (on) r.Root.position = flames[i].centre + Vector3.up * FuelTop(flames[i].centre);
                Shape(r.Body, radius * 0.8f, on ? 40f * Area(radius) : 0f, new Vector2(0.55f, 0.9f) * Mathf.Max(0.6f, height * 0.6f), 2f);
                Shape(r.Tongues, radius * 0.9f, on ? 18f * Area(radius) : 0f, new Vector2(0.35f, 0.6f) * Mathf.Max(0.6f, height * 0.6f), 2f);
                Shape(r.Embers, radius, on ? 25f * Area(radius) : 0f, new Vector2(0.015f, 0.035f));
                Shape(r.Smoke, radius * 0.7f, on ? 6f * Area(radius) : 0f, new Vector2(0.9f, 1.6f) * Mathf.Max(0.8f, radius));
                if (on)
                {
                    var main = r.Body.main; main.startSpeed = new ParticleSystem.MinMaxCurve(0.5f * height, 1.0f * height);
                    var tmain = r.Tongues.main; tmain.startSpeed = new ParticleSystem.MinMaxCurve(0.8f * height, 1.4f * height);
                }
                bool lit = on && lightOrder.TryGetValue(i, out int order);
                var l = r.Lights[0];
                l.enabled = lit;
                if (!lit) continue;
                order = lightOrder[i];
                l.shadows = order < MaxShadowLights ? LightShadows.Soft : LightShadows.None;
                l.transform.localPosition = new Vector3(0f, 0.3f + 0.4f * height, 0f);
                float n = 0.65f * Mathf.PerlinNoise(r.LightSeed, t * 1.3f) + 0.35f * Mathf.PerlinNoise(r.LightSeed + 17f, t * 9f);
                l.intensity = 3000f * Mathf.Clamp(Area(radius), 0.3f, 1.5f) * (0.7f + 0.6f * n);
                l.range = 8f;
            }
            float dt = t - simTime;
            if (!warmed)
            {
                // The house is already burning when the run starts: pre-run every emitter for 2 s so flames and smoke
                // are fully formed in the first frame.
                warmed = true;
                foreach (var r in regions)
                    foreach (var ps in new[] { r.Body, r.Tongues, r.Embers, r.Smoke }) ps.Simulate(2f, true, false, true);
            }
            foreach (var r in regions)
                foreach (var ps in new[] { r.Body, r.Tongues, r.Embers, r.Smoke })
                {
                    if (dt < 0f) ps.Simulate(t, true, true, true);
                    else if (dt > 0f) ps.Simulate(dt, true, false, true);
                }
            simTime = t;
        }

        /// <summary>Height of what is burning at this spot (sofa seat, bed, table top), so flames start on the fuel instead
        /// of inside it. A ray down from just under the ceiling onto the furniture colliders; the floor if nothing is hit.</summary>
        float FuelTop(Vector3 xz)
        {
            var key = new Vector2Int(Mathf.RoundToInt(xz.x * 8f), Mathf.RoundToInt(xz.z * 8f));
            if (fuelTop.TryGetValue(key, out float y)) return y;
            int mask = 1 << LayerMask.NameToLayer("Props");
            y = Physics.Raycast(new Vector3(xz.x, ceiling - 0.15f, xz.z), Vector3.down, out var hit, ceiling, mask)
                ? Mathf.Clamp(hit.point.y, 0f, 1.2f) : 0f;
            fuelTop[key] = y;
            return y;
        }
        readonly Dictionary<Vector2Int, float> fuelTop = new Dictionary<Vector2Int, float>();

        static float Area(float radius) => Mathf.Max(0.15f, radius * radius * 3.1416f);

        static void Shape(ParticleSystem ps, float radius, float rate, Vector2 size, float aspect = 1f)
        {
            var sh = ps.shape; sh.radius = Mathf.Max(0.05f, radius);
            var em = ps.emission; em.rateOverTime = rate;
            var main = ps.main;
            main.startSize3D = aspect != 1f;
            if (aspect != 1f)
            {
                main.startSizeX = new ParticleSystem.MinMaxCurve(size.x, size.y);
                main.startSizeY = new ParticleSystem.MinMaxCurve(size.x * aspect, size.y * aspect);
                main.startSizeZ = new ParticleSystem.MinMaxCurve(size.x, size.y);
            }
            else main.startSize = new ParticleSystem.MinMaxCurve(size.x, size.y);
        }

        Region MakeRegion(int index)
        {
            var go = new GameObject($"FireRegion{index}");
            go.transform.SetParent(root, false);
            var r = new Region { Root = go.transform, LightSeed = index * 13.7f + seed };
            int s = seed * 977 + index * 131;
            r.Body = Emitter(go, "FlameBody", flameBody, s, 16, 4, life: new Vector2(0.7f, 1.1f), maxN: 400, flipbookFps: 28f, upward: true);
            r.Tongues = Emitter(go, "FlameTongues", flameTongue, s + 1, 16, 4, life: new Vector2(0.5f, 0.9f), maxN: 250, flipbookFps: 34f, upward: true);
            r.Embers = Emitter(go, "Embers", ember, s + 2, 1, 1, life: new Vector2(1.5f, 3.5f), maxN: 400, flipbookFps: 0f, upward: true);
            r.Smoke = Emitter(go, "Smoke", smoke, s + 3, 5, 5, life: new Vector2(6f, 10f), maxN: 260, flipbookFps: 0f, upward: true);
            r.Smoke.transform.localPosition = new Vector3(0f, 0.9f, 0f);

            // Flames: bright at the base, orange body, fading tips; slight sideways drift from noise.
            Colour(r.Body, (0f, new Color(1f, 0.9f, 0.6f, 0f)), (0.12f, new Color(1f, 0.75f, 0.4f, 1f)), (0.7f, new Color(1f, 0.45f, 0.12f, 0.7f)), (1f, new Color(0.6f, 0.12f, 0.02f, 0f)));
            Colour(r.Tongues, (0f, new Color(1f, 0.8f, 0.45f, 0f)), (0.2f, new Color(1f, 0.6f, 0.2f, 1f)), (1f, new Color(0.8f, 0.2f, 0.03f, 0f)));
            Colour(r.Embers, (0f, new Color(1f, 0.75f, 0.35f, 1f)), (0.7f, new Color(1f, 0.35f, 0.05f, 0.9f)), (1f, new Color(0.6f, 0.1f, 0f, 0f)));
            Colour(r.Smoke, (0f, new Color(0.05f, 0.045f, 0.04f, 0f)), (0.1f, new Color(0.06f, 0.055f, 0.05f, 0.85f)), (0.7f, new Color(0.08f, 0.075f, 0.07f, 0.6f)), (1f, new Color(0.1f, 0.1f, 0.1f, 0f)));
            Noise(r.Body, 0.35f); Noise(r.Tongues, 0.5f); Noise(r.Embers, 1.4f); Noise(r.Smoke, 0.25f);
            Grow(r.Body, 0.7f, 1.1f, 0.5f); Grow(r.Tongues, 0.8f, 1f, 0.3f); Grow(r.Smoke, 0.5f, 3.5f, 3.5f);
            var sm = r.Smoke.main; sm.startSpeed = new ParticleSystem.MinMaxCurve(0.5f, 0.9f);
            var em = r.Embers.main; em.startSpeed = new ParticleSystem.MinMaxCurve(0.8f, 2.2f); em.gravityModifier = -0.04f;
            var smokeSheet = r.Smoke.textureSheetAnimation; smokeSheet.enabled = true; smokeSheet.numTilesX = 5; smokeSheet.numTilesY = 5;
            smokeSheet.frameOverTime = new ParticleSystem.MinMaxCurve(1f, AnimationCurve.Linear(0f, 0.15f, 1f, 0.95f));

            // Smoke hits the ceiling and spreads sideways under it, the way hot smoke fills a room from the top down.
            var col = r.Smoke.collision; col.enabled = true; col.type = ParticleSystemCollisionType.Planes;
            col.SetPlane(0, ceilingPlane); col.dampen = 0.6f; col.bounce = 0f; col.lifetimeLoss = 0f; col.radiusScale = 0.4f;

            // Unity only emits from a system that has been started. Start and pause each one now; Apply then advances them
            // by sim time with Simulate. (Created stopped and only simulated, they aged no particles and emitted none.)
            foreach (var ps in new[] { r.Body, r.Tongues, r.Embers, r.Smoke }) { ps.Play(false); ps.Pause(false); }

            r.Lights = new Light[1];
            for (int k = 0; k < r.Lights.Length; k++)
            {
                var lgo = new GameObject($"FireLight{k}");
                lgo.transform.SetParent(go.transform, false);
                var l = lgo.AddComponent<Light>();
                l.type = LightType.Point;
                l.color = new Color(1f, 0.5f, 0.18f);
                l.lightUnit = LightUnit.Lumen;
                l.shadows = k == 0 ? LightShadows.Soft : LightShadows.None;
                r.Lights[k] = l;
            }
            return r;
        }

        static ParticleSystem Emitter(GameObject parent, string name, Material mat, int seed, int tilesX, int tilesY,
                                      Vector2 life, int maxN, float flipbookFps, bool upward)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent.transform, false);
            if (upward) go.transform.localRotation = Quaternion.Euler(-90f, 0f, 0f);
            var ps = go.AddComponent<ParticleSystem>();
            ps.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
            ps.useAutoRandomSeed = false;
            ps.randomSeed = (uint)seed;
            var main = ps.main;
            main.playOnAwake = false; main.loop = true; main.duration = 5f;
            main.startLifetime = new ParticleSystem.MinMaxCurve(life.x, life.y);
            main.startRotation = new ParticleSystem.MinMaxCurve(-0.25f, 0.25f);
            main.simulationSpace = ParticleSystemSimulationSpace.World;
            main.maxParticles = maxN;
            // A cone around the emitter's +Z, which the -90 degree X rotation points up: particles start anywhere on
            // the burning disc and rise. (A circle shape emits outward in its own plane, which laid flames on the floor.)
            var sh = ps.shape; sh.shapeType = ParticleSystemShapeType.Cone; sh.angle = 8f; sh.radiusThickness = 1f;
            if (tilesX * tilesY > 1 && flipbookFps > 0f)
            {
                var tsa = ps.textureSheetAnimation;
                tsa.enabled = true; tsa.numTilesX = tilesX; tsa.numTilesY = tilesY;
                tsa.timeMode = ParticleSystemAnimationTimeMode.FPS; tsa.fps = flipbookFps;
                tsa.startFrame = new ParticleSystem.MinMaxCurve(0f, tilesX * tilesY - 1);
                tsa.cycleCount = 1;
            }
            var r = go.GetComponent<ParticleSystemRenderer>();
            r.renderMode = ParticleSystemRenderMode.Billboard;
            r.sharedMaterial = mat;
            r.shadowCastingMode = ShadowCastingMode.Off;
            r.maxParticleSize = 3f;
            return ps;
        }

        static void Colour(ParticleSystem ps, params (float t, Color c)[] keys)
        {
            var g = new Gradient();
            var ck = new GradientColorKey[keys.Length]; var ak = new GradientAlphaKey[keys.Length];
            for (int i = 0; i < keys.Length; i++) { ck[i] = new GradientColorKey(keys[i].c, keys[i].t); ak[i] = new GradientAlphaKey(keys[i].c.a, keys[i].t); }
            g.SetKeys(ck, ak);
            var col = ps.colorOverLifetime; col.enabled = true; col.color = g;
        }
        static void Noise(ParticleSystem ps, float strength)
        {
            var nz = ps.noise; nz.enabled = true; nz.strength = strength; nz.frequency = 1.2f; nz.scrollSpeed = 0.5f;
            nz.quality = ParticleSystemNoiseQuality.Medium;
        }
        static void Grow(ParticleSystem ps, float start, float end, float multiplier)
        {
            var sz = ps.sizeOverLifetime; sz.enabled = true;
            sz.size = new ParticleSystem.MinMaxCurve(Mathf.Max(1f, multiplier), AnimationCurve.EaseInOut(0f, start / Mathf.Max(1f, multiplier), 1f, end / Mathf.Max(1f, multiplier)));
        }

        static void Materials()
        {
            if (flameBody != null) return;
            var add = Shader.Find("SmokeRescue/ParticleAdditive");
            var alpha = Shader.Find("SmokeRescue/ParticleSmoke");
            if (add == null || alpha == null) throw new System.InvalidOperationException("Particle shaders missing.");
            Material M(Shader s, string tex, float intensity)
            {
                var m = new Material(s) { name = tex };
                m.SetTexture("_MainTex", Resources.Load<Texture2D>($"Fire/{tex}"));
                if (m.HasProperty("_Intensity")) m.SetFloat("_Intensity", intensity);
                return m;
            }
            flameBody = M(add, "FlameHollow_16x4", 14f);
            flameTongue = M(add, "Flame02_16x4", 18f);
            ember = new Material(add) { name = "Ember" };
            ember.SetTexture("_MainTex", Atmosphere.SoftDot());
            ember.SetFloat("_Intensity", 40f);
            smoke = M(alpha, "SmokeBurst_5x5", 1f);
            smoke.SetColor("_Tint", new Color(0.08f, 0.075f, 0.07f, 0.95f));
        }
    }
}
