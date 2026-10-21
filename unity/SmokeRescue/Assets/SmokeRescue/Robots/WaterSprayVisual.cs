using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

namespace SmokeRescue.Robots
{
    /// <summary>
    /// What water from the nozzle looks like. Visual only: the fire simulation gets the water through FireSim.ScheduleWater.
    /// A narrow-fog stream (config/actuators.json nozzle: half_angle_deg cone, range_m throw): droplets leave the gripper
    /// inside the cone, tilted LaunchUpDeg up, at the speed whose drag-free arc reaches the floor range_m away (plus
    /// DragMargin for air drag), fall under gravity, and are drawn as short streaks; a fine mist drifts and spreads around
    /// the stream; droplets that hit a wall, the floor or furniture die there and throw small splashes. The materials are
    /// HDRP/Lit transparent, so the torch and the fire light the water: the earlier single unlit strip rendered as a
    /// glowing pale-blue ribbon under the dark room's exposure (user, 2026-10-22). Particles advance on simulation time
    /// (Step), as FireVisuals does, and keep advancing after the nozzle closes so drops in flight land.
    /// </summary>
    public sealed class WaterSprayVisual
    {
        const float LaunchUpDeg = 5f, DragMargin = 1.1f;
        readonly GameObject root;
        readonly ParticleSystem drops, mist, splash;
        readonly float halfAngleDeg, rangeM;
        bool flowing;

        public WaterSprayVisual(Transform parent, float halfAngleDeg, float rangeM, int seed)
        {
            this.halfAngleDeg = halfAngleDeg; this.rangeM = rangeM;
            root = new GameObject("WaterSpray");
            root.transform.SetParent(parent, false);
            var dot = SoftDot();
            // The project's alpha-blended particle shader (the smoke uses it): texture alpha shapes each particle and the
            // colour-over-lifetime fades it. HDRP/Lit ignored both and drew hard-edged cards (first capture, 2026-10-22).
            var dropMat = Particle("WaterDrops", dot, new Color(.62f, .70f, .78f, .55f));
            var mistMat = Particle("WaterMist", Resources.Load<Texture2D>("Fire/SmokeCloud_8x8") ?? dot, new Color(.55f, .58f, .62f, .22f));
            drops = System(root.transform, "Drops", dropMat, seed, 2500);
            mist = System(root.transform, "Mist", mistMat, seed + 1, 400);
            splash = System(drops.transform, "Splash", dropMat, seed + 2, 1500);

            var main = drops.main;
            main.startLifetime = new ParticleSystem.MinMaxCurve(.7f, 1.0f);
            main.startSize = new ParticleSystem.MinMaxCurve(.018f, .04f);
            main.gravityModifier = 1f;
            var em = drops.emission; em.rateOverTime = 1500f;
            var sh = drops.shape; sh.shapeType = ParticleSystemShapeType.Cone; sh.angle = halfAngleDeg; sh.radius = .012f; sh.radiusThickness = 1f;
            var size = drops.sizeOverLifetime; size.enabled = true; size.size = new ParticleSystem.MinMaxCurve(1f, AnimationCurve.Linear(0f, 1f, 1f, .5f));
            Fade(drops, .9f, .6f);
            var r = drops.GetComponent<ParticleSystemRenderer>();
            r.renderMode = ParticleSystemRenderMode.Stretch; r.velocityScale = .025f; r.lengthScale = 1.5f;
            var col = drops.collision;
            col.enabled = true; col.type = ParticleSystemCollisionType.World; col.mode = ParticleSystemCollisionMode.Collision3D;
            col.quality = ParticleSystemCollisionQuality.Medium; col.lifetimeLoss = 1f; col.bounce = .05f; col.dampen = .8f;
            int robots = LayerMask.NameToLayer("Robots");
            col.collidesWith = robots >= 0 ? ~(1 << robots) : ~0;   // not the robot's own arm and body at the nozzle
            var sub = drops.subEmitters; sub.enabled = true;
            sub.AddSubEmitter(splash, ParticleSystemSubEmitterType.Collision, ParticleSystemSubEmitterProperties.InheritNothing, .35f);

            var sm = splash.main;
            sm.startLifetime = new ParticleSystem.MinMaxCurve(.15f, .35f); sm.startSpeed = new ParticleSystem.MinMaxCurve(.6f, 1.8f);
            sm.startSize = new ParticleSystem.MinMaxCurve(.012f, .028f); sm.gravityModifier = 1f;
            var se = splash.emission; se.rateOverTime = 0f; se.SetBursts(new[] { new ParticleSystem.Burst(0f, 3, 6) });
            var ss = splash.shape; ss.shapeType = ParticleSystemShapeType.Sphere; ss.radius = .03f;   // broken up on the surface, every way; those that head into it are hidden by it
            var sr = splash.GetComponent<ParticleSystemRenderer>(); sr.renderMode = ParticleSystemRenderMode.Stretch; sr.velocityScale = .03f; sr.lengthScale = 1.2f;

            var mm = mist.main;
            mm.startLifetime = new ParticleSystem.MinMaxCurve(.7f, 1.3f); mm.startSize = new ParticleSystem.MinMaxCurve(.08f, .16f);
            mm.gravityModifier = .15f; mm.startRotation = new ParticleSystem.MinMaxCurve(0f, 2f * Mathf.PI);
            var me = mist.emission; me.rateOverTime = 140f;
            var msh = mist.shape; msh.shapeType = ParticleSystemShapeType.Cone; msh.angle = halfAngleDeg * 1.4f; msh.radius = .02f;
            Fade(mist, .8f, .1f);
            var mg = mist.sizeOverLifetime; mg.enabled = true; mg.size = new ParticleSystem.MinMaxCurve(1f, AnimationCurve.Linear(0f, 1f, 1f, 4.5f));
            var drag = mist.limitVelocityOverLifetime; drag.enabled = true; drag.drag = 2.2f; drag.multiplyDragByParticleSize = false;
            var nz = mist.noise; nz.enabled = true; nz.strength = .35f; nz.frequency = 1.1f; nz.scrollSpeed = .4f;
            var tsa = mist.textureSheetAnimation; tsa.enabled = true; tsa.numTilesX = 8; tsa.numTilesY = 8;
            tsa.timeMode = ParticleSystemAnimationTimeMode.Lifetime; tsa.startFrame = new ParticleSystem.MinMaxCurve(0f, 63f);
            tsa.frameOverTime = new ParticleSystem.MinMaxCurve(0f); tsa.cycleCount = 1;
        }

        /// <summary>Open the nozzle at the gripper, pointing along headingRad (0 = +z, toward +x positive), level with the
        /// floor plus LaunchUpDeg. The launch speed is set from the nozzle's height above the floor (house floor at y = 0).</summary>
        public void Begin(Vector3 nozzle, float headingRad)
        {
            Aim(nozzle, headingRad);
            float h = Mathf.Max(.2f, nozzle.y), up = LaunchUpDeg * Mathf.Deg2Rad, cos = Mathf.Cos(up);
            float v = Mathf.Sqrt(-Physics.gravity.y * rangeM * rangeM / (2f * cos * cos * (h + rangeM * Mathf.Tan(up)))) * DragMargin;
            var main = drops.main; main.startSpeed = new ParticleSystem.MinMaxCurve(.9f * v, 1.05f * v);
            var mm = mist.main; mm.startSpeed = new ParticleSystem.MinMaxCurve(.35f * v, .6f * v);
            if (!flowing) { drops.Play(true); mist.Play(true); }
            flowing = true;
        }

        public void Aim(Vector3 nozzle, float headingRad)
        {
            root.transform.SetPositionAndRotation(nozzle, Quaternion.Euler(-LaunchUpDeg, headingRad * Mathf.Rad2Deg, 0f));
        }

        /// <summary>Close the nozzle: no new water, the drops in flight finish their arcs.</summary>
        public void End()
        {
            if (!flowing) return;
            flowing = false;
            drops.Stop(true, ParticleSystemStopBehavior.StopEmitting); mist.Stop(true, ParticleSystemStopBehavior.StopEmitting);
        }

        public bool Visible => flowing || drops.particleCount > 0 || mist.particleCount > 0 || splash.particleCount > 0;
        /// <summary>Live droplet, mist and splash particles (diagnostics and evidence captures).</summary>
        public (int drops, int mist, int splash) Counts => (drops.particleCount, mist.particleCount, splash.particleCount);

        /// <summary>Advance on simulation time; call every tick while Visible.</summary>
        public void Step(float dt)
        {
            if (!Visible || !(dt > 0)) return;
            drops.Simulate(dt, true, false, false);
            mist.Simulate(dt, true, false, false);
        }

        /// <summary>Alpha ramps in over the first tenth of life, holds, and fades to zero (no pop when a particle dies).</summary>
        static void Fade(ParticleSystem ps, float peak, float holdUntil)
        {
            var g = new Gradient();
            g.SetKeys(new[] { new GradientColorKey(Color.white, 0f), new GradientColorKey(Color.white, 1f) },
                      new[] { new GradientAlphaKey(0f, 0f), new GradientAlphaKey(peak, .1f), new GradientAlphaKey(peak, holdUntil), new GradientAlphaKey(0f, 1f) });
            var col = ps.colorOverLifetime; col.enabled = true; col.color = g;
        }

        static ParticleSystem System(Transform parent, string name, Material mat, int seed, int maxN)
        {
            var go = new GameObject(name); go.transform.SetParent(parent, false);
            var ps = go.AddComponent<ParticleSystem>();
            ps.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
            ps.useAutoRandomSeed = false; ps.randomSeed = (uint)seed;
            var main = ps.main;
            main.playOnAwake = false; main.loop = true; main.duration = 5f; main.maxParticles = maxN;
            main.simulationSpace = ParticleSystemSimulationSpace.World;
            var r = go.GetComponent<ParticleSystemRenderer>();
            r.sharedMaterial = mat; r.shadowCastingMode = ShadowCastingMode.Off; r.receiveShadows = true; r.maxParticleSize = 2f;
            return ps;
        }

        static Material Particle(string name, Texture2D tex, Color tint)
        {
            var shader = Shader.Find("SmokeRescue/ParticleSmoke");
            if (shader == null) throw new System.InvalidOperationException("SmokeRescue/ParticleSmoke shader missing.");
            var m = new Material(shader) { name = name, renderQueue = (int)RenderQueue.Transparent };
            m.SetTexture("_MainTex", tex); m.SetColor("_Tint", tint);
            return m;
        }

        static Material Lit(string name, Texture2D tex, Color color, float smoothness)
        {
            var shader = Shader.Find("HDRP/Lit");
            if (shader == null) throw new System.InvalidOperationException("HDRP/Lit shader missing.");
            var m = new Material(shader) { name = name };
            m.SetFloat("_SurfaceType", 1f);            // transparent
            m.SetFloat("_BlendMode", 0f);              // alpha
            m.SetFloat("_DoubleSidedEnable", 1f);
            m.SetFloat("_EnableFogOnTransparent", 1f);
            m.SetColor("_BaseColor", color);
            m.SetTexture("_BaseColorMap", tex);
            m.SetFloat("_Smoothness", smoothness);
            m.SetFloat("_Metallic", 0f);
            HDMaterial.ValidateMaterial(m);
            return m;
        }

        static Texture2D softDot;
        static Texture2D SoftDot()
        {
            if (softDot != null) return softDot;
            const int n = 64; var px = new Color[n * n];
            for (int y = 0; y < n; y++) for (int x = 0; x < n; x++)
            {
                float u = x / (n - 1f) * 2f - 1f, v = y / (n - 1f) * 2f - 1f, a = Mathf.Clamp01(1f - Mathf.Sqrt(u * u + v * v));
                px[y * n + x] = new Color(1f, 1f, 1f, a * a * (3f - 2f * a));
            }
            softDot = new Texture2D(n, n, TextureFormat.RGBA32, true) { wrapMode = TextureWrapMode.Clamp, name = "WaterDot" };
            softDot.SetPixels(px); softDot.Apply();
            return softDot;
        }
    }
}
