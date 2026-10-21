using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace SmokeRescue.Robots
{
    /// <summary>
    /// What water from the nozzle looks like. Visual only: the fire simulation gets the water through FireSim.ScheduleWater.
    /// A narrow-fog stream (config/actuators.json nozzle: half_angle_deg cone, range_m throw): droplets leave the gripper
    /// inside the cone, tilted LaunchUpDeg up, at a speed derived from the throw and nozzle height. Alpha-blended,
    /// unlit streaks mark the stream; swept raycasts produce reflected splashes and small mist puffs on impact.
    /// Step advances every system on simulation time, including drops still in flight after the nozzle closes.
    /// </summary>
    public sealed class WaterSprayVisual
    {
        // A slight upward pitch clears the wrist. The speed allowance offsets the modest droplet drag below.
        const float LaunchUpDeg = 5f, DragMargin = 1.1f;
        // At 2200 visual drops/s this holds the full flight, with room for the high edge of the fog cone.
        const int MaxDrops = 3200;
        // Bound physics queries even at capture steps of 1/15 s. Deferred drops retain their unchecked travel.
        const int MaxRaycastsPerStep = 384;
        // A 2 cm floor skin also catches drops when the visual floor has no collider.
        const float FloorY = .02f;
        // Start secondary particles 8 mm off the hit so their cards do not intersect the struck face.
        const float ImpactOffset = .008f;
        readonly GameObject root;
        readonly ParticleSystem drops, mist, splash, impactMist;
        readonly float halfAngleDeg, rangeM;
        readonly int collisionMask;
        readonly ParticleSystem.Particle[] particles = new ParticleSystem.Particle[MaxDrops];
        Dictionary<uint, Vector3> previous = new Dictionary<uint, Vector3>(MaxDrops);
        Dictionary<uint, Vector3> next = new Dictionary<uint, Vector3>(MaxDrops);
        readonly uint impactSeed;
        uint impactSequence;
        int sweepStart;
        bool flowing;

        public WaterSprayVisual(Transform parent, float halfAngleDeg, float rangeM, int seed)
        {
            this.halfAngleDeg = halfAngleDeg; this.rangeM = rangeM;
            impactSeed = unchecked((uint)seed);
            int robots = LayerMask.NameToLayer("Robots");
            collisionMask = robots >= 0 ? ~(1 << robots) : ~0;
            root = new GameObject("WaterSpray");
            root.transform.SetParent(parent, false);
            var dot = SoftDot();
            // The project's alpha-blended particle shader (the smoke uses it): texture alpha shapes each particle and the
            // colour-over-lifetime fades it. HDRP/Lit ignored both and drew hard-edged cards (first capture, 2026-10-22).
            // Blue-grey and stronger coverage stay visible on black smoke and attenuate bright orange fire.
            // The shader uses ordinary alpha blending; none of these particles adds light.
            var dropMat = Particle("WaterDrops", dot, new Color(.42f, .58f, .68f, .85f));
            var cloud = Resources.Load<Texture2D>("Fire/SmokeCloud_8x8");
            var mistMat = Particle("WaterMist", cloud ?? dot, new Color(.55f, .60f, .64f, .13f));
            drops = System(root.transform, "Drops", dropMat, seed, MaxDrops);
            mist = System(root.transform, "Mist", mistMat, seed + 1, 400);
            // Siblings are simulated separately so recursive simulation cannot advance impact effects twice.
            splash = System(root.transform, "Splash", dropMat, seed + 2, 1500);
            impactMist = System(root.transform, "ImpactMist", mistMat, seed + 3, 300);

            var main = drops.main;
            // Each visible streak represents many physical drops, wide enough to survive the 640 px sensor frame.
            main.startSize = new ParticleSystem.MinMaxCurve(.025f, .045f);
            main.gravityModifier = 1f;
            // More streaks in the 12 mm nozzle core; the configured cone spreads them apart with distance.
            var em = drops.emission; em.rateOverTime = 2200f;
            var sh = drops.shape; sh.shapeType = ParticleSystemShapeType.Cone; sh.angle = halfAngleDeg; sh.radius = .012f; sh.radiusThickness = 1f;
            // Break the stream into smaller drops downstream. A short fade-in exposes the near-nozzle core.
            var size = drops.sizeOverLifetime; size.enabled = true; size.size = new ParticleSystem.MinMaxCurve(1f, AnimationCurve.Linear(0f, 1f, 1f, .4f));
            Fade(drops, 1f, .65f, .015f);
            var r = drops.GetComponent<ParticleSystemRenderer>();
            // About 35 ms of travel per streak keeps individual drops readable without drawing a solid ribbon.
            r.renderMode = ParticleSystemRenderMode.Stretch; r.velocityScale = .035f; r.lengthScale = 2f;
            var dropDrag = drops.limitVelocityOverLifetime;
            dropDrag.enabled = true; dropDrag.drag = .18f; dropDrag.multiplyDragByParticleSize = false;
            // Use drag alone. The module's default speed limit must not override the calculated throw speed.
            dropDrag.dampen = 0f;
            // Unity's world collisions do not report impacts during editor Simulate calls. Step owns impacts.
            var col = drops.collision; col.enabled = false;
            var sub = drops.subEmitters; sub.enabled = false;

            var sm = splash.main;
            sm.startLifetime = new ParticleSystem.MinMaxCurve(.15f, .35f); sm.startSpeed = new ParticleSystem.MinMaxCurve(.6f, 1.8f);
            sm.startSize = new ParticleSystem.MinMaxCurve(.012f, .028f); sm.gravityModifier = 1f;
            var se = splash.emission; se.enabled = false;
            var ss = splash.shape; ss.enabled = false;
            Fade(splash, 1f, .25f, 0f);
            var sr = splash.GetComponent<ParticleSystemRenderer>(); sr.renderMode = ParticleSystemRenderMode.Stretch; sr.velocityScale = .03f; sr.lengthScale = 1.2f;

            var mm = mist.main;
            mm.startLifetime = new ParticleSystem.MinMaxCurve(.7f, 1.3f); mm.startSize = new ParticleSystem.MinMaxCurve(.08f, .16f);
            mm.gravityModifier = .15f; mm.startRotation = new ParticleSystem.MinMaxCurve(0f, 2f * Mathf.PI);
            // Fine airborne spray supplements the stream; most visible mist now appears at actual impacts.
            var me = mist.emission; me.rateOverTime = 65f;
            var msh = mist.shape; msh.shapeType = ParticleSystemShapeType.Cone; msh.angle = halfAngleDeg * 1.4f; msh.radius = .02f;
            Fade(mist, .8f, .1f);
            var mg = mist.sizeOverLifetime; mg.enabled = true; mg.size = new ParticleSystem.MinMaxCurve(1f, AnimationCurve.Linear(0f, 1f, 1f, 2.5f));
            var drag = mist.limitVelocityOverLifetime; drag.enabled = true; drag.drag = 2.2f; drag.multiplyDragByParticleSize = false;
            var nz = mist.noise; nz.enabled = true; nz.strength = .35f; nz.frequency = 1.1f; nz.scrollSpeed = .4f;
            CloudFrames(mist, cloud != null);

            // A small, short-lived puff marks the wet impact rather than hiding it in a large smoke-like cloud.
            var im = impactMist.main;
            im.startLifetime = new ParticleSystem.MinMaxCurve(.18f, .32f);
            im.startSize = new ParticleSystem.MinMaxCurve(.045f, .09f); im.gravityModifier = .1f;
            var ie = impactMist.emission; ie.enabled = false;
            var ish = impactMist.shape; ish.enabled = false;
            var ig = impactMist.sizeOverLifetime; ig.enabled = true;
            ig.size = new ParticleSystem.MinMaxCurve(1f, AnimationCurve.Linear(0f, 1f, 1f, 2f));
            Fade(impactMist, .65f, .15f, 0f);
            CloudFrames(impactMist, cloud != null);

            // Start once, then let Step own time, as FireVisuals does. Closing the nozzle disables emission.
            // Keep playback paused so Simulate can advance drops and impact effects after End.
            foreach (var ps in new[] { drops, mist, splash, impactMist })
            {
                ps.Play(false);
                ps.Pause(false);
                var emission = ps.emission; emission.enabled = false;
            }
        }

        /// <summary>Open the nozzle at the gripper, pointing along headingRad (0 = +z, toward +x positive), level with the
        /// floor plus LaunchUpDeg. The launch speed is set from the nozzle's height above the floor (house floor at y = 0).</summary>
        public void Begin(Vector3 nozzle, float headingRad)
        {
            Aim(nozzle, headingRad);
            float h = Mathf.Max(.2f, nozzle.y), up = LaunchUpDeg * Mathf.Deg2Rad, cos = Mathf.Cos(up);
            float v = Mathf.Sqrt(-Physics.gravity.y * rangeM * rangeM / (2f * cos * cos * (h + rangeM * Mathf.Tan(up)))) * DragMargin;
            var main = drops.main; main.startSpeed = new ParticleSystem.MinMaxCurve(.9f * v, 1.05f * v);
            // Use the fastest drop at the upper cone edge to bound floor arrival. Add 150 ms for coarse capture
            // steps and drag. Droplets normally die at impacts, rather than fading out before they can land.
            float gravity = -Physics.gravity.y;
            float upwardSpeed = 1.05f * v * Mathf.Sin(up + halfAngleDeg * Mathf.Deg2Rad);
            main.startLifetime = (upwardSpeed + Mathf.Sqrt(upwardSpeed * upwardSpeed + 2f * gravity * h)) / gravity + .15f;
            var mm = mist.main; mm.startSpeed = new ParticleSystem.MinMaxCurve(.35f * v, .6f * v);
            var emission = drops.emission; emission.enabled = true;
            var mistEmission = mist.emission; mistEmission.enabled = true;
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
            var emission = drops.emission; emission.enabled = false;
            var mistEmission = mist.emission; mistEmission.enabled = false;
        }

        public bool Visible => flowing || drops.particleCount > 0 || mist.particleCount > 0 || splash.particleCount > 0 || impactMist.particleCount > 0;
        /// <summary>Live droplet, mist and splash particles (diagnostics and evidence captures).</summary>
        public (int drops, int mist, int splash) Counts => (drops.particleCount, mist.particleCount + impactMist.particleCount, splash.particleCount);

        /// <summary>Advance on simulation time; call every tick while Visible.</summary>
        public void Step(float dt)
        {
            if (!Visible || !(dt > 0) || !float.IsFinite(dt)) return;
            // Snapshot exact positions before Simulate. Particle array indices can change when particles die,
            // so match by the particle's fixed random seed, never by its array index.
            int count = drops.GetParticles(particles);
            for (int i = 0; i < count; i++)
                if (!previous.ContainsKey(particles[i].randomSeed)) previous.Add(particles[i].randomSeed, particles[i].position);
            drops.Simulate(dt, false, false, false);
            mist.Simulate(dt, false, false, false);
            splash.Simulate(dt, false, false, false);
            impactMist.Simulate(dt, false, false, false);
            DetectImpacts(dt);
        }

        void DetectImpacts(float dt)
        {
            int count = drops.GetParticles(particles);
            next.Clear();
            bool killed = false;
            int start = count > 0 ? sweepStart % count : 0;
            for (int j = 0; j < count; j++)
            {
                int i = (start + j) % count;
                var p = particles[i];
                if (p.remainingLifetime <= 0f) continue;
                if (!previous.TryGetValue(p.randomSeed, out var from))
                {
                    // Born within this Simulate call. Backtrack only its age, not a whole step, to the nozzle.
                    // Gravity gives the birth segment; drag over this first partial step is a small approximation.
                    float age = Mathf.Min(dt, p.startLifetime - p.remainingLifetime);
                    from = p.position - p.velocity * age + Physics.gravity * (.5f * age * age);
                }
                bool checkedWorld = j < MaxRaycastsPerStep;
                // Defer the floor too when the ray budget is spent, otherwise a skipped wall could be crossed
                // and the same drop could splash on the floor behind it before the wall is checked.
                if (checkedWorld && Impact(from, p.position, out var point, out var normal))
                {
                    EmitImpact(point, normal, p.velocity, p.randomSeed);
                    // SetParticles documents a negative lifetime as removal.
                    p.remainingLifetime = -1f; particles[i] = p; killed = true;
                }
                else next[p.randomSeed] = checkedWorld ? p.position : from;
            }
            if (killed) drops.SetParticles(particles, count);
            // Rotate the query order. Keep deferred starts, and discard dead seeds, without allocating each step.
            sweepStart = count > 0 ? (start + Mathf.Min(count, MaxRaycastsPerStep)) % count : 0;
            var swap = previous; previous = next; next = swap;
        }

        bool Impact(Vector3 from, Vector3 to, out Vector3 point, out Vector3 normal)
        {
            var travel = to - from;
            float distance = travel.magnitude;
            bool hit = false;
            point = default; normal = default;
            // Ignore trigger volumes such as sensor regions, as well as the robot's own arm and body.
            if (distance > 0f && Physics.Raycast(from, travel / distance, out var info,
                    distance, collisionMask, QueryTriggerInteraction.Ignore))
            {
                point = info.point; normal = info.normal; distance = info.distance; hit = true;
            }
            if (to.y <= FloorY)
            {
                float fraction = from.y > FloorY ? (from.y - FloorY) / (from.y - to.y) : 0f;
                var floor = Vector3.Lerp(from, to, fraction); floor.y = FloorY;
                if (!hit || (floor - from).sqrMagnitude < distance * distance)
                { point = floor; normal = Vector3.up; hit = true; }
            }
            return hit;
        }

        void EmitImpact(Vector3 point, Vector3 normal, Vector3 incoming, uint dropSeed)
        {
            // Mix with the FNV prime before xorshift so sequential drop IDs and impact IDs do not cancel out.
            // Local state makes impacts repeat without touching UnityEngine.Random or wall-clock time.
            uint state = unchecked(((dropSeed ^ impactSeed) * 16777619u) ^ ++impactSequence);
            if (state == 0) state = 1;
            var reflected = Vector3.Reflect(incoming.normalized, normal);
            // Three secondary drops per representative impact limit particle work for the 1 L/s spray.
            for (int i = 0; i < 3; i++)
            {
                // A random hemisphere adds breakup. Reject any component pointing into the struck face.
                float z = 2f * Random01(ref state) - 1f, azimuth = 2f * Mathf.PI * Random01(ref state);
                float radius = Mathf.Sqrt(1f - z * z);
                var spread = new Vector3(radius * Mathf.Cos(azimuth), z, radius * Mathf.Sin(azimuth));
                if (Vector3.Dot(spread, normal) < 0f) spread = -spread;
                // The normal bias lifts grazing impacts clear of the face instead of skimming into it.
                var direction = (reflected + spread + normal * .35f).normalized;
                var emit = new ParticleSystem.EmitParams {
                    position = point + normal * ImpactOffset,
                    velocity = direction * Mathf.Lerp(.5f, 2f, Random01(ref state)),
                    randomSeed = state, applyShapeToPosition = false
                };
                splash.Emit(emit, 1);
            }
            var puff = new ParticleSystem.EmitParams {
                position = point + normal * ImpactOffset,
                velocity = normal * .2f, randomSeed = state, applyShapeToPosition = false
            }; // The fine impact mist drifts off the face at 20 cm/s rather than following the fast stream.
            impactMist.Emit(puff, 1);
        }

        static float Random01(ref uint state)
        {
            // xorshift32 shifts, then use 24 bits so conversion to float cannot round up to 1.
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            return (state & 0x00ffffffu) / 16777216f;
        }

        /// <summary>Alpha ramps in for fadeIn of the life, holds, then fades to zero.</summary>
        static void Fade(ParticleSystem ps, float peak, float holdUntil, float fadeIn = .1f)
        {
            var g = new Gradient();
            var alpha = fadeIn > 0f
                ? new[] { new GradientAlphaKey(0f, 0f), new GradientAlphaKey(peak, fadeIn), new GradientAlphaKey(peak, holdUntil), new GradientAlphaKey(0f, 1f) }
                : new[] { new GradientAlphaKey(peak, 0f), new GradientAlphaKey(peak, holdUntil), new GradientAlphaKey(0f, 1f) };
            g.SetKeys(new[] { new GradientColorKey(Color.white, 0f), new GradientColorKey(Color.white, 1f) }, alpha);
            var col = ps.colorOverLifetime; col.enabled = true; col.color = g;
        }

        static void CloudFrames(ParticleSystem ps, bool atlas)
        {
            // SmokeCloud is an 8 by 8 atlas. The single-dot fallback must use its whole texture.
            var tsa = ps.textureSheetAnimation; tsa.enabled = atlas;
            if (!atlas) return;
            tsa.numTilesX = 8; tsa.numTilesY = 8;
            tsa.timeMode = ParticleSystemAnimationTimeMode.Lifetime; tsa.startFrame = new ParticleSystem.MinMaxCurve(0f, 63f);
            tsa.frameOverTime = new ParticleSystem.MinMaxCurve(0f); tsa.cycleCount = 1;
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
