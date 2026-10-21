using System.Collections.Generic;
using NUnit.Framework;
using SmokeRescue.Editor;
using SmokeRescue.Robots;
using UnityEngine;
using UnityEngine.SceneManagement;
using Object = UnityEngine.Object;

namespace SmokeRescue.Tests.EditMode
{
    public class WaterSprayVisualTests
    {
        GameObject root;
        readonly Vector3 origin = new Vector3(1000f, 10f, 1000f);

        [SetUp] public void SetUp()
        {
            ProjectSetup.EnsureLayers();
            root = new GameObject("Water spray tests");
            Assert.That(root.scene.GetPhysicsScene(), Is.EqualTo(Physics.defaultPhysicsScene),
                "Physics.Raycast queries the default physics scene, which must contain the fixture");
        }

        [TearDown] public void TearDown()
        {
            var materials = new HashSet<Material>();
            foreach (var renderer in root.GetComponentsInChildren<ParticleSystemRenderer>())
                materials.Add(renderer.sharedMaterial);
            Object.DestroyImmediate(root);
            foreach (var material in materials) Object.DestroyImmediate(material);
            Physics.SyncTransforms();
        }

        WaterSprayVisual Water(out ParticleSystem drops)
        {
            var holder = new GameObject("Nozzle");
            holder.transform.SetParent(root.transform);
            var water = new WaterSprayVisual(holder.transform, 12f, 5f, 7);
            water.Begin(origin, 0f);
            water.End();
            drops = holder.transform.Find("WaterSpray/Drops").GetComponent<ParticleSystem>();
            return water;
        }

        BoxCollider Box(int layer, bool trigger = false)
        {
            var go = new GameObject("Impact target");
            go.transform.SetParent(root.transform);
            go.transform.position = origin + Vector3.forward;
            go.layer = layer;
            var box = go.AddComponent<BoxCollider>();
            box.size = new Vector3(2f, 2f, .1f);
            box.isTrigger = trigger;
            Physics.SyncTransforms();
            Assert.That(Physics.Raycast(origin, Vector3.forward, out var hit, 2f, 1 << layer,
                QueryTriggerInteraction.Collide), Is.True, "the target must be visible to physics before stepping");
            Assert.That(hit.collider, Is.EqualTo(box));
            return box;
        }

        static void Drop(ParticleSystem drops, Vector3 position, Vector3 velocity, uint seed = 123u)
        {
            int before = drops.particleCount;
            drops.Emit(new ParticleSystem.EmitParams {
                position = position, velocity = velocity, startLifetime = 2f,
                startSize = .03f, randomSeed = seed, applyShapeToPosition = false
            }, 1);
            Assert.That(drops.particleCount, Is.EqualTo(before + 1),
                "Emit must create the injected drop before Step can test its impact");
        }

        static int LiveDrops(ParticleSystem drops)
        {
            var particles = new ParticleSystem.Particle[drops.main.maxParticles];
            int n = drops.GetParticles(particles), alive = 0;
            for (int i = 0; i < n; i++) if (particles[i].remainingLifetime > 0f) alive++;
            return alive;
        }

        [Test] public void BeginAndStepEmitDropsAndMistAcrossSuccessiveEditModeSteps()
        {
            var water = Water(out var drops);
            water.Begin(origin, 0f);
            water.Step(.1f);
            Assert.That(LiveDrops(drops), Is.GreaterThan(0), "the first Step must emit drops");
            Assert.That(water.Counts.mist, Is.GreaterThan(0), "the first Step must emit nozzle mist");
            int firstCount = LiveDrops(drops);
            water.Step(.1f);
            Assert.That(LiveDrops(drops), Is.GreaterThan(firstCount),
                "the next Step must continue emission from the paused simulation state");
        }

        [Test] public void ClosingNozzlePreservesFlightAndReopeningKeepsExistingDrops()
        {
            var water = Water(out var drops);
            water.Begin(origin, 0f);
            water.Step(.1f);
            var before = new ParticleSystem.Particle[drops.main.maxParticles];
            int count = drops.GetParticles(before);
            Assert.That(count, Is.GreaterThan(0), "close the nozzle with real emitted drops in flight");
            water.End();
            Assert.That(LiveDrops(drops), Is.EqualTo(count), "End must preserve existing drops");
            water.Step(.05f);
            var after = new ParticleSystem.Particle[drops.main.maxParticles];
            Assert.That(drops.GetParticles(after), Is.EqualTo(count),
                "a closed nozzle must advance its drops without emitting new ones");
            var bySeed = new Dictionary<uint, ParticleSystem.Particle>();
            for (int i = 0; i < count; i++) bySeed.Add(before[i].randomSeed, before[i]);
            for (int i = 0; i < count; i++)
            {
                Assert.That(bySeed.TryGetValue(after[i].randomSeed, out var old), Is.True,
                    "closing must retain each drop's identity");
                Assert.That(after[i].remainingLifetime, Is.EqualTo(old.remainingLifetime - .05f).Within(.0001f));
                Assert.That(Vector3.Distance(after[i].position, old.position), Is.GreaterThan(0f));
            }
            water.Begin(origin, 0f);
            Assert.That(LiveDrops(drops), Is.EqualTo(count), "reopening must preserve drops in flight");
            water.Step(.05f);
            Assert.That(LiveDrops(drops), Is.GreaterThan(count), "reopening must resume emission");
        }

        [Test] public void NearNozzleDropsStayNarrowAndShortAtTheConfiguredThrowSpeed()
        {
            var water = Water(out var drops);
            // Use the capture's 1 m nozzle height, rather than the elevated impact-test fixture.
            water.Begin(new Vector3(origin.x, 1f, origin.z), 0f);
            water.Step(.1f);
            var particles = new ParticleSystem.Particle[drops.main.maxParticles];
            int n = drops.GetParticles(particles);
            Assert.That(n, Is.GreaterThan(0));
            var renderer = drops.GetComponent<ParticleSystemRenderer>();
            Assert.That(renderer.renderMode, Is.EqualTo(ParticleSystemRenderMode.Stretch));
            for (int i = 0; i < n; i++)
            {
                float width = particles[i].GetCurrentSize(drops);
                // Bound the unprojected card, including both size and speed contributions to stretch.
                float length = width * renderer.lengthScale + particles[i].velocity.magnitude * renderer.velocityScale;
                Assert.That(width, Is.LessThan(.024f), "a drop must be narrower than the nozzle diameter");
                Assert.That(length, Is.LessThan(.15f), "near-camera drops must not span tens of centimetres");
            }
            float alpha = renderer.sharedMaterial.GetColor("_Tint").a;
            Assert.That(1f - Mathf.Pow(1f - alpha, 4f), Is.LessThan(.9f),
                "four overlapping dot centres must still transmit some of the fire behind them");
            water.End();
        }

        [Test] public void SweptDropDiesAtThinBoxAndEmitsReflectedSplashAfterNozzleCloses()
        {
            // Crossing the whole 10 cm box in one step catches endpoint-only collision checks.
            Box(LayerMask.NameToLayer("Props"));
            var water = Water(out var drops);
            Drop(drops, origin, Vector3.forward * 10f);
            water.Step(.12f);
            Assert.That(LiveDrops(drops), Is.Zero);
            Assert.That(water.Counts.splash, Is.GreaterThan(0));
            Assert.That(water.Counts.mist, Is.GreaterThan(0), "an impact also emits a mist puff");
            var splash = drops.transform.parent.Find("Splash").GetComponent<ParticleSystem>();
            var particles = new ParticleSystem.Particle[splash.main.maxParticles];
            int n = splash.GetParticles(particles);
            for (int i = 0; i < n; i++)
            {
                Assert.That(particles[i].position.z, Is.LessThan(origin.z + .95f));
                Assert.That(particles[i].velocity.z, Is.LessThan(0f), "splash leaves the front of the box");
                Assert.That(particles[i].velocity.magnitude, Is.InRange(.5f, 2f));
            }
        }

        [Test] public void FallingDropHitsFloorWithoutACollider()
        {
            var water = Water(out var drops);
            Drop(drops, new Vector3(origin.x, .1f, origin.z), Vector3.down * 2f);
            water.Step(.1f);
            Assert.That(LiveDrops(drops), Is.Zero);
            Assert.That(water.Counts.splash, Is.GreaterThan(0));
        }

        [Test] public void DropsBornDuringSimulateAreCheckedOnTheirFirstStep()
        {
            Box(LayerMask.NameToLayer("Props"));
            var water = Water(out var drops);
            // Keep this below the ray budget to isolate the first-step birth segment from deferred queries.
            var emission = drops.emission; emission.rateOverTime = 100f;
            water.Begin(origin, 0f);
            water.Step(.4f);
            Assert.That(water.Counts.splash, Is.GreaterThan(0));
            var particles = new ParticleSystem.Particle[drops.main.maxParticles];
            int n = drops.GetParticles(particles);
            for (int i = 0; i < n; i++)
                if (particles[i].remainingLifetime > 0f)
                    Assert.That(particles[i].position.z, Is.LessThan(origin.z + .95f));
        }

        [TestCase(true)]
        [TestCase(false)]
        public void RobotCollidersAndTriggersDoNotStopWater(bool robot)
        {
            Box(LayerMask.NameToLayer(robot ? "Robots" : "Props"), trigger: !robot);
            var water = Water(out var drops);
            Drop(drops, origin, Vector3.forward * 10f);
            water.Step(.12f);
            Assert.That(LiveDrops(drops), Is.EqualTo(1));
            Assert.That(water.Counts.splash, Is.Zero);
        }

        [Test] public void ImpactEffectsRepeatForTheSameSeedAndSimulationSteps()
        {
            Box(LayerMask.NameToLayer("Props"));
            var first = Water(out var a);
            var second = Water(out var b);
            Drop(a, origin, Vector3.forward * 10f);
            Drop(b, origin, Vector3.forward * 10f);
            first.Step(.12f); second.Step(.12f);
            first.Step(.04f); second.Step(.04f);
            Assert.That(first.Counts.splash, Is.GreaterThan(0));
            Assert.That(first.Counts, Is.EqualTo(second.Counts));
            var sa = a.transform.parent.Find("Splash").GetComponent<ParticleSystem>();
            var sb = b.transform.parent.Find("Splash").GetComponent<ParticleSystem>();
            var pa = new ParticleSystem.Particle[sa.main.maxParticles];
            var pb = new ParticleSystem.Particle[sb.main.maxParticles];
            int n = sa.GetParticles(pa);
            Assert.That(sb.GetParticles(pb), Is.EqualTo(n));
            for (int i = 0; i < n; i++)
            {
                Assert.That(pa[i].position, Is.EqualTo(pb[i].position));
                Assert.That(pa[i].velocity, Is.EqualTo(pb[i].velocity));
                Assert.That(pa[i].remainingLifetime, Is.EqualTo(pb[i].remainingLifetime));
            }
            first.Step(3f);
            Assert.That(first.Visible, Is.False, "closed nozzle effects finish on simulation time");
        }

        [Test] public void SkippedRaycastsKeepTheirTravelUntilTheNextBudgetIsAvailable()
        {
            Box(LayerMask.NameToLayer("Props"));
            var water = Water(out var drops);
            // More drops than the per-step ray budget. The second step must sweep the deferred travel,
            // even though the skipped drops have already crossed the box by then.
            for (uint i = 1; i <= 600; i++) Drop(drops, origin, Vector3.forward * 10f, i);
            water.Step(.12f);
            water.Step(.01f);
            Assert.That(LiveDrops(drops), Is.Zero);
            Assert.That(water.Counts.splash, Is.EqualTo(1800), "all 600 impacts must retain their three splash drops");
        }

        [Test] public void SplashCapacityRetainsAFullDeferredFlightOfImpacts()
        {
            Box(LayerMask.NameToLayer("Props"));
            var water = Water(out var drops);
            // Fill a whole flight, then drain the ray budget before any 150 ms splash can expire.
            int count = drops.main.maxParticles;
            for (uint i = 1; i <= count; i++) Drop(drops, origin, Vector3.forward * 10f, i);
            water.Step(.12f);
            for (int i = 0; i < 8; i++) water.Step(.001f);
            Assert.That(LiveDrops(drops), Is.Zero);
            Assert.That(water.Counts.splash, Is.EqualTo(3 * count), "the deferred impacts must not lose secondary drops");
            // Add the 770 further impacts that can arrive during a 350 ms life at 2200 drops/s.
            // Compress their arrival so even the shortest-lived existing splash must remain in the pool.
            for (uint i = 1; i <= 770; i++) Drop(drops, origin + Vector3.forward * .94f, Vector3.forward * 10f, i);
            water.Step(.002f);
            water.Step(.001f);
            water.Step(.001f);
            Assert.That(LiveDrops(drops), Is.Zero);
            Assert.That(water.Counts.splash, Is.EqualTo(3 * count + 2310),
                "a deferred flight plus continued arrivals must retain every splash drop");
        }
    }
}
