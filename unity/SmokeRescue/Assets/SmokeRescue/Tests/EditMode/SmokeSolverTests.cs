using System.Linq;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SmokeRescue.World;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    public class SmokeSolverTests
    {
        readonly List<string> temporaryFiles = new List<string>();
        string firePath, smokePath;

        [SetUp]
        public void SetUp()
        {
            var fire = JObject.Parse(File.ReadAllText(Path.Combine(FloorPlan.ConfigDir, "fire.json")));
            fire["ignitions"] = JArray.Parse("[{\"room\":\"living\",\"at\":[3.625,1.875],\"time_s\":0,\"radius_m\":0.01}]");
            // One fuel-bearing ignition. The neighbour is inert, so smoke there cannot be local production.
            fire["spread"]["neighbour_heat_per_s"] = 0;
            fire["spread"]["gas_surface_heat_per_s"] = 0;
            fire["spread"]["flashover_surface_heat_c_s"] = 0;
            // These tests are about the smoke layer, so the source burns at its full fuel rate: under the medium t-squared
            // cap it releases about 5 kW at 20 s, too little gas to fill a layer within the test.
            fire["spread"]["growth_alpha_kw_s2"] = 0;
            firePath = Temporary(fire);
            smokePath = Path.Combine(FloorPlan.ConfigDir, "smoke.json");
        }

        string Temporary(JObject json)
        {
            string path = Path.Combine(Path.GetTempPath(), "smoke-solver-" + Guid.NewGuid().ToString("N") + ".json");
            File.WriteAllText(path, json.ToString()); temporaryFiles.Add(path); return path;
        }

        [TearDown]
        public void TearDown()
        {
            foreach (string file in temporaryFiles) File.Delete(file);
            temporaryFiles.Clear();
        }

        static FloorPlan Plan(string door = null, bool neighbour = false)
        {
            var p = new FloorPlan
            {
                footprint = new float[] { 0, 0, neighbour ? 8 : 4, 4 }, wall_height = 2.8f,
                door_height = 2.1f, interior_wall_thickness = .15f, exterior_wall_thickness = .25f,
                rooms = new List<FloorPlan.Room>
                {
                    new FloorPlan.Room { id = "living", rect = new float[] { 0, 0, 4, 4 },
                        floor = "wood_oak_planks", walls = "plaster_white", ceiling = "plaster_white" }
                }
            };
            if (neighbour) p.rooms.Add(new FloorPlan.Room { id = "hallway", rect = new float[] { 4, 0, 8, 4 },
                floor = "stone_tiles_large", walls = "plaster_white", ceiling = "plaster_white" });
            if (door != null) p.doors.Add(new FloorPlan.Door { id = "opening", between = new[] { "living", neighbour ? "hallway" : "outside" },
                wall = "x", at = new float[] { 4, 2 }, width = 1, state = door });
            return p;
        }

        SmokeSolver Solver(FloorPlan p, string config = null, float headStart = 0) =>
            SmokeSolver.Load(p, firePath, config ?? smokePath, 7, headStart);

        static double RoomMass(SmokeSolver solver, string room)
        {
            double mass = 0;
            for (int z = 0; z < solver.Height; z++) for (int x = 0; x < solver.Width; x++)
            {
                var cell = solver.GetCell(x, z);
                if (cell.Room == room) mass += cell.UpperMassKg + cell.LowerMassKg;
            }
            return mass;
        }

        [Test]
        public void SealedMassMatchesIndependentlyMeasuredBurntFuelWithinOnePercent()
        {
            var p = Plan(); var fire = FireSim.Load(p, firePath, 7);
            double before = fire.GetCell(14, 7).FuelRemainingKg;
            fire.AdvanceTo(20);
            double expected = (before - fire.GetCell(14, 7).FuelRemainingKg) * .015;
            var smoke = Solver(p); smoke.AdvanceTo(20);
            // Removing injection, changing its units, or losing soot during transport breaks this comparison.
            Assert.That(expected, Is.GreaterThan(0));
            Assert.That(smoke.TotalMassKg, Is.EqualTo(expected).Within(expected * .01));
            Assert.That(smoke.TotalVentedKg, Is.Zero);
        }

        [TestCase(null)]
        [TestCase("closed")]
        public void WallAndClosedDoorKeepInertNeighbourExactlyEmpty(string state)
        {
            var smoke = Solver(Plan(state, true)); smoke.AdvanceTo(20);
            Assert.That(RoomMass(smoke, "living"), Is.GreaterThan(0));
            Assert.That(RoomMass(smoke, "hallway"), Is.Zero);
        }

        [TestCase("open")]
        [TestCase("removed")]
        [TestCase("half")]
        public void DoorPassesSmokeIntoAnInertNeighbour(string state)
        {
            var smoke = Solver(Plan(state, true)); smoke.AdvanceTo(20);
            // With no fuel in the neighbour, removing cross-door transport makes this fail.
            Assert.That(RoomMass(smoke, "hallway"), Is.GreaterThan(1e-9));
        }

        [Test]
        public void HalfDoorHasHalfThePhysicalApertureEvenBelowOneCellWidth()
        {
            var p = Plan("open", true); p.doors[0].width = .2f;
            var full = Solver(p);
            var halfPlan = JsonConvert.DeserializeObject<FloorPlan>(JsonConvert.SerializeObject(p));
            halfPlan.doors[0].state = "half";
            var half = Solver(halfPlan);
            Assert.That(full.OpenWidthM("opening"), Is.EqualTo(.2).Within(1e-6));
            Assert.That(half.OpenWidthM("opening"), Is.EqualTo(.1).Within(1e-6));
            full.AdvanceTo(5); half.AdvanceTo(5);
            Assert.That(RoomMass(half, "hallway"), Is.GreaterThan(0));
            Assert.That(RoomMass(full, "hallway"), Is.GreaterThan(RoomMass(half, "hallway")));
        }

        [Test]
        public void UpperLayerIsDenserAndItsInterfaceDescendsAsFireAddsGas()
        {
            var smoke = Solver(Plan()); smoke.AdvanceTo(.2);
            double early = smoke.GetCell(14, 7).InterfaceHeightM;
            smoke.AdvanceTo(3);
            var c = smoke.GetCell(14, 7);
            Assert.That(c.InterfaceHeightM, Is.LessThan(early - .001));
            Assert.That(c.UpperConcentrationKgM3, Is.GreaterThan(c.LowerConcentrationKgM3));
            Assert.That(c.LowerMassKg, Is.GreaterThan(0), "Near-flame mixing must move soot to the lower layer.");
            var upper = new Vector3(3.625f, 2.79f, 1.875f);
            var lower = new Vector3(3.625f, .01f, 1.875f);
            Assert.That(smoke.Density(upper, 3), Is.GreaterThan(smoke.Density(lower, 3)));
            Assert.That(smoke.Density(upper, 3), Is.EqualTo(c.UpperConcentrationKgM3 * 8700 * 3).Within(1e-5));
        }

        [TestCase(false)]
        [TestCase(true)]
        public void OpenExteriorDoorOrWindowLosesMassAndAccountsForIt(bool window)
        {
            var p = Plan("open");
            if (window) { p.windows.Add(p.doors[0]); p.doors.Clear(); }
            var vented = Solver(p); var sealedRoom = Solver(Plan());
            vented.AdvanceTo(20); sealedRoom.AdvanceTo(20);
            Assert.That(vented.TotalMassKg, Is.LessThan(sealedRoom.TotalMassKg * .99));
            Assert.That(vented.TotalVentedKg, Is.GreaterThan(0));
            Assert.That(vented.TotalMassKg + vented.TotalVentedKg,
                Is.EqualTo(vented.TotalInjectedKg).Within(vented.TotalInjectedKg * .01));
        }

        [Test]
        public void TwoRunsAndDifferentQueryCadencesHaveIdenticalStateBytes()
        {
            var a = Solver(Plan("open", true)); var b = Solver(Plan("open", true));
            for (int n = 1; n <= 200; n++) b.AdvanceTo(n * .1);
            a.AdvanceTo(20);
            CollectionAssert.AreEqual(a.StateBytes(), b.StateBytes());
            Assert.That(a.TotalMassKg, Is.GreaterThan(0));
        }

        [Test]
        public void RecentRewindRestoresStateAndReplaysFewerThanOneSnapshotInterval()
        {
            var a = Solver(Plan("open", true)); var forward = Solver(Plan("open", true));
            var point = new Vector3(3.625f, 2.7f, 1.875f);
            float expected = forward.Density(point, 8.7f);
            byte[] bytes = forward.StateBytes();
            a.AdvanceTo(12); float actual = a.Density(point, 8.7f);
            Assert.That(expected, Is.GreaterThan(0));
            Assert.AreEqual(expected, actual);
            CollectionAssert.AreEqual(bytes, a.StateBytes());
            Assert.That(a.LastReplaySteps, Is.LessThan(20));
            a.AdvanceTo(12); forward.AdvanceTo(12);
            CollectionAssert.AreEqual(forward.StateBytes(), a.StateBytes());
        }

        [Test]
        public void OldRewindBeyondRetainedHistoryReplaysFromZero()
        {
            var j = JObject.Parse(File.ReadAllText(smokePath)); j["snapshot_count"] = 2;
            var a = Solver(Plan(), Temporary(j)); var b = Solver(Plan());
            a.AdvanceTo(10); a.AdvanceTo(.7); b.AdvanceTo(.7);
            CollectionAssert.AreEqual(b.StateBytes(), a.StateBytes());
        }

        [Test]
        public void HeadStartAndInvalidPositionsUseTheSamplingContract()
        {
            var a = Solver(Plan(), headStart: 5); var b = Solver(Plan());
            var p = new Vector3(3.625f, 2.7f, 1.875f);
            Assert.AreEqual(b.Density(p, 7), a.Density(p, 2));
            CollectionAssert.AreEqual(b.StateBytes(), a.StateBytes());
            Assert.That(a.Density(new Vector3(-.1f, 1, 1), 2), Is.Zero);
            Assert.That(a.Density(new Vector3(3.625f, -.01f, 1.875f), 2), Is.Zero);
            Assert.That(a.Density(new Vector3(3.625f, 2.81f, 1.875f), 2), Is.Zero);
            Assert.Throws<ArgumentOutOfRangeException>(() => a.Density(p, float.NaN));
        }

        [Test]
        public void FluidFieldUsesPhysicalDensityWithoutRoomMinimumOrUnitClamp()
        {
            var p = Plan();
            var settings = new Scenario.Smoke { model = "fluid", global_haze = .7f };
            var field = new SmokeField(p, settings) { Override = (point, time) => 0 };
            var sample = new Vector3(1, 1, 1);
            Assert.That(field.Density(sample), Is.Zero);
            field.Override = (point, time) => 4;
            Assert.That(field.Density(sample), Is.EqualTo(4));
            Assert.That(SmokeField.MeanFreePath(4, true), Is.EqualTo(.75f));
            settings.model = "rooms";
            Assert.That(field.Density(sample), Is.EqualTo(1), "The legacy override path keeps its clamp.");
        }

        [Test]
        public void DisablingPlumeDivergenceChangesVelocityAndSmokeTransport()
        {
            var j = JObject.Parse(File.ReadAllText(smokePath)); j["plume_entrainment_m3_kj"] = 0;
            var still = Solver(Plan(), Temporary(j)); var plume = Solver(Plan());
            still.AdvanceTo(3); plume.AdvanceTo(3);
            var c = plume.GetCell(13, 7);
            Assert.That(Math.Abs(c.VelocityX) + Math.Abs(c.VelocityZ), Is.GreaterThan(1e-6));
            Assert.That(Math.Abs(still.GetCell(13, 7).VelocityX) + Math.Abs(still.GetCell(13, 7).VelocityZ), Is.Zero);
        }

        [Test]
        public void DiffusionPassesSmokeWithoutAnyPlumeVelocity()
        {
            var j = JObject.Parse(File.ReadAllText(smokePath)); j["plume_entrainment_m3_kj"] = 0;
            var diffusion = Solver(Plan("open", true), Temporary(j));
            j["smoke_diffusion_m2_s"] = 0;
            var stationary = Solver(Plan("open", true), Temporary(j));
            diffusion.AdvanceTo(5); stationary.AdvanceTo(5);
            Assert.That(RoomMass(diffusion, "hallway"), Is.GreaterThan(1e-9));
            Assert.That(RoomMass(stationary, "hallway"), Is.Zero);
        }

        [Test]
        public void AdvectionPassesSmokeWithoutScalarDiffusion()
        {
            var j = JObject.Parse(File.ReadAllText(smokePath)); j["smoke_diffusion_m2_s"] = 0;
            var moving = Solver(Plan("open", true), Temporary(j));
            j["plume_entrainment_m3_kj"] = 0;
            var stationary = Solver(Plan("open", true), Temporary(j));
            moving.AdvanceTo(20); stationary.AdvanceTo(20);
            Assert.That(RoomMass(moving, "hallway"), Is.GreaterThan(1e-9));
            Assert.That(RoomMass(stationary, "hallway"), Is.Zero);
        }

        [Test]
        public void FullUpperLayerSpillsIntoLowerLayerAndPreservesMass()
        {
            var j = JObject.Parse(File.ReadAllText(smokePath));
            j["smoke_diffusion_m2_s"] = 0; j["max_advection_speed_m_s"] = 1e-12;
            j["flame_mixing_per_s"] = 0; j["interface_mixing_per_s"] = 0;
            var smoke = Solver(Plan(), Temporary(j)); smoke.AdvanceTo(20);
            var cell = smoke.GetCell(14, 7);
            Assert.That(cell.InterfaceHeightM, Is.EqualTo(.05).Within(1e-6));
            Assert.That(cell.LowerMassKg, Is.GreaterThan(0));
            Assert.That(smoke.TotalMassKg, Is.EqualTo(smoke.TotalInjectedKg).Within(1e-12));
        }

        [Test]
        public void WaterSuppressionInvalidatesSmokeHistoryAndUsesRetainedHeatRelease()
        {
            var smoke = Solver(Plan()); smoke.AdvanceTo(10);
            Assert.That(smoke.TotalMassKg, Is.GreaterThan(0));
            smoke.ScheduleWater(0, new Vector2(3.625f, 1.875f), 0, .1f, 180, 10, 20);
            smoke.AdvanceTo(10);
            Assert.That(smoke.TotalInjectedKg, Is.Zero);
            Assert.That(smoke.TotalMassKg, Is.Zero);
            smoke.AdvanceTo(5);
            Assert.That(smoke.TotalInjectedKg, Is.Zero);
        }

        [Test]
        public void LiveWaterCommandsPreservePastStateAndStopFutureSootProduction()
        {
            var smoke = Solver(Plan()); smoke.AdvanceTo(2);
            byte[] before = smoke.StateBytes(); double injected = smoke.TotalInjectedKg;
            smoke.ScheduleWater(2, new Vector2(3.625f, 1.875f), 0, .1f, 180, 10, 20);
            CollectionAssert.AreEqual(before, smoke.StateBytes());
            smoke.AdvanceTo(3);
            Assert.That(smoke.TotalInjectedKg, Is.EqualTo(injected));
            Assert.That(smoke.TotalMassKg, Is.EqualTo(injected).Within(1e-12));
        }

        [Test]
        public void ParallelSamplesAtOneStepMatchSerialSamplesExactly()
        {
            // Lock-free reads at the current step (the RGB-D noise model samples from parallel rows): every value read by
            // 8 threads at once equals the serial locked read of the same point.
            var smoke = SmokeSolver.Load(FloorPlan.Load(), Path.Combine(FloorPlan.ConfigDir, "fire.json"), smokePath, 7);
            const float t = 40f;
            var points = new System.Collections.Generic.List<UnityEngine.Vector3>();
            for (float x = .2f; x < 20f; x += .37f) for (float z = .2f; z < 14f; z += .41f) points.Add(new UnityEngine.Vector3(x, 2.4f, z));
            var serial = new float[points.Count];
            for (int i = 0; i < points.Count; i++) serial[i] = smoke.Density(points[i], t);
            Assert.That(serial.Max(), Is.GreaterThan(0f), "smoke exists at 40 s");
            var parallel = new float[points.Count];
            System.Threading.Tasks.Parallel.For(0, points.Count, new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = 8 },
                i => parallel[i] = smoke.Density(points[i], t));
            CollectionAssert.AreEqual(serial, parallel);
        }

        [Test]
        public void HouseGridStepTimeIsLoggedWithAGenerousEditorBudget()
        {
            var smoke = SmokeSolver.Load(FloorPlan.Load(), Path.Combine(FloorPlan.ConfigDir, "fire.json"), smokePath, 7);
            smoke.AdvanceTo(1); // Warm up code paths and allocations outside the measured interval.
            var watch = Stopwatch.StartNew(); smoke.AdvanceTo(3); watch.Stop();
            double milliseconds = watch.Elapsed.TotalMilliseconds / 20;
            TestContext.WriteLine("Smoke house grid " + smoke.Width + "x" + smoke.Height +
                ", mean milliseconds per 0.1 s step including FireSim and checkpoints: " + milliseconds.ToString("F3"));
            // A 500 ms ceiling allows five times the simulated interval for unoptimized editor/CI builds.
            // This is a runaway-cost guard. The logged measurement determines real-time suitability.
            Assert.That(milliseconds, Is.LessThan(500));
        }
    }
}
