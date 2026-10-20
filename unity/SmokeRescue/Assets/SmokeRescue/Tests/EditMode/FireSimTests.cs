using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SmokeRescue.World;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    // No GameObjects, scene, physics, clock, or editor menu is needed by these tests.
    public class FireSimTests
    {
        string firePath;
        readonly List<string> temporaryConfigs = new List<string>();

        [SetUp]
        public void SetUp() => firePath = Path.Combine(FloorPlan.ConfigDir, "fire.json");

        [TearDown]
        public void TearDown()
        {
            foreach (string path in temporaryConfigs) File.Delete(path);
            temporaryConfigs.Clear();
        }

        FireSim House(int seed = 7) => FireSim.Load(FloorPlan.Load(), firePath, seed);

        string Config(Action<JObject> change)
        {
            var json = JObject.Parse(File.ReadAllText(firePath));
            change(json);
            string path = Path.Combine(Path.GetTempPath(), "smoke-fire-" + Guid.NewGuid().ToString("N") + ".json");
            File.WriteAllText(path, json.ToString());
            temporaryConfigs.Add(path);
            return path;
        }

        static FloorPlan TwoRooms(string state, string axis = "x")
        {
            bool x = axis == "x";
            return new FloorPlan
            {
                footprint = new float[] { 0, 0, x ? 8 : 4, x ? 4 : 8 },
                wall_height = 2.8f, interior_wall_thickness = .15f,
                exterior_wall_thickness = .25f, door_height = 2.1f,
                rooms = new List<FloorPlan.Room>
                {
                    new FloorPlan.Room { id = "living", rect = new float[] { 0, 0, 4, 4 },
                        floor = "wood_oak_planks", walls = "wallpaper_deep_green", ceiling = "plaster_white" },
                    new FloorPlan.Room { id = "hallway", rect = x ? new float[] { 4, 0, 8, 4 } : new float[] { 0, 4, 4, 8 },
                        floor = "wood_herringbone", walls = "wallpaper_hallway_damask", ceiling = "plaster_white" }
                },
                doors = new List<FloorPlan.Door>
                {
                    new FloorPlan.Door { id = "test_door", between = new[] { "living", "hallway" },
                        wall = axis, at = x ? new float[] { 4, 2 } : new float[] { 2, 4 }, width = 1, state = state }
                }
            };
        }

        string SmallFire() => Config(j => j["ignitions"] = JArray.Parse(
            "[{\"room\":\"living\",\"at\":[2,2],\"time_s\":0,\"radius_m\":0.3}]"));

        [Test]
        public void SameSeedHasIdenticalStateHashesAt60And240Seconds()
        {
            var a = House(); var b = House();
            foreach (double t in new[] { 60d, 240d })
            {
                a.AdvanceTo(t); b.AdvanceTo(t);
                Assert.AreEqual(a.StateHash(), b.StateHash(), "t=" + t);
            }
        }

        [Test]
        public void IrregularCadenceAndRepeatedTimesGiveTheSameBytes()
        {
            var direct = House(); var stepped = House();
            for (int n = 1; n < 240000 / 37; n++) stepped.AdvanceTo(n * .037);
            stepped.AdvanceTo(240); direct.AdvanceTo(240);
            Assert.AreEqual(direct.StateHash(), stepped.StateHash());
            string hash = stepped.StateHash();
            stepped.AdvanceTo(240); stepped.AdvanceTo(240);
            Assert.AreEqual(hash, stepped.StateHash());
            Assert.AreEqual(direct.ToJson(), stepped.ToJson());
        }

        [Test]
        public void TimeIsQuantizedAndBackwardOrNonfiniteTimeIsRejected()
        {
            var sim = House();
            sim.AdvanceTo(.099); Assert.AreEqual(0, sim.Time);
            sim.AdvanceTo(.1); Assert.AreEqual(.1, sim.Time, 1e-9);
            Assert.Throws<ArgumentOutOfRangeException>(() => sim.AdvanceTo(.09));
            Assert.Throws<ArgumentOutOfRangeException>(() => sim.AdvanceTo(double.NaN));
            Assert.Throws<ArgumentOutOfRangeException>(() => sim.AdvanceTo(double.PositiveInfinity));
        }

        [Test]
        public void SeedsChangeTheMaterialVariationWithoutUsingGlobalRandom()
        {
            var a = House(7); var b = House(8);
            a.AdvanceTo(60); b.AdvanceTo(60);
            Assert.AreNotEqual(a.ToJson(), b.ToJson());
            Assert.AreNotEqual(a.GetCell(10, 10).FuelRemainingKg, b.GetCell(10, 10).FuelRemainingKg);
        }

        [TestCase("x")]
        [TestCase("z")]
        public void ClosedDoorOrSolidWallBlocksFireAndSmoke(string axis)
        {
            var plan = TwoRooms("closed", axis);
            var sim = FireSim.Load(plan, SmallFire(), 7);
            sim.AdvanceTo(600);
            Assert.That(sim.GetRoomState("living").FlashoverTime, Is.GreaterThanOrEqualTo(0));
            AssertRoomUnburnt(sim, "hallway");
            Assert.AreEqual(20, sim.GetRoomState("hallway").UpperTemperatureC);
            Assert.AreEqual(0, sim.GetRoomState("hallway").SmokeDensity);
            plan.doors.Clear();
            sim = FireSim.Load(plan, SmallFire(), 7); sim.AdvanceTo(600);
            AssertRoomUnburnt(sim, "hallway");
        }

        [TestCase("x", "open")]
        [TestCase("z", "removed")]
        [TestCase("x", "half")]
        public void FirstHallwayFlameCrossesAtTheOpening(string axis, string state)
        {
            var sim = FireSim.Load(TwoRooms(state, axis), SmallFire(), 7);
            bool reached = false;
            for (int tick = 1; tick <= 6000 && !reached; tick++)
            {
                sim.AdvanceTo(tick * .1);
                for (int z = 0; z < sim.Height; z++) for (int x = 0; x < sim.Width; x++)
                {
                    var c = sim.GetCell(x, z);
                    if (c.Room != "hallway" || c.FlameIntensity <= 0) continue;
                    var p = sim.CellCentre(x, z);
                    float across = axis == "x" ? p.x : p.y;
                    float along = axis == "x" ? p.y : p.x;
                    Assert.That(across, Is.EqualTo(4.125f).Within(.001));
                    Assert.That(Math.Abs(along - 2), Is.LessThanOrEqualTo(state == "half" ? .25 : .5));
                    reached = true;
                }
            }
            Assert.IsTrue(reached, "An open-door positive control must spread.");
        }

        [Test]
        public void LivingRoomCannotReachTheRealHallwayWithAllItsExitsClosed()
        {
            var plan = FloorPlan.Load();
            foreach (var d in plan.doors.Where(d => d.between.Contains("living"))) d.state = "closed";
            var sim = FireSim.Load(plan, firePath, 7); sim.AdvanceTo(360);
            AssertRoomUnburnt(sim, "hallway");
            var open = House(); open.AdvanceTo(360);
            Assert.IsTrue(Cells(open).Any(c => c.Room == "hallway" && c.BurntFraction > 0));
        }

        [Test]
        public void IgnitionRoomReachesFlashoverBetween180And300Seconds()
        {
            var sim = House(); sim.AdvanceTo(300);
            Assert.That(sim.GetRoomState("living").FlashoverTime, Is.InRange(180d, 300d));
        }

        [Test]
        public void FireReachesEveryCombustibleRoomWithinSixMinutes()
        {
            var sim = House(); sim.AdvanceTo(360);
            foreach (string room in new[] { "living", "foyer", "kitchen", "hallway", "bedroom_1", "closet", "study", "bedroom_2", "laundry" })
                Assert.IsTrue(Cells(sim).Any(c => c.Room == room && c.BurntFraction > 0), room);
            AssertRoomUnburnt(sim, "bathroom");
        }

        [Test]
        public void FurnitureAddsFuelOnlyInsideItsRotatedRoomFootprint()
        {
            var plan = TwoRooms("open");
            var baseline = FireSim.Load(plan, SmallFire(), 7);
            plan.landmarks.Add(new FloorPlan.Landmark { room = "living", kind = "red_velvet_sofa", at = new float[] { 2, 2 }, yaw = 90 });
            var furnished = FireSim.Load(plan, SmallFire(), 7);
            Assert.Greater(furnished.GetCell(8, 11).InitialFuelKg, baseline.GetCell(8, 11).InitialFuelKg);
            Assert.AreEqual(baseline.GetCell(11, 8).InitialFuelKg, furnished.GetCell(11, 8).InitialFuelKg);
            for (int z = 0; z < furnished.Height; z++) for (int x = 16; x < furnished.Width; x++)
                Assert.AreEqual(baseline.GetCell(x, z).InitialFuelKg, furnished.GetCell(x, z).InitialFuelKg);
        }

        [TestCase("step_s", 0)]
        [TestCase("cell_size_m", -1)]
        [TestCase("flashover_threshold_c", 2000)]
        public void InvalidConfigurationIsRejected(string field, double value)
        {
            string path = Config(j => j[field] = value);
            Assert.Throws<InvalidDataException>(() => FireSim.Load(FloorPlan.Load(), path, 7));
        }

        [Test]
        public void HotLayerDescendsAndCrawlingAirIsCooler()
        {
            var sim = House(); sim.AdvanceTo(30);
            float early = sim.GetRoomState("living").LayerHeight;
            sim.AdvanceTo(120);
            var room = sim.GetRoomState("living");
            Assert.That(room.LayerHeight, Is.LessThan(early));
            Assert.That(room.SmokeDensity, Is.GreaterThan(0));
            Assert.Less(sim.AirTemperatureC(new Vector3(6, .3f, 4)), sim.AirTemperatureC(new Vector3(6, 2.2f, 4)));
            Assert.Less(sim.SmokeDensity(new Vector3(6, .3f, 4)), sim.SmokeDensity(new Vector3(6, 2.2f, 4)));
        }

        [Test]
        public void CollapseRequiresContinuousHotExposureAndProtectsDoorApproaches()
        {
            var rules = JObject.Parse(File.ReadAllText(firePath));
            double threshold = (double)rules["collapse"]["temperature_c"];
            double duration = (double)rules["collapse"]["duration_s"];
            int max = (int)rules["collapse"]["max_events"];
            var sim = House(); var hotSince = new Dictionary<string, double>();
            int seen = 0;
            var plan = FloorPlan.Load();
            for (int tick = 1; tick <= 6000; tick++)
            {
                sim.AdvanceTo(tick * .1);
                foreach (var room in sim.RoomStates)
                {
                    if (room.UpperTemperatureC < threshold) hotSince.Remove(room.Room);
                    else if (!hotSince.ContainsKey(room.Room)) hotSince[room.Room] = sim.Time;
                }
                while (seen < sim.Collapses.Count)
                {
                    var e = sim.Collapses[seen++];
                    Assert.IsTrue(hotSince.ContainsKey(e.Room));
                    Assert.That(e.Time - hotSince[e.Room], Is.GreaterThanOrEqualTo(duration - .100001));
                    Assert.That(e.HotExposureSeconds, Is.GreaterThanOrEqualTo(duration));
                    foreach (var d in plan.doors.Where(d => d.between.Contains(e.Room)))
                    {
                        float lo = d.RunsAlongX ? e.CellRect.xMin : e.CellRect.yMin;
                        float hi = d.RunsAlongX ? e.CellRect.xMax : e.CellRect.yMax;
                        float centre = d.RunsAlongX ? d.at[0] : d.at[1];
                        Assert.IsTrue(hi <= centre - .45f || lo >= centre + .45f, d.id);
                    }
                }
                Assert.That(seen, Is.LessThanOrEqualTo(max));
            }
            Assert.That(seen, Is.GreaterThan(0), "Collapse checks need a positive control.");
            Assert.AreEqual(seen, sim.Collapses.Select(e => e.CellRect).Distinct().Count());
        }

        [Test]
        public void ADelayedIgnitionDoesNotStartEarlyAndLegacyFiresAreIgnored()
        {
            string path = Config(j => j["ignitions"][0]["time_s"] = 10);
            var sim = FireSim.Load(FloorPlan.Load(), path, 7);
            sim.AdvanceTo(9.9); Assert.IsEmpty(sim.ActiveFlames());
            sim.AdvanceTo(10); Assert.IsNotEmpty(sim.ActiveFlames());
            Assert.AreEqual(0, sim.FlameIntensity(new Vector2(17.5f, 1.2f)));
        }

        [Test]
        public void JsonIsDeterministicAndStaticDumpUsesTheSameSolver()
        {
            var plan = FloorPlan.Load(); var sim = FireSim.Load(plan, firePath, 7); sim.AdvanceTo(60);
            string dump = FireSim.DumpJson(plan, firePath, 7, 60);
            Assert.AreEqual(sim.ToJson(), dump);
            var j = JObject.Parse(dump);
            Assert.AreEqual(1, (int)j["version"]);
            Assert.AreEqual(sim.Width * sim.Height, j["cells"].Count());
            Assert.AreEqual(10, j["rooms"].Count());
            Assert.IsNotEmpty(j["walls"]);
        }

        [Test]
        public void SixHundredSecondsStayFiniteBoundedAndConsumeFuelMonotonically()
        {
            var sim = House();
            float[] fuel = Cells(sim).Select(c => c.FuelRemainingKg).ToArray();
            for (int second = 10; second <= 600; second += 10)
            {
                sim.AdvanceTo(second);
                int i = 0;
                foreach (var c in Cells(sim))
                {
                    CheckRange(c.FuelRemainingKg, 0, fuel[i] + .00001f);
                    CheckRange(c.SurfaceTemperatureC, 20, 1100);
                    CheckRange(c.FlameIntensity, 0, 1); CheckRange(c.BurntFraction, 0, 1);
                    fuel[i++] = c.FuelRemainingKg;
                }
                foreach (var r in sim.RoomStates)
                {
                    CheckRange(r.UpperTemperatureC, 20, 1100); CheckRange(r.LowerTemperatureC, 20, r.UpperTemperatureC);
                    CheckRange(r.LayerHeight, 0, 2.8f); CheckRange(r.SmokeDensity, 0, 1);
                }
                foreach (var f in sim.ActiveFlames())
                {
                    CheckRange(f.Intensity, 0, 1); CheckRange(f.Radius, 0, sim.CellSize);
                    CheckRange(sim.AirTemperatureC(f.Centre + Vector3.up * .3f), 20, 1100);
                    CheckRange(sim.SmokeDensity(f.Centre + Vector3.up * 2.2f), 0, 1);
                }
            }
            Assert.AreEqual(20, sim.SurfaceTemperatureC(new Vector2(-1, -1)));
            Assert.AreEqual(20, sim.AirTemperatureC(new Vector3(3, 4, 3)));
            Assert.AreEqual(0, sim.SmokeDensity(new Vector3(-1, 1, -1)));
        }

        static IEnumerable<FireSim.CellState> Cells(FireSim sim)
        {
            for (int z = 0; z < sim.Height; z++) for (int x = 0; x < sim.Width; x++) yield return sim.GetCell(x, z);
        }
        static void AssertRoomUnburnt(FireSim sim, string room)
        {
            foreach (var c in Cells(sim).Where(c => c.Room == room))
            {
                Assert.AreEqual(0, c.BurntFraction); Assert.AreEqual(0, c.FlameIntensity);
            }
        }
        static void CheckRange(float value, float min, float max)
        {
            if (float.IsNaN(value) || float.IsInfinity(value) || value < min || value > max)
                Assert.Fail("Value " + value + " outside [" + min + ", " + max + "]");
        }
    }
}
