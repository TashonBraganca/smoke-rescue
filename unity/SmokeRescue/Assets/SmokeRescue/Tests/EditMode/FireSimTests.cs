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
            // Only the living-room ignition: the scenario's burning hallway wood pile is a separate fire by design.
            var cfg = Newtonsoft.Json.Linq.JObject.Parse(File.ReadAllText(firePath));
            cfg["ignitions"] = new Newtonsoft.Json.Linq.JArray(cfg["ignitions"].Where(i => (string)i["room"] == "living"));
            string livingOnly = Path.Combine(Path.GetTempPath(), "fire_living_only.json");
            File.WriteAllText(livingOnly, cfg.ToString());
            var sim = FireSim.Load(plan, livingOnly, 7); sim.AdvanceTo(360);
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
            Assert.AreEqual(-1, sim.GetRoomState("bathroom").FlashoverTime);
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
        public void ADisplacedDoorCannotExchangeGasThroughAnUnrelatedWall()
        {
            var plan = TwoRooms("open");
            plan.doors[0].at = new float[] { 7, 2 };
            Assert.Throws<InvalidDataException>(() => FireSim.Load(plan, SmallFire(), 7));
        }

        [Test]
        public void MissingFurnitureDefinitionsAreRejected()
        {
            var plan = FloorPlan.Load();
            plan.landmarks[0].kind = "undefined_fuel";
            Assert.Throws<InvalidDataException>(() => FireSim.Load(plan, firePath, 7));
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
            Assert.AreEqual(plan.rooms.Count, j["rooms"].Count());
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

        // Isolate spray effects from room heating; the reignition test restores neighbour heating.
        string WaterFire(bool neighbour = false) => Config(j =>
        {
            j["spread"]["neighbour_heat_per_s"] = neighbour ? 1 : 0;
            j["spread"]["gas_surface_heat_per_s"] = 0;
            j["spread"]["flashover_surface_heat_c_s"] = 0;
            j["ignitions"] = JArray.Parse(neighbour
                ? "[{\"room\":\"living\",\"at\":[2.125,2.125],\"time_s\":0,\"radius_m\":0},{\"room\":\"living\",\"at\":[2.375,2.125],\"time_s\":0,\"radius_m\":0}]"
                : "[{\"room\":\"living\",\"at\":[2.125,2.125],\"time_s\":0,\"radius_m\":0}]");
        });

        static void TargetWater(FireSim sim, double flow = 1, double start = 0, double duration = 10) =>
            sim.ScheduleWater(start, new Vector2(2.125f, 1.875f), 0, .3f, 1, flow, duration);

        [Test]
        public void OneLitrePerSecondForTenSecondsExtinguishesButZeroFlowDoesNot()
        {
            string path = WaterFire();
            var wet = FireSim.Load(TwoRooms("closed"), path, 7);
            var dry = FireSim.Load(TwoRooms("closed"), path, 7);
            TargetWater(wet); TargetWater(dry, 0);
            wet.AdvanceTo(10); dry.AdvanceTo(10);
            Assert.AreEqual(0, wet.GetCell(8, 8).FlameIntensity, "The cone must extinguish the burning target.");
            Assert.Less(wet.GetCell(8, 8).SurfaceTemperatureC, 330, "Extinction includes cooling below wood ignition.");
            Assert.Greater(dry.GetCell(8, 8).FlameIntensity, 0, "Zero-flow control must still burn.");
            Assert.Greater(wet.GetCell(8, 8).FuelRemainingKg, dry.GetCell(8, 8).FuelRemainingKg,
                "Water must stop burning before fuel exhaustion.");
        }

        [TestCase(2.125f, 1.625f)]
        [TestCase(2.125f, 2.625f)]
        public void CellsBehindNozzleOrBeyondRangeAreUnchanged(float x, float z)
        {
            string path = WaterFire();
            var json = JObject.Parse(File.ReadAllText(path));
            ((JArray)json["ignitions"]).Add(new JObject { ["room"] = "living", ["at"] = new JArray(x, z),
                ["time_s"] = 0, ["radius_m"] = 0 });
            File.WriteAllText(path, json.ToString());
            var wet = FireSim.Load(TwoRooms("closed"), path, 7);
            var dry = FireSim.Load(TwoRooms("closed"), path, 7);
            TargetWater(wet); wet.AdvanceTo(10); dry.AdvanceTo(10);
            int cx = (int)(x / wet.CellSize), cz = (int)(z / wet.CellSize);
            Assert.AreEqual(dry.GetCell(cx, cz).FuelRemainingKg, wet.GetCell(cx, cz).FuelRemainingKg,
                "An excluded cell must consume exactly the same fuel.");
            Assert.AreEqual(dry.GetCell(cx, cz).SurfaceTemperatureC, wet.GetCell(cx, cz).SurfaceTemperatureC);
            Assert.AreEqual(dry.GetCell(cx, cz).FlameIntensity, wet.GetCell(cx, cz).FlameIntensity);
            Assert.AreEqual(0, wet.GetCell(8, 8).FlameIntensity, "The in-cone positive control must be extinguished.");
        }

        [Test]
        public void WaterSchedulesGiveIdenticalHashesAcrossAdvanceCadencesAtSixtySeconds()
        {
            string path = WaterFire(); var plan = TwoRooms("closed");
            var a = FireSim.Load(plan, path, 7); var b = FireSim.Load(plan, path, 7);
            var dry = FireSim.Load(plan, path, 7);
            // Start-time ordering must work even if the future command is submitted first.
            TargetWater(a, .5, 20, 1); TargetWater(a, 1, .05, 10);
            TargetWater(b, 1, .05, 10); TargetWater(b, .5, 20, 1);
            for (int n = 1; n <= 600; n++) b.AdvanceTo(n * .1);
            a.AdvanceTo(60); dry.AdvanceTo(60);
            Assert.AreEqual(a.StateHash(), b.StateHash(), "Same sprays must survive different advance and submission order.");
            Assert.AreNotEqual(a.StateHash(), dry.StateHash());
            Assert.Greater(a.GetCell(8, 8).FuelRemainingKg, dry.GetCell(8, 8).FuelRemainingKg,
                "Hash differences must reflect suppression, not just the stored command log.");
        }

        [Test]
        public void HeatSourceReplaysSpraysAfterRewindingFromSixtyToFiveSeconds()
        {
            var plan = FloorPlan.Load();
            var direct = new FireSimHeat(plan, 7, 0); var rewind = new FireSimHeat(plan, 7, 0);
            foreach (var heat in new[] { direct, rewind })
                heat.ScheduleWater(0, new Vector2(3.125f, .875f), 0, .5f, 5, 1, 10);
            direct.Sample(Vector2.zero, 60); rewind.Sample(Vector2.zero, 60);
            string first = rewind.Sim.StateHash();
            rewind.Sample(Vector2.zero, 5); rewind.Sample(Vector2.zero, 60);
            Assert.AreEqual(first, rewind.Sim.StateHash(), "Reload must restore every spray before replay.");
            Assert.AreEqual(direct.Sim.StateHash(), rewind.Sim.StateHash());
            Assert.Greater(rewind.Sim.TotalWaterEnergyRemovedKj, 0, "Replayed commands must actually absorb heat.");
        }

        [Test]
        public void WetFuelResistsReignitionBesideBurningNeighbourUntilDry()
        {
            string path = WaterFire(true);
            var wet = FireSim.Load(TwoRooms("closed"), path, 7);
            var dry = FireSim.Load(TwoRooms("closed"), path, 7);
            TargetWater(wet); wet.AdvanceTo(10); dry.AdvanceTo(10);
            Assert.AreEqual(0, wet.GetCell(8, 8).FlameIntensity);
            Assert.Greater(dry.GetCell(8, 8).FlameIntensity, 0, "No-water control keeps burning.");
            Assert.Greater(wet.GetCell(9, 8).FlameIntensity, 0, "Neighbour must remain a live ignition source.");
            float fuel = wet.GetCell(8, 8).FuelRemainingKg;
            bool reheated = false;
            for (int second = 11; second <= 30; second++)
            {
                wet.AdvanceTo(second);
                reheated |= wet.GetCell(8, 8).SurfaceTemperatureC >= 330;
                Assert.AreEqual(0, wet.GetCell(8, 8).FlameIntensity, "Wet fuel must not relight after the spray stops.");
                Assert.AreEqual(fuel, wet.GetCell(8, 8).FuelRemainingKg);
            }
            Assert.IsTrue(reheated, "The wet timer must block ignition even after neighbour heat raises temperature above ignition.");
            // Change only the dry-out time to prove that cooling alone cannot explain the result.
            var json = JObject.Parse(File.ReadAllText(path)); json["water"]["wet_seconds_per_litre"] = .001;
            File.WriteAllText(path, json.ToString());
            var fastDry = FireSim.Load(TwoRooms("closed"), path, 7);
            TargetWater(fastDry); fastDry.AdvanceTo(10);
            Assert.AreEqual(0, fastDry.GetCell(8, 8).FlameIntensity);
            fastDry.AdvanceTo(30);
            Assert.Greater(fastDry.GetCell(8, 8).FlameIntensity, 0, "The same cooled fuel must relight when its wet time expires.");
        }

        [Test]
        public void WaterEnergyCannotExceedDeliveredLitresTimesConfiguredCapacity()
        {
            string path = WaterFire(); var sim = FireSim.Load(TwoRooms("closed"), path, 7);
            var water = JObject.Parse(File.ReadAllText(path))["water"];
            double perLitre = (double)water["latent_heat_kj_l"] + (double)water["sensible_heat_kj_l"];
            double efficiency = (double)water["application_efficiency"];
            // Short, off-grid spray exercises partial ticks rather than rounding up the water supply.
            TargetWater(sim, .2, .035, .14);
            for (int n = 1; n <= 20; n++)
            {
                sim.AdvanceTo(n * .1);
                double litres = .2 * Math.Max(0, Math.Min(sim.Time, .175) - .035);
                Assert.LessOrEqual(sim.TotalWaterEnergyRemovedKj, litres * perLitre * efficiency + 1e-8,
                    "Absorbed energy cannot exceed the effective water budget, including partial ticks.");
            }
            Assert.Greater(sim.TotalWaterEnergyRemovedKj, 0, "A zero-energy implementation must fail this check.");
            Assert.LessOrEqual(sim.TotalWaterEnergyRemovedKj, .2 * .14 * perLitre);
            double stopped = sim.TotalWaterEnergyRemovedKj;
            sim.AdvanceTo(10); Assert.AreEqual(stopped, sim.TotalWaterEnergyRemovedKj, "Cooling stops at the spray end.");
        }

        [TestCase("x", "closed")]
        [TestCase("z", "closed")]
        [TestCase("x", "open")]
        [TestCase("z", "half")]
        public void WaterSightUsesDoorWidthsAndUnityYaw(string axis, string state)
        {
            bool east = axis == "x";
            string path = WaterFire(); var json = JObject.Parse(File.ReadAllText(path));
            json["ignitions"] = JArray.Parse(east
                ? "[{\"room\":\"hallway\",\"at\":[4.125,2.125],\"time_s\":0,\"radius_m\":0}]"
                : "[{\"room\":\"hallway\",\"at\":[2.125,4.125],\"time_s\":0,\"radius_m\":0}]");
            File.WriteAllText(path, json.ToString());
            var sim = FireSim.Load(TwoRooms(state, axis), path, 7);
            sim.ScheduleWater(0, east ? new Vector2(3.875f, 2.125f) : new Vector2(2.125f, 3.875f),
                east ? Mathf.PI / 2 : 0, .3f, 1, 1, 10);
            sim.AdvanceTo(10);
            var target = sim.GetCell(east ? 16 : 8, east ? 8 : 16);
            Assert.AreEqual(state == "closed", target.FlameIntensity > 0,
                "A closed wall blocks the stream; an open aperture admits it along the stated yaw.");
            if (state != "closed") Assert.Greater(sim.TotalWaterEnergyRemovedKj, 0);
            else
            {
                // Positive control prevents a no-water implementation passing the closed-door case.
                var open = FireSim.Load(TwoRooms("open", axis), path, 7);
                open.ScheduleWater(0, east ? new Vector2(3.875f, 2.125f) : new Vector2(2.125f, 3.875f),
                    east ? Mathf.PI / 2 : 0, .3f, 1, 1, 10);
                open.AdvanceTo(10);
                Assert.AreEqual(0, open.GetCell(east ? 16 : 8, east ? 8 : 16).FlameIntensity);
            }
        }

        [Test]
        public void WaterAtAmbientOnlyWetsAndBlocksADelayedIgnition()
        {
            string path = WaterFire();
            var json = JObject.Parse(File.ReadAllText(path));
            json["ignitions"][0]["time_s"] = 1;
            File.WriteAllText(path, json.ToString());
            var wet = FireSim.Load(TwoRooms("closed"), path, 7);
            var dry = FireSim.Load(TwoRooms("closed"), path, 7);
            TargetWater(wet, 1, 0, .5);
            float fuel = wet.GetCell(8, 8).FuelRemainingKg;
            wet.AdvanceTo(.5);
            Assert.AreEqual(20, wet.GetCell(8, 8).SurfaceTemperatureC, "Cold cells cannot cool below ambient.");
            Assert.AreEqual(0, wet.TotalWaterEnergyRemovedKj, "Ambient, unlit fuel supplies no energy to absorb.");
            Assert.AreEqual(fuel, wet.GetCell(8, 8).FuelRemainingKg);
            wet.AdvanceTo(2); dry.AdvanceTo(2);
            Assert.AreEqual(0, wet.GetCell(8, 8).FlameIntensity, "Wetting must inhibit a later ignition after the stream stops.");
            Assert.Greater(dry.GetCell(8, 8).FlameIntensity, 0, "The delayed dry ignition must burn.");
        }

        [Test]
        public void WaterRejectsInvalidArgumentsAndPastRequestsWithoutChangingState()
        {
            var sim = FireSim.Load(TwoRooms("closed"), WaterFire(), 7);
            string initial = sim.StateHash();
            foreach (double bad in new[] { -1d, double.NaN, double.PositiveInfinity, double.NegativeInfinity })
            {
                Assert.Throws<ArgumentOutOfRangeException>(() => TargetWater(sim, start: bad));
                Assert.Throws<ArgumentOutOfRangeException>(() => TargetWater(sim, flow: bad));
                Assert.Throws<ArgumentOutOfRangeException>(() => TargetWater(sim, duration: bad));
                Assert.Throws<ArgumentOutOfRangeException>(() => sim.ScheduleWater(0, Vector2.zero, (float)bad, 1, 1, 1, 1));
                Assert.Throws<ArgumentOutOfRangeException>(() => sim.ScheduleWater(0, Vector2.zero, 0, (float)bad, 1, 1, 1));
                Assert.Throws<ArgumentOutOfRangeException>(() => sim.ScheduleWater(0, Vector2.zero, 0, 1, (float)bad, 1, 1));
                Assert.Throws<ArgumentOutOfRangeException>(() => sim.ScheduleWater(0, new Vector2((float)bad, 0), 0, 1, 1, 1, 1));
                Assert.Throws<ArgumentOutOfRangeException>(() => sim.ScheduleWater(0, new Vector2(0, (float)bad), 0, 1, 1, 1, 1));
            }
            Assert.Throws<ArgumentOutOfRangeException>(() => sim.ScheduleWater(0, Vector2.zero, float.NaN, 1, 1, 1, 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => sim.ScheduleWater(0, Vector2.zero, 0, -1, 1, 1, 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => sim.ScheduleWater(0, Vector2.zero, 0, 1, -1, 1, 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => sim.ScheduleWater(0, Vector2.zero, 0, 1, 181, 1, 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => sim.ScheduleWater(0, new Vector2(float.PositiveInfinity, 0), 0, 1, 1, 1, 1));
            Assert.AreEqual(initial, sim.StateHash(), "Rejected schedules must not mutate the replay log.");
            sim.AdvanceTo(.15);
            Assert.Throws<ArgumentOutOfRangeException>(() => TargetWater(sim, start: .14));
            TargetWater(sim, start: .15); sim.AdvanceTo(10.2);
            Assert.Greater(sim.TotalWaterEnergyRemovedKj, 0, "A schedule at the last requested time must work.");
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
