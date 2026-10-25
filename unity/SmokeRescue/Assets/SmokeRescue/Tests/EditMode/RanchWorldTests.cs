using System;
using System.IO;
using System.Linq;
using NUnit.Framework;
using SmokeRescue.World;

namespace SmokeRescue.Tests.EditMode
{
    public partial class BrainTests
    {
        [Test]
        public void RanchScenariosLoadPeopleAndFireAgainstTheirSelectedHouse()
        {
            foreach (string name in new[] { "family_ranch", "long_ranch" })
            {
                var scenario = Scenario.Load($"scenario_{name}_full.json");
                Assert.AreEqual($"floorplan_{name}.json", Scenario.ReadFloorPlanFile($"scenario_{name}_full.json"));
                var plan = FloorPlan.Load(scenario.FloorPlanPath);
                Assert.AreEqual("fixed", scenario.survivor_placement);
                Assert.AreEqual(5, scenario.survivors.Select(s => s.room).Distinct().Count());
                Assert.IsTrue(scenario.survivors.All(s => plan.FindRoom(s.room) != null));
                Assert.AreEqual(600f, scenario.duration_s);
                Assert.AreEqual(480f, scenario.search_s);
                Assert.IsTrue(scenario.time_limit);
                Assert.AreEqual($"fire_{name}.json", scenario.fire.config_file);
                Assert.DoesNotThrow(() => FireSim.Load(plan, scenario.fire.ConfigPath, scenario.seed));
            }
        }

        [Test]
        public void RanchFloorplanSelectionRejectsPathsOutsideConfig()
        {
            foreach (string file in new[] { "../floorplan.json", "folder/floorplan.json", "folder\\floorplan.json", "brain.json", "floorplan.txt", "", null })
            {
                var scenario = new Scenario { floorplan_file = file };
                Assert.Throws<InvalidDataException>(() => { var path = scenario.FloorPlanPath; });
            }
        }

        [Test]
        public void RanchFloorplanSelectionReadsInheritanceWithoutPlacingPeople()
        {
            string dir = Path.Combine(Directory.GetParent(FloorPlan.ConfigDir).FullName, "data", "runs", "ranch-selection-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            try
            {
                File.WriteAllText(Path.Combine(dir, "parent.json"), "{\"floorplan_file\":\"floorplan_long_ranch.json\",\"survivor_placement\":\"random\"}");
                File.WriteAllText(Path.Combine(dir, "child.json"), "{\"inherits\":\"parent.json\"}");
                Assert.AreEqual("floorplan_long_ranch.json", Scenario.ReadFloorPlanFile(Path.Combine(dir, "child.json")));
            }
            finally { Directory.Delete(dir, true); }
        }

        [Test]
        public void RanchSupportPreservesTheDefaultHouseSelection()
        {
            Assert.AreEqual("floorplan.json", Scenario.ReadFloorPlanFile("scenario_house_onedoor_seed934_full.json"));
            Assert.AreEqual(Path.Combine(FloorPlan.ConfigDir, "floorplan.json"), new Scenario().FloorPlanPath);
        }
    }
}
