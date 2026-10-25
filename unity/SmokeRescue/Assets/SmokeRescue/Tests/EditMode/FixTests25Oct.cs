using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using SmokeRescue.Brain;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    /// <summary>Fixes from the 25 Oct root-cause analysis of data/recordings/unity_house_onedoor_seed934_full_1520_5of5. Each test
    /// names the time and count that motivated it; the matching config keys carry the same evidence in their *_reason.</summary>
    public sealed partial class BrainTests
    {
        [Test]
        public void AnAgentActionResultTakesTheBrainsClockNotAStaleHostTime()
        {
            // F8. Brain-only seed 934: four "spray_water succeeded" rows were written with t = 0.0 at 371 to 467 s.
            var c = Settings(); c.grid_size_m = 12;
            var b = new RobotBrain(Calibration(), c, "A", 0); Room(b, -4, 4, -4, 4);
            for (int i = 0; i <= 60; i++) DriveTick(b, i);
            b.DrainEvents();
            b.LogAgentAction(0.0, "spray_water succeeded (target 11deg)");
            var e = b.DrainEvents().Single(x => x.kind == "agent_order");
            Assert.That(e.t, Is.GreaterThanOrEqualTo(2.95), "stamped with the brain's clock (3.0 s), not 0.0");
        }
    }
}
