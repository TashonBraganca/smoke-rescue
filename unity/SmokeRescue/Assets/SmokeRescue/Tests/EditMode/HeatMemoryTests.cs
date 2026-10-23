using System.Linq;
using NUnit.Framework;
using SmokeRescue.Brain;
using SmokeRescue.Sensors;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    public sealed partial class BrainTests
    {
        /// <summary>Codex HeatCheck (23 Oct): a cell the camera saw at 500 C and 3 s later at 20 C stayed fire at 499 C,
        /// because cool pixels never reached ObserveHeat; a partner's old 500 C held against a fresh local 20 C too. Seed 934,
        /// robot A: 21 routes cut by fire margins while its own air never passed 44 C. Controls: with no cool look the cell
        /// stays fire; a partner report newer than the cool look heats it again.</summary>
        [Test]
        public void ACellSeenCoolAfterBurningStopsBeingFireAndAnOlderPartnerReportDoesNotHoldIt()
        {
            var s = Calibration(); var c = Settings();
            var depth = Enumerable.Repeat((ushort)2000, s.camera.width * s.camera.height).ToArray();
            ThermalFrame Thermal(double t, float celsius) => new ThermalFrame(t, s.thermal.width, s.thermal.height,
                Enumerable.Repeat((ushort)Mathf.RoundToInt((celsius + 273.15f) * 10), s.thermal.width * s.thermal.height).ToArray());
            RgbdFrame Rgbd(double t) => new RgbdFrame(t, s.camera.width, s.camera.height, new byte[depth.Length * 3], depth);
            // Internal members, reached by reflection from the test assembly as elsewhere in these tests.
            var project = typeof(ThermalPerception).GetMethod("ProjectHeat", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
            var peerHeat = typeof(OccupancyGrid).GetMethod("ObservePeerHeat", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            void Project(ThermalFrame th, RgbdFrame rgb, OccupancyGrid g) => project.Invoke(null, new object[] { th, rgb, s, c, new BrainPose(), g });
            void Peer(OccupancyGrid g, int i, float temp, double at) => peerHeat.Invoke(g, new object[] { i, temp, at, at });

            var grid = new OccupancyGrid(c);
            grid.DecayTo(0); Project(Thermal(0, 500), Rgbd(0), grid);
            int cell = Enumerable.Range(0, grid.Count).First(grid.Dangerous);
            grid.DecayTo(3);
            Assert.That(grid.Dangerous(cell), Is.True, "control: not looked at again, still fire");
            Project(Thermal(3, 20), Rgbd(3), grid);
            Assert.That(grid.Dangerous(cell), Is.False, "seen at 20 C 3 s after 500 C: the fire there is out");

            var peer = new OccupancyGrid(c);
            peer.DecayTo(0); Peer(peer, cell, 500, 0);
            Assert.That(peer.Dangerous(cell), Is.True, "control: the partner's report alone counts");
            peer.DecayTo(3); peer.ObserveHeat(cell, 20, 3);
            Assert.That(peer.Dangerous(cell), Is.False, "this robot's newer cool look beats the partner's older 500 C");
            Peer(peer, cell, 500, 4);
            Assert.That(peer.Dangerous(cell), Is.True, "control: a partner report newer than the cool look heats it again");
        }
    }
}
