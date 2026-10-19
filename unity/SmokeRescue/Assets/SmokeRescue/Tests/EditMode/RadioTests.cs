using NUnit.Framework;
using SmokeRescue.Editor;
using SmokeRescue.Radio;
using SmokeRescue.World;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    /// <summary>C6 acceptance: five hand-placed pairs give the expected ok / distance / walls results.</summary>
    public class RadioTests
    {
        GameObject house;
        RadioModel radio;

        [SetUp]
        public void SetUp()
        {
            ProjectSetup.EnsureLayers();
            house = HouseBuilder.Build(FloorPlan.Load(), MaterialLibrary.GreyBox);
            Physics.SyncTransforms();
            radio = new RadioModel(12f, 1);
        }

        [TearDown]
        public void TearDown() => Object.DestroyImmediate(house);

        static Vector3 P(float x, float z) => new Vector3(x, 0f, z);

        [TestCase(10f, 2f, 10f, 4f, "ok", 0, TestName = "Same room, foyer")]
        [TestCase(10f, 3f, 10f, 7f, "ok", 0, TestName = "Foyer to hallway through the archway")]
        [TestCase(7f, 2f, 2f, 11f, "walls", 2, TestName = "Living to bedroom 1 through two walls")]
        [TestCase(14f, 3f, 14f, 7f, "ok", 1, TestName = "Kitchen to hallway through one wall")]
        [TestCase(1f, 1f, 19f, 13f, "distance", -1, TestName = "Opposite corners, beyond 12 m")]
        public void LinkRule(float ax, float az, float bx, float bz, string reason, int walls)
        {
            var link = radio.Evaluate(P(ax, az), P(bx, bz));
            Assert.AreEqual(reason, link.reason, $"distance {link.distance:0.00} m, walls {link.walls}");
            if (walls >= 0) Assert.AreEqual(walls, link.walls);
            Assert.AreEqual(reason == "ok", link.delivered);
            Assert.AreEqual(0.050 + 0.001 * link.distance, link.delay, 1e-9);
        }
    }
}
