using System.Collections.Generic;
using System.Linq;
using System.Text;
using NUnit.Framework;
using SmokeRescue.Editor;
using SmokeRescue.World;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    /// <summary>C1 acceptance: door gap widths, a capsule through every door, deterministic rebuilds.</summary>
    public class HouseBuilderTests
    {
        FloorPlan plan;
        GameObject house;

        [SetUp]
        public void SetUp()
        {
            ProjectSetup.EnsureLayers();
            plan = FloorPlan.Load();
            house = HouseBuilder.Build(plan, MaterialLibrary.GreyBox);
            Physics.SyncTransforms();
        }

        [TearDown]
        public void TearDown()
        {
            if (house != null) Object.DestroyImmediate(house);
        }

        static (Vector3 along, Vector3 across) Axes(FloorPlan.Door d) =>
            d.RunsAlongX ? (Vector3.right, Vector3.forward) : (Vector3.forward, Vector3.right);

        static Vector3 Centre(FloorPlan.Door d, float y) => new Vector3(d.at[0], y, d.at[1]);

        [Test]
        public void EveryDoorGapMatchesItsWidthWithin2cm()
        {
            int wallMask = 1 << LayerMask.NameToLayer(HouseBuilder.WallsLayer);
            var failures = new List<string>();
            foreach (var d in plan.doors)
            {
                var (along, _) = Axes(d);
                // Exterior walls sit inside the footprint, so sample the gap on the wall's centre plane.
                var line = HouseBuilder.WallLines(plan, Rect.MinMaxRect(plan.footprint[0], plan.footprint[1], plan.footprint[2], plan.footprint[3]))
                    .First(l => l.alongX == d.RunsAlongX && Mathf.Abs(l.at - (d.RunsAlongX ? d.at[1] : d.at[0])) < 1e-3f);
                Vector3 c = d.RunsAlongX ? new Vector3(d.at[0], 1.0f, line.centre) : new Vector3(line.centre, 1.0f, d.at[1]);
                bool hitA = Physics.Raycast(c, along, out var a, 5f, wallMask);
                bool hitB = Physics.Raycast(c, -along, out var b, 5f, wallMask);
                float gap = (hitA ? a.distance : 5f) + (hitB ? b.distance : 5f);
                if (Mathf.Abs(gap - d.width) > 0.02f) failures.Add($"{d.id}: gap {gap:0.000} m, expected {d.width:0.000} m");
            }
            Assert.IsEmpty(failures, string.Join("\n", failures));
        }

        [Test]
        public void CapsuleWalksThroughEveryDoorWithoutCollision()
        {
            // Robot-sized capsule: 0.36 m wide, body from 0.15 m to 0.65 m above the floor.
            const float radius = 0.18f;
            var failures = new List<string>();
            foreach (var d in plan.doors.Where(d => d.state != "closed"))
            {
                var (_, across) = Axes(d);
                Vector3 start = Centre(d, 0f) - across * 1.0f;
                Vector3 p1 = start + Vector3.up * (0.15f + radius), p2 = start + Vector3.up * (0.65f - radius);
                if (Physics.CapsuleCast(p1, p2, radius, across, out var hit, 2.0f))
                    failures.Add($"{d.id}: hit {hit.collider.name} at {hit.distance:0.00} m");
                // Negative control: the same cast beside the door goes into solid wall and must hit,
                // otherwise this test could pass on a house with no walls at all.
                var (along, _) = Axes(d);
                Vector3 offset = along * (d.width / 2 + 0.6f);
                if (!Physics.CapsuleCast(p1 + offset, p2 + offset, radius, across, out _, 2.0f))
                    failures.Add($"{d.id}: control cast beside the door hit nothing");
            }
            Assert.IsEmpty(failures, string.Join("\n", failures));
        }

        [Test]
        public void RebuildingTwiceGivesAnIdenticalHierarchy()
        {
            string first = Fingerprint(house);
            Object.DestroyImmediate(house);
            house = HouseBuilder.Build(plan, MaterialLibrary.GreyBox);
            Assert.AreEqual(first, Fingerprint(house));
        }

        [Test]
        public void WallsAreOnTheWallsLayerAndFloorsCoverThePlan()
        {
            int walls = LayerMask.NameToLayer(HouseBuilder.WallsLayer);
            var wallObjs = house.transform.Find("Walls").GetComponentsInChildren<BoxCollider>();
            Assert.That(wallObjs.Length, Is.GreaterThan(20));
            Assert.That(wallObjs.All(c => c.gameObject.layer == walls));
            float floorArea = house.transform.Find("Floors").GetComponentsInChildren<MeshFilter>()
                .Sum(m => m.sharedMesh.bounds.size.x * m.sharedMesh.bounds.size.z);
            Assert.AreEqual(280f, floorArea, 0.01f);
        }

        static string Fingerprint(GameObject root)
        {
            var sb = new StringBuilder();
            foreach (var t in root.GetComponentsInChildren<Transform>())
            {
                sb.Append(t.name).Append('|').Append(t.position.ToString("F4")).Append('|').Append(t.rotation.eulerAngles.ToString("F3"));
                var mf = t.GetComponent<MeshFilter>();
                if (mf) sb.Append('|').Append(mf.sharedMesh.bounds.ToString("F4"));
                sb.Append('\n');
            }
            return sb.ToString();
        }
    }
}
