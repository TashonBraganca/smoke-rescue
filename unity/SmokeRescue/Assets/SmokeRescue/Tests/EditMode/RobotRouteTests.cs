using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using SmokeRescue.Editor;
using SmokeRescue.Robots;
using SmokeRescue.World;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    /// <summary>C3 acceptance: both robots drive their full haze routes within limits, without touching walls or props.</summary>
    public class RobotRouteTests
    {
        const float Dt = 1f / 60f;

        [Test]
        public void BothRobotsDriveTheirRoutesWithinLimitsWithoutCollisions() => DriveRoutes(withFurniture: false);

        /// <summary>Same routes with every landmark and survivor in place (needs assets_cache/ from tools/fetch_*.py).</summary>
        [Test]
        public void BothRobotsMissAllFurnitureAndSurvivors() => DriveRoutes(withFurniture: true);

        static void DriveRoutes(bool withFurniture)
        {
            ProjectSetup.EnsureLayers();
            var plan = FloorPlan.Load();
            var scenario = Scenario.Load("scenario_haze_2agents.json");
            var limits = RobotLimits.Load();
            var existingHouse = GameObject.Find(HouseBuilder.RootName);
            if (existingHouse != null) existingHouse.SetActive(false);
            var house = HouseBuilder.Build(plan, MaterialLibrary.GreyBox);
            GameObject furniture = null, survivors = null;
            var existingFurniture = GameObject.Find(FurnitureBuilder.RootName);
            var existingSurvivors = GameObject.Find(ScenePreview.SurvivorsRoot);
            if (withFurniture)
            {
                if (existingFurniture != null) existingFurniture.SetActive(false);
                if (existingSurvivors != null) existingSurvivors.SetActive(false);
                furniture = FurnitureBuilder.Place(plan);
                ScenePreview.PlaceSurvivors();
                survivors = GameObject.Find(ScenePreview.SurvivorsRoot);
                Physics.SyncTransforms();
                // Control: furniture must be visible to the overlap queries, or this test proves nothing.
                // The living-room sofa landmark stands at (3.0, 1.2); a robot-sized capsule there must hit it.
                int props = LayerMask.GetMask(HouseBuilder.PropsLayer);
                var hit = Physics.OverlapCapsule(new Vector3(3f, 0.3f, 1.0f), new Vector3(3f, 0.3f, 1.4f), 0.18f, props);
                Assert.IsNotEmpty(hit, "control failed: no Props collider at the sofa, furniture is not in the physics scene");
            }
            var root = new GameObject("Robots");
            try
            {
                Physics.SyncTransforms();
                int mask = LayerMask.GetMask(HouseBuilder.WallsLayer, HouseBuilder.PropsLayer);
                var report = new List<string>();
                foreach (var spec in scenario.robots)
                {
                    var robot = Robot.Create(spec, scenario.look_around, plan, root.transform);
                    float t = 0f, prevV = 0f, prevYaw = robot.Driver.YawDeg;
                    float maxV = 0f, maxW = 0f, maxA = 0f, maxBob = 0f, closest = float.MaxValue;
                    var hits = new HashSet<string>();
                    while (!robot.Driver.Finished && t < 600f)
                    {
                        robot.Step(Dt);
                        t += Dt;
                        var d = robot.Driver;
                        maxV = Mathf.Max(maxV, d.Speed);
                        maxW = Mathf.Max(maxW, Mathf.Abs(Mathf.DeltaAngle(prevYaw, d.YawDeg)) / Dt);
                        maxA = Mathf.Max(maxA, Mathf.Abs(d.Speed - prevV) / Dt);
                        maxBob = Mathf.Max(maxBob, Mathf.Abs(robot.Body.localPosition.y));
                        prevV = d.Speed; prevYaw = d.YawDeg;
                        // Body footprint: 0.36 m wide capsule from 0.1 m to 0.6 m height, 0.75 m long.
                        Vector3 c = robot.transform.position, f = robot.transform.forward;
                        Vector3 p1 = c + Vector3.up * 0.3f - f * 0.2f, p2 = c + Vector3.up * 0.3f + f * 0.2f;
                        foreach (var col in Physics.OverlapCapsule(p1, p2, 0.18f, mask)) hits.Add(col.name);
                        foreach (var col in Physics.OverlapSphere(c + Vector3.up * 0.3f, 0.6f, mask))
                            closest = Mathf.Min(closest, Vector3.Distance(col.ClosestPoint(c + Vector3.up * 0.3f), c + Vector3.up * 0.3f));
                    }
                    report.Add($"{spec.id}: finished={robot.Driver.Finished} t={t:0.0}s dist={robot.Driver.Distance:0.0}m " +
                               $"maxV={maxV:0.000} maxW={maxW:0.0}deg/s maxA={maxA:0.00} bob={maxBob * 100:0.00}cm clearance={closest:0.000}m hits=[{string.Join(",", hits)}]");
                    Assert.IsTrue(robot.Driver.Finished, $"{spec.id} did not finish its route: {report.Last()}");
                    Assert.LessOrEqual(t, scenario.duration_s, $"{spec.id} route longer than the scenario: {report.Last()}");
                    Assert.LessOrEqual(maxV, limits.maxForward + 1e-4f, report.Last());
                    Assert.LessOrEqual(maxW, limits.maxYawRateDeg + 0.5f, report.Last());
                    Assert.LessOrEqual(maxA, limits.maxAccel + 0.01f, report.Last());
                    Assert.Less(maxBob, 0.02f, report.Last());
                    Assert.Greater(maxBob, 0.003f, $"gait bob not visible: {report.Last()}");
                    Assert.IsEmpty(hits, report.Last());
                }
                Debug.Log($"[SmokeRescue] C3 route report (furniture {withFurniture})\n" + string.Join("\n", report));
            }
            finally
            {
                Object.DestroyImmediate(root);
                Object.DestroyImmediate(house);
                if (existingHouse != null) existingHouse.SetActive(true);
                if (withFurniture)
                {
                    if (furniture != null) Object.DestroyImmediate(furniture);
                    if (survivors != null) Object.DestroyImmediate(survivors);
                    // Rebuild the scene's own copies so the test leaves the open scene as it found it.
                    if (existingFurniture != null) { Object.DestroyImmediate(existingFurniture); FurnitureBuilder.Place(plan); }
                    if (existingSurvivors != null) { Object.DestroyImmediate(existingSurvivors); ScenePreview.PlaceSurvivors(); }
                }
            }
        }
    }
}
