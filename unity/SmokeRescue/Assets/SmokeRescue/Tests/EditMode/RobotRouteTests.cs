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
        public void OneGateGivesTheBrainsThePadsNeverTheHouseCoordinates()
        {
            // One entrance (22 Oct): both brains start in the gate frame from their pads. The spawn's house coordinates
            // (10, -2.4 here, plus placement error) must never reach a brain. Controls: a robot without a pad is not a gate
            // launch, and asking for its gate pose fails rather than falling back to house coordinates.
            var a = new SmokeRescue.World.Scenario.Robot { id = "robot_A", spawn = new SmokeRescue.World.Scenario.Spawn { at = new[] { 10f, -1.2f }, yaw = 0 }, launch_pad = new[] { 0f, 0f, 0f } };
            var b = new SmokeRescue.World.Scenario.Robot { id = "robot_B", spawn = new SmokeRescue.World.Scenario.Spawn { at = new[] { 10.04f, -2.43f }, yaw = 1.3f }, launch_pad = new[] { 0f, -1.2f, 0f } };
            Assert.That(SmokeRescue.Recording.RecordingSession.GateLaunch(new[] { a, b }), Is.True);
            var pose = SmokeRescue.Recording.RecordingSession.EntryFor(b).pose;
            Assert.That(pose.position.x, Is.EqualTo(0f).Within(1e-6f)); Assert.That(pose.position.y, Is.EqualTo(-1.2f).Within(1e-6f));
            Assert.That(pose.yawRad, Is.EqualTo(0f).Within(1e-6f), "the pad's heading, not the placement error");
            var c = new SmokeRescue.World.Scenario.Robot { id = "robot_C", spawn = new SmokeRescue.World.Scenario.Spawn { at = new[] { 3f, 4f }, yaw = 0 } };
            Assert.That(SmokeRescue.Recording.RecordingSession.GateLaunch(new[] { a, c }), Is.False, "control: separate launches");
            Assert.Throws<System.ArgumentException>(() => SmokeRescue.Recording.RecordingSession.EntryFor(c));
        }

        [Test]
        public void ARadioMessageAsksForADecisionUnlessSomethingMoreUrgentWaits()
        {
            // Codex design review (22 Oct): a partner's proposal or report waited in the inbox for some other event or the
            // 20 s period. Now it asks for a decision; a waiting person or heat event keeps its place (control).
            var message = new Newtonsoft.Json.Linq.JObject { ["from"] = "robot_B", ["kind"] = "proposal", ["text"] = "split" };
            var asked = SmokeRescue.Recording.RecordingSession.MessageTrigger(null, message);
            Assert.That(asked.Value<string>("kind"), Is.EqualTo("team_message"));
            Assert.That(asked.Value<string>("reason"), Does.Contain("robot_B").And.Contain("proposal"));
            var urgent = Newtonsoft.Json.Linq.JObject.FromObject(new { kind = "avoid_fire", reason = "heat", numbers = new { } });
            Assert.That(SmokeRescue.Recording.RecordingSession.MessageTrigger(urgent, message), Is.SameAs(urgent), "control: heat first");
            var periodic = Newtonsoft.Json.Linq.JObject.FromObject(new { kind = "no_progress", reason = "x", numbers = new { } });
            Assert.That(SmokeRescue.Recording.RecordingSession.MessageTrigger(periodic, message).Value<string>("kind"), Is.EqualTo("team_message"));
        }

        [Test]
        public void AnArrivalTheAgentAskedForIsDecidedAtOnceOtherEventsWaitTheGap()
        {
            // Held-out seeds 915 to 917: robots stood 28 to 58 s per run after agent turns, waiting out the 4 s gap between
            // agent calls and then a 5 s hold. An arrival (or the candidate view the agent set up) is decided at once;
            // controls: other events still wait the gap, nothing is asked without an event before the period.
            Assert.That(SmokeRescue.Recording.RecordingSession.AgentDue("agent_goal_reached", 1.0), Is.True);
            Assert.That(SmokeRescue.Recording.RecordingSession.AgentDue("survivor_view_ready", .5), Is.True);
            Assert.That(SmokeRescue.Recording.RecordingSession.AgentDue("survivor_candidate", 1.0), Is.False, "control: other events wait the gap");
            Assert.That(SmokeRescue.Recording.RecordingSession.AgentDue("survivor_candidate", 4.5), Is.True);
            Assert.That(SmokeRescue.Recording.RecordingSession.AgentDue(null, 10), Is.False, "control: no event, no call before the period");
            Assert.That(SmokeRescue.Recording.RecordingSession.AgentDue(null, 20), Is.True);
        }

        [Test]
        public void RandomPeopleArePlacedOncePerScenarioSoTheTruthFileMatchesTheScene()
        {
            // Held-out seed 914 runs 3 and 4: the scenario was loaded to build the scene and again inside the recorder; the
            // second load saw the people the first had placed and chose other spots, so oracle/survivors.json disagreed with
            // the scene by up to 4.7 m. A random placement is now made once per file and seed and reused.
            // Control: forgetting the placement lets a changed scene move the people.
            const string file = "scenario_house_heldout_seed914.json";
            try
            {
                Scenario.ForgetRandomPlacements(); Scenario.PlacementBlocked = null;
                var first = Scenario.Load(file).survivors.Select(s => new Vector2(s.at[0], s.at[1])).ToList();
                Scenario.PlacementBlocked = q => Vector2.Distance(q, first[0]) < 2f;   // the scene changed: person 0's spot is now taken
                var again = Scenario.Load(file).survivors.Select(s => new Vector2(s.at[0], s.at[1])).ToList();
                Assert.That(again, Is.EqualTo(first), "the second load reuses the first placement");
                Scenario.ForgetRandomPlacements();
                var fresh = Scenario.Load(file).survivors.Select(s => new Vector2(s.at[0], s.at[1])).ToList();
                Assert.That(Vector2.Distance(fresh[0], first[0]), Is.GreaterThanOrEqualTo(2f), "control: placed afresh, the changed scene moves person 0");
            }
            finally { Scenario.PlacementBlocked = null; Scenario.ForgetRandomPlacements(); }
        }

        [Test]
        public void RandomPeopleReachEveryRoomInProportionToItsFloor()
        {
            // One-door runs 922 to 932: someone was in the laundry in all nine runs and no one ever in the bathroom, closet or
            // hallway: nearby seeds gave correlated first draws, and the room was drawn before the point. Over 300 seeds each
            // room in the search area must get people, none more than 1.6 times its share of floor (sampling noise at n = 1500
            // is about 0.1 of a share).
            var saved = Scenario.PlacementBlocked; Scenario.PlacementBlocked = null;
            try
            {
                var scenario = Scenario.Load("scenario_house_onedoor_seed932_nolimit.json");
                var plan = FloorPlan.Load(); var a = scenario.search_area;
                var rooms = plan.rooms.Where(r => r.Rect.xMin >= a[0] - 1e-3f && r.Rect.yMin >= a[1] - 1e-3f && r.Rect.xMax <= a[2] + 1e-3f && r.Rect.yMax <= a[3] + 1e-3f).ToList();
                float total = rooms.Sum(r => (r.Rect.width - 1.2f) * (r.Rect.height - 1.2f));
                var counts = rooms.ToDictionary(r => r.id, r => 0); int n = 0;
                var place = typeof(Scenario).GetMethod("PlaceSurvivorsRandomly", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                for (int seed = 1; seed <= 300; seed++)
                {
                    scenario.seed = seed; place.Invoke(scenario, new object[] { plan });
                    foreach (var p in scenario.survivors) { counts[p.room]++; n++; }
                }
                var report = string.Join(", ", rooms.Select(r => $"{r.id} {counts[r.id]} ({100.0 * counts[r.id] / n:F1} % vs floor {100.0 * (r.Rect.width - 1.2f) * (r.Rect.height - 1.2f) / total:F1} %)"));
                foreach (var r in rooms)
                {
                    double share = counts[r.id] / (double)n, floor = (r.Rect.width - 1.2f) * (r.Rect.height - 1.2f) / total;
                    Assert.That(counts[r.id], Is.GreaterThan(0), $"{r.id} never gets anyone: {report}");
                    Assert.That(share, Is.LessThan(1.6 * floor + .01), $"{r.id} over-represented: {report}");
                }
                TestContext.WriteLine(report);
            }
            finally { Scenario.PlacementBlocked = saved; Scenario.ForgetRandomPlacements(); }
        }

        [Test]
        public void RandomlyPlacedPeopleStayInsideTheScenarioSearchArea()
        {
            // Held-out run 1: four of five randomly placed people landed in the apartment, office and workshop of the larger
            // building, beyond the house the robots search. Every random person must be inside search_area.
            var scenario = Scenario.Load("scenario_house_heldout.json");
            Assert.That(scenario.survivor_placement, Is.EqualTo("random"));
            var a = scenario.search_area;
            foreach (var s in scenario.survivors)
                Assert.That(s.at[0] >= a[0] && s.at[0] <= a[2] && s.at[1] >= a[1] && s.at[1] <= a[3], Is.True,
                    $"{s.id} at ({s.at[0]:F2}, {s.at[1]:F2}) in {s.room}");
        }

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
                               $"maxV={maxV:0.000} maxW={maxW:0.0}deg/s maxA={maxA:0.00} bob={maxBob * 100:0.00}cm clearance={closest:0.000}m hits=[{string.Join(",", hits)}] " +
                               $"at=({robot.Driver.Position.x:0.00},{robot.Driver.Position.y:0.00}) target={robot.Driver.TargetIndex} " +
                               $"blocked={robot.Driver.CollisionSteps * Dt:0.0}s");
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
