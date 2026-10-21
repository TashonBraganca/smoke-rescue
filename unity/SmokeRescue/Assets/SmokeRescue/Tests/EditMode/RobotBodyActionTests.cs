using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SmokeRescue.Editor;
using SmokeRescue.Robots;
using SmokeRescue.World;
using UnityEngine;
using Object = UnityEngine.Object;

namespace SmokeRescue.Tests.EditMode
{
    public class RobotBodyActionTests
    {
        GameObject root;
        readonly Vector3 origin = new Vector3(1000, 0, 1000);
        [SetUp] public void SetUp() { ProjectSetup.EnsureLayers(); root = new GameObject("Body action tests"); }
        [TearDown] public void TearDown() { Object.DestroyImmediate(root); Physics.SyncTransforms(); }

        Robot MakeRobot()
        {
            return Robot.Create(new Scenario.Robot { id = "test", color = "#FFFFFF",
                spawn = new Scenario.Spawn { at = new[] { origin.x, origin.z }, yaw = 0 },
                route = new List<float[]> { new[] { origin.x, origin.z + 3 } } },
                new Scenario.LookAround(), FloorPlan.Load(), root.transform);
        }

        [Test] public void TorchModesAndAimHavePhysicalEffects()
        {
            var r = MakeRobot(); r.SetTorch("wide", 0, 0);
            float wide = r.Headlight.spotAngle, power = r.Headlight.intensity, range = r.Headlight.range;
            r.SetTorch("narrow", -100, 100);
            Assert.That(r.Headlight.spotAngle, Is.LessThan(wide));
            Assert.That(r.Headlight.range, Is.GreaterThan(range));
            Assert.That(r.TorchPitchDeg, Is.EqualTo(-30)); Assert.That(r.TorchYawDeg, Is.EqualTo(45));
            Assert.That(Vector3.Angle(r.Headlight.transform.localRotation * Vector3.forward,
                Quaternion.Euler(30, 45, 0) * Vector3.forward), Is.LessThan(.001));
            r.SetTorch("low", 0, 0);
            Assert.That(r.Headlight.intensity, Is.LessThan(power)); Assert.That(r.TorchPitchDeg, Is.LessThan(0));
            r.SetTorch("off", 100, -100); Assert.That(r.Headlight.enabled, Is.False);
            Assert.That(r.TorchPitchDeg, Is.EqualTo(15)); Assert.That(r.TorchYawDeg, Is.EqualTo(-45));
            r.SetTorch("wide", 0, 0); Assert.That(r.Headlight.enabled, Is.True);
            Assert.Throws<ArgumentException>(() => r.SetTorch("laser", 0, 0));
            Assert.Throws<ArgumentOutOfRangeException>(() => r.SetTorch("wide", float.NaN, 0));
        }

        [TestCase(.5f, .32f)] [TestCase(1.6f, .25f)]
        public void StowedArmIsOutsideBothCameraFrusta(float height, float forward)
        {
            var r = MakeRobot(); r.CameraMount.localPosition = new Vector3(0, height, forward);
            r.Arm.Stow();
            var cfg = JObject.Parse(System.IO.File.ReadAllText(System.IO.Path.Combine(FloorPlan.ConfigDir, "sensors.json")));
            foreach (string name in new[] { "camera", "thermal" })
            {
                var go = new GameObject(name); go.transform.SetParent(r.CameraMount, false);
                if (name == "thermal") go.transform.localPosition = Vector3.right * cfg[name].Value<float>("mount_right_offset_m");
                var camera = go.AddComponent<Camera>(); camera.enabled = false;
                camera.fieldOfView = cfg[name].Value<float>("vertical_fov_deg");
                camera.aspect = cfg[name].Value<float>("width") / cfg[name].Value<float>("height");
                camera.nearClipPlane = name == "camera" ? .01f : .05f; camera.farClipPlane = 60;
                var planes = GeometryUtility.CalculateFrustumPlanes(camera);
                foreach (var renderer in r.Arm.Root.GetComponentsInChildren<Renderer>())
                {
                    Assert.That(GeometryUtility.TestPlanesAABB(planes, renderer.bounds), Is.False, name + " sees " + renderer.name);
                    Assert.That(renderer.gameObject.layer, Is.EqualTo(LayerMask.NameToLayer("Robots")));
                }
            }
            Assert.That(r.Arm.Root.GetComponentsInChildren<Collider>(), Is.Empty);
        }

        [Test] public void ArmIkReachesAWorldPointAndRejectsAnUnreachablePoint()
        {
            var r = MakeRobot(); var target = r.Arm.Root.TransformPoint(new Vector3(.3f, .3f, .5f));
            Assert.That(r.Arm.Reach(target), Is.True);
            Assert.That(Vector3.Distance(r.Arm.GripperPosition, target), Is.LessThan(.001f));
            Assert.That(r.Arm.Reach(origin + Vector3.up * 10), Is.False);
            r.Arm.Stow(); Assert.That(r.Arm.IsStowed, Is.True);
        }

        ObstacleBody Obstacle(string id, string kind, Vector3 position)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube); go.name = id;
            go.transform.SetParent(root.transform); go.transform.position = position;
            go.transform.localScale = new Vector3(.3f, .4f, .3f); go.layer = LayerMask.NameToLayer("Props");
            var body = go.AddComponent<ObstacleBody>(); body.Initialize(id, kind, ObstacleRuntime.LoadDefinitions());
            return body;
        }

        [Test] public void CrateSlidesPointSixMetresAndRubbleRefuses()
        {
            var crate = Obstacle("crate", "obstacle_burnt_crate", origin + new Vector3(0, .2f, .6f));
            Obstacle("rubble", "obstacle_rubble", origin + new Vector3(3, .2f, 0));
            var runtime = new ObstacleRuntime(root.transform);
            Assert.That(runtime.TryFindMovableAhead(origin, 0, 1)?.Id, Is.EqualTo("crate"));
            Assert.That(runtime.TryFindMovableAhead(origin, 180, 1), Is.Null);
            var start = crate.transform.position;
            Assert.That(runtime.Push("crate", Vector2.up, .6f, 1), Is.True, runtime.FailureReason);
            Assert.That(crate.transform.position, Is.EqualTo(start)); runtime.Step(.5f);
            Assert.That(crate.transform.position.z, Is.EqualTo(start.z + .3f).Within(.0001)); runtime.Step(.5f);
            Assert.That(crate.transform.position.z, Is.EqualTo(start.z + .6f).Within(.0001));
            Assert.That(runtime.Push("rubble", Vector2.up, .6f, 1), Is.False);
            StringAssert.Contains("heavy", runtime.FailureReason);
        }

        [Test] public void PushRefusesAWallAtDestinationOrAcrossItsPath()
        {
            var crate = Obstacle("crate", "obstacle_burnt_crate", origin + new Vector3(0, .2f, .6f));
            var wall = GameObject.CreatePrimitive(PrimitiveType.Cube); wall.transform.SetParent(root.transform);
            wall.transform.position = origin + new Vector3(0, 1, 1.2f);
            wall.transform.localScale = new Vector3(2, 2, .02f); wall.layer = LayerMask.NameToLayer("Walls");
            var runtime = new ObstacleRuntime(root.transform); var start = crate.transform.position;
            Assert.That(runtime.Push("crate", Vector2.up, .6f, 1), Is.False); StringAssert.Contains("blocked", runtime.FailureReason);
            Assert.That(runtime.Push("crate", Vector2.up, 2, 1), Is.False);
            runtime.Step(10); Assert.That(crate.transform.position, Is.EqualTo(start));
        }

        [Test] public void ExtractGripsTheNearestPartOfALyingPerson()
        {
            // Run 26: the arm aimed at the centre of a lying person's bounds, a metre past their near edge, and failed with
            // "contact point is out of the arm's reach". It grips the nearest part of the body. Control: a person whose near
            // edge is beyond reach_m is refused.
            string Lift(float nearEdgeM)
            {
                var r = MakeRobot();
                var people = new GameObject("Survivors"); people.transform.SetParent(root.transform);
                var person = new GameObject("test_person"); person.transform.SetParent(people.transform);
                var box = person.AddComponent<BoxCollider>(); box.size = new Vector3(.5f, .3f, 1.7f);   // lying along the heading
                person.transform.position = r.transform.position + new Vector3(0, .15f, nearEdgeM + .85f);
                Physics.SyncTransforms();
                var actions = new RobotActions(r, null, null, null, new RescueConfig { reach_m = 1.6f, grip_s = 2f, tow_offset_m = .9f, carry_speed_mps = .25f });
                if (!actions.BeginExtract()) { Object.DestroyImmediate(people); return "refused: " + actions.FailureReason; }
                for (int i = 0; i < 60 && actions.IsRunning; i++) { r.Step(.1f); actions.Step(.1f); }
                string result = actions.Status + (actions.FailureReason != null ? ": " + actions.FailureReason : "");
                Object.DestroyImmediate(people);
                return result;
            }
            Assert.That(Lift(.7f), Is.EqualTo("succeeded"), "near edge within the arm's links");
            Assert.That(Lift(1.9f), Does.StartWith("refused"), "control: near edge beyond the arm");
        }

        [Test] public void ClearDebrisHoldsDriverThenResumesAndSequenceRepeatsExactly()
        {
            var first = RunClear(); Object.DestroyImmediate(root); root = new GameObject("Repeated body action tests");
            var second = RunClear(); CollectionAssert.AreEqual(first, second);
        }

        List<Vector3> RunClear()
        {
            var r = MakeRobot(); var crate = Obstacle("crate", "obstacle_burnt_crate", origin + new Vector3(0, .2f, .65f));
            var runtime = new ObstacleRuntime(root.transform); var actions = new RobotActions(r, null, runtime);
            Assert.That(actions.Begin("clear_debris"), Is.True, actions.FailureReason);
            var poses = new List<Vector3>(); Vector3 initial = r.transform.position;
            for (int i = 0; i < 45; i++)
            {
                r.Step(.1f); runtime.Step(.1f); actions.Step(.1f);
                poses.Add(crate.transform.position); poses.Add(r.Arm.GripperPosition); poses.Add(r.transform.position);
                if (actions.IsRunning) Assert.That(r.transform.position, Is.EqualTo(initial));
            }
            Assert.That(actions.Status, Is.EqualTo("succeeded"), actions.FailureReason);
            Assert.That(crate.transform.position.z, Is.EqualTo(origin.z + 1.25f).Within(.0002));
            Assert.That(r.Driver.HoldPosition, Is.False); r.Step(.1f); Assert.That(r.Driver.Speed, Is.GreaterThan(0));
            return poses;
        }
    }
}
