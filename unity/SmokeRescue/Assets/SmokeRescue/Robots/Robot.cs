using System.IO;
using Newtonsoft.Json.Linq;
using SmokeRescue.World;
using UnityEngine;

namespace SmokeRescue.Robots
{
    /// <summary>
    /// One quadruped robot in the scene. Hierarchy:
    ///   Robot_&lt;id&gt;            root, on the floor, yaw only (the kinematic body frame)
    ///     Body                  gait bob and sway applied here (visual + camera ride on it)
    ///       Visual              placeholder or downloaded quadruped mesh
    ///       CameraMount         0.5 m up, 0.32 m forward, pitched down 10 degrees (config/sensors.json)
    ///         Headlight         40 degree spot
    /// Stepped explicitly by the simulation clock via Step(dt); it never moves itself in Update.
    /// </summary>
    public class Robot : MonoBehaviour
    {
        public string RobotId { get; private set; }
        public Color Colour { get; private set; }
        public RobotDriver Driver { get; private set; }
        public Transform Body { get; private set; }
        public Transform CameraMount { get; private set; }
        public Light Headlight { get; private set; }

        /// <summary>Gait bob amplitude in metres (peak), and sway in degrees. Spec: 1 to 2 cm, 1 to 2 degrees.</summary>
        public const float BobAmplitude = 0.007f;
        public const float SwayDegrees = 1.2f;

        public static Robot Create(Scenario.Robot spec, Scenario.LookAround look, FloorPlan plan, Transform parent)
        {
            var cfg = JObject.Parse(File.ReadAllText(Path.Combine(FloorPlan.ConfigDir, "sensors.json")));
            var mount = cfg["camera"]["mount"];
            var head = cfg["headlight"];

            var root = new GameObject($"Robot_{spec.id}");
            root.transform.SetParent(parent, false);
            root.layer = LayerMask.NameToLayer("Robots");
            var robot = root.AddComponent<Robot>();
            robot.RobotId = spec.id;
            robot.Colour = spec.Colour;
            robot.Driver = new RobotDriver(spec, look, RobotLimits.Load(), plan);

            var col = root.AddComponent<CapsuleCollider>();
            col.direction = 2;                  // along z, the body's long axis
            col.radius = 0.18f;
            col.height = 0.75f;
            col.center = new Vector3(0f, 0.4f, 0f);
            var rb = root.AddComponent<Rigidbody>();
            rb.isKinematic = true;
            rb.useGravity = false;

            robot.Body = new GameObject("Body").transform;
            robot.Body.SetParent(root.transform, false);
            var visual = PlaceholderVisual(spec.Colour);
            visual.transform.SetParent(robot.Body, false);

            robot.CameraMount = new GameObject("CameraMount").transform;
            robot.CameraMount.SetParent(robot.Body, false);
            robot.CameraMount.localPosition = new Vector3(0f, mount.Value<float>("height_m"), mount.Value<float>("forward_offset_m"));
            robot.CameraMount.localRotation = Quaternion.Euler(mount.Value<float>("pitch_down_deg"), 0f, 0f);

            var lightGo = new GameObject("Headlight");
            lightGo.transform.SetParent(robot.CameraMount, false);
            var light = lightGo.AddComponent<Light>();
            light.type = LightType.Spot;
            light.spotAngle = head.Value<float>("cone_deg");
            light.innerSpotAngle = head.Value<float>("cone_deg") * 0.6f;
            light.range = head.Value<float>("range_m");
            light.lightUnit = UnityEngine.Rendering.LightUnit.Lumen;
            light.intensity = head.Value<float>("intensity_lumen");
            light.shadows = LightShadows.Soft;
            robot.Headlight = light;

            robot.ApplyPose();
            return robot;
        }

        public void Step(float dt)
        {
            Driver.Step(dt);
            ApplyPose();
        }

        public void ApplyPose()
        {
            transform.SetPositionAndRotation(new Vector3(Driver.Position.x, 0f, Driver.Position.y), Quaternion.Euler(0f, Driver.YawDeg, 0f));
            // Trot: two footfalls per gait cycle, so the body bobs at twice the gait frequency, and sways once per cycle.
            // Scaled by speed so a robot standing still (or turning in place slowly) is steady.
            float moving = Mathf.Clamp01(Mathf.Abs(Driver.Speed) / 0.3f + Mathf.Abs(Driver.YawRateDeg) / 120f);
            float bob = BobAmplitude * moving * Mathf.Sin(2f * Driver.GaitPhase);
            float sway = SwayDegrees * moving * Mathf.Sin(Driver.GaitPhase);
            Body.localPosition = new Vector3(0f, bob, 0f);
            Body.localRotation = Quaternion.Euler(0f, 0f, sway);
        }

        /// <summary>Box body, four legs and a colour band: stands in until the downloaded quadruped is wired in.</summary>
        static GameObject PlaceholderVisual(Color colour)
        {
            var visual = new GameObject("Visual");
            var dark = new Color(0.12f, 0.12f, 0.13f);
            // Torso stops at z = 0.28 and its top at y = 0.47, so the camera (y 0.5, z 0.32, 40 degrees to the
            // bottom of its view) never sees its own body.
            Part(visual, PrimitiveType.Cube, "Torso", new Vector3(0f, 0.40f, -0.035f), new Vector3(0.3f, 0.14f, 0.63f), new Color(0.82f, 0.8f, 0.74f));
            Part(visual, PrimitiveType.Cube, "Band", new Vector3(0f, 0.40f, -0.035f), new Vector3(0.305f, 0.05f, 0.4f), colour);
            Part(visual, PrimitiveType.Cube, "SensorHead", new Vector3(0f, 0.44f, 0.29f), new Vector3(0.2f, 0.08f, 0.06f), dark);
            foreach (var x in new[] { -0.12f, 0.12f })
                foreach (var z in new[] { -0.27f, 0.27f })
                    Part(visual, PrimitiveType.Cylinder, "Leg", new Vector3(x, 0.17f, z), new Vector3(0.05f, 0.17f, 0.05f), dark);
            return visual;
        }

        static void Part(GameObject parent, PrimitiveType type, string name, Vector3 pos, Vector3 scale, Color c)
        {
            var go = GameObject.CreatePrimitive(type);
            go.name = name;
            Object.DestroyImmediate(go.GetComponent<Collider>());
            go.transform.SetParent(parent.transform, false);
            go.transform.localPosition = pos;
            go.transform.localScale = scale;
            go.layer = LayerMask.NameToLayer("Robots");
            var mat = new Material(Shader.Find("HDRP/Lit"));
            mat.SetColor("_BaseColor", c);
            go.GetComponent<MeshRenderer>().sharedMaterial = mat;
        }
    }
}
