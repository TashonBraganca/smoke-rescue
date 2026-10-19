using System.Collections.Generic;
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
            var visual = BodyPrefab != null ? QuadrupedVisual(BodyPrefab, spec.Colour, robot) : PlaceholderVisual(spec.Colour);
            visual.transform.SetParent(robot.Body, false);

            robot.CameraMount = new GameObject("CameraMount").transform;
            robot.CameraMount.SetParent(robot.Body, false);
            robot.CameraMount.localPosition = new Vector3(0f, mount.Value<float>("height_m"), mount.Value<float>("forward_offset_m"));
            robot.CameraMount.localRotation = Quaternion.Euler(mount.Value<float>("pitch_down_deg"), 0f, 0f);

            // The camera sits on the front face of the body (SIMULATION_SPEC 4.1). With a downloaded body that is
            // longer than the spec's 0.32 m offset assumes, move the mount forward so the camera never sees its own shell.
            float front = FrontOfBody(visual);
            if (front + 0.03f > robot.CameraMount.localPosition.z)
                robot.CameraMount.localPosition = new Vector3(0f, robot.CameraMount.localPosition.y, front + 0.03f);

            var lightGo = new GameObject("Headlight");
            lightGo.transform.SetParent(robot.CameraMount, false);
            var light = lightGo.AddComponent<Light>();
            light.type = LightType.Spot;
            // Floodlight, not the spec's hard 40 degree spot: the sharp edge of a 40 degree cone gave ORB a ring of
            // strong corners that move with the camera, and ORB-SLAM3 lost tracking on them during turns (clear run,
            // robot_A at 12.8 s and 74.7 s). A 100 degree cone with a broad falloff has no edge inside the 75 x 60 view.
            light.spotAngle = 100f;
            light.innerSpotAngle = 15f;
            light.range = head.Value<float>("range_m");
            light.lightUnit = UnityEngine.Rendering.LightUnit.Lumen;
            light.intensity = head.Value<float>("intensity_lumen");
            light.shadows = LightShadows.Soft;
            robot.Headlight = light;

            robot.ApplyPose();
            return robot;
        }

        /// <summary>Downloaded quadruped model (set by the editor before robots are created). Null uses the placeholder box body.</summary>
        public static GameObject BodyPrefab;
        /// <summary>Target body length in metres (SIMULATION_SPEC 4.1: about 0.7 m).</summary>
        public const float BodyLength = 0.75f;

        readonly List<(Transform t, Quaternion rest, float phase, float amp)> legJoints = new List<(Transform, Quaternion, float, float)>();

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
            // Trot: diagonal pairs (FL+RR, FR+RL) swing in opposite phase; knees lag hips by a quarter cycle.
            foreach (var (t, rest, phase, amp) in legJoints)
                t.localRotation = rest * Quaternion.Euler(amp * moving * Mathf.Sin(Driver.GaitPhase + phase), 0f, 0f);
        }

        /// <summary>Instantiates the rigged quadruped, scales it to BodyLength, puts its feet on the floor, adds a colour band.</summary>
        static GameObject QuadrupedVisual(GameObject prefab, Color colour, Robot robot)
        {
            var visual = new GameObject("Visual");
            var model = Object.Instantiate(prefab, visual.transform, false);
            model.name = "Quadruped";
            Orient(model);
            var b = Bounds(model);
            float length = Mathf.Max(b.size.z, 1e-3f);
            model.transform.localScale *= BodyLength / length;
            b = Bounds(model);
            model.transform.localPosition -= new Vector3(b.center.x, b.min.y, b.center.z);
            foreach (var t in model.GetComponentsInChildren<Transform>())
            {
                t.gameObject.layer = LayerMask.NameToLayer("Robots");
                string n = t.name;
                bool upper = n.Contains("upperLeg"), lower = n.Contains("lowerLeg");
                if (!upper && !lower) continue;
                bool pairA = n.Contains("_FL") || n.Contains("_RR");
                float phase = (pairA ? 0f : Mathf.PI) + (lower ? Mathf.PI / 2f : 0f);
                robot.legJoints.Add((t, t.localRotation, phase, upper ? 18f : 22f));
            }
            b = Bounds(model);
            Part(visual, PrimitiveType.Cube, "Band", new Vector3(0f, b.max.y + 0.005f, -0.05f), new Vector3(b.size.x * 0.5f, 0.012f, b.size.z * 0.35f), colour);
            // ID letter on both flanks (SIMULATION_SPEC 4.1), from the robot id's last character.
            string id = robot.RobotId.Length > 0 ? robot.RobotId.Substring(robot.RobotId.Length - 1) : "?";
            var tex = LetterTexture(id[0], colour);
            foreach (var side in new[] { -1f, 1f })
            {
                var q = GameObject.CreatePrimitive(PrimitiveType.Quad);
                Object.DestroyImmediate(q.GetComponent<Collider>());
                q.name = $"IdPlate_{(side < 0 ? "L" : "R")}";
                q.transform.SetParent(visual.transform, false);
                q.transform.localPosition = new Vector3(side * (b.size.x / 2f + 0.004f), b.center.y + b.size.y * 0.25f, -0.05f);
                q.transform.localRotation = Quaternion.Euler(0f, side < 0 ? 90f : -90f, 0f);
                q.transform.localScale = new Vector3(0.16f, 0.16f, 1f);
                var m = new Material(Shader.Find("HDRP/Unlit"));
                m.SetTexture("_UnlitColorMap", tex);
                m.SetColor("_UnlitColor", Color.white);
                UnityEngine.Rendering.HighDefinition.HDMaterial.ValidateMaterial(m);
                q.GetComponent<MeshRenderer>().sharedMaterial = m;
                q.layer = LayerMask.NameToLayer("Robots");
            }
            return visual;
        }

        /// <summary>
        /// Downloaded rigs come in arbitrary axes. Pick the pitch (0, 90, -90, 180 degrees) that puts the lower-leg
        /// joints lowest (feet down), then turn 180 degrees about y if the front legs (FL, FR) are behind the rear ones.
        /// </summary>
        static void Orient(GameObject model)
        {
            var lower = new List<Transform>();
            var front = new List<Transform>();
            var rear = new List<Transform>();
            foreach (var t in model.GetComponentsInChildren<Transform>())
            {
                if (t.name.Contains("lowerLeg")) lower.Add(t);
                if (t.name.Contains("upperLeg") && (t.name.Contains("_FL") || t.name.Contains("_FR"))) front.Add(t);
                if (t.name.Contains("upperLeg") && (t.name.Contains("_RL") || t.name.Contains("_RR"))) rear.Add(t);
            }
            if (lower.Count == 0) return;
            var root = model.transform;
            var start = root.localRotation;
            Quaternion best = start;
            float bestScore = float.MaxValue;
            foreach (var pitch in new[] { 0f, 90f, -90f, 180f })
            foreach (var roll in new[] { 0f, 90f, -90f })
            {
                root.localRotation = Quaternion.Euler(pitch, 0f, roll) * start;
                var b = Bounds(model);
                float feet = 0f;
                foreach (var t in lower) feet += t.position.y;
                feet /= lower.Count;
                // Lowest feet relative to the body, and the body longer along z than tall.
                float score = (feet - b.center.y) / Mathf.Max(b.size.y, 1e-3f) + (b.size.z < b.size.y ? 1f : 0f);
                if (score < bestScore) { bestScore = score; best = root.localRotation; }
            }
            root.localRotation = best;
            if (front.Count > 0 && rear.Count > 0)
            {
                float fz = 0f, rz = 0f;
                foreach (var t in front) fz += t.position.z;
                foreach (var t in rear) rz += t.position.z;
                if (fz / front.Count < rz / rear.Count) root.localRotation = Quaternion.Euler(0f, 180f, 0f) * root.localRotation;
            }
        }

        /// <summary>Forward extent (local z) of the whole body. A rigged body is often one skinned mesh, so
        /// filtering renderers by height (torso only) can miss it entirely.</summary>
        static float FrontOfBody(GameObject visual)
        {
            float front = float.MinValue;
            foreach (var r in visual.GetComponentsInChildren<Renderer>())
            {
                var b = r.bounds;
                for (int i = 0; i < 8; i++)
                {
                    var corner = new Vector3((i & 1) == 0 ? b.min.x : b.max.x, (i & 2) == 0 ? b.min.y : b.max.y, (i & 4) == 0 ? b.min.z : b.max.z);
                    front = Mathf.Max(front, visual.transform.InverseTransformPoint(corner).z);
                }
            }
            return front == float.MinValue ? 0f : front;
        }

        /// <summary>A 5x7 bitmap letter in the robot colour on black, 64 px, for the ID plates.</summary>
        static Texture2D LetterTexture(char c, Color colour)
        {
            var glyphs = new Dictionary<char, string[]>
            {
                ['A'] = new[] { "01110", "10001", "10001", "11111", "10001", "10001", "10001" },
                ['B'] = new[] { "11110", "10001", "10001", "11110", "10001", "10001", "11110" },
                ['C'] = new[] { "01111", "10000", "10000", "10000", "10000", "10000", "01111" },
                ['D'] = new[] { "11110", "10001", "10001", "10001", "10001", "10001", "11110" },
            };
            var g = glyphs.TryGetValue(char.ToUpperInvariant(c), out var rows) ? rows : glyphs['A'];
            const int n = 64;
            var tex = new Texture2D(n, n, TextureFormat.RGBA32, false) { filterMode = FilterMode.Point, name = $"Id_{c}" };
            var px = new Color[n * n];
            for (int y = 0; y < n; y++)
            for (int x = 0; x < n; x++)
            {
                int gx = (x - 12) / 8, gy = 6 - (y - 4) / 8;
                bool on = x >= 12 && gx < 5 && y >= 4 && gy >= 0 && gy < 7 && g[gy][gx] == '1';
                px[y * n + x] = on ? colour : new Color(0.06f, 0.06f, 0.07f);
            }
            tex.SetPixels(px);
            tex.Apply();
            return tex;
        }

        static Bounds Bounds(GameObject go)
        {
            var rs = go.GetComponentsInChildren<Renderer>();
            if (rs.Length == 0) return new Bounds(go.transform.position, Vector3.zero);
            var b = rs[0].bounds;
            for (int i = 1; i < rs.Length; i++) b.Encapsulate(rs[i].bounds);
            var parent = go.transform.parent;
            if (parent == null) return b;
            // Return bounds in the parent's frame (the Visual root sits at the robot origin, yaw 0 at creation).
            return new Bounds(parent.InverseTransformPoint(b.center), b.size);
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
