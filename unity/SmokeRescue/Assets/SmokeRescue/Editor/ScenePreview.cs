using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using SmokeRescue.Robots;
using SmokeRescue.Sensors;
using SmokeRescue.World;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

namespace SmokeRescue.Editor
{
    /// <summary>Evidence renders for the robot body and survivors, and survivor placement.</summary>
    public static class ScenePreview
    {
        public const string SurvivorsRoot = "Survivors";

        [MenuItem("Tools/Smoke Rescue/Capture Robot Preview")]
        public static void RobotPreview()
        {
            var plan = FloorPlan.Load();
            var scenario = Scenario.Load("scenario_haze_2agents.json");
            Robot.BodyPrefab = FurnitureBuilder.RobotBodyPrefab();
            var root = new GameObject("RobotPreview");
            try
            {
                var robot = Robot.Create(scenario.robots[0], scenario.look_around, plan, root.transform);
                for (int i = 0; i < 600 * 9; i++) robot.Step(1f / 600f);   // 9 s along robot_A's route: mid-foyer, mid-stride
                var p = robot.transform.position;
                var side = robot.transform.right;
                EvidenceCapture.Render("c3_robot_side.png", p + side * 1.6f + Vector3.up * 0.6f,
                    Quaternion.LookRotation(p + Vector3.up * 0.35f - (p + side * 1.6f + Vector3.up * 0.6f)), 50f, 960, 720);
                EvidenceCapture.Render("c3_robot_front.png", p + robot.transform.forward * 1.8f + Vector3.up * 0.7f,
                    Quaternion.LookRotation(p + Vector3.up * 0.35f - (p + robot.transform.forward * 1.8f + Vector3.up * 0.7f)), 50f, 960, 720);
                for (int k = 0; k < 4; k++)
                {
                    for (int i = 0; i < 25; i++) robot.Step(1f / 600f);
                    EvidenceCapture.Render($"c3_gait_{k}.png", p + side * 1.4f + Vector3.up * 0.4f,
                        Quaternion.LookRotation(robot.transform.position + Vector3.up * 0.3f - (p + side * 1.4f + Vector3.up * 0.4f)), 45f, 480, 360);
                }
                // Its own camera: the body must not appear in the sensor image.
                var camGo = new GameObject("SensorCamera");
                camGo.transform.SetParent(robot.CameraMount, false);
                camGo.AddComponent<Camera>();
                camGo.AddComponent<HDAdditionalCameraData>();
                var rgbd = camGo.AddComponent<RgbdCamera>();
                rgbd.Initialize(SensorConfig.Load(), 3);
                rgbd.NoiseEnabled = false;
                var vb = new Bounds(robot.transform.position, Vector3.zero);
                foreach (var r in robot.GetComponentsInChildren<Renderer>()) vb.Encapsulate(r.bounds);
                var ray = new Ray(rgbd.transform.position, rgbd.transform.forward);
                string hitName = "nothing";
                foreach (var h in Physics.RaycastAll(ray, 1.0f)) hitName += $" | {h.collider.name} at {h.distance:0.000}";
                var rends = new System.Text.StringBuilder();
                foreach (var r in robot.GetComponentsInChildren<Renderer>())
                {
                    var lp = rgbd.transform.InverseTransformPoint(r.bounds.center);
                    if (lp.z > 0f && lp.z < 0.6f) rends.Append($"{r.name} centre_cam=({lp.x:0.00},{lp.y:0.00},{lp.z:0.00}) size={r.bounds.size}; ");
                }
                Debug.Log($"[SmokeRescue] mount local {robot.CameraMount.localPosition} robot bounds {vb} cam world {rgbd.transform.position} fwd {rgbd.transform.forward} raycast {hitName}\nrenderers in front of camera: {rends}");
                var f = rgbd.CaptureSync(0);
                var png = SensorPng.EncodeRgbAsync(f.Rgb, f.Width, f.Height).Result;
                File.WriteAllBytes(Path.Combine(EvidenceCapture.EvidenceDir, "c3_robot_own_view.png"), png);
                Debug.Log($"[SmokeRescue] Robot preview written; leg joints animated: {robot.GetComponentsInChildren<Transform>().Count(t => t.name.Contains("upperLeg") || t.name.Contains("lowerLeg"))}");
            }
            finally { Object.DestroyImmediate(root); }
        }

        /// <summary>
        /// Robot's-eye views: a sensor camera (scene lighting, exposure loop warmed up over 12 frames, noise off) at each
        /// pose in data/evidence/custom_views.json; writes data/evidence/sensor/&lt;name&gt;.png. Builds the haze atmosphere.
        /// </summary>
        [MenuItem("Tools/Smoke Rescue/Capture Sensor Views")]
        public static void SensorViews()
        {
            var plan = FloorPlan.Load();
            var sc = Scenario.Load("scenario_haze_2agents.json");
            FurnitureBuilder.PlaceScenarioObstacles(sc);
            var atm = Atmosphere.Build(plan, sc);
            var views = JArray.Parse(File.ReadAllText(Path.Combine(EvidenceCapture.EvidenceDir, "custom_views.json")));
            var dir = Path.Combine(EvidenceCapture.EvidenceDir, "sensor");
            Directory.CreateDirectory(dir);
            var go = new GameObject("SensorViewCamera");
            try
            {
                go.AddComponent<Camera>();
                go.AddComponent<HDAdditionalCameraData>();
                var rgbd = go.AddComponent<RgbdCamera>();
                rgbd.Initialize(SensorConfig.Load(), 1);
                rgbd.NoiseEnabled = false;
                rgbd.LocalDensity = atm.Field.Density;
                var light = new GameObject("Headlight").AddComponent<Light>();
                light.transform.SetParent(go.transform, false);
                light.type = LightType.Spot; light.spotAngle = 100f; light.innerSpotAngle = 15f; light.range = 10f;
                light.lightUnit = UnityEngine.Rendering.LightUnit.Lumen; light.intensity = 900f;
                foreach (JObject v in views)
                {
                    var e = v["eye"].ToObject<float[]>(); var t = v["target"].ToObject<float[]>();
                    var eye = new Vector3(e[0], e[1], e[2]);
                    go.transform.SetPositionAndRotation(eye, Quaternion.LookRotation(new Vector3(t[0], t[1], t[2]) - eye));
                    RgbdFrame f = null;
                    atm.Apply(119f);   // let flames, smoke and heat build up before the frames that count
                    for (int i = 0; i < 12; i++) { atm.Apply(120f + i / 15f); f = rgbd.CaptureSync(120.0 + i / 15.0); }
                    File.WriteAllBytes(Path.Combine(dir, v.Value<string>("name") + ".png"), SensorPng.EncodeRgbAsync(f.Rgb, f.Width, f.Height).Result);
                }
            }
            finally { Object.DestroyImmediate(go); }
        }

        /// <summary>Survivors from the haze scenario, laid on their side on the floor (the downloaded figures are standing;
        /// proper lying, slumped and curled poses need rig posing and are logged as a known issue).</summary>
        /// <summary>People are solid. The glTF importer brought skin and clothes in alpha-blended (11 of 12 survivor
        /// sub-meshes transparent, no depth write), so the sensor depth pass saw straight through every person: one-door
        /// seed 922, a woman kneeling 2.8 m from robot A was absent from its depth image, her colour detection took the
        /// burning sofa behind her as its range (4.45 m), and the robot walked inside its camera's blind floor before
        /// looking again. Each such material is replaced on this instance (the imported asset is untouched) by an opaque
        /// copy with alpha clipping, so hair cards stay cut. Returns how many materials changed.</summary>
        /// <summary>An opaque HDRP/Lit copy of a glTF metallic-roughness material: base colour (texture and factor, alpha
        /// forced to 1), normal map, metallic and smoothness, double-sided as the source; alpha clipping at 0.5 keeps hair and
        /// lash cut-outs.</summary>
        static Material OpaqueLit(Material m)
        {
            var lit = new Material(Shader.Find("HDRP/Lit")) { name = m.name + " (solid lit)" };
            Texture T(string n) => m.HasProperty(n) ? m.GetTexture(n) : null;
            var col = m.HasProperty("baseColorFactor") ? m.GetColor("baseColorFactor") : Color.white; col.a = 1;
            lit.SetColor("_BaseColor", col);
            if (T("baseColorTexture") != null) lit.SetTexture("_BaseColorMap", T("baseColorTexture"));
            if (T("normalTexture") != null) lit.SetTexture("_NormalMap", T("normalTexture"));
            lit.SetFloat("_Metallic", m.HasProperty("metallicFactor") ? m.GetFloat("metallicFactor") : 0f);
            lit.SetFloat("_Smoothness", 1f - (m.HasProperty("roughnessFactor") ? m.GetFloat("roughnessFactor") : .7f));
            lit.SetFloat("_AlphaCutoff", .5f);
            lit.SetFloat("_DoubleSidedEnable", 1f);
            return lit;
        }
        public static int SolidBodies(GameObject go)
        {
            int changed = 0;
            foreach (var r in go.GetComponentsInChildren<Renderer>(true))
            {
                var mats = r.sharedMaterials; bool any = false;
                for (int i = 0; i < mats.Length; i++)
                {
                    var m = mats[i];
                    if (m == null || !m.HasProperty("_SurfaceType") || m.GetFloat("_SurfaceType") == 0) continue;
                    // A transparent glTF shader-graph material ignores a per-material opaque override: its own graph keeps
                    // drawing the transparent pass, so a body was solid to the depth camera (depth write on) and a near-
                    // invisible ghost to the colour camera and the colour person detector (seed 926: the "Debra" model, s2 and
                    // s6, all nine materials alphaMode BLEND; the evidence render showed nothing where she lay). Draw such
                    // bodies with HDRP/Lit, opaque, from the same textures; alpha-clip only where the texture has cut-outs.
                    var solid = m.shader.name.StartsWith("Shader Graphs/glTF") ? OpaqueLit(m) : new Material(m) { name = m.name + " (solid)" };
                    solid.SetFloat("_SurfaceType", 0);
                    if (solid.HasProperty("_AlphaCutoffEnable")) solid.SetFloat("_AlphaCutoffEnable", 1);
                    if (solid.HasProperty("_ZWrite")) solid.SetFloat("_ZWrite", 1);
                    solid.renderQueue = -1;
                    UnityEditor.Rendering.HighDefinition.HDShaderUtils.ResetMaterialKeywords(solid);
                    mats[i] = solid; any = true; changed++;
                }
                if (any) r.sharedMaterials = mats;
            }
            return changed;
        }

        [MenuItem("Tools/Smoke Rescue/Place Survivors")]
        public static void PlaceSurvivors() => PlaceSurvivors(Scenario.Load("scenario_haze_2agents.json"));

        /// <summary>Places this scenario's survivors (fixed or seeded random spots) and gives them body heat for the
        /// thermal cameras. Called before every recording, because random placement differs per seed.</summary>
        public static void PlaceSurvivors(Scenario scenario)
        {
            var old = GameObject.Find(SurvivorsRoot);
            if (old != null) Object.DestroyImmediate(old);
            var cfg = JObject.Parse(File.ReadAllText(Path.Combine(FloorPlan.ConfigDir, "props.json")));
            var models = (cfg["survivor_candidates"] as JArray)?.Cast<JObject>().ToList();
            var root = new GameObject(SurvivorsRoot);
            int[] pick = { 0, 1, 2, 0 };    // candidate 4 is in a T-pose; reuse candidate 1
            for (int i = 0; i < scenario.survivors.Count; i++)
            {
                var s = scenario.survivors[i];
                // Pre-posed survivors (tools/blender/pose_survivor.py) stand on their own floor contact; otherwise fall
                // back to laying a standing figure on its back.
                var posedPart = cfg["survivors_posed"]?[s.id] as JObject;
                var posed = posedPart != null ? FurnitureBuilder.LoadPart(posedPart) : null;
                var prefab = posed != null ? posed : FurnitureBuilder.LoadPart(models[pick[i % pick.Length]]);
                if (prefab == null) { Debug.LogWarning($"[SmokeRescue] survivor model missing for {s.id}"); continue; }
                var go = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
                SolidBodies(go);
                go.name = $"Survivor_{s.id}_{s.pose}";
                go.transform.SetParent(root.transform, false);
                var lay = posed != null ? Quaternion.identity : Quaternion.Euler(90f, 0f, 90f);
                go.transform.SetPositionAndRotation(new Vector3(s.at[0], 0f, s.at[1]), Quaternion.Euler(0f, s.yaw, 0f) * lay);
                var b = FurnitureBuilder.RenderBounds(go);
                go.transform.position += new Vector3(s.at[0] - b.center.x, -b.min.y + 0.01f, s.at[1] - b.center.z);
                foreach (var t in go.GetComponentsInChildren<Transform>()) t.gameObject.layer = LayerMask.NameToLayer(HouseBuilder.PropsLayer);
                b = FurnitureBuilder.RenderBounds(go);
                var col = go.AddComponent<BoxCollider>();
                col.center = go.transform.InverseTransformPoint(b.center);
                var size = go.transform.InverseTransformVector(b.size);
                col.size = new Vector3(Mathf.Abs(size.x), Mathf.Abs(size.y), Mathf.Abs(size.z));
                EvidenceCapture.Render($"c12_{s.id}.png", new Vector3(s.at[0], 1.5f, s.at[1]) + Quaternion.Euler(0f, s.yaw + 90f, 0f) * Vector3.forward * 1.8f,
                    Quaternion.LookRotation(new Vector3(s.at[0], 0.2f, s.at[1]) - (new Vector3(s.at[0], 1.5f, s.at[1]) + Quaternion.Euler(0f, s.yaw + 90f, 0f) * Vector3.forward * 1.8f)), 60f, 640, 480);
            }
            Debug.Log($"[SmokeRescue] Placed {root.transform.childCount} survivors");
        }
    }
}

namespace SmokeRescue.Editor
{
    public static class ShowcaseMenu
    {
        [MenuItem("Tools/Smoke Rescue/Setup Showcase Camera")]
        public static void Setup()
        {
            var old = GameObject.Find("ShowcaseCamera");
            if (old != null) Object.DestroyImmediate(old);
            foreach (var c in Object.FindObjectsByType<Camera>(FindObjectsInactive.Exclude))
                if (c.CompareTag("MainCamera")) c.gameObject.SetActive(false);
            var go = new GameObject("ShowcaseCamera") { tag = "MainCamera" };
            go.AddComponent<Camera>();
            go.AddComponent<UnityEngine.Rendering.HighDefinition.HDAdditionalCameraData>();
            go.AddComponent<SmokeRescue.Recording.ShowcaseView>();
            Debug.Log("[SmokeRescue] Showcase camera added; enter play mode to run it");
        }
    }
}
