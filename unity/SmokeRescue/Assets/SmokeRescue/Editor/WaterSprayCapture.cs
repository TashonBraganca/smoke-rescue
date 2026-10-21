using System.IO;
using Newtonsoft.Json.Linq;
using SmokeRescue.Robots;
using SmokeRescue.Sensors;
using SmokeRescue.World;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

namespace SmokeRescue.Editor
{
    /// <summary>
    /// Evidence for the water spray visual: a robot's-eye sensor frame (scene lighting, the robot's 40 degree torch, the
    /// exposure loop warmed up, noise off) and a side view, mid-spray at the burning living room sofa of the haze
    /// scenario, after 0.4 s and 1.2 s of flow and 0.6 s after the nozzle closes. Writes data/evidence/water/*.png. The
    /// nozzle and the heading are set as a robot would hold them (gripper about 1 m up, ahead of the camera).
    /// </summary>
    public static class WaterSprayCapture
    {
        [MenuItem("Tools/Smoke Rescue/Diagnostics/Capture Water Spray")]
        public static void Capture()
        {
            var plan = FloorPlan.Load();
            var sc = Scenario.Load("scenario_haze_2agents.json");
            FurnitureBuilder.PlaceScenarioObstacles(sc);
            var atm = Atmosphere.Build(plan, sc);
            var nozzleCfg = JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath, "..", "..", "..", "config", "actuators.json")))["nozzle"];
            var head = JObject.Parse(File.ReadAllText(Path.Combine(Application.dataPath, "..", "..", "..", "config", "sensors.json")))["headlight"];
            var dir = Path.Combine(EvidenceCapture.EvidenceDir, "water");
            Directory.CreateDirectory(dir);
            var holder = new GameObject("WaterSprayCaptureRobot");
            var camGo = new GameObject("WaterSprayCaptureCamera");
            try
            {
                // Robot standing in the living room east of the sofa fire, facing west toward it.
                var body = new Vector3(6.2f, 0f, 2.2f); float headingRad = -Mathf.PI / 2f;
                var forward = new Vector3(Mathf.Sin(headingRad), 0f, Mathf.Cos(headingRad));
                holder.transform.position = body;
                var nozzle = body + forward * .55f + new Vector3(-.08f, 1.0f, 0f);
                var eye = body + forward * .25f + Vector3.up * 1.6f;
                camGo.transform.SetPositionAndRotation(eye, Quaternion.LookRotation(forward) * Quaternion.Euler(10f, 0f, 0f));
                camGo.AddComponent<Camera>(); camGo.AddComponent<HDAdditionalCameraData>();
                var rgbd = camGo.AddComponent<RgbdCamera>();
                rgbd.Initialize(SensorConfig.Load(), 1); rgbd.NoiseEnabled = false; rgbd.LocalDensity = atm.Field.Density;
                var torch = new GameObject("Torch").AddComponent<Light>(); torch.transform.SetParent(camGo.transform, false);
                torch.type = LightType.Spot; torch.spotAngle = head.Value<float>("cone_deg"); torch.innerSpotAngle = head.Value<float>("inner_cone_deg");
                torch.range = head.Value<float>("range_m"); torch.lightUnit = UnityEngine.Rendering.LightUnit.Lumen; torch.intensity = head.Value<float>("intensity_lumen");
                torch.shadows = LightShadows.Soft;
                var water = new WaterSprayVisual(holder.transform, nozzleCfg.Value<float>("half_angle_deg"), nozzleCfg.Value<float>("range_m"), 7);
                double t = 10.0; atm.Apply((float)t);   // the sofa fire still small: the water is what is judged
                for (int i = 0; i < 12; i++) { t += 1 / 15.0; atm.Apply((float)t); rgbd.CaptureSync(t); }   // exposure settles before the water
                water.Begin(nozzle, headingRad);
                void Run(double seconds, string name)
                {
                    RgbdFrame f = null;
                    for (double s = 0; s < seconds - 1e-6; s += 1 / 15.0) { t += 1 / 15.0; atm.Apply((float)t); water.Step(1 / 15f); f = rgbd.CaptureSync(t); }
                    File.WriteAllBytes(Path.Combine(dir, name + "_robot.png"), SensorPng.EncodeRgbAsync(f.Rgb, f.Width, f.Height).Result);
                    Debug.Log($"[SmokeRescue] water {name}: particles {water.Counts}");
                    var side = nozzle - forward * .6f + new Vector3(0f, .9f, 1.6f);   // behind and beside the robot, looking along the stream
                    EvidenceCapture.Render(Path.Combine("water", name + "_side.png"), side, Quaternion.LookRotation(nozzle + forward * 2.0f + Vector3.down * .3f - side), 60f, 960, 720, sceneLighting: true);
                }
                Run(.4, "a_0.4s");
                Run(.8, "b_1.2s");
                water.End();
                Run(.6, "c_closed_0.6s");
                Debug.Log($"[SmokeRescue] Water spray captures written to {dir}");
            }
            finally { Object.DestroyImmediate(camGo); Object.DestroyImmediate(holder); }
        }
    }
}
