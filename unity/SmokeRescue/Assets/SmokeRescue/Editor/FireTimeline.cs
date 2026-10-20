using System.IO;
using SmokeRescue.Sensors;
using SmokeRescue.World;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

namespace SmokeRescue.Editor
{
    /// <summary>Snapshots of the spreading fire on the robots' clock: a cutaway overview of the house plus robot-height
    /// colour and thermal views, written to data/evidence/fire_timeline/.</summary>
    public static class FireTimeline
    {
        static readonly float[] Times = { 0f, 45f, 90f, 135f, 180f };
        static readonly (string name, Vector3 eye, Vector3 target)[] Views =
        {
            ("living", new Vector3(6.5f, 1.6f, 3.5f), new Vector3(2f, 0.8f, 2f)),
            ("hallway", new Vector3(9f, 1.6f, 7f), new Vector3(2f, 1.2f, 7f)),
        };

        [MenuItem("Tools/Smoke Rescue/Capture Fire Timeline")]
        public static void Capture() => Capture("scenario_haze_2agents.json", "fire_timeline");

        [MenuItem("Tools/Smoke Rescue/Diagnostics/Capture Fire Timeline (static fires)")]
        public static void CaptureStatic() => Capture("scenario_clear_2agents.json", "fire_timeline_static");

        static void Capture(string scenarioFile, string folder)
        {
            var plan = FloorPlan.Load();
            var scenario = Scenario.Load(scenarioFile);
            var cfg = SensorConfig.Load();
            var atm = Atmosphere.Build(plan, scenario);
            ThermalPreview.TagSurvivors();
            string dir = Path.Combine(EvidenceCapture.EvidenceDir, folder);
            Directory.CreateDirectory(dir);

            var holder = new GameObject("FireTimelineRig");
            try
            {
                var rgbGo = new GameObject("Rgb"); rgbGo.transform.SetParent(holder.transform, false);
                rgbGo.AddComponent<Camera>(); rgbGo.AddComponent<HDAdditionalCameraData>();
                var rgbd = rgbGo.AddComponent<RgbdCamera>();
                rgbd.Initialize(cfg, 3);
                rgbd.NoiseEnabled = false;
                rgbd.LocalDensity = atm.Field.Density;
                var light = new GameObject("Headlight").AddComponent<Light>();
                light.transform.SetParent(rgbGo.transform, false);
                light.type = LightType.Spot; light.spotAngle = 100f; light.innerSpotAngle = 15f; light.range = 10f;
                light.lightUnit = UnityEngine.Rendering.LightUnit.Lumen; light.intensity = 900f;

                var thGo = new GameObject("Thermal"); thGo.transform.SetParent(holder.transform, false);
                thGo.AddComponent<Camera>(); thGo.AddComponent<HDAdditionalCameraData>();
                var thermal = thGo.AddComponent<ThermalCamera>();
                thermal.Initialize(cfg.thermal, 3);

                var ovGo = new GameObject("Overview"); ovGo.transform.SetParent(holder.transform, false);
                var ov = ovGo.AddComponent<Camera>();
                var ovHd = ovGo.AddComponent<HDAdditionalCameraData>();
                ovHd.clearColorMode = HDAdditionalCameraData.ClearColorMode.Color;
                ovHd.backgroundColorHDR = new Color(0.01f, 0.01f, 0.015f);
                ov.fieldOfView = 45f; ov.farClipPlane = 200f;
                ov.cullingMask = ~(1 << LayerMask.NameToLayer(HouseBuilder.CeilingLayer)) & ~(1 << LayerMask.NameToLayer("ThermalOnly"));
                ov.aperture = 1.4f; ov.iso = 100; ov.shutterSpeed = 1.96f / Mathf.Pow(2f, 1.8f);
                var centre = new Vector3((plan.footprint[0] + plan.footprint[2]) / 2f, 0f, (plan.footprint[1] + plan.footprint[3]) / 2f);
                ovGo.transform.position = centre + new Vector3(0f, 21f, -9f);
                ovGo.transform.LookAt(centre);
                var ovRt = new RenderTexture(1280, 720, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
                ov.targetTexture = ovRt;

                foreach (float t in Times)
                {
                    Settle(atm, t);
                    float now = t;
                    for (int k = 0; k < 6; k++) { now += 1f / 15f; atm.Apply(now); ov.Render(); }
                    Save(ovRt, Path.Combine(dir, $"t{t:000}_overview.png"));
                    // Third view: aimed at the biggest burning region right now, from 3 m away at eye height.
                    var views = new System.Collections.Generic.List<(string, Vector3, Vector3)>(Views);
                    (Vector3 c, float r, float h, float temp) big = default;
                    foreach (var f in atm.Thermal.Source.Flames(t)) if (f.radius * f.height > big.r * big.h) big = f;
                    if (big.r > 0f)
                    {
                        var toward = new Vector3(centre.x - big.c.x, 0f, centre.z - big.c.z).normalized;
                        views.Add(("biggest_fire", big.c + toward * 3f + Vector3.up * 1.6f, big.c + Vector3.up * 0.7f));
                    }
                    foreach (var (name, eye, target) in views)
                    {
                        rgbGo.transform.SetPositionAndRotation(eye, Quaternion.LookRotation(target - eye));
                        thGo.transform.SetPositionAndRotation(eye + rgbGo.transform.right * cfg.thermal.mount_right_offset_m, rgbGo.transform.rotation);
                        RgbdFrame f = null;
                        for (int i = 0; i < 10; i++) { now += 1f / 15f; atm.Apply(now); f = rgbd.CaptureSync(now); }   // exposure settles
                        atm.Apply(now);
                        File.WriteAllBytes(Path.Combine(dir, $"t{t:000}_{name}.png"), SensorPng.EncodeRgbAsync(f.Rgb, f.Width, f.Height).Result);
                        File.WriteAllBytes(Path.Combine(dir, $"t{t:000}_{name}_thermal.png"), ThermalPreview.Ironbow(thermal.CaptureSync(now)));
                    }
                    var fs = atm.FireSpread;
                    string state = fs == null ? "static fires" :
                        $"fire time {fs.FireTime(t):0} s, burning blocks {CountActive(atm, t)}, collapses {fs.Sim.Collapses.Count}";
                    Debug.Log($"[SmokeRescue] fire timeline t={t:0} s: {state}");
                }
                ovRt.Release();
            }
            finally { Object.DestroyImmediate(holder); }
        }

        [MenuItem("Tools/Smoke Rescue/Diagnostics/Fire State")]
        public static void DebugState() { DebugState("scenario_haze_2agents.json"); DebugState("scenario_clear_2agents.json"); }

        static void DebugState(string scenarioFile)
        {
            var plan = FloorPlan.Load();
            var atm = Atmosphere.Build(plan, Scenario.Load(scenarioFile));
            foreach (float t in new[] { 0f, 45f, 90f })
            {
                Settle(atm, t);
                var fl = atm.Thermal.Source.Flames(t);
                int on = 0; float maxR = 0, maxH = 0;
                foreach (var f in fl) if (f.radius > 0) { on++; maxR = Mathf.Max(maxR, f.radius); maxH = Mathf.Max(maxH, f.height); }
                int systems = 0, alive = 0, emitting = 0;
                var root = atm.transform.Find("FireVisuals");
                foreach (var ps in root.GetComponentsInChildren<ParticleSystem>(true))
                {
                    systems++; alive += ps.particleCount;
                    if (ps.emission.rateOverTimeMultiplier > 0) emitting++;
                }
                var first = root.childCount > 1 ? root.GetChild(1).position : Vector3.zero;
                foreach (var ps in root.GetComponentsInChildren<ParticleSystem>(true))
                {
                    if (ps.name != "FlameBody" && ps.name != "Smoke") continue;
                    var buf = new ParticleSystem.Particle[ps.main.maxParticles];
                    int live = ps.GetParticles(buf);
                    var rr = ps.GetComponent<ParticleSystemRenderer>();
                    Debug.Log($"[SmokeRescue] fire state {scenarioFile} t={t} {ps.transform.parent.name}/{ps.name}: live {live}, playing {ps.isPlaying} paused {ps.isPaused} stopped {ps.isStopped} " +
                              $"time {ps.time:0.00} rate {ps.emission.rateOverTimeMultiplier:0.0} enabled {ps.emission.enabled} active {ps.gameObject.activeInHierarchy} " +
                              $"renderer {rr.enabled} mat {(rr.sharedMaterial ? rr.sharedMaterial.name : "null")} tex {(rr.sharedMaterial && rr.sharedMaterial.mainTexture ? rr.sharedMaterial.mainTexture.name : "null")} pos {ps.transform.position}");
                    break;
                }
                Debug.Log($"[SmokeRescue] fire state {scenarioFile} t={t}: slots {fl.Count}, burning {on}, max radius {maxR:0.00} height {maxH:0.00}; " +
                          $"particle systems {systems}, emitting {emitting}, live particles {alive}; first region at {first}");
            }
        }

        /// <summary>Jump to one second before t, then advance in 1/15 s frames like a recording does, so flames and smoke
        /// are in the same state a recording would show at t.</summary>
        static void Settle(Atmosphere atm, float t)
        {
            float start = Mathf.Max(0f, t - 1f);
            atm.Apply(start);
            for (int i = 1; i <= 15 && start + i / 15f <= t + 1e-4f; i++) atm.Apply(start + i / 15f);
            atm.Apply(t);
        }

        static int CountActive(Atmosphere atm, float t)
        {
            int n = 0;
            foreach (var f in atm.Thermal.Source.Flames(t)) if (f.radius > 0f) n++;
            return n;
        }

        static void Save(RenderTexture rt, string path)
        {
            var tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
            RenderTexture.active = rt;
            tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            RenderTexture.active = null;
            tex.Apply();
            File.WriteAllBytes(path, tex.EncodeToPNG());
            Object.DestroyImmediate(tex);
        }
    }
}
