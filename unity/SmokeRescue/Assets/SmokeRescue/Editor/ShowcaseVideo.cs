using System.Collections.Generic;
using System.IO;
using SmokeRescue.Robots;
using SmokeRescue.Sensors;
using SmokeRescue.World;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

namespace SmokeRescue.Editor
{
    /// <summary>
    /// Offline render of the showcase view (C15) on the simulation clock: orbit camera over the cutaway house,
    /// both robots walking the haze routes, and each robot's live RGB and depth as picture-in-picture.
    /// Writes data/evidence/showcase_frames/f00000.jpg ...; tools encode them with ffmpeg.
    /// </summary>
    public static class ShowcaseVideo
    {
        const int W = 1280, H = 720, PipW = 256, PipH = 192;

        [MenuItem("Tools/Smoke Rescue/Render Showcase Video")]
        public static void Render() => Render("scenario_haze_2agents.json", 150f);

        [MenuItem("Tools/Smoke Rescue/Render Showcase Video (10 s test)")]
        public static void RenderTest() => Render("scenario_haze_2agents.json", 10f);

        public static void Render(string scenarioFile, float seconds)
        {
            var plan = FloorPlan.Load();
            var scenario = Scenario.Load(scenarioFile);
            var cfg = SensorConfig.Load();
            FurnitureBuilder.PlaceScenarioObstacles(scenario);
            var atm = Atmosphere.Build(plan, scenario);
            string outDir = Path.Combine(EvidenceCapture.EvidenceDir, "showcase_frames");
            if (Directory.Exists(outDir)) Directory.Delete(outDir, true);
            Directory.CreateDirectory(outDir);

            Robot.BodyPrefab = FurnitureBuilder.RobotBodyPrefab();
            var old = GameObject.Find("Robots");
            if (old != null) Object.DestroyImmediate(old);
            var root = new GameObject("Robots");
            var rigs = new List<(Robot robot, RgbdCamera rgbd)>();
            for (int i = 0; i < scenario.robots.Count; i++)
            {
                var r = Robot.Create(scenario.robots[i], scenario.look_around, plan, root.transform);
                var go = new GameObject("SensorCamera");
                go.transform.SetParent(r.CameraMount, false);
                go.AddComponent<Camera>();
                go.AddComponent<HDAdditionalCameraData>();
                var rgbd = go.AddComponent<RgbdCamera>();
                rgbd.Initialize(cfg, 100 + i);
                rgbd.NoiseEnabled = false;
                rgbd.LocalDensity = atm.Field.Density;
                rigs.Add((r, rgbd));
            }

            var camGo = new GameObject("ShowcaseRenderCamera");
            var cam = camGo.AddComponent<Camera>();
            var hd = camGo.AddComponent<HDAdditionalCameraData>();
            hd.clearColorMode = HDAdditionalCameraData.ClearColorMode.Color;
            hd.backgroundColorHDR = new Color(0.01f, 0.01f, 0.015f);
            cam.fieldOfView = 45f;
            cam.farClipPlane = 200f;
            cam.cullingMask = ~(1 << LayerMask.NameToLayer(HouseBuilder.CeilingLayer)) & ~(1 << LayerMask.NameToLayer("ThermalOnly"));
            cam.aperture = 1.4f; cam.iso = 100; cam.shutterSpeed = 1.96f / Mathf.Pow(2f, 1.8f);
            var rt = new RenderTexture(W, H, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
            cam.targetTexture = rt;
            var frame = new Texture2D(W, H, TextureFormat.RGB24, false);
            var centre = new Vector3((plan.footprint[0] + plan.footprint[2]) / 2f, 0f, (plan.footprint[1] + plan.footprint[3]) / 2f);
            var last = new RgbdFrame[rigs.Count];

            const int stepsPerFrame = 40;     // 600 Hz robot clock, 15 Hz video clock
            int frames = Mathf.CeilToInt(seconds * 15f);
            try
            {
                for (int n = 0; n < frames; n++)
                {
                    float t = n / 15f;
                    if (n > 0)
                        for (int s = 0; s < stepsPerFrame; s++)
                            foreach (var (r, _) in rigs) r.Step(1f / 600f);
                    atm.Apply(t);
                    for (int i = 0; i < rigs.Count; i++) last[i] = rigs[i].rgbd.CaptureSync(t);

                    // Slow orbit, 120 s per revolution, high enough to see every room through the missing ceiling.
                    float a = 2f * Mathf.PI * (t / 120f) + 0.6f;
                    cam.transform.position = centre + new Vector3(Mathf.Cos(a) * 13f, 19f, Mathf.Sin(a) * 11f - 2f);
                    cam.transform.LookAt(centre + Vector3.down * 0.5f);
                    for (int k = 0; k < (n == 0 ? 6 : 1); k++) cam.Render();
                    RenderTexture.active = rt;
                    frame.ReadPixels(new Rect(0, 0, W, H), 0, 0);
                    RenderTexture.active = null;

                    var px = frame.GetPixels32();
                    for (int i = 0; i < rigs.Count; i++)
                    {
                        int x0 = W - 2 * PipW - 24, y0 = H - 16 - (i + 1) * (PipH + 26);   // bottom-left origin
                        Blit(px, last[i], x0, y0, depth: false);
                        Blit(px, last[i], x0 + PipW + 8, y0, depth: true);
                        Bar(px, x0, y0 + PipH + 4, 2 * PipW + 8, 14, rigs[i].robot.Colour);
                    }
                    frame.SetPixels32(px);
                    frame.Apply(false);
                    File.WriteAllBytes(Path.Combine(outDir, $"f{n:D5}.jpg"), frame.EncodeToJPG(88));
                    if (n % 150 == 0) Debug.Log($"[SmokeRescue] showcase video t={t:0.0}s frame {n}/{frames}");
                    if (rigs.TrueForAll(r => r.robot.Driver.Finished) && n > 0 && t > seconds - 0.1f) break;
                }
            }
            finally
            {
                rt.Release();
                Object.DestroyImmediate(camGo);
                Object.DestroyImmediate(frame);
            }
            Debug.Log($"[SmokeRescue] showcase frames written to {outDir}");
        }

        static void Blit(Color32[] dst, RgbdFrame f, int x0, int y0, bool depth)
        {
            for (int y = 0; y < PipH; y++)
            for (int x = 0; x < PipW; x++)
            {
                int sx = x * f.Width / PipW, sy = (PipH - 1 - y) * f.Height / PipH;   // frame rows are top first
                int si = sy * f.Width + sx;
                Color32 c;
                if (!depth) c = new Color32(f.Rgb[si * 3], f.Rgb[si * 3 + 1], f.Rgb[si * 3 + 2], 255);
                else
                {
                    ushort mm = f.Depth[si];
                    if (mm == 0) c = new Color32(0, 0, 0, 255);
                    else
                    {
                        float z = Mathf.Clamp01(mm / 6000f);
                        c = new Color32((byte)(255 * Mathf.Clamp01(1.5f - Mathf.Abs(4f * z - 3f))),
                                        (byte)(255 * Mathf.Clamp01(1.5f - Mathf.Abs(4f * z - 2f))),
                                        (byte)(255 * Mathf.Clamp01(1.5f - Mathf.Abs(4f * z - 1f))), 255);
                    }
                }
                dst[(y0 + y) * W + x0 + x] = c;
            }
        }

        static void Bar(Color32[] dst, int x0, int y0, int w, int h, Color colour)
        {
            Color32 c = colour;
            for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                dst[(y0 + y) * W + x0 + x] = c;
        }
    }
}
