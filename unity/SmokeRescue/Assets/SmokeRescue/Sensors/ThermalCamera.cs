using System;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

namespace SmokeRescue.Sensors
{
    public sealed class ThermalFrame
    {
        public readonly double Timestamp;
        public readonly int Width, Height;
        /// <summary>Apparent temperature in kelvin x 10 (3101 = 37 C). Row zero is the top.</summary>
        public readonly ushort[] DeciKelvin;
        public ThermalFrame(double timestamp, int width, int height, ushort[] deciKelvin)
        { Timestamp = timestamp; Width = width; Height = height; DeciKelvin = deciKelvin; }
        public float Celsius(int x, int y) => DeciKelvin[y * Width + x] / 10f - 273.15f;
    }

    /// <summary>
    /// Long-wave infrared camera beside the RGB-D camera. It sees surface temperature through smoke (smoke barely
    /// absorbs at 8 to 14 um) but the hot gas layer itself glows. Renders on demand with a blocking readback, like
    /// RgbdCamera.CaptureSync. Robot-side output only: apparent temperatures, never object identities.
    /// </summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(Camera), typeof(HDAdditionalCameraData))]
    public sealed class ThermalCamera : MonoBehaviour
    {
        public SensorConfig.ThermalConfig Config { get; private set; }
        public Camera Camera { get; private set; }
        public bool NoiseEnabled { get; set; } = true;
        public RenderTexture TemperatureTexture => tempTarget;
        RenderTexture colorTarget, tempTarget, depthBuffer;
        CustomPassVolume passVolume;
        ThermalPass pass;
        System.Random rng;
        float[] fixedPattern;

        public void Initialize(SensorConfig.ThermalConfig config, int seed)
        {
            ReleaseResources();
            Config = config;
            rng = new System.Random(seed);
            Camera = GetComponent<Camera>();
            Camera.enabled = false;
            Camera.fieldOfView = config.vertical_fov_deg;
            Camera.aspect = (float)config.width / config.height;
            Camera.nearClipPlane = 0.05f;
            Camera.farClipPlane = 60f;
            Camera.allowMSAA = false;
            Camera.allowDynamicResolution = false;
            Camera.cullingMask = ~0;    // includes the ThermalOnly layer (flame bodies) that colour cameras skip
            var hd = GetComponent<HDAdditionalCameraData>();
            hd.antialiasing = HDAdditionalCameraData.AntialiasingMode.None;
            hd.clearColorMode = HDAdditionalCameraData.ClearColorMode.Color;
            hd.backgroundColorHDR = Color.black;
            hd.customRenderingSettings = true;
            foreach (var (field, on) in new[] {
                (FrameSettingsField.CustomPass, true), (FrameSettingsField.Postprocess, false), (FrameSettingsField.Volumetrics, false),
                (FrameSettingsField.ShadowMaps, false), (FrameSettingsField.SSAO, false), (FrameSettingsField.SSR, false),
                (FrameSettingsField.Decals, false), (FrameSettingsField.TransparentObjects, false) })
            {
                hd.renderingPathCustomFrameSettingsOverrideMask.mask[(uint)field] = true;
                hd.renderingPathCustomFrameSettings.SetEnabled(field, on);
            }
            colorTarget = Target(GraphicsFormat.R8G8B8A8_UNorm, GraphicsFormat.None, "Thermal unused colour");
            tempTarget = Target(GraphicsFormat.R32_SFloat, GraphicsFormat.None, "Thermal temperature C");
            depthBuffer = Target(GraphicsFormat.None, GraphicsFormat.D32_SFloat, "Thermal depth");
            Camera.targetTexture = colorTarget;
            passVolume = gameObject.AddComponent<CustomPassVolume>();
            passVolume.isGlobal = true;
            passVolume.targetCamera = Camera;
            passVolume.injectionPoint = CustomPassInjectionPoint.AfterOpaqueDepthAndNormal;
            pass = new ThermalPass { name = "Thermal", SensorCamera = Camera, Destination = tempTarget, DepthBuffer = depthBuffer };
            passVolume.customPasses.Add(pass);

            // Fixed-pattern noise of an uncooled microbolometer after factory calibration: a small per-pixel offset.
            var fp = new System.Random(seed * 7919 + 17);
            fixedPattern = new float[config.width * config.height];
            for (int i = 0; i < fixedPattern.Length; i++) fixedPattern[i] = (float)(Gauss(fp) * config.fixed_pattern_k);
        }

        RenderTexture Target(GraphicsFormat color, GraphicsFormat depth, string label)
        {
            var desc = new RenderTextureDescriptor(Config.width, Config.height)
            { graphicsFormat = color, depthStencilFormat = depth, msaaSamples = 1, useMipMap = false, useDynamicScale = false };
            var rt = new RenderTexture(desc) { name = label, filterMode = FilterMode.Point };
            rt.Create();
            return rt;
        }

        /// <summary>Renders and reads back on the spot. Apparent temperature with NETD noise and fixed-pattern offset.</summary>
        public ThermalFrame CaptureSync(double simTimestamp)
        {
            if (!(RenderPipelineManager.currentPipeline is HDRenderPipeline))
                throw new InvalidOperationException("Thermal capture requires the active HDRP pipeline.");
            int before = pass.RenderCount;
            pass.Armed = true;
            try { RenderPipeline.SubmitRenderRequest(Camera, new RenderPipeline.StandardRequest { destination = colorTarget }); }
            finally { pass.Armed = false; }
            if (pass.RenderCount == before) throw new InvalidOperationException("HDRP did not execute the thermal custom pass.");
            var request = AsyncGPUReadback.Request(tempTarget, 0);
            request.WaitForCompletion();
            if (request.hasError) throw new InvalidOperationException("Thermal GPU readback failed.");
            var raw = request.GetData<float>().ToArray();   // a managed copy: the rows below are read from worker threads
            int w = Config.width, h = Config.height;
            var dk = new ushort[w * h];
            // Rows in parallel, each with a stream seeded from this camera's stream and the row (as RgbdNoise), so a frame
            // repeats for the same seed (perf.txt: about 5 of 68 wall s per 10 sim s serial).
            ulong frameSeed = ((ulong)(uint)rng.Next() << 32) | (uint)rng.Next();
            float netd = Config.netd_k, top = Config.range_max_c; bool noisy = NoiseEnabled;
            System.Threading.Tasks.Parallel.For(0, h, y =>
            {
                var row = new PixelRandom(frameSeed ^ ((ulong)(y + 1) * 0x9E3779B97F4A7C15UL));
                for (int x = 0; x < w; x++)
                {
                    float c = raw[(h - 1 - y) * w + x];
                    if (float.IsNaN(c)) c = 10f;
                    c = Mathf.Min(c, top);    // the sensor saturates at the top of its range
                    if (noisy) c += row.Gaussian() * netd + fixedPattern[y * w + x];
                    float k10 = Mathf.Clamp((c + 273.15f) * 10f, 0f, 65535f);
                    dk[y * w + x] = (ushort)Mathf.RoundToInt(k10);
                }
            });
            return new ThermalFrame(simTimestamp, w, h, dk);
        }

        static double Gauss(System.Random r)
        {
            double u1 = 1.0 - r.NextDouble(), u2 = r.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }

        void OnDestroy() => ReleaseResources();
        void ReleaseResources()
        {
            if (Camera != null) Camera.targetTexture = null;
            if (passVolume != null) { if (Application.isPlaying) Destroy(passVolume); else DestroyImmediate(passVolume); }
            foreach (var rt in new[] { colorTarget, tempTarget, depthBuffer })
                if (rt != null) { rt.Release(); CoreUtils.Destroy(rt); }
            colorTarget = tempTarget = depthBuffer = null;
        }
    }
}
