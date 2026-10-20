using System;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

namespace SmokeRescue.Sensors
{
    public sealed class RgbdFrame
    {
        public readonly double Timestamp;
        public readonly int Width, Height;
        /// <summary>Owned managed buffers. RGB is sRGB RGB24, depth is optical z in mm. Row zero is the top.</summary>
        public readonly byte[] Rgb;
        public readonly ushort[] Depth;
        public RgbdFrame(double timestamp, int width, int height, byte[] rgb, ushort[] depth)
        { Timestamp = timestamp; Width = width; Height = height; Rgb = rgb; Depth = depth; }
    }
    /// <summary>Attach at the camera mount. Capture is explicit; this component never advances a simulation clock.</summary>
    [DisallowMultipleComponent]
    [RequireComponent(typeof(Camera), typeof(HDAdditionalCameraData))]
    public sealed class RgbdCamera : MonoBehaviour
    {
        public SensorConfig Config { get; private set; }
        public Camera Camera { get; private set; }
        public bool NoiseEnabled { get; set; } = true;
        /// <summary>Exposure in EV100, applied through the camera's physical parameters (f/1.4, ISO 100).</summary>
        public float ExposureEv { get; set; } = 3f;
        /// <summary>Deterministic exposure loop: after each frame, move EV toward a mid-grey mean (damped, clamped).</summary>
        public bool AutoExposure { get; set; } = true;
        public const float ExposureMinEv = 0f, ExposureMaxEv = 12f, TargetGrey = 105f / 255f;
        public bool IsCapturing { get; private set; }
        public Func<Vector3, float> LocalDensity { get; set; } = _ => 0;
        /// <summary>Caller supplies normalized proximity to fire for configured exposure flicker.</summary>
        public float NearFire { get; set; }
        public event Action<RgbdFrame> FrameReady;
        /// <summary>Last rendered RGB and linear-depth targets, for on-screen display (showcase picture-in-picture).</summary>
        public RenderTexture ColorTexture => colorTarget;
        public RenderTexture DepthTexture => depthTarget;
        public event Action<string> CaptureFailed;
        RenderTexture colorTarget, depthTarget;
        CustomPassVolume passVolume;
        LinearEyeDepthPass depthPass;
        RgbdNoise noise;
        byte[] rgb;
        float[] depth;
        Pose capturePose;
        double timestamp;
        bool rgbDone, depthDone, failed;
        int pendingReadbacks;
        bool disposing;
        bool captureNoise;
        float captureFire;
        Func<Vector3, float> captureDensity;

        void Awake() { Initialize(SensorConfig.Load(), 0); }
        public void Initialize(SensorConfig config, int seed)
        {
            if (IsCapturing) throw new InvalidOperationException("Cannot reconfigure a pending capture.");
            config.Validate();
            ReleaseResources();
            disposing = false;
            Config = config;
            noise = new RgbdNoise(config, seed);
            Camera = GetComponent<Camera>();
            Camera.enabled = false;
            Camera.usePhysicalProperties = false;
            Camera.orthographic = false;
            Camera.fieldOfView = config.camera.vertical_fov_deg;
            Camera.aspect = (float)config.camera.width / config.camera.height;
            Camera.nearClipPlane = Mathf.Min(0.01f, config.depth.min_m * 0.5f);
            // RGB must see the whole house; depth beyond max_m is invalidated by the noise model, not by clipping.
            // A 9 m far plane left far walls undrawn and the clear colour showed through doorways.
            Camera.farClipPlane = 60f;
            Camera.allowMSAA = false;
            Camera.allowDynamicResolution = false;
            Camera.rect = new Rect(0, 0, 1, 1);
            int thermalOnly = LayerMask.NameToLayer("ThermalOnly");
            if (thermalOnly >= 0) Camera.cullingMask &= ~(1 << thermalOnly);   // flame bodies exist only for thermal cameras
            Camera.ResetProjectionMatrix();
            var hd = GetComponent<HDAdditionalCameraData>();
            hd.antialiasing = HDAdditionalCameraData.AntialiasingMode.None;
            hd.clearColorMode = HDAdditionalCameraData.ClearColorMode.Color;   // night outside: black, never a sky colour
            hd.backgroundColorHDR = Color.black;
            hd.customRenderingSettings = true;
            SetFrameSetting(hd, FrameSettingsField.CustomPass, true);
            // Exposure comes from the scene's Atmosphere volume (fixed EV100, so frames stay deterministic).
            // With exposure control off HDRP ignores that volume and the headlight spot clips to white.
            SetFrameSetting(hd, FrameSettingsField.ExposureControl, true);
            SetFrameSetting(hd, FrameSettingsField.Postprocess, false);
            SetFrameSetting(hd, FrameSettingsField.DepthOfField, false);
            SetFrameSetting(hd, FrameSettingsField.MotionBlur, false);
            SetFrameSetting(hd, FrameSettingsField.LensDistortion, false);
            SetFrameSetting(hd, FrameSettingsField.ChromaticAberration, false);
            SetFrameSetting(hd, FrameSettingsField.Antialiasing, false);
            colorTarget = CreateTarget(config.camera, GraphicsFormat.R8G8B8A8_SRGB, "Sensor RGB");
            depthTarget = CreateTarget(config.camera, GraphicsFormat.R32_SFloat, "Sensor optical depth");
            Camera.targetTexture = colorTarget;
            passVolume = gameObject.AddComponent<CustomPassVolume>();
            passVolume.isGlobal = true;
            passVolume.targetCamera = Camera;
            passVolume.injectionPoint = CustomPassInjectionPoint.AfterOpaqueDepthAndNormal;
            depthPass = new LinearEyeDepthPass { name = "C4 optical depth", SensorCamera = Camera, Destination = depthTarget };
            passVolume.customPasses.Add(depthPass);
        }
        static void SetFrameSetting(HDAdditionalCameraData hd, FrameSettingsField field, bool enabled)
        {
            hd.renderingPathCustomFrameSettingsOverrideMask.mask[(uint)field] = true;
            hd.renderingPathCustomFrameSettings.SetEnabled(field, enabled);
        }
        static RenderTexture CreateTarget(SensorConfig.CameraConfig config, GraphicsFormat format, string label)
        {
            var target = new RenderTexture(new RenderTextureDescriptor(config.width, config.height)
            { graphicsFormat = format, depthStencilFormat = GraphicsFormat.None, msaaSamples = 1, useMipMap = false, useDynamicScale = false })
            { name = label, filterMode = FilterMode.Point };
            target.Create();
            return target;
        }
        /// <summary>Returns false if the previous readbacks are pending. Caller must wait and retry the SAME sim timestamp.
        /// Call on Unity's main thread outside render callbacks. Camera pose is sampled at this call.</summary>
        public bool Capture(double simTimestamp)
        {
            if (disposing || !isActiveAndEnabled) throw new InvalidOperationException("Sensor is disabled or destroyed.");
            if (IsCapturing) return false;
            if (double.IsNaN(simTimestamp) || double.IsInfinity(simTimestamp) || simTimestamp < 0)
                throw new ArgumentOutOfRangeException(nameof(simTimestamp));
            if (!(RenderPipelineManager.currentPipeline is HDRenderPipeline))
                throw new InvalidOperationException("RGB-D capture requires the active HDRP pipeline.");
            IsCapturing = true; rgbDone = depthDone = failed = false;
            timestamp = simTimestamp;
            capturePose = new Pose(transform.position, transform.rotation);
            ApplyExposure();
            captureNoise = NoiseEnabled; captureFire = NearFire; captureDensity = LocalDensity;
            depthPass.Armed = true;
            int previousRenderCount = depthPass.RenderCount;
            try
            {
                RenderPipeline.SubmitRenderRequest(Camera, new RenderPipeline.StandardRequest { destination = colorTarget });
                if (depthPass.RenderCount == previousRenderCount)
                    throw new InvalidOperationException("HDRP did not execute the sensor depth custom pass. Enable custom passes in the pipeline asset.");
                pendingReadbacks = 2;
                AsyncGPUReadback.Request(colorTarget, 0, OnColor);
                AsyncGPUReadback.Request(depthTarget, 0, OnDepth);
            }
            catch
            {
                IsCapturing = false;
                throw;
            }
            finally { depthPass.Armed = false; }
            return true;
        }
        /// <summary>
        /// Recorder path: render and read back on the spot, blocking until both images are on the CPU.
        /// Does not depend on editor or player ticks, so a fixed-clock recording runs as fast as the GPU allows.
        /// Returns the processed frame; FrameReady is not raised.
        /// </summary>
        public RgbdFrame CaptureSync(double simTimestamp)
        {
            RgbdFrame result = null;
            string failure = null;
            void Ready(RgbdFrame f) => result = f;
            void Failed(string m) => failure = m;
            var handlers = FrameReady;
            FrameReady = Ready;
            CaptureFailed += Failed;
            try
            {
                if (!Capture(simTimestamp)) throw new InvalidOperationException("A capture is already pending.");
                AsyncGPUReadback.WaitAllRequests();
                if (failure != null) throw new InvalidOperationException(failure);
                if (result == null) throw new InvalidOperationException("Readback finished without a frame.");
                return result;
            }
            finally
            {
                FrameReady = handlers;
                CaptureFailed -= Failed;
            }
        }

        void OnColor(AsyncGPUReadbackRequest request)
        {
            pendingReadbacks--;
            if (disposing) { ReleaseWhenReadbacksFinish(); return; }
            if (request.hasError) { Fail("RGB GPU readback failed."); return; }
            var raw = request.GetData<byte>();
            int w = Config.camera.width, h = Config.camera.height;
            rgb = new byte[w * h * 3];
            for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int source = ((h - 1 - y) * w + x) * 4;
                int destination = (y * w + x) * 3;
                rgb[destination] = raw[source]; rgb[destination + 1] = raw[source + 1]; rgb[destination + 2] = raw[source + 2];
            }
            rgbDone = true;
            Complete();
        }
        void OnDepth(AsyncGPUReadbackRequest request)
        {
            pendingReadbacks--;
            if (disposing) { ReleaseWhenReadbacksFinish(); return; }
            if (request.hasError) { Fail("Depth GPU readback failed."); return; }
            var raw = request.GetData<float>();
            int w = Config.camera.width, h = Config.camera.height;
            depth = new float[w * h];
            for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) depth[y * w + x] = raw[(h - 1 - y) * w + x];
            depthDone = true;
            Complete();
        }
        void Fail(string message)
        {
            failed = true;
            if (pendingReadbacks == 0) IsCapturing = false;
            CaptureFailed?.Invoke(message);
        }
        void Complete()
        {
            if (failed) { if (pendingReadbacks == 0) IsCapturing = false; return; }
            if (!rgbDone || !depthDone) return;
            try
            {
                if (AutoExposure) UpdateExposure(rgb);
                var millimetres = noise.Process(rgb, depth, capturePose, captureNoise, captureDensity, captureFire);
                var frame = new RgbdFrame(timestamp, Config.camera.width, Config.camera.height, rgb, millimetres);
                IsCapturing = false; rgb = null; depth = null;
                FrameReady?.Invoke(frame);
            }
            catch (Exception error)
            {
                IsCapturing = false;
                CaptureFailed?.Invoke(error.ToString());
            }
        }
        void ApplyExposure()
        {
            // EV100 = log2(N^2 / t) at ISO 100, so t = N^2 / 2^EV.
            Camera.aperture = 1.4f;
            Camera.iso = 100;
            Camera.shutterSpeed = 1.96f / Mathf.Pow(2f, ExposureEv);
        }

        /// <summary>Mean linear luminance of a pixel subsample against mid-grey; half a step per frame toward it.</summary>
        void UpdateExposure(byte[] frameRgb)
        {
            double sum = 0;
            int n = 0;
            for (int i = 0; i < frameRgb.Length; i += 3 * 37)
            {
                float lum = (0.2126f * frameRgb[i] + 0.7152f * frameRgb[i + 1] + 0.0722f * frameRgb[i + 2]) / 255f;
                sum += Mathf.Pow(Mathf.Max(lum, 1f / 255f), 2.2f);
                n++;
            }
            float meanLin = (float)(sum / Mathf.Max(n, 1));
            float targetLin = Mathf.Pow(TargetGrey, 2.2f);
            ExposureEv = Mathf.Clamp(ExposureEv + 0.6f * Mathf.Log(meanLin / targetLin, 2f), ExposureMinEv, ExposureMaxEv);
        }

        void OnDestroy() { disposing = true; ReleaseWhenReadbacksFinish(); }
        void ReleaseWhenReadbacksFinish() { if (pendingReadbacks == 0) ReleaseResources(); }
        void ReleaseResources()
        {
            if (Camera != null) Camera.targetTexture = null;
            if (passVolume != null)
            {
                passVolume.enabled = false;
                if (Application.isPlaying) Destroy(passVolume); else DestroyImmediate(passVolume);
            }
            if (colorTarget != null) { colorTarget.Release(); CoreUtils.Destroy(colorTarget); }
            if (depthTarget != null) { depthTarget.Release(); CoreUtils.Destroy(depthTarget); }
        }
    }
}
