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
        public bool IsCapturing { get; private set; }
        public Func<Vector3, float> LocalDensity { get; set; } = _ => 0;
        /// <summary>Caller supplies normalized proximity to fire for configured exposure flicker.</summary>
        public float NearFire { get; set; }
        public event Action<RgbdFrame> FrameReady;
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
            Camera.farClipPlane = config.depth.max_m + 1;
            Camera.allowMSAA = false;
            Camera.allowDynamicResolution = false;
            Camera.rect = new Rect(0, 0, 1, 1);
            Camera.ResetProjectionMatrix();
            var hd = GetComponent<HDAdditionalCameraData>();
            hd.antialiasing = HDAdditionalCameraData.AntialiasingMode.None;
            hd.customRenderingSettings = true;
            SetFrameSetting(hd, FrameSettingsField.CustomPass, true);
            // Fixed neutral exposure. Scene volumes cannot change the camera's exposure multiplier.
            SetFrameSetting(hd, FrameSettingsField.ExposureControl, false);
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
