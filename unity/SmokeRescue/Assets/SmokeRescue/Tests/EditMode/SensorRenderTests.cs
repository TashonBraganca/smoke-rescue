using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using NUnit.Framework;
using SmokeRescue.Sensors;
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;
using UnityEngine.SceneManagement;
using UnityEngine.TestTools;
using Object = UnityEngine.Object;

namespace SmokeRescue.Tests.EditMode
{
    public class SensorRenderTests
    {
        readonly List<Object> owned = new List<Object>();
        GameObject root;
        SensorConfig config;
        RgbdCamera sensor;
        Vector3 marker;
        static string EvidenceDir => Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", "..", "data", "evidence"));

        [UnitySetUp]
        public IEnumerator SetUp()
        {
            config = SensorConfig.Load();
            // Edit mode cannot create runtime scenes; the geometry is isolated by distance (1 km from the house)
            // and by layer 30, which is the only layer the test cameras render.
            root = new GameObject("C4 test geometry");
            sensor = MakeSensor("C4 camera A", new Vector3(1000.7f, 2.1f, 1000.3f), Quaternion.Euler(13, 31, 0));
            var pose = sensor.transform;
            marker = pose.TransformPoint(new Vector3(0.445f, 0.275f, 3));
            MakeQuad("Background at six metres", pose.TransformPoint(new Vector3(0, 0, 6)), pose.rotation,
                new Vector2(14, 10), Checker(new Color(0.03f, 0.3f, 0.04f), new Color(0.1f, 0.65f, 0.2f)), 12);
            MakeQuad("Marker at three metres", marker, pose.rotation, new Vector2(0.3f, 0.3f),
                Checker(new Color(1, 0.03f, 0.01f), new Color(0.65f, 0.02f, 0.01f)), 2);
            yield return null;
        }
        RgbdCamera MakeSensor(string label, Vector3 position, Quaternion rotation)
        {
            var go = new GameObject(label);
            go.transform.SetParent(root.transform);
            go.transform.SetPositionAndRotation(position, rotation);
            var result = go.AddComponent<RgbdCamera>();
            result.Initialize(config, 17);
            result.NoiseEnabled = false;
            result.Camera.cullingMask = 1 << 30;
            result.Camera.clearFlags = CameraClearFlags.SolidColor;
            result.Camera.backgroundColor = Color.black;
            return result;
        }
        Texture2D Checker(Color a, Color b)
        {
            var texture = new Texture2D(64, 64, TextureFormat.RGBA32, false);
            var pixels = new Color[64 * 64];
            for (int y = 0; y < 64; y++)
            for (int x = 0; x < 64; x++) pixels[y * 64 + x] = ((x / 8 + y / 8) % 2 == 0) ? a : b;
            texture.SetPixels(pixels); texture.Apply();
            owned.Add(texture);
            return texture;
        }
        void MakeQuad(string label, Vector3 position, Quaternion rotation, Vector2 size, Texture texture, float repeats)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Quad);
            go.name = label; go.layer = 30;
            go.transform.SetParent(root.transform);
            go.transform.SetPositionAndRotation(position, rotation);
            go.transform.localScale = new Vector3(size.x, size.y, 1);
            var material = new Material(Shader.Find("HDRP/Unlit"));
            material.SetColor("_UnlitColor", Color.white);
            material.SetTexture("_UnlitColorMap", texture);
            material.SetTextureScale("_UnlitColorMap", Vector2.one * repeats);
            HDMaterial.ValidateMaterial(material);
            go.GetComponent<Renderer>().sharedMaterial = material;
            owned.Add(material);
        }
        IEnumerator Capture(RgbdCamera camera, double time, Action<RgbdFrame> receive)
        {
            RgbdFrame frame = null;
            string failure = null;
            void Ready(RgbdFrame value) { frame = value; }
            void Failed(string value) { failure = value; }
            camera.FrameReady += Ready; camera.CaptureFailed += Failed;
            try
            {
                Assert.That(camera.Capture(time), Is.True);
                var timer = Stopwatch.StartNew();
                while (frame == null && failure == null && timer.Elapsed.TotalSeconds < 30) yield return null;
                Assert.That(failure, Is.Null);
                Assert.That(frame, Is.Not.Null, "RGB/depth readbacks timed out.");
                receive(frame);
            }
            finally { camera.FrameReady -= Ready; camera.CaptureFailed -= Failed; }
        }
        [UnityTest]
        public IEnumerator RenderedProjectionAndDepthAgreeWithConvertedPose()
        {
            RgbdFrame frame = null;
            yield return Capture(sensor, 0, f => frame = f);
            var k = config.camera;
            var housePose = FrameConvert.Pose(sensor.transform.position, sensor.transform.rotation);
            var pointHouse = FrameConvert.Point(marker);
            var optical = Quaternion.Inverse(housePose.rotation) * (pointHouse - housePose.position);
            var projected = FrameConvert.Project(optical, k);
            var screen = sensor.Camera.WorldToScreenPoint(marker);
            // Unity pixel centres are .5. Subtract .5 before the top-first row flip.
            var unityPixel = new Vector2(screen.x - 0.5f, k.height - 0.5f - screen.y);
            float projectionError = Vector2.Distance(projected, unityPixel);
            int u = Mathf.RoundToInt(projected.x), v = Mathf.RoundToInt(projected.y);
            int index = v * k.width + u;
            Assert.That(projectionError, Is.LessThan(0.5f));
            Assert.That(frame.Depth[index], Is.GreaterThan(0));
            Assert.That(frame.Rgb[index * 3], Is.GreaterThan(frame.Rgb[index * 3 + 1] * 3), "Red marker is absent at the top-first projected pixel.");
            var reconstructedOptical = FrameConvert.BackProject(u, v, frame.Depth[index] * 0.001f, k);
            var reconstructed = housePose.position + housePose.rotation * reconstructedOptical;
            float depthErrorCm = Vector3.Distance(reconstructed, pointHouse) * 100;
            Assert.That(depthErrorCm, Is.LessThan(1));
            Assert.That(frame.Depth.Distinct().Count(), Is.GreaterThan(1), "Depth image is constant.");
            // An off-axis point on the background must still be 6 m, not the longer ray distance.
            Assert.That(frame.Depth[40 * k.width + 50], Is.EqualTo(6000).Within(2));
            double mean = frame.Rgb.Average(b => (double)b);
            double variance = frame.Rgb.Average(b => (b - mean) * (b - mean));
            Assert.That(variance, Is.GreaterThan(100), "RGB is blank.");
            Assert.That(sensor.Camera.enabled, Is.False);
            string report = string.Format(CultureInfo.InvariantCulture,
                "projection_error_px={0:F9}\ndepth_error_cm={1:F6}\nprojected_uv={2:F6},{3:F6}\nunity_uv={4:F6},{5:F6}\nsampled_pixel={6},{7}\nsampled_depth_mm={8}\nrgb_variance={9:F3}\n",
                projectionError, depthErrorCm, projected.x, projected.y, unityPixel.x, unityPixel.y, u, v, frame.Depth[index], variance);
            Directory.CreateDirectory(EvidenceDir);
            File.WriteAllText(Path.Combine(EvidenceDir, "c4_projection.txt"), report);
            UnityEngine.Debug.Log("[C4 projection] " + report);
        }
        [UnityTest]
        public IEnumerator TwoCamerasCaptureTenSimulationSecondsWithoutGaps()
        {
            var other = MakeSensor("C4 camera B", sensor.transform.position + sensor.transform.right * 0.12f, sensor.transform.rotation);
            other.NoiseEnabled = sensor.NoiseEnabled = true;
            int count = Mathf.RoundToInt(10 * config.camera.rate_hz);
            int receivedA = 0, receivedB = 0;
            RgbdFrame last = null;
            string failure = null;
            sensor.FrameReady += f => { Assert.That(f.Timestamp, Is.EqualTo(receivedA / (double)config.camera.rate_hz).Within(1e-9)); receivedA++; last = f; };
            other.FrameReady += f => { Assert.That(f.Timestamp, Is.EqualTo(receivedB / (double)config.camera.rate_hz).Within(1e-9)); receivedB++; };
            sensor.CaptureFailed += value => failure = value;
            other.CaptureFailed += value => failure = value;
            double submitMs = 0;
            var timer = Stopwatch.StartNew();
            for (int i = 0; i < count; i++)
            {
                double timestamp = i / (double)config.camera.rate_hz;
                var submitTimer = Stopwatch.StartNew();
                Assert.That(sensor.Capture(timestamp), Is.True);
                Assert.That(other.Capture(timestamp), Is.True);
                submitMs += submitTimer.Elapsed.TotalMilliseconds;
                while ((sensor.IsCapturing || other.IsCapturing) && failure == null && timer.Elapsed.TotalSeconds < 180) yield return null;
                Assert.That(failure, Is.Null);
                Assert.That(timer.Elapsed.TotalSeconds, Is.LessThan(180), "Two-camera capture timed out.");
            }
            timer.Stop();
            double pairedFps = count / timer.Elapsed.TotalSeconds;
            Assert.That(receivedA, Is.EqualTo(count)); Assert.That(receivedB, Is.EqualTo(count));
            var rgbTask = SensorPng.EncodeRgbAsync(last.Rgb, last.Width, last.Height);
            var view = new byte[last.Depth.Length * 3];
            for (int i = 0; i < last.Depth.Length; i++)
            {
                if (last.Depth[i] == 0) continue;
                float z = last.Depth[i] / (config.depth.max_m * 1000);
                view[i * 3] = (byte)(255 * (1 - z)); view[i * 3 + 1] = (byte)(255 * z); view[i * 3 + 2] = 80;
            }
            var depthTask = SensorPng.EncodeRgbAsync(view, last.Width, last.Height);
            var rawDepthTask = SensorPng.EncodeDepthAsync(last.Depth, last.Width, last.Height);
            while (!rgbTask.IsCompleted || !depthTask.IsCompleted || !rawDepthTask.IsCompleted) yield return null;
            Directory.CreateDirectory(EvidenceDir);
            File.WriteAllBytes(Path.Combine(EvidenceDir, "c4_rgb.png"), rgbTask.GetAwaiter().GetResult());
            File.WriteAllBytes(Path.Combine(EvidenceDir, "c4_depth.png"), depthTask.GetAwaiter().GetResult());
            File.WriteAllBytes(Path.Combine(EvidenceDir, "c4_depth_mm.png"), rawDepthTask.GetAwaiter().GetResult());
            string report = string.Format(CultureInfo.InvariantCulture,
                "scene=isolated textured quads, not House\nnoise=enabled\nframes_per_camera={0}\nsim_duration_s=10\nwall_duration_s={1:F6}\npaired_fps={2:F6}\nindividual_camera_frames_per_s={3:F6}\n", count, timer.Elapsed.TotalSeconds, pairedFps, pairedFps * 2) + $"submit_ms_per_pair={submitMs / count:F3}\n";
            File.WriteAllText(Path.Combine(EvidenceDir, "c4_capture.txt"), report);
            UnityEngine.Debug.Log("[C4 capture] " + report);
            // The async path waits on editor ticks, which macOS throttles while the editor is in the background,
            // so its fps measures the editor, not the sensor. Speed is asserted on CaptureSync, the recorder's path
            // (SynchronousCaptureRunsFastEnoughForRecording). This test asserts gap-free timestamps and frame counts.
        }
        [Test]
        public void SynchronousCaptureRunsFastEnoughForRecording()
        {
            var other = MakeSensor("C4 camera B sync", sensor.transform.position + sensor.transform.right * 0.12f, sensor.transform.rotation);
            int count = 60;
            var timer = Stopwatch.StartNew();
            for (int i = 0; i < count; i++)
            {
                double t = i / (double)config.camera.rate_hz;
                var a = sensor.CaptureSync(t);
                var b = other.CaptureSync(t);
                Assert.That(a.Timestamp, Is.EqualTo(t));
                Assert.That(b.Depth.Length, Is.EqualTo(config.camera.width * config.camera.height));
            }
            double pairedFps = count / timer.Elapsed.TotalSeconds;
            string report = string.Format(CultureInfo.InvariantCulture, "sync_paired_fps={0:F3}\nsync_pairs={1}\n", pairedFps, count);
            File.AppendAllText(Path.Combine(EvidenceDir, "c4_capture.txt"), report);
            UnityEngine.Debug.Log("[C4 sync capture] " + report);
            Assert.That(pairedFps, Is.GreaterThanOrEqualTo(5), report);
        }

        [UnityTearDown]
        public IEnumerator TearDown()
        {
            if (root != null) { if (Application.isPlaying) Object.Destroy(root); else Object.DestroyImmediate(root); }
            foreach (var item in owned) if (item != null) { if (Application.isPlaying) Object.Destroy(item); else Object.DestroyImmediate(item); }
            owned.Clear();
            yield return null;
        }
    }
}
