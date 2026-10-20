using System.IO;
using SmokeRescue.World;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

namespace SmokeRescue.Editor
{
    /// <summary>
    /// Renders evidence images to data/evidence/. Inspection rig: a shadowless sun (lights the interior
    /// through the roof) and instant auto exposure, so images show geometry and materials clearly.
    /// Set sceneLighting to render with only the scene's own lights, fog and exposure instead.
    /// </summary>
    public static class EvidenceCapture
    {
        public static string EvidenceDir => Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", "..", "data", "evidence"));

        public static string Render(string relPath, Vector3 position, Quaternion rotation, float vfov = 60f,
            int width = 960, int height = 720, bool sceneLighting = false, bool hideCeiling = false, bool orthographic = false, float orthoSize = 5f)
        {
            var go = new GameObject("EvidenceCamera");
            try
            {
                var cam = go.AddComponent<Camera>();
                cam.transform.SetPositionAndRotation(position, rotation);
                cam.fieldOfView = vfov;
                cam.nearClipPlane = 0.05f;
                cam.orthographic = orthographic;
                cam.orthographicSize = orthoSize;
                cam.cullingMask = ~(1 << LayerMask.NameToLayer("ThermalOnly"));
                if (hideCeiling) cam.cullingMask &= ~(1 << LayerMask.NameToLayer(HouseBuilder.CeilingLayer));
                var hd = go.AddComponent<HDAdditionalCameraData>();
                hd.clearColorMode = HDAdditionalCameraData.ClearColorMode.Color;
                hd.backgroundColorHDR = Color.black;
                if (!sceneLighting)
                {
                    var sun = new GameObject("EvidenceSun").AddComponent<Light>();
                    sun.transform.SetParent(go.transform, false);
                    sun.type = LightType.Directional;
                    sun.transform.rotation = Quaternion.Euler(55f, 35f, 0f);
                    sun.shadows = LightShadows.None;
                    sun.intensity = 8000f;
                    var volume = go.AddComponent<Volume>();
                    volume.isGlobal = true;
                    volume.priority = 1000f;
                    var profile = ScriptableObject.CreateInstance<VolumeProfile>();
                    var exposure = profile.Add<Exposure>(true);
                    // Fixed EV for an 8000 lux shadowless sun. Auto exposure did not settle once the scene's
                    // Atmosphere volume existed, and captures came out white.
                    exposure.mode.Override(ExposureMode.Fixed);
                    exposure.fixedExposure.Override(11f);
                    profile.Add<Fog>(true).enabled.Override(false);
                    volume.sharedProfile = profile;
                }
                var rt = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
                cam.targetTexture = rt;
                for (int i = 0; i < 8; i++) cam.Render();   // settle exposure and temporal history
                RenderTexture.active = rt;
                var tex = new Texture2D(width, height, TextureFormat.RGB24, false);
                tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
                tex.Apply();
                RenderTexture.active = null;
                string path = Path.Combine(EvidenceDir, relPath);
                Directory.CreateDirectory(Path.GetDirectoryName(path));
                File.WriteAllBytes(path, tex.EncodeToPNG());
                Object.DestroyImmediate(tex);
                rt.Release();
                return path;
            }
            finally
            {
                Object.DestroyImmediate(go);
            }
        }

        /// <summary>Renders the views listed in data/evidence/custom_views.json: [{"name","eye":[x,y,z],"target":[x,y,z],"fov"}].</summary>
        [MenuItem("Tools/Smoke Rescue/Capture Custom Views")]
        public static void CaptureCustomViews()
        {
            var views = Newtonsoft.Json.Linq.JArray.Parse(File.ReadAllText(Path.Combine(EvidenceDir, "custom_views.json")));
            foreach (Newtonsoft.Json.Linq.JObject v in views)
            {
                var e = v["eye"].ToObject<float[]>(); var t = v["target"].ToObject<float[]>();
                var eye = new Vector3(e[0], e[1], e[2]);
                Render($"custom/{v.Value<string>("name")}.png", eye, Quaternion.LookRotation(new Vector3(t[0], t[1], t[2]) - eye),
                    v.Value<float?>("fov") ?? 60f, 800, 600, sceneLighting: v.Value<bool?>("scene_lighting") ?? false);
            }
        }

        /// <summary>One image per landmark, from 2.4 m in front of it at eye height, looking back at it.</summary>
        [MenuItem("Tools/Smoke Rescue/Capture Landmarks")]
        public static void CaptureLandmarks()
        {
            var plan = FloorPlan.Load();
            foreach (var lm in plan.landmarks)
            {
                var at = new Vector3(lm.at[0], 0f, lm.at[1]);
                var fwd = Quaternion.Euler(0f, lm.yaw, 0f) * Vector3.forward;
                var eye = at + fwd * 2.4f + Vector3.up * 1.4f;
                // Keep the eye inside the room: clamp to the room rectangle with a 0.2 m margin.
                var r = plan.FindRoom(lm.room).Rect;
                eye.x = Mathf.Clamp(eye.x, r.xMin + 0.2f, r.xMax - 0.2f);
                eye.z = Mathf.Clamp(eye.z, r.yMin + 0.2f, r.yMax - 0.2f);
                var look = Quaternion.LookRotation(at + Vector3.up * 0.7f - eye);
                Render($"c2_landmarks/{lm.room}_{lm.kind}.png", eye, look, 70f, 640, 480);
            }
            Debug.Log($"[SmokeRescue] Captured {plan.landmarks.Count} landmark views to {EvidenceDir}/c2_landmarks");
        }
    }
}
