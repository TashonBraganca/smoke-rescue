using SmokeRescue.Sensors;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

namespace SmokeRescue.Editor
{
    /// <summary>Checks sensor image orientation inside the house: depth at a pixel must match a physics raycast through it.</summary>
    public static class SensorDiagnostics
    {
        /// <summary>Returns the worst |depth - raycast z| in metres over a grid of pixels; logs the per-row picture.</summary>
        public static float CheckOrientation(RgbdCamera sensor, out string report)
        {
            var cfg = sensor.Config.camera;
            var frame = sensor.CaptureSync(0);
            var cam = sensor.Camera;
            float worst = 0f;
            var sb = new System.Text.StringBuilder();
            foreach (int v in new[] { 40, 140, 240, 340, 440 })
                foreach (int u in new[] { 80, 320, 560 })
                {
                    // OpenCV pixel (u, v), top-left origin, centres on integers -> Unity screen point (x right, y up, centres at +0.5).
                    var ray = cam.ScreenPointToRay(new Vector3(u + 0.5f, cfg.height - 0.5f - v, 0f));
                    if (!Physics.Raycast(ray, out var hit, 20f)) continue;
                    float zTrue = Vector3.Dot(hit.point - cam.transform.position, cam.transform.forward);
                    ushort mm = frame.Depth[v * cfg.width + u];
                    if (mm == 0 || zTrue < cfg.width * 0 + 0.25f || zTrue > 7.5f) continue;
                    float err = Mathf.Abs(mm * 0.001f - zTrue);
                    worst = Mathf.Max(worst, err);
                    sb.Append($"v={v} u={u} depth={mm * 0.001f:0.000} ray={zTrue:0.000} hit={hit.collider.name}\n");
                }
            report = sb.ToString();
            return worst;
        }

        [MenuItem("Tools/Smoke Rescue/Diagnostics/Sensor Orientation")]
        public static void Run()
        {
            foreach (var solid in new[] { false, true })
            {
                var go = new GameObject("DiagCamera");
                try
                {
                    go.transform.SetPositionAndRotation(new Vector3(10f, 0.5f, 1.0f), Quaternion.Euler(10f, 0f, 0f));
                    go.AddComponent<Camera>();
                    go.AddComponent<HDAdditionalCameraData>();
                    var s = go.AddComponent<RgbdCamera>();
                    s.Initialize(SensorConfig.Load(), 5);
                    s.NoiseEnabled = false;
                    if (solid) { s.Camera.clearFlags = CameraClearFlags.SolidColor; s.Camera.backgroundColor = Color.black; }
                    Physics.SyncTransforms();
                    float worst = CheckOrientation(s, out var report);
                    Debug.Log($"[SmokeRescue] Orientation check solidClear={solid}: worst error {worst:0.000} m\n{report}");
                }
                finally { Object.DestroyImmediate(go); }
            }
        }
    }
}
