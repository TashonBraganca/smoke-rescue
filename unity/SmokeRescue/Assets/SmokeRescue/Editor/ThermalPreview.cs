using System.IO;
using Newtonsoft.Json.Linq;
using SmokeRescue.Sensors;
using SmokeRescue.World;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

namespace SmokeRescue.Editor
{
    /// <summary>Thermal camera checks: renders the views in data/evidence/custom_views.json with the thermal sensor and
    /// writes data/evidence/thermal/&lt;name&gt;.png (ironbow, for people) and &lt;name&gt;_dk.png (16-bit kelvin x 10, the sensor data).</summary>
    public static class ThermalPreview
    {
        [MenuItem("Tools/Smoke Rescue/Capture Thermal Views")]
        public static void Capture()
        {
            var plan = FloorPlan.Load();
            var cfg = SensorConfig.Load();
            float t = 120f;
            var sc = Scenario.Load("scenario_haze_2agents.json");
            FurnitureBuilder.PlaceScenarioObstacles(sc);
            var atm = Atmosphere.Build(plan, sc);
            atm.Apply(t);
            var holder = new GameObject("ThermalPreview");
            TagSurvivors();
            var views = JArray.Parse(File.ReadAllText(Path.Combine(EvidenceCapture.EvidenceDir, "custom_views.json")));
            var dir = Path.Combine(EvidenceCapture.EvidenceDir, "thermal");
            Directory.CreateDirectory(dir);
            var go = new GameObject("ThermalViewCamera");
            go.transform.SetParent(holder.transform, false);
            try
            {
                go.AddComponent<Camera>();
                go.AddComponent<HDAdditionalCameraData>();
                var cam = go.AddComponent<ThermalCamera>();
                cam.Initialize(cfg.thermal, 1);
                foreach (JObject v in views)
                {
                    var e = v["eye"].ToObject<float[]>(); var tg = v["target"].ToObject<float[]>();
                    var eye = new Vector3(e[0], e[1], e[2]);
                    go.transform.SetPositionAndRotation(eye, Quaternion.LookRotation(new Vector3(tg[0], tg[1], tg[2]) - eye));
                    var f = cam.CaptureSync(t);
                    string name = v.Value<string>("name");
                    File.WriteAllBytes(Path.Combine(dir, name + "_dk.png"), SensorPng.EncodeDepthAsync(f.DeciKelvin, f.Width, f.Height).Result);
                    File.WriteAllBytes(Path.Combine(dir, name + ".png"), Ironbow(f));
                    float min = float.MaxValue, max = float.MinValue;
                    foreach (var dk in f.DeciKelvin) { float c = dk / 10f - 273.15f; min = Mathf.Min(min, c); max = Mathf.Max(max, c); }
                    Debug.Log($"[SmokeRescue] thermal {name}: {min:0.0} to {max:0.0} C");
                }
            }
            finally { Object.DestroyImmediate(holder); }
        }

        /// <summary>Survivors are warm bodies: 34 C skin and clothing surface.</summary>
        public static void TagSurvivors()
        {
            var root = GameObject.Find(ScenePreview.SurvivorsRoot);
            if (root == null) return;
            var cfg = Newtonsoft.Json.Linq.JObject.Parse(File.ReadAllText(Path.Combine(FloorPlan.ConfigDir, "props.json")))["survivor_thermal"] as Newtonsoft.Json.Linq.JObject;
            foreach (Transform s in root.transform)
                if (cfg != null) ThermalBody.AddPerson(s.gameObject, cfg); else ThermalBody.Add(s.gameObject, 34f);
        }

        /// <summary>FLIR-style ironbow false colour with histogram-equalised gain (as thermal cameras use, so a 34 C body
        /// and a 600 C fire are both readable in one frame), as an RGB PNG.</summary>
        public static byte[] Ironbow(ThermalFrame f)
        {
            var lut = EqualisationLut(f.DeciKelvin);
            var tex = new Texture2D(f.Width, f.Height, TextureFormat.RGB24, false);
            var px = new Color32[f.Width * f.Height];
            for (int y = 0; y < f.Height; y++)
            for (int x = 0; x < f.Width; x++)
                px[(f.Height - 1 - y) * f.Width + x] = IronbowColour(lut[f.DeciKelvin[y * f.Width + x]]);
            tex.SetPixels32(px);
            tex.Apply();
            var png = tex.EncodeToPNG();
            Object.DestroyImmediate(tex);
            return png;
        }

        /// <summary>Plateau histogram equalisation: each 0.1 K bin is capped so a large uniform area (walls) cannot take
        /// the whole colour range. Returns a 0..1 lookup indexed by kelvin x 10.</summary>
        public static float[] EqualisationLut(ushort[] dk)
        {
            var hist = new int[65536];
            foreach (var v in dk) hist[v]++;
            int plateau = Mathf.Max(4, dk.Length / 300);
            var lut = new float[65536];
            double total = 0;
            for (int i = 0; i < hist.Length; i++) total += Mathf.Min(hist[i], plateau);
            double acc = 0;
            for (int i = 0; i < hist.Length; i++) { acc += Mathf.Min(hist[i], plateau); lut[i] = (float)(acc / System.Math.Max(total, 1)); }
            return lut;
        }

        static readonly Color[] Stops =
        {
            new Color(0f, 0f, 0f), new Color(0.13f, 0f, 0.36f), new Color(0.55f, 0f, 0.6f), new Color(0.85f, 0.2f, 0.25f),
            new Color(0.98f, 0.55f, 0f), new Color(1f, 0.85f, 0.15f), new Color(1f, 1f, 0.9f)
        };
        public static Color32 IronbowColour(float v)
        {
            float s = v * (Stops.Length - 1);
            int i = Mathf.Min((int)s, Stops.Length - 2);
            return Color.Lerp(Stops[i], Stops[i + 1], s - i);
        }
    }
}
