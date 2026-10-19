using System.Collections.Generic;
using SmokeRescue.Radio;
using SmokeRescue.Robots;
using SmokeRescue.Sensors;
using SmokeRescue.World;
using UnityEngine;
using UnityEngine.Rendering.HighDefinition;

namespace SmokeRescue.Recording
{
    /// <summary>
    /// C15 showcase view (play mode): a slow orbit over the cutaway house (ceilings hidden in this camera only),
    /// the robots driving their scenario routes in real time, each robot's live RGB and depth as picture-in-picture,
    /// and a status strip per robot: ID, speed, smoke density at its position, radio link state.
    /// Display only; recordings come from RecordingSession, not from this view.
    /// </summary>
    public class ShowcaseView : MonoBehaviour
    {
        public string scenarioFile = "scenario_haze_2agents.json";
        public float orbitPeriodS = 90f;
        public float orbitRadius = 15f;
        public float height = 17f;

        Camera cam;
        Atmosphere atmosphere;
        RadioModel radio;
        readonly List<(Robot robot, RgbdCamera rgbd)> robots = new List<(Robot, RgbdCamera)>();
        Vector3 centre;
        float simT, frameAccum;
        Material depthMat;
        GUIStyle style;
        readonly Queue<float> frameTimes = new Queue<float>();
        public float MeasuredFps { get; private set; }

        void Start()
        {
            var plan = FloorPlan.Load();
            var scenario = Scenario.Load(scenarioFile);
            atmosphere = Atmosphere.Build(plan, scenario);
            radio = new RadioModel(scenario.radio.range_m, scenario.radio.max_walls);
            centre = new Vector3((plan.footprint[0] + plan.footprint[2]) / 2f, 0f, (plan.footprint[1] + plan.footprint[3]) / 2f);
            var root = GameObject.Find("Robots");
            if (root != null) Destroy(root);
            root = new GameObject("Robots");
            var cfg = SensorConfig.Load();
            for (int i = 0; i < scenario.robots.Count; i++)
            {
                var r = Robot.Create(scenario.robots[i], scenario.look_around, plan, root.transform);
                var go = new GameObject("SensorCamera");
                go.transform.SetParent(r.CameraMount, false);
                go.AddComponent<Camera>();
                go.AddComponent<HDAdditionalCameraData>();
                var rgbd = go.AddComponent<RgbdCamera>();
                rgbd.Initialize(cfg, scenario.seed * 1000 + i);
                rgbd.NoiseEnabled = false;
                robots.Add((r, rgbd));
            }
            cam = gameObject.GetComponent<Camera>() ?? gameObject.AddComponent<Camera>();
            if (GetComponent<HDAdditionalCameraData>() == null) gameObject.AddComponent<HDAdditionalCameraData>();
            cam.fieldOfView = 50f;
            cam.cullingMask = ~(1 << LayerMask.NameToLayer(HouseBuilder.CeilingLayer));
            cam.farClipPlane = 200f;
            var depthShader = Shader.Find("Hidden/SmokeRescue/DepthColormap");
            if (depthShader != null) depthMat = new Material(depthShader);
        }

        void Update()
        {
            float dt = Mathf.Min(Time.deltaTime, 0.05f);
            frameTimes.Enqueue(Time.unscaledDeltaTime);
            while (frameTimes.Count > 60) frameTimes.Dequeue();
            float sum = 0f;
            foreach (var f in frameTimes) sum += f;
            MeasuredFps = frameTimes.Count / Mathf.Max(sum, 1e-4f);

            // Robots step at a fixed 600 Hz inside each frame so their motion matches the recordings.
            int steps = Mathf.Max(1, Mathf.RoundToInt(dt * RecordingSession.ClockHz));
            for (int s = 0; s < steps; s++)
                foreach (var (r, _) in robots) r.Step(1f / RecordingSession.ClockHz);
            simT += dt;
            atmosphere.Apply(simT);
            frameAccum += dt;
            if (frameAccum >= 1f / 15f)
            {
                frameAccum = 0f;
                foreach (var (_, rgbd) in robots) if (!rgbd.IsCapturing) rgbd.Capture(simT);
            }
            float a = 2f * Mathf.PI * simT / orbitPeriodS;
            transform.position = centre + new Vector3(Mathf.Cos(a) * orbitRadius, height, Mathf.Sin(a) * orbitRadius);
            transform.LookAt(centre + Vector3.down * 1f);
            if (Mathf.FloorToInt(simT / 5f) != Mathf.FloorToInt((simT - dt) / 5f))
                Debug.Log($"[SmokeRescue] Showcase fps {MeasuredFps:0.0} at sim {simT:0.0} s");
        }

        void OnGUI()
        {
            style ??= new GUIStyle(GUI.skin.label) { fontSize = 15, normal = { textColor = Color.white } };
            float w = Screen.width * 0.2f, h = w * 0.75f, pad = 10f;
            for (int i = 0; i < robots.Count; i++)
            {
                var (r, rgbd) = robots[i];
                float y = pad + i * (h + 34f);
                float x = Screen.width - 2 * w - 2 * pad;
                if (rgbd.ColorTexture != null) GUI.DrawTexture(new Rect(x, y, w, h), rgbd.ColorTexture, ScaleMode.ScaleToFit, false);
                if (rgbd.DepthTexture != null)
                {
                    if (depthMat != null) Graphics.DrawTexture(new Rect(x + w + pad, y, w, h), rgbd.DepthTexture, depthMat);
                    else GUI.DrawTexture(new Rect(x + w + pad, y, w, h), rgbd.DepthTexture, ScaleMode.ScaleToFit, false);
                }
                string link = "no other robot";
                foreach (var (o, _) in robots)
                    if (o != r) { var l = radio.Evaluate(r.transform.position, o.transform.position); link = l.delivered ? $"link ok {l.distance:0.0} m" : $"link down ({l.reason})"; }
                float smoke = atmosphere.Field.Density(r.transform.position + Vector3.up * 0.5f);
                var c = r.Colour;
                GUI.color = c;
                GUI.Box(new Rect(x, y + h + 2f, 2 * w + pad, 26f), GUIContent.none);
                GUI.color = Color.white;
                GUI.Label(new Rect(x + 6f, y + h + 4f, 2 * w, 24f),
                    $"{r.RobotId}   {r.Driver.Speed:0.00} m/s   smoke {smoke:0.00}   {link}", style);
            }
            GUI.Label(new Rect(pad, Screen.height - 30f, 600f, 24f), $"Smoke Rescue simulation · sim {simT:0.0} s · {MeasuredFps:0} fps", style);
        }
    }
}
