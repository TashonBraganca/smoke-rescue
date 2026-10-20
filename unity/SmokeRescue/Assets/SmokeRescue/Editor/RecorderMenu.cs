using SmokeRescue.Recording;
using SmokeRescue.World;
using UnityEditor;
using UnityEngine;

namespace SmokeRescue.Editor
{
    /// <summary>Recording entry points. Each run rebuilds robots and atmosphere from config, then records on a fixed clock.</summary>
    public static class RecorderMenu
    {
        /// <summary>Use the downloaded quadruped body instead of the placeholder box.</summary>
        public static bool UseQuadruped = true;
        [MenuItem("Tools/Smoke Rescue/Record/Clear 10 s test")]
        public static void ClearTest() => Run("scenario_clear_2agents.json", "unity_clear_test10", 10f);

        [MenuItem("Tools/Smoke Rescue/Record/Clear 10 s test, no noise")]
        public static void ClearTestNoNoise() => Run("scenario_clear_2agents.json", "unity_clear_test10_nonoise", 10f, noise: false);

        [MenuItem("Tools/Smoke Rescue/Record/Haze 20 s test")]
        public static void HazeTest() => Run("scenario_haze_2agents.json", "unity_haze_test20", 20f);

        [MenuItem("Tools/Smoke Rescue/Record/Clear full run")]
        public static void ClearFull() => Run("scenario_clear_2agents.json", "unity_clear");

        [MenuItem("Tools/Smoke Rescue/Record/Haze full run")]
        public static void HazeFull() => Run("scenario_haze_2agents.json", "unity_haze");

        // Autonomous robots sharing maps in the 2x2 building; new run ids so earlier recordings stay intact.
        [MenuItem("Tools/Smoke Rescue/Record/Haze autonomous 20 s test")]
        public static void HazeAutoTest() => Run("scenario_haze_2agents.json", "unity_haze_auto_test20", 20f);

        [MenuItem("Tools/Smoke Rescue/Record/Haze autonomous full run")]
        public static void HazeAutoFull() => Run("scenario_haze_2agents.json", "unity_haze_auto_v1");

        // Same runs with the LLM agent layer on (brain/brain/agent/server.py must be serving on port 8080).
        [MenuItem("Tools/Smoke Rescue/Record/Haze agents 20 s test")]
        public static void HazeAgentsTest() => WithAgents(() => Run("scenario_haze_2agents.json", "unity_haze_agents_test20", 20f));

        [MenuItem("Tools/Smoke Rescue/Record/Haze agents full run")]
        public static void HazeAgentsFull() => WithAgents(() => Run("scenario_haze_2agents.json", "unity_haze_agents_v1"));

        // One robot, torch off, enters a room whose lights failed; a person is inside. The agent must measure the
        // image, switch the torch on, and find the person.
        [MenuItem("Tools/Smoke Rescue/Record/Dark room torch demo (agents)")]
        public static void DarkTorchDemo() => WithAgents(() => Run("scenario_dark_torch.json", "unity_dark_torch_agents", 45f));

        [MenuItem("Tools/Smoke Rescue/Record/Dark room torch demo (no agents)")]
        public static void DarkTorchBaseline() => Run("scenario_dark_torch.json", "unity_dark_torch_reflex", 45f);

        // Both robots in the original house only, so they cover it, meet and merge within the run.
        [MenuItem("Tools/Smoke Rescue/Record/House agents run")]
        public static void HouseAgents() => WithAgents(() => Run("scenario_house_agents.json", "unity_house_agents_v1"));

        [MenuItem("Tools/Smoke Rescue/Record/House reflex 90 s check (no agents)")]
        public static void HouseReflexCheck() => Run("scenario_house_agents.json", "unity_house_reflex_check", 90f);

        [MenuItem("Tools/Smoke Rescue/Record/House reflex run (no agents)")]
        public static void HouseReflex() => Run("scenario_house_agents.json", "unity_house_reflex_v1");

        static void WithAgents(System.Action run)
        {
            RecordingSession.AgentUrl = System.Environment.GetEnvironmentVariable("SMOKE_AGENT_URL") ?? "http://localhost:8080";
            try { run(); } finally { RecordingSession.AgentUrl = null; }
        }

        static void Run(string scenario, string runId, float maxSeconds = float.PositiveInfinity, bool noise = true)
        {
            // The live colour person detector runs in the vision service; without it robots rely on thermal alone.
            string vision = System.Environment.GetEnvironmentVariable("SMOKE_AGENT_URL") ?? "http://localhost:8080";
            bool up = false;
            try { using (var http = new System.Net.Http.HttpClient { Timeout = System.TimeSpan.FromSeconds(5) }) up = http.GetAsync(vision.TrimEnd('/') + "/health").Result.IsSuccessStatusCode; }
            catch (System.Exception) { up = false; }
            RecordingSession.PeopleUrl = up ? vision : null;
            Debug.Log(up ? $"[SmokeRescue] colour person detector at {vision}" : $"[SmokeRescue] WARNING: vision service {vision} not reachable; colour person detection is off");
            SmokeRescue.Robots.Robot.BodyPrefab = UseQuadruped ? FurnitureBuilder.RobotBodyPrefab() : null;
            var sc = Scenario.Load(scenario);
            FurnitureBuilder.PlaceScenarioObstacles(sc);
            ScenePreview.PlaceSurvivors(sc);
            ThermalPreview.TagSurvivors();
            var r = RecordingSession.Record(scenario, runId, maxSeconds, noise, msg => Debug.Log($"[SmokeRescue] {runId} {msg}"));
            Debug.Log($"[SmokeRescue] Recording done: {r.dir}");
        }
    }
}
