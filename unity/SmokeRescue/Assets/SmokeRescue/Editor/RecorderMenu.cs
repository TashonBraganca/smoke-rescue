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

        // Same house and rules, fire lit as the robots enter (Tashon approved it for measuring the live map link).
        [MenuItem("Tools/Smoke Rescue/Record/House held-out run (agents, random people)")]
        public static void HouseHeldOut() => WithAgents(() => Run("scenario_house_heldout.json", "unity_house_heldout_v1"));

        [MenuItem("Tools/Smoke Rescue/Record/House held-out seed 914 run (agents, random people)")]
        public static void HouseHeldOut914() => WithAgents(() => Run("scenario_house_heldout_seed914.json", "unity_house_heldout_seed914"));
        [MenuItem("Tools/Smoke Rescue/Record/House held-out seed 915 run (agents, random people)")]
        public static void HouseHeldOut915() => WithAgents(() => Run("scenario_house_heldout_seed915.json", "unity_house_heldout_seed915"));

        [MenuItem("Tools/Smoke Rescue/Record/House hidden corner run (agents)")]
        public static void HouseHiddenCorner() => WithAgents(() => Run("scenario_house_hidden_corner.json", "unity_house_hidden_v1"));

        [MenuItem("Tools/Smoke Rescue/Record/House agents slow fire run")]
        public static void HouseAgentsSlowFire() => WithAgents(() => Run("scenario_house_agents_slowfire.json", "unity_house_slowfire_v1"));

        [MenuItem("Tools/Smoke Rescue/Record/House reflex 90 s check (no agents)")]
        public static void HouseReflexCheck() => Run("scenario_house_agents.json", "unity_house_reflex_check", 90f);

        [MenuItem("Tools/Smoke Rescue/Record/House reflex run (no agents)")]
        public static void HouseReflex() => Run("scenario_house_agents.json", "unity_house_reflex_v1");

        // The recording runs over many editor updates (see RecordingSession.RecordSteps), so the agent URL is cleared when
        // the recording ends, not when the menu call returns.
        static void WithAgents(System.Action run)
        {
            RecordingSession.AgentUrl = System.Environment.GetEnvironmentVariable("SMOKE_AGENT_URL") ?? "http://localhost:8080";
            try { run(); } catch { RecordingSession.AgentUrl = null; throw; }
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
            // People are placed clear of the house's own furniture (Props layer), with a body-sized box. People and scenario
            // debris left from an earlier recording are not furniture: they changed where the next run placed people.
            int props = 1 << LayerMask.NameToLayer("Props");
            Physics.SyncTransforms();
            Scenario.PlacementBlocked = q =>
            {
                foreach (var hit in Physics.OverlapBox(new Vector3(q.x, .35f, q.y), new Vector3(.9f, .3f, .9f), Quaternion.identity, props, QueryTriggerInteraction.Ignore))
                {
                    bool scenarioObject = false;
                    for (var t = hit.transform; t != null; t = t.parent) if (t.name == ScenePreview.SurvivorsRoot || t.name == "Obstacles") { scenarioObject = true; break; }
                    if (!scenarioObject) return true;
                }
                return false;
            };
            var sc = Scenario.Load(scenario);
            FurnitureBuilder.PlaceScenarioObstacles(sc);
            int warm = FurnitureBuilder.ApplyWarmObjects(scenario);
            if (warm > 0) Debug.Log($"[SmokeRescue] {warm} warm objects that are not people");
            ScenePreview.PlaceSurvivors(sc);
            ThermalPreview.TagSurvivors();
            if (running != null) { RecordingSession.AgentUrl = null; throw new System.InvalidOperationException("A recording is already running."); }
            var steps = RecordingSession.RecordSteps(scenario, runId, maxSeconds, noise, msg => Debug.Log($"[SmokeRescue] {runId} {msg}")).GetEnumerator();
            running = () =>
            {
                bool more; RecordingSession.Result done = null;
                try { more = steps.MoveNext(); done = more ? steps.Current : null; }
                catch (System.Exception e) { Debug.LogException(e); more = false; }
                if (more && done == null) return;
                UnityEditor.EditorApplication.update -= running; running = null; RecordingSession.AgentUrl = null; steps.Dispose();
                Debug.Log(done != null ? $"[SmokeRescue] Recording done: {done.dir}" : $"[SmokeRescue] Recording {runId} stopped without a result");
            };
            UnityEditor.EditorApplication.update += running;
        }
        static UnityEditor.EditorApplication.CallbackFunction running;
    }
}
