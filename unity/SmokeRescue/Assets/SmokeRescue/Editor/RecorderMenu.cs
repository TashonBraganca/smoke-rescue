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

        static void Run(string scenario, string runId, float maxSeconds = float.PositiveInfinity, bool noise = true)
        {
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
