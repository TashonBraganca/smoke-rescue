using SmokeRescue.Recording;
using UnityEditor;
using UnityEngine;

namespace SmokeRescue.Editor
{
    /// <summary>Recording entry points. Each run rebuilds robots and atmosphere from config, then records on a fixed clock.</summary>
    public static class RecorderMenu
    {
        [MenuItem("Tools/Smoke Rescue/Record/Clear 10 s test")]
        public static void ClearTest() => Run("scenario_clear_2agents.json", "unity_clear_test10", 10f);

        [MenuItem("Tools/Smoke Rescue/Record/Clear 10 s test, no noise")]
        public static void ClearTestNoNoise() => Run("scenario_clear_2agents.json", "unity_clear_test10_nonoise", 10f, noise: false);

        [MenuItem("Tools/Smoke Rescue/Record/Clear full run")]
        public static void ClearFull() => Run("scenario_clear_2agents.json", "unity_clear_seed7");

        [MenuItem("Tools/Smoke Rescue/Record/Haze full run")]
        public static void HazeFull() => Run("scenario_haze_2agents.json", "unity_haze_seed7");

        static void Run(string scenario, string runId, float maxSeconds = float.PositiveInfinity, bool noise = true)
        {
            var r = RecordingSession.Record(scenario, runId, maxSeconds, noise, msg => Debug.Log($"[SmokeRescue] {runId} {msg}"));
            Debug.Log($"[SmokeRescue] Recording done: {r.dir}");
        }
    }
}
