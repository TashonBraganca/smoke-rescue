using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using UnityEditor;
using UnityEngine;

namespace SmokeRescue.Editor
{
    /// <summary>Evaluator diagnostic: where the people's bodies actually are in the scene (render-bounds centre), written to
    /// data/evidence/scene_survivors.json. Held-out seed 914 runs 3 and 4 wrote oracle positions that disagreed with the scene.</summary>
    public static class DumpSceneSurvivors
    {
        [MenuItem("Tools/Smoke Rescue/Diagnostics/Dump Scene Survivors")]
        public static void Dump()
        {
            var root = GameObject.Find(ScenePreview.SurvivorsRoot);
            var rows = new List<object>();
            if (root != null)
                foreach (Transform child in root.transform)
                {
                    var rs = child.GetComponentsInChildren<Renderer>(); if (rs.Length == 0) continue;
                    var b = rs[0].bounds; foreach (var r in rs) b.Encapsulate(r.bounds);
                    var parts = child.name.Split('_');
                    rows.Add(new { id = parts.Length > 1 ? parts[1] : child.name, name = child.name, position_house = new[] { b.center.x, 0f, b.center.z } });
                }
            string path = Path.Combine(EvidenceCapture.EvidenceDir, "scene_survivors.json");
            File.WriteAllText(path, JsonConvert.SerializeObject(rows, Formatting.Indented));
            Debug.Log($"[SmokeRescue] {rows.Count} scene survivors written to {path}");
        }
    }
}
