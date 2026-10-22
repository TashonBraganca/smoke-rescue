using System.Linq;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using SmokeRescue.Editor;
using SmokeRescue.World;
using UnityEngine;

namespace SmokeRescue.Tests.EditMode
{
    /// <summary>What the colour camera renders each survivor model with, after ScenePreview.SolidBodies: shader, surface
    /// type, alpha clipping, render queue and base alpha per material. Diagnostic, run by hand.</summary>
    public sealed class SurvivorMaterialDiagnostics
    {
        [Test, Explicit("diagnostic: survivor materials after SolidBodies")]
        public void ReportSurvivorMaterials()
        {
            var cfg = JObject.Parse(System.IO.File.ReadAllText(System.IO.Path.Combine(FloorPlan.ConfigDir, "props.json")));
            var sb = new System.Text.StringBuilder();
            foreach (var id in new[] { "s1", "s2", "s3" })
            {
                var prefab = FurnitureBuilder.LoadPart((JObject)cfg["survivors_posed"][id]);
                var go = (GameObject)UnityEditor.PrefabUtility.InstantiatePrefab(prefab);
                int changed = ScenePreview.SolidBodies(go);
                sb.Append($"| {id} changed={changed}: ");
                foreach (var r in go.GetComponentsInChildren<Renderer>(true))
                    foreach (var m in r.sharedMaterials.Where(x => x != null))
                    {
                        string P(string n) => m.HasProperty(n) ? m.GetFloat(n).ToString("0.##") : "-";
                        string a = m.HasProperty("_BaseColor") ? m.GetColor("_BaseColor").a.ToString("0.##") : "-";
                        sb.Append($"[{m.name} sh={m.shader.name} surf={P("_SurfaceType")} cut={P("_AlphaCutoffEnable")}/{P("_AlphaCutoff")} q={m.renderQueue} a={a} kw={string.Join(",", m.shaderKeywords.Where(k => k.Contains("TRANSP") || k.Contains("ALPHA") || k.Contains("BLEND")))}] ");
                    }
                go.transform.position = new Vector3(0, 0, 0);
                EvidenceCapture.Render($"diag_{id}_material.png", new Vector3(0, 1.4f, -2.2f), Quaternion.LookRotation(new Vector3(0, .3f, 0) - new Vector3(0, 1.4f, -2.2f)), 60f, 640, 480);
                Object.DestroyImmediate(go);
            }
            Assert.Fail(sb.ToString());
        }
    }
}
