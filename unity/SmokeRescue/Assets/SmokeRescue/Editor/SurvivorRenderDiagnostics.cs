using UnityEditor;
using UnityEngine;

namespace SmokeRescue.Editor
{
    /// <summary>Diagnostics: how the placed people render (renderer type, motion vectors, shader, queue), to find why the
    /// sensor depth pass does not contain them (one-door seed 922: a kneeling woman absent from depth at 3.3 m).</summary>
    public static class SurvivorRenderDiagnostics
    {
        [MenuItem("Tools/Smoke Rescue/Diagnostics/Survivor renderers")]
        public static void Dump()
        {
            var root = GameObject.Find(ScenePreview.SurvivorsRoot);
            if (root == null) { Debug.Log("[SurvivorRender] no survivors in the scene"); return; }
            foreach (var r in root.GetComponentsInChildren<Renderer>(true))
            {
                string mats = "";
                foreach (var m in r.sharedMaterials)
                    if (m != null) mats += $"{m.shader.name}|queue={m.renderQueue}|surface={(m.HasProperty("_SurfaceType") ? m.GetFloat("_SurfaceType") : -1)}|zwrite={(m.HasProperty("_ZWrite") ? m.GetFloat("_ZWrite") : -1)}; ";
                Debug.Log($"[SurvivorRender] {r.name} {r.GetType().Name} enabled={r.enabled} mv={r.motionVectorGenerationMode} layer={LayerMask.LayerToName(r.gameObject.layer)} shadow={r.shadowCastingMode} mats={mats}");
            }
        }
    }
}
