using System.IO;
using SmokeRescue.World;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace SmokeRescue.Editor
{
    /// <summary>Tools > Smoke Rescue > Build House: rebuilds the house in the open scene from config/floorplan.json.</summary>
    public static class HouseMenu
    {
        public const string ScenePath = "Assets/SmokeRescue/Scenes/House.unity";

        [MenuItem("Tools/Smoke Rescue/Build House")]
        public static void BuildTextured() => Rebuild(greyBox: false);

        [MenuItem("Tools/Smoke Rescue/Build House (Grey Box)")]
        public static void BuildGreyBox() => Rebuild(greyBox: true);

        public static GameObject Rebuild(bool greyBox)
        {
            var old = GameObject.Find(HouseBuilder.RootName);
            if (old != null) Object.DestroyImmediate(old);
            var plan = FloorPlan.Load();
            var house = HouseBuilder.Build(plan, greyBox ? MaterialLibrary.GreyBox : MaterialLibrary.Get);
            foreach (var t in house.GetComponentsInChildren<Transform>())
                GameObjectUtility.SetStaticEditorFlags(t.gameObject, StaticEditorFlags.ContributeGI | StaticEditorFlags.OccluderStatic | StaticEditorFlags.OccludeeStatic | StaticEditorFlags.BatchingStatic);
            EditorSceneManager.MarkSceneDirty(house.scene);
            Debug.Log($"[SmokeRescue] Built house: {house.GetComponentsInChildren<MeshRenderer>().Length} renderers, {plan.rooms.Count} rooms, {plan.doors.Count} doors");
            return house;
        }

        /// <summary>Batchmode entry: open or create the House scene, build, save.</summary>
        public static void BuildAndSave()
        {
            Directory.CreateDirectory(Path.GetDirectoryName(ScenePath));
            if (File.Exists(ScenePath)) EditorSceneManager.OpenScene(ScenePath);
            else EditorSceneManager.SaveScene(EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects), ScenePath);
            Rebuild(greyBox: false);
            EditorSceneManager.SaveScene(EditorSceneManager.GetActiveScene());
        }

        /// <summary>Batchmode entry: build, save, and write the top-down evidence image.</summary>
        public static void BuildSaveAndCapture()
        {
            BuildAndSave();
            CaptureTopDown();
        }

        /// <summary>Renders a top-down orthographic view of the house without ceilings to data/evidence/.</summary>
        [MenuItem("Tools/Smoke Rescue/Capture Top-Down")]
        public static void CaptureTopDown()
        {
            var plan = FloorPlan.Load();
            float w = plan.footprint[2] - plan.footprint[0], d = plan.footprint[3] - plan.footprint[1];
            var go = new GameObject("TopDownCapture");
            // The roof would shadow the whole interior from the sun; switch ceiling shadows off for the capture.
            int ceilingLayer = LayerMask.NameToLayer(HouseBuilder.CeilingLayer);
            var ceilings = System.Array.FindAll(Object.FindObjectsByType<MeshRenderer>(FindObjectsInactive.Exclude),
                r => r.gameObject.layer == ceilingLayer);
            foreach (var r in ceilings) r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            try
            {
                var cam = go.AddComponent<Camera>();
                cam.orthographic = true;
                cam.orthographicSize = d / 2 + 1f;
                cam.aspect = (w + 2f) / (d + 2f);
                cam.transform.position = new Vector3(plan.footprint[0] + w / 2, 30f, plan.footprint[1] + d / 2);
                cam.transform.rotation = Quaternion.Euler(90f, 0f, 0f);
                cam.cullingMask = ~(1 << LayerMask.NameToLayer(HouseBuilder.CeilingLayer));
                cam.clearFlags = CameraClearFlags.SolidColor;
                cam.backgroundColor = Color.black;
                int px = 1600, py = Mathf.RoundToInt(px * (d + 2f) / (w + 2f));
                var rt = new RenderTexture(px, py, 24);
                cam.targetTexture = rt;
                // Capture rig: a shadowless sun and instant auto exposure, so the plan reads clearly whatever
                // the scene's own lighting is. HDRP needs a few frames to settle exposure history.
                var sun = new GameObject("CaptureSun").AddComponent<Light>();
                sun.transform.SetParent(go.transform, false);
                sun.type = LightType.Directional;
                sun.transform.rotation = Quaternion.Euler(70f, 30f, 0f);
                sun.shadows = LightShadows.None;
                sun.intensity = 20000f;
                var volume = go.AddComponent<UnityEngine.Rendering.Volume>();
                volume.isGlobal = true;
                volume.priority = 1000f;
                var profile = ScriptableObject.CreateInstance<UnityEngine.Rendering.VolumeProfile>();
                var exposure = profile.Add<UnityEngine.Rendering.HighDefinition.Exposure>(true);
                exposure.mode.Override(UnityEngine.Rendering.HighDefinition.ExposureMode.Automatic);
                exposure.adaptationMode.Override(UnityEngine.Rendering.HighDefinition.AdaptationMode.Fixed);
                exposure.limitMin.Override(-10f);
                exposure.limitMax.Override(20f);
                var fog = profile.Add<UnityEngine.Rendering.HighDefinition.Fog>(true);
                fog.enabled.Override(false);
                volume.sharedProfile = profile;
                for (int i = 0; i < 8; i++) cam.Render();
                RenderTexture.active = rt;
                var tex = new Texture2D(px, py, TextureFormat.RGB24, false);
                tex.ReadPixels(new Rect(0, 0, px, py), 0, 0);
                tex.Apply();
                RenderTexture.active = null;
                string outDir = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", "..", "data", "evidence"));
                Directory.CreateDirectory(outDir);
                string path = Path.Combine(outDir, "c1_topdown.png");
                File.WriteAllBytes(path, tex.EncodeToPNG());
                Object.DestroyImmediate(tex);
                rt.Release();
                Debug.Log($"[SmokeRescue] Top-down capture written: {path}");
            }
            finally
            {
                Object.DestroyImmediate(go);
                foreach (var r in ceilings) r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.On;
            }
        }
    }
}
