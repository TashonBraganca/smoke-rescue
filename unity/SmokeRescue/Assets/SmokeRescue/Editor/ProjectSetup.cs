using System.IO;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

namespace SmokeRescue.Editor
{
    /// <summary>
    /// One-time project configuration, safe to rerun. Batchmode entry:
    ///   Unity -batchmode -projectPath unity/SmokeRescue -executeMethod SmokeRescue.Editor.ProjectSetup.Run -quit
    /// </summary>
    public static class ProjectSetup
    {
        public const string SettingsDir = "Assets/SmokeRescue/Settings";
        public const string HdrpAssetPath = SettingsDir + "/HDRenderPipelineAsset.asset";

        static readonly (int index, string name)[] Layers =
        {
            (8, "Walls"), (9, "Ceiling"), (10, "Robots"), (11, "Props"),
        };

        [MenuItem("Tools/Smoke Rescue/Setup Project")]
        public static void Run()
        {
            EnsureLayers();
            var asset = EnsureHdrpAsset();
            GraphicsSettings.defaultRenderPipeline = asset;
            for (int i = 0; i < QualitySettings.names.Length; i++)
            {
                QualitySettings.SetQualityLevel(i, false);
                QualitySettings.renderPipeline = asset;
            }
            PlayerSettings.colorSpace = ColorSpace.Linear;
            PlayerSettings.runInBackground = true;   // showcase and play-mode tests keep running when the editor is not focused
            AssetDatabase.SaveAssets();
            Debug.Log($"[SmokeRescue] Project setup done. Pipeline: {GraphicsSettings.defaultRenderPipeline?.name}");
        }

        [MenuItem("Tools/Smoke Rescue/Setup Layers")]
        public static void EnsureLayers()
        {
            var tagManager = new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/TagManager.asset")[0]);
            var layers = tagManager.FindProperty("layers");
            foreach (var (index, name) in Layers)
            {
                var prop = layers.GetArrayElementAtIndex(index);
                if (prop.stringValue == name) continue;
                if (!string.IsNullOrEmpty(prop.stringValue))
                    Debug.LogWarning($"[SmokeRescue] Layer {index} was '{prop.stringValue}', now '{name}'");
                prop.stringValue = name;
            }
            tagManager.ApplyModifiedPropertiesWithoutUndo();
        }

        static HDRenderPipelineAsset EnsureHdrpAsset()
        {
            Directory.CreateDirectory(SettingsDir);
            var asset = AssetDatabase.LoadAssetAtPath<HDRenderPipelineAsset>(HdrpAssetPath);
            if (asset == null)
            {
                asset = ScriptableObject.CreateInstance<HDRenderPipelineAsset>();
                AssetDatabase.CreateAsset(asset, HdrpAssetPath);
            }
            // Features the simulation needs: volumetric fog and local fog volumes for smoke, custom passes for
            // linear depth capture, decals for soot, and enough shadowed lights for fires and headlights.
            var so = new SerializedObject(asset);
            void Set(string path, bool value)
            {
                var p = so.FindProperty(path);
                if (p == null) { Debug.LogWarning($"[SmokeRescue] HDRP setting not found: {path}"); return; }
                p.boolValue = value;
            }
            Set("m_RenderPipelineSettings.supportVolumetrics", true);
            Set("m_RenderPipelineSettings.supportVolumetricClouds", false);
            Set("m_RenderPipelineSettings.supportCustomPass", true);
            Set("m_RenderPipelineSettings.supportDecals", true);
            Set("m_RenderPipelineSettings.supportSSAO", true);
            Set("m_RenderPipelineSettings.supportSSR", true);
            Set("m_RenderPipelineSettings.supportMotionVectors", true);
            so.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(asset);
            return asset;
        }
    }
}
