using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

namespace SmokeRescue.Sensors
{
    /// <summary>Draws every opaque renderer with the thermal override material into a float target (degrees C).
    /// Each pass belongs to one thermal camera.</summary>
    [System.Serializable]
    public sealed class ThermalPass : CustomPass
    {
        public Camera SensorCamera;
        public RenderTexture Destination, DepthBuffer;
        public bool Armed;
        public float BackgroundC = 10f;    // night air outside the windows
        public int RenderCount { get; private set; }
        Material material;
        protected override void Setup(ScriptableRenderContext renderContext, CommandBuffer cmd)
        {
            var shader = Shader.Find("Hidden/SmokeRescue/ThermalOverride");
            if (shader == null) throw new System.InvalidOperationException("Missing ThermalOverride shader.");
            material = CoreUtils.CreateEngineMaterial(shader);
        }
        protected override void Execute(CustomPassContext ctx)
        {
            if (!Armed || ctx.hdCamera.camera != SensorCamera || Destination == null) return;
            CoreUtils.SetRenderTarget(ctx.cmd, Destination, DepthBuffer, ClearFlag.All, new Color(BackgroundC, 0, 0, 0));
            CustomPassUtils.DrawRenderers(ctx, ~0, RenderQueueType.AllOpaque, material);
            RenderCount++;
        }
        protected override void Cleanup() { CoreUtils.Destroy(material); }
    }
}
