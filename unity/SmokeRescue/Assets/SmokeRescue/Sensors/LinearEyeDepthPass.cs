using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;

namespace SmokeRescue.Sensors
{
    /// <summary>Exports opaque geometry eye z before fog. Each pass belongs to one sensor camera.</summary>
    public sealed class LinearEyeDepthPass : CustomPass
    {
        public Camera SensorCamera;
        public RenderTexture Destination;
        public bool Armed;
        public int RenderCount { get; private set; }
        Material material;
        protected override void Setup(ScriptableRenderContext renderContext, CommandBuffer cmd)
        {
            var shader = Shader.Find("Hidden/SmokeRescue/LinearEyeDepth");
            if (shader == null) throw new System.InvalidOperationException("Missing LinearEyeDepth shader.");
            material = CoreUtils.CreateEngineMaterial(shader);
        }
        protected override void Execute(CustomPassContext ctx)
        {
            if (!Armed || ctx.hdCamera.camera != SensorCamera || Destination == null) return;
            CoreUtils.SetRenderTarget(ctx.cmd, Destination, ClearFlag.Color, Color.black);
            CoreUtils.DrawFullScreen(ctx.cmd, material);
            RenderCount++;
        }
        protected override void Cleanup() { CoreUtils.Destroy(material); }
    }
}
