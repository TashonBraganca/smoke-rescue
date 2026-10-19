Shader "Hidden/SmokeRescue/LinearEyeDepth"
{
    HLSLINCLUDE
    #pragma vertex Vert
    #pragma target 4.5
    #pragma only_renderers metal vulkan d3d11
    #include "Packages/com.unity.render-pipelines.high-definition/Runtime/RenderPipeline/RenderPass/CustomPass/CustomPassCommon.hlsl"
    float DepthPass(Varyings input) : SV_Target
    {
        UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX(input);
        float deviceDepth = LoadCameraDepth(input.positionCS.xy);
        return deviceDepth == UNITY_RAW_FAR_CLIP_VALUE ? 0 : LinearEyeDepth(deviceDepth, _ZBufferParams);
    }
    ENDHLSL
    SubShader
    {
        Tags { "RenderPipeline" = "HDRenderPipeline" }
        Pass
        {
            Name "Linear eye depth"
            ZWrite Off ZTest Always Cull Off Blend Off
            HLSLPROGRAM
            #pragma fragment DepthPass
            ENDHLSL
        }
    }
    Fallback Off
}
