// Thermal (long-wave infrared) camera: every opaque renderer is drawn with this override material, which writes
// apparent temperature in degrees C to a float target instead of colour.
//   surface  = max(per-renderer base temperature, world heat field at this point)
//   gas      = hot smoke layer between the camera and the surface adds its own emission along the ray
// The heat field is a top-down texture the world updates from the fire state:
//   R surface temperature C, G upper-layer gas temperature C, B layer height above floor m, A gas absorption per m.
// Temperatures mix linearly, not as radiance (T^4); a documented simplification.
Shader "Hidden/SmokeRescue/ThermalOverride"
{
    Properties
    {
        _ThermalBaseC("Base temperature C", Float) = 20
    }
    HLSLINCLUDE
    #pragma target 4.5
    #pragma only_renderers metal vulkan d3d11
    #pragma multi_compile_instancing
    ENDHLSL
    SubShader
    {
        Tags { "RenderPipeline" = "HDRenderPipeline" }
        Pass
        {
            Name "FirstPass"
            Tags { "LightMode" = "FirstPass" }
            Blend Off
            ZWrite On
            ZTest LEqual
            Cull Off
            HLSLPROGRAM
            #define ATTRIBUTES_NEED_NORMAL
            #define VARYINGS_NEED_POSITION_WS
            #include "Packages/com.unity.render-pipelines.high-definition/Runtime/RenderPipeline/RenderPass/CustomPass/CustomPassRenderers.hlsl"

            TEXTURE2D(_ThermalField);
            float4 _ThermalFieldRect;      // x0, z0, 1/width, 1/depth in metres
            float _ThermalAmbientC;
            float _ThermalBaseC;           // per renderer through a MaterialPropertyBlock (people, robots, flames)

            void GetSurfaceAndBuiltinData(FragInputs fragInputs, float3 viewDirection, inout PositionInputs posInput,
                                          out SurfaceData surfaceData, out BuiltinData builtinData)
            {
                float3 p = GetAbsolutePositionWS(fragInputs.positionRWS);
                float3 c = GetAbsolutePositionWS(float3(0, 0, 0));
                float2 uv = (p.xz - _ThermalFieldRect.xy) * _ThermalFieldRect.zw;
                float4 field = SAMPLE_TEXTURE2D_LOD(_ThermalField, s_linear_clamp_sampler, uv, 0);
                float layerH = field.b, gasC = field.g;

                float surf = max(_ThermalBaseC, field.r);
                // Objects sit a little above or below the air around them (thermal mass, last sun, appliances): a fixed
                // per-object offset of up to 1.5 C, hashed from the object's origin, so furniture separates from walls.
                float3 origin = GetAbsolutePositionWS(GetObjectToWorldMatrix()._m03_m13_m23);
                float h = frac(sin(dot(origin, float3(12.9898, 78.233, 37.719))) * 43758.5453);
                surf += (h - 0.5) * 3.0 * step(_ThermalBaseC, 25);   // people, robots and flames keep their own temperature
                // Walls and ceiling inside the hot layer heat toward the gas temperature.
                surf = max(surf, lerp(_ThermalAmbientC, gasC, 0.7 * saturate((p.y - layerH) / 0.4)));

                // Emission of the hot layer along the camera ray: the part of the segment above the layer height.
                float len = length(p - c);
                float lo = min(c.y, p.y), hi = max(c.y, p.y);
                float above = hi <= layerH ? 0 : (lo >= layerH ? 1 : (hi - layerH) / max(hi - lo, 1e-4));
                float eps = 1 - exp(-field.a * len * above);
                float t = lerp(surf, gasC, eps);

                ZERO_BUILTIN_INITIALIZE(builtinData);
                ZERO_INITIALIZE(SurfaceData, surfaceData);
                builtinData.opacity = 1;
                surfaceData.color = float3(t, 0, 0);
            }
            #include "Packages/com.unity.render-pipelines.high-definition/Runtime/RenderPipeline/ShaderPass/ShaderPassForwardUnlit.hlsl"
            #pragma vertex Vert
            #pragma fragment Frag
            ENDHLSL
        }
    }
    Fallback Off
}
