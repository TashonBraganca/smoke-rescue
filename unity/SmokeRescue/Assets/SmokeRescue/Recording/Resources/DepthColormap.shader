// Maps linear optical depth in metres (red channel) to a turbo-like colour ramp for the showcase depth view.
Shader "Hidden/SmokeRescue/DepthColormap"
{
    Properties { _MainTex ("Depth", 2D) = "black" {} _MaxDepth ("Max depth (m)", Float) = 8 }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        ZWrite Off ZTest Always Cull Off
        Pass
        {
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag
            #include "UnityCG.cginc"
            sampler2D _MainTex;
            float _MaxDepth;
            float3 ramp(float t)
            {
                t = saturate(t);
                return saturate(float3(1.5 - abs(4.0 * t - 3.0), 1.5 - abs(4.0 * t - 2.0), 1.5 - abs(4.0 * t - 1.0)));
            }
            fixed4 frag(v2f_img i) : SV_Target
            {
                float z = tex2D(_MainTex, float2(i.uv.x, 1.0 - i.uv.y)).r;
                if (z <= 0.2 || z > _MaxDepth) return fixed4(0, 0, 0, 1);
                return fixed4(ramp(z / _MaxDepth), 1);
            }
            ENDCG
        }
    }
}
