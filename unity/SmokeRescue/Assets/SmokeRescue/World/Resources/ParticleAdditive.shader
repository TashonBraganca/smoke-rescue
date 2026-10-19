// Unlit additive particle shader for fire and embers: texture x vertex colour x intensity.
// Uses the SRPDefaultUnlit pass, which HDRP draws in its transparent pass.
Shader "SmokeRescue/ParticleAdditive"
{
    Properties { _MainTex ("Texture", 2D) = "white" {} _Intensity ("Intensity", Float) = 1 }
    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" "IgnoreProjector"="True" }
        Pass
        {
            Tags { "LightMode"="SRPDefaultUnlit" }
            Blend One One
            ZWrite Off
            Cull Off
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"
            sampler2D _MainTex; float _Intensity;
            struct a2v { float4 vertex : POSITION; float4 color : COLOR; float2 uv : TEXCOORD0; };
            struct v2f { float4 pos : SV_POSITION; float4 color : COLOR; float2 uv : TEXCOORD0; };
            v2f vert(a2v v) { v2f o; o.pos = UnityObjectToClipPos(v.vertex); o.color = v.color; o.uv = v.uv; return o; }
            float4 frag(v2f i) : SV_Target
            {
                float4 t = tex2D(_MainTex, i.uv);
                float3 c = t.rgb * i.color.rgb * t.a * i.color.a * _Intensity;
                return float4(c, 0);
            }
            ENDCG
        }
    }
}
