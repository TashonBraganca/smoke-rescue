#!/bin/bash
# Download selected files from Unity-Technologies/VisualEffectGraph-Samples (Unity Companion License).
# media.githubusercontent.com resolves git-lfs pointers; plain raw works for non-LFS.
BASE=/Volumes/Tashon/opencv/smoke-rescue/assets_cache/vfx_candidates
R=https://media.githubusercontent.com/media/Unity-Technologies/VisualEffectGraph-Samples/master/Assets/Samples
get() { # folder path-under-Samples
  d="$BASE/$1"; mkdir -p "$d"; out="$d/$(basename "$2")"
  p=$(echo "$2" | sed 's/ /%20/g')
  curl -sfL -m 600 -o "$out" "$R/$p" || curl -sfL -m 600 -o "$out" "https://raw.githubusercontent.com/Unity-Technologies/VisualEffectGraph-Samples/master/Assets/Samples/$p" || echo "FAIL $2"
}
f=unityvfxg_SmokePortal_6way
for x in Portal_B3-64_A.tga Portal_B3-64_B.tga Bonefire_E.tga BonefireWind_E.tga TX_Pyro_SteamBall_A_N.tga TX_Pyro_SteamBall_A_P.tga Portal_B3-64_A_MotionVectors_Intensity-1484.exr; do get $f SmokePortal/6way_Textures/$x; done
for x in VFX_PortalSmoke.vfx VFX_BraziersFlames.vfx; do get $f SmokePortal/VFXGraphs/$x; done
f=unityvfxg_BonFire_vfx
for x in BonFire.vfx Shaders/BonfireFlame.shadergraph Shaders/BonfireSmoke.shadergraph Textures/PerlinNoiseRGB.png Textures/smoke-flowmap.png; do get $f BonFire/VFX/$x; done
f=unityvfxg_MagicBook_FireEmbers_Smoke
for x in FireEmbers.vfx Smoke.vfx; do get $f MagicBook/VFX/$x; done
for x in WispySmoke03b_8x8.tga WispySmoke03b_8x8_N.tga; do get $f MagicBook/Art/Textures/$x; done
f=unityvfxg_GenieMagicLamp_Fire7x7
get $f GenieMagicLamp/Art/Fire_7x7.tga
get $f EllenSkinnedMesh/Effects/FireSampling.vfx
