"""Build all contact sheets + manifest.json. Run: python3 -I final_build.py"""
import sys, os, json, glob, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_sheets as bs
from build_sheets import *
SCR = "/tmp/scratch"
CC0 = "CC0 1.0 Universal: https://creativecommons.org/publicdomain/zero/1.0/"
manifest = []; sheets = {}
for asset in bs.A:
    p = os.path.join(BASE, asset["file"])
    if not os.path.exists(p): continue
    a, mode = load(p); H, W, C = a.shape; c, r = asset["grid"]
    ck = checks(a, c, r)
    print("%-30s %5dx%-5d C=%d %-5s grid=%dx%d frame=%.0fx%.0f n=%d near/far=%.2f" % (asset["id"], W, H, C, mode, c, r, W/c, H/r, ck["n"], ck["near"]/max(ck["far"], 1e-9)))
    sheets.setdefault(asset["category"], []).append(tile(asset, a, c, r))
    m = {k: asset[k] for k in ("id", "category", "url", "licence", "attribution", "desc")}
    m.update(local_path=p, width=W, height=H, channels=C, pil_mode_or_dtype=mode, grid="%dx%d" % (c, r), frames=c*r, frame_px="%.0fx%.0f" % (W/c, H/r))
    manifest.append(m)
# --- embers: Kenney sprites + para fire sheets
K = BASE + "/kenney_particle-pack/extracted/PNG (Black background)/"
names = ["spark_01","spark_02","spark_03","spark_04","spark_05","spark_06","spark_07","flare_01","star_01","light_01","fire_01","fire_02","flame_01","flame_02","flame_03","flame_04","flame_05","flame_06","smoke_01","smoke_05"]
tiles = []
for n in names:
    a, m = load(K + n + ".png"); d = cv2.resize(to_display(a[:, :, :3] if a.shape[2] > 3 else a, over=(0, 0, 0)), (180, 180), interpolation=cv2.INTER_AREA)
    d = np.ascontiguousarray(d); label(d, n, 0.4); tiles.append(d)
    print("kenney", n, a.shape, m)
while len(tiles) % 10: tiles.append(np.zeros_like(tiles[0]))
krow = np.vstack([np.hstack(tiles[i:i+10]) for i in range(0, len(tiles), 10)])
P = BASE + "/opengameart_para_particlefx-2/extracted/"
prow = []
for n in ["fire_01", "fire_01b", "fire_01c", "fire_02", "lighter_flame_01"]:
    a, m = load(P + n + ".png"); H, W, C = a.shape
    ck = checks(a, 8, 8)
    print("para", n, W, H, C, m, "near/far=%.2f" % (ck["near"]/max(ck["far"], 1e-9)))
    d = cv2.resize(to_display(a), (360, 360), interpolation=cv2.INTER_AREA); d = np.ascontiguousarray(d); label(d, "para_" + n + " %dx%d C=%d 8x8" % (W, H, C), 0.4); prow.append(d)
    manifest.append(dict(id="opengameart_para_" + n, category="flame", url="https://opengameart.org/content/animated-particle-effects-2", licence=CC0,
        attribution="para (OpenGameArt). CC0, credit not required.", desc="stylised 2D fire sheet, 128px frames", local_path=P + n + ".png", width=W, height=H, channels=C, pil_mode_or_dtype=m, grid="8x8", frames=64, frame_px="%dx%d" % (W//8, H//8)))
prow = np.hstack(prow); 
W2 = max(krow.shape[1], prow.shape[1]); pad = lambda x: np.hstack([x, np.zeros((x.shape[0], W2 - x.shape[1], 3), np.uint8)])
sheets["embers"] = [pad(krow), pad(prow)]
for f in ["spark_01", "flare_01", "fire_01", "flame_01", "smoke_01"]:
    pass
manifest.append(dict(id="kenney_particle-pack", category="embers", url="https://kenney.nl/assets/particle-pack", licence=CC0 + " (License.txt in zip: 'You may use these assets in personal and commercial projects. Credit (Kenney or www.kenney.nl) would be nice but is not mandatory.')",
    attribution="Kenney Vleugels (Kenney.nl), Particle Pack 1.1. CC0, credit optional.", desc="~80 single 512px greyscale particle sprites: sparks, flares, flames, smoke, circles; includes Unity samples",
    local_path=BASE + "/kenney_particle-pack/extracted", width=512, height=512, channels=1, pil_mode_or_dtype="P (palette) PNG", grid="single sprites", frames=1, frame_px="512x512"))
# --- VFX Graph assets sheet (screenshots published by Unity in the repo; the .vfx themselves are not renderable outside Unity)
vf = [BASE + "/unityvfxg_BonFire_vfx/previews/BonFire-screenshot.png", BASE + "/unityvfxg_SmokePortal_6way/previews/Portal-screenshot.png", BASE + "/unityvfxg_SmokePortal_6way/previews/SmokeSample.png", BASE + "/unityvfxg_MagicBook_FireEmbers_Smoke/MagicBook-screenshot.png"]
vt = []
for f in vf:
    a, m = load(f); d = cv2.resize(to_display(a[:, :, :3]), (640, 360), interpolation=cv2.INTER_AREA); d = np.ascontiguousarray(d); label(d, os.path.basename(f) + " (Unity-published screenshot)", 0.45); vt.append(d)
    print("screenshot", f.split("/")[-1], a.shape)
sheets["vfxgraph"] = [np.vstack([np.hstack(vt[:2]), np.hstack(vt[2:])])]
# --- VDB: preview frames from JangaFX's own mp4
vd = []
for t in (0, 2, 4, 6):
    png = SCR + "/camp_%d.png" % t
    if os.path.exists(png):
        a, m = load(png); d = cv2.resize(to_display(a[:, :, :3]), (320, 568), interpolation=cv2.INTER_AREA); d = np.ascontiguousarray(d); label(d, "smallCampfire.mp4 t=%ds (JangaFX preview)" % t, 0.4); vd.append(d)
sheets["vdb"] = [np.hstack(vd)]
os.makedirs(BASE + "/_sheets", exist_ok=True)
for cat, tl in sheets.items():
    W2 = max(t.shape[1] for t in tl)
    tl = [np.hstack([t, np.zeros((t.shape[0], W2 - t.shape[1], 3), np.uint8)]) if t.shape[1] < W2 else t for t in tl]
    img = np.vstack(tl); Image.fromarray(img).save(BASE + "/_sheets/%s.jpg" % cat, quality=90); print("sheet", cat, img.shape)
# --- non-flipbook assets
sp = BASE + "/unityvfxg_SmokePortal_6way/"
manifest += [
 dict(id="unitypkg_VFXGraphAdditions", category="vfxgraph", url="local copy from Unity 6000.6.6f1 install: Unity.app/Contents/Resources/PackageManager/BuiltInPackages/com.unity.visualeffectgraph/Samples~/VFXGraphAdditions (package v17.6.0)", licence=UCL,
      attribution="Unity Technologies", desc="ready-made VFX Graph assets Flames.vfx, Smoke.vfx, Sparks.vfx, Lightning.vfx plus Flame03_Temp_16x5 and WispySmoke03b (color, normal, distortion) textures and subgraphs. Compatible with 17.6 HDRP by construction (shipped with it). Import via Package Manager > Visual Effect Graph > Samples.",
      local_path=BASE + "/unitypkg_VFXGraphAdditions", width=None, height=None, channels=None, pil_mode_or_dtype=None, grid="n/a", frames=None, frame_px=None),
 dict(id="unityvfxg_BonFire_vfx", category="vfxgraph", url="https://github.com/Unity-Technologies/VisualEffectGraph-Samples/tree/master/Assets/Samples/BonFire", licence=UCL + "; repo LICENSE-3rdParty.md: Kenney Castle/Nature Kit CC0",
      attribution="Unity Technologies VisualEffectGraph-Samples", desc="BonFire.vfx + BonfireFlame/BonfireSmoke shader graphs (2019-era, VFX Graph 7.x; will need upgrade/re-save in 17.6, not tested)",
      local_path=BASE + "/unityvfxg_BonFire_vfx", width=None, height=None, channels=None, pil_mode_or_dtype=None, grid="n/a", frames=None, frame_px=None),
 dict(id="unityvfxg_MagicBook_FireEmbers_Smoke", category="vfxgraph", url="https://github.com/Unity-Technologies/VisualEffectGraph-Samples/tree/master/Assets/Samples/MagicBook", licence=UCL,
      attribution="Unity Technologies VisualEffectGraph-Samples", desc="FireEmbers.vfx, Smoke.vfx (+ WispySmoke03b color/normal 512px copies)", local_path=BASE + "/unityvfxg_MagicBook_FireEmbers_Smoke", width=None, height=None, channels=None, pil_mode_or_dtype=None, grid="n/a", frames=None, frame_px=None),
 dict(id="unityvfxg_SmokePortal_6way_vfx", category="vfxgraph", url="https://github.com/Unity-Technologies/VisualEffectGraph-Samples/tree/master/Assets/Samples/SmokePortal", licence=UCL,
      attribution="Unity Technologies VisualEffectGraph-Samples", desc="VFX_PortalSmoke.vfx, VFX_BraziersFlames.vfx (HDRP lit 6-way smoke sample) + Portal_B3-64 motion vector EXR", local_path=sp, width=None, height=None, channels=None, pil_mode_or_dtype=None, grid="n/a", frames=None, frame_px=None),
 dict(id="unityvfxg_GenieMagicLamp_Fire7x7", category="flame", url="https://github.com/Unity-Technologies/VisualEffectGraph-Samples/tree/master/Assets/Samples/GenieMagicLamp/Art/Fire_7x7.tga", licence=UCL,
      attribution="Unity Technologies", desc="Fire_7x7.tga grayscale flame mask 512x1024 (7x7 per filename; frame size 73x146 so very low res) + FireSampling.vfx", local_path=BASE + "/unityvfxg_GenieMagicLamp_Fire7x7/Fire_7x7.tga", width=512, height=1024, channels=4, pil_mode_or_dtype="RGBA", grid="7x7 (from filename, not verified)", frames=49, frame_px="73x146"),
]
for name, nfr, vdesc in (("jangafx_SmallCampfireVDB", 200, "small campfire, 200 VDB frames, channels density/flames/temperature (+ .ember project and mp4 preview), 456 MB unpacked"), ("jangafx_Fire_01_vdb", 101, "EmberGen fire, 101 VDB frames, channels flames/temperature/density, 171 MB (RAR, extracted with bsdtar)")):
    manifest.append(dict(id=name, category="vdb", url="https://jangafx.com/software/embergen/download/free-vdb-animations", licence="CC0 (page: 'licensed as CC0 (Public Domain)'; LICENSE.txt inside archive is the CC0 1.0 legal code)",
        attribution="JangaFX (EmberGen). CC0, credit not required (they ask you to spread the word).", desc=vdesc + ". Unity has no native VDB import: bake to flipbook/3D texture in Blender/Houdini first.",
        local_path=BASE + "/" + name, width=None, height=None, channels=None, pil_mode_or_dtype=None, grid="n/a (volume)", frames=nfr, frame_px=None))
json.dump(manifest, open(BASE + "/manifest.json", "w"), indent=1, default=str)
print("manifest entries", len(manifest))
