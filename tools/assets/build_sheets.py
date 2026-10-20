"""Verify flipbook grids, print dims/channels, build per-category contact sheets and manifest.
Run: python3 -I build_sheets.py [--only id1,id2] [--no-manifest]"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flipbook_lib import *
BASE = "/Volumes/Tashon/opencv/smoke-rescue/assets_cache/vfx_candidates"
UL = "https://unity.com/blog/engine-platform/free-vfx-image-sequences-flipbooks"
UL_LIC = "CC0 1.0 (stated in the Unity blog post: 'image sequences we want to share with you under CC0 license'); https://creativecommons.org/publicdomain/zero/1.0/"
UL_ATTR = "Unity Labs Paris / Thomas Iche (Houdini sims). CC0, credit not required."
UCL = "Unity Companion License (LICENSE.md of package / repo): http://www.unity3d.com/legal/licenses/Unity_Companion_License ; use limited to Unity-dependent projects"
def ul(name, fn, grid, cat, desc, bg="dark"):
    return dict(id="unitylabs_"+name, category=cat, file=f"unitylabs_{name}/extracted/{fn}", grid=grid, bg=bg,
                url=f"https://unity3d.com/files/labs/downloads/vfx/assets01/{name}/{name}-flipbooks.zip", page=UL,
                licence=UL_LIC, attribution=UL_ATTR, desc=desc)
A = [
 ul("Flame02","Flame02_16x4.tga",(16,4),"flame","tall campfire-style flame, 64 frames"),
 ul("Flame03","Flame03_16x4.tga",(16,4),"flame","thin tall flame, 64 frames, low res"),
 ul("Flame03-hollow","Flame03-hollow_16x4.tga",(16,4),"flame","flame with hollow core, orange, RGB only"),
 ul("SmallFlame01","SmallFlame01_16x4.tga",(16,4),"flame","small flame, 64 frames"),
 ul("SmallFlame01-mini","SmallFlame01-mini_16x4.tga",(16,4),"flame","smaller flame variant"),
 ul("FireBall02","FireBall02_8x8.tga",(8,8),"flame","orange fireball / burst, 64 frames"),
 ul("FireBall03","FireBall03_8x8.tga",(8,8),"flame","yellow-white fireball, 64 frames"),
 ul("Explosion02HD","Explosion02HD_5x5.tga",(5,5),"flame","HD explosion, 25 frames at 2048"),
 ul("WispySmoke01","WispySmoke01_8x8.tga",(8,8),"smoke","thin wispy smoke","grey"),
 ul("WispySmoke03b","WispySmoke03b_8x8.tga",(8,8),"smoke","wispy smoke (has normal map in pkg)","grey"),
 ul("Cloud01","Cloud01_8x8.tga",(8,8),"smoke","billowing cloud/smoke puff","grey"),
 ul("Cloud03","Cloud03_8x8.tga",(8,8),"smoke","billowing cloud/smoke puff","grey"),
 ul("Explosion01-nofire","Explosion01-nofire_5x5.tga",(5,5),"smoke","explosion smoke only, 25 frames","grey"),
 ul("DiscSmoke01","DiscSmoke01_16x4.tga",(16,4),"smoke","disc/ground smoke ring, 64 frames","grey"),
 
]
SP = "unityvfxg_SmokePortal_6way"
VG = "https://github.com/Unity-Technologies/VisualEffectGraph-Samples (Assets/Samples/SmokePortal/6way_Textures, git-lfs)"
A += [
 dict(id="unityvfxg_Bonefire_E", category="flame", file=f"{SP}/Bonefire_E.tga", grid=(4,8), bg="dark", url=VG, licence=UCL,
      attribution="Unity Technologies VisualEffectGraph-Samples (BonFire/SmokePortal)", desc="small bonfire flame emissive flipbook, 32 frames, 256px"),
 dict(id="unityvfxg_Pyro_SteamBall_P", category="smoke", file=f"{SP}/TX_Pyro_SteamBall_A_P.tga", grid=(8,8), bg="grey", url=VG, licence=UCL,
      attribution="Unity Technologies VisualEffectGraph-Samples", desc="6-way lit smoke, positive-axis lightmap (RGB) + alpha. Needs HDRP 6-way shader; pair with _N"),
 dict(id="unityvfxg_Pyro_SteamBall_N", category="smoke", file=f"{SP}/TX_Pyro_SteamBall_A_N.tga", grid=(8,8), bg="grey", url=VG, licence=UCL,
      attribution="Unity Technologies VisualEffectGraph-Samples", desc="6-way lit smoke, negative-axis lightmap (RGB)"),
 dict(id="unityvfxg_Portal_B3-64_A", category="smoke", file=f"{SP}/Portal_B3-64_A.tga", grid=(8,8), bg="grey", url=VG, licence=UCL,
      attribution="Unity Technologies VisualEffectGraph-Samples", desc="6-way lit swirling smoke, positive axes + alpha, 2048, with motion vector EXR"),
 dict(id="unitypkg_Flame03_Temp_16x5", category="flame", file="unitypkg_VFXGraphAdditions/Textures/Fire/Flame03_Temp_16x5.png", grid=(16,5), bg="dark",
      url="local: Unity 6000.6.6f1 com.unity.visualeffectgraph 17.6.0 Samples~/VFXGraphAdditions", licence=UCL, attribution="Unity Technologies (derived from Unity Labs Flame03)", desc="temperature/gradient-ramp flame, 80 frames, ships with Flames.vfx"),
 dict(id="unitypkg_WispySmoke03b_N", category="smoke", file="unitypkg_VFXGraphAdditions/Textures/Smoke/WispySmoke03b_8x8_N.png", grid=(8,8), bg="grey",
      url="local: Unity 6000.6.6f1 com.unity.visualeffectgraph 17.6.0 Samples~/VFXGraphAdditions", licence=UCL, attribution="Unity Technologies", desc="normal map for WispySmoke03b"),
 dict(id="unitypkg_Flipbook_Smoke", category="smoke", file="unitypkg_VFXLearningTemplates_flipbooks/TX_Flipbook_Smoke.png", grid=(4,4), bg="grey",
      url="local: Unity 6000.6.6f1 com.unity.visualeffectgraph 17.6.0 Samples~/VFXLearningTemplates", licence=UCL, attribution="Unity Technologies", desc="small smoke flipbook with motion vector pair"),
]
ONLY = None
if "--only" in sys.argv: ONLY = sys.argv[sys.argv.index("--only")+1].split(",")

def cells(a, c, r):
    H, W = a.shape[:2]
    out = []
    for j in range(r):
        for i in range(c):
            y0, y1 = round(j*H/r), round((j+1)*H/r); x0, x1 = round(i*W/c), round((i+1)*W/c)
            out.append(a[y0:y1, x0:x1])
    return out

def checks(a, c, r):
    H, W = a.shape[:2]
    fr = cells(a, c, r); n = len(fr)
    sm = [cv2.resize(f[:, :, :min(4, f.shape[2])], (24, 24), interpolation=cv2.INTER_AREA) for f in fr]
    near = np.mean([np.abs(sm[i]-sm[i+1]).mean() for i in range(n-1)])
    far = np.mean([np.abs(sm[i]-sm[(i+n//2) % n]).mean() for i in range(n)])
    # seam energy: mean |value| in 2px bands at declared cell borders vs same bands shifted half a cell
    def band(off):
        e = []
        for j in range(r):
            for i in range(c):
                y0, x0 = int(round(j*H/r + off*H/r/2)), int(round(i*W/c + off*W/c/2))
                e.append(a[y0:y0+2, :].mean() if False else 0)
        return 0
    lum = a[:, :, :3].max(axis=2)*(a[:, :, 3] if a.shape[2] == 4 else 1)
    xs = [round(i*W/c) for i in range(1, c)]; ys = [round(j*H/r) for j in range(1, r)]
    xh = [round((i+.5)*W/c) for i in range(c)]; yh = [round((j+.5)*H/r) for j in range(r)]
    seam = np.mean([lum[:, x-1:x+1].mean() for x in xs] + [lum[y-1:y+1, :].mean() for y in ys]) if (xs or ys) else 0
    mid = np.mean([lum[:, x-1:x+1].mean() for x in xh] + [lum[y-1:y+1, :].mean() for y in yh])
    return dict(n=n, near=round(float(near), 4), far=round(float(far), 4), seam=round(float(seam), 4), mid=round(float(mid), 4))

def label(img, text, scale=0.5):
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 1, cv2.LINE_AA)

def tile(asset, a, c, r):
    bgc = {"dark": (0.04, 0.04, 0.05), "grey": (0.45, 0.47, 0.5)}[asset["bg"]]
    d = to_display(a, over=bgc)
    H, W = d.shape[:2]
    fr = cells(a, c, r); n = len(fr)
    idx = [0, n//3, 2*n//3, n-1]
    TH = 300
    s = TH / max(H, W)
    atlas = cv2.resize(d, (max(1, int(W*s)), max(1, int(H*s))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((TH+24, 1280, 3), np.uint8); canvas[:] = [int(v*255) for v in bgc]
    canvas[22:22+atlas.shape[0], 0:atlas.shape[1]] = atlas
    x = atlas.shape[1] + 10
    for k in idx:
        f = to_display(fr[k], over=bgc)
        fh, fw = f.shape[:2]
        sc = min(TH / fh, (1280 - x) / 4 / fw * 0.98)
        f2 = cv2.resize(f, (max(1, int(fw*sc)), max(1, int(fh*sc))), interpolation=cv2.INTER_CUBIC)
        canvas[22:22+f2.shape[0], x:x+f2.shape[1]] = f2[:TH]
        x += f2.shape[1] + 6
    label(canvas, "%s  %dx%d C=%d grid %dx%d (frame %.0fx%.0f)" % (asset["id"], W, H, a.shape[2], c, r, W/c, H/r))
    return canvas

if __name__ == "__main__":
    sheets = {}; manifest = []
    for asset in A:
        if ONLY and asset["id"] not in ONLY: continue
        p = os.path.join(BASE, asset["file"])
        if not os.path.exists(p): print("MISSING", p); continue
        a, mode = load(p); H, W, C = a.shape; c, r = asset["grid"]
        ck = checks(a, c, r)
        # alternative grids for comparison
        alt = {}
        for (cc, rr) in [(c*2, r), (c, r*2), (max(1, c//2), r), (c, max(1, r//2))]:
            alt["%dx%d" % (cc, rr)] = checks(a, cc, rr)["near"]
        print("%-32s %5dx%-5d C=%d %-5s grid=%dx%d n=%d near/far=%.2f seam/mid=%.2f | alt near: %s" % (
            asset["id"], W, H, C, mode, c, r, ck["n"], ck["near"]/max(ck["far"], 1e-9), ck["seam"]/max(ck["mid"], 1e-9), alt))
        sheets.setdefault(asset["category"], []).append(tile(asset, a, c, r))
        m = {k: asset[k] for k in ("id", "category", "url", "licence", "attribution", "desc")}
        m.update(local_path=p, width=W, height=H, channels=C, pil_mode_or_dtype=mode, grid="%dx%d" % (c, r), frames=c*r,
                 frame_px="%.0fx%.0f" % (W/c, H/r), verify=ck)
        manifest.append(m)
    os.makedirs(BASE + "/_sheets", exist_ok=True)
    for cat, tiles in sheets.items():
        img = np.vstack(tiles)
        Image.fromarray(img).save(BASE + "/_sheets/%s.jpg" % cat, quality=88)
        print("sheet", cat, img.shape)
    json.dump(manifest, open("/tmp/scratch", "w"), indent=1)
