import sys, glob, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flipbook_lib import *
base = "/Volumes/Tashon/opencv/smoke-rescue/assets_cache/vfx_candidates"
files = sorted(glob.glob(base + "/unitylabs_*/extracted/*.tga") + glob.glob(base + "/unityvfxg_*/*.tga") + glob.glob(base + "/unitypkg_*/Textures/*/*_*.png") + glob.glob(base + "/unitypkg_VFXLearning*/*.png"))
cands = [(c, r) for c in (1,2,4,5,6,7,8,10,12,16,20,32) for r in (1,2,3,4,5,6,7,8,10,12,16)]
for f in files:
    a, mode = load(f)
    H, W, C = a.shape
    g = best_grid(a, cands)[:2]
    al = a[:,:,3] if C == 4 else None
    print(os.path.relpath(f, base), "| %dx%d C=%d mode=%s | max=%.2f mean=%.3f" % (W, H, C, mode, a.max(), a.mean()),
          "| alpha min/max %.2f/%.2f" % (al.min(), al.max()) if al is not None else "",
          "| best grids:", [("%dx%d" % x[0], round(x[3], 3)) for x in g])
