import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flipbook_lib import *
def seam_score(a, c, r):
    H, W = a.shape[:2]
    lum = a[:, :, :3].max(axis=2) * (a[:, :, 3] if a.shape[2] == 4 else 1)
    xs = [round(i*W/c) for i in range(1, c)]; ys = [round(j*H/r) for j in range(1, r)]
    seam = np.mean([lum[:, x-1:x+1].mean() for x in xs] + [lum[y-1:y+1, :].mean() for y in ys])
    return seam / (lum.mean() + 1e-9)
for f in sys.argv[1:]:
    a, _ = load(f); H, W = a.shape[:2]
    res = []
    for c in range(1, 33):
        for r in range(1, 17):
            if c*r < 8 or c*r > 128: continue
            res.append((seam_score(a, c, r), c, r))
    res.sort()
    print(os.path.basename(f), "%dx%d" % (W, H), [("%dx%d" % (c, r), round(s, 3)) for s, c, r in res[:6]])
