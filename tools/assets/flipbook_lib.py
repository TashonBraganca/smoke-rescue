"""Helpers to load and inspect flipbook atlases. Run with python3 -I from another cwd."""
import os, sys
sys.path.append(os.path.expanduser("~/Library/Python/3.14/lib/python/site-packages"))  # trusted user site; -I disables it
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
from PIL import Image
import cv2
Image.MAX_IMAGE_PIXELS = None

def load(path):
    """Return float32 array HxWxC in 0..1 (EXR: may exceed 1) and the native dtype string."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".exr":
        a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if a is None: raise IOError("cv2 could not read " + path)
        if a.ndim == 2: a = a[:, :, None]
        if a.shape[2] >= 3:  # BGR(A) -> RGB(A)
            a = a[:, :, [2, 1, 0] + ([3] if a.shape[2] == 4 else [])]
        return a.astype(np.float32), str(a.dtype)
    im = Image.open(path)
    mode = im.mode
    im.load()
    if mode in ('P', 'PA'): im = im.convert('RGBA')
    a = np.asarray(im)
    if a.dtype == np.uint16: a = a.astype(np.float32) / 65535.0
    else: a = a.astype(np.float32) / 255.0
    if a.ndim == 2: a = a[:, :, None]
    return a, mode

def frames(a, cols, rows):
    H, W = a.shape[:2]
    fh, fw = H // rows, W // cols
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append(a[r*fh:(r+1)*fh, c*fw:(c+1)*fw])
    return out

def coherence(a, cols, rows):
    """Mean abs diff of consecutive frames / mean abs diff of frames far apart. Low = true time sequence."""
    fr = frames(a, cols, rows)
    n = len(fr)
    # downsample for speed
    small = [cv2.resize(f[:, :, :min(f.shape[2], 4)], (32, 32), interpolation=cv2.INTER_AREA) for f in fr]
    near = np.mean([np.abs(small[i] - small[i+1]).mean() for i in range(n-1)])
    far = np.mean([np.abs(small[i] - small[(i + n//2) % n]).mean() for i in range(n)])
    return float(near), float(far)

def best_grid(a, cands):
    H, W = a.shape[:2]
    res = []
    for (c, r) in cands:
        if W % c or H % r: continue
        near, far = coherence(a, c, r)
        res.append(((c, r), near, far, near / (far + 1e-9)))
    return sorted(res, key=lambda x: x[3])

def to_display(a, gain=1.0, over=(0.06, 0.06, 0.07), use_alpha=True):
    """Tonemap float RGB(A) to uint8 RGB composited over a dark background."""
    c = a.shape[2]
    rgb = a[:, :, :3] if c >= 3 else np.repeat(a[:, :, :1], 3, axis=2)
    if c == 4 and use_alpha:
        al = np.clip(a[:, :, 3:4], 0, 1)
        rgb = rgb * al + np.array(over, np.float32) * (1 - al)
    elif c == 2 and use_alpha:  # luminance+alpha
        al = np.clip(a[:, :, 1:2], 0, 1)
        rgb = np.repeat(a[:, :, :1], 3, axis=2) * al + np.array(over, np.float32) * (1 - al)
    rgb = np.clip(rgb * gain, 0, None)
    rgb = rgb / (1 + rgb) * 2 if rgb.max() > 1.5 else np.clip(rgb, 0, 1)  # Reinhard-ish only for HDR
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
