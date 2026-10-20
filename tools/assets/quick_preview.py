import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flipbook_lib import *
out = sys.argv[1]; files = sys.argv[2:]
tiles = []
for f in files:
    a, _ = load(f)
    d = to_display(a)
    h, w = d.shape[:2]
    s = 480 / max(h, w)
    d = cv2.resize(d, (max(1, int(w*s)), max(1, int(h*s))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((520, 500, 3), np.uint8) + 20
    canvas[:d.shape[0], :d.shape[1]] = d
    cv2.putText(canvas, os.path.basename(f)[:40], (4, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    tiles.append(canvas)
cols = 3
while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i+cols]) for i in range(0, len(tiles), cols)]
img = np.vstack(rows)
Image.fromarray(img).save(out, quality=85)
print(img.shape)
