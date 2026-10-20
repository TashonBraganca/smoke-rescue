"""Contact sheet for model candidates using each source site's own preview image.
python3 -I sheet_models_v2.py <category> <out.jpg> [cols]"""
import glob, json, os, sys
from PIL import Image, ImageDraw, ImageFont
ROOT = "/Volumes/Tashon/opencv/smoke-rescue/assets_cache/candidates_v2"
cat, out = sys.argv[1], sys.argv[2]
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 6
CELL = 270
folders = sorted(glob.glob(f"{ROOT}/{cat}/*"))
rows = (len(folders) + cols - 1) // cols
font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 12)
sheet = Image.new("RGB", (cols * CELL, rows * (CELL + 46)), (28, 28, 30))
d = ImageDraw.Draw(sheet)
for i, f in enumerate(folders):
    m = json.load(open(f + "/meta.json")); s = m.get("stats", {})
    x, y = (i % cols) * CELL, (i // cols) * (CELL + 46)
    pv = m.get("preview")
    try:
        im = Image.open(f"{f}/{pv}").convert("RGBA")
        bg = Image.new("RGBA", im.size, (58, 58, 62, 255)); bg.alpha_composite(im); im = bg.convert("RGB")
        im.thumbnail((CELL - 8, CELL - 8)); sheet.paste(im, (x + 4 + (CELL - 8 - im.width) // 2, y + 4 + (CELL - 8 - im.height) // 2))
    except Exception as e:
        d.text((x + 8, y + 8), "no preview", fill=(255, 90, 90), font=font)
    tris = s.get("tris", 0)
    flag = " >300k!" if tris > 300000 else ""
    d.text((x + 6, y + CELL), m["id"][:CELL // 7], fill=(240, 240, 240), font=font)
    d.text((x + 6, y + CELL + 15), f"{tris/1000:.1f}k tris {m['licence']}{flag}", fill=(170, 200, 170), font=font)
    d.text((x + 6, y + CELL + 30), (m.get("label") or m["name"])[:CELL // 7], fill=(200, 200, 140), font=font)
sheet.save(out, quality=88); print(out, sheet.size)
