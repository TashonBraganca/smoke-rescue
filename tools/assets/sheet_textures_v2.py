"""Contact sheet for texture candidates: python3 -I sheet_textures_v2.py <category> <out.jpg>
Shows the albedo tile (a 2x2 repeat crop so tiling and seams are visible). Decals are shown over a checker with their opacity map applied."""
import glob, json, os, sys
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = "/Volumes/Tashon/opencv/smoke-rescue/assets_cache/candidates_v2"
CELL = 380

def pick(folder, pats):
    fs = sorted(glob.glob(folder + "/*"))
    for p in pats:
        for f in fs:
            if p.lower() in os.path.basename(f).lower() and f.lower().endswith((".jpg", ".png")):
                return f
    return None

def render(folder, decal):
    alb = pick(folder, ["albedo", "_color", "basecolor", "_T.jpg", "Coal_Ashes"])
    im = Image.open(alb).convert("RGB")
    if decal:
        op = pick(folder, ["opacity"])
        if op:
            a = Image.open(op).convert("L").resize(im.size)
            chk = Image.new("RGB", im.size, (90, 90, 96))
            d = ImageDraw.Draw(chk)
            for y in range(0, im.size[1], 64):
                for x in range(0, im.size[0], 64):
                    if (x // 64 + y // 64) % 2:
                        d.rectangle([x, y, x + 63, y + 63], fill=(130, 130, 136))
            return Image.composite(im, chk, a)
        return im
    w, h = im.size
    t = Image.new("RGB", (w * 2, h * 2))
    for i in range(2):
        for j in range(2):
            t.paste(im, (i * w, j * h))
    return t

if __name__ == "__main__":
    cat, out = sys.argv[1], sys.argv[2]
    folders = sorted(glob.glob(f"{ROOT}/{cat}/*"))
    cols = 4
    rows = (len(folders) + cols - 1) // cols
    font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13)
    sheet = Image.new("RGB", (cols * CELL, rows * (CELL + 40)), (28, 28, 30))
    d = ImageDraw.Draw(sheet)
    for i, f in enumerate(folders):
        x, y = (i % cols) * CELL, (i // cols) * (CELL + 40)
        im = render(f, cat.endswith("decals"))
        im.thumbnail((CELL - 8, CELL - 8))
        sheet.paste(im, (x + 4, y + 4))
        meta = json.load(open(f + "/meta.json"))
        d.text((x + 6, y + CELL), os.path.basename(f)[:CELL // 7], fill=(240, 240, 240), font=font)
        d.text((x + 6, y + CELL + 18), "%s | %s" % (meta["licence"], "x".join(map(str, Image.open(pick(f, ["albedo", "_color", "basecolor", "_T.jpg", "Coal_Ashes"])).size))), fill=(170, 200, 170), font=font)
    sheet.save(out, quality=88)
    print(out, sheet.size)
