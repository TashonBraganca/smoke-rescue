"""Build a labelled contact sheet from (label, image) pairs.

    python3 -I sheet_v2.py out.jpg cols cell_px label=path [label=path ...]
"""
import sys
from PIL import Image, ImageDraw, ImageFont

def build(out, cols, cell, items):
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
    except Exception:
        font = ImageFont.load_default()
    rows = (len(items) + cols - 1) // cols
    lab_h = 36
    sheet = Image.new("RGB", (cols * cell, rows * (cell + lab_h)), (28, 28, 30))
    d = ImageDraw.Draw(sheet)
    for i, (label, path) in enumerate(items):
        x, y = (i % cols) * cell, (i // cols) * (cell + lab_h)
        try:
            im = Image.open(path).convert("RGBA")
            bg = Image.new("RGBA", im.size, (60, 60, 64, 255))
            bg.alpha_composite(im)
            im = bg.convert("RGB")
            im.thumbnail((cell - 8, cell - 8))
            sheet.paste(im, (x + (cell - im.width) // 2, y + 4 + (cell - 8 - im.height) // 2))
        except Exception as exc:
            d.text((x + 8, y + 8), f"ERR {exc}"[:40], fill=(255, 80, 80), font=font)
        d.text((x + 6, y + cell + 2), label[:cell // 8], fill=(235, 235, 235), font=font)
    sheet.save(out, quality=88)

if __name__ == "__main__":
    out, cols, cell = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    build(out, cols, cell, [tuple(a.split("=", 1)) for a in sys.argv[4:]])
