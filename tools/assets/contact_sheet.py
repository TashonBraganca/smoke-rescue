"""Make labelled contact sheets from Blender renders or explicitly marked source thumbnails."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-thumbnails", action="store_true")
    p.add_argument("--slots", nargs="*")
    a = p.parse_args()
    image_name = "source_thumbnail.jpg" if a.source_thumbnails else "preview.png"
    entries = []
    for meta_path in sorted((ROOT / "assets_cache/candidates").glob("*/*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if a.slots and meta["slot"] not in a.slots:
            continue
        path = meta_path.with_name(image_name)
        if path.exists():
            entries.append((meta, path))
    for page in range(0, len(entries), 16):
        batch = entries[page:page + 16]
        sheet = Image.new("RGB", (1200, 55 + 255 * ((len(batch) + 3) // 4)), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        label = "SOURCE THUMBNAILS. These are not normalized Blender previews." if a.source_thumbnails else "Blender previews. Normalized candidates."
        draw.text((12, 12), label, fill="black")
        for i, (meta, path) in enumerate(batch):
            x, y = (i % 4) * 300, 55 + (i // 4) * 255
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.thumbnail((294, 190))
                sheet.paste(image, (x + (300 - image.width) // 2, y))
            draw.text((x + 5, y + 194), meta["slot"] + " / " + meta["uid"][:8], fill="black")
            draw.text((x + 5, y + 210), meta["name"][:43], fill="black")
            draw.text((x + 5, y + 226), meta["licence"], fill="black")
        out = ROOT / "assets_cache/candidates" / f"{'source_' if a.source_thumbnails else ''}contact_{page // 16 + 1:02d}.jpg"
        sheet.save(out, quality=92)
        print(out)


if __name__ == "__main__":
    main()
