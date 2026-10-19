"""Download CC0 PBR textures named in config/materials.json and pack them for HDRP.

Run with the system Python, which has Scrapling, Pillow and NumPy:

    python3 tools/fetch_assets.py            # every material
    python3 tools/fetch_assets.py carpet_grey wood_herringbone

Output per material, in assets_cache/textures/<name>/:
    albedo.png   sRGB base colour
    normal.png   tangent-space normal, OpenGL convention (Unity's convention)
    mask.png     HDRP mask map: R metallic, G ambient occlusion, B detail mask, A smoothness
    source.json  where each map came from, for CREDITS.md

Raw downloads are cached in assets_cache/raw/ so reruns are free.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scrapling.fetchers import Fetcher

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "materials.json"
RAW = ROOT / "assets_cache" / "raw"
OUT = ROOT / "assets_cache" / "textures"
HEADERS = {"User-Agent": "SmokeRescueSim/0.1 (private research simulation, CC0 asset fetch)"}
RES = "2k"
SIZE = 2048


def fetch_bytes(url: str) -> bytes:
    page = Fetcher.get(url, headers=HEADERS, timeout=120)
    if page.status != 200 or not page.body:
        raise RuntimeError(f"fetch failed {page.status} {url}")
    return page.body


def cached(path: Path, url: str) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch_bytes(url))
    return path


def load(path: Path | None, mode: str) -> np.ndarray | None:
    if path is None:
        return None
    img = Image.open(path).convert(mode)
    if img.size != (SIZE, SIZE):
        img = img.resize((SIZE, SIZE), Image.LANCZOS)
    return np.asarray(img).astype(np.float32) / 255.0


def polyhaven_maps(asset_id: str) -> dict[str, Path | None]:
    files = json.loads(fetch_bytes(f"https://api.polyhaven.com/files/{asset_id}"))
    maps: dict[str, Path | None] = {}
    # Some fabrics ship colour variants (col_01, coll1) instead of a single Diffuse map.
    diffuse_key = next((k for k in ("Diffuse", "col_01", "coll1") if k in files), "Diffuse")
    for key, ph_key in [("albedo", diffuse_key), ("normal", "nor_gl"), ("rough", "Rough"), ("ao", "AO")]:
        entry = files.get(ph_key, {}).get(RES, {}).get("jpg")
        if entry is None:
            maps[key] = None
            continue
        maps[key] = cached(RAW / "polyhaven" / asset_id / f"{ph_key}_{RES}.jpg", entry["url"])
    return maps


def ambientcg_maps(asset_id: str) -> dict[str, Path | None]:
    folder = RAW / "ambientcg" / asset_id
    zpath = cached(folder / f"{asset_id}_2K-JPG.zip", f"https://ambientcg.com/get?file={asset_id}_2K-JPG.zip")
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        maps: dict[str, Path | None] = {}
        for key, suffix in [("albedo", "_Color.jpg"), ("normal", "_NormalGL.jpg"), ("rough", "_Roughness.jpg"), ("ao", "_AmbientOcclusion.jpg")]:
            hit = [n for n in names if n.endswith(suffix)]
            if not hit:
                maps[key] = None
                continue
            target = folder / hit[0]
            if not target.exists():
                target.write_bytes(zf.read(hit[0]))
            maps[key] = target
    return maps


def source_maps(src: dict) -> dict[str, Path | None]:
    if src["source"] == "polyhaven":
        return polyhaven_maps(src["id"])
    if src["source"] == "ambientcg":
        return ambientcg_maps(src["id"])
    raise ValueError(src["source"])


def hex_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def tint(albedo: np.ndarray, color: str) -> np.ndarray:
    """Keep the texture's light variation, move its mean colour to `color`."""
    lin = srgb_to_linear(albedo)
    lum = luminance(lin)
    rel = lum / max(float(lum.mean()), 1e-4)
    return linear_to_srgb(rel[..., None] * srgb_to_linear(hex_rgb(color)))


def paper_variation(paper_albedo: np.ndarray) -> np.ndarray:
    lum = luminance(srgb_to_linear(paper_albedo))
    return (lum / max(float(lum.mean()), 1e-4))[..., None]


def recolor(pattern: np.ndarray, paper: np.ndarray, dark: str, light: str, gamma: float = 1.0) -> np.ndarray:
    lum = luminance(pattern)
    lo, hi = np.percentile(lum, [2, 98])
    t = (np.clip((lum - lo) / max(hi - lo, 1e-4), 0, 1) ** gamma)[..., None]
    lin = (1 - t) * srgb_to_linear(hex_rgb(dark)) + t * srgb_to_linear(hex_rgb(light))
    return linear_to_srgb(lin * paper_variation(paper))


def stripes(paper: np.ndarray, colors: list[str], motif: str, stripe_px: int) -> np.ndarray:
    """Two-tone vertical stripes with a small printed diamond motif every stripe, so the wall has corners to track."""
    x = np.arange(SIZE)
    idx = (x // stripe_px) % len(colors)
    base = np.stack([srgb_to_linear(hex_rgb(c)) for c in colors])[idx]  # (SIZE, 3)
    img = np.broadcast_to(base[None, :, :], (SIZE, SIZE, 3)).copy()
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    period = stripe_px
    cx = (xx // period) * period + period // 2
    cy = (yy // period) * period + period // 2 + ((xx // period) % 2) * (period // 2)
    diamond = (np.abs(xx - cx) + np.abs((yy - cy + period // 2) % period - period // 2)) < period * 0.18
    img[diamond] = srgb_to_linear(hex_rgb(motif))
    edge = (xx % stripe_px) < 3
    img[edge] *= 0.75
    return linear_to_srgb(img * paper_variation(paper))


def build_mask(rough: np.ndarray | None, ao: np.ndarray | None, metallic: float = 0.0, smoothness: float | None = None) -> np.ndarray:
    smooth = 1.0 - (rough if rough is not None else np.full((SIZE, SIZE), 0.6, np.float32))
    if smoothness is not None:
        # Keep the texture's variation, move its mean to the requested smoothness.
        smooth = np.clip(smooth - smooth.mean() + smoothness, 0, 1)
    occl = ao if ao is not None else np.ones((SIZE, SIZE), np.float32)
    zeros = np.zeros((SIZE, SIZE), np.float32)
    return np.stack([zeros + metallic, occl, zeros, smooth], axis=-1)


def save(arr: np.ndarray, path: Path, mode: str) -> None:
    Image.fromarray((np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8), mode).save(path, optimize=False)


def flat_normal() -> np.ndarray:
    n = np.zeros((SIZE, SIZE, 3), np.float32)
    n[..., 0] = n[..., 1] = 0.5
    n[..., 2] = 1.0
    return n


def build(name: str, spec: dict) -> dict:
    kind = spec["source"]
    record: dict = {"name": name, "tile_m": spec["tile_m"], "kind": kind}
    if kind in ("polyhaven", "ambientcg"):
        maps = source_maps(spec)
        albedo = load(maps["albedo"], "RGB")
        normal = load(maps["normal"], "RGB")
        rough = load(maps["rough"], "L")
        ao = load(maps["ao"], "L")
        if "tint" in spec:
            albedo = tint(albedo, spec["tint"])
        record["sources"] = [{"source": kind, "id": spec["id"]}]
    elif kind in ("recolor", "stripes"):
        paper_maps = source_maps(spec["paper"])
        paper = load(paper_maps["albedo"], "RGB")
        normal = load(paper_maps["normal"], "RGB")
        rough = load(paper_maps["rough"], "L")
        ao = load(paper_maps["ao"], "L")
        record["sources"] = [{"source": spec["paper"]["source"], "id": spec["paper"]["id"], "role": "paper"}]
        if kind == "recolor":
            pat_maps = source_maps(spec["pattern"])
            pattern = load(pat_maps["albedo"], "RGB")
            pattern = np.rot90(pattern, k=spec.get("rotate", 0) // 90).copy()
            albedo = recolor(pattern, paper, spec["dark"], spec["light"], spec.get("gamma", 1.0))
            record["sources"].append({"source": spec["pattern"]["source"], "id": spec["pattern"]["id"], "role": "pattern, recoloured"})
        else:
            albedo = stripes(paper, spec["colors"], spec["motif"], spec["stripe_px"])
            record["sources"].append({"source": "generated", "id": "stripes", "role": "pattern"})
    else:
        raise ValueError(kind)

    if albedo is None:
        raise RuntimeError(f"{name}: no albedo map")
    if normal is None:
        normal = flat_normal()
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    save(albedo, out / "albedo.png", "RGB")
    save(normal, out / "normal.png", "RGB")
    save(build_mask(rough, ao, spec.get("metallic", 0.0), spec.get("smoothness")), out / "mask.png", "RGBA")
    record["has"] = {"rough": rough is not None, "ao": ao is not None}

    # Sanity gate: a texture that is nearly flat is useless for visual SLAM.
    lum = luminance(albedo)
    record["albedo_lum_mean"] = round(float(lum.mean()), 3)
    record["albedo_lum_std"] = round(float(lum.std()), 4)
    if record["albedo_lum_std"] < 0.01:
        raise RuntimeError(f"{name}: albedo is almost flat (std {record['albedo_lum_std']})")
    (out / "source.json").write_text(json.dumps(record, indent=2))
    return record


def main(argv: list[str]) -> int:
    cfg = json.loads(CONFIG.read_text())["materials"]
    names = argv or list(cfg)
    failures = 0
    for name in names:
        try:
            rec = build(name, cfg[name])
            print(f"ok   {name:30s} lum {rec['albedo_lum_mean']:.3f} ± {rec['albedo_lum_std']:.3f}  rough={rec['has']['rough']} ao={rec['has']['ao']}")
        except Exception as exc:  # report every failure, keep going
            failures += 1
            print(f"FAIL {name:30s} {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
