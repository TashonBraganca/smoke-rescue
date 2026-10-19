"""Download the Poly Haven models named in config/props.json and normalise them in Blender.

    python3 tools/fetch_models.py              # every model in props.json
    python3 tools/fetch_models.py sofa_02

Steps per model:
1. Download the 2K glTF and its textures into assets_cache/raw/models/<id>/.
2. Run tools/blender/normalize_model.py headless: apply transforms, put the origin on the
   floor contact point, decimate above 150k triangles, export assets_cache/models/<id>.glb.
3. Read back the report Blender writes and print the real-world size, so wrong scale is caught here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scrapling.fetchers import Fetcher

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "assets_cache" / "raw" / "models"
OUT = ROOT / "assets_cache" / "models"
BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
HEADERS = {"User-Agent": "SmokeRescueSim/0.1 (private research simulation, CC0 asset fetch)"}


def fetch(url: str) -> bytes:
    page = Fetcher.get(url, headers=HEADERS, timeout=180)
    if page.status != 200 or not page.body:
        raise RuntimeError(f"fetch failed {page.status} {url}")
    return page.body


def model_ids() -> list[str]:
    cfg = json.loads((ROOT / "config" / "props.json").read_text())
    ids: list[str] = []
    for parts in cfg["landmarks"].values():
        ids += [p["polyhaven"] for p in parts if "polyhaven" in p]
    for names in cfg["filler"].values():
        ids += names
    return sorted(set(ids))


def download(asset_id: str, res: str = "2k", fmt: str = "gltf") -> Path:
    files = json.loads(fetch(f"https://api.polyhaven.com/files/{asset_id}"))
    entry = files[fmt].get(res) or files[fmt]["1k"]
    gltf = entry[fmt]
    folder = RAW / asset_id / fmt
    main = folder / Path(gltf["url"]).name
    if not main.exists():
        folder.mkdir(parents=True, exist_ok=True)
        for rel, inc in gltf.get("include", {}).items():
            dest = folder / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.write_bytes(fetch(inc["url"]))
        main.write_bytes(fetch(gltf["url"]))
    return main


def normalise(asset_id: str, src: Path) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{asset_id}.glb"
    report = OUT / f"{asset_id}.json"
    cmd = [BLENDER, "--background", "--factory-startup", "--python", str(ROOT / "tools" / "blender" / "normalize_model.py"),
           "--", "--input", str(src), "--output", str(out), "--report", str(report)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not report.exists():
        raise RuntimeError(f"blender failed for {asset_id}: {res.stderr[-800:] or res.stdout[-800:]}")
    return json.loads(report.read_text())


def main(argv: list[str]) -> int:
    ids = argv or model_ids()
    failures = 0
    for asset_id in ids:
        try:
            try:
                rep = normalise(asset_id, download(asset_id))
            except RuntimeError as exc:
                # Blender's glTF importer fails on a few Poly Haven files; their FBX imports fine.
                print(f"     {asset_id}: glTF import failed, retrying with FBX ({str(exc).splitlines()[-1][:80]})")
                rep = normalise(asset_id, download(asset_id, fmt="fbx"))
            w, h, d = rep["size_m"]
            print(f"ok   {asset_id:32s} {w:5.2f} x {h:5.2f} x {d:5.2f} m  tris {rep['tris_before']:>7d} -> {rep['tris_after']:>7d}")
        except Exception as exc:
            failures += 1
            print(f"FAIL {asset_id:32s} {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
