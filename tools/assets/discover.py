"""Search all Objaverse 1.0 annotation shards, retaining source evidence in the cache.

Run with UV_CACHE_DIR=/private/tmp/smoke-assets-uv uv run --no-project --with requests
python tools/assets/discover.py. Downloads are resumable and never use the brain environment.
"""
import concurrent.futures
import gzip
import json
import math
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "assets_cache/research"
HF = "https://huggingface.co/datasets/allenai/objaverse/resolve/main/"
QUERIES = {
    "fridge": r"fridge|refrigerator",
    "kitchen_island": r"kitchen.?island|kitchen.?counter|kitchen.?cabinet|kitchen.?unit",
    "range_hood": r"range.?hood|extractor|cooker.?hood|kitchen.?hood|exhaust.?hood",
    "clawfoot_bathtub": r"claw.?foot|bathtub|bath.?tub",
    "sink_with_mirror": r"bathroom.?sink|bathroom.?vanity|washbasin|wash.?basin|vanity|sink",
    "washing_machine": r"washing.?machine|\bwasher\b",
    "clothes_dryer": r"tumble.?dry|clothes.?dry|\bdryer\b",
    "water_heater": r"water.?heater|\bboiler\b|hot.?water|\bheater\b|\bgeyser\b",
    "bunk_bed": r"bunk.?bed|bunkbed",
    "coat_rack": r"coat.?rack|coat.?stand|hat.?stand|hat.?rack",
    "clothes_rack": r"clothes.?rack|clothing.?rack|garment.?rack|garment.?rail|clothes.?rail|clothing.?rail|clothes.?stand|coat.?rail|hanger.?rack|clothing.?stand|clothes.?hanger|clothing.?display",
    "antique_globe": r"\bglobe\b",
    "fireplace": r"fireplace|fire.?place",
    "rug": r"\brug\b|\bcarpet\b|persian.?rug",
    "toy_chest": r"toy.?chest|toy.?box|wooden.?chest|storage.?chest",
    "red_sofa": r"\bsofa\b|\bcouch\b",
    "robot": r"\bunitree\b|\bgo1\b|\bgo2\b|boston.?dynamics|\bspot\b|quadruped|robot.?dog",
    "survivor": r"\bscan.?person\b|\brigged.?man\b|\brigged.?woman\b|\brigged.?human\b|\bman\b|\bwoman\b|\bhuman\b|\bperson\b",
}
PATTERNS = {k: re.compile(v, re.I) for k, v in QUERIES.items()}


def get(url, path=None):
    if path and path.exists():
        return path.read_bytes()
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=(20, 180), headers={"User-Agent": "SmokeRescueAssetResearch/1.0"})
            r.raise_for_status()
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(path.suffix + ".part")
                tmp.write_bytes(r.content)
                tmp.replace(path)
            return r.content
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def score(a, slot):
    name = a["name"].lower()
    text = name + " " + " ".join(t["name"] for t in a.get("tags", []))
    result = math.log1p(a.get("likeCount", 0)) * 6 + math.log1p(a.get("viewCount", 0))
    result += 14 if PATTERNS[slot].search(name) else 0
    result += 5 if 5000 <= a.get("faceCount", 0) <= 300000 else -6
    result += 8 if a.get("archives", {}).get("glb", {}).get("textureCount", 0) else -15
    result -= 35 if re.search(r"low.?poly|styliz|stylis|cartoon|voxel|minecraft|diorama|pixel", text) else 0
    if slot == "survivor":
        result += 15 if a.get("animationCount", 0) else 0
        result -= 40 if re.search(r"zombie|nude|naked|anatomy|superhero|soldier|knight|monster|skeleton|statue|bust|head|anime", name) else 0
    if slot == "red_sofa":
        result += 18 if "red" in text or "chesterfield" in text else 0
    if slot == "robot":
        result += 30 if re.search(r"unitree|boston|robot.?dog|quadruped|spot.?robot", text) else -25
    return round(result, 3)


def scan(entry):
    filename = entry["path"].split("/")[-1]
    path = CACHE / filename
    data = json.loads(gzip.decompress(get(HF + entry["path"], path)))
    hits = {k: [] for k in QUERIES}
    for uid, a in data.items():
        if a.get("license") not in {"by", "cc0"} or a.get("isAgeRestricted"):
            continue
        text = a["name"] + " " + " ".join(t["name"] for t in a.get("tags", []))
        for slot, pattern in PATTERNS.items():
            if pattern.search(text):
                hits[slot].append(dict(a, metadata_shard=entry["path"], search_score=score(a, slot)))
    return filename, hits


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    listing = json.loads(get("https://huggingface.co/api/datasets/allenai/objaverse/tree/main/metadata", CACHE / "metadata_listing.json"))
    combined = {k: [] for k in QUERIES}
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(scan, e): e for e in listing if e["path"].endswith(".json.gz")}
        for n, f in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                name, hits = f.result()
                for k, values in hits.items():
                    combined[k].extend(values)
                print(f"{n}/{len(futures)} {name}", flush=True)
            except Exception as exc:
                failures.append({"shard": futures[f]["path"], "error": str(exc)})
            if n % 10 == 0 or n == len(futures):
                for v in combined.values():
                    v.sort(key=lambda a: a["search_score"], reverse=True)
                (CACHE / "search_results.json").write_text(json.dumps(combined, indent=2))
    (CACHE / "search_failures.json").write_text(json.dumps(failures, indent=2))
    print({k: len(v) for k, v in combined.items()})


if __name__ == "__main__":
    main()
