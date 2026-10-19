"""Download, normalize, render and report reviewable asset candidates.

Use the same isolated uv command as discover.py. Subcommands are fetch, process,
catalogue and verify. Source thumbnails are labelled separately from Blender renders.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import html
import json
import shutil
import struct
import subprocess
from pathlib import Path

from discover import CACHE, HF, ROOT, get

OUT = ROOT / "assets_cache/candidates"
BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
CHOICES = Path(__file__).with_name("choices.json")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def glb_document(path):
    with path.open("rb") as f:
        magic, version, size = struct.unpack("<4sII", f.read(12))
        if magic != b"glTF" or version != 2 or size != path.stat().st_size:
            raise ValueError(f"invalid GLB header: {path}")
        length, kind = struct.unpack("<II", f.read(8))
        if kind != 0x4E4F534A:
            raise ValueError("GLB has no JSON chunk")
        return json.loads(f.read(length))


def fetch_one(slot, spec, annotation, object_paths):
    uid = annotation["uid"]
    folder = OUT / slot / uid
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / "annotation.json", annotation)
    live = json.loads(get(annotation["uri"], folder / "source_api.json"))
    licence = live.get("license", {})
    url = licence.get("url", "").replace("http://", "https://").rstrip("/")
    allowed = {"https://creativecommons.org/licenses/by/4.0": "CC-BY-4.0",
               "https://creativecommons.org/licenses/by/3.0": "CC-BY-3.0",
               "https://creativecommons.org/publicdomain/zero/1.0": "CC0-1.0"}
    if url not in allowed:
        raise ValueError(f"unapproved or unknown licence {url!r}")
    source = folder / "source.glb"
    download_url = HF + object_paths[uid]
    get(download_url, source)
    document = glb_document(source)
    thumbnails = live.get("thumbnails", {}).get("images", []) or annotation.get("thumbnails", {}).get("images", [])
    if thumbnails:
        thumbnail = min(thumbnails, key=lambda t: abs(t["width"] - 640))
        try:
            get(thumbnail["url"], folder / "source_thumbnail.jpg")
        except Exception:
            pass
    settings = dict(spec, **spec.get("settings", {}).get(uid, {}))
    meta = {
        "uid": uid, "slot": slot, "name": live.get("name", annotation["name"]),
        "source": "Objaverse 1.0 / Sketchfab", "url": annotation["viewerUrl"],
        "download_url": download_url, "author": live["user"].get("displayName") or live["user"]["username"],
        "author_url": live["user"]["profileUrl"], "licence": allowed[url], "licence_url": url + "/",
        "accessed": datetime.date.today().isoformat(), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "faces_source_annotation": annotation.get("faceCount"), "faces_before": None, "faces_after": None,
        "size_m": None, "has_textures": bool(document.get("textures")),
        "rigged_source": bool(document.get("skins")), "animation_count_source": len(document.get("animations", [])),
        "node_names": [n.get("name", "") for n in document.get("nodes", [])],
        "like_count": annotation.get("likeCount"), "view_count": annotation.get("viewCount"),
        "target_axis": settings["axis"], "target_size_m": settings["target"],
        "processing_options": {k: settings[k] for k in ["keep_mesh", "drop_mesh", "scale_reference"] if k in settings},
        "yaw_degrees": spec.get("yaw", {}).get(uid, 0),
        "status": "downloaded", "review": "pending", "notes": ["Front orientation requires visual review."],
    }
    if not meta["has_textures"]:
        meta["notes"].append("Untextured fallback. Retain only if no suitable textured alternative exists.")
    if (folder / "meta.json").exists():
        previous = read(folder / "meta.json")
        for key in ["faces_before", "faces_after", "size_m", "status", "review", "notes", "normalization"]:
            if key in previous:
                meta[key] = previous[key]
    write(folder / "meta.json", meta)
    return f"downloaded {slot}/{uid} {meta['name']} {meta['licence']}"


def fetch():
    import gzip
    annotations = {a["uid"]: a for values in read(CACHE / "search_results.json").values() for a in values}
    paths = json.loads(gzip.decompress(get(HF + "object-paths.json.gz", CACHE / "object-paths.json.gz")))
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for slot, spec in read(CHOICES).items():
            for uid in spec["uids"]:
                futures[pool.submit(fetch_one, slot, spec, annotations[uid], paths)] = (slot, uid)
        for f in concurrent.futures.as_completed(futures):
            try:
                print(f.result(), flush=True)
            except Exception as exc:
                slot, uid = futures[f]
                failures.append({"slot": slot, "uid": uid, "error": str(exc)})
                print(f"FAILED {slot}/{uid}: {exc}", flush=True)
    write(OUT / "fetch_failures.json", failures)
    catalogue()
    return bool(failures)


def process(slots):
    catalogue()
    for path in sorted(OUT.glob("*/*/meta.json")):
        meta = read(path)
        if meta.get("review") == "rejected":
            continue
        if slots and meta["slot"] not in slots:
            continue
        folder = path.parent
        if meta["status"] == "ready" and (folder / "preview.png").exists():
            continue
        cmd = [BLENDER, "--background", "--factory-startup", "--threads", "4", "--python", str(ROOT / "tools/blender/normalize_model.py"),
               "--", "--input", str(folder / "source.glb"), "--output", str(folder / "model.glb"),
               "--report", str(folder / "normalization.json"), "--target-axis", meta["target_axis"],
               "--target-size", str(meta["target_size_m"]), "--yaw", str(meta["yaw_degrees"]),
               "--preview", str(folder / "preview.png")]
        for key, value in meta.get("processing_options", {}).items():
            cmd += ["--" + key.replace("_", "-"), value]
        with (folder / "blender.log").open("w") as log:
            run = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        if run.returncode or not (folder / "preview.png").exists():
            meta["status"] = "processing_failed"
            write(path, meta)
            catalogue()
            raise RuntimeError(f"Blender failed for {folder}; inspect blender.log. Stop before repeating a startup failure.")
        report = read(folder / "normalization.json")
        meta.update(faces_before=report["tris_before"], faces_after=report["tris_after"], size_m=report["size_m"],
                    has_textures=report["has_textures"], rigged=report["rigged"], status="ready", normalization=report)
        write(path, meta)
        print(f"ready {meta['slot']}/{meta['uid']} {meta['size_m']}", flush=True)
        catalogue()
    catalogue()


def catalogue():
    reviews = read(Path(__file__).with_name("reviews.json")) if Path(__file__).with_name("reviews.json").exists() else {}
    index = {"generated": datetime.datetime.now(datetime.timezone.utc).isoformat(), "slots": {}}
    rows = []
    for slot, spec in read(CHOICES).items():
        candidates = []
        for uid in spec["uids"]:
            path = OUT / slot / uid / "meta.json"
            if path.exists():
                meta = read(path)
                meta.update(reviews.get(uid, {}))
                write(path, meta)
                candidates.append(meta)
        selection = spec.get("recommended")
        pick = next((c for c in candidates if c["uid"] == selection), None)
        index["slots"][slot] = {"candidates": candidates, "recommended_uid": selection,
                                "reason": spec.get("reason", "No recommendation before rendered review."),
                                "recommendation_status": "provisional_source_thumbnail_review" if pick and pick.get("review") != "accepted" else "render_reviewed" if pick else "missing",
                                "ready_count": sum(c["status"] == "ready" for c in candidates),
                                "shortfall": max(0, 3 - sum(c.get("review") != "rejected" for c in candidates))}
        cards = []
        rows.append(f"{slot}. Pick: {pick['name'] if pick else 'pending'}. {index['slots'][slot]['reason']}")
        for c in candidates:
            relative = f"{slot}/{c['uid']}"
            image = "preview.png" if (OUT / relative / "preview.png").exists() else "source_thumbnail.jpg"
            label = "Blender render" if image == "preview.png" else "Source thumbnail only. Normalization/render pending."
            details = f"{c['licence']}; size {c['size_m']}; triangles {c['faces_before']} to {c['faces_after']}; source faces {c['faces_source_annotation']}; {c['status']}; review {c['review']}"
            rows.append(f"  {c['name']} | {details} | {c['url']}")
            notes = " ".join(c.get("notes", []))
            picture = f'<a href="{relative}/{image}"><img loading="lazy" src="{relative}/{image}" alt="{html.escape(c["name"], quote=True)}"></a>' if (OUT / relative / image).exists() else '<p>No source thumbnail available.</p>'
            cards.append(f'<article>{picture}<h3>{html.escape(c["name"])}</h3><p>{html.escape(label)}</p><p>{html.escape(details)}</p><p>{html.escape(notes)}</p><a href="{html.escape(c["url"], quote=True)}">Source and author</a> <a href="{relative}/meta.json">Metadata</a></article>')
        rows.append("")
        index["slots"][slot]["gallery_html"] = "".join(cards)
    sections = []
    for slot, data in index["slots"].items():
        sections.append(f'<section id="{slot}"><h2>{slot.replace("_", " ")}</h2><p>{html.escape(data["reason"])}</p><div class="grid">{data.pop("gallery_html")}</div></section>')
    OUT.mkdir(parents=True, exist_ok=True)
    index["rejected"] = [read(p) for p in sorted((ROOT / "assets_cache/rejected").glob("*/*/meta.json"))]
    write(OUT / "index.json", index)
    (OUT / "report.txt").write_text("\n".join(rows))
    (OUT / "gallery.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Smoke Rescue asset choices</title><style>body{font:16px system-ui;background:#eee;color:#222;margin:32px}h1,h2,h3{font-weight:600}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}article{background:white;padding:16px}img{width:100%;aspect-ratio:4/3;object-fit:contain;background:#ccc}p{line-height:1.5}a{color:#164e80}section{margin:40px 0}</style><h1>Smoke Rescue asset choices</h1><p>Dimensions are width, height, depth in metres. Source thumbnails and Blender renders are labelled. Pending candidates are not production assets.</p>' + "".join(sections) + '</html>')


def credits():
    """Append attribution for every retained GLB, including rejected research downloads."""
    target = ROOT / "CREDITS.md"
    existing = target.read_text()
    rows = []
    for path in sorted(list(OUT.glob("*/*/meta.json")) + list((ROOT / "assets_cache/rejected").glob("*/*/meta.json"))):
        c = read(path)
        if c["uid"] in existing:
            continue
        name = c["name"].replace("|", "&#124;").replace("\n", " ")
        author = c["author"].replace("|", "&#124;").replace("\n", " ")
        use = "rejected research download" if c.get("review") == "rejected" else "candidate review; modified exports pending"
        rows.append(f"| {name} by {author} | {c['url']} | [{c['licence']}]({c['licence_url']}) | {c['slot']} {use} |")
    if rows:
        with target.open("a") as f:
            if not existing.endswith("\n"):
                f.write("\n")
            f.write("\n".join(rows) + "\n")
    print(f"Appended {len(rows)} attribution rows.")


def archive():
    """Move excluded research downloads within the ignored asset cache."""
    active = {(slot, uid) for slot, spec in read(CHOICES).items() for uid in spec["uids"]}
    reviews = read(Path(__file__).with_name("reviews.json"))
    count = 0
    for path in list(OUT.glob("*/*/meta.json")):
        c = read(path)
        if (c["slot"], c["uid"]) in active:
            continue
        c.update(reviews.get(c["uid"], {}))
        c["review"] = "rejected"
        write(path, c)
        destination = ROOT / "assets_cache/rejected" / c["slot"] / c["uid"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise RuntimeError(f"archive destination already exists: {destination}")
        shutil.move(str(path.parent), str(destination))
        count += 1
    print(f"Archived {count} excluded downloads.")
    catalogue()


def verify(require_ready=False):
    index = read(OUT / "index.json")
    failures = []
    total = ready = 0
    for slot, data in index["slots"].items():
        for c in data["candidates"]:
            total += 1
            folder = OUT / slot / c["uid"]
            source = folder / "source.glb"
            glb_document(source)
            assert hashlib.sha256(source.read_bytes()).hexdigest() == c["sha256"]
            assert c["licence"] in {"CC0-1.0", "CC-BY-4.0", "CC-BY-3.0"}
            if c["status"] == "ready":
                ready += 1
                glb_document(folder / "model.glb")
                from PIL import Image
                with Image.open(folder / "preview.png") as image:
                    assert image.size == (640, 480)
                assert c["faces_after"] <= 150000 and min(c["size_m"]) >= 0
                assert abs(c["normalization"]["floor_min_z"]) < 0.001
            elif require_ready and c.get("review") != "rejected":
                failures.append(f"{slot}/{c['uid']} is {c['status']}")
        if len([c for c in data["candidates"] if c.get("review") != "rejected"]) < 3:
            failures.append(f"{slot} has fewer than three non-rejected candidates")
    print(json.dumps({"downloaded": total, "ready": ready, "failures": failures}, indent=2))
    return bool(failures)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["fetch", "process", "catalogue", "verify", "credits", "archive"])
    p.add_argument("--slots", nargs="*")
    p.add_argument("--require-ready", action="store_true")
    a = p.parse_args()
    if a.command == "fetch":
        return fetch()
    if a.command == "process":
        process(a.slots)
    elif a.command == "catalogue":
        catalogue()
    elif a.command == "verify":
        return verify(a.require_ready)
    elif a.command == "credits":
        credits()
    elif a.command == "archive":
        archive()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
