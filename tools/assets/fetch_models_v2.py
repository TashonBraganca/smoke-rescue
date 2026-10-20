"""Download model candidates into assets_cache/candidates_v2/<category>/<source>_<id>/.

Poly Haven: 2k glTF + textures (CC0).  Objaverse: GLB from the allenai/objaverse HF dataset; licence/author from the
Sketchfab metadata shipped with the dataset (only CC0 / CC-BY accepted). Files are saved, never executed.
Usage: python3 -I fetch_models_v2.py jobs.json   (jobs: [{"src":"polyhaven","id":..,"cat":..} | {"src":"objaverse","id":uid,"cat":..,"label":..,"shard":..,"name":..,"user":..,"url":..,"lic":..,"thumb":..}])
"""
import json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
ROOT = "/Volumes/Tashon/opencv/smoke-rescue/assets_cache/candidates_v2"
UA = {"User-Agent": "Mozilla/5.0 SmokeRescueSim/0.1 (private research simulation)"}

def http(url, timeout=300):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)

def save(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with http(url) as r, open(path + ".part", "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    os.replace(path + ".part", path)

def polyhaven(j):
    aid = j["id"]
    files = json.load(http(f"https://api.polyhaven.com/files/{aid}"))
    info = json.load(http(f"https://api.polyhaven.com/info/{aid}"))
    folder = f"{ROOT}/{j['cat']}/polyhaven_{aid}"
    ent = files["gltf"].get("2k") or files["gltf"]["1k"]
    g = ent["gltf"]
    for rel, inc in g.get("include", {}).items():
        save(inc["url"], f"{folder}/{rel}")
    main = f"{folder}/{os.path.basename(g['url'])}"
    save(g["url"], main)
    save(f"https://cdn.polyhaven.com/asset_img/thumbs/{aid}.png?width=512&height=512", f"{folder}/preview.png")
    authors = ", ".join(info.get("authors", {}).keys())
    meta = dict(id=f"polyhaven_{aid}", category=j["cat"], source_url=f"https://polyhaven.com/a/{aid}", licence="CC0",
                attribution=f"{info['name']} by {authors} (Poly Haven, CC0; credit optional)", name=info["name"], model_file=os.path.basename(g["url"]), preview="preview.png")
    json.dump(meta, open(f"{folder}/meta.json", "w"), indent=1)
    return folder

def objaverse(j):
    uid = j["id"]
    folder = f"{ROOT}/{j['cat']}/objaverse_{uid[:8]}"
    save(f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/{j['shard']}/{uid}.glb", f"{folder}/{uid}.glb")
    if j.get("thumb"):
        try:
            save(j["thumb"], f"{folder}/preview.jpg")
        except Exception:
            pass
    lic = {"by": "CC-BY 4.0", "cc0": "CC0"}[j["lic"]]
    meta = dict(id=f"objaverse_{uid[:8]}", category=j["cat"], source_url=j["url"], licence=lic,
                attribution=f"\"{j['name']}\" by {j['user']} ({j['url']}), {lic}", name=j["name"], objaverse_uid=uid, model_file=f"{uid}.glb",
                preview="preview.jpg" if os.path.exists(f"{folder}/preview.jpg") else None, label=j.get("label"))
    json.dump(meta, open(f"{folder}/meta.json", "w"), indent=1)
    return folder

def run(j):
    try:
        return j["id"], {"polyhaven": polyhaven, "objaverse": objaverse}[j["src"]](j)
    except Exception as e:
        return j["id"], "FAIL %r" % e

if __name__ == "__main__":
    jobs = json.load(open(sys.argv[1]))
    with ThreadPoolExecutor(4) as ex:
        for r in ex.map(run, jobs):
            print(r)
