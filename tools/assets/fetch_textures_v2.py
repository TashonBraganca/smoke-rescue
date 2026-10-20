"""Download texture candidates (burn damage) into assets_cache/candidates_v2/<category>/<source>_<id>/.

Sources: Poly Haven (CC0), ambientCG (CC0), 3dtextures.me (CC0, public Google Drive files),
OpenGameArt (per-asset licence). Writes meta.json beside each asset. Downloads are only saved,
never executed. Zips are extracted with python zipfile after a name check.
"""
import json, os, re, sys, zipfile, urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = "/Volumes/Tashon/opencv/smoke-rescue/assets_cache/candidates_v2"
UA = {"User-Agent": "Mozilla/5.0 SmokeRescueSim/0.1 (private research simulation)"}

def http(url, timeout=180):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)

def save(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with http(url) as r, open(path + ".part", "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    os.replace(path + ".part", path)
    return path

def write_meta(folder, meta):
    json.dump(meta, open(os.path.join(folder, "meta.json"), "w"), indent=1)

def polyhaven(aid, category, res="2k"):
    files = json.load(http(f"https://api.polyhaven.com/files/{aid}"))
    info = json.load(http(f"https://api.polyhaven.com/info/{aid}"))
    folder = f"{ROOT}/{category}/polyhaven_{aid}"
    maps = {}
    for key, label in [("Diffuse", "albedo"), ("nor_gl", "normal"), ("Rough", "roughness"), ("AO", "ao"), ("Displacement", "height")]:
        ent = files.get(key, {})
        r = ent.get(res) or ent.get("1k")
        if not r:
            continue
        url = r["jpg"]["url"]
        save(url, f"{folder}/{aid}_{label}.jpg")
        maps[label] = f"{aid}_{label}.jpg"
    save(f"https://cdn.polyhaven.com/asset_img/thumbs/{aid}.png?width=512&height=512", f"{folder}/preview.png")
    authors = ", ".join(info.get("authors", {}).keys())
    write_meta(folder, dict(id=f"polyhaven_{aid}", category=category, source_url=f"https://polyhaven.com/a/{aid}", licence="CC0",
                            attribution=f"{info['name']} by {authors} (Poly Haven, CC0; credit optional)", name=info["name"], maps_files=maps, preview="preview.png"))
    return folder

def ambientcg(aid, category, res="2K"):
    folder = f"{ROOT}/{category}/ambientcg_{aid}"
    z = f"{folder}/{aid}_{res}-JPG.zip"
    save(f"https://ambientcg.com/get?file={aid}_{res}-JPG.zip", z)
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            assert not n.startswith("/") and ".." not in n, n
        zf.extractall(folder)
    save(f"https://acg-media.struffelproductions.com/file/ambientCG-Web/media/thumbnail/512-JPG-242424/{aid}.jpg", f"{folder}/preview.jpg")
    write_meta(folder, dict(id=f"ambientcg_{aid}", category=category, source_url=f"https://ambientcg.com/view?id={aid}", licence="CC0",
                            attribution="ambientCG.com (CC0; credit optional)", name=aid, preview="preview.jpg"))
    return folder

def drive(fid, path):
    return save(f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t", path)

def threedt(slug, page, files, preview, category, licence_note=""):
    folder = f"{ROOT}/{category}/3dtextures_{slug}"
    maps = {}
    for label, (fid, fname) in files.items():
        drive(fid, f"{folder}/{fname}")
        maps[label] = fname
    save(preview, f"{folder}/preview.png")
    write_meta(folder, dict(id=f"3dtextures_{slug}", category=category, source_url=page, licence="CC0",
                            attribution="3dtextures.me by Katsukagi (CC0; credit optional)", name=slug, maps_files=maps, preview="preview.png"))
    return folder

def oga(slug, page, urls, licence, attribution, category):
    folder = f"{ROOT}/{category}/oga_{slug}"
    maps = {}
    for label, u in urls.items():
        fn = os.path.basename(u.split("?")[0]).replace("%20", "_")
        save(u, f"{folder}/{fn}")
        maps[label] = fn
    write_meta(folder, dict(id=f"oga_{slug}", category=category, source_url=page, licence=licence, attribution=attribution, name=slug, maps_files=maps))
    return folder

JOBS = []
def job(f, *a, **k):
    JOBS.append((f, a, k))

C = "burn_textures"
for a in ["wood_peeling_paint_weathered", "burned_ground_01", "dark_wooden_planks", "rubble", "decrepit_wallpaper",
          "damaged_plaster", "rough_plaster_broken", "concrete_wall_006", "dirty_carpet", "peeling_painted_wall"]:
    job(polyhaven, a, C)
job(ambientcg, "Lava002", C)
job(ambientcg, "Lava003", C)
D = "burn_decals"
for a in ["Leaking001", "Leaking004", "Leaking006"]:
    job(ambientcg, a, D)
job(threedt, "burnt-wood-001", "https://3dtextures.me/2018/01/20/burnt-wood-001/",
    {"albedo": ("1gHlh9ruRtW-MK0lninHcB4QG7O8jyZMd", "Burnt_wood_001_COLOR.jpg"), "height": ("1RGwiCmNvnb2mzJy0uzBcwMDZvBzad_ao", "Burnt_wood_001_DISP.png"),
     "normal": ("1NT-cETlQVDro7LtitC3OQROKr_ZDzX6b", "Burnt_wood_001_NORM.jpg"), "ao": ("1ytgGaCdYso1_xYldGTzbDqs21m6LUgf-", "Burnt_wood_001_OCC.jpg"),
     "roughness": ("1knJv2pCLUsqTkFB9VlDpfrmoSO46sVqN", "Burnt_wood_001_ROUGH.jpg")},
    "https://3dtextures.me/wp-content/uploads/2018/01/material_333.png", C)
job(threedt, "wood-023", "https://3dtextures.me/2021/11/30/wood-023/",
    {"albedo": ("1N9yV9kpaptY9BPry2IPFzr0lJDJ_M86-", "Wood_023_basecolor.jpg"), "height": ("1VGqMMq2vPP_FC76D_98jcadY_k2ZZyxL", "Wood_023_height.png"),
     "normal": ("1aN39bk8HL0F8jmJVI0Kbgimrd3qYvdkl", "Wood_023_normal.jpg"), "ao": ("1D3T1opxeUjI8U16ydxZvRH9rpl1UtU50", "Wood_023_ambientOcclusion.jpg"),
     "roughness": ("1NmBABDLWcUV6Echr5O1K8jZe0J1KAUg1", "Wood_023_roughness.jpg")},
    "https://3dtextures.me/wp-content/uploads/2021/11/Material_1980.jpg", C)
OGA = "https://opengameart.org/sites/default/files/"
job(oga, "burned-wood-1", "https://opengameart.org/content/burned-wood-texture-seamless-texture-with-normalmap",
    {"albedo": OGA + "BurnedWood1_T.jpg", "normal": OGA + "BurnedWood1_N.jpg", "specular": OGA + "BurnedWood1_S.jpg"}, "CC-BY 3.0", "OpenGameArt 'Burned Wood Texture - Seamless Texture with normalmap' (see page for author), CC-BY 3.0", C)
job(oga, "burned-wood-2", "https://opengameart.org/content/burned-wood-texture-seamless-texture-with-normalmap-0",
    {"albedo": OGA + "BurnedWood2_T.jpg", "normal": OGA + "BurnedWood2_N.jpg", "specular": OGA + "BurnedWood2_S.jpg"}, "CC-BY 3.0", "OpenGameArt 'Burned Wood Texture - Seamless Texture with normalmap' (see page for author), CC-BY 3.0", C)
job(oga, "burnt-coal-ashes-1024", "https://opengameart.org/content/burnt-coal-ashes-tiling-1024",
    {"albedo": OGA + "Burnt%20Coal%20Ashes%20Tiling.jpg"}, "CC0", "OpenGameArt Burnt Coal Ashes Tiling 1024 (CC0)", C)

def run(j):
    f, a, k = j
    try:
        return f.__name__, a[0], f(*a, **k)
    except Exception as e:
        return f.__name__, a[0], "FAIL %r" % e

if __name__ == "__main__":
    only = sys.argv[1:]
    jobs = [j for j in JOBS if not only or j[1][0] in only]
    with ThreadPoolExecutor(4) as ex:
        for r in ex.map(run, jobs):
            print(r)
