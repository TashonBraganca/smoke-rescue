"""Rebuild stats for new models, then write assets_cache/candidates_v2/manifest.json from every meta.json.
python3 -I build_manifest_v2.py"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gltf_stats
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
ROOT = "/Volumes/Tashon/opencv/smoke-rescue/assets_cache/candidates_v2"
EXP = {'sofa':2.0,'armchair':0.9,'tv+tv_stand':1.6,'monitor':0.6,'fridge':1.8,'microwave':0.5,'stove':0.9,'kitchen_units':3.0,'double_bed':2.0,'kids_bunk_bed':2.0,'toy':0.3,'desk':1.4,'office_chair':1.0,'washer':0.85,'curtains':2.2,'rug':2.0,'bookcase':1.8,'floor_lamp':1.6,'shoes':0.3,'backpack':0.45}
EXPA = {'objaverse_c7544d91':2.0}
KEYS = {"albedo": ["albedo", "_color", "basecolor", "_t.jpg"], "normal": ["normal", "_norm", "_n.jpg"], "roughness": ["rough"], "ao": ["_ao", "occ", "ambientocclusion"],
        "height": ["height", "displacement", "_disp"], "opacity": ["opacity"], "emission": ["emission"], "specular": ["_s.jpg"]}
out = []
for d in sorted(glob.glob(ROOT + "/*/*")):
    if os.path.basename(os.path.dirname(d)).startswith("_") or not os.path.exists(d + "/meta.json"):
        continue
    m = json.load(open(d + "/meta.json"))
    m["local_path"] = d
    m["size_mb"] = round(sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs) / 1e6, 1)
    if m["category"].endswith("models"):
        if "stats" not in m or "tris" not in m["stats"]:
            m["stats"] = gltf_stats.stats(os.path.join(d, m["model_file"]))
        s = m["stats"]; m["triangles"] = s["tris"]; m["bbox_m"] = s["bbox_m"]; m["over_300k_tris"] = s["tris"] > 300000
        if m["id"].startswith("objaverse"):
            e = EXP.get(m.get("label")) or EXPA.get(m["id"]) or m.get("expected_max_dim_m")
            if e:
                m["expected_max_dim_m"] = e; m["suggested_scale_to_metres"] = float("%.4g" % (e / max(s["bbox_m"])))
            m["scale_note"] = "file units are arbitrary (Objaverse); bbox_m is in file units, apply suggested_scale_to_metres (a guess from typical real size)"
        else:
            m["scale_note"] = "glTF units are metres (Poly Haven)"
    else:
        maps = {}; dims = {}
        for f in sorted(os.listdir(d)):
            lf = f.lower()
            if not lf.endswith((".jpg", ".png")) or f.startswith("preview") or lf.startswith(("leaking00", "lava00")) and lf.endswith(".png") and "_" not in f:
                continue
            for k, pats in KEYS.items():
                if k not in maps and any(p in lf for p in pats):
                    maps[k] = f; dims[k] = "x".join(map(str, Image.open(os.path.join(d, f)).size)); break
        m["maps"] = sorted(maps); m["map_files"] = maps; m["map_pixels"] = dims
        m["pixel_size"] = dims.get("albedo")
    json.dump(m, open(d + "/meta.json", "w"), indent=1)
    out.append(m)
json.dump(out, open(ROOT + "/manifest.json", "w"), indent=1)
print(len(out), "entries;", round(sum(m["size_mb"] for m in out), 1), "MB")
