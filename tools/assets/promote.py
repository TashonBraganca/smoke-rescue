"""Select a rendered candidate and replace its cached prop without changing Unity code.

Run with --slot SLOT --uid UID after inspecting the candidate's Blender preview.
The command refuses candidates without normalized geometry and a studio render.
"""
import argparse
import json
import shutil
from pathlib import Path

from pipeline import OUT, ROOT, glb_document, read, write

BINDINGS = {
    "fridge": ("tall_steel_fridge", "fridge"),
    "kitchen_island": ("kitchen_island", "kitchen_island"),
    "range_hood": ("stove_and_hood", "range_hood"),
    "clawfoot_bathtub": ("clawfoot_bathtub", "clawfoot_bathtub"),
    "sink_with_mirror": ("sink_with_mirror", "sink_with_mirror"),
    "washing_machine": ("washer_and_dryer", "washer"),
    "clothes_dryer": ("washer_and_dryer", "dryer"),
    "water_heater": ("water_heater_and_pipes", "water_heater"),
    "bunk_bed": ("bunk_bed", "bunk_bed"),
    "coat_rack": ("coat_rack_with_coats", "coat_rack"),
    "clothes_rack": ("clothes_racks_both_sides", "clothes_rack"),
    "antique_globe": ("antique_globe", "globe_stand"),
    "fireplace": ("brick_fireplace", "fireplace_brick"),
    "rug": ("large_rug_patterned", "rug_patterned"),
    "toy_chest": ("toy_chest_and_toys", "toy_chest"),
    "red_sofa": ("red_velvet_sofa", "red_sofa"),
}


def promote(slot, uid):
    if not uid.isalnum() or slot not in BINDINGS:
        raise ValueError("Choose a prop slot and an alphanumeric candidate UID. Robot and survivor choices stay in the catalogue.")
    folder = OUT / slot / uid
    meta = read(folder / "meta.json")
    if meta["status"] != "ready" or meta.get("review") == "rejected":
        raise ValueError("Candidate is not normalized and rendered. Run pipeline.py process first.")
    if not (folder / "preview.png").is_file() or not (folder / "normalization.json").is_file():
        raise ValueError("Candidate is missing its preview or normalization report.")
    report = read(folder / "normalization.json")
    if report["tris_after"] > 150000 or abs(report["floor_min_z"]) > 0.001:
        raise ValueError("Candidate exceeds the triangle budget or has an invalid floor origin.")
    glb_document(folder / "model.glb")
    config_path = ROOT / "config/props.json"
    config = read(config_path)
    landmark, prop = BINDINGS[slot]
    downloaded = str((folder / "model.glb").relative_to(ROOT / "assets_cache"))
    if slot in {"toy_chest", "red_sofa"}:
        toys = [p for p in config["landmarks"][landmark] if p.get("polyhaven") in {"rubber_duck_toy", "football"}]
        config["landmarks"][landmark] = [{"blender": prop, "downloaded": downloaded}]
        if slot == "toy_chest" and uid != "8ce01f03c4c54c1bb69f6ba47a590586":
            config["landmarks"][landmark].extend(toys)
    else:
        found = False
        for part in config["landmarks"][landmark]:
            matches = part.get("blender") == prop
            if slot == "washing_machine":
                matches = matches and part.get("variant") != "dryer"
            elif slot == "clothes_dryer":
                matches = matches or (part.get("blender") == "washer" and part.get("variant") == "dryer")
            if matches:
                part["blender"] = prop
                part.pop("variant", None)
                part["downloaded"] = downloaded
                found = True
        if not found:
            raise ValueError(f"Expected {prop} in landmark {landmark}; inspect config before changing it.")
    destination = ROOT / "assets_cache/models" / f"blender_{prop}.glb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(folder / "model.glb", destination)
    write(destination.with_suffix(".json"), dict(report, name=prop, output=str(destination), candidate=uid))
    write(config_path, config)
    print(f"Selected {meta['name']} for {landmark}. Rebuild through build_props.py to reproduce it.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--slot", choices=BINDINGS, required=True)
    p.add_argument("--uid", required=True)
    a = p.parse_args()
    promote(a.slot, a.uid)


if __name__ == "__main__":
    main()
