"""Build the props Poly Haven does not have, from simple parts with real PBR textures.

    Blender --background --factory-startup --python tools/blender/build_props.py -- [name ...]

Writes assets_cache/models/blender_<name>.glb and a .json size report per prop.
Conventions match normalize_model.py: metres, origin at the floor contact centre, front faces
Blender -Y, which is glTF +Z and Unity +Z after glTFast import.
Textures come from assets_cache/textures/<material>/ built by tools/fetch_assets.py, and UVs are
cube-projected at each material's real-world tile size from config/materials.json.
"""

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "assets_cache" / "textures"
OUT = ROOT / "assets_cache" / "models"
MATCFG = json.loads((ROOT / "config" / "materials.json").read_text())["materials"]

_materials = {}


def material(name, tint=None, roughness=None, metallic=None):
    """Principled material from a built texture folder. Optional tint multiplies the base colour."""
    key = (name, tint, roughness, metallic)
    if key in _materials:
        return _materials[key]
    mat = bpy.data.materials.new(f"{name}{'_' + tint if tint else ''}")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    folder = TEX / name
    img = nt.nodes.new("ShaderNodeTexImage")
    img.image = bpy.data.images.load(str(folder / "albedo.png"))
    if tint:
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs["Factor"].default_value = 1.0
        h = tint.lstrip("#")
        mix.inputs["B"].default_value = (*[(int(h[i:i + 2], 16) / 255) ** 2.2 for i in (0, 2, 4)], 1)
        nt.links.new(img.outputs["Color"], mix.inputs["A"])
        nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    else:
        nt.links.new(img.outputs["Color"], bsdf.inputs["Base Color"])
    nimg = nt.nodes.new("ShaderNodeTexImage")
    nimg.image = bpy.data.images.load(str(folder / "normal.png"))
    nimg.image.colorspace_settings.name = "Non-Color"
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(nimg.outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    spec = MATCFG.get(name, {})
    bsdf.inputs["Roughness"].default_value = roughness if roughness is not None else 1.0 - spec.get("smoothness", 0.35)
    bsdf.inputs["Metallic"].default_value = metallic if metallic is not None else spec.get("metallic", 0.0)
    mat["tile_m"] = spec.get("tile_m", 1.0)
    _materials[key] = mat
    return mat


def plain(name, rgb, roughness=0.5, metallic=0.0):
    key = ("plain", name)
    if key in _materials:
        return _materials[key]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    mat["tile_m"] = 1.0
    _materials[key] = mat
    return mat


def finish(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.cube_project(cube_size=float(mat.get("tile_m", 1.0)))
    bpy.ops.object.mode_set(mode="OBJECT")
    return obj


def box(size, center, mat, bevel=0.0):
    """size (x width, y depth, z height), center in Blender coords."""
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    o = bpy.context.active_object
    o.scale = size
    finish(o, mat)
    if bevel > 0:
        m = o.modifiers.new("bevel", "BEVEL")
        m.width = bevel
        m.segments = 2
        bpy.ops.object.modifier_apply(modifier=m.name)
    return o


def cyl(radius, depth, center, mat, axis="Z", verts=24):
    rot = {"Z": (0, 0, 0), "X": (0, math.pi / 2, 0), "Y": (math.pi / 2, 0, 0)}[axis]
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, location=center, rotation=rot, vertices=verts)
    return finish(bpy.context.active_object, mat)


def sphere(radius, center, mat, scale=(1, 1, 1)):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=center, segments=32, ring_count=16)
    o = bpy.context.active_object
    o.scale = scale
    return finish(o, mat)


# ---------------------------------------------------------------- props


def fireplace_brick():
    brick = material("prop_brick_red")
    wood = material("prop_wood_cabinet")
    soot = plain("soot_black", (0.02, 0.018, 0.016), 0.95)
    box((0.45, 0.5, 1.15), (-0.575, 0, 0.575), brick)
    box((0.45, 0.5, 1.15), (0.575, 0, 0.575), brick)
    box((0.7, 0.5, 0.45), (0, 0, 0.925), brick)
    box((0.7, 0.12, 0.7), (0, 0.19, 0.35), soot)  # back of the firebox
    box((1.7, 0.6, 0.08), (0, -0.03, 0.14 / 2), brick)  # hearth
    box((1.85, 0.3, 0.07), (0, -0.05, 1.19), wood)  # mantel
    box((1.6, 0.4, 1.55), (0, 0.06, 1.2 + 0.775), brick)  # chimney breast to the ceiling


def rug_patterned():
    box((2.8, 2.0, 0.012), (0, 0, 0.006), material("prop_rug_red"))


def coat_rack():
    wood = material("prop_wood_cabinet")
    cyl(0.025, 1.8, (0, 0, 0.9), wood)
    for a in range(4):
        ang = a * math.pi / 2
        b = box((0.45, 0.04, 0.03), (0.2 * math.cos(ang), 0.2 * math.sin(ang), 0.02), wood)
        b.rotation_euler.z = ang
    fabrics = [material("prop_coat_wool"), material("prop_denim"), material("prop_coat_wool", tint="#7A2A22"), material("prop_coat_wool", tint="#2F4A3A")]
    for i, f in enumerate(fabrics):
        ang = i * math.pi / 2 + math.pi / 4
        x, y = 0.17 * math.cos(ang), 0.17 * math.sin(ang)
        c = cyl(0.13, 0.85, (x, y, 1.2), f, verts=16)
        c.scale = (1.0, 0.55, 1.0)
        c.rotation_euler.z = ang + math.pi / 2


def kitchen_island():
    box((2.0, 0.9, 0.86), (0, 0, 0.43), material("prop_wood_cabinet"), bevel=0.01)
    box((2.12, 1.02, 0.04), (0, 0, 0.88), material("prop_marble_top"), bevel=0.005)
    steel = material("prop_steel_brushed", roughness=0.35)
    for x in (-0.66, 0.0, 0.66):
        box((0.3, 0.02, 0.02), (x, -0.46, 0.75), steel)


def fridge():
    steel = material("prop_steel_brushed", roughness=0.3)
    dark = plain("rubber_dark", (0.03, 0.03, 0.03), 0.8)
    box((0.9, 0.7, 1.85), (0, 0, 0.925), steel, bevel=0.01)
    box((0.006, 0.01, 1.7), (0, -0.351, 0.95), dark)  # door gap
    for x in (-0.06, 0.06):
        cyl(0.012, 0.9, (x, -0.38, 1.05), steel)


def range_hood():
    steel = material("prop_steel_brushed", roughness=0.3)
    box((0.9, 0.5, 0.15), (0, 0, 0.075), steel)
    box((0.32, 0.26, 0.9), (0, 0.1, 0.6), steel)


def clawfoot_bathtub():
    enamel = material("prop_enamel_white")
    brass = plain("brass", (0.78, 0.6, 0.25), 0.3, 1.0)
    t = sphere(0.5, (0, 0, 0.42), enamel, scale=(1.7, 0.75, 0.62))
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    for v in t.data.vertices:  # cut the sphere into a tub: flatten the top half into a rim
        if v.co.z > 0.0:
            v.co.z = 0.0
    box((1.55, 0.6, 0.02), (0, 0, 0.42), plain("water_dark", (0.55, 0.6, 0.62), 0.05))
    for x in (-0.6, 0.6):
        for y in (-0.22, 0.22):
            sphere(0.06, (x, y, 0.06), brass)


def sink_with_mirror():
    enamel = material("prop_enamel_white")
    wood = material("prop_wood_cabinet")
    mirror = plain("mirror", (0.9, 0.9, 0.9), 0.02, 1.0)
    cyl(0.1, 0.75, (0, 0.05, 0.375), enamel)
    box((0.58, 0.46, 0.16), (0, 0, 0.83), enamel, bevel=0.03)
    box((0.66, 0.04, 0.86), (0, 0.2, 1.62), wood)
    box((0.56, 0.02, 0.76), (0, 0.175, 1.62), mirror)


def clothes_rack():
    steel = material("prop_steel_brushed", roughness=0.4)
    for x in (-0.6, 0.6):
        cyl(0.015, 1.7, (x, 0, 0.85), steel)
        box((0.05, 0.5, 0.03), (x, 0, 0.015), steel)
    cyl(0.012, 1.25, (0, 0, 1.68), steel, axis="X")
    tints = [None, "#8A2E2E", "#2E4F8A", None, "#5E6B3A", "#2A2A2A", "#9A7B4F", None]
    for i, tint in enumerate(tints):
        fab = material("prop_denim" if i % 3 == 0 else "prop_coat_wool", tint=tint)
        box((0.06, 0.48, 0.85 + 0.1 * (i % 3)), (-0.52 + i * 0.15, 0, 1.2 - 0.05 * (i % 3)), fab)


def globe_stand():
    wood = material("prop_wood_cabinet")
    for a in range(3):
        ang = a * 2 * math.pi / 3
        leg = cyl(0.018, 0.62, (0.12 * math.cos(ang), 0.12 * math.sin(ang), 0.3), wood)
        leg.rotation_euler = (0.25 * math.sin(ang), -0.25 * math.cos(ang), 0)
    cyl(0.03, 0.12, (0, 0, 0.62), wood)
    sphere(0.25, (0, 0, 0.92), material("prop_globe_map"))
    ring = cyl(0.27, 0.02, (0, 0, 0.92), plain("brass", (0.78, 0.6, 0.25), 0.3, 1.0), axis="Y", verts=48)
    ring.scale = (1, 1, 0.08)


def bunk_bed():
    wood = material("prop_wood_cabinet")
    for x in (-0.98, 0.98):
        for y in (-0.48, 0.48):
            box((0.07, 0.07, 1.75), (x, y, 0.875), wood)
    for z, blanket in ((0.3, material("prop_blanket_blue")), (1.25, material("prop_denim", tint="#C9A13A"))):
        box((2.0, 1.0, 0.06), (0, 0, z), wood)
        box((1.9, 0.92, 0.16), (0, 0, z + 0.11), blanket, bevel=0.03)
        box((0.4, 0.6, 0.1), (-0.72, 0, z + 0.24), material("prop_enamel_white"), bevel=0.04)
    for x in (-0.98, 0.98):
        box((0.04, 1.0, 0.18), (x, 0, 1.55), wood)
    for z in (0.5, 0.8, 1.1):  # ladder rungs on the front
        cyl(0.015, 0.42, (0.75, -0.53, z), wood, axis="X")


def washer(variant="washer"):
    enamel = material("prop_enamel_white")
    steel = material("prop_steel_brushed", roughness=0.3)
    glass = plain("door_glass", (0.05, 0.06, 0.07), 0.05)
    box((0.6, 0.6, 0.85), (0, 0, 0.425), enamel, bevel=0.01)
    cyl(0.2, 0.03, (0, -0.31, 0.42), steel, axis="Y", verts=40)
    cyl(0.16, 0.035, (0, -0.315, 0.42), glass if variant == "washer" else enamel, axis="Y", verts=40)
    box((0.56, 0.02, 0.1), (0, -0.305, 0.78), plain("panel_grey", (0.35, 0.36, 0.38), 0.4))
    for i in range(3):
        cyl(0.025, 0.02, (0.12 + 0.07 * i, -0.32, 0.78), steel, axis="Y")


def water_heater():
    enamel = material("prop_enamel_white", tint="#D8D6CF")
    copper = plain("copper", (0.72, 0.38, 0.22), 0.35, 1.0)
    cyl(0.28, 1.5, (0, 0, 0.8), enamel, verts=40)
    cyl(0.29, 0.05, (0, 0, 1.56), enamel, verts=40)
    for x in (-0.1, 0.1):
        cyl(0.018, 1.25, (x, 0, 2.18), copper)
        cyl(0.018, 0.5, (x, 0.25, 2.78), copper, axis="Y")
    box((0.12, 0.08, 0.16), (0, -0.29, 0.35), plain("panel_grey", (0.35, 0.36, 0.38), 0.4))


PROPS = {
    "fireplace_brick": fireplace_brick,
    "rug_patterned": rug_patterned,
    "coat_rack": coat_rack,
    "kitchen_island": kitchen_island,
    "fridge": fridge,
    "range_hood": range_hood,
    "clawfoot_bathtub": clawfoot_bathtub,
    "sink_with_mirror": sink_with_mirror,
    "clothes_rack": clothes_rack,
    "globe_stand": globe_stand,
    "bunk_bed": bunk_bed,
    "washer": washer,
    "dryer": lambda: washer("dryer"),
    "water_heater": water_heater,
}


def world_bbox(objs):
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for o in objs:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            lo = Vector(map(min, lo, w))
            hi = Vector(map(max, hi, w))
    return lo, hi


def build(name):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _materials.clear()
    PROPS[name]()
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    bpy.ops.object.select_all(action="SELECT")
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    lo, hi = world_bbox(meshes)
    if lo.z < -0.005:
        raise RuntimeError(f"{name} goes below the floor: {lo.z:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"blender_{name}.glb"
    bpy.ops.export_scene.gltf(filepath=str(out), export_format="GLB", export_yup=True)
    size = hi - lo
    report = {"name": name, "output": str(out), "size_m": [round(size.x, 3), round(size.z, 3), round(size.y, 3)],
              "parts": len(meshes), "tris": sum(len(p.vertices) - 2 for o in meshes for p in o.data.polygons)}
    (OUT / f"blender_{name}.json").write_text(json.dumps(report, indent=2))
    print("REPORT", json.dumps(report))


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    for name in argv or list(PROPS):
        build(name)


main()
