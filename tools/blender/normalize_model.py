"""Normalise one downloaded model for Unity. Run headless:

    Blender --background --factory-startup --python tools/blender/normalize_model.py -- \
        --input model.gltf --output model.glb --report model.json

Applies all transforms, joins nothing, moves the origin to the centre of the floor contact
(bounding-box bottom centre), decimates meshes above 150k triangles in total, and exports a GLB
with embedded textures. glTF is +Y up, metres, which glTFast imports into Unity unchanged.
"""

import argparse
import json
import sys

import bpy
from mathutils import Matrix, Vector

MAX_TRIS = 150_000


def args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--report", required=True)
    return p.parse_args(argv)


def tri_count(objs):
    total = 0
    for o in objs:
        for poly in o.data.polygons:
            total += len(poly.vertices) - 2
    return total


def world_bbox(objs):
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for o in objs:
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            lo = Vector(map(min, lo, w))
            hi = Vector(map(max, hi, w))
    return lo, hi


def main():
    a = args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if a.input.lower().endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=a.input)
    else:
        bpy.ops.import_scene.gltf(filepath=a.input)

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("no meshes imported")

    # Apply transforms on everything so the GLB has identity nodes with real-size geometry.
    bpy.ops.object.select_all(action="DESELECT")
    for o in bpy.context.scene.objects:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    # Instanced meshes share data; transform_apply refuses multi-user data.
    bpy.ops.object.make_single_user(object=True, obdata=True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # Unparent while keeping transforms, so moving the origin below moves every part.
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")

    tris_before = tri_count(meshes)
    if tris_before > MAX_TRIS:
        ratio = MAX_TRIS / tris_before
        for o in meshes:
            mod = o.modifiers.new("decimate", "DECIMATE")
            mod.ratio = ratio
            bpy.context.view_layer.objects.active = o
            bpy.ops.object.modifier_apply(modifier=mod.name)
    tris_after = tri_count(meshes)

    # Blender is Z-up. Floor contact = min z; centre in x and y.
    lo, hi = world_bbox(meshes)
    shift = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
    for o in meshes:
        o.data.transform(Matrix.Translation(-shift))
        o.location = (0, 0, 0)
    # Remove empties left over from the import hierarchy.
    for o in [o for o in bpy.context.scene.objects if o.type != "MESH"]:
        bpy.data.objects.remove(o, do_unlink=True)

    lo, hi = world_bbox(meshes)
    size = hi - lo
    bpy.ops.export_scene.gltf(filepath=a.output, export_format="GLB", export_apply=True, export_yup=True)

    report = {
        "input": a.input,
        "output": a.output,
        "size_m": [round(size.x, 3), round(size.z, 3), round(size.y, 3)],  # width (x), height (up), depth
        "tris_before": tris_before,
        "tris_after": tris_after,
        "meshes": len(meshes),
        "materials": sorted({s.material.name for o in meshes for s in o.material_slots if s.material}),
    }
    with open(a.report, "w") as f:
        json.dump(report, f, indent=2)
    print("REPORT", json.dumps(report))


main()
