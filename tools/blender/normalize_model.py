"""Normalise one downloaded model for Unity. Run headless:

    Blender --background --factory-startup --python tools/blender/normalize_model.py -- \
        --input model.gltf --output model.glb --report model.json

Applies all transforms, joins nothing, moves the origin to the centre of the floor contact
(bounding-box bottom centre), decimates meshes above 150k triangles in total, and exports a GLB
with embedded textures. glTF is +Y up, metres, which glTFast imports into Unity unchanged.
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

MAX_TRIS = 150_000


def args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--target-size", type=float)
    p.add_argument("--target-axis", choices=["width", "height", "depth", "longest", "horizontal"], default="height")
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--preview")
    p.add_argument("--keep-mesh", help="Regex matched against a mesh and its parent names")
    p.add_argument("--drop-mesh", help="Regex matched against a mesh and its parent names")
    p.add_argument("--scale-reference", help="Measure the target axis on these meshes, then scale the entire retained asset")
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

    def matches(obj, pattern):
        while obj is not None:
            if re.search(pattern, obj.name):
                return True
            obj = obj.parent
        return False

    for obj in list(bpy.context.scene.objects):
        if obj.type == "MESH" and ((a.keep_mesh and not matches(obj, a.keep_mesh)) or
                                   (a.drop_mesh and matches(obj, a.drop_mesh))):
            bpy.data.objects.remove(obj, do_unlink=True)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("no meshes imported")

    # Keep skinning and joint names on downloaded characters and robots.
    rigged = any(o.type == "ARMATURE" for o in bpy.context.scene.objects)
    roots = [o for o in bpy.context.scene.objects if o.parent is None]
    turn = Matrix.Rotation(math.radians(a.yaw), 4, "Z")
    for o in roots:
        o.matrix_world = turn @ o.matrix_world
    bpy.context.view_layer.update()
    references = [o for o in meshes if matches(o, a.scale_reference)] if a.scale_reference else meshes
    if not references:
        raise ValueError("scale reference did not match any retained meshes")
    lo, hi = world_bbox(references)
    size = hi - lo
    extent = {"width": size.x, "height": size.z, "depth": size.y,
              "longest": max(size), "horizontal": max(size.x, size.y)}[a.target_axis]
    if extent <= 1e-8 or (a.target_size is not None and a.target_size <= 0):
        raise ValueError("model has no usable extent or target size is invalid")
    factor = a.target_size / extent if a.target_size else 1.0
    lo, hi = world_bbox(meshes)
    shift = Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
    normalise = Matrix.Scale(factor, 4) @ Matrix.Translation(-shift)
    for o in roots:
        o.matrix_world = normalise @ o.matrix_world
    bpy.context.view_layer.update()

    # Bake static meshes after clearing parents; doing this in reverse can move children.
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    # Instanced meshes share data; transform_apply refuses multi-user data.
    bpy.ops.object.make_single_user(object=True, obdata=True)
    if not rigged:
        bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    tris_before = tri_count(meshes)
    if tris_before > MAX_TRIS:
        ratio = MAX_TRIS / tris_before
        for o in meshes:
            mod = o.modifiers.new("decimate", "DECIMATE")
            mod.ratio = ratio
            bpy.context.view_layer.objects.active = o
            bpy.ops.object.modifier_apply(modifier=mod.name)
    tris_after = tri_count(meshes)

    # Imported lamps and cameras do not belong in a simulation prop.
    for o in [o for o in bpy.context.scene.objects if o.type in {"LIGHT", "CAMERA"} or
              (not rigged and o.type != "MESH")]:
        bpy.data.objects.remove(o, do_unlink=True)

    lo, hi = world_bbox(meshes)
    size = hi - lo
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=a.output, export_format="GLB", export_apply=True, export_yup=True)

    report = {
        "input": a.input,
        "output": a.output,
        "size_m": [round(size.x, 3), round(size.z, 3), round(size.y, 3)],  # width (x), height (up), depth
        "tris_before": tris_before,
        "tris_after": tris_after,
        "meshes": len(meshes),
        "rigged": rigged,
        "armatures": [{"name": o.name, "bones": len(o.data.bones)} for o in bpy.context.scene.objects if o.type == "ARMATURE"],
        "scale_factor": factor,
        "floor_min_z": round(lo.z, 6),
        "has_textures": any(n.type == "TEX_IMAGE" and n.image for o in meshes for s in o.material_slots if s.material and s.material.use_nodes for n in s.material.node_tree.nodes),
        "materials": sorted({s.material.name for o in meshes for s in o.material_slots if s.material}),
    }
    with open(a.report, "w") as f:
        json.dump(report, f, indent=2)
    print("REPORT", json.dumps(report))
    if a.preview:
        render_preview(a.preview, lo, hi)


def render_preview(path, lo, hi):
    """Render with CPU Cycles so another Blender GUI and its Metal context stay untouched."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 12
    scene.cycles.use_denoising = True
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    if scene.world is None:
        scene.world = bpy.data.worlds.new("Preview world")
    scene.world.color = (0.22, 0.22, 0.22)
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.32, 0.32, 0.32, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
    centre = (lo + hi) / 2
    extent = max(hi - lo)
    bpy.ops.object.camera_add(location=centre + Vector((1.3, -1.9, 1.05)) * extent)
    camera = bpy.context.object
    camera.rotation_euler = (centre - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = extent * 1.65
    camera.data.clip_end = max(100, extent * 20)
    scene.camera = camera
    for location, power, size in [((1, -2, 3), 500, 2.2), ((-2, -1, 1.5), 250, 2), ((1, 2, 2), 400, 1.8)]:
        bpy.ops.object.light_add(type="AREA", location=centre + Vector(location) * extent)
        lamp = bpy.context.object
        lamp.data.energy = power * extent * extent
        lamp.data.shape = "DISK"
        lamp.data.size = size * extent
        lamp.rotation_euler = (centre - lamp.location).to_track_quat("-Z", "Y").to_euler()
    bpy.ops.mesh.primitive_plane_add(size=max(200, extent * 20), location=(0, 0, lo.z - 0.003))
    ground = bpy.context.object
    mat = bpy.data.materials.new("Preview grey floor")
    mat.diffuse_color = (0.24, 0.24, 0.24, 1)
    ground.data.materials.append(mat)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(path)
    try:
        scene.render.engine = "CYCLES"
        bpy.ops.render.render(write_still=True)
    except RuntimeError:
        scene.render.engine = "CYCLES"
        scene.cycles.use_denoising = False
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
