"""Pose a rigged survivor, bake the pose into static meshes, lay it on the floor, export GLB.

    Blender --background --factory-startup --python tools/blender/pose_survivor.py -- \
        --input model.glb --output posed.glb --pose curled_on_side [--preview preview.png]

Each joint is bent toward a world direction (for example "thigh toward the front"), with the rotation axis
computed from the bone's own direction, so rig roll and axis conventions cannot flip a bend. Characters
from the glTF importer face -Y. Stray meshes that are not skinned to the armature are removed.
"""

import argparse
import math
import sys

import bpy
from mathutils import Matrix, Vector

FRONT, BACK, UP, DOWN = Vector((0, -1, 0)), Vector((0, 1, 0)), Vector((0, 0, 1)), Vector((0, 0, -1))
LEFT, RIGHT = Vector((1, 0, 0)), Vector((-1, 0, 0))   # character's own left is +X when facing -Y

NAMES = {
    "l_thigh": ["LeftUpLeg", "L_Thigh_"], "r_thigh": ["RightUpLeg", "R_Thigh_"],
    "l_shin": ["LeftLeg_", "L_Calf_"], "r_shin": ["RightLeg_", "R_Calf_"],
    "l_arm": ["LeftArm_", "L_Upperarm_"], "r_arm": ["RightArm_", "R_Upperarm_"],
    "l_fore": ["LeftForeArm", "L_Forearm_"], "r_fore": ["RightForeArm", "R_Forearm_"],
    "spine": ["Spine1_", "Spine01_", "Spine_0"], "neck": ["Neck_", "NeckTwist01"],
}

# (bone, direction to bend toward, degrees)
POSES = {
    "lying_on_side": [("l_thigh", FRONT, 40), ("r_thigh", FRONT, 15), ("l_shin", BACK, 60), ("r_shin", BACK, 25),
                      ("l_arm", FRONT, 70), ("r_arm", FRONT, 40), ("l_fore", FRONT, 40), ("spine", FRONT, 10)],
    "slumped_against_wall": [("l_thigh", FRONT, 85), ("r_thigh", FRONT, 80), ("l_shin", BACK, 45), ("r_shin", BACK, 60),
                             ("spine", FRONT, 25), ("neck", FRONT, 35), ("l_arm", LEFT, 15), ("r_arm", RIGHT, 15),
                             ("l_fore", FRONT, 30), ("r_fore", FRONT, 30)],
    "collapsed_face_down": [("l_arm", UP, 150), ("r_arm", RIGHT, 30), ("l_fore", FRONT, 40), ("r_fore", BACK, 30),
                            ("l_thigh", FRONT, 25), ("l_shin", BACK, 45), ("neck", LEFT, 30)],
    "curled_on_side": [("l_thigh", FRONT, 95), ("r_thigh", FRONT, 85), ("l_shin", BACK, 110), ("r_shin", BACK, 105),
                       ("spine", FRONT, 35), ("neck", FRONT, 30), ("l_arm", FRONT, 70), ("r_arm", FRONT, 65),
                       ("l_fore", UP, 80), ("r_fore", UP, 80)],
}
# Whole-body rotation after posing, applied in world space: lie on the right side (rotate about the front axis),
# sit (no change), lie face down (rotate about the left axis so the front faces the floor).
LAY = {"lying_on_side": [(FRONT, 90)], "slumped_against_wall": [], "collapsed_face_down": [(LEFT, 90)], "curled_on_side": [(FRONT, 90)]}


def find(arm, key):
    for pat in NAMES[key]:
        for pb in arm.pose.bones:
            n = pb.name
            if pat in n and "scaleCompensation" not in n and "Twist" not in n.replace("NeckTwist01", "") and "end" not in n.lower():
                return pb
    return None


def bend(arm, pb, toward: Vector, deg: float) -> None:
    if pb is None:
        return
    bpy.context.view_layer.update()
    mw = arm.matrix_world
    head, tail = mw @ pb.head, mw @ pb.tail
    d = (tail - head).normalized()
    axis_w = d.cross(toward)
    if axis_w.length < 1e-4:
        return
    axis_w.normalize()
    # Express the world rotation in armature space (armatures from glTF can carry scale and rotation).
    R_w = Matrix.Rotation(math.radians(deg), 4, axis_w)
    inv = mw.inverted()
    R_a = inv @ Matrix.Translation(head) @ R_w @ Matrix.Translation(-head) @ mw
    pb.matrix = R_a @ pb.matrix
    bpy.context.view_layer.update()


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--input"); ap.add_argument("--output"); ap.add_argument("--pose"); ap.add_argument("--preview", default="")
    a = ap.parse_args(argv)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=a.input)
    arm = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
    for o in [o for o in bpy.context.scene.objects if o.type == "MESH" and not any(m.type == "ARMATURE" for m in o.modifiers)]:
        print("REMOVE stray mesh", o.name, [round(x, 2) for x in o.dimensions])
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    for key, toward, deg in POSES[a.pose]:
        pb = find(arm, key)
        before = arm.matrix_world @ pb.tail if pb else None
        bend(arm, pb, toward, deg)
        if pb:
            after = arm.matrix_world @ pb.tail
            print("BEND", key, pb.name, "tail moved", [round(x, 2) for x in (after - before)])
    bpy.ops.object.mode_set(mode="OBJECT")
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    # Bake: evaluated (posed) mesh in world space becomes a new static mesh.
    deps = bpy.context.evaluated_depsgraph_get()
    baked = []
    for o in meshes:
        ev = o.evaluated_get(deps)
        me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=deps)
        me.transform(o.matrix_world)
        nb = bpy.data.objects.new(o.name + "_posed", me)
        bpy.context.scene.collection.objects.link(nb)
        baked.append(nb)
    for o in [o for o in bpy.context.scene.objects if o not in baked]:
        bpy.data.objects.remove(o, do_unlink=True)
    R = Matrix.Identity(4)
    for axis, deg in LAY[a.pose]:
        R = Matrix.Rotation(math.radians(deg), 4, axis) @ R
    for o in baked:
        o.data.transform(R)
    lo = Vector((1e9,) * 3); hi = Vector((-1e9,) * 3)
    for o in baked:
        for v in o.data.vertices:
            lo = Vector(map(min, lo, v.co)); hi = Vector(map(max, hi, v.co))
    shift = Matrix.Translation(-Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z)))
    for o in baked:
        o.data.transform(shift)
    size = hi - lo
    print("SIZE", round(size.x, 3), round(size.y, 3), round(size.z, 3))
    bpy.ops.export_scene.gltf(filepath=a.output, export_format="GLB", export_yup=True)
    if a.preview:
        cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
        bpy.context.scene.collection.objects.link(cam)
        dist = max(size.x, size.y, size.z) * 1.6 + 0.6
        cam.location = (dist * 0.7, -dist * 0.7, dist * 0.55)
        direction = Vector((0, 0, size.z * 0.4)) - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        bpy.context.scene.camera = cam
        sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
        sun.data.energy = 4
        bpy.context.scene.collection.objects.link(sun)
        sun.rotation_euler = (math.radians(40), 0, math.radians(30))
        sc = bpy.context.scene
        for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
            try:
                sc.render.engine = eng
                break
            except TypeError:
                continue
        sc.render.resolution_x, sc.render.resolution_y = 480, 360
        sc.world = bpy.data.worlds.new("w")
        sc.world.color = (0.45, 0.45, 0.45)
        sc.render.filepath = a.preview
        bpy.ops.render.render(write_still=True)


main()
