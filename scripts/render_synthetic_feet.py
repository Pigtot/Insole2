"""Phase D -- realistic synthetic foot renders in Blender (headless).

    /Applications/Blender.app/Contents/MacOS/Blender --background --factory-startup \
        --python scripts/render_synthetic_feet.py -- \
        --mesh data/raw/foot3d_multiview/.../0036/mesh.obj \
        --out experiments/focus_baseline/synthetic/0036 --views 24

Why this exists
---------------
Milestone 4 measured 2.93 mm chamfer *using the dataset's own camera poses*. An
end-to-end phone capture has to estimate cameras too, and that is precisely the
stage that failed on flat clay renders: COLMAP found no good initial image pair.

Rendering a **known** mesh realistically gives a test with ground truth at both
ends -- we know the true geometry, and COLMAP has to work out the cameras. The
error against the input mesh then quantifies what camera estimation costs.

Because the input mesh is known, ground-truth cameras are deliberately *not*
exported: COLMAP estimating them is the thing under test.

Realism notes
-------------
Foot3D scans carry geometry only -- no vertex colours, no UVs, no material. So
appearance is procedural: a Principled BSDF with subsurface scattering, noise-
driven colour and roughness variation, area lighting and a textured floor. This
mirrors how FOCUS's own SynFoot training data was produced, which is the
distribution the TOC predictor expects.

Runs on Cycles with Metal (Apple M1 Max GPU); falls back to CPU if unavailable.
"""

import argparse
import math
import sys

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", type=int, default=24)
    ap.add_argument("--res-x", type=int, default=480)
    ap.add_argument("--res-y", type=int, default=640)
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--elevation", type=float, default=18.0, help="degrees above floor")
    ap.add_argument("--radius-scale", type=float, default=1.5)
    ap.add_argument("--light-power", type=float, default=1.0,
                    help="multiplier on scene-scaled light energy")
    ap.add_argument("--exposure", type=float, default=0.0, help="stops")
    ap.add_argument("--engine", default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def enable_metal():
    """Use the M1 GPU for Cycles when it is there; say so either way."""
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == "METAL"]
        for d in prefs.devices:
            d.use = (d.type == "METAL")
        if gpus:
            bpy.context.scene.cycles.device = "GPU"
            print(f"RENDER_DEVICE METAL ({gpus[0].name})")
            return True
    except Exception as exc:
        print(f"RENDER_DEVICE metal unavailable: {exc}")
    bpy.context.scene.cycles.device = "CPU"
    print("RENDER_DEVICE CPU")
    return False


def import_foot(path):
    bpy.ops.wm.obj_import(filepath=path)
    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not objs:
        raise RuntimeError(f"no mesh imported from {path}")
    foot = objs[0]
    foot.name = "Foot"
    # Centre on origin and drop the lowest point onto z=0, so the foot rests on
    # the floor the way a photographed foot does.
    bpy.context.view_layer.objects.active = foot
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bb = [foot.matrix_world @ Vector(c) for c in foot.bound_box]
    lo = Vector((min(p.x for p in bb), min(p.y for p in bb), min(p.z for p in bb)))
    hi = Vector((max(p.x for p in bb), max(p.y for p in bb), max(p.z for p in bb)))
    centre = (lo + hi) / 2
    foot.location = (-centre.x, -centre.y, -lo.z)
    bpy.ops.object.transform_apply(location=True)
    bpy.ops.object.shade_smooth()
    return foot, (hi - lo).length


def _set(node, name, value):
    """Set a Principled BSDF socket if this Blender version has it."""
    if name in node.inputs:
        node.inputs[name].default_value = value
        return True
    return False


def skin_material(seed=0):
    mat = bpy.data.materials.new("Skin")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    _set(bsdf, "Base Color", (0.80, 0.61, 0.52, 1.0))
    _set(bsdf, "Roughness", 0.52)
    _set(bsdf, "Specular IOR Level", 0.35) or _set(bsdf, "Specular", 0.35)
    # Subsurface: skin is translucent, and flat diffuse is what made the earlier
    # clay renders read as plastic.
    if not _set(bsdf, "Subsurface Weight", 0.22):
        _set(bsdf, "Subsurface", 0.22)
    _set(bsdf, "Subsurface Radius", (0.012, 0.0055, 0.0035))

    # Fine noise on colour and roughness: a perfectly uniform surface gives a
    # feature-free image, which is part of why the clay renders failed.
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = 120.0
    tex.inputs["Detail"].default_value = 8.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.74, 0.55, 0.47, 1)
    ramp.color_ramp.elements[1].color = (0.87, 0.68, 0.58, 1)
    nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    nt.links.new(tex.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def floor_and_light(size, power=1.0):
    bpy.ops.mesh.primitive_plane_add(size=size * 8, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    mat = bpy.data.materials.new("FloorMat")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    _set(bsdf, "Roughness", 0.85)
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = 22.0
    tex.inputs["Detail"].default_value = 6.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.32, 0.31, 0.30, 1)
    ramp.color_ramp.elements[1].color = (0.55, 0.54, 0.52, 1)
    nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    floor.data.materials.append(mat)

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.42, 0.46, 0.52, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.45
    bpy.context.scene.world = world

    # Area-light power must scale with the SCENE, not be a fixed wattage: a foot
    # is ~0.25 m across, so a light that suits a room blows the exposure out
    # completely (the first attempt rendered pure white).
    for loc, rel in (((size * 2, size * 1.4, size * 2.4), 30.0),
                     ((-size * 2, -size * 1.2, size * 2.0), 14.0)):
        bpy.ops.object.light_add(type="AREA", location=loc)
        light = bpy.context.active_object
        light.data.energy = rel * power * (size ** 2)
        light.data.size = size * 1.5
        d = Vector((0, 0, size * 0.3)) - Vector(loc)
        light.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def add_camera():
    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 40.0
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    return cam


def main():
    args = parse_args()
    clear_scene()
    scene = bpy.context.scene
    scene.render.engine = args.engine
    if args.engine == "CYCLES":
        enable_metal()
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
    scene.render.resolution_x = args.res_x
    scene.render.resolution_y = args.res_y
    scene.render.image_settings.file_format = "PNG"

    foot, diag = import_foot(args.mesh)
    foot.data.materials.clear()
    foot.data.materials.append(skin_material(args.seed))
    floor_and_light(diag, args.light_power)
    scene.view_settings.exposure = args.exposure
    cam = add_camera()

    target = Vector((0, 0, diag * 0.12))
    radius = diag * args.radius_scale
    elev = math.radians(args.elevation)

    import os
    os.makedirs(args.out, exist_ok=True)
    # A perfect circle at fixed elevation and radius is a poorly conditioned
    # camera configuration for structure-from-motion, and it is also unlike a real
    # hand-held capture. Vary elevation and distance deterministically (seeded) so
    # the baseline geometry is better spread while staying reproducible.
    import random as _random
    rng = _random.Random(args.seed)
    for i in range(args.views):
        az = 2 * math.pi * i / args.views
        el = elev + math.radians(rng.uniform(-8.0, 14.0))
        rad = radius * rng.uniform(0.85, 1.18)
        cam.data.lens = 40.0 * rng.uniform(0.92, 1.08)
        cam.location = (rad * math.cos(az) * math.cos(el),
                        rad * math.sin(az) * math.cos(el),
                        target.z + rad * math.sin(el))
        cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = os.path.join(args.out, f"{i:06d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"RENDERED {i + 1}/{args.views}", flush=True)
    print(f"DONE {args.views} views -> {args.out}")


if __name__ == "__main__":
    main()
