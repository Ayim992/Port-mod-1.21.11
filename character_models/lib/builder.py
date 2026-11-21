"""Builds one character .blend from a character spec module."""
import os
import time
import numpy as np
import bpy

from . import anatomy, sdf, voxel, faces, rig as rigmod
from .blend import (PX, reset_scene, collection, make_mesh, voxel_remesh, laplacian_smooth,
                    paint_object, colors_to_layers, body_material, image_from_array, head_material,
                    head_mesh, setup_render, render_views)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "out")

BODY_STEP = 0.12      # marching-cubes sample spacing (px)
REMESH = 0.11         # final voxel remesh size (px) -> ~6 mm


def log(*a):
    print("[build]", *a, flush=True)


def _ellip_falloff(P, c, radii, scale=1.2, front_axis=None, front_sign=1):
    r = np.linalg.norm((P - c) / (np.asarray(radii) * scale), axis=1)
    w = np.clip(1.0 - r, 0, 1)
    w = w * w * (3 - 2 * w)  # smoothstep
    w = np.clip(w * 1.6, 0, 1)
    if front_axis is not None:
        side = (P[:, front_axis] - c[front_axis]) * front_sign
        w *= np.clip(side / 0.8 + 0.6, 0, 1)
    return w


def build(spec, render=True, samples=40, width=900):
    t0 = time.time()
    p = spec.P
    name = spec.NAME
    reset_scene()
    col = collection(name)
    geo = collection(f"{name}_geo", col)

    # ---------------------------------------------------------------- body
    shape, lm = anatomy.body_shape(p)
    if hasattr(spec, "body_mod"):
        shape = spec.body_mod(shape, lm, p)
    v, f = sdf.mesh_from_sdf(shape, (-8.5, -7.5, -0.4), (8.5, 7.5, 25.6), BODY_STEP)
    body = make_mesh(f"{name}_Body", v, f, geo)
    voxel_remesh(body, REMESH)
    laplacian_smooth(body, 2, 0.4)
    log("body", len(body.data.vertices), "verts", f"{time.time() - t0:.1f}s")
    paint_object(body, lambda P, N, layer: spec.paint(P, N, layer, "body", lm))

    # ---------------------------------------------------------------- arms
    arms = {}
    for side, s in (("L", 1), ("R", -1)):
        ash, (piv, d, c) = anatomy.arm_shape(p, s)
        lo = c - 8.5
        hi = c + 8.5
        v, f = sdf.mesh_from_sdf(ash, lo, hi, 0.1)
        ob = make_mesh(f"{name}_Arm.{side}", v, f, geo)
        voxel_remesh(ob, 0.1)
        paint_object(ob, lambda P, N, layer, side=side: spec.paint(P, N, layer, f"arm_{side}", lm))
        arms[side] = ob
    log("arms done", f"{time.time() - t0:.1f}s")

    # ---------------------------------------------------------------- head
    head = head_mesh(f"{name}_Head", geo, (0, 0, 28), 4.0)
    face_rgba, face_emit = faces.atlas(spec.face)
    side_rgba, side_emit = faces.sides_atlas(spec.head_sides())
    tex_dir = os.path.join(OUT, "textures")
    img_f = image_from_array(f"{name}_face", face_rgba, os.path.join(tex_dir, f"{name}_face.png"))
    img_fe = image_from_array(f"{name}_face_emit", face_emit, os.path.join(tex_dir, f"{name}_face_emit.png"))
    img_s = image_from_array(f"{name}_head", side_rgba, os.path.join(tex_dir, f"{name}_head.png"))
    img_se = image_from_array(f"{name}_head_emit", side_emit, os.path.join(tex_dir, f"{name}_head_emit.png"))

    # ---------------------------------------------------------------- props (voxel / boxes)
    props = []
    for pr in spec.props(lm, p):
        if pr["kind"] == "voxel":
            occ, lo = sdf.voxelize(pr["shape"], pr["lo"], pr["hi"], pr["vox"])
            v, f, cols, em = voxel.mesh_voxels(occ, lo, pr["vox"], pr["color"], pr.get("emit"))
        else:
            v, f, cols, em = voxel.boxes_mesh(pr["boxes"])
        if len(f) == 0:
            log("WARN empty prop", pr["name"])
            continue
        ob = make_mesh(f"{name}_{pr['name']}", v, f, geo, smooth=False)
        colors_to_layers(ob, cols, em)
        props.append((ob, pr))
    log("props", len(props), f"{time.time() - t0:.1f}s")

    # ---------------------------------------------------------------- rig
    bones, extra = anatomy.rig_bones(p)
    if hasattr(spec, "rig_mod"):
        bones, extra = spec.rig_mod(bones, extra, p)
    meta = rigmod.build_metarig(col, bones, extra)
    rig, res = rigmod.generate(meta, col, f"{name}_rig")
    log("rig", res, len(rig.data.bones), "bones")
    rigmod.add_rig_prop(rig, "outfit", 1, 0, 1, "1 = dressed, 0 = nude", integer=True)
    rigmod.add_rig_prop(rig, "expression", 0, 0, len(faces.EXPRESSIONS) - 1,
                        "Face: " + ", ".join(f"{i}={e}" for i, e in enumerate(faces.EXPRESSIONS)),
                        integer=True)
    for prop, default, desc in getattr(spec, "RIG_TOGGLES", []):
        rigmod.add_rig_prop(rig, prop, default, 0, 1, desc, integer=True)

    # ---------------------------------------------------------------- materials
    emit_k = getattr(spec, "EMIT_STRENGTH", 6.0)
    mat_body = body_material(f"{name}_Skin", rig, emit_strength=emit_k)
    for ob in [body, arms["L"], arms["R"]] + [o for o, _ in props]:
        ob.data.materials.append(mat_body)
    mat_face = head_material(f"{name}_Face", img_f, img_fe, len(faces.EXPRESSIONS), rig, "expression",
                             emit_strength=emit_k)
    mat_head = head_material(f"{name}_HeadSides", img_s, img_se, 1, emit_strength=emit_k)
    mat_head.node_tree.nodes["Mapping"].inputs["Scale"].default_value = (1, 1, 1)
    head.data.materials.append(mat_face)
    head.data.materials.append(mat_head)

    # ---------------------------------------------------------------- skinning
    cz = p["crotch_z"]
    up_fade = lambda lo_z, span: (lambda P: np.clip((lo_z + span - P[:, 2]) / span, 0.02, 1.0))
    down_fade = lambda lo_z, span: (lambda P: np.clip((P[:, 2] - lo_z) / span, 0.0, 1.0))
    body_bones = ["DEF-spine", "DEF-spine.001", "DEF-spine.002", "DEF-spine.003", "DEF-spine.004",
                  "DEF-spine.005"]
    for s in ("L", "R"):
        body_bones += [f"DEF-thigh.{s}", f"DEF-thigh.{s}.001", f"DEF-shin.{s}", f"DEF-shin.{s}.001",
                       f"DEF-foot.{s}", f"DEF-toe.{s}", f"DEF-breast.{s}", f"DEF-butt.{s}"]
    body_bones.append("DEF-belly")
    masks = {}
    for s in ("L", "R"):
        masks[f"DEF-thigh.{s}"] = up_fade(cz - 0.6, 2.2)
        masks[f"DEF-foot.{s}"] = up_fade(0.0, 1.6)
        masks[f"DEF-toe.{s}"] = lambda P: np.zeros(len(P))
        masks[f"DEF-breast.{s}"] = lambda P: np.zeros(len(P))
        masks[f"DEF-butt.{s}"] = lambda P: np.zeros(len(P))
    masks["DEF-belly"] = lambda P: np.zeros(len(P))
    masks["DEF-spine"] = down_fade(cz - 2.5, 2.5)
    masks["DEF-spine.005"] = down_fade(24.0, 0.3)
    bc = lm["breast_centers"]
    btc = lm["butt_centers"]
    overrides = [
        ("DEF-breast.L", lambda P: _ellip_falloff(P, bc[0], p["breast_r"], 1.25, 1, -1)),
        ("DEF-breast.R", lambda P: _ellip_falloff(P, bc[1], p["breast_r"], 1.25, 1, -1)),
        ("DEF-butt.L", lambda P: _ellip_falloff(P, btc[0], p["butt_r"], 1.2, 1, 1) *
         np.clip((P[:, 0] + 0.3) / 0.6, 0, 1)),
        ("DEF-butt.R", lambda P: _ellip_falloff(P, btc[1], p["butt_r"], 1.2, 1, 1) *
         np.clip((-P[:, 0] + 0.3) / 0.6, 0, 1)),
        ("DEF-belly", lambda P: 0.75 * _ellip_falloff(P, np.array([0, p["belly_y"], p["belly_z"]]),
                                                      p["belly_r"], 1.15, 1, -1)),
    ]
    st = rigmod.skin(body, rig, body_bones, masks=masks, overrides=overrides, smooth_iter=14,
                     side_soft=0.15)
    log("body skin", st)
    for side in ("L", "R"):
        ab = [f"DEF-shoulder.{side}", f"DEF-upper_arm.{side}", f"DEF-upper_arm.{side}.001",
              f"DEF-forearm.{side}", f"DEF-forearm.{side}.001", f"DEF-hand.{side}"]
        am = {f"DEF-shoulder.{side}": lambda P: np.full(len(P), 0.05)}
        st = rigmod.skin(arms[side], rig, ab, masks=am, smooth_iter=6, side_soft=50)
        log("arm skin", side, st)
    rigmod.rigid(head, rig, "DEF-spine.006")
    for ob, pr in props:
        rigmod.rigid(ob, rig, pr.get("bone", "DEF-spine.006"))
        if pr.get("toggle"):
            rigmod.drive_visibility(ob, rig, pr["toggle"], pr.get("toggle_on", 1))

    # ---------------------------------------------------------------- finish
    for ob in geo.objects:
        ob.hide_select = False
    rig.show_in_front = True
    bpy.context.view_layer.objects.active = rig
    os.makedirs(OUT, exist_ok=True)
    blend_path = os.path.join(OUT, f"{name}.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path, compress=True)
    log("saved", blend_path, f"{time.time() - t0:.1f}s")

    renders = []
    if render:
        setup_render((width, width * 4 // 3), samples)
        rdir = os.path.join(OUT, "renders")
        rcfg = getattr(spec, "RENDER", {})
        for outfit in (1, 0):
            rig["outfit"] = outfit
            bpy.context.view_layer.update()
            tag = "dressed" if outfit else "nude"
            renders += render_views(os.path.join(rdir, f"{name}_{tag}"),
                                    [("front", 0, 5), ("q3", 35, 8), ("side", 90, 5), ("back", 180, 5)],
                                    target=(0, 0, rcfg.get("target_z", 0.95)), ortho_scale=rcfg.get("ortho", 2.3))
        rig["outfit"] = 1
    return blend_path, renders
