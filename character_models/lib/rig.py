"""Rigify metarig fitting, generation, and geometric skin weights."""
import numpy as np
import bpy
from mathutils import Vector
from .blend import PX


def build_metarig(col, bones_px, extra_bones=()):
    """bones_px: {metarig bone name: (head_px, tail_px)}; .L entries are mirrored to .R.
    extra_bones: list of dict(name, head, tail, parent, rigify_type, mirror=True)."""
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.armature_basic_human_metarig_add()
    meta = bpy.context.object
    meta.name = "metarig"
    for c in list(meta.users_collection):
        c.objects.unlink(meta)
    col.objects.link(meta)
    bpy.context.view_layer.objects.active = meta
    bpy.ops.object.mode_set(mode="EDIT")
    eb = meta.data.edit_bones

    def mir(v):
        return (-v[0], v[1], v[2])

    for name, (h, t) in bones_px.items():
        targets = [(name, h, t)]
        if name.endswith(".L"):
            targets.append((name[:-2] + ".R", mir(h), mir(t)))
        for n, hh, tt in targets:
            b = eb[n]
            b.head = Vector(hh) * PX
            b.tail = Vector(tt) * PX
    # keep connected chains consistent: children heads follow parent tails where connected
    for b in eb:
        if b.use_connect and b.parent:
            b.head = b.parent.tail
    # recompute rolls so limbs bend the right way
    for side in ("L", "R"):
        for n in (f"thigh.{side}", f"shin.{side}", f"foot.{side}", f"toe.{side}"):
            eb[n].align_roll(Vector((0, -1, 0)))
        for n in (f"upper_arm.{side}", f"forearm.{side}", f"hand.{side}"):
            eb[n].align_roll(Vector((0, 1, 0)))
    for ex in extra_bones:
        sides = [("", 1)]
        if ex.get("mirror", True) and ex["name"].endswith(".L"):
            sides = [(".L", 1), (".R", -1)]
        for suffix, s in sides:
            nm = ex["name"][:-2] + suffix if suffix else ex["name"]
            b = eb.new(nm)
            h, t = ex["head"], ex["tail"]
            b.head = Vector((h[0] * s, h[1], h[2])) * PX
            b.tail = Vector((t[0] * s, t[1], t[2])) * PX
            par = ex["parent"]
            if par.endswith(".L") and s < 0:
                par = par[:-2] + ".R"
            b.parent = eb[par]
            b.roll = 0.0
    bpy.ops.object.mode_set(mode="OBJECT")
    for ex in extra_bones:
        names = [ex["name"]]
        if ex.get("mirror", True) and ex["name"].endswith(".L"):
            names.append(ex["name"][:-2] + ".R")
        for nm in names:
            pb = meta.pose.bones[nm]
            pb.rigify_type = ex.get("rigify_type", "basic.super_copy")
            if pb.rigify_type == "basic.super_copy":
                pb.rigify_parameters.make_deform = True
                pb.rigify_parameters.make_control = True
    return meta


def generate(meta, col, name):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = meta
    meta.select_set(True)
    before = {o.name for o in bpy.data.objects}
    res = bpy.ops.pose.rigify_generate()
    rig = getattr(meta.data, "rigify_target_rig", None)
    if rig is None:
        rig = next(o for o in bpy.data.objects if o.name not in before and o.type == "ARMATURE")
    rig.name = name
    rig.data.name = name
    for c in list(rig.users_collection):
        c.objects.unlink(rig)
    col.objects.link(rig)
    meta.hide_set(True)
    meta.hide_render = True
    return rig, res


def deform_bones_px(rig):
    out = {}
    for b in rig.data.bones:
        if b.use_deform:
            out[b.name] = (np.array(b.head_local) / PX, np.array(b.tail_local) / PX)
    return out


def _seg_dist(P, a, b):
    ba = b - a
    t = np.clip(((P - a) @ ba) / max(ba @ ba, 1e-9), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ba), axis=1), t


def vertex_positions_px(ob):
    me = ob.data
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    return co.reshape(-1, 3).astype(np.float64) / PX


def edges(ob):
    e = np.empty(len(ob.data.edges) * 2, np.int32)
    ob.data.edges.foreach_get("vertices", e)
    return e.reshape(-1, 2)


def smooth_weights(W, E, iterations, factor=0.5):
    n = W.shape[0]
    deg = np.bincount(E.ravel(), minlength=n).astype(np.float64)
    for _ in range(iterations):
        acc = np.zeros_like(W)
        np.add.at(acc, E[:, 0], W[E[:, 1]])
        np.add.at(acc, E[:, 1], W[E[:, 0]])
        W = W + factor * (acc / np.maximum(deg, 1)[:, None] - W)
    return W


def skin(ob, rig, bone_names, power=4.0, smooth_iter=6, masks=None, side_soft=0.6, overrides=None,
         max_influences=4):
    """Geometric skin weights.

    bone_names: deform bones allowed for this object.
    masks: {bone: fn(P)->multiplier (V,)} to shape influence regions.
    overrides: list of (bone, fn(P)->weight (V,) in 0..1): bone gets that weight, others scaled by 1-w.
    """
    P = vertex_positions_px(ob)
    bones = deform_bones_px(rig)
    names = [n for n in bone_names if n in bones]
    W = np.zeros((len(P), len(names)))
    for j, n in enumerate(names):
        a, b = bones[n]
        d, _ = _seg_dist(P, a, b)
        w = 1.0 / np.maximum(d, 0.15) ** power
        if n.endswith(".L"):
            w *= np.clip((P[:, 0] + side_soft) / (2 * side_soft), 0, 1)
        elif n.endswith(".R"):
            w *= np.clip((-P[:, 0] + side_soft) / (2 * side_soft), 0, 1)
        if masks and n in masks:
            w *= masks[n](P)
        W[:, j] = w
    W /= np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    E = edges(ob)
    if smooth_iter:
        W = smooth_weights(W, E, smooth_iter)
    if overrides:
        for bone, fn in overrides:
            if bone not in names:
                continue
            j = names.index(bone)
            w = np.clip(fn(P), 0, 1)
            W *= (1 - w)[:, None]
            W[:, j] += w
    # prune and limit influences
    if W.shape[1] > max_influences:
        idx = np.argsort(-W, axis=1)[:, max_influences:]
        np.put_along_axis(W, idx, 0.0, axis=1)
    W[W < 0.01] = 0.0
    W /= np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    _write_groups(ob, names, W)
    _add_armature(ob, rig)
    return {"verts": len(P), "unweighted": int((W.sum(axis=1) < 0.99).sum())}


def rigid(ob, rig, bone):
    n = len(ob.data.vertices)
    vg = ob.vertex_groups.get(bone) or ob.vertex_groups.new(name=bone)
    vg.add(list(range(n)), 1.0, "REPLACE")
    _add_armature(ob, rig)


def _write_groups(ob, names, W):
    for j, n in enumerate(names):
        col = W[:, j]
        nz = np.nonzero(col)[0]
        if len(nz) == 0:
            continue
        vg = ob.vertex_groups.get(n) or ob.vertex_groups.new(name=n)
        # group vertices by quantized weight to reduce API calls
        q = np.round(col[nz], 3)
        for val in np.unique(q):
            ids = nz[q == val].tolist()
            vg.add(ids, float(val), "REPLACE")


def _add_armature(ob, rig):
    if not any(m.type == "ARMATURE" for m in ob.modifiers):
        m = ob.modifiers.new("Armature", "ARMATURE")
        m.object = rig
        m.use_deform_preserve_volume = True
    ob.parent = rig
    ob.matrix_parent_inverse = rig.matrix_world.inverted()


def add_rig_prop(rig, name, default, lo, hi, desc, integer=False):
    rig[name] = int(default) if integer else float(default)
    ui = rig.id_properties_ui(name)
    ui.update(min=lo, max=hi, soft_min=lo, soft_max=hi, description=desc)


def drive_visibility(ob, rig, prop, show_when=1):
    """Hide object when rig[prop] != show_when (viewport + render)."""
    for path in ("hide_viewport", "hide_render"):
        fc = ob.driver_add(path)
        d = fc.driver
        d.type = "SCRIPTED"
        v = d.variables.new()
        v.name = "v"
        v.targets[0].id = rig
        v.targets[0].data_path = f'["{prop}"]'
        d.expression = f"round(v) != {show_when}"
