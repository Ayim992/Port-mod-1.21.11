"""Exact importer for GeckoLib / Bedrock `.geo.json` models and `.animation.json` animations.

Every cube keeps its exact origin, size, inflate, pivot, rotation and per-face UV. Every bone
becomes an armature bone at its exact pivot and rest orientation; cubes are bound 100% to their
bone, exactly as GeckoLib renders them. Animations are evaluated the way GeckoLib does
(euler sum of rest + animated rotation, per-segment easing, molang constants) and baked
to Blender actions, one keyframe per frame.

Coordinate chain: Bedrock -> Blockbench internal (x mirrored, rot X/Y negated, ZYX order)
-> Blender (X = -x_int, Y = z_int, Z = y_int; face north -> -Y). 1 px = 1/16 m.
"""
import json
import math
import re

import numpy as np
import bpy
from mathutils import Matrix

PXM = 1.0 / 16.0  # Minecraft: 16 px = 1 block = 1 m

# internal (three.js / Blockbench) -> Blender, proper rotation
_M3 = np.array([[-1.0, 0, 0], [0, 0, 1.0], [0, 1.0, 0]])
_M4 = np.eye(4)
_M4[:3, :3] = _M3
_M4i = np.linalg.inv(_M4)


# --------------------------------------------------------------------------- math
def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


ROT_SIGNS = (-1.0, -1.0, 1.0)


def rot_internal(bedrock_deg):
    """Bedrock euler (deg) -> internal rotation matrix (ZYX order, signs per ROT_SIGNS)."""
    x, y, z = (math.radians(v) * s for v, s in zip(bedrock_deg, ROT_SIGNS))
    return _rz(z) @ _ry(y) @ _rx(x)


def T(v):
    m = np.eye(4)
    m[:3, 3] = v
    return m


def R4(r3):
    m = np.eye(4)
    m[:3, :3] = r3
    return m


def S4(s):
    return np.diag([s[0], s[1], s[2], 1.0])


def piv_internal(p):
    return np.array([-p[0], p[1], p[2]], dtype=np.float64)


# --------------------------------------------------------------------------- geometry
def load_geo(path):
    d = json.load(open(path))
    g = d["minecraft:geometry"][0]
    desc = g["description"]
    bones = {}
    order = []
    for b in g["bones"]:
        bones[b["name"]] = b
        order.append(b["name"])
    return desc, bones, order


def topo_order(bones, order):
    out, seen = [], set()

    def visit(n):
        if n in seen:
            return
        p = bones[n].get("parent")
        if p and p in bones:
            visit(p)
        seen.add(n)
        out.append(n)

    for n in order:
        visit(n)
    return out


def rest_world(bones, order):
    """Internal-space 4x4 rest transform of every bone (absolute pivots, parent-composed)."""
    W = {}
    for n in topo_order(bones, order):
        b = bones[n]
        p = piv_internal(b.get("pivot", [0, 0, 0]))
        L = T(p) @ R4(rot_internal(b.get("rotation", [0, 0, 0]))) @ T(-p)
        par = b.get("parent")
        W[n] = (W[par] if par in W else np.eye(4)) @ L
    return W


# face -> corner selectors (TL, TR, BR, BL as seen from outside), each corner = (xi, yi, zi) in {0,1}
_FACE_CORNERS = {
    "north": [(1, 1, 0), (0, 1, 0), (0, 0, 0), (1, 0, 0)],
    "south": [(0, 1, 1), (1, 1, 1), (1, 0, 1), (0, 0, 1)],
    "east": [(1, 1, 1), (1, 1, 0), (1, 0, 0), (1, 0, 1)],
    "west": [(0, 1, 0), (0, 1, 1), (0, 0, 1), (0, 0, 0)],
    "up": [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],
    "down": [(0, 0, 1), (1, 0, 1), (1, 0, 0), (0, 0, 0)],
}
_FACE_AXIS = {"north": 2, "south": 2, "east": 0, "west": 0, "up": 1, "down": 1}


def _cube_box(c):
    o = np.asarray(c["origin"], dtype=np.float64)
    s = np.asarray(c["size"], dtype=np.float64)
    inf = float(c.get("inflate", 0.0))
    lo = np.array([-(o[0] + s[0]), o[1], o[2]]) - inf
    hi = np.array([-o[0], o[1] + s[1], o[2] + s[2]]) + inf
    C = np.eye(4)
    if "rotation" in c:
        cp = piv_internal(c.get("pivot", [0, 0, 0]))
        C = T(cp) @ R4(rot_internal(c["rotation"])) @ T(-cp)
    return lo, hi, C


def solid_cubes(bones, order, W, bone_filter):
    """Inverse transforms + bounds of every cube with volume (internal space)."""
    inv, los, his = [], [], []
    for n in order:
        if not bone_filter(n):
            continue
        for c in bones[n].get("cubes") or []:
            lo, hi, C = _cube_box(c)
            if np.all(hi - lo > 1e-4):
                inv.append(np.linalg.inv(W[n] @ C))
                los.append(lo)
                his.append(hi)
    return np.asarray(inv), np.asarray(los), np.asarray(his)


def _inside_any(p, solids, margin=1e-4):
    inv, los, his = solids
    if len(inv) == 0:
        return False
    q = inv[:, :3, :3] @ p + inv[:, :3, 3]
    return bool(np.any(np.all((q > los + margin) & (q < his - margin), axis=1)))


def cubes_to_mesh(bones, order, W, tex_w, tex_h, bone_filter, cull_buried_planes=True, moving=()):
    """Return verts (Blender meters), quads, per-loop uvs, per-vertex bone index, bone names.

    Zero-thickness cubes (decal planes) get both faces like in game, except a face that points
    into another cube: Minecraft never shows it, but in a path tracer it shadows the skin under
    the decal (black patches)."""
    verts, faces, uvs, vbone = [], [], [], []
    names = []
    solids = solid_cubes(bones, order, W, bone_filter) if cull_buried_planes else None
    culled = 0
    for n in order:
        if not bone_filter(n):
            continue
        b = bones[n]
        cubes = b.get("cubes") or []
        if not cubes:
            continue
        bi = len(names)
        names.append(n)
        Wb = W[n]
        for c in cubes:
            o = np.asarray(c["origin"], dtype=np.float64)
            s = np.asarray(c["size"], dtype=np.float64)
            inf = float(c.get("inflate", 0.0))
            lo = np.array([-(o[0] + s[0]), o[1], o[2]]) - inf
            hi = np.array([-o[0], o[1] + s[1], o[2] + s[2]]) + inf
            C = np.eye(4)
            if "rotation" in c:
                cp = piv_internal(c.get("pivot", [0, 0, 0]))
                C = T(cp) @ R4(rot_internal(c["rotation"])) @ T(-cp)
            X = _M4 @ Wb @ C
            uvd = c.get("uv")
            if not isinstance(uvd, dict):
                continue  # box-UV cubes are not used by these models
            ext = hi - lo
            # paper-thin planes: pull the two faces 0.01 px apart so the one facing the camera
            # always wins (coincident faces flicker or go black in a path tracer)
            flat = ext <= 1e-6
            lo = lo - 0.01 * flat
            hi = hi + 0.01 * flat
            for face, spec in uvd.items():
                if face not in _FACE_CORNERS:
                    continue
                # skip faces with zero area (the two faces across a flat plane are kept)
                ax = _FACE_AXIS[face]
                other = [i for i in range(3) if i != ax]
                if ext[other[0]] <= 1e-9 or ext[other[1]] <= 1e-9:
                    continue
                # (pieces that animations slide out, e.g. alternate eyes/mouths/blush parked inside
                # the head, are never culled)
                if solids is not None and ext[ax] <= 1e-6 and n not in moving:
                    sign = 1.0 if face in ("south", "east", "up") else -1.0
                    cen = (lo + hi) / 2
                    cen[ax] += sign * 0.08
                    pw = (Wb @ C @ np.r_[cen, 1.0])[:3]
                    if _inside_any(pw, solids):
                        culled += 1
                        continue
                u, v = spec["uv"]
                w, h = spec.get("uv_size", [0, 0])
                u1, v1, u2, v2 = u, v, u + w, v + h
                corner_uv = [(u1, v1), (u2, v1), (u2, v2), (u1, v2)]
                base = len(verts)
                for sel in _FACE_CORNERS[face]:
                    pt = np.array([hi[i] if sel[i] else lo[i] for i in range(3)] + [1.0])
                    verts.append((X @ pt)[:3] * PXM)
                    vbone.append(bi)
                # CCW from outside: TL, BL, BR, TR
                faces.append((base, base + 3, base + 2, base + 1))
                for k in (0, 3, 2, 1):
                    cu, cv = corner_uv[k]
                    uvs.append((cu / tex_w, 1.0 - cv / tex_h))
    if culled:
        print(f"[import] culled {culled} buried decal faces", flush=True)
    return np.asarray(verts), np.asarray(faces), np.asarray(uvs), np.asarray(vbone), names


def build_mesh_object(name, col, verts, faces, uvs, vbone, names, material):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    me.loops.add(len(faces) * 4)
    me.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
    me.polygons.add(len(faces))
    me.polygons.foreach_set("loop_start", np.arange(0, len(faces) * 4, 4, dtype=np.int32))
    me.update(calc_edges=True)
    uvl = me.uv_layers.new(name="UVMap")
    uvl.data.foreach_set("uv", uvs.astype(np.float32).ravel())
    me.validate()
    me.shade_flat()
    me.materials.append(material)
    ob = bpy.data.objects.new(name, me)
    col.objects.link(ob)
    for j, n in enumerate(names):
        vg = ob.vertex_groups.new(name=n)
        ids = np.nonzero(vbone == j)[0].tolist()
        if ids:
            vg.add(ids, 1.0, "REPLACE")
    return ob


def texture_material(name, image_path, rgba_dir=None):
    # Palette PNGs with tRNS transparency lose alpha in Blender's loader: convert to RGBA first.
    from PIL import Image
    import os
    rgba_dir = rgba_dir or os.path.join(os.path.dirname(image_path), "..", "..", "out", "textures")
    os.makedirs(rgba_dir, exist_ok=True)
    rgba_path = os.path.abspath(os.path.join(rgba_dir, f"{name}.png"))
    Image.open(image_path).convert("RGBA").save(rgba_path)
    img = bpy.data.images.load(rgba_path)
    img.alpha_mode = "STRAIGHT"
    img.pack()
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Closest"
    tex.location = (-400, 200)
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    # Minecraft culls back faces: through alpha cutouts you must not see the cube interiors.
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    geo.location = (-400, -200)
    inv = nt.nodes.new("ShaderNodeMath")
    inv.operation = "SUBTRACT"
    inv.inputs[0].default_value = 1.0
    inv.location = (-220, -200)
    nt.links.new(geo.outputs["Backfacing"], inv.inputs[1])
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.location = (-60, -100)
    nt.links.new(tex.outputs["Alpha"], mul.inputs[0])
    nt.links.new(inv.outputs[0], mul.inputs[1])
    # hard cutout like the game (alpha test at 0.5 instead of blending)
    cut = nt.nodes.new("ShaderNodeMath")
    cut.operation = "GREATER_THAN"
    cut.inputs[1].default_value = 0.5
    cut.location = (100, -100)
    nt.links.new(mul.outputs[0], cut.inputs[0])
    nt.links.new(cut.outputs[0], bsdf.inputs["Alpha"])
    bsdf.inputs["Roughness"].default_value = 0.6
    bsdf.inputs["Specular IOR Level"].default_value = 0.25
    mat.use_backface_culling = True
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED"
    return mat


# --------------------------------------------------------------------------- armature
_Q = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])  # bone local axes -> (X, Z, -Y): bone Y points up


def bone_rest_blender(Wb, pivot_int):
    """Blender armature-space rest matrix for a bone (head at pivot, Y axis = model up)."""
    head = (_M4 @ Wb @ np.r_[pivot_int, 1.0])[:3] * PXM
    rot = _M3 @ Wb[:3, :3] @ _M3.T @ _Q
    A = np.eye(4)
    A[:3, :3] = rot
    A[:3, 3] = head
    return A


def build_armature(name, col, bones, order, W, bone_len=0.08):
    arm = bpy.data.armatures.new(name)
    rig = bpy.data.objects.new(name, arm)
    col.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    A = {}
    for n in topo_order(bones, order):
        b = bones[n]
        eb = arm.edit_bones.new(n)
        Ab = bone_rest_blender(W[n], piv_internal(b.get("pivot", [0, 0, 0])))
        A[n] = Ab
        eb.head = (0, 0, 0)
        eb.tail = (0, bone_len, 0)
        eb.matrix = Matrix(Ab.tolist())
        par = b.get("parent")
        if par and par in arm.edit_bones:
            eb.parent = arm.edit_bones[par]
            eb.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    arm.display_type = "STICK"
    rig.show_in_front = True
    return rig, A


def bind(ob, rig):
    m = ob.modifiers.new("Armature", "ARMATURE")
    m.object = rig
    ob.parent = rig


# --------------------------------------------------------------------------- animation
def _ease(name, t):
    t = min(max(t, 0.0), 1.0)
    if not name or name == "linear":
        return t
    pi = math.pi
    if name == "step":
        return 0.0 if t < 1 else 1.0
    base = name.replace("easeInOut", "").replace("easeIn", "").replace("easeOut", "")
    f = {
        "Sine": lambda x: 1 - math.cos(x * pi / 2),
        "Quad": lambda x: x * x,
        "Cubic": lambda x: x ** 3,
        "Quart": lambda x: x ** 4,
        "Quint": lambda x: x ** 5,
        "Expo": lambda x: 0.0 if x == 0 else 2 ** (10 * x - 10),
        "Circ": lambda x: 1 - math.sqrt(max(0.0, 1 - x * x)),
        "Back": lambda x: 2.70158 * x ** 3 - 1.70158 * x * x,
        "Elastic": lambda x: 0.0 if x in (0, 1) else
        -(2 ** (10 * x - 10)) * math.sin((x * 10 - 10.75) * (2 * pi / 3)),
        "Bounce": None,
    }.get(base)
    if base == "Bounce":
        def out_b(x):
            n1, d1 = 7.5625, 2.75
            if x < 1 / d1:
                return n1 * x * x
            if x < 2 / d1:
                x -= 1.5 / d1
                return n1 * x * x + 0.75
            if x < 2.5 / d1:
                x -= 2.25 / d1
                return n1 * x * x + 0.9375
            x -= 2.625 / d1
            return n1 * x * x + 0.984375
        f = lambda x: 1 - out_b(1 - x)
    if f is None:
        return t
    if name.startswith("easeInOut"):
        return f(2 * t) / 2 if t < 0.5 else 1 - f(2 - 2 * t) / 2
    if name.startswith("easeIn"):
        return f(t)
    if name.startswith("easeOut"):
        return 1 - f(1 - t)
    return t


_NUM_EXPR = re.compile(r"^[\d\s.+\-*/()]+$")


def _num(x):
    """Numbers, or GeckoLib/molang strings. Only constant arithmetic is evaluated;
    identifiers (query.*, stray easing names) count as 0, which matches how these files use them."""
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    s = re.sub(r"[A-Za-z_][A-Za-z_0-9.]*(\([^)]*\))?", "0", s)
    if not s:
        return 0.0
    if _NUM_EXPR.match(s):
        try:
            return float(eval(s, {"__builtins__": {}}, {}))
        except Exception:
            return 0.0
    return 0.0


def _vec(v):
    if isinstance(v, (int, float, str)):
        n = _num(v)
        return [n, n, n]
    return [_num(x) for x in v]


def parse_channel(ch):
    """-> sorted list of (time, vec3, easing)."""
    if isinstance(ch, (list, int, float, str)):
        return [(0.0, _vec(ch), None)]
    keys = []
    for t, kv in ch.items():
        try:
            tt = float(t)
        except ValueError:
            continue
        if isinstance(kv, dict):
            vec = kv.get("vector", kv.get("post", kv.get("pre", [0, 0, 0])))
            keys.append((tt, _vec(vec), kv.get("easing")))
        else:
            keys.append((tt, _vec(kv), None))
    if not keys and "vector" in ch:
        keys.append((0.0, _vec(ch["vector"]), ch.get("easing")))
    keys.sort(key=lambda k: k[0])
    return keys


def sample(keys, t):
    if not keys:
        return None
    if t <= keys[0][0]:
        return np.array(keys[0][1])
    for i in range(1, len(keys)):
        t0, v0, _ = keys[i - 1]
        t1, v1, e1 = keys[i]
        if t <= t1:
            span = t1 - t0
            a = 1.0 if span <= 0 else (t - t0) / span
            a = _ease(e1, a)
            return np.array(v0) + (np.array(v1) - np.array(v0)) * a
    return np.array(keys[-1][1])


def bake_animations(rig, A, W, bones, order, anim_path, fps=24, prefix=""):
    anims = json.load(open(anim_path))["animations"]
    topo = topo_order(bones, order)
    animated_any = set()
    parsed = {}
    for an, data in anims.items():
        ch = {}
        for bn, bd in (data.get("bones") or {}).items():
            if bn not in bones:
                continue
            ch[bn] = {k: parse_channel(v) for k, v in bd.items() if k in ("rotation", "position", "scale")}
            animated_any.add(bn)
        parsed[an] = (data, ch)
    Ainv = {n: np.linalg.inv(A[n]) for n in A}
    rest_local = {}
    for n in topo:
        b = bones[n]
        p = piv_internal(b.get("pivot", [0, 0, 0]))
        rest_local[n] = (p, np.array(b.get("rotation", [0, 0, 0]), dtype=np.float64))
    made = []
    for an, (data, ch) in parsed.items():
        length = float(data.get("animation_length", 0) or 0)
        if length <= 0:
            ts = [t for c in ch.values() for keys in c.values() for (t, _, _) in keys]
            length = max(ts) if ts else 0.0
        nframes = max(1, int(round(length * fps)) + 1)
        short = an.split(".", 1)[-1] if an.startswith("animation.") else an
        act = bpy.data.actions.new(prefix + short)
        act.use_fake_user = True
        act.frame_range = (0, nframes - 1)
        act.use_frame_range = True
        act["loop"] = str(data.get("loop", False))
        act["source"] = an
        slot = act.slots.new(id_type="OBJECT", name=rig.name)
        layer = act.layers.new("Layer")
        strip = layer.strips.new(type="KEYFRAME")
        bag = strip.channelbag(slot, ensure=True)
        frames = np.arange(nframes)
        for n in sorted(animated_any):
            c = ch.get(n, {})
            p, rrest = rest_local[n]
            locs = np.zeros((nframes, 3))
            quats = np.zeros((nframes, 4))
            scls = np.ones((nframes, 3))
            Lrest = T(p) @ R4(rot_internal(rrest)) @ T(-p)
            Lrest_inv = np.linalg.inv(Lrest)
            par = bones[n].get("parent")
            # the animated delta lives in the parent's rest frame
            Wp = W[par] if par in W else np.eye(4)
            Wp_inv = np.linalg.inv(Wp)
            for fi in frames:
                t = fi / fps
                rot = sample(c.get("rotation"), t)
                pos = sample(c.get("position"), t)
                scl = sample(c.get("scale"), t)
                rtot = rrest + (rot if rot is not None else 0.0)
                L = T(p) @ R4(rot_internal(rtot))
                if scl is not None:
                    L = L @ S4(scl)
                L = L @ T(-p)
                if pos is not None:
                    L = T(np.array([-pos[0], pos[1], pos[2]])) @ L
                # delta in parent rest frame -> Blender armature space -> bone basis
                Dloc = Wp @ L @ Lrest_inv @ Wp_inv
                Dbl = _M4 @ Dloc @ _M4i
                Dbl[:3, 3] *= PXM
                basis = Ainv[n] @ Dbl @ A[n]
                mb = Matrix(basis.tolist())
                lo, q, sc = mb.decompose()
                locs[fi] = lo
                quats[fi] = q
                scls[fi] = sc
            # keep quaternion signs continuous
            for fi in range(1, nframes):
                if np.dot(quats[fi], quats[fi - 1]) < 0:
                    quats[fi] = -quats[fi]
            grp = bag.groups.new(n)
            pb_path = f'pose.bones["{n}"]'
            for prop, arr in (("location", locs), ("rotation_quaternion", quats), ("scale", scls)):
                for i in range(arr.shape[1]):
                    vals = arr[:, i]
                    if np.allclose(vals, vals[0], atol=1e-6):
                        kf = np.array([[0, vals[0]]], dtype=np.float32)
                    else:
                        kf = np.stack([frames, vals], axis=1).astype(np.float32)
                    fc = bag.fcurves.new(f"{pb_path}.{prop}", index=i)
                    fc.group = grp
                    fc.keyframe_points.add(len(kf))
                    fc.keyframe_points.foreach_set("co", kf.ravel())
                    fc.keyframe_points.foreach_set("interpolation", [1] * len(kf))  # LINEAR
                    fc.update()
        made.append((act, slot))
    return made



def moving_bones(bones, anim_paths):
    """Bones (and their descendants) that any animation translates or scales."""
    hit = set()
    for p in anim_paths:
        try:
            anims = json.load(open(p))["animations"]
        except (OSError, ValueError, KeyError):
            continue
        for a in anims.values():
            for bn, ch in (a.get("bones") or {}).items():
                if bn in bones and ("position" in ch or "scale" in ch):
                    hit.add(bn)
    kids = {}
    for n, b in bones.items():
        kids.setdefault(b.get("parent"), []).append(n)
    out, stack = set(), list(hit)
    while stack:
        n = stack.pop()
        if n not in out:
            out.add(n)
            stack += kids.get(n, [])
    return out
