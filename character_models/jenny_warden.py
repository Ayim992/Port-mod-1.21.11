"""Jenny x Warden: reverse-cowgirl ride in the Deep Dark (SlipperyT-style shot).

    python3 jenny_warden.py            # builds out/Jenny_Warden.blend
    python3 jenny_warden.py --sheet    # + contact sheet
    python3 jenny_warden.py --render   # + out/renders/Jenny_Warden.mp4 (letterboxed)

The Warden is built from boxes in Minecraft proportions (scaled so he fits Jenny), lies on his back
on a sculk floor; Jenny kneels astride his hips facing the camera, hands behind her head, and rides.
Legs and arms are IK-solved every frame (knees on his hips, hands behind her head). Breasts and butt
are driven by a damped-spring simulation of her vertical motion, so every drop shakes them.
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib.blend import setup_render  # noqa: E402
import titty_drop as td  # noqa: E402  (shared helpers: apply, solve_arm, hand_tip_rest, subtree, smooth)

NUDE = "Jenny_Nude_SlipperyT"
FPS = 24
END = 119                 # 5 s
PERIOD = 0.62             # seconds per ride stroke
PX = 1.0 / 16.0
K = PX * 0.62             # warden pixel size (scaled to Jenny)


# ------------------------------------------------------------------ mesh helpers
def box_mesh(name, boxes, mat):
    """boxes: list of (center xyz, half xyz, rgb, emit)"""
    unit = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    fidx = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    V, F, C, E = [], [], [], []
    for c, h, rgb, em in boxes:
        b = len(V)
        V += list(unit * np.asarray(h) + np.asarray(c))
        for f in fidx:
            F.append(tuple(i + b for i in f))
            C.append(rgb)
            E.append(em)
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in V], [], F)
    lin = lambda x: x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
    col = me.color_attributes.new("Col", "FLOAT_COLOR", "CORNER")
    emi = me.color_attributes.new("Emit", "FLOAT_COLOR", "CORNER")
    cd, ed = [], []
    for rgb, em in zip(C, E):
        for _ in range(4):
            cd += [lin(rgb[0]), lin(rgb[1]), lin(rgb[2]), 1.0]
            ed += [em, em, em, 1.0]
    col.data.foreach_set("color", cd)
    emi.data.foreach_set("color", ed)
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def attr_material(name, emit_rgb=(0.15, 0.95, 1.0), emit_k=6.0, rough=0.55):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bs = nt.nodes["Principled BSDF"]
    a = nt.nodes.new("ShaderNodeAttribute"); a.attribute_name = "Col"
    e = nt.nodes.new("ShaderNodeAttribute"); e.attribute_name = "Emit"
    nt.links.new(a.outputs["Color"], bs.inputs["Base Color"])
    mul = nt.nodes.new("ShaderNodeMix"); mul.data_type = "RGBA"; mul.blend_type = "MULTIPLY"
    mul.inputs["Factor"].default_value = 1.0
    mul.inputs[6].default_value = (*emit_rgb, 1)
    nt.links.new(e.outputs["Color"], mul.inputs[7])
    nt.links.new(mul.outputs[2], bs.inputs["Emission Color"])
    bs.inputs["Emission Strength"].default_value = emit_k
    bs.inputs["Roughness"].default_value = rough
    # pulsing soul glow
    fc = bs.inputs["Emission Strength"].driver_add("default_value")
    fc.driver.type = "SCRIPTED"
    fc.driver.expression = f"{emit_k} * (0.75 + 0.25 * sin(frame / 3.1))"
    return mat


def empty(name, parent=None, loc=(0, 0, 0)):
    e = bpy.data.objects.new(name, None)
    e.empty_display_size = 0.05
    bpy.context.scene.collection.objects.link(e)
    if parent:
        e.parent = parent
    e.location = loc
    return e


# ------------------------------------------------------------------ warden
TEAL = (0.05, 0.22, 0.27)
TEAL_D = (0.03, 0.14, 0.18)
TEAL_L = (0.09, 0.33, 0.38)
SOUL = (0.35, 0.95, 1.0)
DARK = (0.01, 0.03, 0.05)


def k(*v):
    return tuple(x * K for x in v)


def warden():
    mat = attr_material("Warden_mat")
    rng = np.random.default_rng(7)
    root = empty("Warden_root")
    # ---- body (standing local coords, feet at z=0, front -Y), pivot at hips
    hips = empty("W_hips", root, k(0, 0, 13))
    body = []
    body.append((k(0, 0, 10.5), k(9, 5.5, 10.5), TEAL, 0))
    # skin mottling
    for _ in range(40):
        x, z = rng.uniform(-8.5, 8.5), rng.uniform(0.5, 20.5)
        body.append((k(x, -5.55, z), k(rng.uniform(.5, 1.5), .05, rng.uniform(.5, 1.5)),
                     TEAL_D if rng.random() < .6 else TEAL_L, 0))
    # ribcage: dark ribs over glowing souls on the chest
    body.append((k(0, -5.6, 14), k(6.5, .06, 5), SOUL, 1.0))
    for i, z in enumerate((10.5, 12.5, 14.5, 16.5, 18.5)):
        body.append((k(-3.6, -5.7, z), k(2.6, .12, .45), DARK, 0))
        body.append((k(3.6, -5.7, z), k(2.6, .12, .45), DARK, 0))
    body.append((k(0, -5.7, 14.5), k(.6, .12, 5.2), DARK, 0))
    box_mesh("Warden_Body", body, mat).parent = hips
    # ---- head
    head = empty("W_head", hips, k(0, 0, 21))
    hb = [(k(0, 0, 8), k(8, 5, 8), TEAL, 0),
          (k(0, -5.05, 4), k(5.5, .06, 1.2), DARK, 0),          # mouth slit
          (k(-3, -5.05, 9), k(1.6, .06, 1.6), TEAL_D, 0),
          (k(3, -5.05, 9), k(1.6, .06, 1.6), TEAL_D, 0)]
    box_mesh("Warden_Head", hb, mat).parent = head
    tendrils = []
    for s in (1, -1):
        t = empty(f"W_tendril{'L' if s > 0 else 'R'}", head, k(s * 8, 0, 13))
        tb = [(k(s * 7, 0, 0), k(7, .25, 6), TEAL_L, 0),
              (k(s * 5, -.3, 0), k(3, .05, 3), SOUL, 0.8)]
        box_mesh(f"Warden_Tendril{'L' if s > 0 else 'R'}", tb, mat).parent = t
        tendrils.append(t)
    # ---- arms (pivot at shoulder)
    arms = []
    for s in (1, -1):
        a = empty(f"W_arm{'L' if s > 0 else 'R'}", hips, k(s * 13, 0, 19))
        ab = [(k(0, 0, -12), k(4, 4, 14), TEAL, 0), (k(0, -4.05, -20), k(3, .05, 3), TEAL_D, 0)]
        box_mesh(f"Warden_Arm{'L' if s > 0 else 'R'}", ab, mat).parent = a
        arms.append(a)
    # ---- legs (pivot at hip joint)
    for s in (1, -1):
        lg = empty(f"W_leg{'L' if s > 0 else 'R'}", root, k(s * 5.5, 0, 13))
        box_mesh(f"Warden_Leg{'L' if s > 0 else 'R'}", [(k(0, 0, -6.5), k(3, 3, 6.5), TEAL, 0)], mat).parent = lg
    # ---- shaft: tapered, veined, glowing tip; points along -Y (forward) from the groin
    shaft = empty("W_shaft", hips, k(0, -5.5, 1.5))
    sb = []
    n = 9
    for i in range(n):
        t = i / (n - 1)
        r = 1.55 - 0.35 * t + (0.25 if i == n - 2 else 0)
        sb.append((k(0, -(i + 0.5) * 1.05, 0), k(r, 0.55, r), TEAL_D if i % 3 else TEAL, 0))
        sb.append((k(r * 0.6, -(i + 0.5) * 1.05, r * 0.95), k(.22, .5, .12), SOUL, 0.9))  # vein
    sb.append((k(0, -n * 1.05 - .4, 0), k(1.25, .7, 1.25), (0.12, 0.55, 0.62), 0.35))  # tip
    box_mesh("Warden_Shaft", sb, mat).parent = shaft
    # lie down: up -> +Y (head away from camera), front -> +Z
    root.rotation_euler = (math.radians(-90), 0, 0)
    root.location = (0, -13 * K, 5.5 * K)        # crotch over the world origin, back on the floor
    return root, hips, head, arms, tendrils, shaft


# ------------------------------------------------------------------ set
def pixel_image(name, size, fn):
    img = bpy.data.images.new(name, size, size)
    px = np.zeros((size, size, 4))
    rng = np.random.default_rng(3)
    for y in range(size):
        for x in range(size):
            px[y, x] = (*fn(x, y, rng), 1)
    img.pixels = px.ravel()
    img.pack()
    return img


def textured_plane(name, size, loc, rot, img, tiles, emit=0.0):
    bpy.ops.mesh.primitive_plane_add(size=size, location=loc, rotation=rot)
    ob = bpy.context.object
    ob.name = name
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bs = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (tiles, tiles, 1)
    tx = nt.nodes.new("ShaderNodeTexImage")
    tx.image = img
    tx.interpolation = "Closest"
    nt.links.new(tc.outputs["UV"], mp.inputs[0])
    nt.links.new(mp.outputs[0], tx.inputs[0])
    nt.links.new(tx.outputs["Color"], bs.inputs["Base Color"])
    bs.inputs["Roughness"].default_value = 0.9
    if emit:
        nt.links.new(tx.outputs["Color"], bs.inputs["Emission Color"])
        bs.inputs["Emission Strength"].default_value = emit
    ob.data.materials.append(mat)
    return ob


def deep_dark():
    sc = bpy.context.scene

    def sculk(x, y, rng):
        if rng.random() < 0.035:
            return (0.12, 0.55, 0.65)                     # soul specks
        v = rng.uniform(0.006, 0.03)
        return (v * 0.5, v * 1.2, v * 2.2)

    def deepslate(x, y, rng):
        col = (x // 4) % 3
        v = [0.012, 0.02, 0.016][col] + rng.uniform(-0.004, 0.006)
        if (y + col * 5) % 13 == 0:
            v *= 0.5
        return (v * 0.6, v * 1.1, v * 2.4)

    floor = pixel_image("sculk", 64, sculk)
    wall = pixel_image("deepslate", 64, deepslate)
    textured_plane("SculkFloor", 24, (0, 0, 0), (0, 0, 0), floor, 6, emit=0.4)
    textured_plane("DeepslateWall", 24, (0, 4.5, 6), (math.radians(90), 0, 0), wall, 6, emit=0.25)
    w = sc.world or bpy.data.worlds.new("World")
    sc.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.01, 0.025, 0.06, 1)
    bg.inputs[1].default_value = 0.6
    for name, kind, energy, loc, rot, col, size in (
            ("Key", "AREA", 70, (0.9, -2.6, 2.9), (55, 0, 18), (0.55, 0.75, 1.0), 2.0),
            ("Rim", "AREA", 220, (-1.4, 1.8, 2.2), (-60, 0, -35), (0.25, 0.85, 1.0), 1.5),
            ("Fill", "AREA", 12, (0, -3, 0.6), (85, 0, 0), (0.3, 0.45, 0.9), 3.0)):
        ld = bpy.data.lights.new(name, kind)
        ld.energy = energy
        ld.color = col
        ld.size = size
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        lo.rotation_euler = [math.radians(a) for a in rot]
        sc.collection.objects.link(lo)
    cd = bpy.data.cameras.new("ShotCam")
    cd.lens = 32
    cd.dof.use_dof = True
    cd.dof.aperture_fstop = 2.2
    cam = bpy.data.objects.new("ShotCam", cd)
    sc.collection.objects.link(cam)
    target = Vector((0, 0.1, 1.08))
    cam.location = Vector((0.08, -3.05, 0.78))
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    cd.dof.focus_distance = (target - cam.location).length
    sc.camera = cam


# ------------------------------------------------------------------ jenny
def vulva_point():
    sm = bpy.data.objects[f"{NUDE}_Smooth"]
    gi = {g.name: g.index for g in sm.vertex_groups}
    ids = {gi[n] for n in ("Lside", "Rside") if n in gi}
    co = np.array([v.co[:] for v in sm.data.vertices if v.groups and v.groups[0].group in ids])
    c = co.mean(axis=0)
    return Vector((0.0, c[1] + 0.01, co[:, 2].min() + 0.25 * (co[:, 2].max() - co[:, 2].min())))


def face_from(rig, action, frame):
    sc = bpy.context.scene
    face = td.subtree(rig, "eyes") | td.subtree(rig, "mouth") | td.subtree(rig, "blush")
    ad = rig.animation_data
    act = bpy.data.actions[action]
    ad.action = act
    ad.action_slot = act.slots[0]
    sc.frame_set(frame)
    pose = {pb.name: pb.matrix_basis.copy() for pb in rig.pose.bones if pb.name in face}
    ad.action = None
    return pose


def leg_tip_rest(rig, side):
    body = bpy.data.objects[f"{NUDE}_Body"]
    gi = {g.name: g.index for g in body.vertex_groups}
    idx = {gi[b] for b in td.subtree(rig, f"shin{side}") if b in gi}
    co = [v.co.copy() for v in body.data.vertices if v.groups and v.groups[0].group in idx]
    zmin = min(c.z for c in co)
    low = [c for c in co if c.z < zmin + 0.03]
    return sum(low, Vector()) / len(low)


def solve_leg(rig, side, knee_t, foot_t, guess, foot_tip):
    """thigh (x,y,z) + shin x so the knee pivot and foot reach their targets (rig-local, metres)."""
    from scipy.optimize import minimize
    pt, ps = rig.pose.bones[f"leg{side}"], rig.pose.bones[f"shin{side}"]
    pt.rotation_mode = ps.rotation_mode = "XYZ"
    knee_rest = ps.bone.head_local.copy()

    def err(x):
        pt.rotation_euler = Euler([math.radians(a) for a in x[:3]], "XYZ")
        ps.rotation_euler = Euler((math.radians(x[3]), 0, 0), "XYZ")
        bpy.context.view_layer.update()
        kp = (pt.matrix @ pt.bone.matrix_local.inverted()) @ knee_rest
        fp = (ps.matrix @ ps.bone.matrix_local.inverted()) @ foot_tip
        return (kp - knee_t).length ** 2 + 0.5 * (fp - foot_t).length ** 2 + 1e-7 * float(np.dot(x, x))

    r = minimize(err, guess, method="Nelder-Mead", options={"xatol": .05, "fatol": 1e-11, "maxiter": 900})
    err(r.x)
    return list(r.x), math.sqrt(max(r.fun, 0))


def ride_height(t):
    """vulva height above the shaft base: slow rise, gravity drop, hard stop at the bottom"""
    p = (t / PERIOD) % 1.0
    lo, hi = 0.0, 0.25
    if p < 0.58:
        return lo + (hi - lo) * td.smooth(0, 1, p / 0.58)
    q = (p - 0.58) / 0.42
    return hi - (hi - lo) * q * q


def jiggle(acc, w=2 * math.pi * 3.1, zeta=0.17, dt=1 / FPS, sub=8):
    """damped spring driven by the body's vertical acceleration; returns displacement per frame (m)"""
    x = v = 0.0
    out = []
    h = dt / sub
    for a in acc:
        for _ in range(sub):
            v += (-w * w * x - 2 * zeta * w * v - a) * h
            x += v * h
        out.append(x)
    return np.array(out)


def main():
    args = sys.argv[1:]
    bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "out", f"{NUDE}.blend"))
    sc = bpy.context.scene
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 0, END
    rig = bpy.data.objects[f"{NUDE}_rig"]
    for ob in bpy.data.objects:   # only Jenny's smooth body
        if ob.type == "MESH" and any(s in ob.name for s in ("Steve", "Armor", "Blocky")):
            ob.hide_render = ob.hide_viewport = True
            if ob.animation_data:
                for fc in list(ob.animation_data.drivers):
                    ob.driver_remove(fc.data_path, fc.array_index)
            ob.hide_render = ob.hide_viewport = True
    face = face_from(rig, "jenny.doggyslow", 28)
    for pb in rig.pose.bones:
        pb.matrix_basis = face.get(pb.name, Matrix.Identity(4))

    wroot, whips, whead, warms, wtend, wshaft = warden()
    deep_dark()
    sc.frame_set(0)
    bpy.context.view_layer.update()
    shaft_base = Vector((0, 0, 5.5 * K * 2 + 0.01))   # top of his belly at the groin
    vul = vulva_point()

    # vertical motion and the jiggle it causes
    t = np.arange(END + 1) / FPS
    hgt = np.array([ride_height(x) for x in t])
    acc = np.gradient(np.gradient(hgt, 1 / FPS), 1 / FPS)
    jb = jiggle(acc)
    jb_r = np.concatenate([[0, 0], jb[:-2]])            # right breast lags 2 frames
    jbut = jiggle(acc, w=2 * math.pi * 3.6, zeta=0.22)

    tipL, tipF = td.hand_tip_rest(rig, "L"), leg_tip_rest(rig, "L")
    g_arm, g_leg = [-160.0, -20.0, -100.0], [-60.0, 0.0, 30.0, 100.0]
    errs = []
    for f in range(END + 1):
        h = hgt[f]
        # Jenny's whole rig: vulva over the shaft, height h above its base, slight sway
        rig.location = Vector((0.02 * math.sin(f / 9), 0, 0)) + shaft_base + Vector((0, 0, h)) - vul
        rig.keyframe_insert("location", frame=f)
        sc.frame_set(f)
        P = {}
        p = (t[f] / PERIOD) % 1.0
        lean = -7 + 3 * math.sin(2 * math.pi * p)
        P["upperBody"] = ((0, 0, 0), (lean, 4 * math.sin(f / 11), 0), (1, 1, 1))
        P["head"] = ((0, 0, 0), (-14 + 5 * math.sin(2 * math.pi * p - 1), 0, 8 + 3 * math.sin(f / 13)), (1, 1, 1))
        for side, s, jj in (("L", 1, jb), ("R", -1, jb_r)):
            lift = float(np.clip(jj[f] / PX, -2.4, 2.4))
            sq = 1 - 0.1 * max(0, -lift) / 2 + 0.06 * max(0, lift) / 2
            wsc = 1 / math.sqrt(sq)
            P[f"boob{side}"] = ((0, lift, 0), (-lift * 8, 0, s * 3 * math.sin(f / 5 + s)), (wsc, sq, wsc))
            e = float(np.clip(jbut[f] / PX, -2, 2)) * 0.05
            P[f"cheek{side}"] = ((0, 0, 0), (0, 0, 0), (1 + e, 1 - e, 1 + e))
        td.apply(rig, P)
        bpy.context.view_layer.update()
        # legs: knees on his hips beside the shaft, shins tucked back along his sides
        inv = rig.matrix_world.inverted()
        for side, s in (("L", 1), ("R", -1)):
            knee_w = Vector((s * 0.37, -0.20, 0.40 + 0.55 * h))
            foot_w = Vector((s * 0.46, 0.30, 0.22))
            sol, e = solve_leg(rig, side, inv @ knee_w, inv @ foot_w, g_leg if side == "L" else
                               [g_leg[0], -g_leg[1], -g_leg[2], g_leg[3]], tipF if side == "L"
                               else Vector((-tipF.x, tipF.y, tipF.z)))
            if side == "L":
                g_leg = sol
            errs.append(e)
            P[f"leg{side}"] = ((0, 0, 0), tuple(sol[:3]), (1, 1, 1))
            P[f"shin{side}"] = ((0, 0, 0), (sol[3], 0, 0), (1, 1, 1))
        # hands behind her head (rest-space targets; the arm IK works in the rig's own frame)
        # elbows out wide, forearms folded behind her head (picked from an arm-pose grid)
        sol = [-155 + 4 * math.sin(2 * math.pi * p), -50, -125 + 5 * math.sin(2 * math.pi * p - 1)]
        for side, s in (("L", 1), ("R", -1)):
            P[f"arm{side}"] = ((0, 0, 0), (sol[0], 0, s * sol[1]), (1, 1, 1))
            P[f"lowerArm{side}"] = ((0, 0, 0), (sol[2], 0, 0), (1, 1, 1))
        td.apply(rig, P)
        for name in set(P) | set(face):
            pb = rig.pose.bones.get(name)
            if pb is None:
                continue
            rp = "rotation_euler" if pb.rotation_mode == "XYZ" else "rotation_quaternion"
            for path in ("location", rp, "scale"):
                pb.keyframe_insert(path, frame=f, group=name)
        # warden: hips meet her, head rolls back, tendrils twitch, hands squeeze her thighs
        whips.location = Vector((0, 0, 13 * K)) + Vector((0, 0.03 * (1 - h / 0.25), 0))
        whips.keyframe_insert("location", frame=f)
        whead.rotation_euler = (math.radians(-18 + 6 * math.sin(f / 7)), 0, math.radians(5 * math.sin(f / 17)))
        whead.keyframe_insert("rotation_euler", frame=f)
        for i, tt in enumerate(wtend):
            s = 1 if i == 0 else -1
            tt.rotation_euler = (0, math.radians(s * (8 * math.sin(f / 2.3 + i) + 10)), 0)
            tt.keyframe_insert("rotation_euler", frame=f)
        for i, a in enumerate(warms):
            s = 1 if i == 0 else -1
            a.rotation_euler = (math.radians(-16 - 3 * math.sin(2 * math.pi * p)), 0, math.radians(s * 7))
            a.keyframe_insert("rotation_euler", frame=f)
    print(f"[warden] leg IK max error {max(errs) * 100:.1f} cm", flush=True)

    out_blend = os.path.join(HERE, "out", "Jenny_Warden.blend")
    bpy.ops.wm.save_as_mainfile(filepath=out_blend, compress=True)
    print("[warden] saved", out_blend, flush=True)

    def setup(res, samples):
        setup_render(res, samples)
        sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.01, 0.025, 0.06, 1)
        sc.world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
        sc.cycles.transparent_max_bounces = 128
        sc.view_settings.view_transform = "AgX"
        for n in ("KeyLight", "FillLight"):
            if n in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects[n])

    if "--sheet" in args:
        setup((480, 240), 12)
        frames = [0, 4, 8, 12, 15, 18, 22, 26, 40, 60, 80, 100]
        paths = []
        for f in frames:
            sc.frame_set(f)
            sc.render.filepath = os.path.join(HERE, "out", "renders", "warden", f"f{f:03d}.png")
            bpy.ops.render.render(write_still=True)
            paths.append(sc.render.filepath)
        from PIL import Image
        ims = [Image.open(p) for p in paths]
        w, hh = ims[0].size
        sheet = Image.new("RGB", (w * 4, hh * 3))
        for i, im in enumerate(ims):
            sheet.paste(im, ((i % 4) * w, (i // 4) * hh))
        sheet.save(os.path.join(HERE, "out", "renders", "warden_sheet.png"))
        print("[warden] sheet", flush=True)
    if "--render" in args:
        setup((960, 480), 20)
        fdir = os.path.join(HERE, "out", "renders", "warden_frames")
        os.makedirs(fdir, exist_ok=True)
        for fn in os.listdir(fdir):
            os.remove(os.path.join(fdir, fn))
        sc.render.image_settings.file_format = "PNG"
        sc.render.filepath = os.path.join(fdir, "f")
        bpy.ops.render.render(animation=True)
        mp4 = os.path.join(HERE, "out", "renders", "Jenny_Warden.mp4")
        import subprocess
        # two loops of the 5 s take, letterboxed to 16:9 like the film
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-stream_loop", "1", "-framerate", str(FPS),
                        "-start_number", "0", "-i", os.path.join(fdir, "f%04d.png"),
                        "-vf", "pad=960:540:0:30:black", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "18", mp4], check=True)
        print("[warden] video", mp4, flush=True)


if __name__ == "__main__":
    main()
