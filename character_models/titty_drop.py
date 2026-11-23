"""Short "titty drop" for the SlipperyT Jenny: she grabs the front of her sweater, pulls it up over
her breasts and keeps holding it bunched above her chest while they drop out and bounce.

    python3 titty_drop.py            # builds out/Jenny_TittyDrop.blend
    python3 titty_drop.py --sheet    # + contact sheet of key frames
    python3 titty_drop.py --render   # + out/renders/Jenny_TittyDrop.mp4

How the shirt works: the dressed and nude SlipperyT models share one animation. Before the drop
the dressed model is shown. From the drop on, the scene shows the dressed model above the chest
(sweater shoulders, sleeves, collar) and below the waist (pants, shoes), the nude model in
between (bare chest and breasts), and a rolled-up sweater band around the upper chest that her
hands hold. Breasts are driven per frame by a damped spring (big drop, overshoot, decaying
bounces, left/right slightly out of phase, squash/stretch, sway).
"""
import math
import os
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Euler, Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib.blend import setup_render  # noqa: E402

PX = 1.0 / 16.0
FPS = 24
END = 132
GRAB0, GRAB1 = 14, 26          # hands go to the hem
PULL1 = 40                     # shirt fully up
SWAP = 40                      # breasts released
SHIMMY0, SHIMMY1 = 84, 108
BAND_Z = 1.415                 # sweater rolled up to here (m, rest)
WAIST_Z = 0.96                 # pants start below this

NUDE = "Jenny_Nude_SlipperyT"
DRESSED = "Jenny_Dressed_SlipperyT"


# ------------------------------------------------------------------ scene
def load():
    bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "out", f"{NUDE}.blend"))
    path = os.path.join(HERE, "out", f"{DRESSED}.blend")
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.collections = [DRESSED]
    bpy.context.scene.collection.children.link(dst.collections[0])
    sc = bpy.context.scene
    sc.render.fps = FPS
    sc.frame_start, sc.frame_end = 0, END
    return bpy.data.objects[f"{NUDE}_rig"], bpy.data.objects[f"{DRESSED}_rig"]


def subtree(rig, name):
    out, stack = set(), [name]
    while stack:
        n = stack.pop()
        b = rig.data.bones.get(n)
        if b and n not in out:
            out.add(n)
            stack += [c.name for c in b.children]
    return out


def split(ob, keep_fn, new_name):
    """Copy `ob` keeping only faces where keep_fn(center_z, bone_name) is True."""
    me = ob.data.copy()
    cp = ob.copy()
    cp.data = me
    cp.name = new_name
    if cp.animation_data:
        for fc in list(cp.animation_data.drivers):
            cp.driver_remove(fc.data_path, fc.array_index)
    for c in ob.users_collection:
        c.objects.link(cp)
    gname = {g.index: g.name for g in cp.vertex_groups}
    bm = bmesh.new()
    bm.from_mesh(me)
    dl = bm.verts.layers.deform.active
    kill = []
    for f in bm.faces:
        v0 = f.verts[0]
        bone = ""
        if dl is not None and len(v0[dl]):
            bone = gname.get(max(v0[dl].items(), key=lambda kv: kv[1])[0], "")
        if not keep_fn(f.calc_center_median().z, bone):
            kill.append(f)
    bmesh.ops.delete(bm, geom=kill, context="FACES")
    bm.to_mesh(me)
    bm.free()
    return cp


def band_mesh(name, rig, ref_objs):
    """Rolled-up sweater: a lumpy knit tube hugging the chest just above the breasts."""
    pts = []
    for ob in ref_objs:
        co = np.array([v.co[:] for v in ob.data.vertices])
        if len(co):
            pts.append(co[np.abs(co[:, 2] - BAND_Z) < 0.03])
    pts = np.concatenate(pts)
    cx, cy = 0.0, (pts[:, 1].min() + pts[:, 1].max()) / 2
    rx = np.abs(pts[:, 0]).max() + 0.012
    ry_f = cy - pts[:, 1].min() + 0.02
    ry_b = pts[:, 1].max() - cy + 0.012
    nu, nv, tube = 96, 12, 0.026
    rng = np.random.default_rng(4)
    bump = rng.normal(0, 1, 24)
    verts, faces = [], []
    for i in range(nu):
        a = 2 * math.pi * i / nu
        lump = 1 + 0.25 * sum(bump[k] * math.sin((k + 2) * a + bump[k + 12]) for k in range(6)) / 6
        ry = ry_f if math.sin(a) < 0 else ry_b
        c = Vector((cx + rx * math.cos(a), cy + ry * math.sin(a), BAND_Z + 0.012 * math.sin(3 * a)))
        nrm = Vector((math.cos(a) / rx, math.sin(a) / ry, 0)).normalized()
        for j in range(nv):
            b = 2 * math.pi * j / nv
            off = nrm * math.cos(b) * tube * lump + Vector((0, 0, 1)) * math.sin(b) * tube * 1.25 * lump
            verts.append(c + off)
    for i in range(nu):
        for j in range(nv):
            a, b = i * nv + j, i * nv + (j + 1) % nv
            c2, d = ((i + 1) % nu) * nv + (j + 1) % nv, ((i + 1) % nu) * nv + j
            faces.append((a, d, c2, b))
    me = bpy.data.meshes.new(name)
    me.from_pydata([v[:] for v in verts], [], faces)
    me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    mat = bpy.data.materials.new(name + "_mat")
    mat.use_nodes = True
    nt = mat.node_tree
    bs = nt.nodes["Principled BSDF"]
    bs.inputs["Roughness"].default_value = 0.85
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "Z"
    wave.inputs["Scale"].default_value = 30
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs[6].default_value = (0.04, 0.028, 0.06, 1)
    mix.inputs[7].default_value = (0.07, 0.052, 0.10, 1)
    nt.links.new(wave.outputs["Fac"], mix.inputs["Factor"])
    nt.links.new(mix.outputs[2], bs.inputs["Base Color"])
    me.materials.append(mat)
    vg = ob.vertex_groups.new(name="torso")
    vg.add(list(range(len(verts))), 1.0, "REPLACE")
    m = ob.modifiers.new("Armature", "ARMATURE")
    m.object = rig
    ob.parent = rig
    return ob


def set_vis(ob, frames_visible):
    if ob.animation_data:
        for fc in list(ob.animation_data.drivers):
            if fc.data_path in ("hide_render", "hide_viewport"):
                ob.driver_remove(fc.data_path)
    for f, vis in frames_visible:
        ob.hide_render = ob.hide_viewport = not vis
        ob.keyframe_insert("hide_render", frame=f)
        ob.keyframe_insert("hide_viewport", frame=f)


# ------------------------------------------------------------------ motion helpers
def smooth(a, b, t):
    t = min(max(t, 0.0), 1.0)
    t = t * t * (3 - 2 * t)
    return a + (b - a) * t


def keys(f, pts):
    if f <= pts[0][0]:
        return pts[0][1]
    for (f0, v0), (f1, v1) in zip(pts, pts[1:]):
        if f <= f1:
            if isinstance(v0, (tuple, list)):
                return tuple(smooth(a, b, (f - f0) / (f1 - f0)) for a, b in zip(v0, v1))
            return smooth(v0, v1, (f - f0) / (f1 - f0))
    return pts[-1][1]


def spring(t, amp, freq, decay, phase=0.0):
    if t < 0:
        return 0.0
    return amp * math.exp(-decay * t) * math.cos(2 * math.pi * freq * t + phase)


# left-hand targets (right mirrored), metres in rest space
HAND_PATH = [(GRAB1, (0.135, -0.33, 1.03)), (31, (0.16, -0.43, 1.18)),
             (PULL1, (0.13, -0.27, BAND_Z + 0.015)), (END, (0.13, -0.27, BAND_Z + 0.015))]


def body_pose(f):
    P = {}
    sh = 0.0
    if SHIMMY0 <= f <= SHIMMY1:
        k = (f - SHIMMY0) / (SHIMMY1 - SHIMMY0)
        sh = math.sin(k * 2 * math.pi * 3) * math.sin(k * math.pi)
    lean = keys(f, [(0, 0), (GRAB1, 5), (PULL1, -3), (SWAP + 6, -8), (SWAP + 22, -4), (END, -4)])
    P["upperBody"] = ((0, 0, 0), (lean, sh * 8, 0), (1, 1, 1))
    P["head"] = ((0, 0, 0), (keys(f, [(0, 0), (GRAB1, 12), (PULL1, -2), (SWAP + 10, 8), (END, 6)]),
                            -sh * 6, keys(f, [(0, 0), (SWAP + 8, 5), (END, 4)])), (1, 1, 1))
    return P, sh


def breast_pose(f):
    P = {}
    t_rel = (f - SWAP) / FPS
    for side, s, ph in (("L", 1, 0.0), ("R", -1, 0.55)):
        if f <= SWAP:
            lift = keys(f, [(0, 0), (31, 0.1), (PULL1, 1.3)])  # the hem drags them up
        else:
            lift = spring(t_rel, 1.3, 2.9, 3.3, phase=ph * 0.35)
        sway = spring(t_rel, 4.0, 2.2, 2.6, ph + 1.2) if f > SWAP else 0.0
        if SHIMMY0 <= f <= SHIMMY1 + 30:
            tt = (f - SHIMMY0) / FPS
            fade = math.exp(-max(0.0, f - SHIMMY1) / 10)
            lift += 0.22 * math.sin(tt * 2 * math.pi * 6.0 - ph) * fade
            sway += s * 9.0 * math.sin(tt * 2 * math.pi * 3.0 - 0.6 - ph) * fade
        squash = 1.0 - 0.12 * max(0.0, -lift) / 1.3 + 0.07 * max(0.0, lift) / 1.3
        pitch = -lift * 9.0 + (spring(t_rel - 0.04, 6.0, 2.9, 3.0, ph) if f > SWAP else 0.0)
        w = 1.0 / math.sqrt(max(squash, 0.5))
        P[f"boob{side}"] = ((0, lift, 0), (pitch, 0, sway * s), (w, squash, w))
    return P


def apply(rig, P):
    for name, (loc, rot, scl) in P.items():
        pb = rig.pose.bones.get(name)
        if pb is None:
            continue
        pb.rotation_mode = "XYZ"
        pb.location = Vector(loc) * PX
        pb.rotation_euler = Euler([math.radians(a) for a in rot], "XYZ")
        pb.scale = scl


def hand_tip_rest(rig, side):
    """lowest point of the arm box at rest (the hand)"""
    body = bpy.data.objects[f"{NUDE}_Body"]
    gi = {g.name: g.index for g in body.vertex_groups}
    idx = {gi[b] for b in subtree(rig, f"lowerArm{side}") if b in gi}
    co = [v.co.copy() for v in body.data.vertices if v.groups and v.groups[0].group in idx]
    zmin = min(c.z for c in co)
    low = [c for c in co if c.z < zmin + 0.02]
    return sum(low, Vector()) / len(low)


def solve_arm(rig, side, target, guess, tip):
    from scipy.optimize import minimize
    pa, pl = rig.pose.bones[f"arm{side}"], rig.pose.bones[f"lowerArm{side}"]
    pa.rotation_mode = pl.rotation_mode = "XYZ"
    s = 1 if side == "L" else -1
    t = Vector((s * target[0], target[1], target[2]))

    def err(x):
        pa.rotation_euler = Euler((math.radians(x[0]), 0, math.radians(s * x[1])), "XYZ")
        pl.rotation_euler = Euler((math.radians(x[2]), 0, 0), "XYZ")
        bpy.context.view_layer.update()
        p = (pl.matrix @ pl.bone.matrix_local.inverted()) @ tip
        return (p - t).length ** 2 + 1e-7 * (x[1] ** 2)

    r = minimize(err, guess, method="Nelder-Mead", options={"xatol": 0.05, "fatol": 1e-10, "maxiter": 500})
    err(r.x)
    return list(r.x), math.sqrt(max(r.fun, 0.0))


def bake(nude, dressed):
    sc = bpy.context.scene
    sc.frame_set(0)
    face = {pb.name: pb.matrix_basis.copy() for pb in nude.pose.bones
            if pb.matrix_basis != Matrix.Identity(4)}
    for rig in (nude, dressed):
        ad = rig.animation_data_create()
        ad.action = None
        for pb in rig.pose.bones:
            pb.matrix_basis = face.get(pb.name, Matrix.Identity(4))
    tip = hand_tip_rest(nude, "L")
    # hand solutions along the path (ease from rest into the first grab)
    sol_at = {}
    guess = [-30.0, -10.0, -70.0]
    errs = []
    for f in range(GRAB1, END + 1):
        tgt = keys(f, HAND_PATH)
        sol, e = solve_arm(nude, "L", tgt, guess, tip)
        guess = sol
        sol_at[f] = sol
        errs.append(e)
    print(f"[drop] hand IK max error {max(errs) * 100:.1f} cm", flush=True)
    for f in range(0, GRAB1):
        w = smooth(0, 1, (f - GRAB0) / (GRAB1 - GRAB0))
        sol_at[f] = [a * w for a in sol_at[GRAB1]]
    for name in ("armL", "armR", "lowerArmL", "lowerArmR"):
        nude.pose.bones[name].matrix_basis = Matrix.Identity(4)
    last = None
    for f in range(0, END + 1):
        P, sh = body_pose(f)
        P.update(breast_pose(f))
        a = sol_at[f]
        for side, s in (("L", 1), ("R", -1)):
            P[f"arm{side}"] = ((0, 0, 0), (a[0], 0, s * a[1]), (1, 1, 1))
            P[f"lowerArm{side}"] = ((0, 0, 0), (a[2], 0, 0), (1, 1, 1))
        apply(nude, P)
        last = P
        for name in list(P) + list(face):
            pb = nude.pose.bones.get(name)
            if pb is None:
                continue
            rp = "rotation_euler" if pb.rotation_mode == "XYZ" else "rotation_quaternion"
            for path in ("location", rp, "scale"):
                pb.keyframe_insert(path, frame=f, group=name)
    act = nude.animation_data.action
    act.name = "jenny.titty_drop"
    act_d = act.copy()
    act_d.name = "jenny.titty_drop.dressed"
    dressed.animation_data.action = act_d
    dressed.animation_data.action_slot = act_d.slots[0]
    for name in last:
        if name in dressed.pose.bones:
            dressed.pose.bones[name].rotation_mode = "XYZ"
    return act


def build_shirt_swap(nude, dressed):
    arm = subtree(nude, "armL") | subtree(nude, "armR")
    nb = bpy.data.objects[f"{NUDE}_Body"]
    db = bpy.data.objects[f"{DRESSED}_Body"]
    ns = bpy.data.objects[f"{NUDE}_Smooth"]
    ds = bpy.data.objects[f"{DRESSED}_Smooth"]
    boobs = subtree(nude, "boobs") | subtree(dressed, "boobs")
    keep_on = split(db, lambda z, b: z > BAND_Z or z < WAIST_Z or b in arm, f"{DRESSED}_KeepOn")
    chest = split(nb, lambda z, b: WAIST_Z <= z <= BAND_Z and b not in arm, f"{NUDE}_Chest")
    n_boobs = split(ns, lambda z, b: b in boobs, f"{NUDE}_Breasts")
    d_rest = split(ds, lambda z, b: b not in boobs, f"{DRESSED}_SmoothRest")
    band = band_mesh("RolledSweater", nude, [chest, n_boobs])
    for ob in list(bpy.data.objects):
        if ob.type != "MESH":
            continue
        n = ob.name
        if n in (db.name, ds.name):
            set_vis(ob, [(0, True), (SWAP, False)])
        elif n in (keep_on.name, chest.name, n_boobs.name, band.name, d_rest.name):
            set_vis(ob, [(0, False), (SWAP, True)])
        else:  # nude originals, blocky parts, armor slots, partner
            set_vis(ob, [(0, False)])


def camera():
    sc = bpy.context.scene
    cd = bpy.data.cameras.new("ShotCam")
    cd.lens = 55
    cam = bpy.data.objects.new("ShotCam", cd)
    sc.collection.objects.link(cam)
    target = Vector((0, -0.05, 1.22))
    cam.location = Vector((0.65, -2.3, 1.42))
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.camera = cam
    for name, energy, rot, col in (("Key", 3.0, (55, 0, -35), (1.0, 0.86, 0.72)),
                                   ("Rim", 2.0, (65, 0, 150), (1.0, 0.62, 0.45))):
        ld = bpy.data.lights.new(name, "SUN")
        ld.energy = energy
        ld.color = col
        ld.angle = math.radians(8)
        lo = bpy.data.objects.new(name, ld)
        lo.rotation_euler = [math.radians(a) for a in rot]
        sc.collection.objects.link(lo)


def world():
    sc = bpy.context.scene
    w = sc.world or bpy.data.worlds.new("World")
    sc.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.42, 0.32, 0.45, 1)
    w.node_tree.nodes["Background"].inputs[1].default_value = 0.8
    sc.cycles.transparent_max_bounces = 128


def main():
    args = sys.argv[1:]
    nude, dressed = load()
    bake(nude, dressed)
    build_shirt_swap(nude, dressed)
    camera()
    sc = bpy.context.scene
    out_blend = os.path.join(HERE, "out", "Jenny_TittyDrop.blend")
    bpy.ops.wm.save_as_mainfile(filepath=out_blend, compress=True)
    print("[drop] saved", out_blend, flush=True)
    if "--sheet" in args:
        setup_render((360, 360), 10)
        world()
        frames = [0, GRAB0 + 6, GRAB1, 31, 36, PULL1 - 1, SWAP + 2, SWAP + 4, SWAP + 7, SWAP + 12, 70, 96]
        paths = []
        for f in frames:
            sc.frame_set(f)
            sc.render.filepath = os.path.join(HERE, "out", "renders", "drop", f"f{f:03d}.png")
            bpy.ops.render.render(write_still=True)
            paths.append(sc.render.filepath)
        from PIL import Image, ImageDraw
        ims = [Image.open(p) for p in paths]
        w, h = ims[0].size
        sheet = Image.new("RGB", (w * 6, h * 2))
        d = ImageDraw.Draw(sheet)
        for i, (im, fr) in enumerate(zip(ims, frames)):
            sheet.paste(im, ((i % 6) * w, (i // 6) * h))
            d.text(((i % 6) * w + 4, (i // 6) * h + 4), f"f{fr}", fill="white")
        sheet.save(os.path.join(HERE, "out", "renders", "drop_sheet.png"))
        print("[drop] sheet", flush=True)
    if "--render" in args:
        setup_render((640, 640), 16)
        world()
        fdir = os.path.join(HERE, "out", "renders", "drop_frames")
        os.makedirs(fdir, exist_ok=True)
        for fn in os.listdir(fdir):
            os.remove(os.path.join(fdir, fn))
        sc.render.image_settings.file_format = "PNG"
        sc.render.filepath = os.path.join(fdir, "f")
        bpy.ops.render.render(animation=True)
        mp4 = os.path.join(HERE, "out", "renders", "Jenny_TittyDrop.mp4")
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-start_number", "0",
                        "-i", os.path.join(fdir, "f%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "18", mp4], check=True)
        print("[drop] video", mp4, flush=True)


if __name__ == "__main__":
    main()
