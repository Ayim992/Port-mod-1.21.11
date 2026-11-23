"""Short "titty drop" animation for the SlipperyT Jenny.

    python3 titty_drop.py            # builds out/Jenny_TittyDrop.blend
    python3 titty_drop.py --sheet    # + a contact sheet of key frames
    python3 titty_drop.py --render   # + renders out/renders/Jenny_TittyDrop.mp4

Dressed and nude SlipperyT Jenny share one action; the dressed one is shown until the sweater
clears her chest, then the nude one takes over and the breasts drop and bounce.

Jiggle is a damped spring per breast (SlipperyT-style: big first drop, overshoot, quick
decaying bounces, left/right slightly out of phase, squash on the way down, stretch on the way
up, a little sideways sway), keyed every frame so the motion is exact.
"""
import math
import os
import sys

import bpy
from mathutils import Euler, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib.blend import setup_render  # noqa: E402

PX = 1.0 / 16.0
FPS = 24
END = 132
LIFT0, LIFT1, SWAP = 22, 36, 37       # grab -> sweater up -> breasts released
SHIMMY0, SHIMMY1 = 78, 104

NUDE = "Jenny_Nude_SlipperyT"
DRESSED = "Jenny_Dressed_SlipperyT"


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


# ------------------------------------------------------------------ curves
def smooth(a, b, t):
    t = min(max(t, 0.0), 1.0)
    t = t * t * (3 - 2 * t)
    return a + (b - a) * t


def keys(f, pts):
    """piecewise smooth interpolation through (frame, value) points"""
    if f <= pts[0][0]:
        return pts[0][1]
    for (f0, v0), (f1, v1) in zip(pts, pts[1:]):
        if f <= f1:
            return smooth(v0, v1, (f - f0) / (f1 - f0))
    return pts[-1][1]


def spring(t, amp, freq, decay, phase=0.0):
    """damped oscillation, t in seconds since release"""
    if t < 0:
        return 0.0
    return amp * math.exp(-decay * t) * math.cos(2 * math.pi * freq * t + phase)


# ------------------------------------------------------------------ pose per frame
def pose_at(f):
    """dict bone -> (loc(px, bone space), rot euler(deg), scale)"""
    P = {}
    t_rel = (f - SWAP) / FPS

    # arms: hang -> grab hem -> rip sweater overhead -> drop to hips / behind head
    swing = keys(f, [(0, 0), (12, -8), (LIFT0, -38), (LIFT1, -172), (SWAP + 6, -178),
                     (SWAP + 16, -150), (SWAP + 26, -150), (62, -165), (END, -165)])
    inward = keys(f, [(0, 0), (LIFT0, -14), (LIFT1, -6), (SWAP + 16, 18), (END, 18)])
    elbow = keys(f, [(0, 0), (LIFT0, -25), (LIFT1, -10), (SWAP + 16, -95), (62, -110), (END, -110)])
    for side, s in (("L", 1), ("R", -1)):
        P[f"arm{side}"] = ((0, 0, 0), (swing, 0, s * inward), (1, 1, 1))
        P[f"lowerArm{side}"] = ((0, 0, 0), (elbow, 0, 0), (1, 1, 1))

    # body: lean into the pull, arch back on release, shimmy later
    sh = 0.0
    if SHIMMY0 <= f <= SHIMMY1:
        k = (f - SHIMMY0) / (SHIMMY1 - SHIMMY0)
        sh = math.sin(k * 2 * math.pi * 3) * math.sin(k * math.pi)
    lean = keys(f, [(0, 0), (LIFT0, 6), (LIFT1, -4), (SWAP + 6, -9), (SWAP + 20, -3), (END, -3)])
    P["upperBody"] = ((0, 0, 0), (lean, sh * 7, 0), (1, 1, 1))
    P["head"] = ((0, 0, 0), (keys(f, [(0, 0), (LIFT1, -10), (SWAP + 12, 6), (END, 4)]),
                            -sh * 5, keys(f, [(0, 0), (SWAP + 10, 6), (END, 5)])), (1, 1, 1))

    # breasts: dragged up by the sweater, released at SWAP, then spring
    for side, s, ph in (("L", 1, 0.0), ("R", -1, 0.55)):
        lift = keys(f, [(0, 0), (LIFT0 + 4, 0), (LIFT1, 1.25), (SWAP, 1.35)]) if f <= SWAP else 0.0
        if f > SWAP:
            # first big drop + bounces (freq ~3 Hz like SlipperyT's heavy wobble)
            lift = spring(t_rel, 1.35, 2.9, 3.4, phase=ph * 0.35)
        squash = 1.0 - 0.12 * max(0.0, -lift) / 1.35 + 0.08 * max(0.0, lift) / 1.35
        pitch = -lift * 9.0 + (spring(t_rel - 0.04, 6.0, 2.9, 3.0, ph) if f > SWAP else 0.0)
        sway = spring(t_rel, 4.0, 2.2, 2.6, ph + 1.2) if f > SWAP else 0.0
        if SHIMMY0 <= f <= SHIMMY1 + 30:
            tt = (f - SHIMMY0) / FPS
            sway += s * 9.0 * math.sin(tt * 2 * math.pi * 3.0 - 0.6 - ph) * math.exp(-max(0.0, f - SHIMMY1) / 10)
            lift += 0.25 * math.sin(tt * 2 * math.pi * 6.0 - ph) * math.exp(-max(0.0, f - SHIMMY1) / 10)
        w = 1.0 / math.sqrt(max(squash, 0.5))
        P[f"boob{side}"] = ((0, lift, 0), (pitch, 0, sway * s), (w, squash, w))
    return P


def apply(rig, f):
    for name, (loc, rot, scl) in pose_at(f).items():
        pb = rig.pose.bones.get(name)
        if pb is None:
            continue
        pb.rotation_mode = "XYZ"
        pb.location = Vector(loc) * PX
        pb.rotation_euler = Euler([math.radians(a) for a in rot], "XYZ")
        pb.scale = scl


def bake(nude, dressed):
    sc = bpy.context.scene
    face_src = nude.animation_data.action if nude.animation_data else None
    face_pose = {}
    if face_src is not None:
        sc.frame_set(0)
        for pb in nude.pose.bones:
            if pb.matrix_basis != pb.matrix_basis.__class__.Identity(4):
                face_pose[pb.name] = pb.matrix_basis.copy()
    act = bpy.data.actions.new("jenny.titty_drop")
    act.use_fake_user = True
    for rig in (nude, dressed):
        ad = rig.animation_data_create()
        ad.action = None
        for pb in rig.pose.bones:
            pb.matrix_basis = face_pose.get(pb.name, pb.matrix_basis.__class__.Identity(4))
    touched = set(pose_at(0).keys()) | set(face_pose.keys())
    for f in range(0, END + 1):
        apply(nude, f)
        for name in touched:
            pb = nude.pose.bones.get(name)
            if pb is None:
                continue
            pb.rotation_mode = "XYZ" if name in pose_at(0) else pb.rotation_mode
            for path in ("location", "rotation_euler" if pb.rotation_mode == "XYZ" else "rotation_quaternion",
                         "scale"):
                pb.keyframe_insert(path, frame=f, group=name)
        nude.animation_data.action = nude.animation_data.action or act
    act = nude.animation_data.action
    act.name = "jenny.titty_drop"
    # the dressed rig gets its own copy (an action slot is tied to the object it was made for)
    act_d = act.copy()
    act_d.name = "jenny.titty_drop.dressed"
    dressed.animation_data.action = act_d
    dressed.animation_data.action_slot = act_d.slots[0]
    for name in pose_at(0):  # the keys are eulers on these bones
        if name in dressed.pose.bones:
            dressed.pose.bones[name].rotation_mode = "XYZ"
    # swap dressed -> nude when the sweater clears her chest
    for ob in bpy.data.objects:
        if ob.type != "MESH" or "Armor" in ob.name or "Steve" in ob.name or "Blocky" in ob.name:
            continue
        is_dressed = ob.name.startswith(DRESSED)
        if ob.animation_data:  # the smooth/blocky drivers would override the swap keys
            for path in ("hide_render", "hide_viewport"):
                ob.driver_remove(path)
        for f, vis in ((0, is_dressed), (SWAP, not is_dressed)):
            ob.hide_render = not vis
            ob.hide_viewport = not vis
            ob.keyframe_insert("hide_render", frame=f)
            ob.keyframe_insert("hide_viewport", frame=f)
    return act


def camera():
    sc = bpy.context.scene
    cd = bpy.data.cameras.new("ShotCam")
    cd.lens = 50
    cam = bpy.data.objects.new("ShotCam", cd)
    sc.collection.objects.link(cam)
    target = Vector((0, 0, 1.25))
    cam.location = Vector((0.75, -2.35, 1.45))
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.camera = cam
    # soft key + rim like SlipperyT's sunset shots
    for name, energy, rot, col in (("Key", 3.0, (55, 0, -35), (1.0, 0.86, 0.72)),
                                   ("Rim", 2.0, (65, 0, 150), (1.0, 0.62, 0.45))):
        ld = bpy.data.lights.new(name, "SUN")
        ld.energy = energy
        ld.color = col
        ld.angle = math.radians(8)
        lo = bpy.data.objects.new(name, ld)
        lo.rotation_euler = [math.radians(a) for a in rot]
        sc.collection.objects.link(lo)
    world = sc.world or bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.42, 0.32, 0.45, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.8
    return cam


def main():
    args = sys.argv[1:]
    nude, dressed = load()
    bake(nude, dressed)
    camera()
    sc = bpy.context.scene
    out_blend = os.path.join(HERE, "out", "Jenny_TittyDrop.blend")
    bpy.ops.wm.save_as_mainfile(filepath=out_blend, compress=True)
    print("[drop] saved", out_blend, flush=True)
    if "--sheet" in args:
        setup_render((360, 360), 10)
        sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.42, 0.32, 0.45, 1)
        sc.cycles.transparent_max_bounces = 128
        frames = [0, 14, LIFT0, 30, LIFT1, SWAP + 2, SWAP + 4, SWAP + 6, SWAP + 9, SWAP + 13, 70, 88]
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
        sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.42, 0.32, 0.45, 1)
        sc.cycles.transparent_max_bounces = 128
        fdir = os.path.join(HERE, "out", "renders", "drop_frames")
        os.makedirs(fdir, exist_ok=True)
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
