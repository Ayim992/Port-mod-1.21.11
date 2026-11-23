"""Render the face (head bones only posed) for frames of many actions: python3 facescan.py Jenny_Nude_SlipperyT"""
import os, sys, math
import bpy
from mathutils import Vector
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from lib.blend import setup_render, render_views
title = sys.argv[1]
bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "out", f"{title}.blend"))
sc = bpy.context.scene; rig = bpy.data.objects[f"{title}_rig"]
face = set()
def walk(b):
    face.add(b.name)
    for c in b.children: walk(c)
for n in ("eyes", "mouth", "blush", "heart"):
    walk(rig.data.bones[n])
setup_render((260, 260), 10); sc.cycles.transparent_max_bounces = 128
ad = rig.animation_data
names = [a.name for a in bpy.data.actions if a.name.startswith("jenny.")]
paths = []; labels = []
for nm in names:
    act = bpy.data.actions[nm]
    f0, f1 = act.frame_range
    for f in (f0, (f0 + f1) / 2):
        ad.action = act; ad.action_slot = act.slots[0]
        sc.frame_set(int(f))
        mats = {pb.name: pb.matrix_basis.copy() for pb in rig.pose.bones if pb.name in face}
        ad.action = None
        for pb in rig.pose.bones:
            pb.matrix_basis = mats.get(pb.name, pb.matrix_basis.__class__.Identity(4))
        bpy.context.view_layer.update()
        out = os.path.join(HERE, "out", "renders", "faces", f"{nm}_{int(f):03d}")
        paths += render_views(out, [("f", 0, 0)], target=(0, -0.3, 1.73), ortho_scale=0.62)
        labels.append(f"{nm[6:]}@{int(f)}")
from PIL import Image, ImageDraw
ims = [Image.open(p) for p in paths]; w, h = ims[0].size; cols = 8
rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (w * cols, (h + 16) * rows), "white"); d = ImageDraw.Draw(sheet)
for i, (im, lb) in enumerate(zip(ims, labels)):
    x, y = (i % cols) * w, (i // cols) * (h + 16)
    sheet.paste(im, (x, y + 16)); d.text((x + 3, y + 2), f"{i}:{lb}", fill="black")
sheet.save(os.path.join(HERE, "out", "renders", "face_scan.png")); print("RESULT", len(ims))
