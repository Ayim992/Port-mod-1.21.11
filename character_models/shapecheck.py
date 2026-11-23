"""Overlay the smooth breasts' outline (red) on the original blocky breasts, front and side.
    python3 shapecheck.py Jenny_Nude_SlipperyT
"""
import os, sys
import numpy as np
import bpy
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from lib.blend import setup_render, render_views
title = sys.argv[1]
bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "out", f"{title}.blend"))
sc = bpy.context.scene; rig = bpy.data.objects[f"{title}_rig"]
if rig.animation_data: rig.animation_data.action = None
for pb in rig.pose.bones: pb.matrix_basis.identity()
setup_render((500, 500), 16); sc.cycles.transparent_max_bounces = 128
# only body + breasts: hide arms by rendering the chest region tight; mask via holdout of everything else is overkill
views = [("front", 0, 0), ("side", 90, 0), ("q3", 40, 0)]
imgs = {}
for sm in (0, 1):
    rig["smooth"] = sm
    bpy.context.view_layer.update()
    p = render_views(os.path.join(HERE, "out", "renders", f"shape_{title}_{sm}"), views, target=(0, -0.12, 1.28), ortho_scale=0.8)
    imgs[sm] = [np.array(Image.open(x).convert("RGB")).astype(int) for x in p]
row = []
for b, s in zip(imgs[0], imgs[1]):
    bg = s[2, 2]
    diff = np.abs(s - b).sum(-1) > 40          # where the smooth version differs from blocky
    out = b.copy()
    out[diff] = (out[diff] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(int)
    row.append(np.concatenate([b, s, out], axis=0))
sheet = np.concatenate(row, axis=1).astype(np.uint8)
Image.fromarray(sheet).save(os.path.join(HERE, "out", "renders", f"{title}_shapecheck.png"))
print("RESULT ok")
