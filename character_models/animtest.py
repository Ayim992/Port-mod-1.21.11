"""Render frames of baked actions to verify them.

    python3 animtest.py Jenny_Nude doggyslow,idle,walk 3        # 3 frames per action
"""
import math
import os
import sys

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib.blend import setup_render, render_views  # noqa: E402

title, names, n = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3])
bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "out", f"{title}.blend"))
sc = bpy.context.scene
rig = bpy.data.objects[f"{title}_rig"]
setup_render((420, 420), 12)
sc.cycles.transparent_max_bounces = 128
paths = []
for nm in names:
    act = next(a for a in bpy.data.actions if a.name == nm)
    ad = rig.animation_data
    ad.action = act
    ad.action_slot = act.slots[0]
    scene_anim = any(k in nm for k in ("doggy", "blowjob", "paizuri", "missionary", "cowgirl", "anal"))
    rig["steve"] = 1 if scene_anim else 0
    f0, f1 = act.frame_range
    for i in range(n):
        f = int(f0 + (f1 - f0) * i / max(n - 1, 1))
        sc.frame_set(f)
        out = os.path.join(HERE, "out", "renders", f"anim_{title}_{nm}_{f:03d}")
        paths += render_views(out, [("q3", 40, 12)], target=(0, -0.3, 0.9), ortho_scale=3.2)
from PIL import Image  # noqa: E402
ims = [Image.open(p) for p in paths]
w, h = ims[0].size
cols = n
sheet = Image.new("RGB", (w * cols, h * len(names)))
for i, im in enumerate(ims):
    sheet.paste(im, ((i % cols) * w, (i // cols) * h))
sheet.save(os.path.join(HERE, "out", "renders", f"{title}_anim_sheet.png"))
print("RESULT ok")
