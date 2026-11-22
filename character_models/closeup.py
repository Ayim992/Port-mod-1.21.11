"""Close-up renders of chest, crotch and butt: python3 closeup.py Jenny_Nude [smooth]"""
import os, sys, math
import bpy
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from lib.blend import setup_render, render_views
title = sys.argv[1]; smooth = int(sys.argv[2]) if len(sys.argv) > 2 else 1
bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "out", f"{title}.blend"))
sc = bpy.context.scene; rig = bpy.data.objects[f"{title}_rig"]
if rig.animation_data: rig.animation_data.action = None
for pb in rig.pose.bones: pb.matrix_basis.identity()
if "smooth" in rig: rig["smooth"] = smooth
setup_render((500, 500), 24); sc.cycles.transparent_max_bounces = 128
tag = "smooth" if smooth else "blocky"
paths = []
paths += render_views(os.path.join(HERE, "out", "renders", f"close_{title}_{tag}_chest"), [("front", 0, 5), ("q3", 40, 5)], target=(0, 0, 1.32), ortho_scale=0.75)
paths += render_views(os.path.join(HERE, "out", "renders", f"close_{title}_{tag}_crotch"), [("front", 0, 5), ("low", 20, -25)], target=(0, 0, 0.72), ortho_scale=0.55)
paths += render_views(os.path.join(HERE, "out", "renders", f"close_{title}_{tag}_butt"), [("back", 180, 5), ("q3b", 140, 5)], target=(0, 0, 0.85), ortho_scale=0.75)
from PIL import Image
ims = [Image.open(p) for p in paths]; w, h = ims[0].size
s = Image.new("RGB", (w * 2, h * 3)); [s.paste(im, ((i % 2) * w, (i // 2) * h)) for i, im in enumerate(ims)]
s.save(os.path.join(HERE, "out", "renders", f"{title}_closeups_{tag}.png")); print("RESULT ok")
