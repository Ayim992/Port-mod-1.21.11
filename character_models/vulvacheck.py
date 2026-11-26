"""Crotch close-ups, original cubes vs smooth: python3 vulvacheck.py out.png"""
import bpy, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from lib.blend import setup_render, render_views
bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "out", "Jenny_Final.blend"))
sc = bpy.context.scene; rig = bpy.data.objects["Jenny_rig"]; rig["outfit"] = 0; sc.frame_set(0)
setup_render((380, 380), 24); sc.cycles.transparent_max_bounces = 128
paths = []
for sm in (0, 1):
    rig["smooth"] = sm; bpy.context.view_layer.update()
    paths += render_views(os.path.join(HERE, "out", "renders", f"vul_{sm}"), [("f", 0, 0), ("low", 0, -35), ("q", 35, -15)],
                          target=(0, -0.1, 0.66), ortho_scale=0.32)
from PIL import Image
ims = [Image.open(p) for p in paths]; s = Image.new("RGB", (380 * 3, 760))
for i, im in enumerate(ims): s.paste(im, ((i % 3) * 380, (i // 3) * 380))
s.save(sys.argv[1]); print("RESULT ok")
