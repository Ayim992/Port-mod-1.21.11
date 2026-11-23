"""Nipple and chest close-ups: python3 nipcheck.py Jenny_Nude_SlipperyT"""
import bpy, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.blend import setup_render, render_views
t = sys.argv[1]
bpy.ops.wm.open_mainfile(filepath=f"out/{t}.blend")
sc = bpy.context.scene; rig = bpy.data.objects[f"{t}_rig"]
if rig.animation_data: rig.animation_data.action = None
for pb in rig.pose.bones: pb.matrix_basis.identity()
setup_render((500, 500), 32); sc.cycles.transparent_max_bounces = 128
render_views("out/renders/nip_new", [("f", 0, 0), ("q", 35, 0), ("s", 80, 0)], target=(0.08, -0.3, 1.27), ortho_scale=0.3)
render_views("out/renders/chest_new", [("f", 0, 5), ("q", 35, 5)], target=(0, -0.2, 1.3), ortho_scale=0.75)
from PIL import Image
ims = [Image.open(f"out/renders/nip_new_{v}.png") for v in ("f", "q", "s")] + [Image.open(f"out/renders/chest_new_{v}.png") for v in ("f", "q")]
s = Image.new("RGB", (1500, 1000), "white"); [s.paste(im, ((i % 3) * 500, (i // 3) * 500)) for i, im in enumerate(ims)]
s.save("out/renders/nip_new.png"); print("RESULT ok")
