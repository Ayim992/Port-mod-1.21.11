"""Butt close-ups across animations: python3 buttanim.py Jenny_Nude_SlipperyT out.png"""
import bpy, sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from lib.blend import setup_render, render_views
title, outp = sys.argv[1], sys.argv[2]
bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "out", f"{title}.blend"))
sc = bpy.context.scene; rig = bpy.data.objects[f"{title}_rig"]; sm = bpy.data.objects[f"{title}_Smooth"]
rig["smooth"] = int(os.environ.get("SMOOTH", "1"))
setup_render((340, 340), 12); sc.cycles.transparent_max_bounces = 128
gi = [g.index for g in sm.vertex_groups if g.name.startswith(("innerCheek", "midSection"))]
ids = [v.index for v in sm.data.vertices if any(g.group in gi and g.weight > 0.5 for g in v.groups)]
paths = []
for act, f in (("jenny.face_smug", 0), ("jenny.doggyslow", 20), ("jenny.doggyfast", 8), ("jenny.sit", 20),
               ("jenny.ride", 30), ("jenny.blowjobsuck", 10), ("jenny.strip", 60), ("jenny.walk", 10)):
    a = bpy.data.actions[act]; rig.animation_data.action = a; rig.animation_data.action_slot = a.slots[0]
    sc.frame_set(f)
    pb = rig.pose.bones["midSection3"]
    from mathutils import Vector
    c = rig.matrix_world @ (pb.matrix @ pb.bone.matrix_local.inverted()) @ Vector((0.0, 0.175, 0.82))
    paths += render_views(os.path.join(HERE, "out", "renders", "ba", f"{act}"), [("a", 160, 15)], target=tuple(c), ortho_scale=0.9)
from PIL import Image
ims = [Image.open(p) for p in paths]; s = Image.new("RGB", (340 * 4, 680))
for i, im in enumerate(ims): s.paste(im, ((i % 4) * 340, (i // 4) * 340))
s.save(outp); print("RESULT ok")
