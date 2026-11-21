"""Stress-pose a built character and render it, to check skin deformation.

    python3 posetest.py Jenny
"""
import math
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib.blend import setup_render, render_views  # noqa: E402

name = sys.argv[1]
bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "out", f"{name}.blend"))
rig = bpy.data.objects[f"{name}_rig"]
pb = rig.pose.bones
for s in ("L", "R"):
    pb[f"thigh_parent.{s}"]["IK_FK"] = 1.0
    pb[f"upper_arm_parent.{s}"]["IK_FK"] = 1.0


def rot(bone, x=0, y=0, z=0):
    b = pb[bone]
    b.rotation_mode = "XYZ"
    b.rotation_euler = (math.radians(x), math.radians(y), math.radians(z))


poses = {
    # FK axes (probed): thigh -X = forward, -Z(L)/+Z(R) = spread; shin/forearm +X = bend;
    # upper_arm +X = forward, +Z(L)/-Z(R) = raise sideways.
    "spread": lambda: (rot("thigh_fk.L", -20, 0, -42), rot("thigh_fk.R", -20, 0, 42),
                       rot("shin_fk.L", 35), rot("shin_fk.R", 35),
                       rot("upper_arm_fk.L", 0, 0, 75), rot("upper_arm_fk.R", 0, 0, -75),
                       rot("forearm_fk.L", 60), rot("forearm_fk.R", 60)),
    "kneel": lambda: (rot("thigh_fk.L", -95), rot("thigh_fk.R", 30),
                      rot("shin_fk.L", 110), rot("shin_fk.R", 100),
                      rot("upper_arm_fk.L", 150), rot("upper_arm_fk.R", -40),
                      rot("chest", 15), rot("breast.L", 12)),
}
setup_render((420, 560), 16)
for pname, fn in poses.items():
    for b in pb:
        b.rotation_mode = "QUATERNION"
        b.rotation_quaternion = (1, 0, 0, 0)
        b.location = (0, 0, 0)
    fn()
    bpy.context.view_layer.update()
    rig["outfit"] = int(os.environ.get("OUTFIT", "0"))
    render_views(os.path.join(HERE, "out", "renders", f"{name}_pose_{pname}"),
                 [("front", 0, 5), ("q3", 40, 10), ("side", 90, 5)], target=(0, 0, 0.85), ortho_scale=2.5)
print("RESULT ok")

from PIL import Image  # noqa: E402
ims = [Image.open(os.path.join(HERE, "out", "renders", f"{name}_pose_{p}_{v}.png"))
       for p in poses for v in ("front", "q3", "side")]
w, h = ims[0].size
sheet = Image.new("RGB", (w * 3, h * len(poses)))
for i, im in enumerate(ims):
    sheet.paste(im, ((i % 3) * w, (i // 3) * h))
sheet.save(os.path.join(HERE, "out", "renders", f"{name}_pose_sheet.png"))
