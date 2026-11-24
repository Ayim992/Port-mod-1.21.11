"""Hip/butt close-ups of the final Jenny (dressed): python3 hipcheck.py [outfit] [tag]"""
import bpy, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from lib.blend import setup_render, render_views
outfit = int(sys.argv[1]) if len(sys.argv) > 1 else 1
tag = sys.argv[2] if len(sys.argv) > 2 else "now"
bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "out", "Jenny_Final.blend"))
sc = bpy.context.scene; rig = bpy.data.objects["Jenny_rig"]; rig["outfit"] = outfit; rig["smooth"] = int(os.environ.get("SMOOTH", "1")); sc.frame_set(0)
setup_render((420, 420), 16); sc.cycles.transparent_max_bounces = 128
views = [("f", 0, 5), ("q", 40, 5), ("s", 90, 5), ("bq", 140, 5), ("b", 180, 5), ("bq2", 220, 5)]
paths = render_views(os.path.join(HERE, "out", "renders", f"hip_{tag}"), views, target=(0, 0, 0.92), ortho_scale=0.95)
from PIL import Image
ims = [Image.open(p) for p in paths]; w, h = ims[0].size
s = Image.new("RGB", (w * 3, h * 2)); [s.paste(im, ((i % 3) * w, (i // 3) * h)) for i, im in enumerate(ims)]
s.save(os.path.join(HERE, "out", "renders", f"hip_{tag}.png")); print("RESULT ok")
