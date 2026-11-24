"""Rebuild a .mimodel in Blender (inverse of export_mimodel / Blockbench's exporter) and render it,
to check the export against the original model.   python3 mimodel_check.py out/mineimator/X/X.mimodel"""
import json, os, shutil, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from export_mimodel import box_faces

path = sys.argv[1]
m = json.load(open(path))
S = m["texture_size"][0]
bones = []

def conv(v):
    return np.array([-v[0], v[1], -v[2]], float)

def walk(part, parent, parent_int):
    o_int = parent_int + conv(part["position"])
    b = {"name": part["name"], "pivot": [-o_int[0], o_int[1], o_int[2]]}
    if parent:
        b["parent"] = parent
    if "rotation" in part:
        r = part["rotation"]
        b["rotation"] = [r[0], -r[1], -r[2]]
    cubes = []
    for sh in part.get("shapes", []):
        a, z = conv(sh["from"]), conv(sh["to"])
        co = o_int + conv(sh.get("position", [0, 0, 0]))
        lo, hi = np.minimum(a, z) + co, np.maximum(a, z) + co
        w, h, d = hi - lo
        c = {"origin": [-hi[0], lo[1], lo[2]], "size": [w, h, d], "pivot": [-co[0], co[1], co[2]]}
        if "rotation" in sh:
            r = sh["rotation"]
            c["rotation"] = [r[0], -r[1], -r[2]]
        if "inflate" in sh:
            c["inflate"] = sh["inflate"]
        u0, v0 = sh["uv"][0] - d, sh["uv"][1] - d
        c["uv"] = {f: {"uv": [u0 + fr[0], v0 + fr[1]], "uv_size": list(sz)} for f, (fr, sz) in box_faces(w, h, d).items()}
        cubes.append(c)
    if cubes:
        b["cubes"] = cubes
    bones.append(b)
    for ch in part.get("parts", []):
        walk(ch, part["name"], o_int)

for p in m["parts"]:
    walk(p, None, np.zeros(3))
d = os.path.join(HERE, "sources", "mi_check")
os.makedirs(d, exist_ok=True)
geo = {"format_version": "1.12.0", "minecraft:geometry": [{"description": {"identifier": "geometry.mi_check",
       "texture_width": S, "texture_height": S}, "bones": bones}]}
json.dump(geo, open(os.path.join(d, "mi_checknude.geo.json"), "w"))
shutil.copy(os.path.join(os.path.dirname(path), m["texture"]), os.path.join(d, "mi_check.png"))
print("RESULT", len(bones), "bones")
