"""Export the GeckoLib Jenny/Bia models to Mine-imator (.mimodel + texture).

    python3 export_mimodel.py jenny nude [--style SlipperyT]
    -> out/mineimator/Jenny_Nude/Jenny_Nude.mimodel + Jenny_Nude.png

Mine-imator blocks only support box UV, while the mod model uses free per-face UVs on a 4096 px
texture. So every cube gets its own box-UV slot in a new atlas, and each face of the slot is
resampled from that face's original texture rectangle (flips and stretches included).

Coordinates follow Blockbench's official Mine-imator exporter (JannisX11/blockbench-plugins,
mimodel_format.js): Bedrock -> Blockbench (x mirrored, rot x/y negated) -> Mine-imator (x and z
negated); block UV = box-UV offset + depth. Every bone becomes a part at its exact pivot and rest
rotation, every cube a block with its exact size, inflate, pivot and rotation, so parts can be
posed in Mine-imator the same way the mod animates them. The default face is the sheet face
(half-lidded eyes, blush) from sources/<char>/faces.animation.json.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SRC = os.path.join(HERE, "sources")
OUT = os.path.join(HERE, "out")
DENSITY = 32            # atlas texels per model pixel
SUB = 4                 # sub-samples per texel axis (keeps thin decal lines like the mouth)
PAD = 2 / DENSITY       # gap between cube slots (model px)

SKIP_PREFIX = ("armor", "item", "custom")


def box_faces(w, h, d):
    """Blockbench box-UV face rects relative to the uv offset: from (u, v), signed size."""
    return {
        "east": ((0, d), (d, h)),
        "west": ((d + w, d), (d, h)),
        "up": ((d + w, d), (-w, -d)),
        "down": ((d + 2 * w, 0), (-w, d)),
        "south": ((2 * d + w, d), (w, h)),
        "north": ((d, d), (w, h)),
    }


def subtree(bones, root):
    kids = {}
    for n, b in bones.items():
        kids.setdefault(b.get("parent"), []).append(n)
    out, stack = set(), [root]
    while stack:
        n = stack.pop()
        if n in bones and n not in out:
            out.add(n)
            stack += kids.get(n, [])
    return out


class Packer:
    """Shelf packer for (w, h) rects in model px."""

    def __init__(self, width):
        self.W = width
        self.x = self.y = self.row = 0.0

    def place(self, w, h):
        w, h = w + PAD, h + PAD
        if self.x + w > self.W:
            self.x, self.y, self.row = 0.0, self.y + self.row, 0.0
        pos = (self.x, self.y)
        self.x += w
        self.row = max(self.row, h)
        return pos


def snap(v):
    return math.ceil(v * DENSITY) / DENSITY


def export(char, variant, style=None):
    geo = json.load(open(os.path.join(SRC, char, f"{char}{variant}.geo.json")))["minecraft:geometry"][0]
    desc, blist = geo["description"], geo["bones"]
    bones = {b["name"]: b for b in blist}
    tw, th = desc.get("texture_width", 64), desc.get("texture_height", 64)
    title = f"{char.capitalize()}_{variant.capitalize()}" + (f"_{style}" if style else "")
    tex_path = os.path.join(OUT, "textures", f"{title}_src.png")
    if not os.path.exists(tex_path):
        tex_path = os.path.join(SRC, char, f"{char}.png")
    src = np.array(Image.open(tex_path).convert("RGBA"))
    sh, sw = src.shape[:2]
    kx, ky = sw / tw, sh / th

    skip = subtree(bones, "steve") if "steve" in bones else set()
    for n in bones:
        if n.lower().startswith(SKIP_PREFIX):
            skip |= subtree(bones, n)
    # parents before children (the file order is not guaranteed)
    order, seen = [], set()

    def visit(n):
        if n in seen or n not in bones:
            return
        p = bones[n].get("parent")
        if p:
            visit(p)
        seen.add(n)
        order.append(bones[n])

    for b in blist:
        visit(b["name"])
    keep = [b for b in order if b["name"] not in skip]

    # ---- default face offsets
    offs = {}
    fpath = os.path.join(SRC, char, "faces.animation.json")
    if os.path.exists(fpath):
        anim = next(iter(json.load(open(fpath))["animations"].values()))
        for bn, ch in anim.get("bones", {}).items():
            pos = ch.get("position")
            scl = ch.get("scale")
            offs[bn] = (pos, scl)

    # ---- culling like the Blender import: a paper-thin decal face that points into another cube is
    # left transparent (otherwise it z-fights with / shows through the face it sits on)
    import bpy  # noqa: F401  (geckolib needs it importable)
    from lib import geckolib as gl
    border = [b["name"] for b in blist]
    W = gl.rest_world(bones, border)
    solids = gl.solid_cubes(bones, border, W, lambda n: n not in skip)
    moving = gl.moving_bones(bones, [os.path.join(SRC, char, f"{char}.animation.json"),
                                     os.path.join(SRC, char, "faces.animation.json")])
    moving &= subtree(bones, "head")

    def culled(bone, c, face):
        lo, hi, C = gl._cube_box(c)
        ext = hi - lo
        ax = gl._FACE_AXIS[face]
        if ext[ax] > 1e-6 or bone in moving:
            return False
        sign = 1.0 if face in ("south", "east", "up") else -1.0
        cen = (lo + hi) / 2
        cen[ax] += sign * 0.08
        pw = (W[bone] @ C @ np.r_[cen, 1.0])[:3]
        return gl._inside_any(pw, solids)

    # ---- atlas layout
    cubes = []
    for b in keep:
        for c in b.get("cubes") or []:
            if isinstance(c.get("uv"), dict):
                w, h, d = (float(v) for v in c["size"])
                cubes.append((b["name"], c, w, h, d))
    area = sum((2 * d + 2 * w + PAD) * (d + h + PAD) for _, _, w, h, d in cubes)
    S = 64
    while True:
        pk = Packer(S)
        order = sorted(range(len(cubes)), key=lambda i: -(cubes[i][4] + cubes[i][3]))
        slots = {}
        for i in order:
            _, _, w, h, d = cubes[i]
            slots[i] = pk.place(snap(2 * d + 2 * w), snap(d + h))
        if pk.y + pk.row <= S:
            break
        S *= 2 if area > S * S * 0.5 else 1.25
        S = int(math.ceil(S / 16) * 16)
    R = int(S * DENSITY)
    atlas = np.zeros((R, R, 4), np.uint8)

    def paint(dst_from, dst_size, src_uv, src_size):
        (x0, y0), (dw, dh) = dst_from, dst_size
        if abs(dw) < 1e-9 or abs(dh) < 1e-9:
            return
        xa, xb = sorted((x0, x0 + dw))
        ya, yb = sorted((y0, y0 + dh))
        ix = np.arange(int(math.floor(xa * DENSITY)), int(math.ceil(xb * DENSITY)))
        iy = np.arange(int(math.floor(ya * DENSITY)), int(math.ceil(yb * DENSITY)))
        if len(ix) == 0 or len(iy) == 0:
            return
        off = (np.arange(SUB) + 0.5) / SUB
        cx = (ix[:, None] + off[None, :]).ravel() / DENSITY
        cy = (iy[:, None] + off[None, :]).ravel() / DENSITY
        s = np.clip((cx - x0) / dw, 0, 1)
        t = np.clip((cy - y0) / dh, 0, 1)
        px = np.clip(((src_uv[0] + s * src_size[0]) * kx).astype(int), 0, sw - 1)
        py = np.clip(((src_uv[1] + t * src_size[1]) * ky).astype(int), 0, sh - 1)
        smp = src[np.ix_(py, px)].reshape(len(iy), SUB, len(ix), SUB, 4).transpose(0, 2, 1, 3, 4)
        smp = smp.reshape(len(iy), len(ix), SUB * SUB, 4).astype(np.float64)
        op = smp[..., 3] > 127
        n = op.sum(-1)
        col = (smp[..., :3] * op[..., None]).sum(-2) / np.maximum(n, 1)[..., None]
        out = np.zeros((len(iy), len(ix), 4), np.uint8)
        out[..., :3] = col.astype(np.uint8)
        out[..., 3] = np.where(n >= 0.3 * SUB * SUB, 255, 0)  # majority-ish: keeps thin lines, no fringe
        ix = np.clip(ix, 0, R - 1)
        iy = np.clip(iy, 0, R - 1)
        atlas[np.ix_(iy, ix)] = out

    # ---- parts
    def pivot_i(p):  # Blockbench internal
        return np.array([-p[0], p[1], p[2]], float)

    def conv(v):  # Blockbench internal -> Mine-imator
        return [round(float(v[0]) * -1, 5), round(float(v[1]), 5), round(float(v[2]) * -1, 5)]

    def rot_conv(r):  # Bedrock rot -> Blockbench internal (-x,-y,z) -> Mine-imator (negate x,z)
        return [round(float(r[0]), 5), round(-float(r[1]), 5), round(-float(r[2]), 5)]

    parts = {}
    roots = []
    cube_index = {id(c): i for i, (_, c, *_r) in enumerate(cubes)}
    for b in keep:
        n = b["name"]
        piv = pivot_i(b.get("pivot", [0, 0, 0]))
        par = b.get("parent")
        ppiv = pivot_i(bones[par].get("pivot", [0, 0, 0])) if par in bones and par not in skip else np.zeros(3)
        part = {"name": n, "position": conv(piv - ppiv)}
        if any(b.get("rotation", [0, 0, 0])):
            part["rotation"] = rot_conv(b["rotation"])
        if n in offs:
            pos, scl = offs[n]
            if pos:
                part["position"] = [round(a + o, 5) for a, o in zip(part["position"], [pos[0], pos[1], -pos[2]])]
            if scl:
                part["scale"] = [float(v) for v in (scl if isinstance(scl, list) else [scl] * 3)]
        shapes = []
        for c in b.get("cubes") or []:
            if id(c) not in cube_index:
                continue
            i = cube_index[id(c)]
            _, _, w, h, d = cubes[i]
            o = np.asarray(c["origin"], float)
            fr = np.array([-(o[0] + w), o[1], o[2]])
            to = np.array([-o[0], o[1] + h, o[2] + d])
            co = pivot_i(c["pivot"]) if "pivot" in c else piv
            a, z = conv(fr - co), conv(to - co)
            lo = [min(p, q) for p, q in zip(a, z)]
            hi = [max(p, q) for p, q in zip(a, z)]
            ox, oy = slots[i]
            ox, oy = snap(ox), snap(oy)
            faces = box_faces(w, h, d)
            for fname, spec in c["uv"].items():
                if fname not in faces or culled(n, c, fname):
                    continue
                (fx, fy), fsize = faces[fname]
                paint((ox + fx, oy + fy), fsize, spec["uv"], spec.get("uv_size", [0, 0]))
            shape = {"type": "block", "from": lo, "to": hi, "uv": [round(ox + d, 5), round(oy + d, 5)]}
            if c.get("inflate"):
                shape["inflate"] = float(c["inflate"])
            pos = conv(co - piv)
            if any(abs(v) > 1e-9 for v in pos):
                shape["position"] = pos
            if any(c.get("rotation", [0, 0, 0])):
                shape["rotation"] = rot_conv(c["rotation"])
            shapes.append(shape)
        if shapes:
            part["shapes"] = shapes
        parts[n] = part
        if par in parts:
            parts[par].setdefault("parts", []).append(part)
        else:
            roots.append(part)

    # grow every painted face by one texel into empty space, so sampling right at a face edge
    # never hits a transparent texel (shows up as light seams on block edges)
    a = atlas[..., 3] > 0
    grown = atlas.copy()
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        src_sh = np.roll(atlas, (dy, dx), axis=(0, 1))
        take = (~a) & (src_sh[..., 3] > 0) & (grown[..., 3] == 0)
        grown[take] = src_sh[take]
    atlas = grown
    model = {"name": title.replace("_", " "), "texture": f"{title}.png",
             "texture_size": [S, S], "parts": roots}
    odir = os.path.join(OUT, "mineimator", title)
    os.makedirs(odir, exist_ok=True)
    Image.fromarray(atlas, "RGBA").save(os.path.join(odir, f"{title}.png"))
    with open(os.path.join(odir, f"{title}.mimodel"), "w") as f:
        json.dump(model, f, indent=1)
    print(f"[mimodel] {title}: {len(parts)} parts, {len(cubes)} blocks, atlas {R}px "
          f"(texture_size {S}) -> {odir}", flush=True)
    return os.path.join(odir, f"{title}.mimodel")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("char")
    ap.add_argument("variant")
    ap.add_argument("--style", default=None)
    a = ap.parse_args()
    export(a.char, a.variant, a.style)


if __name__ == "__main__":
    main()
