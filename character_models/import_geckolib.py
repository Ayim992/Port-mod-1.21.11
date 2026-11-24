"""Convert the exact GeckoLib (sexmod) Jenny / Bia models into rigged, animated .blend files.

    python3 import_geckolib.py jenny            # -> out/Jenny_Nude.blend, out/Jenny_Dressed.blend
    python3 import_geckolib.py bia --no-anim
    python3 import_geckolib.py jenny --render  # also renders review views

Sources live in sources/<name>/: <name>nude.geo.json, <name>dressed.geo.json,
<name>.animation.json, <name>.png
"""
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from lib import geckolib as gl  # noqa: E402
from lib.blend import setup_render, render_views  # noqa: E402

SRC = os.path.join(HERE, "sources")
OUT = os.path.join(HERE, "out")


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


def add_prop(rig, name, default, desc):
    rig[name] = int(default)
    rig.id_properties_ui(name).update(min=0, max=1, soft_min=0, soft_max=1, description=desc)


def drive_hide(ob, rig, prop, hide_when=0):
    for path in ("hide_viewport", "hide_render"):
        d = ob.driver_add(path).driver
        d.type = "SCRIPTED"
        v = d.variables.new()
        v.name = "v"
        v.targets[0].id = rig
        v.targets[0].data_path = f'["{prop}"]'
        d.expression = f"round(v) == {hide_when}"


BREAST_SMOOTH = 40


def smooth_lobes(bones, variant, style=None):
    """Which bone subtrees become smooth lobes. Armor overlays, nipples and the interior/anus
    pieces stay as the original cubes."""
    def keep(n):
        low = n.lower()
        return not low.startswith("armor") and "nipple" not in low
    out = []
    px = gl.PXM
    if "boobs" in bones:
        bs = {n for n in subtree(bones, "boobs") if keep(n)}
        nip = {n for n in subtree(bones, "boobs") if "nipple" in n.lower()}
        if style and variant == "dressed":
            # a shirt stretches across both breasts: one lobe spanning the cleavage (no split down the
            # middle), one plain sweater colour
            out.append(dict(name="breasts", bones=bs | nip, center_shift=(0, 1.3 * px, 0),
                            subdiv=7, smooth=60, cone=1.6, uniform_color=True))
            bs = set()
        for side, s in (("L", 1), ("R", -1)) if bs else ():
            # the nipple/areola cubes are part of the breast surface (leaving them out leaves a dent)
            out.append(dict(name=f"breast.{side}", bones=bs | nip, split_x=s, center_shift=(0, 1.3 * px, 0),
                            subdiv=7 if style else 6, smooth=BREAST_SMOOTH * (4 if style else 1), cone=2.0, full_breast=True, press=0.55,
                            soft_nipple=bool(style), uniform_skin=bool(style), areola_radius=0.75, areola_dark=0.92,
                            nipple_radius=0.32, nipple_dark=0.9,
                            nipple_height=0.24 if style else 0.0))
            if nip:
                # plus a firmer nipple on top so it keeps poking out after smoothing
                if not style:  # SlipperyT's Jenny has flat, wide areolas without a peg nipple
                    out.append(dict(name=f"nipple.{side}", bones=nip, split_x=s, center_shift=(0, 0.9 * px, 0),
                                    subdiv=6, smooth=6, cone=3.0, weight_smooth=2))
    if "booty" in bones:
        bs = {n for n in subtree(bones, "booty") if keep(n)}
        for side, s in (("L", 1), ("R", -1)):
            out.append(dict(name=f"butt.{side}", bones=bs, split_x=s, center_shift=(0, -1.4 * px, 0.2 * px),
                            subdiv=6, smooth=40, cone=2.0, skin_match=bool(style)))
    if variant == "nude" and "vagina" in bones:
        skip = ("asshole", "fuckhole", "inside")
        bs = {n for n in subtree(bones, "vagina")
              if keep(n) and not any(n.lower().startswith(k) for k in skip)}
        out.append(dict(name="vulva", bones=bs, center_shift=(0, 0.9 * px, 0.6 * px),
                        subdiv=6, smooth=30, cone=2.4, weight_smooth=4,
                        skin_match=bool(style), flatten=(0.55, 0.85) if style else None,
                        slit=0.22 if style else None))
    return out


def paint_sheet_blush(tex_path, out_path):
    """Jenny's `blush` cube only carries a faint skin shade; paint two pink cheek ovals like the
    June 2020 sheet (outside them the plane is transparent, so the face shows through)."""
    from PIL import Image
    a = np.array(Image.open(tex_path).convert("RGBA")).astype(np.float64)
    s = 64  # texels per model pixel in this 4096 texture
    u0, v0, w, h = 43.875, 14.11719, 5.25, 1.98438
    x0, y0, x1, y1 = int(u0 * s), int(v0 * s), int((u0 + w) * s), int((v0 + h) * s)
    reg = a[y0:y1, x0:x1]
    H, Wd = reg.shape[:2]
    yy, xx = np.mgrid[0:H, 0:Wd]
    pink = np.array([236.0, 132.0, 142.0])
    alpha = np.zeros((H, Wd))
    for cx in (0.16, 0.84):
        d = ((xx / Wd - cx) / 0.13) ** 2 + ((yy / H - 0.78) / 0.26) ** 2
        alpha = np.maximum(alpha, np.clip(1.4 * (1 - d), 0, 1))
    skin = np.array([248.0, 202.0, 168.0])
    col = skin * (1 - 0.5 * alpha[..., None]) + pink * (0.5 * alpha[..., None])
    reg[..., :3] = col
    reg[..., 3] = np.where(alpha > 0.15, 255, 0)
    a[y0:y1, x0:x1] = reg
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(a.astype(np.uint8), "RGBA").save(out_path)
    return out_path


def convert(char, variant, anim=True, render=False, fps=24, smooth=True, style=None):
    title = f"{char.capitalize()}_{variant.capitalize()}" + (f"_{style}" if style else "")
    geo = os.path.join(SRC, char, f"{char}{variant}.geo.json")
    tex = os.path.join(SRC, char, f"{char}.png")
    anim_path = os.path.join(SRC, char, f"{char}.animation.json")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.fps = fps
    # many stacked alpha-cutout cubes: Cycles turns rays black once it runs out of transparent bounces
    sc.cycles.transparent_max_bounces = 128
    sc.cycles.max_bounces = 16
    if hasattr(sc, "eevee"):
        sc.eevee.use_shadows = True
    col = bpy.data.collections.new(title)
    sc.collection.children.link(col)

    desc, bones, order = gl.load_geo(geo)
    W = gl.rest_world(bones, order)
    tw, th = desc.get("texture_width", 64), desc.get("texture_height", 64)
    if style and char == "jenny":
        tex = paint_sheet_blush(tex, os.path.join(OUT, "textures", f"{title}_src.png"))
    mat = gl.texture_material(f"{title}_tex", tex)

    steve = subtree(bones, "steve")
    armor = set()
    for n in bones:
        if n.lower().startswith(("armor", "customshoe", "item")):
            armor |= subtree(bones, n)
    genitals = set()
    for n in ("cock", "ballL", "ballR"):
        genitals |= subtree(bones, n) if n in bones else set()
    groups = {
        "Body": lambda n: n not in steve and n not in armor,
        "Steve": lambda n: n in steve and n not in genitals,
        "SteveCock": lambda n: n in genitals,
        "ArmorSlots": lambda n: n in armor,
    }
    # the mod skins the partner with the player's skin: use a generated Steve-style skin
    from lib import steve_skin
    skin_png = os.path.join(OUT, "textures", "partner_skin.png")
    os.makedirs(os.path.dirname(skin_png), exist_ok=True)
    if not os.path.exists(skin_png):
        steve_skin.make(skin_png)
    mat_skin = gl.texture_material(f"{title}_partner_skin", skin_png, rgba_dir=os.path.join(OUT, "textures"))
    mat_flesh = bpy.data.materials.new(f"{title}_partner_flesh")
    mat_flesh.use_nodes = True
    fb = mat_flesh.node_tree.nodes["Principled BSDF"]
    fb.inputs["Base Color"].default_value = (0.78, 0.42, 0.36, 1)
    fb.inputs["Roughness"].default_value = 0.5
    mats = {"Steve": mat_skin, "SteveCock": mat_flesh}

    # only face pieces (alternate eyes/mouths/blush parked inside the head) are exempt from culling;
    # other animated parts keep it (e.g. the dark crotch pieces under the dressed pants)
    moving = gl.moving_bones(bones, [anim_path, os.path.join(SRC, char, "faces.animation.json")])
    moving &= subtree(bones, "head")
    rig, A = gl.build_armature(f"{title}_rig", col, bones, order, W)
    objs = {}
    for gname, flt in groups.items():
        v, f, uv, vb, names = gl.cubes_to_mesh(bones, order, W, tw, th, flt, moving=moving)
        if len(f) == 0:
            continue
        ob = gl.build_mesh_object(f"{title}_{gname}", col, v, f, uv, vb, names, mats.get(gname, mat))
        gl.bind(ob, rig)
        objs[gname] = ob
        print(f"[import] {gname}: {len(names)} bones with cubes, {len(f)} faces", flush=True)

    if smooth and "Body" in objs:
        from PIL import Image
        from lib import smooth_regions as sr
        tex_arr = np.array(Image.open(tex).convert("RGBA"))
        smat = sr.color_material(f"{title}_smooth")
        sob, region = sr.build_smooth(title, col, objs["Body"], rig, smooth_lobes(bones, variant, style),
                                      tex_arr, smat)
        if sob is not None:
            blocky = sr.split_blocky(objs["Body"], region, title, col)
            objs["Smooth"], objs["BlockyParts"] = sob, blocky
            add_prop(rig, "smooth", 1, "1 = smooth breasts/butt/vulva, 0 = the original blocky cubes")
            drive_hide(sob, rig, "smooth", 0)
            drive_hide(blocky, rig, "smooth", 1)

    if style:
        from lib import stylize
        # off by default: it tints the breasts' creases pink (the breasts are one plain colour)
        add_prop(rig, "pink_shading", 0, "SlipperyT-style pink shading in skin creases (0 = off)")
        rig.id_properties_ui("pink_shading").update(subtype="FACTOR")
        # only on the smooth parts: AO on the cube body counts its hidden alpha-cut interior as creases
        if "Smooth" in objs:
            stylize.add_skin_tint(objs["Smooth"].data.materials[0], rig, ao_distance=0.18)

    add_prop(rig, "steve", 0, "Show the Steve partner model used by the scene animations")
    for k in ("Steve", "SteveCock"):
        if k in objs:
            drive_hide(objs[k], rig, "steve")
    if "ArmorSlots" in objs:
        # armor/item overlay slots are only drawn by the mod when armor is worn
        objs["ArmorSlots"].hide_viewport = True
        objs["ArmorSlots"].hide_render = True

    made = []
    if anim and os.path.exists(anim_path):
        made = gl.bake_animations(rig, A, W, bones, order, anim_path, fps=fps)
        print(f"[import] baked {len(made)} animations", flush=True)
        face_path = os.path.join(SRC, char, "faces.animation.json")
        faces = []
        if os.path.exists(face_path):
            # static face poses (the sheet's half-lidded, blushing smug face); the default look
            faces = gl.bake_animations(rig, A, W, bones, order, face_path, fps=fps)
            made += faces
        ad = rig.animation_data_create()
        idle = faces[0] if faces else next(((a, s) for a, s in made if a.name.endswith("idle")), made[0])
        ad.action, ad.action_slot = idle
        sc.frame_start, sc.frame_end = int(idle[0].frame_range[0]), int(idle[0].frame_range[1])

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{title}.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path, compress=True)
    print(f"[import] saved {path}", flush=True)

    if render:
        setup_render((700, 933), 32)
        sc.cycles.transparent_max_bounces = 128
        ad = rig.animation_data
        keep = (ad.action, ad.action_slot) if ad else None
        if ad:
            ad.action = None
        for pb in rig.pose.bones:
            pb.matrix_basis.identity()
        bpy.context.view_layer.update()
        rdir = os.path.join(OUT, "renders")
        render_views(os.path.join(rdir, f"{title}"),
                     [("front", 0, 5), ("q3", 35, 8), ("side", 90, 5), ("back", 180, 5)],
                     target=(0, 0, 1.02), ortho_scale=2.5)
        if keep:
            ad.action, ad.action_slot = keep
    return path, made


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("characters", nargs="+")
    ap.add_argument("--variants", default="nude,dressed")
    ap.add_argument("--no-anim", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--blocky", action="store_true", help="skip the smooth lobes")
    ap.add_argument("--style", default=None, help="e.g. SlipperyT: film-look shading + flat wide areolas")
    a = ap.parse_args(argv)
    for c in a.characters:
        for v in a.variants.split(","):
            p, made = convert(c, v, anim=not a.no_anim, render=a.render, smooth=not a.blocky,
                              style=a.style)
            print(f"RESULT {c} {v} {p} actions={len(made)}", flush=True)


if __name__ == "__main__":
    main()
