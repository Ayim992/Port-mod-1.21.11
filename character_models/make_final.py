"""Combine the SlipperyT Jenny variants into one polished file: out/Jenny_Final.blend

    python3 make_final.py

One rig to animate (Jenny_rig, all 40 actions), with an `outfit` switch: the dressed model rides on
a hidden second armature whose bones copy Jenny_rig's bones, so every action works dressed or nude.
Rig props: outfit (0 nude / 1 dressed), smooth (1 smooth breasts/butt/vulva, 0 original cubes),
steve (partner), pink_shading (0..1). Bones are sorted into coloured bone collections.
"""
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
NUDE, DRESSED = "Jenny_Nude_SlipperyT", "Jenny_Dressed_SlipperyT"


def subtree(rig, names):
    out, stack = set(), list(names)
    while stack:
        n = stack.pop()
        b = rig.data.bones.get(n)
        if b and n not in out:
            out.add(n)
            stack += [c.name for c in b.children]
    return out


def vis_driver(ob, rig, expr, props):
    if ob.animation_data:
        for fc in list(ob.animation_data.drivers):
            if fc.data_path in ("hide_viewport", "hide_render"):
                ob.driver_remove(fc.data_path)
    for path in ("hide_viewport", "hide_render"):
        d = ob.driver_add(path).driver
        d.type = "SCRIPTED"
        for p in props:
            v = d.variables.new()
            v.name = p
            v.targets[0].id = rig
            v.targets[0].data_path = f'["{p}"]'
        d.expression = expr


def retarget_drivers(idblock, old, new):
    ad = getattr(idblock, "animation_data", None)
    if not ad:
        return
    for fc in ad.drivers:
        for v in fc.driver.variables:
            for t in v.targets:
                if t.id == old:
                    t.id = new


def main():
    bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "out", f"{NUDE}.blend"))
    sc = bpy.context.scene
    with bpy.data.libraries.load(os.path.join(HERE, "out", f"{DRESSED}.blend"), link=False) as (src, dst):
        dst.collections = [DRESSED]
    sc.collection.children.link(dst.collections[0])
    rig = bpy.data.objects[f"{NUDE}_rig"]
    drig = bpy.data.objects[f"{DRESSED}_rig"]

    # ---- names and collections
    rig.name = rig.data.name = "Jenny_rig"
    bpy.data.collections[NUDE].name = "Jenny"
    bpy.data.collections[DRESSED].name = "Jenny_Outfit"
    for ob in list(bpy.data.objects):
        for a, b in ((NUDE + "_", "Jenny_"), (DRESSED + "_", "Jenny_Dressed_")):
            if ob.name.startswith(a):
                ob.name = b + ob.name[len(a):]
    drig.name = "Jenny_Dressed_rig (follows Jenny_rig)"

    # ---- dressed armature follows the main rig bone-for-bone
    drig.animation_data_create().action = None
    for pb in drig.pose.bones:
        pb.matrix_basis.identity()
        if pb.name in rig.pose.bones:
            c = pb.constraints.new("COPY_TRANSFORMS")
            c.target = rig
            c.subtarget = pb.name
            # copy the pose, not the world matrix: a few bones (boobs, booty, cheeks) have
            # different rest pivots in the two variants
            c.owner_space = c.target_space = "LOCAL"
    drig.hide_set(True)
    drig.hide_render = True
    for idb in list(bpy.data.objects) + list(bpy.data.materials):
        retarget_drivers(idb, drig, rig)
        if getattr(idb, "node_tree", None):
            retarget_drivers(idb.node_tree, drig, rig)

    # ---- outfit switch
    rig["outfit"] = 1
    rig.id_properties_ui("outfit").update(min=0, max=1, description="1 = dressed (sweater, pants), 0 = nude")
    nb, ns, nk = (bpy.data.objects[f"Jenny_{n}"] for n in ("Body", "Smooth", "BlockyParts"))
    db, ds, dk = (bpy.data.objects[f"Jenny_Dressed_{n}"] for n in ("Body", "Smooth", "BlockyParts"))
    vis_driver(nb, rig, "round(outfit) == 1", ["outfit"])
    vis_driver(ns, rig, "round(outfit) == 1 or round(smooth) == 0", ["outfit", "smooth"])
    vis_driver(nk, rig, "round(outfit) == 1 or round(smooth) == 1", ["outfit", "smooth"])
    vis_driver(db, rig, "round(outfit) == 0", ["outfit"])
    vis_driver(ds, rig, "round(outfit) == 0 or round(smooth) == 0", ["outfit", "smooth"])
    vis_driver(dk, rig, "round(outfit) == 0 or round(smooth) == 1", ["outfit", "smooth"])
    for n in ("Steve", "SteveCock", "ArmorSlots"):      # the dressed copies are never needed
        ob = bpy.data.objects[f"Jenny_Dressed_{n}"]
        if ob.animation_data:
            for fc in list(ob.animation_data.drivers):
                ob.driver_remove(fc.data_path)
        ob.hide_viewport = ob.hide_render = True
    # partner / armour helpers in their own collection
    extra = bpy.data.collections.new("Partner & armor slots")
    sc.collection.children.link(extra)
    for n in ("Jenny_Steve", "Jenny_SteveCock", "Jenny_ArmorSlots", "Jenny_Dressed_Steve",
              "Jenny_Dressed_SteveCock", "Jenny_Dressed_ArmorSlots"):
        ob = bpy.data.objects[n]
        for c in list(ob.users_collection):
            c.objects.unlink(ob)
        extra.objects.link(ob)

    # ---- bone collections
    arm = rig.data
    for bc in list(arm.collections):
        arm.collections.remove(bc)
    groups = [
        ("Partner (Steve)", subtree(rig, ["steve"]), "THEME11", False),
        ("Armor & items", subtree(rig, [b.name for b in arm.bones
                                         if b.name.lower().startswith(("armor", "item", "custom"))]), "THEME14", False),
        ("Face", subtree(rig, ["eyes", "mouth", "blush", "heart"]), "THEME09", True),
        ("Breasts", subtree(rig, ["boobs"]), "THEME01", True),
        ("Butt", subtree(rig, ["booty"]), "THEME06", True),
        ("Genitals", subtree(rig, ["vagina"]), "THEME02", True),
        ("Arms", subtree(rig, ["armL", "armR"]), "THEME03", True),
        ("Legs", subtree(rig, ["legL", "legR"]), "THEME04", True),
    ]
    taken = set()
    for name, bones, theme, visible in groups:
        bc = arm.collections.new(name)
        bc.is_visible = visible
        for n in bones - taken:
            bc.assign(arm.bones[n])
            arm.bones[n].color.palette = theme
        taken |= bones
    core = arm.collections.new("Body & head")
    for b in arm.bones:
        if b.name not in taken:
            core.assign(b)
            b.color.palette = "THEME07"
    arm.display_type = "STICK"

    # ---- defaults
    ad = rig.animation_data
    face = bpy.data.actions.get("jenny.face_smug")
    if face:
        ad.action = face
        ad.action_slot = face.slots[0]
    rig["smooth"] = 1
    rig["steve"] = 0
    rig["pink_shading"] = 0
    sc.render.fps = 24
    sc.frame_start, sc.frame_end = 0, 120
    bpy.context.view_layer.objects.active = rig
    for ob in sc.objects:
        ob.select_set(ob == rig)
    for area_scr in bpy.data.screens:      # open in Material preview so textures show
        for area in area_scr.areas:
            if area.type == "VIEW_3D":
                for sp in area.spaces:
                    if sp.type == "VIEW_3D":
                        sp.shading.type = "MATERIAL"
    bpy.ops.file.pack_all()
    out = os.path.join(HERE, "out", "Jenny_Final.blend")
    bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
    print("[final] saved", out, len(bpy.data.actions), "actions", flush=True)


if __name__ == "__main__":
    main()
