"""Blender-side helpers: meshes, color layers, materials, head textures, rendering."""
import os
import math
import numpy as np
import bpy
import bmesh
from mathutils import Vector, Matrix

PX = 1.8 / 32.0  # one Minecraft pixel in meters (character is 32 px = 1.8 m tall)


# --------------------------------------------------------------------------- scene
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    import addon_utils
    addon_utils.enable("rigify", default_set=True)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.render.fps = 24
    return sc


def collection(name, parent=None):
    col = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    parent = parent or bpy.context.scene.collection
    if col.name not in parent.children:
        parent.children.link(col)
    return col


# --------------------------------------------------------------------------- meshes
def make_mesh(name, verts_px, faces, col, smooth=True):
    me = bpy.data.meshes.new(name)
    verts = np.asarray(verts_px, dtype=np.float64) * PX
    faces = np.asarray(faces)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    nloops = faces.shape[1]
    me.loops.add(len(faces) * nloops)
    me.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
    me.polygons.add(len(faces))
    me.polygons.foreach_set("loop_start", np.arange(0, len(faces) * nloops, nloops, dtype=np.int32))
    me.update(calc_edges=True)
    me.validate()
    ob = bpy.data.objects.new(name, me)
    col.objects.link(ob)
    if smooth:
        me.shade_smooth()
    else:
        me.shade_flat()
    return ob


def voxel_remesh(ob, size_px, smooth=True):
    mod = ob.modifiers.new("Remesh", "REMESH")
    mod.mode = "VOXEL"
    mod.voxel_size = size_px * PX
    mod.adaptivity = 0.0
    mod.use_smooth_shade = smooth
    apply_modifiers(ob)


def apply_modifiers(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ev)
    old = ob.data
    ob.modifiers.clear()
    ob.data = me
    bpy.data.meshes.remove(old)


def laplacian_smooth(ob, iterations=2, factor=0.5):
    me = ob.data
    n = len(me.vertices)
    co = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3).astype(np.float64)
    e = np.empty(len(me.edges) * 2, np.int32)
    me.edges.foreach_get("vertices", e)
    e = e.reshape(-1, 2)
    deg = np.bincount(e.ravel(), minlength=n).astype(np.float64)
    for _ in range(iterations):
        acc = np.zeros_like(co)
        np.add.at(acc, e[:, 0], co[e[:, 1]])
        np.add.at(acc, e[:, 1], co[e[:, 0]])
        avg = acc / np.maximum(deg, 1)[:, None]
        co = co + factor * (avg - co)
    me.vertices.foreach_set("co", co.astype(np.float32).ravel())
    me.update()


def face_centers_normals_px(ob):
    me = ob.data
    nf = len(me.polygons)
    c = np.empty(nf * 3, np.float32)
    nrm = np.empty(nf * 3, np.float32)
    me.polygons.foreach_get("center", c)
    me.polygons.foreach_get("normal", nrm)
    return c.reshape(-1, 3).astype(np.float64) / PX, nrm.reshape(-1, 3).astype(np.float64)


def set_face_colors(ob, name, face_rgba):
    """Write a per-face color into a CORNER-domain float color attribute."""
    me = ob.data
    attr = me.color_attributes.get(name) or me.color_attributes.new(name, "FLOAT_COLOR", "CORNER")
    rgba = np.asarray(face_rgba, dtype=np.float32)
    if rgba.shape[1] == 3:
        rgba = np.concatenate([rgba, np.ones((len(rgba), 1), np.float32)], axis=1)
    totals = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("loop_total", totals)
    per_loop = np.repeat(rgba, totals, axis=0)
    attr.data.foreach_set("color", per_loop.ravel())
    return attr


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def hexcol(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def paint_object(ob, paint_fn, layers=("Col", "ColNude")):
    """paint_fn(P_px, N, layer) -> (rgb (F,3) sRGB 0..1, emit (F,))"""
    P, N = face_centers_normals_px(ob)
    for layer in layers:
        rgb, emit = paint_fn(P, N, layer)
        lin = srgb_to_linear(rgb)
        set_face_colors(ob, layer, lin)
        e = np.repeat(np.asarray(emit, dtype=np.float64)[:, None], 3, axis=1)
        set_face_colors(ob, "Emit" + layer[3:], e)


def colors_to_layers(ob, face_rgb, face_emit, layers=("Col", "ColNude")):
    lin = srgb_to_linear(np.asarray(face_rgb)[:, :3])
    e = np.repeat(np.asarray(face_emit, dtype=np.float64)[:, None], 3, axis=1)
    for layer in layers:
        set_face_colors(ob, layer, lin)
        set_face_colors(ob, "Emit" + layer[3:], e)


# --------------------------------------------------------------------------- materials
def _driver_prop(target_id, path, index, rig, prop, expr="v"):
    fc = target_id.driver_add(path, index) if index is not None else target_id.driver_add(path)
    drv = fc.driver
    drv.type = "SCRIPTED"
    for v in list(drv.variables):
        drv.variables.remove(v)
    var = drv.variables.new()
    var.name = "v"
    var.type = "SINGLE_PROP"
    var.targets[0].id_type = "OBJECT"
    var.targets[0].id = rig
    var.targets[0].data_path = f'["{prop}"]'
    drv.expression = expr
    return fc


def body_material(name, rig=None, emit_strength=6.0, roughness=0.55, sheen=0.0):
    """Color-attribute material with a driven dressed/nude switch."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (700, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (350, 0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Specular IOR Level"].default_value = 0.35
    bsdf.inputs["Emission Strength"].default_value = emit_strength
    a_d = nt.nodes.new("ShaderNodeAttribute"); a_d.attribute_name = "Col"; a_d.location = (-500, 200)
    a_n = nt.nodes.new("ShaderNodeAttribute"); a_n.attribute_name = "ColNude"; a_n.location = (-500, 0)
    e_d = nt.nodes.new("ShaderNodeAttribute"); e_d.attribute_name = "Emit"; e_d.location = (-500, -200)
    e_n = nt.nodes.new("ShaderNodeAttribute"); e_n.attribute_name = "EmitNude"; e_n.location = (-500, -400)
    outfit = nt.nodes.new("ShaderNodeValue"); outfit.name = "Outfit"; outfit.label = "Outfit (driven)"
    outfit.location = (-500, 400); outfit.outputs[0].default_value = 1.0
    mixc = nt.nodes.new("ShaderNodeMix"); mixc.data_type = "RGBA"; mixc.location = (-200, 150)
    nt.links.new(outfit.outputs[0], mixc.inputs["Factor"])
    nt.links.new(a_n.outputs["Color"], mixc.inputs[6])
    nt.links.new(a_d.outputs["Color"], mixc.inputs[7])
    mixe = nt.nodes.new("ShaderNodeMix"); mixe.data_type = "RGBA"; mixe.location = (-200, -250)
    nt.links.new(outfit.outputs[0], mixe.inputs["Factor"])
    nt.links.new(e_n.outputs["Color"], mixe.inputs[6])
    nt.links.new(e_d.outputs["Color"], mixe.inputs[7])
    emul = nt.nodes.new("ShaderNodeMix"); emul.data_type = "RGBA"; emul.blend_type = "MULTIPLY"
    emul.inputs["Factor"].default_value = 1.0; emul.location = (80, -200)
    nt.links.new(mixc.outputs[2], emul.inputs[6])
    nt.links.new(mixe.outputs[2], emul.inputs[7])
    nt.links.new(mixc.outputs[2], bsdf.inputs["Base Color"])
    nt.links.new(emul.outputs[2], bsdf.inputs["Emission Color"])
    nt.links.new(bsdf.outputs[0], out.inputs[0])
    if rig is not None:
        _driver_prop(outfit.outputs[0], "default_value", None, rig, "outfit")
    return mat


def image_from_array(name, arr, path):
    """arr: (H,W,4) uint8, row 0 = top."""
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr, "RGBA").save(path)
    img = bpy.data.images.load(path)
    img.name = name
    img.pack()
    return img


def head_material(name, img, emit_img, tiles, rig=None, prop=None, emit_strength=6.0):
    """Image material. tiles = number of horizontal tiles (expression atlas) driven by rig[prop]."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (700, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (350, 0)
    bsdf.inputs["Roughness"].default_value = 0.6
    bsdf.inputs["Specular IOR Level"].default_value = 0.3
    bsdf.inputs["Emission Strength"].default_value = emit_strength
    uv = nt.nodes.new("ShaderNodeUVMap"); uv.location = (-700, 0)
    mp = nt.nodes.new("ShaderNodeMapping"); mp.name = "Mapping"; mp.location = (-500, 0)
    mp.inputs["Scale"].default_value = (1.0 / tiles, 1.0, 1.0)
    nt.links.new(uv.outputs[0], mp.inputs[0])
    tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = img; tex.interpolation = "Closest"
    tex.location = (-250, 100)
    nt.links.new(mp.outputs[0], tex.inputs[0])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    if emit_img is not None:
        et = nt.nodes.new("ShaderNodeTexImage"); et.image = emit_img; et.interpolation = "Closest"
        et.location = (-250, -200)
        nt.links.new(mp.outputs[0], et.inputs[0])
        nt.links.new(et.outputs["Color"], bsdf.inputs["Emission Color"])
    nt.links.new(bsdf.outputs[0], out.inputs[0])
    if rig is not None and prop is not None:
        _driver_prop(mp.inputs["Location"], "default_value", 0, rig, prop, f"round(v)/{tiles}")
    return mat


def head_mesh(name, col, center_px, half_px):
    """Cube with UVs: front face -> slot 0 full UV; others -> slot 1 tiles (L,R,Back,Top,Bottom)."""
    cx, cy, cz = center_px
    h = half_px
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.new("UVMap")

    def v(x, y, z):
        return bm.verts.new(((cx + x) * PX, (cy + y) * PX, (cz + z) * PX))

    # face: corners listed bottom-left, bottom-right, top-right, top-left as seen from outside
    spec = [
        ("front", [(-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, h)], 0, None),
        ("right", [(-h, h, -h), (-h, -h, -h), (-h, -h, h), (-h, h, h)], 1, 0),   # character right (-X)
        ("left", [(h, -h, -h), (h, h, -h), (h, h, h), (h, -h, h)], 1, 1),        # character left (+X)
        ("back", [(h, h, -h), (-h, h, -h), (-h, h, h), (h, h, h)], 1, 2),
        ("top", [(-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)], 1, 3),
        ("bottom", [(-h, h, -h), (h, h, -h), (h, -h, -h), (-h, -h, -h)], 1, 4),
    ]
    for _, corners, slot, tile in spec:
        vs = [v(*c) for c in corners]
        f = bm.faces.new(vs)
        f.material_index = slot
        uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
        for loop, (u, w) in zip(f.loops, uvs):
            if tile is None:
                loop[uvl].uv = (u, w)
            else:
                loop[uvl].uv = ((tile + u) / 5.0, w)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    col.objects.link(ob)
    return ob


# --------------------------------------------------------------------------- render
def setup_render(res=(900, 1200), samples=48):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs[0].default_value = (0.62, 0.64, 0.68, 1)
    bg.inputs[1].default_value = 0.9
    if "KeyLight" not in bpy.data.objects:
        ld = bpy.data.lights.new("KeyLight", "SUN")
        ld.energy = 3.2
        ld.angle = math.radians(12)
        lo = bpy.data.objects.new("KeyLight", ld)
        sc.collection.objects.link(lo)
        lo.rotation_euler = (math.radians(50), 0, math.radians(-30))
        fd = bpy.data.lights.new("FillLight", "SUN")
        fd.energy = 1.0
        fo = bpy.data.objects.new("FillLight", fd)
        sc.collection.objects.link(fo)
        fo.rotation_euler = (math.radians(60), 0, math.radians(150))


def render_views(path_prefix, views, target=(0, 0, 0.95), dist=4.2, ortho_scale=2.3):
    """views: list of (label, azimuth_deg, elevation_deg). azimuth 0 = front (camera at -Y)."""
    sc = bpy.context.scene
    cam_d = bpy.data.cameras.get("ReviewCam") or bpy.data.cameras.new("ReviewCam")
    cam_d.type = "ORTHO"
    cam_d.ortho_scale = ortho_scale
    cam = bpy.data.objects.get("ReviewCam") or bpy.data.objects.new("ReviewCam", cam_d)
    if cam.name not in sc.collection.objects:
        sc.collection.objects.link(cam)
    sc.camera = cam
    paths = []
    t = Vector(target)
    for label, az, el in views:
        a, e = math.radians(az), math.radians(el)
        d = Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))
        cam.location = t + d * dist
        cam.rotation_euler = (t - cam.location).to_track_quat("-Z", "Y").to_euler()
        p = f"{path_prefix}_{label}.png"
        sc.render.filepath = p
        bpy.ops.render.render(write_still=True)
        paths.append(p)
    return paths
