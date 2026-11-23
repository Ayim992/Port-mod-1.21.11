"""Smooth replacements for the round parts of the GeckoLib models (breasts, butt, vulva).

The mod builds these shapes as shells of thin alpha-cut planes, which read as faceted steps up close.
For each lobe we:
  1. sample the visible (opaque-texel) surface of the original cubes at sub-pixel density,
  2. take the outermost surface along every direction from a center inside the lobe (a star-shaped
     radial field on an icosphere), so the silhouette and volume stay the same,
  3. smooth that field to remove the steps, while directions with no surface (inside the body) are filled,
  4. colour each new vertex from the nearest original texel, and weight it to the bones of the
     nearest original cubes, smoothed over the new surface.
The blocky originals are kept in a separate object; the rig's `smooth` property switches between them.
"""
import math

import numpy as np
import bmesh
import bpy
from mathutils import Vector
from scipy.spatial import cKDTree

from .geckolib import PXM


# --------------------------------------------------------------------------- sampling
def vertex_bone_index(ob):
    me = ob.data
    vb = np.full(len(me.vertices), -1, np.int32)
    for v in me.vertices:
        if v.groups:
            vb[v.index] = v.groups[0].group
    return vb


def sample_surface(ob, face_mask, tex, step_px=0.08, alpha_min=128):
    """Sample opaque texels on the selected faces of `ob` (rest pose).
    Returns positions (N,3) m, colours (N,3) sRGB 0..1, face normals (N,3), vertex-group index (N,)."""
    me = ob.data
    H, W = tex.shape[:2]
    uvd = me.uv_layers[0].data
    vb = vertex_bone_index(ob)
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    P, C, Nn, B = [], [], [], []
    step = step_px * PXM
    for p in me.polygons:
        if not face_mask[p.index] or p.loop_total != 4:
            continue
        li = list(p.loop_indices)
        vi = list(p.vertices)
        q = co[vi]
        uv = np.array([uvd[l].uv for l in li])
        a, b, d = q[0], q[1], q[3]
        nu = max(2, int(math.ceil(np.linalg.norm(b - a) / step)))
        nv = max(2, int(math.ceil(np.linalg.norm(d - a) / step)))
        s = (np.arange(nu) + 0.5) / nu
        t = (np.arange(nv) + 0.5) / nv
        S_, T_ = np.meshgrid(s, t, indexing="ij")
        S_, T_ = S_.ravel()[:, None], T_.ravel()[:, None]
        pos = (1 - S_) * (1 - T_) * q[0] + S_ * (1 - T_) * q[1] + S_ * T_ * q[2] + (1 - S_) * T_ * q[3]
        tuv = (1 - S_) * (1 - T_) * uv[0] + S_ * (1 - T_) * uv[1] + S_ * T_ * uv[2] + (1 - S_) * T_ * uv[3]
        x = np.clip((tuv[:, 0] * W).astype(int), 0, W - 1)
        y = np.clip(((1 - tuv[:, 1]) * H).astype(int), 0, H - 1)
        px = tex[y, x]
        keep = px[:, 3] >= alpha_min
        if not keep.any():
            continue
        P.append(pos[keep])
        C.append(px[keep, :3] / 255.0)
        Nn.append(np.repeat(np.array(p.normal)[None], keep.sum(), axis=0))
        B.append(np.full(keep.sum(), vb[vi[0]]))
    if not P:
        return None
    return np.concatenate(P), np.concatenate(C), np.concatenate(Nn), np.concatenate(B)


# --------------------------------------------------------------------------- radial lobe
def icosphere(subdiv):
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=subdiv, radius=1.0)
    bm.verts.ensure_lookup_table()
    V = np.array([v.co[:] for v in bm.verts])
    F = np.array([[v.index for v in f.verts] for f in bm.faces])
    E = np.array([[e.verts[0].index, e.verts[1].index] for e in bm.edges])
    bm.free()
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return V, F, E


def _laplace(values, E, n, iters, factor=0.5, fixed=None):
    deg = np.bincount(E.ravel(), minlength=n).astype(np.float64)
    v = values.copy()
    for _ in range(iters):
        acc = np.zeros_like(v)
        np.add.at(acc, E[:, 0], v[E[:, 1]])
        np.add.at(acc, E[:, 1], v[E[:, 0]])
        avg = acc / np.maximum(deg, 1)[(slice(None),) + (None,) * (v.ndim - 1)]
        nv = v + factor * (avg - v)
        if fixed is not None:
            nv[fixed] = v[fixed]
        v = nv
    return v


def radial_lobe(P, center, subdiv=5, cone_deg=2.6, smooth_iters=14, fill_scale=0.55):
    D, F, E = icosphere(subdiv)
    rel = P - center
    rad = np.linalg.norm(rel, axis=1)
    dirs = rel / np.maximum(rad, 1e-9)[:, None]
    tree = cKDTree(dirs)
    chord = 2 * math.sin(math.radians(cone_deg) / 2)
    hits = tree.query_ball_point(D, chord)
    r = np.full(len(D), np.nan)
    for i, h in enumerate(hits):
        if h:
            r[i] = rad[h].max()
    known = ~np.isnan(r)
    if known.sum() < 10:
        return None
    base = np.nanmedian(r) * fill_scale
    r[~known] = base
    # diffuse into the unseen (inside-the-body) directions, then shrink them so they stay hidden
    r = _laplace(r, E, len(D), 80, 0.5, fixed=known)
    r[~known] = np.minimum(r[~known], base * 1.15)
    r = _laplace(r, E, len(D), smooth_iters, 0.5)
    V = center + D * r[:, None]
    return V, F, E, known


# --------------------------------------------------------------------------- build
def _mesh(name, V, F):
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], F.tolist())
    me.update()
    me.shade_smooth()
    return me


def build_smooth(title, col, body, rig, lobes, tex, material):
    """lobes: list of dict(name, bones(set), center_shift(vec m), split_x(+1/-1/0), subdiv, smooth, cone).
    Returns (smooth_object, region_face_mask)."""
    me = body.data
    vb = vertex_bone_index(body)
    gnames = {g.index: g.name for g in body.vertex_groups}
    face_bone = np.array([gnames.get(vb[p.vertices[0]], "") for p in me.polygons])
    allV, allF, allC, allW = [], [], [], []
    region_mask = np.zeros(len(me.polygons), bool)
    wnames = []
    for lb in lobes:
        mask = np.isin(face_bone, list(lb["bones"]))
        if lb.get("split_x"):
            cx = np.array([p.center.x for p in me.polygons])
            mask &= (cx * lb["split_x"]) > -1e-4
        if not mask.any():
            continue
        smp = sample_surface(body, mask, tex)
        if smp is None:
            continue
        P, C, N, B = smp
        center = P.mean(axis=0) + np.asarray(lb.get("center_shift", (0, 0, 0)))
        res = radial_lobe(P, center, lb.get("subdiv", 5), lb.get("cone", 2.6), lb.get("smooth", 14))
        if res is None:
            continue
        V, F, E, known = res
        region_mask |= mask
        # colour from outward-facing samples only (side faces of plates carry darker edge texels),
        # averaged over the nearest few; bones from the nearest sample of any kind
        radial = (P - center) / np.maximum(np.linalg.norm(P - center, axis=1, keepdims=True), 1e-9)
        outward = np.einsum("ij,ij->i", N, radial) > 0.35
        if outward.sum() < 50:
            outward[:] = True
        Po, Co = P[outward], C[outward]
        ctree = cKDTree(Po)
        dist, cidx = ctree.query(V, k=4)
        w = 1.0 / np.maximum(dist, 1e-5)
        col_v = (Co[cidx] * w[..., None]).sum(axis=1) / w.sum(axis=1, keepdims=True)
        tree = cKDTree(P)
        _, pick = tree.query(V, k=1)
        bone_v = B[pick]
        names = sorted({gnames[b] for b in np.unique(bone_v)})
        Wl = np.zeros((len(V), len(names)))
        for j, n in enumerate(names):
            gi = next(k for k, v in gnames.items() if v == n)
            Wl[:, j] = bone_v == gi
        Wl = _laplace(Wl, E, len(V), lb.get("weight_smooth", 6), 0.5)
        Wl /= np.maximum(Wl.sum(axis=1, keepdims=True), 1e-9)
        if lb.get("areola_scale") and any("nipple" in gnames.get(b, "").lower() for b in np.unique(B)):
            # SlipperyT-style areola: wider, flat, same colour as the model's own areola
            nipm = np.array(["nipple" in gnames.get(b, "").lower() for b in B])
            tip = P[nipm].mean(axis=0)
            d = np.linalg.norm(V - tip, axis=1)
            core = d < 0.55 * PXM
            if core.any():
                acol = np.median(col_v[core], axis=0)
                R = lb["areola_scale"] * PXM
                # skin colour from a ring well outside the original areola: lets the areola shrink too
                ring = (d > 2.0 * PXM) & (d < 2.6 * PXM)
                skin = np.median(col_v[ring], axis=0) if ring.any() else None
                if skin is not None:
                    outer = (d > R) & (d < 1.9 * PXM)
                    col_v[outer] = skin
                t = np.clip((R - d) / (0.15 * PXM), 0, 1)[:, None]
                col_v = col_v * (1 - t) + acol * t
                # flatten the old raised areola plate so its edge doesn't show as a ridge,
                # and soften the colour edge
                zone = d < 2.2 * PXM
                V = _laplace(V, E, len(V), 40, 0.5, fixed=~zone)
                col_v = _laplace(col_v, E, len(V), 2, 0.5, fixed=~(d < 2.4 * PXM))
                # small raised nipple: a soft dome over the areola, pushed out along the surface normal
                h = lb.get("nipple_height", 0.0) * PXM
                if h > 0:
                    nr = R * 1.05
                    dd = np.linalg.norm(V - tip, axis=1)
                    prof = np.where(dd < nr, 0.5 * (1 + np.cos(np.pi * dd / nr)), 0.0)
                    nrm = (V - center) / np.linalg.norm(V - center, axis=1, keepdims=True)
                    V = V + nrm * (h * prof)[:, None]
        allV.append(V)
        allF.append(F + sum(len(v) for v in allV[:-1]))
        allC.append(col_v)
        allW.append((names, Wl))
        print(f"[smooth] {lb['name']}: {len(P)} samples, {known.mean():.0%} of directions seen, "
              f"{len(V)} verts, bones {names[:6]}{'…' if len(names) > 6 else ''}", flush=True)
    if not allV:
        return None, region_mask
    V = np.concatenate(allV)
    F = np.concatenate(allF)
    sm = _mesh(f"{title}_Smooth", V, F)
    ob = bpy.data.objects.new(f"{title}_Smooth", sm)
    col.objects.link(ob)
    # colour attribute (linear)
    Cs = np.concatenate(allC)
    lin = np.where(Cs <= 0.04045, Cs / 12.92, ((Cs + 0.055) / 1.055) ** 2.4)
    attr = sm.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    attr.data.foreach_set("color", np.concatenate([lin, np.ones((len(lin), 1))], axis=1).astype(np.float32).ravel())
    # weights
    off = 0
    for (names, Wl), Vl in zip(allW, allV):
        for j, n in enumerate(names):
            vg = ob.vertex_groups.get(n) or ob.vertex_groups.new(name=n)
            w = Wl[:, j]
            for val in np.unique(np.round(w[w > 0.01], 3)):
                ids = (np.nonzero(np.round(w, 3) == val)[0] + off).tolist()
                vg.add(ids, float(val), "REPLACE")
        off += len(Vl)
    sm.materials.append(material)
    m = ob.modifiers.new("Armature", "ARMATURE")
    m.object = rig
    ob.parent = rig
    return ob, region_mask


def split_blocky(body, region_mask, title, col):
    """Move the replaced faces out of the body into their own object (kept for the blocky look)."""
    blocky = body.copy()
    blocky.data = body.data.copy()
    blocky.name = f"{title}_BlockyParts"
    blocky.data.name = blocky.name
    col.objects.link(blocky)
    for ob, keep in ((body, ~region_mask), (blocky, region_mask)):
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        bm.faces.ensure_lookup_table()
        kill = [f for f in bm.faces if not keep[f.index]]
        bmesh.ops.delete(bm, geom=kill, context="FACES")
        bm.to_mesh(ob.data)
        bm.free()
    return blocky


def color_material(name, roughness=0.6):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    a = nt.nodes.new("ShaderNodeAttribute")
    a.attribute_name = "Col"
    a.location = (-400, 200)
    nt.links.new(a.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Specular IOR Level"].default_value = 0.3
    return mat
