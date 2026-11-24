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


def _fit_sphere(X):
    """Least-squares sphere through points X (N,3) -> (center, radius)."""
    A = np.c_[2 * X, np.ones(len(X))]
    b = (X * X).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    return c, math.sqrt(max(sol[3] + c @ c, 1e-12))


def _ray_sphere(o, D, c, R):
    """Distance along unit rays D from o (inside the sphere) to the sphere surface."""
    oc = o - c
    b = D @ oc
    q = oc @ oc - R * R
    return -b + np.sqrt(np.maximum(b * b - q, 0.0))


def radial_lobe(P, center, subdiv=5, cone_deg=2.6, smooth_iters=14, fill_scale=0.55, roundness=0.0,
                mirror_x=0, body_y=1.0):
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
    if mirror_x:
        # complete the hidden inner side (toward the other breast) from the outer side, mirrored
        # through the lobe's own center: a full breast instead of a half cut flat at the midline
        # body_y: which way the body is (+Y behind the breasts, -Y in front of the butt)
        inner = (~known) & (D[:, 0] * mirror_x < 0) & (D[:, 1] * body_y < 0.35)
        mt = cKDTree(D)
        _, mi = mt.query(D * np.array([-1.0, 1.0, 1.0]))
        take = inner & known[mi]
        r[take] = r[mi[take]]
        known = known | take
        # a full breast reaches past the midline (where the other one pushes back): stretch the
        # mirrored inner side so it overlaps by ~0.35 px; squish_against_plane then flattens the contact
        if take.any():
            reach = (D[take, 0] * -mirror_x) * r[take]          # how far toward the midline it goes
            need = abs(center[0]) + 0.35 * PXM
            k = need / max(reach.max(), 1e-9)
            if k > 1:
                w = np.clip(-D[:, 0] * mirror_x, 0, 1)            # 0 at the front, 1 straight inward
                r[take] = r[take] * (1 + (k - 1) * w[take])
    base = np.nanmedian(r) * fill_scale
    r[~known] = base
    # diffuse into the unseen (inside-the-body) directions, then shrink them so they stay hidden
    r = _laplace(r, E, len(D), 80, 0.5, fixed=known)
    r[~known] = np.minimum(r[~known], base * 1.15)
    if roundness > 0:
        # pull the shape toward the best-fit sphere of its visible surface (film Jenny's round breasts)
        sc, sr = _fit_sphere(center + D[known] * r[known, None])
        if np.linalg.norm(center - sc) < sr:
            rs = _ray_sphere(center, D, sc, sr)
            blend = np.where(known, roundness, 0.0)
            # unseen (inside-body) directions may only shrink toward the sphere
            r_new = r * (1 - blend) + rs * blend
            r_new[~known] = np.minimum(r[~known], rs[~known])
            r = r_new
    r = _laplace(r, E, len(D), smooth_iters, 0.5)
    V = center + D * r[:, None]
    return V, F, E, known


def squish_against_plane(V, E, side, gap, bulge=0.35, iters=12, out_y=-1.0):
    """Soft-body style contact: everything that crosses the midline (x=0) toward the other breast is
    flattened onto the contact plane, the displaced flesh pushes forward and outward a little, and the
    surface around the contact is relaxed so the cleavage reads as two soft breasts pressed together."""
    V = V.copy()
    x = V[:, 0] * side
    pen = np.maximum(0.0, gap - x)
    if not pen.any():
        return V
    # flatten onto the contact plane
    V[:, 0] += side * pen
    # volume goes forward (-Y) and slightly outward/up, scaled by how deep the contact was
    V[:, 1] += out_y * bulge * pen   # displaced flesh bulges away from the body
    V[:, 2] += 0.15 * bulge * pen
    # relax the skin around the contact, keep the contact plane
    zone = np.abs(V[:, 0]) < (gap + 0.6 * PXM)
    for _ in range(iters):
        V = _laplace(V, E, len(V), 1, 0.5, fixed=~zone)
        x = V[:, 0] * side
        V[:, 0] += side * np.maximum(0.0, gap - x)
    return V


# --------------------------------------------------------------------------- build
def _mesh(name, V, F):
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], F.tolist())
    me.update()
    me.shade_smooth()
    return me


SKIN_REF = [None]


def waistband_sheet(me, body, dark, cen, tex, gnames, col, z_lo_ref):
    """A smooth pants band over the lower back, replacing the jagged waistband cubes: a thin slab on
    the back plane from just below the cheeks' top to the top of the waistband, rounded top edge."""
    ids = np.nonzero(dark)[0]
    co = np.array([v.co[:] for v in me.vertices])
    vids = np.unique(np.concatenate([list(me.polygons[i].vertices) for i in ids]))
    pts = co[vids]
    yb = float(np.median(cen[ids, 1])) + 0.06 * PXM
    z0, z1 = min(z_lo_ref - 2.5 * PXM, pts[:, 2].min()), pts[:, 2].max()
    # width of the back itself just above the band (the side straps must not widen it)
    nrm = np.array([p.normal[:] for p in me.polygons])
    above = (nrm[:, 1] > 0.5) & (cen[:, 2] > z1) & (cen[:, 2] < z1 + 1.5 * PXM) & (np.abs(cen[:, 0]) < 3.5 * PXM)
    if above.any():
        av = np.unique(np.concatenate([list(me.polygons[i].vertices) for i in np.nonzero(above)[0]]))
        half = np.abs(co[av, 0]).max() - 0.05 * PXM
    else:
        half = np.abs(pts[:, 0]).max()
    x0, x1 = -half, half
    nx, nz = 48, 16
    xs = np.linspace(x0, x1, nx)
    zs = np.linspace(z0, z1, nz)
    V, F = [], []
    # outer sheet; the top two rows curl into the body for a rounded edge
    for j, z in enumerate(zs):
        curl = max(0.0, (j - (nz - 3)) / 2.0)
        for x in xs:
            edge = min(1.0, min(x - x0, x1 - x) / (0.5 * PXM))     # ends tuck in too
            V.append((x, yb - 0.25 * PXM * (curl ** 2) - 0.25 * PXM * (1 - edge) ** 2, z - 0.1 * PXM * curl))
    for j in range(nz - 1):
        for i in range(nx - 1):
            a = j * nx + i
            F.append((a, a + 1, a + nx + 1))
            F.append((a, a + nx + 1, a + nx))
    V = np.array(V)
    F = np.array(F)
    smp = sample_surface(body, dark, tex)
    if smp is None:
        return None
    P, _, _, B = smp
    _, nn = cKDTree(P).query(V)
    bone_v = B[nn]
    names = sorted({gnames[b] for b in np.unique(bone_v)})
    W = np.zeros((len(V), len(names)))
    for j, n in enumerate(names):
        gi = next(k for k, v in gnames.items() if v == n)
        W[:, j] = bone_v == gi
    return V, F, np.tile(col, (len(V), 1)), (names, W)


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
        dark = np.zeros(len(me.polygons), bool)
        if lb.get("absorb_up"):
            # pull the dark (pants) pieces just above the lobe into it, so the smooth surface runs
            # up to the top of the waistband instead of stopping under a jagged ledge
            cen = np.array([p.center[:] for p in me.polygons])
            nrm = np.array([p.normal[:] for p in me.polygons])
            ztop = cen[mask, 2].max()
            xw = np.abs(cen[mask, 0]).max() + 0.3 * PXM
            ymid = cen[mask, 1].mean()
            cand = (~mask) & (cen[:, 2] > ztop - 1.5 * PXM) & (cen[:, 2] < ztop + lb["absorb_up"] * PXM) \
                & (np.abs(cen[:, 0]) < xw) & (cen[:, 1] > 0.5 * PXM) & (nrm[:, 1] > 0.3)
            if cand.any():
                H, Wt = tex.shape[:2]
                uvd = me.uv_layers[0].data
                for i in np.nonzero(cand)[0]:
                    p = me.polygons[i]
                    uv = np.mean([uvd[l].uv[:] for l in p.loop_indices], axis=0)
                    px = tex[min(H - 1, int((1 - uv[1]) * H)), min(Wt - 1, int(uv[0] * Wt))]
                    dark[i] = px[3] > 128 and px[:3].mean() < 70
                dark &= nrm[:, 1] > 0.5          # only the back-facing waistband pieces
                print(f"[smooth] {lb['name']}: {int(dark.sum())} waistband faces replaced", flush=True)
        smp = sample_surface(body, mask, tex)
        if smp is None:
            continue
        P, C, N, B = smp
        center = P.mean(axis=0) + np.asarray(lb.get("center_shift", (0, 0, 0)))
        Pall, Ball = P, B
        if lb.get("full_breast") and lb.get("split_x"):
            # the mod closes each breast with a flat wall at the cleavage: leave it out of the shape
            # (it is rebuilt by mirroring the outer side), but keep it for colour/bone lookups
            keep = P[:, 0] * lb["split_x"] > lb.get("mid_exclude", 0.9) * PXM
            P = P[keep]
        res = radial_lobe(P, center, lb.get("subdiv", 5), lb.get("cone", 2.6), lb.get("smooth", 14),
                          roundness=lb.get("roundness", 0.0),
                          mirror_x=lb.get("split_x", 0) if lb.get("full_breast") else 0,
                          body_y=lb.get("body_y", 1.0))
        if res is None:
            continue
        V, F, E, known = res
        P = Pall
        if lb.get("keep_top"):
            # keep the original cubes along the top edge (e.g. the pants waistband): smoothing rounds
            # that edge down and would leave notches where skin shows through
            cz = np.array([p.center.z for p in me.polygons])
            ztop = cz[mask].max()
            mask = mask & (cz < ztop - lb["keep_top"] * PXM)
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
        if lb.get("uniform_color"):
            col_v[:] = np.median(col_v, axis=0)
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
        if lb.get("soft_nipple") and any("nipple" in gnames.get(b, "").lower() for b in np.unique(B)):
            # SlipperyT Jenny: no areola disc. The skin around the nipple takes the plain skin colour,
            # the mod's raised areola plate is relaxed away *radially* (a position Laplacian flattens
            # the patch into a disc that shades darker), and the nipple is a small gaussian swell
            # that blends into the breast with no rim.
            nipm = np.array(["nipple" in gnames.get(b, "").lower() for b in B])
            tip = P[nipm].mean(axis=0)
            d = np.linalg.norm(V - tip, axis=1)
            ring = (d > 2.0 * PXM) & (d < 2.6 * PXM)
            if ring.any():
                skin = np.median(col_v[ring], axis=0)
                SKIN_REF[0] = skin
                core = d < 0.5 * PXM
                acol0 = np.median(col_v[core], axis=0) if core.any() else skin * 0.85
                if lb.get("uniform_skin"):
                    # one plain skin colour over the whole breast; only areola + nipple differ
                    col_v[:] = skin
                else:
                    col_v[d < 2.0 * PXM] = skin
                zone = d < 2.2 * PXM
                rel = V - center
                rr = np.linalg.norm(rel, axis=1)
                dirs = rel / rr[:, None]
                rr = _laplace(rr, E, len(V), 60, 0.5, fixed=~zone)
                V = center + dirs * rr[:, None]
                col_v = _laplace(col_v, E, len(V), 3, 0.5, fixed=~(d < 2.4 * PXM))
                tip = V[np.argmin(np.linalg.norm(V - tip, axis=1))]
                dd = np.linalg.norm(V - tip, axis=1)
                ar = lb.get("areola_radius", 0.0) * PXM
                if ar > 0:
                    # flat, plain, darker disc with a crisp edge (colour only, follows the breast curve)
                    acol = acol0 * lb.get("areola_dark", 1.0)
                    t = np.clip((ar - dd) / (0.1 * PXM) + 0.5, 0, 1)[:, None]
                    col_v = col_v * (1 - t) + acol * t
                nr = lb.get("nipple_radius", 0.0) * PXM
                h = lb.get("nipple_height", 0.0) * PXM
                if nr > 0 and h > 0:
                    # small round nipple standing out of the areola: flat-topped with a rounded edge
                    edge = 0.35 * nr
                    x = np.clip((nr - dd) / edge, 0, 1)
                    prof = x * x * (3 - 2 * x)
                    V = V + dirs * (h * prof)[:, None]
                    ncol = acol0 * lb.get("areola_dark", 1.0) * lb.get("nipple_dark", 0.92)
                    t = np.clip((nr + 0.03 * PXM - dd) / (0.04 * PXM), 0, 1)[:, None]
                    col_v = col_v * (1 - t) + ncol * t
        if lb.get("full_breast") and lb.get("split_x"):
            # push the two full breasts into each other, then let the contact flatten them
            # (stretch the inner side toward the midline about the outer edge, so the outer
            # silhouette stays exactly where the original model has it)
            V = V.copy()
            sx = lb["split_x"]
            xs = V[:, 0] * sx
            xo, xi = xs.max(), xs.min()
            k = (xo - xi + lb.get("press", 0.0) * PXM) / max(xo - xi, 1e-9)
            V[:, 0] = sx * (xo - (xo - xs) * k)
            V = squish_against_plane(V, E, lb["split_x"], gap=0.03 * PXM, bulge=lb.get("bulge", 0.35),
                                     out_y=-lb.get("body_y", 1.0))
        if lb.get("skin_match") and SKIN_REF[0] is not None:
            # same plain skin colour as the breasts (the mod texture paints these darker and orange)
            col_v[:] = SKIN_REF[0]
        if lb.get("flatten"):
            # press the mound back into the body: squash depth toward its deepest (inner) side
            fy, fz = lb["flatten"]
            yb = V[:, 1].max()
            V = V.copy()
            V[:, 1] = yb - (yb - V[:, 1]) * fy
            zc = V[:, 2].mean()
            V[:, 2] = zc + (V[:, 2] - zc) * fz
            if lb.get("slit") and SKIN_REF[0] is not None:
                # soft cleft down the middle of the front
                front = V[:, 1] < (yb - 0.35 * (yb - V[:, 1].min()))
                w = np.clip(1 - np.abs(V[:, 0]) / (lb["slit"] * PXM), 0, 1) * front
                V[:, 1] += 0.18 * PXM * w
                col_v = col_v * (1 - 0.35 * w[:, None]) + np.array([0.80, 0.45, 0.42]) * (0.35 * w[:, None])
        if lb.get("scale") or lb.get("offset"):
            # reshape the lobe toward the film proportions: scale about its own base, then move
            sc = np.asarray(lb.get("scale", 1.0), dtype=np.float64)
            V = center + (V - center) * sc + np.asarray(lb.get("offset", (0, 0, 0)))
        if lb.get("absorb_up") and dark.any():
            region_mask |= dark
            wb = waistband_sheet(me, body, dark, cen, tex, gnames, np.median(col_v, axis=0), ztop)
            if wb is not None:
                wV, wF, wC, (wnames, wW) = wb
                allV.append(wV)
                allF.append(wF + sum(len(v) for v in allV[:-1]))
                allC.append(wC)
                allW.append((wnames, wW))
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
