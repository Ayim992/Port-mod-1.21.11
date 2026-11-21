"""Greedy-free voxel mesher: occupancy grid -> quads with per-face colors.

Produces the stepped, pixel-edged look of Mine-imator / Blockbench props.
"""
import numpy as np

# (axis, sign) -> quad corner offsets in unit cube, counter-clockwise seen from outside
_FACES = {
    (0, +1): [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
    (0, -1): [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
    (1, +1): [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
    (1, -1): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
    (2, +1): [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
    (2, -1): [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
}


def mesh_voxels(occ, lo, vox, color_fn, emit_fn=None):
    """occ: bool (nx,ny,nz). color_fn(centers (N,3), normals (N,3)) -> (N,4) RGBA.
    Returns verts (V,3), faces (F,4), colors (F,4), emit (F,)"""
    padded = np.pad(occ, 1)
    quads, centers, normals = [], [], []
    for (ax, sgn), corners in _FACES.items():
        shift = [0, 0, 0]
        shift[ax] = sgn
        nb = np.roll(padded, -sgn, axis=ax)[1:-1, 1:-1, 1:-1]
        exposed = occ & ~nb
        idx = np.argwhere(exposed)
        if len(idx) == 0:
            continue
        c = np.asarray(corners, dtype=np.float64)
        q = (idx[:, None, :] + c[None, :, :]) * vox + lo
        quads.append(q)
        cen = (idx + 0.5) * vox + lo
        n = np.zeros_like(cen)
        n[:, ax] = sgn
        centers.append(cen + n * 0.5 * vox)
        normals.append(n)
    if not quads:
        return np.zeros((0, 3)), np.zeros((0, 4), int), np.zeros((0, 4)), np.zeros(0)
    quads = np.concatenate(quads)
    centers = np.concatenate(centers)
    normals = np.concatenate(normals)
    flat = quads.reshape(-1, 3)
    key = np.round(flat / vox * 2).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    verts = np.zeros((len(uniq), 3))
    verts[inv.ravel()] = flat
    faces = inv.reshape(-1, 4)
    cols = color_fn(centers, normals)
    emit = emit_fn(centers, normals) if emit_fn else np.zeros(len(faces))
    return verts, faces, cols, emit


def boxes_mesh(boxes):
    """boxes: list of dict(c=(x,y,z), half=(hx,hy,hz), color=(r,g,b,a), emit=0, rot=3x3 or None)
    Returns verts, faces, colors, emit."""
    verts, faces, cols, emit = [], [], [], []
    base = 0
    unit = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float64)
    fidx = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    for b in boxes:
        v = unit * np.asarray(b["half"])
        if b.get("rot") is not None:
            v = v @ np.asarray(b["rot"]).T
        v = v + np.asarray(b["c"])
        verts.append(v)
        for f in fidx:
            faces.append(tuple(i + base for i in f))
            fc = b.get("face_colors", {})
            cols.append(b["color"])
            emit.append(b.get("emit", 0.0))
        base += 8
    return np.concatenate(verts), np.asarray(faces), np.asarray(cols, dtype=np.float64), np.asarray(emit)
