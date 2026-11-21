"""Numpy signed-distance primitives and operators.

All coordinates are in Minecraft pixel units (1 px = 1/32 of character height).
Character faces -Y, character's left is +X, feet at Z=0.
Every primitive takes p with shape (..., 3) and returns distances with shape (...).
"""
import numpy as np


def _rel(p, c):
    return p - np.asarray(c, dtype=np.float64)


def rot_matrix(rx=0.0, ry=0.0, rz=0.0):
    """XYZ euler (degrees) -> 3x3 rotation matrix."""
    rx, ry, rz = np.radians([rx, ry, rz])
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return mz @ my @ mx


def _local(p, c, rot):
    q = _rel(p, c)
    if rot is not None:
        q = q @ np.asarray(rot)  # inverse rotation (row vectors)
    return q


def sphere(p, c, r):
    return np.linalg.norm(_rel(p, c), axis=-1) - r


def ellipsoid(p, c, r, rot=None):
    q = _local(p, c, rot)
    r = np.asarray(r, dtype=np.float64)
    k0 = np.linalg.norm(q / r, axis=-1)
    k1 = np.linalg.norm(q / (r * r), axis=-1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def round_box(p, c, half, radius=0.0, rot=None):
    q = np.abs(_local(p, c, rot)) - (np.asarray(half, dtype=np.float64) - radius)
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside - radius


def capsule(p, a, b, r):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    pa = p - a
    ba = b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0.0, 1.0)
    return np.linalg.norm(pa - h[..., None] * ba, axis=-1) - r


def cylinder_z(p, c, r, half_h):
    q = _rel(p, c)
    d = np.stack([np.linalg.norm(q[..., :2], axis=-1) - r, np.abs(q[..., 2]) - half_h], axis=-1)
    return np.minimum(np.max(d, axis=-1), 0.0) + np.linalg.norm(np.maximum(d, 0.0), axis=-1)


def loft(p, table, cap_round=0.0):
    """Vertical superellipse loft.

    table rows: (z, cx, cy, half_w, front, back, exponent[, half_w_negx])
      front = extent toward -Y, back = extent toward +Y; half_w_negx overrides the
      extent toward -X (used to keep a thigh gap while thighs stay thick on the outside).
    Cross sections are interpolated linearly in z. Flat caps at the ends
    (optionally rounded by cap_round).
    """
    t = np.asarray(table, dtype=np.float64)
    z = p[..., 2]
    zs = t[:, 0]
    cx = np.interp(z, zs, t[:, 1])
    cy = np.interp(z, zs, t[:, 2])
    hw = np.interp(z, zs, t[:, 3])
    fr = np.interp(z, zs, t[:, 4])
    bk = np.interp(z, zs, t[:, 5])
    n = np.interp(z, zs, t[:, 6])
    x = p[..., 0] - cx
    y = p[..., 1] - cy
    depth = np.where(y < 0.0, fr, bk)
    if t.shape[1] > 7:  # optional separate half width toward -X
        hw = np.where(x < 0.0, np.interp(z, zs, t[:, 7]), hw)
    u = np.abs(x) / hw
    v = np.abs(y) / depth
    rho = (u ** n + v ** n) ** (1.0 / n)
    scale = np.minimum(hw, depth)
    d_xy = (rho - 1.0) * scale
    d_z = np.maximum(zs[0] - z, z - zs[-1])
    if cap_round > 0.0:
        return smax(d_xy, d_z, cap_round)
    return np.maximum(d_xy, d_z)


def smin(a, b, k):
    if k <= 0.0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def union(*ds):
    out = ds[0]
    for d in ds[1:]:
        out = np.minimum(out, d)
    return out


class Shape:
    """Lazy SDF tree so the same description can be evaluated on any grid."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, p):
        return self.fn(p)

    def smin(self, other, k):
        return Shape(lambda p: smin(self(p), other(p), k))

    def smax(self, other, k):
        return Shape(lambda p: smax(self(p), other(p), k))

    def __or__(self, other):
        return Shape(lambda p: np.minimum(self(p), other(p)))

    def __sub__(self, other):
        return Shape(lambda p: np.maximum(self(p), -other(p)))

    def cut(self, other, k):
        return Shape(lambda p: smax(self(p), -other(p), k))


def S(fn, *args, **kw):
    return Shape(lambda p: fn(p, *args, **kw))


def mirror_x(fn_factory):
    """Union of a shape built for +X and its mirror."""
    right = fn_factory(1.0)
    left = fn_factory(-1.0)
    return right | left


def grid_eval(shape, lo, hi, step):
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    dims = np.ceil((hi - lo) / step).astype(int) + 1
    axes = [lo[i] + np.arange(dims[i]) * step for i in range(3)]
    out = np.empty(dims, dtype=np.float32)
    # evaluate slab by slab along z to bound memory
    gx, gy = np.meshgrid(axes[0], axes[1], indexing="ij")
    for k, zv in enumerate(axes[2]):
        pts = np.stack([gx, gy, np.full_like(gx, zv)], axis=-1)
        out[:, :, k] = shape(pts)
    return out, lo


def mesh_from_sdf(shape, lo, hi, step):
    from skimage.measure import marching_cubes

    vol, origin = grid_eval(shape, lo, hi, step)
    verts, faces, _, _ = marching_cubes(vol, level=0.0, spacing=(step, step, step))
    verts += origin
    # marching_cubes winding is inward for "negative inside"; flip
    faces = faces[:, ::-1]
    return verts, faces


def voxelize(shape, lo, hi, vox):
    """Boolean occupancy on a grid of cube size `vox`, sampled at cell centers."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    dims = np.ceil((hi - lo) / vox).astype(int)
    axes = [lo[i] + (np.arange(dims[i]) + 0.5) * vox for i in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return shape(g) <= 0.0, lo


def round_cone(p, a, b, r1, r2):
    """Capsule with different end radii (Inigo Quilez)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ba = b - a
    l2 = ba @ ba
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = p - a
    y = pa @ ba
    z = y - l2
    xv = pa * l2 - y[..., None] * ba
    x2 = np.sum(xv * xv, axis=-1)
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    d3 = (np.sqrt(x2 * a2 * il2) + y * rr) * il2 - r1
    d1 = np.sqrt(x2 + z2) * il2 - r2
    d2 = np.sqrt(x2 + y2) * il2 - r1
    out = np.where(np.sign(y) * a2 * y2 < k, d2, d3)
    out = np.where(np.sign(z) * a2 * z2 > k, d1, out)
    return out


def chain(p, pts, radii):
    """Union of round cones through a polyline with per-point radii."""
    d = None
    for i in range(len(pts) - 1):
        di = round_cone(p, pts[i], pts[i + 1], radii[i], radii[i + 1])
        d = di if d is None else np.minimum(d, di)
    return d
