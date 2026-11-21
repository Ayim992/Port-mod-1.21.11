"""Vectorized paint helpers. Inputs are face centers P (F,3) in px and normals N (F,3)."""
import numpy as np


def hx(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


class Layer:
    """Accumulates color + emission for a face set."""

    def __init__(self, n, base):
        self.rgb = np.tile(hx(base), (n, 1))
        self.emit = np.zeros(n)

    def put(self, mask, col, emit=None):
        if np.isscalar(mask) and mask:
            mask = np.ones(len(self.rgb), bool)
        self.rgb[mask] = hx(col)
        if emit is not None:
            self.emit[mask] = emit

    def blend(self, mask_w, col):
        w = np.clip(mask_w, 0, 1)[:, None]
        self.rgb = self.rgb * (1 - w) + hx(col)[None, :] * w

    def out(self):
        return self.rgb, self.emit


def hash2(a, b, seed=0):
    """Deterministic pseudo-random in [0,1) from integer grids."""
    h = (a.astype(np.int64) * 73856093) ^ (b.astype(np.int64) * 19349663) ^ (seed * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFF) / 65536.0


def cell(v, size=0.5):
    return np.floor(v / size).astype(np.int64)


def stripes(v, period, duty=0.5, offset=0.0):
    return ((v - offset) / period) % 1.0 < duty


def jagged_below(P, z_line, amp=0.5, cell_size=0.5, seed=1, axis_u=0):
    """Mask of points below a pixel-jagged horizontal edge."""
    u = cell(P[:, axis_u], cell_size)
    jit = (hash2(u, np.zeros_like(u), seed) - 0.5) * 2 * amp
    jit = np.round(jit / cell_size) * cell_size
    return P[:, 2] < z_line + jit


def disc_on_surface(P, N, center, radius, facing=None):
    """Points within radius of center (3D distance); optionally require normal facing a direction."""
    d = np.linalg.norm(P - center, axis=1)
    m = d < radius
    if facing is not None:
        m &= (N @ np.asarray(facing)) > 0.0
    return m


def front(N, thr=-0.2):
    return N[:, 1] < thr


def back(N, thr=0.2):
    return N[:, 1] > thr


def vulva(L, P, lm, inner="c56a72", outer="eaa69c", width=0.55):
    if "vulva_top" not in lm:
        return
    t, b = lm["vulva_top"], lm["vulva_bottom"]
    zlo, zhi = b[2] - 0.2, t[2] + 0.15
    region = (P[:, 2] > zlo) & (P[:, 2] < zhi) & (P[:, 1] < t[1] + 1.4) & (P[:, 1] > t[1] - 0.6)
    L.put(region & (np.abs(P[:, 0]) < width), outer)
    L.put(region & (np.abs(P[:, 0]) < 0.2), inner)


def nipples(L, P, N, lm, areola, nipple, r_areola=0.78, r_nip=0.34, emit=None):
    for n in lm["nipples"]:
        L.put(disc_on_surface(P, N, n, r_areola, (0, -1, 0)), areola, emit)
        L.put(disc_on_surface(P, N, n, r_nip, (0, -1, 0)), nipple, emit)


def navel(L, P, lm, col, r=0.32):
    if "navel" in lm:
        L.put(np.linalg.norm(P - lm["navel"], axis=1) < r, col)
