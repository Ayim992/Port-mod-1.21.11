"""Smooth remesh of whole body regions (hips, thighs) from the exact cube volumes.

The cubes of a bone group are voxelized in the model's own space, blurred (so the pixel steps melt
into curves) and turned back into a surface with marching cubes. Extra boxes can be added to the
volume first, e.g. to let the thighs meet in the middle like in SlipperyT's renders.
"""
import numpy as np
from scipy import ndimage
from skimage.measure import marching_cubes

from . import geckolib as gl


def _inside(pts, inv, lo, hi):
    """pts (N,3) internal px -> bool (N,), any cube contains the point"""
    out = np.zeros(len(pts), bool)
    for k in range(len(inv)):
        q = pts @ inv[k, :3, :3].T + inv[k, :3, 3]
        out |= np.all((q >= lo[k]) & (q <= hi[k]), axis=1)
    return out


def remesh(bones, order, W, names, extra_boxes=(), y_range=None, voxel=0.1, sigma_px=0.35, pad=1.0):
    """Returns (verts in Blender metres, triangle faces) or None.
    extra_boxes: [(from_xyz, to_xyz)] in Bedrock pixel coordinates (added to the volume)."""
    inv, lo, hi = gl.solid_cubes(bones, order, W, lambda n: n in names)
    if len(inv) == 0:
        return None
    # world-space bounds of all cube corners
    corners = []
    for k in range(len(inv)):
        M = np.linalg.inv(inv[k])
        for c in np.array(np.meshgrid([0, 1], [0, 1], [0, 1])).T.reshape(-1, 3):
            p = np.where(c, hi[k], lo[k])
            corners.append(M[:3, :3] @ p + M[:3, 3])
    corners = np.array(corners)
    bmin, bmax = corners.min(0) - pad, corners.max(0) + pad
    xs = [np.arange(bmin[i], bmax[i], voxel) + voxel / 2 for i in range(3)]
    G = np.stack(np.meshgrid(*xs, indexing="ij"), -1).reshape(-1, 3)
    occ = _inside(G, inv, lo, hi)
    for f, t in extra_boxes:
        fi = np.array([-t[0], f[1], f[2]], float)
        ti = np.array([-f[0], t[1], t[2]], float)
        occ |= np.all((G >= fi) & (G <= ti), axis=1)
    occ = occ.reshape(len(xs[0]), len(xs[1]), len(xs[2])).astype(np.float32)
    if y_range is not None:  # internal y is up
        ys = xs[1]
        keep = (ys >= y_range[0]) & (ys <= y_range[1])
        occ[:, ~keep, :] = 0
    occ = ndimage.gaussian_filter(occ, sigma_px / voxel)
    if occ.max() < 0.5:
        return None
    v, f, _, _ = marching_cubes(occ, 0.5, spacing=(voxel, voxel, voxel))
    v += bmin + voxel / 2
    # internal px -> Blender metres
    V = (v @ gl._M3.T) * gl.PXM
    F = f[:, ::-1]   # outward winding after the x mirror in _M3
    return V, F
