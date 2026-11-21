"""Pixel-art head textures. 32x32 texels per head face (4 texels per Minecraft pixel).

Expression atlas order (rig property `expression`):
  0 neutral   1 blink   2 lidded (bedroom eyes)   3 smile (open mouth)
  4 moan      5 ahegao (eyes rolled up, tongue out)   6 heart eyes
"""
import numpy as np

EXPRESSIONS = ["neutral", "blink", "lidded", "smile", "moan", "ahegao", "hearts"]
T = 32


def rgba(h, a=255):
    h = h.lstrip("#")
    return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a], np.uint8)


class Canvas:
    def __init__(self, base, w=T, h=T):
        self.c = np.zeros((h, w, 4), np.uint8)
        self.c[:] = rgba(base)
        self.e = np.zeros((h, w, 4), np.uint8)
        self.e[..., 3] = 255

    def rect(self, x0, y0, x1, y1, col, emit=False, alpha=1.0):
        """Inclusive texel rect, row 0 = top."""
        x0, x1 = max(0, min(x0, x1)), min(self.c.shape[1] - 1, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(self.c.shape[0] - 1, max(y0, y1))
        cc = rgba(col) if isinstance(col, str) else np.asarray(col, np.uint8)
        reg = self.c[y0:y1 + 1, x0:x1 + 1]
        if alpha >= 1.0:
            reg[:] = cc
        else:
            reg[..., :3] = (reg[..., :3] * (1 - alpha) + cc[:3] * alpha).astype(np.uint8)
        if emit:
            self.e[y0:y1 + 1, x0:x1 + 1, :3] = cc[:3]

    def px(self, x, y, col, emit=False, alpha=1.0):
        self.rect(x, y, x, y, col, emit, alpha)

    def pts(self, pts, col, emit=False):
        for x, y in pts:
            self.px(x, y, col, emit)

    def noise(self, col, amount, seed=0, mask=None):
        rng = np.random.default_rng(seed)
        n = rng.random(self.c.shape[:2]) < amount
        if mask is not None:
            n &= mask
        self.c[n, :3] = rgba(col)[:3]

    def mirror_copy(self):
        """Mirror left half onto right half (for symmetric faces)."""
        w = self.c.shape[1]
        self.c[:, w // 2:] = self.c[:, :w // 2][:, ::-1]
        self.e[:, w // 2:] = self.e[:, :w // 2][:, ::-1]


# ----------------------------------------------------------------------------- eyes
def eye(cv, x0, y0, style, expr, flip=False):
    """Draw one eye whose box is 6 wide x 6 tall at (x0,y0) (top-left), iris toward the nose
    when flip=False is the character's right eye (viewer left)."""
    s = style
    W, H = 6, 6

    def X(dx):  # horizontal coordinate with mirroring
        return x0 + (W - 1 - dx if flip else dx)

    lash = s["lash"]
    if expr == "blink":
        for dx in range(W):
            dy = 4 if dx in (0, W - 1) else 5
            cv.px(X(dx), y0 + dy, lash)
        cv.px(X(-1), y0 + 3, lash)
        return
    if s.get("glow"):
        # ender style: glowing horizontal slit
        top = {"lidded": 2, "moan": 2, "ahegao": 1}.get(expr, 1)
        for dx in range(-1, W + 1):
            cv.rect(X(dx), y0 + top, X(dx), y0 + 4, s["iris"], emit=True)
        cv.rect(X(1), y0 + top + (1 if expr != "ahegao" else 0), X(4), y0 + top + 1, s["iris_hi"], emit=True)
        if expr == "ahegao":
            cv.rect(X(0), y0 + 2, X(5), y0 + 2, s["iris_hi"], emit=True)
        if expr == "hearts":
            cv.pts([(X(1), y0 + 2), (X(2), y0 + 3), (X(3), y0 + 2), (X(2), y0 + 4)], "ff9df5", emit=True)
        for dx in range(-2, W + 2, 2):
            cv.px(X(dx), y0 + top - 1, lash)
        cv.px(X(-2), y0 + top - 2, lash)
        cv.px(X(W + 1), y0 + top - 2, lash)
        return
    # sclera
    cv.rect(X(0), y0 + 1, X(W - 1), y0 + H - 1, s["sclera"])
    # iris (inner 4 columns, toward the nose)
    ix0, ix1 = 2, 5
    iy0, iy1 = 1, 5
    if expr == "ahegao":
        iy0, iy1 = 0, 2
    cv.rect(X(ix0), y0 + iy0, X(ix1), y0 + iy1, s["iris"])
    cv.rect(X(ix0), y0 + iy0, X(ix1), y0 + iy0, s["iris_dark"])
    if iy1 - iy0 >= 3:
        cv.rect(X(ix0), y0 + iy1, X(ix1), y0 + iy1, s["iris_light"])
        cv.rect(X(3), y0 + iy0 + 1, X(4), y0 + iy1 - 1, s["pupil"])
        cv.px(X(4), y0 + iy0 + 1, "ffffff")
    if expr == "hearts":
        cv.rect(X(ix0), y0 + 1, X(ix1), y0 + 5, s["iris"])
        heart = [(2, 2), (4, 2), (2, 3), (3, 3), (4, 3), (5, 3), (3, 4), (4, 4), (5, 2)]
        for dx, dy in heart:
            cv.px(X(dx), y0 + dy, "ff4f9a")
        cv.px(X(2), y0 + 2, "ffc0dc")
    # upper lid / lashes
    lid_rows = {"lidded": 2, "moan": 3}.get(expr, 0)
    if lid_rows:
        cv.rect(X(0), y0 + 1, X(W - 1), y0 + lid_rows, s["lid"])
    cv.rect(X(-1), y0 + lid_rows, X(W - 1), y0 + lid_rows, lash)
    cv.px(X(-1), y0 + lid_rows - 1, lash)
    if s.get("lower_lash"):
        cv.px(X(0), y0 + H, lash)


def mouth(cv, cx, y, expr, style):
    m = style
    if expr in ("neutral", "blink", "lidded"):
        if m.get("cat"):
            cv.pts([(cx - 3, y), (cx - 2, y + 1), (cx - 1, y), (cx, y), (cx + 1, y + 1), (cx + 2, y)], m["line"])
        else:
            cv.rect(cx - 2, y, cx + 1, y, m["line"])
        if expr == "lidded":
            cv.px(cx + 2, y - 1, m["line"])
    elif expr in ("smile", "hearts"):
        cv.rect(cx - 4, y - 1, cx + 3, y + 2, m["inside"])
        cv.rect(cx - 4, y - 1, cx + 3, y - 1, m["teeth"])
        cv.rect(cx - 2, y + 2, cx + 1, y + 2, m["tongue"])
        cv.px(cx - 4, y + 2, m["skin"]); cv.px(cx + 3, y + 2, m["skin"])
    elif expr == "moan":
        cv.rect(cx - 2, y - 1, cx + 1, y + 2, m["inside"])
        cv.rect(cx - 1, y + 1, cx, y + 2, m["tongue"])
        cv.px(cx - 2, y - 1, m["skin"]); cv.px(cx + 1, y - 1, m["skin"])
    elif expr == "ahegao":
        cv.rect(cx - 4, y - 1, cx + 3, y + 1, m["inside"])
        cv.rect(cx - 2, y, cx + 1, y + 5, m["tongue"])
        cv.rect(cx - 2, y + 5, cx + 1, y + 5, m["tongue_dark"])
        cv.px(cx, y + 1, m["tongue_dark"]); cv.px(cx, y + 2, m["tongue_dark"])
        cv.px(cx + 3, y + 2, "cfe8ff"); cv.px(cx + 3, y + 3, "cfe8ff")


def blush(cv, x0, y0, w, h, col, strength):
    cv.rect(x0, y0, x0 + w - 1, y0 + h - 1, col, alpha=strength)


def brows(cv, x0, y0, col, expr, flip=False, length=5):
    worried = expr in ("moan", "ahegao")
    for i in range(length):
        x = x0 + (length - 1 - i if flip else i)
        dy = 0
        if worried:
            dy = -1 if i >= length - 2 else 0
        elif expr == "lidded":
            dy = 1 if i >= length - 2 else 0
        cv.px(x, y0 + dy, col)


def atlas(draw_fn):
    """draw_fn(expr) -> Canvas. Returns (color atlas, emit atlas) as uint8 arrays."""
    cols, emis = [], []
    for ex in EXPRESSIONS:
        cv = draw_fn(ex)
        cols.append(cv.c)
        emis.append(cv.e)
    return np.concatenate(cols, axis=1), np.concatenate(emis, axis=1)


def sides_atlas(canvases):
    """canvases in order right(-X), left(+X), back, top, bottom."""
    return (np.concatenate([c.c for c in canvases], axis=1),
            np.concatenate([c.e for c in canvases], axis=1))
