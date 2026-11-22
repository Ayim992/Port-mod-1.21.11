"""Generate a classic Steve-style 64x64 player skin (standard layout, overlay layer empty).

Replace out/textures/partner_skin.png with any 64x64 Minecraft skin to change the partner.
"""
import numpy as np
from PIL import Image

SKIN = (198, 150, 128)
SKIN_D = (170, 120, 98)
HAIR = (59, 42, 26)
SHIRT = (0, 170, 170)
SHIRT_D = (0, 140, 140)
PANTS = (58, 58, 153)
PANTS_D = (45, 45, 125)
SHOE = (85, 85, 85)


def _box(img, x0, y0, w, h, d, cols):
    """Paint one box in standard skin layout. cols: dict(top,bottom,right,front,left,back) -> fn(x,y,w,h)->rgb."""
    rects = {
        "top": (x0 + d, y0, w, d), "bottom": (x0 + d + w, y0, w, d),
        "right": (x0, y0 + d, d, h), "front": (x0 + d, y0 + d, w, h),
        "left": (x0 + d + w, y0 + d, d, h), "back": (x0 + 2 * d + w, y0 + d, w, h),
    }
    for face, (rx, ry, rw, rh) in rects.items():
        for yy in range(rh):
            for xx in range(rw):
                img[ry + yy, rx + xx] = (*cols(face, xx, yy, rw, rh), 255)


def make(path):
    img = np.zeros((64, 64, 4), np.uint8)
    rng = np.random.default_rng(3)

    def jitter(c, k=8):
        return tuple(int(np.clip(v + rng.integers(-k, k + 1), 0, 255)) for v in c)

    def head(face, x, y, w, h):
        if face == "top":
            return jitter(HAIR)
        if face == "bottom":
            return jitter(SKIN_D)
        if face in ("back",) or (face in ("left", "right") and (y < 3 or x > 4 if face == "right" else x < 3)):
            return jitter(HAIR)
        if face == "front":
            if y < 2:
                return jitter(HAIR)
            if y == 2 and x in (0, 7):
                return jitter(HAIR)
            if y == 4 and x in (1, 6):
                return (255, 255, 255)
            if y == 4 and x in (2, 5):
                return (73, 59, 145)
            if y == 5 and x in (3, 4):
                return (148, 96, 72)
            if y == 6 and 2 <= x <= 5:
                return (110, 65, 45)
            if y >= 6:
                return jitter((120, 80, 55))
        return jitter(SKIN)

    def body(face, x, y, w, h):
        if face == "front" and y == 0 and 2 <= x <= 5:
            return jitter(SKIN)
        return jitter(SHIRT if y < h - 1 else SHIRT_D)

    def arm(face, x, y, w, h):
        if face == "top":
            return jitter(SHIRT)
        if face == "bottom":
            return jitter(SKIN_D)
        return jitter(SHIRT if y < 4 else SKIN)

    def leg(face, x, y, w, h):
        if face == "bottom":
            return jitter(SHOE)
        if face == "top":
            return jitter(PANTS)
        return jitter(SHOE if y >= h - 2 else (PANTS if y < h - 3 else PANTS_D))

    _box(img, 0, 0, 8, 8, 8, head)        # head
    _box(img, 16, 16, 8, 12, 4, body)     # body
    _box(img, 40, 16, 4, 12, 4, arm)      # right arm
    _box(img, 32, 48, 4, 12, 4, arm)      # left arm
    _box(img, 0, 16, 4, 12, 4, leg)       # right leg
    _box(img, 16, 48, 4, 12, 4, leg)      # left leg
    Image.fromarray(img, "RGBA").save(path)
    return path
