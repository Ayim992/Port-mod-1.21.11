"""Marie Prowell, the Ender Witch. Near-black skin, magenta glow (eyes, sigils, nipples),
bandage wraps, belt with glowing potions, loincloth, boots, huge bent witch hat with a lantern."""
import numpy as np
from lib import anatomy, sdf
from lib.sdf import S
from lib.paint import Layer, hx, stripes, hash2, nipples, vulva, navel, front
from lib.faces import Canvas, eye, mouth, blush, brows

NAME = "Marie"
EMIT_STRENGTH = 7.0
RENDER = dict(target_z=1.12, ortho=2.65)
RIG_TOGGLES = [("hat", 1, "Witch hat + lantern")]
P = anatomy.params(
    hip_w=5.25, waist_w=3.3, chest_w=3.9, hip_back=2.3,
    breast_r=(2.05, 1.85, 1.95), breast_x=1.95, breast_z=19.4, breast_y=-1.5,
    butt_r=(2.7, 2.45, 2.6), butt_x=2.0, butt_z=11.8, butt_y=1.65,
    belly_r=(2.8, 1.4, 2.0), belly_y=-0.85,
    thigh_w=2.9, thigh_depth=2.45, leg_x_top=2.75,
)

SKIN = "1e1b24"
SKIN_D = "141218"
GLOW = "d23cff"
GLOW_HI = "f6a6ff"
WRAP = "b49c74"
WRAP_D = "8a7452"
BELT = "4a3222"
BUCKLE = "c9a54a"
BOOT = "5b3d27"
BOOT_D = "3c281a"
HAT = "2b2534"
HAT_EDGE = "3b3247"
HAT_BAND = "6e4b2d"


def _band(v, center, half):
    return np.abs(v - center) < half


def paint(P_, N, layer, part, lm):
    L = Layer(len(P_), SKIN)
    x, y, z = P_[:, 0], P_[:, 1], P_[:, 2]
    dressed = layer == "Col"
    g = 1.0  # emission factor for glow
    if part == "body":
        nip_z = lm["nipples"][0][2]
        if dressed:
            # boots
            L.put(z < 4.3, BOOT)
            L.put((z > 3.6) & (z < 4.3), BOOT_D)
            L.put(z < 0.35, BOOT_D)
            lace = front(N, -0.5) & (z > 1.0) & (z < 3.5) & (np.abs(np.abs(x) - 2.05) < 0.35)
            L.put(lace & stripes(z, 0.6, 0.4), WRAP)
            # thigh wraps (diagonal bandage)
            xl = np.abs(x) - 2.7
            L.put(_band(z, 7.1 + 0.4 * xl, 0.32) & (z > 5.5) & (z < 9.0), WRAP)
            L.put(_band(z, 6.2 - 0.3 * xl, 0.22) & (z > 5.5) & (z < 9.0), WRAP_D)
            # belt
            L.put(_band(z, 11.35, 0.42), BELT)
            L.put(_band(z, 11.35, 0.42) & front(N, -0.6) & (np.abs(x) < 0.55), BUCKLE)
            # loin wrap under the belt
            L.put((z > 9.0) & (z < 11.0) & (np.abs(x) < 1.6) & (y < 0), WRAP_D)
            # chest wraps: crossing straps + band over the bust
            L.put(_band(z, 19.6 + 0.95 * x, 0.38) & (z > 15.5) & (z < 23.0), WRAP)
            L.put(_band(z, 19.6 - 0.95 * x, 0.38) & (z > 15.5) & (z < 23.0), WRAP)
            L.put(_band(z, nip_z, 0.7) & (y < 0.5), WRAP)
            L.put(_band(z, nip_z - 0.62, 0.12) & (y < 0.5), WRAP_D)
            L.put(_band(z, 17.3, 0.3), WRAP_D)
        else:
            nipples(L, P_, N, lm, GLOW, GLOW_HI, r_areola=0.7, r_nip=0.32, emit=g)
            vulva(L, P_, lm, inner=GLOW_HI, outer=GLOW)
            if "vulva_top" in lm:
                t = lm["vulva_top"]
                m = (np.abs(x) < 0.5) & (z < t[2] + 0.15) & (z > lm["vulva_bottom"][2] - 0.2) & (y < t[1] + 1.4)
                L.emit[m] = g * 0.8
        navel(L, P_, lm, SKIN_D)
    else:
        side = 1 if part.endswith("L") else -1
        # position along arm (0 = top, 1 = hand)
        u = (24.0 - z) / 12.0
        if dressed:
            L.put((u > 0.55) & (u < 0.93) & stripes(z, 0.9, 0.45), WRAP)
            L.put((u > 0.88) & (u < 0.93), WRAP_D)
        if side == 1:
            # glowing glyph on the upper left arm (outer front)
            m = (u > 0.12) & (u < 0.48) & (N[:, 1] < -0.4) & (np.abs(x - 6.2) < 0.75)
            code = hash2(np.floor(x / 0.25).astype(np.int64), np.floor(z / 0.25).astype(np.int64), 11)
            col = m & ((np.abs(x - 6.2) < 0.13) | (code < 0.25))
            L.put(col, GLOW, emit=g)
    return L.out()


def rig_mod(bones, extra, p):
    cz = p["crotch_z"]
    extra = list(extra) + [
        dict(name="cloth_front", head=(0, -2.3, 11.0), tail=(0, -2.5, 8.0), parent="spine", mirror=False),
        dict(name="cloth_back", head=(0, 3.6, 11.0), tail=(0, 3.8, 8.0), parent="spine", mirror=False),
        dict(name="hat", head=(0, 0.4, 32.4), tail=(0, 0.4, 36.0), parent="spine.006", mirror=False),
    ]
    return bones, extra


# ----------------------------------------------------------------------------- props
def _hat():
    brim = S(sdf.ellipsoid, (0, 0.3, 32.35), (9.6, 9.4, 0.5))
    brim_cut = S(sdf.round_box, (0, 0.3, 31.6), (12, 12, 0.6))
    pts = [(0, 0.3, 32.3), (0, 0.9, 35.6), (-0.4, 2.4, 38.3), (-1.5, 4.4, 39.7), (-2.7, 6.2, 39.6),
           (-3.4, 7.2, 38.7)]
    rad = [4.9, 3.7, 2.5, 1.5, 0.8, 0.35]
    cone = sdf.Shape(lambda q: sdf.chain(q, pts, rad))
    return (brim - brim_cut) | cone


def _hat_color(c, n):
    rgb = np.tile(hx(HAT), (len(c), 1))
    r = np.hypot(c[:, 0], c[:, 1] - 0.3)
    rgb[(r > 8.6) & (c[:, 2] < 33.0)] = hx(HAT_EDGE)
    rgb[(c[:, 2] > 32.8) & (c[:, 2] < 34.0) & (r < 5.6)] = hx(HAT_BAND)
    k = hash2(np.floor(c[:, 0] / 0.5).astype(np.int64), np.floor(c[:, 2] / 0.5).astype(np.int64), 5)
    rgb[(k < 0.12) & (c[:, 2] > 34.0)] = hx(HAT_EDGE)
    return np.concatenate([rgb, np.ones((len(c), 1))], axis=1)


def _lantern():
    c = np.array([7.6, -3.6, 29.5])
    dark = (0.16, 0.13, 0.12, 1)
    glow = (1.0, 0.62, 0.22, 1)
    return [
        dict(c=c + (0, 0, 2.2), half=(0.08, 0.08, 0.75), color=dark),      # chain
        dict(c=c + (0, 0, 1.2), half=(0.55, 0.55, 0.15), color=dark),      # cap
        dict(c=c, half=(0.45, 0.45, 0.85), color=glow, emit=12.0),          # glass/flame
        dict(c=c + (0, 0, -0.95), half=(0.6, 0.6, 0.15), color=dark),      # base
        dict(c=c + (0.47, 0.47, 0), half=(0.07, 0.07, 0.9), color=dark),
        dict(c=c + (-0.47, 0.47, 0), half=(0.07, 0.07, 0.9), color=dark),
        dict(c=c + (0.47, -0.47, 0), half=(0.07, 0.07, 0.9), color=dark),
        dict(c=c + (-0.47, -0.47, 0), half=(0.07, 0.07, 0.9), color=dark),
    ]


def _potions():
    out = []
    cols = [((0.3, 1.0, 0.42, 1), (5.0, -1.2, 10.2)), ((0.3, 0.72, 1.0, 1), (5.35, 0.3, 10.0)),
            ((1.0, 0.55, 0.2, 1), (4.7, -2.4, 10.45))]
    for col, c in cols:
        c = np.array(c)
        out += [dict(c=c, half=(0.5, 0.5, 0.55), color=col, emit=8.0),
                dict(c=c + (0, 0, 0.75), half=(0.2, 0.2, 0.22), color=(0.85, 0.9, 0.95, 1)),
                dict(c=c + (0, 0, 1.05), half=(0.22, 0.22, 0.1), color=(0.45, 0.3, 0.18, 1))]
    pouch = np.array([-5.1, -0.6, 10.3])
    out += [dict(c=pouch, half=(0.55, 0.9, 0.75), color=(0.29, 0.2, 0.13, 1)),
            dict(c=pouch + (0, 0, 0.7), half=(0.6, 0.95, 0.12), color=(0.22, 0.15, 0.1, 1))]
    return out


def _cloth(sign):
    yb = -2.35 if sign < 0 else 3.75
    col = (0.63, 0.54, 0.4, 1)
    dcol = (0.5, 0.42, 0.3, 1)
    boxes = [dict(c=(0, yb, 9.4), half=(1.35, 0.12, 1.7), color=col),
             dict(c=(0, yb + sign * 0.02, 7.5), half=(1.1, 0.12, 0.3), color=dcol)]
    # ragged bottom
    for i, dx in enumerate((-0.9, -0.3, 0.4, 0.95)):
        boxes.append(dict(c=(dx, yb, 7.0 - 0.25 * (i % 2)), half=(0.25, 0.11, 0.3), color=col))
    return boxes


def props(lm, p):
    return [
        dict(name="Hat", kind="voxel", shape=_hat(), lo=(-10.5, -10.5, 31.4), hi=(10.5, 10.5, 41.0),
             vox=0.4, color=_hat_color, bone="DEF-hat", toggle="hat"),
        dict(name="Lantern", kind="boxes", boxes=_lantern(), bone="DEF-hat", toggle="hat"),
        dict(name="Potions", kind="boxes", boxes=_potions(), bone="DEF-spine", toggle="outfit"),
        dict(name="ClothFront", kind="boxes", boxes=_cloth(-1), bone="DEF-cloth_front", toggle="outfit"),
        dict(name="ClothBack", kind="boxes", boxes=_cloth(1), bone="DEF-cloth_back", toggle="outfit"),
    ]


# ----------------------------------------------------------------------------- face
EYE = dict(glow=True, iris=GLOW, iris_hi=GLOW_HI, lash="050407")
MOUTH = dict(line="3a1a44", inside="2a0d33", teeth="d7c8e8", tongue=GLOW, tongue_dark="a020c8", skin=SKIN)


def face(expr):
    cv = Canvas(SKIN)
    # forehead sigil
    ring = [(14, 2), (15, 2), (16, 2), (17, 2), (13, 3), (18, 3), (12, 4), (19, 4), (12, 5), (19, 5),
            (12, 6), (19, 6), (13, 7), (18, 7), (14, 8), (15, 8), (16, 8), (17, 8)]
    cv.pts(ring, GLOW, emit=True)
    cv.rect(15, 4, 16, 6, GLOW_HI, emit=True)
    cv.rect(10, 5, 11, 5, GLOW, emit=True)
    cv.rect(20, 5, 21, 5, GLOW, emit=True)
    cv.rect(6, 0, 6, 11, GLOW, emit=True)
    cv.rect(25, 0, 25, 11, GLOW, emit=True)
    eye(cv, 5, 13, EYE, expr)
    eye(cv, 21, 13, EYE, expr, flip=True)
    if expr in ("smile", "moan", "ahegao", "hearts"):
        mouth(cv, 16, 25, expr, MOUTH)
    if expr in ("moan", "ahegao", "lidded", "hearts"):
        blush(cv, 4, 21, 6, 2, "6a1d7a", 0.6)
        blush(cv, 22, 21, 6, 2, "6a1d7a", 0.6)
    return cv


def head_sides():
    sides = []
    for i in range(5):
        cv = Canvas(SKIN)
        cv.noise(SKIN_D, 0.2, seed=i)
        if i == 0:  # glyph on her right side
            cv.rect(6, 12, 6, 18, GLOW, emit=True)
            cv.rect(8, 13, 10, 13, GLOW, emit=True)
            cv.rect(8, 15, 9, 16, GLOW, emit=True)
            cv.rect(11, 17, 13, 17, GLOW, emit=True)
            cv.rect(15, 12, 15, 18, GLOW, emit=True)
        sides.append(cv)
    return sides
