"""Bia Prowell, the Teddy Bear. Brown plush fur with pixel-jagged tan patches, round ears,
leaf sprouts, white flower, big blue eyes, tan muzzle, lilac bikini bottom with a leaf, round tail."""
import numpy as np
from lib import anatomy, sdf
from lib.sdf import S
from lib.paint import Layer, hx, stripes, hash2, nipples, vulva, navel, front
from lib.faces import Canvas, eye, mouth, blush, brows

NAME = "Bia"
RENDER = dict(target_z=1.0, ortho=2.4)
P = anatomy.params(
    hip_w=5.45, waist_w=3.95, chest_w=4.05, hip_back=2.35, torso_front=2.1,
    breast_r=(2.0, 1.7, 1.85), breast_x=1.95, breast_z=19.1, breast_y=-1.55, breast_droop=9,
    butt_r=(2.75, 2.45, 2.65), butt_x=2.05, butt_z=11.7, butt_y=1.65,
    belly_r=(3.3, 1.75, 2.6), belly_y=-0.75, belly_z=13.4,
    thigh_w=3.0, thigh_depth=2.5, leg_x_top=2.8,
    arm_half=(1.85, 1.8, 6.0), arm_pivot=(5.65, 0.0, 22.4),
)

FUR = "7b4a29"
FUR_D = "5f381e"
TAN = "c79a63"
TAN_D = "b0844f"
PAW = "9c7445"
PINK = "d98b80"
PANTY = "a597e4"
PANTY_D = "8778cc"
LEAF = "6ec23e"
LEAF_L = "a3e26f"
LEAF_D = "479a28"


def _jag(P_, amp=0.5, seed=3, size=0.5):
    """Per-pixel-cell jitter in [-amp, amp], quantized to the cell size."""
    a = np.floor(P_[:, 0] / size).astype(np.int64) * 7 + np.floor(P_[:, 1] / size).astype(np.int64)
    b = np.floor(P_[:, 2] / size).astype(np.int64)
    return np.round((hash2(a, b, seed) - 0.5) * 2 * amp / size) * size


def paint(P_, N, layer, part, lm):
    L = Layer(len(P_), FUR)
    x, y, z = P_[:, 0], P_[:, 1], P_[:, 2]
    dressed = layer == "Col"
    cz = 9.7
    # sparse darker fur pixels
    L.put(hash2(np.floor(x / 0.5).astype(np.int64) * 31 + np.floor(y / 0.5).astype(np.int64),
                np.floor(z / 0.5).astype(np.int64), 2) < 0.035, FUR_D)
    if part == "body":
        # tan belly/chest plate with pixel-jagged edge
        e = (x / 3.0) ** 2 + ((z - 16.2) / 6.6) ** 2
        plate = (e < 1.0 + _jag(P_, 0.18, 4, 0.5)) & (N[:, 1] < -0.15)
        L.put(plate, TAN)
        # under-bust and waist seam shading like the sheet
        nz = lm["nipples"][0][2]
        L.put(plate & (np.abs(z - (nz - 1.55)) < 0.14), TAN_D)
        # tan lower legs with jagged top
        sock = z < 3.9 + _jag(P_, 0.5, 6, 0.5)
        L.put(sock, TAN)
        L.put(z < 0.45, PAW)
        # inner thigh tan
        L.put((z < cz - 0.2) & (z > 4.5) & (np.abs(x) < 1.2 + _jag(P_, 0.25, 8)) & (N[:, 0] * x < 0), TAN)
        navel(L, P_, lm, TAN_D)
        if dressed:
            # bikini bottom: front triangle, side strings, back panel
            fr = (y < 0.5) & (z > cz - 0.4) & (z < cz + 1.7) & (np.abs(x) < 1.0 + (z - (cz - 0.4)) * 1.25)
            bk = (y > 0.5) & (z > cz + 0.2) & (z < cz + 1.7) & (np.abs(x) < 1.2 + (z - cz) * 1.3)
            strap = (z > cz + 1.35) & (z < cz + 1.75)
            L.put(fr | bk | strap, PANTY)
            L.put(strap & (np.abs(x) > 2.0), PANTY_D)
            # leaf patch over the crotch
            lc = np.array([0.0, 0.0, cz + 0.45])
            u = x / 0.95
            v = (z - lc[2]) / 1.15
            leaf = (u ** 2 + v ** 2 < 1.0) & (y < 0.3)
            L.put(leaf, LEAF)
            L.put(leaf & (np.abs(x) < 0.1), LEAF_D)
            L.put(leaf & (u ** 2 + v ** 2 > 0.65) & (x > 0), LEAF_L)
        else:
            nipples(L, P_, N, lm, "c48a72", PINK, r_areola=0.6, r_nip=0.28)
            vulva(L, P_, lm, inner="b85f66", outer=PINK, width=0.5)
    else:
        u = (24.0 - z) / 12.0
        L.put(u > 0.8 + _jag(P_, 0.04, 9, 0.5) / 12, TAN)
        L.put(u > 0.97, PAW)
    return L.out()


def rig_mod(bones, extra, p):
    extra = list(extra) + [
        dict(name="tail", head=(0, 2.9, 12.6), tail=(0, 4.6, 12.6), parent="spine", mirror=False),
        dict(name="ear.L", head=(3.4, 0.5, 31.5), tail=(3.6, 0.5, 34.0), parent="spine.006"),
        dict(name="leaves", head=(0, 0.3, 32.0), tail=(0, 0.3, 35.0), parent="spine.006", mirror=False),
    ]
    return bones, extra


# ----------------------------------------------------------------------------- props
def _ear(s):
    return S(sdf.round_box, (s * 3.55, 0.4, 32.6), (1.25, 0.6, 1.3), 0.55)


def _ear_color(s):
    def fn(c, n):
        rgb = np.tile(hx(FUR), (len(c), 1))
        inner = (n[:, 1] < -0.5) & (np.abs(c[:, 0] - s * 3.55) < 0.8) & (c[:, 2] > 32.0) & (c[:, 2] < 33.5)
        rgb[inner] = hx(TAN)
        return np.concatenate([rgb, np.ones((len(c), 1))], axis=1)
    return fn


def _leaves():
    a = S(sdf.ellipsoid, (-1.0, 0.4, 34.2), (0.45, 1.5, 2.1), sdf.rot_matrix(0, -28, 10))
    b = S(sdf.ellipsoid, (1.15, 0.1, 34.4), (0.45, 1.35, 2.3), sdf.rot_matrix(0, 32, -12))
    stem = S(sdf.round_box, (0, 0.3, 32.4), (0.35, 0.35, 0.6), 0.1)
    return a | b | stem


def _leaf_color(c, n):
    rgb = np.tile(hx(LEAF), (len(c), 1))
    rgb[c[:, 2] > 35.0] = hx(LEAF_L)
    rgb[c[:, 2] < 33.0] = hx(LEAF_D)
    k = hash2(np.floor(c[:, 0] / 0.3).astype(np.int64), np.floor(c[:, 2] / 0.3).astype(np.int64), 12)
    rgb[(k < 0.15) & (c[:, 2] > 33.0)] = hx(LEAF_D)
    return np.concatenate([rgb, np.ones((len(c), 1))], axis=1)


def _tufts():
    fur = tuple(np.r_[hx(FUR), 1])
    furd = tuple(np.r_[hx(FUR_D), 1])
    boxes = []
    for s in (1, -1):
        for (dy, z, ln) in ((-2.6, 25.4, 0.9), (-1.6, 26.3, 0.6), (-2.2, 27.1, 0.75), (1.0, 25.6, 0.6)):
            boxes.append(dict(c=(s * (4.0 + ln / 2), dy, z), half=(ln / 2, 0.3, 0.28), color=fur))
        boxes.append(dict(c=(s * 4.25, -0.3, 24.6), half=(0.3, 0.6, 0.3), color=furd))
    for dx in (-2.5, -0.5, 1.8):
        boxes.append(dict(c=(dx, -4.15, 24.2), half=(0.35, 0.25, 0.35), color=fur))
    return boxes


def _flower():
    c = np.array([2.6, -4.4, 30.4])
    white = (0.97, 0.97, 0.97, 1)
    return [dict(c=c, half=(1.35, 0.22, 0.55), color=white),
            dict(c=c, half=(0.55, 0.22, 1.35), color=white),
            dict(c=c + (0, -0.02, 0), half=(0.95, 0.2, 0.95), color=(0.9, 0.9, 0.93, 1)),
            dict(c=c + (0, -0.2, 0), half=(0.4, 0.1, 0.4), color=(0.62, 0.56, 0.94, 1))]


def _collar():
    outer = S(sdf.round_box, (0, 0.1, 23.75), (1.95, 1.8, 0.32), 0.1)
    inner = S(sdf.round_box, (0, 0.1, 23.75), (1.68, 1.53, 0.6), 0.0)
    return outer - inner


def props(lm, p):
    rgba1 = lambda h: (lambda c, n: np.tile(np.r_[hx(h), 1], (len(c), 1)))
    return [
        dict(name="Ear.L", kind="voxel", shape=_ear(1), lo=(1.5, -1.5, 30.5), hi=(5.5, 2.5, 34.5),
             vox=0.25, color=_ear_color(1), bone="DEF-ear.L"),
        dict(name="Ear.R", kind="voxel", shape=_ear(-1), lo=(-5.5, -1.5, 30.5), hi=(-1.5, 2.5, 34.5),
             vox=0.25, color=_ear_color(-1), bone="DEF-ear.R"),
        dict(name="Leaves", kind="voxel", shape=_leaves(), lo=(-3.5, -2.5, 31.5), hi=(3.5, 3.0, 37.5),
             vox=0.25, color=_leaf_color, bone="DEF-leaves"),
        dict(name="Tufts", kind="boxes", boxes=_tufts()),
        dict(name="Flower", kind="boxes", boxes=_flower()),
        dict(name="Tail", kind="voxel", shape=S(sdf.sphere, (0, 3.75, 12.6), 1.05),
             lo=(-1.5, 2.0, 11.0), hi=(1.5, 5.5, 14.2), vox=0.25,
             color=lambda c, n: np.tile(np.r_[hx(TAN), 1], (len(c), 1)), bone="DEF-tail"),
        dict(name="Collar", kind="voxel", shape=_collar(), lo=(-2.5, -2.5, 23.2), hi=(2.5, 2.5, 24.3),
             vox=0.125, color=rgba1("1b191f"), bone="DEF-spine.004"),
        dict(name="CollarTag", kind="boxes", bone="DEF-spine.004",
             boxes=[dict(c=(0, -1.95, 23.35), half=(0.35, 0.12, 0.35), color=(0.6, 0.48, 1.0, 1))]),
    ]


# ----------------------------------------------------------------------------- face
EYE = dict(sclera="ffffff", iris="2f86e8", iris_dark="1b4fa6", iris_light="85c8ff", pupil="10213f",
           lash="1a110c", lid=FUR, lower_lash=True)
MOUTH = dict(line="2a1610", inside="6e1f24", teeth="ffffff", tongue="e57a86", tongue_dark="bf5465",
             skin=TAN, cat=False)


def face(expr):
    cv = Canvas(FUR)
    rng = np.random.default_rng(1)
    cv.noise(FUR_D, 0.05, seed=2)
    brows(cv, 6, 9, "2a1810", expr)
    brows(cv, 21, 9, "2a1810", expr, flip=True)
    eye(cv, 6, 12, EYE, expr)
    eye(cv, 20, 12, EYE, expr, flip=True)
    # muzzle
    cv.rect(10, 20, 21, 30, TAN)
    cv.rect(9, 22, 22, 29, TAN)
    cv.rect(14, 21, 17, 22, "15100d")
    cv.rect(15, 23, 16, 23, "15100d")
    mouth(cv, 16, 26, expr, MOUTH)
    if expr in ("neutral", "blink"):
        cv.rect(14, 26, 18, 26, "2a1610")
        cv.px(17, 27, "ffffff")
    k = {"lidded": 0.55, "moan": 0.7, "ahegao": 0.8, "hearts": 0.65}.get(expr, 0.45)
    blush(cv, 4, 20, 5, 2, "ff7f8a", k)
    blush(cv, 23, 20, 5, 2, "ff7f8a", k)
    cv.rect(22, 24, 25, 25, "ecd9b5")  # band-aid
    cv.px(23, 24, "d8c29a"); cv.px(24, 25, "d8c29a")
    return cv


def head_sides():
    out = []
    for i in range(5):
        cv = Canvas(FUR)
        cv.noise(FUR_D, 0.08, seed=10 + i)
        if i == 4:
            cv = Canvas(TAN)
        elif i in (0, 1, 2):
            cv.rect(0, 29, 31, 31, FUR_D)
        out.append(cv)
    return out
