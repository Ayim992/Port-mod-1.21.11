"""Jenny Belle (human). Latest sheet: dark slate crop sweater, striped thigh-highs, sneakers,
brown bob with bangs, pink/yellow flower on her right side, choker with gold tag."""
import numpy as np
from lib import anatomy, sdf
from lib.sdf import S
from lib.paint import Layer, hx, stripes, hash2, cell, nipples, navel, vulva, front
from lib.faces import Canvas, eye, mouth, blush, brows

NAME = "Jenny"
P = anatomy.params(
    hip_w=5.6, waist_w=3.4, chest_w=4.0, hip_back=2.4,
    breast_r=(2.3, 2.1, 2.15), breast_x=2.05, breast_z=19.3, breast_y=-1.45, breast_droop=7,
    butt_r=(2.85, 2.6, 2.7), butt_x=2.05, butt_z=11.8, butt_y=1.75,
    belly_r=(3.0, 1.55, 2.2), belly_y=-0.8,
    thigh_w=3.05, thigh_depth=2.55, leg_x_top=2.85,
)

SKIN = "f4c4a2"
SKIN_SHADE = "e3a988"
SWEATER = "222d3a"
SWEATER_B = "2a3747"
SWEATER_HEM = "18212c"
SOCK_A = "2b3548"
SOCK_B = "171d2a"
SHOE = "1c2130"
SOLE = "eceef2"
HAIR = "6e4428"
HAIR_DARK = "57341f"
HAIR_LIGHT = "8a5735"


def paint(P_, N, layer, part, lm):
    L = Layer(len(P_), SKIN)
    x, y, z = P_[:, 0], P_[:, 1], P_[:, 2]
    dressed = layer == "Col"
    if part == "body":
        if not dressed:
            nipples(L, P_, N, lm, "d8927b", "c47562")
        vulva(L, P_, lm)
        navel(L, P_, lm, SKIN_SHADE)
        # thigh-high socks (on in both layers, like the sheet)
        sock = z < 8.3
        L.put(sock, SOCK_A)
        L.put(sock & stripes(z, 0.75, 0.5), SOCK_B)
        L.put((z > 7.7) & (z < 8.3), SOCK_B)
        # sneakers
        shoe = z < 1.35
        L.put(shoe, SHOE)
        L.put(z < 0.4, SOLE)
        lace = shoe & front(N, -0.5) & (z > 0.55) & (z < 1.25) & (np.abs(np.abs(x) - 2.05) < 0.8)
        L.put(lace & stripes(z, 0.35, 0.5, 0.55), SOLE)
        L.put(shoe & (np.abs(N[:, 0]) > 0.6) & (z > 0.5) & (z < 1.0) & (np.abs(y) < 0.6), SOLE)
        if dressed:
            sw = z > 17.25
            L.put(sw, SWEATER)
            L.put(sw & stripes(z, 1.0, 0.5), SWEATER_B)
            L.put((z > 17.25) & (z < 17.75), SWEATER_HEM)
    else:
        if dressed:
            L.put(np.ones(len(z), bool), SWEATER)
            L.put(stripes(z, 1.0, 0.5), SWEATER_B)
            L.put(z < 12.9, SWEATER_HEM)
    return L.out()


def body_mod(shape, lm, p):
    return shape


# ----------------------------------------------------------------------------- props
def _hair_shape():
    head_c = np.array([0.0, 0.0, 28.0])
    shell = S(sdf.round_box, (0, 0.2, 28.45), (4.5, 4.45, 4.05), 0.35)
    inner = S(sdf.round_box, head_c, (3.98, 3.98, 3.98), 0.0)

    def fn(q):
        d = np.maximum(shell(q), -inner(q))
        x, y, z = q[..., 0], q[..., 1], q[..., 2]
        # bangs line: swept toward her left, pixel jagged
        u = np.floor((x + 8) / 0.5)
        jag = (hash2(u.astype(np.int64), np.zeros_like(u, dtype=np.int64), 7) - 0.5) * 0.9
        bang = 30.2 + 0.32 * x + np.round(jag * 2) / 2
        face_open = (y < -3.3) & (z < bang) & (np.abs(x) < 3.05)
        d = np.where(face_open, 1.0, d)
        # side locks a bit longer in front
        d = np.where((z < 24.6) & (y > -3.3), 1.0, d)
        return d
    return sdf.Shape(fn)


def _hair_color(c, n):
    rgb = np.tile(hx(HAIR), (len(c), 1))
    u = np.floor(c[:, 0] / 0.5).astype(np.int64) + np.floor(c[:, 1] / 0.5).astype(np.int64)
    r = hash2(u, np.floor(c[:, 2] / 1.5).astype(np.int64), 3)
    rgb[r < 0.3] = hx(HAIR_DARK)
    rgb[(c[:, 2] > 31.8) & (r > 0.75)] = hx(HAIR_LIGHT)
    return np.concatenate([rgb, np.ones((len(c), 1))], axis=1)


def _collar_shape():
    outer = S(sdf.round_box, (0, 0.1, 23.75), (1.95, 1.8, 0.32), 0.1)
    inner = S(sdf.round_box, (0, 0.1, 23.75), (1.68, 1.53, 0.6), 0.0)
    return outer - inner


def props(lm, p):
    pink, pink_d, yellow = (0.88, 0.2, 0.54, 1), (0.74, 0.12, 0.42, 1), (0.95, 0.82, 0.18, 1)
    fc = np.array([-2.35, -4.55, 30.55])
    flower = [
        dict(c=fc, half=(1.45, 0.25, 0.7), color=pink),
        dict(c=fc, half=(0.7, 0.25, 1.45), color=pink),
        dict(c=fc + (0, 0.05, 0), half=(1.05, 0.22, 1.05), color=pink_d),
        dict(c=fc + (0, -0.25, 0), half=(0.6, 0.12, 0.6), color=yellow),
    ]
    return [
        dict(name="Hair", kind="voxel", shape=_hair_shape(), lo=(-5, -5, 23.5), hi=(5, 5, 33),
             vox=0.25, color=_hair_color),
        dict(name="Flower", kind="boxes", boxes=flower),
        dict(name="Collar", kind="voxel", shape=_collar_shape(), lo=(-2.5, -2.5, 23.2), hi=(2.5, 2.5, 24.3),
             vox=0.125, color=lambda c, n: np.tile([0.09, 0.09, 0.12, 1], (len(c), 1)),
             bone="DEF-spine.004"),
        dict(name="CollarTag", kind="boxes",
             boxes=[dict(c=(0, -1.95, 23.35), half=(0.32, 0.12, 0.38), color=(0.93, 0.6, 0.18, 1))],
             bone="DEF-spine.004"),
    ]


# ----------------------------------------------------------------------------- face
EYE = dict(sclera="ffffff", iris="3d7be0", iris_dark="25429a", iris_light="8cc0ff", pupil="17204a",
           lash="2a1a18", lid="e8b394")
MOUTH = dict(line="5a2a22", inside="7a2430", teeth="ffffff", tongue="e8798a", tongue_dark="c45468",
             skin=SKIN, cat=True)


def face(expr):
    cv = Canvas(SKIN)
    brows(cv, 6, 9, HAIR_DARK, expr)
    brows(cv, 21, 9, HAIR_DARK, expr, flip=True)
    eye(cv, 6, 11, EYE, expr)
    eye(cv, 20, 11, EYE, expr, flip=True)
    k = {"lidded": 0.45, "moan": 0.6, "ahegao": 0.7, "hearts": 0.55}.get(expr, 0.32)
    blush(cv, 4, 18, 6, 2, "f07c84", k)
    blush(cv, 22, 18, 6, 2, "f07c84", k)
    mouth(cv, 16, 23, expr, MOUTH)
    # hair edges visible on face sides
    cv.rect(0, 0, 2, 31, HAIR)
    cv.rect(29, 0, 31, 31, HAIR)
    cv.rect(0, 0, 31, 4, HAIR)
    return cv


def head_sides():
    sides = []
    for i in range(5):
        cv = Canvas(HAIR)
        cv.noise(HAIR_DARK, 0.25, seed=i)
        if i == 4:
            cv = Canvas(SKIN)
        sides.append(cv)
    return sides
