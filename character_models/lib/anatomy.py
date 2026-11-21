"""Parametric curvy Minecraft-style female body as an SDF, plus rig landmarks.

All values in Minecraft pixels. 8 px head on top of a 24 px body (32 px total = 1.8 m).
"""
import math
import numpy as np
from . import sdf
from .sdf import S, Shape

DEFAULT = dict(
    # torso half widths
    hip_w=5.3, waist_w=3.55, chest_w=4.0, shoulder_w=4.0,
    torso_front=2.0, torso_back=2.0, hip_back=2.3,
    crotch_z=9.7,
    # breasts
    breast_r=(2.05, 1.9, 1.95), breast_x=1.95, breast_z=19.4, breast_y=-1.55, breast_yaw=9.0,
    breast_droop=6.0, breast_k=0.35, nipple_r=0.32,
    # butt
    butt_r=(2.6, 2.3, 2.5), butt_x=2.0, butt_z=11.7, butt_y=1.55, butt_k=0.7,
    # belly
    belly_r=(3.0, 1.6, 2.2), belly_z=13.3, belly_y=-0.75, belly_k=1.1,
    # legs
    thigh_w=2.95, knee_w=2.1, calf_w=2.05, foot_w=2.0, leg_depth=2.05, thigh_depth=2.45,
    leg_x_top=2.75, leg_x_knee=2.25, leg_x_foot=2.05, leg_k=0.8,
    # arms
    arm_half=(1.75, 1.75, 6.0), arm_round=0.45, arm_angle=7.0, arm_pivot=(5.55, 0.0, 22.4),
    # anatomy details
    vulva=True, navel=True,
)


def params(**over):
    p = dict(DEFAULT)
    p.update(over)
    return p


def torso_shape(p):
    hw, ww, cw, sw = p["hip_w"], p["waist_w"], p["chest_w"], p["shoulder_w"]
    tf, tb, hb = p["torso_front"], p["torso_back"], p["hip_back"]
    cz = p["crotch_z"]
    table = [
        (cz - 0.1, 0, 0.05, 1.5, 1.35, 1.25, 2.2),
        (cz + 0.9, 0, 0.1, hw * 0.80, tf * 0.95, hb * 0.9, 2.4),
        (cz + 2.3, 0, 0.1, hw, tf * 1.02, hb, 2.6),
        (cz + 3.8, 0, 0.05, hw * 0.9, tf * 1.0, tb * 1.0, 2.8),
        (15.6, 0, 0.0, ww, tf * 0.9, tb * 0.88, 3.0),
        (17.6, 0, 0.0, (ww + cw) / 2, tf * 0.95, tb * 0.95, 3.4),
        (20.0, 0, 0.0, cw, tf, tb, 4.2),
        (22.4, 0, 0.0, sw, tf * 0.95, tb * 0.95, 5.5),
        (23.4, 0, 0.0, sw - 0.35, tf * 0.88, tb * 0.88, 5.5),
    ]
    torso = Shape(lambda q: sdf.loft(q, table, cap_round=0.6))
    neck = S(sdf.round_box, (0, 0.1, 23.9), (1.7, 1.55, 0.9), 0.35)
    return torso | neck


def breast_centers(p):
    out = []
    for s in (1, -1):
        out.append(np.array([s * p["breast_x"], p["breast_y"], p["breast_z"]]))
    return out


def nipple_points(p):
    pts = []
    for s, c in zip((1, -1), breast_centers(p)):
        rot = sdf.rot_matrix(p["breast_droop"], 0, s * p["breast_yaw"])
        front = rot @ np.array([0, -p["breast_r"][1], -0.15])
        pts.append(c + front)
    return pts


def breasts_shape(p):
    parts = []
    for s, c in zip((1, -1), breast_centers(p)):
        rot = sdf.rot_matrix(p["breast_droop"], 0, s * p["breast_yaw"])
        parts.append(S(sdf.ellipsoid, c, p["breast_r"], rot))
    b = parts[0] | parts[1]
    if p["nipple_r"] > 0:
        for n in nipple_points(p):
            b = b.smin(S(sdf.ellipsoid, n, (p["nipple_r"], 0.28, p["nipple_r"])), 0.12)
    return b


def butt_centers(p):
    return [np.array([s * p["butt_x"], p["butt_y"], p["butt_z"]]) for s in (1, -1)]


def butt_shape(p):
    parts = [S(sdf.ellipsoid, c, p["butt_r"], sdf.rot_matrix(-8, 0, s * -6))
             for s, c in zip((1, -1), butt_centers(p))]
    return parts[0] | parts[1]


def leg_table(p, s, gap=0.2):
    """Rows for one leg. Inner extent is clamped so the thighs never fuse below the crotch
    (fused thighs turn into a skin web when the legs spread)."""
    cz = p["crotch_z"]
    rows = [
        (0.0, p["leg_x_foot"], 0.0, p["foot_w"], p["leg_depth"], p["leg_depth"], 9.0),
        (4.0, p["leg_x_foot"], 0.0, p["calf_w"], p["leg_depth"], p["leg_depth"] * 1.02, 7.0),
        (5.6, p["leg_x_knee"], -0.05, p["knee_w"], p["leg_depth"] * 1.02, p["leg_depth"] * 1.05, 4.5),
        (7.6, (p["leg_x_knee"] + p["leg_x_top"]) / 2, 0.0, (p["knee_w"] + p["thigh_w"]) / 2 + 0.1,
         (p["leg_depth"] + p["thigh_depth"]) / 2, (p["leg_depth"] + p["thigh_depth"]) / 2 + 0.05, 2.7),
        (cz, p["leg_x_top"], 0.05, p["thigh_w"], p["thigh_depth"], p["thigh_depth"] + 0.1, 2.4),
        (cz + 2.6, p["leg_x_top"] - 0.3, 0.1, p["thigh_w"] * 0.9, p["thigh_depth"] * 0.95,
         p["thigh_depth"], 2.4),
    ]
    out = []
    for z, cx, cy, hw, fr, bk, n in rows:
        inner = min(hw, cx - gap) if z <= cz else hw
        outer = hw
        if s > 0:   # +X leg: inner side is toward -X
            out.append((z, cx, cy, outer, fr, bk, n, inner))
        else:       # -X leg: mirrored, inner side is toward +X
            out.append((z, -cx, cy, inner, fr, bk, n, outer))
    return out


def legs_shape(p):
    L = Shape(lambda q: sdf.loft(q, leg_table(p, 1)))
    R = Shape(lambda q: sdf.loft(q, leg_table(p, -1)))
    return L | R


def surface_y(shape, x, z, front=True, y0=-8.0, steps=200):
    """March along +/-Y at (x, z) to find the body surface."""
    ys = np.linspace(y0 if front else -y0, 0.0, steps)
    pts = np.stack([np.full_like(ys, x), ys, np.full_like(ys, z)], axis=-1)
    d = shape(pts)
    i = int(np.argmax(d <= 0.0))
    return float(ys[i])


def body_shape(p):
    torso = torso_shape(p)
    body = torso.smin(S(sdf.ellipsoid, (0, p["belly_y"], p["belly_z"]), p["belly_r"]), p["belly_k"])
    body = body.smin(butt_shape(p), p["butt_k"])
    body = body.smin(legs_shape(p), p["leg_k"])
    body = body.smin(breasts_shape(p), p["breast_k"])
    lm = {}
    if p["vulva"]:
        cz = p["crotch_z"]
        fy = surface_y(body, 0.0, cz + 0.5)
        mound = S(sdf.ellipsoid, (0, fy + 0.45, cz + 0.65), (1.05, 0.55, 0.8))
        body = body.smin(mound, 0.35)
        fy2 = surface_y(body, 0.0, cz + 0.4)
        slit = S(sdf.capsule, (0, fy2 - 0.1, cz + 0.85), (0, fy2 + 0.9, cz - 0.15), 0.11)
        body = body.cut(slit, 0.09)
        lm["vulva_top"] = np.array([0, fy2, cz + 0.85])
        lm["vulva_bottom"] = np.array([0, fy2 + 0.9, cz - 0.2])
    if p["navel"]:
        ny = surface_y(body, 0.0, p["belly_z"] + 1.0)
        body = body.cut(S(sdf.ellipsoid, (0, ny - 0.05, p["belly_z"] + 1.0), (0.18, 0.3, 0.32)), 0.08)
        lm["navel"] = np.array([0, ny, p["belly_z"] + 1.0])
    lm["nipples"] = nipple_points(p)
    lm["breast_centers"] = breast_centers(p)
    lm["butt_centers"] = butt_centers(p)
    return body, lm


# --------------------------------------------------------------------------- arms
def arm_frame(p, s):
    piv = np.array([s * p["arm_pivot"][0], p["arm_pivot"][1], p["arm_pivot"][2]])
    a = math.radians(p["arm_angle"])
    d = np.array([s * math.sin(a), 0.0, -math.cos(a)])
    return piv, d


def arm_shape(p, s):
    piv, d = arm_frame(p, s)
    hx, hy, hz = p["arm_half"]
    top_off = 1.6  # pivot sits this far below the top of the arm box
    center = piv + d * (hz - top_off)
    rot = sdf.rot_matrix(0, -s * p["arm_angle"], 0)
    return S(sdf.round_box, center, (hx, hy, hz), p["arm_round"], rot), (piv, d, center)


def arm_local(P, p, s):
    """Return (u along arm 0=top..1=bottom, lateral x, depth y) for arm points P."""
    piv, d = arm_frame(p, s)
    hz = p["arm_half"][2]
    top = piv - d * 1.6
    rel = P - top
    along = rel @ d
    a = math.radians(p["arm_angle"])
    side = np.array([s * math.cos(a), 0.0, math.sin(a)])  # outward, perpendicular to the arm
    return along / (2 * hz), rel @ side, rel[:, 1]


# --------------------------------------------------------------------------- rig landmarks
def rig_bones(p):
    cz = p["crotch_z"]
    piv, d = arm_frame(p, 1)
    hz = p["arm_half"][2]
    elbow = piv + d * (hz - 1.6) + np.array([0, 0.2, 0])
    wrist = piv + d * (2 * hz - 1.6 - 2.2)
    hand_end = piv + d * (2 * hz - 1.6 - 0.3)
    lx = p["leg_x_knee"]
    bones = {
        "spine": ((0, 0.2, cz + 0.6), (0, 0.15, 13.6)),
        "spine.001": ((0, 0.15, 13.6), (0, 0.1, 16.2)),
        "spine.002": ((0, 0.1, 16.2), (0, 0.1, 19.2)),
        "spine.003": ((0, 0.1, 19.2), (0, 0.15, 23.2)),
        "spine.004": ((0, 0.15, 23.2), (0, 0.1, 24.0)),
        "spine.005": ((0, 0.1, 24.0), (0, 0.05, 24.6)),
        "spine.006": ((0, 0.05, 24.6), (0, 0.05, 31.0)),
        "shoulder.L": ((0.5, -0.3, 22.9), (piv[0] - 1.6, 0.2, 22.9)),
        "upper_arm.L": (tuple(piv), tuple(elbow)),
        "forearm.L": (tuple(elbow), tuple(wrist)),
        "hand.L": (tuple(wrist), tuple(hand_end)),
        "breast.L": (tuple(breast_centers(p)[0] + np.array([0, 1.2, 0.2])), tuple(nipple_points(p)[0])),
        "pelvis.L": ((0, 0.2, cz + 0.6), (2.6, -0.6, cz + 2.6)),
        "thigh.L": ((p["leg_x_top"] - 0.3, 0.1, cz + 0.5), (lx, -0.25, 5.6)),
        "shin.L": ((lx, -0.25, 5.6), (p["leg_x_foot"], 0.15, 1.1)),
        "foot.L": ((p["leg_x_foot"], 0.15, 1.1), (p["leg_x_foot"], -1.3, 0.35)),
        "toe.L": ((p["leg_x_foot"], -1.3, 0.35), (p["leg_x_foot"], -2.0, 0.35)),
        "heel.02.L": ((p["leg_x_foot"] - 0.9, 1.2, 0.0), (p["leg_x_foot"] + 0.9, 1.2, 0.0)),
    }
    extra = []
    for c in butt_centers(p)[:1]:
        extra.append(dict(name="butt.L", head=tuple(c - np.array([0, 1.4, -0.3])),
                          tail=tuple(c + np.array([0, p["butt_r"][1] * 0.9, -0.2])), parent="spine"))
    extra.append(dict(name="belly", head=(0, 0.4, p["belly_z"]), tail=(0, -2.4, p["belly_z"]),
                      parent="spine.001", mirror=False))
    return bones, extra
