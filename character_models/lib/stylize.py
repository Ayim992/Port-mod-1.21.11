"""SlipperyT-style shading: a soft pink tint in the creases of skin (under the breasts, belly folds,
armpits, shoulders), done in the shader with ambient occlusion so it follows every pose.

Only skin-coloured pixels are tinted (hue/saturation/value mask), clothes and hair are untouched.
Strength is driven by the rig property `pink_shading` (0..1).
"""
import bpy

PINK = (0.80, 0.33, 0.32, 1.0)  # linear of sRGB #e89c9a


def add_skin_tint(mat, rig=None, prop="pink_shading", strength=0.85, ao_distance=0.12):
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    link = next((l for l in nt.links if l.to_socket == bsdf.inputs["Base Color"]), None)
    if link is None:
        return
    src = link.from_socket
    nt.links.remove(link)
    n = nt.nodes
    sep = n.new("ShaderNodeSeparateColor"); sep.mode = "HSV"; sep.location = (-200, 420)
    nt.links.new(src, sep.inputs[0])

    def cmp(sock, op, val, loc):
        m = n.new("ShaderNodeMath"); m.operation = op; m.inputs[1].default_value = val; m.location = loc
        nt.links.new(sock, m.inputs[0]); return m.outputs[0]

    def mul(a, b, loc):
        m = n.new("ShaderNodeMath"); m.operation = "MULTIPLY"; m.location = loc
        nt.links.new(a, m.inputs[0]); nt.links.new(b, m.inputs[1]); return m.outputs[0]

    h_ok = cmp(sep.outputs[0], "LESS_THAN", 0.12, (0, 520))
    s_lo = cmp(sep.outputs[1], "GREATER_THAN", 0.10, (0, 470))
    s_hi = cmp(sep.outputs[1], "LESS_THAN", 0.70, (0, 420))
    v_ok = cmp(sep.outputs[2], "GREATER_THAN", 0.35, (0, 370))
    mask = mul(mul(h_ok, s_lo, (160, 500)), mul(s_hi, v_ok, (160, 400)), (300, 450))
    ao = n.new("ShaderNodeAmbientOcclusion"); ao.location = (-200, 250)
    ao.inputs["Distance"].default_value = ao_distance
    ao.samples = 16
    inv = n.new("ShaderNodeMath"); inv.operation = "SUBTRACT"; inv.inputs[0].default_value = 1.0
    inv.location = (0, 250); nt.links.new(ao.outputs["AO"], inv.inputs[1])
    # sharpen the falloff a little so it reads as a painted crease, not grime
    pw = n.new("ShaderNodeMath"); pw.operation = "POWER"; pw.inputs[1].default_value = 0.7
    pw.location = (160, 250); nt.links.new(inv.outputs[0], pw.inputs[0])
    k = n.new("ShaderNodeValue"); k.name = "PinkStrength"; k.label = "Pink shading (driven)"
    k.outputs[0].default_value = strength; k.location = (160, 150)
    fac = mul(mul(pw.outputs[0], mask, (340, 300)), k.outputs[0], (480, 300))
    clamp = n.new("ShaderNodeClamp"); clamp.location = (620, 300); nt.links.new(fac, clamp.inputs[0])
    mix = n.new("ShaderNodeMix"); mix.data_type = "RGBA"; mix.blend_type = "MIX"; mix.location = (760, 200)
    nt.links.new(clamp.outputs[0], mix.inputs["Factor"])
    nt.links.new(src, mix.inputs[6])
    mix.inputs[7].default_value = PINK
    nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    bsdf.location = (1000, 0)
    if rig is not None:
        fc = k.outputs[0].driver_add("default_value")
        d = fc.driver; d.type = "SCRIPTED"
        v = d.variables.new(); v.name = "v"; v.targets[0].id = rig; v.targets[0].data_path = f'["{prop}"]'
        d.expression = f"v * {strength}"
