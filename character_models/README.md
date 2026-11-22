# Character models: Jenny, Marie, Bia

## Exact models (from the sexmod GeckoLib files): `out/Jenny_*.blend`, `out/Bia_*.blend`
`import_geckolib.py` converts `sources/<name>/` (geo.json + animation.json + texture) 1:1:
- **Geometry:** every cube keeps its exact origin, size, inflate, pivot, rotation and per-face UV on the original 4096² texture.
  Scale is 1 px = 1/16 m, as in Minecraft.
- **Rig:** each GeckoLib bone becomes an armature bone at its exact pivot and rest orientation. Cubes are 100% bound to their bone,
  exactly how the mod deforms them (including `boobs`, `booty`/`cheek*`, `curves*`, `side*`, `vagina`, the face expression bones and so on).
- **Animations:** every mod animation is baked to an action, one key per frame at 24 fps. Jenny has 39 (doggy*, blowjob*, paizuri*, missionary, strip,
  idle, walk…). Bia has 41 (anal*, prone_doggy*, talk_*, headpat, sitdown…). Easing curves and constant molang are evaluated like GeckoLib does.
  Pick one in the Action editor on `<Name>_<Variant>_rig`. A loop's frame range is stored on the action, and `act["loop"]` records the loop mode.
- **Partner:** the `steve` bone tree is the partner used by the scene animations. Turn on the rig's `steve` property to show it.
  It uses a standard 64×64 player-skin layout (`out/textures/partner_skin.png`, a generated Steve-style skin), so swap in any skin you like.
- `ArmorSlots`: the mod's armor/item overlay cubes. They're hidden because the game only draws them when armor is worn.
- Rendering notes: the material does back-face culling and a 0.5 alpha cutout like Minecraft, and transparent bounces are raised to 128.
  Paper-thin decal planes that face into the body are dropped, and plane faces are pulled 0.01 px apart; otherwise Cycles shows black patches.

```
python3 import_geckolib.py jenny bia            # both variants each; --variants nude|dressed, --no-anim, --render
python3 animtest.py Jenny_Nude jenny.doggyslow 3  # render frames of an action
```

## Procedural models (earlier approximation)

Procedural, rigged Blender 5.3 models built from the reference sheets.
Each `out/<Name>.blend` is self-contained: textures are packed and the metarig is kept for regeneration.

## Open and animate
1. Open `out/Jenny.blend` (or Marie/Bia). When Blender asks, **allow scripts**; otherwise the Rigify `rig_ui.py` panel
   (IK/FK switches, snapping) won't load. You can also enable *Preferences > Save & Load > Auto Run Python Scripts*.
2. Select `<Name>_rig` and enter Pose Mode. It's a full Rigify rig:
   - Legs: IK by default (`foot_ik.L/R`, knee poles, foot roll via `foot_heel_ik`). Set `IK_FK` = 1 on `thigh_parent.*` for FK.
   - Arms: IK and FK (`upper_arm_parent.*` → `IK_FK`).
   - Torso: `torso`, `hips`, `chest`, `neck`, `head`, plus tweak bones.
   - Soft-body controls: `breast.L/R`, `butt.L/R`, `belly`, and **jiggle** keyframes or the Wiggle 2 add-on go on these.
   - Extras: Bia has `tail`, `ear.L/R`, `leaves`. Marie has `hat`, `cloth_front`, `cloth_back`.
3. Custom properties on the rig object (N panel > Item > Properties), all animatable:

| Property | Values |
|---|---|
| `outfit` | 1 dressed / 0 nude (switches skin layers and clothing props) |
| `expression` | 0 neutral, 1 blink, 2 lidded, 3 smile, 4 moan, 5 ahegao, 6 heart eyes |
| `hat` (Marie) | 1 shows hat + lantern |

## Structure
- `Body`: one watertight mesh with weights to the DEF bones. Its colors live in color attributes `Col` / `ColNude` and `Emit` / `EmitNude`,
  so you can Vertex Paint to touch them up.
- `Arm.L/R`: Minecraft-style separate arm boxes that bend at the elbow.
- `Head`: an 8×8×8 px cube. The front face uses an expression atlas (`out/textures/<Name>_face.png`, 32 px per expression).
  Edit the PNG and reload it, or texture-paint it directly.
- Props (hair, ears, leaves, hat, collar, potions…): voxel meshes rigidly bound to their bone.
- Scale: 1 Minecraft pixel = 1.8/32 m, so characters are 1.8 m tall.

## Rebuild / tweak
```
pip install bpy==5.3.0 scikit-image
python3 build.py all                 # or: jenny marie bia; flags: --no-render --samples N --width W
python3 posetest.py Jenny            # stress-pose deformation renders
python3 sheet.py Jenny               # contact sheet of the turnaround renders
```
Proportions are in `characters/<name>.py` → `P = anatomy.params(...)`: breast/butt/hip/thigh sizes, crotch height, arm width.
Paint rules (`paint()`), props (`props()`), and pixel faces (`face()`) live in the same file.
Shared code: `lib/anatomy.py` (body SDF + rig landmarks), `lib/rig.py` (Rigify + skin weights), `lib/faces.py` (expressions).
