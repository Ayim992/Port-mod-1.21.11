"""Tile review renders into one contact sheet: python3 sheet.py Jenny"""
import sys, glob, os
from PIL import Image
name = sys.argv[1]
rows = []
for tag in ("dressed", "nude"):
    ims = [Image.open(f"out/renders/{name}_{tag}_{v}.png") for v in ("front", "q3", "side", "back")]
    rows.append(ims)
w, h = rows[0][0].size
sheet = Image.new("RGB", (w * 4, h * len(rows)), "white")
for r, ims in enumerate(rows):
    for c, im in enumerate(ims):
        sheet.paste(im, (c * w, r * h))
out = f"out/renders/{name}_sheet.png"
sheet.save(out)
print(out)
