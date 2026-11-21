"""Build rigged character .blend files headless.

    python3 build.py jenny marie bia      # or: all
    python3 build.py jenny --no-render
    python3 build.py jenny --samples 24

Needs the `bpy` (Blender 5.3 as a Python module) and `scikit-image` pip packages,
or run inside Blender:  blender -b --factory-startup --python-exit-code 1 -P build.py -- jenny
"""
import argparse
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lib import builder  # noqa: E402

ALL = ["jenny", "marie", "bia"]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("characters", nargs="+")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--width", type=int, default=900)
    a = ap.parse_args(argv)
    names = ALL if a.characters == ["all"] else a.characters
    for n in names:
        spec = importlib.import_module(f"characters.{n}")
        path, renders = builder.build(spec, render=not a.no_render, samples=a.samples, width=a.width)
        print(f"RESULT {n} {path} renders={len(renders)}", flush=True)


if __name__ == "__main__":
    main()
