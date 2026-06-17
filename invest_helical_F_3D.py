#!/usr/bin/env python3
"""
invest_helical_F_3D — interactive triage of helical-filament segments.

author: Wen-Lu Chung

Opens a Dynamo `averages` folder (the two refined_table_ref_00X_iteYYYY.tbl
files) OR a RELION 5 tomography particles .star, reconstructs the per-filament
helical roll check from subtomo_averaging/msa_human.ipynb, and lets you mark bad
segments for removal:

  * overview: a grid of per-filament roll-vs-position panels + the tomogram XY
    map; hover a panel to highlight that filament on the map, click to drill in.
  * detail (per filament): roll vs position, residual-to-model, and the XY map,
    all linked — point at a segment in one and it lights up in all three. Mark
    segments with a rubber-band drag, "Select all", or by clicking points.

Twist, rise and pixel size are live controls in both windows: retune them and
every model overlay refits instantly, at real Angstrom scale.

The marked segments are written continuously to a plain text file (one id per
line: the Dynamo tag / col 1 for Dynamo, the _rlnTomoParticleId for RELION).
That file is both the resume list (reloaded on the next launch) and the export
you feed to your own .tbl / .star cleanup.

Usage:
    # Dynamo (auto-detected from the folder)
    python invest_helical_F_3D.py /path/to/.../ite_0004/averages [--tomo 1]
    # RELION (auto-detected from the .star extension)
    python invest_helical_F_3D.py /path/to/particles_relion5.star
        [--twist -1.4] [--rise 4.75] [--pixelsize 7.92] [--cols 5]
        [--out remove_list.txt]

Built for `ssh -XY`: PyQt6 + pyqtgraph (raster, no OpenGL). Requires Python 3.9+.
"""

from __future__ import annotations

import sys

if sys.version_info < (3, 9):
    sys.stderr.write("invest_helical_F_3D requires Python 3.9+ "
                     "(running {0}.{1}).\n".format(*sys.version_info[:2]))
    sys.exit(1)

import argparse
import os

DEFAULT_PIXELSIZE = 7.92          # A/px (bin-4); used for Dynamo when not given


def _import_or_die(relion: bool):
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        import pyqtgraph  # noqa: F401
        from PyQt6 import QtWidgets  # noqa: F401
        if relion:
            import eulerangles  # noqa: F401
    except ImportError as e:
        sys.stderr.write(
            f"missing dependency: {e}\n"
            "Install into your venv:  pip install -r requirements.txt\n")
        sys.exit(1)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Interactive helical-filament segment triage (mark for removal).")
    ap.add_argument("path", help="Dynamo project folder (…/abp_align_eo, shows "
                                 "per-iteration paths), a single averages folder, "
                                 "or a RELION .star file")
    fmt = ap.add_mutually_exclusive_group()
    fmt.add_argument("--dynamo", action="store_true",
                     help="treat `path` as a Dynamo averages folder")
    fmt.add_argument("--relion", action="store_true",
                     help="treat `path` as a RELION 5 particles .star")
    ap.add_argument("--tomo", default=None,
                    help="tomogram to load (Dynamo: col 20 id; RELION: TomoName); "
                         "default: only/first present")
    ap.add_argument("--twist", type=float, default=-1.4, help="helix twist, deg/subunit")
    ap.add_argument("--rise", type=float, default=4.75, help="helix rise, Angstrom/subunit")
    ap.add_argument("--pixelsize", type=float, default=None,
                    help="Angstrom/px; RELION default reads the star optics, "
                         f"Dynamo default {DEFAULT_PIXELSIZE}")
    ap.add_argument("--map", default=None,
                    help="reference average map (.mrc/.map/.em) for the 3D view "
                         "density overlay; can also be loaded from the 3D window")
    ap.add_argument("--cols", type=int, default=5, help="panel columns in the overview")
    ap.add_argument("--out", default=None,
                    help="remove-list path (default: <dir>/remove_list.txt)")
    ap.add_argument("--no-temp", action="store_true",
                    help="do not write temp.tbl into a Dynamo folder")
    glgrp = ap.add_mutually_exclusive_group()
    glgrp.add_argument("--gl", action="store_true",
                       help="force-enable the OpenGL 3D view (local display / VirtualGL)")
    glgrp.add_argument("--no-gl", action="store_true",
                       help="disable OpenGL/View-3D; avoids GLX crashes over ssh -XY")
    return ap.parse_args(argv)


def resolve_gl(args) -> bool:
    """Whether the OpenGL 3D view can run. Default: off over SSH (forwarded X
    rarely has usable GLX and Qt aborts trying), on locally. --gl / --no-gl and a
    pre-set QT_XCB_GL_INTEGRATION override."""
    if args.no_gl:
        return False
    if args.gl:
        return True
    env = os.environ.get("QT_XCB_GL_INTEGRATION")
    if env is not None:
        return env != "none"
    return not os.environ.get("SSH_CONNECTION")


def detect_format(args) -> str:
    if args.relion:
        return "relion"
    if args.dynamo:
        return "dynamo"
    if os.path.isfile(args.path) and args.path.lower().endswith(".star"):
        return "relion"
    if os.path.isdir(args.path):
        from relion_star import is_refine_job
        return "relion" if is_refine_job(args.path) else "dynamo"
    sys.stderr.write(
        f"cannot tell if {args.path!r} is Dynamo or RELION; "
        "pass --dynamo or --relion.\n")
    sys.exit(1)


def resolve_pixelsize(args, fmt: str) -> float:
    if args.pixelsize is not None:
        return args.pixelsize
    if fmt == "relion":
        from relion_star import final_star, image_pixel_size
        px = image_pixel_size(final_star(args.path))
        if px is not None:
            sys.stderr.write(f"pixel size from star optics: {px} A/px\n")
            return px
        sys.stderr.write("no pixel size in star optics; assuming 1.0 A/px "
                         "(pass --pixelsize)\n")
        return 1.0
    sys.stderr.write(f"assuming pixel size {DEFAULT_PIXELSIZE} A/px "
                     "(pass --pixelsize to override)\n")
    return DEFAULT_PIXELSIZE


def choose_tomogram(path, fmt):
    """Startup chooser when --tomo is omitted and several tomograms exist."""
    from PyQt6 import QtWidgets
    if fmt == "relion":
        from relion_star import final_star, tomogram_names
        tomos = tomogram_names(final_star(path))
    else:
        from dynamo_table import available_tomograms
        tomos = available_tomograms(path)
    if len(tomos) == 1:
        return tomos[0]
    item, ok = QtWidgets.QInputDialog.getItem(
        None, "invest_helical_F_3D", "Tomogram to load:",
        [str(t) for t in tomos], 0, False)
    if not ok:
        sys.exit(0)
    # map the chosen string back to the native id type
    return next(t for t in tomos if str(t) == item)


def main(argv=None):
    args = parse_args(argv)
    fmt = detect_format(args)
    _import_or_die(relion=(fmt == "relion"))

    import pyqtgraph as pg
    from PyQt6 import QtWidgets
    from dynamo_table import load_dataset
    from selection_store import SelectionStore
    from plot_common import ModelParams
    from overview_window import OverviewWindow

    if fmt == "dynamo" and not os.path.isdir(args.path):
        sys.stderr.write(f"not a folder: {args.path}\n")
        sys.exit(1)
    if fmt == "relion" and not os.path.exists(args.path):
        sys.stderr.write(f"not found: {args.path}\n")
        sys.exit(1)

    # Decide on OpenGL BEFORE the QApplication: over SSH, disabling the xcb GLX
    # integration keeps the 2D app from aborting on "Could not initialize GLX".
    gl_enabled = resolve_gl(args)
    if not gl_enabled:
        os.environ["QT_XCB_GL_INTEGRATION"] = "none"
    elif os.environ.get("QT_XCB_GL_INTEGRATION") == "none":
        del os.environ["QT_XCB_GL_INTEGRATION"]        # --gl overrides a stale 'none'

    pg.setConfigOptions(antialias=False, useOpenGL=False, background="w", foreground="k")

    # Every View-3D window is its own GLViewWidget with its own GL context, but
    # pyqtgraph caches each compiled shader program's GL id globally. Opening a
    # second 3D window (e.g. another filament) would then glUseProgram() an id
    # from the first window's context -> GLError 1281 (invalid value), spammed on
    # every repaint. Sharing GL contexts app-wide makes those ids valid in every
    # window. Must be set before the QApplication is constructed.
    if gl_enabled:
        from PyQt6 import QtCore
        QtWidgets.QApplication.setAttribute(
            QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QtWidgets.QApplication(sys.argv[:1])

    pixelsize = resolve_pixelsize(args, fmt)
    tomo = args.tomo if args.tomo is not None else choose_tomogram(args.path, fmt)

    ds = load_dataset(args.path, fmt, tomo, args.twist, args.rise, pixelsize,
                      write_temp=not args.no_temp)
    params = ModelParams(ds)
    out_dir = args.path if os.path.isdir(args.path) else os.path.dirname(args.path)
    out = args.out or os.path.join(out_dir, "remove_list.txt")
    store = SelectionStore(out)

    map_volume = map_voxel = None
    if args.map:
        from volume_io import read_volume
        map_volume, map_voxel = read_volume(args.map)
        if map_voxel is None:
            map_voxel = pixelsize
        sys.stderr.write(
            f"loaded map {args.map}: {map_volume.shape}, voxel {map_voxel} A/px\n")

    win = OverviewWindow(ds, store, params, cols=args.cols,
                         map_volume=map_volume, map_voxel=map_voxel,
                         gl_enabled=gl_enabled)
    win.show()
    sys.stderr.write(
        f"loaded {fmt} tomo {ds.tomo}: {len(ds.filaments)} filaments, "
        f"{ds.n_segments} segments. remove list -> {out} "
        f"({store.count()} already marked)\n")
    if not gl_enabled:
        sys.stderr.write(
            "3D view disabled (no usable OpenGL — typical over ssh -XY); 2D triage "
            "works. Run locally, or with --gl / VirtualGL, for View 3D.\n")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
