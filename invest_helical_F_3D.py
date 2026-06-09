#!/usr/bin/env python3
"""
invest_helical_F_3D — interactive triage of helical-filament segments.

author: Wen-Lu Chung

Opens a Dynamo `averages` folder (the two refined_table_ref_00X_iteYYYY.tbl
files), reconstructs the per-filament helical roll check from
subtomo_averaging/msa_human.ipynb, and lets you mark bad segments for removal:

  * overview: a grid of per-filament roll-vs-position panels + the tomogram XY
    map; hover a panel to highlight that filament on the map, click to drill in.
  * detail (per filament): roll vs position, residual-to-model, and the XY map,
    all linked — point at a segment in one and it lights up in all three. Mark
    segments with a rubber-band drag, "Select all", or by clicking points.

The marked segments are written continuously to a plain text file (one Dynamo
tag / col 1 per line). That file is both the resume list (reloaded on the next
launch) and the export you feed to your own .tbl cleanup.

Usage:
    python invest_helical_F_3D.py /path/to/.../ite_0004/averages [--tomo 1]
        [--twist -1.4] [--rise 0.6] [--cols 5] [--out remove_list.txt]

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


def _import_or_die():
    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401
        import pyqtgraph  # noqa: F401
        from PyQt6 import QtWidgets  # noqa: F401
    except ImportError as e:
        sys.stderr.write(
            f"missing dependency: {e}\n"
            "Install into your venv:  pip install -r requirements.txt\n")
        sys.exit(1)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Interactive helical-filament segment triage (mark for removal).")
    ap.add_argument("folder", help="Dynamo averages folder with refined_table_ref_*.tbl")
    ap.add_argument("--tomo", type=int, default=None,
                    help="tomogram id (col 20) to load; default: only/first present")
    ap.add_argument("--twist", type=float, default=-1.4, help="helix twist, deg/subunit")
    ap.add_argument("--rise", type=float, default=0.6, help="helix rise, px/subunit")
    ap.add_argument("--cols", type=int, default=5, help="panel columns in the overview")
    ap.add_argument("--out", default=None,
                    help="remove-list path (default: <folder>/remove_list.txt)")
    ap.add_argument("--no-temp", action="store_true",
                    help="do not write temp.tbl into the folder")
    return ap.parse_args(argv)


def choose_tomogram(folder, app):
    """Startup chooser when --tomo is omitted and several tomograms exist."""
    from PyQt6 import QtWidgets
    from dynamo_table import available_tomograms
    tomos = available_tomograms(folder)
    if len(tomos) == 1:
        return tomos[0]
    item, ok = QtWidgets.QInputDialog.getItem(
        None, "invest_helical_F_3D", "Tomogram to load:",
        [str(t) for t in tomos], 0, False)
    if not ok:
        sys.exit(0)
    return int(item)


def main(argv=None):
    args = parse_args(argv)
    _import_or_die()

    import pyqtgraph as pg
    from PyQt6 import QtWidgets
    from dynamo_table import load_dataset
    from selection_store import SelectionStore
    from overview_window import OverviewWindow

    if not os.path.isdir(args.folder):
        sys.stderr.write(f"not a folder: {args.folder}\n")
        sys.exit(1)

    pg.setConfigOptions(antialias=False, useOpenGL=False, background="w", foreground="k")
    app = QtWidgets.QApplication(sys.argv[:1])

    tomo = args.tomo if args.tomo is not None else choose_tomogram(args.folder, app)
    model_rate = args.twist / args.rise

    ds = load_dataset(args.folder, tomo, model_rate, write_temp=not args.no_temp)
    out = args.out or os.path.join(args.folder, "remove_list.txt")
    store = SelectionStore(out)

    win = OverviewWindow(ds, store, cols=args.cols)
    win.show()
    sys.stderr.write(
        f"loaded tomo {ds.tomo}: {len(ds.filaments)} filaments, "
        f"{len(ds.table)} segments. remove list -> {out} "
        f"({store.count()} already marked)\n")
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
