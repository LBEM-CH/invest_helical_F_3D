#!/usr/bin/env python3
"""
relion2dynamo — convert RELION 5 (tomography) particle poses to a Dynamo .tbl.

author: Wen-Lu Chung

Reuses the exact relion -> dynamo pose conversion that invest_helical_F_3D uses
(relion_star.relion_to_dynamo_eulers), so a table written here is identical to
the app's temp.tbl. The Dynamo angles land in columns 7-9; from there the
standard Dynamo -> ChimeraX placement is:

    sed -E 's/[+-][0-9]+\\.?[0-9]*([eE][+-]?[0-9]+)?i//g' temp.tbl > cleaned_table.tbl
    awk '{tmp=$7; $7=-$9-180; $9=-tmp+180; print}' cleaned_table.tbl > modified_table.tbl

Dynamo .tbl columns filled (1-based): tag=1, aligned/averaged=2,3, euler=7-9,
cc=10, tomo=20, class=22, filament(HelicalTubeID)=23, xyz=24-26. Shifts (cols
4-6) are left at 0 -- this table carries the orientation/position, not the
sub-pixel origin refinement.

CLI:
    python relion2dynamo.py particles.star [-o temp.tbl] [--tomo TOMONAME]
API:
    from relion2dynamo import convert, assemble_table
    table, tomo_map = convert("particles.star", out_path="temp.tbl")
"""

from __future__ import annotations

import numpy as np

# Dynamo .tbl is 35 columns wide; 0-based positions of the fields we set.
_TBL_NCOLS = 35
_C_TAG = 0
_C_ALIGNED = 1
_C_AVERAGED = 2
_C_EULER = slice(6, 9)        # cols 7-9: tdrot, tilt, narot (ZXZ extrinsic)
_C_CC = 9
_C_TOMO = 19                  # col 20
_C_CLASS = 21                 # col 22
_C_FIL = 22                   # col 23 (annotation = HelicalTubeID)
_C_XYZ = slice(23, 26)        # cols 24-26


def assemble_table(pid, eulers, tube, xyz, tomo_id=None) -> np.ndarray:
    """Build a Dynamo .tbl array (N x 35) from per-particle fields.

    pid     : (N,) tag (rlnTomoParticleId).
    eulers  : (N,3) Dynamo ZXZ-extrinsic angles (deg).
    tube    : (N,) filament id (rlnHelicalTubeID) -> col 23.
    xyz     : (N,3) coordinates (px) -> cols 24-26.
    tomo_id : (N,) integer tomogram id for col 20 (default all 1).
    Rows are returned sorted by tag, matching the Dynamo loader.
    """
    pid = np.asarray(pid)
    eulers = np.asarray(eulers, dtype=float)
    tube = np.asarray(tube)
    xyz = np.asarray(xyz, dtype=float)
    n = len(pid)
    tomo_id = np.ones(n) if tomo_id is None else np.asarray(tomo_id, dtype=float)

    tbl = np.zeros((n, _TBL_NCOLS), dtype=float)
    tbl[:, _C_TAG] = pid
    tbl[:, _C_ALIGNED] = 1.0
    tbl[:, _C_AVERAGED] = 1.0
    tbl[:, _C_EULER] = eulers
    tbl[:, _C_CC] = 1.0
    tbl[:, _C_TOMO] = tomo_id
    tbl[:, _C_CLASS] = 1.0
    tbl[:, _C_FIL] = tube
    tbl[:, _C_XYZ] = xyz
    return tbl[np.argsort(tbl[:, _C_TAG], kind="stable")]


def convert(star_path: str, out_path: str | None = None, tomo: str | None = None):
    """Convert a RELION 5 tomography .star to a Dynamo .tbl.

    star_path : path to a particles .star.
    out_path  : if given, write the .tbl there (space-separated, %g).
    tomo      : restrict to one rlnTomoName; default: all tomograms, each given a
                sequential integer id in col 20.

    Returns (table, tomo_map) where table is the (N, 35) array and tomo_map maps
    {col20_int: rlnTomoName}.
    """
    from relion_star import (read_star, _Table, _alignment_eulers,
                             relion_to_dynamo_eulers)

    t = _Table(read_star(star_path)["data_particles"])
    names = t.col("rlnTomoName", dtype=str)
    if tomo is None:
        mask = np.ones(len(names), dtype=bool)
    else:
        mask = names == str(tomo)
        if not mask.any():
            raise ValueError(f"tomogram {tomo!r} not in star "
                             f"(have {sorted(set(names.tolist()))})")

    pid = t.col("rlnTomoParticleId", dtype=float).astype(int)[mask]
    xyz = np.column_stack([t.col("rlnCoordinateX"), t.col("rlnCoordinateY"),
                           t.col("rlnCoordinateZ")])[mask]
    subtomo = np.column_stack([t.col("rlnTomoSubtomogramRot"),
                               t.col("rlnTomoSubtomogramTilt"),
                               t.col("rlnTomoSubtomogramPsi")])[mask]
    align = _alignment_eulers(t)[mask]
    if t.has("rlnHelicalTubeID"):
        tube = t.col("rlnHelicalTubeID", dtype=float).astype(int)[mask]
    else:
        tube = np.ones(int(mask.sum()), dtype=int)

    eulers = relion_to_dynamo_eulers(subtomo, align)

    sel_names = names[mask]
    uniq = sorted(set(sel_names.tolist()))
    tnum = {nm: i + 1 for i, nm in enumerate(uniq)}
    tomo_id = np.array([tnum[nm] for nm in sel_names], dtype=float)

    table = assemble_table(pid, eulers, tube, xyz, tomo_id)
    if out_path:
        np.savetxt(out_path, table, fmt="%g")
    return table, {i + 1: nm for i, nm in enumerate(uniq)}


def main(argv=None):
    import argparse
    import os
    import sys

    ap = argparse.ArgumentParser(
        description="Convert a RELION 5 tomography particles .star to a Dynamo .tbl.")
    ap.add_argument("star", help="RELION 5 particles .star file")
    ap.add_argument("-o", "--out", default=None,
                    help="output .tbl (default: <star_dir>/temp.tbl)")
    ap.add_argument("--tomo", default=None,
                    help="restrict to one rlnTomoName (default: all tomograms)")
    args = ap.parse_args(argv)

    out = args.out or os.path.join(os.path.dirname(args.star) or ".", "temp.tbl")
    table, tomo_map = convert(args.star, out_path=out, tomo=args.tomo)
    sys.stderr.write(
        f"wrote {out}: {table.shape[0]} particles x {table.shape[1]} columns\n")
    for tid, nm in tomo_map.items():
        sys.stderr.write(f"  col 20 tomo {tid} = {nm}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
