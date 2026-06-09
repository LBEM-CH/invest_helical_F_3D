#!/usr/bin/env python3
"""
Dynamo table loading for invest_helical_F_3D.

author: Wen-Lu Chung

Mirrors the notebook: concatenate refined_table_ref_001 + ref_002 from the given
averages folder, sort by tag (col 1), filter to one tomogram, and group into
filaments with the per-filament helical fit precomputed once at load.

.tbl columns (0-based):  tag=0,  ZXZ euler=6:9,  tomo=19,  filament id=22,  coords=23:26
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np

from helix_geom import fit_filament

# Column indices (0-based) into the Dynamo .tbl.
COL_TAG = 0
COL_EULER = slice(6, 9)
COL_TOMO = 19
COL_FIL = 22
COL_XYZ = slice(23, 26)


@dataclass
class Filament:
    """One filament, all per-segment arrays ordered head->tail by real position."""
    fid: int
    tags: np.ndarray        # (N,) Dynamo tags (col 0), int, ordered head->tail
    pos: np.ndarray         # (N,) position along axis (px), centered at the middle
    phi: np.ndarray         # (N,) measured roll (deg)
    delta: np.ndarray       # (N,) residual to the screw model (deg)
    xy: np.ndarray          # (N, 2) tomogram X,Y for the map, ordered head->tail
    phi0: float             # model phase (deg)

    @property
    def n(self) -> int:
        return len(self.tags)


@dataclass
class Dataset:
    """All filaments of one tomogram, plus the working table for export bookkeeping."""
    folder: str
    tomo: int
    model_rate: float
    table: np.ndarray                 # working rows (this tomogram), sorted by tag (str dtype)
    filaments: list[Filament]

    @property
    def pos_halfspan(self) -> float:
        """Largest |pos| across filaments with >=5 segments -> shared x-scale (px)."""
        h = 0.0
        for f in self.filaments:
            if f.n >= 5:
                h = max(h, float(np.abs(f.pos).max()))
        return h * 1.05 if h > 0 else 1.0

    def all_tags(self) -> np.ndarray:
        return self.table[:, COL_TAG].astype(float).astype(int)


def find_ref_tables(folder: str) -> list[str]:
    """Locate the two refined_table_ref_00X_iteYYYY.tbl files in `folder`."""
    hits = sorted(glob.glob(os.path.join(folder, "refined_table_ref_*_ite_*.tbl")))
    if not hits:
        raise FileNotFoundError(
            f"no refined_table_ref_*_ite_*.tbl found in {folder!r}")
    return hits


def load_dataset(folder: str, tomo: int | None, model_rate: float,
                 write_temp: bool = True) -> Dataset:
    """Load + concat + sort the ref tables, pick a tomogram, build filaments.

    tomo : tomogram id to keep; if None, the only/first tomogram is used.
    Writes temp.tbl (the working rows) into `folder` like the notebook, unless
    write_temp is False.
    """
    tables = find_ref_tables(folder)
    par = np.concatenate(
        [np.loadtxt(t, comments="#", dtype=str, ndmin=2) for t in tables], axis=0)
    par = par[np.argsort(par[:, COL_TAG].astype(int))]            # sort by tag

    tomo_ids = np.unique(par[:, COL_TOMO].astype(float).astype(int))
    if tomo is None:
        tomo = int(tomo_ids[0])
    elif tomo not in tomo_ids:
        raise ValueError(f"tomogram {tomo} not in table (have {tomo_ids.tolist()})")

    table = par[par[:, COL_TOMO].astype(float).astype(int) == tomo, :]
    if write_temp:
        np.savetxt(os.path.join(folder, "temp.tbl"), table, delimiter=" ", fmt="%s")

    fil_col = table[:, COL_FIL].astype(float).astype(int)
    filaments: list[Filament] = []
    for fid in np.unique(fil_col):
        rows = table[fil_col == fid, :]
        tags = rows[:, COL_TAG].astype(float).astype(int)
        xyz = rows[:, COL_XYZ].astype(float)
        if len(rows) < 5:
            # too short to fit an axis; keep raw order so it still shows/marks.
            filaments.append(Filament(
                fid=int(fid), tags=tags,
                pos=np.zeros(len(rows)), phi=np.full(len(rows), np.nan),
                delta=np.full(len(rows), np.nan), xy=xyz[:, :2], phi0=float("nan")))
            continue
        eulers = rows[:, COL_EULER].astype(float)
        fit = fit_filament(xyz, eulers, model_rate)
        o = fit["order"]
        filaments.append(Filament(
            fid=int(fid), tags=tags[o], pos=fit["pos"], phi=fit["phi"],
            delta=fit["delta"], xy=xyz[o, :2], phi0=fit["phi0"]))

    return Dataset(folder=folder, tomo=int(tomo), model_rate=model_rate,
                   table=table, filaments=filaments)


def available_tomograms(folder: str) -> list[int]:
    """Tomogram ids present across the ref tables (for the startup chooser)."""
    tables = find_ref_tables(folder)
    par = np.concatenate(
        [np.loadtxt(t, comments="#", dtype=str, ndmin=2) for t in tables], axis=0)
    return sorted(int(x) for x in np.unique(par[:, COL_TOMO].astype(float).astype(int)))
