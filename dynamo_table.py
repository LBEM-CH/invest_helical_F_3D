#!/usr/bin/env python3
"""
Dataset loading for invest_helical_F_3D (Dynamo .tbl or RELION .star).

author: Wen-Lu Chung

Both inputs are funnelled into the same Dataset of Filaments so the GUI never
cares which format it came from:

  Dynamo : concatenate refined_table_ref_001 + ref_002, sort by tag (col 1),
           filter to one tomogram (col 20), group into filaments (col 23).
           .tbl columns (0-based): tag=0, ZXZ euler=6:9, tomo=19, fil=22, xyz=23:26
  RELION : read one tomogram's particles from a RELION 5 star, group by
           _rlnHelicalTubeID; relion_star converts the poses to the SAME Dynamo
           ZXZ-extrinsic convention before they reach helix_geom.

The pose-only part of the fit (position along the axis, measured roll) is done
once at load. The model overlay (phi0, residual delta) and the Angstrom scaling
of position both depend on twist / rise / pixel-size, which the GUI can retune
live -- Dataset.set_params() recomputes just that cheap part across filaments.
"""

from __future__ import annotations

import glob
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np

from helix_geom import fit_model, fit_pose, roll_from_eulers

# Dynamo .tbl column indices (0-based).
COL_TAG = 0
COL_EULER = slice(6, 9)
COL_TOMO = 19
COL_FIL = 22
COL_XYZ = slice(23, 26)


@dataclass
class Filament:
    """One filament, all per-segment arrays ordered head->tail by real position.

    pos_px and phi are pose-only (independent of twist/rise/pixel-size). pos
    (Angstrom), phi0 and delta are filled by apply_model() and refreshed whenever
    the model parameters change.
    """
    fid: int
    tags: np.ndarray            # (N,) Dynamo tag / RELION TomoParticleId, int
    pos_px: np.ndarray          # (N,) position along axis (px), centered, model-free
    phi: np.ndarray             # (N,) measured roll (deg), model-free
    xy: np.ndarray              # (N, 2) tomogram X,Y (px), ordered head->tail
    xyz: np.ndarray             # (N, 3) tomogram X,Y,Z (px), ordered head->tail (for 3D)
    eulers: np.ndarray          # (N, 3) ZXZ-extrinsic pose (deg), ordered (for 3D glyphs)
    fittable: bool = False      # n >= 5: an axis could be fit
    pos: np.ndarray = field(default_factory=lambda: np.array([]))   # (N,) position (Angstrom)
    delta: np.ndarray = field(default_factory=lambda: np.array([])) # (N,) residual to model (deg)
    phi0: float = float("nan")  # model phase (deg)
    axis: np.ndarray = None     # (3,) unit filament axis (fittable only); for iteration roll
    traj_roll: np.ndarray = None  # (n_iter, N) roll per Dynamo iteration, ordered like tags; or None
    traj_iters: list = None     # iteration numbers matching traj_roll rows; or None

    @property
    def n(self) -> int:
        return len(self.tags)

    def apply_model(self, rate: float, pixelsize: float) -> None:
        """Refresh the model-dependent arrays for a new rate / pixel-size."""
        self.pos = self.pos_px * pixelsize
        if self.fittable:
            m = fit_model(self.pos, self.phi, rate)
            self.phi0 = m["phi0"]
            self.delta = m["delta"]
        else:
            self.phi0 = float("nan")
            self.delta = np.full(self.n, np.nan)


@dataclass
class Dataset:
    """All filaments of one tomogram, plus the live helix parameters."""
    source: str                 # folder (Dynamo) or .star path (RELION)
    fmt: str                    # "dynamo" | "relion"
    tomo: object                # tomogram id: int (Dynamo) or str (RELION TomoName)
    twist: float                # deg / subunit
    rise: float                 # Angstrom / subunit
    pixelsize: float            # Angstrom / px
    n_segments: int
    filaments: list[Filament]

    @property
    def model_rate(self) -> float:
        """Screw slope in deg azimuth per Angstrom (twist / rise)."""
        return self.twist / self.rise

    def set_params(self, twist: float, rise: float, pixelsize: float) -> None:
        self.twist, self.rise, self.pixelsize = twist, rise, pixelsize
        self.recompute()

    def recompute(self) -> None:
        for f in self.filaments:
            f.apply_model(self.model_rate, self.pixelsize)

    @property
    def pos_halfspan(self) -> float:
        """Largest |pos| (Angstrom) across fittable filaments -> shared x-scale."""
        h = 0.0
        for f in self.filaments:
            if f.fittable and len(f.pos):
                h = max(h, float(np.abs(f.pos).max()))
        return h * 1.05 if h > 0 else 1.0


def _build_filament(fid, tags, xyz, eulers) -> Filament:
    """Pose-only build shared by both formats. eulers are ZXZ-extrinsic (deg)."""
    tags = np.asarray(tags).astype(int)
    xyz = np.asarray(xyz, dtype=float)
    eulers = np.asarray(eulers, dtype=float)
    if len(tags) < 5:
        # too short to fit an axis; keep raw order so it still shows / can be marked.
        return Filament(fid=int(fid), tags=tags,
                        pos_px=np.zeros(len(tags)), phi=np.full(len(tags), np.nan),
                        xy=xyz[:, :2], xyz=xyz, eulers=eulers, fittable=False)
    fp = fit_pose(xyz, eulers)
    o = fp["order"]
    return Filament(fid=int(fid), tags=tags[o], pos_px=fp["pos"], phi=fp["phi"],
                    xy=xyz[o, :2], xyz=xyz[o], eulers=eulers[o], fittable=True,
                    axis=fp["axis"])


# --- Dynamo --------------------------------------------------------------------
def find_ref_tables(folder: str) -> list[str]:
    """Locate the refined_table_ref_00X_iteYYYY.tbl files in `folder`."""
    hits = sorted(glob.glob(os.path.join(folder, "refined_table_ref_*_ite_*.tbl")))
    if not hits:
        raise FileNotFoundError(
            f"no refined_table_ref_*_ite_*.tbl found in {folder!r}")
    return hits


def _iter_num(table_path: str):
    m = re.search(r"_ite_(\d+)\.tbl$", os.path.basename(table_path))
    return int(m.group(1)) if m else None


def find_iteration_folders(path: str) -> list[tuple[int, str]]:
    """Discover Dynamo iteration `averages/` folders under `path`, oldest->newest.

    Accepts the project folder (…/abp_align_eo, with results/ite_*/averages), the
    results/ folder, or a single averages folder. Returns [(iter_num, folder), …]
    for iterations whose averages hold a NON-EMPTY refined table — so the empty
    N+1 placeholder iteration is dropped automatically.
    """
    direct = glob.glob(os.path.join(path, "refined_table_ref_*_ite_*.tbl"))
    cands = [path] if direct else sorted(
        glob.glob(os.path.join(path, "results", "ite_*", "averages")) +
        glob.glob(os.path.join(path, "ite_*", "averages")))
    out = []
    for d in cands:
        tbls = [t for t in glob.glob(os.path.join(d, "refined_table_ref_*_ite_*.tbl"))
                if os.path.getsize(t) > 0]
        nums = [n for n in (_iter_num(t) for t in tbls) if n is not None]
        if nums:
            out.append((max(nums), d))
    if not out:
        raise FileNotFoundError(
            f"no Dynamo iteration tables (results/ite_*/averages/) under {path!r}")
    out.sort(key=lambda x: x[0])
    return out


def _load_concat_table(folder: str, tomo):
    """Concat the ref tables in `folder`, sort by tag, filter to one tomogram.
    Returns (tomo_id, table). tomo=None picks the first tomogram present."""
    tables = find_ref_tables(folder)
    par = np.concatenate(
        [np.loadtxt(t, comments="#", dtype=str, ndmin=2) for t in tables], axis=0)
    par = par[np.argsort(par[:, COL_TAG].astype(int))]
    tomo_ids = np.unique(par[:, COL_TOMO].astype(float).astype(int))
    if tomo is None:
        tomo = int(tomo_ids[0])
    elif int(tomo) not in tomo_ids:
        raise ValueError(f"tomogram {tomo} not in table (have {tomo_ids.tolist()})")
    table = par[par[:, COL_TOMO].astype(float).astype(int) == int(tomo), :]
    return int(tomo), table


def available_tomograms(path: str) -> list[int]:
    """Tomogram ids present (final iteration), for the startup chooser."""
    final_folder = find_iteration_folders(path)[-1][1]
    tables = find_ref_tables(final_folder)
    par = np.concatenate(
        [np.loadtxt(t, comments="#", dtype=str, ndmin=2) for t in tables], axis=0)
    return sorted(int(x) for x in np.unique(par[:, COL_TOMO].astype(float).astype(int)))


def _euler_map(tables, tomo_id) -> dict:
    """tag -> ZXZ euler triple, from a list of .tbl files filtered to one tomogram."""
    par = np.concatenate(
        [np.loadtxt(t, comments="#", dtype=str, ndmin=2) for t in tables], axis=0)
    par = par[par[:, COL_TOMO].astype(float).astype(int) == int(tomo_id)]
    tags = par[:, COL_TAG].astype(float).astype(int)
    return dict(zip(tags.tolist(), par[:, COL_EULER].astype(float)))


def _attach_trajectories(filaments, folders, tomo_id) -> None:
    """For each fittable filament, fill traj_roll: the per-segment roll measured at
    iteration 0 (the starting_values that seeded iteration 1) and every iteration,
    about the (fixed) final axis. Segments are matched by tag."""
    labels, maps = [], []
    # iteration 0: starting_values sit next to the FIRST iteration's averages
    start_dir = os.path.join(os.path.dirname(folders[0][1]), "starting_values")
    start_tbls = [t for t in sorted(glob.glob(
        os.path.join(start_dir, "starting_table_ref_*_ite_*.tbl")))
        if os.path.getsize(t) > 0]
    if start_tbls:
        labels.append(0)
        maps.append(_euler_map(start_tbls, tomo_id))
    for it, d in folders:
        labels.append(it)
        maps.append(_euler_map(find_ref_tables(d), tomo_id))

    nan3 = np.full(3, np.nan)
    for fil in filaments:
        if not fil.fittable:
            continue
        traj = np.full((len(maps), fil.n), np.nan)
        for ii, emap in enumerate(maps):
            eul = np.array([emap.get(int(t), nan3) for t in fil.tags])   # (N, 3)
            valid = np.isfinite(eul[:, 0])
            if valid.any():
                traj[ii, valid] = roll_from_eulers(eul[valid], fil.axis)
        traj[-1] = fil.phi                       # final row == the dot (path ends on it)
        fil.traj_roll = traj
        fil.traj_iters = labels


def _build_dynamo(path: str, tomo, write_temp: bool):
    folders = find_iteration_folders(path)
    final_folder = folders[-1][1]
    n_tables = len(find_ref_tables(final_folder))
    kind = "single-table" if n_tables == 1 else f"{n_tables}-table (concatenated)"
    sys.stderr.write(
        f"Dynamo {kind} job — iterations {[it for it, _ in folders]}\n")
    tomo_id, table = _load_concat_table(final_folder, tomo)
    if write_temp:
        np.savetxt(os.path.join(final_folder, "temp.tbl"), table, delimiter=" ", fmt="%s")

    fil_col = table[:, COL_FIL].astype(float).astype(int)
    filaments: list[Filament] = []
    for fid in np.unique(fil_col):
        rows = table[fil_col == fid, :]
        filaments.append(_build_filament(
            fid,
            rows[:, COL_TAG].astype(float).astype(int),
            rows[:, COL_XYZ].astype(float),
            rows[:, COL_EULER].astype(float)))
    if len(folders) > 1:                         # earlier iterations -> convergence trails
        _attach_trajectories(filaments, folders, tomo_id)
    return tomo_id, len(table), filaments


# --- RELION --------------------------------------------------------------------
def relion_to_dynamo_table(d: dict) -> np.ndarray:
    """Dynamo .tbl array from the relion-derived particle dict (single tomogram).

    Thin wrapper over relion2dynamo.assemble_table so the GUI's temp.tbl and the
    standalone relion2dynamo CLI always produce identical tables.
    """
    from relion2dynamo import assemble_table
    return assemble_table(d["pid"], d["eulers"], d["tube"], d["xyz"])


def _attach_relion_trajectories(filaments, iter_stars, tomo_name) -> None:
    """RELION analogue of _attach_trajectories: per fittable filament, roll at every
    refinement iteration (run_it000 = the start) about the fixed final axis, matched
    by _rlnTomoParticleId. Each iteration star's pose is converted to Dynamo eulers
    by load_particles, so the roll is computed identically to the final set."""
    from relion_star import load_particles
    labels, maps = [], []
    for label, star in iter_stars:
        _, d = load_particles(star, tomo_name)
        labels.append(label)
        maps.append(dict(zip(d["pid"].astype(int).tolist(), d["eulers"])))
    nan3 = np.full(3, np.nan)
    for fil in filaments:
        if not fil.fittable:
            continue
        traj = np.full((len(maps), fil.n), np.nan)
        for ii, emap in enumerate(maps):
            eul = np.array([emap.get(int(t), nan3) for t in fil.tags])   # (N, 3)
            valid = np.isfinite(eul[:, 0])
            if valid.any():
                traj[ii, valid] = roll_from_eulers(eul[valid], fil.axis)
        traj[-1] = fil.phi                       # final row == the dot (path ends on it)
        fil.traj_roll = traj
        fil.traj_iters = labels


def _build_relion(path: str, tomo, write_temp: bool = True):
    from relion_star import final_star, iteration_stars, load_particles
    star = final_star(path)
    iters = iteration_stars(path)
    if iters:
        sys.stderr.write(
            f"RELION refinement job — iterations {[it for it, _ in iters]} (+ run_data)\n")
    tomo_name, d = load_particles(star, None if tomo is None else str(tomo))
    if write_temp:
        out = os.path.join(os.path.dirname(star) or ".", "temp.tbl")
        try:
            np.savetxt(out, relion_to_dynamo_table(d), fmt="%g")
        except OSError as e:                         # job folders are often read-only
            sys.stderr.write(f"could not write {out} ({e}); skipping temp.tbl\n")
    filaments: list[Filament] = []
    for fid in np.unique(d["tube"]):
        sel = d["tube"] == fid
        filaments.append(_build_filament(
            fid, d["pid"][sel], d["xyz"][sel], d["eulers"][sel]))
    if len(iters) > 1:                               # earlier iterations -> convergence trails
        _attach_relion_trajectories(filaments, iters, tomo_name)
    return tomo_name, d["n"], filaments


# --- dispatch ------------------------------------------------------------------
def load_dataset(source: str, fmt: str, tomo, twist: float, rise: float,
                 pixelsize: float, write_temp: bool = True) -> Dataset:
    """Load a Dynamo folder or a RELION .star into a Dataset and fit it.

    fmt       : "dynamo" or "relion".
    tomo      : tomogram id to keep (int for Dynamo, TomoName str for RELION);
                None -> the only/first tomogram.
    twist     : deg / subunit.   rise : Angstrom / subunit.   pixelsize : A/px.
    write_temp: Dynamo only -- write temp.tbl (the working rows) into the folder.
    """
    if fmt == "relion":
        tomo_id, n_seg, fils = _build_relion(source, tomo, write_temp)
    elif fmt == "dynamo":
        tomo_id, n_seg, fils = _build_dynamo(source, tomo, write_temp)
    else:
        raise ValueError(f"unknown format {fmt!r}")
    ds = Dataset(source=source, fmt=fmt, tomo=tomo_id, twist=twist, rise=rise,
                 pixelsize=pixelsize, n_segments=n_seg, filaments=fils)
    ds.recompute()
    return ds
