#!/usr/bin/env python3
"""
Dataset loading for Rohlex (Dynamo .tbl or RELION .star).

author: Wen-Lu Chung

Both inputs are funnelled into the same Dataset of Filaments so the GUI never
cares which format it came from:

  Dynamo : concatenate refined_table_ref_001 + ref_002, sort by tag (col 1),
           filter to one tomogram (col 20), group into filaments (col 23).
           .tbl columns (0-based): tag=0, shift=3:6, ZXZ euler=6:9, tomo=19,
           fil=22, xyz=23:26. The position used is xyz + shift (see COL_SHIFT).
  RELION : read one tomogram's particles from a RELION 5 star, group by
           _rlnHelicalTubeID; relion_star converts the poses to the SAME Dynamo
           ZXZ-extrinsic convention before they reach helix_geom.

The pose-only part of the fit (position along the axis, measured roll) is done
once at load. The model overlay (phi0, residual delta) and the Angstrom scaling
of position both depend on twist / rise / pixel-size, which the GUI can retune
live -- Dataset.set_params() recomputes just that cheap part across filaments.
"""

from __future__ import annotations

import copy
import glob
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np

from helix_geom import (fit_pose, roll_from_eulers, dominant_phase,
                        polarity_dyad_angle, dynamo_rotation, oriented_axis)

# A real tilt flip is a head<->tail dyad: the two polarity groups' mean z-axes are
# ~antiparallel (angle near 180). Below this the split is spurious (z ~perp to the
# axis, polarity sign = noise) and gets no flipped register. See polarity_dyad_angle.
MIN_DYAD_DEG = 150.0

# Dynamo .tbl column indices (0-based).
COL_TAG = 0
COL_SHIFT = slice(3, 6)
COL_EULER = slice(6, 9)
COL_TOMO = 19
COL_FIL = 22
COL_XYZ = slice(23, 26)

# A Dynamo particle sits at cols 24-26 (the pick) PLUS cols 4-6 (the refined shift,
# same tomogram frame, signed) -- that is what dtcrop re-extracts at and what any
# tbl->star converter writes out. Both parts are needed here, because the pose is
# refined TOGETHER with the shift: the roll a segment ends up with belongs to the
# SHIFTED position, so measuring it against the raw pick decouples roll from
# position and destroys the screw. On the actin test set the shifts average 3.9 px
# (62 A, ~2.3 rises) and reach 12 px, i.e. >360 deg of roll at 166 deg / 27 A --
# ignoring them turned twist 166 into noise (roll concentration 0.31 -> 0.51 and
# the recovered twist -2.7 -> +160 once they are added). Adding them also leaves
# the filament straighter than the picks alone (perpendicular RMS 5.1 vs 5.3 A;
# SUBTRACTING them gives 13.2 A, which is how the sign was pinned).
#
# This is the opposite of the RELION path, which deliberately drops
# _rlnOrigin*Angst -- see the note in relion_star.load_particles. Those shifts are
# in the SUBTOMOGRAM frame and are not tomogram coordinates; Dynamo's are.


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
    polarity: np.ndarray = field(default_factory=lambda: np.array([]))  # (N,) +1/-1 sign(z.axis)
    flipped: np.ndarray = field(default_factory=lambda: np.array([], bool))  # (N,) minority polarity
    pos: np.ndarray = field(default_factory=lambda: np.array([]))   # (N,) position (Angstrom)
    delta: np.ndarray = field(default_factory=lambda: np.array([])) # (N,) residual to model (deg)
    phi0: float = float("nan")  # model phase of the majority register (deg)
    phi0_flip: float = float("nan")  # phase of the flipped (minority-polarity) register (deg)
    axis: np.ndarray = None     # (3,) unit filament axis (fittable only); for iteration roll
    axis_oriented: np.ndarray = None  # (3,) axis pinned to the majority pose direction
    traj_roll: np.ndarray = None  # (n_iter, N) roll per Dynamo iteration, ordered like tags; or None
    traj_iters: list = None     # iteration numbers matching traj_roll rows; or None
    traj_eulers: np.ndarray = None  # (n_iter, N, 3) pose per iteration, ordered like tags; or None
    traj_pos_px: np.ndarray = None  # (n_iter, N) position along the fixed axis per iteration; or None

    @property
    def n(self) -> int:
        return len(self.tags)

    def at_iteration(self, k: int, rate: float, pixelsize: float) -> "Filament":
        """A view of this filament with the pose AND position it had at trajectory row `k`.

        The pick never moves, but Dynamo refines a SHIFT alongside the angles and the
        position that matters here is pick + shift (see COL_SHIFT), so a segment does
        travel along its filament between iterations -- 2 px on average over the first
        iteration of the actin set, i.e. ~200 deg of roll at 166 deg / 27 A. Both the
        pose-derived arrays and pos_px are therefore recomputed; everything else (tags,
        the fitted axis, the head->tail ordering) is shared unchanged, which is what
        makes stepping through iterations cheap. Positions are projected onto the FINAL
        axis and centroid so the trail is comparable row to row. Returns a shallow copy,
        so the caller can hand it to any plot/fit helper that takes a Filament; `self`
        is returned untouched when no trajectory is attached.

        Rows for a tag missing at that iteration are NaN (see _attach_trajectories);
        NaN flows through scipy's Rotation without raising, so such segments simply
        read as NaN roll / off-axis tilt rather than breaking the view.
        """
        if self.traj_eulers is None or not self.fittable:
            return self
        eul = np.asarray(self.traj_eulers[k], float)
        f = copy.copy(self)
        f.eulers = eul
        if self.traj_pos_px is not None:
            f.pos_px = np.asarray(self.traj_pos_px[k], float)
        f.phi = roll_from_eulers(eul, self.axis)
        z = dynamo_rotation(eul).as_matrix()[:, :, 2]
        pol = np.sign(z @ np.asarray(self.axis, float))
        pol[pol == 0] = 1.0
        f.polarity = pol
        # axis_oriented stays the FINAL-pose orientation (copied): the dark/amber
        # split must mean the same thing at every iteration to be comparable.
        f.apply_model(rate, pixelsize)
        return f

    def apply_model(self, rate: float, pixelsize: float) -> None:
        """Refresh the model-dependent arrays for a new rate / pixel-size.

        The main screw phase (phi0, the black line) is the center of the DENSEST
        residual cluster -- a robust "majority" that outliers or a rot/tilt split
        can't drag off, rather than a plain mean over a count-based majority. The
        dominant polarity of that cluster defines the correct polarity; segments
        of the opposite polarity are the tilt-flip (pink) register and get their
        own robust phase (phi0_flip). The rot-flip register is just phi0 + 180.
        """
        self.pos = self.pos_px * pixelsize
        if not self.fittable:
            self.phi0 = self.phi0_flip = float("nan")
            self.flipped = np.zeros(self.n, bool)
            self.delta = np.full(self.n, np.nan)
            return
        self.phi0, main_in = dominant_phase(self.pos, self.phi, rate)
        # correct polarity = the dominant polarity within the main cluster; the
        # opposite-polarity segments are the tilt-flipped ones.
        pol = self.polarity if self.polarity.size == self.n else np.ones(self.n)
        main_pol = 1.0 if pol[main_in].sum() >= 0 else -1.0
        self.flipped = pol != main_pol
        self.delta = ((self.phi - (rate * self.pos + self.phi0) + 180) % 360) - 180
        # Reject a spurious split: only a real head<->tail dyad (mean z-axes
        # ~antiparallel) counts as a tilt flip; a noise split (z ~perp to the axis)
        # has the two groups far from antiparallel -> no flipped register.
        if self.flipped.sum() >= 3 and polarity_dyad_angle(self.eulers, self.flipped) < MIN_DYAD_DEG:
            self.flipped = np.zeros(self.n, bool)
        self.phi0_flip = (dominant_phase(self.pos[self.flipped], self.phi[self.flipped], rate)[0]
                          if self.flipped.sum() >= 3 else float("nan"))



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
    def iter_labels(self) -> list:
        """Iteration numbers of the attached trajectory rows (oldest->newest), or []
        when the dataset was loaded without --iteration-paths. The LAST row is the
        working set (what every non-iteration view shows)."""
        for f in self.filaments:
            if f.traj_iters:
                return list(f.traj_iters)
        return []

    @property
    def n_iterations(self) -> int:
        return len(self.iter_labels)

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
    # `flipped` (which polarity is the tilt-flip side) is decided in apply_model,
    # where it can lean on the robust main-register fit; here we just carry polarity.
    return Filament(fid=int(fid), tags=tags[o], pos_px=fp["pos"], phi=fp["phi"],
                    xy=xyz[o, :2], xyz=xyz[o], eulers=eulers[o], fittable=True,
                    axis=fp["axis"], polarity=fp["polarity"],
                    axis_oriented=oriented_axis(eulers[o], fp["axis"]))


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


def _euler_map(tables, tomo_id) -> tuple[dict, dict]:
    """(tag -> ZXZ euler triple, tag -> xyz+shift), from .tbl files of one tomogram."""
    par = np.concatenate(
        [np.loadtxt(t, comments="#", dtype=str, ndmin=2) for t in tables], axis=0)
    par = par[par[:, COL_TOMO].astype(float).astype(int) == int(tomo_id)]
    tags = par[:, COL_TAG].astype(float).astype(int).tolist()
    xyz = par[:, COL_XYZ].astype(float) + par[:, COL_SHIFT].astype(float)
    return dict(zip(tags, par[:, COL_EULER].astype(float))), dict(zip(tags, xyz))


def _fill_trajectories(filaments, labels, maps, xyz_maps=None) -> None:
    """Attach the per-iteration poses (traj_eulers), rolls (traj_roll) and positions
    (traj_pos_px) to every fittable filament. Shared by the Dynamo and RELION paths,
    which differ only in how they build `maps` -- one {tag: euler triple} per
    iteration, oldest->newest.

    Rolls are measured about the FIXED final axis, and `xyz_maps` (optional, one
    {tag: xyz} per iteration) is projected onto that same axis and centroid, so both
    halves of the trail are read in one frame and are comparable row to row. Refits
    per iteration would let the axis wobble and mix that wobble into the trail; the
    pick-to-pick scatter about the axis is only ~5 A, so a fixed axis costs nothing.
    Without `xyz_maps` (RELION) the position is held at its final value.
    Segments missing from an iteration stay NaN in every array.
    """
    nan3 = np.full(3, np.nan)
    for fil in filaments:
        if not fil.fittable:
            continue
        eulers = np.full((len(maps), fil.n, 3), np.nan)
        traj = np.full((len(maps), fil.n), np.nan)
        pos = np.full((len(maps), fil.n), np.nan)
        centroid, axis = fil.xyz.mean(0), np.asarray(fil.axis, float)
        for ii, emap in enumerate(maps):
            eul = np.array([emap.get(int(t), nan3) for t in fil.tags])   # (N, 3)
            eulers[ii] = eul
            valid = np.isfinite(eul[:, 0])
            if valid.any():
                traj[ii, valid] = roll_from_eulers(eul[valid], fil.axis)
            if xyz_maps is not None:
                xyz = np.array([xyz_maps[ii].get(int(t), nan3) for t in fil.tags])
                pos[ii] = (xyz - centroid) @ axis
        # The last row IS the working set every other view shows, so pin it to the
        # loaded poses rather than to the last iteration file -- the two can differ
        # (RELION reads run_data.star, whose row is labelled with the last run_it).
        eulers[-1] = fil.eulers
        traj[-1] = fil.phi
        pos[-1] = fil.pos_px
        fil.traj_eulers = eulers
        fil.traj_roll = traj
        fil.traj_pos_px = pos if xyz_maps is not None else None
        fil.traj_iters = labels


def _attach_trajectories(filaments, folders, tomo_id) -> None:
    """Dynamo: per-iteration poses from iteration 0 (the starting_values that seeded
    iteration 1) and every refined iteration. Segments are matched by tag."""
    labels, maps, xyz_maps = [], [], []

    def add(label, tables):
        emap, xmap = _euler_map(tables, tomo_id)
        labels.append(label)
        maps.append(emap)
        xyz_maps.append(xmap)

    # iteration 0: starting_values sit next to the FIRST iteration's averages
    start_dir = os.path.join(os.path.dirname(folders[0][1]), "starting_values")
    start_tbls = [t for t in sorted(glob.glob(
        os.path.join(start_dir, "starting_table_ref_*_ite_*.tbl")))
        if os.path.getsize(t) > 0]
    if start_tbls:
        add(0, start_tbls)
    for it, d in folders:
        add(it, find_ref_tables(d))
    _fill_trajectories(filaments, labels, maps, xyz_maps)


def _build_dynamo(path: str, tomo, write_temp: bool, trails: bool = False):
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
            rows[:, COL_XYZ].astype(float) + rows[:, COL_SHIFT].astype(float),
            rows[:, COL_EULER].astype(float)))
    if trails and len(folders) > 1:              # earlier iterations -> convergence trails
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
    """RELION analogue of _attach_trajectories: poses at every refinement iteration
    (run_it000 = the start), matched by _rlnTomoParticleId. Each iteration star is
    read through load_particles, so its poses reach the Dynamo convention exactly
    the way the final set does."""
    from relion_star import load_particles
    labels, maps = [], []
    for label, star in iter_stars:
        _, d = load_particles(star, tomo_name)
        labels.append(label)
        maps.append(dict(zip(d["pid"].astype(int).tolist(), d["eulers"])))
    _fill_trajectories(filaments, labels, maps)


def _build_relion(path: str, tomo, write_temp: bool = True, trails: bool = False):
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
    if trails and len(iters) > 1:                    # earlier iterations -> convergence trails
        _attach_relion_trajectories(filaments, iters, tomo_name)
    return tomo_name, d["n"], filaments


# --- deferred trajectory loading ------------------------------------------------
def _iteration_sources(ds):
    """The per-iteration inputs `ds.source` holds, WITHOUT reading any of them.

    Just a glob, so the GUI can ask "is there an iteration history here?" for free and
    defer the (multi-second) read until the user actually asks for it.
    """
    try:
        if ds.fmt == "relion":
            from relion_star import iteration_stars
            return iteration_stars(ds.source)
        return find_iteration_folders(ds.source)
    except (OSError, ValueError):
        return []


def can_attach_trajectories(ds) -> bool:
    """True if `ds.source` offers more than one iteration to step through."""
    return len(_iteration_sources(ds)) > 1


def attach_trajectories(ds) -> int:
    """Read every iteration of `ds.source` and attach the per-iteration poses/rolls.

    Exactly what load_dataset(trails=True) does, factored out so it can be triggered
    later from the GUI -- the read is far too slow to do on every startup, but making
    it a launch-only flag hid the feature from anyone who did not know to pass it.
    Returns the number of iterations attached (0 if there is no history).
    """
    srcs = _iteration_sources(ds)
    if len(srcs) < 2:
        return 0
    if ds.fmt == "relion":
        _attach_relion_trajectories(ds.filaments, srcs, ds.tomo)
    else:
        _attach_trajectories(ds.filaments, srcs, ds.tomo)
    return ds.n_iterations


# --- dispatch ------------------------------------------------------------------
def load_dataset(source: str, fmt: str, tomo, twist: float, rise: float,
                 pixelsize: float, write_temp: bool = True,
                 trails: bool = False) -> Dataset:
    """Load a Dynamo folder or a RELION .star into a Dataset and fit it.

    fmt       : "dynamo" or "relion".
    tomo      : tomogram id to keep (int for Dynamo, TomoName str for RELION);
                None -> the only/first tomogram.
    twist     : deg / subunit.   rise : Angstrom / subunit.   pixelsize : A/px.
    write_temp: Dynamo only -- write temp.tbl (the working rows) into the folder.
    trails    : attach per-iteration convergence trails (off by default; the
                earlier-iteration scan is skipped unless requested).
    """
    if fmt == "relion":
        tomo_id, n_seg, fils = _build_relion(source, tomo, write_temp, trails)
    elif fmt == "dynamo":
        tomo_id, n_seg, fils = _build_dynamo(source, tomo, write_temp, trails)
    else:
        raise ValueError(f"unknown format {fmt!r}")
    ds = Dataset(source=source, fmt=fmt, tomo=tomo_id, twist=twist, rise=rise,
                 pixelsize=pixelsize, n_segments=n_seg, filaments=fils)
    ds.recompute()
    return ds
