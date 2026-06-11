#!/usr/bin/env python3
"""
RELION 5 (tomography) STAR reading for invest_helical_F_3D.

author: Wen-Lu Chung

A RELION subtomogram particle file stores poses differently from Dynamo, so we
convert them back to the Dynamo ZXZ-extrinsic convention and feed the *same*
helix_geom.fit_pose() the Dynamo path uses. That conversion is the exact inverse
of the Dynamo -> RELION pipeline in subtomo_averaging/msa_human.ipynb:

  forward (how the .star was made):
    1. dynamo (ZXZ extrinsic) --convert_eulers('dynamo','warp')--> warp (zyz
       intrinsic) = the aligned/real pose  r_aligned
    2. store with a fixed +90 deg prior removed:
         r_subtomo = r_aligned * r_prior.inv(),   r_prior = R_zyz(prior angles)
       -> _rlnTomoSubtomogramRot/Tilt/Psi   (prior kept in _rlnAngle*Prior)

  inverse (what we do here):
    1. r_aligned = r_subtomo * r_prior            (undo the +90 deg prior)
    2. dynamo = convert_eulers(r_aligned 'zyz', 'warp', 'dynamo')

The per-particle pose lives in _rlnTomoSubtomogram*. The alignment is taken from
_rlnAngleRot/Tilt/Psi when present, else from _rlnAngleRotPrior/TiltPrior/PsiPrior
(the assumed pose) -- which in these files is the fixed (0, 90, 0).

We read:
  _rlnTomoName            tomogram id (string)
  _rlnTomoParticleId      per-tomogram particle id -> the remove-list tag
  _rlnCoordinateX/Y/Z     particle coordinates (px)
  _rlnHelicalTubeID       filament grouping
  _rlnTomoSubtomogram*    per-particle pose
  _rlnAngle*[Prior]       alignment / assumed pose
and pull the pixel size from the optics block (_rlnImagePixelSize).
"""

from __future__ import annotations

import numpy as np


def read_star(path: str) -> dict:
    """Parse every `loop_` table in a STAR file.

    Returns {block_name: {"cols": [name, ...], "rows": [[token, ...], ...]}},
    e.g. block_name == "data_particles". Column names keep the leading rln but
    drop the leading underscore ("rlnTomoName"). Non-loop blocks are ignored.
    """
    with open(path) as fh:
        lines = fh.readlines()

    blocks: dict[str, dict] = {}
    i, n = 0, len(lines)
    while i < n:
        s = lines[i].strip()
        if not s.startswith("data_"):
            i += 1
            continue
        block_name = s
        j = i + 1
        while j < n and not lines[j].strip():
            j += 1
        if j >= n or lines[j].strip() != "loop_":
            i = j                                  # non-loop block (key/value); skip
            continue
        # header lines: "_rlnName #k" (the index is advisory; we use order)
        cols: list[str] = []
        k = j + 1
        while k < n and lines[k].lstrip().startswith("_"):
            cols.append(lines[k].split()[0].lstrip("_"))
            k += 1
        # data rows until a blank line or the next block
        rows: list[list[str]] = []
        while k < n:
            t = lines[k].strip()
            if not t or t.startswith("data_") or t.startswith("loop_"):
                break
            rows.append(t.split())
            k += 1
        blocks[block_name] = {"cols": cols, "rows": rows}
        i = k
    return blocks


class _Table:
    """Thin column-name accessor over a parsed loop block."""

    def __init__(self, block: dict):
        self.cols = block["cols"]
        self.rows = block["rows"]
        self.idx = {name: k for k, name in enumerate(self.cols)}

    def has(self, name: str) -> bool:
        return name in self.idx

    def col(self, name: str, dtype=float) -> np.ndarray:
        k = self.idx[name]
        return np.array([r[k] for r in self.rows], dtype=dtype)

    def first(self, name: str, dtype=float):
        return dtype(self.rows[0][self.idx[name]])


def tomogram_names(path: str) -> list[str]:
    """Distinct _rlnTomoName values, for the startup chooser."""
    t = _Table(read_star(path)["data_particles"])
    return sorted(set(t.col("rlnTomoName", dtype=str).tolist()))


def image_pixel_size(path: str) -> float | None:
    """_rlnImagePixelSize from the optics block (A/px), or None if absent."""
    blocks = read_star(path)
    if "data_optics" not in blocks:
        return None
    opt = _Table(blocks["data_optics"])
    if opt.has("rlnImagePixelSize"):
        return float(opt.first("rlnImagePixelSize"))
    if opt.has("rlnTomoTiltSeriesPixelSize"):
        return float(opt.first("rlnTomoTiltSeriesPixelSize"))
    return None


def _alignment_eulers(t: _Table) -> np.ndarray:
    """(N,3) zyz alignment angles: the refined _rlnAngle* if present, else the
    assumed _rlnAngle*Prior (RELION's preassumed pose)."""
    if t.has("rlnAngleRot") and t.has("rlnAngleTilt") and t.has("rlnAnglePsi"):
        names = ("rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi")
    else:
        names = ("rlnAngleRotPrior", "rlnAngleTiltPrior", "rlnAnglePsiPrior")
    return np.column_stack([t.col(nm) for nm in names])


def relion_to_dynamo_eulers(subtomo_zyz: np.ndarray,
                            align_zyz: np.ndarray) -> np.ndarray:
    """Inverse of the notebook's Dynamo->RELION pose pipeline.

    subtomo_zyz : (N,3) _rlnTomoSubtomogramRot/Tilt/Psi  (zyz intrinsic, deg)
    align_zyz   : (N,3) _rlnAngle*[Prior]                (zyz intrinsic, deg)
    Returns (N,3) Dynamo ZXZ-extrinsic Euler angles (deg).
    """
    import warnings

    from scipy.spatial.transform import Rotation as R
    from eulerangles import convert_eulers

    r_subtomo = R.from_euler("zyz", subtomo_zyz, degrees=True)
    r_prior = R.from_euler("zyz", align_zyz, degrees=True)
    r_aligned = r_subtomo * r_prior                      # undo step 8b (+90 prior)
    # Tilt near 90 deg trips scipy's gimbal-lock warning, but as_euler still
    # returns a triple for the SAME rotation, so convert_eulers recovers the
    # correct Dynamo matrix (verified: the known helix rise reappears).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warp = r_aligned.as_euler("zyz", degrees=True)   # rlnAngle* equivalent
    return convert_eulers(warp, source_meta="warp", target_meta="dynamo")


def load_particles(path: str, tomo_name: str | None):
    """Read one tomogram's particles.

    Returns (tomo_name, dict) where dict has:
      tube   : (N,) filament id (rlnHelicalTubeID, or all-1 if absent)
      pid    : (N,) rlnTomoParticleId  -> remove-list tag
      xyz    : (N,3) rlnCoordinateX/Y/Z (px)
      eulers : (N,3) Dynamo ZXZ-extrinsic angles (deg)
      n      : int particle count
    """
    t = _Table(read_star(path)["data_particles"])
    names = t.col("rlnTomoName", dtype=str)
    if tomo_name is None:
        tomo_name = sorted(set(names.tolist()))[0]
    elif tomo_name not in set(names.tolist()):
        raise ValueError(f"tomogram {tomo_name!r} not in star "
                         f"(have {sorted(set(names.tolist()))})")
    sel = names == tomo_name

    pid = t.col("rlnTomoParticleId", dtype=float).astype(int)[sel]
    xyz = np.column_stack([t.col("rlnCoordinateX"), t.col("rlnCoordinateY"),
                           t.col("rlnCoordinateZ")])[sel]
    subtomo = np.column_stack([t.col("rlnTomoSubtomogramRot"),
                               t.col("rlnTomoSubtomogramTilt"),
                               t.col("rlnTomoSubtomogramPsi")])[sel]
    align = _alignment_eulers(t)[sel]
    if t.has("rlnHelicalTubeID"):
        tube = t.col("rlnHelicalTubeID", dtype=float).astype(int)[sel]
    else:
        tube = np.ones(int(sel.sum()), dtype=int)

    eulers = relion_to_dynamo_eulers(subtomo, align)
    return tomo_name, dict(tube=tube, pid=pid, xyz=xyz, eulers=eulers,
                           n=int(sel.sum()))
