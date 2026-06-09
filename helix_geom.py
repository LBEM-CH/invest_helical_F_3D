#!/usr/bin/env python3
"""
Helical-filament geometry for invest_helical_F_3D.

author: Wen-Lu Chung

These functions are lifted verbatim (in spirit) from the per-filament "helical
roll check" worked out in subtomo_averaging/msa_human.ipynb, so the app and the
notebook stay numerically identical.

Per filament we:
  * project the 3D segment coordinates onto the SVD-fitted axis -> real signed
    position along the filament (px), with the middle (centroid) at 0,
  * carry the reference x-axis through each particle's pose and read the azimuth
    (roll) about the filament axis,
  * compare that roll to the known screw  roll = RATE*pos + phi0, where
    RATE = TWIST / RISE (deg per px).

Coordinates are taken from the 3D xyz (NOT the segment sequence index): a segment
whose alignment drifted lands at its true (wrong) position and leaves the model.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

# Known helix of the human filament (deg, px per subunit). Defaults; the CLI can
# override them, so nothing downstream should hard-code these values.
TWIST = -1.4
RISE = 0.6


def rate(twist: float = TWIST, rise: float = RISE) -> float:
    """Model slope: deg of azimuth per px along the axis."""
    return twist / rise


def axis_and_pos(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the filament axis through the middle and project to a 1D position.

    xyz : (N, 3) tomogram coordinates.
    Returns (n, pos): n = unit axis (head<->tail, sign arbitrary), pos = signed
    arc-length along it (px) with 0 at the centroid (the middle of the filament).
    """
    c = xyz.mean(0)                       # middle of the filament
    _, _, vt = np.linalg.svd(xyz - c)
    n = vt[0]                             # principal axis = head<->tail direction
    pos = (xyz - c) @ n                   # real signed position along axis (px)
    return n, pos


def roll_about_axis(D: Rot, n: np.ndarray) -> np.ndarray:
    """Azimuth (roll) of each pose about the filament axis n, in (-180, 180].

    Builds an orthonormal frame (e1, e2) perpendicular to n, carries the
    reference x-axis [1,0,0] through each particle's rotation D, and reads its
    angle in that frame.
    """
    e1 = np.cross(n, [0, 0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(n, [0, 1.0, 0])
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    v = D.apply(np.tile([1.0, 0, 0], (len(D), 1)))   # reference x carried into tomo frame
    return np.degrees(np.arctan2(v @ e2, v @ e1))


def fit_filament(xyz: np.ndarray, eulers: np.ndarray,
                 model_rate: float) -> dict:
    """Full per-filament fit.

    xyz     : (N, 3) coordinates.
    eulers  : (N, 3) ZXZ Euler angles (deg) for each segment.
    model_rate : deg azimuth per px (RATE = twist/rise).

    Returns a dict with everything the views need, all ordered head->tail by the
    real position:
      order : (N,) indices that sort the input into head->tail order
      pos   : (N,) real position along axis (px), centered at the middle
      phi   : (N,) measured roll about the axis (deg)
      phi0  : float circular-mean model phase (deg)
      delta : (N,) circular (phi - model) in (-180, 180]  (residual to the screw)
      axis  : (3,) unit filament axis
    """
    n, pos = axis_and_pos(xyz)
    order = np.argsort(pos)
    pos = pos[order]
    D = Rot.from_euler('ZXZ', eulers[order], degrees=True)
    phi = roll_about_axis(D, n)
    phi0 = np.degrees(np.angle(np.exp(1j * np.radians(phi - model_rate * pos)).mean()))
    model = model_rate * pos + phi0
    delta = ((phi - model + 180) % 360) - 180
    return dict(order=order, pos=pos, phi=phi, phi0=float(phi0), delta=delta, axis=n)


def model_line(pos_span: float, phi0: float, model_rate: float,
               npts: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Dashed model curve across [-pos_span, pos_span], wrapped to (-180,180].

    Returns (xx, model) with NaNs inserted at the +/-180 wraps so a line plot
    does not draw vertical strokes across the panel.
    """
    xx = np.linspace(-pos_span, pos_span, npts)
    model = ((model_rate * xx + phi0 + 180) % 360) - 180
    model[np.r_[False, np.abs(np.diff(model)) > 180]] = np.nan
    return xx, model
