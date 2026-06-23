#!/usr/bin/env python3
"""
Helical-filament geometry for invest_helical_F_3D.

author: Wen-Lu Chung

These functions are lifted verbatim (in spirit) from the per-filament "helical
roll check" worked out in subtomo_averaging/msa_human.ipynb, so the app and the
notebook stay numerically identical.

Per filament we:
  * project the 3D segment coordinates onto the SVD-fitted axis -> real signed
    position along the filament, with the middle (centroid) at 0,
  * carry the reference x-axis through each particle's pose and read the azimuth
    (roll) about the filament axis,
  * compare that roll to the known screw  roll = RATE*pos + phi0, where
    RATE = TWIST / RISE (deg per Angstrom).

Units: positions are carried in Angstrom (raw pixel coordinates * pixel size),
so RATE = twist[deg/subunit] / rise[A/subunit] is deg-per-Angstrom and the whole
fit is at real physical scale regardless of binning.

The pose-only quantities (position along the axis, measured roll) do NOT depend
on twist/rise/pixel-size, so fit_pose() is computed once at load. Only the model
overlay (phi0 phase, residual delta) depends on the rate -- fit_model() is cheap
and re-run whenever the user retunes twist/rise/pixel-size in the GUI.

Coordinates are taken from the 3D xyz (NOT the segment sequence index): a segment
whose alignment drifted lands at its true (wrong) position and leaves the model.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

# Known helix of the human filament. TWIST is deg/subunit, RISE is Angstrom per
# subunit. Defaults; the CLI can override them, so nothing downstream should
# hard-code these values.
TWIST = -1.4
RISE = 4.75


def model_rate(twist: float = TWIST, rise: float = RISE) -> float:
    """Screw slope: deg of azimuth per Angstrom along the axis (twist / rise)."""
    return twist / rise


def axis_and_pos(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the filament axis through the middle and project to a 1D position.

    xyz : (N, 3) coordinates (any single length unit).
    Returns (n, pos): n = unit axis (head<->tail, sign arbitrary), pos = signed
    arc-length along it (same unit as xyz) with 0 at the centroid.
    """
    c = xyz.mean(0)                       # middle of the filament
    _, _, vt = np.linalg.svd(xyz - c)
    n = vt[0]                             # principal axis = head<->tail direction
    pos = (xyz - c) @ n                   # real signed position along axis
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


def fit_pose(xyz: np.ndarray, eulers: np.ndarray) -> dict:
    """Pose-only, rate-independent part of the per-filament fit.

    xyz    : (N, 3) coordinates (px); the caller scales pos to Angstrom.
    eulers : (N, 3) ZXZ-extrinsic Euler angles (deg) -- Dynamo convention. The
             Relion path converts its angles to this same convention first, so
             this function is identical for both inputs.

    Returns, all ordered head->tail by real position:
      order    : (N,) indices that sort the input into head->tail order
      pos      : (N,) real position along axis (same unit as xyz), centered
      phi      : (N,) measured roll about the axis (deg)
      axis     : (3,) unit filament axis
      polarity : (N,) sign of (particle z-axis . filament axis), +1/-1. Which way
                 each particle points along the axis; a flipped subset (opposite
                 sign) is a polarity (perpendicular-dyad) ambiguity, not a roll
                 difference -- see invest_helical's flipped-register overlay.
    """
    n, pos = axis_and_pos(xyz)
    order = np.argsort(pos)
    pos = pos[order]
    D = Rot.from_euler('ZXZ', eulers[order], degrees=True)
    phi = roll_about_axis(D, n)
    zaxis = D.as_matrix()[:, :, 2]              # each particle's z-axis in tomo frame
    polarity = np.sign(zaxis @ n)
    polarity[polarity == 0] = 1.0
    return dict(order=order, pos=pos, phi=phi, axis=n, polarity=polarity)


def roll_from_eulers(eulers: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Measured roll (deg) for ZXZ-extrinsic eulers about a fixed filament axis.

    Used to recompute each segment's roll at earlier Dynamo iterations (same axis,
    different pose) without redoing the SVD fit.
    """
    D = Rot.from_euler('ZXZ', np.asarray(eulers, float), degrees=True)
    return roll_about_axis(D, axis)


def fit_model(pos: np.ndarray, phi: np.ndarray, rate: float) -> dict:
    """Fit the screw phase to measured rolls, given the rate.

    pos  : (N,) position along the axis (same unit the rate is per).
    phi  : (N,) measured roll (deg).
    rate : screw slope (deg per unit of pos).

    Returns:
      phi0  : float circular-mean model phase (deg)
      delta : (N,) circular (phi - model) in (-180, 180]  (residual to the screw)
    """
    phi0 = np.degrees(np.angle(np.exp(1j * np.radians(phi - rate * pos)).mean()))
    model = rate * pos + phi0
    delta = ((phi - model + 180) % 360) - 180
    return dict(phi0=float(phi0), delta=delta)


def _densest_window(ang: np.ndarray, halfwidth: float) -> np.ndarray:
    """Boolean mask of the most-populated circular window (center = some sample,
    members within +/- halfwidth deg). A robust 'mode' for wrapped angles: a
    minority of outliers or a symmetry-split sub-cluster can't capture it."""
    ang = np.asarray(ang, float)
    if ang.size == 0:
        return np.zeros(0, bool)
    d = np.abs(((ang[:, None] - ang[None, :] + 180.0) % 360.0) - 180.0)
    within = d <= halfwidth
    return within[:, int(np.argmax(within.sum(0)))]


def dominant_phase(pos: np.ndarray, phi: np.ndarray, rate: float,
                   halfwidth: float = 25.0) -> tuple[float, np.ndarray]:
    """Robust screw phase: the center of the densest cluster of residuals
    (phi - rate*pos), so outliers / a +180 (rot) or tilt sub-population cannot
    drag the main line off (unlike a plain mean over a count-based 'majority').
    Returns (phi0_deg, inlier_mask)."""
    pos = np.asarray(pos, float)
    phi = np.asarray(phi, float)
    if phi.size == 0:
        return float("nan"), np.zeros(0, bool)
    r = (phi - rate * pos) % 360.0
    inliers = _densest_window(r, halfwidth)
    phi0 = np.degrees(np.angle(np.exp(1j * np.radians(r[inliers])).mean()))
    return float(phi0), inliers


def register_flip_rotation(eulers: np.ndarray, pos: np.ndarray, axis: np.ndarray,
                           flipped: np.ndarray, rate: float, phi0_main: float,
                           phi0_flip: float, halfwidth: float = 25.0):
    """The rotation S that maps the flipped (pink) register onto the main (black)
    register, measured from this filament's own data.

    Both registers are de-screwed (the position-dependent helical roll removed)
    and averaged, S = mean(main) * mean(flipped)^-1. Each group's mean is taken
    over the segments whose de-screwed roll is near the GIVEN line phase --
    phi0_main (black) for the main polarity, phi0_flip (pink) for the flipped --
    so S is anchored to the drawn lines. (Using each polarity's densest cluster
    instead would misfire when the main polarity has a second, off-register
    cluster: S would target that cluster, not black.) Empirically S is a ~180 deg
    rotation about an axis perpendicular to the filament axis (the polarity dyad).
    Returns a scipy Rotation, or None if a group is empty / phi0_flip is unset.
    """
    flipped = np.asarray(flipped, bool)
    if (~flipped).sum() < 1 or flipped.sum() < 1 or not np.isfinite(phi0_flip):
        return None
    D = Rot.from_euler('ZXZ', np.asarray(eulers, float), degrees=True)
    descrew = Rot.from_rotvec((-np.radians(rate * np.asarray(pos, float)))[:, None]
                              * np.asarray(axis, float)[None, :])
    Dt = descrew * D                                   # de-screwed orientations
    roll = roll_about_axis(Dt, axis)                   # de-screwed roll ~ (phi - rate*pos)

    def mean_near(group, center):
        gi = np.where(group)[0]
        d = np.abs(((roll[gi] - center + 180.0) % 360.0) - 180.0)
        keep = gi[d <= halfwidth]
        return Dt[keep if keep.size else gi].mean()    # fallback: whole group

    # The natural rotation between the two registers -- NOT snapped to 180/perp:
    # for some filaments the pink<->black relation is a rotation about an axis far
    # from perpendicular (even near-parallel to the filament axis), and forcing it
    # perpendicular breaks tilt-flip. Exact involution / sequence-closure is handled
    # by the caller tracking the discrete flip STATE (recomputing each pose from the
    # original), not by squaring this operator -- so S only has to map pink -> black.
    return mean_near(~flipped, phi0_main) * mean_near(flipped, phi0_flip).inv()


def flipped_eulers(eulers: np.ndarray, pos: np.ndarray, axis: np.ndarray,
                   rate: float, S, to_majority) -> np.ndarray:
    """Apply the register flip to segments -> new ZXZ-extrinsic eulers (deg).

    For each segment, S (or S^-1) is conjugated by the screw rotation at that
    segment's position, so the flipped segment lands on the OTHER register at its
    own position -- only the 3 angles change, the position does not.

      to_majority : per-segment bool. True  -> apply S    (minority -> majority),
                                       False -> apply S^-1 (majority -> minority).
    """
    eulers = np.atleast_2d(np.asarray(eulers, float))
    pos = np.atleast_1d(np.asarray(pos, float))
    to_majority = np.atleast_1d(np.asarray(to_majority, bool))
    D = Rot.from_euler('ZXZ', eulers, degrees=True)
    descrew = Rot.from_rotvec((-np.radians(rate * pos))[:, None]
                              * np.asarray(axis, float)[None, :])
    Sarr = Rot.concatenate([S if t else S.inv() for t in to_majority])
    Dnew = descrew.inv() * Sarr * descrew * D
    return Dnew.as_euler('ZXZ', degrees=True)


def rot_flip_eulers(eulers: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Apply the C2-about-axis ambiguity: rotate each pose 180 deg about the
    filament axis -> new ZXZ-extrinsic eulers (deg).

    Unlike the polarity (tilt) flip this keeps the particle z-axis pointing the
    same way; it only shifts the measured roll by exactly +180 deg. So the
    rot-flipped register is just the main screw line offset by 180 deg. Position
    is unchanged.
    """
    eulers = np.atleast_2d(np.asarray(eulers, float))
    D = Rot.from_euler('ZXZ', eulers, degrees=True)
    R = Rot.from_rotvec(np.pi * np.asarray(axis, float))   # 180 deg about the axis
    return (R * D).as_euler('ZXZ', degrees=True)


def model_line(pos_span: float, phi0: float, rate: float,
               npts: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Dashed model curve across [-pos_span, pos_span], wrapped to (-180,180].

    Returns (xx, model) with NaNs inserted at the +/-180 wraps so a line plot
    does not draw vertical strokes across the panel.
    """
    xx = np.linspace(-pos_span, pos_span, npts)
    model = ((rate * xx + phi0 + 180) % 360) - 180
    model[np.r_[False, np.abs(np.diff(model)) > 180]] = np.nan
    return xx, model
