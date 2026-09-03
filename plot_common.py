#!/usr/bin/env python3
"""
Shared pyqtgraph helpers for Rohlex.

author: Wen-Lu Chung
"""

from __future__ import annotations

import functools

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from helix_geom import roll_from_eulers, axis_tilt, oriented_axis, TILT_CONE


def _flip_overrides(fil, store):
    """(indices, angles) of this filament's committed flips as arrays, or (None, None).

    Returning them batched lets the effective_* helpers rebuild ALL flipped poses in
    one vectorised rotation instead of one scipy Rotation per flipped segment -- these
    run on every restyle (and on every step of the iteration slider), so a per-segment
    loop over a heavily flipped filament was the dominant redraw cost.
    """
    if not (fil.fittable and fil.axis is not None and store.flip_count()):
        return None, None
    idx, ang = [], []
    for i, t in enumerate(fil.tags):
        a = store.get_flip(int(t))
        if a is not None:
            idx.append(i)
            ang.append(a)
    if not idx:
        return None, None
    return np.asarray(idx, int), np.asarray(ang, float)


def effective_phi(fil, store, use_flips: bool = True) -> np.ndarray:
    """Roll per segment with committed flips applied: a flipped tag reads the roll of
    its stored (flipped) angles, everything else its original roll. Shared by the
    overview and detail plots so both show flips the same way.

    use_flips=False returns the raw roll. The stored flip angles are derived from the
    FINAL poses, so they are meaningless against an earlier iteration's poses -- the
    iteration views pass False rather than mixing the two.
    """
    phi = fil.phi.astype(float).copy()
    if not use_flips:
        return phi
    idx, ang = _flip_overrides(fil, store)
    if idx is not None:
        phi[idx] = roll_from_eulers(ang, fil.axis)
    return phi


def _oriented_axis(fil) -> np.ndarray:
    """The filament axis flipped toward its majority pose direction -- a FIXED
    reference, so tilts measured from different pose sets (committed flips, earlier
    iterations) stay comparable and a flip visibly moves a dot dark<->amber.

    Prefers the value pinned at load from the final poses (Filament.axis_oriented);
    falls back to computing it for filament-likes that lack the field."""
    n = getattr(fil, "axis_oriented", None)
    if n is not None:
        return np.asarray(n, float)
    return oriented_axis(fil.eulers, fil.axis)


def effective_eulers(fil, store, use_flips: bool = True) -> np.ndarray:
    """(N, 3) raw Dynamo angles with committed flips applied -- a flipped tag reads its
    stored (flipped) triple, everything else its original. The pose-level counterpart of
    effective_phi/effective_tilt, for views that need the whole orientation rather than
    one derived angle (the 3D pointing arrows, the export path)."""
    eul = np.asarray(fil.eulers, float).copy()
    if not use_flips:
        return eul
    idx, ang = _flip_overrides(fil, store)
    if idx is not None:
        eul[idx] = ang
    return eul


def effective_tilt(fil, store, use_flips: bool = True) -> np.ndarray:
    """Tilt-to-axis angle per segment (deg, [0,180]) with committed flips applied: a
    flipped tag reads the tilt of its stored (flipped) pose. Measured against the fixed
    majority-oriented axis, so flipping a dark segment shows up as it turning amber.
    use_flips=False gives the raw tilt (see effective_phi)."""
    n = _oriented_axis(fil)
    tilt = axis_tilt(fil.eulers, n, orient=False)
    if not use_flips:
        return tilt
    idx, ang = _flip_overrides(fil, store)
    if idx is not None:
        tilt[idx] = axis_tilt(ang, n, orient=False)
    return tilt


# --- unwrapped roll -------------------------------------------------------------
# The roll panels plot the roll UNWRAPPED head-to-tail instead of wrapped into
# (-180, 180]. Wrapping is what made a fast screw unreadable: at twist 166 / rise 27
# the model turns once every 58.6 A, and an overview panel spanning 1237 A in 190 px
# gives 9 px per turn -- the dashed model came out as ~21 near-vertical strokes and
# nothing could be judged against it. Unwrapped, the same filament is one straight
# climb of ~3500 deg and the model is a single straight line at any twist.
#
# unwrap_roll() is MODEL-FREE on purpose: it consults the roll only, never twist,
# rise or pixel size. So the points do not move when those are retuned -- only the
# model line does. An unwrap that snapped each point to the nearest branch OF THE
# MODEL was tried and rejected: it draws a perfect straight line for ANY twist you
# type (verified on the amyloid set, a deliberately wrong 166/26 model scored 105 deg
# against 98 for the correct one), so it can never show that the model is wrong.
#
# Known limitation: a step to the branch nearest zero loses a whole turn whenever the
# alignment advances two subunits between neighbouring picks (the true +332 reads as
# -28). On the actin set that pulls the apparent twist from 166 down to ~110 on the
# filaments with plateaus. The picture stays honest -- it is the alignment's own
# jumps that are being drawn -- but do not read a precise twist off the slope.

def unwrap_roll(phi: np.ndarray) -> np.ndarray:
    """Roll unwrapped head-to-tail: each step taken to the branch nearest zero.

    Model-free (see the note above). NaN entries -- segments missing at an earlier
    iteration -- keep their NaN and are bridged, so a gap does not desynchronise
    everything after it.
    """
    phi = np.asarray(phi, float)
    out = np.full(phi.shape, np.nan)
    ok = np.isfinite(phi)
    if not ok.any():
        return out
    p = phi[ok]
    out[ok] = p[0] + np.r_[0.0, np.cumsum(((np.diff(p) + 180.0) % 360.0) - 180.0)]
    return out


def on_branch(y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """`y` moved by whole turns onto the turn nearest `ref`, elementwise.

    Lets a derived roll -- a flip's new angle, an earlier iteration's pose, the ghost
    of an old one -- be drawn in the same unwrapped frame as the main curve without
    re-unwrapping it (which would let one changed segment shift everything after it).
    """
    y = np.asarray(y, float)
    ref = np.asarray(ref, float)
    shift = np.where(np.isfinite(ref) & np.isfinite(y),
                     360.0 * np.round((ref - y) / 360.0), 0.0)
    return y + shift


def unwrapped_phase(u: np.ndarray, pos: np.ndarray, rate: float, phi0: float) -> float:
    """`phi0` lifted by whole turns so the straight model line sits on the unwrapped
    data. Changes the model by a multiple of 360 only, so the residual, the tilt
    colours and the auto-exclude are untouched -- this moves the drawn line, nothing
    else. Returns phi0 unchanged when there is nothing to anchor to.
    """
    u = np.asarray(u, float)
    ok = np.isfinite(u)
    if not np.isfinite(phi0) or not ok.any():
        return phi0
    off = np.median(u[ok] - rate * np.asarray(pos, float)[ok] - phi0)
    return float(phi0 + 360.0 * np.round(off / 360.0))


# A viridis-like colormap defined explicitly so we don't depend on matplotlib
# (pyqtgraph's get('viridis') needs matplotlib/colorcet installed).
_VIRIDIS = pg.ColorMap(
    [0.0, 0.25, 0.5, 0.75, 1.0],
    [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
)
MARK_COLOR = (220, 30, 30)          # red: marked for removal
HILITE_PEN = pg.mkPen((255, 140, 0), width=2.5)   # orange ring: linked hover

# Shared toolbar-button look. Mixing stylesheet-styled and native-default buttons in
# one row makes them render at different heights (ragged toolbar); giving every button
# the same padding + border + radius keeps the row aligned. Colored buttons layer their
# own background on this SAME geometry (see helpers below) so heights still match.
_BTN_GEOM = "padding: 3px 9px; border: 1px solid {bd}; border-radius: 3px;"
PLAIN_BTN_QSS = (
    "QPushButton { " + _BTN_GEOM.format(bd="#9aa3b0") + " background: #f3f4f6; } "
    "QPushButton:hover { background: #e6ebf2; } "
    "QPushButton:disabled { color: rgba(0,0,0,90); border-color: #cccccc; }")


def color_button_qss(rgb, checked_rgb=None) -> str:
    """Toolbar-button stylesheet tinted `rgb`, sharing PLAIN_BTN_QSS geometry so a
    colored button lines up with the plain ones. `checked_rgb` styles the :checked
    state of a checkable button (and leaves it neutral when unchecked)."""
    if checked_rgb is None:
        r, g, b = rgb
        qss = ("QPushButton { " + _BTN_GEOM.format(bd=f"rgb({r},{g},{b})")
               + f" background: rgb({r},{g},{b}); color: white; }} "
               f"QPushButton:disabled {{ background: rgba({r},{g},{b},70); "
               "color: rgba(255,255,255,150); border-color: #cccccc; }")
    else:
        cr, cg, cb = checked_rgb
        qss = ("QPushButton { " + _BTN_GEOM.format(bd="#9aa3b0") + " background: #f3f4f6; } "
               "QPushButton:hover { background: #e6ebf2; } "
               f"QPushButton:checked {{ background: rgb({cr},{cg},{cb}); color: white; "
               f"border-color: rgb({cr},{cg},{cb}); }}")
    return qss


@functools.lru_cache(maxsize=4096)
def shared_brush(rgb) -> "pg.mkBrush":
    """A CACHED QBrush per colour — never build one per point.

    pyqtgraph's symbol atlas keys rendered symbols on brush/pen object IDENTITY
    (SymbolAtlas._keys stamps an _id on each instance), so a fresh mkBrush per point
    makes every point a unique symbol: it re-rasterises the whole scatter on every
    redraw and grows the atlas without bound. Handing back the SAME object for the
    same colour collapses a panel to its handful of real symbols. Measured on a 2982-
    segment tomogram: ~134k renderSymbol calls per 45 redraws -> a few dozen, and the
    iteration slider from ~133 ms/step to ~4 ms/step.

    Callers must not mutate the returned brush.
    """
    return pg.mkBrush(*rgb)


def viridis_rgba(values: np.ndarray) -> np.ndarray:
    """(N, 4) float RGBA: viridis across the min..max of `values` (e.g. position)."""
    v = np.asarray(values, float)
    rng = np.ptp(v)
    norm = (v - v.min()) / rng if rng > 0 else np.zeros_like(v)
    return np.array([_VIRIDIS.map(float(x), mode="float") for x in norm])


def pos_brushes(pos: np.ndarray, marked_mask: np.ndarray):
    """One QBrush per point: viridis by position, or red where marked.

    The ramp is quantised to 256 steps so the brushes come from the shared cache --
    a continuous ramp would otherwise give every point its own symbol in pyqtgraph's
    atlas (see shared_brush). 256 steps is finer than the eye resolves on a dot."""
    p = np.asarray(pos, float)
    rng = np.ptp(p)
    norm = (p - p.min()) / rng if rng > 0 else np.zeros_like(p)
    out = []
    for v, m in zip(norm, marked_mask):
        if m:
            out.append(shared_brush(MARK_COLOR))
        else:
            c = _VIRIDIS.map(round(float(v) * 255) / 255.0, mode="byte")
            out.append(shared_brush(tuple(int(x) for x in c)))
    return out


# --- register ("tilt") coloring for the roll panels --------------------------
# The XY maps can't show how far a segment's roll sits from the main helix
# register, so the roll / residual plots encode it as color instead. Three
# DEFINED bins (not a spectrum), keyed to |residual to the main model line|
# folded into [0, 180]:
#   on the main register (<= REG_ZONE)      -> dark  : the "good" angle, recedes
#   ~180 off / flipped   (>= 180 - REG_ZONE) -> amber : strongest contrast, flip these
#   anything between                         -> grey  : "doesn't matter" angles
# Marked-for-removal still overrides to red. REG_ZONE is the on/off-register
# tolerance (deg): a segment within it of the main line reads as on-register, within
# it of 180 reads as flipped.
REG_ZONE = 30.0
_REG_MAIN = (40, 44, 58)            # near-black slate: on the main register
_REG_FLIP = (240, 168, 0)           # amber: ~180 off -> the strongest-contrast "flip me" dots
_REG_OFF = (162, 167, 176)          # muted grey: off-register but not 180 -> doesn't matter


def register_brushes(resid: np.ndarray, marked_mask):
    """One QBrush per point, colored by |residual to the main model line| (deg): dark on
    the main register, amber at ~180 (flipped), muted grey between; red where marked."""
    d = np.abs(((np.asarray(resid, float) + 180.0) % 360.0) - 180.0)   # fold to [0, 180]
    brushes = []
    for di, m in zip(d, marked_mask):
        if m:
            c = MARK_COLOR
        elif di <= REG_ZONE:
            c = _REG_MAIN
        elif di >= 180.0 - REG_ZONE:
            c = _REG_FLIP
        else:
            c = _REG_OFF
        brushes.append(shared_brush(c))
    return brushes


# The geometric tilt angle (pose z-axis vs coordinate-fit filament axis) is shown as
# exactly THREE discrete colors -- one per on-screen category, matching the view
# checkboxes -- never a continuous ramp:
#   dark  (<= TILT_ZONE of 0)          axis-aligned  (majority / "good" polarity)
#   amber (>= 180 - TILT_ZONE)         antiparallel  (the tilt-flipped register)
#   grey  (in between)                 off-axis      (bad tilt: neither)
# Marked-for-removal overrides to red -> 4 colors total (red is the selection).
CAT_DARK, CAT_BAD, CAT_AMBER = 0, 1, 2
CAT_NAMES = ("dark", "bad-tilt", "amber")
_CAT_RGB = (_REG_MAIN, _REG_OFF, _REG_FLIP)          # indexed by CAT_*

# A segment "fits the tilt axis" (dark or amber) when its tilt is within TILT_ZONE of
# 0 or 180; otherwise it is off-axis grey. Drives "exclude bad tilt" and gates flip-all.
# Same cone the majority-polarity vote counts within (helix_geom.TILT_CONE), so the
# dark/amber membership and the majority orientation stay consistent.
TILT_ZONE = TILT_CONE


def tilt_off_axis(tilt_deg: np.ndarray) -> np.ndarray:
    """Boolean mask: True where the tilt is off-axis (grey) -- neither aligned (~0) nor
    antiparallel (~180) within TILT_ZONE. `min(tilt, 180-tilt)` is the distance to the
    nearer pole, so it's independent of the arbitrary axis-sign orientation."""
    t = np.asarray(tilt_deg, float)
    return np.minimum(t, 180.0 - t) > TILT_ZONE


def tilt_category(tilt_deg: np.ndarray) -> np.ndarray:
    """(N,) int CAT_* per segment: CAT_DARK (tilt <= TILT_ZONE), CAT_AMBER (tilt >=
    180 - TILT_ZONE), else CAT_BAD (off-axis grey). Pass the oriented/effective tilt."""
    t = np.asarray(tilt_deg, float)
    cat = np.full(t.shape, CAT_BAD, int)
    cat[t <= TILT_ZONE] = CAT_DARK
    cat[t >= 180.0 - TILT_ZONE] = CAT_AMBER
    return cat


def tilt_counts(tilt_deg: np.ndarray):
    """(n_dark, n_bad, n_amber): how many segments fall in each tilt category."""
    cat = tilt_category(tilt_deg)
    return (int((cat == CAT_DARK).sum()), int((cat == CAT_BAD).sum()),
            int((cat == CAT_AMBER).sum()))


def category_visible_mask(tilt_deg: np.ndarray, show) -> np.ndarray:
    """(N,) bool: keep a segment iff its category's checkbox is on.
    show = (dark_on, bad_on, amber_on), ordered by CAT_*."""
    cat = tilt_category(tilt_deg)
    if cat.size == 0:
        return np.zeros(0, bool)
    s = (bool(show[0]), bool(show[1]), bool(show[2]))
    return np.array([s[c] for c in cat], bool)


def tilt_brushes(tilt_deg: np.ndarray, marked_mask):
    """One QBrush per point: exactly 3 DISCRETE tilt colors -- dark (axis-aligned),
    grey (off-axis bad tilt), amber (antiparallel) -- red where marked. No ramp."""
    cat = tilt_category(tilt_deg)
    return [shared_brush(MARK_COLOR) if m else shared_brush(_CAT_RGB[c])
            for c, m in zip(cat, marked_mask)]


class SelectableViewBox(pg.ViewBox):
    """ViewBox where rubber-band selection is always active (no mode toggle):

      * left-drag  emits `regionSelected(QRectF)`   -> mark the enclosed points,
      * right-drag emits `regionDeselected(QRectF)` -> unmark the enclosed points.

    Both rects are in data coordinates and emitted on release. Drag never pans or
    zooms; zooming is the scroll wheel only (the default ViewBox wheelEvent).
    """
    regionSelected = QtCore.pyqtSignal(object)
    regionDeselected = QtCore.pyqtSignal(object)

    def mouseDragEvent(self, ev, axis=None):
        btn = ev.button()
        left = btn == QtCore.Qt.MouseButton.LeftButton
        right = btn == QtCore.Qt.MouseButton.RightButton
        if left or right:
            ev.accept()
            self.updateScaleBox(ev.buttonDownPos(), ev.pos())   # reuse the zoom rubber band
            if ev.isFinish():
                self.rbScaleBox.hide()
                p0 = self.mapToView(ev.buttonDownPos())
                p1 = self.mapToView(ev.pos())
                rect = QtCore.QRectF(p0, p1).normalized()
                (self.regionSelected if left else self.regionDeselected).emit(rect)
        else:
            super().mouseDragEvent(ev, axis=axis)   # middle-drag etc. keep default


class ModelParams(QtCore.QObject):
    """Live helix parameters, shared by every window.

    Thin Qt wrapper around the Dataset: the values live on the Dataset (so the
    fit machinery reads them straight off it), and `changed` fires after a
    set+recompute so all open views redraw their model overlays in lockstep.
    """
    changed = QtCore.pyqtSignal()

    def __init__(self, ds):
        super().__init__()
        self.ds = ds

    @property
    def twist(self) -> float:
        return self.ds.twist

    @property
    def rise(self) -> float:
        return self.ds.rise

    @property
    def pixelsize(self) -> float:
        return self.ds.pixelsize

    @property
    def model_rate(self) -> float:
        return self.ds.model_rate

    def update(self, twist=None, rise=None, pixelsize=None) -> None:
        self.ds.set_params(
            self.ds.twist if twist is None else twist,
            self.ds.rise if rise is None else rise,
            self.ds.pixelsize if pixelsize is None else pixelsize)
        self.changed.emit()


class ParamBar(QtWidgets.QWidget):
    """twist / rise / pixel-size spin boxes bound to a ModelParams.

    Editing a box retunes the shared params (which refits + signals); the boxes
    also re-sync from `params.changed` so the two windows' bars stay identical.
    """

    def __init__(self, params: ModelParams, parent=None):
        super().__init__(parent)
        self.params = params
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.sp_twist = self._spin(-180.0, 180.0, 3, 0.1, " °/sub")
        self.sp_rise = self._spin(0.001, 10000.0, 3, 0.1, " Å/sub")
        self.sp_px = self._spin(0.001, 1000.0, 4, 0.01, " Å/px")
        for label, sp in (("twist", self.sp_twist), ("rise", self.sp_rise),
                          ("pixel", self.sp_px)):
            lay.addWidget(QtWidgets.QLabel(label))
            lay.addWidget(sp)
        self._sync()
        for sp in (self.sp_twist, self.sp_rise, self.sp_px):
            sp.valueChanged.connect(self._emit)
        params.changed.connect(self._sync)

    @staticmethod
    def _spin(lo, hi, decimals, step, suffix):
        sp = QtWidgets.QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(decimals)
        sp.setSingleStep(step)
        sp.setSuffix(suffix)
        sp.setKeyboardTracking(False)   # fire on enter/focus-out, not per keystroke
        return sp

    def _emit(self, *_):
        self.params.update(twist=self.sp_twist.value(), rise=self.sp_rise.value(),
                           pixelsize=self.sp_px.value())

    def _sync(self):
        for sp, val in ((self.sp_twist, self.params.twist),
                        (self.sp_rise, self.params.rise),
                        (self.sp_px, self.params.pixelsize)):
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)
