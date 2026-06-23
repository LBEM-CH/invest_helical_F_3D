#!/usr/bin/env python3
"""
Shared pyqtgraph helpers for invest_helical_F_3D.

author: Wen-Lu Chung
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from helix_geom import roll_from_eulers


def effective_phi(fil, store) -> np.ndarray:
    """Roll per segment with committed flips applied: a flipped tag reads the roll of
    its stored (flipped) angles, everything else its original roll. Shared by the
    overview and detail plots so both show flips the same way."""
    phi = fil.phi.astype(float).copy()
    if fil.fittable and fil.axis is not None and store.flip_count():
        for i, t in enumerate(fil.tags):
            ang = store.get_flip(int(t))
            if ang is not None:
                phi[i] = roll_from_eulers(np.asarray(ang, float)[None, :], fil.axis)[0]
    return phi

# A viridis-like colormap defined explicitly so we don't depend on matplotlib
# (pyqtgraph's get('viridis') needs matplotlib/colorcet installed).
_VIRIDIS = pg.ColorMap(
    [0.0, 0.25, 0.5, 0.75, 1.0],
    [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
)
MARK_COLOR = (220, 30, 30)          # red: marked for removal
HILITE_PEN = pg.mkPen((255, 140, 0), width=2.5)   # orange ring: linked hover


def viridis_rgba(values: np.ndarray) -> np.ndarray:
    """(N, 4) float RGBA: viridis across the min..max of `values` (e.g. position)."""
    v = np.asarray(values, float)
    rng = np.ptp(v)
    norm = (v - v.min()) / rng if rng > 0 else np.zeros_like(v)
    return np.array([_VIRIDIS.map(float(x), mode="float") for x in norm])


def pos_brushes(pos: np.ndarray, marked_mask: np.ndarray):
    """One QBrush per point: viridis by position, or red where marked."""
    p = np.asarray(pos, float)
    rng = np.ptp(p)
    norm = (p - p.min()) / rng if rng > 0 else np.zeros_like(p)
    brushes = []
    for v, m in zip(norm, marked_mask):
        brushes.append(pg.mkBrush(MARK_COLOR) if m else pg.mkBrush(_VIRIDIS.map(float(v), mode="qcolor")))
    return brushes


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
