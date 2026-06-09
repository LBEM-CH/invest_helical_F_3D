#!/usr/bin/env python3
"""
Shared pyqtgraph helpers for invest_helical_F_3D.

author: Wen-Lu Chung
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore

# A viridis-like colormap defined explicitly so we don't depend on matplotlib
# (pyqtgraph's get('viridis') needs matplotlib/colorcet installed).
_VIRIDIS = pg.ColorMap(
    [0.0, 0.25, 0.5, 0.75, 1.0],
    [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)],
)
MARK_COLOR = (220, 30, 30)          # red: marked for removal
HILITE_PEN = pg.mkPen((255, 140, 0), width=2.5)   # orange ring: linked hover


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
