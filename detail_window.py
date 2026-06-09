#!/usr/bin/env python3
"""
Per-filament detail window for invest_helical_F_3D.

author: Wen-Lu Chung

Three linked plots for one filament:
  plot1: roll vs real position + dashed screw model
  plot2: residual (delta) to the model vs position + zero line
  plot3: the filament's XY map (segments in order)

Pointing at a segment in any plot highlights the SAME segment in all three.
Marking for removal: rubber-band drag (Select mode), the Select-all button, or
clicking a single point toggles it. Marks restyle live across every plot and the
overview, and are persisted by the SelectionStore.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from dynamo_table import Filament
from helix_geom import model_line
from plot_common import HILITE_PEN, SelectableViewBox, pos_brushes
from selection_store import SelectionStore


class _Panel:
    """One scatter plot with marking + hover highlight, over arrays (x, y, tags)."""

    def __init__(self, glw: pg.GraphicsLayoutWidget, row: int, col: int,
                 title: str, xlabel: str, ylabel: str):
        self.vb = SelectableViewBox()
        self.plot = glw.addPlot(row=row, col=col, viewBox=self.vb, title=title)
        self.plot.setLabel("bottom", xlabel)
        self.plot.setLabel("left", ylabel)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setMenuEnabled(False)          # no right-click context menu (it's distracting)
        # hoverable for the linked-hover signal, but hoverSize=-1 (default) so the
        # dot itself does NOT grow -- the highlight ring is the only hover cue.
        self.scatter = pg.ScatterPlotItem(size=10, hoverable=True, pen=pg.mkPen(None))
        self.highlight = pg.ScatterPlotItem(size=18, pen=HILITE_PEN,
                                            brush=pg.mkBrush(None))
        self.plot.addItem(self.scatter)
        # ignoreBounds: the hover ring must not affect auto-range, else the axes
        # jump slightly every time it appears.
        self.plot.addItem(self.highlight, ignoreBounds=True)
        self.x = np.array([])
        self.y = np.array([])
        self.tags = np.array([])

    def set_data(self, x, y, tags):
        self.x, self.y, self.tags = np.asarray(x), np.asarray(y), np.asarray(tags)

    def restyle(self, store: SelectionStore):
        marked = np.array([store.is_marked(t) for t in self.tags], dtype=bool)
        brushes = pos_brushes(self.x, marked)
        spots = [dict(pos=(float(x), float(y)), data=int(t), brush=b,
                      size=(14 if m else 10))
                 for x, y, t, b, m in zip(self.x, self.y, self.tags, brushes, marked)]
        self.scatter.setData(spots=spots)

    def show_hover(self, idx):
        if idx is None or idx >= len(self.x):
            self.highlight.setData([])
        else:
            self.highlight.setData(x=[float(self.x[idx])], y=[float(self.y[idx])])

    def tags_in_rect(self, rect: QtCore.QRectF):
        xmin, xmax = rect.left(), rect.right()
        ymin, ymax = rect.top(), rect.bottom()
        sel = (self.x >= xmin) & (self.x <= xmax) & (self.y >= ymin) & (self.y <= ymax)
        return self.tags[sel].tolist()


class DetailWindow(QtWidgets.QMainWindow):

    def __init__(self, fil: Filament, model_rate: float, pos_halfspan: float,
                 store: SelectionStore, parent=None):
        super().__init__(parent)
        self.fil = fil
        self.store = store
        self.setWindowTitle(f"filament {fil.fid}  (n={fil.n})")
        self.resize(1300, 520)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)

        # --- toolbar ---------------------------------------------------------
        bar = QtWidgets.QHBoxLayout()
        self.btn_all = QtWidgets.QPushButton("Select all in filament")
        self.btn_all.clicked.connect(lambda: self.store.add(self.fil.tags.tolist()))
        self.btn_clear = QtWidgets.QPushButton("Clear filament")
        self.btn_clear.clicked.connect(lambda: self.store.remove(self.fil.tags.tolist()))
        self.btn_home = QtWidgets.QPushButton("Home (reset view)")
        self.btn_home.clicked.connect(self._reset_view)
        self.btn_back = QtWidgets.QPushButton("← Back to overview")
        self.btn_back.clicked.connect(self.close)
        self.readout = QtWidgets.QLabel("hover a segment…")
        self.readout.setMinimumWidth(360)
        for w in (self.btn_all, self.btn_clear, self.btn_home, self.btn_back):
            bar.addWidget(w)
        bar.addWidget(self.readout)
        bar.addStretch(1)
        outer.addLayout(bar)

        # --- plots -----------------------------------------------------------
        glw = pg.GraphicsLayoutWidget()
        outer.addWidget(glw, 1)
        self.p1 = _Panel(glw, 0, 0, f"fil {fil.fid}: roll vs position",
                         "position along axis (px)", "roll (deg)")
        self.p2 = _Panel(glw, 0, 1, "residual to model",
                         "position along axis (px)", "delta: data - model (deg)")
        self.p3 = _Panel(glw, 0, 2, "XY map", "X (px)", "Y (px)")
        self.panels = [self.p1, self.p2, self.p3]

        self.p1.set_data(fil.pos, fil.phi, fil.tags)
        self.p2.set_data(fil.pos, fil.delta, fil.tags)
        self.p3.set_data(fil.xy[:, 0], fil.xy[:, 1], fil.tags)
        self.p2.plot.addLine(y=0, pen=pg.mkPen("k", style=QtCore.Qt.PenStyle.DashLine))
        self.p3.vb.setAspectLocked(True)
        self.p3.plot.plot(fil.xy[:, 0], fil.xy[:, 1],
                          pen=pg.mkPen((150, 150, 150), width=1))   # connecting line

        # dashed screw model on plot1
        if fil.n >= 5 and np.isfinite(fil.phi0):
            xx, model = model_line(pos_halfspan, fil.phi0, model_rate)
            self.p1.plot.plot(xx, model, connect="finite",
                              pen=pg.mkPen("k", width=1.6, style=QtCore.Qt.PenStyle.DashLine))

        # --- wiring ----------------------------------------------------------
        for p in self.panels:
            p.scatter.sigHovered.connect(self._on_hover)
            p.scatter.sigClicked.connect(self._on_click)
            p.vb.regionSelected.connect(self._on_select)
            p.vb.regionDeselected.connect(self._on_deselect)
        self.store.changed.connect(self._restyle_all)
        self._restyle_all()
        self.statusBar().showMessage(
            "left-drag = mark   |   right-drag = unmark   |   scroll = zoom   |   click = toggle one")

    # --- interaction ---------------------------------------------------------
    def _reset_view(self):
        for p in self.panels:
            p.vb.autoRange()

    def _on_hover(self, scatter, points, ev=None):
        if len(points) == 0:
            return
        idx = points[0].index()
        for p in self.panels:
            p.show_hover(idx)
        t = int(self.fil.tags[idx])
        self.readout.setText(
            f"tag {t}   pos={self.fil.pos[idx]:+.1f}px   "
            f"roll={self.fil.phi[idx]:+.1f}°   delta={self.fil.delta[idx]:+.1f}°"
            f"   {'[MARKED]' if self.store.is_marked(t) else ''}")

    def _on_click(self, scatter, points, ev=None):
        if len(points):
            self.store.toggle(int(points[0].data()))

    def _panel_of(self, vb):
        for p in self.panels:
            if p.vb is vb:
                return p
        return None

    def _on_select(self, rect: QtCore.QRectF):
        p = self._panel_of(self.sender())          # select against the emitting panel's coords
        if p:
            tags = p.tags_in_rect(rect)
            if tags:
                self.store.add(tags)

    def _on_deselect(self, rect: QtCore.QRectF):
        p = self._panel_of(self.sender())
        if p:
            tags = p.tags_in_rect(rect)
            if tags:
                self.store.remove(tags)

    def _restyle_all(self):
        for p in self.panels:
            p.restyle(self.store)
