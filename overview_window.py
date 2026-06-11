#!/usr/bin/env python3
"""
Overview window for invest_helical_F_3D.

author: Wen-Lu Chung

Left: a scrollable grid of per-filament panels (roll vs real position + screw
model), one per filament in the tomogram. Right: the XY projection map of the
whole tomogram, plus the live twist / rise / pixel-size controls. Hovering a
panel highlights that filament on the map; clicking a panel opens its detail
window. Panels whose filament has marked segments turn red so triage progress is
visible at a glance. Retuning the parameters refits every panel instantly.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from dynamo_table import Dataset
from detail_window import DetailWindow
from helix_geom import model_line
from plot_common import ModelParams, ParamBar, pos_brushes
from selection_store import SelectionStore

_PANEL_W, _PANEL_H = 190, 150
_DASH = pg.mkPen("k", width=1.6, style=QtCore.Qt.PenStyle.DashLine)


class _MiniPanel:
    """Small roll-vs-position panel for one filament in the overview grid."""

    def __init__(self, glw, row, col, fil):
        self.fil = fil
        self.plot = glw.addPlot(row=row, col=col)
        self.plot.setMouseEnabled(False, False)        # panels are for glance/click, not zoom
        self.plot.hideButtons()
        self.plot.setMenuEnabled(False)                # no right-click context menu
        self.plot.setYRange(-180, 180, padding=0)
        self.plot.getAxis("bottom").setStyle(showValues=False)
        self.plot.getAxis("left").setStyle(showValues=False)
        self.scatter = pg.ScatterPlotItem(size=6, pen=pg.mkPen(None))
        self.plot.addItem(self.scatter)
        self.model_item = self.plot.plot([], [], connect="finite", pen=_DASH)
        self.vb = self.plot.getViewBox()

    def redraw_model(self, rate, halfspan):
        """Rescale x to the shared span and redraw the dashed screw (rate-dependent)."""
        self.plot.setXRange(-halfspan, halfspan, padding=0)
        if self.fil.fittable and np.isfinite(self.fil.phi0):
            xx, model = model_line(halfspan, self.fil.phi0, rate)
            self.model_item.setData(xx, model)
        else:
            self.model_item.setData([], [])

    def restyle(self, store):
        marked = np.array([store.is_marked(t) for t in self.fil.tags], dtype=bool)
        if self.fil.fittable and np.isfinite(self.fil.phi0):
            self.scatter.setData(x=self.fil.pos, y=self.fil.phi,
                                 brush=pos_brushes(self.fil.pos, marked))
        else:
            self.scatter.setData([])
        color = "#dc1e1e" if marked.any() else "#222222"
        self.plot.setTitle(f"fil {self.fil.fid} (n={self.fil.n})", color=color, size="8pt")


class OverviewWindow(QtWidgets.QMainWindow):

    def __init__(self, ds: Dataset, store: SelectionStore, params: ModelParams,
                 cols: int = 5, map_volume=None, map_voxel=None, gl_enabled: bool = True):
        super().__init__()
        self.ds = ds
        self.store = store
        self.params = params
        self.map_volume = map_volume
        self.map_voxel = map_voxel
        self.gl_enabled = gl_enabled
        self.halfspan = ds.pos_halfspan
        self.detail = None
        self.setWindowTitle(
            f"invest_helical_F_3D — {ds.fmt} tomo {ds.tomo} — {len(ds.filaments)} filaments")
        self.resize(1500, 850)

        splitter = QtWidgets.QSplitter()
        self.setCentralWidget(splitter)

        # --- left: scrollable grid of filament panels ------------------------
        self.glw = pg.GraphicsLayoutWidget()
        nrows = int(np.ceil(len(ds.filaments) / cols))
        self.glw.setFixedSize(cols * _PANEL_W, nrows * _PANEL_H)
        self.panels: list[_MiniPanel] = []
        for k, fil in enumerate(ds.filaments):
            self.panels.append(_MiniPanel(self.glw, k // cols, k % cols, fil))
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.glw)
        scroll.setWidgetResizable(False)
        splitter.addWidget(scroll)

        # --- right: controls + tomogram XY map -------------------------------
        right = QtWidgets.QWidget()
        rlay = QtWidgets.QVBoxLayout(right)
        rlay.addWidget(ParamBar(params))
        self.status = QtWidgets.QLabel("hover a filament panel…")
        rlay.addWidget(self.status)
        map_glw = pg.GraphicsLayoutWidget()
        rlay.addWidget(map_glw, 1)
        self.map = map_glw.addPlot(title=f"tomo {ds.tomo} — XY view")
        self.map.setLabel("bottom", "X (px)")
        self.map.setLabel("left", "Y (px)")
        self.map.getViewBox().setAspectLocked(True)
        self.map.setMenuEnabled(False)                 # no right-click context menu
        self._draw_map()
        self.map_hl = pg.ScatterPlotItem(size=12, pen=pg.mkPen((255, 140, 0), width=2),
                                         brush=pg.mkBrush(None))
        self.map.addItem(self.map_hl, ignoreBounds=True)   # don't let hover shift the map view
        splitter.addWidget(right)
        splitter.setSizes([950, 550])

        # --- wiring ----------------------------------------------------------
        self.glw.scene().sigMouseMoved.connect(self._on_move)
        self.glw.scene().sigMouseClicked.connect(self._on_click)
        self.store.changed.connect(self._restyle_all)
        self.params.changed.connect(self._on_params)
        self._on_params()                              # initial model draw + ranges

    # --- map -----------------------------------------------------------------
    def _draw_map(self):
        cmap = [pg.intColor(i, hues=max(9, len(self.ds.filaments))) for i in
                range(len(self.ds.filaments))]
        for i, fil in enumerate(self.ds.filaments):
            xy = fil.xy
            self.map.plot(xy[:, 0], xy[:, 1], pen=pg.mkPen(cmap[i], width=1))
            self.map.addItem(pg.ScatterPlotItem(x=xy[:, 0], y=xy[:, 1], size=5,
                                                brush=pg.mkBrush(cmap[i]), pen=pg.mkPen(None)))
            txt = pg.TextItem(str(fil.fid), color="k", anchor=(0, 1))
            txt.setPos(float(xy[0, 0]), float(xy[0, 1]))
            self.map.addItem(txt)

    def _panel_at(self, scene_pos):
        for p in self.panels:
            if p.vb.sceneBoundingRect().contains(scene_pos):
                return p
        return None

    # --- interaction ---------------------------------------------------------
    def _on_move(self, scene_pos):
        p = self._panel_at(scene_pos)
        if p is None:
            self.map_hl.setData([])
            return
        self.map_hl.setData(x=p.fil.xy[:, 0], y=p.fil.xy[:, 1])
        nmark = sum(self.store.is_marked(t) for t in p.fil.tags)
        self.status.setText(f"filament {p.fil.fid}  (n={p.fil.n}, marked={nmark})  "
                            f"— click to open")

    def _on_click(self, ev):
        p = self._panel_at(ev.scenePos())
        if p is None:
            return
        self.detail = DetailWindow(p.fil, self.params, self.store,
                                   map_volume=self.map_volume, map_voxel=self.map_voxel,
                                   gl_enabled=self.gl_enabled, parent=self)
        self.detail.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.detail.show()

    def _on_params(self):
        """Parameters changed: rescale to the new Angstrom span and refit overlays."""
        self.halfspan = self.ds.pos_halfspan
        for p in self.panels:
            p.redraw_model(self.ds.model_rate, self.halfspan)
            p.restyle(self.store)

    def _restyle_all(self):
        for p in self.panels:
            p.restyle(self.store)
        self.setWindowTitle(
            f"invest_helical_F_3D — {self.ds.fmt} tomo {self.ds.tomo} — "
            f"{len(self.ds.filaments)} filaments — {self.store.count()} marked")
