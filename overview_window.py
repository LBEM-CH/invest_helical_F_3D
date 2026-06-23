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
from helix_geom import (model_line, register_flip_rotation, flipped_eulers,
                        rot_flip_eulers)
from plot_common import ModelParams, ParamBar, effective_phi, pos_brushes
from selection_store import SelectionStore

_PANEL_W, _PANEL_H = 190, 150
_DASH = pg.mkPen("k", width=1.6, style=QtCore.Qt.PenStyle.DashLine)
_EXCL_ZONE = 20.0          # deg: mark for removal when |roll - black| exceeds this


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
            self.scatter.setData(x=self.fil.pos, y=effective_phi(self.fil, store),
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
        top = QtWidgets.QHBoxLayout()
        top.addWidget(ParamBar(params))
        # tilt-flip / rot-flip are independent ON/OFF switches: checked applies that flip
        # to every segment in its register zone, unchecked removes it. Separate because
        # the 180° rot is only valid where the fibril has C2; tilt (polarity) is general.
        self.btn_tiltall = QtWidgets.QPushButton("tilt-flip all")
        self.btn_tiltall.setCheckable(True)
        self.btn_tiltall.setToolTip("Switch: tilt-flip (polarity) every pink/both-register "
                                    "segment onto black. Click again to remove all tilt flips.")
        self.btn_tiltall.setStyleSheet(
            "QPushButton { padding: 3px 9px; } "
            "QPushButton:checked { background-color: rgb(235,64,170); color: white; }")
        self.btn_tiltall.toggled.connect(lambda on: self._toggle_all("tilt", on))
        self.btn_rotall = QtWidgets.QPushButton("rot-flip all")
        self.btn_rotall.setCheckable(True)
        self.btn_rotall.setToolTip("Switch: rot-flip (180° about axis) every blue/both-register "
                                   "segment onto black. Use only where the fibril has C2. "
                                   "Click again to remove all rot flips.")
        self.btn_rotall.setStyleSheet(
            "QPushButton { padding: 3px 9px; } "
            "QPushButton:checked { background-color: rgb(25,55,200); color: white; }")
        self.btn_rotall.toggled.connect(lambda on: self._toggle_all("rot", on))
        self.btn_autoexcl = QtWidgets.QPushButton("auto-exclude")
        self.btn_autoexcl.setToolTip("Mark for removal every segment whose (flipped) roll "
                                     "is more than ±20° off its black line. Run AFTER you've "
                                     "flipped — only the black line matters now.")
        self.btn_autoexcl.setStyleSheet(
            "QPushButton { background-color: rgb(200,40,40); color: white; "
            "border-radius: 3px; padding: 3px 9px; }")
        self.btn_autoexcl.clicked.connect(self._auto_exclude)
        self.btn_clearsel = QtWidgets.QPushButton("clear selection")
        self.btn_clearsel.setToolTip("Clear all removal marks (unselect every segment).")
        self.btn_clearsel.clicked.connect(lambda *_: self.store.clear())
        top.addWidget(self.btn_tiltall)
        top.addWidget(self.btn_rotall)
        top.addWidget(self.btn_autoexcl)
        top.addWidget(self.btn_clearsel)
        top.addStretch(1)
        rlay.addLayout(top)
        self.status = QtWidgets.QLabel("hover a filament panel…")
        rlay.addWidget(self.status)
        map_glw = pg.GraphicsLayoutWidget()
        rlay.addWidget(map_glw, 1)
        self.map = map_glw.addPlot(title=f"tomo {ds.tomo} — XY view")
        self.map.setLabel("bottom", "X (px)")
        self.map.setLabel("left", "Y (px)")
        mvb = self.map.getViewBox()
        mvb.setAspectLocked(True)
        mvb.setMouseEnabled(True, True)                # left-drag pans, wheel zooms
        mvb.setMouseMode(pg.ViewBox.PanMode)
        self.map.setMenuEnabled(True)                  # right-click menu (incl. "View All")
        self._draw_map()
        self.map.autoRange()                           # fit all filaments on open
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

    def _toggle_all(self, which, on, zone=20.0):
        """ON/OFF switch for one flip type, with priority black > blue(rot) > pink(tilt):
        a black-zone segment is never flipped; a blue-zone segment is a rot target; a
        pink-zone segment (and NOT also in the blue zone) is a tilt target. The both /
        purple register is NOT auto-flipped here -- only manually in the detail window.
        Each toggle owns a single-bit state for its own disjoint zone."""
        rate = self.ds.model_rate
        nset = 0
        set_map, clear = {}, []
        for f in self.ds.filaments:
            if not (f.fittable and np.isfinite(f.phi0)):
                continue
            rb = ((f.phi - (rate * f.pos + f.phi0) + 180) % 360) - 180
            in_black = np.abs(rb) <= zone
            in_blue = np.abs(((rb - 180 + 180) % 360) - 180) <= zone
            if which == "rot":
                target = in_blue & ~in_black
                if not target.any():
                    continue
                pose, bits = rot_flip_eulers(f.eulers, f.axis), (0, 1)
            else:                                          # tilt
                if not np.isfinite(f.phi0_flip):
                    continue
                rp = ((f.phi - (rate * f.pos + f.phi0_flip) + 180) % 360) - 180
                target = (np.abs(rp) <= zone) & ~in_black & ~in_blue   # priority: blue beats pink
                if not target.any():
                    continue
                S = register_flip_rotation(f.eulers, f.pos, f.axis, f.flipped, rate,
                                           f.phi0, f.phi0_flip)
                if S is None:
                    continue
                pose = flipped_eulers(f.eulers, f.pos, f.axis, rate, S, np.ones(f.n, bool))
                bits = (1, 0)
            for i in np.where(target)[0]:
                t = int(f.tags[i])
                if on:
                    set_map[t] = (bits[0], bits[1], tuple(pose[i]))
                    nset += 1
                else:
                    clear.append(t)
        self.store.replace_flips(set_map, clear)
        self.status.setText(f"{which}-flip all {'ON' if on else 'OFF'}: "
                            f"{nset if on else 0} {which}-zone segments "
                            f"{'flipped' if on else 'cleared'}")

    def _auto_exclude(self, *_):
        """Mark for removal every segment whose effective (flipped) roll is more than
        _EXCL_ZONE deg off its black line. Meant to run after flipping: by then only
        the black register is "good", so anything still off it is bad. (The *_ swallows
        the bool that QPushButton.clicked emits, so it can't become the threshold.)"""
        zone = _EXCL_ZONE
        rate = self.ds.model_rate
        bad, nfil = [], 0
        for f in self.ds.filaments:
            if not (f.fittable and np.isfinite(f.phi0)):
                continue                                   # no black line to test against
            resid = ((effective_phi(f, self.store) - (rate * f.pos + f.phi0) + 180) % 360) - 180
            bad.extend(int(t) for t in f.tags[np.abs(resid) > zone])
            nfil += 1
        self.store.add(bad)
        self.status.setText(
            f"auto-exclude: marked {len(bad)} segments >±{int(zone)}° off black "
            f"across {nfil} filaments  ({self.store.count()} total marked)")

    def _restyle_all(self):
        for p in self.panels:
            p.restyle(self.store)
        self.setWindowTitle(
            f"invest_helical_F_3D — {self.ds.fmt} tomo {self.ds.tomo} — "
            f"{len(self.ds.filaments)} filaments — {self.store.count()} marked")
