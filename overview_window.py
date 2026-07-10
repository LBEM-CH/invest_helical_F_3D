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
                        rot_flip_eulers, axis_tilt)
from plot_common import (ModelParams, ParamBar, effective_phi, effective_tilt,
                        tilt_brushes, tilt_off_axis, TILT_ZONE, PLAIN_BTN_QSS,
                        color_button_qss)
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
        if self.fil.fittable and self.fil.axis is not None:
            phi = effective_phi(self.fil, store)
            tilt = effective_tilt(self.fil, store)   # flip-aware pose z-axis vs filament axis
            self.scatter.setData(x=self.fil.pos, y=phi,
                                 brush=tilt_brushes(tilt, marked))
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
        # Two control rows so the XY map below gets full width: row 1 = helix params,
        # row 2 = the flip / exclude actions. (One long row squeezed the map.)
        row1 = QtWidgets.QHBoxLayout()
        row1.addWidget(ParamBar(params))
        row1.addStretch(1)
        rlay.addLayout(row1)
        row2 = QtWidgets.QHBoxLayout()
        # tilt-flip / rot-flip are independent ON/OFF switches. They act ONLY on the
        # dark (axis-aligned) and amber (antiparallel) segments -- the ones that fit the
        # filament tilt axis; grey off-axis segments are skipped as irrelevant.
        self.btn_tiltall = QtWidgets.QPushButton("tilt-flip all")
        self.btn_tiltall.setCheckable(True)
        self.btn_tiltall.setToolTip("Switch: tilt-flip (polarity) every AMBER (antiparallel) "
                                    "segment onto the dark axis-aligned majority. Click again "
                                    "to remove all tilt flips. Grey off-axis segments ignored.")
        self.btn_tiltall.setStyleSheet(color_button_qss(None, checked_rgb=(235, 64, 170)))
        self.btn_tiltall.toggled.connect(lambda on: self._toggle_all("tilt", on))
        self.btn_rotall = QtWidgets.QPushButton("rot-flip all")
        self.btn_rotall.setCheckable(True)
        self.btn_rotall.setToolTip("Switch: rot-flip (180° about axis) every blue/both-register "
                                   "dark/amber segment onto black. Use only where the fibril has "
                                   "C2. Click again to remove all rot flips. Grey segments ignored.")
        self.btn_rotall.setStyleSheet(color_button_qss(None, checked_rgb=(25, 55, 200)))
        self.btn_rotall.toggled.connect(lambda on: self._toggle_all("rot", on))
        self.btn_badtilt = QtWidgets.QPushButton("exclude bad tilt")
        self.btn_badtilt.setToolTip("Mark for removal every segment whose pose is neither "
                                    "aligned nor antiparallel to its filament axis (off-axis "
                                    "grey tilt). Leaves each filament with only dark + amber.")
        self.btn_badtilt.setStyleSheet(color_button_qss((150, 75, 55)))
        self.btn_badtilt.clicked.connect(self._exclude_bad_tilt)
        self.btn_autoexcl = QtWidgets.QPushButton("auto-exclude")
        self.btn_autoexcl.setToolTip("Mark for removal every segment whose (flipped) roll "
                                     "is more than ±20° off its black line. Run AFTER you've "
                                     "flipped — only the black line matters now.")
        self.btn_autoexcl.setStyleSheet(color_button_qss((200, 40, 40)))
        self.btn_autoexcl.clicked.connect(self._auto_exclude)
        self.btn_clearsel = QtWidgets.QPushButton("clear selection")
        self.btn_clearsel.setToolTip("Clear all removal marks (unselect every segment).")
        self.btn_clearsel.setStyleSheet(PLAIN_BTN_QSS)
        self.btn_clearsel.clicked.connect(lambda *_: self.store.clear())
        self.btn_autofit = QtWidgets.QPushButton("auto-fit twist")
        self.btn_autofit.setToolTip("Estimate the helix twist that best fits every "
                                    "filament's roll at once (robust to outlier tubes), "
                                    "then apply it to the live twist / rise.")
        self.btn_autofit.setStyleSheet(color_button_qss((25, 110, 200)))
        self.btn_autofit.clicked.connect(self._open_autofit)
        for w in (self.btn_tiltall, self.btn_rotall, self.btn_badtilt,
                  self.btn_autoexcl, self.btn_autofit, self.btn_clearsel):
            row2.addWidget(w)
        row2.addStretch(1)
        rlay.addLayout(row2)
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
        self._refresh_title()                          # show marked count + stale flag now
        # If the resume flips were written under an older pose convention, their cached
        # angles can't be trusted -- warn once (after the window is up) so they get
        # regenerated rather than silently misleading the display / auto-exclude.
        if self.store.flip_count() and getattr(self.store, "flips_stale", False):
            QtCore.QTimer.singleShot(0, self._warn_stale_flips)

    def _warn_stale_flips(self):
        # NON-MODAL: a blocking QMessageBox.warning() runs its own nested event loop,
        # and over `ssh -XY` it can open behind / off-screen -> the app looks frozen
        # ("the loop is always running"). show() on a non-modal box never blocks; the
        # title bar (see _restyle_all) carries a persistent fallback indicator.
        mb = QtWidgets.QMessageBox(
            QtWidgets.QMessageBox.Icon.Warning,
            "Flipped list is from an older convention",
            f"{self.store.flip_count()} flips were loaded from flipped_list.txt, but the "
            "file predates the current pose convention, so its angles may place segments "
            "on the wrong register — and mislead auto-exclude.\n\n"
            "Regenerate them: toggle “tilt-flip all” / “rot-flip all” here, or re-flip per "
            "filament in a detail window. Saving any flip rewrites the file in the current "
            "convention and clears this warning.",
            QtWidgets.QMessageBox.StandardButton.Ok, self)
        mb.setModal(False)
        mb.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        mb.show()

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

    def _open_autofit(self, *_):
        """Open the modeless auto-fit-twist dialog (imported lazily; keeps startup lean)."""
        from twist_fit_dialog import TwistFitDialog
        self.autofit = TwistFitDialog(self.ds, self.params, parent=self)
        self.autofit.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.autofit.show()

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
        Each toggle owns a single-bit state for its own disjoint zone.

        ON only flips segments that aren't already flipped onto their register (resume
        files / manual flips stay put -- re-setting them is idempotent). OFF "resumes
        everything": it clears EVERY flip of this register, not just the ones currently
        in the target zone. Loaded flips can fall outside today's zone (they were saved
        under different params), so a zone-only clear would strand them flipped and the
        switch would never fully flip back."""
        # OFF clears the whole register so "click again" reliably undoes the flip-all.
        # tilt owns the pure (1, 0) state, rot the pure (0, 1) state; manual both-register
        # (1, 1) flips belong to neither switch and are left untouched.
        if not on:
            want = (1, 0) if which == "tilt" else (0, 1)
            clear = [t for t in self.store.flip_tags() if self.store.get_state(t) == want]
            self.store.replace_flips({}, clear)
            self.status.setText(f"{which}-flip all OFF: {len(clear)} {which} flips cleared")
            return
        rate = self.ds.model_rate
        nset = 0
        set_map = {}
        for f in self.ds.filaments:
            if not (f.fittable and np.isfinite(f.phi0) and f.axis is not None):
                continue
            # tilt colour of every segment (pose z-axis vs the fitted axis).
            tilt_ang = axis_tilt(f.eulers, f.axis)
            if which == "rot":
                # rot still works on the roll registers, restricted to the on-axis
                # (dark/amber) segments -- off-axis grey ones are irrelevant.
                on_axis = ~tilt_off_axis(tilt_ang)
                rb = ((f.phi - (rate * f.pos + f.phi0) + 180) % 360) - 180
                in_black = np.abs(rb) <= zone
                in_blue = np.abs(((rb - 180 + 180) % 360) - 180) <= zone
                target = in_blue & ~in_black & on_axis
                if not target.any():
                    continue
                pose, bits = rot_flip_eulers(f.eulers, f.axis), (0, 1)
            else:                                          # tilt: flip AMBER -> dark
                # Pure polarity view: target the AMBER (antiparallel) segments -- the
                # minority pointing the wrong way -- and flip them onto the dark
                # (axis-aligned) majority. No roll / register (magenta-line) criterion
                # anymore -- tilt colour alone decides.
                target = tilt_ang >= 180.0 - TILT_ZONE
                if not target.any():
                    continue
                S = register_flip_rotation(f.eulers, f.pos, f.axis, f.flipped, rate,
                                           f.phi0, f.phi0_flip)
                if S is None:
                    continue
                # Reverse polarity per segment: S maps the flipped group -> main, S^-1 the
                # main group -> flipped. to_majority=f.flipped picks the right direction for
                # each so EVERY targeted (amber) segment ends up axis-aligned (dark).
                pose = flipped_eulers(f.eulers, f.pos, f.axis, rate, S, f.flipped)
                bits = (1, 0)
            for i in np.where(target)[0]:
                t = int(f.tags[i])
                set_map[int(t)] = (bits[0], bits[1], tuple(pose[i]))
                nset += 1
        self.store.replace_flips(set_map)
        self.status.setText(f"{which}-flip all ON: {nset} {which}-zone segments flipped")

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

    def _exclude_bad_tilt(self, *_):
        """Mark for removal every segment whose pose is off-axis (grey tilt): neither
        aligned nor antiparallel to its filament's fitted axis. Leaves each filament
        with only its dark (aligned) and amber (antiparallel) segments -- the ones the
        polarity/rot flips can act on. (*_ swallows the clicked bool.)"""
        bad, nfil = [], 0
        for f in self.ds.filaments:
            if not (f.fittable and f.axis is not None):
                continue                                   # no axis fit -> no tilt to test
            off = tilt_off_axis(axis_tilt(f.eulers, f.axis))
            bad.extend(int(t) for t in f.tags[off])
            nfil += 1
        self.store.add(bad)
        self.status.setText(
            f"exclude bad tilt: marked {len(bad)} off-axis (grey) segments "
            f"across {nfil} filaments  ({self.store.count()} total marked)")

    def _refresh_title(self):
        stale = " — ⚠ flipped_list.txt outdated: regenerate flips" if (
            self.store.flip_count() and getattr(self.store, "flips_stale", False)) else ""
        self.setWindowTitle(
            f"invest_helical_F_3D — {self.ds.fmt} tomo {self.ds.tomo} — "
            f"{len(self.ds.filaments)} filaments — {self.store.count()} marked{stale}")

    def _restyle_all(self):
        for p in self.panels:
            p.restyle(self.store)
        self._refresh_title()
