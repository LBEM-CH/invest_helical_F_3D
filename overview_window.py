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
                        color_button_qss, category_visible_mask,
                        CAT_DARK, CAT_BAD, CAT_AMBER)
from selection_store import SelectionStore

_PANEL_W, _PANEL_H = 190, 150
_DASH = pg.mkPen("k", width=1.6, style=QtCore.Qt.PenStyle.DashLine)
_EXCL_ZONE = 20.0          # deg: mark for removal when |roll - black| exceeds this
_MODEL_PTS = 600           # model-line samples per overview panel (they are ~190 px wide)


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
        # Last values pushed to Qt. Re-setting an unchanged title or x-range is not
        # free (setTitle re-renders HTML), and with ~80 panels re-set on every step of
        # the iteration slider that was a measurable slice of the redraw.
        self._title_key = None
        self._halfspan = None

    # `fil` is passed in rather than read off self.fil: when the iteration slider is
    # off the final position it is that iteration's view of the same filament (same
    # tags and positions, earlier poses). self.fil stays the working-set filament.
    def redraw_model(self, fil, rate, halfspan):
        """Rescale x to the shared span and redraw the dashed screw (rate-dependent)."""
        if halfspan != self._halfspan:
            self._halfspan = halfspan
            self.plot.setXRange(-halfspan, halfspan, padding=0)
        if fil.fittable and np.isfinite(fil.phi0):
            # _MODEL_PTS, not the 2000-point default: these panels are ~190 px wide,
            # so a finer curve costs redraw time nobody can see.
            xx, model = model_line(halfspan, fil.phi0, rate, npts=_MODEL_PTS)
            self.model_item.setData(xx, model)
        else:
            self.model_item.setData([], [])

    # `tilt` is supplied by the window (cached per iteration -- see _tilt_for) rather
    # than recomputed here: it costs a scipy Rotation build per panel, which with ~80
    # panels was the largest non-Qt cost of a slider step. None = not fittable.
    def restyle(self, fil, store, tilt, show=(True, True, True), use_flips=True):
        marked = np.array([store.is_marked(t) for t in fil.tags], dtype=bool)
        if tilt is not None:
            phi = effective_phi(fil, store, use_flips)
            vis = category_visible_mask(tilt, show)  # hide unchecked dark/amber/bad-tilt
            self.scatter.setData(x=fil.pos[vis], y=phi[vis],
                                 brush=tilt_brushes(tilt[vis], marked[vis]))
        else:
            self.scatter.setData([])
        key = (bool(marked.any()), int(fil.fid), int(fil.n))
        if key != self._title_key:
            self._title_key = key
            self.plot.setTitle(f"fil {fil.fid} (n={fil.n})",
                               color="#dc1e1e" if key[0] else "#222222", size="8pt")


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
        # Show/hide each tilt category (dark / amber / bad-tilt) across every panel.
        row1.addWidget(QtWidgets.QLabel("show:"))
        self.chk_show = {}
        for code, name in ((CAT_DARK, "dark"), (CAT_AMBER, "amber"), (CAT_BAD, "bad-tilt")):
            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(True)
            cb.toggled.connect(self._restyle_all)
            self.chk_show[code] = cb
            row1.addWidget(cb)
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
        rlay.addLayout(self._build_iter_row())
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
        self._on_iter_changed(self.iter_idx)           # iteration labels + button states
        self._refresh_title()                          # show marked count + stale flag now
        # If the resume flips were written under an older pose convention, their cached
        # angles can't be trusted -- warn once (after the window is up) so they get
        # regenerated rather than silently misleading the display / auto-exclude.
        if self.store.flip_count() and getattr(self.store, "flips_stale", False):
            QtCore.QTimer.singleShot(0, self._warn_stale_flips)

    # --- iteration slider ----------------------------------------------------
    def _build_iter_row(self):
        """Scrub every panel back through the refinement iterations.

        The point is to WATCH the roll-vs-position slope change as refinement
        proceeds: a RELION helical refinement that applies the symmetry folds the
        accumulated phase, so the line flattens iteration by iteration toward a slope
        that is an artefact of the search, not the filament (see docs/pose_and_roll.md
        §8). The measured-slope readout puts a number on that flattening.
        """
        row = QtWidgets.QHBoxLayout()
        labels = self.ds.iter_labels
        # keep the real tooltips: the flip buttons swap to a "why disabled" message
        # off the final iteration and must be able to swap back.
        self._flip_tips = {w: w.toolTip() for w in (self.btn_tiltall, self.btn_rotall)}
        self.iter_idx = max(len(labels) - 1, 0)      # start on the working set
        row.addWidget(QtWidgets.QLabel("iteration"))
        self.sl_iter = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_iter.setMinimumWidth(200)
        self.sl_iter.setRange(0, max(len(labels) - 1, 0))
        self.sl_iter.setValue(self.iter_idx)
        self.sl_iter.setPageStep(1)
        self.sl_iter.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.sl_iter.setTickInterval(1)
        self.sl_iter.valueChanged.connect(self._on_iter_changed)
        row.addWidget(self.sl_iter, 1)
        self.lbl_iter = QtWidgets.QLabel()
        self.lbl_iter.setMinimumWidth(96)
        row.addWidget(self.lbl_iter)
        self.btn_final = QtWidgets.QPushButton("final")
        self.btn_final.setStyleSheet(PLAIN_BTN_QSS)
        self.btn_final.setToolTip("jump back to the working set (the last iteration)")
        self.btn_final.clicked.connect(
            lambda *_: self.sl_iter.setValue(self.sl_iter.maximum()))
        row.addWidget(self.btn_final)
        # Reading every iteration takes seconds, so it is NOT done at startup unless
        # --iteration-paths was passed. When the source has a history we offer to load
        # it here instead of leaving a dead slider and an unexplained "n/a".
        self.btn_loadit = QtWidgets.QPushButton("load iterations")
        self.btn_loadit.setStyleSheet(color_button_qss((25, 110, 200)))
        self.btn_loadit.setToolTip(
            "Read every refinement iteration of this job (takes a few seconds), then "
            "enable the slider. Same as launching with --iteration-paths.")
        self.btn_loadit.clicked.connect(self._load_iterations)
        self.btn_loadit.setVisible(False)
        row.addWidget(self.btn_loadit)
        self.lbl_slope = QtWidgets.QLabel()
        self.lbl_slope.setToolTip(
            "Consensus twist measured from the rolls SHOWN, by the same robust "
            "per-filament slope vote the auto-fit uses. Compare it across iterations "
            "to see how far refinement has flattened the screw.")
        row.addWidget(self.lbl_slope)
        row.addStretch(1)

        if not labels:
            from dynamo_table import can_attach_trajectories
            for w in (self.sl_iter, self.btn_final):
                w.setEnabled(False)
            if can_attach_trajectories(self.ds):
                tip = ("This job has an iteration history — press “load iterations” to "
                       "read it and enable the slider.")
                self.btn_loadit.setVisible(True)
                self.lbl_iter.setText("not loaded")
            else:
                tip = ("No iteration history at this path. Point the tool at the RELION "
                       "refinement JOB FOLDER (the one holding run_it*_data.star), or a "
                       "Dynamo project folder, to step through iterations.")
                self.lbl_iter.setText("n/a")
            self.sl_iter.setToolTip(tip)
            self.lbl_iter.setToolTip(tip)
        else:
            self.sl_iter.setToolTip(
                "Show every panel's roll as it was at this refinement iteration. "
                "Positions never move between iterations — only the poses — so only "
                "the roll and its colouring change.")
        # Recomputing a whole iteration is far cheaper than refitting the twist, so the
        # panels follow the slider immediately and the slope readout trails on a short
        # debounce -- dragging never waits on a fit that is about to be superseded.
        self._slope_timer = QtCore.QTimer(self)
        self._slope_timer.setSingleShot(True)
        self._slope_timer.setInterval(250)
        self._slope_timer.timeout.connect(self._update_slope_label)
        self._iter_cache: dict[int, list] = {}
        self._slope_cache: dict[int, float] = {}
        self._tilt_cache: dict[int, list] = {}
        return row

    def _load_iterations(self, *_):
        """Read the iteration history now and bring the slider to life."""
        from dynamo_table import attach_trajectories
        self.status.setText("reading every iteration of this job…")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        QtWidgets.QApplication.processEvents()        # let the message paint first
        try:
            n = attach_trajectories(self.ds)
        except Exception as e:                        # noqa: BLE001 - report, don't die
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(self, "load iterations",
                                          f"could not read the iteration history:\n{e}")
            self.status.setText("could not load iterations")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if not n:
            self.status.setText("no iteration history found at this path")
            return
        self._iter_cache.clear()
        self._slope_cache.clear()
        self._tilt_cache.clear()
        self.iter_idx = n - 1
        self.sl_iter.blockSignals(True)               # setRange/setValue must not fire
        self.sl_iter.setRange(0, n - 1)               # a redraw before the state is set
        self.sl_iter.setValue(n - 1)
        self.sl_iter.blockSignals(False)
        for w in (self.sl_iter, self.btn_final):
            w.setEnabled(True)
        self.sl_iter.setToolTip(
            "Show every panel's roll as it was at this refinement iteration. "
            "Positions never move between iterations — only the poses.")
        self.btn_loadit.setVisible(False)
        self._on_iter_changed(self.iter_idx)
        self.status.setText(f"loaded {n} iterations — drag the slider to step through them")

    def _is_final(self) -> bool:
        """True when the panels show the working set (so flips/poses are the real ones)."""
        return self.iter_idx >= self.ds.n_iterations - 1

    def _view_fils(self):
        """The filaments as of the displayed iteration (the dataset's own on final).

        Cached per iteration: rebuilding one costs a pose->roll pass plus a robust
        phase fit per filament, which is fine once but not on every slider step.
        Cleared whenever twist/rise/pixel-size change, since the phase fit depends
        on the rate.
        """
        if self._is_final() or not self.ds.n_iterations:
            return self.ds.filaments
        k = self.iter_idx
        if k not in self._iter_cache:
            self._iter_cache[k] = [f.at_iteration(k, self.ds.model_rate,
                                                  self.ds.pixelsize)
                                   for f in self.ds.filaments]
        return self._iter_cache[k]

    def _tilt_for(self, fils):
        """Per-filament tilt-to-axis angle for the displayed iteration, cached.

        Tilt depends only on the poses, the fixed majority-oriented axis and the
        committed flips — none of which change as the slider moves — so scrubbing
        reuses these arrays instead of rebuilding a scipy Rotation per panel per step.
        Invalidated on any store change (a flip alters tilt) and on any parameter
        change (which rebuilds the iteration views outright).
        """
        k = self.iter_idx
        if k not in self._tilt_cache:
            use_flips = self._use_flips()
            self._tilt_cache[k] = [
                effective_tilt(f, self.store, use_flips)
                if (f.fittable and f.axis is not None) else None
                for f in fils]
        return self._tilt_cache[k]

    def _use_flips(self) -> bool:
        """Committed flips are corrections to the FINAL poses, so they are applied
        only there -- an earlier iteration shows its own raw rolls."""
        return self._is_final()

    def _on_iter_changed(self, value):
        self.iter_idx = int(value)
        labels = self.ds.iter_labels
        if labels:
            it = labels[self.iter_idx]
            self.lbl_iter.setText(f"it {it}" + ("  (final)" if self._is_final() else ""))
        # Pose-writing actions must not run against an earlier iteration's poses.
        final = self._is_final()
        for w in (self.btn_tiltall, self.btn_rotall):
            w.setEnabled(final)
            w.setToolTip(self._flip_tips[w] if final else
                         "Disabled while viewing an earlier iteration — a flip rewrites "
                         "the pose, which is only meaningful on the working set. Press "
                         "“final” first.")
        self._redraw_all()
        if not self._is_final():
            self.status.setText(
                f"showing iteration {labels[self.iter_idx]} — marks still apply to the "
                "same segments; opening a filament shows the final iteration")
        self.lbl_slope.setText("measuring…")
        self._slope_timer.start()

    def _update_slope_label(self):
        """Measured twist for the displayed iteration, against it0 and the final."""
        if not self.ds.n_iterations:
            self.lbl_slope.setText("")
            return
        last = self.ds.n_iterations - 1
        try:
            here, first, final = (self._slope_at(k)
                                  for k in (self.iter_idx, 0, last))
        except Exception as e:                        # noqa: BLE001 - readout only
            self.lbl_slope.setText(f"slope: n/a ({e})")
            return
        self.lbl_slope.setText(
            f"<b>slope here: {here:+.2f} °/sub</b> "
            f"<span style='color:#888'>(it0 {first:+.2f} → final {final:+.2f})</span>")
        self.lbl_slope.setTextFormat(QtCore.Qt.TextFormat.RichText)

    def _slope_at(self, k: int) -> float:
        if k not in self._slope_cache:
            from twist_fit import quick_twist
            fils = (self.ds.filaments if k >= self.ds.n_iterations - 1 else
                    [f.at_iteration(k, self.ds.model_rate, self.ds.pixelsize)
                     for f in self.ds.filaments])
            self._slope_cache[k] = float(quick_twist(fils, self.ds.rise))
        return self._slope_cache[k]

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
        # Both caches are keyed on the rate: the per-iteration views hold a phase fit
        # and the slope readouts a twist, so a retune invalidates them.
        self._iter_cache.clear()
        self._slope_cache.clear()
        self._tilt_cache.clear()
        self._redraw_all()
        if self.ds.n_iterations:
            self._slope_timer.start()

    def _redraw_all(self):
        """Redraw model lines + dots for the currently displayed iteration."""
        fils = self._view_fils()
        tilts = self._tilt_for(fils)
        use_flips, show = self._use_flips(), self._show()
        for p, f, t in zip(self.panels, fils, tilts):
            p.redraw_model(f, self.ds.model_rate, self.halfspan)
            p.restyle(f, self.store, t, show, use_flips)

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
        use_flips = self._use_flips()
        bad, nfil = [], 0
        # Judged against the iteration on screen: what you see off the line is what
        # gets marked. Tags are the same at every iteration, so marks made at
        # different iterations simply accumulate into the one remove list.
        for f in self._view_fils():
            if not (f.fittable and np.isfinite(f.phi0)):
                continue                                   # no black line to test against
            resid = ((effective_phi(f, self.store, use_flips)
                      - (rate * f.pos + f.phi0) + 180) % 360) - 180
            bad.extend(int(t) for t in f.tags[np.abs(resid) > zone])
            nfil += 1
        self.store.add(bad)
        self.status.setText(
            f"auto-exclude: marked {len(bad)} segments >±{int(zone)}° off black "
            f"across {nfil} filaments{self._iter_note()}  "
            f"({self.store.count()} total marked)")

    def _exclude_bad_tilt(self, *_):
        """Mark for removal every segment whose pose is off-axis (grey tilt): neither
        aligned nor antiparallel to its filament's fitted axis. Leaves each filament
        with only its dark (aligned) and amber (antiparallel) segments -- the ones the
        polarity/rot flips can act on. (*_ swallows the clicked bool.)"""
        use_flips = self._use_flips()
        bad, nfil = [], 0
        for f in self._view_fils():                        # the iteration on screen
            if not (f.fittable and f.axis is not None):
                continue                                   # no axis fit -> no tilt to test
            off = tilt_off_axis(effective_tilt(f, self.store, use_flips))  # matches the grey dots
            bad.extend(int(t) for t in f.tags[off])
            nfil += 1
        self.store.add(bad)
        self.status.setText(
            f"exclude bad tilt: marked {len(bad)} off-axis (grey) segments "
            f"across {nfil} filaments{self._iter_note()}  "
            f"({self.store.count()} total marked)")

    def _iter_note(self) -> str:
        """' (at iteration N)' when the panels are not showing the working set."""
        if self._is_final() or not self.ds.n_iterations:
            return ""
        return f" (at iteration {self.ds.iter_labels[self.iter_idx]})"

    def _refresh_title(self):
        stale = " — ⚠ flipped_list.txt outdated: regenerate flips" if (
            self.store.flip_count() and getattr(self.store, "flips_stale", False)) else ""
        self.setWindowTitle(
            f"invest_helical_F_3D — {self.ds.fmt} tomo {self.ds.tomo} — "
            f"{len(self.ds.filaments)} filaments — {self.store.count()} marked{stale}")

    def _show(self):
        """(dark_on, bad_on, amber_on) from the view checkboxes, ordered by CAT_*."""
        return (self.chk_show[CAT_DARK].isChecked(),
                self.chk_show[CAT_BAD].isChecked(),
                self.chk_show[CAT_AMBER].isChecked())

    def _restyle_all(self):
        # Driven by store changes, which can include a flip -> the cached tilts may be
        # stale. (Also the path for the show-category checkboxes, where they are not,
        # but dropping them is cheap next to being wrong.)
        self._tilt_cache.clear()
        fils = self._view_fils()
        tilts = self._tilt_for(fils)
        use_flips, show = self._use_flips(), self._show()
        for p, f, t in zip(self.panels, fils, tilts):
            p.restyle(f, self.store, t, show, use_flips)
        self._refresh_title()
