#!/usr/bin/env python3
"""
Per-filament detail window for Rohlex.

author: Wen-Lu Chung

Three linked plots for one filament:
  plot1: roll vs real position + dashed screw model
  plot2: residual (delta) to the model vs position + zero line
  plot3: the filament's XY map (segments in order)

Pointing at a segment in any plot highlights the SAME segment in all three.
Marking for removal: rubber-band drag (Select mode), the Select-all button, or
clicking a single point toggles it. Marks restyle live across every plot and the
overview, and are persisted by the SelectionStore. The twist / rise / pixel-size
bar retunes the model live and is kept in sync with the overview's bar.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

from dynamo_table import Filament
from helix_geom import polarity_flip_eulers, rot_flip_eulers, axis_tilt
from plot_common import (HILITE_PEN, ModelParams, ParamBar, SelectableViewBox,
                         effective_phi, effective_tilt, pos_brushes, tilt_brushes,
                         tilt_off_axis, TILT_ZONE, PLAIN_BTN_QSS, color_button_qss,
                         category_visible_mask, tilt_counts, tilt_category,
                         unwrap_roll, on_branch, unwrapped_phase,
                         CAT_DARK, CAT_BAD, CAT_AMBER)

_DASH = pg.mkPen("k", width=1.6, style=QtCore.Qt.PenStyle.DashLine)
_DARK_RGB = (40, 44, 58)                             # matches the tilt "dark" (axis-aligned) dots
_AMBER_RGB = (240, 168, 0)                           # matches the tilt "amber" (antiparallel) dots
_FLIP_RGB = (235, 64, 170)                           # pink: the tilt-flip (polarity) register
_ROT_RGB = (25, 55, 200)                             # dark blue: the rot-flip (+180) register
_BOTH_RGB = (140, 50, 200)                           # purple: tilt+rot (both) register
_DASH_BOTH = pg.mkPen(_BOTH_RGB, width=1.6, style=QtCore.Qt.PenStyle.DashLine)
_DASH_FLIP = pg.mkPen(_FLIP_RGB, width=1.6, style=QtCore.Qt.PenStyle.DashLine)
_GHOST_PEN = pg.mkPen((235, 64, 170, 120), width=1.3, style=QtCore.Qt.PenStyle.DashLine)  # old-roll ghost
_DASH_ROT = pg.mkPen(_ROT_RGB, width=1.6, style=QtCore.Qt.PenStyle.DashLine)


def _color_button(btn, rgb):
    """Tint a button to match its dashed register line (muted when disabled), using
    the shared toolbar geometry so it lines up with the plain buttons."""
    btn.setStyleSheet(color_button_qss(rgb))
_TRAJ_PEN = pg.mkPen((80, 80, 80, 160), width=1.6)   # grey iteration trails (thicker, antialiased)


def _iter_path_xy(pos, traj_roll, u_final):
    """Connect each segment's per-iteration rolls, in the UNWRAPPED frame.

    Every iteration's roll is placed on the turn nearest that segment's FINAL
    unwrapped value (u_final), so the trail ends exactly on its coloured dot and can
    never wander more than half a turn from it. The old +/-180 seam-splitting is gone
    with the wrapping that required it. A NaN separates segments, for one
    connect='finite' item.
    """
    n_iter, N = traj_roll.shape
    xs, ys = [], []
    for j in range(N):
        x = float(pos[j])
        ref = u_final[j]
        for i in range(n_iter):
            r = traj_roll[i, j]
            if not np.isfinite(r) or not np.isfinite(ref):
                continue
            xs.append(x); ys.append(float(r + 360.0 * round((ref - r) / 360.0)))
        xs.append(np.nan); ys.append(np.nan)                   # separate segments
    return np.asarray(xs), np.asarray(ys)


def _iter_start_xy(pos, traj_roll, u_final):
    """Positions for the starting-value markers: row 0 (iteration 0, the start that
    seeded iteration 1). Intermediate iterations get NO marker -- only the line --
    and the final iteration is the colored data dot. Placed in the same unwrapped
    frame as the trail. Returns (xs, ys)."""
    start = traj_roll[0]
    xs, ys = [], []
    for j in range(len(start)):
        r, ref = start[j], u_final[j]
        if np.isfinite(r) and np.isfinite(ref):
            xs.append(float(pos[j]))
            ys.append(float(r + 360.0 * round((ref - r) / 360.0)))
    return xs, ys


class _Panel:
    """One scatter plot with marking + hover highlight, over arrays (x, y, tags)."""

    def __init__(self, glw: pg.GraphicsLayoutWidget, row: int, col: int,
                 title: str, xlabel: str, ylabel: str):
        self.vb = SelectableViewBox()
        self.plot = glw.addPlot(row=row, col=col, viewBox=self.vb, title=title)
        self.plot.setLabel("bottom", xlabel)
        self.plot.setLabel("left", ylabel)
        self.plot.setMenuEnabled(False)          # no right-click context menu (it's distracting)
        # hoverable for the linked-hover signal, but hoverSize=-1 (default) so the
        # dot itself does NOT grow -- the highlight ring is the only hover cue. tip=None
        # suppresses the x/y/data tooltip block (the readout label shows it instead).
        self.scatter = pg.ScatterPlotItem(size=10, hoverable=True, tip=None,
                                          pen=pg.mkPen(None))
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

    def restyle(self, store, brushes=None, visible=None):
        """Recolor + resize the dots. `brushes` (one per point) overrides the default
        viridis-by-position fill -- the roll / residual panels pass register colors.
        `visible` (bool per point, same order as self.tags) hides unchecked tilt
        categories; None shows all."""
        marked = np.array([store.is_marked(t) for t in self.tags], dtype=bool)
        if brushes is None:
            brushes = pos_brushes(self.x, marked)
        spots = [dict(pos=(float(x), float(y)), data=int(t), brush=b,
                      size=(14 if m else 10))
                 for i, (x, y, t, b, m) in enumerate(
                     zip(self.x, self.y, self.tags, brushes, marked))
                 if visible is None or visible[i]]
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

    def __init__(self, fil: Filament, params: ModelParams, store,
                 map_volume=None, map_voxel=None, gl_enabled=True, parent=None):
        super().__init__(parent)
        self.fil = fil
        self.params = params
        self.store = store
        self.map_volume = map_volume
        self.map_voxel = map_voxel
        self.gl_enabled = gl_enabled
        self.view3d = None
        self.setWindowTitle(f"filament {fil.fid}  (n={fil.n})")
        self.resize(1300, 560)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)

        # --- toolbar (two rows: select/remove/navigate above, flip/params below) --
        bar1 = QtWidgets.QHBoxLayout()
        bar2 = QtWidgets.QHBoxLayout()
        self.btn_all = QtWidgets.QPushButton("Select all in filament")
        self.btn_all.clicked.connect(lambda: self.store.add(self.fil.tags.tolist()))
        self.btn_clear = QtWidgets.QPushButton("Clear filament")
        self.btn_clear.clicked.connect(lambda: self.store.remove(self.fil.tags.tolist()))
        self.btn_badtilt = QtWidgets.QPushButton("exclude bad tilt")
        self.btn_badtilt.setToolTip("Mark for removal every off-axis (grey) segment in this "
                                    "filament — pose neither aligned nor antiparallel to the "
                                    "filament axis. Leaves only dark + amber.")
        self.btn_badtilt.setStyleSheet(color_button_qss((150, 75, 55)))
        self.btn_badtilt.clicked.connect(self._exclude_bad_tilt)
        # remove-by-color: mark every DARK (axis-aligned) or every AMBER (antiparallel)
        # segment for removal in one click -- the same tilt colors the dots show.
        self.btn_rmdark = QtWidgets.QPushButton("remove dark")
        self.btn_rmdark.setToolTip("Mark for removal every DARK (axis-aligned) segment in "
                                   "this filament — the ones on the main tilt register.")
        self.btn_rmdark.setStyleSheet(color_button_qss(_DARK_RGB))
        self.btn_rmdark.clicked.connect(lambda: self._remove_color("dark"))
        self.btn_rmamber = QtWidgets.QPushButton("remove amber")
        self.btn_rmamber.setToolTip("Mark for removal every AMBER (antiparallel) segment in "
                                    "this filament — the polarity-flipped register.")
        self.btn_rmamber.setStyleSheet(color_button_qss(_AMBER_RGB))
        self.btn_rmamber.clicked.connect(lambda: self._remove_color("amber"))
        self.btn_home = QtWidgets.QPushButton("Home (reset view)")
        self.btn_home.clicked.connect(self._reset_view)
        self.btn_back = QtWidgets.QPushButton("← Back to overview")
        self.btn_back.clicked.connect(self.close)
        self.btn_3d = QtWidgets.QPushButton("View 3D")
        self.btn_3d.clicked.connect(self._open_3d)
        if not self.gl_enabled:
            self.btn_3d.setEnabled(False)
            self.btn_3d.setToolTip("OpenGL unavailable (typical over ssh -XY). "
                                   "Run locally, or with --gl / VirtualGL, for the 3D view.")
        # flip mode: a checkable toggle (selection re-flips instead of excludes) plus
        # the action button that commits the staged selection onto the other register.
        self.btn_flipmode = QtWidgets.QPushButton("⇋ Flip mode")
        self.btn_flipmode.setCheckable(True)
        self.btn_flipmode.setToolTip(
            "Selection re-flips segments (polarity dyad) instead of marking them for "
            "removal. Rubber-band to stage, then press Flip.")
        self.btn_flipmode.setStyleSheet(color_button_qss(None, checked_rgb=(218, 135, 30)))
        self.btn_flipmode.toggled.connect(self._set_mode)
        self.btn_doflip = QtWidgets.QPushButton("tilt-flip")
        self.btn_doflip.setToolTip("Tilt-flip the staged segments: the polarity dyad "
                                   "(~180° about an axis ⊥ the filament). Onto the pink "
                                   "register; angles change, position stays.")
        self.btn_doflip.setEnabled(False)
        self.btn_doflip.clicked.connect(self._commit_flip)
        _color_button(self.btn_doflip, _FLIP_RGB)      # dark pink, matches its register line
        self.btn_rotflip = QtWidgets.QPushButton("rot-flip")
        self.btn_rotflip.setToolTip("Rot-flip the staged segments: rotate 180° about the "
                                    "filament axis (C2). Onto the dark-blue register "
                                    "(main line + 180°); polarity unchanged.")
        self.btn_rotflip.setEnabled(False)
        self.btn_rotflip.clicked.connect(self._commit_rot_flip)
        _color_button(self.btn_rotflip, _ROT_RGB)      # dark blue, matches its register line
        self.btn_resume = QtWidgets.QPushButton("↺ Resume")
        self.btn_resume.setToolTip("Undo: revert every flipped segment in this filament "
                                   "back to its original angles.")
        self.btn_resume.setEnabled(False)
        self.btn_resume.clicked.connect(self._resume_flips)
        self.chk_traj = QtWidgets.QCheckBox("Iteration paths")
        _has_traj = fil.traj_roll is not None
        self.chk_traj.setChecked(_has_traj)
        self.chk_traj.setEnabled(_has_traj)
        self.chk_traj.setToolTip(
            (f"per-segment roll path: start (grey) → "
             f"{sum(1 for i in fil.traj_iters if i != 0)} iterations → final (colored dot)")
            if _has_traj else
            "point the tool at the Dynamo project folder (…/abp_align_eo) "
            "to see how each segment's roll converged from its starting value")
        self.chk_traj.toggled.connect(self._refit)
        for w in (self.btn_all, self.btn_clear, self.btn_home, self.btn_back, self.btn_3d,
                  self.btn_resume):
            w.setStyleSheet(PLAIN_BTN_QSS)            # uniform height with the colored buttons
        # Row 1: select + remove-by-color + navigation.
        for w in (self.btn_all, self.btn_clear, self.btn_badtilt, self.btn_rmdark,
                  self.btn_rmamber, self.btn_home, self.btn_back, self.btn_3d):
            bar1.addWidget(w)
        # Show/hide each tilt category; the label carries this filament's live count.
        bar1.addWidget(QtWidgets.QLabel("show:"))
        self.chk_show = {}
        for code, name in ((CAT_DARK, "dark"), (CAT_AMBER, "amber"), (CAT_BAD, "bad-tilt")):
            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(True)
            cb.toggled.connect(self._restyle_all)
            self.chk_show[code] = cb
            bar1.addWidget(cb)
        bar1.addStretch(1)
        # Row 2: flip machinery + iteration paths + the live twist/rise/pixel bar + readout.
        for w in (self.btn_flipmode, self.btn_doflip, self.btn_rotflip, self.btn_resume):
            bar2.addWidget(w)
        bar2.addWidget(self.chk_traj)
        bar2.addWidget(ParamBar(params))
        self.readout = QtWidgets.QLabel("hover a segment…")
        self.readout.setMinimumWidth(340)
        bar2.addWidget(self.readout)
        bar2.addStretch(1)
        outer.addLayout(bar1)
        outer.addLayout(bar2)

        # --- plots -----------------------------------------------------------
        glw = pg.GraphicsLayoutWidget()
        outer.addWidget(glw, 1)
        self.p1 = _Panel(glw, 0, 0, f"fil {fil.fid}: roll vs position",
                         "position along axis (Å)", "unwrapped roll (deg)")
        self.p2 = _Panel(glw, 0, 1, "residual to model",
                         "position along axis (Å)", "delta: data - model (deg)")
        self.p3 = _Panel(glw, 0, 2, "XY map", "X (px)", "Y (px)")
        self.panels = [self.p1, self.p2, self.p3]

        self.p2.plot.addLine(y=0, pen=pg.mkPen("k", style=QtCore.Qt.PenStyle.DashLine))
        self.p3.vb.setAspectLocked(True)
        self.p3.set_data(fil.xy[:, 0], fil.xy[:, 1], fil.tags)
        self.p3.plot.plot(fil.xy[:, 0], fil.xy[:, 1],
                          pen=pg.mkPen((150, 150, 150), width=1))   # connecting line
        self.traj_item = pg.PlotDataItem([], [], connect="finite", pen=_TRAJ_PEN,
                                         antialias=True)
        self.traj_item.setZValue(-10)                  # beneath the dots and model line
        # ignoreBounds: trails must not drive auto-range -- the view stays on the span
        # of the unwrapped roll. Each trail is placed within half a turn of its own
        # dot (_iter_path_xy), so it is in view anyway.
        self.p1.plot.addItem(self.traj_item, ignoreBounds=True)
        # grey marker at the starting values (iteration 0); the final iter is the
        # colored dot, and intermediate iters are line-only.
        self.traj_nodes = pg.ScatterPlotItem(size=7, brush=pg.mkBrush(105, 105, 105, 235),
                                             pen=pg.mkPen(None), hoverable=False)
        self.traj_nodes.setZValue(-5)                  # above the trail line, below the dots
        self.p1.plot.addItem(self.traj_nodes, ignoreBounds=True)
        self.model_item = self.p1.plot.plot([], [], connect="finite", pen=_DASH)
        # the flipped (polarity-reversed) register: a pink dashed line marking where
        # the flipped segments sit. Rings are NOT shown by default -- only after a
        # flip, as faint pink dashed "ghosts" at each segment's OLD roll.
        self.model_flip_item = self.p1.plot.plot([], [], connect="finite", pen=_DASH_FLIP)
        # the rot-flip register (180 deg about the axis) is just the main line + 180.
        self.model_rot_item = self.p1.plot.plot([], [], connect="finite", pen=_DASH_ROT)
        # the both register (tilt + rot) is the pink line + 180.
        self.model_both_item = self.p1.plot.plot([], [], connect="finite", pen=_DASH_BOTH)
        self.ghost_marker = pg.ScatterPlotItem(size=15, symbol="o", hoverable=False,
                                               pen=_GHOST_PEN, brush=pg.mkBrush(None))
        self.ghost_marker.setZValue(-4)
        self.staged_marker = pg.ScatterPlotItem(size=17, symbol="s", hoverable=False,
                                                pen=pg.mkPen(_FLIP_RGB, width=2),
                                                brush=pg.mkBrush(None))
        self.staged_marker.setZValue(-3)
        for it in (self.ghost_marker, self.staged_marker):
            self.p1.plot.addItem(it, ignoreBounds=True)

        # --- flip state ------------------------------------------------------
        self.mode = "exclude"                          # or "flip"
        self.flip_staged: set[int] = set()             # staged for flip, not yet committed

        # --- wiring ----------------------------------------------------------
        for p in self.panels:
            p.scatter.sigHovered.connect(self._on_hover)
            p.scatter.sigClicked.connect(self._on_click)
            p.vb.regionSelected.connect(self._on_select)
            p.vb.regionDeselected.connect(self._on_deselect)
        self.store.changed.connect(self._refresh_data)
        self.params.changed.connect(self._refit)
        self._refit()                                  # fill roll/residual + model
        self.statusBar().showMessage(
            "left-drag = mark   |   right-drag = unmark   |   scroll = zoom   |   click = toggle one")

    def _open_3d(self):
        """Open this filament's 3D view (GL imported lazily so 2D never needs it)."""
        try:
            from view3d_window import View3DWindow
        except ImportError as e:
            QtWidgets.QMessageBox.warning(
                self, "View 3D", f"3D view needs PyOpenGL:\n{e}\n\npip install PyOpenGL")
            return
        self.view3d = View3DWindow(self.fil, self.params.ds, self.store,
                                   volume=self.map_volume, map_voxel=self.map_voxel,
                                   parent=self)
        self.view3d.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        # drop our reference once it closes, so hover-linking stops touching it
        self.view3d.destroyed.connect(lambda *_: setattr(self, "view3d", None))
        self.view3d.show()

    # --- interaction ---------------------------------------------------------
    def _refit(self):
        """Parameters changed (or first show): redraw the rate-dependent overlays
        (model line, flipped-register line, iteration trails), then refresh data."""
        if self.fil.fittable and np.isfinite(self.fil.phi0):
            xx, phi0 = self._model_xx(), self._model_phase()
            self.model_item.setData(xx, self.params.model_rate * xx + phi0)
            self.model_rot_item.setData(                # dark-blue rot-flip register (+180)
                xx, self.params.model_rate * xx + phi0 + 180.0)
        else:
            self.model_item.setData([], [])
            self.model_rot_item.setData([], [])
        self._refit_flip()
        # per-iteration roll-convergence trails (uses current Angstrom x). These are
        # the ORIGINAL rolls and are never altered by a flip.
        if self.fil.traj_roll is not None and self.chk_traj.isChecked():
            u = self._unwrapped()
            xs, ys = _iter_path_xy(self.fil.pos, self.fil.traj_roll, u)
            self.traj_item.setData(xs, ys)
            nx, ny = _iter_start_xy(self.fil.pos, self.fil.traj_roll, u)
            if nx:
                self.traj_nodes.setData(x=nx, y=ny)
            else:
                self.traj_nodes.setData([])
        else:
            self.traj_item.setData([], [])
            self.traj_nodes.setData([])
        self._refresh_data()

    # --- unwrapped-roll frame -------------------------------------------------
    # The roll panel plots the roll UNWRAPPED (plot_common.unwrap_roll) so a fast
    # screw stays readable; every register line is therefore one straight line and
    # every marker has to be placed on the same turn as the data it belongs to.
    def _unwrapped(self):
        """The filament's raw roll, unwrapped head-to-tail. Model-free and stable:
        it does not move when twist / rise / pixel size are retuned."""
        return unwrap_roll(self.fil.phi)

    def _model_xx(self):
        h = self.params.ds.pos_halfspan
        return np.array([-h, h])

    def _model_phase(self, phi0=None):
        """A register's phase lifted by whole turns onto the unwrapped data, so its
        straight line is drawn where the points are. The model is unchanged modulo
        360, so the residual panel and the tilt colours are untouched."""
        if phi0 is None:
            phi0 = self.fil.phi0
        return unwrapped_phase(self._unwrapped(), self.fil.pos,
                               self.params.model_rate, phi0)

    def _refit_flip(self):
        """Draw the polarity-flipped register as a pink dashed line and note it in
        the title (no rings -- those appear only as post-flip ghosts)."""
        fil = self.fil
        if fil.fittable and np.isfinite(fil.phi0_flip):
            xx = self._model_xx()
            phi0f = self._model_phase(fil.phi0_flip)
            rate = self.params.model_rate
            self.model_flip_item.setData(xx, rate * xx + phi0f)
            # tilt + rot register (pink + 180)
            self.model_both_item.setData(xx, rate * xx + phi0f + 180.0)
            doff = ((fil.phi0_flip - fil.phi0 + 180) % 360) - 180
            self.p1.plot.setTitle(
                f"fil {fil.fid}: roll vs position   "
                f"<span style='color:#eb40aa'>(flipped register: "
                f"{int(fil.flipped.sum())} seg, Δφ={doff:+.0f}°)</span>")
        else:
            self.model_flip_item.setData([], [])
            self.model_both_item.setData([], [])
            self.p1.plot.setTitle(f"fil {fil.fid}: roll vs position")

    # --- flip machinery -------------------------------------------------------
    def _effective_arrays(self):
        """(roll, residual) with committed flips applied -- flipped tags read their
        NEW roll (from the stored flipped angles); everything else is the original.

        The roll comes back in the UNWRAPPED frame for the panel; the residual is
        computed from the wrapped roll exactly as before, so marking, the tilt colours
        and the auto-exclude threshold are unaffected by the display change.
        """
        fil = self.fil
        phi = effective_phi(fil, self.store)
        if fil.fittable and np.isfinite(fil.phi0):
            model = self.params.model_rate * fil.pos + fil.phi0
            delta = ((phi - model + 180) % 360) - 180
        else:
            delta = np.full(fil.n, np.nan)
        return on_branch(phi, self._unwrapped()), delta

    def _refresh_data(self):
        """Push effective roll/residual into the panels and redraw flip markers."""
        phi, delta = self._effective_arrays()
        self.p1.set_data(self.fil.pos, phi, self.fil.tags)
        self.p2.set_data(self.fil.pos, delta, self.fil.tags)
        self._draw_flip_markers(phi)
        self._restyle_all()

    def _draw_flip_markers(self, phi_eff):
        """Ghost rings at the OLD roll of committed flips; squares stay on every
        flipped (committed) OR staged segment so you can see what is/was flipped."""
        fil = self.fil
        u = self._unwrapped()          # ghosts sit at the OLD roll, in the panel's frame
        gx, gy, sx, sy = [], [], [], []
        for i, t in enumerate(fil.tags):
            t = int(t)
            flipped = self.store.is_flipped(t)
            if flipped:
                gx.append(float(fil.pos[i])); gy.append(float(u[i]))
            if flipped or t in self.flip_staged:
                sx.append(float(fil.pos[i])); sy.append(float(phi_eff[i]))
        self.ghost_marker.setData(x=gx, y=gy)
        self.staged_marker.setData(x=sx, y=sy)

    def _set_mode(self, flip_on):
        self.mode = "flip" if flip_on else "exclude"
        self.btn_doflip.setEnabled(flip_on)
        self.btn_rotflip.setEnabled(flip_on)
        self.btn_resume.setEnabled(flip_on)
        if not flip_on and self.flip_staged:
            self.flip_staged.clear()
            self._draw_flip_markers(self.p1.y)
        self.statusBar().showMessage(
            "FLIP mode: left-drag = stage, then tilt-flip / rot-flip (commits and clears "
            "the selection). Right-drag = unstage (won't unflip); click a flipped dot or "
            "Resume to undo." if flip_on else
            "left-drag = mark   |   right-drag = unmark   |   scroll = zoom   |   click = toggle one")

    def _toggle_flip(self, which):
        """Toggle the tilt- or rot-bit of each staged segment's (tilt, rot) state. The
        pose is recomputed from the ORIGINAL angles for the resulting state. Committing
        FREES the selection (clears the staged set), so the next rubber-band picks a
        fresh group instead of re-flipping the one just committed. Re-select the same
        segments to compose a second flip (e.g. tilt then rot) or to undo."""
        if not self.flip_staged:
            return
        fil = self.fil
        idx = np.array([i for i, t in enumerate(fil.tags)
                        if int(t) in self.flip_staged], int)
        orig = fil.eulers[idx]
        # Identical operation to "tilt-flip all": reverse each staged segment where it
        # stands, no fit and no rate involved (helix_geom.polarity_flip_eulers). It is
        # always available -- unlike the old register flip it needs neither a second
        # polarity group in this filament nor a fittable axis.
        tilt = polarity_flip_eulers(orig)
        pose = {(0, 0): orig, (1, 0): tilt}
        if fil.fittable and fil.axis is not None:          # rot needs the filament axis
            pose[(0, 1)] = rot_flip_eulers(orig, fil.axis)
            pose[(1, 1)] = rot_flip_eulers(tilt, fil.axis)
        set_map, clear = {}, []
        for k, i in enumerate(idx):
            t = int(fil.tags[i])
            tt, rt = self.store.get_state(t)
            if which == "tilt":
                tt ^= 1
            else:
                rt ^= 1
            if (tt, rt) == (0, 0):
                clear.append(t)
            elif (tt, rt) in pose:                     # rot states need an axis; skip if absent
                set_map[t] = (tt, rt, tuple(pose[(tt, rt)][k]))
        self.flip_staged.clear()                       # commit frees the selection (point 3/4)
        self.store.replace_flips(set_map, clear)       # one save+signal -> refresh redraws markers

    def _commit_flip(self):
        """Tilt-flip (polarity dyad): toggle the tilt bit of the staged segments."""
        self._toggle_flip("tilt")

    def _commit_rot_flip(self):
        """Rot-flip (180° about the filament axis): toggle the rot bit of the staged."""
        if self.fil.fittable:
            self._toggle_flip("rot")

    def _resume_flips(self):
        """Undo: revert every flipped segment in this filament to its original angles."""
        self.flip_staged.clear()
        here = [int(t) for t in self.fil.tags if self.store.is_flipped(int(t))]
        if here:
            self.store.unflip(here)                    # -> changed -> _refresh_data
        else:
            self._draw_flip_markers(self.p1.y)

    def _exclude_bad_tilt(self, *_):
        """Mark for removal this filament's off-axis (grey) segments: pose neither
        aligned nor antiparallel to the fitted axis. Leaves only dark + amber. Uses the
        SAME flip-aware effective tilt the dots are colored by, so it marks exactly the
        grey dots shown (raw eulers would miss committed flips)."""
        if not (self.fil.fittable and self.fil.axis is not None):
            return
        off = tilt_off_axis(effective_tilt(self.fil, self.store))
        self.store.add([int(t) for t in self.fil.tags[off]])

    def _remove_color(self, which):
        """Mark for removal every segment of ONE tilt color in this filament and add it
        to the remove list: `dark` = axis-aligned (tilt ≈ 0), `amber` = antiparallel
        (tilt ≈ 180). Uses the SAME flip-aware effective tilt the dots are colored by,
        so what you remove matches what you see (a committed flip moves a dot dark↔amber
        and this follows it)."""
        if not (self.fil.fittable and self.fil.axis is not None):
            return
        tilt = effective_tilt(self.fil, self.store)          # flip-aware, [0, 180]
        sel = (tilt <= TILT_ZONE) if which == "dark" else (tilt >= 180.0 - TILT_ZONE)
        tags = [int(t) for t in self.fil.tags[sel]]
        self.store.add(tags)
        self.statusBar().showMessage(
            f"removed {len(tags)} {which} segment(s) → remove list "
            f"({self.store.count()} total marked)")

    def _reset_view(self):
        for p in self.panels:
            p.vb.autoRange()

    def _on_hover(self, scatter, points, ev=None):
        if len(points) == 0:
            return
        # Map by TAG, not by points[0].index(): when a tilt category is hidden the
        # scatter holds only the visible subset, so its spot index no longer lines up
        # with the full fil.tags / panel arrays. The spot's data (tag) is stable.
        t = int(points[0].data())
        matches = np.where(self.fil.tags.astype(int) == t)[0]
        if matches.size == 0:
            return
        idx = int(matches[0])
        for p in self.panels:
            p.show_hover(idx)
        if self.view3d is not None:                # mirror the hover onto the 3D scene
            self.view3d.highlight_tags([t])
        flags = [s for s, on in (("MARKED", self.store.is_marked(t)),
                                 ("FLIPPED", self.store.is_flipped(t)),
                                 ("STAGED", t in self.flip_staged)) if on]
        self.readout.setText(
            f"tag {t}   pos={self.fil.pos[idx]:+.1f}Å   "
            f"roll={self.p1.y[idx]:+.1f}°   delta={self.p2.y[idx]:+.1f}°   "
            + " ".join(f"[{s}]" for s in flags))

    def _on_click(self, scatter, points, ev=None):
        if not points:
            return
        t = int(points[0].data())
        if self.mode == "exclude":
            self.store.toggle(t)
        elif self.store.is_flipped(t):
            self.store.unflip([t])                 # click a flipped point -> unflip it
        else:
            self.flip_staged.symmetric_difference_update({t})   # toggle staged
            self._draw_flip_markers(self.p1.y)

    def _panel_of(self, vb):
        for p in self.panels:
            if p.vb is vb:
                return p
        return None

    def _on_select(self, rect: QtCore.QRectF):
        p = self._panel_of(self.sender())          # select against the emitting panel's coords
        if not p:
            return
        tags = p.tags_in_rect(rect)
        if not tags:
            return
        if self.mode == "flip":
            self.flip_staged.update(int(t) for t in tags)
            self._draw_flip_markers(self.p1.y)
        else:
            self.store.add(tags)

    def _on_deselect(self, rect: QtCore.QRectF):
        p = self._panel_of(self.sender())
        if not p:
            return
        tags = p.tags_in_rect(rect)
        if not tags:
            return
        if self.mode == "flip":
            # Right-drag only UNSTAGES (point 5): it must not flip committed segments
            # back. To undo a flip, click a flipped dot or press Resume.
            self.flip_staged.difference_update(int(t) for t in tags)
            self._draw_flip_markers(self.p1.y)
        else:
            self.store.remove(tags)

    def _show(self):
        """(dark_on, bad_on, amber_on) from the view checkboxes, ordered by CAT_*."""
        return (self.chk_show[CAT_DARK].isChecked(),
                self.chk_show[CAT_BAD].isChecked(),
                self.chk_show[CAT_AMBER].isChecked())

    def _update_cat_labels(self, tilt):
        """Put this filament's live segment count of each tilt category on its checkbox,
        plus the cos-weighted score (Σ|cos tilt|) for dark and amber -- the weight each
        register casts in the majority vote (the larger is the dominant polarity)."""
        nd, nb, na = tilt_counts(tilt)
        cat = tilt_category(tilt)
        acos = np.abs(np.cos(np.radians(np.asarray(tilt, float))))
        dark_cos = float(acos[cat == CAT_DARK].sum())
        amber_cos = float(acos[cat == CAT_AMBER].sum())
        self.chk_show[CAT_DARK].setText(f"dark ({nd}, cos {dark_cos:.1f})")
        self.chk_show[CAT_AMBER].setText(f"amber ({na}, cos {amber_cos:.1f})")
        self.chk_show[CAT_BAD].setText(f"bad-tilt ({nb})")

    def _restyle_all(self):
        # All three panels -- roll (p1), residual (p2) AND the XY map (p3) -- are colored
        # by the geometric tilt angle (pose z-axis vs the coordinate-fit filament axis):
        # dark along the axis, amber antiparallel (flipped), grey perpendicular. Sharing
        # the color lets the map show the dark/amber split the roll plots reveal.
        if self.fil.fittable and self.fil.axis is not None:
            tilt = effective_tilt(self.fil, self.store)   # flip-aware: flipped dots turn amber
            marked = np.array([self.store.is_marked(int(t)) for t in self.fil.tags], dtype=bool)
            brushes = tilt_brushes(tilt, marked)
            vis = category_visible_mask(tilt, self._show())   # hide unchecked categories
            self._update_cat_labels(tilt)                     # show live per-category counts
            self.p1.restyle(self.store, brushes, vis)
            self.p2.restyle(self.store, brushes, vis)
            self.p3.restyle(self.store, brushes, vis)
        else:
            self.p1.restyle(self.store)
            self.p2.restyle(self.store)
            self.p3.restyle(self.store)
