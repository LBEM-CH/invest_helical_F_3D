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
overview, and are persisted by the SelectionStore. The twist / rise / pixel-size
bar retunes the model live and is kept in sync with the overview's bar.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets
from scipy.spatial.transform import Rotation as Rot

from dynamo_table import Filament
from helix_geom import (model_line, register_flip_rotation, flipped_eulers,
                        rot_flip_eulers)
from plot_common import (HILITE_PEN, ModelParams, ParamBar, SelectableViewBox,
                         effective_phi, pos_brushes)

_DASH = pg.mkPen("k", width=1.6, style=QtCore.Qt.PenStyle.DashLine)
_FLIP_RGB = (235, 64, 170)                           # pink: the tilt-flip (polarity) register
_ROT_RGB = (25, 55, 200)                             # dark blue: the rot-flip (+180) register
_BOTH_RGB = (140, 50, 200)                           # purple: tilt+rot (both) register
_DASH_BOTH = pg.mkPen(_BOTH_RGB, width=1.6, style=QtCore.Qt.PenStyle.DashLine)
_DASH_FLIP = pg.mkPen(_FLIP_RGB, width=1.6, style=QtCore.Qt.PenStyle.DashLine)
_GHOST_PEN = pg.mkPen((235, 64, 170, 120), width=1.3, style=QtCore.Qt.PenStyle.DashLine)  # old-roll ghost
_DASH_ROT = pg.mkPen(_ROT_RGB, width=1.6, style=QtCore.Qt.PenStyle.DashLine)


def _color_button(btn, rgb):
    """Tint a button to match its dashed register line (muted when disabled)."""
    r, g, b = rgb
    btn.setStyleSheet(
        f"QPushButton {{ background-color: rgb({r},{g},{b}); color: white; "
        f"border: 1px solid rgb({r},{g},{b}); border-radius: 3px; padding: 3px 9px; }}"
        f"QPushButton:disabled {{ background-color: rgba({r},{g},{b},70); "
        f"color: rgba(255,255,255,150); }}")
_TRAJ_PEN = pg.mkPen((80, 80, 80, 160), width=1.6)   # grey iteration trails (thicker, antialiased)


def _iter_path_xy(pos, traj_roll):
    """Connect each segment's per-iteration rolls, keeping every value in (-180,180]
    (the real roll domain -- no >180 numbers). A step that crosses the +/-180 seam is
    drawn as two pieces: up to the edge, then in from the opposite edge (a NaN break
    between), so the trail "wraps" instead of running off-scale, and no dot is left
    unlinked. A NaN also separates segments, for one connect='finite' item.
    """
    n_iter, N = traj_roll.shape
    xs, ys = [], []
    for j in range(N):
        x = float(pos[j])
        prev = None
        for i in range(n_iter):
            r = traj_roll[i, j]
            if not np.isfinite(r):
                prev = None
                continue
            if prev is not None:
                step = ((r - prev + 180.0) % 360.0) - 180.0    # short signed move
                if prev + step > 180.0:                        # crossed the top seam
                    xs += [x, np.nan, x]; ys += [180.0, np.nan, -180.0]
                elif prev + step < -180.0:                     # crossed the bottom seam
                    xs += [x, np.nan, x]; ys += [-180.0, np.nan, 180.0]
            xs.append(x); ys.append(float(r))
            prev = r
        xs.append(np.nan); ys.append(np.nan)                   # separate segments
    return np.asarray(xs), np.asarray(ys)


def _iter_start_xy(pos, traj_roll):
    """Positions for the starting-value markers: row 0 (iteration 0, the start that
    seeded iteration 1). Intermediate iterations get NO marker -- only the line --
    and the final iteration is the colored data dot. Returns (xs, ys)."""
    start = traj_roll[0]
    xs, ys = [], []
    for j in range(len(start)):
        r = start[j]
        if np.isfinite(r):
            xs.append(float(pos[j])); ys.append(float(r))
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

    def restyle(self, store):
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
                  self.btn_flipmode, self.btn_doflip, self.btn_rotflip, self.btn_resume):
            bar.addWidget(w)
        bar.addWidget(self.chk_traj)
        bar.addWidget(ParamBar(params))
        self.readout = QtWidgets.QLabel("hover a segment…")
        self.readout.setMinimumWidth(340)
        bar.addWidget(self.readout)
        bar.addStretch(1)
        outer.addLayout(bar)

        # --- plots -----------------------------------------------------------
        glw = pg.GraphicsLayoutWidget()
        outer.addWidget(glw, 1)
        self.p1 = _Panel(glw, 0, 0, f"fil {fil.fid}: roll vs position",
                         "position along axis (Å)", "roll (deg)")
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
        # ignoreBounds: trails must not drive auto-range -- the view stays on the
        # roll domain (~[-180,180]); genuinely big-wandering trails just run off it.
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
        self._S = False                                # lazy register-flip rotation cache

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
        self._S = False                                # rate changed -> recompute flip op
        if self.fil.fittable and np.isfinite(self.fil.phi0):
            xx, model = model_line(self.params.ds.pos_halfspan, self.fil.phi0,
                                   self.params.model_rate)
            self.model_item.setData(xx, model)
            xr, rot = model_line(self.params.ds.pos_halfspan, self.fil.phi0 + 180.0,
                                 self.params.model_rate)
            self.model_rot_item.setData(xr, rot)       # dark-blue rot-flip register (+180)
        else:
            self.model_item.setData([], [])
            self.model_rot_item.setData([], [])
        self._refit_flip()
        # per-iteration roll-convergence trails (uses current Angstrom x). These are
        # the ORIGINAL rolls and are never altered by a flip.
        if self.fil.traj_roll is not None and self.chk_traj.isChecked():
            xs, ys = _iter_path_xy(self.fil.pos, self.fil.traj_roll)   # true rolls, seam-aware
            self.traj_item.setData(xs, ys)
            nx, ny = _iter_start_xy(self.fil.pos, self.fil.traj_roll)
            if nx:
                self.traj_nodes.setData(x=nx, y=ny)
            else:
                self.traj_nodes.setData([])
        else:
            self.traj_item.setData([], [])
            self.traj_nodes.setData([])
        self._refresh_data()

    def _refit_flip(self):
        """Draw the polarity-flipped register as a pink dashed line and note it in
        the title (no rings -- those appear only as post-flip ghosts)."""
        fil = self.fil
        if fil.fittable and np.isfinite(fil.phi0_flip):
            xx, model = model_line(self.params.ds.pos_halfspan, fil.phi0_flip,
                                   self.params.model_rate)
            self.model_flip_item.setData(xx, model)
            xb, modelb = model_line(self.params.ds.pos_halfspan, fil.phi0_flip + 180.0,
                                    self.params.model_rate)
            self.model_both_item.setData(xb, modelb)   # tilt + rot register (pink + 180)
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
    def _flip_rotation(self):
        """Lazily compute (and cache) S: the rotation mapping this filament's flipped
        register onto the majority one. None if both polarities aren't present."""
        if self._S is False:
            fil = self.fil
            ok = (fil.fittable and getattr(fil, "flipped", np.array([])).size == fil.n
                  and fil.flipped.any() and (~fil.flipped).any())
            self._S = (register_flip_rotation(fil.eulers, fil.pos, fil.axis, fil.flipped,
                                              self.params.model_rate, fil.phi0, fil.phi0_flip)
                       if ok else None)
        return self._S

    def _effective_arrays(self):
        """(roll, residual) with committed flips applied -- flipped tags read their
        NEW roll (from the stored flipped angles); everything else is the original."""
        fil = self.fil
        phi = effective_phi(fil, self.store)
        if fil.fittable and np.isfinite(fil.phi0):
            model = self.params.model_rate * fil.pos + fil.phi0
            delta = ((phi - model + 180) % 360) - 180
        else:
            delta = np.full(fil.n, np.nan)
        return phi, delta

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
        gx, gy, sx, sy = [], [], [], []
        for i, t in enumerate(fil.tags):
            t = int(t)
            flipped = self.store.is_flipped(t)
            if flipped:
                gx.append(float(fil.pos[i])); gy.append(float(fil.phi[i]))
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
            "FLIP mode: left-drag = stage, then tilt-flip / rot-flip (they compose; "
            "click the same flip twice to undo). Right-drag = unflip." if flip_on else
            "left-drag = mark   |   right-drag = unmark   |   scroll = zoom   |   click = toggle one")

    @staticmethod
    def _toggled_state(state, which):
        """Toggle one bit of a flip state. state in {none,tilt,rot,both}; which in
        {tilt,rot}. Each bit toggles independently, so the same flip twice cancels."""
        tilt = state in ("tilt", "both")
        rot = state in ("rot", "both")
        if which == "tilt":
            tilt = not tilt
        else:
            rot = not rot
        return {(0, 0): "none", (1, 0): "tilt", (0, 1): "rot", (1, 1): "both"}[(int(tilt), int(rot))]

    def _toggle_flip(self, which):
        """Toggle the tilt- or rot-bit of each staged segment's flip state. Every pose
        is recomputed from the ORIGINAL angles for the resulting state, so flips
        compose AND applying the same flip twice returns exactly to the original
        (regardless of whether the operators are perfect involutions). The staged set
        is kept, so the buttons can be clicked repeatedly on the same selection."""
        if not self.flip_staged:
            return
        fil = self.fil
        S = self._flip_rotation()
        if which == "tilt" and S is None:
            QtWidgets.QMessageBox.information(
                self, "tilt-flip", "No flipped register detected in this filament — "
                "both polarities must be present to define the tilt flip.")
            return
        idx = np.array([i for i, t in enumerate(fil.tags)
                        if int(t) in self.flip_staged], int)
        orig = fil.eulers[idx]
        rate = self.params.model_rate
        cand = {"none": orig, "rot": rot_flip_eulers(orig, fil.axis)}
        if S is not None:
            tilt = flipped_eulers(orig, fil.pos[idx], fil.axis, rate, S, np.ones(len(idx), bool))
            cand["tilt"] = tilt
            cand["both"] = rot_flip_eulers(tilt, fil.axis)
        Rcand = {s: Rot.from_euler('ZXZ', v, degrees=True) for s, v in cand.items()}
        set_map, clear = {}, []
        for k, i in enumerate(idx):
            t = int(fil.tags[i])
            cur = self.store.get_flip(t)
            Rc = Rot.from_euler('ZXZ', np.asarray(cur if cur else orig[k], float), degrees=True)
            state = min(Rcand, key=lambda s: float((Rcand[s][k] * Rc.inv()).magnitude()))
            new = self._toggled_state(state, which)
            if new == "none":
                clear.append(t)
            elif new in cand:                          # tilt/both need S; skip if absent
                set_map[t] = tuple(cand[new][k])
        self.store.replace_flips(set_map, clear)       # one save+signal; staged kept

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
            self.flip_staged.difference_update(int(t) for t in tags)
            committed = [t for t in tags if self.store.is_flipped(int(t))]
            if committed:
                self.store.unflip(committed)       # -> changed -> _refresh_data
            else:
                self._draw_flip_markers(self.p1.y)
        else:
            self.store.remove(tags)

    def _restyle_all(self):
        for p in self.panels:
            p.restyle(self.store)
