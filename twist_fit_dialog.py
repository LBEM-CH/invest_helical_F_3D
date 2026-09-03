#!/usr/bin/env python3
"""
Auto-fit-twist dialog for Rohlex.

author: Wen-Lu Chung

A thin Qt front-end over twist_fit: runs the global fit, draws the concentration-
vs-twist curve with the consensus peak and each filament's own vote (outliers in
red), and applies the result to the shared live parameters. Two modes:

  * Fix rise : hold rise, fit twist = rate * rise (the robust default).
  * Free     : gradient-ascend (twist, rise) from a start point with a step size
               ("learn rate"), globally anchored so it can't stick on an alias.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

import twist_fit as tf
from plot_common import ModelParams, PLAIN_BTN_QSS, color_button_qss

_INLIER = (40, 160, 70)
_OUTLIER = (215, 40, 40)


class TwistFitDialog(QtWidgets.QDialog):
    """Modeless dialog: auto-fit the helix twist across all filaments."""

    def __init__(self, ds, params: ModelParams, parent=None):
        super().__init__(parent)
        self.ds = ds
        self.params = params
        self.result: tf.TwistFitResult | None = None
        self.setWindowTitle("auto-fit twist")
        self.resize(720, 560)

        lay = QtWidgets.QVBoxLayout(self)

        # --- controls --------------------------------------------------------
        ctl = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Fix rise (fit twist)", "Free (fit twist + rise)"])
        self.mode.currentIndexChanged.connect(self._mode_changed)
        ctl.addWidget(QtWidgets.QLabel("mode"))
        ctl.addWidget(self.mode)

        self.sp_rise = self._spin(0.1, 10000.0, 3, 0.05, params.rise, " Å/sub")
        self._riselbl = QtWidgets.QLabel("rise")
        ctl.addWidget(self._riselbl)
        ctl.addWidget(self.sp_rise)

        # free-only controls (start twist, learn rate, iterations)
        self.sp_twist0 = self._spin(tf.TWIST_MIN, tf.TWIST_MAX, 3, 0.1, params.twist, " °/sub")
        self.sp_lr = self._spin(0.001, 5.0, 3, 0.05, 0.10, "")
        self.sp_iters = QtWidgets.QSpinBox()
        self.sp_iters.setRange(10, 5000)
        self.sp_iters.setValue(300)
        self._free_w = []
        for lbl, w in (("start twist", self.sp_twist0), ("learn rate", self.sp_lr),
                       ("iters", self.sp_iters)):
            la = QtWidgets.QLabel(lbl)
            ctl.addWidget(la)
            ctl.addWidget(w)
            self._free_w += [la, w]

        self.cb_rot = QtWidgets.QCheckBox("fold rot-180 (C2)")
        self.cb_rot.setToolTip("Also fold a +180° (rot) split onto the main register. "
                               "Only valid when the fibril has C2 symmetry — otherwise a "
                               "+180 particle is genuinely wrong, not a symmetry. Off by "
                               "default. (Tilt-flip of the antiparallel group is always on.)")
        ctl.addWidget(self.cb_rot)

        ctl.addStretch(1)
        self.btn_run = QtWidgets.QPushButton("run fit")
        self.btn_run.setStyleSheet(color_button_qss((25, 110, 200)))
        self.btn_run.clicked.connect(self.run_fit)
        ctl.addWidget(self.btn_run)
        lay.addLayout(ctl)

        # --- plot ------------------------------------------------------------
        self.plot = pg.PlotWidget(background="w")
        self.plot.setLabel("bottom", "twist (deg / subunit)")
        self.plot.setLabel("left", "filament-vote density  /  linearity")
        self.plot.setYRange(0.0, 1.05, padding=0)
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.curve = self.plot.plot([], [], pen=pg.mkPen((25, 110, 200), width=2))
        self.dots = pg.ScatterPlotItem(size=9, pen=pg.mkPen("w", width=0.5))
        self.plot.addItem(self.dots)
        self.vpeak = pg.InfiniteLine(angle=90, movable=False,
                                     pen=pg.mkPen(_INLIER, width=2))
        self.vcur = pg.InfiniteLine(angle=90, movable=False,
                                    pen=pg.mkPen((120, 120, 120), width=1.2,
                                                 style=QtCore.Qt.PenStyle.DashLine))
        self.plot.addItem(self.vpeak, ignoreBounds=True)
        self.plot.addItem(self.vcur, ignoreBounds=True)
        lay.addWidget(self.plot, 1)

        # --- verdict banner + result text + actions -------------------------
        self.verdict = QtWidgets.QLabel("")
        self.verdict.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.verdict.setWordWrap(True)
        lay.addWidget(self.verdict)
        self.info = QtWidgets.QLabel("")
        self.info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.info.setWordWrap(True)
        lay.addWidget(self.info)

        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        self.btn_apply = QtWidgets.QPushButton("apply to twist / rise")
        self.btn_apply.setStyleSheet(color_button_qss((40, 160, 70)))
        self.btn_apply.clicked.connect(self._apply)
        self.btn_apply.setEnabled(False)
        self.btn_close = QtWidgets.QPushButton("close")
        self.btn_close.setStyleSheet(PLAIN_BTN_QSS)
        self.btn_close.clicked.connect(self.close)
        btns.addWidget(self.btn_apply)
        btns.addWidget(self.btn_close)
        lay.addLayout(btns)

        self._mode_changed()
        QtCore.QTimer.singleShot(0, self.run_fit)     # fit once on open

    @staticmethod
    def _spin(lo, hi, dec, step, val, suffix):
        sp = QtWidgets.QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(dec)
        sp.setSingleStep(step)
        sp.setValue(val)
        sp.setSuffix(suffix)
        sp.setKeyboardTracking(False)
        return sp

    def _mode_changed(self, *_):
        free = self.mode.currentIndex() == 1
        for w in self._free_w:
            w.setVisible(free)
        self._riselbl.setText("start rise" if free else "rise")

    # --- run + draw ----------------------------------------------------------
    def run_fit(self, *_):
        free = self.mode.currentIndex() == 1
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            rot = self.cb_rot.isChecked()
            if free:
                self.result = tf.fit_free(
                    self.ds.filaments, twist0=self.sp_twist0.value(),
                    rise0=self.sp_rise.value(), lr=self.sp_lr.value(),
                    iters=self.sp_iters.value(), rot_flip=rot)
            else:
                self.result = tf.fit_fixed_rise(self.ds.filaments,
                                                self.sp_rise.value(), rot_flip=rot)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._draw()

    def _draw(self):
        r = self.result
        if r is None or not np.isfinite(r.twist):
            self.info.setText("<b>no fittable filaments</b>")
            self.btn_apply.setEnabled(False)
            return
        self.curve.setData(r.grid, r.curve)
        own = np.array([f.twist for f in r.per_fil])
        peak = np.array([f.peakscore for f in r.per_fil])
        out = np.array([f.outlier for f in r.per_fil])
        self.dots.setData(
            x=own, y=peak,
            brush=[pg.mkBrush(*(_OUTLIER if o else _INLIER)) for o in out])
        self.vpeak.setValue(r.twist)
        self.vcur.setValue(self.params.twist)
        # focus the x-view on the peak but keep a little context
        lo = max(tf.TWIST_MIN, r.twist - 3.0)
        hi = min(tf.TWIST_MAX, r.twist + 3.0)
        self.plot.setXRange(lo, hi, padding=0)

        outtxt = (", ".join(str(o) for o in r.outliers)) if r.outliers else "none"
        note = ("" if r.mode == "fix_rise" else
                "<br>Free mode fits <i>rate</i> (twist/rise are degenerate from roll); "
                "rise is anchored to your start point.")
        st = r.stats
        # verdict banner: green when trustworthy, red when the fit is unreliable
        if st is not None:
            bg, fg = (("#e6f4ea", "#1b7a34") if st.reliable else ("#fdecea", "#b3261e"))
            self.verdict.setStyleSheet(
                f"background:{bg}; color:{fg}; border:1px solid {fg}; "
                "border-radius:4px; padding:5px 8px; font-weight:bold;")
            self.verdict.setText(st.verdict)
        self.info.setText(
            f"<b>twist = {r.twist:+.3f} ± {r.unc:.3f} °/subunit</b> "
            f"<span style='color:#888'>(95% CI on consensus)</span> &nbsp; "
            f"rate = {r.rate:+.5f} °/Å, rise = {r.rise:.3f} Å/sub<br>"
            + (f"<b>median linearity {st.peak_concentration:.2f}</b> "
               f"<span style='color:#888'>(1 = perfectly straight roll-vs-pos)</span>, "
               f"<b>{st.agree_frac*100:.0f}%</b> of filaments vote within "
               f"{tf.AGREE_TOL:.1f}° of consensus; vote 95% range "
               f"[{st.vote_lo:+.2f}, {st.vote_hi:+.2f}]°.<br>"
               f"<span style='color:#888'>dynamic local-slope fit, expanded to the longest "
               f"reliable baseline (~{st.median_span:.0f} Å); 2nd (antiparallel) group also "
               f"voted in {st.n_flipped} filament(s); rot-180 folding "
               f"{'ON' if r.rot_flip else 'off'}.</span><br>"
               if st is not None else "")
            + f"consensus of <b>{r.n_inlier}/{len(r.per_fil)}</b> filaments; "
            f"outliers excluded: <span style='color:#d72828'>{outtxt}</span>. "
            f"<span style='color:#888'>each dot = one filament's slope vote (y = its "
            f"linearity); red = outlier; grey dashed = current twist "
            f"{self.params.twist:+.3f}.</span>{note}")
        self.btn_apply.setEnabled(True)
        # A failed fit stays applyable (user may know better) but warns on click.
        self.btn_apply.setText("apply to twist / rise" if (st is None or st.reliable)
                               else "apply anyway…")

    def _apply(self):
        r = self.result
        if r is None or not np.isfinite(r.twist):
            return
        if r.stats is not None and not r.stats.reliable:
            ok = QtWidgets.QMessageBox.warning(
                self, "Apply an unreliable fit?",
                "This fit did not pass the reliability check:\n\n"
                f"{r.stats.verdict}\n\nApply twist "
                f"{r.twist:+.3f} / rise {r.rise:.3f} anyway?",
                QtWidgets.QMessageBox.StandardButton.Apply
                | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Cancel)
            if ok != QtWidgets.QMessageBox.StandardButton.Apply:
                return
        self.params.update(twist=round(r.twist, 3), rise=round(r.rise, 3))
        self.vcur.setValue(self.params.twist)
        self.info.setText(self.info.text() +
                          "<br><b style='color:#28a745'>applied.</b>")
