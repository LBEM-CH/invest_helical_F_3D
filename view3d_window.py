#!/usr/bin/env python3
"""
In-app 3D view for invest_helical_F_3D.

author: Wen-Lu Chung

A self-contained OpenGL scene (pyqtgraph.opengl) showing the tomogram in 3D:

  * every particle as a point, colored by filament, with marked-for-removal
    points drawn red and updated live as you triage;
  * each filament's backbone as a 3D line;
  * for the selected filament, a pose triad per particle (carried x/y/z axes of
    its rotation) so you can see how the subunits are oriented -- the x-axis
    visibly screws around the filament, matching the 2D roll check;
  * optionally the refined average map (an .mrc/.em isosurface) placed and
    oriented at each particle of the selected filament.

Needs OpenGL (PyOpenGL), which the 2D windows deliberately avoid -- so this is
imported lazily, only when "View 3D" is clicked. It is crisp locally but may be
sluggish over plain `ssh -XY` (VirtualGL helps).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6 import QtCore, QtWidgets
from scipy.spatial.transform import Rotation as Rot

from plot_common import viridis_rgba
from helix_geom import dynamo_rotation

_HORIZONTAL = QtCore.Qt.Orientation.Horizontal

# axis -> RGBA for the pose triad (x red, y green, z blue)
_AXIS_RGBA = np.array([[1, 0, 0, 1.], [0, 0.8, 0, 1.], [0.2, 0.4, 1, 1.]])
_MARK_RGBA = (0.86, 0.12, 0.12, 1.0)
# placed-density surface color (medium blue-grey: light but with enough body to
# read against the white background under the built-in 'shaded' lighting)
_MESH_RGBA = (0.55, 0.62, 0.78, 1.0)


def _rotations(eulers: np.ndarray, invert: bool) -> Rot:
    # Same convention as the roll measurement: raw Dynamo angles through the
    # ArtiaX/ChimeraX placement transform (+ extrinsic zxz), so the triads match
    # how ArtiaX places the density. The toggle still exposes the transpose escape.
    R = dynamo_rotation(np.asarray(eulers, float))
    return R.inv() if invert else R


def pose_triads(xyz: np.ndarray, eulers: np.ndarray, length: float,
                invert: bool = False):
    """Line-segment endpoints + colors for a per-particle orientation triad.

    Returns (pts, colors) shaped (6N, 3) and (6N, 4): for each particle three
    segments center->center+length*(carried x/y/z). Feed to GLLinePlotItem with
    mode="lines".
    """
    xyz = np.asarray(xyz, float)
    R = _rotations(eulers, invert)
    axes = np.stack([R.apply([1., 0, 0]), R.apply([0, 1., 0]), R.apply([0, 0, 1.])], 1)
    # axes: (N, 3 ax-dirs, 3 xyz)
    pts = np.empty((len(xyz), 3, 2, 3))
    pts[:, :, 0, :] = xyz[:, None, :]                       # segment start = center
    pts[:, :, 1, :] = xyz[:, None, :] + length * axes       # end = center + L*axis
    cols = np.broadcast_to(_AXIS_RGBA[None, :, None, :], (len(xyz), 3, 2, 4))
    return pts.reshape(-1, 3), cols.reshape(-1, 4)


def isosurface(volume: np.ndarray, level: float):
    """Marching-cubes isosurface (via pyqtgraph), centered on the box middle.

    pyqtgraph returns vertices in array-index order (axis0, axis1, axis2), which
    for an mrc/em volume is (z, y, x). We reverse them to physical (x, y, z) so
    the pose rotation in place_meshes -- which is in tomogram x/y/z -- acts on the
    right axes. Without this the map's z (helical) axis would be placed as if it
    were x, so the density points along the wrong axis no matter the Euler pose.
    Face winding is reversed as well to keep the surface normals pointing outward
    after the axis swap.

    Returns (verts, faces) with verts in voxel units shifted so the box center
    is the origin -- ready to rotate about the particle center.
    """
    vol = np.ascontiguousarray(volume, dtype="float32")
    verts, faces = pg.isosurface(vol, float(level))
    verts = verts[:, ::-1]                                  # (z,y,x) -> (x,y,z)
    verts = verts - (np.array(vol.shape[::-1]) - 1) / 2.0   # center; shape -> (nx,ny,nz)
    faces = faces[:, ::-1]                                  # outward winding after swap
    return np.ascontiguousarray(verts), np.ascontiguousarray(faces)


def place_meshes(verts: np.ndarray, faces: np.ndarray, xyz: np.ndarray,
                 eulers: np.ndarray, scale: float, invert: bool = False):
    """Replicate one mesh at every particle, rotated by its pose.

    Returns merged (Vtot, 3) vertices and (Ftot, 3) faces. verts are in voxel
    units (centered); scale converts voxels -> tomogram px.
    """
    xyz = np.asarray(xyz, float)
    M = _rotations(eulers, invert).as_matrix()              # (N, 3, 3)
    # rotate the shared mesh into each pose, scale to px, translate to center
    placed = np.einsum("nij,vj->nvi", M, verts * scale) + xyz[:, None, :]  # (N, V, 3)
    n, v = placed.shape[:2]
    all_verts = placed.reshape(-1, 3)
    all_faces = (faces[None, :, :] + (np.arange(n)[:, None, None] * v)).reshape(-1, 3)
    return all_verts, all_faces


def _face_budget(n_particles: int, n_faces: int, cap: int = 1_500_000) -> int:
    """Particle stride so the merged mesh stays under `cap` faces."""
    if n_particles * n_faces <= cap:
        return 1
    return int(np.ceil(n_particles * n_faces / cap))


class View3DWindow(QtWidgets.QMainWindow):
    """3D scene for ONE filament, opened from its detail window.

    Scoped to a single filament so only that filament's density is ever built --
    fast, and exactly the unit you are triaging. Points are colored by position
    (viridis), marked-for-removal points red and live; pose triads and an optional
    placed density isosurface show the orientations.
    """

    def __init__(self, fil, ds, store, volume=None, map_voxel=None, parent=None):
        super().__init__(parent)
        self.fil = fil
        self.ds = ds
        self.store = store
        self.volume = volume                       # (nz,ny,nx) float or None
        # voxel size of the map (A/px); default to the tomogram pixel size so the
        # density lands at 1:1 when the .mrc header has no usable voxel size.
        self.map_voxel = float(map_voxel) if map_voxel else float(ds.pixelsize)
        self.scale = 1.0                            # voxel -> tomogram-px (set below)
        self._iso = None                            # cached (verts, faces) for current level
        self.setWindowTitle(f"3D — filament {fil.fid}  (n={fil.n})")
        self.resize(1100, 820)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        lay = QtWidgets.QHBoxLayout(central)
        lay.addWidget(self._build_controls())       # creates self.sp_voxel
        self._recompute_scale()
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor("w")
        lay.addWidget(self.view, 1)

        self.scatter = gl.GLScatterPlotItem(pos=fil.xyz, size=9.0, pxMode=True)
        self.view.addItem(self.scatter)
        if fil.n >= 2:
            self.view.addItem(gl.GLLinePlotItem(pos=fil.xyz, color=(0.5, 0.5, 0.5, 1.0),
                                                width=1.0, antialias=True))
        self.glyphs = gl.GLLinePlotItem(mode="lines", width=2.0, antialias=True)
        self.view.addItem(self.glyphs)
        # placed reference density: built-in 'shaded' lighting, medium blue-grey.
        self.mesh = gl.GLMeshItem(smooth=True, shader="shaded",
                                  color=_MESH_RGBA, glOptions="opaque")
        self.view.addItem(self.mesh)
        # orange marker for the segment currently pointed at in the 2D detail window
        self.hover = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), size=18.0,
                                          color=(1.0, 0.5, 0.0, 1.0), pxMode=True)
        self.view.addItem(self.hover)

        self._center_camera()
        self.store.changed.connect(self._recolor)
        self._recolor()
        self._refresh()

    def _build_controls(self):
        w = QtWidgets.QWidget()
        w.setFixedWidth(240)
        v = QtWidgets.QVBoxLayout(w)
        v.addWidget(QtWidgets.QLabel(f"<b>filament {self.fil.fid}</b>  (n={self.fil.n})"))

        self.btn_home = QtWidgets.QPushButton("Reset view (home)")
        self.btn_home.setToolTip("recenter and reframe the filament (undo zoom/rotate)")
        self.btn_home.clicked.connect(self._center_camera)
        v.addWidget(self.btn_home)

        self.chk_pose = QtWidgets.QCheckBox("Show pose triads")
        self.chk_pose.setChecked(True)
        self.chk_pose.toggled.connect(self._refresh)
        v.addWidget(self.chk_pose)

        v.addWidget(QtWidgets.QLabel("triad length (px)"))
        self.sl_len = QtWidgets.QSlider(_HORIZONTAL)
        self.sl_len.setRange(2, 60)
        self.sl_len.setValue(16)
        self.sl_len.valueChanged.connect(self._refresh)
        v.addWidget(self.sl_len)

        self.btn_map = QtWidgets.QPushButton("Load map (.mrc/.em)…")
        self.btn_map.clicked.connect(self._load_map)
        v.addWidget(self.btn_map)

        v.addWidget(QtWidgets.QLabel("map voxel size (Å/px)"))
        self.sp_voxel = QtWidgets.QDoubleSpinBox()
        self.sp_voxel.setRange(0.001, 1000.0)
        self.sp_voxel.setDecimals(3)
        self.sp_voxel.setSingleStep(0.01)
        self.sp_voxel.setKeyboardTracking(False)
        self.sp_voxel.setValue(self.map_voxel)
        self.sp_voxel.valueChanged.connect(self._on_voxel_changed)
        v.addWidget(self.sp_voxel)

        self.chk_density = QtWidgets.QCheckBox("Show density")
        self.chk_density.setEnabled(self.volume is not None)
        self.chk_density.toggled.connect(self._refresh)
        v.addWidget(self.chk_density)

        v.addWidget(QtWidgets.QLabel("iso threshold (σ above mean)"))
        self.sl_iso = QtWidgets.QSlider(_HORIZONTAL)
        self.sl_iso.setRange(0, 80)                 # /10 -> 0.0 .. 8.0 sigma
        self.sl_iso.setValue(20)
        self.sl_iso.setEnabled(self.volume is not None)
        self.sl_iso.valueChanged.connect(self._on_iso_changed)
        v.addWidget(self.sl_iso)

        v.addWidget(QtWidgets.QLabel("placement convention"))
        self.cb_conv = QtWidgets.QComboBox()
        # With the volume axes decoded correctly (see isosurface), "as-is (D)" is
        # the placement that matches ChimeraX: the map's z (helical) axis lands on
        # the filament. "inverse (Dᵀ)" is the transpose, kept as an escape hatch.
        self.cb_conv.addItems(["pose: as-is (D)", "pose: inverse (Dᵀ)"])
        self.cb_conv.setCurrentIndex(0)
        self.cb_conv.currentIndexChanged.connect(self._refresh)
        v.addWidget(self.cb_conv)

        self.lbl_info = QtWidgets.QLabel("")
        self.lbl_info.setWordWrap(True)
        v.addWidget(self.lbl_info)
        v.addStretch(1)
        return w

    # --- scene ---------------------------------------------------------------
    def _center_camera(self):
        """Frame the whole filament from a fixed default angle (also the Home reset)."""
        c = self.fil.xyz.mean(0)
        span = float(np.ptp(self.fil.xyz, axis=0).max())
        self.view.opts["center"] = pg.Qt.QtGui.QVector3D(*c)
        self.view.setCameraPosition(distance=max(span * 1.6, 10.0),
                                    azimuth=45, elevation=30)

    def _recolor(self):
        marked = np.array([self.store.is_marked(t) for t in self.fil.tags], bool)
        cols = viridis_rgba(self.fil.pos_px)
        cols[marked] = _MARK_RGBA
        self.scatter.setData(color=cols)

    def _invert(self):
        # index 0 = as-is D (default, ChimeraX-matching); index 1 = inverse Dᵀ
        return self.cb_conv.currentIndex() == 1

    def _recompute_scale(self):
        """voxel -> tomogram-px factor from the (possibly user-overridden) voxel size."""
        vx = self.sp_voxel.value()
        self.scale = (vx / self.ds.pixelsize) if self.ds.pixelsize else 1.0

    def _on_voxel_changed(self):
        self.map_voxel = self.sp_voxel.value()
        self._recompute_scale()
        self._refresh()

    def highlight_tags(self, tags):
        """Light up the given segment tags (orange) -- the live 2D->3D hover link."""
        if tags is None or len(tags) == 0:
            self.hover.setData(pos=np.zeros((0, 3)))
            return
        mask = np.isin(self.fil.tags, np.asarray(tags))
        self.hover.setData(pos=self.fil.xyz[mask] if mask.any() else np.zeros((0, 3)))

    def _load_map(self):
        # Freeze GL repaints while the modal file dialog spins its own event loop;
        # on macOS a repaint there hits a context whose shader programs aren't
        # valid and PyOpenGL spams GLError 1281 (non-fatal, but noisy).
        self.view.setUpdatesEnabled(False)
        try:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Reference map", "",
                "Density maps (*.mrc *.map *.rec *.em);;All files (*)")
        finally:
            self.view.setUpdatesEnabled(True)
        if not path:
            return
        from volume_io import read_volume
        try:
            vol, voxel = read_volume(path)
        except Exception as e:                      # noqa: BLE001 - surface any read error
            QtWidgets.QMessageBox.warning(self, "Load map", f"could not read map:\n{e}")
            return
        self.volume = vol
        self.map_voxel = float(voxel) if voxel else float(self.ds.pixelsize)
        self.sp_voxel.blockSignals(True)
        self.sp_voxel.setValue(self.map_voxel)      # reflect the header voxel in the box
        self.sp_voxel.blockSignals(False)
        self._recompute_scale()
        self._iso = None
        self.chk_density.setEnabled(True)
        self.sl_iso.setEnabled(True)
        self.chk_density.setChecked(True)
        self._refresh()

    def _on_iso_changed(self):
        self._iso = None                            # threshold changed -> recompute
        self._refresh()

    def closeEvent(self, ev):
        """Tear the GL scene down before the C++ widget dies.

        On macOS a queued paintGL can fire while the window is closing, after the
        GL context is gone, and PyOpenGL then raises GLError 1281 (invalid
        program) from glUseProgram. Disabling updates, dropping the live store
        connection, and removing the items first avoids that stray repaint.
        """
        try:
            self.store.changed.disconnect(self._recolor)
        except (TypeError, RuntimeError):
            pass
        try:
            self.view.setUpdatesEnabled(False)
            for it in list(self.view.items):
                self.view.removeItem(it)
        except Exception:                           # noqa: BLE001 - teardown is best-effort
            pass
        super().closeEvent(ev)

    def _refresh(self):
        f = self.fil
        if self.chk_pose.isChecked() and f.n:
            pts, cols = pose_triads(f.xyz, f.eulers, float(self.sl_len.value()),
                                    self._invert())
            self.glyphs.setData(pos=pts, color=cols)
        else:
            self.glyphs.setData(pos=np.zeros((0, 3)))
        info = f"{f.n} particles"
        if self.chk_density.isChecked() and self.volume is not None and f.n:
            if self._iso is None:
                lvl = self.volume.mean() + (self.sl_iso.value() / 10.0) * self.volume.std()
                self._iso = isosurface(self.volume, lvl)
            verts, faces = self._iso
            stride = _face_budget(f.n, len(faces))
            xyz, eul = f.xyz[::stride], f.eulers[::stride]
            mv, mf = place_meshes(verts, faces, xyz, eul, self.scale, self._invert())
            self.mesh.setMeshData(vertexes=mv, faces=mf)
            self.mesh.setVisible(True)
            info += (f"  |  density: {len(xyz)} copies (stride {stride}), {len(mf):,} faces"
                     f"  |  voxel {self.map_voxel:.3g} Å/px → scale {self.scale:.3g}")
        else:
            self.mesh.setVisible(False)
        self.lbl_info.setText(info)
