# invest_helical_F_3D

Interactive triage of helical-filament segments from a Dynamo **or** RELION 5
subtomogram-averaging project. It reconstructs the per-filament *helical roll check*
(subunit roll vs. real position along the filament axis, compared to the known screw) and
lets you mark bad segments for removal. Built to run over `ssh -XY` and stay responsive
(PyQt6 + pyqtgraph, raster renderer, no OpenGL).

## What it does

For one tomogram it loads either input — the two `refined_table_ref_00X_iteYYYY.tbl` tables
from a Dynamo `averages` folder, or a RELION 5 tomography particles `.star` — and for each
filament:

- fits the axis through the 3D coordinates (centroid = middle, SVD = head↔tail direction),
- projects each segment onto the axis → **real position** (Å, via the pixel size),
- reads the subunit **roll** about the axis,
- compares to the screw model `roll = (twist/rise)·pos + phase`.

**Inputs.** Dynamo: concatenates and sorts the ref tables by tag, groups filaments by
column 23, poses from the ZXZ-extrinsic Eulers. RELION: groups by `_rlnHelicalTubeID`,
coordinates from `_rlnCoordinateX/Y/Z`, and converts the poses
(`_rlnTomoSubtomogram*` combined with `_rlnAngle*`, or `_rlnAngle*Prior` when no refined
angles exist) back to the Dynamo convention so both paths share one fit. The pixel size is
read from the star optics block (override with `--pixelsize`).

A segment whose alignment drifted lands at its true (wrong) position and visibly leaves the
model line — that's what you mark.

### Overview window
- A scrollable grid of per-filament panels (roll vs position + dashed model).
- The tomogram **XY map** on the right.
- **Hover** a panel → that filament lights up on the map.
- **Click** a panel → open its detail window.
- Panels with marked segments turn **red**.

### Detail window (one filament)
- Three linked plots: **roll vs position**, **residual to model**, **XY map**.
- Point at a segment in any plot → it highlights in **all three**, with a readout
  (tag, position, roll, delta).
- **Mark for removal:** **left-drag** a rubber-band box to mark, **right-drag** to unmark,
  click *Select all in filament*, or click individual points to toggle. *Clear filament*
  unmarks the whole tube. Marks are red everywhere and saved immediately.

### Live helix controls
Both windows carry **twist / rise / pixel size** spin boxes (rise in Å, pixel size in
Å/px). Retune any of them and every model overlay refits instantly at real Å scale; the two
windows stay in sync. Use this to dial the screw onto the data before triaging.

### 3D view (per filament)
The **View 3D** button lives in the per-filament *detail* window (click a panel first), so
the 3D scene is always scoped to **one filament** — which keeps it fast, since only that
filament's density is ever built. It shows the filament's particles as points colored by
position (marked-for-removal in **red**, live), the 3D backbone, and a **pose triad** per
particle so you can see the orientations screw along the axis. If you pass a refined average
map (`--map run_class001.mrc`, or load one from the 3D window's "Load map…"), its
**isosurface** is placed and oriented at every particle of that filament, with a threshold
slider and a convention toggle (as-is vs inverse) to confirm the placement looks continuous.

The app **launches in 2D**; 3D is opened on demand from a filament. It needs OpenGL
(`PyOpenGL`), which the 2D windows avoid. Forwarded X (`ssh -XY`) usually has no usable GLX,
and Qt would otherwise **abort at startup** trying to initialize it — so over SSH the app
disables the xcb GLX integration (2D runs fine) and **greys out the View 3D button**. To use
3D, run locally, or force it with `--gl` (e.g. under VirtualGL / `vglrun`); `--no-gl` forces
it off. The OpenGL/mrc packages are imported only when you actually open the 3D view.

## Output / resume

Marked segments are written continuously to a plain text file (default
`<dir>/remove_list.txt`), **one id per line**, sorted — the Dynamo tag (column 1) for
Dynamo, or the `_rlnTomoParticleId` for RELION. This file is:

- the **resume list** — reloaded automatically on the next launch, and
- the **export** — feed it to your own `.tbl` / `.star` cleanup (the app does *not* edit
  the input).

For Dynamo it also writes `temp.tbl` (the sorted working rows for the chosen tomogram),
matching the notebook; disable with `--no-temp`.

## Install

Dependencies (both methods below install all of them):

| | packages | needed for |
|---|---|---|
| core | numpy, scipy, PyQt6, pyqtgraph | the 2D app |
| RELION | eulerangles | `--relion` (RELION→Dynamo pose conversion) |
| 3D view | PyOpenGL, mrcfile | the per-filament "View 3D" (OpenGL + reading the map) |

### Recommended: conda (self-contained, no root)

PyQt6 from PyPI needs the system library `libxcb-cursor0` at runtime (and the 3D view needs
system OpenGL), which are often missing on clusters where you can't `sudo apt install` them.
The conda-forge packages bundle `libxcb-cursor` and a working OpenGL/Mesa stack, so the
environment is fully self-contained — the right install path for a shared / no-sudo machine.

```bash
cd ~/LBEM/invest_helical_F_3D
conda env create -f environment.yml     # creates env "invest_helical" with everything
conda activate invest_helical
```

### Alternative: pip venv

Lean, but relies on system libraries being present (`libxcb-cursor0` for PyQt6 ≥ 6.5, and
`libGL`/GLX for the 3D view):

```bash
cd ~/LBEM/invest_helical_F_3D
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # core + eulerangles + PyOpenGL + mrcfile
```

If launching fails with *"xcb-cursor0 or libxcb-cursor0 is needed to load the Qt xcb
platform plugin"*, either `sudo apt install libxcb-cursor0` (one-time, all users), or — with
no root — drop the single library into your home dir and point `LD_LIBRARY_PATH` at it:

```bash
cd /tmp && apt-get download libxcb-cursor0 && dpkg -x libxcb-cursor0_*.deb x
mkdir -p ~/.local/lib && cp x/usr/lib/*/libxcb-cursor.so.0* ~/.local/lib/
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
```

The bundled `run.sh` sets that `LD_LIBRARY_PATH` for you and runs the venv Python. The 2D app
needs none of this — the 3D extras (PyOpenGL, mrcfile, OpenGL libs) are only touched when you
click **View 3D**, so a missing OpenGL stack never blocks 2D triage.

## Run

The GUI needs a display, so connect with X11 forwarding first:

```bash
ssh -XY user@cluster
conda activate invest_helical          # or: source .venv/bin/activate
cd ~/LBEM/invest_helical_F_3D

# Dynamo (auto-detected from the folder)
python invest_helical_F_3D.py \
  /mnt/.../dynamo_project_b4/abp_align_eo/results/ite_0004/averages \
  --tomo 1 --twist -1.4 --rise 4.75 --pixelsize 7.92

# RELION 5 (auto-detected from the .star extension)
python invest_helical_F_3D.py \
  /mnt/.../warp/relion_b4_clean/particles_new_relion5.star \
  --twist -1.4 --rise 4.75 --out ~/relion_remove_list.txt
```

With the pip venv (no conda) on this cluster, swap the launch line for `./run.sh <path>
[options]`, which adds the local `libxcb-cursor` shim automatically.

The format is inferred from `path` (`.star` → RELION, folder → Dynamo); force it with
`--dynamo` / `--relion`.

Options: `--tomo` (Dynamo col-20 id or RELION `_rlnTomoName`; else only/first or a chooser),
`--twist` (°/subunit), `--rise` (Å/subunit), `--pixelsize` (Å/px; RELION default from the
star optics, Dynamo default 7.92), `--map` (reference average `.mrc`/`.em` for the 3D
density overlay), `--cols` (overview columns), `--out` (remove-list path; point it somewhere
writable if the input sits on read-only storage), `--no-temp`. Twist, rise and pixel size
are also live in the GUI.

Requires Python 3.9+. `--relion` additionally needs the `eulerangles` package (included in
both `environment.yml` and `requirements.txt`).
