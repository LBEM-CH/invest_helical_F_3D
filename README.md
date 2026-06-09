# invest_helical_F_3D

Interactive triage of helical-filament segments from a Dynamo subtomogram-averaging
project. It reconstructs the per-filament *helical roll check* (subunit roll vs. real
position along the filament axis, compared to the known screw) and lets you mark bad
segments for removal. Built to run over `ssh -XY` and stay responsive (PyQt6 + pyqtgraph,
raster renderer, no OpenGL).

## What it does

For one tomogram it loads the two `refined_table_ref_00X_iteYYYY.tbl` tables from a Dynamo
`averages` folder, concatenates and sorts them by tag (like the notebook), and for each
filament:

- fits the axis through the 3D coordinates (centroid = middle, SVD = head↔tail direction),
- projects each segment onto the axis → **real position** (px),
- reads the subunit **roll** about the axis,
- compares to the screw model `roll = (twist/rise)·pos + phase`.

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
- **Mark for removal:** toggle *Select mode* and drag a rubber-band box, click *Select all
  in filament*, or click individual points. *Clear filament* unmarks. Marks are red
  everywhere and saved immediately.

## Output / resume

Marked segments are written continuously to a plain text file (default
`<folder>/remove_list.txt`), **one Dynamo tag (column 1) per line**, sorted. This file is:

- the **resume list** — reloaded automatically on the next launch, and
- the **export** — feed it to your own `.tbl` cleanup (the app does *not* edit the tables).

It also writes `temp.tbl` (the sorted working rows for the chosen tomogram), matching the
notebook; disable with `--no-temp`.

## Install

```bash
cd ~/LBEM/invest_helical_F_3D
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
# over ssh with X11 forwarding:
ssh -XY user@cluster
source ~/LBEM/invest_helical_F_3D/.venv/bin/activate

python invest_helical_F_3D.py \
  /mnt/.../dynamo_project_b4/abp_align_eo/results/ite_0004/averages \
  --tomo 1 --twist -1.4 --rise 0.6
```

Options: `--tomo` (id, else only/first or a chooser), `--twist`/`--rise` (screw),
`--cols` (overview columns), `--out` (remove-list path), `--no-temp`.

Requires Python 3.9+.
