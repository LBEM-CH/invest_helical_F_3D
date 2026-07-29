# How a segment's pose becomes a point on the roll plot

*Reference for `invest_helical_F_3D` — the per-filament "helical roll check".*

This document explains, stage by stage, how each subtomogram segment is turned into
one dot on the **roll-vs-position** plot: how the pose (Euler angles) is interpreted,
how the filament axis is found from the coordinates, and how the along-axis rotation
(the "roll") is measured and compared to the reference helical screw. Every claim
here is backed by a ground-truth synthetic test (§7).

---

## 1. What we are measuring, in one line

A helix is a screw: as you walk along the filament axis by a distance `Δz`, each
successive subunit is rotated about that axis by

```
Δφ  =  RATE · Δz ,        RATE = TWIST / RISE      (deg per Ångström)
```

where **TWIST** is deg/subunit and **RISE** is Å/subunit. So if we can measure, for
every segment, (a) its **position** along the axis and (b) its **azimuthal rotation
(roll)** about that axis, a good filament produces a straight line of slope `RATE` on
a roll-vs-position plot. Deviations from that line are what the tool triages.

Two facts make this robust:

- **Position and roll are pose-only** (independent of twist/rise/pixel size), so they
  are computed once at load in `fit_pose`.
- Only the **overlay** (`φ₀`, the residual `δ`) depends on RATE, and is recomputed
  live when the user retunes twist/rise/pixel size (`fit_model`, `apply_model`).

Units: positions are carried in **Ångström** = `pixel_coordinate × pixel_size`, so
RATE is deg/Å and the whole fit is at physical scale regardless of binning.

---

## 2. Inputs and the common internal representation

Two file types are funnelled into the **same** internal representation — a set of
per-filament `(xyz, eulers)` arrays in the **Dynamo ZXZ-extrinsic** convention:

| input | pose source | conversion |
|---|---|---|
| Dynamo `.tbl` | cols 7–9 (`tdrot, tilt, narot`) | used directly (already Dynamo ZXZ) |
| RELION 5 `.star` | `rlnTomoSubtomogram*` + `rlnAngle*` | converted to Dynamo (§3.3) |

`xyz` comes from `rlnCoordinateX/Y/Z` (RELION) or table cols 24–26 (Dynamo) — the
**picked** coordinates. Because the axis and position are read from these
coordinates, a segment whose alignment drifted lands at its true (possibly wrong)
position and visibly leaves the model line.

---

## 3. Stage A — from stored angles to a rotation object

**Goal:** turn the stored Euler triple into a scipy `Rotation` `D` that maps the
*reference frame* into the *tomogram frame*, i.e. `v_tomo = D · v_reference`.

### 3.1 The Dynamo convention

`eulerangles` defines the two conventions we touch as:

```
dynamo : axes = zxz , intrinsic = False  (EXTRINSIC), active = False
warp   : axes = ZYZ , intrinsic = True   (INTRINSIC)
```

`active = False` means the plain matrix `R.from_euler('zxz', angles)` is the
**passive** (reference-frame) rotation. For plotting a particle we want the
**active** rotation (where the particle actually points), which is its **inverse**.

### 3.2 The ArtiaX transform *is* that inverse

The code builds every rotation through `dynamo_rotation` (helix_geom.py:72):

```python
def _artiax_eulers(eulers):            # (tdrot, tilt, narot) ->
    td, ti, na = eulers.T
    return [-na - 180,  ti,  -td + 180]      # read as extrinsic 'zxz'

def dynamo_rotation(eulers):
    return Rot.from_euler('zxz', _artiax_eulers(eulers), degrees=True)
```

This triple manipulation was borrowed from an **ArtiaX display-bug workaround** (a
fix for how ArtiaX/ChimeraX visualises Dynamo tables: swap cols 7↔9, negate and
±180). Algebraically it is **exactly a matrix transpose / inverse**:

```
zxz[ -na-180, ti, -td+180 ]  ==  ( zxz[ td, ti, na ] )⁻¹
```

**Verified:** `max| dynamo_rotation(e) − R.from_euler('zxz', e).inv() | = 1.3e-15`.

So the "ArtiaX hand-flip" and the "passive→active inverse" are the *same operation*
arrived at two ways. Practical consequence: it makes a genuinely right-handed helix
read as **+1.4** (not the mirrored −1.4). It is **not** a handedness/mirror flip —
every mirror form (`diag(1,1,−1)·M·diag(1,1,−1)`, etc.) is ≈ 2.0 away; only the
transpose matches to 0. Keep this straight when reasoning about twist sign: adding a
real mirror here would silently flip the sign.

The inverse, `rotation_to_dynamo_eulers` (helix_geom.py:79), writes a rotation back
to raw Dynamo angles so flipped poses can be exported to a `.tbl`. Round-trip:
`dynamo_rotation(rotation_to_dynamo_eulers(R)) == R` to **8.9e-14°** — the export
path is loss-less for flipped *and* untouched segments.

### 3.3 RELION → Dynamo (the reverse of the notebook pipeline)

RELION stores the pose split across two columns. The final placement of a refined
particle is `rlnTomoSubtomogram* × rlnAngle*`; the app inverts the notebook's
forward pipeline exactly (`relion_to_dynamo_eulers`, relion_star.py:188):

```python
r_subtomo = R.from_euler('zyz', rlnTomoSubtomogram*)   # lowercase 'zyz'
r_prior   = R.from_euler('zyz', align)                 # rlnAngle*, see below
r_aligned = r_subtomo * r_prior                        # the full pose
warp      = r_aligned.as_euler('zyz')
dynamo    = convert_eulers(warp, 'warp' -> 'dynamo')   # eulerangles package
```

- **`align` = `rlnAngleRot/Tilt/Psi` when present, else `rlnAngle*Prior`**
  (`_alignment_eulers`, relion_star.py:163). Semantics (authoritative):
  `rlnAngle*` is the refined answer; `rlnAngle*Prior` is only the **search centre**
  of the iteration, used solely as the fallback for `run_it000` / pre-refinement
  stars that carry no `rlnAngle*`.
- The lowercase `'zyz'` spelling paired with the `r_subtomo * r_prior` order is
  algebraically identical to the textbook-correct `'ZYZ'` + reverse order; it is the
  **only** convention/order combination that reproduces the validated MSA result
  (+1.28), so it is correct but *fragile* — do not edit spelling or order in
  isolation.
- **Exactness proof:** running the notebook's forward step
  (`r_subtomo = r_aligned · r_prior⁻¹`) and then this reverse recovers the original
  angles to **2.5e-14°**. The two are literal inverses.

The refined shifts `rlnOrigin*Angst` are **deliberately not applied** to the
coordinates: they live in the subtomogram frame, so subtracting them from tomogram
coordinates is invalid, and doing so measurably degrades a known-truth control. Their
along-axis component is tiny anyway (~1 % of the filament span).

---

## 4. Stage B — the filament axis from the coordinates ("major tilt")

`axis_and_pos` (helix_geom.py:87) fits the axis by SVD and projects onto it:

```python
c   = xyz.mean(0)                 # centroid: mean of ALL segments, not the middle one
n   = svd(xyz - c).Vt[0]          # principal axis (unit) = head<->tail direction
pos = (xyz - c) @ n               # signed arc-length along the axis, 0 at centre
```

Key properties:

- The axis is derived **only from coordinates** — completely independent of the
  poses. This is what lets the roll survive a tilted or mis-aligned pose (§7).
- The **sign of `n` is arbitrary** (SVD gives a direction, not an orientation). This
  does **not** bias the twist: flipping `n` negates `pos` **and** transforms the roll
  in a compensating way, so the fitted slope is unchanged (verified: reversing a
  filament's coordinate order flips the axis, `n·n' = −1`, but the recovered twist
  stays `+1.400`). Where a *consistent* orientation is needed across filaments
  (e.g. the tilt colouring), `axis_tilt(..., orient=True)` points `n` toward the
  majority pose direction.

`pos` here is in pixels; the caller multiplies by `pixel_size` to get Ångström
(`Dataset.apply_model`, dynamo_table.py).

---

## 5. Stage C — the along-axis rotation (the "roll")

`roll_about_axis` (helix_geom.py:101) measures each pose's azimuth about the axis:

```python
e1 = normalize( n × ẑ )           # (falls back to n × ŷ if n ∥ ẑ)
e2 = n × e1                        # (e1, e2, n) is right-handed
v  = D · [1, 0, 0]                # the particle x-axis, carried into the tomo frame
roll = atan2( v·e2 , v·e1 )       # azimuth in the plane ⊥ n, in (−180, 180]
```

Why the **particle x-axis** as the reference vector: for a helical subunit the pose
**z-axis points along the filament**, so the **x-axis is ⊥ to the axis** and its
azimuth about `n` *is* the screw phase. As the screw advances, `v` rotates about `n`
and `roll` advances at exactly `RATE`.

- The frame `(e1, e2, n)` is right-handed (`e1 × e2 = n`), so the sign of `roll`
  is consistent.
- If the pose is **tilted** off the axis, `v` gains a small component along `n`; the
  azimuth of its in-plane projection still advances at `RATE` (verified stable to a
  60° pose tilt, §7). The construction only fails if `v` were *parallel* to `n`,
  which never happens for real helical particles.

`fit_pose` (helix_geom.py:117) assembles the pose-only outputs, ordered head→tail:

```python
n, pos   = axis_and_pos(xyz)
order    = argsort(pos)                      # head -> tail
D        = dynamo_rotation(eulers[order])
phi      = roll_about_axis(D, n)             # measured roll (deg)
polarity = sign( D.z_axis · n )              # +1/-1: which way each subunit points
```

`polarity` distinguishes the two head↔tail orientations; a minority with the
opposite sign is a **polarity (perpendicular-dyad) flip**, handled separately by the
flipped-register overlay — it is a pointing difference, not a roll difference.

---

## 6. The reference screw and the overlay

Given RATE, `fit_model` / `dominant_phase` fit the phase of the line:

```
φ₀    = circular-mean of (φ − RATE·pos)      # or centre of the densest cluster
model = RATE·pos + φ₀                         # the dashed reference line
δ     = wrap(φ − model)  ∈ (−180, 180]        # residual to the screw
```

`φ₀` is a **free per-filament nuisance** (each filament has its own phase), which is
why the plot is unbiased with respect to any reference segment. `dominant_phase`
uses the densest residual cluster rather than a plain mean so a `+180` (rot) split or
an outlier sub-population cannot drag the line off.

The dashed model line is drawn by `model_line`, which wraps to (−180, 180] and
breaks the line with `NaN`s at each ±180 seam so no vertical strokes are drawn.

---

## 7. Verification against ground truth

A synthetic helix of **known** twist is built (particle z-axis along a chosen filament
axis, x-axis azimuth `= TWIST·i + phase`), converted to Dynamo angles with
`rotation_to_dynamo_eulers`, and pushed through the *real* `fit_pose`. The recovered
twist is `slope(φ vs pos) × RISE`.

| test | expected | recovered |
|---|---|---|
| 12 cases: axis ∈ {ẑ, tilted, …} × twist ∈ {+1.4, −1.4, −0.96, +0.7} | input | **input, to 3 dp** |
| filament axis recovery (`n · n_true`) | 1 | **1.0000** |
| reverse coordinate order (SVD axis flips, `n·n' = −1`) | same sign | **+1.400 → +1.400** |
| global pose tilt 0 → 60° | +1.4 | **+1.400 throughout** |
| roll noise 5 / 10 / 20 / 30° | +1.4 | +1.394 / +1.389 / +1.378 / +1.361 |
| extraction spacing 1 / 0.5 / 0.25 subunit | +1.4 | **exact** |

Interpretation: the geometry (pose convention, axis from coordinates, along-axis
roll) is mathematically exact and robust to axis-sign ambiguity, pose tilt, noise
(mild attenuation only), and oversampling.

---

## 8. Scope and known limitations

The geometry above is correct wherever the segment poses carry the *accumulated*
helical phase:

- **Dynamo tables** — valid.
- **RELION pre-refinement / `run_it000` stars** (no helical symmetry applied) —
  valid; recovers ~90 % of the reference twist.

It is **not** a twist estimator for a RELION `--helix` refinement with
`do_apply_helical_symmetry = Yes`. There the refined `rlnAngleRot` is only defined
**modulo the screw operator** (a particle at `(φ, z)` is equivalent to one at
`(φ + TWIST, z + RISE)`), so the offset search folds the accumulated phase and the
measured slope is attenuated (~40–50 % of the reference) — this is regression
dilution from the symmetry, **not** an error in the pipeline, and **not** the
filament failing to twist. For such jobs the twist is already reported by RELION's
own symmetry search as `rlnHelicalTwist` / `rlnHelicalRise` in
`run_it*_model.star`. A separate limitation: filaments spanning much less than one
helical period (`360/RATE`) under-determine the slope regardless of convention.

---

## 9. File / function map

| stage | function | file:line |
|---|---|---|
| Dynamo angles → rotation | `dynamo_rotation` / `_artiax_eulers` | helix_geom.py:65,72 |
| rotation → Dynamo angles | `rotation_to_dynamo_eulers` | helix_geom.py:79 |
| RELION → Dynamo angles | `relion_to_dynamo_eulers` | relion_star.py:188 |
| RELION alignment column choice | `_alignment_eulers` | relion_star.py:163 |
| axis + position from coords | `axis_and_pos` | helix_geom.py:87 |
| along-axis roll | `roll_about_axis` | helix_geom.py:101 |
| pose-only assembly | `fit_pose` | helix_geom.py:117 |
| pose z-axis vs filament axis | `axis_tilt` | helix_geom.py:156 |
| screw phase + residual | `fit_model` / `dominant_phase` | helix_geom.py:175,204 |
| Å scaling + model refresh | `Filament.apply_model` | dynamo_table.py |
