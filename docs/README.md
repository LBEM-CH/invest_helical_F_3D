# Measuring the Helical Screw from Subtomogram Orientations

*Methods and results in paper form. Equations are numbered for reference.*

---

## 1. What we measure

A helix can be described as a screw. Moving by one subunit along the filament corresponds to an axial step, the **rise** h (ångström), and a rotation, the **twist** τ (degrees).

The rotation per unit length is therefore:

**g = τ / h**  [degrees per ångström]  (1)

For each segment i, we measure two quantities:

* its position along the filament, **zᵢ**
* its rotation about the filament axis, **φᵢ**

For a correctly aligned filament, these quantities follow a straight line:

**φᵢ = g zᵢ + φ₀ (mod 360°)**  (2)

The helix determines the slope **g**.

The offset **φ₀** is free and is fitted separately for each filament, because each filament may start at a different phase of the screw.

Importantly, **zᵢ** is determined from the particle coordinates, not from the orientation. A wrongly aligned segment therefore remains at its true physical position but moves away from the expected screw line.

Angles are periodic. For example, 350° and 10° are only 20° apart, not 340°.

We therefore use:

**wrap(α)**

to add or subtract complete 360° turns until the angle lies between −180° and +180°.

In other words, **wrap(α)** gives the shortest angular distance.

---

## 2. Reading the data

Both supported project types are ultimately reduced to the same representation: for each filament, a list of particle coordinates and a corresponding list of orientations.

### Dynamo

Use the last refinement iteration containing a non-empty table.

If the refinement contains multiple references, concatenate all reference tables from that iteration so that all classes are included.

Sort the particles by particle tag, select one tomogram, and group the particles into filaments using the filament column.

The particle position is:

**picked coordinate + refined shift**

from the same Dynamo table.

The orientation and shift were refined together, so the refined orientation belongs to the shifted position.

### RELION

Select the particles belonging to one tomogram and group them by helical tube.

The final orientation is constructed from the two stored orientation components described in Section 3.

Do **not** apply the refined RELION particle shifts to the tomogram coordinates. These shifts are defined within the extracted subtomogram rather than in the coordinate system of the original tomogram.

Finally, positions are converted from voxels to ångström using the pixel size **s**.

This makes the measured slope a physical quantity and removes any dependence on binning.

---

## 3. Turning the stored angles into an orientation

Dynamo stores angles describing the rotation of the reference frame.

Here, however, we need the opposite transformation: the rotation that places the reference average into the tomogram.

The required rotation is therefore the inverse:

**Rᵢ = Rᵉˣᵗ_zxz(−narotᵢ − 180°, tiltᵢ, −tdrotᵢ + 180°) = [Rᵉˣᵗ_zxz(eᵢ)]⁻¹**  (3)

Here, **zxz extrinsic** means three consecutive rotations about the fixed z, x, and z axes.

Equation (3) is an inverse transformation, not a mirror operation.

It is exactly invertible, so corrected orientations can later be converted back into ordinary Dynamo table angles without loss.

If this inversion is omitted, the measured sense of the screw is reversed.

### RELION orientations

For RELION, first combine the two stored orientation components:

**Rᵢ(aligned) = Rᵉˣᵗ_zyz(subtomogram angles) × Rᵉˣᵗ_zyz(alignment angles)**  (4)

The resulting rotation is rewritten as a Dynamo Euler-angle triple and then passed through Equation (3).

From this point onward, Dynamo and RELION data follow the same analysis path.

Use refined alignment angles when they are available.

If they are absent, as in the first refinement iteration, use the corresponding prior angles.

---

## 4. Finding the filament axis

For each filament, let **c** be the mean of all particle coordinates.

The filament axis is defined as the direction along which the particle coordinates show the greatest spread.

This is the first right singular vector of the centred coordinates:

**n̂ = arg max Σ [(xᵢ − c) · u]²**, with **||u|| = 1**

and

**zᵢ = s (xᵢ − c) · n̂**  (5)

The centre **c** is calculated from all particles and is not itself a particle.

It simply defines where:

**z = 0**

Moving the origin along the filament shifts every **zᵢ** by the same amount.

This shift is absorbed into the filament-specific offset **φ₀**, so it has no effect on the measured slope.

Using all particles rather than only the two endpoints makes the axis estimate more robust.

Filaments containing fewer than five segments are not given a fitted axis.

They remain in the data set and can still be marked manually.

### Axis direction

The fitted filament axis is initially a direction rather than an arrow.

Reversing it changes the signs of the associated coordinates and phases consistently, so the measured screw slope remains unchanged.

For comparison of particle pointing directions, however, a consistent arrow is useful.

We therefore orient the axis toward the direction in which most well-aligned particles already point:

**σ = sign [ Σ cos θᵢ ]**  (6)

where the sum includes only particles satisfying:

**|cos θᵢ| ≥ cos 30°**

and

**cos θᵢ = (Rᵢ ẑ) · n̂**

If no particle qualifies:

**σ = +1**

Only particles within 30° of either direction of the filament axis participate in this vote.

A particle pointing approximately sideways does not provide a meaningful vote for filament direction.

The quantities **zᵢ**, **φᵢ**, and the pointing direction **pᵢ** use the original axis **n̂**.

Only the angle to the oriented axis, **θᵢ**, uses **σn̂**.

---

## 5. Measuring the rotation about the filament axis

To measure rotation around the filament, construct two perpendicular directions spanning the plane normal to the filament axis:

**e₁ = (n̂ × ẑ) / ||n̂ × ẑ||**

**e₂ = n̂ × e₁**  (7)

If the cross product with **ẑ** vanishes, use **ŷ** instead.

Next, carry the reference x-direction through the orientation of each segment:

**vᵢ = Rᵢ x̂**

Its angle in the plane perpendicular to the filament is:

**φᵢ = atan2(vᵢ · e₂, vᵢ · e₁)**  (8)

For a helical reference, the reference z-axis runs approximately along the filament.

Its x-axis therefore lies across the filament, and the angle of this axis around the filament gives the rotational phase of the screw.

A segment does not need to point perfectly along the fitted filament axis.

If it is tilted, part of **vᵢ** projects onto **n̂**, but its remaining component in the perpendicular plane still follows the screw rotation **g**.

The construction fails only when **vᵢ** is exactly parallel to the filament axis, which does not occur for helical subunits.

### Pointing direction and angle to the axis

Two additional quantities are calculated for every segment.

The pointing direction is:

**pᵢ = sign[(Rᵢ ẑ) · n̂]**

and indicates which way the particle points along the filament.

The angle to the oriented filament axis is:

**θᵢ = arccos[(Rᵢ ẑ) · σn̂]**

A segment is classified as:

* **along the axis** if **θᵢ ≤ 30°**
* **reversed** if **θᵢ ≥ 150°**
* **off-axis** otherwise

Off-axis segments do not provide usable pointing-direction information.

---

## 6. Fitting the screw line

For a given rise and twist, the screw slope **g** is fixed.

Only the filament-specific phase offset **φ₀** needs to be fitted.

To remove the expected screw rotation from each segment, calculate:

**rᵢ = (φᵢ − g zᵢ) mod 360°**  (9)

If all particles follow the same screw, the values of **rᵢ** should cluster around one phase.

To prevent a minority of poorly aligned particles from shifting the fitted line, find the circular window with half-width:

**w = 25°**

that contains the largest number of **rᵢ** values.

Call this set **M**.

The filament offset is then:

**φ₀ = arg[(1 / |M|) Σᵢ∈M exp(i rᵢ)]**

and the residual of each particle from the fitted screw is:

**δᵢ = wrap(φᵢ − g zᵢ − φ₀)**  (10)

A conventional mean over all particles could be shifted by a poorly aligned population or by a group approximately 180° away from the main register.

The densest-window procedure instead identifies the dominant screw register first.

Particles outside it remain visible as large values of **δᵢ**.

### Main and reversed registers

The set **M** also determines the main pointing register.

The main register is defined by the pointing direction shared by most particles **inside M**, rather than by the majority direction of the entire filament.

Particles pointing in the opposite direction may form a second line with the same slope but a different offset.

Such a split is accepted only if the two groups genuinely point in opposite directions.

If at least three reversed particles are present, require:

**angle(mean direction of main group, mean direction of opposite group) ≥ 150°**  (11)

Otherwise, the split is discarded.

This prevents a filament containing particles that point approximately sideways from being divided artificially into two groups simply because noise changes the sign of **pᵢ**.

Groups of one or two reversed segments keep their label but are not assigned a separate fitted line.

### Unwrapping for plotting

For visualization, the measured phase is unwrapped along the filament:

**u₁ = φ₁**

**uᵢ = uᵢ₋₁ + wrap(φᵢ − φᵢ₋₁)**  (12)

The values **uᵢ** form the vertical axis of the screw plots.

Unlike **φᵢ**, they are not folded back into a single 360° range.

Instead, they accumulate successive angular changes and can therefore increase through many turns along a filament.

Without unwrapping, a rapidly rotating helix repeatedly crosses the 0°/360° boundary and appears as many almost vertical strokes.

After unwrapping, the same data form a continuous line.

Equation (12) uses only the measured orientations.

It does not depend on an assumed twist.

Changing the assumed twist therefore changes the fitted screw line, not the plotted data.

An incorrect assumed twist remains visible as a mismatch between the data and the line.

### Sampling condition

The unwrapping procedure assumes that neighbouring particles rotate by less than half a turn:

**|g| × Δz < 180°**  (13)

Equivalently:

**Δz < ½ × helical period**

where **Δz** is the axial separation between neighbouring segments.

If this condition is violated, one or more complete turns can be lost during unwrapping.

The resulting slope may be wrong in both magnitude and sign.

This is an aliasing problem: once the information has been lost through undersampling, no estimator can recover it.

The failure can also be silent.

The resulting screw plot may still appear as a straight line even though its slope is incorrect.

Section 13 shows the size of this effect and the check that should be performed.

The residual **δᵢ** is different from the unwrapped plotting coordinate.

It remains wrapped because only its absolute angular distance from the fitted screw is required.

---

## 7. Measuring the twist when it is not known

When the twist is unknown, it is estimated independently for each filament.

Within each filament, particles are first separated by pointing direction.

A pointing group must contain at least eight particles to be analysed independently.

The two pointing groups are allowed to have different phase offsets but are expected to share the same screw slope.

### Initial slope from neighbouring segments

Provided that Equation (13) holds, neighbouring particles cannot hide an entire turn.

An initial slope can therefore be estimated directly from neighbouring phase differences:

**g⁽⁰⁾ = median { wrap(φᵢ₊₁ − φᵢ) / (zᵢ₊₁ − zᵢ) }**  (14)

using only neighbour pairs satisfying:

**Δzᵢ > 0.3 Å**

Using the median makes the estimate robust to a small number of bad neighbour pairs.

Pairs separated by less than 0.3 Å are ignored.

If fewer than three valid neighbour pairs remain:

**g⁽⁰⁾ = 0**

Because this estimate is constructed entirely from neighbouring differences, it directly inherits the sampling condition of Equation (13).

### Refining the slope over longer distances

The measured phases are next unwrapped relative to the initial slope:

**ψᵢ = φᵢ + 360 × round[(g⁽⁰⁾ zᵢ − φᵢ) / 360]**

A second slope estimate uses pairs separated by more than 30 Å:

**g⁽¹⁾ = median { (ψⱼ − ψᵢ) / (zⱼ − zᵢ) }**

for pairs satisfying:

**i < j**

and

**|zⱼ − zᵢ| > 30 Å**

The longer distance gives a larger lever arm and therefore a more precise slope estimate.

However, this refined value is accepted only if it remains close to the neighbour-based estimate:

**g = g⁽¹⁾ if |g⁽¹⁾ − g⁽⁰⁾| ≤ 0.10 degrees/Å**

otherwise:

**g = g⁽⁰⁾**  (15)

The long-range refinement is applied once rather than iteratively.

Because the number of particle pairs grows approximately as the square of the number of segments, filaments containing more than 150 segments are reduced to 150 evenly spaced particles for this calculation.

This retains the full filament length while removing unnecessary closely spaced pairs.

### Straightness of the screw line

The consistency of the particles with the fitted screw is quantified by:

**C = (1 / N) × |Σ exp[i(ψᵢ − g zᵢ)]|**  (16)

with:

**0 ≤ C ≤ 1**

Here:

* **C = 1** means that all particles lie on the same screw line
* **C = 0** means that their phases are completely dispersed

### Combining the filament estimates

Each filament contributes one twist estimate.

If both pointing groups contain enough particles to be analysed separately, their slopes are combined using:

**group length × C**

as the weight.

Their straightness scores are combined using group length as the weight.

The filament twist is:

**τ_f = g_f h**

with weight:

**w_f = L_f C_f**

where **L_f** is the total axial span of the filament.

Long, well-defined filaments therefore contribute more strongly than short or poorly coherent filaments.

If neither pointing group contains at least eight particles, all particles in that filament are analysed together.

### Removing filament-level outliers

Let **τ̃** be the weighted median of the filament estimates.

A filament is rejected when:

**|τ_f − τ̃| > 3 × max[1.4826 × median(|τ_f − τ̃|), 0.20°]**  (17)

The factor 1.4826 converts a median absolute deviation into its standard-deviation equivalent.

The 0.20° floor prevents the rejection criterion from becoming unrealistically narrow when the filament estimates agree extremely closely.

Rejected filaments are reported.

The final twist estimate is the weighted median of the remaining filament estimates.

### Estimating uncertainty

Two independent uncertainty measures are calculated, and the larger is reported.

The first is obtained by resampling the set of filaments:

**B = 400 times**

The second asks a different question:

**Does the complete data set actually identify one particular twist?**

For every candidate twist, calculate:

**Q(τ) = (1 / F) Σ_f [(1 / N_f) × |Σᵢ exp{i[φᵢ − (τ / h)zᵢ]}|]**  (18)

The function is evaluated at 1201 candidate twist values between:

**−6° and +6° per subunit**

Let:

**τ\* = the value of τ that maximizes Q(τ)**

Then define:

**P = Q(τ\*) − median[Q(τ) for |τ − τ\*| > 1°]**

and:

**I = {τ : Q(τ) ≥ Q(τ\*) − 0.05 P}**  (19)

The value **P** measures how strongly the best twist rises above the background.

The interval **I** contains twist values that fit almost as well as the optimum.

Agreement between filaments alone is not sufficient evidence for a well-determined twist.

Multiple filaments may share the same limitation and therefore agree on a value even when the data themselves do not uniquely determine it.

In such a case, the resampling uncertainty may be small while **Q(τ)** remains broad or flat.

A twist estimate is therefore considered reliable only when **all** of the following conditions are satisfied:

* at least half of the total weight lies within **0.5°** of the reported twist
* median **C ≥ 0.30**
* resampling half-width **≤ 0.50°**
* **P ≥ 0.30**
* half-width of **I ≤ 0.30°**
* the reported twist lies inside **I**

If any condition fails, the failed criterion is reported.

---

## 8. Choosing which segments to remove

Segments can be marked manually or by either of two independent automatic criteria:

**Rule (a): |δᵢ| > 20°**

**Rule (b): min(θᵢ, 180° − θᵢ) > 30°**  (20)

Rule (a) identifies particles that lie away from the fitted screw.

Rule (b) identifies particles that point sideways and therefore belong to neither axial pointing direction.

Rule (a) is evaluated against the main screw line rather than the second line defined in Equation (11).

A particle that is merely reversed would therefore initially appear far from the main line.

For this reason, Rule (a) is applied after the orientation correction described in Equation (21), and uses the corrected orientation.

After this correction, only particles that remain inconsistent with the screw are marked by Rule (a).

Marked particles are written to a list of particle identifiers, one identifier per line.

The original input tables are never modified.

The application instead writes new output files alongside the input:

* the list of marked particles
* the list of corrected orientations
* for a Dynamo project, unless disabled, a copy of the working table rows

### Correcting orientations instead of removing particles

Two 180° orientation corrections are available.

The first is:

**R′ = R × Rₓ(180°)**

The second is:

**R″ = R_n̂(180°) × R**  (21)

The first operation reverses the direction in which the segment points.

It acts in the segment's own coordinate frame and therefore does not depend on the filament geometry, particle position, or helical twist.

It changes:

**θ → 180° − θ**

while leaving:

**φ unchanged**

The second operation represents the two-fold ambiguity around the filament axis.

It leaves:

**θ unchanged**

while changing:

**φ → φ + 180°**

The two operations commute.

Both corrected rotations can be converted back into ordinary table Euler angles.

---

## 9. The two algorithms

### Algorithm 1 — Geometry of one filament

Steps 1–5 are calculated once when the data are loaded.

Step 6 is repeated whenever the twist, rise, or pixel size is changed.

1. Read the particle coordinates and orientations. Construct **Rᵢ** using Equation (3), preceded by Equation (4) for RELION data.

2. If the filament contains fewer than five segments, keep the filament but do not fit an axis.

3. Fit the filament axis using Equation (5). At this stage, project the particle coordinates onto the axis in voxel units and sort the particles along the axis.

4. Construct the perpendicular basis using Equation (7), then calculate the rotational phases using Equation (8).

5. Record the pointing direction **pᵢ**, the axis sense **σ** from Equation (6), and the angle **θᵢ**.

6. Convert the projected coordinates from voxels to ångström using the pixel size **s**, completing Equation (5).

Set:

**g = τ / h**

Fit **φ₀** and **δᵢ** using Equations (9) and (10).

Determine the main register from **M**, and fit a reversed register if Equation (11) permits it.

The pixel-size scaling is performed in Step 6 rather than Step 3 because pixel size is a live parameter.

Steps 1–5 therefore contain only quantities that are independent of user-adjustable helical parameters.

### Algorithm 2 — Twist of one tomogram

1. For each fitted filament, divide particles by pointing direction. Keep pointing groups containing at least eight particles. If neither group qualifies, use all particles together.

2. For each group, calculate the neighbour-based slope using Equation (14), unwrap the phases to this estimate, refine it once using Equation (15), thinning to 150 particles where necessary, and calculate the straightness score using Equation (16).

3. Combine the pointing groups into one filament estimate:

**τ_f = g_f h**

with weight:

**w_f = L_f C_f**

4. Reject filament-level outliers using Equation (17), then calculate the weighted median of the remaining filament estimates.

5. Resample the filament set 400 times to obtain the first uncertainty estimate. Calculate **Q(τ)** using Equation (18), followed by **P** and **I** using Equation (19).

6. Report the final twist with the larger of the two uncertainty estimates and evaluate all reliability conditions from Section 7.

The orientations determine only:

**g = τ / h**

Twist and rise therefore cannot be determined independently from the orientations alone.

Either hold the rise fixed and report:

**τ = g h**

or, when both values are varied, report the **(τ, h)** pair nearest the chosen starting values along the line of constant **g**.

---

## 10. Parameter values

| Symbol | Value | Meaning |
| ------ | ----: | ------- |
| s | from the file, or 7.92 Å/voxel | pixel size |
| h | 4.75 Å/subunit | rise, fixed unless fitted |
| — | 5 segments | minimum number of segments for fitting an axis |
| w | 25° | half-width of the offset window |
| — | 30° | half-angle of the on-axis cone |
| — | 150° | minimum opposition required for a reversed group |
| — | 3 segments | minimum reversed-group size for a separate fit |
| — | 20° | removal threshold on the residual δᵢ, Rule (a) |
| — | 8 segments | minimum number of segments for a pointing group |
| — | 0.3 Å, 3 pairs | minimum neighbour separation and pair count |
| — | 30 Å | minimum long-range pair separation |
| — | 150 segments | thinning threshold |
| — | 0.10 degrees/Å | allowed difference between g⁽⁰⁾ and g⁽¹⁾ |
| — | 3, floor 0.20° | filament-outlier criterion |
| B | 400 | number of filament resamples |
| — | −6° to +6°, 1201 points | twist search grid |
| — | 0.05 | drop used to define I |
| — | 0.50, 0.30, 0.50°, 0.30, 0.30° | reliability limits |

---

## 11. Validation against known twist

Synthetic filaments with known twist were generated by aligning the reference z-direction with a chosen filament axis and rotating the reference x-direction by a fixed amount per subunit.

The resulting orientations were written as ordinary table Euler angles and then read back using the same functions applied to real data.

Unless otherwise stated, each simulated filament contains 300 particles spanning approximately 1400 Å.

| Test | Expected | Result |
| ---- | -------: | -----: |
| Four twists × three filament-axis directions | input value | input value, to three decimals |
| Fitted axis versus true axis | 1 | 1.0000 |
| Segment order reversed and fitted axis flips | +1.400 | +1.400 |
| Segments tilted 0°–60° from the axis | +1.400 | +1.4000 throughout |
| Segments sampled every 1, ½, ¼ subunit | +1.400 | exact |
| Stored orientations used without the inverse in Equation (3) | — | −1.333, reversed sense |

These tests show that the geometry recovers the known twist exactly and is unaffected by the arbitrary sign of the fitted filament axis, by particle tilt relative to the filament, or by oversampling.

### Angular noise

Twenty synthetic filaments were passed through the estimator from Section 7.

The calculation was repeated five times for each noise level.

The mean estimated twists were:

**+1.400, +1.400, +1.399, +1.350, +1.408° per subunit**

for angular noise levels of:

**0°, 5°, 10°, 20°, and 30°**

respectively.

The variation between repeats increased with noise:

* approximately ±0.003° at 5° noise
* approximately ±0.017° at 10° noise
* −0.29° to +0.11° at 30° noise

Increasing orientation noise mainly reduces precision.

An individual noisy data set can nevertheless fall above or below the true value.

---

## 12. Results

The method was applied to the Dynamo refinement of tomogram 1.

The analysis used:

* four refinement iterations
* two concatenated reference tables
* pixel size = **7.92 Å/voxel**
* fixed rise = **4.75 Å/subunit**

### The data

The tomogram contains:

**42 filaments**

and:

**8484 segments**

All 42 filaments contain enough segments to fit an axis.

Individual filaments contain 129–449 segments, with a median of 199.

Their axial spans range from 847 to 3961 Å, with a median of 1577 Å.

The median lateral distance of particles from the fitted filament axis is:

**19.5 Å**

The median axial separation between neighbouring particles is:

**1.64 Å**

At the fitted screw slope, this corresponds to only:

**0.45°**

of rotation between neighbouring particles, far below the 180° sampling limit in Equation (13).

The sampling condition is therefore satisfied with a wide margin.

The filaments are also longer than one complete screw turn, which corresponds to:

**1322 Å**

for this data set.

### Twist

The estimator from Section 7 gives:

**+1.293 ± 0.170° per subunit**

The result passes all reliability conditions.

Of the 42 filaments:

**41 are retained as inliers**

Their individual twist estimates are tightly grouped:

* weighted median: **+1.29°**
* robust standard deviation: **0.09°**
* central range: **+0.88° to +1.47°**
* **99%** of the total weight lies within 0.5° of the reported result

Filament resampling gives a half-width of:

**0.033°**

The independent **Q(τ)** profile peaks at approximately:

**+1.29°**

with prominence:

**P = 0.45**

and a supported interval:

**I = [+1.12°, +1.46°]**

This interval contains the fitted twist.

The two independent uncertainty measures therefore agree.

The larger value:

**0.170°**

is reported as the final uncertainty.

One filament is rejected by Equation (17):

**filament 38**

which gives a twist estimate of:

**+0.63°**

from 232 segments.

The measured value of approximately +1.29° is lower than the assumed twist of +1.4°.

The data also fit the measured value better.

At:

**τ = +1.29°**

69% of segments lie within 20° of the main screw line.

At:

**τ = +1.4°**

the corresponding fraction is 62%.

### Register and pointing direction

Equation (11) accepts no reversed group in any filament.

The data set therefore contains no detected polarity split: all particles point in the same direction along their respective filaments.

Rule (b) marks:

**4%**

of particles as sideways.

Rule (a) marks:

**38%**

of particles at the assumed twist.

Because no second register is detected that could explain these particles, they remain outside the fitted screw.

The median filament straightness score is:

**C = 0.61**

---

## 13. Where the method does not apply

### Helical symmetry imposed during refinement

The method assumes that the stored particle orientations retain the running rotational phase of the helix.

This is true for Dynamo tables and for RELION refinements in which helical symmetry was **not** imposed.

If helical symmetry is imposed during refinement, the rotation around the filament becomes equivalent up to one screw step.

A particle at:

**(φ, z)**

and one at:

**(φ + τ, z + h)**

are then treated as equivalent.

As a result, the running phase folds back and the measured slope becomes too shallow.

For such refinements, obtain the twist from the refinement's own helical-symmetry search instead of using this method.

### Segment spacing

Equation (13) must be satisfied:

**|g| × Δz < 180°**

Consider a synthetic helix with:

**τ = 166° per subunit**

and:

**h = 27 Å**

Sampling once per subunit correctly returns:

**+166.00°**

Sampling every second subunit produces a true step of:

**+332°**

After wrapping:

**+332° → −28°**

and the measured twist becomes:

**−14.00°**

which is wrong in both sign and magnitude.

Sampling every third subunit gives:

**+46.00°**

The aliasing boundary is equally sharp:

**179° → +179.00°**

but:

**181° → −179.00°**

The important point is that the plotted screw, the initial slope estimate, and the final slope can all be wrong by the same amount.

They therefore agree with one another and provide no internal warning.

The sampling condition can fail for two reasons:

1. the screw rotates too rapidly
2. neighbouring picked particles skip one or more subunits

The second case can occur locally even within an otherwise well-sampled filament if an alignment advances by two subunits between neighbouring picks.

Before trusting the fitted slope, compare the median separation between neighbouring particles with half of the helical period.

### Filament length

Short filaments reduce precision rather than imposing an absolute length cutoff.

The method therefore does not apply a direct minimum-length test.

Instead, insufficient filament length appears through the reliability criteria in Section 7, particularly through the width of **I**.

To test this behaviour, 25 synthetic filaments with 10° of orientation noise were analysed at a true twist of:

**+1.4° per subunit**

For this twist, one complete screw turn spans:

**1221 Å**

| Span | Fraction of one turn | Fitted twist | Half-width of I | Reliable |
| ---: | -------------------: | -----------: | --------------: | :------: |
| 90 Å | 7% | +0.954° | 0.77° | no |
| 185 Å | 15% | +1.412° | 0.75° | no |
| 470 Å | 38% | +1.400° | 0.56° | no |
| 945 Å | 77% | +1.386° | 0.29° | yes |
| 1420 Å | 116% | +1.393° | 0.20° | yes |
| 2845 Å | 233% | +1.397° | 0.10° | yes |

The fitted value remains close to the true twist down to approximately one third of a turn.

What deteriorates first is the uncertainty.

As the filament becomes shorter, the supported interval broadens from approximately:

**±0.10°**

to:

**±0.77°**

With the present reliability criteria, the result becomes reliable at approximately three quarters of one turn.

Below roughly one tenth of a turn, the fitted value itself also begins to drift substantially.

A short filament therefore does not make a screw slope mathematically impossible.

Instead, it makes that slope increasingly poorly resolved.

The reliability analysis is designed to report this loss of information rather than allowing it to fail silently.

In the experimental data analysed here, the median filament span is:

**1577 Å**

corresponding to approximately:

**119% of one screw turn**

and therefore lying within the reliable range observed in the synthetic tests.

---

## 14. Handedness

Equation (2) measures the screw in the coordinate frame in which the tomogram was reconstructed.

If this coordinate frame is mirrored relative to the physical specimen, the sign of the screw reported outside this analysis must be inverted.

The correct transformation is:

**φᵢ → −φᵢ**

while:

**zᵢ, θᵢ, and pᵢ remain unchanged**

The rotation about the filament axis is the only quantity in this analysis whose sign changes under this mirror.

Simply changing:

**τ → −τ**

is not sufficient.

If only τ were inverted, the measured particle phases would still progress in the original direction and the calculated residuals **δᵢ** would therefore become incorrect.

After the complete mirror transformation:

* every **|δᵢ|** remains unchanged
* every particle classification from Section 5 remains unchanged
* only the relevant signs are reversed
