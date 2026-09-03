#!/usr/bin/env python3
"""
Global twist auto-fit for Rohlex.

author: Wen-Lu Chung

Estimates the helix twist that best explains the measured per-segment roll across
ALL filaments of the tomogram at once, so the user no longer has to dial twist by
eye. Pure numpy — no Qt — so it stays testable and the dialog just drives it.

What we measure: the local SLOPE, expanded to the longest reliable baseline
---------------------------------------------------------------------------
Each segment has a position `pos` (Angstrom along the SVD filament axis) and a
measured roll `phi` (deg). The screw model is  phi = rate*pos + phi0  with
rate = twist / rise (deg/Angstrom). The quantity the data fixes is the SLOPE
dphi/dpos = rate.

A single global circular-resultant fit (concentration of phi - rate*pos over the
whole tomogram) works only when each filament spans much MORE than one helical
period (360/rate). When filaments are SHORT relative to the period — e.g. tau,
where a ~470 A filament turns only ~70 deg — roll noise accumulates into a spurious
steep ramp and the resultant locks onto the wrong (too-steep) rate. Verified: on
tau the resultant gives twist ~ -3.5, while the true slope is ~ -0.7.

So we fit the slope DYNAMICALLY, the way the eye does it (user's design):
  1. SEED from adjacent segments — a short, wrap-unambiguous baseline that pins the
     coarse slope (robust median; noise-resistant).
  2. UNWRAP the roll along the sequence toward the seed, then EXPAND once with a
     robust Theil-Sen slope over the full baseline (longest lever = most precise).
  3. GUARD: accept the expansion only if it stays within REFINE_TOL of the seed;
     otherwise keep the seed. And do it ONCE — never iterate to convergence, because
     iterating compounds a tiny steepening bias into a runaway (verified: iterating
     drives tau from -0.8 to -1.1). One guarded step refines MSA (+1.15 -> +1.30)
     without moving tau (-0.77).
This gives the right slope in BOTH regimes from the same code: tau ~ -0.77, MSA
~ +1.30 (span*confidence-weighted median over filaments).

The 2nd group votes, it is not merged  (user: "flip the 2nd group first")
    The antiparallel (polarity-flipped) segments are the same screw pointing the
    other way, at the same slope. Each polarity group casts its OWN slope vote —
    which uses the 2nd group's data at full weight AND survives the case where the
    flip is spatially segregated to one end (tau: flip/position correlation 0.72),
    where merging into one register would help but a plain fit on either half alone
    would not. (An explicit register merge was tried and REJECTED: it biases the
    slope, because the flip is computed at an approximate rate.) Rot-flip (+180 / C2)
    folding is an optional per-group toggle, off by default.

Consensus + robustness  (user: "longest slope", "outliers can interrupt")
    Consensus is the median of filament votes weighted by span*confidence (long,
    linear filaments count most). Filaments far from the median (MAD) are rejected
    and reported. Twist and rise are degenerate from roll (only the ratio appears):
    fix_rise gives twist = rate*rise; free projects the start onto the measured-rate
    ridge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TWIST_MIN, TWIST_MAX = -6.0, 6.0
N_GRID = 1201                      # grid for the display density only
MIN_SEG_SLOPE = 8                  # a polarity group needs this many segments to vote a slope
_MIN_PAIR_A = 30.0                 # min pair baseline (A) for Theil-Sen slopes
REFINE_TOL = 0.10                  # deg/A: accept the expansion only within this of the seed
_MAD_K = 3.0                       # reject a filament > K robust-sigma from the median
_MAD_FLOOR = 0.20                  # deg: don't reject inside this even if MAD is tiny
_KDE_BW = 0.20                     # twist-density kernel bandwidth (deg) for display

N_BOOT = 400                       # bootstrap resamples of the filament set for the CI
AGREE_TOL = 0.5                    # deg: a filament "agrees" if its vote is within this
CONF_MIN = 0.30                    # min median linearity to trust the fit
AGREE_MIN = 0.50                   # min weighted fraction agreeing to trust the fit
CI_MAX = 0.50                      # deg: max bootstrap CI half-width to call it reliable

# --- is the twist DETERMINED by the data at all? -----------------------------
# Filament agreement is NOT enough: every filament can share the same systematic
# limitation and agree on a value the data does not actually pin down (verified on
# tau Class3D job062 — 79/81 filaments agreed on -0.47, bootstrap CI +/-0.09, but the
# fit-quality-vs-twist profile was FLAT from -0.15 to -1.0, so -0.47 and -1.0 are
# indistinguishable there). So we also profile the fit quality across twist and take
# its PROMINENCE (peak above floor) and WIDTH as the honest determination/uncertainty.
PROFILE_DROP = 0.05                # interval = twists fitting within 5% of the peak
PROM_MIN = 0.30                    # min profile prominence for the twist to be determined
WIDTH_MAX = 0.30                   # deg: max profile half-width to call it reliable


@dataclass
class FilFit:
    """One filament's slope vote and how linear it is."""
    fid: int
    twist: float          # this filament's dynamic-slope vote (deg/subunit)
    peakscore: float      # linearity: roll concentration about its fitted line, [0, 1]
    agree: float          # |vote - consensus| (deg) — small = agrees
    n: int
    span: float = 0.0     # baseline along the axis (A) — its lever / vote weight
    outlier: bool = False


@dataclass
class FitStats:
    """Statistics that say whether the consensus twist is trustworthy."""
    peak_concentration: float     # median filament linearity, [0, 1]
    prominence: float             # fit-quality profile peak above its floor — is twist DETERMINED?
    ci95: float                   # bootstrap 95% half-width on the consensus twist (deg)
    interval_lo: float            # profile-likelihood interval: twists fitting within PROFILE_DROP
    interval_hi: float            # of the peak. THE honest uncertainty (bootstrap alone lies).
    profile_peak: float           # twist at the profile maximum (deg/subunit)
    vote_spread: float            # robust std (1.4826*MAD) of per-filament votes (deg)
    vote_lo: float                # 2.5th percentile of per-filament votes (deg)
    vote_hi: float                # 97.5th percentile of per-filament votes (deg)
    agree_frac: float             # weighted fraction voting within AGREE_TOL of consensus
    median_span: float            # median filament length (A); vs period tells if resolvable
    n_flipped: int                # filaments whose 2nd (antiparallel) group also voted
    reliable: bool
    verdict: str


@dataclass
class TwistFitResult:
    twist: float
    rate: float
    rise: float
    unc: float
    grid: np.ndarray                  # twist grid for the vote-density curve
    curve: np.ndarray                 # span*conf-weighted density of filament votes, peak-normed
    per_fil: list
    n_inlier: int
    outliers: list
    mode: str                         # "fix_rise" | "free"
    spacing_med: float
    rot_flip: bool = False
    stats: FitStats = field(default=None)
    traj: np.ndarray = field(default=None)


# --- dynamic slope: seed from local pairs, expand once to the longest baseline -
def _adjacent_seed(pos, phi):
    """Coarse slope (deg/A) from adjacent pairs — a short, wrap-safe baseline."""
    o = np.argsort(pos)
    d = ((np.diff(phi[o]) + 180.0) % 360.0) - 180.0
    dp = np.diff(pos[o])
    m = dp > 0.3
    return float(np.median(d[m] / dp[m])) if m.sum() >= 3 else 0.0


def _theilsen(p, y):
    """Median pairwise slope over baselines > _MIN_PAIR_A (longest levers dominate)."""
    n = len(p)
    if n > 150:                                    # cap pairs for speed; keep the span
        idx = np.linspace(0, n - 1, 150).astype(int)
        p, y = p[idx], y[idx]
        n = len(p)
    dp = p[None, :] - p[:, None]
    dy = y[None, :] - y[:, None]
    m = np.triu(np.ones((n, n), bool), 1) & (np.abs(dp) > _MIN_PAIR_A)
    return float(np.median(dy[m] / dp[m])) if m.any() else float("nan")


def _group_slope(pos, phi, rot_flip=False):
    """Dynamic slope of one polarity group: seed + ONE guarded Theil-Sen refinement.
    Returns (rate deg/A, confidence [0,1], span A)."""
    o = np.argsort(pos)
    p, ph = pos[o].astype(float), phi[o].astype(float).copy()
    span = float(p.max() - p.min()) or 1.0
    seed = _adjacent_seed(p, ph)
    if rot_flip:                                   # fold a +180 (rot/C2) split, de-screwed
        res = ((ph - seed * p) + 180.0) % 360.0 - 180.0
        c = np.degrees(np.angle(np.exp(1j * np.radians(res)).mean()))
        d = ((res - c + 180.0) % 360.0) - 180.0
        ph = np.where(np.abs(d) > 90.0, ph - 180.0, ph)
        seed = _adjacent_seed(p, ph)
    s = seed if np.isfinite(seed) else 0.0
    unw = ph + 360.0 * np.round((s * p - ph) / 360.0)      # unwrap toward the seed
    unw = unw - np.median(unw - s * p)
    s2 = _theilsen(p, unw)                                 # expand to the full baseline
    s_use = s2 if (np.isfinite(s2) and abs(s2 - s) <= REFINE_TOL) else s
    unw = ph + 360.0 * np.round((s_use * p - ph) / 360.0)
    conf = float(abs(np.exp(1j * np.radians(unw - s_use * p)).sum()) / len(p))
    return s_use, conf, span


# --- weighted robust helpers -------------------------------------------------
def _wmedian(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    if x.size == 0:
        return float("nan")
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    if cw[-1] <= 0:
        return float(np.median(x))
    return float(x[o][np.searchsorted(cw, cw[-1] / 2.0)])


def _median_spacing(filaments):
    ds = [float(np.median(np.diff(np.sort(f.pos)))) for f in filaments
          if getattr(f, "fittable", False) and len(f.pos) > 1]
    return float(np.median(ds)) if ds else float("nan")


def _median_span(filaments):
    sp = [float(f.pos.max() - f.pos.min()) for f in filaments
          if getattr(f, "fittable", False) and len(f.pos) > 1]
    return float(np.median(sp)) if sp else float("nan")


# --- statistics + verdict ----------------------------------------------------
def _compute_stats(votes, weights, confs, inlier, consensus, median_span, n_flipped,
                   prof_peak=float("nan"), prom=float("nan"),
                   lo=float("nan"), hi=float("nan")):
    votes = np.asarray(votes, float)
    w = np.asarray(weights, float) * inlier
    within = np.abs(votes - consensus) <= AGREE_TOL
    agree_frac = float(w[within].sum() / max(w.sum(), 1e-9))
    med_conf = float(np.median(np.asarray(confs)[inlier])) if inlier.any() else 0.0

    vin, win = votes[inlier], w[inlier]
    vote_spread = float(1.4826 * np.median(np.abs(vin - np.median(vin)))) if vin.size else float("nan")
    vlo, vhi = (float(np.percentile(vin, 2.5)), float(np.percentile(vin, 97.5))) \
        if vin.size else (float("nan"), float("nan"))

    if vin.size >= 2:
        rng = np.random.default_rng(0)
        m = vin.size
        peaks = [_wmedian(vin[b], win[b]) for b in
                 (rng.integers(0, m, m) for _ in range(N_BOOT))]
        ci95 = float((np.percentile(peaks, 97.5) - np.percentile(peaks, 2.5)) / 2.0)
    else:
        ci95 = float("nan")

    # The honest +/-: the profile-likelihood half-width, or the bootstrap CI, whichever
    # is LARGER. Bootstrap alone under-reports when all filaments share a systematic
    # limitation (they agree on a value the data does not determine).
    half = (hi - lo) / 2.0 if np.isfinite(lo) and np.isfinite(hi) else float("nan")
    ci_ok = not np.isfinite(ci95) or ci95 <= CI_MAX
    prom_ok = not np.isfinite(prom) or prom >= PROM_MIN
    width_ok = not np.isfinite(half) or half <= WIDTH_MAX
    # The slope estimate must lie inside the range the data actually supports. If the
    # dynamic fit lands outside the profile interval the two measurements contradict
    # each other -- never report that as reliable.
    agrees_profile = (not np.isfinite(lo)) or (lo <= consensus <= hi)
    reliable = (agree_frac >= AGREE_MIN) and (med_conf >= CONF_MIN) and ci_ok \
        and prom_ok and width_ok and agrees_profile
    if reliable:
        verdict = "RELIABLE — the data pins the twist and the filaments agree."
    else:
        why = []
        if not prom_ok:
            why.append(f"the data barely distinguishes twists (profile prominence "
                       f"{prom:.2f}) — the twist is NOT determined by these rolls")
        if not width_ok:
            why.append(f"any twist in [{lo:+.2f}, {hi:+.2f}]° fits essentially as well")
        if agree_frac < AGREE_MIN:
            why.append(f"filament slopes disagree (only {agree_frac*100:.0f}% within "
                       f"±{AGREE_TOL:.1f}° of consensus)")
        if med_conf < CONF_MIN:
            why.append(f"rolls are not linear (median linearity {med_conf:.2f})")
        if not ci_ok:
            why.append(f"unstable consensus (bootstrap CI ±{ci95:.2f}°)")
        if not agrees_profile:
            why.append(f"the slope fit ({consensus:+.2f}°) falls OUTSIDE the range the "
                       f"data supports [{lo:+.2f}, {hi:+.2f}]° — the two disagree")
        hint = ""
        if not (prom_ok and width_ok) and np.isfinite(median_span):
            hint = (f" Filaments span only ~{median_span:.0f} Å — too little turn to fix the "
                    "slope. Trust a real-space/3D symmetry search over this fit.")
        verdict = "UNRELIABLE — " + "; ".join(why) + "." + hint

    return FitStats(peak_concentration=med_conf, prominence=prom, ci95=ci95,
                    interval_lo=lo, interval_hi=hi, profile_peak=prof_peak,
                    vote_spread=vote_spread, vote_lo=vlo, vote_hi=vhi,
                    agree_frac=agree_frac, median_span=median_span,
                    n_flipped=n_flipped, reliable=reliable, verdict=verdict)


def _quality_profile(fils, twists, rise):
    """Mean per-filament roll concentration as a function of candidate twist.

    This is the profile likelihood of the twist: how well the WHOLE dataset's rolls
    line up on a screw of that slope. Its peak height above the floor says whether
    the twist is determined at all, and the width of its top says how precisely —
    which filament-to-filament agreement cannot tell you (they can agree on a value
    the data does not pin down). Vectorised over the twist grid.
    """
    rates = twists / rise
    tot = np.zeros_like(twists)
    n = 0
    for f in fils:
        pos = f.pos.astype(float)
        phi = f.phi.astype(float)
        if len(pos) < 2:
            continue
        res = np.radians(phi[:, None] - rates[None, :] * pos[:, None])   # (segs, grid)
        tot += np.abs(np.exp(1j * res).sum(0)) / len(pos)
        n += 1
    return tot / max(n, 1)


def _profile_stats(twists, profile):
    """(peak_twist, prominence, lo, hi) of the fit-quality profile."""
    k = int(np.argmax(profile))
    peak = float(profile[k])
    off = np.abs(twists - twists[k]) > 1.0
    floor = float(np.median(profile[off])) if off.any() else float(profile.min())
    prom = peak - floor
    inside = twists[profile >= peak - PROFILE_DROP * prom] if prom > 0 else twists
    return float(twists[k]), prom, float(inside.min()), float(inside.max())


def _vote_density(twists, votes, weights, inlier):
    """Peak-normalised span*conf-weighted Gaussian density of filament votes."""
    curve = np.zeros_like(twists)
    for v, w, keep in zip(votes, weights, inlier):
        if keep:
            curve += w * np.exp(-0.5 * ((twists - v) / _KDE_BW) ** 2)
    mx = curve.max()
    return curve / mx if mx > 0 else curve


# --- per-filament pass -------------------------------------------------------
def _fit_votes(filaments, rise, rot_flip):
    """One dynamic-slope vote per filament (both polarity groups voting, combined)."""
    fils = [f for f in filaments if getattr(f, "fittable", False)
            and f.axis is not None and np.isfinite(f.phi).any()]
    if not fils:
        return None
    votes, confs, spans, ids, nn, nflip = [], [], [], [], [], 0
    for f in fils:
        pos, phi = f.pos.astype(float), f.phi.astype(float)
        fl = f.flipped if getattr(f, "flipped", None) is not None and f.flipped.size == f.n \
            else np.zeros(f.n, bool)
        groups = [g for g in (~fl, fl) if g.sum() >= MIN_SEG_SLOPE]
        if not groups:
            groups = [np.ones(f.n, bool)]
        gsl, gcf, gsp = [], [], []
        for g in groups:
            sl, cf, sp = _group_slope(pos[g], phi[g], rot_flip)
            gsl.append(sl); gcf.append(cf); gsp.append(sp)
        gsl, gcf, gsp = np.array(gsl), np.array(gcf), np.array(gsp)
        w = gsp * np.clip(gcf, 1e-3, None)
        slope = float(np.average(gsl, weights=w))          # both groups share the slope
        votes.append(slope * rise)
        confs.append(float(np.average(gcf, weights=gsp)))
        spans.append(float(pos.max() - pos.min()))
        ids.append(int(f.fid)); nn.append(f.n)
        nflip += int(len(groups) > 1)
    return fils, np.array(votes), np.array(confs), np.array(spans), ids, nn, nflip


def _assemble(rise, twists, votes, confs, spans, ids, nn, nflip, fils, mode,
              rot_flip, traj=None):
    weights = spans * np.clip(confs, 1e-3, None)
    med = _wmedian(votes, weights)
    mad = float(np.median(np.abs(votes - med)))
    scale = max(1.4826 * mad, _MAD_FLOOR)
    outlier_mask = np.abs(votes - med) > _MAD_K * scale
    if outlier_mask.all():
        outlier_mask[:] = False
    inlier = ~outlier_mask

    twist = _wmedian(votes[inlier], weights[inlier])
    rate = twist / rise
    # Profile the fit quality across twist: does the data actually DETERMINE the twist?
    profile = _quality_profile(fils, twists, rise)
    prof_peak, prom, lo, hi = _profile_stats(twists, profile)
    stats = _compute_stats(votes, weights, confs, inlier, twist,
                           _median_span(fils), nflip, prof_peak, prom, lo, hi)
    half = (hi - lo) / 2.0 if np.isfinite(lo) and np.isfinite(hi) else float("nan")
    cands = [v for v in (stats.ci95, half) if np.isfinite(v)]
    unc = max(cands) if cands else scale       # honest +/-: widest of the two measures
    curve = _vote_density(twists, votes, weights, inlier)

    per_fil = [FilFit(fid=ids[i], twist=float(votes[i]), peakscore=float(confs[i]),
                      agree=float(abs(votes[i] - twist)), n=int(nn[i]),
                      span=float(spans[i]), outlier=bool(outlier_mask[i]))
               for i in range(len(ids))]
    outliers = [ids[i] for i in range(len(ids)) if outlier_mask[i]]
    return TwistFitResult(twist=twist, rate=rate, rise=rise, unc=unc,
                          grid=twists, curve=curve, per_fil=per_fil,
                          n_inlier=int(inlier.sum()), outliers=outliers,
                          mode=mode, spacing_med=_median_spacing(fils),
                          rot_flip=rot_flip, stats=stats, traj=traj)


# --- public API --------------------------------------------------------------
def quick_twist(filaments, rise: float, rot_flip: bool = False) -> float:
    """Just the consensus twist (deg/subunit), for live readouts.

    Same measurement as fit_fixed_rise -- per-filament dynamic slope votes, MAD
    outlier rejection, span*confidence-weighted median -- but WITHOUT the quality
    profile and bootstrap, which dominate that function's cost (the profile alone
    evaluates every filament against a 1201-point twist grid). Use this where a
    number is wanted per interaction and the reliability verdict is not; use
    fit_fixed_rise when the answer is going to be applied or trusted.
    """
    got = _fit_votes(filaments, rise, rot_flip)
    if got is None:
        return float("nan")
    _, votes, confs, spans = got[:4]
    weights = spans * np.clip(confs, 1e-3, None)
    med = _wmedian(votes, weights)
    scale = max(1.4826 * float(np.median(np.abs(votes - med))), _MAD_FLOOR)
    inlier = np.abs(votes - med) <= _MAD_K * scale
    if not inlier.any():
        inlier[:] = True
    return _wmedian(votes[inlier], weights[inlier])


def fit_fixed_rise(filaments, rise: float, working_rate: float = None,
                   rot_flip: bool = False, tmin: float = TWIST_MIN,
                   tmax: float = TWIST_MAX, n_grid: int = N_GRID) -> TwistFitResult:
    """Dynamic-slope twist fit at a fixed rise (span*confidence-weighted consensus).
    `working_rate` is accepted for API symmetry but unused — the seed is self-determined."""
    twists = np.linspace(tmin, tmax, n_grid)
    got = _fit_votes(filaments, rise, rot_flip)
    if got is None:
        return TwistFitResult(float("nan"), float("nan"), rise, float("nan"),
                              twists, np.zeros_like(twists), [], 0, [],
                              "fix_rise", float("nan"), rot_flip)
    fils, votes, confs, spans, ids, nn, nflip = got
    return _assemble(rise, twists, votes, confs, spans, ids, nn, nflip, fils,
                     "fix_rise", rot_flip)


def fit_free(filaments, twist0: float, rise0: float, lr: float = 0.1,
             iters: int = 200, rot_flip: bool = False, tmin: float = TWIST_MIN,
             tmax: float = TWIST_MAX, n_grid: int = N_GRID) -> TwistFitResult:
    """Free (twist + rise): the data fixes only rate = twist/rise, so measure the rate
    (dynamic slope) and project the start (twist0, rise0) onto that ridge — the nearest
    (twist, rise) with the measured ratio. `lr`/`iters` accepted for API symmetry."""
    base = fit_fixed_rise(filaments, rise0, None, rot_flip, tmin, tmax, n_grid)
    if not np.isfinite(base.rate):
        return base
    r = base.rate                                     # measured slope (deg/A), rise-independent
    s = (r * twist0 + rise0) / (r ** 2 + 1.0)         # Euclidean-nearest ridge point
    t = r * s
    res = fit_fixed_rise(filaments, s, None, rot_flip, tmin, tmax, n_grid)
    res.twist, res.rate, res.rise = t, r, s
    res.mode = "free"
    res.traj = np.array([(twist0, rise0), (t, s)])
    return res
