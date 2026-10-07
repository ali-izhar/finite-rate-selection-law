"""Per-field intervals and the summary of the field grid.

Predicted direction, *compression toward 1/2*:
    k in {0,1,2,3}, q_eq > 1/2 : q_model interval entirely below q_eq's, U_k < a_k
    k in {4,5,6,7}, q_eq < 1/2 : q_model interval entirely above q_eq's, L_k > b_k
ANTI(k) is the opposite separation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logit

from . import protocol as D
from . import equilibrium as Q
from .stats import Interval, logit_variance, recovery_ratio, weighted_slope_through_origin, wilson


@dataclass
class PointResult:
    """Everything the paper reports at one grid point."""

    k: int
    g_b: float
    field_g: float
    n: int
    successes: int
    q_model: Interval
    q_eq: Interval          # centre + the error envelope [a_k, b_k]
    sep: bool
    anti: bool
    upper_below_half: bool
    recovery: Interval
    per_seed: list[tuple[int, int, int, float, float, float, float]]
    # (seed_index, successes, n, q_hat, lower, upper, q_eq_seed)


def pooled_mbar(
    mbar: dict[int, float], se: dict[int, float], n_by_seed: dict[int, int]
) -> tuple[float, float]:
    """n-weighted mean of the per-seed bar m, and its standard error.

    NOTE: q_eq is fixed per *pattern set* (route (b)), but the primary interval is
    *pooled* over sets.  The pooled q_eq is therefore evaluated
    at the n-weighted mean of the per-seed bar m, with the corresponding weighted
    SE.  The choice is immaterial: the per-seed bar m spread is ~1e-4 while the
    1% systematic allowance is 8.9e-3.
    """
    seeds = sorted(n_by_seed)
    w = np.array([n_by_seed[s] for s in seeds], dtype=np.float64)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    m = np.array([mbar[s] for s in seeds])
    e = np.array([se[s] for s in seeds])
    return float(np.dot(w, m)), float(np.sqrt(np.dot(w**2, e**2)))


def analyse_point(
    k: int, successes_by_seed: dict[int, int], n_by_seed: dict[int, int],
    mbar: dict[int, float], se_mbar: dict[int, float],
    *, n_spins: int = D.N_SPINS, beta: float = D.BETA_DEEP, z: float = D.Z_SCORE,
) -> PointResult:
    """Wilson interval, q_eq envelope, SEP/ANTI, and the recovery ratio at point k."""
    g = float(D.G_GRID[k])
    n = int(sum(n_by_seed.values()))
    s = int(sum(successes_by_seed.values()))
    q_model = wilson(s, n, z)

    m_pool, se_pool = pooled_mbar(mbar, se_mbar, n_by_seed)
    a, centre, b = Q.qeq_envelope(
        n_spins, beta, g, m_pool, se_pool, z=z, systematic=D.SYSTEMATIC_ENVELOPE
    )
    q_eq = Interval(centre, a, b)

    if k <= 3:
        sep = q_model.upper < a
        anti = q_model.lower > b
    else:
        sep = q_model.lower > b
        anti = q_model.upper < a

    per_seed = []
    for si in sorted(n_by_seed):
        ci = wilson(successes_by_seed[si], n_by_seed[si], z)
        per_seed.append(
            (si, successes_by_seed[si], n_by_seed[si], ci.point, ci.lower, ci.upper,
             Q.qeq_route_b(n_spins, beta, g, mbar[si]))
        )

    return PointResult(
        k=k, g_b=float(D.GB_GRID[k]), field_g=g, n=n, successes=s,
        q_model=q_model, q_eq=q_eq, sep=bool(sep), anti=bool(anti),
        upper_below_half=bool(q_model.upper < 0.5),
        recovery=recovery_ratio(q_model, q_eq), per_seed=per_seed,
    )


@dataclass
class Verdict:
    d1_ceiling_holds: bool
    d1_reason: str
    d2_separation_asserted: bool
    d2_reason: str
    d3_pooled_c: Interval
    points: list[PointResult]


def decide(points: list[PointResult]) -> Verdict:
    """D1 (primary), D2 (dose-response), D3 (recovery ratio).  D4 runs no test."""
    anti_points = [p.k for p in points if p.anti]
    upper_fail = [p.k for p in points if p.k >= 4 and not p.upper_below_half]
    d1 = not anti_points and not upper_fail
    d1_reason = (
        "no point separates in the anti-predicted direction and q_model's upper "
        "limit is below 1/2 at every k >= 4"
        if d1 else
        (f"ANTI fires at k={anti_points}; " if anti_points else "")
        + (f"q_model upper limit >= 1/2 at k={upper_fail}" if upper_fail else "")
    )

    sep_ks = [p.k for p in points if p.sep]
    sep_hi = [k for k in sep_ks if k <= 3]
    sep_lo = [k for k in sep_ks if k >= 4]
    d2 = len(sep_ks) >= 6 and len(sep_hi) >= 2 and len(sep_lo) >= 2
    d2_reason = (
        f"SEP holds at {len(sep_ks)}/8 points (k={sep_ks}); "
        f"{len(sep_hi)} with q_eq > 1/2, {len(sep_lo)} with q_eq < 1/2; "
        f"rule needs >= 6 including >= 2 each side"
    )

    x = np.array([logit(p.q_eq.point) for p in points])
    y = np.array([logit(min(max(p.q_model.point, 1e-12), 1 - 1e-12)) for p in points])
    v = np.array([logit_variance(p.successes, p.n) for p in points])
    c = weighted_slope_through_origin(x, y, v)

    return Verdict(d1, d1_reason, d2, d2_reason, c, points)


def report(v: Verdict) -> str:  # pragma: no cover - presentation only
    lines = []
    lines.append("D2  dose-response table (pooled over seeds)")
    lines.append(
        f"{'k':>2} {'g_b':>11} {'G':>12} {'n':>6} {'succ':>6} "
        f"{'q_model':>9} {'[L':>9} {'U]':>9} {'q_eq':>9} {'[a':>9} {'b]':>9} "
        f"{'SEP':>4} {'ANTI':>5} {'c_k':>7} {'[c_lo':>7} {'c_hi]':>7}"
    )
    for p in v.points:
        lines.append(
            f"{p.k:2d} {p.g_b:11.4e} {p.field_g:12.5e} {p.n:6d} {p.successes:6d} "
            f"{p.q_model.point:9.5f} {p.q_model.lower:9.5f} {p.q_model.upper:9.5f} "
            f"{p.q_eq.point:9.5f} {p.q_eq.lower:9.5f} {p.q_eq.upper:9.5f} "
            f"{'yes' if p.sep else 'no':>4} {'YES' if p.anti else 'no':>5} "
            f"{p.recovery.point:7.4f} {p.recovery.lower:7.4f} {p.recovery.upper:7.4f}"
        )
    lines.append("")
    lines.append(f"D1  ceiling holds = {v.d1_ceiling_holds}  ({v.d1_reason})")
    lines.append(f"D2  separation across the grid asserted = {v.d2_separation_asserted}")
    lines.append(f"    {v.d2_reason}")
    lines.append(
        f"D3  pooled recovery ratio c = {v.d3_pooled_c.point:.4f} "
        f"[{v.d3_pooled_c.lower:.4f}, {v.d3_pooled_c.upper:.4f}]  "
        f"(c < 1 shortfall, c = 1 attainment, c > 1 ceiling violation)"
    )
    lines.append(
        "    NOTE: that interval is statistical only -- it propagates the binomial "
        "variance of\n    logit q_model. The 1% systematic allowance on "
        "bar m scales\n    every logit q_eq, hence c, by at most 1%: "
        f"a systematic band of [{v.d3_pooled_c.point/1.01:.4f}, "
        f"{v.d3_pooled_c.point/0.99:.4f}] around the point estimate."
    )
    lines.append(
        "D4  selectivity is not tested: on this grid the type of error "
        "is not an independent variable."
    )
    lines.append("")
    lines.append("D2  per-draw (per-seed) q_model, as the main text requires")
    lines.append(f"{'k':>2} " + " ".join(f"s{p[0]:02d}" for p in v.points[0].per_seed))
    for p in v.points:
        lines.append(f"{p.k:2d} " + " ".join(f"{row[3]:.3f}" for row in p.per_seed))
    return "\n".join(lines)
