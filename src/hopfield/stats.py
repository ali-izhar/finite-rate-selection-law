"""Wilson score intervals and the derived intervals and slopes the analysis needs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logit


@dataclass(frozen=True)
class Interval:
    """A point estimate with a two-sided interval."""

    point: float
    lower: float
    upper: float

    def covers(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return f"{self.point:.5f} [{self.lower:.5f}, {self.upper:.5f}]"


def wilson(successes: int, trials: int, z: float = 1.96) -> Interval:
    """Wilson score interval, at ``z = 1.96``.

    STATUS: exact -- closed-form algebra, no approximation beyond the score
    statistic's own normal reference.
    """
    if trials <= 0:
        return Interval(float("nan"), 0.0, 1.0)
    if not 0 <= successes <= trials:
        raise ValueError(f"successes={successes} outside [0, {trials}]")
    n = float(trials)
    p_hat = successes / n
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)) / denom
    return Interval(p_hat, float(centre - half), float(centre + half))


def mover_sum(a: Interval, b: Interval) -> Interval:
    """Interval for ``p_a + p_b`` from two independent Wilson intervals (check 3).

    The MOVER (method of variance estimates recovery) combination of Newcombe:
    the lower limit uses each part's distance to its own lower limit, the upper
    limit its distance to its own upper limit.  Preferred over a Wald sum
    because it inherits Wilson's boundary behaviour, which matters at
    ``q_eq = 0.027``.

    STATUS: cited construction (Newcombe 1998), exact algebra given the parts.
    """
    point = a.point + b.point
    lower = point - float(np.hypot(a.point - a.lower, b.point - b.lower))
    upper = point + float(np.hypot(a.upper - a.point, b.upper - b.point))
    return Interval(point, lower, upper)


def _safe_logit(value: float) -> float:
    return float(logit(min(max(value, 1e-12), 1.0 - 1e-12)))


def recovery_ratio(q_model: Interval, q_eq: Interval) -> Interval:
    """c = logit(q_model) / logit(q_eq) with an interval, D3.

    Both intervals are transformed by the monotone ``logit`` and the quotient is
    taken over the four corners.  Valid because no grid point's ``logit q_eq``
    interval contains zero (the smallest is ``|logit q_eq| = 0.511`` at
    ``k = 3, 4`` against an envelope half-width of ~0.011).

    STATUS: exact interval arithmetic on the two input intervals.
    """
    num = [_safe_logit(q_model.lower), _safe_logit(q_model.upper)]
    den = [_safe_logit(q_eq.lower), _safe_logit(q_eq.upper)]
    if min(den) <= 0.0 <= max(den):
        raise ValueError("logit q_eq interval straddles 0; the ratio is unbounded")
    corners = [a / b for a in num for b in den]
    point = _safe_logit(q_model.point) / _safe_logit(q_eq.point)
    return Interval(point, float(min(corners)), float(max(corners)))


def weighted_slope_through_origin(
    x: np.ndarray, y: np.ndarray, var_y: np.ndarray, z: float = 1.96
) -> Interval:
    """Weighted least-squares slope ``c`` of ``y = c x`` through the origin, D3.

    ``x = logit q_eq``, ``y = logit q_model``, ``var_y`` the delta-method
    variance ``1 / (n p_hat (1 - p_hat))`` of ``logit q_hat``.  Weights are
    ``1 / var_y``; the reported interval is ``c +- z / sqrt(sum w x^2)``.

    STATUS: exact algebra given the inputs; the delta-method variance of
    ``logit q_hat`` is asymptotic in ``n`` (n >= 2000 here).
    """
    w = 1.0 / np.asarray(var_y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    denom = float(np.sum(w * x * x))
    slope = float(np.sum(w * x * y) / denom)
    half = z / np.sqrt(denom)
    return Interval(slope, slope - half, slope + half)


def logit_variance(successes: int, trials: int) -> float:
    """Delta-method variance of ``logit(k/n)``: ``1 / (n p (1-p))``.

    Uses the Wilson centre rather than the raw ``k/n`` so that the variance stays
    finite at ``k = 0`` or ``k = n``.
    """
    est = wilson(successes, trials)
    p = min(max(est.lower / 2.0 + est.upper / 2.0, 1e-9), 1.0 - 1e-9)
    return float(1.0 / (trials * p * (1.0 - p)))
