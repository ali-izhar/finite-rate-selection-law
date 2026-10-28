"""The one estimator of the simplified Hopfield test (docs/protocol-simple.md).

counts has shape (K sets, 2 signs, 2): for set s and field sign j (0: +|G|, 1: -|G|),
counts[s, j, 0] = runs labelled pattern 1 (largest |m^mu| is mu = 1, no tie, m^1 != 0),
counts[s, j, 1] = of those, runs whose m^1 has the sign of the field.

    c_hat = logit( sum counts[..., 1] / sum counts[..., 0] ) / L_eq

with L_eq the exact CW equilibrium sign log-odds at beta_f (results/simple/predictions.json).
Interval: percentile cluster bootstrap over pattern sets (a resampled set brings both signs).
"""
from __future__ import annotations

import numpy as np
from scipy.special import logit

from src.simple import design as D


def c_hat(counts: np.ndarray, l_eq: float) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    return float(logit(counts[..., 1].sum() / counts[..., 0].sum()) / l_eq)


def estimate(counts: np.ndarray, l_eq: float, n_boot: int = D.N_BOOT,
             seed: int = D.BOOT_SEED, level: float = 0.95) -> dict:
    counts = np.asarray(counts, dtype=np.float64)
    k = counts.shape[0]
    per_set = counts.sum(axis=1)                       # (K, 2): signs pooled
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, k, size=(n_boot, k))
    tot = per_set[draws].sum(axis=1)                   # (n_boot, 2)
    boot = logit(tot[:, 1] / tot[:, 0]) / l_eq
    lo, hi = np.percentile(boot, [50 * (1 - level), 50 * (1 + level)])
    n1 = counts[..., 0].sum(axis=0); m1 = counts[..., 1].sum(axis=0)
    return dict(c=c_hat(counts, l_eq), lo=float(lo), hi=float(hi), n_pattern1=int(n1.sum()),
                q=float(m1.sum() / n1.sum()),
                q_plus=float(m1[0] / n1[0]), q_minus=float(m1[1] / n1[1]))


def reading(est: dict, c_pred: float) -> dict:
    """Ratio to the frozen prediction and the pre-registered PASS flag (interval contains it)."""
    return dict(ratio=est["c"] / c_pred, ratio_lo=est["lo"] / c_pred, ratio_hi=est["hi"] / c_pred,
                passed=bool(est["lo"] <= c_pred <= est["hi"]))
