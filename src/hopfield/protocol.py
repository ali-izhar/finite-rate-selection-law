"""Constants of the Hopfield experiment (paper Sec. 4, App. B, Tables 3-5).

Every number the sampler, the pipeline and the analysis use is defined here.
The values were fixed before the field-grid runs; changing one changes the
experiment.
"""

from __future__ import annotations

from typing import Final

import numpy as np

# ---------------------------------------------------------------- model ----
N_SPINS: Final[int] = 2048
"""N."""

N_PATTERNS: Final[int] = 3
"""p.  Loading alpha = p/N."""

ALPHA: Final[float] = N_PATTERNS / N_SPINS
"""1.46484e-3."""

# ------------------------------------------------------------ annealing ----
LADDER: Final[tuple[float, ...]] = (0.60, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.40, 1.60)
"""The 10 rungs.  Both descents use this ladder, ascending in beta."""

K_REDESCENT: Final[tuple[int, ...]] = (15, 30, 59, 107, 355, 145, 65, 31, 16, 11)
"""K(beta) = ceil(6 tau_ex(beta, 2048)) from the exact eigensolve (src.gap).

``sampler_checks.check_ladder_budget`` recomputes it
from ``src.gap`` and asserts agreement (a cross-check, not one of G1-G4)."""

TAU_EX_SPEC: Final[tuple[float, ...]] = (
    2.4984, 4.9696, 9.6820, 17.6916, 59.0024, 24.1475, 10.7311, 5.1194, 2.5406, 1.6946,
)
"""tau_ex(beta, 2048) from the exact eigensolve."""

BUDGET_CONSTANT: Final[float] = 6.0
"""C in K(beta) = ceil(C tau_ex)."""

K_SCARCE: Final[int] = 2
"""Sweeps per rung of the scarce initial descent.  Total 20 sweeps."""

BETA_REHEAT: Final[float] = 0.7
"""Reheat (paper Sec. 4)."""

K_REHEAT: Final[int] = 30
"""Reheat (paper Sec. 4)."""

SWEEPS_SCARCE: Final[int] = K_SCARCE * len(LADDER)          # 20
SWEEPS_REDESCENT: Final[int] = sum(K_REDESCENT)             # 834
SWEEPS_PER_CHAIN: Final[int] = SWEEPS_SCARCE + K_REHEAT + SWEEPS_REDESCENT  # 884

# ----------------------------------------------------------------- grid ----
DELTA: Final[float] = 1.75e-4
"""Grid increment in g_b."""

G0: Final[float] = 3.5 * DELTA
"""Reference tilt g_0 = 6.125e-4."""

N_GRID: Final[int] = 8

GB_GRID: Final[np.ndarray] = np.arange(N_GRID, dtype=np.float64) * DELTA
"""g_b^(k) = k * Delta, k = 0..7."""

G_GRID: Final[np.ndarray] = G0 - GB_GRID
"""G^(k) = g_0 - g_b^(k).  +6.125e-4 ... -6.125e-4."""

G_CAL: Final[float] = 0.0
"""Calibration point: g_b = g_0, hence G = 0, q_eq = 1/2 exactly."""

QEQ_SPEC: Final[tuple[float, ...]] = (
    0.9727397, 0.9277980, 0.8222999, 0.6249634, 0.3750366, 0.1777001, 0.0722020, 0.0272603,
)
"""q_eq^(k) route (a) at N=2048."""

MBAR_CW_SPEC: Final[float] = 0.8905367
"""bar m_CW(2048, 1.6), used by check G4."""

# ---------------------------------------------------------------- seeds ----
SEED_BASE: Final[int] = 20260823
SEED_STRIDE: Final[int] = 1009
N_SEEDS: Final[int] = 20
MAX_SEEDS: Final[int] = 25
"""Top-up cap."""


def seed_of(s: int) -> int:
    """seed_s = 20260823 + 1009 s, s = 0, 1, 2, ...  (paper Table 5)."""
    return SEED_BASE + SEED_STRIDE * s


# --------------------------------------------------------------- chains ----
B_DEFAULT: Final[int] = 2000
B_CAP: Final[int] = 6000
B_STEP: Final[int] = 500
FC_DEFAULT_THRESHOLD: Final[float] = 0.10
POOLED_TARGET: Final[int] = 2000
"""Pooled n target per grid point."""


def chains_per_seed(f: float, c: float) -> int:
    """B from the pilot run.

    ``B = 2000`` when ``f c >= 0.10``; otherwise ``ceil(200/(f c))`` rounded up
    to the next multiple of 500 and capped at 6000.
    """
    if f <= 0.0 or c <= 0.0:
        raise ValueError(f"pilot returned a degenerate f={f}, c={c}")
    if f * c >= FC_DEFAULT_THRESHOLD:
        return B_DEFAULT
    raw = int(np.ceil(200.0 / (f * c)))
    rounded = int(np.ceil(raw / B_STEP)) * B_STEP
    return min(rounded, B_CAP)


# ------------------------------------------------------------- pre-runs ----
B_PILOT: Final[int] = 2000
"""(P1) budget pilot: one seed, 2000 chains, full protocol at G = 0."""

PILOT_SEED_INDEX: Final[int] = 0
"""(P1) uses one pattern set: the first, s = 0.

The pilot runs at G = 0 only, so the choice cannot inform the results."""

B_CAL_CHAINS: Final[int] = 256
K_BURN: Final[int] = 60
K_MEAS: Final[int] = 200
BETA_MEASURE: Final[float] = 1.60
"""(P2) overlap calibration."""

# ------------------------------------------------------------ intervals ----
Z_SCORE: Final[float] = 1.96
SYSTEMATIC_ENVELOPE: Final[float] = 0.01
"""The 1% systematic allowance on bar m."""

GATE_G1_TOL: Final[float] = 1e-6
GATE_G4_ROUTE_TOL: Final[float] = 1e-10
GATE_G4_MBAR_TOL: Final[float] = 5.0 * ALPHA * MBAR_CW_SPEC
"""6.52e-3."""

BETA_DEEP: Final[float] = 1.60
