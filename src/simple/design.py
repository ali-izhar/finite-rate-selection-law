"""Constants of the simplified Hopfield test (docs/protocol-simple.md, locked 2026-10-27).

Changing a value here changes the experiment.  Run counts are in results/simple/predictions.json,
computed by ``python -m src.simple.predict`` from the precision target below.
"""
from __future__ import annotations

import numpy as np

BETA0 = 0.7            # ramp start; spins start independent and unbiased at this beta
BETA_F = 1.6           # ramp end; labels and the equilibrium reference are taken here
GAMMA = 0.004          # cooling rate, beta per sweep (one sweep = N single-spin updates)
NG = 6.125e-4 * 2048   # N*G fixed at every N, as in paper Table 2 (G = 6.125e-4 at N = 2048)

ROWS = ((2048, 1), (2048, 2), (2048, 3), (2048, 4),
        (1024, 1), (1024, 3), (4096, 1), (4096, 3))   # (N, p)

N_SETS = 50            # pattern sets per row, s = 0..49
SEED = 20261027        # pattern set s at size N: default_rng([SEED, N, s]), first p of 4 rows
HALF_WIDTH = 0.03      # target 95% half-width, as a fraction of the predicted c
N_BOOT = 10_000        # cluster bootstrap resamples
BOOT_SEED = 7
# Amendment 1 (2026-10-27, after the pilot, before any row ran): every row runs twice the counts of
# predictions.json, because the pilot showed between-set spread that can push p = 3 past 3%.
RUN_FACTOR = 2
# Amendment 2 (same date): the four gate rows G1 (3 rows) and G2 (1 row) are read at a joint 95%
# level, each at 1 - 0.05/4 = 98.75% (Bonferroni); a 95% interval per row fails by chance 19% of the time.
GATE_LEVEL = 0.9875


def field(n_spins: int) -> float:
    """|G| at size N (N*G fixed)."""
    return NG / n_spins


def n_updates(n_spins: int) -> int:
    """Updates in one run: N (beta_f - beta0) / gamma, an integer at every N used."""
    return int(round(n_spins * (BETA_F - BETA0) / GAMMA))


def beta_of_update(n: np.ndarray | int, n_spins: int) -> np.ndarray | float:
    """Inverse temperature of update n = 0..M-1: the ramp at the midpoint of the update."""
    m = n_updates(n_spins)
    return BETA0 + (BETA_F - BETA0) * (np.asarray(n) + 0.5) / m


def patterns(n_spins: int, s: int, p: int) -> np.ndarray:
    """Pattern set s: p i.i.d. +-1 patterns of length N, shape (p, N), int8.

    The first p of four rows drawn from one generator, so the sets are nested in p.
    """
    rng = np.random.default_rng([SEED, n_spins, s])
    return (2 * rng.integers(0, 2, size=(4, n_spins)) - 1).astype(np.int8)[:p]


def torch_seed(n_spins: int, p: int) -> int:
    """Seed of the GPU random stream of row (N, p)."""
    return SEED + 10 * n_spins + p
