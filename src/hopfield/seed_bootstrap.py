"""The field-grid slope c and its pattern-set bootstrap interval (paper App. B).

Reads results/hopfield/field_grid_analysis.json (per-set counts, 20 pattern sets
x 8 fields) and prints, with no rerun:

    sample-weighted c   = 0.1994
    equal-weight c      = 0.2101
    bootstrap 95%       = [0.1908, 0.2095]

Each pattern set's slope through the origin of logit q_chain against logit q_eq
weights a field by the inverse squared logit half-width of its 95% Wilson
interval. The grid estimate weights the sets by their retained samples. The
bootstrap resamples the 20 sets 20,000 times (seed 20260825).

    python -m src.hopfield.seed_bootstrap
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[2]
N_BOOT, SEED = 20_000, 20260825


def set_slopes() -> tuple[np.ndarray, np.ndarray]:
    """Per-set slopes and retained-sample weights, in set order."""
    points = json.loads((ROOT / "results/hopfield/field_grid_analysis.json").read_text())["points"]
    sets = sorted({int(r[0]) for p in points for r in p["per_seed"]})
    slopes, weights = [], []
    for s in sets:
        # per_seed row: set, successes, n, q, Wilson lower, Wilson upper, q_eq of the set
        r = np.array([next(r for r in p["per_seed"] if r[0] == s) for p in points])
        r = r[r[:, 2] > 0]
        x, y = logit(r[:, 6]), logit(r[:, 3])
        w = ((logit(r[:, 5]) - logit(r[:, 4])) / (2 * 1.96)) ** -2
        slopes.append(float(np.sum(w * x * y) / np.sum(w * x * x)))
        weights.append(float(r[:, 2].sum()))
    return np.array(slopes), np.array(weights)


def main() -> None:
    slopes, weights = set_slopes()
    ids = np.random.default_rng(SEED).integers(0, len(slopes), (N_BOOT, len(slopes)))
    boot = (slopes[ids] * weights[ids]).sum(axis=1) / weights[ids].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"pattern sets: {len(slopes)}, bootstrap resamples: {N_BOOT}")
    print(f"sample-weighted c = {np.average(slopes, weights=weights):.4f}")
    print(f"equal-weight c    = {slopes.mean():.4f}")
    print(f"bootstrap 95%     = [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
