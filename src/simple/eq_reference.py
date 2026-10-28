"""Exact Hopfield equilibrium sign log-odds at p = 3, against the CW value (paper App. B).

    python -m src.simple.eq_reference [N ...]     # default 512 1024 -> results/simple/eq_reference_p3.json

In the variables y_i = xi^1_i x_i, the spins split into four sets by (xi^1 xi^2, xi^1 xi^3).  The
energy depends on the configuration only through the four set sums, so the equilibrium law of the
label (largest |m^mu| is mu = 1, and the sign of m^1) is an exact sum over the four set counts,
weighted by binomial factors.  The ratio to the exact CW log-odds L_eq(N) is the error of using the
CW denominator in c-hat.  Pattern sets and constants are those of src/simple/design.py; the run
takes about 4 s per set at N = 512 and 80 s per set at N = 1024 on one core.
"""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.special import gammaln, logsumexp

from src.simple import design as D
from src.simple.predict import eq_law, sign_probs

OUT = os.path.join("results", "simple", "eq_reference_p3.json")
SIGNS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def block_sizes(xi: np.ndarray) -> list[int]:
    e2, e3 = xi[0] * xi[1], xi[0] * xi[2]
    return [int(((e2 == a) & (e3 == b)).sum()) for a, b in SIGNS]


def hopfield_leq(nb: list[int], n: int, beta: float, g: float) -> float:
    """log P(label (1,+)) - log P(label (1,-)) at equilibrium, ties excluded."""
    j = 1.0 / (n - 1)
    s = [2 * np.arange(m + 1) - m for m in nb]
    lb = [gammaln(m + 1) - gammaln(np.arange(m + 1) + 1) - gammaln(m - np.arange(m + 1) + 1) for m in nb]
    b, c, d = np.meshgrid(s[1], s[2], s[3], indexing="ij")
    lbcd = lb[1][:, None, None] + lb[2][None, :, None] + lb[3][None, None, :]
    acc = {1: [], -1: []}
    for ka, sa in enumerate(s[0]):
        m1, m2, m3 = sa + b + c + d, sa + b - c - d, sa - b + c - d
        lw = lb[0][ka] + lbcd + 0.5 * beta * j * (m1**2 + m2**2 + m3**2) + beta * g * m1
        win = (np.abs(m1) > np.abs(m2)) & (np.abs(m1) > np.abs(m3))
        for sg in (1, -1):
            sel = win & (sg * m1 > 0)
            if sel.any():
                acc[sg].append(logsumexp(lw[sel]))
    return float(logsumexp(acc[1]) - logsumexp(acc[-1]))


def cw_leq(n: int) -> float:
    lp, lm = sign_probs(eq_law(n, 1.0 / (n - 1), D.BETA_F, D.field(n)), n)
    return float(np.log(lp / lm))


def _one(args):
    n, s = args
    nb = block_sizes(D.patterns(n, s, 3).astype(int))
    return n, s, nb, hopfield_leq(nb, n, D.BETA_F, D.field(n))


def main(sizes=(512, 1024), workers: int = 12) -> dict:
    out = {"beta_f": D.BETA_F, "NG": D.NG, "pattern_sets": D.N_SETS}
    jobs = [(n, s) for n in sizes for s in range(D.N_SETS)]
    with Pool(workers) as pool:
        res = pool.map(_one, jobs)
    for n in sizes:
        l_cw = cw_leq(n)
        part = sorted((s, nb, h) for m, s, nb, h in res if m == n)
        ratios = [h / l_cw for _, _, h in part]
        out[str(n)] = dict(L_eq_CW=l_cw, blocks=[nb for _, nb, _ in part],
                           L_eq_Hopfield=[h for _, _, h in part], ratios=ratios,
                           mean=float(np.mean(ratios)), min=float(np.min(ratios)), max=float(np.max(ratios)))
        print(f"N={n}: Hopfield/CW mean {np.mean(ratios):.5f}, range {np.min(ratios):.5f}-{np.max(ratios):.5f}")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT}")
    return out


if __name__ == "__main__":
    main(tuple(int(a) for a in sys.argv[1:]) or (512, 1024))
