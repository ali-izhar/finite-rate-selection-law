"""Sign changes of m after the selection window, on the nine linear ramps of the exact chain.

    python -m src.switching      # writes results/experiments/switching.json

The selection law assumes that a run keeps the sign of m that it has when it leaves the selection
window.  This module tests that with the exact (N+1)-state chain and no Monte Carlo.  At a split
point beta_s the law over k is cut into its m < 0 and m > 0 parts, each part is propagated to the
end of the ramp, and the mass that ends with the other sign is the fraction of runs that switch
after beta_s.  Split points: 1 + 3 sqrt(gamma) (three window widths after the transition) and 1.1.
Same ramp, field and integrator as src.cw_ramp, whose c_exact each row reproduces.

Paper: Sec. 3 ("rarely switch once the modes have separated"; "On the slowest ramp at N=1024, 12% of
runs still change the sign of m after beta = 1 + 3 sqrt(gamma)") and App. A.  Checked by checks.claims
(A25-A26).
"""
import argparse
import json

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import logit

from src.cw_ramp import banded_IE, magnetisation, q_eq
from src.rates import flip_rates
from src.simple.design import field

OUT = 'results/experiments/switching.json'


def evolve(P, beta, h, t, dt):
    """src.cw_ramp.evolve for several columns at once (P has shape (N+1, columns))."""
    N = P.shape[0] - 1
    n = int(round(t / dt)); dt = t / n
    A1 = banded_IE(N, beta, h, dt); A2 = banded_IE(N, beta, h, dt / 2)
    for _ in range(n):
        a = solve_banded((1, 1), A1, P)
        b = solve_banded((1, 1), A2, solve_banded((1, 1), A2, P))
        P = 2 * b - a
    return P


def switched(G, gamma, N, splits, b0=0.7, b1=1.6, db=0.005, dt=0.02):
    """(c_exact, [fraction of runs whose sign at each split differs from the final sign])."""
    m = magnetisation(N)
    wp, wm = flip_rates(N, b0, b0 * G, check=False)
    logpi = np.zeros(N + 1); logpi[1:] = np.cumsum(np.log(wp[:-1]) - np.log(wm[1:]))
    p = np.exp(logpi - logpi.max()); p /= p.sum()
    P = p[:, None]
    betas = np.arange(b0, b1 + 1e-12, db)
    tseg = db / gamma
    at = []                                   # grid beta at which each split was taken, in order
    todo = sorted(splits)
    for b in betas[:-1]:
        while todo and b >= todo[0] - 1e-12:
            full = P[:, 0]
            P = np.column_stack([P, full * (m < 0), full * (m > 0)])
            at.append(float(b)); todo.pop(0)
        bb = b + db / 2
        P = evolve(P, bb, bb * G, tseg, min(dt, tseg / 4))
    c = logit(P[m > 0, 0].sum()) / logit(q_eq(G, beta=b1, N=N))
    frac = [float(P[m > 0, 1 + 2 * j].sum() + P[m < 0, 2 + 2 * j].sum()) for j in range(len(at))]
    return float(c), at, frac


def main(out=OUT):
    rows = []
    for N in (1024, 2048, 4096):
        G = field(N)
        for gamma in (4e-3, 1e-3, 2.5e-4):
            splits = {'window3': 1 + 3 * np.sqrt(gamma), 'beta1.1': 1.1}
            order = sorted(splits, key=splits.get)
            c, at, frac = switched(G, gamma, N, list(splits.values()))
            row = dict(N=N, gamma=gamma, N_gamma=N * gamma, c_exact=c)
            for name, b, f in zip(order, at, frac):
                row[name] = dict(beta_split=b, switched=f)
            rows.append(row)
            print(row, flush=True)
    json.dump(rows, open(out, 'w'), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sign changes after the selection window on the nine linear ramps.")
    ap.add_argument("--out", default=OUT, help=f"output JSON (default {OUT})")
    main(ap.parse_args().out)
