"""Linear ramps of the exact Curie-Weiss chain (paper Table 2, Fig. 1b, Fig. 2a).

The probability vector over k (up-spin count, N+1 states) is propagated in continuous time (one
unit = one sweep) while beta rises linearly from b0 to b1 at rate gamma, with the field h = beta*G.
No Monte Carlo, no fit.  Integrator: implicit Euler with Richardson extrapolation (second order,
L-stable, so the O(N) fast modes are damped, not aliased).  Banded tridiagonal solves.

Kondepudi-Nelson (1983) for  dm/dt = (beta(t)-1) m - m^3/3 + h + sqrt(2D) xi,  beta = 1 + gamma t:
   P(favoured) = Phi( h (pi/gamma)^{1/4} / sqrt(D) ),   D = 1/N (heat-bath Glauber, sweep units)
so  logit q_model ~ (4/sqrt(2 pi)) * G * sqrt(N) * (pi/gamma)^{1/4}   for small G,
and c_KN = logit q_model / logit q_eq  with  logit q_eq = 2 N beta_f G mbar_f.

    python -m src.cw_ramp      # writes results/experiments/cw_linear_sweep.json
"""
import argparse
import json

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import logit, ndtr

from src.rates import flip_rates
from src.simple.design import field

OUT = 'results/experiments/cw_linear_sweep.json'


def banded_IE(N, beta, h, dt):
    """ab-form of (I - dt Q^T) for solve_banded with (l,u)=(1,1)."""
    wp, wm = flip_rates(N, beta, h, check=False)
    ab = np.zeros((3, N + 1))
    ab[1] = 1.0 + dt * (wp + wm)          # diagonal
    ab[0, 1:] = -dt * wm[1:]              # superdiag of Q^T: (Q^T)[k,k+1] = wm[k+1]
    ab[2, :-1] = -dt * wp[:-1]            # subdiag: (Q^T)[k+1,k] = wp[k]
    return ab


def evolve(p, beta, h, t, dt):
    """Propagate the law p over k = 0..N (N = len(p) - 1) for time t at fixed beta, h."""
    N = len(p) - 1
    n = int(round(t / dt)); dt = t / n
    A1 = banded_IE(N, beta, h, dt); A2 = banded_IE(N, beta, h, dt / 2)
    for _ in range(n):
        a = solve_banded((1, 1), A1, p)
        b = solve_banded((1, 1), A2, solve_banded((1, 1), A2, p))
        p = 2 * b - a
    return p


def magnetisation(N):
    """m = 2k/N - 1 for k = 0..N."""
    return 2 * np.arange(N + 1) / N - 1


def q_eq(G, beta, N):
    """Equilibrium mass of m > 0 at inverse temperature beta and field h = beta*G."""
    wp, wm = flip_rates(N, beta, beta * G, check=False)
    logpi = np.zeros(N + 1)
    logpi[1:] = np.cumsum(np.log(wp[:-1]) - np.log(wm[1:]))
    pi = np.exp(logpi - logpi.max()); pi /= pi.sum()
    return pi[magnetisation(N) > 0].sum()


def linear_sweep(G, gamma, N, b0=0.7, b1=1.6, db=0.005, dt=0.02):
    """(q_model, q_eq) after a linear ramp from an equilibrium start at b0 to b1."""
    wp, wm = flip_rates(N, b0, b0*G, check=False)
    logpi = np.zeros(N+1); logpi[1:] = np.cumsum(np.log(wp[:-1]) - np.log(wm[1:]))
    p = np.exp(logpi - logpi.max()); p /= p.sum()
    betas = np.arange(b0, b1 + 1e-12, db)
    tseg = db / gamma
    for b in betas[:-1]:
        bb = b + db/2
        p = evolve(p, bb, bb*G, tseg, min(dt, tseg/4))
    q = p[magnetisation(N) > 0].sum()
    qe = q_eq(G, beta=b1, N=N)
    return q, qe


def main(out=OUT):
    rows = []
    for N in (1024, 2048, 4096):
        G = field(N)
        for gamma in (4e-3, 1e-3, 2.5e-4):
            q, qe = linear_sweep(G, gamma, N)
            c = logit(q)/logit(qe)
            x = G*np.sqrt(N)*(np.pi/gamma)**0.25
            q_kn = ndtr(x); c_kn = logit(q_kn)/logit(qe)
            rows.append(dict(N=N, gamma=gamma, G=float(G), q_exact=float(q), q_KN=float(q_kn), c_exact=float(c), c_KN=float(c_kn), ratio=float(c/c_kn)))
            print(rows[-1], flush=True)
    json.dump(rows, open(out, 'w'), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Nine linear ramps of the exact Curie-Weiss chain (paper Table 2).")
    ap.add_argument("--out", default=OUT, help=f"output JSON (default {OUT})")
    main(ap.parse_args().out)
