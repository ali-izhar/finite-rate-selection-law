"""Zero-parameter prediction of q_model(G) on the exact lumped Curie-Weiss chain.

Propagates the full probability vector over k (up-spin count, N+1 states) through
the Hopfield protocol (paper Sec. 4) in continuous time (one unit = one sweep), with the
field h = beta*G at every rung (Hopfield energy E_G = E_0 - G N m^1, so the CW
tilt is beta*G).  No Monte Carlo, no fit.

Integrator: implicit Euler with Richardson extrapolation (second order, L-stable,
so the O(N) fast modes are damped, not aliased).  Banded tridiagonal solves.
N defaults to the paper's 2048 (protocol.N_SPINS); protocol() and q_eq() take N= for other sizes.

checks.claims calls protocol(G0, dt=0.02) for c_CW = 0.120.  Run as a script,
it writes results/experiments/exact_protocol.json (c_CW at each field; Table 3).
"""
import sys, json, time
import numpy as np
sys.path.insert(0, '.')
from scipy.linalg import solve_banded
from scipy.special import logit
from scipy.stats import binom
from src.rates import flip_rates
from src.hopfield import protocol as D

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

def protocol(G, dt=0.01, scale=1.0, ladder=D.LADDER, kred=D.K_REDESCENT, k_scarce=D.K_SCARCE,
             beta_rh=D.BETA_REHEAT, k_rh=D.K_REHEAT, N=D.N_SPINS):
    m = magnetisation(N)
    p = binom.pmf(np.arange(N + 1), N, 0.5)                    # 1. cold start
    for beta in ladder:                                        # 2. scarce descent
        p = evolve(p, beta, beta * G, k_scarce, dt)
    mis = (m < 0)                                              # 3. keep mis-committed
    f = p[mis].sum()
    p = np.where(mis, p, 0.0); p /= p.sum()
    p = evolve(p, beta_rh, beta_rh * G, k_rh, dt)              # 4. reheat
    for beta, K in zip(ladder, kred):                          # 5. redescent
        p = evolve(p, beta, beta * G, K * scale, dt)
    return f, p[m > 0].sum(), p

def q_eq(G, beta=D.BETA_DEEP, N=D.N_SPINS):
    wp, wm = flip_rates(N, beta, beta * G, check=False)
    logpi = np.zeros(N + 1)
    logpi[1:] = np.cumsum(np.log(wp[:-1]) - np.log(wm[1:]))
    pi = np.exp(logpi - logpi.max()); pi /= pi.sum()
    return pi[magnetisation(N) > 0].sum()

if __name__ == '__main__':
    v = json.load(open('results/hopfield/field_grid_analysis.json'))
    # convergence check at k=0
    G0 = v['points'][0]['field_g']
    for dt in (0.04, 0.02, 0.01):
        t0 = time.time(); f, q, _ = protocol(G0, dt=dt)
        print(f"dt={dt}: f_mis={f:.5f} q_model={q:.6f}  ({time.time()-t0:.1f}s)", flush=True)
    print(f"{'k':>2} {'G':>11} {'q_eq_exact':>10} {'q_eq_spec':>10} {'q_pred':>8} {'q_meas':>8} {'lo':>7} {'hi':>7} {'c_pred':>7} {'c_meas':>7}")
    rows = []
    for pt in v['points']:
        G = pt['field_g']; kk = pt['k']
        f, qm, _ = protocol(G, dt=0.01)
        qe = q_eq(G)
        c_pred = logit(qm) / logit(qe)
        rows.append(dict(k=kk, G=G, q_eq=qe, q_pred=qm, q_meas=pt['q_model']['point'], lo=pt['q_model']['lower'], hi=pt['q_model']['upper'], c_pred=c_pred, c_meas=pt['recovery']['point'], f_mis=f))
        print(f"{kk:2d} {G:+.4e} {qe:10.5f} {D.QEQ_SPEC[kk]:10.5f} {qm:8.4f} {pt['q_model']['point']:8.4f} {pt['q_model']['lower']:7.4f} {pt['q_model']['upper']:7.4f} {c_pred:7.4f} {pt['recovery']['point']:7.4f}   f_mis={f:.4f}", flush=True)
    json.dump(rows, open('results/experiments/exact_protocol.json', 'w'), indent=1)
