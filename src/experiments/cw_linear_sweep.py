"""Kondepudi-Nelson test on the exact chain: linear sweep of beta through 1 at rate gamma.

KN (1983) for  dm/dt = (beta(t)-1) m - m^3/3 + h + sqrt(2D) xi,  beta = 1 + gamma t:
   P(favoured) = Phi( h (pi/gamma)^{1/4} / sqrt(D) ),   D = 1/N (heat-bath Glauber, sweep units)
so  logit q_model ~ (4/sqrt(2 pi)) * G * sqrt(N) * (pi/gamma)^{1/4}   for small G,
and c_KN = logit q_model / logit q_eq  with  logit q_eq = 2 N beta_f G mbar_f.
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
from scipy.special import logit, ndtr
import src.experiments.exact_protocol as E
from src.rates import flip_rates
from src.hopfield import protocol as D

def linear_sweep(G, gamma, N, b0=0.7, b1=1.6, db=0.005, dt=0.02):
    # start in equilibrium at b0 (what the reheat stage produces)
    wp, wm = flip_rates(N, b0, b0*G, check=False)
    logpi = np.zeros(N+1); logpi[1:] = np.cumsum(np.log(wp[:-1]) - np.log(wm[1:]))
    p = np.exp(logpi - logpi.max()); p /= p.sum()
    betas = np.arange(b0, b1 + 1e-12, db)
    tseg = db / gamma
    for b in betas[:-1]:
        bb = b + db/2
        p = E.evolve(p, bb, bb*G, tseg, min(dt, tseg/4))
    q = p[E.magnetisation(N) > 0].sum()
    qe = E.q_eq(G, beta=b1, N=N)
    return q, qe


if __name__ == "__main__":
    rows = []
    for N in (1024, 2048, 4096):
        G = D.G0 * 2048 / N
        for gamma in (4e-3, 1e-3, 2.5e-4):
            q, qe = linear_sweep(G, gamma, N)
            c = logit(q)/logit(qe)
            x = G*np.sqrt(N)*(np.pi/gamma)**0.25
            q_kn = ndtr(x); c_kn = logit(q_kn)/logit(qe)
            rows.append(dict(N=N, gamma=gamma, G=float(G), q_exact=float(q), q_KN=float(q_kn), c_exact=float(c), c_KN=float(c_kn), ratio=float(c/c_kn)))
            print(rows[-1], flush=True)
    json.dump(rows, open('results/experiments/cw_linear_sweep.json','w'), indent=1)
