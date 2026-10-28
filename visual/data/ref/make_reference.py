"""Reference values for lib/sim.test.mjs, computed with the paper's own code in ../src.

Run from the paper repo root (the parent of visual/):
    python visual/data/ref/make_reference.py
Writes visual/data/ref/reference.json.  Every value below comes from a function in
../src or from SciPy; nothing is re-derived here.
"""
import json
import sys

import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp, ndtr, logit
from scipy.stats import binom

sys.path.insert(0, '.')
from src.rates import flip_rates                       # noqa: E402
from src.barrier import log_pi_closed_form            # noqa: E402
from src.hopfield.equilibrium import qeq_route_a                   # noqa: E402
from src.hopfield import protocol as D                       # noqa: E402
import src.experiments.exact_protocol as E              # noqa: E402

ref = {"command": "python visual/data/ref/make_reference.py (from the repository root)"}

# 1. rates and stationary law ---------------------------------------------------------
pis = []
for N in (64, 256):
    for beta, h in ((0.8, 0.0), (1.2, 0.0), (1.6, 0.05), (1.0, -0.02)):
        wp, wm = flip_rates(N, beta, h)
        lp = log_pi_closed_form(N, beta, h)
        lp = lp - logsumexp(lp)
        pis.append(dict(N=N, beta=beta, h=h, W_plus=wp.tolist(), W_minus=wm.tolist(), log_pi=lp.tolist()))
ref["pi"] = pis

# 2. q_eq (route a, favoured mode m > 0, tie excluded, h = beta G) --------------------
qeqs = []
for N in (64, 256, 2048):
    for beta, G in ((1.6, D.G0), (1.6, D.G0 * 2048 / N), (1.2, 0.01), (1.6, -0.003)):
        qeqs.append(dict(N=N, beta=beta, G=G, q_eq=qeq_route_a(N, beta, G)))
ref["q_eq"] = qeqs

# 3. exact continuous-time propagation through the paper protocol ---------------------
def run_protocol(N, G, dt=0.02, scale=1.0):
    f, q, p = E.protocol(G, dt=dt, scale=scale, N=N)
    qe = E.q_eq(G, N=N)
    return dict(N=N, G=G, dt=dt, scale=scale, f_mis=float(f), q_chain=float(q), q_eq=float(qe),
                c=float(logit(q) / logit(qe)))

ref["protocol"] = [run_protocol(256, D.G0 * 8), run_protocol(512, D.G0 * 4)]

# a single level from the binomial start, to test the integrator alone
N = 128
p0 = binom.pmf(np.arange(N + 1), N, 0.5)
p1 = E.evolve(p0, 1.1, 1.1 * 0.003, 25.0, 0.02)
ref["evolve"] = dict(N=N, beta=1.1, h=1.1 * 0.003, t=25.0, dt=0.02, p=p1.tolist())

# 4. Gaussian order statistics A_p ---------------------------------------------------
F = lambda x, p: 1.0 - (2.0 * ndtr(x) - 1.0) ** p
ref["A_p"] = {p: quad(F, 0, 40, args=(p,), epsabs=1e-14, epsrel=1e-13, limit=200)[0] / np.sqrt(2 / np.pi)
              for p in range(1, 7)}

# 5. special functions ---------------------------------------------------------------
xs = [-38.0, -20.0, -8.0, -7.1, -5.0, -3.0, -1.0, -0.3, 0.0, 0.2, 0.7, 1.5, 2.5, 4.0, 7.0, 7.08, 9.0]
# Phi oracle: mpmath at 40 digits (scipy's ndtr itself is off by ~2e-13 relative below x = -30)
import mpmath as mp
mp.mp.dps = 40
ref["ndtr"] = [[x, float(mp.ncdf(mp.mpf(x)))] for x in xs + [-36.83, -33.42, -12.5, -2.6, -2.4, 2.45, 2.55]]

json.dump(ref, open('visual/data/ref/reference.json', 'w'))
print("wrote visual/data/ref/reference.json")
print("A_p:", ref["A_p"])
print("protocol:", ref["protocol"])
