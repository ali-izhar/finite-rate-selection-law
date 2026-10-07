"""c(N, schedule scale) on the exact chain (paper Table 3): test c ~ N^-1/2 and c ~ scale^-1/4.

Writes results/experiments/cw_scaling.json; visual/data/make_exports.py copies it into
visual/data/cw_scaling.json for the visual edition.
"""
import sys, json
sys.path.insert(0, '.')
from scipy.special import logit
import src.experiments.exact_protocol as E
from src.hopfield import protocol as D
rows = []
for N in (512, 1024, 2048, 4096, 8192):
    G = D.G0 * 2048 / N
    f, q, _ = E.protocol(G, dt=0.02, N=N)
    qe = E.q_eq(G, N=N)
    c = logit(q)/logit(qe)
    rows.append(dict(N=N, scale=1.0, G=float(G), q_model=float(q), q_eq=float(qe), c=float(c))); print(rows[-1], flush=True)
for scale in (0.25, 0.5, 2.0, 4.0, 8.0):
    f, q, _ = E.protocol(D.G0, dt=0.02, scale=scale)
    qe = E.q_eq(D.G0); c = logit(q)/logit(qe)
    rows.append(dict(N=2048, scale=scale, G=float(D.G0), q_model=float(q), q_eq=float(qe), c=float(c))); print(rows[-1], flush=True)
json.dump(rows, open('results/experiments/cw_scaling.json','w'), indent=1)
