"""Pattern-set analysis of the p = 1, 2, 3, 4 runs (paper Table 1).

Reads pattern_count_p{p}.json (12 pattern draws x 8000 chains per sign of G)
from $HOPFIELD_OUT (default results/experiments) and writes
draw_level_analysis.json to the same folder.  For each p: the symmetrised q per draw, c per draw
(logit q / logit q_eq), the pooled c with a Wilson interval, and a draw-level
(cluster) bootstrap.  Ratios c(p)/c(p=1) with a cluster bootstrap interval,
against the order-statistic amplification E[max_p |Z|]/E|Z|.
"""
import json, os, sys
import numpy as np
from scipy.special import logit
from scipy.stats import norm
from scipy.integrate import quad
sys.path.insert(0, '.')
from src.hopfield.stats import wilson

L_EQ = float(logit(0.9727397))
F = lambda x: 2 * norm.cdf(x) - 1
AMP = {p: quad(lambda x: 1 - F(x) ** p, 0, 20)[0] / np.sqrt(2 / np.pi) for p in (1, 2, 3, 4)}
C_CW = 0.1198
rng = np.random.default_rng(2026); B_BOOT = 20000
DIR = os.environ.get('HOPFIELD_OUT', 'results/experiments')

def load(p):
    f = os.path.join(DIR, f'pattern_count_p{p}.json')
    if not os.path.exists(f): return None
    v = json.load(open(f)); ret = np.array(v['n_ret']); cor = np.array(v['n_cor'])
    s = cor[0] + (ret[1] - cor[1]); n = ret[0] + ret[1]          # per draw, symmetrised
    return s, n

def c_of(s, n): return float(logit(s / n) / L_EQ)

runs = {p: load(p) for p in (1, 2, 3, 4)}
runs = {p: v for p, v in runs.items() if v is not None}
if not runs: sys.exit(f'no pattern_count_p*.json in {DIR}')
per_draw = {p: np.array([c_of(si, ni) for si, ni in zip(*runs[p])]) for p in runs}
pooled = {p: c_of(runs[p][0].sum(), runs[p][1].sum()) for p in runs}
boot = {}
for p, (s, n) in runs.items():
    J = len(s); idx = rng.integers(0, J, size=(B_BOOT, J))
    boot[p] = np.array([c_of(s[i].sum(), n[i].sum()) for i in idx])
print(f"{'p':>2} {'amp':>6} {'c_pred':>7} {'c_pooled':>9} {'Wilson95':>17} {'draw-boot95':>17} {'draw sd':>8}  n_chains")
out = {}
for p in sorted(runs):
    s, n = runs[p]; w = wilson(int(s.sum()), int(n.sum()))
    lo, hi = np.percentile(boot[p], [2.5, 97.5])
    print(f"{p:2d} {AMP[p]:6.3f} {C_CW*AMP[p]:7.4f} {pooled[p]:9.4f} [{logit(w.lower)/L_EQ:.4f},{logit(w.upper)/L_EQ:.4f}] [{lo:.4f},{hi:.4f}] {per_draw[p].std(ddof=1):8.4f}  {int(n.sum())}")
    out[p] = dict(amplification=AMP[p], c_pred=C_CW * AMP[p], c_pooled=pooled[p],
                  wilson=[float(logit(w.lower) / L_EQ), float(logit(w.upper) / L_EQ)],
                  draw_boot=[float(lo), float(hi)], per_draw=per_draw[p].tolist(), n=int(n.sum()))
if 1 in runs:
    print("\nratios to p=1 (draw-level bootstrap, draws resampled independently per p):")
    for p in sorted(runs):
        if p == 1: continue
        r = boot[p] / boot[1]; lo, hi = np.percentile(r, [2.5, 97.5])
        z = (pooled[p] / pooled[1] - AMP[p]) / r.std(ddof=1)
        print(f"  c({p})/c(1) = {pooled[p]/pooled[1]:.3f}  boot95 [{lo:.3f},{hi:.3f}]  sd {r.std(ddof=1):.3f}   predicted {AMP[p]:.3f}  ({z:+.2f} sd)")
        out[p]['ratio_to_p1'] = dict(point=pooled[p] / pooled[1], boot95=[float(lo), float(hi)], sd=float(r.std(ddof=1)),
                                     predicted=AMP[p], z=float(z))
json.dump(out, open(os.path.join(DIR, 'draw_level_analysis.json'), 'w'), indent=1)
