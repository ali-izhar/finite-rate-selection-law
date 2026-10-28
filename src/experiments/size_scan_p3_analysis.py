"""c(N) measured at p = 3 (size_scan_p3.json, plus the N=2048 12-draw run) against
the exact chain's c_CW(N) x 1.662 (exact_protocol_sizes.json) and against the
N^(-1/2) law of c ~ (N^2 gamma)^(-1/4) at a fixed schedule (paper Table 1, Sec. 5).
Reads the measured runs from $HOPFIELD_OUT (default results/experiments) and writes
size_scan_p3_analysis.json to the same folder; the exact chain is read from results/."""
import json, os, sys
import numpy as np
from scipy.special import logit
from scipy.integrate import quad
from scipy.stats import norm
sys.path.insert(0, '.')
from src.hopfield.stats import wilson

F = lambda x: 2 * norm.cdf(x) - 1
AMP3 = quad(lambda x: 1 - F(x) ** 3, 0, 20)[0] / np.sqrt(2 / np.pi)
ex = {int(k): v for k, v in json.load(open('results/experiments/exact_protocol_sizes.json')).items()}
DIR = os.environ.get('HOPFIELD_OUT', 'results/experiments')
meas = {}
if os.path.exists(os.path.join(DIR, 'size_scan_p3.json')):
    for k, v in json.load(open(os.path.join(DIR, 'size_scan_p3.json'))).items():
        meas[int(k)] = v
if os.path.exists(os.path.join(DIR, 'pattern_count_p3.json')):
    meas[2048] = json.load(open(os.path.join(DIR, 'pattern_count_p3.json')))
if not meas: sys.exit(f'no size_scan_p3.json or pattern_count_p3.json in {DIR}')
rng = np.random.default_rng(7)
print(f"{'N':>5} {'q_eq':>8} {'c_CW':>7} {'c_pred':>7} {'c_meas':>7} {'Wilson95':>17} {'draw-boot95':>17}   c/c(2048): meas  pred  N^-1/2")
rows = {}
for N in sorted(meas):
    v = meas[N]; ret = np.array(v['n_ret']); cor = np.array(v['n_cor'])
    s = cor[0] + (ret[1] - cor[1]); n = ret[0] + ret[1]
    l_eq = logit(ex[N]['q_eq']); c = float(logit(s.sum() / n.sum()) / l_eq)
    w = wilson(int(s.sum()), int(n.sum()))
    J = len(s); idx = rng.integers(0, J, size=(20000, J))
    bs = np.array([logit(s[i].sum() / n[i].sum()) / l_eq for i in idx]); lo, hi = np.percentile(bs, [2.5, 97.5])
    rows[N] = dict(c_meas=c, c_pred=ex[N]['c_cw'] * AMP3, c_cw=ex[N]['c_cw'], q_eq=ex[N]['q_eq'],
                   wilson=[float(logit(w.lower) / l_eq), float(logit(w.upper) / l_eq)], draw_boot=[float(lo), float(hi)], n=int(n.sum()))
for N in sorted(rows):
    r = rows[N]; r0 = rows.get(2048)
    rel = f"{r['c_meas']/r0['c_meas']:.3f} {r['c_pred']/r0['c_pred']:.3f} {(N/2048)**-0.5:.3f}" if r0 else ""
    print(f"{N:5d} {r['q_eq']:8.5f} {r['c_cw']:7.4f} {r['c_pred']:7.4f} {r['c_meas']:7.4f} [{r['wilson'][0]:.4f},{r['wilson'][1]:.4f}] [{r['draw_boot'][0]:.4f},{r['draw_boot'][1]:.4f}]   {rel}")
json.dump(rows, open(os.path.join(DIR, 'size_scan_p3_analysis.json'), 'w'), indent=1)
