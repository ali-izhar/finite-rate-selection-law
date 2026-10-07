"""c_CW(N): the exact two-mode chain run through the experiment's own schedule at
several N (paper Table 1).  Same protocol as exact_protocol.py, with N a
parameter; the ladder, its dwell counts (fixed at the N=2048 values), G, the
reheat and the brief cooling are the same at every N.

Writes results/experiments/exact_protocol_sizes.json, which size_scan_p3_analysis.py,
sampler_clock.py and checks.claims read.
Prints c_CW(N) = logit q_model / logit q_eq at G = G0 and the ratio to N=2048,
against the (N^2 gamma)^(-1/4) law's N^(-1/2) at fixed schedule.

    python -m src.experiments.exact_protocol_sizes [512,1024,2048,4096] [dt=0.02]
"""
import sys, json, time
sys.path.insert(0, '.')
from scipy.special import logit
import src.experiments.exact_protocol as E
from src.hopfield import protocol as D

sizes = [int(s) for s in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['512', '1024', '2048', '4096'])]
dt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
G = D.G0; out = {}
for N in sizes:
    t0 = time.time()
    f, qm, _ = E.protocol(G, dt, N=N); qe = E.q_eq(G, N=N); c = float(logit(qm) / logit(qe))
    out[N] = dict(G=G, dt=dt, f_mis=float(f), q_model=float(qm), q_eq=float(qe), c_cw=c)
    print(f"N={N}: f_mis={f:.4f} q_eq={qe:.5f} q_model={qm:.5f} c_CW={c:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open('results/experiments/exact_protocol_sizes.json', 'w'), indent=1)
c0 = out.get(2048, {}).get('c_cw')
if c0:
    for N in sizes:
        print(f"N={N}: c_CW/c_CW(2048) = {out[N]['c_cw']/c0:.4f}   N^-1/2 law: {(N/2048)**-0.5:.4f}")
