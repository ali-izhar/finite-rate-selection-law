"""GPU reruns of the Hopfield protocol (torch): pattern counts, size scans, one-pattern runs.

Each mode calls src.hopfield.pipeline.run_protocol at G = +/-6.125e-4 and writes
one file to $HOPFIELD_OUT (default results/experiments; set a scratch folder to
keep the saved results).  Run from the repository root:

    python -m src.experiments.hopfield_runs draws P
        p = P, pattern sets s = 3-14, 8000 runs per set and field sign, N = 2048
        (torch seed 9000 + P) -> pattern_count_p{P}.json: per-set n_ret (retained)
        and n_cor (retained, positive sign), shape (2 signs, 12 sets).
        P = 1..4 give the measured rows of paper Table 1.
    python -m src.experiments.hopfield_runs draws P N1,N2,... [B]
        the same at other N (torch seed 4000 + N) -> size_scan_p{P}.json, keyed by N.
        P = 3, N = 1024,4096 gives the size rows of Table 1.
    python -m src.experiments.hopfield_runs size-p1 [N1,N2,...] [B]
        p = 1, pattern set s = 0, 150,000 runs per field sign, N = 512..4096
        (torch seed 31337 + N) -> size_scan_p1.json, keyed by N: the pooled q and
        its logit.  sampler_clock.py divides these c(N) by c_CW(N) (paper Sec. 5,
        App. B); the N = 1024 logit enters App. C.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.special import logit

sys.path.insert(0, ".")
from src.hopfield import protocol as D
from src.hopfield import reference as H
from src.hopfield.pipeline import run_protocol

OUT = os.environ.get("HOPFIELD_OUT", "results/experiments")
G = D.G0
FIELDS = np.array([+G, -G])


def _out(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def _sizes(arg):
    return [int(s) for s in arg.split(",")]


def draws(P, sizes=None, B=8000):
    """p = P over the 12 draws (seeds 3-14): Table 1 at N = 2048, or the N scan at `sizes`."""
    seeds = list(range(3, 15))

    def run(n, torch_seed):
        t0 = time.time()
        xi_all = np.stack([H.patterns(D.seed_of(s), n, P) for s in seeds])
        _, n_ret, n_cor = run_protocol(xi_all, FIELDS, B, torch_seed=torch_seed)
        qp = n_cor[0].sum() / n_ret[0].sum(); qm = n_cor[1].sum() / n_ret[1].sum()
        print(f"N={n} p={P} POOLED: q+={qp:.4f} q-={qm:.4f} sym={(qp+1-qm)/2:.4f}  n={n_ret.sum()}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        return n_ret, n_cor, time.time() - t0

    if sizes is None:
        n_ret, n_cor, wall = run(D.N_SPINS, 9000 + P)
        out = dict(p=P, seeds=seeds, B=B, fields=FIELDS.tolist(), n_ret=n_ret.tolist(),
                   n_cor=n_cor.tolist(), wall=wall)
        json.dump(out, open(_out(f"pattern_count_p{P}.json"), "w"), indent=1)
        for j, s in enumerate(seeds):
            qp = n_cor[0, j] / n_ret[0, j]; qm = n_cor[1, j] / n_ret[1, j]
            print(f"p={P} seed={s:2d}: q+={qp:.4f} (n={n_ret[0,j]})  q-={qm:.4f} (n={n_ret[1,j]})  "
                  f"sym={(qp+1-qm)/2:.4f}", flush=True)
        return
    if P == 1:   # size_scan_p1.json has another format and other seeds; size-p1 owns it
        sys.exit("draws 1 with an N list would overwrite size_scan_p1.json; use size-p1")
    out = {}
    for n in sizes:
        n_ret, n_cor, wall = run(n, 4000 + n)
        out[n] = dict(p=P, N=n, seeds=seeds, B=B, fields=FIELDS.tolist(), n_ret=n_ret.tolist(),
                      n_cor=n_cor.tolist(), wall=wall)
        json.dump(out, open(_out(f"size_scan_p{P}.json"), "w"), indent=1)


def size_p1(sizes=(512, 1024, 2048, 4096), B=150_000):
    """p = 1, one draw (seed 0), at each N in `sizes`."""
    out = {}
    for n in sizes:
        t0 = time.time()
        xi = H.patterns(D.seed_of(0), n, 1)[None]                       # (1 draw, 1 pattern, n)
        _, n_ret, n_cor = run_protocol(xi, FIELDS, B, torch_seed=31337 + n)
        n_ret, n_cor = n_ret[:, 0], n_cor[:, 0]                         # (2 signs,)
        s = int(n_cor[0] + (n_ret[1] - n_cor[1])); tot = int(n_ret.sum())
        q = s / tot
        se = np.sqrt(q * (1 - q) / tot)
        out[n] = {"q": q, "n": tot, "logit": float(logit(q)),
                  "logit_lo": float(logit(q - 1.96 * se)), "logit_hi": float(logit(q + 1.96 * se)),
                  "n_ret": n_ret.tolist(), "n_cor": n_cor.tolist(), "wall": time.time() - t0}
        print(f"N={n:5d}: q={q:.5f} +-{1.96*se:.5f}  logit={logit(q):.5f} "
              f"[{logit(q-1.96*se):.5f},{logit(q+1.96*se):.5f}]  n={tot}  ({time.time()-t0:.0f}s)",
              flush=True)
        json.dump(out, open(_out("size_scan_p1.json"), "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["draws"] and len(a) >= 2:
        draws(int(a[1]), _sizes(a[2]) if len(a) > 2 else None, int(a[3]) if len(a) > 3 else 8000)
    elif a[:1] == ["size-p1"]:
        size_p1(_sizes(a[1]) if len(a) > 1 else (512, 1024, 2048, 4096), int(a[2]) if len(a) > 2 else 150_000)
    else:
        sys.exit(__doc__)
