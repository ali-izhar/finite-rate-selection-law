"""Frozen predictions of the simplified Hopfield test (docs/protocol-simple.md).

    python -m src.simple.predict            # writes results/simple/predictions.json (CPU, ~15 min)

Everything is computed on exact birth-death chains, without sampling:

* c_CW(N): the CW chain of the GPU sampler itself -- one uniformly chosen spin per update,
  heat bath, coupling 1/(N-1), N updates per sweep, beta at the midpoint of each update
  (design.beta_of_update), spins independent and unbiased at beta0.  The law of the
  up-spin count k is propagated update by update (a tridiagonal kernel per update: no ODE, no
  time step).  By the gauge identity y = xi^1 * x (theory.md Sec. 1) this is ALSO the exact
  law of the p = 1 Hopfield sampler, for every pattern.
* p = 2: in the gauge the spins split into blocks A (xi^1 = xi^2) and B (xi^1 = -xi^2) that do
  not interact; each is a CW chain with coupling 2/(N-1) and field G.  The marginal of each
  block under the global random-site chain is propagated exactly (a block is hit with
  probability n_block/N per update).  The prediction treats the two blocks as independent;
  the only dependence is through the shared count of updates, O(1/N).
* p >= 3: A_p * c_CW(N), A_p = E max_{mu<=p}|Z_mu| / E|Z| (paper Eq. amplification).
* L_eq(N): the exact CW equilibrium sign log-odds at beta_f; the estimator's denominator for
  every p.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import logit, ndtr
from scipy.stats import binom

from src.simple import design as D

OUT = os.path.join("results", "simple", "predictions.json")


# --------------------------------------------------------------------------- chains
def _rows(sizes, couplings, g, n_global):
    """Padded per-row constants of the up/down probabilities of one update."""
    sizes = np.asarray(sizes)
    nmax = int(sizes.max())
    k = np.arange(nmax + 1, dtype=np.float64)[None, :]
    n = sizes[:, None].astype(np.float64)
    J = np.asarray(couplings, dtype=np.float64)[:, None]
    inside = k <= n
    a_up = J * (2 * k - n + 1) + g          # local field seen by a down spin (gauge frame)
    a_dn = J * (2 * k - n - 1) + g          # local field seen by an up spin
    w_up = np.where(inside, (n - k) / n_global, 0.0)   # P(pick a down spin of this block)
    w_dn = np.where(inside, k / n_global, 0.0)         # P(pick an up spin of this block)
    return a_up, a_dn, w_up, w_dn, inside


def ramp_laws(sizes, couplings, n_global, g, start=None):
    """Exact law of k for each row after one ramp of design.n_updates(n_global) updates.

    Row r is a CW block of sizes[r] spins with pair coupling couplings[r] and field g, embedded
    in a system of n_global spins updated by uniform random site selection.  ``start`` is the
    initial law per row (default: Binomial(n, 1/2), independent unbiased spins).
    """
    a_up, a_dn, w_up, w_dn, inside = _rows(sizes, couplings, g, n_global)
    if start is None:
        kk = np.arange(a_up.shape[1])
        p = np.stack([np.where(kk <= n, binom.pmf(kk, n, 0.5), 0.0) for n in sizes])
    else:
        p = np.array(start, dtype=np.float64)
    m = D.n_updates(n_global)
    betas = D.beta_of_update(np.arange(m), n_global)
    up = np.empty_like(p); dn = np.empty_like(p); stay = np.empty_like(p)
    for b in betas:
        np.tanh(b * a_up, out=up); up += 1.0; up *= 0.5 * w_up        # P(k -> k+1)
        np.tanh(b * a_dn, out=dn); np.subtract(1.0, dn, out=dn); dn *= 0.5 * w_dn   # P(k -> k-1)
        np.add(up, dn, out=stay); np.subtract(1.0, stay, out=stay)
        new = p * stay
        new[:, 1:] += p[:, :-1] * up[:, :-1]
        new[:, :-1] += p[:, 1:] * dn[:, 1:]
        p = new
    return p


def eq_law(n, coupling, beta, g):
    """Equilibrium law of k for one CW block (detailed balance of the same kernel)."""
    a_up, a_dn, w_up, w_dn, _ = _rows([n], [coupling], g, n)
    up = 0.5 * w_up[0] * (1 + np.tanh(beta * a_up[0]))
    dn = 0.5 * w_dn[0] * (1 - np.tanh(beta * a_dn[0]))
    lp = np.zeros(n + 1)
    lp[1:] = np.cumsum(np.log(up[:n]) - np.log(dn[1:n + 1]))
    pi = np.exp(lp - lp.max())
    return pi / pi.sum()


def sign_probs(p, n):
    """(P(S>0), P(S<0)) with S = 2k - n, for one row of length >= n+1."""
    k = np.arange(n + 1)
    return p[: n + 1][2 * k > n].sum(), p[: n + 1][2 * k < n].sum()


# ------------------------------------------------------------------------- A_p and law
def amplification(p):
    """A_p = E max_{mu<=p} |Z_mu| / E|Z| for i.i.d. standard normal Z."""
    emax = quad(lambda x: 1.0 - (2 * ndtr(x) - 1) ** p, 0, np.inf)[0]
    return emax / np.sqrt(2 / np.pi)


def c_selection_law(n_spins, l_eq):
    a = D.field(n_spins) * np.sqrt(n_spins) * (np.pi / D.GAMMA) ** 0.25
    return float(logit(ndtr(a)) / l_eq), float(a)


# --------------------------------------------------------------------------- run counts
def disorder_sd(n_spins, p, sd_p2=None):
    """Assumed between-set sd of the per-set c, as a fraction of c (sets K's share of error).

    p = 1: 0 (gauge: every pattern gives the same chain).  p = 2: the exact sd over the frozen
    sets.  p >= 3: the linear-Gaussian model of theory.md Sec. 2c (4% at N*gamma = 8.2, ~1.3% at
    32.8, ~9% at 2), scaled as (8.2/(N gamma))^0.65 sqrt(p/3).  Speculative; it only sets n.
    """
    if p == 1:
        return 0.0
    if p == 2:
        return sd_p2
    return 0.04 * (8.2 / (n_spins * D.GAMMA)) ** 0.65 * np.sqrt(p / 3)


def runs_needed(c_pred, l_eq, p, sd_rel):
    """Runs per (set, sign) so that the 95% half-width of c is <= HALF_WIDTH * c_pred."""
    se_t = D.HALF_WIDTH / 1.96
    room = se_t**2 - sd_rel**2 / D.N_SETS
    if room <= 0:
        raise ValueError("disorder alone exceeds the precision target; raise N_SETS")
    x = c_pred * l_eq
    q = 1 / (1 + np.exp(-x))
    n_label1 = 1.0 / (q * (1 - q) * x**2 * room)
    total = n_label1 * p                      # pattern 1 wins in ~1/p of the runs
    per_cell = int(np.ceil(total / (2 * D.N_SETS) / 100.0) * 100)
    return per_cell


# --------------------------------------------------------------------------------- main
def main(updates_per_second=1.5e8):
    t0 = time.time()
    out = dict(design={k: getattr(D, k) for k in
                       ("BETA0", "BETA_F", "GAMMA", "NG", "N_SETS", "SEED", "HALF_WIDTH")},
               rows=[], checks={})
    sizes = sorted({n for n, _ in D.ROWS})
    ref = {}
    for n in sizes:
        g = D.field(n)
        J = 1.0 / (n - 1)
        lp, lm = sign_probs(eq_law(n, J, D.BETA_F, g), n)
        l_eq = float(np.log(lp / lm))
        start_rows = None
        rows_sizes, rows_J = [n], [J]
        if n == 2048:   # start-law check: equilibrium at beta0 instead of independent spins
            rows_sizes, rows_J = [n, n], [J, J]
            kk = np.arange(n + 1)
            start_rows = [binom.pmf(kk, n, 0.5), eq_law(n, J, D.BETA0, g)]
        laws = ramp_laws(rows_sizes, rows_J, n, g, start=start_rows)
        qp, qm = sign_probs(laws[0], n)
        c_cw = float(np.log(qp / qm) / l_eq)
        ref[n] = dict(c_CW=c_cw, L_eq=l_eq, q_CW=float(qp / (qp + qm)))
        if n == 2048:
            qp2, qm2 = sign_probs(laws[1], n)
            out["checks"]["c_CW_eq_start_at_beta0"] = float(np.log(qp2 / qm2) / l_eq)
        print(f"N={n}: c_CW={c_cw:.6f}  L_eq={l_eq:.6f}  [{time.time()-t0:.0f}s]", flush=True)

    # p = 2 at N = 2048: exact two-block prediction for each frozen pattern set
    n = 2048; g = D.field(n); Jb = 2.0 / (n - 1)
    n_a = np.array([int((D.patterns(n, s, 2).prod(axis=0) > 0).sum()) for s in range(D.N_SETS)])
    blocks = sorted(set(n_a.tolist()) | set((n - n_a).tolist()))
    laws = ramp_laws(blocks, [Jb] * len(blocks), n, g)
    marg = {b: sign_probs(laws[i], b) for i, b in enumerate(blocks)}
    eqm = {b: sign_probs(eq_law(b, Jb, D.BETA_F, g), b) for b in blocks}
    win, corr, lset, leq_h = [], [], [], []
    for na in n_a:
        (ap, am), (bp, bm) = marg[int(na)], marg[int(n - na)]
        win.append(ap * bp + am * bm); corr.append(ap * bp)
        lset.append(np.log(ap / am) + np.log(bp / bm))
        (eap, eam), (ebp, ebm) = eqm[int(na)], eqm[int(n - na)]
        leq_h.append(np.log(eap / eam) + np.log(ebp / ebm))
    win, corr, lset, leq_h = map(np.array, (win, corr, lset, leq_h))
    q_pool = corr.sum() / win.sum()
    c2 = float(logit(q_pool) / ref[n]["L_eq"])
    c2_sets = lset / ref[n]["L_eq"]
    sd_p2 = float(c2_sets.std(ddof=1) / c2_sets.mean())
    out["checks"]["p2"] = dict(
        n_A=n_a.tolist(), c_per_set=c2_sets.tolist(), sd_rel=sd_p2,
        A2_cCW=float(amplification(2) * ref[n]["c_CW"]),
        L_eq_Hopfield_over_CW_mean=float((leq_h / ref[n]["L_eq"]).mean()),
        P_pattern1_wins_mean=float(win.mean()))
    print(f"p=2 exact blocks: c={c2:.6f}  (A_2 c_CW={amplification(2)*ref[n]['c_CW']:.6f}) "
          f"per-set sd={100*sd_p2:.2f}%  [{time.time()-t0:.0f}s]", flush=True)

    tot_hours = 0.0
    for n, p in D.ROWS:
        r = ref[n]
        a_p = float(amplification(p))
        c_sel, a = c_selection_law(n, r["L_eq"])
        if p == 1:
            pred, kind = r["c_CW"], "exact (gauge identity)"
        elif p == 2:
            pred, kind = c2, "exact two-block chain"
        else:
            pred, kind = a_p * r["c_CW"], "A_p * c_CW"
        sd = disorder_sd(n, p, sd_p2)
        per_cell = runs_needed(pred, r["L_eq"], p, sd)
        runs = per_cell * 2 * D.N_SETS
        upd = runs * D.n_updates(n)
        hours = upd / updates_per_second / 3600
        tot_hours += hours
        out["rows"].append(dict(
            N=n, p=p, G=D.field(n), N_gamma=n * D.GAMMA, updates_per_run=D.n_updates(n),
            c_CW=r["c_CW"], L_eq=r["L_eq"], A_p=a_p, c_pred=pred, prediction=kind,
            A_p_c_sel=a_p * c_sel, a=a, logit_q_pred=pred * r["L_eq"],
            q_pred=float(1 / (1 + np.exp(-pred * r["L_eq"]))),
            disorder_sd_assumed=sd, runs_per_set_per_sign=per_cell, runs=runs,
            spin_updates=float(upd), gpu_hours_at_1p5e8=hours))
        print(f"N={n} p={p}: pred c={pred:.5f} ({kind}); A_p={a_p:.4f}; A_p c_sel={a_p*c_sel:.5f}; "
              f"runs={runs} ({per_cell}/set/sign); {hours:.2f} GPU h", flush=True)
    out["gpu_hours_total_at_1p5e8"] = tot_hours
    out["wall_seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"total {tot_hours:.1f} GPU h at {updates_per_second:.1e} updates/s; wrote {OUT}")


if __name__ == "__main__":
    main()
