"""Category (a): numbers recomputed live from exact chains in src/.

Every number comes from the functions the experiments use: ``src.cw_ramp``
(the linear ramps of Table 1), ``src.barrier`` (switching times, the zero-magnetization mass),
``src.simple.predict`` (the discrete-update chain of the Hopfield test: c_CW, the exact two-block
prediction, the equilibrium log-odds and A_p) and ``src.simple.reference`` (brute force at N = 10).
"""

from __future__ import annotations

import contextlib
import io
import math

import numpy as np
from scipy.special import logit, logsumexp, ndtr
from scipy.stats import binom

from src.cw_ramp import linear_sweep
from src.simple import design as SD
from src.simple.predict import amplification, eq_law, ramp_laws, sign_probs

from . import Result, check_bool, check_rounding

G0 = 0.0006125          # |G| at N = 2048; N G = 1.2544 at every N
SIZES = (1024, 2048, 4096)


def _c_cw(n: int) -> tuple[float, float, float]:
    """(c_CW, L_eq, c with an equilibrium start at beta0) for the discrete sampler's CW chain."""
    g, j = SD.field(n), 1.0 / (n - 1)
    lp, lm = sign_probs(eq_law(n, j, SD.BETA_F, g), n)
    l_eq = float(np.log(lp / lm))
    kk = np.arange(n + 1)
    laws = ramp_laws([n, n], [j, j], n, g, start=[binom.pmf(kk, n, 0.5), eq_law(n, j, SD.BETA0, g)])
    c = [float(np.log(np.divide(*sign_probs(law, n))) / l_eq) for law in laws]
    return c[0], l_eq, c[1]


def two_block(n: int = 2048, sets=range(SD.N_SETS)) -> dict:
    """Exact two-pattern prediction for frozen pattern sets (App. B), as src.simple.predict."""
    g, jb = SD.field(n), 2.0 / (n - 1)
    n_a = np.array([int((SD.patterns(n, s, 2).prod(axis=0) > 0).sum()) for s in sets])
    blocks = sorted(set(n_a.tolist()) | set((n - n_a).tolist()))
    laws = ramp_laws(blocks, [jb] * len(blocks), n, g)
    marg = {b: sign_probs(laws[i], b) for i, b in enumerate(blocks)}
    eqm = {b: sign_probs(eq_law(b, jb, SD.BETA_F, g), b) for b in blocks}
    win, corr, lset, leq = [], [], [], []
    for na in n_a:
        (ap, am), (bp, bm) = marg[int(na)], marg[int(n - na)]
        win.append(ap * bp + am * bm)
        corr.append(ap * bp)
        lset.append(np.log(ap / am) + np.log(bp / bm))
        (eap, eam), (ebp, ebm) = eqm[int(na)], eqm[int(n - na)]
        leq.append(np.log(eap / eam) + np.log(ebp / ebm))
    return dict(n_a=n_a, blocks=blocks, q_pool=float(np.sum(corr) / np.sum(win)),
                lset=np.array(lset), leq=np.array(leq))


def checks(shared: dict) -> list[Result]:
    out: list[Result] = []

    # ------------------------------------------ Sec. 2: worked example
    out.append(check_rounding("A00", "Sec. 1", "1.32", 4.0 ** 0.2, "4^0.2: equilibrium odds 4:1 at c = 0.2"))

    # ------------------------------------------ Sec. 3 and Table 1: linear ramps of the exact chain
    rows = []
    for N in SIZES:
        G = G0 * 2048 / N
        for gamma in (4e-3, 1e-3, 2.5e-4):
            q, qe = linear_sweep(G, gamma, N)
            a = G * math.sqrt(N) * (math.pi / gamma) ** 0.25
            rows.append((N, gamma, float(logit(q) / logit(qe)), float(logit(ndtr(a)) / logit(qe))))
    shared["linear"] = rows
    excess = [100.0 * (c / ck - 1.0) for _, _, c, ck in rows]
    out.append(check_bool(
        "A14", "Sec. 3, Fig. 1c caption, Table 1", "exact c exceeds the law by 0.23-1.81% on all nine ramps (within 2%)",
        min(excess) > 0 and f"{min(excess):.2f}" == "0.23" and f"{max(excess):.2f}" == "1.81",
        f"{min(excess):.3f}% .. {max(excess):.3f}%", "all positive; ends round to 2 dp",
        "nine linear ramps of the exact chain vs logit Phi(G sqrt N (pi/gamma)^1/4) / logit q_eq",
    ))
    out.append(check_rounding("A15", "Sec. 3", "0.26", min(N * g for N, g, _, _ in rows),
                              "smallest N gamma over the nine ramps"))
    for i, (N, g, c, _) in enumerate(r for r in rows if round(r[0] ** 2 * r[1]) == 4194):
        out.append(check_rounding(f"A16{'abc'[i]}", "Table 1", ("0.09370", "0.09372", "0.09361")[i], c,
                                  f"exact chain, N={N}, gamma={g:g}"))
    ratios = {N: max(c for n, _, c, _ in rows if n == N) / min(c for n, _, c, _ in rows if n == N)
              for N in SIZES}
    out.append(check_bool(
        "A16d", "Abstract, Sec. 3", "for a weak field (N G = 1.2544), sixteen times slower cooling only doubles the fraction",
        all(round(v) == 2 for v in ratios.values()),
        ", ".join(f"N={N}: {v:.3f}" for N, v in ratios.items()), "each ratio rounds to 2",
        "exact chain: c(gamma=0.00025) / c(gamma=0.004) at each N (a sixteenfold slower linear ramp)",
    ))
    a_fig = G0 * math.sqrt(2048) * (math.pi / 0.001) ** 0.25
    out.append(check_rounding("A16e", "Sec. 3", "58", 100.0 * float(ndtr(a_fig)),
                              "q_sel = Phi(a) in percent, a = G sqrt(N) (pi/gamma)^1/4 at N=2048, gamma=0.001, G=0.0006125"))
    out.append(check_rounding("A16f", "Sec. 3", "0.208", a_fig, "a for Fig. 1b"))
    out.append(check_rounding("A16g", "Sec. 3", "5", 1.0 / a_fig,
                              "noise about five times stronger than the field: 1/a"))
    out.append(check_rounding("A22", "Sec. 3", "3600", (1.6 - 0.7) / 2.5e-4,
                              "longest ramp: (1.6 - 0.7) / 0.00025 expected sweeps"))

    # ------------------------------------------ Sec. 3, App. A: switching time and zero-magnetization mass
    from src.barrier import log_mfpt, log_pi_closed_form
    lp = log_pi_closed_form(2048, 1.1)
    k_mode = int(np.argmax(lp[:1024]))
    switch = math.exp(log_mfpt(2048, 1.1, 0.0, k_mode, 2048 - k_mode))
    out.append(check_rounding("A21", "Sec. 3", "5", switch / 1e7,
                              "zero-field mean passage time between mode centres, N=2048, beta=1.1, in units of 1e7 sweeps (src.barrier)"))
    worst = max((log_pi_closed_form(n, 1.6)[n // 2] - logsumexp(log_pi_closed_form(n, 1.6))) / math.log(10)
                for n in (512, 1024, 2048, 4096, 16384))
    out.append(check_bool("A23", "App. A", "zero-magnetization mass below 10^-36 at beta_f = 1.6, N >= 512",
                          worst < -36, f"10^{worst:.1f} at most", "log10 < -36",
                          "src.barrier.log_pi_closed_form, N = 512..16384"))

    # ------------------------------------------ Sec. 3: sign changes after the selection window
    lp1 = log_pi_closed_form(1024, 1.1)
    k1 = int(np.argmax(lp1[:512]))
    out.append(check_rounding("A24", "Sec. 3", "5", math.exp(log_mfpt(1024, 1.1, 0.0, k1, 1024 - k1)) / 1e4,
                              "zero-field mean passage time between mode centres, N=1024, beta=1.1, in units of 1e4 sweeps (src.barrier)"))
    import json
    from . import ROOT
    sw = json.loads((ROOT / "results/experiments/switching.json").read_text())
    saved = {(r["N"], r["gamma"]): r["c_exact"] for r in json.loads((ROOT / "results/experiments/cw_linear_sweep.json").read_text())}
    late = [max((r["window3"], r["beta1.1"]), key=lambda d: d["beta_split"])["switched"] for r in sw]
    same = max(abs(r["c_exact"] - saved[(r["N"], r["gamma"])]) for r in sw)
    out.append(check_bool(
        "A25", "Abstract, Sec. 3", "runs rarely switch once the modes have separated",
        max(late) < 0.01 and same < 1e-9, f"at most {100 * max(late):.2f}% of runs on the nine ramps",
        "below 1% after the later of beta = 1.1 and 1 + 3 sqrt(gamma); c_exact equals the saved sweep",
        "results/experiments/switching.json (python -m src.switching)"))
    slow = {(r["N"], r["gamma"]): r["window3"]["switched"] for r in sw}
    out.append(check_bool(
        "A26", "Sec. 3, App. A", "on the slowest ramp at N = 1024, 12% of runs still change the sign of m three window widths after the transition",
        round(100 * slow[(1024, 2.5e-4)]) == 12 and slow[(2048, 2.5e-4)] > 0.02,
        f"{100 * slow[(1024, 2.5e-4)]:.1f}% (N=1024), {100 * slow[(2048, 2.5e-4)]:.1f}% (N=2048) at gamma=0.00025",
        "rounds to 12% (N=1024); above 2% (N=2048), after beta = 1 + 3 sqrt(gamma)",
        "results/experiments/switching.json (python -m src.switching)"))

    # ------------------------------------------ Sec. 4, Eq. 5: amplification
    amp = {p: float(amplification(p)) for p in (1, 2, 3, 4)}
    shared["amp"] = amp
    out.append(check_rounding("A09", "Sec. 4", "1.414", amp[2], "A_2 (= sqrt 2)"))
    out.append(check_rounding("A10", "Sec. 4", "1.662", amp[3], "A_3 by quadrature"))
    out.append(check_rounding("A11", "Sec. 4", "1.836", amp[4], "A_4 by quadrature"))

    # ------------------------------------------ Table 3, App. A, App. C: the discrete sampler's CW chain
    cw = {n: _c_cw(n) for n in SIZES}
    shared["c_cw"] = {n: v[0] for n, v in cw.items()}
    shared["l_eq"] = {n: v[1] for n, v in cw.items()}
    src_cw = "src.simple.predict.ramp_laws: CW chain advanced update by update, no time step"
    for n, printed in zip(SIZES, ("0.0937", "0.0659", "0.0464")):
        out.append(check_rounding(f"A30_{n}", "Table 3 (p = 1 prediction)", printed, cw[n][0],
                                  f"{src_cw}, N={n}"))
    table2 = {n: next(c for N, g, c, _ in rows if N == n and g == 4e-3) for n in SIZES}
    out.append(check_bool(
        "A31", "App. A", "the discrete-update chain reproduces the gamma = 0.004 column of Table 1 to the printed digits",
        all(f"{cw[n][0]:.5f}" == f"{table2[n]:.5f}" for n in SIZES),
        ", ".join(f"{cw[n][0]:.6f} vs {table2[n]:.6f}" for n in SIZES), "equal at 5 dp",
        "src.simple.predict.ramp_laws vs src.cw_ramp",
    ))
    out.append(check_bool(
        "A32", "App. C", "independent-spin start and equilibrium at beta = 0.7 give the same c to six digits",
        f"{cw[2048][0]:.6g}" == f"{cw[2048][2]:.6g}", f"{cw[2048][0]:.9f} vs {cw[2048][2]:.9f}",
        "equal at 6 significant digits", "exact CW chain at N = 2048, both start laws",
    ))
    for n, printed in zip(SIZES, ("3.5742", "3.5747", "3.5749")):
        out.append(check_rounding(f"A33_{n}", "App. C", printed, cw[n][1],
                                  f"exact CW equilibrium sign log-odds at beta_f = 1.6, N={n}"))
    a_run = {n: SD.field(n) * math.sqrt(n) * (math.pi / SD.GAMMA) ** 0.25 for n in SIZES}
    for n, printed in zip(SIZES, ("0.21", "0.15", "0.10")):
        out.append(check_rounding(f"A34_{n}", "App. C", printed, a_run[n], f"a = G sqrt(N) (pi/gamma)^1/4, N={n}"))
    expo = 0.5 * math.log2(cw[1024][0] / cw[4096][0])
    out.append(check_rounding("A35", "App. C", "0.506", expo, "exact CW size exponent: log2(c_CW(1024)/c_CW(4096)) / 2"))
    for (n, p), printed in (((1024, 3), "0.1558"), ((2048, 3), "0.1095"), ((4096, 3), "0.0772"), ((2048, 4), "0.1209")):
        out.append(check_rounding(f"A36_{n}_{p}", "Table 3", printed, amp[p] * cw[n][0], f"A_{p} c_CW at N={n}"))

    # ------------------------------------------ Sec. 4, App. C, App. B: exact two-pattern prediction
    # The full 50-set propagation takes minutes (src.simple.predict); the pooled value and the per-set
    # values come from results/simple/predictions.json, and two sets (the extreme block sizes) are
    # recomputed live here and must equal the stored per-set values.
    import json
    from . import ROOT
    p2 = json.loads((ROOT / "results/simple/predictions.json").read_text())
    p2rows = {(r["N"], r["p"]): r for r in p2["rows"]}
    p2 = p2["checks"]["p2"]
    l2 = cw[2048][1]
    n_a = np.array(p2["n_A"])
    pick = [int(np.argmin(n_a)), int(np.argmax(n_a))]
    tb = two_block(sets=pick)
    live_gap = max(abs(tb["lset"][i] / l2 - p2["c_per_set"][s]) for i, s in enumerate(pick))
    out.append(check_bool("A37live", "App. B", "exact two-block chain (stored per-set values reproduced)",
                          live_gap < 1e-12, f"sets {pick}: max diff {live_gap:.1e}", "< 1e-12",
                          "src.simple.predict.ramp_laws on the blocks of two pattern sets vs predictions.json"))
    c2 = float(p2rows[(2048, 2)]["c_pred"])
    shared["c_p2"] = c2
    tb = dict(n_a=n_a, leq=np.array([1.0]) * p2["L_eq_Hopfield_over_CW_mean"] * l2)
    out.append(check_rounding("A37", "App. B", "0.09363", c2,
                              "exact two-block chain pooled over the 50 pattern sets (predictions.json; A37live)"))
    out.append(check_rounding("A37t", "Table 3", "0.0936", c2, "as A37"))
    a2c = amp[2] * cw[2048][0]
    out.append(check_rounding("A39", "App. B", "0.09316", a2c, "A_2 c_CW at N = 2048"))
    out.append(check_rounding("A39b", "App. B", "0.5", 100.0 * (1.0 - a2c / c2),
                              "A_2 c_CW below the exact two-block value, percent"))
    live_na = [int((SD.patterns(2048, s, 2).prod(axis=0) > 0).sum()) for s in range(SD.N_SETS)]
    assert live_na == list(map(int, tb["n_a"])), "stored block sizes differ from the pattern sets"
    sizes = live_na + [2048 - x for x in live_na]
    out.append(check_bool("A40", "App. B", "blocks of 944-1104 spins in our pattern sets",
                          (min(sizes), max(sizes)) == (944, 1104), f"{min(sizes)}..{max(sizes)}", "equality",
                          "both blocks of the 50 sets at N = 2048: counts of xi^1 xi^2 = +1 and -1"))

    # ------------------------------------------ the chain itself (App. A)
    from src.validate import main as validate_main
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = validate_main()
    passed = "82/82 checks passed" in buf.getvalue()
    out.append(check_bool(
        "A20", "App. A", "rates, detailed balance, drift and noise of the exact chain",
        rc == 0 and passed, "82/82 checks passed" if passed else buf.getvalue().strip().splitlines()[-1],
        "all checks", "python -m src (src/validate.py)",
    ))
    return out
