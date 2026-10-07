"""Category (a): numbers recomputed live from the exact (N+1)-state chain in src/.

Every number comes from the functions the experiments use: ``src.gap``
(relaxation times), ``src.experiments.exact_protocol`` (c_CW and the linear
ramps) and ``src.hopfield.equilibrium`` (q_eq).
"""

from __future__ import annotations

import contextlib
import io
import math

from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import logit, ndtr
from scipy.stats import norm

from src.gap import gap_full, gap_well
from src.hopfield import protocol as D
from src.hopfield.equilibrium import qeq_route_a
import src.experiments.exact_protocol as E
from src.experiments.cw_linear_sweep import linear_sweep

from . import Result, check_bool, check_rounding, check_sci


def tracking(n: int, beta: float, h: float = 0.0) -> float:
    """Relaxation time in sweeps: inverse gap of the chain reflected at m = 0."""
    return 1.0 / gap_well(n, beta, h)


def peak(n: int) -> tuple[float, float]:
    """(beta, tracking time) at the finite-N maximum, by bounded 1-D maximisation."""
    r = minimize_scalar(lambda b: -tracking(n, b), bounds=(0.9, 1.25),
                        method="bounded", options={"xatol": 1e-7})
    return float(r.x), float(-r.fun)


def amplification(p: int) -> float:
    """E[max_p |Z|] / E|Z| for p independent standard normals (Eq. 6, App. C)."""
    F = lambda x: 2.0 * norm.cdf(x) - 1.0          # noqa: E731  CDF of |Z|
    return quad(lambda x: 1.0 - F(x) ** p, 0.0, 20.0)[0] / math.sqrt(2.0 / math.pi)


def checks(shared: dict) -> list[Result]:
    out: list[Result] = []
    REL = "docs/relaxation-times.md"

    # ------------------------------------------ relaxation times
    beta_pk, tau_pk = peak(16384)
    out.append(check_rounding("A01", REL, "76.94", tau_pk,
                              "max over beta of 1/gap_well(16384, beta), bounded maximiser"))
    out.append(check_rounding("A02", REL, "1.012", beta_pk, "argmax of the same"))
    out.append(check_rounding("A03", "App. B", "1.6946", tracking(D.N_SPINS, D.BETA_DEEP),
                              "1/gap_well(2048, 1.6): reflected-chain relaxation time"))
    inv = [1.0 / gap_full(n, 1.0) for n in (1024, 2048, 4096)]
    out.append(check_bool(
        "A04", "App. A", "42.2, 59.0, 82.7 sweeps", [f"{t:.1f}" for t in inv] == ["42.2", "59.0", "82.7"],
        ", ".join(f"{t:.3f}" for t in inv), "each rounds to 1 dp",
        "1/gap_full(N, 1.0) for N = 1024, 2048, 4096",
    ))

    # ------------------------------------------ c_CW = 0.120 (Sec. 4, Table 3)
    _, q_model, _ = E.protocol(D.G0, dt=0.02)
    q_eq = float(E.q_eq(D.G0))
    c_cw = float(logit(q_model) / logit(q_eq))
    shared["c_cw"], shared["logit_q_eq"] = c_cw, float(logit(q_eq))
    out.append(check_rounding(
        "A05", "Sec. 4, Table 1 caption", "0.120", c_cw,
        "src.experiments.exact_protocol.protocol(G0, dt=0.02): logit(q_model)/logit(q_eq)",
    ))
    _, q_half, _ = E.protocol(D.G0, dt=0.01)
    out.append(check_sci(
        "A06", "Table 3 caption", "2.8", -11, abs(c_cw - float(logit(q_half) / logit(q_eq))),
        "|c_CW(dt=0.02) - c_CW(dt=0.01)|, src.experiments.exact_protocol.protocol(G0, dt)",
    ))
    out.append(check_rounding("A07", "App. C", "3.575", float(logit(q_eq)),
                              "logit of the exact CW q_eq at N=2048, beta=1.6, G=0.0006125"))

    # ------------------------------------------ amplification (Sec. 4, Eq. 6)
    amp = {p: amplification(p) for p in (1, 2, 3, 4)}
    shared["amp"] = amp
    out.append(check_rounding("A08", "Sec. 4", "1.414", amp[2], "E[max_2|Z|]/E|Z| (= sqrt 2)"))
    out.append(check_rounding("A09", "Sec. 4", "1.662", amp[3], "E[max_3|Z|]/E|Z| by quadrature"))
    out.append(check_rounding("A10", "Sec. 4", "1.836", amp[4], "E[max_4|Z|]/E|Z|"))
    for i, (p, printed) in enumerate(((2, "0.169"), (3, "0.199"), (4, "0.220"))):
        out.append(check_rounding(f"A1{1 + i}", "Sec. 4, Table 1" if p == 3 else "Table 1", printed,
                                  c_cw * amp[p], f"c_CW x A_{p}"))

    # ------------------------------------------ linear ramps (Sec. 3, Table 2)
    rows = []
    for N in (1024, 2048, 4096):
        G = D.G0 * 2048 / N
        for gamma in (4e-3, 1e-3, 2.5e-4):
            q, qe = linear_sweep(G, gamma, N)
            a = G * math.sqrt(N) * (math.pi / gamma) ** 0.25
            rows.append((N, gamma, float(logit(q) / logit(qe)), float(logit(ndtr(a)) / logit(qe))))
    excess = [100.0 * (c / ck - 1.0) for _, _, c, ck in rows]
    out.append(check_bool(
        "A14", "Sec. 3, Table 2", "exceeds the prediction in all nine cases, by 0.23-1.81%",
        min(excess) > 0 and f"{min(excess):.2f}" == "0.23" and f"{max(excess):.2f}" == "1.81",
        f"{min(excess):.3f}% .. {max(excess):.3f}%", "all positive; ends round to 2 dp",
        "nine linear ramps of the exact chain vs logit Phi(G sqrt N (pi/gamma)^1/4) / logit q_eq",
    ))
    out.append(check_rounding("A15", "Sec. 3", "0.256", min(N * g for N, g, _, _ in rows),
                              "smallest N gamma over the nine ramps"))
    for i, (N, g, c, _) in enumerate(r for r in rows if round(r[0] ** 2 * r[1]) == 4194):
        out.append(check_rounding(f"A16{'abc'[i]}", "Table 2", ("0.09370", "0.09372", "0.09361")[i], c,
                                  f"exact chain, N={N}, gamma={g:g}"))

    # ------------------------------------------ schedule and reference (Tables 3-4)
    tau = [1.0 / (gap_full(D.N_SPINS, b) if b <= 1.0 else gap_well(D.N_SPINS, b)) for b in D.LADDER]
    dwell = tuple(int(math.ceil(D.BUDGET_CONSTANT * t)) for t in tau)
    out.append(check_bool(
        "A17", "Sec. 4, Table 3", "dwell ceil(6 tau) sweeps, 834 in total",
        dwell == D.K_REDESCENT and sum(dwell) == 834, f"{dwell}, sum {sum(dwell)}", "tuple equality",
        "tau = 1/gap_full for beta <= 1, 1/gap_well above (N = 2048, G = 0)",
    ))
    tau_rh = 1.0 / gap_full(D.N_SPINS, D.BETA_REHEAT)
    out.append(check_sci(
        "A18", "Table 4 caption", "1.2", -4, math.exp(-D.K_REHEAT / tau_rh),
        "exp(-30/tau), tau = 1/gap_full(2048, 0.7)", f"tau = {tau_rh:.3f} sweeps",
    ))
    q_eqs = [qeq_route_a(D.N_SPINS, D.BETA_DEEP, float(g)) for g in D.G_GRID]
    above, below = sum(q > 0.5 for q in q_eqs), sum(q < 0.5 for q in q_eqs)
    out.append(check_bool(
        "A19", "Sec. 4, Table 4", "eight nonzero fields, four on each side of q_eq = 1/2",
        len(q_eqs) == 8 and above == 4 and below == 4,
        f"{len(q_eqs)} fields, {above} above 1/2, {below} below", "counts",
        "src.hopfield.equilibrium.qeq_route_a(2048, 1.6, G) over src.hopfield.protocol.G_GRID",
    ))

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
