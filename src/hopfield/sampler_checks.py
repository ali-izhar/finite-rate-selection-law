"""Four checks of the sampler and of the saved counts.

    G1  detailed balance by enumeration at N <= 10, p = 2, to <= 1e-6, plus
        equivalence of the fast O(Np) sweep and the dense ``J s`` sweep.
    G2  the pooled q_model Wilson interval at G = 0 covers 1/2.
    G3  pooled q_model^(k) + q_model^(7-k) covers 1 for k = 0,1,2,3.
    G4  the two q_eq routes agree to 1e-10, and every seed's bar m sits within
        5 alpha bar m_CW = 6.52e-3 of 0.8905367.

The analysis is read only after all four pass; a failure is a bug, not a
finding.  These functions report; they never repair.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.special import logsumexp

from . import protocol as D
from . import reference as H
from . import equilibrium as Q
from .sampler import BatchedHopfield, acceptance_table
from .stats import mover_sum, wilson


@dataclass
class GateResult:
    """One check's outcome, with the numbers it was read from."""

    name: str
    passed: bool
    detail: str
    values: dict = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


# ============================================================== G1 ==========
def _enumerate(xi: np.ndarray, field_g: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """All 2^N states and their normalised Boltzmann probabilities."""
    n = xi.shape[1]
    bits = ((np.arange(2**n)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
    states = (bits * 2 - 1).astype(np.int8)
    log_w = np.array([-beta * H.energy(xi, s, field_g) for s in states])
    return states, np.exp(log_w - logsumexp(log_w))


def g1a_detailed_balance(
    n_spins: int = 10, n_patterns: int = 2, seed: int = 20260823,
    betas: tuple[float, ...] = (0.7, 1.6), fields: tuple[float, ...] = (0.0, 6.125e-4, 0.05),
    tol: float = D.GATE_G1_TOL,
) -> GateResult:
    """G1, first half: the implemented kernel is reversible w.r.t. e^{-beta E_G}.

    The single-site kernel is assembled from the *production* acceptance table
    (``sampler.acceptance_table``), so this tests the object the GPU actually
    looks up, not a re-derivation of it.  Two quantities are reported: the
    largest detailed-balance residual ``|pi(s)P(s,s') - pi(s')P(s',s)|`` and the
    largest stationarity residual ``|(pi P - pi)_s|``.

    STATUS: exact enumeration, float64 evaluation.
    """
    xi = H.patterns(seed, n_spins, n_patterns)
    off = n_patterns * n_spins + n_patterns
    worst_db = 0.0
    worst_stat = 0.0
    worst_at = ()
    for beta in betas:
        for g in fields:
            tab_g = acceptance_table(n_spins, n_patterns, beta, np.array([g]))[0]  # (2, W)
            states, pi = _enumerate(xi, g, beta)
            n_states = states.shape[0]
            kernel = np.zeros((n_states, n_states))
            for a in range(n_states):
                s = states[a]
                proj = H.projections(xi, s)
                for i in range(n_spins):
                    num = int(xi[:, i].astype(np.int64) @ proj) - n_patterns * int(s[i])
                    p_up = tab_g[1 if xi[0, i] > 0 else 0, num + off]
                    b = a ^ (1 << i)      # bit i of the index is spin i
                    stay_plus = states[a][i] == 1
                    kernel[a, b] += (1.0 - p_up if stay_plus else p_up) / n_spins
                    kernel[a, a] += (p_up if stay_plus else 1.0 - p_up) / n_spins
            flux = pi[:, None] * kernel
            db = float(np.abs(flux - flux.T).max())
            stat = float(np.abs(pi @ kernel - pi).max())
            if max(db, stat) > max(worst_db, worst_stat):
                worst_at = (beta, g)
            worst_db = max(worst_db, db)
            worst_stat = max(worst_stat, stat)
    ok = worst_db <= tol and worst_stat <= tol
    return GateResult(
        "G1a detailed balance (enumeration)", ok,
        f"N={n_spins}, p={n_patterns}, betas={betas}, G in {fields}: "
        f"max |pi P - (pi P)^T| = {worst_db:.3e}, max |pi P - pi| = {worst_stat:.3e}, "
        f"tol = {tol:.0e} (worst at beta={worst_at[0] if worst_at else '-'}, "
        f"G={worst_at[1] if worst_at else '-'})",
        {"db_residual": worst_db, "stationarity_residual": worst_stat, "tol": tol},
    )


def g1b_fast_vs_dense(
    n_spins: int = 64, n_patterns: int = 3, seed: int = 20260823,
    beta: float = 1.6, field_g: float = 6.125e-4, n_sweeps: int = 20,
) -> GateResult:
    """G1, second half: the O(Np) carried-projection sweep == the dense ``J s`` sweep.

    "it is exactly equivalent to forming the dense J and
    computing J s, and the equivalence must be asserted in a test, not assumed."
    Both routes form the same *integer* numerator, so the assertion is bitwise
    equality of the states, not a tolerance.

    STATUS: exact.
    """
    xi = H.patterns(seed, n_spins, n_patterns)
    a_matrix = H.interaction_matrix(xi)
    rng = np.random.default_rng(1)
    s0 = (rng.integers(0, 2, n_spins, dtype=np.int8) * 2 - 1).astype(np.int8)
    fast_s, fast_c = s0.copy(), H.projections(xi, s0)
    dense_s, dense_c = s0.copy(), H.projections(xi, s0)
    max_num_gap = 0
    for _ in range(n_sweeps):
        perm = rng.permutation(n_spins)
        unif = rng.random(n_spins)
        fast_s, fast_c = H.sweep(xi, fast_s, fast_c, beta, field_g, perm, unif)
        dense_s, dense_c = H.sweep(
            xi, dense_s, dense_c, beta, field_g, perm, unif, dense=True, a_matrix=a_matrix
        )
        max_num_gap = max(
            max_num_gap,
            int(np.abs(
                H.field_numerator_fast(xi, fast_c, fast_s)
                - H.field_numerator_dense(a_matrix, fast_s)
            ).max()),
        )
    same = bool(np.array_equal(fast_s, dense_s) and np.array_equal(fast_c, dense_c))
    return GateResult(
        "G1b fast O(Np) sweep == dense J s sweep", same,
        f"N={n_spins}, p={n_patterns}, {n_sweeps} sweeps: states identical={same}, "
        f"max |numerator_fast - numerator_dense| = {max_num_gap} (integer, exact)",
        {"identical": same, "max_numerator_gap": max_num_gap},
    )


def g1c_gpu_vs_reference(
    n_spins: int = 32, n_patterns: int = 3, n_chains: int = 7,
    beta: float = 1.6, fields: tuple[float, ...] = (6.125e-4, -6.125e-4),
    n_sweeps: int = 6, seed: int = 20260823, device: str = "cuda",
) -> GateResult:
    """G1, extension: the GPU production kernel == the NumPy reference sweep.

    Both are driven by the *same* supplied permutation and uniforms, so agreement
    must be bit for bit.  This is the check that the object which will actually
    run 2.4e8 chain-sweeps is the sampler the paper describes.

    STATUS: exact.
    """
    rng = np.random.default_rng(seed)
    n_seed = 3
    xi_all = np.stack([H.patterns(D.seed_of(s), n_spins, n_patterns) for s in range(n_seed)])
    seed_idx = rng.integers(0, n_seed, n_chains)
    cond_idx = rng.integers(0, len(fields), n_chains)
    sim = BatchedHopfield(
        xi_all, seed_idx, cond_idx, np.array(fields), device=device, torch_seed=0
    )
    s0 = (rng.integers(0, 2, (n_chains, n_spins), dtype=np.int8) * 2 - 1).astype(np.int8)
    sim.set_state(s0)

    perm = np.stack([rng.permutation(n_spins) for _ in range(n_chains)]).astype(np.int16)
    unif = rng.random((n_sweeps, n_chains, n_spins))
    sim.run_sweeps(beta, n_sweeps, perm_override=perm, unif_override=unif)

    ref_s = s0.copy()
    ref_c = np.stack([H.projections(xi_all[seed_idx[b]], s0[b]) for b in range(n_chains)])
    for w in range(n_sweeps):
        for b in range(n_chains):
            ref_s[b], ref_c[b] = H.sweep(
                xi_all[seed_idx[b]], ref_s[b], ref_c[b], beta, fields[cond_idx[b]],
                perm[b], unif[w, b],
            )
    gpu_s = sim.state.cpu().numpy()
    gpu_c = sim.projections()
    ok = bool(np.array_equal(gpu_s, ref_s) and np.array_equal(gpu_c, ref_c))
    n_diff = int((gpu_s != ref_s).sum())
    return GateResult(
        "G1c GPU kernel == NumPy reference (same random stream)", ok,
        f"N={n_spins}, p={n_patterns}, {n_chains} chains, {n_sweeps} sweeps, "
        f"G in {fields}: differing spins = {n_diff} of {gpu_s.size}, "
        f"projections identical = {bool(np.array_equal(gpu_c, ref_c))}",
        {"identical": ok, "n_differing_spins": n_diff},
    )


def g1d_empirical_stationary(
    n_spins: int = 10, n_patterns: int = 2, beta: float = 1.6, field_g: float = 0.05,
    n_chains: int = 200_000, n_burn: int = 60, n_meas: int = 120,
    n_groups: int = 200, seed: int = 20260823, device: str = "cuda",
) -> GateResult:
    """G1, supporting: the GPU sampler's empirical law against e^{-beta E_G}.

    A sampling check cannot reach the 1e-6 of G1 -- the exact 1e-6 claim is
    carried by ``g1a``.  Its standard error must also not be the naive
    multinomial one: successive sweeps of the same chain are autocorrelated, so
    that band is too narrow.  The chains are split into ``n_groups``
    *independent* groups and the standard error is taken across groups, which
    needs no autocorrelation model at all.  The check is read at 5 sigma.

    STATUS: numerical (Monte Carlo), reported with an autocorrelation-free
    standard error.
    """
    xi = H.patterns(seed, n_spins, n_patterns)[None, ...]
    _, pi = _enumerate(xi[0], field_g, beta)
    sim = BatchedHopfield(
        xi, np.zeros(n_chains, dtype=np.int64), np.zeros(n_chains, dtype=np.int64),
        np.array([field_g]), device=device, torch_seed=7,
    )
    sim.cold_start()
    sim.run_sweeps(beta, n_burn)
    n_states = 2**n_spins
    pow2 = torch.from_numpy(2 ** np.arange(n_spins)).to(sim.device).to(torch.int64)
    grp = (torch.arange(n_chains, device=sim.device) * n_groups) // n_chains
    counts = torch.zeros(n_groups * n_states, dtype=torch.int64, device=sim.device)
    for _ in range(n_meas):
        sim.run_sweeps(beta, 1)
        idx = ((sim.state.to(torch.int64) + 1) // 2 * pow2).sum(dim=1)
        counts += torch.bincount(grp * n_states + idx, minlength=n_groups * n_states)
    per_group = counts.reshape(n_groups, n_states).cpu().numpy().astype(np.float64)
    per_group /= per_group.sum(axis=1, keepdims=True)
    emp = per_group.mean(axis=0)
    se = per_group.std(axis=0, ddof=1) / np.sqrt(n_groups)
    dev = np.abs(emp - pi)
    z = dev / np.maximum(se, 1e-300)
    drift = sim.projection_drift()
    ok = bool(z.max() <= 5.0 and drift == 0)
    return GateResult(
        "G1d empirical law of the GPU sampler vs Boltzmann", ok,
        f"N={n_spins}, p={n_patterns}, beta={beta}, G={field_g}, "
        f"{n_chains * n_meas:.2e} draws in {n_groups} independent groups: "
        f"max |emp - pi| = {dev.max():.3e}, max across-group z = {z.max():.2f} sigma "
        f"(threshold 5), median SE = {np.median(se):.2e}; "
        f"carried-vs-recomputed projection drift = {drift}",
        {"max_abs_dev": float(dev.max()), "max_z": float(z.max()),
         "projection_drift": drift},
    )


# ============================================================== G2 ==========
def gate_g2(successes: int, trials: int, z: float = D.Z_SCORE) -> GateResult:
    """G2: the pooled q_model Wilson interval at G = 0 covers 1/2."""
    ci = wilson(successes, trials, z)
    ok = ci.covers(0.5)
    return GateResult(
        "G2 exact null at G = 0", ok,
        f"pooled k/n = {successes}/{trials}, q_model = {ci}, covers 1/2 = {ok}",
        {"q_model": ci.point, "lower": ci.lower, "upper": ci.upper, "n": trials},
    )


# ============================================================== G3 ==========
def gate_g3(successes: list[int], trials: list[int], z: float = D.Z_SCORE) -> GateResult:
    """G3: q_model^(k) + q_model^(7-k) covers 1 for k = 0,1,2,3."""
    rows = []
    ok = True
    for k in range(4):
        a = wilson(successes[k], trials[k], z)
        b = wilson(successes[7 - k], trials[7 - k], z)
        s = mover_sum(a, b)
        covers = s.covers(1.0)
        ok = ok and covers
        rows.append((k, s, covers))
    detail = "; ".join(
        f"k={k}+{7-k}: {s.point:.5f} [{s.lower:.5f}, {s.upper:.5f}] "
        f"{'covers 1' if c else 'MISSES 1'}"
        for k, s, c in rows
    )
    n_miss = sum(1 for _, _, c in rows if not c)
    detail += (
        f"  [{n_miss}/4 pairs miss.  NOTE: the gate as specified reads four "
        f"nominally-95% intervals, so even a perfect implementation misses at "
        f"least one with probability 1 - 0.95^4 = 0.19; a single marginal miss "
        f"is weak evidence of a defect, and the miss size in sigma is the number "
        f"to read.]"
    )
    for k, s, c in rows:
        if not c:
            sigma = abs(s.point - 1.0) / max((s.upper - s.lower) / (2 * z), 1e-12)
            detail += f"  k={k}+{7-k} misses by {sigma:.2f} sigma."
    return GateResult(
        "G3 odd symmetry q(k) + q(7-k) = 1", ok, detail,
        {"sums": [(k, s.point, s.lower, s.upper, c) for k, s, c in rows],
         "n_miss": n_miss},
    )


# ============================================================== G4 ==========
def gate_g4(
    mbar_seeds: dict[int, float], *, n_spins: int = D.N_SPINS, beta: float = D.BETA_DEEP,
) -> GateResult:
    """G4: the two q_eq routes agree, and every seed's bar m is within 6.52e-3.

    NOTE: the gate's literal route-agreement tolerance is 1e-10.  It is not
    attainable: the weak-field route
    carries an O(N^{-2}) residual which that remark tabulates as -2.123e-8 at
    N = 2048.  Both the literal outcome and a theory-consistent one (tolerance
    1e-7, comfortably above the quoted 2e-8) are returned; the literal outcome is
    the ``passed`` field, and the discrepancy is reported, never widened away.
    """
    mbar_cw = Q.mbar_cw(n_spins, beta)
    gaps = [
        abs(Q.qeq_route_a(n_spins, beta, g) - Q.qeq_route_b(n_spins, beta, g, mbar_cw))
        for g in D.G_GRID
    ]
    worst_route = float(max(gaps))
    literal_ok = worst_route <= D.GATE_G4_ROUTE_TOL
    theory_ok = worst_route <= 1e-7

    dev = {s: abs(m - D.MBAR_CW_SPEC) for s, m in mbar_seeds.items()}
    worst_seed = max(dev, key=dev.get) if dev else -1
    worst_dev = dev[worst_seed] if dev else float("nan")
    mbar_ok = bool(dev) and worst_dev <= D.GATE_G4_MBAR_TOL

    return GateResult(
        "G4 q_eq routes agree + per-seed bar m within 5 alpha", bool(literal_ok and mbar_ok),
        f"max |route(a) - route(b)| over the grid = {worst_route:.3e} "
        f"(literal tol {D.GATE_G4_ROUTE_TOL:.0e}: {'PASS' if literal_ok else 'FAIL'}; "
        f"theory-consistent tol 1e-7: {'PASS' if theory_ok else 'FAIL'}); "
        f"max_s |bar m_seed - {D.MBAR_CW_SPEC}| = {worst_dev:.3e} at s={worst_seed}, "
        f"tol {D.GATE_G4_MBAR_TOL:.3e}: {'PASS' if mbar_ok else 'FAIL'}",
        {
            "route_gap": worst_route,
            "route_literal_pass": bool(literal_ok),
            "route_theory_pass": bool(theory_ok),
            "mbar_max_dev": float(worst_dev),
            "mbar_pass": bool(mbar_ok),
            "mbar_cw_computed": mbar_cw,
        },
    )


# =================================================== interval machinery ======
def check_interval_machinery(z: float = D.Z_SCORE) -> GateResult:
    """Verify ``stats.wilson`` against an independent root-find of the score equation.

    The Wilson limits are *defined* as the two roots of
    ``|p_hat - p| = z sqrt(p(1-p)/n)``; ``stats.wilson`` uses the closed form.
    Solving the definition numerically with Brent's method is an oracle that
    shares no algebra with the implementation.  Two published reference
    intervals are checked as well.

    Not one of G1-G4; the decision rule is read entirely through these
    intervals, so they get their own equivalence test.
    """
    from scipy.optimize import brentq

    worst = 0.0
    for n in (7, 40, 137, 2000, 8317):
        for k in (0, 1, n // 3, n // 2, n - 1, n):
            ci = wilson(k, n, z)
            p_hat = k / n

            def score(p: float, p_hat: float = p_hat, n: int = n) -> float:
                return (p_hat - p) ** 2 - z * z * p * (1.0 - p) / n

            # The two roots bracket p_hat, and score(0), score(1) >= 0.  At p_hat = 0
            # the lower root is 0 itself and at p_hat = 1 the upper root is 1, so
            # bracketing on p_hat would let Brent converge onto that *endpoint* root
            # and return the wrong one.  Nudging the interior point off the boundary
            # keeps score < 0 there (true whenever the nudge is below
            # z^2/(n + z^2), i.e. 4.6e-4 at the largest n used) and picks out the
            # two roots unambiguously.
            mid = min(max(p_hat, 1e-6), 1.0 - 1e-6)
            assert score(mid) < 0.0, (n, k, mid, score(mid))
            lo = brentq(score, 0.0, mid, xtol=1e-16, rtol=8.9e-16)
            hi = brentq(score, mid, 1.0, xtol=1e-16, rtol=8.9e-16)
            worst = max(worst, abs(ci.lower - lo), abs(ci.upper - hi))
    ref = wilson(50, 100, 1.96)
    ref_ok = abs(ref.lower - 0.4038) < 5e-5 and abs(ref.upper - 0.5962) < 5e-5
    ref2 = wilson(0, 100, 1.96)
    ref2_ok = abs(ref2.lower - 0.0) < 1e-12 and abs(ref2.upper - 0.03698) < 5e-5

    a, b = wilson(600, 1000), wilson(400, 1000)
    s = mover_sum(a, b)
    sym = abs(s.lower - mover_sum(b, a).lower) < 1e-15
    ok = worst <= 1e-12 and ref_ok and ref2_ok and sym and s.covers(1.0)
    return GateResult(
        "interval machinery (Wilson / MOVER) verified", ok,
        f"max |closed form - Brent root of the score equation| = {worst:.3e} "
        f"(tol 1e-12); wilson(50,100) = [{ref.lower:.4f}, {ref.upper:.4f}] vs "
        f"[0.4038, 0.5962]: {ref_ok}; wilson(0,100) upper = {ref2.upper:.5f} vs "
        f"0.03698: {ref2_ok}; MOVER symmetric = {sym}, covers 1 on a 0.6+0.4 "
        f"pair = {s.covers(1.0)}",
        {"max_wilson_residual": worst},
    )


# ================================================== ladder cross-check ======
def check_ladder_budget() -> GateResult:
    """Recompute K(beta) = ceil(6 tau_ex(beta, 2048)) from the exact eigensolve (src.gap).

    Not one of G1-G4; a cross-check that the transcribed ladder is the ladder the
    exact chain produces.
    """
    from ..gap import gap_full, gap_well

    rows = []
    ok = True
    for beta, k_spec in zip(D.LADDER, D.K_REDESCENT, strict=True):
        tau = 1.0 / (gap_full(D.N_SPINS, beta, 0.0) if beta <= 1.0
                     else gap_well(D.N_SPINS, beta, 0.0))
        k = int(np.ceil(D.BUDGET_CONSTANT * tau))
        ok = ok and k == k_spec
        rows.append((beta, tau, k, k_spec))
    detail = ", ".join(f"beta={b:.2f}: tau={t:.4f} K={k} (protocol {ks})" for b, t, k, ks in rows)
    return GateResult("ladder budget K(beta) reproduced", ok, detail, {"rows": rows})


__all__ = [
    "GateResult",
    "check_interval_machinery",
    "check_ladder_budget",
    "g1a_detailed_balance",
    "g1b_fast_vs_dense",
    "g1c_gpu_vs_reference",
    "g1d_empirical_stationary",
    "gate_g2",
    "gate_g3",
    "gate_g4",
]
