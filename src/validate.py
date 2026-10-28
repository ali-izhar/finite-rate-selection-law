"""Analytic and numerical checks of the exact Curie-Weiss chain in ``src``.

Run from the repository root::

    python -m src

Each row reports the expected value, the computed value and the tolerance.
The paper uses this chain in Sec. 2-3 and App. A.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from scipy.special import logsumexp

from .barrier import (
    BARRIER_IDENTITY_CONST,
    Barrier,
    arrhenius_log_prefactor,
    assert_barrier_identity,
    barrier,
    continuum_barrier,
    log_mfpt,
    log_pi_closed_form,
    log_potential_exact,
)
from .brute_force import birth_death_spectrum, glauber_spectrum
from .gap import (
    eigensolve_max_N,
    gap_full,
    gap_well,
    resolution_floor,
    top_eigenvalues,
)
from .landau import (
    drift,
    drift_d1,
    kappa,
    landau_F,
    landau_F_d2,
    log_potential,
    m_star,
    stationary_magnetisations,
    susceptibility,
)
from .normal_form import richardson_rates
from .rates import (
    BarrierIdentityError,
    NumericalFloorError,
    RateBoundError,
    assert_rate_bounds,
    flip_rates,
    jump_moments,
    symmetric_generator,
)

LAMBDA_1_INF = 0.790154
LAMBDA_2_INF = 2.571336

_BARRIER_GRID = (
    (1.05, 0.0),
    (1.3, 0.0),
    (1.3, 0.02),
    (1.5, 0.0),
    (1.5, 0.05),
    (2.0, 0.0),
    (2.0, 0.1),
    (3.0, 0.0),
    (3.0, 0.1),
    (4.0, 0.0),
)
"""Barrier cases: beyond the spinodal field a tilt destroys the metastable well,
so h is paired with beta rather than swept independently."""


@dataclass
class Row:
    section: str
    check: str
    expected: str
    computed: str
    metric: str
    passed: bool


@dataclass
class Report:
    rows: list[Row] = field(default_factory=list)

    def numeric(
        self,
        section: str,
        check: str,
        expected: Any,
        computed: Any,
        *,
        rtol: float = 0.0,
        atol: float = 0.0,
    ) -> None:
        exp = np.atleast_1d(np.asarray(expected, dtype=np.float64))
        got = np.atleast_1d(np.asarray(computed, dtype=np.float64))
        error = float(np.max(np.abs(got - exp)))
        tol = atol + rtol * float(np.max(np.abs(exp)))
        self.rows.append(
            Row(
                section=section,
                check=check,
                expected=_fmt(exp),
                computed=_fmt(got),
                metric=f"|err| {error:.2e} <= {tol:.2e}",
                passed=bool(error <= tol),
            )
        )

    def predicate(
        self, section: str, check: str, expected: str, computed: str, passed: bool
    ) -> None:
        self.rows.append(
            Row(section, check, expected, computed, "boolean", bool(passed))
        )

    def raises(
        self,
        section: str,
        check: str,
        exception: type[BaseException],
        call: Callable[[], Any],
    ) -> None:
        try:
            call()
        except exception:
            outcome, passed = f"{exception.__name__} raised", True
        except Exception as exc:  # noqa: BLE001 - a wrong exception is a real failure
            outcome, passed = f"{type(exc).__name__} raised", False
        else:
            outcome, passed = "returned without raising", False
        self.predicate(section, check, f"{exception.__name__}", outcome, passed)

    @property
    def failures(self) -> list[Row]:
        return [r for r in self.rows if not r.passed]

    def render(self) -> str:
        widths = [72, 26, 26, 26, 4]
        header = ("check", "expected", "computed", "metric", "")
        lines: list[str] = []
        current = None
        rule = "-" * (sum(widths) + 5 * 2)
        lines.append(rule)
        lines.append(_line(header, widths))
        lines.append(rule)
        for row in self.rows:
            if row.section != current:
                current = row.section
                lines.append("")
                lines.append(f"== {current}")
            lines.append(
                _line(
                    (
                        "  " + row.check,
                        row.expected,
                        row.computed,
                        row.metric,
                        "PASS" if row.passed else "FAIL",
                    ),
                    widths,
                )
            )
        lines.append("")
        lines.append(rule)
        lines.append(f"{len(self.rows) - len(self.failures)}/{len(self.rows)} checks passed")
        return "\n".join(lines)


def _line(cells: Sequence[str], widths: Sequence[int]) -> str:
    return "  ".join(str(c)[:w].ljust(w) for c, w in zip(cells, widths))


def _fmt(values: np.ndarray) -> str:
    if values.size == 1:
        return f"{float(values[0]):.9g}"
    head = ", ".join(f"{v:.6g}" for v in values[:3])
    return f"[{head}{', ...' if values.size > 3 else ''}]"


# ----------------------------------------------------------------

def check_lumping(report: Report) -> None:
    section = "Exact lumping to the count chain"
    worst_containment = 0.0
    worst_gap = 0.0
    for N in (4, 5, 6, 7, 8):
        for beta in (0.5, 1.0, 2.0):
            for h in (0.0, 0.3):
                full = glauber_spectrum(N, beta, h)
                lumped = birth_death_spectrum(N, beta, h)
                distance = np.abs(full[None, :] - lumped[:, None]).min(axis=1)
                worst_containment = max(worst_containment, float(distance.max()))
                worst_gap = max(worst_gap, abs(float(-full[-2]) - float(-lumped[-2])))
    report.numeric(
        section,
        "spec(G) contained in spec(L_N), N in 4..8, beta in {0.5,1,2}, h in {0,0.3}",
        0.0,
        worst_containment,
        atol=1e-13,
    )
    report.numeric(
        section,
        "gap(L_N) == gap(G) on the same grid (equality is not proved in general)",
        0.0,
        worst_gap,
        atol=1e-13,
    )


# ----------------------------------------------------------------

def check_stationary_law(report: Report) -> None:
    section = "Stationary law"
    small, large, worst_db = 0.0, 0.0, 0.0
    for N, beta, h in (
        (64, 0.5, 0.0),
        (64, 2.0, 0.05),
        (1024, 2.0, 0.0),
        (1024, 0.9, 0.1),
        (4096, 2.0, 0.0),
    ):
        cumulative = -log_potential_exact(N, beta, h)
        closed = log_pi_closed_form(N, beta, h)
        log_pi = cumulative - logsumexp(cumulative)
        difference = log_pi - (closed - logsumexp(closed))
        error = float(np.abs(difference).max())
        if N <= 64:
            small = max(small, error)
        else:
            large = max(large, error / float(np.abs(log_pi).max()))

        # Detailed balance is checked against the closed form of pi, not against
        # the cumulative construction, which would restate its own definition.
        w_plus, w_minus = flip_rates(N, beta, h)
        independent = closed - logsumexp(closed)
        lhs = independent[:N] + np.log(w_plus[:N])
        rhs = independent[1:] + np.log(w_minus[1:])
        scale = max(1.0, float(np.abs(independent).max()))
        worst_db = max(worst_db, float(np.abs(lhs - rhs).max()) / scale)
    report.numeric(
        section,
        "log pi by cumulative log(W+/W-) vs the closed form, N=64 (absolute)",
        0.0,
        small,
        atol=1e-13,
    )
    # Absolute agreement to 1e-13 at every N cannot hold once log pi spans
    # hundreds of nats: at N=4096, beta=2 the range is 1344, so a cumulative sum
    # of N terms carries ~sqrt(N) eps * range ~ 1e-11 of rounding.  Agreement is at machine precision relative to that range,
    # which is the strongest statement float64 admits.
    report.numeric(
        section,
        "  ... N in {1024, 4096}, relative to the range of log pi (absolute 1e-13 is unattainable)",
        0.0,
        large,
        atol=1e-14,
    )
    report.numeric(
        section,
        "detailed balance pi_k W+_k = pi_{k+1} W-_{k+1} with pi in closed form, "
        "relative to the range of log pi",
        0.0,
        worst_db,
        atol=1e-14,
    )

    N, beta, h = 12, 1.7, 0.05
    tri = symmetric_generator(N, beta, h)
    dense_sym = np.diag(tri.diag) + np.diag(tri.offdiag, 1) + np.diag(tri.offdiag, -1)
    w_plus, w_minus = flip_rates(N, beta, h)
    generator = np.diag(-(w_plus + w_minus)) + np.diag(w_plus[:N], 1) + np.diag(w_minus[1:], -1)
    report.numeric(
        section,
        "spec(S) == spec(G) for the unsymmetrised tridiagonal, N=12",
        0.0,
        np.max(np.abs(np.sort(np.linalg.eigvalsh(dense_sym)) - np.sort(np.linalg.eigvals(generator).real))),
        atol=1e-12,
    )


# ----------------------------------------------------------------

def check_gap_amplitudes(report: Report) -> None:
    section = "Spectral-gap amplitudes"

    report.numeric(
        section,
        "gap_full(beta=0) at N in {16,64,1024}; Ehrenfest spectrum {0,-1,...,-N}",
        [1.0, 1.0, 1.0],
        [gap_full(N, 0.0) for N in (16, 64, 1024)],
        atol=1e-12,
    )
    spectrum = -top_eigenvalues(symmetric_generator(8, 0.0, 0.0), 9)
    report.numeric(
        section,
        "full spectrum at beta=0, N=8 equals {0,1,...,8}",
        np.arange(9.0),
        np.sort(spectrum),
        atol=1e-12,
    )

    betas = (0.5, 0.8, 0.9)
    computed = [gap_full(65536, b) for b in betas]
    report.numeric(
        section,
        "gap_full at N=65536, beta in {0.5,0.8,0.9} vs 1-beta (4 digits)",
        [1 - b for b in betas],
        computed,
        rtol=2e-3,
    )
    report.numeric(
        section,
        "  ... reproduces the recorded eigensolve values",
        [0.5000038, 0.2000390, 0.1001109],
        computed,
        atol=1e-7,
    )

    computed = [gap_well(65536, b) for b in betas]
    report.numeric(
        section,
        "gap_well, wall at m=0, N=65536, beta<1 vs 2(1-beta) (3 digits)",
        [2 * (1 - b) for b in betas],
        computed,
        rtol=3e-3,
    )
    report.numeric(
        section,
        "  ... reproduces the recorded eigensolve values",
        [0.9989292, 0.3999015, 0.2003713],
        computed,
        atol=1e-7,
    )

    betas = (1.02, 1.05, 1.1, 1.2, 1.5)
    computed = [gap_well(500_000, b) for b in betas]
    report.numeric(
        section,
        "gap_well, wall at m=0, N=5e5, beta>1 vs beta-1 (3 digits)",
        [b - 1 for b in betas],
        computed,
        rtol=6e-3,
    )
    # NOTE: the recorded row is quoted at N = 5e5.  At beta = 1.02 the gap is
    # still 5e-4 short of beta-1 there and rises monotonically with N
    # (0.0198937, 0.0199465, 0.0199728 at N = 5e5, 1e6, 2e6), so the recorded
    # 0.019899 corresponds to a slightly larger N than stated.  The tolerance
    # here is set by that residual finite-size drift, not by the solver.
    report.numeric(
        section,
        "  ... reproduces the recorded eigensolve values",
        [0.019899, 0.049947, 0.099944, 0.199890, 0.499603],
        computed,
        atol=1e-5,
    )
    report.numeric(
        section,
        "  ... gap_well(beta=1.02) rises to beta-1 = 0.02 as N grows to 4e6",
        [0.0198937, 0.0199465, 0.0199728, 0.0199861],
        [gap_well(N, 1.02) for N in (500_000, 1_000_000, 2_000_000, 4_000_000)],
        atol=1e-6,
    )

    betas = (1.2, 1.5, 2.0, 3.0)
    # NOTE: gap_well -> kappa is an N -> infinity statement; the residual falls
    # like 1/N (2.6e-3, 6.6e-4, 1.6e-4, 4.1e-5 at N = 16384..1048576, beta=1.2),
    # so the tolerance is the finite-size correction at the N actually used.
    report.numeric(
        section,
        "gap_well, wall at m*/2, beta>1 vs kappa(beta) of the drift, N=262144",
        [kappa(b) for b in betas],
        [gap_well(262144, b, m_wall=m_star(b) / 2) for b in betas],
        rtol=3e-4,
    )
    residuals = [
        abs(gap_well(N, 1.2, m_wall=m_star(1.2) / 2) - kappa(1.2)) for N in (16384, 65536, 262144)
    ]
    report.predicate(
        section,
        "  ... that residual falls by at least 3 per factor 4 in N, so the limit is kappa",
        "ratio >= 3",
        f"{residuals[0] / residuals[1]:.1f}, {residuals[1] / residuals[2]:.1f}",
        residuals[0] / residuals[1] >= 3 and residuals[1] / residuals[2] >= 3,
    )
    # The recorded gap at beta=1.2, 0.320246, is reproduced exactly; it agrees
    # with kappa(1.2) = 0.3204568 only to four figures; the fifth is the
    # finite-N residual above.
    report.numeric(
        section,
        "wall at m*/2, beta=1.2, N=65536: reproduces the recorded 0.320246",
        0.320246,
        gap_well(65536, 1.2, m_wall=m_star(1.2) / 2),
        atol=1e-6,
    )
    report.numeric(
        section,
        "wall at m=0, beta=1.2, N=65536: first four rates are the two ladders",
        [0.19963, 0.32025, 0.59862, 0.64002],
        -top_eigenvalues(symmetric_generator(65536, 1.2, 0.0, k_wall=32768), 5)[1:],
        atol=1e-5,
    )
    report.numeric(
        section,
        "beta=2 exact gap_well = 0.833612 while |1-beta| = 1 (kappa, not the linear amplitude)",
        0.833612,
        gap_well(65536, 2.0),
        atol=1e-6,
    )
    report.numeric(
        section,
        "van Hove: |A'(m*)| == (1-m*^2)/chi",
        [(1 - m_star(b) ** 2) / susceptibility(b) for b in (1.2, 2.0, 3.0)],
        [kappa(b) for b in (1.2, 2.0, 3.0)],
        rtol=1e-13,
    )

def check_bounds(report: Report) -> None:
    """The two-sided bounds of Sec. "Two chains, not one", swept rather than spot-checked."""
    section = "Two-sided gap bounds"

    rng = np.random.default_rng(0)
    ratios = []
    for _ in range(200):
        N = 2 * int(rng.integers(1, 300))
        beta = float(rng.uniform(0.0, 6.0))
        ratios.append(gap_well(N, beta) * N**2 / (0.5 * (1 - np.tanh(beta))))
    report.predicate(
        section,
        "gap_well >= c_beta/N^2 over 200 seeded (N, beta) draws, h=0, N even",
        "min ratio >= 1",
        f"min ratio {min(ratios):.1f}",
        min(ratios) >= 1.0,
    )

    slack = []
    skipped = 0
    for N in range(8, 140, 4):
        for beta in (1.05, 1.2, 1.5, 2.0, 3.0, 4.0):
            ell = -log_potential_exact(N, beta, 0.0)
            log_pi = ell - logsumexp(ell)
            k_0 = N // 2
            if np.exp(log_pi[k_0]) > 0.5:
                continue
            try:
                gap = gap_full(N, beta)
            except NumericalFloorError:
                skipped += 1
                continue
            w_plus, _ = flip_rates(N, beta, 0.0)
            slack.append(np.log(gap) - (np.log(8.0) + log_pi[k_0] + np.log(w_plus[k_0])))
    report.predicate(
        section,
        f"gap_full <= 8 pi_k0 W+_k0 wherever the eigensolve is resolvable ({len(slack)} cases, "
        f"{skipped} refused by the noise-floor guard)",
        "max log-slack <= 0",
        f"{max(slack):.3f}",
        max(slack) <= 0.0,
    )

    # The well-gap bound assumes h = 0, where Phi_N is unimodal on {N/2,...,N}.  An
    # opposing tilt makes it bimodal, the well chain regains an exponentially
    # small mode, and gap_well is no longer licensed.
    turning_points = {
        h: int(np.sum(np.diff(np.sign(np.diff(log_potential_exact(256, 2.0, h)[128:]))) != 0))
        for h in (0.0, -0.3)
    }
    report.predicate(
        section,
        "Phi_N unimodal on {N/2,...,N} at h=0, bimodal at h=-0.3 (N=256, beta=2)",
        "1 turning point at h=0, 2 at h=-0.3",
        f"{turning_points[0.0]} at h=0, {turning_points[-0.3]} at h=-0.3",
        turning_points[0.0] == 1 and turning_points[-0.3] == 2,
    )
    tilted = [gap_well(N, 2.0, -0.3, check=False) for N in (128, 256, 520)]
    report.predicate(
        section,
        "  ... so the tilted well gap decays exponentially and the noise-floor guard refuses it at N=1024",
        "decaying, then refused",
        f"{tilted[0]:.1e} -> {tilted[2]:.1e}, then refused",
        tilted[0] > tilted[1] > tilted[2] and _refuses(lambda: gap_well(1024, 2.0, -0.3)),
    )


def _refuses(call: Callable[[], Any]) -> bool:
    try:
        call()
    except NumericalFloorError:
        return True
    return False


# ----------------------------------------------------------------

def check_critical_amplitudes(report: Report) -> None:
    section = "Universal amplitudes at beta = 1"
    sizes = (1024, 4096, 16384, 65536, 262144)
    full = np.array([gap_full(N, 1.0) * np.sqrt(N) for N in sizes])
    well = np.array([gap_well(N, 1.0) * np.sqrt(N) for N in sizes])

    report.numeric(
        section,
        "gap_full*sqrt(N) at beta=1 reproduces the recorded table",
        [0.75779, 0.77364, 0.78182, 0.78597, 0.78806],
        full,
        atol=1e-5,
    )
    report.numeric(
        section,
        "gap_well*sqrt(N) at beta=1 reproduces the recorded table",
        [2.47287, 2.52228, 2.54703, 2.55931, 2.56538],
        well,
        atol=1e-5,
    )
    # The finite-size correction is O(N^{-1/2}), so one Richardson step in that
    # variable across a factor 4 in N removes it: g_inf = 2 g(4N) - g(N).
    report.numeric(
        section,
        "Richardson limit of gap_full*sqrt(N) vs lambda_1^inf = 0.790154 (odd mode)",
        LAMBDA_1_INF,
        2 * full[-1] - full[-2],
        rtol=1e-4,
    )
    report.numeric(
        section,
        "Richardson limit of gap_well*sqrt(N) vs lambda_2^inf = 2.571336 (even mode)",
        LAMBDA_2_INF,
        2 * well[-1] - well[-2],
        rtol=1e-4,
    )
    report.numeric(
        section,
        "gap_well/gap_full at beta=1 -> lambda_2/lambda_1 = 3.25422",
        3.25422,
        well[-1] / full[-1],
        rtol=3e-3,
    )

    rates = richardson_rates(3, half_width=12.0, n_grid=10000)
    report.numeric(
        section,
        "independent oracle: grid diagonalisation of the normal form gives the two constants",
        [LAMBDA_1_INF, LAMBDA_2_INF],
        rates[1:],
        rtol=2e-5,
    )
    spread = np.array(
        [richardson_rates(3, half_width=L, n_grid=10000)[1:] for L in (10.0, 12.0, 14.0, 16.0)]
    )
    report.numeric(
        section,
        "  ... stable to 1e-5 under box half-width L in [10,16]",
        0.0,
        spread.max(axis=0) - spread.min(axis=0),
        atol=1e-5,
    )

    sizes_fit = (64, 256, 1024, 4096, 16384, 65536)
    tau = np.array([1.0 / gap_well(N, 1.0) for N in sizes_fit])
    slope_all = float(np.polyfit(np.log(sizes_fit), np.log(tau), 1)[0])
    slope_top = float(np.polyfit(np.log(sizes_fit[-3:]), np.log(tau[-3:]), 1)[0])
    report.numeric(
        section,
        "log-log slope of tau_well(beta=1) over N in 64..65536",
        0.4798,
        slope_all,
        atol=5e-4,
    )
    report.numeric(
        section,
        "  ... over the three largest sizes (biased low by O(N^{-1/2}))",
        0.4947,
        slope_top,
        atol=5e-4,
    )
    report.predicate(
        section,
        "  ... both slopes lie in (0.47, 0.50), i.e. approach 1/2 from below",
        "0.47 < slope_all < slope_top < 0.5",
        f"{slope_all:.4f} < {slope_top:.4f}",
        0.47 < slope_all < slope_top < 0.5,
    )


# ---------------------------------------------------------------- Ledger: Landau functional and pitchfork

def check_pitchfork(report: Report) -> None:
    section = "Landau functional and pitchfork"
    betas = (0.5, 0.9, 1.0, 1.5, 2.0, 4.0)
    report.numeric(
        section,
        "F''(0)|_{h=0} = 1 - beta",
        [1 - b for b in betas],
        [float(landau_F_d2(0.0, b)) for b in betas],
        rtol=1e-14,
        atol=1e-15,
    )
    report.predicate(
        section,
        "one stationary point for beta <= 1, three for beta > 1",
        "1, 1, 1, 3, 3, 3",
        ", ".join(str(len(stationary_magnetisations(b))) for b in betas),
        [len(stationary_magnetisations(b)) for b in betas] == [1, 1, 1, 3, 3, 3],
    )
    # F = m^4/12 - m^6/45 + 17 m^8/2520 at beta=1, so the two-term series is
    # short of the truth by 1.73e-16 at m = 0.02.  That truncation, not the
    # evaluation, sets the tolerance: F itself is accurate to 2e-20 here.
    report.numeric(
        section,
        "at beta=1, F(m) = m^4/12 - m^6/45 + O(m^8)",
        [(m**4) / 12 - (m**6) / 45 for m in (0.01, 0.02)],
        [float(landau_F(m, 1.0)) for m in (0.01, 0.02)],
        atol=2e-16,
    )
    report.numeric(
        section,
        "  ... with the m^8 term restored, agreement is limited only by float64",
        [(m**4) / 12 - (m**6) / 45 + 17 * m**8 / 2520 for m in (0.01, 0.02)],
        [float(landau_F(m, 1.0)) for m in (0.01, 0.02)],
        atol=1e-19,
    )
    # The next two rows check leading-order expansions in (beta-1), so their
    # relative error is O(beta-1) by construction: 1.8e-3 at beta=1.001,
    # 1.8e-4 at beta=1.0001.  The tolerances are those residuals, not fitted.
    report.numeric(
        section,
        "m*^2 = 3(beta-1)/beta^3 (1 + O(beta-1)), supercritical",
        [3 * (b - 1) / b**3 for b in (1.0001, 1.001)],
        [m_star(b) ** 2 for b in (1.0001, 1.001)],
        rtol=3e-3,
    )
    report.numeric(
        section,
        "kappa(beta) = (beta-1)(3/beta^2 - 1) + O((beta-1)^2)",
        [(b - 1) * (3 / b**2 - 1) for b in (1.0001, 1.001)],
        [kappa(b) for b in (1.0001, 1.001)],
        rtol=3e-3,
    )
    report.numeric(
        section,
        "  ... so the ordered-side amplitude kappa/(beta-1) -> 2 while the "
        "disordered-side |A'(0)|/(1-beta) = 1",
        [2.0, 1.0],
        [kappa(1.0001) / 1e-4, abs(float(drift_d1(0.0, 0.9999))) / 1e-4],
        atol=1e-3,
    )
    # The leading drift power p is measured, not assumed: the general exponent formula then
    # fixes (a, b) = (1/(p+1), (p-1)/(p+1)), giving (1/4, 1/2) here.
    m = np.array([1e-4, 1e-3])
    p_pitchfork = float(np.diff(np.log(np.abs(drift(m, 1.0))))[0] / np.diff(np.log(m))[0])
    report.numeric(
        section,
        "measured leading drift power p: 3 at beta=1 (Z_2)",
        [3.0],
        [p_pitchfork],
        atol=1e-3,
    )
    report.numeric(
        section,
        "  ... the general exponent formula then gives (a, b) = (1/4, 1/2)",
        [0.25, 0.5],
        [
            1 / (round(p_pitchfork) + 1),
            (round(p_pitchfork) - 1) / (round(p_pitchfork) + 1),
        ],
        rtol=1e-15,
    )


# ----------------------------------------------------------------

def check_barrier_ratio(report: Report) -> None:
    section = "Barrier identity Delta_Phi = beta Delta_F"
    cases = ((1.05, 0.0), (1.3, 0.02), (1.5, 0.05), (2.0, 0.0), (2.0, 0.1), (3.0, 0.1), (4.0, 0.0))
    computed = [continuum_barrier(b, h) for b, h in cases]
    report.numeric(
        section,
        "Delta_Phi / Delta_F equals beta exactly, symmetric and tilted",
        [b for b, _ in cases],
        [c[0] / c[1] for c in computed],
        rtol=1e-12,
    )
    report.numeric(
        section,
        "  ... reproduces the recorded Delta_Phi column",
        [0.00176834, 0.03508836, 0.07536426, 0.32652389, 0.23633122, 0.71243306, 1.30718913],
        [c[0] for c in computed],
        atol=1e-8,
    )
    report.numeric(
        section,
        "  ... reproduces the recorded Delta_F column",
        [0.00168413, 0.02699104, 0.05024284, 0.16326194, 0.11816561, 0.23747769, 0.32679728],
        [c[1] for c in computed],
        atol=1e-8,
    )
    report.numeric(
        section,
        "away from stationary points Phi != beta F: beta=2, m=0.5",
        [-0.119188, -0.183781],
        [float(log_potential(0.5, 2.0)), 2.0 * float(landau_F(0.5, 2.0))],
        atol=1e-6,
    )
    report.numeric(
        section,
        "tilted-chain cross-check (the 10^1994 sweeps): beta=2, h=0.05 gives Delta_Phi and N*Delta_Phi/log10",
        [0.280030, 1992.5],
        [continuum_barrier(2.0, 0.05)[0], 16384 * continuum_barrier(2.0, 0.05)[0] / np.log(10)],
        atol=0.1,
    )


# ----------------------------------------------------------------

def check_mfpt(report: Report) -> None:
    section = "Exact mean first-passage time and Arrhenius law"

    for beta, sizes in ((2.0, (24, 32, 40, 48)), (1.5, (48, 64, 80, 128))):
        products = []
        for N in sizes:
            b = barrier(N, beta)
            products.append(gap_full(N, beta) * np.exp(log_mfpt(N, beta, 0.0, b.k_wrong, b.k_right)))
        report.numeric(
            section,
            f"gap_full * T_MFPT -> 2 at beta={beta}, N in {sizes}",
            2.0,
            products,
            atol=1.1e-2 if beta == 1.5 else 1e-3,
        )
    report.numeric(
        section,
        "  ... reproduces the recorded products at beta=2",
        [2.00046, 2.00005, 2.00000, 2.00000],
        [
            gap_full(N, 2.0) * np.exp(log_mfpt(N, 2.0, 0.0, barrier(N, 2.0).k_wrong, barrier(N, 2.0).k_right))
            for N in (24, 32, 40, 48)
        ],
        atol=1e-5,
    )

    sizes = (64, 256, 1024, 4096, 16384)
    log10_T = []
    log10_gap = []
    for N in sizes:
        b = barrier(N, 2.0)
        lt = log_mfpt(N, 2.0, 0.0, b.k_wrong, b.k_right)
        log10_T.append(lt / np.log(10.0))
        log10_gap.append((np.log(2.0) - lt) / np.log(10.0))
    report.numeric(
        section,
        "log10 T at beta=2 reproduces the recorded values, N up to 16384",
        [10.3, 37.5, 146.4, 582.1, 2324.6],
        log10_T,
        atol=0.05,
    )
    report.numeric(
        section,
        "  ... log10 gap_full = log10(2/T) column",
        [-10.0, -37.2, -146.1, -581.8, -2324.3],
        log10_gap,
        atol=0.05,
    )
    report.predicate(
        section,
        "log T stays finite in float64 at N=16384 and N=65536 (log-space telescoping)",
        "finite",
        "finite"
        if all(
            np.isfinite(log_mfpt(N, 2.0, 0.0, barrier(N, 2.0).k_wrong, barrier(N, 2.0).k_right))
            for N in (16384, 65536)
        )
        else "overflowed",
        all(
            np.isfinite(log_mfpt(N, 2.0, 0.0, barrier(N, 2.0).k_wrong, barrier(N, 2.0).k_right))
            for N in (16384, 65536)
        ),
    )

    betas = (1.3, 1.5, 2.0, 3.0, 4.0)
    predicted = [arrhenius_log_prefactor(b) for b in betas]
    report.numeric(
        section,
        "predicted log-prefactor log(2 pi / sqrt(Phi''_w Phi''_t))",
        [2.43888, 1.76711, 0.68552, -0.78731, -2.01322],
        predicted,
        atol=1e-5,
    )
    report.numeric(
        section,
        "Phi''(m*) at beta in {1.3,1.5,2,3,4}",
        [1.00198, 2.30408, 10.02125, 95.31925, 737.73012],
        [1.0 / (1.0 - m_star(b) ** 2) - b for b in betas],
        atol=1e-4,
    )
    for N, tol in ((4096, 3e-2), (65536, 3e-3)):
        measured = []
        for b in betas:
            bar = barrier(N, b)
            measured.append(log_mfpt(N, b, 0.0, bar.k_wrong, bar.k_right) - N * bar.delta_phi_N)
        report.numeric(
            section,
            f"measured log T - N Delta_Phi_N at N={N} vs the Kramers prefactor",
            predicted,
            measured,
            atol=tol,
        )
        recorded = (
            [2.44039, 1.76853, 0.68625, -0.79207, -2.03645]
            if N == 4096
            else [2.43899, 1.76719, 0.68586, -0.78701, -2.01142]
        )
        report.numeric(
            section,
            f"  ... reproduces the recorded measurement column at N={N}",
            recorded,
            measured,
            atol=1e-4,
        )


# ----------------------------------------------------------------

def check_guards(report: Report) -> None:
    section = "Runtime guards"

    report.raises(
        section,
        "gap_full(N=128, beta=2) refuses the noise-floor plateau",
        NumericalFloorError,
        lambda: gap_full(128, 2.0),
    )
    report.raises(
        section,
        "gap_full(N=112, beta=2) refuses (raw eigensolve overstates by 654x)",
        NumericalFloorError,
        lambda: gap_full(112, 2.0),
    )
    report.raises(
        section,
        "gap_full(N=96, beta=2) refuses (raw eigensolve overstates by 2.1x)",
        NumericalFloorError,
        lambda: gap_full(96, 2.0),
    )
    report.predicate(
        section,
        "gap_full(N=64, beta=2) is still resolvable and passes the guard",
        "returns a value",
        f"{gap_full(64, 2.0):.4e}",
        gap_full(64, 2.0) > 0,
    )
    report.predicate(
        section,
        "gap_well is licensed at every beta and N *at h=0*: no guard fires at N=5e5, beta=4",
        "returns a value",
        f"{gap_well(500_000, 4.0):.6f}",
        gap_well(500_000, 4.0) > 0,
    )

    raw = [gap_full(N, 2.0, check=False) for N in (64, 80, 88, 96, 112, 128)]
    report.numeric(
        section,
        "unguarded eigensolve reproduces the plateau-then-sign-flip",
        [9.598e-11, 5.191e-13, 3.776e-14, 5.861e-15, 4.939e-15, -4.239e-15],
        raw,
        rtol=1e-3,
    )
    truth = []
    for N in (64, 80, 88, 96, 112, 128):
        b = barrier(N, 2.0)
        truth.append(2.0 * np.exp(-log_mfpt(N, 2.0, 0.0, b.k_wrong, b.k_right)))
    report.numeric(
        section,
        "  ... against the log-space 2/T column, which stays correct throughout",
        [9.598e-11, 5.185e-13, 3.809e-14, 2.798e-15, 1.509e-17, 8.132e-20],
        truth,
        rtol=1e-3,
    )
    report.numeric(
        section,
        "eigensolve_max_N predicts failure near N = 96 / 264 / 40 at beta = 2 / 1.5 / 3",
        [96.0, 264.0, 40.0],
        [eigensolve_max_N(b) for b in (2.0, 1.5, 3.0)],
        atol=6.0,
    )

    report.raises(
        section,
        "rate-bound guard fires on a negative rate",
        RateBoundError,
        lambda: assert_rate_bounds(8, np.array([-1.0, 1.0]), np.array([1.0, 1.0])),
    )
    report.raises(
        section,
        "rate-bound guard fires when W+_k + W-_k exceeds N",
        RateBoundError,
        lambda: assert_rate_bounds(8, np.array([5.0, 1.0]), np.array([4.0, 1.0])),
    )
    report.predicate(
        section,
        "rate-bound guard holds for the real rates over beta in [0,4], h in [-1,1], N in 2..4096",
        "no violation",
        "no violation",
        _rates_bounded(),
    )
    report.raises(
        section,
        "barrier-identity guard fires when Delta_Phi_N is inconsistent with beta Delta_F",
        BarrierIdentityError,
        lambda: assert_barrier_identity(
            Barrier(4096, 2.0, 0.0, 87, 2048, 4009, -0.9575, 0.0, 0.9575, 0.65, 0.3265, 0.1633)
        ),
    )
    report.raises(
        section,
        "barrier-identity guard fires on a non-positive Delta_Phi_N",
        BarrierIdentityError,
        lambda: assert_barrier_identity(
            Barrier(4096, 2.0, 0.0, 87, 2048, 4009, -0.9575, 0.0, 0.9575, 0.0, 0.3265, 0.1633)
        ),
    )
    margins = []
    for beta, h in _BARRIER_GRID:
        for N in (32, 64, 4096, 65536):
            b = barrier(N, beta, h)
            residual = abs(b.delta_phi_N - beta * b.delta_F)
            margins.append(BARRIER_IDENTITY_CONST * np.log(N) / N / max(residual, 1e-300))
    report.predicate(
        section,
        "barrier-identity guard holds with margin >= 2 over beta in [1.05,4], tilted and not, N in 32..65536",
        "margin >= 2",
        f"min margin {min(margins):.2f}",
        min(margins) >= 2.0,
    )

    tri = symmetric_generator(4096, 2.0, 0.0)
    report.predicate(
        section,
        "Gershgorin ||S||_2 respects the 2N bound the noise-floor threshold uses",
        f"<= {tri.norm_bound:.0f}",
        f"{tri.gershgorin_norm:.1f}",
        tri.gershgorin_norm <= tri.norm_bound,
    )
    report.predicate(
        section,
        "resolution floor at N=4096 equals 10 eps (2N)",
        f"{10 * np.finfo(float).eps * 2 * 4096:.4e}",
        f"{resolution_floor(tri):.4e}",
        np.isclose(resolution_floor(tri), 10 * np.finfo(float).eps * 2 * 4096, rtol=1e-15),
    )


def _rates_bounded() -> bool:
    for N in (2, 3, 8, 64, 4096):
        for beta in (0.0, 0.5, 1.0, 2.0, 4.0):
            for h in (-1.0, 0.0, 0.3, 1.0):
                w_plus, w_minus = flip_rates(N, beta, h)
                if w_plus.min() < 0 or w_minus.min() < 0:
                    return False
                if (w_plus + w_minus).max() > N * (1 + 8 * np.finfo(float).eps):
                    return False
    return True


# ---------------------------------------------------------------- exactness of the rates

def check_jump_moments(report: Report) -> None:
    section = "Exact rates vs their continuum limits"
    computed = []
    for N in (100, 1000, 10000):
        A_N, _ = jump_moments(N, 1.3, 0.0)
        computed.append(A_N[int(round(N * (1 + 0.3) / 2))])
    report.numeric(
        section,
        "exact A_N(0.3) at beta=1.3, N = 1e2, 1e3, 1e4",
        [0.07130979, 0.07135973, 0.07136022],
        computed,
        atol=1e-8,
    )
    report.numeric(
        section,
        "  ... against the continuum drift A(0.3) = tanh(1.3*0.3) - 0.3",
        0.07136023,
        float(drift(0.3, 1.3)),
        atol=1e-8,
    )
    diffusion = []
    for N in (100, 1000, 10000):
        _, B_N = jump_moments(N, 1.0, 0.0)
        diffusion.append(B_N[N // 2])
    report.numeric(
        section,
        "exact B_N(0) at beta=1 vs 2/N",
        [2.0202e-2, 2.0020e-3, 2.0002e-4],
        diffusion,
        rtol=1e-4,
    )
    # The next term is 62 m^9/2835, so the series truncated at m^7 is good to
    # 1.1e-17 at m = 0.02 -- the tolerance is set by that, not chosen.
    series = [-(m**3) / 3 + 2 * m**5 / 15 - 17 * m**7 / 315 for m in (0.01, 0.02)]
    report.numeric(
        section,
        "cubic normal form: A(m) = -m^3/3 + 2m^5/15 - 17m^7/315 + O(m^9) at beta=1",
        series,
        [float(drift(m, 1.0)) for m in (0.01, 0.02)],
        atol=1e-15,
    )
    report.predicate(
        section,
        "  ... the linear and quadratic terms vanish identically (Z_2 symmetry at h=0)",
        "|A(m)/m^3 + 1/3| < 1e-4 for m <= 1e-2",
        f"{abs(float(drift(1e-2, 1.0)) / 1e-6 + 1 / 3):.2e}",
        abs(float(drift(1e-2, 1.0)) / 1e-6 + 1 / 3) < 1e-4,
    )


def main() -> int:
    report = Report()
    check_lumping(report)
    check_stationary_law(report)
    check_gap_amplitudes(report)
    check_bounds(report)
    check_critical_amplitudes(report)
    check_pitchfork(report)
    check_barrier_ratio(report)
    check_mfpt(report)
    check_guards(report)
    check_jump_moments(report)
    print(report.render())
    if report.failures:
        print("\nFAILURES:")
        for row in report.failures:
            print(f"  [{row.section}] {row.check}")
            print(f"      expected {row.expected}  computed {row.computed}  ({row.metric})")
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
