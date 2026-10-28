"""Mode-switching barrier: exact finite-N log-potential, log-space MFPT, and the
Arrhenius asymptotic.

The paper uses the mean passage time between mode centres (Sec. 3, App. A).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, logsumexp

from .landau import landau_F, log_potential, log_potential_d2, stationary_magnetisations
from .rates import BarrierIdentityError, flip_rates

BARRIER_IDENTITY_CONST: float = 3.0
"""Constant C in the barrier-identity guard bound |Delta_Phi_N - beta Delta_F| <= C log N / N.

The log N / N rate is the Stirling correction to the continuum potential;
C is not fixed by theory.  C = 3 is the smallest round value leaving a margin of at least 2.5
over beta in [1.05, 4], h in [0, 0.1] and N in [32, 65536]; the worst case is
beta = 4, N = 32.  ``validate.py`` reports the realised margin so the choice
stays auditable.
"""


def log_potential_exact(N: int, beta: float, h: float = 0.0) -> np.ndarray:
    """Phi_N(k) = -log pi_k, up to an additive constant, built from the exact rates.

    ell_k - ell_{k-1} = log W^+_{k-1} - log W^-_k, accumulated
    once.  Phi_N is thus defined from the rates, not assumed equal to any
    continuum formula.

    STATUS: exact at every finite N.
    """
    w_plus, w_minus = flip_rates(N, beta, h)
    increments = np.log(w_plus[:N]) - np.log(w_minus[1:])
    ell = np.concatenate(([0.0], np.cumsum(increments)))
    return -ell


def log_pi_closed_form(N: int, beta: float, h: float = 0.0) -> np.ndarray:
    """log pi_k directly, up to an additive constant.

    The closed form. Paper Fig. 1a and the visual edition's reference data use it
    directly; ``src.validate`` also uses it as an oracle for ``log_potential_exact``.

    STATUS: exact.
    """
    k = np.arange(N + 1, dtype=np.float64)
    log_binom = gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1)
    return log_binom + beta * ((2 * k - N) ** 2 - N) / (2 * (N - 1)) + h * (2 * k - N)


def log_mfpt(N: int, beta: float, h: float, k_from: int, k_to: int) -> float:
    """log T(k_from -> k_to), evaluated in log space.

    The additive constant in ell cancels between the two logsumexp calls, so no
    quantity ever leaves float64 range -- exact for N well past 16384, where the
    MFPT itself is ~10^2324.

    The formula is stated for upward passage with a reflecting boundary at 0, so
    ``k_from < k_to``.  A tilt with h < 0 puts the metastable well on the right;
    use h -> -h and k -> N-k, which the chain's symmetry makes exact.

    STATUS: exact at every finite N; the float64 evaluation is
    numerical but free of cancellation.
    """
    if not 0 <= k_from < k_to <= N:
        raise ValueError(f"need 0 <= k_from < k_to <= N, got {k_from}, {k_to}, {N}")
    w_plus, _ = flip_rates(N, beta, h)
    ell = -log_potential_exact(N, beta, h)
    cumulative = np.logaddexp.accumulate(ell)
    j = np.arange(k_from, k_to)
    terms = -np.log(w_plus[j]) - ell[j] + cumulative[j]
    return float(logsumexp(terms))


@dataclass(frozen=True)
class Barrier:
    """Finite-N and continuum barriers between the metastable well and the top."""

    N: int
    beta: float
    h: float
    k_wrong: int
    k_top: int
    k_right: int
    m_wrong: float
    m_top: float
    m_right: float
    delta_phi_N: float
    delta_phi: float
    delta_F: float


def barrier(N: int, beta: float, h: float = 0.0, *, check: bool = True) -> Barrier:
    """Delta_Phi_N from the exact rates, plus its continuum partners.

    The wrong (metastable) well is the one with the larger Phi; at h = 0 they
    are degenerate and the left one is taken.  With ``check=True`` the barrier-identity guard
    assertions fire.

    STATUS: Delta_Phi_N exact at finite N; Delta_Phi and Delta_F asymptotic.
    """
    delta_phi, delta_F, m_wrong, m_top, m_right = continuum_barrier(beta, h)

    # NOTE: alpha and gamma are the indices of the
    # continuum wells -+m*, not as the finite-N minima of Phi_N.  The two differ
    # by one lattice site and shift log T - N Delta_Phi_N by about 1e-3; the
    # recorded prefactor table uses the -+m* convention.
    phi = log_potential_exact(N, beta, h)
    k_wrong = int(round(0.5 * N * (1.0 + m_wrong)))
    k_right = int(round(0.5 * N * (1.0 + m_right)))
    lo, hi = sorted((k_wrong, k_right))
    k_top = int(lo + np.argmax(phi[lo : hi + 1]))

    delta_phi_N = float((phi[k_top] - phi[k_wrong]) / N)

    result = Barrier(
        N=N,
        beta=beta,
        h=h,
        k_wrong=k_wrong,
        k_top=k_top,
        k_right=k_right,
        m_wrong=float(m_wrong),
        m_top=float(m_top),
        m_right=float(m_right),
        delta_phi_N=delta_phi_N,
        delta_phi=delta_phi,
        delta_F=delta_F,
    )
    if check and beta > 1.0:
        assert_barrier_identity(result)
    return result


def assert_barrier_identity(b: Barrier) -> None:
    """Barrier-identity guard: Delta_Phi_N > 0 and |Delta_Phi_N - beta Delta_F| <= C log N / N."""
    if not np.isfinite(b.delta_phi_N) or b.delta_phi_N <= 0.0:
        raise BarrierIdentityError(
            f"barrier-identity guard: Delta_Phi_N={b.delta_phi_N!r} is not positive at "
            f"N={b.N}, beta={b.beta}, h={b.h}"
        )
    tolerance = BARRIER_IDENTITY_CONST * float(np.log(b.N)) / b.N
    residual = abs(b.delta_phi_N - b.beta * b.delta_F)
    if residual > tolerance:
        raise BarrierIdentityError(
            f"barrier-identity guard: |Delta_Phi_N - beta Delta_F| = {residual:.6e} exceeds "
            f"C log N / N = {tolerance:.6e} at N={b.N}, beta={b.beta}, h={b.h}. "
            f"Delta_Phi = beta Delta_F holds exactly at stationary points."
        )


def continuum_barrier(beta: float, h: float = 0.0) -> tuple[float, float, float, float, float]:
    """(Delta_Phi, Delta_F, m_wrong, m_top, m_right) from the continuum stationary points.

    The wrong well is the metastable one, i.e. the minimum with the larger Phi;
    at h = 0 the two are exactly degenerate and the left one is taken, which
    keeps the passage upward.

    STATUS: asymptotic (Phi and F are N -> infinity objects); their ratio is
    exact at stationary points.
    """
    roots = stationary_magnetisations(beta, h)
    if len(roots) != 3:
        raise ValueError(
            f"no barrier at beta={beta}, h={h}: {len(roots)} stationary point(s)"
        )
    m_lo, m_mid, m_hi = roots
    if log_potential(m_lo, beta, h) >= log_potential(m_hi, beta, h):
        m_wrong, m_right = m_lo, m_hi
    else:
        m_wrong, m_right = m_hi, m_lo
    delta_phi = float(log_potential(m_mid, beta, h) - log_potential(m_wrong, beta, h))
    delta_F = float(landau_F(m_mid, beta, h) - landau_F(m_wrong, beta, h))
    return delta_phi, delta_F, float(m_wrong), float(m_mid), float(m_right)


def arrhenius_log_prefactor(beta: float, h: float = 0.0) -> float:
    """log of the Kramers prefactor 2 pi / sqrt(Phi''(m_well) |Phi''(m_top)|).

    The prefactor is stated at h = 0, where both wells are +-m*; for h != 0 the
    departure (metastable) well is used.

    STATUS: asymptotic -- a Laplace/steepest-descent estimate with an asserted,
    numerically checked O(1/N) error, not a proof.
    """
    _, _, m_wrong, m_top, _ = continuum_barrier(beta, h)
    curvature_well = float(log_potential_d2(m_wrong, beta))
    curvature_top = abs(float(log_potential_d2(m_top, beta)))
    return float(np.log(2.0 * np.pi) - 0.5 * (np.log(curvature_well) + np.log(curvature_top)))
