"""Continuum objects: the Landau functional F, the log-potential Phi and the drift A.

The drift A(m) = tanh(beta m + h) - m is derived in paper App. A.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

_EDGE = 1.0 - 1e-15
_RTOL = 4.0 * float(np.finfo(np.float64).eps)


def log_cosh(z: np.ndarray | float) -> np.ndarray | float:
    """log cosh(z), accurate at both ends of the range.

    The large-|z| form |z| + log1p(e^{-2|z|}) - log 2 subtracts two quantities of
    size log 2, so for small |z| it returns a value of size z^2/2 with an
    absolute error of ~1e-16.  That destroys F(m) = m^2/2 - log cosh(m)/beta
    near m = 0, which is precisely where the pitchfork lives.  Below |z| = 1 the
    identity log cosh z = log1p(2 sinh^2(z/2)) keeps full relative precision.
    """
    a = np.abs(np.asarray(z, dtype=np.float64))
    small = np.log1p(2.0 * np.sinh(np.where(a < 1.0, a, 0.0) / 2.0) ** 2)
    large = a + np.log1p(np.exp(-2.0 * np.where(a < 1.0, 1.0, a))) - np.log(2.0)
    return np.where(a < 1.0, small, large)


def landau_F(m: np.ndarray | float, beta: float, h: float = 0.0) -> np.ndarray | float:
    """F(m) = m^2/2 - log cosh(beta m + h)/beta.

    STATUS: asymptotic -- F is the N -> infinity Laplace/Hubbard-Stratonovich
    free energy, not a finite-N object.
    """
    return 0.5 * np.asarray(m) ** 2 - log_cosh(beta * np.asarray(m) + h) / beta


def landau_F_d2(m: np.ndarray | float, beta: float, h: float = 0.0) -> np.ndarray | float:
    """F''(m) = 1 - beta sech^2(beta m + h)."""
    return 1.0 - beta * (1.0 - np.tanh(beta * np.asarray(m) + h) ** 2)


def log_potential(m: np.ndarray | float, beta: float, h: float = 0.0) -> np.ndarray | float:
    """Phi(m) = log(1-m^2)/2 + m artanh(m) - beta m^2/2 - h m.

    STATUS: asymptotic -- the N -> infinity limit of Phi_N(k)/N, with an
    O(log N / N) Stirling correction.  The finite-N object is
    ``barrier.log_potential_exact``.
    """
    m = np.asarray(m, dtype=np.float64)
    return 0.5 * np.log1p(-(m**2)) + m * np.arctanh(m) - 0.5 * beta * m**2 - h * m


def log_potential_d2(m: np.ndarray | float, beta: float) -> np.ndarray | float:
    """Phi''(m) = 1/(1-m^2) - beta."""
    return 1.0 / (1.0 - np.asarray(m) ** 2) - beta


def drift(m: np.ndarray | float, beta: float, h: float = 0.0) -> np.ndarray | float:
    """A(m) = tanh(beta m + h) - m.

    STATUS: asymptotic -- the O(1/N) correction to the exact jump moment
    A_N is dropped.
    """
    return np.tanh(beta * np.asarray(m) + h) - np.asarray(m)


def drift_d1(m: np.ndarray | float, beta: float, h: float = 0.0) -> np.ndarray | float:
    """A'(m) = beta sech^2(beta m + h) - 1."""
    return beta * (1.0 - np.tanh(beta * np.asarray(m) + h) ** 2) - 1.0


def stationary_magnetisations(beta: float, h: float = 0.0) -> tuple[float, ...]:
    """Solutions of the self-consistency equation m = tanh(beta m + h), ascending.

    These are exactly the stationary points of both F and Phi.  One root for beta <= 1; three for beta > 1 and |h| below the
    spinodal field.

    STATUS: exact root-finding on the continuum drift.
    """
    def g(m: float) -> float:
        return float(np.tanh(beta * m + h) - m)

    if beta <= 1.0 and h == 0.0:
        return (0.0,)

    # tanh saturates in float64 once |beta m + h| exceeds artanh(1 - 1e-15) ~ 17.6.
    # Past that, g has no sign change at +-_EDGE, every stationary point sits
    # within one ulp of +-1, and scipy's brentq reports only the opaque
    # "f(a) and f(b) must have different signs".  Name the real cause instead.
    if not (g(-_EDGE) > 0.0 > g(_EDGE)):
        raise ValueError(
            f"the stationary points of m = tanh(beta m + h) are not representable in "
            f"float64 at beta={beta}, h={h}: tanh saturates for |beta m + h| >~ "
            f"{float(np.arctanh(_EDGE)):.4f}, so |m*| is within one ulp of 1 and every "
            f"continuum quantity built on it (kappa, chi, Delta_Phi, Delta_F, the Kramers "
            f"prefactor) would be meaningless. Use the finite-N objects in barrier.py."
        )

    if beta <= 1.0:
        return (brentq(g, -_EDGE, _EDGE, xtol=1e-16, rtol=_RTOL),)

    u_c = np.arctanh(np.sqrt(1.0 - 1.0 / beta))
    m_a = max(-_EDGE, (-u_c - h) / beta)
    m_b = min(_EDGE, (u_c - h) / beta)
    if g(m_a) >= 0.0 or g(m_b) <= 0.0:
        return (brentq(g, -_EDGE, _EDGE, xtol=1e-16, rtol=_RTOL),)
    if h == 0.0:
        m_star = brentq(g, m_b, _EDGE, xtol=1e-16, rtol=_RTOL)
        return (-m_star, 0.0, m_star)
    return (
        brentq(g, -_EDGE, m_a, xtol=1e-16, rtol=_RTOL),
        brentq(g, m_a, m_b, xtol=1e-16, rtol=_RTOL),
        brentq(g, m_b, _EDGE, xtol=1e-16, rtol=_RTOL),
    )


def m_star(beta: float) -> float:
    """The positive solution of m = tanh(beta m) at h = 0; 0 for beta <= 1."""
    roots = stationary_magnetisations(beta, 0.0)
    return float(roots[-1])


def kappa(beta: float) -> float:
    """kappa(beta) = |A'(m*)|, the in-well linearised rate, beta > 1.

    STATUS: exact given the continuum drift A; the finite-N gap approaches it
 rather than equalling it.
    """
    if beta <= 1.0:
        raise ValueError(f"kappa is defined for beta > 1, got {beta}")
    return float(abs(drift_d1(m_star(beta), beta)))


def susceptibility(beta: float) -> float:
    """chi = d m*/d h at h = 0.  STATUS: exact.

    chi diverges as 1/|1-beta| at the critical point, so beta = 1 has no finite
    value; it is refused rather than allowed to raise ZeroDivisionError.
    """
    ms = m_star(beta) ** 2
    denominator = 1.0 - beta * (1.0 - ms)
    if denominator == 0.0:
        raise ValueError(
            f"the Curie-Weiss susceptibility diverges at beta={beta} "
            f"(chi = 1/(1-beta) for beta < 1)"
        )
    return float((1.0 - ms) / denominator)
