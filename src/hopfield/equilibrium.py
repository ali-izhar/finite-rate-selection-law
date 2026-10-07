"""The equilibrium reference q_eq(G), by two routes (paper App. B).

Route (a)  exact finite-N Curie-Weiss mass split, built on the closed form
           ``src.barrier.log_pi_closed_form`` (paper Eq. 10).
Route (b)  the weak-field law  q_eq = sigmoid(2 beta N bar m G).

The Curie-Weiss code is used here only: the Hopfield model at field ``G xi^1``
maps onto Curie-Weiss at ``h = beta G`` for the reference.  Nothing in this
module touches the Hopfield Monte Carlo dynamics.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp

from ..barrier import log_pi_closed_form


def _split_indices(n_spins: int) -> tuple[np.ndarray, np.ndarray]:
    """Index masks for m^1 < 0 and m^1 > 0; the tie set k = N/2 is excluded.

    The labels exclude {m^1 = 0}; at N = 2048, beta = 1.6 its equilibrium
    mass is 10^{-139.0}, so the exclusion is not a truncation of any
    quantity this experiment can resolve.
    """
    k = np.arange(n_spins + 1)
    return k < n_spins / 2.0, k > n_spins / 2.0


def qeq_route_a(n_spins: int, beta: float, field_g: float) -> float:
    """q_eq by the exact finite-N mass split.

        q_eq = [1 + exp(LSE_{k<N/2} log pi_k - LSE_{k>N/2} log pi_k)]^{-1}

    with log pi from the closed form at ``h = beta * field_g``.  The
    additive constant in log pi cancels between the two log-sum-exps, so no
    quantity leaves float64 range.

    STATUS: numerical -- an exact finite-N formula evaluated in float64.
    """
    log_pi = log_pi_closed_form(n_spins, beta, beta * field_g)
    lo, hi = _split_indices(n_spins)
    return float(expit(logsumexp(log_pi[hi]) - logsumexp(log_pi[lo])))


def qeq_route_b(n_spins: int, beta: float, field_g: float, mbar: float) -> float:
    """q_eq by the tilt-mass law.

        q_eq = sigmoid(2 beta N bar m G)

    STATUS: exact to O(N^{-2}); at N = 2048 the residual
    against route (a) is ~2e-8.
    """
    return float(expit(2.0 * beta * n_spins * mbar * field_g))


def tilt_u(n_spins: int, beta: float, field_g: float, mbar: float) -> float:
    """The dimensionless tilt u = beta N bar m G; q_eq = sigmoid(2u)."""
    return float(beta * n_spins * mbar * field_g)


def mbar_cw(n_spins: int, beta: float) -> float:
    """bar m = E[m | m > 0] under the exact finite-N Curie-Weiss law at h = 0.

    This is the kappa_1 for the Curie-Weiss surrogate,
    i.e. the ``bar m_CW`` (0.8905367 at N = 2048,
    beta = 1.6).

    STATUS: numerical -- exact finite-N pi evaluated in float64.
    """
    log_pi = log_pi_closed_form(n_spins, beta, 0.0)
    _, hi = _split_indices(n_spins)
    k = np.arange(n_spins + 1, dtype=np.float64)
    m = 2.0 * k / n_spins - 1.0
    w = np.exp(log_pi[hi] - logsumexp(log_pi[hi]))
    return float(np.dot(w, m[hi]))


def qeq_envelope(
    n_spins: int, beta: float, field_g: float, mbar: float, se_mbar: float,
    *, z: float = 1.96, systematic: float = 0.01,
) -> tuple[float, float, float]:
    """The q_eq error envelope [a_k, b_k], plus the centre.

        delta L = 2 beta N |G| (z SE(bar m) + systematic * bar m)

    in logit space; the returned triple is ``(a_k, q_eq, b_k)``.

    STATUS: design -- the 1% systematic term is a design choice, not a
    measured error.
    """
    centre = 2.0 * beta * n_spins * mbar * field_g
    half = 2.0 * beta * n_spins * abs(field_g) * (z * se_mbar + systematic * mbar)
    return float(expit(centre - half)), float(expit(centre)), float(expit(centre + half))
