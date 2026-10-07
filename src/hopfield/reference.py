"""The Hopfield model and its exact leave-one-out Gibbs sweep (paper Eq. 5, App. B).

This module is the *reference* implementation: plain NumPy, one chain at a time,
written to be obviously the formula.  ``sampler`` is the batched GPU production
kernel, and ``sampler_checks.g1c_gpu_vs_reference`` asserts the two agree bit for bit on a supplied
random stream.

Nothing here is imported from the Curie-Weiss package ``src``: the energy, the
local field and the dynamics are Hopfield objects.  The Curie-Weiss code enters
only through ``equilibrium``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit


def patterns(seed: int, n_spins: int, n_patterns: int) -> np.ndarray:
    """p i.i.d. Rademacher patterns, shape ``(p, N)``, dtype int8.

    the pN entries are i.i.d. with P(xi = +-1) = 1/2.
    Drawn with ``numpy.random.default_rng(seed)`` as the protocol requires.

    STATUS: design (the disorder draw).
    """
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, size=(n_patterns, n_spins), dtype=np.int8) * 2 - 1).astype(np.int8)


def interaction_matrix(xi: np.ndarray) -> np.ndarray:
    """A_ij = sum_mu xi^mu_i xi^mu_j with A_ii = 0, as exact int64.

    The Hebbian coupling is J = A / N.  A is kept integral so that
    the dense and the fast local field agree *bitwise*, not merely to rounding
    (the equivalence is asserted, not assumed).

    STATUS: exact.
    """
    a = xi.astype(np.int64).T @ xi.astype(np.int64)
    np.fill_diagonal(a, 0)
    return a


def energy(xi: np.ndarray, state: np.ndarray, field_g: float) -> float:
    """E_G(s) = -1/2 sum_{i != j} J_ij s_i s_j - G xi^1.s.

    STATUS: exact.
    """
    a = interaction_matrix(xi)
    s = state.astype(np.int64)
    quad = float(s @ a @ s) / xi.shape[1]
    return -0.5 * quad - field_g * float(xi[0].astype(np.int64) @ s)


def field_numerator_fast(xi: np.ndarray, proj: np.ndarray, state: np.ndarray) -> np.ndarray:
    """N * (H_i - G xi^1_i) = sum_mu xi^mu_i c_mu - p s_i, the O(Np) route.

    ``proj`` is c_mu = xi^mu . s, carried and updated incrementally
.  Integer-valued, hence exact.

    STATUS: exact.
    """
    p = xi.shape[0]
    return xi.astype(np.int64).T @ proj.astype(np.int64) - p * state.astype(np.int64)


def field_numerator_dense(a_matrix: np.ndarray, state: np.ndarray) -> np.ndarray:
    """N * (H_i - G xi^1_i) = (A s)_i, the dense route.

    STATUS: exact; equals ``field_numerator_fast`` identically because
    (A s)_i = sum_mu xi^mu_i (c_mu - xi^mu_i s_i) = sum_mu xi^mu_i c_mu - p s_i.
    """
    return a_matrix @ state.astype(np.int64)


def p_plus(local_h: np.ndarray | float, beta: float) -> np.ndarray | float:
    """P(s_i = +1 | s_{-i}) = (1 + tanh(beta H_i)) / 2 = sigmoid(2 beta H_i).

    ``scipy.special.expit`` is used rather than the tanh
    form: (1 + tanh x)/2 cancels catastrophically as tanh x -> -1, and the
    deepest rung reaches beta H ~ -4.8.

    STATUS: exact formula, stable float64 evaluation.
    """
    return expit(2.0 * beta * np.asarray(local_h, dtype=np.float64))


def sweep(
    xi: np.ndarray,
    state: np.ndarray,
    proj: np.ndarray,
    beta: float,
    field_g: float,
    perm: np.ndarray,
    unif: np.ndarray,
    *, dense: bool = False, a_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One sweep: N sequential single-spin heat-bath updates in the order ``perm``.

    One sweep is N single-spin updates in a fresh random permutation.  ``unif`` supplies the N uniforms, so the sweep is a
    deterministic function of its arguments and can be compared bit for bit
    against the GPU kernel.

    With ``dense=True`` the local field is recomputed from the dense A at every
    site instead of from the carried projections -- the O(N^2 p) oracle for the
    equivalence assertion.

    STATUS: exact (this kernel is reversible w.r.t. mu_{N,beta}).
    """
    n = xi.shape[1]
    p = xi.shape[0]
    state = state.astype(np.int8).copy()
    proj = proj.astype(np.int64).copy()
    xi64 = xi.astype(np.int64)
    if dense and a_matrix is None:
        a_matrix = interaction_matrix(xi)
    for t in range(n):
        i = int(perm[t])
        if dense:
            num_i = int(a_matrix[i] @ state.astype(np.int64))
        else:
            num_i = int(xi64[:, i] @ proj) - p * int(state[i])
        h_i = num_i / n + field_g * float(xi[0, i])
        s_new = np.int8(1) if unif[t] < p_plus(h_i, beta) else np.int8(-1)
        delta = int(s_new) - int(state[i])
        if delta:
            state[i] = s_new
            proj += xi64[:, i] * delta
    return state, proj


def projections(xi: np.ndarray, state: np.ndarray) -> np.ndarray:
    """c_mu = xi^mu . s, exact int64."""
    return xi.astype(np.int64) @ state.astype(np.int64)
