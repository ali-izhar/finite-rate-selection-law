"""Brute-force 2^N Glauber generator, for validating the exact lumping.

Feasible only for N <= 8.

This module exists solely as an independent oracle: it never touches the
birth-death machinery, so agreement is evidence of the lumping rather than of a
shared implementation.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def glauber_spectrum(N: int, beta: float, h: float = 0.0) -> np.ndarray:
    """Full spectrum of the 2^N-state Glauber generator L_N, ascending.

    L_N is symmetrised in l^2(mu_N) before diagonalising.  Detailed balance
 makes the symmetrised off-diagonal entry
    sqrt(c_i(s) c_i(s^(i))), so mu_N itself is never formed.

    STATUS: exact for the finite N it can reach.
    """
    if N > 12:
        raise ValueError(f"2^{N} states is not a brute force any more")
    size = 1 << N
    states = ((np.arange(size)[:, None] >> np.arange(N)[None, :]) & 1) * 2 - 1
    magnetisation = states.sum(axis=1)

    S = np.zeros((size, size), dtype=np.float64)
    for i in range(N):
        local_field = (magnetisation - states[:, i]) / (N - 1)
        p_up = 0.5 * (1.0 + np.tanh(beta * local_field + h))
        rate = np.where(states[:, i] == -1, p_up, 1.0 - p_up)
        neighbour = np.arange(size) ^ (1 << i)
        S[np.arange(size), np.arange(size)] -= rate
        S[np.arange(size), neighbour] = np.sqrt(rate * rate[neighbour])
    return eigh(S, eigvals_only=True)


def birth_death_spectrum(N: int, beta: float, h: float = 0.0) -> np.ndarray:
    """Full spectrum of the lumped chain, by dense diagonalisation, ascending."""
    from .rates import symmetric_generator

    tri = symmetric_generator(N, beta, h, k_wall=0)
    dense = np.diag(tri.diag) + np.diag(tri.offdiag, 1) + np.diag(tri.offdiag, -1)
    return eigh(dense, eigvals_only=True)
