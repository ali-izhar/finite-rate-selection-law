"""Exact spectral gaps of the two birth-death chains, with the resolution-floor guards."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh_tridiagonal

from .landau import log_potential, m_star
from .rates import (
    EPS_MACH,
    NumericalFloorError,
    Tridiagonal,
    symmetric_generator,
    wall_index,
)

GUARD_FACTOR: float = 10.0


def top_eigenvalues(tri: Tridiagonal, count: int = 2) -> np.ndarray:
    """The ``count`` algebraically largest eigenvalues of S, descending.

    Eigenvectors are never formed; they would cost O(N^2) memory.
    """
    n = tri.size
    if count > n:
        raise ValueError(f"asked for {count} eigenvalues of a {n}x{n} matrix")
    vals = eigh_tridiagonal(
        tri.diag,
        tri.offdiag,
        eigvals_only=True,
        select="i",
        select_range=(n - count, n - 1),
    )
    return np.asarray(vals)[::-1]


def resolution_floor(tri: Tridiagonal) -> float:
    """The noise-floor guard threshold 10 * eps_mach * ||S||_2, with ||S||_2 <= 2N."""
    return GUARD_FACTOR * EPS_MACH * tri.norm_bound


def spectral_gap(tri: Tridiagonal, *, check: bool = True) -> float:
    """gap = |second-largest eigenvalue of G|, by exact eigensolve.

    With ``check=True`` the positivity and noise-floor guards fire as exceptions.  They
    must not be disabled outside a test that deliberately inspects the failure
    mode: past the floor the eigensolve returns a
    plausible-looking plateau, not an error.

    STATUS: numerical (exact formula evaluated in float64).
    """
    lam_0, lam_1 = top_eigenvalues(tri, 2)
    gap = -float(lam_1)
    if check:
        assert_gap_resolvable(gap, tri, lam_0=float(lam_0))
    return gap


def assert_gap_resolvable(gap: float, tri: Tridiagonal, *, lam_0: float | None = None) -> None:
    """Positivity and noise-floor guards."""
    floor = resolution_floor(tri)
    if not np.isfinite(gap) or gap <= 0.0:
        raise NumericalFloorError(
            f"positivity guard: gap={gap!r} is not positive at N={tri.n_spins}, k_wall={tri.k_wall}. "
            f"The gap is positive at every finite N, so this is a numerical failure."
        )
    if gap < floor:
        raise NumericalFloorError(
            f"noise-floor guard: gap={gap:.6e} at N={tri.n_spins}, k_wall={tri.k_wall} is below the "
            f"float64 resolution floor 10*eps*||S||_2 = {floor:.6e}. "
            f"Past the floor the eigensolve plateaus instead of erroring. "
            f"Use the log-space MFPT (barrier.log_mfpt) instead."
        )
    if lam_0 is not None and abs(lam_0) > floor:
        raise NumericalFloorError(
            f"positivity guard: lambda_0={lam_0:.6e} at N={tri.n_spins} exceeds the resolution floor "
            f"{floor:.6e}; the simple zero eigenvalue of G was not resolved."
        )


def gap_full(N: int, beta: float, h: float = 0.0, *, check: bool = True) -> float:
    """gap of the full chain on {0,...,N}.

    Valid for beta <= 1 at any N.  For beta > 1 this route is valid
    only below the floor (N Delta_Phi <~ 36 - log 2N);
    past it the noise-floor guard raises rather than returning the noise-floor plateau.

    STATUS: numerical.
    """
    return spectral_gap(symmetric_generator(N, beta, h, k_wall=0), check=check)


def gap_well(
    N: int, beta: float, h: float = 0.0, *, m_wall: float = 0.0, check: bool = True
) -> float:
    """gap of the well-restricted chain, reflecting wall at magnetisation ``m_wall``.

    Licensed (gap >= c_beta/N^2, no exponential factor) for
    all beta and all N **at h = 0 only**.  The bound needs Phi_N unimodal on {N/2,...,N}, which an
    opposing tilt destroys: at beta = 2, h = -0.3 it already fails at N = 30, a
    second minimum appears inside the well, and the well gap becomes
    exponentially small again -- 1.3e-10 at N = 512, five hundred times *below*
    the c_beta/N^2 bound, and still passing the noise-floor guard.  At h != 0 this
    function is therefore no safer than ``gap_full``: check unimodality of
    ``barrier.log_potential_exact(N, beta, h)[N // 2:]`` first, or use the
    log-space MFPT route.

    The wall location changes the amplitude and must be reported with the number:
    the wall at m=0 sits on a fixed point of the drift
    and retains that fixed point's even ladder.  ``m_wall = 0`` requires N even
.

    STATUS: numerical.
    """
    k_wall = wall_index(N, m_wall)
    return spectral_gap(symmetric_generator(N, beta, h, k_wall=k_wall), check=check)


def eigensolve_max_N(beta: float, h: float = 0.0) -> float:
    """Largest N at which the full-chain eigensolve still resolves the gap.

    Solves N * Delta_Phi = log(1/eps_mach) - log(2N) by iteration.
    Reported for planning only; the binding check is the noise-floor guard.

    STATUS: asymptotic.
    """
    if beta <= 1.0:
        return float("inf")
    ms = m_star(beta)
    delta_phi = float(log_potential(0.0, beta, h) - log_potential(ms, beta, h))
    n = 100.0
    for _ in range(200):
        n = (np.log(1.0 / EPS_MACH) - np.log(2.0 * n)) / delta_phi
    return float(n)
