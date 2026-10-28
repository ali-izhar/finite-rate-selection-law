"""Exact Curie-Weiss birth-death rates and the symmetrised tridiagonal generator.

The rates are paper Eq. (8) (App. A).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS_MACH: float = float(np.finfo(np.float64).eps)


class RateBoundError(RuntimeError):
    """A rate array violated the exact bounds (rate-bound guard)."""


class NumericalFloorError(RuntimeError):
    """A computed gap sits at or below the float64 resolution floor.

    Past the floor the eigensolve does not error; it plateaus at ~eps_mach*||S||_2 and then changes sign. Raising is the only
    correct response; returning the plateau is the trap this guard exists for.
    """


class BarrierIdentityError(RuntimeError):
    """Delta_Phi_N failed the beta*Delta_F relation (barrier-identity guard)."""


def flip_rates(
    N: int, beta: float, h: float = 0.0, *, check: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Exact leave-one-out Glauber birth and death rates.

    Returns ``(W_plus, W_minus)``, each shape ``(N+1,)``, indexed by the up-spin
    count k.  The two tanh arguments differ -- (2k-N+1)/(N-1) for a birth,
    (2k-N-1)/(N-1) for a death -- and both differ from beta*m; that distinction
    is what makes detailed balance hold identically.

    STATUS: exact at every finite N, beta, h.
    """
    if N < 2:
        raise ValueError(f"N must be at least 2, got {N}")
    k = np.arange(N + 1, dtype=np.float64)
    w_plus = 0.5 * (N - k) * (1.0 + np.tanh(beta * (2.0 * k - N + 1.0) / (N - 1.0) + h))
    w_minus = 0.5 * k * (1.0 - np.tanh(beta * (2.0 * k - N - 1.0) / (N - 1.0) + h))
    w_plus[N] = 0.0
    w_minus[0] = 0.0
    if check:
        assert_rate_bounds(N, w_plus, w_minus)
    return w_plus, w_minus


def assert_rate_bounds(N: int, w_plus: np.ndarray, w_minus: np.ndarray) -> None:
    """Runtime rate-bound guard: W+ , W- >= 0 and W+_k + W-_k <= N."""
    if not (np.all(np.isfinite(w_plus)) and np.all(np.isfinite(w_minus))):
        raise RateBoundError(f"non-finite rate at N={N}")
    if w_plus.min() < 0.0 or w_minus.min() < 0.0:
        raise RateBoundError(
            f"negative rate at N={N}: min W+={w_plus.min():.3e}, min W-={w_minus.min():.3e}"
        )
    total_max = float((w_plus + w_minus).max())
    if total_max > N * (1.0 + 8.0 * EPS_MACH):
        raise RateBoundError(f"max(W+ + W-)={total_max!r} exceeds N={N}")


def wall_index(N: int, m_wall: float) -> int:
    """State index of a reflecting wall at intensive magnetisation m_wall (m = 2k/N - 1).

    Rounding is half-up, hence monotone in N.  Python's built-in ``round`` is
    half-to-even, which at m_wall = 0 sends N = 4j+1 down and N = 4j+3 up, so the
    wall lands on opposite sides of m = 0 depending on N mod 4.  The in-well
    amplitude depends on the wall location, so this produced a ~1.5% parity sawtooth in gap_well across consecutive N.

    The reflected chain needs N even for the wall at m = 0 -- the only m_wall an
    integer k can realise exactly.  Odd N is refused there rather than silently
    displaced by one site.
    """
    if N < 2:
        raise ValueError(f"N must be at least 2, got {N}")
    if m_wall == 0.0 and N % 2 == 1:
        raise ValueError(
            f"the wall at m=0 requires N even; got N={N}, for which "
            f"m = 2k/N - 1 = 0 has no integer solution. Displacing the wall by one site "
            f"changes the amplitude; pass an explicit m_wall instead."
        )
    k = int(np.floor(0.5 * N * (1.0 + m_wall) + 0.5))
    if not 0 <= k < N:
        raise ValueError(f"wall at m={m_wall} maps to k={k}, outside [0, {N})")
    return k


@dataclass(frozen=True)
class Tridiagonal:
    """Symmetric tridiagonal S on states {k_wall, ..., N}."""

    diag: np.ndarray
    offdiag: np.ndarray
    n_spins: int
    k_wall: int

    @property
    def size(self) -> int:
        return int(self.diag.size)

    @property
    def norm_bound(self) -> float:
        """The bound ||S||_2 <= 2N, used by the noise-floor guard."""
        return 2.0 * self.n_spins

    @property
    def gershgorin_norm(self) -> float:
        """Gershgorin bound on ||S||_2, computed from the assembled matrix."""
        radius = np.abs(self.diag).copy()
        radius[:-1] += np.abs(self.offdiag)
        radius[1:] += np.abs(self.offdiag)
        return float(radius.max())


def symmetric_generator(
    N: int, beta: float, h: float = 0.0, *, k_wall: int = 0
) -> Tridiagonal:
    """Assemble S = D^{1/2} G D^{-1/2} directly from the rates.

    ``k_wall > 0`` imposes the reflecting wall by
    setting W^-_{k_wall} := 0; ``k_wall = 0`` gives the full chain.
    The stationary law pi is never formed: S contains no reference to it, which
    keeps the assembly free of overflow.

    STATUS: exact.
    """
    w_plus, w_minus = flip_rates(N, beta, h)
    w_minus = w_minus.copy()
    w_minus[k_wall] = 0.0
    diag = -(w_plus[k_wall:] + w_minus[k_wall:])
    offdiag = np.sqrt(w_plus[k_wall:N] * w_minus[k_wall + 1 :])
    tri = Tridiagonal(diag, offdiag, n_spins=N, k_wall=k_wall)
    if tri.gershgorin_norm > tri.norm_bound * (1.0 + 8.0 * EPS_MACH):
        raise RateBoundError(
            f"Gershgorin ||S||={tri.gershgorin_norm!r} exceeds the 2N bound "
            f"that the noise-floor guard relies on"
        )
    return tri


def jump_moments(N: int, beta: float, h: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Exact first two jump moments A_N(m), B_N(m), indexed by k.

    m = 2k/N - 1.  These converge to the continuum drift and diffusion
    with an O(1/N) error.

    STATUS: exact at every finite N.
    """
    w_plus, w_minus = flip_rates(N, beta, h)
    return (2.0 / N) * (w_plus - w_minus), (4.0 / N**2) * (w_plus + w_minus)
