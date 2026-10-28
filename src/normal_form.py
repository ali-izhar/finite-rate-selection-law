"""Relaxation rates of the van Kampen normal form at beta = 1.

These two constants are the sharpest available check on the chain
implementation: they fix the amplitudes, not just the exponent.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh_tridiagonal


def normal_form_rates(
    count: int = 3, *, half_width: float = 12.0, n_grid: int = 20000
) -> np.ndarray:
    """Lowest ``count`` relaxation rates of L_inf = -(u^3/3) d_u + d_u^2.

    Diagonalises the equivalent Schroedinger operator
    H = -d_u^2 + (u^3/3)^2/4 - u^2/2 on [-L, L] with Dirichlet ends, by
    second-order finite differences.  Rates are returned relative to the
    numerical ground state, whose exact value is 0.

    STATUS: numerical.  The two constants are known to five digits from
    exactly this route and are not available in closed form.
    """
    u = np.linspace(-half_width, half_width, n_grid + 2)[1:-1]
    step = float(u[1] - u[0])
    potential = (u**3 / 3.0) ** 2 / 4.0 - u**2 / 2.0
    diag = 2.0 / step**2 + potential
    offdiag = np.full(u.size - 1, -1.0 / step**2)
    vals = eigh_tridiagonal(
        diag, offdiag, eigvals_only=True, select="i", select_range=(0, count - 1)
    )
    return np.asarray(vals) - vals[0]


def richardson_rates(count: int = 3, *, half_width: float = 12.0, n_grid: int = 10000) -> np.ndarray:
    """``normal_form_rates`` with the O(step^2) finite-difference error extrapolated away.

    STATUS: numerical.
    """
    coarse = normal_form_rates(count, half_width=half_width, n_grid=n_grid)
    fine = normal_form_rates(count, half_width=half_width, n_grid=2 * n_grid)
    return (4.0 * fine - coarse) / 3.0
