"""Exact Curie-Weiss chain: rates, spectral gaps, first passage, and their checks.

Modules
-------
``rates``        exact birth-death rates (paper Eqs. 8-9) and the symmetrised
                 tridiagonal generator S
``gap``          spectral gaps of the full and reflected chains, with
                 resolution-floor guards
``landau``       continuum objects: F, Phi, the drift A
``barrier``      finite-N log-potential, log-space mean first-passage time,
                 Arrhenius asymptotics
``normal_form``  relaxation rates of the normal form at beta = 1
``brute_force``  independent 2^N Gibbs generator, a check of the lumping
``cw_ramp``      linear ramps of the exact chain (paper Table 2)
``validate``     the checks run by ``python -m src``

Subpackage
----------
``simple``       the Hopfield test of the written protocol (docs/protocol-simple.md)

Run the checks::

    python -m src
"""

from __future__ import annotations

# No re-exports: submodules are imported explicitly (``from src.gap import gap_well``)
# so that importing one does not pull in the others.
