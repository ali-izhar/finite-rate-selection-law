"""The Hopfield model, its GPU sampler and the field-grid experiment (paper Sec. 4, App. B).

Modules
-------
``protocol``        the experiment's constants (schedule, fields, seeds, run counts)
``equilibrium``     the equilibrium reference q_eq, exact CW and weak-field routes
``reference``       NumPy reference model: patterns, local fields, one Gibbs sweep
``sampler``         batched GPU (torch) sampler; bit-for-bit equal to ``reference``
                    on a supplied random stream
``stats``           Wilson intervals, interval sums, slope estimators
``sampler_checks``  checks G1-G4 of the sampler and the saved counts
``pipeline``        the pre-runs, the field-grid driver and ``run_protocol``
                    (the same protocol, unlogged, for ``src.experiments``)
``analysis``        per-field intervals and the summary of the field grid
``seed_bootstrap``  the slope c and its pattern-set bootstrap interval

Run the field grid in this order (GPU; output folder ``$HOPFIELD_OUT``,
default ``results/hopfield``)::

    python -m src.hopfield static
    python -m src.hopfield pre
    python -m src.hopfield grid
    python -m src.hopfield analyse
"""

from __future__ import annotations

# No re-exports: submodules are imported explicitly, as in the src package.
