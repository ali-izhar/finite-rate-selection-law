"""Category (c): rows whose evidence needs a GPU rerun or external data.

Each row names the command that regenerates the saved result; the saved
result itself is checked in ``stored.py``.
"""

from __future__ import annotations

from . import Result, unverified


def checks(shared: dict) -> list[Result]:
    return [
        unverified(
            "C01", "Table 1", "measured c for p = 1..4 at N = 2048, and p = 3 at N = 1024, 4096",
            "python -m src.experiments.hopfield_runs draws P (P = 1..4; 10-21 min GPU each); "
            "... draws 3 1024,4096 (about 48 min GPU)",
            "needs a GPU; the saved counts reproduce Table 1 in B04-B07i and B10",
        ),
        unverified(
            "C02", "Abstract, Sec. 4, Fig. 2b, Table 4", "the field grid (eight fields, twenty pattern sets)",
            "python -m src.hopfield static | pre | grid | analyse (about 37 min GPU for grid)",
            "needs a GPU; the saved per-set counts reproduce B01-B03 and Table 4",
        ),
        unverified(
            "C03", "Sec. 5, App. B, App. C", "one-pattern runs at N = 512..4096",
            "python -m src.experiments.hopfield_runs size-p1 N (about 24 min GPU at N = 1024)",
            "needs a GPU; the saved counts reproduce B13 and B17",
        ),
        unverified(
            "C04", "docs/mnist-illustration.md", "the 24-run MNIST ensemble of the figure",
            "python -m src.experiments.dbm_reheat (needs raw MNIST)",
            "src.dbm re-implements the original sampler, so a rerun reproduces the ensemble "
            "statistically, not byte for byte; the saved run is checked in B30",
        ),
    ]
