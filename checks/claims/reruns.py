"""Category (c): rows whose evidence needs a GPU rerun.

Each row names the command that regenerates the saved result; the saved result itself is
checked in ``stored.py``.
"""

from __future__ import annotations

from . import Result, unverified


def checks(shared: dict) -> list[Result]:
    return [
        unverified(
            "C10", "Table 1, Table 3, Fig. 2b", "the eight pre-registered Hopfield rows",
            "python -m src.simple rows --batch 131072 (GPU, about 2.9 h), then python -m src.simple analyse",
            "needs a GPU; the saved counts reproduce Table 1 and Table 3 in B41-B57; a rerun with new "
            "seeds (--seed-offset 1) reproduces them statistically, not bit for bit",
        ),
        unverified(
            "C11", "App. B", "gate G0: the GPU sampler equals the NumPy reference bit for bit",
            "python -m src.simple g0 (GPU, seconds); python -m pytest -q src/simple/tests (with torch)",
            "needs torch; the saved result is checked in B54",
        ),
    ]
