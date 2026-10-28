"""Category (c): rows whose evidence needs a GPU rerun.

Each row names the command that regenerates the saved result; the saved result itself is
checked in ``stored.py``.
"""

from __future__ import annotations

from . import Result, unverified


def checks(shared: dict) -> list[Result]:
    return [
        unverified(
            "C10", "Table 3, Fig. 2b", "the eight Hopfield rows of the protocol",
            "python -m src.simple rows --batch 131072 (GPU, about 2.9 h), then python -m src.simple analyse",
            "needs a GPU; the saved counts reproduce Table 3 in B41-B56; a rerun with new "
            "seeds (--seed-offset 1) reproduces them statistically, not bit for bit",
        ),
    ]
