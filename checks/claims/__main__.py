"""Run every claim check and print the table.

    python -m checks.claims

Exit status is 1 when any row is FAIL.  UNVERIFIED rows (category c) do not
change the exit status; they are listed with the reason.
"""

from __future__ import annotations

import sys
import time

from . import FAIL, UNVERIFIED, Result
from . import exact_chain, posthoc, reruns, stored

COLS = (("id", 6), ("printed", 28), ("recomputed", 32), ("tolerance", 24), ("status", 11))


def _cell(text: str, width: int) -> str:
    text = str(text)
    return (text[: width - 2] + "..") if len(text) > width else text.ljust(width)


def render(rows: list[Result]) -> str:
    head = "".join(_cell(name, w) for name, w in COLS) + "source"
    lines = [head, "-" * (len(head) + 40)]
    for r in rows:
        cells = (r.id, r.printed, r.recomputed, r.tolerance, r.status)
        lines.append("".join(_cell(c, w) for c, (_, w) in zip(cells, COLS)) + r.source)
    return "\n".join(lines)


def main() -> int:
    shared: dict = {}
    rows: list[Result] = []
    for module in (exact_chain, stored, posthoc, reruns):
        t0 = time.time()
        rows.extend(module.checks(shared))
        print(f"[{module.__name__.split('.')[-1]}: {time.time() - t0:.1f} s]", file=sys.stderr)

    print(render(rows))
    fails = [r for r in rows if r.status == FAIL]
    unv = [r for r in rows if r.status == UNVERIFIED]
    print()
    print(f"{len(rows) - len(fails) - len(unv)} PASS, {len(fails)} FAIL, {len(unv)} UNVERIFIED")
    if fails:
        print("\nFAIL details:")
        for r in fails:
            print(f"  {r.id} [{r.location}] printed {r.printed}, recomputed {r.recomputed}")
            if r.note:
                print(f"      {r.note}")
    if unv:
        print("\nUNVERIFIED (category c):")
        for r in unv:
            print(f"  {r.id} [{r.location}] {r.printed}\n      how: {r.source}\n      why: {r.note}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
