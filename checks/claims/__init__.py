"""Recompute every number the paper prints and compare it at the printed precision.

    python -m checks.claims

A value printed as ``0.199`` passes when the recomputed value rounds to 0.199;
no tolerance is wider than the printed precision unless the row says why.

  (a) ``exact_chain``  recomputed live from the exact (N+1)-state chain in ``src/``
  (b) ``stored``       recomputed from saved results, with no rerun
  (c) ``reruns``       needs a GPU rerun or external data; reported as UNVERIFIED

``checks/claims.md`` lists every row with its location in the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PASS, FAIL, UNVERIFIED = "PASS", "FAIL", "UNVERIFIED"


@dataclass
class Result:
    id: str
    location: str
    printed: str
    recomputed: str
    tolerance: str
    status: str
    source: str
    note: str = ""


def decimals(printed: str) -> int:
    """Number of digits after the decimal point in a printed value."""
    text = printed.replace(",", "").strip().lstrip("+-")
    return len(text.split(".")[1]) if "." in text else 0


def rounds_to(value: float, printed: str) -> bool:
    """True when ``value`` rounds to ``printed`` at the printed precision.

    Half a unit in the last printed place, plus 1e-12 for float noise.
    """
    target = float(printed.replace(",", ""))
    return abs(value - target) <= 0.5 * 10.0 ** (-decimals(printed)) + 1e-12


def fmt(value: float, printed: str, extra: int = 2) -> str:
    """Show the recomputed value with ``extra`` digits beyond the printed ones."""
    return f"{value:.{decimals(printed) + extra}f}"


def check_rounding(
    id: str, location: str, printed: str, value: float, source: str, note: str = ""
) -> Result:
    d = decimals(printed)
    return Result(
        id, location, printed, fmt(value, printed),
        f"rounds to {d} dp" if d else "rounds to integer",
        PASS if rounds_to(value, printed) else FAIL, source, note,
    )


def check_interval(
    id: str, location: str, lo_printed: str, hi_printed: str,
    lo: float, hi: float, source: str, note: str = "", slack: float = 0.0,
) -> Result:
    """Both endpoints at printed precision; ``slack`` widens for parsed input."""
    d = decimals(lo_printed)
    ok = (abs(lo - float(lo_printed)) <= 0.5 * 10.0 ** (-d) + slack + 1e-12
          and abs(hi - float(hi_printed)) <= 0.5 * 10.0 ** (-d) + slack + 1e-12)
    tol = f"ends round to {d} dp" + (f" (+{slack:g} parse)" if slack else "")
    return Result(
        id, location, f"[{lo_printed}, {hi_printed}]",
        f"[{lo:.{d + 2}f}, {hi:.{d + 2}f}]", tol, PASS if ok else FAIL, source, note,
    )


def check_pow10(
    id: str, location: str, exponent: int, log10_value: float, source: str, note: str = ""
) -> Result:
    return Result(
        id, location, f"10^{exponent}", f"10^{log10_value:.2f}",
        "log10 rounds to integer",
        PASS if round(log10_value) == exponent else FAIL, source, note,
    )


def check_sci(
    id: str, location: str, mantissa: str, exponent: int, value: float,
    source: str, note: str = "",
) -> Result:
    """A value printed as ``mantissa x 10^exponent``; the mantissa sets the precision."""
    scaled = value / 10.0 ** exponent
    return Result(
        id, location, f"{mantissa}e{exponent:+d}", f"{value:.3e}",
        f"mantissa rounds to {decimals(mantissa)} dp",
        PASS if rounds_to(scaled, mantissa) else FAIL, source, note,
    )


def check_bool(
    id: str, location: str, printed: str, ok: bool, recomputed: str,
    tolerance: str, source: str, note: str = "",
) -> Result:
    return Result(id, location, printed, recomputed, tolerance, PASS if ok else FAIL, source, note)


def unverified(id: str, location: str, printed: str, source: str, note: str) -> Result:
    return Result(id, location, printed, "-", "-", UNVERIFIED, source, note)


def sig1(value: float) -> float:
    """Round to one significant figure (for words such as 'sixty-fold')."""
    if value == 0.0:
        return 0.0
    exp = math.floor(math.log10(abs(value)))
    return round(value / 10.0 ** exp) * 10.0 ** exp
