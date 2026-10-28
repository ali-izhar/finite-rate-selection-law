"""Figure colours, sampled from Crameri's Scientific Colour Maps (cmcrameri).

Crameri, Shephard and Heron, Nat. Commun. 11, 5444 (2020): perceptually uniform,
colour-blind safe, monotone in lightness along a sequential ramp.

  oslo    sequential; density images and ordered N
  vik     diverging; the two signs
  batlow  sequential; one warm accent at 0.70
"""
from __future__ import annotations

from cmcrameri import cm as _cm


def scico(palette: str, x: float) -> tuple[float, float, float]:
    """Sample a named palette at x in [0, 1]."""
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"palettes are defined on [0, 1], got {x}")
    r, g, b, _ = getattr(_cm, palette)(float(x))
    return (float(r), float(g), float(b))


def hex_of(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{round(255 * c):02x}" for c in rgb)


INK = "#2d3142"
ACCENT = hex_of(scico("batlow", 0.70))
MODE_POS = hex_of(scico("vik", 0.32))       # + sign
MODE_NEG = hex_of(scico("vik", 0.70))       # - sign


def oslo_ramp(frac: float) -> str:
    """Ordered-N ramp, dark = large: frac in [0, 1] maps oslo 0.70 -> 0.15."""
    return hex_of(scico("oslo", 0.70 - 0.55 * float(frac)))
