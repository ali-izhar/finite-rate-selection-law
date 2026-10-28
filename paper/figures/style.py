"""Shared typography and colours for the six-inch figures of the paper."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.text import Text

from colormaps import ACCENT, INK, MODE_NEG, MODE_POS, oslo_ramp, scico, hex_of

BLUE = hex_of(scico("vik", 0.22))
SIZE_COLORS = [oslo_ramp(0.12 + 0.88*i/3) for i in range(4)]
RULE = "#9a9ca0"


def configure():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Liberation Serif"],
        "mathtext.fontset": "stix", "font.size": 9,
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "axes.linewidth": 0.65,
        "pdf.fonttype": 42, "ps.fonttype": 42, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.labelpad": 3, "xtick.major.pad": 3, "ytick.major.pad": 3,
        "lines.solid_capstyle": "round", "savefig.dpi": 300, "svg.hashsalt": "figures",
    })


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(RULE)
    ax.tick_params(length=3, width=0.65, color=RULE)
    ax.set_axisbelow(True)


def panel_heading(fig, ax, label, title, y=0.976):
    x = ax.get_position().x0
    fig.text(x, y, label, fontsize=10, fontweight="bold", va="top")
    fig.text(x + 0.039, y, title, fontsize=10, va="top")


def save(fig, out, name, formats=("pdf", "png"), dpi=300):
    # Check visible labels at their final physical size, rather than shrinking
    # the canvas to fit them. Tick labels are created by draw().
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for obj in fig.findobj(Text):
        if not obj.get_visible() or not obj.get_text().strip():
            continue
        assert obj.get_fontsize() >= 9, (name, obj.get_text(), obj.get_fontsize())
        box = obj.get_window_extent(renderer)
        assert box.x0 >= -0.5 and box.y0 >= -0.5, (name, obj.get_text(), box)
        assert box.x1 <= bounds.x1+0.5 and box.y1 <= bounds.y1+0.5, (name, obj.get_text(), box)
    for suffix in formats:
        # Raster and web formats get an opaque white page, readable on dark themes.
        options = {} if suffix == "pdf" else {"facecolor": "white", "transparent": False}
        if suffix == "png":
            options["dpi"] = dpi
        # No timestamps, so a rebuild reproduces the committed files byte for byte.
        options["metadata"] = {"pdf": {"CreationDate": None}, "svg": {"Date": None}}.get(suffix, {})
        fig.savefig(out / f"{name}.{suffix}", **options)
    plt.close(fig)
