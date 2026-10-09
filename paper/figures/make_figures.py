"""Figures 1 and 2 of the paper, drawn at their final six-inch width.

Reads saved results; the equilibrium peaks and selection Gaussian are
analytical illustrations. No simulation is run.
Run from the repository root with:
    python paper/figures/make_figures.py [--out DIR]
The default --out is paper/figures/; it overwrites the PDFs the paper includes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import NullLocator
from scipy.special import logit, ndtr

from style import (
    ACCENT, BLUE, INK, MODE_NEG, MODE_POS, MUTED, SIZE_COLORS,
    configure, panel_heading, save, style_axis,
)


def read(root, path):
    return json.loads((root / path).read_text())


def mechanism(root, out):
    sys.path.insert(0, str(root))
    from src.barrier import log_pi_closed_form

    paths = read(root, "visual/data/fork_trajectories.json")
    beta, trajectories = np.array(paths["beta"]), np.array(paths["m"])
    assert trajectories.shape == (14, 360)
    assert int((trajectories[:, -1] > 0).sum()) == 9
    # Exact equilibrium at N=1024, G=0: the magnetization of the most probable
    # count at each beta (one peak at 0 below the transition, two above it),
    # not an estimate from the fourteen diffusion paths.
    log_mass = np.column_stack([log_pi_closed_form(1024, b) for b in beta])
    m_peak = np.abs(2.0 * log_mass.argmax(axis=0) / 1024 - 1.0)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6, 1.96))
    fig.subplots_adjust(left=0.077, right=0.982, bottom=0.245, top=0.835, wspace=0.38)
    panel_heading(fig, ax, "(a)", "Cooling at zero field")
    panel_heading(fig, bx, "(b)", "A field biases the sign")
    ax.axvline(1, color="0.55", lw=0.85, ls=(0, (3.5, 2.5)))
    for m in trajectories:
        line, = ax.plot(beta, m, color=MODE_POS if m[-1] > 0 else MODE_NEG, lw=0.8, alpha=0.95)
    for sign in (1, -1):
        ax.plot(beta, sign * m_peak, color=INK, lw=0.9, ls=(0, (2.2, 1.6)), zorder=3)
    ax.text(0.716, 0.9, r"$N=1024$, $G=0$", color=INK, va="top")
    ax.text(1.017, 0.9, r"$\beta=1$", color=INK, va="top")
    ax.text(1.333, 0.28, "+ sign", color=MODE_POS, ha="right", va="center")
    ax.text(1.333, -0.31, "− sign", color=MODE_NEG, ha="right", va="center")
    ax.annotate("cooling", xy=(0.96, -0.83), xytext=(0.72, -0.83), color="0.4",
                va="center", arrowprops=dict(arrowstyle="->", lw=0.7, color="0.4"))
    ax.set(xlim=(0.7, 1.35), ylim=(-1, 1), yticks=[-1, 0, 1],
           xlabel=r"Inverse temperature $\beta$", ylabel=r"Magnetization $m$")
    ax.set_xticks([0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3],
                  labels=["0.7", "0.8", "0.9", "1.0", "1.1", "1.2", "1.3"])

    linear = read(root, "results/experiments/cw_linear_sweep.json")
    row = next(z for z in linear if z["N"] == 2048 and z["gamma"] == 0.001)
    a = row["G"] * np.sqrt(row["N"]) * (np.pi / row["gamma"]) ** 0.25
    assert abs(ndtr(a) - row["q_KN"]) < 1e-12
    u = np.sort(np.r_[np.linspace(-3.5, 3.8, 1000), 0, a])
    density = np.exp(-0.5*(u-a)**2) / np.sqrt(2*np.pi)
    bx.fill_between(u, density, where=u<=0, color=MODE_NEG, alpha=0.45)
    bx.fill_between(u, density, where=u>=0, color=MODE_POS, alpha=0.50)
    bx.plot(u, density, color=INK, lw=1.55)
    bx.axvline(0, color="0.55", lw=0.7)
    bx.plot([a, a], [0, 1/np.sqrt(2*np.pi)], color=BLUE, lw=1, ls=(0, (1.2, 1.8)))
    bx.annotate(r"mean $a=%.3f$" % a, xy=(a, 0.403), xytext=(1.75, 0.455),
                ha="center", va="center", color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.65))
    bx.text(-0.95, 0.045, "− sign", color=INK, ha="center")
    bx.text(1.25, 0.045, "+ sign", color=INK, ha="center")
    bx.annotate(r"$q_{\rm sel}=%.3f$" % ndtr(a), xy=(1.1, 0.16), xytext=(2.12, 0.30),
                ha="center", color=BLUE, fontsize=10,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.65))
    bx.set(xlim=(-3.5, 3.8), ylim=(0, 0.50), xticks=[-3, 0, 3], yticks=[0, 0.2, 0.4],
           xlabel=r"Standardized early amplitude $U$", ylabel="Probability density")
    for axis in (ax, bx):
        style_axis(axis)
    save(fig, out, "mechanism")


def validation(root, out):
    linear = read(root, "results/experiments/cw_linear_sweep.json")
    grid = read(root, "results/hopfield/field_grid_analysis.json")["points"]
    original = read(root, "results/figures/field_grid_points.json")
    fitted = original["recovery_ratio_plotted"]
    prediction = original["prediction_drawn"]["c_pred"]
    # Only numerical fields are reused; checks/reanalysis.py recomputes the slope.
    assert len(grid) == 8
    assert abs(fitted["point"] - 0.1994302352904474) < 1e-12
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6, 2.30))
    fig.subplots_adjust(left=0.083, right=0.982, bottom=0.215, top=0.72, wspace=0.38)
    panel_heading(fig, ax, "(a)", "CW: linear cooling")
    panel_heading(fig, bx, "(b)", r"Hopfield: $p=3$, $N=2048$")
    for n, color, marker in zip([1024, 2048, 4096], SIZE_COLORS[1:], ["o", "s", "^"]):
        rows = sorted([z for z in linear if z["N"] == n], key=lambda z: z["gamma"])
        rates = np.geomspace(0.00025, 0.004, 160)
        denominator = logit(rows[0]["q_KN"]) / rows[0]["c_KN"]
        curve = logit(ndtr(rows[0]["G"] * np.sqrt(n) * (np.pi / rates) ** 0.25)) / denominator
        ax.plot(rates, curve, color=color, lw=1.65)
        ax.plot([z["gamma"] for z in rows], [z["c_exact"] for z in rows],
                marker, color=color, ms=4.8, markerfacecolor="white", mew=1.15,
                label=f"{n}")
    ax.set_xscale("log")
    ax.set(xlim=(0.00019, 0.0052), ylim=(0.025, 0.21),
           xlabel=r"Cooling rate $\gamma$ (per expected sweep)", ylabel=r"Fraction $c$")
    ax.set_xticks([0.00025, 0.001, 0.004], labels=[r"$2.5\!\times\!10^{-4}$", r"$10^{-3}$", r"$4\!\times\!10^{-3}$"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([0.05, 0.10, 0.15, 0.20])
    sizes = ax.legend(title="$N$", title_fontsize=9, loc="upper right", frameon=False,
              handlelength=0.9, handletextpad=0.35, borderpad=0, labelspacing=0.2)
    ax.add_artist(sizes)
    ax.legend(handles=[Line2D([], [], color=INK, lw=1.65, label="Selection law"),
                       Line2D([], [], color=INK, marker="o", mfc="white", mew=1.1,
                              ms=4.6, lw=0, label="Exact calculation")],
              loc="lower left", bbox_to_anchor=(0, 1.04), frameon=False,
              handlelength=1.8, handletextpad=0.6, borderpad=0, borderaxespad=0,
              labelspacing=0.25)
    xx = np.linspace(-3.95, 3.95, 300)
    bx.axhline(0, color="0.85", lw=0.65)
    bx.axvline(0, color="0.85", lw=0.65)
    bx.plot(xx, xx, color="0.55", ls=(0, (4, 3)), lw=1.1)
    bx.fill_between(xx, np.minimum(fitted["lower"] * xx, fitted["upper"] * xx),
                    np.maximum(fitted["lower"] * xx, fitted["upper"] * xx),
                    color=ACCENT, alpha=0.3)
    bx.plot(xx, fitted["point"] * xx, color=ACCENT, lw=2.3)
    bx.plot(xx, prediction * xx, color=INK, ls=(0, (1.2, 1.6)), lw=1.4)
    x = np.array([logit(z["q_eq"]["point"]) for z in grid])
    y = np.array([logit(z["q_model"]["point"]) for z in grid])
    xerr = np.array([[x[i] - logit(z["q_eq"]["lower"]) for i,z in enumerate(grid)],
                     [logit(z["q_eq"]["upper"]) - x[i] for i,z in enumerate(grid)]])
    yerr = np.array([[y[i] - logit(z["q_model"]["lower"]) for i,z in enumerate(grid)],
                     [logit(z["q_model"]["upper"]) - y[i] for i,z in enumerate(grid)]])
    bx.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="s", color=ACCENT,
                markeredgecolor=INK, ms=4.7, mew=0.8, ecolor=INK, lw=0.8, capsize=2,
                zorder=4)
    bx.text(1.15, 0.88, r"$c=1$", color=MUTED, va="center", ha="left")
    measured_handle = (Patch(facecolor=ACCENT, alpha=0.3, edgecolor="none"),
                       Line2D([], [], color=ACCENT, lw=2.3))
    predicted_handle = Line2D([], [], color=INK, ls=(0, (1.2, 1.6)), lw=1.4)
    bx.legend(handles=[measured_handle, predicted_handle],
              labels=["Measured 0.199, 95% CI", "Predicted 0.199"],
              handler_map={tuple: HandlerTuple(ndivide=1)},
              loc="lower left", bbox_to_anchor=(0, 1.04), frameon=False,
              handlelength=1.8, handletextpad=0.6, borderpad=0, borderaxespad=0,
              labelspacing=0.25)
    bx.set(xlim=(-3.95, 3.95), ylim=(-1.04, 1.04), xticks=[-3, 0, 3], yticks=[-1, 0, 1],
           xlabel=r"Equilibrium log-odds $\operatorname{logit}q_{\rm eq}$",
           ylabel=r"Sampled log-odds $\operatorname{logit}q_{\rm chain}$")
    for axis in (ax, bx):
        style_axis(axis)
    save(fig, out, "validation")


def make_figures(root, out):
    configure()
    out.mkdir(parents=True, exist_ok=True)
    mechanism(root, out)
    validation(root, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    make_figures(args.root, args.out)
