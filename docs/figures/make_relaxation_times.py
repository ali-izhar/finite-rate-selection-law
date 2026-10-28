"""Draw docs/figures/relaxation-times.svg: reflected-chain relaxation times and the schedule.

Reads saved results only. Run from the repository root:
    python docs/figures/make_relaxation_times.py [--out DIR]
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
from matplotlib.ticker import NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "paper/figures"))
from style import ACCENT, BLUE, SIZE_COLORS, configure, panel_heading, save, style_axis


def read(root, path):
    return json.loads((root / path).read_text())


def relaxation(root, out):
    track = read(root, "visual/data/tracking_time.json")
    schedule = read(root, "visual/data/schedule.json")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6, 1.96))
    fig.subplots_adjust(left=0.077, right=0.982, bottom=0.245, top=0.835, wspace=0.38)
    panel_heading(fig, ax, "(a)", "Reflected-chain relaxation")
    panel_heading(fig, bx, "(b)", r"Schedule at $N=2048$")
    ax.axvline(1, color="0.55", lw=0.85, ls=(0, (3.5, 2.5)))
    for n, color in zip([256, 1024, 4096, 16384], SIZE_COLORS):
        t = np.array(track["tau"][str(n)])
        ax.semilogy(track["beta"], t, color=color, lw=1.8)
        k = int(t.argmax())
        label = r"$N=16384$" if n == 16384 else f"{n}"
        ax.annotate(label, xy=(track["beta"][k], t[k]),
                    xytext=(7, 2.4), textcoords="offset points", color=color, va="bottom")
    ax.set(xlim=(0.7, 1.35), ylim=(1.4, 175),
           xlabel=r"Inverse temperature $\beta$", ylabel=r"$\tau_{\rm ref}$ (sweeps)")
    ax.set_xticks([0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3],
                  labels=["0.7", "0.8", "0.9", "1.0", "1.1", "1.2", "1.3"])
    ladder, tau = np.asarray(schedule["ladder"]), np.asarray(schedule["tau"])
    assert np.array_equal(np.ceil(6*tau).astype(int), schedule["K"])
    bx.semilogy(ladder, tau, color="0.63", ls=(0, (1.3, 2.1)), lw=1.05)
    for mask, color, marker, label in ((ladder<=1, BLUE, "o", "Full chain"),
                                       (ladder>1, ACCENT, "s", "Reflected chain")):
        bx.semilogy(ladder[mask], tau[mask], marker, color=color,
                    ms=5, markerfacecolor="white", mew=1.25, label=label)
    bx.axvline(1, color="0.55", lw=0.85, ls=(0, (3.5, 2.5)))
    bx.set(xlim=(0.56, 1.65), ylim=(1.4, 175), xticks=[0.6, 0.8, 1, 1.2, 1.4, 1.6],
           xlabel=r"Inverse temperature $\beta$", ylabel=r"$\tau_{\rm des}$ (sweeps)")
    bx.legend(loc="upper right", frameon=False, handlelength=0.9, handletextpad=0.4,
              borderpad=0, labelspacing=0.35, borderaxespad=0.25)
    for axis in (ax, bx):
        axis.set_yticks([2, 10, 100], labels=["2", "10", "100"])
        axis.yaxis.set_minor_locator(NullLocator())
        style_axis(axis)
    save(fig, out, "relaxation-times", formats=("svg",))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    configure()
    args.out.mkdir(parents=True, exist_ok=True)
    relaxation(args.root, args.out)
