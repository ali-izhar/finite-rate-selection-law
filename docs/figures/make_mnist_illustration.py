"""Draw docs/figures/mnist-illustration.png from the saved 24-run ensemble.

Shading marks reheating only. Run from the repository root:
    python docs/figures/make_mnist_illustration.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "paper/figures"))
from style import BLUE, INK, MODE_NEG, MODE_POS, MUTED, ORANGE, configure, save, style_axis


def make_figure(root: Path, out: Path) -> None:
    meta = json.loads((root / "results/figures/mnist_illustration.json").read_text())
    with np.load(root / "results/experiments/dbm_reheat.npz") as saved:
        frames = saved["frames"]
        snapshots = saved["snap_sweep"]
        probabilities = saved["probs"]
        betas = saved["betas"]
    margin = probabilities[:, :, 3] - probabilities[:, :, 8]
    assert margin.shape == (24, 701)
    assert meta["chains_shown"] == [0, 16]
    cols = np.array(meta["cols_sweep"])
    crossing = meta["sweep_beta_star"]  # inherited axis knot, not an inferred transition
    assert abs(betas[crossing] - 0.9333) < 0.005
    knots_s = np.array([0, 300, 400, crossing, 700])
    knots_x = np.array([0, 2, 8, 13, 16])
    coordinate = lambda s: np.interp(s, knots_s, knots_x)
    configure()
    fig = plt.figure(figsize=(6, 2.10))
    left, width, top = 0.09, 0.888, 0.865
    cell_w, cell_h, gap = width/16, width/16*6/2.10, 0.004
    rows = [top - (i + 1)*cell_h for i in range(2)]
    grey = np.array(matplotlib.colors.to_rgb("#ededed"))
    fig.patches.append(plt.Rectangle(
        (left + 2*cell_w, rows[1]), 6*cell_w-gap, 2*cell_h-gap,
        transform=fig.transFigure, facecolor=grey, edgecolor="none", zorder=0))
    for i, chain in enumerate(meta["chains_shown"]):
        for j, sweep in enumerate(cols):
            index = int(np.argmin(abs(snapshots-sweep)))
            assert snapshots[index] == sweep
            pixels = frames[chain, index].reshape(28, 28).astype(float)
            bg = grey if 2 <= j < 8 else np.ones(3)
            axis = fig.add_axes([left+j*cell_w, rows[i], cell_w-gap, cell_h-gap], zorder=1)
            axis.imshow((1-pixels)[:, :, None]*bg[None, None, :], interpolation="none")
            axis.set(xticks=[], yticks=[])
            for spine in axis.spines.values():
                spine.set(linewidth=0.4, color="#c4c4c4")
    ax = fig.add_axes([left, 0.18, width, 0.28])
    x = coordinate(np.arange(701))
    ax.axvspan(2, 8, color=grey, lw=0)
    for sweep in (300, 400, 600):
        ax.axvline(coordinate(sweep), lw=0.5, color="0.72", ls=(0, (2, 3)), zorder=0)
    ax.axhline(0, color=INK, lw=0.7, zorder=3)
    for values in margin:
        points = np.column_stack((x, values)).reshape(-1, 1, 2)
        segments = np.concatenate((points[:-1], points[1:]), axis=1)
        colors = np.where(values[:-1] >= 0, MODE_POS, MODE_NEG)
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=0.65,
                                        alpha=0.68, rasterized=False))
    ax.set(xlim=(0, 16), ylim=(-1.16, 1.16), yticks=[-1, 0, 1])
    ax.set_ylabel(r"$P_{\rm clf}(3)-P_{\rm clf}(8)$", labelpad=2, fontsize=9)
    ax.set_xlabel("Sweep (nonuniform spacing)", labelpad=2)
    ticks = [0, 300, 400, 600, 700]  # the 454 knot stays internal
    ax.set_xticks(coordinate(ticks), labels=[str(t) for t in ticks])
    style_axis(ax)
    ax.tick_params(length=2.5, width=0.65)
    ax.text(0.15, 0.70, "3 over 8", color=BLUE, va="center")
    ax.text(coordinate(600) - 0.12, -0.75, "8 over 3", color=ORANGE, ha="right", va="center")
    annotation_y = (0.46+rows[1])/2
    fig.text(left, annotation_y, "All 24 runs", color=MUTED, va="center")
    fig.text(left + width*7.8/16, annotation_y, "Shading: reheat stage", color=MUTED,
             ha="center", va="center")
    stages = [(0, 300, r"initial $\beta=2$", "left"),
              (300, 400, r"reheat $\beta=0.55$", "center"),
              (400, 600, "cool, biased toward 3", "center"),
              (600, 700, "settle", "right")]
    for start, end, label, alignment in stages:
        x0, x1 = left + width*coordinate(start)/16, left + width*coordinate(end)/16
        # Stage bars use the very same time mapping as the traces below.
        fig.add_artist(Line2D([x0+0.003, x1-0.003], [top+0.017]*2,
                             transform=fig.transFigure, color="0.65", lw=0.65))
        location = {"left": x0, "right": x1, "center": (x0+x1)/2}[alignment]
        fig.text(location, top+0.045, label, ha=alignment, va="bottom")
    fig.text(0.042, (rows[1]+rows[0]+cell_h)/2, "2 of 24 runs", rotation=90,
             ha="center", va="center", color=MUTED)
    out.mkdir(parents=True, exist_ok=True)
    save(fig, out, "mnist-illustration", formats=("png",), dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    make_figure(args.root, args.out)
