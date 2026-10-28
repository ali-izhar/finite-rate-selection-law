"""The paper's two figures, drawn at the six-inch text width: teaser and competition.

teaser.pdf       Story: runs split into two modes near beta = 1; a weak field shifts the early
                 amplitude by a fraction of its spread; the fraction c grows as the fourth root
                 of the ramp length and stays far below 1.
competition.pdf  Story: the largest of three amplitudes has a larger mean size than one; pooled
                 over pattern sets the measured c is within 6% of the predictions; single
                 pattern sets scatter, and c is lower where pattern 1 wins more runs.

The panels carry data, axis labels, series labels and reference lines only; readings and
numbers belong to the caption and the text.

Reads saved results only; no simulation is run. Every plotted value is asserted against the
numbers the paper prints (Tables 1, 3 and 4, Fig. 1 caption, Eq. for A_p) before it is drawn.
Run from the repository root with:
    python paper/figures/make_figures.py [--out DIR]
The default --out is paper/figures/.
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
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import NullFormatter, NullLocator
from scipy.integrate import quad
from scipy.special import expit, logit, ndtr
from scipy.stats import norm, spearmanr

from style import INK, MODE_NEG, MODE_POS, configure, style_axis
import make_fork

# Entity -> colour and marker; one meaning per hue.
#   mode (sign)   + MODE_POS (blue), - MODE_NEG (terracotta)
#   system size   N = 1024, 2048, 4096 -> a neutral ramp, light to dark, markers o, s, ^
#   exact rows    INK; p = 1 open circle, p = 2 filled diamond
#   A_p rows      AP (plum), in all three competition panels; p = 3 square, p = 4 triangle
#   context       greys
# AP was chosen against a deep teal: smallest CIELAB distance to the two mode colours and the ink
# under normal vision / deuteranopia / protanopia (Machado et al. 2009) is 45 / 24 / 30 for
# #7b3f8c and 32 / 25 / 28 for #1b7a6e; its lightness (L* 37) differs from the modes' (63) by 26.
AP = "#7b3f8c"
GREY = "0.40"          # reference labels
REF = "0.55"           # reference lines
LABEL, NOTE = 9, 8     # axis/tick/heading size and series-label size, in points on the page
LW = 1.4               # every data curve
DASH = (0, (3.5, 2.5))
DOTS = (0, (1.2, 1.8))
PROBLEMS = []


def tint(color, f):
    """Mix a colour toward white; f = 1 is white."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


SIZES = [(1024, tint(INK, 0.50), "o"), (2048, tint(INK, 0.26), "s"), (4096, INK, "^")]
MARKS = {1: dict(marker="o", color=INK, mfc="white", ms=4.6),
         2: dict(marker="D", color=INK, mfc=INK, ms=3.5),
         3: dict(marker="s", color=AP, mfc=AP, ms=4.2),
         4: dict(marker="^", color=AP, mfc=AP, ms=5.0)}

# ---- The numbers the paper prints (main.tex). The script asserts the data against them. ----
TABLE_RAMPS = {  # N -> gamma -> (exact c, percentage excess over the law)
    1024: {0.004: ("0.09370", "0.93"), 0.001: ("0.13393", "1.81"), 0.00025: ("0.18868", "1.03")},
    2048: {0.004: ("0.06587", "0.46"), 0.001: ("0.09372", "0.97"), 0.00025: ("0.13314", "1.23")},
    4096: {0.004: ("0.04645", "0.23"), 0.001: ("0.06589", "0.49"), 0.00025: ("0.09361", "0.86")},
}
TABLE_RESULTS = {  # row -> predicted c, measured c, lo, hi, ratio
    "N1024_p1": ("0.0937", "0.0928", "0.0909", "0.0947", "0.990"),
    "N2048_p1": ("0.0659", "0.0659", "0.0647", "0.0672", "1.001"),
    "N4096_p1": ("0.0464", "0.0472", "0.0464", "0.0480", "1.015"),
    "N2048_p2": ("0.0936", "0.0946", "0.0928", "0.0965", "1.011"),
    "N1024_p3": ("0.1558", "0.1518", "0.1463", "0.1571", "0.975"),
    "N2048_p3": ("0.1095", "0.1090", "0.1059", "0.1123", "0.995"),
    "N4096_p3": ("0.0772", "0.0758", "0.0734", "0.0782", "0.982"),
    "N2048_p4": ("0.1209", "0.1137", "0.1069", "0.1210", "0.940"),
}
TABLE4 = {  # row -> win share min, max, Spearman
    "N1024_p3": ("0.075", "0.692", "-0.75"),
    "N2048_p3": ("0.168", "0.566", "-0.69"),
    "N4096_p3": ("0.229", "0.522", "-0.56"),
    "N2048_p4": ("0.087", "0.713", "-0.72"),
}
A_PRINTED = {1: "1.000", 2: "1.414", 3: "1.662", 4: "1.836"}


def read(root, path):
    return json.loads((root / path).read_text())


def same(value, printed):
    """True if `value` rounds to the printed string."""
    digits = len(printed.split(".")[1])
    return f"{value:.{digits}f}" == printed


def heading(fig, ax, label, title, y=0.975):
    x = ax.get_position().x0
    bold = fig.text(x, y, label, fontsize=LABEL, fontweight="bold", va="top")
    fig.canvas.draw()
    width = bold.get_window_extent(fig.canvas.get_renderer()).width / fig.bbox.width
    fig.text(x + width + 0.008, y, title, fontsize=LABEL, va="top")


def save(fig, out, name):
    """Check the text (size, inside the page, no two labels overlapping), then write PDF and PNG."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes, problems = [], []
    for obj in fig.findobj(Text):
        if not obj.get_visible() or not obj.get_text().strip():
            continue
        box = obj.get_window_extent(renderer)
        if obj.get_fontsize() < NOTE:
            problems.append(("too small", obj.get_text(), obj.get_fontsize()))
        if (box.x0 < -0.5 or box.y0 < -0.5 or box.x1 > fig.bbox.x1 + 0.5
                or box.y1 > fig.bbox.y1 + 0.5):
            problems.append(("outside the page", obj.get_text(), box.extents.round(1).tolist()))
        boxes.append((obj.get_text(), box))
    for i, (s, a) in enumerate(boxes):
        for t, b in boxes[i + 1:]:
            if a.overlaps(b):
                problems.append(("labels overlap", s, t))
    fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(out / f"{name}.png", dpi=300, facecolor="white", transparent=False)
    plt.close(fig)
    PROBLEMS.extend((name, *z) for z in problems)   # raised by make_figures, after both files exist
    print(f"wrote {out / name}.pdf and .png  ({fig.get_size_inches()[0]:.2f} x "
          f"{fig.get_size_inches()[1]:.2f} in)")


# --------------------------------------------------------------------------------------------
def teaser(root, out):
    sys.path.insert(0, str(root))
    from src.barrier import log_pi_closed_form
    from src.cw_ramp import q_eq

    paths = read(root, "results/figures/fork_trajectories.json")
    linear = read(root, "results/experiments/cw_linear_sweep.json")
    predictions = {(z["N"], z["p"]): z for z in read(root, "results/simple/predictions.json")["rows"]}
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(6, 2.1),
                                     gridspec_kw=dict(width_ratios=[1, 0.95, 1.10]))
    fig.subplots_adjust(left=0.066, right=0.992, bottom=0.225, top=0.855, wspace=0.48)
    heading(fig, ax, "(a)", "Paths during cooling")
    heading(fig, bx, "(b)", "Early amplitude")
    heading(fig, cx, "(c)", "Fraction kept")

    # (a) the fourteen saved paths of make_fork.py (N = 1024, G = 0, NumPy seed 4), unsmoothed.
    beta, traj = np.array(paths["beta"]), np.array(paths["m"])
    assert paths["N"] == 1024 and traj.shape == (14, 360) and paths["ended_up"] == 9
    assert int((traj[:, -1] > 0).sum()) == 9
    assert np.allclose(beta, make_fork.BETAS, atol=1e-6)
    # Selection window: |t| < T = gamma^(-1/2), i.e. |beta - 1| < sqrt(gamma), with gamma the
    # rate of the paths' ramp (one beta step per SPB sweeps). An order-of-magnitude band.
    rate = (make_fork.BETAS[1] - make_fork.BETAS[0]) / make_fork.SPB
    half = np.sqrt(rate)
    ax.axvspan(1 - half, 1 + half, color="0.5", alpha=0.22, lw=0, zorder=0)
    ax.axvline(1, color=REF, lw=0.7, ls=DASH, zorder=1)
    for m in traj:
        ax.plot(beta, m, color=MODE_POS if m[-1] > 0 else MODE_NEG, lw=0.5, alpha=0.8)
    # Exact equilibrium at N = 1024, G = 0: the most probable magnetization at each beta.
    log_mass = np.column_stack([log_pi_closed_form(1024, b) for b in beta])
    m_peak = np.abs(2.0 * log_mass.argmax(axis=0) / 1024 - 1.0)
    for sign in (1, -1):
        ax.plot(beta, sign * m_peak, color=INK, lw=0.8, ls=(0, (2.2, 1.6)), zorder=3)
    ax.text(1.335, 0.30, "+ mode", color=MODE_POS, ha="right", va="center", fontsize=NOTE)
    ax.text(1.335, -0.30, "− mode", color=MODE_NEG, ha="right", va="center", fontsize=NOTE)
    ax.set(xlim=(0.7, 1.35), ylim=(-1, 1), yticks=[-1, 0, 1],
           xlabel=r"Inverse temperature $\beta$", ylabel=r"Magnetization $m$")
    ax.set_xticks([0.7, 1.0, 1.3], labels=["0.7", "1.0", "1.3"])
    ax.yaxis.labelpad = -2
    print(f"teaser a: 14 paths, 9 end at m > 0; band |beta - 1| < sqrt(gamma) = {half:.4f} "
          f"(gamma = {rate:.3e} per sweep)")

    # (b) U ~ N(a, 1) on the ramp of the Fig. 1 caption: N = 2048, gamma = 0.001, NG = 1.2544.
    row = next(z for z in linear if z["N"] == 2048 and z["gamma"] == 0.001)
    a = row["G"] * np.sqrt(row["N"]) * (np.pi / row["gamma"]) ** 0.25
    q = ndtr(a)
    assert abs(row["N"] * row["G"] - 1.2544) < 1e-12 and abs(q - row["q_KN"]) < 1e-12
    assert same(a, "0.208") and same(q, "0.582") and f"{100 * q:.0f}" == "58"
    # Not drawn; for the caption and the text: the model's probability of the + mode at beta = 1.6.
    q_model = q_eq(row["G"], 1.6, 2048)
    assert abs(logit(q_model) - predictions[(2048, 1)]["L_eq"]) < 1e-9 and same(logit(q_model), "3.5747")
    assert abs(logit(row["q_KN"]) / row["c_KN"] - logit(q_model)) < 1e-9
    assert abs(q_model - expit(3.574684631714542)) < 1e-9
    u = np.sort(np.r_[np.linspace(-3.5, 3.8, 1000), 0, a])
    density = norm.pdf(u - a)
    bx.fill_between(u, density, where=u <= 0, color=MODE_NEG, alpha=0.40, lw=0)
    bx.fill_between(u, density, where=u >= 0, color=MODE_POS, alpha=0.40, lw=0)
    bx.plot(u, density, color=INK, lw=LW)
    bx.axvline(0, color=REF, lw=0.7)
    bx.plot([a, a], [0, norm.pdf(0)], color=INK, lw=0.9, ls=DOTS)
    bx.text(-2.45, 0.17, "− mode", color=MODE_NEG, ha="center", va="center", fontsize=NOTE)
    bx.text(2.75, 0.17, "+ mode", color=MODE_POS, ha="center", va="center", fontsize=NOTE)
    bx.set(xlim=(-3.5, 3.8), ylim=(0, 0.45), xticks=[-3, 0, 3], yticks=[0, 0.2, 0.4],
           xlabel=r"Early amplitude $U$", ylabel="Probability density")
    bx.set_xticklabels(["−3", "0", "3"])
    print(f"teaser b: a = {a:.4f}, q_sel = {q:.4f}, exact chain q = {row['q_exact']:.4f}; model "
          f"q_eq = {q_model:.4f} (logit {logit(q_model):.4f}); c = {row['c_exact']:.4f} "
          f"(N = 2048, gamma = 0.001, NG = 1.2544, beta 0.7 to 1.6)")

    # (c) exact c (markers) and the selection law (lines) against the ramp length 0.9 / gamma.
    span = 1.6 - 0.7
    for n, color, marker in SIZES:
        rows = sorted([z for z in linear if z["N"] == n], key=lambda z: z["gamma"])
        assert [z["gamma"] for z in rows] == [0.00025, 0.001, 0.004]
        assert [round(span / z["gamma"]) for z in rows] == [3600, 900, 225]
        for z in rows:
            printed = TABLE_RAMPS[n][z["gamma"]]
            assert abs(n * z["G"] - 1.2544) < 1e-12
            assert same(z["c_exact"], printed[0]), (n, z["gamma"], z["c_exact"])
            assert same(100 * (z["c_exact"] / z["c_KN"] - 1), printed[1]), (n, z["gamma"])
            a_z = z["G"] * np.sqrt(n) * (np.pi / z["gamma"]) ** 0.25
            assert abs(ndtr(a_z) - z["q_KN"]) < 1e-12
            print(f"teaser c: N = {n} gamma = {z['gamma']:<7} sweeps = {span / z['gamma']:6.0f}  "
                  f"exact c = {z['c_exact']:.5f}  law = {z['c_KN']:.5f}  "
                  f"excess = {100 * (z['c_exact'] / z['c_KN'] - 1):.2f}%")
        rates = np.geomspace(0.00025, 0.004, 160)
        l_eq = logit(rows[0]["q_KN"]) / rows[0]["c_KN"]      # the law's denominator, logit q_eq
        law = logit(ndtr(rows[0]["G"] * np.sqrt(n) * (np.pi / rates) ** 0.25)) / l_eq
        for z in rows:                                        # the line passes through c_KN
            assert abs(np.interp(np.log(z["gamma"]), np.log(rates), law) - z["c_KN"]) < 2e-6
        cx.plot(span / rates, law, color=color, lw=LW)
        cx.plot([span / z["gamma"] for z in rows], [z["c_exact"] for z in rows], marker,
                color=color, ms=4.6, markerfacecolor="white", mew=1.1, zorder=4)
        cx.annotate(f"$N={n}$" if n == 1024 else f"{n}", xy=(span / rows[0]["gamma"], rows[0]["c_exact"]),
                    xytext=(6, 0), textcoords="offset points", ha="left", va="center",
                    color=color, fontsize=NOTE)
    # Readings for the caption (not drawn): 16 times more sweeps, about twice the fraction.
    c = {(z["N"], z["gamma"]): z["c_exact"] for z in linear}
    for n in (1024, 2048, 4096):
        gain = c[(n, 0.00025)] / c[(n, 0.004)]
        assert abs(gain - 2) < 0.03 and abs(np.log(gain) / np.log(16) - 0.25) < 0.005, n
        print(f"teaser c: N = {n}: c(3600 sweeps) / c(225 sweeps) = {gain:.3f}, "
              f"log-log slope = {np.log(gain) / np.log(16):.4f}")
    assert same(c[(2048, 0.004)], "0.06587") and same(c[(2048, 0.00025)], "0.13314")
    # Slope reference: a rise of 4^(1/4) over a fourfold run, drawn below the N = 4096 line.
    x0, x1, y0 = 700.0, 2800.0, 0.0445
    cx.plot([x0, x1, x1, x0], [y0, y0, y0 * 4 ** 0.25, y0], color=GREY, lw=0.6,
            solid_joinstyle="miter")
    cx.text(x1 * 1.10, y0 * 4 ** 0.125, "1/4", color=GREY, fontsize=NOTE, ha="left", va="center")
    cx.axhline(1, color=REF, lw=0.7, ls=DASH, zorder=1)
    cx.set_xscale("log")
    cx.set_yscale("log")
    cx.set(xlim=(170, 15500), ylim=(0.03, 1.25),
           xlabel="Ramp length (sweeps)", ylabel=r"Fraction $c$")
    cx.set_xticks([225, 900, 3600], labels=["225", "900", "3600"])
    cx.set_yticks([0.05, 0.1, 0.2, 0.5, 1], labels=["0.05", "0.1", "0.2", "0.5", "1"])
    for axis in (cx.xaxis, cx.yaxis):
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())
    cx.legend(handles=[Line2D([], [], color=INK, lw=LW, label="selection law"),
                       Line2D([], [], color=INK, marker="o", mfc="white", mew=1.1, ms=4.6, lw=0,
                              label="exact chain")],
              loc="upper left", bbox_to_anchor=(0.02, 0.90), frameon=False, fontsize=NOTE,
              handlelength=1.5, handletextpad=0.5, borderpad=0, borderaxespad=0, labelspacing=0.25)
    cx.yaxis.labelpad = 1
    for axis in (ax, bx, cx):
        style_axis(axis)
    save(fig, out, "teaser")


# --------------------------------------------------------------------------------------------
def amplification(p):
    """A_p = E max_{mu <= p} |Z_mu| / E|Z| by quadrature, the two means and the density."""
    dens = lambda x: p * 2 * norm.pdf(x) * (2 * norm.cdf(x) - 1) ** (p - 1)
    mass = quad(dens, 0, np.inf)[0]
    assert abs(mass - 1) < 1e-10
    mean_max = quad(lambda x: x * dens(x), 0, np.inf)[0]
    mean_one = quad(lambda x: x * 2 * norm.pdf(x), 0, np.inf)[0]
    assert abs(mean_one - np.sqrt(2 / np.pi)) < 1e-12
    return mean_max / mean_one, mean_one, mean_max, dens


def competition(root, out):
    sys.path.insert(0, str(root))
    from src.simple.analyse import counts_for_estimator
    from src.simple.estimate import estimate
    from src.simple.predict import amplification as amplification_repo

    analysis = read(root, "results/simple/analysis.json")["rows"]
    predictions = {(z["N"], z["p"]): z for z in read(root, "results/simple/predictions.json")["rows"]}
    posthoc = read(root, "results/posthoc/per_set.json")["rows"]

    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(6, 2.1),
                                     gridspec_kw=dict(width_ratios=[1, 0.86, 1]))
    fig.subplots_adjust(left=0.078, right=0.988, bottom=0.225, top=0.855, wspace=0.62)
    pos = bx.get_position()                      # room for the row names of the forest plot
    bx.set_position([pos.x0 + 0.045, pos.y0, pos.width - 0.045 + 0.030, pos.height])
    pos_c = cx.get_position()
    cx.set_position([pos_c.x0 + 0.020, pos_c.y0, pos_c.width - 0.020, pos_c.height])
    heading(fig, ax, "(a)", "Magnitude of the winning amplitude")
    heading(fig, bx, "(b)", "Pooled over pattern sets")
    heading(fig, cx, "(c)", "Single pattern sets")

    # (a) densities of |Z| and of the largest of three |Z_mu|, with their means.
    for p, printed in A_PRINTED.items():
        a_p = amplification(p)[0]
        assert same(a_p, printed), (p, a_p)
        assert abs(a_p - float(amplification_repo(p))) < 1e-9, p
        print(f"competition a: A_{p} = {a_p:.6f}")
    assert abs(amplification(2)[0] - np.sqrt(2)) < 1e-10
    a3, mean_one, mean_max, dens3 = amplification(3)
    assert abs(a3 - predictions[(2048, 3)]["A_p"]) < 1e-9
    assert same(mean_one, "0.798") and same(mean_max, "1.326")
    x = np.linspace(0, 3.4, 500)
    one = 2 * norm.pdf(x)
    ax.fill_between(x, one, color=INK, alpha=0.10, lw=0)
    ax.fill_between(x, dens3(x), color=AP, alpha=0.16, lw=0)
    ax.plot(x, one, color=INK, lw=LW)
    ax.plot(x, dens3(x), color=AP, lw=LW)
    ax.plot([mean_one] * 2, [0, 2 * norm.pdf(mean_one)], color=INK, lw=0.9, ls=DOTS)
    ax.plot([mean_max] * 2, [0, dens3(mean_max)], color=AP, lw=0.9, ls=DOTS)
    ax.text(0.12, 0.905, "one pattern", color=INK, fontsize=NOTE, ha="left", va="center")
    ax.text(1.95, 0.50, "largest\nof three", color=AP, fontsize=NOTE, ha="left", va="center",
            linespacing=1.05)
    ax.set(xlim=(0, 3.4), ylim=(0, 1.0), xticks=[0, 1, 2, 3], yticks=[0, 0.4, 0.8],
           xlabel="Amplitude magnitude (standard deviations)", ylabel="Probability density")
    print(f"competition a: E|Z| = {mean_one:.4f}, E max_3|Z| = {mean_max:.4f}, ratio = {a3:.4f}")

    # (b) forest plot of the eight rows of Table 3; 95% intervals resample the 50 pattern sets.
    order = ["N1024_p1", "N2048_p1", "N4096_p1", "N2048_p2",
             "N1024_p3", "N2048_p3", "N4096_p3", "N2048_p4"]
    assert sorted(order) == sorted(analysis) == sorted(TABLE_RESULTS)
    ys = [7.6, 6.6, 5.6, 4.35, 2.75, 1.75, 0.75, -0.5]
    bx.axvline(1, color=REF, ls=DASH, lw=0.7, zorder=1)
    for key, y in zip(order, ys):
        row = analysis[key]
        n, p = int(key.split("_p")[0][1:]), int(key.split("_p")[1])
        # The protocol's estimator at 95% for every row.
        est = estimate(counts_for_estimator(read(root, f"results/simple/row_{key}.json")),
                       predictions[(n, p)]["L_eq"], level=0.95)
        assert abs(est["c"] - row["estimate"]["c"]) < 1e-15, key
        assert abs(row["c_pred"] - predictions[(n, p)]["c_pred"]) < 1e-15, key
        r, r_lo, r_hi = (est[k] / row["c_pred"] for k in ("c", "lo", "hi"))
        printed = TABLE_RESULTS[key]
        assert same(row["c_pred"], printed[0]) and same(est["c"], printed[1]), key
        assert same(est["lo"], printed[2]) and same(est["hi"], printed[3]) and same(r, printed[4]), key
        mark = MARKS[p]
        bx.errorbar(r, y, xerr=[[r - r_lo], [r_hi - r]], fmt="none", ecolor=mark["color"],
                    elinewidth=0.9, capsize=2.0, capthick=0.9, zorder=2)
        bx.plot(r, y, lw=0, marker=mark["marker"], color=mark["color"], mfc=mark["mfc"],
                mec=mark["color"], ms=mark["ms"], mew=1.0, zorder=4)
        print(f"competition b: {key}: predicted {row['c_pred']:.4f}  measured {est['c']:.4f} "
              f"[{est['lo']:.4f}, {est['hi']:.4f}]  ratio {r:.3f} [{r_lo:.3f}, {r_hi:.3f}]")
    bx.set_yticks(ys, labels=[k.split("_p")[0][1:] for k in order])
    for label, y in [("$p=1$", 6.6), ("$p=2$", 4.35), ("$p=3$", 1.75), ("$p=4$", -0.5)]:
        bx.annotate(label, xy=(0, y), xycoords=("axes fraction", "data"), xytext=(-27, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=LABEL)
    bx.annotate("$N$", xy=(0, 1), xycoords="axes fraction", xytext=(-3, -3), textcoords="offset points",
                ha="right", va="bottom", fontsize=LABEL)
    bx.text(0.866, 6.6, "exact", color=INK, fontsize=NOTE, ha="left", va="center")
    bx.text(0.866, 1.75, r"$A_p\,c_{\rm CW}$", color=AP, fontsize=NOTE, ha="left", va="center")
    bx.set(xlim=(0.862, 1.052), ylim=(-1.05, 8.6), xlabel="Measured / predicted $c$")
    bx.set_xticks([0.9, 1.0], labels=["0.9", "1.0"])

    # (c) per pattern set: c_s over the prediction against p times pattern 1's win share
    # (src/posthoc.py); under the assumptions of A_p every set lies at (1, 1).
    cx.axhline(1, color=REF, ls=DASH, lw=0.7, zorder=1)
    cx.axvline(1, color=REF, ls=DASH, lw=0.7, zorder=1)
    main = "N2048_p4"
    for key in ["N1024_p3", "N2048_p3", "N4096_p3", "N2048_p4"]:
        n, p = int(key.split("_p")[0][1:]), int(key.split("_p")[1])
        z, pred = posthoc[key], predictions[(n, p)]
        share = np.array(z["win_share"]["per_set"])
        ratio = np.array(z["c_per_set"]) / pred["c_pred"]
        printed = TABLE4[key]
        assert z["sets"] == share.size == ratio.size == 50
        assert same(share.min(), printed[0]) and same(share.max(), printed[1]), key
        assert same(z["spearman_c_vs_win_share"], printed[2]), key
        assert abs(ratio.min() - z["ratio_per_set"]["min"]) < 1e-12
        assert abs(ratio.max() - z["ratio_per_set"]["max"]) < 1e-12
        # Recompute both axes from the saved run counts with the definitions of src/posthoc.py.
        counts = read(root, f"results/simple/row_{key}.json")
        runs_p1 = np.asarray(counts["counts"]["pattern1"], float).sum(axis=1)
        right = np.asarray(counts["counts"]["correct"], float).sum(axis=1)
        runs = np.asarray(counts["counts"]["runs"], float).sum(axis=1)
        wins = np.asarray(counts["winners"], float)[..., 0].sum(axis=1)
        q_set = right / runs_p1
        assert np.allclose(wins / runs, share, atol=1e-12)
        assert np.allclose(logit(q_set) / pred["L_eq"] / pred["c_pred"], ratio, atol=1e-12)
        assert abs(spearmanr(p * share, ratio)[0] - z["spearman_c_vs_win_share"]) < 1e-12
        # Binomial 95% half-width of a set's ratio: 1.96 / sqrt(n q (1 - q)) on the logit.
        half = 1.96 / np.sqrt(runs_p1 * q_set * (1 - q_set)) / (pred["L_eq"] * pred["c_pred"])
        print(f"competition c: {key}: win share {share.min():.3f}-{share.max():.3f} "
              f"(x p: {p * share.min():.2f}-{p * share.max():.2f}), c_s / c_pred "
              f"{ratio.min():.3f}-{ratio.max():.3f}, Spearman {z['spearman_c_vs_win_share']:.2f}, "
              f"median binomial 95% half-width {np.median(half):.3f}")
        if key == main:
            cx.plot(p * share, ratio, lw=0, marker="^", color=AP, mfc=AP, mec="white", ms=4.4,
                    mew=0.4, alpha=0.9, zorder=5)
            # One representative interval: the median half-width over the 50 sets of this row.
            cx.errorbar(2.42, 1.30, yerr=np.median(half), fmt="none", ecolor=AP, elinewidth=0.9,
                        capsize=2.0, capthick=0.9, zorder=3)
        else:
            cx.plot(p * share, ratio, lw=0, marker="s", mfc=tint(AP, 0.42), mec="none", ms=3.0,
                    alpha=0.75, zorder=2)
    cx.plot(2.42, 1.55, lw=0, marker="^", color=AP, mfc=AP, mec="white", ms=4.4, mew=0.4)
    cx.plot(2.42, 1.71, lw=0, marker="s", mfc=tint(AP, 0.42), mec="none", ms=3.0)
    cx.text(2.55, 1.71, "$p=3$", color=tint(AP, 0.25), fontsize=NOTE, ha="left", va="center")
    cx.text(2.55, 1.55, "$p=4$", color=AP, fontsize=NOTE, ha="left", va="center")
    cx.text(2.55, 1.30, "95%", color=AP, fontsize=NOTE, ha="left", va="center")
    cx.set(xlim=(0.0, 3.1), ylim=(0.45, 1.8), xticks=[0, 1, 2, 3], yticks=[0.5, 1.0, 1.5],
           xlabel=r"$p\,\times$ pattern 1's win share", ylabel=r"Set's $c$ / predicted $c$")
    for axis in (ax, bx, cx):
        style_axis(axis)
    bx.spines["left"].set_visible(False)
    bx.tick_params(axis="y", length=0, pad=3)
    save(fig, out, "competition")


def make_figures(root, out):
    configure()
    plt.rcParams.update({"axes.labelsize": LABEL, "xtick.labelsize": LABEL, "ytick.labelsize": LABEL,
                         "legend.fontsize": NOTE})
    out.mkdir(parents=True, exist_ok=True)
    teaser(root, out)
    competition(root, out)
    assert not PROBLEMS, PROBLEMS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    make_figures(args.root, args.out)
