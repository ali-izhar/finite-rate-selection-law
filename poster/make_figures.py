"""Poster figures as hand-written vector SVG at their printed size.

1 SVG user unit = 1 pt (1/72 in). Text stays text; the poster's CSS supplies the fonts (Inter,
Source Serif 4, KaTeX_Math), so build.mjs inlines the SVGs into the page.
Palette: visual/lib/theme.css (the hero uses the dark tokens, the panels the light tokens).
The figures redraw the visual edition's and the film's motifs from the paper's data.

  VH  hero (dark band, 22.5 x 6.5 in): the fork (paper Fig. 1a paths) -> samples grid 67 | model grid 97
      -> one log-odds bar, c ~ 1/5 (Hopfield, Table 4, strongest field)
  Panels: one frame, 6.3 x 4.35 in, for all six (the climax is marked only by its gold card border):
  V1  the race: U ~ N(a, 1), a = 0.208, 58% favored (Fig. 1b)
  V2  slower cooling: exact c against N^2 gamma on linear ramps, with the law (Table 2, Eq. 4)
  V3  reheating: three landscapes (sketch) and c for x1/4, x1, x4 final cooling (Table 3 caption)
  V4  competition: |Z| for one pattern vs the winner of three; A_3 = 1.662 (Eq. 6)
  V5  the test: sampled vs model log-odds over the eight fields (Fig. 2b, Table 4)
  V6  Table 1 as an interval plot

Data: ../visual/data/*.json (exports of results/). Printed numbers are asserted against paper/main.tex.

Run:  python poster/make_figures.py   (or ./poster/build.sh)
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "visual" / "data"
PAPER = (HERE.parent / "paper" / "main.tex").read_text()
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
for old in OUT.glob("V*.svg"):
    old.unlink()
IN = 72.0

# ---------------------------------------------------------------- theme.css light tokens
INK, INK2, INK3 = "#141518", "#3B3D43", "#6A6C72"
RULE, RULE_S = "#DAD8D1", "#A4A39D"
CARD, BG = "#FCFBF8", "#F7F6F2"
CLIMAX = "#FBF6E9"
A, B = "#2350B4", "#C8561E"   # one orange for the other mode, light and dark
A_WASH, B_WASH = "rgba(35,80,180,0.16)", "rgba(200,86,30,0.15)"
NOISE = "#8B8D93"
NOISE_WASH = "rgba(139,141,147,0.18)"
GILT, GILT_INK = "#B08A2E", "#7A5C0E"
GILT_WASH = "rgba(176,138,46,0.20)"
# ---------------------------------------------------------------- theme.css dark tokens (hero band)
D_BG, D_INK, D_INK2, D_INK3 = "#111214", "#E8E7E2", "#BDBCB6", "#8C8B86"
D_RULE, D_RULE_S = "#26272A", "#4C4D50"
D_A, D_B, D_NOISE, D_GILT = "#7EA3F0", B, "#6F7177", "#D9B350"

# type scale (pt; 1 SVG unit = 1 pt): figure labels and numbers FL; axis symbols and display labels FA (the
# poster's key-line size); display numerals FD (the panel-headline size); sub/superscripts FS
FL, FA, FD, FS = 22, 24, 34, 15

SANS = "Inter, sans-serif"
SERIF = "'Source Serif 4', serif"
MATH = "KaTeX_Math, 'Source Serif 4', serif"
MINUS = "−"

FW, FH = 6.3, 4.35          # panel figure frame (in)
HW, HH = 22.5, 6.0         # hero figure (in)


def need(*snips):
    """Every printed number must appear in the paper source."""
    for s in snips:
        assert s in PAPER, f"not in paper/main.tex: {s!r}"


def load(name):
    return json.loads((DATA / name).read_text())


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def T(x, y, s, size, fill=INK2, family=SANS, weight=400, anchor="start", raw=False, extra="", style=""):
    body = s if raw else esc(s)
    plain = re.sub(r"<[^>]+>", "", s)
    nums = "lining-nums proportional-nums" if re.search(r"[^\W\d_]", plain) else "lining-nums tabular-nums"
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" '
            f'style="font-variant-numeric:{nums};{style}" {extra}>{body}</text>')


def halo(x, y, s, size, fill=INK2, family=SANS, weight=400, anchor="start", raw=False, bg=CARD, w=8):
    return (T(x, y, s, size, bg, family, weight, anchor, raw,
              extra=f'stroke="{bg}" stroke-width="{w}" stroke-linejoin="round" class="halo" aria-hidden="true"')
            + T(x, y, s, size, fill, family, weight, anchor, raw))


def mi(s, size=None, fill=None):
    sz = f' font-size="{size}"' if size else ""
    fl = f' fill="{fill}"' if fill else ""
    return f'<tspan font-family="{MATH}" font-style="italic"{sz}{fl}>{esc(s)}</tspan>'


def sub(s, size):
    return f'<tspan font-size="{size}" dy="{size * 0.32:.1f}">{esc(s)}</tspan>'


def L(x1, y1, x2, y2, stroke, w=1.5, dash=None, extra=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" '
            f'stroke-width="{w}"{d} stroke-linecap="butt" {extra}/>')


def P(pts, stroke, w=2.5, dash=None, fill="none", extra=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    p = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
    return (f'<polyline points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{w}" '
            f'stroke-linejoin="round" stroke-linecap="round"{d} {extra}/>')


def poly(pts, fill, extra=""):
    p = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
    return f'<polygon points="{p}" fill="{fill}" {extra}/>'


def rect(x, y, w, h, fill, extra=""):
    return f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" {extra}/>'


def dot(x, y, r, fill, stroke=CARD, sw=2.5):
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def head(x, y, dx, dy, size, fill):
    n = math.hypot(dx, dy)
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    bx, by = x - ux * size, y - uy * size
    w = size * 0.5
    return poly([(x, y), (bx + px * w, by + py * w), (bx - px * w, by - py * w)], fill)


def arrow(x0, y0, x1, y1, col, w=3, hs=14):
    n = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / n, (y1 - y0) / n
    return L(x0, y0, x1 - ux * hs * 0.8, y1 - uy * hs * 0.8, col, w) + head(x1, y1, ux, uy, hs, col)


def svg(name, w_in, h_in, body, label, defs=""):
    w, h = w_in * IN, h_in * IN
    s = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.2f}pt" height="{h:.2f}pt" '
         f'viewBox="0 0 {w:.2f} {h:.2f}" role="img" aria-label="{esc(label)}" class="fig" id="fig-{name}">\n'
         + (f"<defs>{defs}</defs>\n" if defs else "") + "\n".join(body) + "\n</svg>\n")
    (OUT / f"{name}.svg").write_text(s)
    print("wrote", name, f"{w_in:.2f} x {h_in:.2f} in")


def logit(q):
    return math.log(q / (1 - q))


def grid10(x, y, size, n_a, col_a, col_b, gap=3.2):
    """10 x 10 proportion grid, favored cells first (row-major from the top)."""
    cs = size / 10
    out = []
    for k in range(100):
        r, c = divmod(k, 10)
        out.append(rect(x + c * cs, y + r * cs, cs - gap, cs - gap, col_a if k < n_a else col_b, 'rx="1.5"'))
    return out


# ======================================================================== VH: hero (dark band)
def vh():
    W, H = HW * IN, HH * IN
    fk = load("fork_trajectories.json")
    assert fk["N"] == 1024 and fk["paper"] == "Fig. 1a"
    need("Fourteen paths ($N=1024$, $G=0$)")
    betas = np.array(fk["beta"])
    M = np.array(fk["m"])
    fg = load("field_grid_points.json")["points"][0]
    need(".6691 & .9726", "keep about a fifth", "$ 6.125$ & 11085 & 7417 & .6691")
    q_s, q_m = fg["q_chain"]["point"], fg["q_eq"]["point"]
    assert round(q_s * 100) == 67 and round(q_m * 100) == 97
    c_hero = logit(q_s) / logit(q_m)
    assert abs(c_hero - 0.2) < 0.01

    # --- fork plot (left 13 in)
    x0, x1 = 78, 13.0 * IN
    yt, yb = 40, H - 74
    bl, br = 0.7, 1.35
    X = lambda b: x0 + (b - bl) / (br - bl) * (x1 - x0)
    Y = lambda m: yt + (1 - (m + 1) / 2) * (yb - yt)
    defs = ""
    for nm, col in (("gA", D_A), ("gB", D_B)):
        defs += (f'<linearGradient id="{nm}" gradientUnits="userSpaceOnUse" x1="{X(0.975):.1f}" y1="0" '
                 f'x2="{X(1.07):.1f}" y2="0"><stop offset="0" stop-color="#7D7E83"/>'
                 f'<stop offset="1" stop-color="{col}"/></linearGradient>')
    o = []
    wl, wr = X(0.985), X(1.10)
    o.append(rect(wl, yt - 8, wr - wl, yb - yt + 8, "rgba(232,231,226,0.075)"))
    o.append(T((wl + wr) / 2, yt - 16, f'Runs commit near {mi("β", fill=D_INK)} = 1, then rarely switch.', FL, D_INK, SANS, 600,
               "middle", raw=True))
    o.append(L(x0, yb + 12, x1, yb + 12, D_RULE_S, 2))
    for b, lab in ((0.7, "0.7  hot"), (1.0, ""), (1.35, "cold  1.35")):
        o.append(L(X(b), yb + 12, X(b), yb + 22, D_RULE_S, 2))
    o.append(T(X(0.7), yb + 52, "0.7", FL, D_INK2, SANS, 400, "middle"))
    o.append(T(X(1.35), yb + 52, "1.35", FL, D_INK2, SANS, 400, "middle"))
    o.append(T(X(1.0), yb + 52, f'{mi("β", fill=D_INK)} = 1', FL, D_INK, SANS, 600, "middle", raw=True))
    o.append(T(X(1.17), yb + 52, f'cooling: {mi("β")} rises →', FL, D_INK3, SANS, 400, "middle", raw=True))
    o.append(T(X(0.85), yb + 52, f'← hot', FL, D_INK3, SANS, 400, "middle", raw=True))
    o.append(L(x0 - 12, yt, x0 - 12, yb, D_RULE_S, 2))
    for m, lab in ((1, "+1"), (0, "0"), (-1, MINUS + "1")):
        o.append(L(x0 - 22, Y(m), x0 - 12, Y(m), D_RULE_S, 2))
        o.append(T(x0 - 30, Y(m) + 9, lab, FL, D_INK2, SANS, 400, "end"))
    o.append(T(x0 - 50, Y(0.5) + 10, "m", FA, D_INK2, MATH, 400, "middle", style="font-style:italic"))
    o.append(L(x0, Y(0), x1, Y(0), D_RULE, 1.5))
    step = 2
    for row in M:
        col = "url(#gA)" if row[-1] > 0 else "url(#gB)"
        pts = [(X(b), Y(m)) for b, m in zip(betas[::step], row[::step])]
        o.append(P(pts, col, 2.8, extra='stroke-opacity="0.92"'))
    o.append(T(X(0.745), Y(0.36), "undecided", FA, D_INK2, SERIF, 400, "start", style="font-style:italic"))
    o.append(T(X(0.715), Y(0.86), f'{mi("m", fill=D_INK2)}: magnetization, the mean spin', FL, D_INK3, SANS, 600, "start", raw=True))
    o.append(T(X(1.345), Y(0.16), "positive mode", FL, D_A, SANS, 600, "end"))
    o.append(T(X(1.345), Y(-0.16) + 20, "negative mode", FL, D_B, SANS, 600, "end"))
    o.append(T(X(0.715), Y(-0.72), 'Curie–Weiss · commitment (no field yet)', FL, D_INK3, SANS, 600, "start"))

    # --- divider: two separate examples
    xs = 14.3 * IN
    o.append(L((x1 + xs) / 2, 0, (x1 + xs) / 2, H - 8, D_RULE_S, 2))

    # --- right: Hopfield example. Favored share (grids), then log-odds on one scale.
    o.append(T(xs, 26, f'Same effect, Hopfield: {mi("p")} = 3, {mi("N")} = 2,048', FL, D_INK3, SANS, 600, raw=True))
    gs = 2.0 * IN
    gy = 80
    xm = W - gs + 3.2
    for gx, n, name in ((xs, 67, "samples"), (xm, 97, "model")):
        o.append(T(gx, gy - 16, name, FD, D_INK, SERIF, 500))
        o += grid10(gx, gy, gs, n, D_A, D_B)
        o.append(T(gx if n == 67 else W, gy + gs + 56,
                   f'{n}<tspan font-size="{FA}" fill="{D_INK2}" dx="10">of 100 favored</tspan>',
                   60, D_INK, SERIF, 300, "start" if n == 67 else "end", raw=True))
    la_s, la_m = logit(q_s), logit(q_m)
    assert f"{la_s:.2f}" == "0.70" and f"{la_m:.2f}" == "3.57" and f"{c_hero:.3f}" == "0.197"
    o.append(T(xs, gy + gs + 96, "log-odds, favored vs other", FL, D_INK2, SANS, 600))
    bx0, bx1 = xs + 116, W - 4
    S = lambda v: bx0 + v / 3.6 * (bx1 - bx0)
    by, bh, bg = gy + gs + 108, 24, 32
    o.append(T(bx0 - 12, by + 20, "samples", FL, D_GILT, SANS, 700, "end"))
    o.append(rect(bx0, by, S(la_s) - bx0, bh, D_GILT))
    o.append(T(S(la_s) + 10, by + 21, "0.70", FL, D_GILT, SANS, 700))
    o.append(T(S(la_s) + 84, by + 26, f'→  {mi("c", fill=D_GILT)} ≈ 1/5', 34, D_GILT, SERIF, 600, raw=True))
    o.append(T(bx0 - 12, by + bg + 20, "model", FL, D_INK, SANS, 700, "end"))
    o.append(rect(bx0, by + bg, S(la_m) - bx0, bh, "#3A3B40", f'stroke="{D_INK3}" stroke-width="1.5"'))
    o.append(T(S(la_m) - 10, by + bg + 21, "3.57", FL, D_INK, SANS, 700, "end"))
    ya = by + bg + bh + 5
    o.append(L(bx0, ya, bx1, ya, D_RULE_S, 2))
    for v in (0, 1, 2, 3):
        o.append(L(S(v), ya, S(v), ya + 8, D_RULE_S, 2))
        o.append(T(S(v), ya + 30, str(v), FL, D_INK3, SANS, 400, "middle"))
    assert ya + 30 <= H
    svg("VH", HW, HH, o, "Fourteen cooling runs stay undecided until beta = 1, then each commits to the positive "
        "or negative mode. The samples' grid shows 67 of 100 in the favored mode, the model's grid 97 of 100: "
        "the samples keep about a fifth of the model's log-odds.", defs)


# ======================================================================== V1: the race
def v1():
    W, H = FW * IN, FH * IN
    G, N, g = 0.0006125, 2048, 0.001
    a = G * math.sqrt(N) * (math.pi / g) ** 0.25
    assert round(a, 3) == 0.208 and round(norm.cdf(a) * 100) == 58
    need("$a=0.208$", "$58\\%$ of runs", "about five times stronger")
    x0, x1 = 12, W - 12
    ul, ur = -3.3, 3.7
    base, top = 244, 72
    X = lambda u: x0 + (u - ul) / (ur - ul) * (x1 - x0)
    Y = lambda d: base - d / norm.pdf(0) * (base - top)
    us = np.linspace(ul, ur, 300)
    o = []
    left = [(X(u), Y(norm.pdf(u - a))) for u in us if u <= 0] + [(X(0), Y(norm.pdf(a)))]
    right = [(X(0), Y(norm.pdf(a)))] + [(X(u), Y(norm.pdf(u - a))) for u in us if u >= 0]
    o.append(poly([(X(ul), base)] + left + [(X(0), base)], B_WASH))
    o.append(poly([(X(0), base)] + right + [(X(ur), base)], A_WASH))
    o.append(P([(X(u), Y(norm.pdf(u - a))) for u in us], INK, 3))
    o.append(L(x0, base, x1, base, INK2, 2))
    o.append(L(X(0), base, X(0), Y(norm.pdf(a)) - 6, INK3, 2, "6 5"))
    o.append(T(X(0), base + 30, "0", FL, INK3, SANS, 400, "middle"))
    o.append(L(X(a), base, X(a), top - 10, A, 3.5))
    o.append(T(x1, top - 26, f'field’s push {mi("a")} = 0.208', FL, A, SANS, 600, "end", raw=True))
    yh = Y(norm.pdf(1))
    o.append(arrow(X(a), yh, X(a + 1), yh, INK, 2.6, 16))     # one noise unit: mean to +1 sigma
    o.append(T(X(a + 1) + 14, yh - 34, "noise ≈ 5×", FL, INK, SANS, 600, "start"))
    o.append(T(X(a + 1) + 14, yh - 4, "the field", FL, INK, SANS, 600, "start"))
    o.append(T(X(1.2), base - 16, "58%", FD, A, SERIF, 600, "middle"))
    o.append(T(x1, base + 30, "favored mode →", FL, A, SANS, 600, "end"))
    o.append(T(x0, base + 30, "← other mode", FL, B, SANS, 600, "start"))
    o.append(T(W / 2, H - 6, "each run’s early amplitude, in noise units", FL, INK3, SANS, 400, "middle"))
    svg("V1", FW, FH, o, "The early amplitude is Gaussian with mean a = 0.208 and spread 1: the noise is about five "
        "times the field, and 58% of runs end in the favored mode.")


# ======================================================================== V2: slower cooling
TABLE2 = {(1024, .004): (.09370, .93), (1024, .001): (.13393, 1.81), (1024, .00025): (.18868, 1.03),
          (2048, .004): (.06587, .46), (2048, .001): (.09372, .97), (2048, .00025): (.13314, 1.23),
          (4096, .004): (.04645, .23), (4096, .001): (.06589, .49), (4096, .00025): (.09361, .86)}


def v2():
    W, H = FW * IN, FH * IN
    need("0.06587", "0.13314", "0.04645", "0.18868", "\\logit\\qeq=3.575", "only doubles")
    x0, x1, yt, yb = 78, W - 10, 40, H - 102
    lx0, lx1 = math.log10(180), math.log10(1.0e5)
    ly0, ly1 = math.log10(0.040), math.log10(0.23)
    X = lambda v: x0 + (math.log10(v) - lx0) / (lx1 - lx0) * (x1 - x0)
    Y = lambda c: yb - (math.log10(c) - ly0) / (ly1 - ly0) * (yb - yt)
    o = []
    for v in (1e3, 1e4, 1e5):
        o.append(L(X(v), yt, X(v), yb, RULE, 1.2))
    for c in (0.05, 0.1, 0.2):
        o.append(L(x0, Y(c), x1, Y(c), RULE, 1.2))
        o.append(T(x0 - 12, Y(c) + 8, f"{c:g}", FL, INK3, SANS, 400, "end"))
    o.append(L(x0, yb, x1, yb, INK2, 2))
    o.append(L(x0, yt, x0, yb, INK2, 2))
    for v, e in ((1e3, 3), (1e4, 4), (1e5, 5)):
        o.append(T(X(v), yb + 32, f"10<tspan dy='-10' font-size='{FS}'>{e}</tspan>", FL, INK3, SANS, 400,
                   "middle" if e < 5 else "end", raw=True))
    o.append(T(x0 - 12, yt - 14, f'{mi("c", fill=GILT_INK)}', FA, GILT_INK, SERIF, 600, "end", raw=True))
    o.append(T(W / 2 + 20, H - 44, f'{mi("N")}<tspan dy="-10" font-size="{FS}">2</tspan><tspan dy="10">{mi("γ")}</tspan>'
               f':  size² × cooling rate', FL, INK3, SANS, 400, "middle", raw=True))
    NG = 0.0006125 * 2048
    law = lambda v: logit(norm.cdf(NG * math.pi ** 0.25 * v ** -0.25)) / 3.575
    for (N, g), (c, e) in TABLE2.items():
        assert abs(law(N * N * g) / (c / (1 + e / 100)) - 1) < 0.001
    vs = np.logspace(lx0, lx1, 120)
    o.append(P([(X(v), Y(law(v))) for v in vs], INK2, 2.6, "10 7"))
    # the arrow lies ON the law, from 0.066 (fast) to 0.133 (16x slower); the markers sit on top of it
    va, vb = 2048 ** 2 * .004, 2048 ** 2 * .00025
    arr = [(X(v), Y(law(v))) for v in np.logspace(math.log10(va), math.log10(vb), 40)]
    (xa0, ya0), (xb0, yb0) = arr[0], arr[-1]
    ux, uy = xb0 - xa0, yb0 - ya0
    n = math.hypot(ux, uy)
    ux, uy = ux / n, uy / n
    o.append(arrow(xa0 + ux * 14, ya0 + uy * 14, xb0 - ux * 13, yb0 - uy * 13, GILT_INK, 4, 20))
    for (N, g), (c, e) in sorted(TABLE2.items(), key=lambda kv: {1024: 0, 4096: 1, 2048: 2}[kv[0][0]]):
        x, y = X(N * N * g), Y(c)
        if N == 1024:
            o.append(dot(x, y, 11, "none", INK, 2.6))
        elif N == 2048:
            o.append(dot(x, y, 7.5, GILT, CARD, 1.5))
        else:
            o.append(rect(x - 10, y - 10, 20, 20, "none", f'stroke="{INK}" stroke-width="2.6"'))
    xa, ya = X(va), Y(.06587)
    xb, yb2 = X(vb), Y(.13314)
    # two short lines in the empty upper-right corner of the plot; the drawn arrow alone gives the direction
    rx = X(1.0e5)
    o.append(T(rx, Y(0.19), '16× slower cooling:', FL, GILT_INK, SANS, 700, "end"))
    o.append(T(rx, Y(0.19) + 32, f'{mi("c", fill=GILT_INK)} only ×2', FL, GILT_INK, SANS, 700, "end", raw=True))
    o.append(T(xa + 16, ya - 14, "0.066", FL, GILT_INK, SANS, 700, "start"))
    o.append(T(xb - 14, yb2 + 40, "0.133", FL, GILT_INK, SANS, 700, "end"))
    ly = H - 14
    o.append(T(0, ly + 8, f'{mi("N")}:', FL, INK3, SANS, 400, raw=True))
    o.append(dot(52, ly, 9, "none", INK, 2.6))
    o.append(T(66, ly + 8, "1,024", FL, INK2, SANS, 400))
    o.append(dot(150, ly, 9, GILT, CARD, 2.5))
    o.append(T(165, ly + 8, "2,048", FL, INK2, SANS, 400))
    o.append(rect(241.5, ly - 7.5, 15, 15, CARD, f'stroke="{INK}" stroke-width="2.6"'))
    o.append(T(265, ly + 8, "4,096", FL, INK2, SANS, 400))
    o.append(L(352, ly, 392, ly, INK2, 2.6, "10 7"))
    o.append(T(400, ly + 8, "law", FL, INK2, SANS, 400))
    svg("V2", FW, FH, o, "Exact-chain c on linear ramps against N squared times gamma, log-log, on the selection law's "
        "line: at N = 2,048, 16 times slower cooling raises c from 0.066 to 0.133.")


# ======================================================================== V3: reheating
def fe(m, beta, h):
    """Mean-field free energy per spin of Curie-Weiss (sketch)."""
    s = ((1 + m) / 2) * np.log((1 + m) / 2) + ((1 - m) / 2) * np.log((1 - m) / 2)
    return -beta * m * m / 2 - beta * h * m + s


def v3():
    W, H = FW * IN, FH * IN
    need("$c=0.098978$ and $0.146625$", "$1.48$-fold", "$c_{\\mathrm{CW}}=0.120$",
         "reheat} of $30$ sweeps at $\\beta=0.7$")
    o = []
    ms = np.linspace(-0.97, 0.97, 200)
    pw, gapx, py0, ph = 128, (W - 3 * 128) / 2, 40, 72
    xs = [0, pw + gapx, 2 * (pw + gapx)]
    titles = [("1", "cool"), ("2", "reheat"), ("3", "cool again")]
    sc = ph / (fe(np.array([0.0]), 1.6, 0.05)[0] - fe(ms, 1.6, 0.05).min() + 0.10)
    for k, (xL, (n, t), beta) in enumerate(zip(xs, titles, (1.6, 0.7, 1.6))):
        f = fe(ms, beta, 0.05)
        f = f - f.min()
        pts = [(xL + (m + 1) / 2 * pw, py0 + ph - min(v * sc, ph + 6)) for m, v in zip(ms, f)]
        o.append(P(pts, INK, 3))
        tx, anc = (xL + pw / 2, "middle") if k < 2 else (W, "end")
        o.append(T(tx, 24, f'<tspan font-weight="700">{n}</tspan>  {t}', FL, INK, SANS, 500, anc, raw=True))
        if k < 2:   # the unfavored runs go on to the reheat (orange), then to the final cooling
            o.append(arrow(xL + pw + 8, py0 + ph / 2, xL + pw + gapx - 8, py0 + ph / 2, B if k == 0 else INK3,
                           3 if k == 0 else 2.2, 13 if k == 0 else 11))

        def at(m):
            return pts[int(np.argmin(np.abs(ms - m)))]
        if k == 0:
            xb, yb = at(-0.90)
            o.append(dot(xb + 2, yb - 13, 11, B, CARD, 2.5))          # unfavored: goes on
            xa, ya = at(0.92)
            o.append(dot(xa - 2, ya - 13, 11, CARD, A, 3))            # favored: set aside (hollow)
        else:
            xb, yb = at(0.0)
            o.append(dot(xb, yb - 13, 11, NOISE, CARD, 2.5))
            if k == 1:
                o.append(T(xb, py0 + 14, f'{mi("β")} = 0.7', FL, INK3, SANS, 500, "middle", raw=True))
            if k == 2:
                o.append(arrow(xb - 15, yb - 2, xb - 44, yb + 30, B, 2.6, 12))
                o.append(arrow(xb + 15, yb - 2, xb + 44, yb + 34, A, 2.6, 12))
    # one direct label, under the hollow (favored) run of the first landscape
    o.append(T(0, py0 + ph + 30, "favored runs set aside", FL, A, SANS, 600, "start"))
    yh = 206
    bx = 52
    bl = W - bx - 150
    cmax = 0.2
    rs, bh = 30, 25                       # the bars fill the frame down to its bottom
    o.append(T(0, yh, f'{mi("c", fill=GILT_INK)} by length of the final cooling', FL, INK2, SANS, 600, raw=True))
    rows = (("×¼", 0.098978, "0.099"), ("×1", 0.119808, "0.120"), ("×4", 0.146625, "0.147"))
    for i, (lab, v, sv) in enumerate(rows):
        y = yh + 14 + i * rs
        emph = lab == "×1"
        o.append(T(bx - 12, y + 20, lab, FL, INK if emph else INK2, SANS, 700 if emph else 500, "end"))
        o.append(rect(bx, y, v / cmax * bl, bh, GILT if emph else "rgba(176,138,46,0.42)", 'rx="2"'))
        o.append(T(bx + v / cmax * bl + 10, y + 20, sv, FL, GILT_INK if emph else INK2, SANS, 700 if emph else 500))
    o.append(L(bx, yh + 8, bx, yh + 14 + 2 * rs + bh + 4, INK2, 2))
    o.append(T(W, yh + 14 + rs + 6, "16× longer:", FL, GILT_INK, SANS, 700, "end"))
    o.append(T(W, yh + 14 + 2 * rs + 6, "only ×1.48", FL, GILT_INK, SANS, 700, "end"))
    assert yh + 14 + 2 * rs + bh + 4 <= H - 2
    svg("V3", FW, FH, o, "Schematic of the protocol: favored runs are set aside; the rest are reheated to beta = 0.7, where "
        "one well remains, then cooled again and choose again. Exact c is 0.099, 0.120 and 0.147 for a quarter, one "
        "and four times the final cooling.")


# ======================================================================== V4: competition
def v4():
    W, H = FW * IN, FH * IN
    need("$A_3=1.662$", "\\mathbb E\\max_{1\\le\\mu\\le p}|Z_\\mu|}{\\mathbb E|Z|}")
    f1 = lambda z: 2 * norm.pdf(z)
    f3 = lambda z: 3 * (2 * norm.cdf(z) - 1) ** 2 * 2 * norm.pdf(z)
    m1 = quad(lambda z: z * f1(z), 0, np.inf)[0]
    m3 = quad(lambda z: z * f3(z), 0, np.inf)[0]
    assert round(m3 / m1, 3) == 1.662
    x0, x1, base, top = 12, W - 12, 244, 92
    zl, zr = 0, 3.6
    X = lambda z: x0 + (z - zl) / (zr - zl) * (x1 - x0)
    peak = max(f1(0), max(f3(z) for z in np.linspace(0, 3, 300)))
    Y = lambda d: base - d / peak * (base - top)
    zs = np.linspace(zl, zr, 260)
    o = []
    o.append(poly([(X(0), base)] + [(X(z), Y(f1(z))) for z in zs] + [(X(zr), base)], NOISE_WASH))
    o.append(P([(X(z), Y(f1(z))) for z in zs], NOISE, 3))
    o.append(poly([(X(0), base)] + [(X(z), Y(f3(z))) for z in zs] + [(X(zr), base)], A_WASH))
    o.append(P([(X(z), Y(f3(z))) for z in zs], A, 3.4))
    o.append(L(x0, base, x1, base, INK2, 2))
    for z in (0, 1, 2, 3):
        o.append(T(X(z), base + 30, str(z), FL, INK3, SANS, 400, "middle"))
    ya = 46
    o.append(L(X(m1), base, X(m1), ya - 14, INK2, 2.6, "7 6"))
    o.append(L(X(m3), base, X(m3), ya - 14, A, 3.2))
    o.append(arrow(X(m1) + 4, ya, X(m3) - 4, ya, INK, 2.6, 14))
    o.append(T(X(m1) - 12, ya + 9, "averages", FL, INK3, SANS, 500, "end"))
    o.append(T(X(m3) + 14, ya + 9, f'× {mi("A")}{sub("3", FS)}<tspan dy="-5.4"> = 1.662</tspan>', FL, INK, SANS, 700,
               "start", raw=True))
    lx = X(1.9)
    o.append(rect(lx, 100, 30, 18, NOISE_WASH, f'stroke="{NOISE}" stroke-width="2.4"'))
    o.append(T(lx + 42, 117, "one pattern", FL, INK2, SANS, 600))
    o.append(rect(lx, 134, 30, 18, A_WASH, f'stroke="{A}" stroke-width="2.4"'))
    o.append(T(lx + 42, 151, "winner of three", FL, A, SANS, 700))
    o.append(T(W / 2, H - 6, "size of the early overlap, in noise units", FL, INK3, SANS, 400, "middle"))
    svg("V4", FW, FH, o, "Distribution of the size of one Gaussian overlap and of the largest of three: the winner's "
        "average size is 1.662 times larger.")


# ======================================================================== V5: the test
LBL_X = 1.7


def v5():
    W, H = FW * IN, FH * IN
    need("$[0.1908,0.2095]$", "c_{\\mathrm{pred}}=A_3c_{\\mathrm{CW}}=1.662\\times0.120=0.199", "eight nonzero fields")
    pts = load("field_grid_points.json")["points"]
    assert len(pts) == 8
    x0, x1, yt, yb = 112, W - 6, 8, H - 40
    xl, xr, yl, yr = -3.9, 3.9, -0.95, 0.95
    X = lambda v: x0 + (v - xl) / (xr - xl) * (x1 - x0)
    Y = lambda v: yb - (v - yl) / (yr - yl) * (yb - yt)
    o = [f'<clipPath id="c5"><rect x="{x0}" y="{yt}" width="{x1 - x0}" height="{yb - yt}"/></clipPath>']
    for v in (-3, 0, 3):
        o.append(L(X(v), yt, X(v), yb, RULE, 1.2))
        if v:
            o.append(T(X(v), yb + 30, (MINUS + "3") if v < 0 else str(v), FL, INK3, SANS, 400, "middle"))
    o.append(T(X(0), yb + 30, "model’s log-odds", FL, INK3, SANS, 500, "middle"))
    for v in (-0.5, 0, 0.5):
        o.append(L(x0, Y(v), x1, Y(v), RULE, 1.2))
        o.append(T(x0 - 10, Y(v) + 8, (MINUS + "0.5") if v < 0 else f"{v:g}", FL, INK3, SANS, 400, "end"))
    o.append(L(x0, yb, x1, yb, INK2, 2))          # open axes: left and bottom spines only, as in V2 and V6
    o.append(L(x0, yt, x0, yb, INK2, 2))
    o.append(f'<g clip-path="url(#c5)">' + L(X(-1), Y(-1), X(1), Y(1), INK3, 2.4, "3 6") + "</g>")
    fg = load("field_grid_points.json")
    lo, hi, cm, cp = 0.1908, 0.2095, fg["c_measured"]["point"], 1.662 * 0.120
    assert f"{cm:.3f}" == "0.199" and f"{cp:.3f}" == "0.199"
    assert abs(fg["c_measured"]["lower"] - lo) < 1e-4 and abs(fg["c_measured"]["upper"] - hi) < 1e-4
    o.append(poly([(X(xl), Y(lo * xl)), (X(xr), Y(lo * xr)), (X(xr), Y(hi * xr)), (X(xl), Y(hi * xl))],
                  "rgba(176,138,46,0.45)"))                 # measured 95% band
    for v in (lo, hi):
        o.append(L(X(xl), Y(v * xl), X(xr), Y(v * xr), GILT, 1.2))
    o.append(L(X(xl), Y(cm * xl), X(xr), Y(cm * xr), GILT_INK, 2.4))   # measured fit, solid
    for p in pts:
        x, y = p["logit_q_eq"], p["logit_q_chain"]
        ylo, yhi = logit(p["q_chain"]["lower"]), logit(p["q_chain"]["upper"])
        o.append(L(X(x), Y(ylo), X(x), Y(yhi), INK, 2.6))
        o.append(dot(X(x), Y(y), 7, INK, CLIMAX, 2.5))
    o.append(L(X(xl), Y(cp * xl), X(xr), Y(cp * xr), INK, 1.6, "6 6"))   # predicted: thin, dashed, on top
    o.append(T(X(0.50) - 12, Y(0.70), "slope 1", FL, INK3, SANS, 600, "end"))
    o.append(T(X(0.50) - 12, Y(0.70) + 28, "= the model", FL, INK3, SANS, 600, "end"))
    # key in the empty lower-right corner: line samples, no box
    kx, sw = X(-0.12), 36
    ky = [yb - 62, yb - 36, yb - 10]
    assert X(-0.40) < kx                     # right of the slope-1 line at the key's top
    o.append(L(kx, ky[0] - 8, kx + sw, ky[0] - 8, GILT_INK, 2.4))
    o.append(T(kx + sw + 10, ky[0], "measured", FL, GILT_INK, SANS, 700, "start"))
    o.append(rect(kx, ky[1] - 16, sw, 16, "rgba(176,138,46,0.45)", f'stroke="{GILT}" stroke-width="1.2"'))
    o.append(T(kx + sw + 10, ky[1], "95% band", FL, GILT_INK, SANS, 700, "start"))
    o.append(L(kx, ky[2] - 8, kx + sw, ky[2] - 8, INK, 1.6, "6 6"))
    o.append(T(kx + sw + 10, ky[2], "predicted", FL, INK, SANS, 600, "start"))
    o.append(T(24, (yt + yb) / 2, "samples’ log-odds", FL, INK3, SANS, 500, "middle",
               extra=f'transform="rotate(-90 24 {(yt + yb) / 2:.1f})"'))
    svg("V5", FW, FH, o, "Sampled against model log-odds at eight fields: the measured fit has slope 0.199 with 95% "
        "band 0.1908 to 0.2095; the predicted line of slope 0.199 lies inside it; the model itself has slope 1.")


# ======================================================================== V6: Table 1
FH6 = FH - 0.40      # the two-line headline of panel 6 takes 0.40 in from its frame; the frame bottom aligns


def v6():
    W, H = FW * IN, FH6 * IN
    t1 = load("table1.json")["rows"]
    sz = {r["N"]: r for r in load("size_scan.json")["rows"]}
    rows = [(r["p"], 2048, r["c_predicted"], r["c_measured"], r["ci95"]) for r in t1]
    rows += [(3, 1024, sz[1024]["c_predicted"], sz[1024]["c_measured"], sz[1024]["ci95"]),
             (3, 4096, sz[4096]["c_predicted"], sz[4096]["c_measured"], sz[4096]["ci95"])]
    # Table 1 of the paper, as printed
    printed = [("0.120", "0.127", "0.124", "0.131"), ("0.169", "0.188", "0.184", "0.191"),
               ("0.199", "0.201", "0.194", "0.210"), ("0.220", "0.211", "0.191", "0.239"),
               ("0.284", "0.263", "0.247", "0.283"), ("0.150", "0.160", "0.155", "0.165")]
    for (p, N, cp, cm, ci), (sp, sm, sl, sh) in zip(rows, printed):
        assert f"{cp:.3f}" == sp and f"{cm:.3f}" == sm and f"{ci[0]:.3f}" == sl and f"{ci[1]:.3f}" == sh, (p, N)
        need(f"{p} & {N} & {sp} & ${sm}\\ [{sl},{sh}]$")
    gap = rows[1][3] / rows[1][2] - 1
    assert f"{gap:.0%}" == "11%" and max(abs(r[3] / r[2] - 1) for r in rows) == abs(gap)
    cx_set, cx_pred, cx_meas, cx_ci = 0, 134, 248, W - 4      # setting (left) | predicted, measured (centred) | interval (right)
    rh = 28.5
    o = []
    yh = 24
    o.append(T(cx_set, yh, "setting", FL, INK3, SANS, 400, "start"))
    o.append(T(cx_pred, yh, "predicted", FL, INK3, SANS, 400, "middle"))
    o.append(T(cx_meas, yh, "measured", FL, INK3, SANS, 400, "middle"))
    o.append(T(cx_ci, yh, "95% interval", FL, INK3, SANS, 400, "end"))
    o.append(L(0, yh + 12, W, yh + 12, INK2, 2))
    y = yh + 12
    groups = ((f'{mi("N", fill=INK3)} = 2,048, {mi("p", fill=INK3)} =', rows[:4], printed[:4]),
              (f'{mi("p", fill=INK3)} = 3, {mi("N", fill=INK3)} =', rows[4:], printed[4:]))
    for gi, (glab, grows, gprint) in enumerate(groups):
        if gi:
            o.append(L(0, y + 6, W, y + 6, RULE_S, 1.2))
            y += 6
        y += rh
        o.append(T(cx_set, y - 7, glab, FL, INK3, SANS, 600, "start", raw=True))
        for (p, N, cp, cm, ci), (sp, sm, sl, sh) in zip(grows, gprint):
            hot = p == 2
            if hot:   # the largest miss
                o.append(rect(0, y + 2, W, rh, GILT_WASH, 'rx="3"'))
            y += rh
            lab = f"{p}" if gi == 0 else f"{N:,}"
            ink, wt = (INK, 700) if hot else (INK2, 400)
            yb = y - 7
            pn = "font-variant-numeric:lining-nums proportional-nums"
            o.append(T(cx_set + 24, yb, lab, FL, ink, SANS, wt, "start", style=pn))
            o.append(T(cx_pred, yb, sp, FL, ink, SANS, wt, "middle", style=pn))
            o.append(T(cx_meas, yb, sm, FL, GILT_INK if hot else INK, SANS, 700 if hot else 500, "middle", style=pn))
            o.append(T(cx_ci, yb, f"[{sl},{sh}]", FL, ink, SANS, wt, "end", style=pn))
    o.append(L(0, y + 8, W, y + 8, INK2, 2))
    assert y + 9 <= H, (y, H)
    svg("V6", FW, FH6, o, "Table 1: predicted c, measured c and its 95% interval for p = 1 to 4 at N = 2,048 and for "
        "N = 1,024 and 4,096 at p = 3. The largest miss, p = 2, is highlighted: 0.169 predicted, 0.188 measured, 11%.")


if __name__ == "__main__":
    vh(); v1(); v2(); v3(); v4(); v5(); v6()
