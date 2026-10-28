"""Poster figures as hand-written vector SVG at their printed size.

1 SVG user unit = 1 pt (1/72 in). Text stays text; the poster's CSS supplies the fonts (Inter,
Source Serif 4, KaTeX_Math), so build.mjs inlines the SVGs into the page.
Palette: visual/lib/theme.css (the hero uses the dark tokens, the panels the light tokens).

  VH  hero (dark band, 22.5 x 6.0 in): the fork (paper Fig. 1a paths) | samples grid 60 vs model grid 97
      and one log-odds bar, c = 0.109 (Hopfield, p = 3, N = 2,048, Table 1)
  Panels: one frame, 6.3 x 4.35 in (shorter under a two-line headline):
  V1  the race: U ~ N(a, 1), a = 0.208, 58% favored (Fig. 1b)
  V2  slower cooling: exact c against N^2 gamma on linear ramps, with the law (Table 2, Eq. 4)
  V3  one pattern: exact-chain c(N) on the test ramp and the three p = 1 Hopfield rows (Table 1)
  V4  competition: |Z| for one pattern vs the winner of three; A_3 = 1.662 (Eq. 6)
  V5  the test: A_3 c_CW(N) and the three p = 3 rows, on V3's axes (Table 1)
  V6  measured / predicted c for the eight rows of Table 1 (Fig. 2b)

Data: results/simple/{analysis,predictions}.json, results/experiments/cw_linear_sweep.json,
results/figures/fork_trajectories.json. Printed numbers are asserted against paper/main.tex.

Run:  python poster/make_figures.py   (or ./poster/build.sh)
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
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
    return json.loads((RESULTS / name).read_text())


ROWS = load("simple/analysis.json")["rows"]                       # Table 1, keyed N2048_p3 etc.
PRED = {(r["N"], r["p"]): r for r in load("simple/predictions.json")["rows"]}
C_CW = {N: PRED[(N, 1)]["c_CW"] for N in (1024, 2048, 4096)}     # exact CW fraction on the test ramp
L_EQ = {N: PRED[(N, 1)]["L_eq"] for N in (1024, 2048, 4096)}     # exact CW logit q_eq at beta = 1.6



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
    fk = load("figures/fork_trajectories.json")
    assert fk["N"] == 1024 and fk["paper"] == "Fig. 1a"
    need("Fourteen paths ($N=1024$, $G=0$)")
    betas = np.array(fk["beta"])
    M = np.array(fk["m"])
    # Hopfield, p = 3, N = 2,048 (Table 1): measured c = 0.1090 of the exact CW log-odds 3.5747 (App. B).
    # The two shares are derived from these printed numbers: q = logistic(c logit q_eq).
    need("3 & 2048 & $A_3c_{\\mathrm{CW}}$ & 0.1095 & $0.1090\\ [0.1059,0.1123]$", "$3.5747$")
    r3 = ROWS["N2048_p3"]["estimate"]
    c_hero, la_m = 0.1090, 3.5747
    assert f"{r3['c']:.4f}" == "0.1090" and f"{L_EQ[2048]:.4f}" == "3.5747"
    la_s = c_hero * la_m
    q_s, q_m = 1 / (1 + math.exp(-la_s)), 1 / (1 + math.exp(-la_m))
    assert abs(q_s - r3["q"]) < 0.001                     # the pooled share in analysis.json
    n_s, n_m = round(q_s * 100), round(q_m * 100)
    assert (n_s, n_m) == (60, 97)

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
    o.append(T(X(0.85), yb + 52, '← hot', FL, D_INK3, SANS, 400, "middle", raw=True))
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
    for gx, n, name in ((xs, n_s, "samples"), (xm, n_m, "model")):
        o.append(T(gx, gy - 16, name, FD, D_INK, SERIF, 500))
        o += grid10(gx, gy, gs, n, D_A, D_B)
        o.append(T(gx if n == n_s else W, gy + gs + 56,
                   f'{n}<tspan font-size="{FA}" fill="{D_INK2}" dx="10">of 100 favored</tspan>',
                   60, D_INK, SERIF, 300, "start" if n == n_s else "end", raw=True))
    assert f"{la_s:.2f}" == "0.39" and f"{la_m:.2f}" == "3.57"
    o.append(T(xs, gy + gs + 96, "log-odds, favored vs other", FL, D_INK2, SANS, 600))
    bx0, bx1 = xs + 116, W - 4
    S = lambda v: bx0 + v / 3.6 * (bx1 - bx0)
    by, bh, bg = gy + gs + 108, 24, 32
    o.append(T(bx0 - 12, by + 20, "samples", FL, D_GILT, SANS, 700, "end"))
    o.append(rect(bx0, by, S(la_s) - bx0, bh, D_GILT))
    o.append(T(S(la_s) + 10, by + 21, "0.39", FL, D_GILT, SANS, 700))
    o.append(T(S(la_s) + 84, by + 26, f'→  {mi("c", fill=D_GILT)} = 0.109', 34, D_GILT, SERIF, 600, raw=True))
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
        "or negative mode. In a Hopfield model with three patterns, the samples' grid shows 60 of 100 in the favored "
        "mode, the model's grid 97 of 100: the samples keep a fraction c = 0.109 of the model's log-odds.", defs)


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
    need("0.06587", "0.13314", "0.04645", "0.18868", "$3.5742$, $3.5747$ and $3.5749$", "only doubles")
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
    law = lambda v, N=2048: logit(norm.cdf(NG * math.pi ** 0.25 * v ** -0.25)) / L_EQ[N]
    sweep = {(r["N"], r["gamma"]): r["c_exact"] for r in load("experiments/cw_linear_sweep.json")}
    for (N, g), (c, e) in TABLE2.items():
        assert abs(law(N * N * g, N) / (c / (1 + e / 100)) - 1) < 0.001
        assert f"{sweep[(N, g)]:.5f}" == f"{c:.5f}", (N, g)
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


# ======================================================================== V3, V5: c against N, shared axes
SIZES = (1024, 2048, 4096)
FH2 = FH - 0.40      # under a two-line headline the frame is shorter by 0.40 in, so the frame bottoms align


def cn_axes(W, H, bg=CARD):
    """Log-log axes of c against N, the same data range in V3 and V5."""
    x0, x1, yt, yb = 78, W - 10, 34, H - 66
    lx0, lx1 = math.log(1024 / 1.42), math.log(4096 * 1.42)
    ly0, ly1 = math.log(0.040), math.log(0.180)
    X = lambda N: x0 + (math.log(N) - lx0) / (lx1 - lx0) * (x1 - x0)
    Y = lambda c: yb - (math.log(c) - ly0) / (ly1 - ly0) * (yb - yt)
    o = []
    for N in SIZES:
        o.append(L(X(N), yt, X(N), yb, RULE, 1.2))
        o.append(T(X(N), yb + 32, f"{N:,}", FL, INK3, SANS, 400, "middle"))
    for c in (0.05, 0.1, 0.15):
        o.append(L(x0, Y(c), x1, Y(c), RULE, 1.2))
        o.append(T(x0 - 12, Y(c) + 8, f"{c:g}", FL, INK3, SANS, 400, "end"))
    o.append(L(x0, yb, x1, yb, INK2, 2))
    o.append(L(x0, yt, x0, yb, INK2, 2))
    o.append(T(x0 - 12, yt - 4, f'{mi("c", fill=GILT_INK)}', FA, GILT_INK, SERIF, 600, "end", raw=True))
    o.append(T((x0 + x1) / 2, H - 8, f'size {mi("N")} (spins), log scale', FL, INK3, SANS, 400, "middle", raw=True))
    return X, Y, o


def measured(X, Y, o, p, bg):
    """Table 1 rows at one p: gold dots with their 95% intervals."""
    for N in SIZES:
        e = ROWS[f"N{N}_p{p}"]["estimate"]
        o.append(L(X(N), Y(e["lo"]), X(N), Y(e["hi"]), GILT_INK, 3))
        o.append(dot(X(N), Y(e["c"]), 8.5, GILT, bg, 2.5))


def v3():
    W, H = FW * IN, FH * IN
    need("1 & 1024 & exact chain & 0.0937 & $0.0928\\ [0.0909,0.0947]$ & 0.990",
         "1 & 2048 & exact chain & 0.0659 & $0.0659\\ [0.0647,0.0672]$ & 1.001",
         "1 & 4096 & exact chain & 0.0464 & $0.0472\\ [0.0464,0.0480]$ & 1.015",
         "ratios $0.990$--$1.015$", "the CW model is the one-pattern Hopfield model")
    for N, r in zip(SIZES, ("0.990", "1.001", "1.015")):
        assert f"{ROWS[f'N{N}_p1']['reading']['ratio']:.3f}" == r
    X, Y, o = cn_axes(W, H)
    o.append(P([(X(N), Y(C_CW[N])) for N in SIZES], INK, 2.4, "10 7"))     # exact chain: the prediction
    measured(X, Y, o, 1, CARD)
    x0, x1 = X(1024 / 1.42), X(4096 * 1.42)
    # direct labels: the line's above it (upper right), the dots' below it (lower left)
    o.append(T(x1, Y(0.112), f'exact chain, {mi("c", fill=INK)}{sub("CW", FS)}', FL, INK, SANS, 600, "end", raw=True))
    o.append(T(x0 + 14, Y(0.054), "Hopfield runs,", FL, GILT_INK, SANS, 700, "start"))
    o.append(T(x0 + 14, Y(0.054) + 28, "one pattern", FL, GILT_INK, SANS, 700, "start"))
    svg("V3", FW, FH, o, "Exact-chain c on the test ramp at N = 1,024, 2,048 and 4,096, falling as N to the minus one half; "
        "the one-pattern Hopfield runs lie on it, with ratios 0.990 to 1.015.")


# ======================================================================== V4: competition
def v4():
    W, H = FW * IN, FH * IN
    need("$A_3=1.662$", "independent Gaussians of common variance", "\\mathbb E\\max_{1\\le\\mu\\le p}|Z_\\mu|}{\\mathbb E|Z|}")
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


def v5():
    W, H = FW * IN, FH2 * IN
    need("the fraction pooled over all runs is within 3\\% of this\nprediction for three patterns at three sizes", "($\\gamma=0.004$, $\\beta=0.7$ to $1.6$)")
    need("$A_3=1.662$", "falls as $N^{-0.50}$ (exponent interval $[0.47,0.53]$)", "the exact CW chain gives $0.506$",
         "3 & 1024 & $A_3c_{\\mathrm{CW}}$ & 0.1558 & $0.1518\\ [0.1463,0.1571]$ & 0.975",
         "3 & 2048 & $A_3c_{\\mathrm{CW}}$ & 0.1095 & $0.1090\\ [0.1059,0.1123]$ & 0.995",
         "3 & 4096 & $A_3c_{\\mathrm{CW}}$ & 0.0772 & $0.0758\\ [0.0734,0.0782]$ & 0.982")
    A3 = PRED[(2048, 3)]["A_p"]
    assert f"{A3:.3f}" == "1.662" and f"{1.662 * 0.0659:.4f}" == f"{A3 * C_CW[2048]:.4f}" == "0.1095" and f"{C_CW[2048]:.4f}" == "0.0659"
    for N, cp in zip(SIZES, ("0.1558", "0.1095", "0.0772")):
        assert f"{ROWS[f'N{N}_p3']['c_pred']:.4f}" == cp
    ex = load("simple/analysis.json")["size_exponent_p3"]
    assert (f"{ex['value']:.2f}", f"{ex['lo']:.2f}", f"{ex['hi']:.2f}") == ("0.50", "0.47", "0.53")
    X, Y, o = cn_axes(W, H, CLIMAX)
    o.append(P([(X(N), Y(C_CW[N])) for N in SIZES], RULE_S, 2.4))                 # panel 3's line, faint
    o.append(P([(X(N), Y(A3 * C_CW[N])) for N in SIZES], INK, 2.4, "10 7"))       # the prediction
    xm = math.exp((math.log(1024) + math.log(2048)) / 2)
    cg = math.exp((math.log(C_CW[1024]) + math.log(C_CW[2048])) / 2)          # the faint line at xm
    o.append(arrow(X(xm), Y(cg) - 6, X(xm), Y(A3 * cg) + 6, INK, 2.6, 14))
    # the label sits between the two lines: baseline just above the faint line at its left end
    xs = X(xm) + 12
    ns = math.exp(math.log(1024) + (xs - X(1024)) / (X(2048) - X(1024)) * math.log(2))
    cs = math.exp(math.log(C_CW[1024]) + math.log(ns / 1024) / math.log(2) * math.log(C_CW[2048] / C_CW[1024]))
    o.append(T(xs, Y(cs) - 9, f'× {mi("A")}{sub("3", FS)}', FA, INK, SANS, 700, "start", raw=True))
    x0, x1 = X(1024 / 1.42), X(4096 * 1.42)
    o.append(T(x0 + 12, Y(0.059), "one pattern", FL, INK3, SANS, 500, "start"))
    measured(X, Y, o, 3, CLIMAX)
    o.append(T(x1, Y(0.172), "Hopfield runs, three patterns", FL, GILT_INK, SANS, 700, "end"))
    o.append(T(x1, Y(0.172) + 28, "predicted: dashed", FL, INK, SANS, 500, "end"))
    svg("V5", FW, FH2, o, "Predicted c for three patterns, 1.662 times the exact one-pattern line, at N = 1,024, 2,048 "
        "and 4,096, with the three measured values (pooled over pattern sets) and their 95% intervals; each point estimate is within 3%.")


# ======================================================================== V6: the eight ratios (Fig. 2b)
ORDER = ((1024, 1), (2048, 1), (4096, 1), (2048, 2), (1024, 3), (2048, 3), (4096, 3), (2048, 4))
PRINTED_RATIO = ("0.990", "1.001", "1.015", "1.011", "0.975", "0.995", "0.982", "0.940")


def v6():
    W, H = FW * IN, FH * IN
    need("$6\\%$ below $A_4c_{\\mathrm{CW}}$", "$[0.884,1.00]$", "$0.5$--$6\\%$ below", "single sets depart\nfrom it beyond sampling noise",
         "pooled agreement is thus an average over\npattern sets", "describes an average over random pattern sets, not one network",
         "protocol does not support the factor as derived",
         "For $p\\ge3$, $c_{\\mathrm{pred}}=A_pc_{\\mathrm{CW}}$", "4 & 2048 & $A_4c_{\\mathrm{CW}}$ & 0.1209 & $0.1137\\ [0.1069,0.1210]$ & 0.940")
    rows = []
    for (N, p), sr in zip(ORDER, PRINTED_RATIO):
        rd = ROWS[f"N{N}_p{p}"]["reading"]
        assert f"{rd['ratio']:.3f}" == sr, (N, p)
        need(f"{p} & {N} & ")
        rows.append((p, N, rd["ratio"], rd["ratio_lo"], rd["ratio_hi"], sr))
    assert (f"{rows[-1][3]:.3f}", f"{rows[-1][4]:.3f}") == ("0.884", "1.001")
    cx_p, cx_n, x0, x1, cx_r = 14, 132, 172, W - 92, W
    rl, rr = 0.86, 1.06
    X = lambda v: x0 + (v - rl) / (rr - rl) * (x1 - x0)
    rh, yh = 28, 22
    o = []
    o.append(T(cx_p, yh, "p", FA, INK3, MATH, 400, "middle", style="font-style:italic"))
    o.append(T(cx_n, yh, "N", FA, INK3, MATH, 400, "end", style="font-style:italic"))
    o.append(T((x0 + x1) / 2, yh, "measured ÷ predicted", FL, INK3, SANS, 400, "middle"))
    o.append(T(cx_r, yh, "ratio", FL, INK3, SANS, 400, "end"))
    o.append(L(0, yh + 12, W, yh + 12, INK2, 2))
    y = yh + 12
    ys = []
    for i, (p, N, r, lo, hi, sr) in enumerate(rows):
        if i == 4:                                    # p >= 3: the A_p c_CW rows
            o.append(L(0, y + 5, W, y + 5, RULE_S, 1.2))
            y += 10
        hot = p == 4
        if hot:
            o.append(rect(0, y + 1, W, rh, GILT_WASH, 'rx="3"'))
        yc = y + rh / 2 + 1
        ys.append(yc)
        ink, wt = (INK, 700) if hot else (INK2, 400)
        o.append(T(cx_p, yc + 8, str(p), FL, ink, SANS, wt, "middle"))
        o.append(T(cx_n, yc + 8, f"{N:,}", FL, ink, SANS, wt, "end"))
        o.append(T(cx_r, yc + 8, sr, FL, INK if hot else INK2, SANS, 700 if hot else 500, "end"))
        y += rh
    yb = y + 6
    for v in (0.9, 1.0):
        o.append(L(X(v), yh + 14, X(v), yb, RULE if v != 1.0 else INK, 1.2 if v != 1.0 else 2.2,
                   None if v != 1.0 else "8 6"))
    for (p, N, r, lo, hi, sr), yc in zip(rows, ys):
        o.append(L(X(lo), yc, X(hi), yc, GILT_INK, 3))
        o.append(dot(X(r), yc, 7.5, GILT, CARD, 2.2))
    o.append(L(x0, yb, x1, yb, INK2, 2))
    for v in (0.9, 1.0):
        o.append(T(X(v), yb + 30, f"{v:.1f}", FL, INK3, SANS, 400, "middle"))
    o.append(T(X(1.0) + 26, yb + 30, "= predicted", FL, INK3, SANS, 400, "start"))
    assert yb + 30 <= H, (yb, H)
    svg("V6", FW, FH, o, "Measured over predicted c for the eight rows of Table 1, with 95% intervals. The exact rows "
        "(p = 1, 2) lie within 1.5% of 1; the four p >= 3 rows lie below 1, by 0.5 to 6%; p = 4 is lowest, 0.940.")


if __name__ == "__main__":
    vh(); v1(); v2(); v3(); v4(); v5(); v6()
