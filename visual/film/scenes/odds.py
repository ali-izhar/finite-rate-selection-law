"""Scene "odds" — How much preference survives (script.md, scene odds; the page's chapter "odds").

Beats: field (a small bias G on every spin tilts the landscape toward the positive valley)
-> schedule (the paper's ten-step cooling schedule; the widest step at beta = 1)
-> counts (the exact count calculation: N + 1 probabilities over k; the sampler's histogram
   reshapes along the schedule while its positive share moves, then freezes; the model's own
   distribution, dashed; 97.3% for the model, 60.5% for the sampler)
-> logodds (q, log-odds bars 3.57 and 0.43; a share cannot double, log-odds can; doubling G
   doubles both, the ratio stays)
-> c (c = 0.43 / 3.57 = 0.120 on a dial from 0 to 1; odds to the power c: 36:1 -> 1.5:1)
-> flat (c at the paper's eight fields, varying by less than 0.3%; the answer: about an eighth survives).

Numbers (paper Sec. 4, App. A; script.md, Sources):
  ladder, sweeps per step (834), tau             ../../data/schedule.json (src/hopfield/protocol.py)
  q_eq 0.97274, q_chain 0.60546, c_CW 0.119808     ../../data/c_cw.json
  c_CW at the eight fields (0.119498-0.119808)     REPO/results/experiments/exact_protocol.json
  the sampler's histogram over k, the model's      the exact count calculation itself,
      REPO/src/experiments/exact_protocol.py (evolve), run through the paper's whole protocol at
      G = 0.0006125 (brief cooling, removal of m > 0, reheating 30 sweeps at beta 0.7, then the
      834-sweep schedule shown here); its final P(m > 0 | m != 0) is asserted to equal q_chain
      above; the model's equilibrium histogram at beta = 1.6 (detailed balance of the same rates)
      is asserted to give q_eq.
  36:1 and 1.5:1 are 0.97274/0.02726 = 35.7 and 0.60546/0.39454 = 1.53 = 35.7^0.1198.
Illustrations (tagged on screen): the landscape tilt is exaggerated (the real per-spin tilt
beta G = 0.001 is invisible); the doubled field is drawn as an exact doubling of both bars.
  2 x 97.3% = 194.6% is arithmetic (a share cannot double).

Render:  ./build.sh render -l odds   (or -h)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

SCENES = Path(__file__).resolve().parent
sys.path.insert(0, str(SCENES.parent))
from style import *  # noqa: E402,F403

VISUAL = SCENES.parents[1]
VDATA = VISUAL / "data"
# the repository root (results/, src/), read only; FILM_REPO overrides it for a copy outside the repo
REPO = Path(os.environ.get("FILM_REPO", VISUAL.parent))
CACHE = SCENES.parent / "build" / "cache" / "odds_exact_chain.npz"
# a private TeX cache: other scenes render in parallel into media/Tex, and a shared cache can race
config.tex_dir = str(SCENES.parent / "media" / "Tex_odds")

SCHED = json.loads((VDATA / "schedule.json").read_text())
CCW = json.loads((VDATA / "c_cw.json").read_text())
ROWS = json.loads((REPO / "results" / "experiments" / "exact_protocol.json").read_text())
LADDER = SCHED["ladder"]
SWEEPS = SCHED["K"]
G0 = SCHED["G0"]
N = SCHED["N"]
Q_EQ = CCW["runs"][-1]["q_eq"]
Q_CH = CCW["runs"][-1]["q_chain"]
L_EQ = float(np.log(Q_EQ / (1 - Q_EQ)))     # 3.57
L_CH = float(np.log(Q_CH / (1 - Q_CH)))     # 0.43
C_CW = CCW["c_cw"]                          # 0.120
assert SWEEPS[4] == 355 and sum(SWEEPS) == 834 and round(C_CW, 3) == 0.120
assert round(L_EQ, 2) == 3.57 and round(L_CH, 2) == 0.43
assert round(Q_EQ / (1 - Q_EQ)) == 36 and round(Q_CH / (1 - Q_CH), 1) == 1.5
C_FIELDS = np.array([r["c_pred"] for r in ROWS])
G_FIELDS = np.array([r["G"] for r in ROWS])
assert len(ROWS) == 8 and (C_FIELDS.max() - C_FIELDS.min()) / C_FIELDS.max() < 0.003

NBINS = 256             # 8 counts k per bin (2048 / 256), so the bins do not alias
SPIN_PLUS = INK_2       # a +1 spin (as in the commit scene's row)
SPIN_MINUS = "#2D2F33"  # a -1 spin
H_VIS = 0.05            # the drawn tilt (exaggerated)


# ------------------------------------------------------------------ the exact count calculation
def exact_chain_snapshots():
    """Histogram over m of the sampler along the 834-sweep schedule (exact, no sampling), its
    positive share conditioned on m != 0 (the paper's q), and the model's own equilibrium
    histogram at the final beta (same bins, same probability units)."""
    if CACHE.exists():
        z = np.load(CACHE)
        if "heq" in z.files:
            return z["t"], z["h"], z["q"], z["heq"]
    sys.dont_write_bytecode = True          # never write __pycache__ into the repository
    sys.path.insert(0, str(REPO))
    from scipy.stats import binom
    from src.experiments import exact_protocol as E
    from src.rates import flip_rates
    dt, G = 0.02, G0
    m = E.magnetisation(N)
    p = binom.pmf(np.arange(N + 1), N, 0.5)                   # the paper's protocol, steps 1-4
    for b in LADDER:
        p = E.evolve(p, b, b * G, CCW["scarce_sweeps_per_level"], dt)
    p = np.where(m < 0, p, 0.0)
    p /= p.sum()
    p = E.evolve(p, SCHED["beta_reheat"], SCHED["beta_reheat"] * G, SCHED["reheat_sweeps"], dt)
    edges = np.linspace(-1, 1, NBINS + 1)
    idx = np.clip(np.digitize(m, edges) - 1, 0, NBINS - 1)

    def hist(p):
        return np.bincount(idx, weights=p, minlength=NBINS)

    def share(p):
        return p[m > 0].sum() / p[m != 0].sum()

    ts, hs, qs, t = [0.0], [hist(p)], [share(p)], 0.0
    for b, K in zip(LADDER, SWEEPS):                          # step 5: the 834-sweep schedule
        n = max(3, int(np.ceil(K / 4)))
        for _ in range(n):
            p = E.evolve(p, b, b * G, K / n, dt)
            t += K / n
            ts.append(t); hs.append(hist(p)); qs.append(share(p))
    assert abs(qs[-1] - Q_CH) < 1e-6, (qs[-1], Q_CH)
    b = LADDER[-1]                                            # the model at beta = 1.6 (detailed balance)
    wp, wm = flip_rates(N, b, b * G, check=False)
    logpi = np.zeros(N + 1)
    logpi[1:] = np.cumsum(np.log(wp[:-1]) - np.log(wm[1:]))
    pi = np.exp(logpi - logpi.max()); pi /= pi.sum()
    assert abs(share(pi) - Q_EQ) < 1e-6, (share(pi), Q_EQ)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, t=np.array(ts), h=np.array(hs), q=np.array(qs), heq=hist(pi))
    return np.array(ts), np.array(hs), np.array(qs), hist(pi)


def landscape(m, beta, h=0.0):
    """-log p(m) per spin, large-N Curie-Weiss with field h, minus its zero-field minimum."""
    def f(x, hh):
        x = np.clip(x, -0.999, 0.999)
        ent = (1 + x) / 2 * np.log((1 + x) / 2) + (1 - x) / 2 * np.log((1 - x) / 2)
        return -beta * x**2 / 2 - hh * x + ent + np.log(2)
    ms = 0.9
    for _ in range(200):
        ms = np.tanh(beta * ms)
    return f(m, h) - f(ms, 0.0)


def count_grid(n_blue, center, cell=0.24, gap=0.045):
    """10 x 10 grid: round(100 q) blue squares first, in reading order (the page's rule)."""
    g = VGroup()
    step = cell + gap
    for i in range(100):
        r, c = divmod(i, 10)
        g.add(Square(cell, stroke_width=0, fill_color=MODE_A if i < n_blue else MODE_B, fill_opacity=0.92)
              .move_to(center + np.array([(c - 4.5) * step, -(r - 4.5) * step, 0])))
    return g


class Odds(NarratedScene):
    scene_id = "odds"

    def word(self, w: str, n: int = 1) -> float:
        """Start time of the n-th spoken word that matches w (punctuation ignored)."""
        k = 0
        for x in self.timing["words"]:
            if x["w"].strip(".,:;!?").lower() == w.lower():
                k += 1
                if k == n:
                    return x["start"]
        raise KeyError(w)

    def at(self, t: float, *anims, **kw):
        """Start untimed (not cue-bound) animations at a time taken from the word timings."""
        self.wait_until(t)
        self.play(*anims, **kw)

    # ---------------------------------------------------------------- the landscape with arrows
    def make_landscape(self, center, scale, tilt):
        ax = Axes(x_range=[-1, 1, 1], y_range=[-0.05, 0.4, 0.1], x_length=9 * scale, y_length=3.3 * scale,
                  tips=False, axis_config={"stroke_opacity": 0}).move_to(center)
        xs = np.linspace(-0.97, 0.97, 241)

        def part(lo, hi, color):
            sel = xs[(xs >= lo - 1e-9) & (xs <= hi + 1e-9)]
            pts = [ax.c2p(v, landscape(v, 1.6, tilt.get_value())) for v in sel]
            return VMobject(stroke_width=5 * max(scale, 0.6), stroke_color=color).set_points_smoothly(pts)

        curve = always_redraw(lambda: VGroup(part(-0.97, 0.0, MODE_B), part(0.0, 0.97, MODE_A)))
        return ax, curve

    def construct(self):
        ts, hs, qs, heq = exact_chain_snapshots()

        # ============================================================ beat: field
        tilt = ValueTracker(0.0)
        ax, curve = self.make_landscape(DOWN * 0.7, 1.0, tilt)
        base = Line(ax.c2p(-1.02, -0.05), ax.c2p(1.02, -0.05), stroke_color=RULE, stroke_width=2)
        mlab = mtex("m", size=SIZE_BODY, color=INK_2).next_to(base, RIGHT, buff=0.2)
        ticks = VGroup(*[mtex(s, size=SIZE_LABEL, color=INK_3).next_to(ax.c2p(x, -0.05), DOWN, buff=0.16)
                         for s, x in (("-1", -1), ("0", 0), ("+1", 1))])
        ylab = label("height = −log probability", size=SIZE_SMALL).move_to(ax.c2p(-1, 0.42), aligned_edge=UL)
        self.play_at("field", Create(curve), FadeIn(base), FadeIn(mlab), FadeIn(ticks), FadeIn(ylab),
                     run_time=1.4)

        arrows = VGroup(*[Arrow(ORIGIN, RIGHT * 0.5, buff=0, stroke_width=4, color=BIAS,
                                max_tip_length_to_length_ratio=0.35) for _ in range(10)])
        arrows.arrange(RIGHT, buff=0.28).move_to([0, 2.55, 0])
        on_spins = label("on every spin, toward +1", size=SIZE_SMALL, color=BIAS).next_to(arrows, UP, buff=0.2)
        self.at(self.word("small"), LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.1),
                run_time=1.8)
        self.at(self.word("every"), FadeIn(on_spins, shift=UP * 0.1), run_time=0.7)
        Gsym = mtex("G", size=SIZE_HEAD, color=BIAS).next_to(arrows, RIGHT, buff=0.35)
        self.play_at("G", Write(Gsym), run_time=0.8)
        exag = label("tilt exaggerated", size=SIZE_SMALL).to_corner(DL, buff=0.45)
        self.play_at("tilts", tilt.animate.set_value(H_VIS), FadeIn(exag), run_time=2.0)
        self.at(self.word("positive") - 0.05,
                ShowPassingFlash(curve.copy().clear_updaters()[1].set_stroke(MODE_A, width=10), time_width=0.6),
                run_time=0.9)
        field_group = VGroup(base, mlab, ticks, ylab, arrows, on_spins, Gsym, exag)

        # ============================================================ beat: schedule
        sax = Axes(x_range=[0, 834, 100], y_range=[0.5, 1.7, 0.1], x_length=10.4, y_length=4.4,
                   axis_config={"color": RULE, "stroke_width": 2, "include_ticks": False},
                   tips=False).shift(DOWN * 0.15 + RIGHT * 0.3)
        cum = np.concatenate([[0], np.cumsum(SWEEPS)])
        pts = []
        for i, b in enumerate(LADDER):
            pts += [sax.c2p(cum[i], b), sax.c2p(cum[i + 1], b)]
        stair = VMobject(stroke_color=INK, stroke_width=4).set_points_as_corners(pts)
        b_ticks = VGroup(*[mtex(s, size=SIZE_LABEL, color=INK_3).next_to(sax.c2p(0, b), LEFT, buff=0.18)
                           for s, b in (("0.6", 0.6), ("1", 1.0), ("1.6", 1.6))])
        b_lbl = mtex(r"\beta", size=SIZE_BODY, color=INK_2).next_to(sax.c2p(0, 1.72), LEFT, buff=0.2)
        cool = label("cooling ↑", size=SIZE_SMALL).next_to(b_lbl, LEFT, buff=0.25)
        x_lbl = label("sweeps", size=SIZE_LABEL).next_to(sax.x_axis.get_right(), DOWN, buff=0.2).shift(LEFT * 0.4)
        corners = VGroup(*[Dot(sax.c2p(cum[i], b), radius=0.05, color=INK) for i, b in enumerate(LADDER)])
        ten = label("10 steps", size=SIZE_LABEL, color=INK_2).move_to(sax.c2p(110, 1.52))

        # a crossfade: the landscape leaves while the staircase draws (no empty frame)
        quick = lambda t: smooth(min(1.0, 2.5 * t))     # the old picture is gone in the first 40%
        late = lambda t: smooth(max(0.0, (t - 0.3) / 0.7))  # the new one starts after 30%
        self.play_at("schedule", FadeOut(field_group, shift=UP * 0.3, rate_func=quick),
                     FadeOut(curve, shift=UP * 0.3, rate_func=quick), Create(sax, rate_func=late),
                     FadeIn(b_ticks, rate_func=late), FadeIn(b_lbl, rate_func=late), FadeIn(cool, rate_func=late),
                     FadeIn(x_lbl, rate_func=late), Create(stair, rate_func=lambda t: max(0.0, (t - 0.3) / 0.7)),
                     run_time=1.6)
        self.remove(curve)
        self.at(self.word("ten"), FadeIn(corners), FadeIn(ten), run_time=0.6)

        step1 = Line(sax.c2p(cum[4], 1.0), sax.c2p(cum[5], 1.0), stroke_color=INK, stroke_width=9)
        brace = BraceBetweenPoints(sax.c2p(cum[4], 1.0), sax.c2p(cum[5], 1.0), UP, color=INK_3, buff=0.12)
        wid = tex(r"355 sweeps at $\beta = 1$", size=SIZE_LABEL, color=INK_2).next_to(brace, UP, buff=0.1)
        self.play_at("longest", Create(step1), GrowFromCenter(brace), run_time=0.8)
        self.at(self.word("beta"), FadeIn(wid, shift=UP * 0.1), run_time=0.7)
        design = tex(r"\mbox{834 sweeps in all $\cdot$ sweeps per step $= 6\times$ relaxation time, rounded up "
                     r"$\cdot$ a design choice}", size=SIZE_SMALL, color=INK_3).to_edge(DOWN, buff=0.35)
        self.at(self.word("settle") - 0.4, FadeIn(design), run_time=0.8)

        # ============================================================ beat: counts
        mini = VGroup(sax, stair, corners)
        rng = np.random.default_rng(1983)
        spins = rng.random(64) < 0.5
        n_plus = int(spins.sum())
        side, gap = 0.17, 0.035
        row = VGroup(*[Square(side, stroke_color=RULE, stroke_width=1,
                              fill_color=SPIN_PLUS if s else SPIN_MINUS, fill_opacity=1) for s in spins])
        row.arrange(RIGHT, buff=gap).move_to([0, 1.0, 0])
        slots = [row[i].get_center() for i in range(64)]
        order = list(np.where(~spins)[0]) + list(np.where(spins)[0])   # minus left, plus right
        targets = {int(j): slots[r] for r, j in enumerate(order)}
        self.play_at("count",
                     FadeOut(VGroup(b_ticks, b_lbl, cool, x_lbl, ten, step1, brace, wid, design)),
                     mini.animate.scale(0.3).move_to([2.0, 3.15, 0]),
                     FadeIn(row, lag_ratio=0.02), run_time=1.0)
        self.play(*[row[j].animate.move_to(targets[j]) for j in range(64)], run_time=1.6)
        plus_block = VGroup(*[row[j] for j in np.where(spins)[0]])
        kbrace = Brace(plus_block, DOWN, color=INK_3, buff=0.1)
        klab = tex(r"$k$ = number of positive spins", size=SIZE_LABEL, color=INK_2).next_to(kbrace, DOWN, buff=0.12)
        self.at(self.word("positive", 2) - 0.4, GrowFromCenter(kbrace), FadeIn(klab), run_time=0.8)

        # the histogram over k (the sampler's probabilities), right of centre; text column left
        HX0, HX1, HY0, HH = -1.4, 5.8, -1.7, 3.0
        bw = (HX1 - HX0) / NBINS
        mid = (np.linspace(-1, 1, NBINS + 1)[:-1] + np.linspace(-1, 1, NBINS + 1)[1:]) / 2
        hbase = Line([HX0, HY0, 0], [HX1, HY0, 0], stroke_color=RULE, stroke_width=2)
        hlab = VGroup(mtex("0", size=SIZE_SMALL, color=INK_3).next_to(hbase.get_left(), DOWN, buff=0.1),
                      mtex(r"N/2", size=SIZE_SMALL, color=INK_3).next_to(hbase.get_center(), DOWN, buff=0.1),
                      mtex("N", size=SIZE_SMALL, color=INK_3).next_to(hbase.get_right(), DOWN, buff=0.1))
        hk = mtex("k", size=SIZE_LABEL, color=INK_2).next_to(hbase, RIGHT, buff=0.15)
        T0 = self.t("watch") + 0.4          # the schedule runs from here ...
        T1 = self.t("q-eq") - 0.3           # ... to here, at a constant pace
        hscale = ValueTracker(0.0)          # the histogram's drawn height (grows from the tick)

        def sweeps_now():
            return 834.0 * float(np.clip((self.now - T0) / (T1 - T0), 0, 1))

        def hist_now():
            s = sweeps_now()
            j = int(np.clip(np.searchsorted(ts, s), 1, len(ts) - 1))
            a = float(np.clip((s - ts[j - 1]) / (ts[j] - ts[j - 1]), 0, 1))
            h = (1 - a) * hs[j - 1] + a * hs[j]
            return h / h.max() * hscale.get_value()

        def area(h, sel, color):
            xs_ = [HX0 + (i + 0.5) * bw for i in sel]
            ys_ = [HY0 + HH * max(h[i], 1e-4) for i in sel]
            pts = [[xs_[0] - bw / 2, HY0, 0]] + [[x, y, 0] for x, y in zip(xs_, ys_)] + [[xs_[-1] + bw / 2, HY0, 0]]
            return Polygon(*pts, stroke_width=1.5, stroke_color=color, fill_color=color, fill_opacity=0.85)

        neg_i = [i for i in range(NBINS) if mid[i] < 0]
        pos_i = [i for i in range(NBINS) if mid[i] > 0]
        hist = always_redraw(lambda: VGroup(area(hist_now(), neg_i, MODE_B), area(hist_now(), pos_i, MODE_A)))

        def beta_at(s):
            i = int(np.searchsorted(cum, s, side="right") - 1)
            return LADDER[min(max(i, 0), 9)]

        # track: the sorted row shrinks into one tick at k = N/2; the histogram grows from it
        tick = Line([0, 0, 0], [0, 0.22, 0], stroke_color=INK, stroke_width=3).move_to(
            hbase.get_center() + UP * 0.11)
        hname = label("sampler", size=SIZE_LABEL, color=INK_2).move_to([-0.3, 1.75, 0])
        self.play_at("track", ReplacementTransform(row, tick), FadeOut(VGroup(kbrace, klab)),
                     Create(hbase), FadeIn(hlab), FadeIn(hk), run_time=0.9)
        self.add(hist)
        self.play(hscale.animate.set_value(1.0), FadeOut(tick), FadeIn(hname), run_time=0.9)
        np1 = mtex(r"N+1", r"\ \text{probabilities}", size=SIZE_BODY, color=INK).move_to([-4.6, 0.6, 0])
        self.at(self.word("N"), Write(np1), run_time=0.9)
        inst = mtex(r"\text{instead of}\ ", r"2^{N}", size=SIZE_BODY, color=INK_3).next_to(np1, DOWN, buff=0.3)
        self.play_at("exact", Write(inst), run_time=1.0)
        ntag = tex(r"\mbox{$N = 2{,}048$ $\cdot$ $G = 0.0006125$ $\cdot$ computed exactly from the probability of each count $k$}",
                   size=SIZE_SMALL, color=INK_3).to_edge(DOWN, buff=0.3)
        self.play(FadeIn(ntag), run_time=0.6)

        # watch: the playhead runs the schedule; the share moves while the bump splits, then freezes
        head = always_redraw(lambda: Dot(sax.c2p(min(sweeps_now(), 833.9), beta_at(min(sweeps_now(), 833.9))),
                                         radius=0.07, color=INK))
        share = always_redraw(lambda: tex(r"positive: %.1f\%%" % (100 * float(np.interp(sweeps_now(), ts, qs))),
                                          size=SIZE_LABEL, color=MODE_A).move_to([2.3, 1.75, 0]))
        self.play_at("watch", FadeIn(head, scale=0.5), FadeIn(share), run_time=0.4)

        # q-eq: the model's own distribution at beta = 1.6, dashed, on the same scale; the sampler's
        # histogram shrinks to that scale (its positive spike holds 60.5%, the model's 97.3%)
        r = float(heq.max() / hs[-1].max())
        hq = heq / heq.max()
        opts = [[HX0, HY0, 0]] + [[HX0 + (i + 0.5) * bw, HY0 + HH * hq[i], 0] for i in range(NBINS)] + [[HX1, HY0, 0]]
        outline = theory(VMobject().set_points_as_corners(opts).set_stroke(width=3), num_dashes=140)
        x_star = HX0 + (mid[int(np.argmax(hq))] + 1) / 2 * (HX1 - HX0)      # the dashed spike, m = 0.895
        mname = label("model", size=SIZE_LABEL, color=THEORY).next_to([x_star, HY0 + HH, 0], UP, buff=0.12)
        self.play_at("q-eq", FadeOut(VGroup(np1, inst)), hscale.animate.set_value(1.0 / r),
                     Create(outline), FadeIn(mname), run_time=1.2)

        gm = count_grid(round(100 * Q_EQ), np.array([-3.7, -0.35, 0]))
        gs = count_grid(round(100 * Q_CH), np.array([3.7, -0.35, 0]))
        hm = label("model", size=SIZE_LABEL, color=INK_2).next_to(gm, UP, buff=0.25)
        hs_ = label("sampler", size=SIZE_LABEL, color=INK_2).next_to(gs, UP, buff=0.25)
        pm = mtex(r"97.3\%", size=SIZE_HEAD).next_to(gm, DOWN, buff=0.3)
        pc = mtex(r"60.5\%", size=SIZE_HEAD).next_to(gs, DOWN, buff=0.3)
        self.at(self.word("ninety-seven") - 0.15, ReplacementTransform(outline, gm),
                ReplacementTransform(mname, hm), Write(pm), run_time=1.1)

        # q-chain: the histogram pours into the sampler's grid
        frozen = hist.copy().clear_updaters()
        self.remove(hist)
        self.add(frozen)
        share_f = share.copy().clear_updaters()
        self.remove(share)
        self.add(share_f)
        self.play_at("q-chain",
                     ReplacementTransform(frozen[1], VGroup(*gs[:61])),
                     ReplacementTransform(frozen[0], VGroup(*gs[61:])),
                     ReplacementTransform(share_f, pc), ReplacementTransform(hname, hs_),
                     FadeOut(VGroup(hbase, hlab, hk, mini)), FadeOut(head),
                     run_time=1.2)
        self.remove(head)
        self.remove(*frozen)
        self.add(gs)

        # ============================================================ beat: logodds
        qm = mtex("q", "=", size=SIZE_HEAD, color=INK).next_to(pm, LEFT, buff=0.15)
        qc = mtex("q", "=", size=SIZE_HEAD, color=INK).next_to(pc, LEFT, buff=0.15)
        self.play_at("logit", FadeOut(ntag), run_time=0.3)
        self.at(self.word("q"), Write(qm), Write(qc), run_time=0.8)
        formula = mtex(r"\text{log-odds}", "=", r"\log\frac{q}{1-q}", size=SIZE_HEAD).move_to([0, 2.95, 0])
        self.at(self.word("log-odds"), Write(formula[0]), run_time=0.7)
        self.at(self.word("log"), Write(formula[1:]), run_time=2.2)

        # bars: x of log-odds 0 and units per log-odds; room for the doubled bars (7.14)
        X0, UNIT, Y_M, Y_C, BH = -3.6, 1.1, 0.75, -0.55, 0.5
        laxis = NumberLine(x_range=[0, 8, 1], length=8 * UNIT, color=RULE, stroke_width=2,
                           include_numbers=False, tick_size=0.06).move_to([X0 + 4 * UNIT, -1.35, 0])
        lnums = VGroup(*[mtex(str(k), size=SIZE_SMALL, color=INK_3).next_to(laxis.n2p(k), DOWN, buff=0.1)
                         for k in range(0, 9, 2)])
        zero = Line([X0, -1.35, 0], [X0, 1.45, 0], stroke_color=INK_3, stroke_width=2)
        zlab = label("0 = equal modes", size=SIZE_SMALL).next_to(zero, UP, buff=0.1)

        def slices(L, y):
            w = L * UNIT / 100
            return VGroup(*[Rectangle(width=w * 1.03, height=BH, stroke_width=0, fill_color=MODE_A,
                                      fill_opacity=1).move_to([X0 + (i + 0.5) * w, y, 0]) for i in range(100)])

        def solid(L, y, color=MODE_A):
            return Rectangle(width=L * UNIT, height=BH, stroke_width=0, fill_color=color,
                             fill_opacity=1).move_to([X0, y, 0], aligned_edge=LEFT)

        bm_sl, bc_sl = slices(L_EQ, Y_M), slices(L_CH, Y_C)
        bar_m, bar_c = solid(L_EQ, Y_M), solid(L_CH, Y_C)
        num_m = mtex(f"{L_EQ:.2f}", size=SIZE_BODY).next_to(bar_m, RIGHT, buff=0.2)
        num_c = mtex(f"{L_CH:.2f}", size=SIZE_BODY).next_to(bar_c, RIGHT, buff=0.2)
        lm = label("model", size=SIZE_LABEL, color=INK_2).next_to([X0, Y_M, 0], LEFT, buff=0.3)
        lc = label("sampler", size=SIZE_LABEL, color=INK_2).next_to([X0, Y_C, 0], LEFT, buff=0.3)
        self.play_at("bar-model", ReplacementTransform(gm, bm_sl, lag_ratio=0.004),
                     FadeOut(VGroup(pm, qm)), ReplacementTransform(hm, lm),
                     Create(laxis), FadeIn(lnums), Create(zero), FadeIn(zlab),
                     VGroup(gs, pc, qc, hs_).animate.scale(0.55).move_to([5.45, 0.25, 0]),
                     run_time=1.5)
        self.remove(bm_sl); self.add(bar_m)
        self.play(Write(num_m), run_time=0.5)
        self.play_at("bar-chain", ReplacementTransform(gs, bc_sl, lag_ratio=0.004),
                     FadeOut(VGroup(pc, qc)), ReplacementTransform(hs_, lc), run_time=1.4)
        self.remove(bc_sl); self.add(bar_c)
        self.play(Write(num_c), run_time=0.5)

        # why-logit: a share cannot double past 100%; log-odds can
        dbl = mtex(r"2\times 97.3\%", "=", r"194.6\%\ ?", size=SIZE_HEAD, color=INK_3).move_to([3.4, -2.75, 0])
        self.play_at("why-logit", Write(dbl), run_time=1.0)
        strike = Line(dbl.get_left() + LEFT * 0.1, dbl.get_right() + RIGHT * 0.1, stroke_color=INK_2, stroke_width=4)
        self.at(self.word("can", 2) - 0.6, Create(strike), run_time=0.5)

        # exponent: the tilted landscape returns small, top right; the formula moves top left
        tilt2 = ValueTracker(H_VIS)
        ax2, curve2 = self.make_landscape(np.array([4.5, 2.45, 0]), 0.42, tilt2)
        arrows2 = VGroup(*[Arrow(ORIGIN, RIGHT * 0.28, buff=0, stroke_width=3, color=BIAS,
                                 max_tip_length_to_length_ratio=0.35) for _ in range(5)])
        arrows2.arrange(RIGHT, buff=0.14).next_to(ax2, UP, buff=0.0).shift(DOWN * 0.15)
        G2 = mtex("G", size=SIZE_BODY, color=BIAS).next_to(arrows2, LEFT, buff=0.2)
        self.play_at("exponent", formula.animate.scale(0.75).to_corner(UL, buff=0.45),
                     FadeOut(VGroup(dbl, strike)), Create(curve2), FadeIn(arrows2), FadeIn(G2), run_time=1.0)
        self.at(self.word("G", 2) - 0.1, Indicate(G2, color=BIAS, scale_factor=1.4), run_time=0.7)

        G2x = mtex("2G", size=SIZE_BODY, color=BIAS).move_to(G2, aligned_edge=RIGHT)
        illus = label("weak field · illustration", size=SIZE_SMALL).to_corner(DL, buff=0.4)
        x2m = mtex(r"2\times 3.57", size=SIZE_BODY).next_to(solid(2 * L_EQ, Y_M), RIGHT, buff=0.2)
        x2c = mtex(r"2\times 0.43", size=SIZE_BODY).next_to(solid(2 * L_CH, Y_C), RIGHT, buff=0.2)
        self.play_at("doubling", ReplacementTransform(G2, G2x), tilt2.animate.set_value(2 * H_VIS),
                     bar_m.animate.stretch_about_point(2, 0, bar_m.get_left()),
                     bar_c.animate.stretch_about_point(2, 0, bar_c.get_left()),
                     ReplacementTransform(num_m, x2m), ReplacementTransform(num_c, x2c),
                     *[a.animate.scale(1.4, about_point=a.get_start()) for a in arrows2],
                     FadeIn(illus), run_time=1.6)

        frac2 = mtex(r"\frac{2\times 0.43}{2\times 3.57}", "=", r"\frac{0.43}{3.57}",
                     size=SIZE_HEAD, color=INK_2).move_to([3.4, -2.75, 0])
        self.play_at("ratio-fixed", Write(frac2[0]), run_time=0.9)
        num_m2 = mtex(f"{L_EQ:.2f}", size=SIZE_BODY).next_to(solid(L_EQ, Y_M), RIGHT, buff=0.2)
        num_c2 = mtex(f"{L_CH:.2f}", size=SIZE_BODY).next_to(solid(L_CH, Y_C), RIGHT, buff=0.2)
        G2b = mtex("G", size=SIZE_BODY, color=BIAS).move_to(G2x, aligned_edge=RIGHT)
        self.play(Write(frac2[1:]), bar_m.animate.stretch_about_point(0.5, 0, bar_m.get_left()),
                  bar_c.animate.stretch_about_point(0.5, 0, bar_c.get_left()),
                  ReplacementTransform(x2m, num_m2), ReplacementTransform(x2c, num_c2),
                  ReplacementTransform(G2x, G2b), tilt2.animate.set_value(H_VIS), FadeOut(illus),
                  *[a.animate.scale(1 / 1.4, about_point=a.get_start()) for a in arrows2],
                  run_time=1.2)

        # ============================================================ beat: c
        # the left fraction leaves first; 0.43/3.57 stays and gains "c ="; the short bar lies on
        # the long one in gold (12% of it), then faint cuts show about eight copies fit
        ratio = mtex("c", "=", r"\frac{0.43}{3.57}", size=SIZE_HEAD, colors={"c": GILT_TEXT}).move_to([3.4, -2.75, 0])
        lifted = bar_c.copy()
        lay = Rectangle(width=bar_c.width, height=BH, stroke_width=0, fill_color=GILT, fill_opacity=1)
        lay.move_to([X0, Y_M, 0], aligned_edge=LEFT)
        self.play_at("ratio", FadeOut(frac2[:2]), run_time=0.3)
        self.play(frac2[2].animate.move_to(ratio[2]), FadeIn(ratio[:2]), Transform(lifted, lay), run_time=0.7)
        self.remove(frac2[2]); self.add(ratio)
        value = mtex("c", "=", f"{C_CW:.3f}", size=SIZE_HEAD, colors={"c": GILT_TEXT}).move_to(ratio, aligned_edge=LEFT)
        value[2].set_color(GILT_TEXT)
        self.play_at("c", FadeOut(ratio[2]), run_time=0.35)
        self.play(ReplacementTransform(ratio[:2], value[:2]), Write(value[2]), run_time=0.6)
        cuts = VGroup(*[Line([X0 + j * L_CH * UNIT, Y_M - BH / 2, 0], [X0 + j * L_CH * UNIT, Y_M + BH / 2, 0],
                             stroke_color=BG, stroke_width=3) for j in range(1, 9)])
        self.play(LaggedStart(*[Create(x) for x in cuts], lag_ratio=0.25), run_time=1.2)

        # the long bar bends into a dial from 0 (equal modes) to 1 (the model's proportions)
        DC, R = np.array([-3.4, -1.0, 0]), 2.4
        track = AnnularSector(inner_radius=R - 0.07, outer_radius=R + 0.07, start_angle=PI, angle=-PI,
                              arc_center=DC, fill_color=RULE, fill_opacity=1, stroke_width=0)
        kept = AnnularSector(inner_radius=R - 0.09, outer_radius=R + 0.09, start_angle=PI, angle=-PI * C_CW,
                             arc_center=DC, fill_color=GILT, fill_opacity=1, stroke_width=0)
        ang = PI * (1 - C_CW)
        needle = Line(DC, DC + 0.9 * R * np.array([np.cos(ang), np.sin(ang), 0]), stroke_color=GILT, stroke_width=5)
        hub = Dot(DC, radius=0.08, color=GILT)
        d0 = VGroup(mtex("0", size=SIZE_LABEL, color=INK_2), label("equal modes", size=SIZE_SMALL))
        d0.arrange(DOWN, buff=0.08).next_to(DC + LEFT * R, DOWN, buff=0.15)
        d1 = VGroup(mtex("1", size=SIZE_LABEL, color=INK_2), label("the model's proportions", size=SIZE_SMALL))
        d1.arrange(DOWN, buff=0.08).next_to(DC + RIGHT * R, DOWN, buff=0.15)
        vdial = value.copy().move_to(DC + UP * 0.95)
        # one strip bends: a curve of constant curvature whose turning grows from 0 to pi, its
        # length from the bar's to the half circle's, its left end gliding from the bar's 0 to the
        # dial's 0; the gold piece is the same strip's first 12% (L_CH / L_EQ = c), so it stays put
        bend = ValueTracker(0.0)
        P_BAR, P_DIAL = np.array([X0, Y_M, 0]), DC + LEFT * R
        LEN_BAR, LEN_DIAL = L_EQ * UNIT, PI * R

        def strip(s1, half0, half1, c0, c1):
            k = bend.get_value()
            ln = (1 - k) * LEN_BAR + k * LEN_DIAL
            a0, phi = k * PI / 2, k * PI           # start direction; total clockwise turning
            p0 = (1 - k) * P_BAR + k * P_DIAL
            hw = (1 - k) * half0 + k * half1
            us = np.linspace(0, s1 * ln, 64)
            if phi < 1e-4:
                cen = np.array([p0 + u * np.array([np.cos(a0), np.sin(a0), 0]) for u in us])
                dirs = np.full_like(us, a0)
            else:
                kap = phi / ln
                dirs = a0 - kap * us
                cen = np.array([p0 + np.array([np.sin(a0) - np.sin(d), np.cos(d) - np.cos(a0), 0]) / kap
                                for d in dirs])
            nrm = np.stack([-np.sin(dirs), np.cos(dirs), 0 * dirs], axis=1)   # left normal: outward
            pts = list(cen + hw * nrm) + list((cen - hw * nrm)[::-1])
            return Polygon(*pts, stroke_width=0, fill_color=interpolate_color(ManimColor(c0), ManimColor(c1), k), fill_opacity=1)

        bent_m = always_redraw(lambda: strip(1.0, BH / 2, 0.07, MODE_A, RULE))
        bent_c = always_redraw(lambda: strip(C_CW, BH / 2, 0.09, GILT, GILT))
        self.at(self.word("fraction") - 0.35, FadeOut(cuts), FadeOut(num_m2), run_time=0.35)
        self.remove(bar_m, lifted)
        self.add(bent_m, bent_c)
        self.play(bend.animate.set_value(1.0),
                  FadeOut(VGroup(bar_c, num_c2, laxis, lnums, zero, zlab, lm, lc,
                                 curve2, arrows2, G2b, formula), run_time=0.7),
                  value.animate.move_to(vdial),
                  run_time=1.4)
        self.remove(bent_m, bent_c, curve2)
        self.add(track, kept)
        self.play(GrowFromPoint(needle, DC), FadeIn(hub), FadeIn(d0), FadeIn(d1), run_time=0.9)

        power = mtex(r"\text{sampler odds}", "=", r"(\text{model odds})^{", "c", "}",
                     size=SIZE_BODY, colors={"c": GILT_TEXT}).move_to([3.4, 1.6, 0])
        self.play_at("power", Write(power), run_time=1.8)
        ex1 = mtex(r"36:1", r"\;\to\;", r"36^{", "c", r"}:1", r"\approx 1.5:1",
                   size=SIZE_HEAD, colors={"c": GILT_TEXT}).move_to([3.4, 0.35, 0])
        src = mtex(r"97.3:2.7 \approx 36:1", r"\qquad", r"60.5:39.5 \approx 1.5:1",
                   size=SIZE_SMALL, color=INK_3).next_to(ex1, DOWN, buff=0.35)
        self.play_at("odds-example", Write(ex1[:2]), FadeIn(src[0]), run_time=1.0)
        self.at(self.word("becomes") + 0.1, Write(ex1[2:]), FadeIn(src[2]), run_time=1.3)

        # ============================================================ beat: flat
        fax = Axes(x_range=[-7, 7, 1], y_range=[0, 0.2, 0.1], x_length=4.8, y_length=2.6,
                   axis_config={"color": RULE, "stroke_width": 2, "include_ticks": False},
                   tips=False).move_to([4.15, 0.35, 0])
        fy = VGroup(*[mtex(s, size=SIZE_SMALL, color=INK_3).next_to(fax.c2p(-7, v), LEFT, buff=0.12)
                      for s, v in (("0", 0.0), ("0.1", 0.1), ("0.2", 0.2))])
        fyl = mtex("c", size=SIZE_BODY, color=GILT_TEXT).next_to(fax.c2p(-7, 0.2), UP, buff=0.15)
        fxn = VGroup(*[mtex(s, size=SIZE_SMALL, color=INK_3).next_to(fax.c2p(v, 0), DOWN, buff=0.1)
                       for s, v in (("-6", -6), ("0", 0), ("+6", 6))])
        fxl = tex(r"field $G$ ($\times 10^{-4}$)", size=SIZE_LABEL, color=BIAS).next_to(fxn, DOWN, buff=0.12)
        dots = VGroup(*[Dot(fax.c2p(g * 1e4, c), radius=0.07, color=GILT) for g, c in zip(G_FIELDS, C_FIELDS)])
        fline = Line(fax.c2p(-7, C_CW), fax.c2p(7, C_CW), stroke_color=GILT, stroke_width=2, stroke_opacity=0.7)
        zline = Line(fax.c2p(0, 0), fax.c2p(0, 0.2), stroke_color=RULE, stroke_width=1.5)
        var = mtex(r"\text{8 fields:}\ ", "c", r"\ \text{varies by} < 0.3\%", size=SIZE_LABEL, color=INK_2,
                   colors={"c": GILT_TEXT}).next_to(fax, UP, buff=0.45)
        self.play_at("flat", FadeOut(VGroup(power, ex1, src)), run_time=0.5)
        self.play(Create(fax), FadeIn(fy), FadeIn(fyl), FadeIn(fxn), FadeIn(fxl), FadeIn(zline),
                  Create(fline),
                  LaggedStart(*[FadeIn(d, scale=0.4) for d in dots], lag_ratio=0.12), run_time=1.6)
        self.at(self.word("eight") - 0.2, FadeIn(var, shift=UP * 0.1), run_time=0.8)

        # answer: the dial's needle, its c = 0.120 and the flat line pulse together
        self.play_at("answer", Indicate(value, color=GILT_TEXT, scale_factor=1.2),
                     ShowPassingFlash(needle.copy().set_stroke(GILT, width=12), time_width=0.7),
                     ShowPassingFlash(fline.copy().set_stroke(GILT, width=8, opacity=1), time_width=0.7),
                     *[Indicate(d, color=GILT, scale_factor=1.6) for d in dots], run_time=1.4)
        self.at(self.word("eighth") - 0.1, Indicate(kept, color=GILT_TEXT, scale_factor=1.06),
                Indicate(value, color=GILT_TEXT, scale_factor=1.2), run_time=1.2)
