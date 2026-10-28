"""Scene "reheat" — Where the runs start (script.md, scene: reheat; page chapter: reheat).

Beats: protocol -> controls -> mnist. The test's runs start from a deliberate imbalance: brief
cooling, removal of the samples labelled (1,+), reheating at beta = 0.7, then the paper's schedule.
Opposite-field runs mirror each other; that is not proof of fresh starts. A trained Boltzmann
machine shows reselection after reheating (an illustration). The final cooling sets the proportions.

Real numbers and shapes:
  * the protocol line: two sweeps at each of the ten levels 0.6-1.6, 30 sweeps at beta = 0.7, then
    the schedule 15, 30, 59, 107, 355, 145, 65, 31, 16, 11 = 834 sweeps (../data/schedule.json;
    paper App. A-B, Table 3); within each segment the width is proportional to the sweeps;
  * the dot clouds: 100 dots = 100% of the runs; the blue (1,+) dots are the measured share removed
    after the brief cooling, 32,467 of 40,000 kept at G = +0.0006125 (19 of 100 removed) and 34,178
    of 40,000 at G = -0.0006125 (15 of 100) (results/hopfield/field_grid.json, n_miscommitted); which
    dots are blue is a seeded draw;
  * the leftover-imbalance curve, on its own small axes (imbalance 1 -> 0 against the 30 reheating
    sweeps): exp(-s / tau_full), tau_full = 3.327 sweeps = 1 / gap_full(N = 2,048, beta = 0.7)
    (src.gap); exp(-30 / tau) = 1.2e-4 (paper App. B), Curie-Weiss only; tau is not named on screen;
  * the inset run after the removal sits in the negative valley (paper App. A: the removal keeps
    k < N/2), then rolls to the centre at beta = 0.7;
  * 61% and c = 0.120: the exact count calculation through these same steps (../data/c_cw.json:
    q_chain = 0.60546, c_CW = 0.119808; paper Sec. 4);
  * the mirrored bars: the strongest field pair of Table 4 (paper App. B): 7,417 / 11,085 = 66.9%
    positive at +G, 3,886 / 11,801 = 32.9% positive at -G; blue = positive in both bars; the -G bar is
    reflected left-right, so the two blue parts laid end to end fill one bar exactly when the sum is
    100%; the sum 99.8% with the paper's paired-sum interval [0.98620, 1.01058] (paper App. B); the
    removal takes (1,+) runs at both field signs, so a memory would lower both blue parts and push the
    sum below 100% (a direction only, no size drawn);
  * the closing line at beta = 1: the paper says the sign is chosen near beta = 1 (Sec. 3); no width;
  * the landscape is -log p(m) per spin of the large-N Curie-Weiss model without a field (the same
    curve as the commit scene), an illustration;
  * the digit tiles are chain 13 of ../assets/mnist_illustration_frames.png (docs/mnist-illustration.md runs):
    sweeps 250 and 300 at beta = 2, 330, 360 and 390 while reheating at beta = 0.55, then 440, 500
    and 700 while cooling with the bias toward 3. Illustration only: no tallies.

Render:  ./build.sh render -l reheat
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import *  # noqa: E402,F403

VISUAL = Path(__file__).resolve().parents[2]
DATA = VISUAL / "data"
# a private TeX cache: scenes render in parallel into media/, and a shared cache can race
config.tex_dir = str(Path(__file__).resolve().parents[1] / "media" / "Tex_reheat")

SCHED = json.loads((DATA / "schedule.json").read_text())   # ladder, K, reheat (src/hopfield/protocol.py)
CCW = json.loads((DATA / "c_cw.json").read_text())         # exact count calculation of the protocol
LADDER = SCHED["ladder"]                       # 0.6 ... 1.6, ten levels
K = SCHED["K"]                  # 15 ... 11, sum 834
K_BRIEF = CCW["scarce_sweeps_per_level"]     # 2
B_REHEAT, K_REHEAT = SCHED["beta_reheat"], SCHED["reheat_sweeps"]   # 0.7, 30
assert sum(K) == 834 and K_BRIEF == 2 and (B_REHEAT, K_REHEAT) == (0.7, 30)
assert round(CCW["c_cw"], 3) == 0.120 and round(100 * CCW["runs"][-1]["q_chain"]) == 61

TAU_FULL = 3.3271926395265954                  # 1 / gap_full(2048, 0.7), src.gap
assert abs(np.exp(-30 / TAU_FULL) - 1.2e-4) < 0.05e-4

# share of runs labelled (1,+) after the brief cooling (results/hopfield/field_grid.json)
REMOVED_POS = round(100 * (40000 - 32467) / 40000)    # 19 at G = +0.0006125
REMOVED_NEG = round(100 * (40000 - 34178) / 40000)    # 15 at G = -0.0006125
assert (REMOVED_POS, REMOVED_NEG) == (19, 15)

# strongest field pair, paper Table 4
Q_POS = 7417 / 11085                           # 0.6691
Q_NEG = 3886 / 11801                           # 0.3293
SUM_LO, SUM_HI = 0.98620, 1.01058
assert round(Q_POS, 4) == 0.6691 and round(Q_NEG, 4) == 0.3293 and SUM_LO < Q_POS + Q_NEG < SUM_HI

DBM_CHAIN = 13
DBM_SNAPS = [50, 60, 66, 72, 78, 88, 100, 140]  # sweeps 250, 300 | 330, 360, 390 | 440, 500, 700
SEED = 1983


# ------------------------------------------------------------------ models
def mstar(beta):
    """Positive root of m = tanh(beta m) (0 for beta <= 1)."""
    if beta <= 1:
        return 0.0
    m = 0.9
    for _ in range(200):
        m = np.tanh(beta * m)
    return float(m)


def landscape(m, beta):
    """-log p(m) per spin, large-N Curie-Weiss, no field, shifted so its minimum is 0."""
    def f(x):
        x = np.clip(x, -0.999, 0.999)
        ent = (1 + x) / 2 * np.log((1 + x) / 2) + (1 - x) / 2 * np.log((1 - x) / 2)
        return -beta * x ** 2 / 2 + ent + np.log(2)
    return float(f(m) - f(mstar(beta)))


# protocol plot: piecewise sweep axis (brief + reheat on one scale, the schedule compressed)
X0, PER_SWEEP_A, BREAK, PER_SWEEP_B = -6.0, 0.08, 0.25, 0.009
S_BRIEF = K_BRIEF * len(LADDER)                # 20
S_PRE = S_BRIEF + K_REHEAT                     # 50


def X(s):
    if s <= S_PRE:
        return X0 + PER_SWEEP_A * s
    return X0 + PER_SWEEP_A * S_PRE + BREAK + PER_SWEEP_B * (s - S_PRE)


def Y(b):
    return -3.05 + (b - 0.5) * 3.0


def P(s, b):
    return np.array([X(s), Y(b), 0.0])


def brief_pts():
    pts, s = [P(0, LADDER[0])], 0
    for i, b in enumerate(LADDER):
        if i:
            pts.append(P(s, b))
        s += K_BRIEF
        pts.append(P(s, b))
    return pts


def reheat_pts():
    return [P(S_BRIEF, LADDER[-1]), P(S_BRIEF, B_REHEAT), P(S_PRE, B_REHEAT)]


def sched_pts():
    s = S_PRE
    pts = [P(S_PRE, B_REHEAT), P(S_PRE + 1e-6, B_REHEAT), P(S_PRE + 1e-6, LADDER[0])]
    for i, b in enumerate(LADDER):
        if i:
            pts.append(P(s, b))
        s += K[i]
        pts.append(P(s, b))
    return pts


def poly(pts, color=INK, width=4):
    return VMobject(stroke_color=color, stroke_width=width).set_points_as_corners(pts)


# ------------------------------------------------------------------ small parts
def cloud(center, n_blue_idx, spacing=0.19, seed=SEED):
    """100 dots on a 10 x 10 grid (small seeded jitter); returns (group, blue dots, rest)."""
    rng = np.random.default_rng(seed)
    dots = VGroup()
    for k in range(100):
        r, c = divmod(k, 10)
        j = rng.uniform(-0.03, 0.03, 2)
        d = Dot(radius=0.052, color=INK_3)
        d.move_to(np.array(center) + np.array([(c - 4.5) * spacing + j[0], (4.5 - r) * spacing + j[1], 0]))
        dots.add(d)
    blue = VGroup(*[dots[k] for k in n_blue_idx])
    rest = VGroup(*[dots[k] for k in range(100) if k not in set(n_blue_idx)])
    return dots, blue, rest


def field_arrow(start, direction, length=1.0, label_side=RIGHT):
    a = Arrow(start, np.array(start) + direction * length, buff=0, color=BIAS, stroke_width=5,
              max_tip_length_to_length_ratio=0.28)
    sign = "+" if direction[0] > 0 else "-"
    g = mtex(sign + "G", size=SIZE_LABEL, color=BIAS).next_to(a, label_side, buff=0.12)
    return VGroup(a, g)


def tile(arr, size, frame):
    """A digit tile: lit pixels as ink squares on the ground, framed in the given colour."""
    n = arr.shape[0]
    px = size / n
    sq = VGroup()
    for r in range(n):
        for c in range(n):
            if arr[r, c]:
                sq.add(Square(px, stroke_width=0, fill_color=INK, fill_opacity=1)
                       .move_to([(c - n / 2 + 0.5) * px, (n / 2 - r - 0.5) * px, 0]))
    box = Square(size + 0.08, stroke_color=frame, stroke_width=3)
    return VGroup(box, sq)


def dbm_frames():
    from PIL import Image
    im = np.asarray(Image.open(VISUAL / "assets" / "mnist_illustration_frames.png").convert("L")) > 127
    r0 = DBM_CHAIN * 28
    return [im[r0:r0 + 28, s * 28:(s + 1) * 28] for s in DBM_SNAPS]


# ------------------------------------------------------------------ scene
class Reheat(NarratedScene):
    scene_id = "reheat"

    def word(self, w, nth=0):
        hits = [x["start"] for x in self.timing["words"] if x["w"].strip(".,;:'").lower() == w]
        return hits[nth]

    def at(self, t, *anims, **kw):
        """Start secondary animations on a word time (not a cue)."""
        self._wait_to(t)
        if anims:
            self.play(*anims, **kw)

    def till(self, t, *anims, **kw):
        """Play from now so the animations end at time t."""
        rt = round((t - self.now) * config.frame_rate) / config.frame_rate
        self.play(*anims, run_time=max(rt, 2 / config.frame_rate) - 1e-6, **kw)

    # the landscape with a run on it; beta, the run's m and a ghost's m are trackers
    def make_landscape(self, center, width, height):
        ax = Axes(x_range=[-1, 1, 1], y_range=[0, 0.3, 0.1], x_length=width, y_length=height,
                  axis_config={"color": RULE, "stroke_width": 2, "include_ticks": False},
                  tips=False).move_to(center)
        ax.y_axis.set_opacity(0)
        beta = ValueTracker(1.6)
        mv = ValueTracker(-mstar(1.6))
        gv = ValueTracker(-mstar(1.6))
        ghost_on = ValueTracker(0.0)
        dot_on = ValueTracker(1.0)

        def curve():
            b = beta.get_value()
            ms = np.linspace(-0.95, 0.95, 161)
            pts = [ax.c2p(m, min(landscape(m, b), 0.3)) for m in ms]
            return VMobject(stroke_color=INK_2, stroke_width=3 * ax.width / 5).set_points_smoothly(pts)

        def side_color(m):
            if abs(m) < 0.08:
                return INK
            return interpolate_color(ManimColor(INK), ManimColor(MODE_A if m > 0 else MODE_B),
                                     min(1.0, (abs(m) - 0.08) / 0.2))

        def dot_at(tr, opacity):
            m, b = tr.get_value(), beta.get_value()
            r = 0.12 * ax.width / 5
            d = Dot(ax.c2p(m, landscape(m, b)), radius=r, color=side_color(m))
            d.shift(UP * r * 1.05).set_opacity(opacity)
            return d

        crv = always_redraw(curve)
        ghost = always_redraw(lambda: dot_at(gv, 0.35 * ghost_on.get_value()))
        dot = always_redraw(lambda: dot_at(mv, dot_on.get_value()))
        mlab = mtex("m", size=SIZE_SMALL, color=INK_3)
        mlab.add_updater(lambda o: o.next_to(ax.x_axis, RIGHT, buff=0.1))
        group = VGroup(ax, crv, ghost, dot, mlab)
        return group, ax, beta, mv, gv, ghost_on, dot_on

    def construct(self):
        # ============================================================ beat: protocol
        # --- 0 s "leave": a run sits in one mode; reheating lets it out, cooling lets it choose again
        land, lax, beta, mv, gv, ghost_on, dot_on = self.make_landscape([0, 0.35, 0], 6.4, 3.4)
        bread = VGroup(mtex(r"\beta =", size=SIZE_BODY, color=INK_2),
                       DecimalNumber(1.6, num_decimal_places=1, font_size=SIZE_BODY, color=INK_2))
        bread[1].add_updater(lambda d: d.set_value(beta.get_value()))
        bread[1].add_updater(lambda d: d.next_to(bread[0], RIGHT, buff=0.12))
        bread.arrange(RIGHT, buff=0.12).move_to([0, 2.75, 0])
        illus = label("illustration · no field", size=SIZE_SMALL).move_to([0, -1.75, 0])
        self.play_at("leave", Create(lax), Create(land[1]), FadeIn(land[3]), FadeIn(land[4]),
                     FadeIn(bread), FadeIn(illus), run_time=1.0)
        self.add(land[2])

        mv.add_updater(lambda t: t.set_value(-mstar(beta.get_value())))
        gv.add_updater(lambda t: t.set_value(-mstar(beta.get_value())))
        self.add(mv, gv)
        # "raising the temperature again": beta down to 0.7, one valley, the run is free
        self.at(self.word("raising"))
        self.till(self.word("again") + 0.3, beta.animate.set_value(0.7), rate_func=smooth)
        mv.clear_updaters()
        gv.clear_updaters()
        mv.add_updater(lambda t: t.set_value(mstar(beta.get_value())))
        gv.add_updater(lambda t: t.set_value(-mstar(beta.get_value())))
        # "can leave its mode": cooling again, the run falls into a valley; the ghost marks the other
        self.at(self.word("can"), ghost_on.animate.set_value(1.0), run_time=0.3)
        self.till(self.word("mode") + 0.45, beta.animate.set_value(1.6), rate_func=smooth)

        # --- 7.3 s "imbalance": the landscape becomes an inset; the protocol plot and the runs
        y_axis = Line([X0 - 0.25, Y(0.5), 0], [X0 - 0.25, Y(1.72), 0], color=RULE, stroke_width=2)
        x_axis = VGroup(Line([X0 - 0.25, Y(0.5), 0], [X(S_PRE) + 0.08, Y(0.5), 0], color=RULE, stroke_width=2),
                        Line([X(S_PRE) + BREAK - 0.08, Y(0.5), 0], [X(884) + 0.2, Y(0.5), 0], color=RULE,
                             stroke_width=2))
        brk = VGroup(*[Line([0, -0.1, 0], [0.08, 0.1, 0], color=RULE, stroke_width=2)
                       .move_to([X(S_PRE) + 0.07 + 0.11 * k, Y(0.5), 0]) for k in range(2)])
        yt = VGroup(*[mtex(s, size=SIZE_SMALL, color=INK_3).next_to([X0 - 0.25, Y(v), 0], LEFT, buff=0.1)
                      for s, v in (("0.7", 0.7), ("1", 1.0), ("1.6", 1.6))])
        ytk = VGroup(*[Line([X0 - 0.31, Y(v), 0], [X0 - 0.19, Y(v), 0], color=RULE, stroke_width=2)
                       for v in (0.7, 1.0, 1.6)])
        ylab = mtex(r"\beta", size=SIZE_LABEL, color=INK_2).next_to(y_axis, UP, buff=0.1)
        axes_grp = VGroup(y_axis, x_axis, brk, yt, ytk, ylab)

        seg_names = VGroup(
            text("brief cooling", size=SIZE_SMALL, color=INK_2).move_to([(X(0) + X(S_BRIEF)) / 2, -3.33, 0]),
            text("reheating", size=SIZE_SMALL, color=INK_2).move_to([(X(S_BRIEF) + X(S_PRE)) / 2, -3.33, 0]),
            text("the paper's schedule", size=SIZE_SMALL, color=INK_2).move_to([(X(S_PRE) + X(884)) / 2, -3.33, 0]),
        )
        seg_tags = VGroup(
            # the brief-cooling tag is wider than its segment: right edge kept clear of "30 sweeps"
            label("2 sweeps per temperature", size=17).move_to([0, -3.68, 0], aligned_edge=RIGHT)
            .set_x(X(S_BRIEF) + 0.75, direction=RIGHT),
            label("30 sweeps", size=19).move_to([(X(S_BRIEF) + X(S_PRE)) / 2 + 0.37, -3.68, 0]),
            label("834 sweeps · time axis compressed after the break", size=19)
            .move_to([(X(S_PRE) + X(884)) / 2, -3.68, 0]),
        )

        rng = np.random.default_rng(SEED)
        idx_pos = sorted(rng.choice(100, REMOVED_POS, replace=False).tolist())
        idx_neg = sorted(rng.choice(100, REMOVED_NEG, replace=False).tolist())
        cl_center = [(X(0) + X(S_BRIEF)) / 2, 2.25, 0]
        dots, blue, rest = cloud(cl_center, idx_pos)
        cl_tag = label("each dot: 1% of runs", size=19).next_to(dots, RIGHT, buff=0.35).align_to(dots, DOWN)
        farrow = field_arrow([cl_center[0] - 0.7, 3.62, 0], RIGHT, length=0.9)
        hop = label("Hopfield runs", size=SIZE_SMALL, color=INK_2).next_to(dots, RIGHT, buff=0.35).align_to(dots, UP)

        inset_center = [4.75, 2.55, 0]
        self.play_at("imbalance",
                     lax.animate.scale(0.47).move_to(inset_center),
                     FadeOut(bread), illus.animate.scale(0.8).move_to([inset_center[0], 1.55, 0]),
                     ghost_on.animate.set_value(0.0),
                     FadeIn(axes_grp), run_time=1.2)
        self.at(self.word("hopfield"), LaggedStart(*[FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.01),
                FadeIn(hop), FadeIn(cl_tag), GrowArrow(farrow[0]), FadeIn(farrow[1]), run_time=1.2)

        # --- 13.1 s "brief": two sweeps at each of the ten levels
        brief = poly(brief_pts())
        self.play_at("brief", Create(brief), FadeIn(seg_names[0]), FadeIn(seg_tags[0]), run_time=1.45,
                     rate_func=linear)
        # --- 14.7 s "remove": the runs labelled (1,+) light up, then go
        lab_pos = mtex("(1,+)", size=SIZE_LABEL, color=MODE_A).next_to(dots, RIGHT, buff=0.35).align_to(dots, UP)
        cut = Line(P(S_BRIEF, 1.6) + UP * 0.12, [X(S_BRIEF), dots.get_bottom()[1] - 0.08, 0], color=INK_2, stroke_width=2)
        self.play_at("remove", blue.animate.set_color(MODE_A), FadeIn(lab_pos), FadeOut(hop),
                     Create(cut), run_time=0.8)
        # the one label that shows the deliberate imbalance: under the inset tag, next to the orange ball;
        # it stays until the ball rolls to the centre at "one valley"
        only_neg = tex(r"pattern 1: only $-$ left", size=SIZE_SMALL, color=MODE_B) \
            .next_to(illus, DOWN, buff=0.12)
        self.at(self.word("removed"), *[d.animate.scale(0.1).set_opacity(0) for d in blue],
                lab_pos.animate.set_opacity(0.35), dot_on.animate.set_value(0.0), run_time=0.6)
        self.remove(blue)
        # the kept run sits in the negative valley (paper App. A: removal keeps k < N/2)
        mv.clear_updaters()
        mv.add_updater(lambda t: t.set_value(-mstar(beta.get_value())))
        self.play(dot_on.animate.set_value(1.0), FadeIn(only_neg), run_time=0.5)

        # --- 19.6 s "reheat": a drop to beta = 0.7, held
        reheat = poly(reheat_pts())
        self.play_at("reheat", Create(reheat), FadeIn(seg_names[1]), FadeIn(seg_tags[1]),
                     FadeOut(lab_pos), FadeOut(cut), run_time=1.5)
        # "one valley": the inset follows beta down to 0.7; the kept run rolls to the centre (white)
        self.at(self.word("one", 2) - 0.25, beta.animate.set_value(0.7), FadeOut(only_neg), run_time=0.9)
        # the recooling chooses a side again: the run lands in one valley, the ghost marks the other
        mv.clear_updaters()
        mv.add_updater(lambda t: t.set_value(mstar(beta.get_value())))

        # --- 25.0 s "recool": the schedule, widths proportional to its sweeps
        sched = poly(sched_pts())
        self.play_at("recool", Create(sched), FadeIn(seg_names[2]), FadeIn(seg_tags[2]),
                     beta.animate.set_value(1.6), ghost_on.animate.set_value(1.0), run_time=2.3, rate_func=linear)

        # --- 28.2 s "same": the Curie-Weiss numbers came through these same steps
        end = P(S_PRE + sum(K), LADDER[-1])
        cw61 = tex(r"$61\%$ positive", size=SIZE_LABEL, color=MODE_A)
        cwc = mtex("c", "= 0.120", size=SIZE_LABEL, colors={"c": GILT_TEXT})
        cwc[1].set_color(GILT_TEXT)
        VGroup(cw61, cwc).arrange(RIGHT, buff=0.45).next_to(end, UP, buff=0.3).align_to(end + RIGHT * 0.15, RIGHT)
        cw_tag = label("Curie–Weiss, exact, same steps", size=19).next_to(VGroup(cw61, cwc), UP, buff=0.15) \
            .align_to(cwc, RIGHT)
        cw_tick = Line(end + UP * 0.08, end + UP * 0.24, color=INK_3, stroke_width=2)
        self.play_at("same", FadeIn(cw_tag), Create(cw_tick), FadeOut(cl_tag), run_time=0.6)
        self.at(self.word("sixty-one"), Write(cw61), run_time=0.7)
        self.at(self.word("c"), Write(cwc), run_time=0.7)

        # --- 34.4 s "memory": almost none of the imbalance survives reheating (Curie-Weiss estimate)
        # its own axes (imbalance, 1 -> 0, against the 30 reheating sweeps), not the beta axis
        dx0, dx1, dy0, dy1 = -3.95, -2.0, -1.3, -0.2  # "0" sits between the beta ticks, not level with beta = 1
        DXs = lambda s: dx0 + s / K_REHEAT * (dx1 - dx0)  # noqa: E731
        DYv = lambda v: dy0 + v * (dy1 - dy0)  # noqa: E731
        d_axes = VGroup(Line([dx0, dy0, 0], [dx1 + 0.05, dy0, 0], color=RULE, stroke_width=2),
                        Line([dx0, dy0, 0], [dx0, dy1 + 0.05, 0], color=RULE, stroke_width=2))
        d_ticks = VGroup(label("1", size=17).next_to([dx0, DYv(1), 0], LEFT, buff=0.08),
                         label("0", size=17).next_to([dx0, DYv(0), 0], LEFT, buff=0.08),
                         label("30 sweeps", size=17).next_to([dx1, dy0, 0], DOWN, buff=0.08))
        ss = np.linspace(0, K_REHEAT, 121)
        dec = theory(VMobject().set_points_smoothly(
            [[DXs(s), DYv(np.exp(-s / TAU_FULL)), 0] for s in ss]), num_dashes=24)
        dec.set_stroke(width=3)
        dec_tag = label("leftover imbalance · Curie–Weiss estimate", size=19).next_to(d_axes, UP, buff=0.12) \
            .align_to(d_axes, LEFT)
        dec_grp = VGroup(d_axes, d_ticks, dec_tag)
        mem = VGroup(tex(r"after 30 sweeps: about $1.2\times10^{-4}$ of it left", size=21, color=INK_3),
                     tex(r"Curie–Weiss only · not a bound for Hopfield", size=21, color=INK_3)) \
            .arrange(DOWN, buff=0.1, aligned_edge=LEFT).move_to([-1.7, -0.7, 0], aligned_edge=LEFT)
        self.play_at("memory", FadeIn(dec_grp), run_time=0.6)
        self.play(Create(dec), run_time=1.6, rate_func=linear)
        self.at(self.word("survives"), FadeIn(mem), run_time=0.8)

        # ============================================================ beat: controls
        # --- 39.5 s "controls": the same removal at both field signs
        top_c, bot_c = np.array([-5.0, 1.55, 0]), np.array([-5.0, -1.95, 0])
        dots_p, blue_p, rest_p = cloud(top_c, idx_pos)
        dots_n, blue_n, rest_n = cloud(bot_c, idx_neg, seed=SEED + 2)
        fa_p = field_arrow(top_c + np.array([-0.45, 1.3, 0]), RIGHT, length=0.9)
        fa_n = field_arrow(bot_c + np.array([0.45, 1.3, 0]), LEFT, length=0.9, label_side=LEFT)
        protocol_out = VGroup(axes_grp, seg_names, seg_tags, brief, reheat, sched, dec, dec_grp,
                              cw_tag, cw61, cwc, cw_tick, mem)
        self.play_at("controls", FadeOut(protocol_out), FadeOut(lax), FadeOut(land[1]), FadeOut(land[3]),
                     FadeOut(land[4]), FadeOut(illus), FadeOut(rest), ReplacementTransform(farrow, fa_p),
                     FadeIn(dots_p), FadeIn(dots_n), FadeIn(fa_n), run_time=1.3)
        self.remove(land[2])
        lab_p = mtex("(1,+)", size=SIZE_LABEL, color=MODE_A).next_to(dots_p, RIGHT, buff=0.3)
        lab_n = mtex("(1,+)", size=SIZE_LABEL, color=MODE_A).next_to(dots_n, RIGHT, buff=0.3)
        self.at(self.word("directions"), blue_p.animate.set_color(MODE_A), blue_n.animate.set_color(MODE_A),
                FadeIn(lab_p), FadeIn(lab_n), run_time=0.8)
        self.at(self.word("removal"), *[d.animate.scale(0.1).set_opacity(0) for d in [*blue_p, *blue_n]],
                FadeOut(lab_p), FadeOut(lab_n), run_time=0.6)
        self.remove(blue_p, blue_n)

        # --- 44.9 s "mirror": after cooling, the share of each sign among samples labelled pattern 1
        L, H = 5.8, 0.5
        x_l = -1.6

        def bar(q_pos, y):
            a = Rectangle(width=L * q_pos, height=H, stroke_width=0, fill_color=MODE_A, fill_opacity=0.9)
            b = Rectangle(width=L * (1 - q_pos), height=H, stroke_width=0, fill_color=MODE_B, fill_opacity=0.9)
            a.move_to([x_l + L * q_pos / 2, y, 0])
            b.move_to([x_l + L * q_pos + L * (1 - q_pos) / 2, y, 0])
            return VGroup(a, b)

        bar_p, bar_n = bar(Q_POS, top_c[1]), bar(Q_NEG, bot_c[1])
        pct = lambda s, col: tex(s, size=SIZE_LABEL, color=col)  # noqa: E731
        num_p = pct(r"$66.9\%$", BG).move_to(bar_p[0])
        num_n = pct(r"$32.9\%$", BG).move_to(bar_n[0])
        # no label on the orange part of the -G bar: its 67.1% would collide with the test scene's
        # predicted 67.1% at +G (a different quantity); the sum line below uses 66.9% and 32.9% only
        head = text("share of each sign, samples labelled pattern 1", size=SIZE_SMALL, color=INK_2) \
            .move_to([x_l + L / 2, 3.25, 0])
        arr_p = Arrow(dots_p.get_right(), [x_l - 0.1, top_c[1], 0], buff=0.25, color=RULE, stroke_width=3,
                      max_tip_length_to_length_ratio=0.15)
        arr_n = Arrow(dots_n.get_right(), [x_l - 0.1, bot_c[1], 0], buff=0.25, color=RULE, stroke_width=3,
                      max_tip_length_to_length_ratio=0.15)
        after = VGroup(*[label("after reheating", size=17).next_to(a, UP, buff=0.06).shift(DOWN * 0.02)
                         for a in (arr_p, arr_n)],
                       *[label("+ schedule", size=17).next_to(a, DOWN, buff=0.06) for a in (arr_p, arr_n)])
        self.play_at("mirror", GrowArrow(arr_p), GrowArrow(arr_n), FadeIn(after), FadeIn(head),
                     GrowFromEdge(bar_p, LEFT), GrowFromEdge(bar_n, LEFT),
                     FadeIn(VGroup(num_p, num_n), rate_func=lambda t: smooth(max(0, 2 * t - 1))),
                     run_time=1.2)
        # "if the removal left a memory": both fields lost the same positive runs, so a memory
        # would shrink BOTH blue parts (a direction, no size claimed)
        shrink = VGroup(*[Arrow([x_l + L * q - 0.02, y + H / 2 + 0.2, 0], [x_l + L * q - 0.75, y + H / 2 + 0.2, 0],
                                buff=0, color=INK_3, stroke_width=3, max_tip_length_to_length_ratio=0.3)
                          for q, y in ((Q_POS, top_c[1]), (Q_NEG, bot_c[1]))])
        shrink_lab = text("a memory would shrink both blue parts", size=SIZE_SMALL, color=INK_3) \
            .move_to([x_l + L / 2, (top_c[1] + bot_c[1]) / 2 + 0.25, 0])
        self.at(self.word("memory"), GrowArrow(shrink[0]), GrowArrow(shrink[1]), FadeIn(shrink_lab), run_time=0.8)
        # "mirror each other": reflect the -G bar left-right under the +G bar; colours keep their meaning
        flip_y = top_c[1] - H - 0.12
        neg_or = Rectangle(width=L * (1 - Q_NEG), height=H, stroke_width=0, fill_color=MODE_B, fill_opacity=0.9) \
            .move_to([x_l + L * (1 - Q_NEG) / 2, flip_y, 0])
        neg_bl = Rectangle(width=L * Q_NEG, height=H, stroke_width=0, fill_color=MODE_A, fill_opacity=0.9) \
            .move_to([x_l + L * (1 - Q_NEG) + L * Q_NEG / 2, flip_y, 0])
        mir_lab = tex(r"$-G$, mirrored", size=SIZE_SMALL, color=INK_3).next_to(neg_bl, RIGHT, buff=0.2)
        mir_lab[0][1].set_color(BIAS)
        self.at(self.word("mirror") - 0.05,
                Transform(bar_n[0], neg_bl), Transform(bar_n[1], neg_or),
                num_n.animate.move_to(neg_bl),
                FadeIn(mir_lab), FadeOut(arr_n), FadeOut(dots_n), FadeOut(fa_n), FadeOut(after[1]), FadeOut(after[3]),
                FadeOut(shrink), FadeOut(shrink_lab), run_time=1.0)
        guide = DashedLine([x_l + L * Q_POS, top_c[1] + H / 2 + 0.12, 0], [x_l + L * Q_POS, flip_y - H / 2 - 0.12, 0],
                           color=INK_2, stroke_width=2, dash_length=0.05)

        # --- 50.5 s "mirrored": the two blue parts, end to end, fill one bar
        row_y, rh = -0.15, 0.32
        frame = Rectangle(width=L, height=rh, stroke_color=THEORY, stroke_width=2) \
            .move_to([x_l + L / 2, row_y, 0])
        frame = DashedVMobject(frame, num_dashes=60)
        pc = bar_p[0].copy()
        nc = bar_n[0].copy()
        sum_eq = tex(r"$66.9\%$", r"$\,+\,$", r"$32.9\%$", r"$\,=\,99.8\%$", size=SIZE_LABEL, color=INK) \
            .move_to([x_l + L / 2, -0.75, 0])
        sum_eq[0].set_color(MODE_A)
        sum_eq[2].set_color(MODE_A)
        ia_y = -1.85
        lo_v, hi_v = 0.97, 1.03
        IX = lambda v: x_l + 0.4 + (v - lo_v) / (hi_v - lo_v) * (L - 0.8)  # noqa: E731
        iaxis = Line([IX(lo_v), ia_y, 0], [IX(hi_v), ia_y, 0], color=RULE, stroke_width=2)
        iticks = VGroup(*[VGroup(Line([IX(v), ia_y - 0.07, 0], [IX(v), ia_y + 0.07, 0], color=RULE, stroke_width=2),
                                 tex(s, size=19, color=INK_3).move_to([IX(v), ia_y - 0.3, 0]))
                          for v, s in ((0.98, r"$98\%$"), (1.02, r"$102\%$"))])
        one = theory(Line([IX(1.0), ia_y - 0.4, 0], [IX(1.0), ia_y + 0.45, 0]), num_dashes=7)
        one_lab = tex(r"$100\%$: mirror images", size=19, color=THEORY).move_to([IX(1.0), ia_y - 0.62, 0])
        mem_arr = Arrow([IX(0.996), ia_y + 0.3, 0], [IX(0.974), ia_y + 0.3, 0], buff=0, color=INK_3,
                        stroke_width=3, max_tip_length_to_length_ratio=0.2)
        mem_lab = tex(r"memory $\to$ below $100\%$", size=19, color=INK_3).next_to(mem_arr, UP, buff=0.08)
        ibar = Line([IX(SUM_LO), ia_y, 0], [IX(SUM_HI), ia_y, 0], color=INK, stroke_width=6)
        ipt = Dot([IX(Q_POS + Q_NEG), ia_y, 0], radius=0.07, color=INK)
        itag = label("strongest field pair, 95% interval · all four pairs contain 100%", size=19) \
            .move_to([x_l + L / 2, -3.0, 0])
        self.play_at("mirrored", Create(guide), FadeIn(frame),
                     pc.animate.stretch_to_fit_height(rh).move_to([x_l + L * Q_POS / 2, row_y, 0]),
                     nc.animate.stretch_to_fit_height(rh).move_to([x_l + L * Q_POS + L * Q_NEG / 2, row_y, 0]),
                     run_time=0.6)
        join = Line([x_l + L * Q_POS, row_y - rh / 2, 0], [x_l + L * Q_POS, row_y + rh / 2, 0], color=BG, stroke_width=3)
        self.add(join)
        self.play(Write(sum_eq), run_time=0.5)
        self.at(self.word("precision"), Create(iaxis), FadeIn(iticks), Create(one), FadeIn(one_lab),
                GrowArrow(mem_arr), FadeIn(mem_lab), run_time=0.8)
        self.at(self.word("they"), GrowFromCenter(ibar), FadeIn(ipt), FadeIn(itag), run_time=0.6)

        # --- 53.7 s "not-proof": symmetry is not equivalence to fresh starts
        nf = tex(r"mirror images $\neq$ fresh starts", size=SIZE_LABEL, color=INK_2) \
            .move_to([x_l + L / 2, -3.55, 0])
        self.play_at("not-proof", Write(nf), run_time=1.1)

        # ============================================================ beat: mnist
        controls = VGroup(dots_p, rest_p, fa_p, arr_p, after[0], after[2], bar_p, bar_n, num_p, num_n,
                          mir_lab, head, guide, frame, pc, nc, join, sum_eq, iaxis, iticks, one, one_lab, mem_arr, mem_lab,
                          ibar, ipt, itag, nf)
        frames = dbm_frames()
        size, gap = 1.12, 0.3
        xs = [(-3.5 + k) * (size + gap) - (0.15 if k < 2 else (0 if k < 5 else -0.15)) for k in range(8)]
        frame_col = [MODE_B, MODE_B, NOISE, NOISE, NOISE, NOISE, MODE_A, MODE_A]
        tiles = [tile(frames[k], size, frame_col[k]).move_to([xs[k], 0.45, 0]) for k in range(8)]
        tiles[5][0].set_stroke(INK_3)

        def phase(k0, k1, s, y=-0.5):
            l, r = tiles[k0].get_left()[0], tiles[k1].get_right()[0]
            br = VGroup(Line([l, y, 0], [r, y, 0], color=RULE, stroke_width=2),
                        Line([l, y, 0], [l, y + 0.1, 0], color=RULE, stroke_width=2),
                        Line([r, y, 0], [r, y + 0.1, 0], color=RULE, stroke_width=2))
            t = tex(s, size=SIZE_SMALL, color=INK_2).next_to(br, DOWN, buff=0.15)
            return VGroup(br, t)

        ph = [phase(0, 1, r"$\beta = 2$"), phase(2, 4, r"reheating, $\beta = 0.55$"),
              phase(5, 7, r"cooling, bias toward 3")]
        mn_tag = label("illustration · deep Boltzmann machine trained on MNIST · one stored run", size=19).move_to([0, 1.75, 0])
        self.play_at("mnist", FadeOut(controls), run_time=0.8)
        self.at(self.now, FadeIn(tiles[0]), FadeIn(tiles[1]), FadeIn(ph[0]), FadeIn(mn_tag), run_time=0.8)
        # "a reheated eight can become a three under a bias toward threes"
        self.play_at("eights", LaggedStart(*[FadeIn(tiles[k]) for k in (2, 3, 4)], lag_ratio=0.35),
                     FadeIn(ph[1]), run_time=1.2)
        self.at(self.word("become"), LaggedStart(*[FadeIn(tiles[k]) for k in (5, 6, 7)], lag_ratio=0.4),
                run_time=1.3)
        self.at(self.word("bias"), FadeIn(ph[2]), run_time=0.6)

        # --- 65.4 s "answer": reheating frees a run; the final cooling chooses again
        mn = VGroup(*tiles, *ph, mn_tag)
        mini = VGroup(axes_grp, seg_names, seg_tags, brief, reheat, sched)
        lax2_center = [4.75, 2.55, 0]
        marker = Dot(P(S_BRIEF, LADDER[-1]), radius=0.09, color=INK)
        beta.set_value(1.6)
        mv.clear_updaters()
        gv.clear_updaters()
        mv.set_value(-mstar(1.6))
        gv.set_value(-mstar(1.6))
        ghost_on.set_value(0.0)
        lax.move_to(lax2_center)
        illus.move_to([lax2_center[0], 1.55, 0])
        # FadeIn suspends updaters: redraw the run (-m*, orange) and the ghost before they fade in
        land[3].update()
        land[2].update()
        self.play_at("answer", FadeOut(mn), FadeIn(mini), FadeIn(lax), FadeIn(land[1]), FadeIn(land[3]),
                     FadeIn(land[4]), FadeIn(illus), FadeIn(marker), run_time=0.9)
        self.add(land[2])
        mv.add_updater(lambda t: t.set_value(-mstar(beta.get_value())))
        gv.add_updater(lambda t: t.set_value(-mstar(beta.get_value())))
        # "free a run": the marker drops to beta = 0.7 and runs through the reheating
        self.at(self.word("free") - 0.3)
        t_then = self.word("then", 1)
        t_mid = self.now + 0.35 * (t_then - self.now)
        self.till(t_mid, marker.animate.move_to(P(S_BRIEF, B_REHEAT)), beta.animate.set_value(0.7), rate_func=smooth)
        self.till(t_then, marker.animate.move_to(P(S_PRE, B_REHEAT)), rate_func=linear)
        mv.clear_updaters()
        gv.clear_updaters()
        mv.add_updater(lambda t: t.set_value(mstar(beta.get_value())))
        gv.add_updater(lambda t: t.set_value(-mstar(beta.get_value())))
        # "the run then cools and chooses again": the marker walks the schedule; the inset follows its beta
        path = VMobject().set_points_as_corners(sched_pts()[1:])
        marker_beta = lambda: (marker.get_center()[1] + 3.05) / 3.0 + 0.5  # noqa: E731
        beta.add_updater(lambda t: t.set_value(min(1.6, max(0.6, marker_beta()))))
        self.at(self.word("then", 1), ghost_on.animate.set_value(1.0), run_time=0.3)
        self.till(self.t("sets") - 0.1, MoveAlongPath(marker, path), rate_func=linear)
        beta.clear_updaters()

        # --- 71.3 s "sets": the band where the sign is chosen, in that final cooling
        # a line at beta = 1, not a band: the paper says the sign is chosen near beta = 1 (Sec. 3)
        band = DashedLine([X(S_PRE), Y(1.0), 0], [X(884), Y(1.0), 0], color=INK_3, stroke_width=2,
                          dash_length=0.08)
        band_lab = tex(r"the sign is chosen near $\beta = 1$", size=SIZE_SMALL, color=INK_2) \
            .move_to([2.4, Y(0.8), 0])
        self.play_at("sets", Create(band), FadeIn(band_lab), run_time=1.0)
        self.bring_to_front(sched, marker)
