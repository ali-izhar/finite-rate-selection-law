"""Scene "commit" — Where the choice is made (script.md, scene: commit; page chapter "commit").

Spins and configurations, the Curie-Weiss model (every spin favours agreeing with every other),
the inverse temperature beta, the magnetization m, the landscape -log p(m) that splits into two
modes past beta = 1, the Gibbs update and the sweep, and the paper's Fig. 1a runs splitting near
beta = 1.

Real numbers and shapes (no field anywhere in this scene):
  * the 64 spins and the eight rows under them: exact Curie-Weiss draws at the needle's beta (the count k of +1 spins is drawn
    from p(k) ∝ C(64, k) exp(beta S^2 / (2 (N-1))), S = 2k - N, then k random spins are set +1);
  * each draw's dot on the m line is its average spin; the curve is -log p(m) of the same
    distribution (continuous in k through lgamma), minus its minimum, cut above 12; "large beta"
    is 1.3, where the two valleys of 64 spins sit inside the frame;
  * the circle: 16 of the 64 spins; a line is bright when its two spins agree (the Curie-Weiss
    weight is exp[beta (agreeing - disagreeing pairs) / (N - 1)] at zero field);
  * the Gibbs updates: a seeded chain on the 64 spins, every update an exact draw from
    p(x_i = +1 | rest) = 1 / (1 + exp(-2 beta / (N-1) sum_{j != i} x_j)); the split bar above a
    spin is that probability; the anneal runs the paper's ten levels (Table 3) with its sweep
    counts on these 64 spins (its own seed, ANNEAL_SEED, chosen so that the run's last sign change
    of m falls at a level of at most 1.05, where the 64-spin barrier is 0.3 nats: from 1.10 on the
    dot stays in one valley, as the "locked" beat says; simulate() asserts it); the levels above
    1 get more film time than the levels up to 1;
  * the fourteen runs: ../data/fork_trajectories.json, the paper's Fig. 1a setting (unexpanded
    drift -m + tanh(beta m), noise sqrt(2/N), N = 1,024, 6 sweeps at each of 360 temperatures
    0.7-1.35, seed 4); a path takes its end colour after its last zero crossing; the grey band
    spans the fourteen last zero crossings (0.968-1.086); dashed branches are m = tanh(beta m);
  * the inset: the large-N per-spin landscape at beta = 1.35 (shape only, no scale).

Render:  ./build.sh render -l commit   (or -h)
"""
import json
import sys
from math import lgamma
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import *  # noqa: E402,F403

VISUAL = Path(__file__).resolve().parents[2]
# a private TeX cache: other scenes render in parallel into media/Tex, and a shared cache can race
config.tex_dir = str(Path(__file__).resolve().parents[1] / "media" / "Tex_commit")

# ------------------------------------------------------------------ data and models
N = 64                      # spins shown (and simulated) in the row
SEED = 1983
ANNEAL_SEED = 62            # seed of the anneal stretch of the chain (see the docstring)
LAST_FLIP_LEVEL = 5         # index in LEVELS of the last level at which m may change sign (1.05)
B_LO, B_HI = 0.6, 1.6       # the paper's schedule runs beta from 0.6 to 1.6
LEVELS = [0.60, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20, 1.40, 1.60]   # paper Table 3
SWEEPS = [15, 30, 59, 107, 355, 145, 65, 31, 16, 11]                    # = 834
F_LO, F_HI = 0.7, 1.35      # Fig. 1a temperature range
B_SHOW = 1.3                # the "large beta" of the landscape: for 64 spins its two valleys sit
                            # inside the frame (at 1.6 they press against m = +-1)
N_DRAWS = 8                 # exact draws stacked under the row

SQ_PLUS = INK_2             # a +1 spin
SQ_MINUS = "#2D2F33"        # a -1 spin
LOGC = np.array([lgamma(N + 1) - lgamma(k + 1) - lgamma(N - k + 1) for k in range(N + 1)])


def p_count(beta):
    """Exact p(k) of the number of +1 spins, Curie-Weiss, N = 64, zero field."""
    s = 2 * np.arange(N + 1) - N
    lw = LOGC + beta * s ** 2 / (2 * (N - 1))
    p = np.exp(lw - lw.max())
    return p / p.sum()


def neglogp(m, beta):
    """-log p(m) for N = 64 (k = N(1+m)/2 continuous), minus its minimum over [-0.97, 0.97]."""
    def f(mm):
        mm = np.atleast_1d(mm)
        k = N * (1 + mm) / 2
        lc = np.array([lgamma(N + 1) - lgamma(x + 1) - lgamma(N - x + 1) for x in k])
        return -(lc + beta * (N * mm) ** 2 / (2 * (N - 1)))
    grid = np.linspace(-0.97, 0.97, 195)
    return f(m) - f(grid).min()


def mstar(beta):
    """Stable magnetization m = tanh(beta m), large N (0 for beta <= 1)."""
    if beta <= 1.0:
        return 0.0
    m = 0.9
    for _ in range(400):
        m = np.tanh(beta * m)
    return float(m)


def big_n_landscape(m, beta):
    """Large-N per-spin -log p(m) (zero field), minus its minimum: shape only."""
    m = np.clip(m, -0.999, 0.999)
    ent = (1 + m) / 2 * np.log((1 + m) / 2) + (1 - m) / 2 * np.log((1 - m) / 2)
    ms = mstar(beta)
    ent0 = (1 + ms) / 2 * np.log((1 + ms) / 2) + (1 - ms) / 2 * np.log((1 - ms) / 2)
    return (-beta * m ** 2 / 2 + ent) - (-beta * ms ** 2 / 2 + ent0)


def exact_draw(rng, beta):
    k = rng.choice(N + 1, p=p_count(beta))
    x = -np.ones(N, dtype=int)
    x[rng.permutation(N)[:k]] = 1
    return x


def p_plus(x, i, beta):
    h = x.sum() - x[i]
    return 1.0 / (1.0 + np.exp(-2.0 * beta * h / (N - 1)))


# ------------------------------------------------------------------ the scene
class Commit(NarratedScene):
    scene_id = "commit"

    # -------------------------------------------------------------- timing helpers
    def word(self, w: str, nth: int = 0) -> float:
        """Start time of the nth spoken word equal to w (punctuation stripped, case-free)."""
        k = 0
        for item in self.timing["words"]:
            if item["w"].strip(".,:;!?").lower() == w.lower():
                if k == nth:
                    return item["start"]
                k += 1
        raise KeyError(w)

    def play_from(self, t, *anims, run_time=1.0, **kw):
        """Start at time t (seconds); not a cue-bound event."""
        self._wait_to(t)
        self.play(*anims, run_time=run_time, **kw)

    def play_window(self, t0, t1, *anims, **kw):
        """Start at t0 and end at t1, both quantised to whole frames."""
        self._wait_to(t0)
        rt = round((t1 - self.now) * config.frame_rate) / config.frame_rate
        self.play(*anims, run_time=max(rt, 2 * self.dt) - 1e-6, **kw)

    # -------------------------------------------------------------- simulation
    def simulate(self):
        rng = np.random.default_rng(SEED)
        self.x_first = exact_draw(rng, B_LO)
        self.x_configs = [exact_draw(rng, B_LO) for _ in range(2)]
        self.x_low = exact_draw(rng, B_LO)
        self.x_high = exact_draw(rng, B_SHOW)
        self.draws_by_beta = {B_LO: [exact_draw(rng, B_LO) for _ in range(N_DRAWS)],
                              "low2": [exact_draw(rng, B_LO) for _ in range(N_DRAWS)],
                              B_SHOW: [exact_draw(rng, B_SHOW) for _ in range(N_DRAWS)]}

        # the chain: an exact start at 0.6, one shown sweep at 0.6, then the paper's ten levels
        x = exact_draw(rng, B_LO)
        self.chain_start = x.copy()
        self.sweep_order = rng.permutation(N)
        self.sweep_states, self.sweep_p = [x.copy()], []
        for i in self.sweep_order:
            p = p_plus(x, i, B_LO)
            self.sweep_p.append(p)
            x[i] = 1 if rng.random() < p else -1
            self.sweep_states.append(x.copy())
        self.anneal = []                    # per level: states after each sweep
        rng_a = np.random.default_rng(ANNEAL_SEED)
        for b, s in zip(LEVELS, SWEEPS):
            states = []
            for _ in range(s):
                for i in rng_a.permutation(N):
                    x[i] = 1 if rng_a.random() < p_plus(x, i, b) else -1
                states.append(x.copy())
            self.anneal.append(states)
        # no barrier crossing on screen after beta = 1.05: from LEVELS[LAST_FLIP_LEVEL + 1] on, m
        # keeps one nonzero sign (the "locked" beat says switching is slow)
        late = np.array([st.mean() for lv in self.anneal[LAST_FLIP_LEVEL + 1:] for st in lv])
        last_lv = self.anneal[LAST_FLIP_LEVEL][-1].mean()
        assert last_lv != 0 and np.all(np.sign(late) == np.sign(last_lv)), "anneal crosses the barrier late"

    # -------------------------------------------------------------- construct
    def construct(self):
        self.simulate()
        state = {"x": self.x_first.copy()}

        # ============================================================ beat: spins
        pitch, side, row_y = 0.17, 0.135, 2.7
        row = VGroup(*[Square(side, stroke_color=RULE, stroke_width=1, fill_opacity=1)
                       .move_to(np.array([(j - (N - 1) / 2) * pitch, row_y, 0])) for j in range(N)])

        def paint(x):
            for j in range(N):
                row[j].set_fill(SQ_PLUS if x[j] > 0 else SQ_MINUS)
        paint(state["x"])
        ROW_DROP = 2.1              # the row opens near the centre of the frame and rises at "couplings"
        row.shift(DOWN * ROW_DROP)
        self.play_at("spins", LaggedStart(*[FadeIn(s, scale=0.5) for s in row], lag_ratio=0.02),
                     run_time=1.6)

        row_label = tex(r"$N$ spins, each $\pm 1$", size=SIZE_LABEL, color=INK_2) \
            .next_to(row, DOWN, buff=0.22).align_to(row, LEFT)
        leg = VGroup(Square(0.17, stroke_color=RULE, stroke_width=1, fill_color=SQ_PLUS, fill_opacity=1),
                     mtex("+1", size=SIZE_SMALL, color=INK_2),
                     Square(0.17, stroke_color=RULE, stroke_width=1, fill_color=SQ_MINUS, fill_opacity=1),
                     mtex("-1", size=SIZE_SMALL, color=INK_2)).arrange(RIGHT, buff=0.12)
        leg[2].shift(RIGHT * 0.2)
        leg[3].shift(RIGHT * 0.2)
        shown = label("64 shown", size=SIZE_SMALL)
        leg_all = VGroup(leg, shown).arrange(RIGHT, buff=0.45).next_to(row, DOWN, buff=0.22).align_to(row, RIGHT)
        self.play_from(self.word("n") - 0.05, Write(row_label), run_time=1.0)
        self.play_from(self.word("plus"), FadeIn(leg_all, shift=UP * 0.1), run_time=0.8)

        # ============================================================ beat: config
        brace = Brace(row, UP, buff=0.12, color=INK_3, sharpness=1.2)
        cfg = label("a configuration", size=SIZE_LABEL, color=INK_2).next_to(brace, UP, buff=0.08)
        self.play_at("config", GrowFromCenter(brace), FadeIn(cfg), run_time=0.8)
        # another setting of the same spins is another configuration
        for t0, xc in zip((self.word("them") - 0.05, self.word("configuration") + 0.25), self.x_configs):
            self.play_from(t0, *[row[j].animate.set_fill(SQ_PLUS if xc[j] > 0 else SQ_MINUS) for j in range(N)],
                           run_time=0.45)
            state["x"] = xc.copy()

        # ============================================================ beat: couplings
        idx = list(range(0, N, 4))
        cc, rr = np.array([0.0, -0.45, 0]), 1.75
        pos = [cc + rr * np.array([np.cos(PI / 2 - TAU * k / 16), np.sin(PI / 2 - TAU * k / 16), 0])
               for k in range(16)]
        nodes = VGroup(*[row[j].copy() for j in idx])
        row_grp = VGroup(row, row_label, leg_all)
        node_t = VGroup(*[nodes[k].copy().scale(2.0).move_to(pos[k]) for k in range(16)])
        pairs = [(a, b) for a in range(16) for b in range(a + 1, 16)]
        lines = VGroup(*[Line(pos[a], pos[b], stroke_color=RULE, stroke_width=1.3) for a, b in pairs])
        cw_name = text("Curie–Weiss model", size=SIZE_BODY, color=INK_2).move_to(np.array([-4.6, -0.2, 0]))
        self.add(nodes)
        self.play_at("couplings", FadeOut(brace), FadeOut(cfg),
                     row_grp.animate.shift(UP * ROW_DROP).set_opacity(0.35),
                     LaggedStart(*[Transform(nodes[k], node_t[k]) for k in range(16)], lag_ratio=0.03),
                     Write(cw_name), run_time=1.2)
        self.bring_to_back(lines)
        self.play(LaggedStart(*[Create(l) for l in lines], lag_ratio=0.006), run_time=1.0)
        circ_x = np.array([state["x"][j] for j in idx])

        def bond_anims(xs):
            return [l.animate.set_stroke(color=INK_2 if xs[a] == xs[b] else RULE,
                                         opacity=0.85 if xs[a] == xs[b] else 0.35,
                                         width=1.8 if xs[a] == xs[b] else 1.1)
                    for l, (a, b) in zip(lines, pairs)]
        # "agreeing": the pairs that agree light up
        self.play_from(self.word("agreeing") - 0.05, *bond_anims(circ_x), run_time=0.8)
        n_agree = lambda xs: sum(xs[a] == xs[b] for a, b in pairs)  # noqa: E731
        cnt = lambda xs: label(f"{n_agree(xs)} of 120 pairs agree", size=SIZE_SMALL, color=INK_2)  # noqa: E731
        cnt0 = cnt(circ_x).move_to(np.array([4.4, -0.2, 0]))
        self.play_from(self.word("other") - 0.2, FadeIn(cnt0), run_time=0.6)

        # ============================================================ beat: agree
        agree_x = np.ones(16, dtype=int)
        agree_x[[3, 11]] = -1
        more = label("more probable", size=SIZE_LABEL, color=INK).move_to(np.array([4.4, -0.75, 0]))
        cnt1 = cnt(agree_x).move_to(cnt0)
        self.play_at("agree",
                     *[nodes[k].animate.set_fill(SQ_PLUS if agree_x[k] > 0 else SQ_MINUS) for k in range(16)],
                     *bond_anims(agree_x), Transform(cnt0, cnt1), run_time=1.1)
        self.play_from(self.word("more") - 0.1, FadeIn(more, shift=UP * 0.1), run_time=0.6)

        # ============================================================ beat: beta
        beta = ValueTracker(B_LO)
        dial = NumberLine(x_range=[B_LO, B_HI, 0.1], length=7, color=RULE, stroke_width=3,
                          include_ticks=False).move_to(DOWN * 3.35)
        dticks = VGroup(*[Line(dial.n2p(v) + DOWN * 0.09, dial.n2p(v) + UP * 0.09, stroke_color=INK_3,
                               stroke_width=2) for v in (0.6, 1.0, 1.6)])
        dnums = VGroup(*[mtex(s, size=SIZE_SMALL, color=INK_3).next_to(dial.n2p(v), DOWN, buff=0.16)
                         for s, v in (("0.6", 0.6), ("1", 1.0), ("1.6", 1.6))])
        dname = mtex(r"\beta", size=SIZE_BODY, color=INK).next_to(dial, LEFT, buff=0.35)
        needle = always_redraw(lambda: VGroup(
            Line(dial.n2p(beta.get_value()) + DOWN * 0.17, dial.n2p(beta.get_value()) + UP * 0.17,
                 stroke_color=INK, stroke_width=4),
            Dot(dial.n2p(beta.get_value()), radius=0.06, color=INK)))

        # eight more exact draws at the needle's beta, stacked under the row
        blk_y0, blk_pitch = 1.55, 0.2
        block = VGroup(*[VGroup(*[Square(side, stroke_color=RULE, stroke_width=1, fill_opacity=1)
                                  .move_to(np.array([(j - (N - 1) / 2) * pitch, blk_y0 - r * blk_pitch, 0]))
                                  for j in range(N)]) for r in range(N_DRAWS)])
        draws = {"x": self.draws_by_beta[B_LO]}

        def fill_block(xs):
            return [block[r][j].animate.set_fill(SQ_PLUS if xs[r][j] > 0 else SQ_MINUS)
                    for r in range(N_DRAWS) for j in range(N)]
        for r in range(N_DRAWS):
            for j in range(N):
                block[r][j].set_fill(SQ_PLUS if draws["x"][r][j] > 0 else SQ_MINUS)
        blk_tag = tex(r"draws at this $\beta$", size=SIZE_SMALL, color=INK_3).next_to(block, DOWN, buff=0.12).align_to(block, LEFT)
        soft = mtex(r"p(\text{configuration}) \propto e^{\,\beta \cdot \text{agreement}}", size=SIZE_LABEL,
                    color=INK_2).move_to(np.array([0, -0.75, 0]))
        self.play_at("beta", FadeOut(VGroup(lines, nodes, cw_name, cnt0, more)), row_grp.animate.set_opacity(1),
                     Create(dial), FadeIn(dticks), FadeIn(dnums), Write(dname), FadeIn(needle),
                     run_time=1.0)
        self.play(LaggedStart(*[FadeIn(b, shift=DOWN * 0.05) for b in block], lag_ratio=0.12),
                  FadeIn(blk_tag), run_time=1.0)
        self.play_from(self.word("agreement") - 0.1, Write(soft), run_time=1.1)

        def resample(x):
            state["x"] = x.copy()
            return [row[j].animate.set_fill(SQ_PLUS if x[j] > 0 else SQ_MINUS) for j in range(N)]

        broad = label("broad", size=SIZE_SMALL, color=INK_2).next_to(dial, UP, buff=0.3).align_to(dial, LEFT)
        conc = label("concentrated", size=SIZE_SMALL, color=INK_2).next_to(dial.n2p(B_SHOW), UP, buff=0.3)
        # small beta: the draws differ from one another; large beta: each draw mostly agrees
        self.play_at("beta-low", Indicate(needle, color=INK, scale_factor=1.3),
                     *fill_block(self.draws_by_beta["low2"]), *resample(self.x_low), FadeIn(broad), run_time=1.0)
        self.play_at("beta-high", beta.animate.set_value(B_SHOW), *fill_block(self.draws_by_beta[B_SHOW]),
                     *resample(self.x_high), FadeIn(conc), run_time=1.3)
        # above the dial (below it would leave the frame), over "broad" and "concentrated"
        cool = Arrow(dial.n2p(0.68) + UP * 0.85, dial.n2p(1.52) + UP * 0.85, buff=0, stroke_width=3,
                     color=INK_2, max_tip_length_to_length_ratio=0.04)
        cool_lab = label("cooling", size=SIZE_SMALL, color=INK_2).next_to(cool, RIGHT, buff=0.15)
        self.play_at("cooling", GrowArrow(cool), FadeIn(cool_lab), run_time=0.9)

        # ============================================================ beat: m
        mline = NumberLine(x_range=[-1, 1, 1], length=9, color=RULE, stroke_width=2.5, include_ticks=True,
                           tick_size=0.07).move_to(np.array([0, -1.95, 0]))
        mx = mline.n2p
        m_ticks = VGroup(*[mtex(s, size=SIZE_SMALL, color=INK_3).next_to(mx(v), DOWN, buff=0.16)
                           for s, v in (("-1", -1), ("0", 0), ("+1", 1))])
        m_name = mtex("m", size=SIZE_BODY, color=INK).next_to(mline, RIGHT, buff=0.3)
        meq = mtex(r"m = \frac{1}{N}\sum_{i=1}^{N} x_i", size=SIZE_LABEL, color=INK).move_to(np.array([5.9, -0.75, 0]))
        # the needle returns to small beta and the block shows draws there, so the landscape below
        # opens on one valley (its split is shown later, at "two-wells")
        lo_x = self.draws_by_beta[B_LO]
        bm = [float(np.mean(x)) for x in lo_x]                       # small-beta draws: m near 0
        hm = [float(np.mean(x)) for x in self.draws_by_beta[B_SHOW]]  # large-beta draws: m near +-0.8
        dots = VGroup(*[Dot(mx(m) + UP * 0.1 * (1 + (r % 4)), radius=0.08, color=INK) for r, m in enumerate(bm)])
        self.play_at("m-meter", FadeOut(soft), FadeOut(broad), FadeOut(conc), FadeOut(cool), FadeOut(cool_lab),
                     beta.animate.set_value(B_LO), *fill_block(lo_x), run_time=0.8)
        self.play(FadeOut(blk_tag), Create(mline), FadeIn(m_ticks), FadeIn(m_name),
                  *[ReplacementTransform(block[r], dots[r]) for r in range(N_DRAWS)], run_time=1.2)
        self.play(Write(meq), run_time=0.9)
        tint_a = Line(mx(0), mx(1), stroke_color=MODE_A, stroke_width=6)
        tint_b = Line(mx(-1), mx(0), stroke_color=MODE_B, stroke_width=6)
        self.play_at("pos", Create(tint_a), *[d.animate.set_color(MODE_A) for d, m in zip(dots, bm) if m > 0],
                     run_time=0.8)
        self.play_at("neg", Create(tint_b), *[d.animate.set_color(MODE_B) for d, m in zip(dots, bm) if m < 0],
                     run_time=0.8)
        self.bring_to_front(dots)

        # ============================================================ beat: landscape
        YMAX = 12.0
        lax = Axes(x_range=[-1, 1, 1], y_range=[0, YMAX, 4], x_length=9, y_length=3.4, tips=False,
                   axis_config={"stroke_opacity": 0})
        lax.shift(mx(0) - lax.c2p(0, 0))
        xs = np.linspace(-0.97, 0.97, 131)

        def lcol(b):
            a = float(np.clip((b - 1.0) / 0.2, 0, 1))
            return (interpolate_color(ManimColor(INK_2), ManimColor(MODE_B), a),
                    interpolate_color(ManimColor(INK_2), ManimColor(MODE_A), a))

        def curve_mob():
            b = beta.get_value()
            y = neglogp(xs, b)
            cl, cr = lcol(b)
            out = VGroup()
            for sl, col in ((slice(0, 66), cl), (slice(65, None), cr)):
                pts = [lax.c2p(v, w) for v, w in zip(xs[sl], y[sl]) if w <= YMAX]
                out.add(VMobject(stroke_width=4.5, stroke_color=col).set_points_smoothly(pts))
            return out
        curve = always_redraw(curve_mob)
        lname = label("−log probability", size=SIZE_SMALL).move_to(np.array([-6.6, lax.c2p(0, YMAX)[1], 0]),
                                                                   aligned_edge=UL)
        tag_exact = tex(r"exact $\cdot$ $N = 64$ $\cdot$ no field", size=SIZE_SMALL, color=INK_3) \
            .move_to(np.array([6.6, 1.85, 0]), aligned_edge=RIGHT)
        curve.update()
        on_curve = lambda m: lax.c2p(m, min(float(neglogp(m, beta.get_value())[0]), YMAX)) + UP * 0.1  # noqa: E731
        self.play_at("landscape", Create(curve), FadeIn(lname), FadeIn(tag_exact), run_time=1.6)
        # likely values sit in valleys: the draws' dots drop onto the curve, at its valley floors
        self.play_from(self.word("likely"), *[d.animate.move_to(on_curve(m)) for d, m in zip(dots, bm)],
                       run_time=0.9)
        # a scale-only pulse: each dot grows about its own centre, stays on the curve, keeps its mode colour
        self.play_from(self.word("valleys"), *[d.animate(rate_func=there_and_back).scale(1.4) for d in dots],
                       run_time=0.8)

        one = tex(r"one valley, at $m = 0$", size=SIZE_LABEL, color=INK_2).move_to(lax.c2p(0, 6.5))
        one_arr = Arrow(one.get_bottom() + DOWN * 0.06, lax.c2p(0, 0.9), buff=0, stroke_width=3, color=INK_2,
                        max_tip_length_to_length_ratio=0.15)
        self.play_at("one-well", FadeIn(one), GrowArrow(one_arr), run_time=0.8)
        # past beta = 1 the valley splits; the dots become the large-beta draws (paired by rank, so
        # few change side) and ride the curve into the two valleys
        order_lo, order_hi = np.argsort(bm), np.argsort(hm)
        m_from, m_to = np.empty(N_DRAWS), np.empty(N_DRAWS)
        m_from[order_lo] = np.array(bm)[order_lo]
        m_to[order_lo] = np.array(hm)[order_hi]
        mix = ValueTracker(0.0)

        def ride(k):
            def upd(d):
                m = (1 - mix.get_value()) * m_from[k] + mix.get_value() * m_to[k]
                d.move_to(on_curve(m))
                d.set_color(MODE_A if m > 0.02 else MODE_B if m < -0.02 else INK)
            return upd
        for k, d in enumerate(dots):
            d.add_updater(ride(k))
        self.play_at("two-wells", beta.animate.set_value(B_SHOW), mix.animate.set_value(1.0),
                     FadeOut(one), FadeOut(one_arr), run_time=2.2, rate_func=smooth)
        for d in dots:
            d.clear_updaters()
        lab_pos = text("positive mode", size=SIZE_LABEL, color=MODE_A).next_to(mx(0.5), DOWN, buff=0.45)
        lab_neg = text("negative mode", size=SIZE_LABEL, color=MODE_B).next_to(mx(-0.5), DOWN, buff=0.45)
        self.play_at("modes", FadeIn(lab_pos, shift=UP * 0.1), FadeIn(lab_neg, shift=UP * 0.1), run_time=0.8)

        # ============================================================ beat: gibbs
        paint(self.chain_start)
        row_new = VGroup(*[row[j].copy() for j in range(N)])
        paint(state["x"])
        ball = Dot(radius=0.11, color=NOISE)
        ball_m = ValueTracker(float(self.chain_start.mean()))
        ball.add_updater(lambda d: d.move_to(lax.c2p(ball_m.get_value(),
                                                     min(float(neglogp(ball_m.get_value(), beta.get_value())[0]), YMAX)))
                         .shift(UP * 0.12))
        ball.update()
        restart = tex(r"a new run starts at small $\beta$", size=SIZE_SMALL, color=INK_3) \
            .next_to(dial, UP, buff=0.32).align_to(dial, LEFT)
        self.play_at("gibbs", FadeOut(lab_pos), FadeOut(lab_neg), FadeOut(dots), beta.animate.set_value(B_LO),
                     FadeIn(restart),
                     *[Transform(row[j], row_new[j]) for j in range(N)], FadeOut(meq), run_time=1.0)
        self.add(ball)
        self.play(FadeIn(ball, scale=0.4), run_time=0.4)

        # the redraw of one spin: a highlight, the conditional probability as a split bar, the new value
        hl = Square(side + 0.09, stroke_color=INK, stroke_width=3).move_to(row[self.sweep_order[0]])
        bar_w, bar_h = 0.9, 0.14

        def split_bar(j, p):
            c = row[j].get_center() + UP * 0.38
            lt = Rectangle(width=bar_w * p, height=bar_h, stroke_width=0, fill_color=SQ_PLUS, fill_opacity=1)
            dk = Rectangle(width=bar_w * (1 - p), height=bar_h, stroke_width=0, fill_color=SQ_MINUS, fill_opacity=1)
            g = VGroup(lt, dk).arrange(RIGHT, buff=0).move_to(c)
            frame = Rectangle(width=bar_w, height=bar_h, stroke_color=INK_3, stroke_width=1).move_to(c)
            return VGroup(g, frame)
        cond = tex(r"$p(+1 \mid \text{all the others})$", size=SIZE_SMALL, color=INK_2)
        side_of = lambda m: RIGHT if m.get_x() < 2.5 else LEFT  # noqa: E731  (keep the label in frame)
        x_cur = self.chain_start.copy()
        hop_t = [self.word("redraws") - 0.05, self.word("time") - 0.1, self.word("given") - 0.1]
        first_bar = None
        for k, t0 in enumerate(hop_t):
            j = int(self.sweep_order[k])
            sb = split_bar(j, self.sweep_p[k])
            anims = [Create(hl) if k == 0 else hl.animate.move_to(row[j])]
            if first_bar is None:
                cond.next_to(sb, side_of(sb), buff=0.2)
                anims += [FadeIn(sb), FadeIn(cond)]
                first_bar = sb
            else:
                anims += [Transform(first_bar, sb), cond.animate.next_to(sb, side_of(sb), buff=0.2)]
            self.play_from(t0, *anims, run_time=0.4)
            x_cur = self.sweep_states[k + 1]
            self.play(row[j].animate.set_fill(SQ_PLUS if x_cur[j] > 0 else SQ_MINUS),
                      ball_m.animate.set_value(float(x_cur.mean())), run_time=0.35)

        # ============================================================ beat: sweep
        cpos = row.get_bottom() + DOWN * 0.42
        c_pre = tex(r"updates:", size=SIZE_LABEL, color=INK_2)
        c_post = tex(r"of $N$", size=SIZE_LABEL, color=INK_2)
        done = {"k": 3}

        def count_mob():
            num = text("%d" % done["k"], size=SIZE_LABEL * 0.8, color=INK)
            g = VGroup(c_pre.copy(), num, c_post.copy()).arrange(RIGHT, buff=0.14)
            num.align_to(g[0], DOWN)
            return g.move_to(cpos)
        counter = always_redraw(count_mob)

        def fast(mob, a):
            k = min(3 + int(round(a * (N - 3))), N)
            done["k"] = k
            xk = self.sweep_states[k]
            for jj in range(N):
                row[jj].set_fill(SQ_PLUS if xk[jj] > 0 else SQ_MINUS)
            hl.move_to(row[int(self.sweep_order[k - 1])])
            ball_m.set_value(float(xk.mean()))
        self.play_at("sweep", FadeOut(VGroup(row_label, leg_all)), FadeOut(restart), FadeOut(first_bar), FadeOut(cond),
                     FadeIn(counter), run_time=0.3)
        self.play(UpdateFromAlphaFunc(VMobject(), fast), run_time=2.0, rate_func=linear)
        self.remove(counter)
        cstat = count_mob()
        self.add(cstat)
        sweep_lab = tex(r"$N$ updates $=$ 1 sweep", size=SIZE_LABEL, color=INK).move_to(cpos)
        self.play(FadeOut(hl), ReplacementTransform(cstat, sweep_lab), run_time=0.7)

        # ============================================================ beat: anneal
        lev_ticks = VGroup(*[Line(dial.n2p(b) + DOWN * 0.06, dial.n2p(b) + UP * 0.06, stroke_color=INK_3,
                                  stroke_width=1.5) for b in LEVELS])
        # film time per level: 0.18 s up to beta = 1, 0.32 s above (the modes form there)
        dur = np.array([0.18 if b <= 1.0 else 0.32 for b in LEVELS])
        edges = np.concatenate([[0.0], np.cumsum(dur)]) / dur.sum()
        n_lev = len(LEVELS)

        def anneal_step(mob, a):
            li = min(int(np.searchsorted(edges, a, side="right")) - 1, n_lev - 1)
            frac = min((a - edges[li]) / (edges[li + 1] - edges[li]), 1.0)
            beta.set_value(LEVELS[li])
            states = self.anneal[li]
            xk = states[min(int(frac * len(states)), len(states) - 1)]
            for jj in range(N):
                row[jj].set_fill(SQ_PLUS if xk[jj] > 0 else SQ_MINUS)
            ball_m.set_value(float(xk.mean()))
        self.play_at("anneal", FadeIn(lev_ticks), UpdateFromAlphaFunc(VMobject(), anneal_step),
                     run_time=float(dur.sum()), rate_func=linear)
        m_end = float(self.anneal[-1][-1].mean())
        self.play(ball.animate.set_color(MODE_A if m_end > 0 else MODE_B), run_time=0.2)

        # ============================================================ beat: fork
        fd = json.loads((VISUAL / "data" / "fork_trajectories.json").read_text())
        fb = np.array(fd["beta"])
        fm = np.array(fd["m"])
        fax = Axes(x_range=[F_LO, F_HI, 0.05], y_range=[-1, 1, 0.5], x_length=8.4, y_length=5.0, tips=False,
                   axis_config={"stroke_opacity": 0}).move_to(np.array([-1.7, 0.35, 0]))
        x_axis = Line(fax.c2p(F_LO, -1), fax.c2p(F_HI, -1), stroke_color=RULE, stroke_width=3)
        x_ticks = VGroup(*[Line(fax.c2p(v, -1) + DOWN * 0.09, fax.c2p(v, -1) + UP * 0.09, stroke_color=INK_3,
                                stroke_width=2) for v in (0.7, 1.0, 1.35)])
        x_nums = VGroup(*[mtex(s, size=SIZE_SMALL, color=INK_3).next_to(fax.c2p(v, -1), DOWN, buff=0.16)
                          for s, v in (("0.7", 0.7), ("1", 1.0), ("1.35", 1.35))])
        x_name = mtex(r"\beta", size=SIZE_BODY, color=INK).next_to(x_axis, RIGHT, buff=0.3)
        y_up = Line(fax.c2p(F_LO, 0), fax.c2p(F_LO, 1), stroke_color=MODE_A, stroke_width=6)
        y_dn = Line(fax.c2p(F_LO, -1), fax.c2p(F_LO, 0), stroke_color=MODE_B, stroke_width=6)
        y_nums = VGroup(*[mtex(s, size=SIZE_SMALL, color=INK_3).next_to(fax.c2p(F_LO, v), LEFT, buff=0.3)
                          for s, v in (("-1", -1), ("0", 0), ("+1", 1))])
        y_name = mtex("m", size=SIZE_BODY, color=INK).next_to(fax.c2p(F_LO, 1), UP, buff=0.2)
        ftag = tex(r"drift-plus-noise paths $\cdot$ no field $\cdot$ $N = 1{,}024$ $\cdot$ illustration",
                   size=SIZE_SMALL, color=INK_3) \
            .next_to(x_nums, DOWN, buff=0.2).align_to(x_axis, LEFT)
        land_out = VGroup(curve, ball, lname, tag_exact, row, sweep_lab, mline, m_ticks, needle, lev_ticks,
                          dticks, dnums, tint_a, tint_b)
        curve.clear_updaters()
        ball.clear_updaters()
        # three steps, no overlap: the landscape and every dial label leave; the dial line slides
        # into the beta axis while the m axis grows from m = 0; then the new labels write on
        self.play_at("fork", FadeOut(land_out), run_time=0.4)
        self.play(ReplacementTransform(dial, x_axis), ReplacementTransform(dname, x_name),
                  ReplacementTransform(m_name, y_name),
                  GrowFromPoint(y_up, fax.c2p(F_LO, 0)), GrowFromPoint(y_dn, fax.c2p(F_LO, 0)), run_time=0.6)
        self.play(FadeIn(x_ticks), FadeIn(x_nums), FadeIn(y_nums), Write(ftag), run_time=0.5)

        # the fourteen paths, drawn in three stretches; each takes its end colour after its last
        # zero crossing
        i_a = int(np.searchsorted(fb, 0.97))
        i_b = int(np.searchsorted(fb, 1.10))
        last = []
        for traj in fm:
            s = np.sign(traj)
            cr = np.where(s[1:] != s[:-1])[0]
            last.append(int(cr[-1]) + 1 if len(cr) else 0)

        def seg(traj, i0, i1, color):
            return VMobject(stroke_color=color, stroke_width=1.8).set_points_as_corners(
                [fax.c2p(b, m) for b, m in zip(fb[i0:i1 + 1], traj[i0:i1 + 1])])
        s1 = VGroup(*[seg(t, 0, i_a, NOISE) for t in fm])
        s2 = VGroup(*[seg(t, i_a, i_b, NOISE) for t in fm])
        s3 = VGroup(*[seg(t, i_b, len(fb) - 1, MODE_A if t[-1] > 0 else MODE_B) for t in fm])
        # recoloured copies of the grey stretches, from each path's last zero crossing on
        rec = VGroup()
        for t, lc in zip(fm, last):
            if lc < i_b:
                rec.add(seg(t, lc, i_b, MODE_A if t[-1] > 0 else MODE_B))
        self._wait_to(self.t("runs"))
        t_runs = self.now
        self.play_window(t_runs, self.t("split") - 0.1, *[Create(p) for p in s1], rate_func=linear)
        self._log("runs", "start", self.t("runs"), t_runs, self.now)

        b1 = DashedLine(fax.c2p(1.0, -1), fax.c2p(1.0, 1), stroke_color=INK_3, stroke_width=2,
                        dash_length=0.08, dashed_ratio=0.55)
        b1_lab = mtex(r"\beta = 1", size=SIZE_LABEL, color=INK_2).next_to(fax.c2p(1.0, 1), UP, buff=0.12)
        self._wait_to(self.t("split"))
        t_split = self.now
        self.play_window(t_split, self.word("grows") + 0.5, Create(b1), FadeIn(b1_lab),
                         *[Create(p) for p in s2], rate_func=linear)
        self._log("split", "start", self.t("split"), t_split, self.now)

        self.play_at("falls", *[Create(p) for p in s3], FadeIn(rec), run_time=1.8, rate_func=linear)
        bb = np.linspace(1.0, F_HI, 60)
        th_a = theory(VMobject(stroke_width=2.5).set_points_smoothly([fax.c2p(b, mstar(b)) for b in bb]),
                      num_dashes=30)
        th_b = theory(VMobject(stroke_width=2.5).set_points_smoothly([fax.c2p(b, -mstar(b)) for b in bb]),
                      num_dashes=30)
        th_la = label("positive mode", size=SIZE_SMALL, color=MODE_A).next_to(fax.c2p(F_HI, mstar(F_HI)), RIGHT, buff=0.2)
        th_lb = label("negative mode", size=SIZE_SMALL, color=MODE_B).next_to(fax.c2p(F_HI, -mstar(F_HI)), RIGHT, buff=0.2)
        # the valley floors (m = tanh(beta m), dashed) drawn over the runs, with the modes' names
        self.play(Create(th_a), Create(th_b), FadeIn(th_la), FadeIn(th_lb), run_time=0.8)

        # ============================================================ beat: locked
        iax = Axes(x_range=[-1, 1, 1], y_range=[0, 0.16, 0.04], x_length=3.4, y_length=2.1, tips=False,
                   axis_config={"stroke_opacity": 0}).move_to(np.array([5.0, 0.2, 0]))
        ib = 1.35
        ixs = np.linspace(-0.97, 0.97, 121)
        iy = np.minimum(big_n_landscape(ixs, ib), 0.16)
        il = VMobject(stroke_width=3.5, stroke_color=MODE_B).set_points_smoothly(
            [iax.c2p(v, w) for v, w in zip(ixs[:61], iy[:61])])
        ir = VMobject(stroke_width=3.5, stroke_color=MODE_A).set_points_smoothly(
            [iax.c2p(v, w) for v, w in zip(ixs[60:], iy[60:])])
        ims = mstar(ib)
        iv = ValueTracker(ims)
        iball = always_redraw(lambda: Dot(iax.c2p(iv.get_value(), float(big_n_landscape(iv.get_value(), ib))),
                                          radius=0.09, color=MODE_A).shift(UP * 0.1))
        self.play_at("locked", Create(il), Create(ir), run_time=0.9)
        self.play(FadeIn(iball, scale=0.4), run_time=0.3)
        # switching: a push toward the other mode climbs part of the hill and falls back
        sw = label("switching: slow", size=SIZE_SMALL, color=INK_2).next_to(iax.c2p(0, 0.16), UP, buff=0.1)
        self.play_from(self.word("switching") - 0.1, FadeIn(sw),
                       iv.animate(rate_func=there_and_back).set_value(0.28), run_time=1.4)
        # settling: small moves inside the valley
        st = label("settling: fast", size=SIZE_SMALL, color=INK_2).next_to(iax.c2p(ims, 0), DOWN, buff=0.25)
        st.shift(LEFT * max(0.0, st.get_right()[0] - (config.frame_x_radius - 0.45)))
        wig = ValueTracker(0.0)
        iv.add_updater(lambda v: v.set_value(ims + 0.06 * np.sin(TAU * 3 * wig.get_value())
                                             * (1 - wig.get_value())))
        self.play_from(self.word("settling") - 0.1, FadeIn(st), wig.animate.set_value(1.0),
                       run_time=1.1, rate_func=linear)
        iv.clear_updaters()
        iv.set_value(ims)

        # ============================================================ beat: decides
        lo, hi = fb[min(last)], fb[max(last)]
        band = Rectangle(width=fax.c2p(hi, 0)[0] - fax.c2p(lo, 0)[0], height=fax.c2p(0, 1)[1] - fax.c2p(0, -1)[1],
                         stroke_width=0, fill_color=NOISE, fill_opacity=0.28) \
            .move_to(np.array([(fax.c2p(lo, 0)[0] + fax.c2p(hi, 0)[0]) / 2, fax.c2p(0, 0)[1], 0]))
        ends = VGroup(*[Dot(fax.c2p(fb[-1], t[-1]), radius=0.055, color=MODE_A if t[-1] > 0 else MODE_B) for t in fm])
        self.play_at("decides", FadeOut(VGroup(il, ir, iball, sw, st)), FadeIn(band), run_time=0.3)
        self.bring_to_back(band)
        # one run traced from its sign inside the band to its end mode
        # (the run that stays farthest from m = 0 after beta = 1.1, so the trace reads as one choice)
        i_11 = int(np.searchsorted(fb, 1.10))
        k_tr = int(np.argmax([np.min(np.abs(t[i_11:])) for t in fm]))
        t_tr = fm[k_tr]
        col_tr = MODE_A if t_tr[-1] > 0 else MODE_B
        tr = VMobject(stroke_color=col_tr, stroke_width=5).set_points_as_corners(
            [fax.c2p(b, m) for b, m in zip(fb[last[k_tr]:], t_tr[last[k_tr]:])])
        tr_start = Dot(fax.c2p(fb[last[k_tr]], t_tr[last[k_tr]]), radius=0.07, color=col_tr)
        self.play_from(self.word("sign") - 0.05, VGroup(s1, s2, s3, rec).animate.set_stroke(opacity=0.3),
                       FadeIn(tr_start, scale=0.5), Create(tr), run_time=1.6, rate_func=linear)
        self.play_from(self.word("decide") - 0.1, FadeIn(ends, scale=0.5), run_time=0.8)
