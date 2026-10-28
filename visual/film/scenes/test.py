"""Scene "test" (The test; page chapter test): the three-pattern Hopfield test of the prediction
c = 0.120 x 1.662 = 0.199, the one-pattern ratio that moves it outside the interval, and Table 1
(script.md, scene: test, and its Sources table).

Every number on screen is read from the visual edition's data files and checked against the paper:
  ../data/field_grid_points.json   Fig. 2b: eight fields, logit q_eq (reference, with its envelope) and
                              logit q_chain (Wilson 95%); c measured 0.199 [0.1908, 0.2095] (pattern-set
                              resampling), predicted A_3 c_CW = 0.199; at G = 0.0006125: 66.9% vs 97.26%
  ../data/c_cw.json           the Curie-Weiss q_chain = 60.5% at the same field (derived)
  ../data/sampler_clock.json  the one-pattern ratio 1.063 at N = 2,048 and the shifted prediction 0.212
  ../data/table1.json         Table 1 rows p = 1..4 at N = 2,048 (separate runs, pooled fields)
  ../data/size_scan.json      Table 1 rows N = 1,024 and 4,096 at p = 3
Derived on screen: predicted share at the strongest field = logistic(0.199 x logit 0.97274) = 67.1%,
with 0.97274 the Curie-Weiss equilibrium share in c_cw.json (the same 97.3%).
The pattern tiles and the spin row are seeded illustrations, not data.

Render:  PYTHONNOUSERSITE=1 manim -ql --media_dir media scenes/test.py Test
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import *  # noqa: E402,F403

DATA = Path(__file__).resolve().parents[2] / "data"


def load(name):
    return json.loads((DATA / name).read_text())


def logit(q):
    return math.log(q / (1 - q))


def logistic(x):
    return 1 / (1 + math.exp(-x))


F2B = load("field_grid_points.json")
T1 = load("table1.json")
SS = load("size_scan.json")
CLK = load("sampler_clock.json")
CCW = load("c_cw.json")

POINTS = sorted(F2B["points"], key=lambda p: p["G"])          # eight fields, -G .. +G
STRONG = max(F2B["points"], key=lambda p: p["G"])             # G = +0.0006125
G_STRONG = STRONG["G"]
Q_MEAS = STRONG["q_chain"]["point"]                           # 0.669
Q_REF = STRONG["q_eq"]["point"]                               # 0.973
MEAS = F2B["c_measured"]                                      # 0.199 [0.1908, 0.2095]
PRED = F2B["c_predicted"]                                     # 0.199
Q_PRED = logistic(PRED * logit(CCW["runs"][-1]["q_eq"]))      # 0.671 (on the CW 97.3%)
Q_CW = CCW["runs"][-1]["q_chain"]                             # 0.605
R1 = next(r for r in CLK["rows"] if r["N"] == 2048)["r"]      # 1.063
PRED_R = CLK["p3_clock_corrected_prediction_N2048"]           # 0.212
G_MAGS = sorted({round(abs(p["G"]), 10) for p in POINTS})     # four strengths

assert f"{100 * Q_MEAS:.1f}" == "66.9" and f"{100 * Q_REF:.2f}" == "97.26"
assert f"{MEAS['point']:.3f}" == "0.199" and f"{MEAS['lower']:.4f}" == "0.1908" and f"{MEAS['upper']:.4f}" == "0.2095"
assert f"{PRED:.3f}" == "0.199" and f"{100 * Q_PRED:.1f}" == "67.1" and f"{100 * Q_CW:.1f}" == "60.5"
assert f"{R1:.3f}" == "1.063" and f"{PRED_R:.3f}" == "0.212" and abs(PRED * R1 - PRED_R) < 1e-3
assert len(POINTS) == 8 and len(G_MAGS) == 4

LIGHT = "#D9D7D0"   # pattern entry +1 (as in the modes scene)
DARK = "#2C2D31"    # pattern entry -1


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def pct2(q):
    return f"{100 * q:.2f}\\%"


def pct1(q):
    return f"{100 * q:.1f}\\%"


# ------------------------------------------------------------------ small parts
def pattern_block(bits, px):
    g = VGroup()
    n = bits.shape[0]
    for i in range(n):
        for j in range(n):
            g.add(Square(px, stroke_width=0, fill_opacity=1, fill_color=LIGHT if bits[i, j] > 0 else DARK)
                  .move_to([j * px, -i * px, 0]))
    return g


def tri_marker(scale=0.15):
    """The prediction marker: a downward triangle, dashed light grey."""
    t = Triangle(stroke_width=3).rotate(PI).scale(scale)
    return theory(t, num_dashes=9, dashed_ratio=0.6)


def interval_bar(x_lo, x_hi, y, cap=0.14, sw=3.5):
    return VGroup(Line([x_lo, y, 0], [x_hi, y, 0], stroke_width=sw, color=INK),
                  Line([x_lo, y - cap / 2, 0], [x_lo, y + cap / 2, 0], stroke_width=sw, color=INK),
                  Line([x_hi, y - cap / 2, 0], [x_hi, y + cap / 2, 0], stroke_width=sw, color=INK))


def spin_arrow(s, at, h=0.42, color=None):
    """One spin: an arrow up (+1, blue) or down (-1, orange)."""
    a = Arrow(ORIGIN + DOWN * h / 2, ORIGIN + UP * h / 2, buff=0, stroke_width=4,
              max_tip_length_to_length_ratio=0.32, color=color or (MODE_A if s > 0 else MODE_B))
    if s < 0:
        a.rotate(PI)
    return a.move_to(at)


class Test(NarratedScene):
    scene_id = "test"

    def w(self, word, after=0.0):
        """Start time of the first spoken word equal to `word` at or after `after` seconds."""
        for x in self.timing["words"]:
            if x["start"] >= after - 1e-6 and x["w"].strip(".,:;").lower() == word.lower():
                return x["start"]
        raise KeyError(word)

    def construct(self):
        # ============================================================ beat: run
        # the prediction from the modes scene, kept at the top while it is tested
        pred_eq = mtex("0.120", r"\;\times\;", "1.662", r"\;=\;", "0.199", size=SIZE_HEAD)
        pred_eq[0].set_color(GILT_TEXT)
        pred_eq[4].set_color(THEORY)
        pred_eq.move_to(UP * 3.05)
        pred_box = theory(SurroundingRectangle(pred_eq[4], buff=0.12, corner_radius=0.06), num_dashes=26)
        pred_tags = VGroup(label("Curie–Weiss", size=SIZE_SMALL).next_to(pred_eq[0], DOWN, buff=0.2),
                           label("competition", size=SIZE_SMALL).next_to(pred_eq[2], DOWN, buff=0.2),
                           label("prediction", size=SIZE_SMALL, color=THEORY).next_to(pred_eq[4], DOWN, buff=0.2))
        y_tag = min(t.get_y() for t in pred_tags)
        for t in pred_tags:
            t.set_y(y_tag)
        pred = VGroup(pred_eq, pred_box)

        # twenty pattern sets, three random +-1 patterns each (seeded illustration)
        rng = np.random.default_rng(2048)
        tiles = VGroup()
        for _ in range(20):
            blocks = VGroup(*[pattern_block(rng.choice([-1, 1], size=(5, 5)), 0.066) for _ in range(3)])
            blocks.arrange(RIGHT, buff=0.06)
            tiles.add(VGroup(SurroundingRectangle(blocks, buff=0.05, stroke_color=RULE, stroke_width=1.5), blocks))
        tiles.arrange_in_grid(rows=4, cols=5, buff=(0.18, 0.2)).move_to([-3.3, -0.15, 0])
        tiles_tag = label("20 pattern sets · 3 patterns each", size=SIZE_SMALL).next_to(tiles, DOWN, buff=0.3)

        # eight fields: four strengths, each pointing toward pattern 1 (right) or toward its mirror, -pattern 1 (left)
        FX, FY = 4.3, 0.95
        gmax = max(G_MAGS)
        f_right, f_left = VGroup(), VGroup()
        for k, g in enumerate(G_MAGS):
            y = FY - 0.5 * k
            L = 1.55 * g / gmax
            f_right.add(Arrow([FX + 0.08, y, 0], [FX + 0.08 + L, y, 0], buff=0, stroke_width=4,
                              max_tip_length_to_length_ratio=0.18, color=BIAS))
            f_left.add(Arrow([FX - 0.08, y, 0], [FX - 0.08 - L, y, 0], buff=0, stroke_width=4,
                             max_tip_length_to_length_ratio=0.18, color=BIAS))
        f_axis = DashedLine([FX, FY + 0.35, 0], [FX, FY - 1.85, 0], dash_length=0.06, color=RULE, stroke_width=1.5)
        f_head = VGroup(label("8 fields", size=SIZE_LABEL, color=BIAS), mtex("G", size=SIZE_LABEL + 4, color=BIAS)
                        ).arrange(RIGHT, buff=0.15).move_to([FX, FY + 0.75, 0])
        f_ends = VGroup(label("toward −pattern 1", size=SIZE_SMALL).move_to([FX - 0.2, FY - 2.15, 0], aligned_edge=RIGHT),
                        label("toward pattern 1", size=SIZE_SMALL).move_to([FX + 0.2, FY - 2.15, 0], aligned_edge=LEFT))
        settings = tex(r"$p = 3$ \ $\cdot$ \ $N = 2{,}048$ \ $\cdot$ \ 2{,}000 runs per pattern set and field",
                       size=SIZE_LABEL, color=INK_3).move_to(DOWN * 3.3)

        self.play_at("run", FadeIn(pred_eq, shift=DOWN * 0.15), Create(pred_box), FadeIn(pred_tags),
                     LaggedStart(*[FadeIn(t, scale=0.7) for t in tiles], lag_ratio=0.06),
                     run_time=max(1.0, self.t("fields") - 0.45))
        self.play_until("fields", FadeIn(tiles_tag), min_run_time=0.3)
        self.play_at("fields", FadeIn(f_head), Create(f_axis),
                     LaggedStart(*[GrowArrow(a) for a in f_right], lag_ratio=0.15), run_time=1.2)
        self.play(FadeIn(settings), run_time=0.6)
        self._wait_to(self.w("each") - 0.05)
        self.play(LaggedStart(*[GrowArrow(a) for a in f_left], lag_ratio=0.15), FadeIn(f_ends), run_time=1.3)

        # ============================================================ beat: share
        # one bar per proportion: positive sign (blue) and negative sign (orange), 0-100%
        BX0, BX1 = -4.6, 4.0
        BW = BX1 - BX0
        bx = lambda q: BX0 + BW * q  # noqa: E731
        Y_S, Y_R, BH = 0.75, -0.95, 0.5

        strong = f_right[-1]
        g_tag = VGroup(mtex("G", "=", "0.0006125", size=SIZE_LABEL, color=INK_2), label("toward pattern 1", size=SIZE_SMALL)
                       ).arrange(RIGHT, buff=0.25)
        g_tag[0][0].set_color(BIAS)
        g_arrow = Arrow(ORIGIN, RIGHT * 1.0, buff=0, stroke_width=4, max_tip_length_to_length_ratio=0.2, color=BIAS)
        g_grp = VGroup(g_arrow, g_tag).arrange(RIGHT, buff=0.25).move_to([0, 1.95, 0])
        self.play_at("share",
                     FadeOut(tiles, shift=LEFT * 0.3), FadeOut(tiles_tag), FadeOut(settings),
                     FadeOut(VGroup(f_left, f_right[:-1], f_axis, f_head, f_ends)),
                     ReplacementTransform(strong, g_arrow), FadeIn(g_tag),
                     run_time=1.0)

        s_frame = Rectangle(width=BW, height=BH, stroke_color=INK, stroke_width=2).move_to([BX0 + BW / 2, Y_S, 0])
        s_pos = Rectangle(width=BW * Q_MEAS, height=BH, stroke_width=0, fill_color=MODE_A, fill_opacity=0.9
                          ).move_to([bx(Q_MEAS / 2), Y_S, 0])
        s_neg = Rectangle(width=BW * (1 - Q_MEAS), height=BH, stroke_width=0, fill_color=MODE_B, fill_opacity=0.9
                          ).move_to([bx((1 + Q_MEAS) / 2), Y_S, 0])
        s_lab = label("sampled", size=SIZE_LABEL, color=INK).next_to(s_frame, LEFT, buff=0.3)
        s_val = mtex(pct1(Q_MEAS), size=SIZE_LABEL, color=INK).next_to([bx(Q_MEAS), Y_S + BH / 2, 0], UP, buff=0.15)
        sign_keys = VGroup(label("positive", size=SIZE_SMALL, color=BG).move_to([bx(0.0) + 0.65, Y_S, 0]),
                           label("negative", size=SIZE_SMALL, color=BG).move_to([bx(1.0) - 0.7, Y_S, 0]))
        cond = label("samples labelled pattern 1", size=SIZE_SMALL).next_to(s_frame, UP, buff=0.15).set_x(bx(0.17))
        self.play(FadeIn(s_lab), Create(s_frame), FadeIn(cond), run_time=0.8)
        # the positive share fills as it is spoken
        t_num = self.w("sixty-six")
        self._wait_to(t_num - 1.4)
        s_pos.save_state()
        s_pos.stretch(0.001, 0, about_edge=LEFT)
        self.add(s_pos)
        self.play(Restore(s_pos), run_time=1.4, rate_func=smooth)
        self.play(FadeIn(s_neg), Write(s_val), FadeIn(sign_keys), run_time=0.9)

        # the model's own share, dashed light grey: a reference, not a measurement
        r_frame = theory(Rectangle(width=BW, height=BH).move_to([BX0 + BW / 2, Y_R, 0]), num_dashes=60)
        r_pos = Rectangle(width=BW * Q_REF, height=BH, stroke_width=0, fill_color=MODE_A, fill_opacity=0.35
                          ).move_to([bx(Q_REF / 2), Y_R, 0])
        r_neg = Rectangle(width=BW * (1 - Q_REF), height=BH, stroke_width=0, fill_color=MODE_B, fill_opacity=0.35
                          ).move_to([bx((1 + Q_REF) / 2), Y_R, 0])
        r_div = theory(Line([bx(Q_REF), Y_R - BH / 2, 0], [bx(Q_REF), Y_R + BH / 2, 0]), num_dashes=4)
        r_lab = VGroup(label("model’s", size=SIZE_LABEL, color=THEORY),
                       label("reference", size=SIZE_LABEL, color=THEORY)).arrange(DOWN, buff=0.06, aligned_edge=RIGHT)
        r_lab.next_to(r_frame, LEFT, buff=0.3)
        r_val = mtex(pct2(Q_REF), size=SIZE_LABEL, color=THEORY).next_to([bx(Q_REF), Y_R - BH / 2, 0], DOWN, buff=0.15)
        self.play_at("reference", FadeIn(r_lab), Create(r_frame), FadeIn(r_pos), FadeIn(r_neg), Create(r_div),
                     run_time=1.6)
        self._wait_to(self.w("ninety-seven") - 0.1)
        self.play(Write(r_val), run_time=0.7)

        r_val2 = mtex(r"\approx " + pct2(Q_REF), size=SIZE_LABEL, color=THEORY).move_to(r_val, aligned_edge=RIGHT)
        approx_tag = label("approximate: from the mean overlap inside the positive mode", size=SIZE_SMALL
                           ).next_to(r_frame, DOWN, buff=0.75).set_x(BX0 + BW / 2)
        self.play_at("approx", TransformMatchingShapes(r_val, r_val2), FadeIn(approx_tag, shift=UP * 0.1), run_time=1.0)

        # the gap between the two shares
        gap = Rectangle(width=bx(Q_REF) - bx(Q_MEAS), height=Y_S - Y_R - BH, stroke_width=0,
                        fill_color=NOISE, fill_opacity=0.32).move_to([(bx(Q_MEAS) + bx(Q_REF)) / 2, (Y_S + Y_R) / 2, 0])
        gap_edges = VGroup(*[DashedLine([x, Y_S - BH / 2, 0], [x, Y_R + BH / 2, 0], dash_length=0.06,
                                        color=INK_2, stroke_width=1.5) for x in (bx(Q_MEAS), bx(Q_REF))])
        self.play_at("still-far", FadeIn(gap), Create(gap_edges), run_time=1.2)

        # ============================================================ beat: fit
        # the paper's Fig. 2b: sampled log-odds against the reference log-odds, one point per field
        U = 0.78
        ax = Axes(x_range=[-4, 4, 1], y_range=[-2.5, 2.5, 1], x_length=8 * U, y_length=5 * U,
                  tips=False, axis_config=dict(color=RULE, stroke_width=2, include_ticks=True, tick_size=0.05))
        ax.move_to([-3.0, 0.0, 0])
        x_nums = VGroup(*[mtex(str(v), size=SIZE_SMALL, color=INK_3).next_to(ax.c2p(v, -2.5), DOWN, buff=0.12)
                          for v in (-4, -2, 0, 2, 4)])
        y_nums = VGroup(*[mtex(str(v), size=SIZE_SMALL, color=INK_3).next_to(ax.c2p(-4, v), LEFT, buff=0.12)
                          for v in (-2, 0, 2)])
        frame_ticks = VGroup(*[Line(ax.c2p(v, -2.5), ax.c2p(v, -2.5) + DOWN * 0.07, color=RULE, stroke_width=2) for v in (-4, -2, 0, 2, 4)],
                             *[Line(ax.c2p(-4, v), ax.c2p(-4, v) + LEFT * 0.07, color=RULE, stroke_width=2) for v in (-2, 0, 2)])
        x_nums.add(frame_ticks)
        x_lab = label("model’s log-odds (reference)", size=SIZE_SMALL, color=INK_2).next_to(ax, DOWN, buff=0.5)
        y_lab = label("sampled log-odds", size=SIZE_SMALL, color=INK_2).rotate(PI / 2).next_to(ax, LEFT, buff=0.5)
        diag = theory(Line(ax.c2p(-2.5, -2.5), ax.c2p(2.5, 2.5)), num_dashes=34)
        diag_lab = label("model’s proportions (c = 1)", size=SIZE_SMALL, color=THEORY).next_to(ax.c2p(2.5, 2.5), RIGHT, buff=0.12)

        pts = VGroup()
        for p in POINTS:
            x, y = p["logit_q_eq"], p["logit_q_chain"]
            xl, xh = logit(p["q_eq"]["lower"]), logit(p["q_eq"]["upper"])
            yl, yh = logit(p["q_chain"]["lower"]), logit(p["q_chain"]["upper"])
            pts.add(VGroup(Line(ax.c2p(xl, y), ax.c2p(xh, y), stroke_width=4, color=NOISE),
                           Line(ax.c2p(x, yl), ax.c2p(x, yh), stroke_width=2.5, color=INK),
                           Dot(ax.c2p(x, y), radius=0.06, color=INK)))
        strong_pt = pts[[p["G"] for p in POINTS].index(G_STRONG)]

        # the strongest-field point is read off the two bars: x from 97.26%, y from 66.9%
        # the sampled bar stays: it shrinks into the lower right and returns at "units"
        SX0, SX1, SY = 1.2, 6.4, -2.55
        sbar = VGroup(s_frame, s_pos, s_neg)
        sbar_target = sbar.copy().stretch_to_fit_width(SX1 - SX0).stretch_to_fit_height(0.3).move_to([(SX0 + SX1) / 2, SY, 0])
        sx = lambda q: SX0 + (SX1 - SX0) * q  # noqa: E731
        s_val_small = mtex(pct1(Q_MEAS), size=SIZE_SMALL, color=INK).next_to([sx(Q_MEAS), SY + 0.15, 0], UP, buff=0.1)
        bars = VGroup(s_lab, sign_keys, cond, r_frame, r_pos, r_neg, r_div, r_lab, r_val2, approx_tag, gap, gap_edges)
        self.play_at("fit", FadeOut(bars, shift=DOWN * 0.3), FadeOut(g_grp), run_time=0.35)
        self.play(Transform(sbar, sbar_target), ReplacementTransform(s_val, s_val_small),
                  Create(ax), FadeIn(x_nums), FadeIn(y_nums), FadeIn(x_lab), FadeIn(y_lab),
                  run_time=0.65)
        guide = VGroup(DashedLine(ax.c2p(STRONG["logit_q_eq"], 0), strong_pt[2].get_center(),
                                  dash_length=0.05, color=INK_3, stroke_width=1.5),
                       DashedLine(ax.c2p(0, STRONG["logit_q_chain"]), strong_pt[2].get_center(),
                                  dash_length=0.05, color=INK_3, stroke_width=1.5))
        self.play(FadeIn(strong_pt, scale=1.6), Create(guide),
                  LaggedStart(*[FadeIn(q, scale=1.6) for q in pts if q is not strong_pt], lag_ratio=0.12),
                  run_time=0.7)

        # the slope: c, measured (gold), with its band from resampling pattern sets
        X_END = 4.0
        lo, hi, mid = MEAS["lower"], MEAS["upper"], MEAS["point"]
        band = VGroup(Polygon(ax.c2p(0, 0), ax.c2p(X_END, X_END * lo), ax.c2p(X_END, X_END * hi)),
                      Polygon(ax.c2p(0, 0), ax.c2p(-X_END, -X_END * lo), ax.c2p(-X_END, -X_END * hi)))
        band.set_stroke(width=0).set_fill(GILT, opacity=0.28)
        slope = Line(ax.c2p(-X_END, -X_END * mid), ax.c2p(X_END, X_END * mid), stroke_width=3.5, color=GILT)
        self.play_at("slope", FadeOut(guide), Create(diag), FadeIn(diag_lab), run_time=1.0)
        self.play(Create(slope), FadeIn(band), run_time=1.6)
        self.bring_to_front(pts)

        RX = 3.45
        c_read = mtex("c", "=", f3(mid), size=SIZE_HEAD + 4, colors={"c": GILT_TEXT})
        c_read[2].set_color(GILT_TEXT)
        c_read_lab = label("measured", size=SIZE_LABEL, color=INK_2)
        c_head = VGroup(c_read_lab, c_read).arrange(RIGHT, buff=0.3, aligned_edge=DOWN).move_to([RX, 1.35, 0])
        self.play_at("measured", Write(c_read), FadeIn(c_read_lab), Indicate(slope, color=GILT_TEXT, scale_factor=1.0),
                     run_time=1.1)
        ci = mtex(r"95\%:\ [\,", f4(lo), r",\ ", f4(hi), r"\,]", size=SIZE_LABEL, color=INK_2).next_to(c_head, DOWN, buff=0.25)
        ci_tag = label("from resampling the 20 pattern sets", size=SIZE_SMALL).next_to(ci, DOWN, buff=0.2)
        self.play_at("interval", Write(ci), FadeIn(ci_tag), band.animate.set_fill(opacity=0.42), run_time=1.1)
        self.play(band.animate.set_fill(opacity=0.28), run_time=0.5)

        # the prediction lands on the measured slope
        pline = theory(Line(ax.c2p(-X_END, -X_END * PRED), ax.c2p(X_END, X_END * PRED)), num_dashes=40)
        p_read = VGroup(label("predicted", size=SIZE_LABEL, color=THEORY), mtex(f3(PRED), size=SIZE_HEAD, color=THEORY)
                        ).arrange(RIGHT, buff=0.3, aligned_edge=DOWN)
        p_read.move_to(c_head).shift(DOWN * 0.0)
        self.play_at("agree", Indicate(pred_eq[4], color=THEORY, scale_factor=1.25),
                     FadeOut(ci_tag), run_time=0.8)
        p_read.next_to(ci, DOWN, buff=0.3).align_to(c_head, RIGHT)
        p_box = theory(SurroundingRectangle(p_read[1], buff=0.1, corner_radius=0.05), num_dashes=22)
        self.play(TransformFromCopy(pred_eq[4], p_read[1]), FadeIn(p_read[0]), Create(pline), Create(p_box), run_time=1.2)

        # the same agreement as shares at the strongest field, on the share-beat bar: 66.9% measured, 67.1% predicted
        m_tick = Line([sx(Q_MEAS), SY - 0.3, 0], [sx(Q_MEAS), SY + 0.15, 0], stroke_width=3, color=INK)
        m_lab = VGroup(mtex(pct1(Q_MEAS), size=SIZE_SMALL, color=INK), label("measured", size=SIZE_SMALL, color=INK_2)
                       ).arrange(RIGHT, buff=0.12).next_to(m_tick, DOWN, buff=0.06).align_to(m_tick, LEFT).shift(LEFT * 0.05)
        p_tick = theory(Line([sx(Q_PRED), SY - 0.15, 0], [sx(Q_PRED), SY + 0.4, 0]), num_dashes=4)
        p_lab = VGroup(mtex(pct1(Q_PRED), size=SIZE_SMALL, color=THEORY), label("predicted", size=SIZE_SMALL, color=THEORY)
                       ).arrange(RIGHT, buff=0.12).next_to(p_tick, UP, buff=0.06).align_to(p_tick, LEFT).shift(LEFT * 0.05)
        strip_head = VGroup(label("positive share at", size=SIZE_SMALL), mtex("G", "=", "0.0006125", size=SIZE_SMALL, color=INK_3)
                            ).arrange(RIGHT, buff=0.15).next_to(sbar, UP, buff=0.62).set_x((SX0 + SX1) / 2)
        strip_head[1][0].set_color(BIAS)
        self.play_at("units", FadeIn(strip_head, shift=DOWN * 0.1), run_time=0.9)
        self._wait_to(self.w("sixty-seven") - 0.05)
        self.play(Create(p_tick), FadeIn(p_lab), run_time=0.8)
        self._wait_to(self.w("sixty-six", self.t("units")) - 0.05)
        self.play(Create(m_tick), ReplacementTransform(s_val_small, m_lab[0]), FadeIn(m_lab[1]), run_time=0.8)

        # ============================================================ beat: one-pattern
        # the fit becomes a c axis: measured interval (solid) and the prediction (dashed triangle)
        CY = 0.85
        caxis = NumberLine(x_range=[0.18, 0.22, 0.01], length=10, color=RULE, stroke_width=2.5,
                           include_numbers=False, tick_size=0.07).move_to([0, CY, 0])
        CX = lambda v: caxis.n2p(v)[0]  # noqa: E731
        c_nums = VGroup(*[mtex(f"{v:.2f}", size=SIZE_SMALL, color=INK_3).next_to(caxis.n2p(v), DOWN, buff=0.15)
                          for v in (0.18, 0.19, 0.20, 0.21, 0.22)])
        c_sym = mtex("c", size=SIZE_LABEL + 4, color=GILT_TEXT).next_to(caxis, RIGHT, buff=0.25)
        TY = CY + 0.55
        ibar = interval_bar(CX(lo), CX(hi), TY)
        idot = Dot([CX(mid), TY, 0], radius=0.08, color=INK)
        i_lab = label("measured, 95%", size=SIZE_SMALL, color=INK_2).next_to(ibar, LEFT, buff=0.25)
        i_ends = VGroup(mtex(f4(lo), size=SIZE_SMALL, color=INK_2).next_to(ibar[1], DOWN, buff=0.12).shift(LEFT * 0.0),
                        mtex(f4(hi), size=SIZE_SMALL, color=INK_2).next_to(ibar[2], DOWN, buff=0.12))
        tri = tri_marker().move_to([CX(PRED), TY + 0.42, 0])
        tri_lab = label("predicted", size=SIZE_SMALL, color=THEORY).next_to(tri, UP, buff=0.12)
        # the gold measured value becomes the interval's dot and the dashed prediction box becomes the triangle
        # (t = 0, they carry the eye); the rest of the fit fades out in 0.45 s, and the axis text starts at 0.4 s
        fit_all = VGroup(ax, x_nums, y_nums, x_lab, y_lab, diag, diag_lab, pts, band, slope, pline,
                         c_read_lab, c_read[0], c_read[1], ci, p_read, sbar, strip_head, m_tick, m_lab, p_tick, p_lab)
        D = 0.4
        self.play_at("one-pattern", AnimationGroup(
            FadeOut(fit_all, shift=DOWN * 0.3, run_time=0.45), FadeOut(pred_tags, run_time=0.45),
            pred.animate(run_time=1.0).scale(0.72).to_corner(UL, buff=0.45),
            ReplacementTransform(c_read[2], idot, run_time=1.0), ReplacementTransform(p_box, tri, run_time=1.0),
            Succession(Wait(D), AnimationGroup(Create(caxis), FadeIn(c_nums), FadeIn(c_sym), FadeIn(ibar),
                                               FadeIn(i_lab), FadeIn(i_ends), FadeIn(tri_lab), run_time=0.65))))
        self.bring_to_front(idot)

        # one pattern, and a configuration aligned with it (seeded illustration)
        rng2 = np.random.default_rng(1)
        NS = 16
        xi = rng2.choice([-1, 1], size=NS)
        xi[:2] = (1, -1)
        PXL = 0.42
        SPX0 = -(NS - 1) * PXL * 1.12 / 2
        xs = [SPX0 + k * PXL * 1.12 for k in range(NS)]
        P_Y, S_Y = -0.95, -1.75
        pat = VGroup(*[Square(PXL, stroke_width=0, fill_opacity=1, fill_color=LIGHT if v > 0 else DARK).move_to([x, P_Y, 0])
                       for v, x in zip(xi, xs)])
        pat_frame = SurroundingRectangle(pat, buff=0.06, stroke_color=RULE, stroke_width=1.5)
        spins = VGroup(*[spin_arrow(v, [x, S_Y, 0]) for v, x in zip(xi, xs)])
        row_labs = VGroup(label("pattern", size=SIZE_SMALL).next_to(pat_frame, LEFT, buff=0.3),
                          label("spins", size=SIZE_SMALL).next_to(spins, LEFT, buff=0.3).set_x(0))
        row_labs[1].align_to(row_labs[0], RIGHT)
        model_lab = label("Hopfield, one pattern", size=SIZE_LABEL, color=INK).move_to([0, -2.6, 0])
        self.play_at("no-competition", FadeIn(pat), Create(pat_frame), FadeIn(row_labs),
                     LaggedStart(*[GrowArrow(a) for a in spins], lag_ratio=0.03), FadeIn(model_lab), run_time=1.3)

        # flip every spin where the pattern is -1: the pattern state becomes "all spins +1"
        neg = [k for k in range(NS) if xi[k] < 0]
        flips = []
        for k in neg:
            new = spin_arrow(1, [xs[k], S_Y, 0])
            flips.append(Transform(spins[k], new, path_arc=PI))
            flips.append(pat[k].animate.set_fill(LIGHT))
        self.play_at("flip", LaggedStart(*[Indicate(pat[k], color=MODE_B, scale_factor=1.15) for k in neg],
                                         lag_ratio=0.05), run_time=1.0)
        self._wait_to(self.w("minus") - 0.05)
        self.play(LaggedStart(*flips, lag_ratio=0.03), run_time=1.4)
        cw_name = label("Curie–Weiss", size=SIZE_LABEL, color=INK).move_to(model_lab)
        self._wait_to(self.w("curie–weiss") - 0.05)
        self.play(TransformMatchingShapes(model_lab, cw_name), run_time=0.9)
        foot = tex(r"model: $1/N$ vs $1/(N-1)$ \ $\cdot$ \ sampler: update order", size=SIZE_SMALL,
                   color=INK_3).next_to(cw_name, DOWN, buff=0.22)
        self._wait_to(self.w("small", self.t("flip")) - 0.05)
        self.play(FadeIn(foot, shift=UP * 0.1), run_time=0.8)

        # the one-pattern ratio
        ratio = VGroup(label("one pattern, measured:", size=SIZE_LABEL, color=INK_2),
                       mtex("c", "=", f3(R1), r"\times", r"c_{\mathrm{CW}}", size=SIZE_HEAD))
        ratio[1][0].set_color(GILT_TEXT)
        ratio[1][4].set_color(GILT_TEXT)
        ratio.arrange(RIGHT, buff=0.3).move_to([0, -1.0, 0])
        self.play_at("ratio", FadeOut(VGroup(pat, pat_frame, spins, row_labs, foot), shift=DOWN * 0.2),
                     FadeTransform(cw_name, ratio[1][4]), run_time=1.0)
        self.play(Write(ratio[0]), Write(ratio[1][:4]), run_time=1.2)

        # use it as the base: the header 0.120 x 1.662 = 0.199 comes down and its base is multiplied by 1.063
        new_eq = mtex(f3(R1), r"\times", "0.120", r"\times", "1.662", r"\;=\;", f3(PRED_R), size=SIZE_HEAD, color=INK)
        new_eq[2].set_color(GILT_TEXT)
        new_eq[6].set_color(THEORY)
        new_eq.move_to([0, -2.35, 0])
        base_brace = Brace(new_eq[:3], DOWN, buff=0.1, color=INK_3)
        base_lab = VGroup(label("one-pattern", size=SIZE_SMALL, color=INK_2), mtex("c", size=SIZE_SMALL + 4, color=GILT_TEXT)
                          ).arrange(RIGHT, buff=0.1, aligned_edge=DOWN)
        base_lab[1].shift(UP * 0.06)
        base_lab.next_to(base_brace, DOWN, buff=0.1)
        old_val = pred_eq[4].copy().scale(1 / 0.72).move_to(new_eq[6], aligned_edge=LEFT)
        box_down = pred_box.copy().scale(1 / 0.72, about_point=pred_eq[4].get_center()).shift(
            old_val.get_center() - pred_eq[4].get_center())
        box_new = theory(SurroundingRectangle(new_eq[6], buff=0.12, corner_radius=0.06), num_dashes=26)
        self.play_at("shift", Transform(pred_box, box_down),
                     ReplacementTransform(pred_eq[0], new_eq[2]), ReplacementTransform(pred_eq[1], new_eq[3]),
                     ReplacementTransform(pred_eq[2], new_eq[4]), ReplacementTransform(pred_eq[3], new_eq[5]),
                     pred_eq[4].animate.scale(1 / 0.72).move_to(old_val), run_time=1.0)
        self.play(TransformFromCopy(ratio[1][2], new_eq[0]), FadeIn(new_eq[1]),
                  FadeIn(base_brace), FadeIn(base_lab), run_time=1.0)
        ghost = tri.copy().set_stroke(opacity=0.35)
        self.add(ghost)
        self._wait_to(self.w("zero", self.t("shift")) - 0.05)
        tri_lab2 = mtex(f3(PRED_R), size=SIZE_SMALL, color=THEORY)
        self.play(tri.animate.move_to([CX(PRED_R), TY + 0.42, 0]), ReplacementTransform(pred_eq[4], new_eq[6]),
                  Transform(pred_box, box_new), FadeOut(tri_lab), run_time=1.4)
        tri_lab2.next_to(tri, UP, buff=0.12)
        self.play(FadeIn(tri_lab2), run_time=0.4)
        drop = theory(Line(tri.get_bottom() + DOWN * 0.02, [CX(PRED_R), TY - 0.02, 0]), num_dashes=4)
        self.play_at("outside", Create(drop), Indicate(VGroup(ibar[2], i_ends[1]), color=INK, scale_factor=1.3),
                     run_time=0.9)

        # the original match: two approximations that can cancel -- the Curie-Weiss base (too low, by 1.063)
        # and the competition factor (also approximate)
        brace = BraceBetweenPoints([ghost.get_left()[0], TY + 1.0, 0], [tri.get_right()[0], TY + 1.0, 0],
                                   direction=UP, color=INK_3)
        cancel = label("the match at 0.199 can come from approximations that cancel", size=SIZE_SMALL, color=INK_2
                       ).next_to(brace, UP, buff=0.12)
        if cancel.get_right()[0] > 6.8:
            cancel.shift(LEFT * (cancel.get_right()[0] - 6.8))
        comp_brace = Brace(new_eq[4], DOWN, buff=0.1, color=INK_3)
        comp_lab = label("competition: approximate", size=SIZE_SMALL, color=INK_2).next_to(comp_brace, DOWN, buff=0.1)
        comp_lab.align_to(base_lab, DOWN)
        if comp_lab.get_left()[0] < base_lab.get_right()[0] + 0.3:
            comp_lab.shift(RIGHT * (base_lab.get_right()[0] + 0.3 - comp_lab.get_left()[0]))
        self.play_at("cancel", FadeIn(brace), FadeIn(cancel, shift=UP * 0.1),
                     ghost.animate.set_stroke(opacity=0.7), run_time=0.9)
        self._wait_to(self.w("approximations", self.t("cancel")) - 0.05)
        self.play(FadeIn(comp_brace), FadeIn(comp_lab), Indicate(base_lab, color=INK, scale_factor=1.1), run_time=0.9)

        # ============================================================ beat: rows
        # Table 1: predicted (dashed triangle) against measured (solid interval), six settings
        TX0, TX1 = -2.0, 4.4
        tX = lambda v: TX0 + (TX1 - TX0) * (v - 0.10) / 0.20  # noqa: E731
        LAB_X, GAP_X = -6.3, 5.55
        specs = [dict(lab=f"p = {r['p']}", sub=r"$N = 2{,}048$", pred=r["c_predicted"], meas=r["c_measured"],
                      lo=r["ci95"][0], hi=r["ci95"][1]) for r in T1["rows"]]
        for r in SS["rows"]:
            if r["N"] != 2048:
                specs.append(dict(lab=f"N = {r['N']:,}".replace(",", "{,}"), sub=r"$p = 3$", pred=r["c_predicted"],
                                  meas=r["c_measured"], lo=r["ci95"][0], hi=r["ci95"][1]))
        ROW_Y = [2.0, 1.25, 0.5, -0.25, -1.3, -2.05]
        rows = []
        for s, y in zip(specs, ROW_Y):
            lab = VGroup(mtex(s["lab"], size=SIZE_LABEL, color=INK), tex(s["sub"], size=SIZE_SMALL, color=INK_3)
                         ).arrange(RIGHT, buff=0.2, aligned_edge=DOWN)
            lab.move_to([LAB_X, y, 0], aligned_edge=LEFT)
            track = Line([TX0, y, 0], [TX1, y, 0], stroke_width=1, color=GRID)
            ib = interval_bar(tX(s["lo"]), tX(s["hi"]), y, cap=0.12, sw=3)
            d = Dot([tX(s["meas"]), y, 0], radius=0.065, color=INK)
            t = tri_marker(0.12).move_to([tX(s["pred"]), y + 0.24, 0])
            inside = s["lo"] <= s["pred"] <= s["hi"]
            v = round(100 * (s["meas"] / s["pred"] - 1))
            g = (label("inside", size=SIZE_SMALL, color=INK_2) if inside
                 else mtex(("+" if v > 0 else "-") + f"{abs(v)}\\%", size=SIZE_LABEL, color=INK))
            g.move_to([GAP_X, y, 0])
            rows.append(dict(lab=lab, track=track, ib=ib, dot=d, tri=t, gap=g, y=y, inside=inside, v=v))
        assert [r["v"] for r in rows] == [6, 11, 1, -4, -7, 6] and [r["inside"] for r in rows] == [False, False, True, True, False, False]

        T_AX_Y = -2.75
        t_axis = NumberLine(x_range=[0.10, 0.30, 0.05], length=TX1 - TX0, color=RULE, stroke_width=2.5,
                            include_numbers=False, tick_size=0.07).move_to([(TX0 + TX1) / 2, T_AX_Y, 0])
        t_nums = VGroup(*[mtex(f"{v:.2f}", size=SIZE_SMALL, color=INK_3).next_to(t_axis.n2p(v), DOWN, buff=0.15)
                          for v in (0.10, 0.15, 0.20, 0.25, 0.30)])
        t_c = mtex("c", size=SIZE_LABEL + 4, color=GILT_TEXT).next_to(t_axis, RIGHT, buff=0.25)
        head_y = 2.85
        key = VGroup(tri_marker(0.1), label("predicted", size=SIZE_SMALL, color=THEORY),
                     VGroup(interval_bar(0, 0.5, 0, cap=0.1, sw=2.5), Dot([0.25, 0, 0], radius=0.05, color=INK)),
                     label("measured, 95%", size=SIZE_SMALL, color=INK_2)).arrange(RIGHT, buff=0.15)
        key[2].shift(RIGHT * 0.3)
        key[3].shift(RIGHT * 0.3)
        key.move_to([(TX0 + TX1) / 2, head_y, 0])
        h_gap = VGroup(label("measured", size=SIZE_SMALL), label("vs predicted", size=SIZE_SMALL),
                       label("(+ = measured above)", size=SIZE_SMALL, color=INK_3)
                       ).arrange(DOWN, buff=0.05)
        h_gap[2].scale(0.85)
        h_gap.arrange(DOWN, buff=0.05).move_to([GAP_X, head_y, 0])
        if h_gap.get_right()[0] > 6.75:
            h_gap.shift(LEFT * (h_gap.get_right()[0] - 6.75))
        t_tag = tex(r"separate runs \ $\cdot$ \ both field directions pooled \ $\cdot$ \ Curie--Weiss model's share as reference"
                    r" \ $\cdot$ \ so $p = 3$ differs slightly from the fit", size=SIZE_SMALL, color=INK_3).move_to([0, 3.5, 0])
        if t_tag.width > 13.6:
            t_tag.scale_to_fit_width(13.6)
        sep = DashedLine([LAB_X, (ROW_Y[3] + ROW_Y[4]) / 2, 0], [GAP_X + 0.6, (ROW_Y[3] + ROW_Y[4]) / 2, 0],
                         dash_length=0.05, color=RULE, stroke_width=1)

        # the measured interval and the 0.199 mark (the ghost) become the p = 3 row of Table 1
        p3 = rows[2]
        old = [i_lab, i_ends, tri, tri_lab2, drop, brace, cancel, ratio, new_eq, pred_box,
               base_brace, base_lab, comp_brace, comp_lab]
        # one play: the old parts fade out in the first 0.5 s while the interval, the dot, the ghost and the axis
        # carry the eye (t = 0); the new text starts at 0.45 s, after the old text is gone (no pile-up)
        others = [r for r in rows if r is not p3]
        D = 0.45
        self.play_at("rows", AnimationGroup(
            *[FadeOut(m, shift=UP * 0.2, run_time=0.5) for m in old],
            ReplacementTransform(ibar, p3["ib"], run_time=1.0), ReplacementTransform(idot, p3["dot"], run_time=1.0),
            ReplacementTransform(ghost, p3["tri"], run_time=1.0),
            ReplacementTransform(caxis, t_axis, run_time=1.0), FadeOut(c_nums, run_time=0.5),
            ReplacementTransform(c_sym, t_c, run_time=1.0),
            Succession(Wait(D), AnimationGroup(FadeIn(p3["lab"], shift=RIGHT * 0.2), FadeIn(p3["track"]),
                                               FadeIn(t_nums), run_time=0.8)),
            Succession(Wait(D), FadeIn(VGroup(key, h_gap), run_time=1.0)),
            Succession(Wait(D + 0.15), AnimationGroup(FadeIn(t_tag), Create(sep), run_time=1.3)),
            Succession(Wait(D), LaggedStart(*[FadeIn(VGroup(r["lab"], r["track"], r["ib"], r["dot"], r["tri"]),
                                                    shift=RIGHT * 0.2) for r in others],
                                           lag_ratio=0.15, run_time=1.6))))
        self.remove(*old)
        for r in rows:
            self.bring_to_front(r["dot"])

        self.play_at("misses", LaggedStart(*[Write(r["gap"]) for r in rows], lag_ratio=0.12), run_time=1.4)
        r2 = rows[1]
        hl = SurroundingRectangle(VGroup(r2["lab"], r2["gap"]), buff=0.12, corner_radius=0.06,
                                  stroke_color=INK_2, stroke_width=1.5)
        self._wait_to(self.w("eleven") - 0.05)
        self.play(Create(hl), Indicate(r2["gap"], color=INK, scale_factor=1.3),
                  Indicate(VGroup(r2["tri"], r2["dot"]), color=INK, scale_factor=1.3), run_time=1.0)

        # the gain from competition: predictions and measurements both rise from p = 1 to p = 4
        p_path = theory(VMobject().set_points_as_corners([r["tri"].get_center() for r in rows[:4]]), num_dashes=24)
        m_path = VMobject(stroke_color=INK, stroke_width=2).set_points_as_corners([r["dot"].get_center() for r in rows[:4]])
        trend = VGroup(label("more patterns", size=SIZE_SMALL, color=INK_2),
                       VGroup(label("→ larger", size=SIZE_SMALL, color=INK_2),
                              mtex("c", size=SIZE_SMALL + 4, color=GILT_TEXT)).arrange(RIGHT, buff=0.1)).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
        trend.move_to([3.45, (ROW_Y[0] + ROW_Y[1]) / 2, 0])
        self.play_at("answer", FadeOut(hl), Create(p_path), Create(m_path), FadeIn(trend, shift=DOWN * 0.1), run_time=1.6)
        for r in rows:
            self.bring_to_front(r["dot"])
        # the errors: pattern count (p rows), size (N rows), update rule (the p = 1 row)
        self._wait_to(self.w("size", self.t("answer")) - 0.05)
        self.play(Indicate(VGroup(rows[4]["gap"], rows[5]["gap"]), color=INK, scale_factor=1.25), run_time=0.8)
        self._wait_to(self.w("update", self.t("answer")) - 0.05)
        self.play(Indicate(rows[0]["gap"], color=INK, scale_factor=1.25), run_time=0.8)
