"""Scene "modes" (chapter "modes" of the visual edition, components/amplify.js): competing patterns.

Beats (script.md, scene modes): hopfield -> overlap -> condition -> assume -> pull -> amp -> predict.

Numbers and shapes:
  CW tiles              seeded +-1 tiles at m = +-0.891 (tanh(1.6 m) = m, the final temperature)
  patterns, sample      seeded random +-1 tiles (16 x 16 = 256 spins, an illustration of N = 2048)
  sample overlaps       computed from the sample tile itself (179 agree, 77 disagree: m1 = (179-77)/256 = 0.40)
  population strip      60 seeded draws of the independent-Gaussian picture at a = 0.208 (Fig. 1b),
                        tagged "illustration"; only the labels are shown
  bells, draws          standard Gaussians; pattern 1 shifted by 0.6 sd ("enlarged to be visible")
  log-odds at size r    2 r delta / sigma^2 (paper App. C), drawn with the same enlarged delta
  histograms            60,000 seeded Gaussian triples; dashed curves are the exact densities
                        2 phi(r) (all draws) and 6 phi(r) (2 Phi(r) - 1)^2 (draws pattern 1 wins)
  1.662 (A_3)           E[max_3 |Z|] / E|Z| by quadrature, checked against visual/data/table1.json; shown as
                        the ratio of the two mean lines (E|Z| = the one-pattern, no-competition reference)
  c_CW = 0.120          visual/data/c_cw.json (0.119808); prediction 0.120 x 1.662 = 0.199
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import *  # noqa: E402,F403

DATA = Path(__file__).resolve().parents[2] / "data"

LIGHT = "#D9D7D0"   # spin +1
DARK = "#2C2D31"    # spin -1
SIDE = 16
DELTA = 0.6         # field shift of overlap 1, in sd units, enlarged to be visible (tagged)


# ------------------------------------------------------------------ numbers
def phi(x):
    return np.exp(-np.asarray(x) ** 2 / 2) / math.sqrt(2 * math.pi)


def Phi(x):
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2)))


def amp_exact(p: int) -> float:
    """A_p = E[max_p |Z|] / E|Z| = int_0^inf 1 - (2Phi(x)-1)^p dx / sqrt(2/pi)."""
    x = np.linspace(0, 12, 120001)
    f = 1 - (2 * Phi(x) - 1) ** p
    return float(np.trapezoid(f, x) / math.sqrt(2 / math.pi))


C_CW = json.loads((DATA / "c_cw.json").read_text())["c_cw"]          # 0.119808
TABLE1 = json.loads((DATA / "table1.json").read_text())
A3 = amp_exact(3)
assert abs(A3 - next(r for r in TABLE1["rows"] if r["p"] == 3)["amplification"]) < 1e-6
C_PRED = C_CW * A3
assert f"{C_CW:.3f}" == "0.120" and f"{A3:.3f}" == "1.662" and f"{C_PRED:.3f}" == "0.199"
MEAN_ALL = math.sqrt(2 / math.pi)                                      # E|Z| = 0.798
MEAN_WIN = A3 * MEAN_ALL                                               # E max_3 |Z| = 1.326


def cw_mstar(beta: float) -> float:
    m = 0.9
    for _ in range(200):
        m = math.tanh(beta * m)
    return m


M_CW = cw_mstar(1.6)                                                   # 0.891


# ------------------------------------------------------------------ tiles
def make_tile(vals, size, border=RULE, bw=1.5):
    s = size / SIDE
    px = VGroup()
    for k, v in enumerate(vals):
        i, j = divmod(k, SIDE)
        sq = Square(side_length=s * 1.04, stroke_width=0, fill_color=LIGHT if v > 0 else DARK, fill_opacity=1)
        sq.move_to(np.array([(j - (SIDE - 1) / 2) * s, ((SIDE - 1) / 2 - i) * s, 0]))
        px.add(sq)
    frame = Square(side_length=size, stroke_color=border, stroke_width=bw, fill_opacity=0)
    return VGroup(px, frame)


class Modes(NarratedScene):
    scene_id = "modes"

    def word(self, w: str, n: int = 1) -> float:
        """Start time of the n-th spoken word that matches w (punctuation ignored)."""
        k = 0
        for x in self.timing["words"]:
            if x["w"].strip(".,:;!?").lower() == w.lower():
                k += 1
                if k == n:
                    return x["start"]
        raise KeyError(w)

    def word_after(self, w: str, t: float) -> float:
        for x in self.timing["words"]:
            if x["start"] > t and x["w"].strip(".,:;!?").lower() == w.lower():
                return x["start"]
        raise KeyError((w, t))

    def construct(self):
        rng = np.random.default_rng(7)
        pats = rng.choice([-1, 1], size=(3, SIDE * SIDE))

        # ============================================================ Curie-Weiss: one choice
        TS = 1.3
        ROOT = np.array([-0.2, 2.95, 0])
        NY, LY = 1.75, 0.05
        XS = [-4.0, -0.2, 3.6]
        OFF = 0.85

        flips = np.random.default_rng(5).random(SIDE * SIDE) < (1 - M_CW) / 2
        cw_plus = np.where(flips, -1, 1)
        cw_p = make_tile(cw_plus, TS, border=MODE_A, bw=4).move_to([XS[1] - OFF, LY, 0])
        cw_m = make_tile(-cw_plus, TS, border=MODE_B, bw=4).move_to([XS[1] + OFF, LY, 0])
        cw_node = Dot([XS[1], NY, 0], radius=0.07, color=INK_2)
        cw_edges = VGroup(*[Line(cw_node.get_center(), t.get_top(), stroke_color=INK_3, stroke_width=2.5)
                            for t in (cw_p, cw_m)])
        cw_lab = VGroup(mtex("m > 0", size=SIZE_LABEL, color=MODE_A).next_to(cw_p, DOWN, buff=0.18),
                        mtex("m < 0", size=SIZE_LABEL, color=MODE_B).next_to(cw_m, DOWN, buff=0.18))
        head_cw = label("Curie–Weiss", size=SIZE_LABEL, color=INK_2).to_corner(UL, buff=0.55)
        one = label("one choice", size=SIZE_LABEL, color=INK_2).next_to(cw_node, RIGHT, buff=0.35).shift(0.25 * DOWN)

        self.play(FadeIn(head_cw), GrowFromCenter(cw_node), run_time=0.6)
        self.play(Create(cw_edges), FadeIn(cw_p, shift=0.15 * DOWN), FadeIn(cw_m, shift=0.15 * DOWN),
                  FadeIn(cw_lab), run_time=1.2)
        self.wait_until(self.word("one") - 0.1)
        self.play(FadeIn(one, shift=0.1 * LEFT), run_time=0.7)

        # ============================================================ Hopfield: patterns and negatives
        tiles = [make_tile(pats[k], TS, border=MODE_A if k == 0 else RULE, bw=4 if k == 0 else 1.5)
                 .move_to([XS[k] - OFF, LY, 0]) for k in range(3)]
        negs = [make_tile(-pats[k], TS, border=MODE_B if k == 0 else RULE, bw=4 if k == 0 else 1.5)
                .move_to([XS[k] + OFF, LY, 0]) for k in range(3)]
        plabs = VGroup(*[label(f"pattern {k + 1}", size=SIZE_SMALL, color=MODE_A if k == 0 else INK_2)
                         .move_to([XS[k] - OFF, LY - TS / 2 - 0.3, 0]) for k in range(3)])
        nlabs = VGroup(*[label("negative", size=SIZE_SMALL, color=MODE_B if k == 0 else INK_3)
                         .move_to([XS[k] + OFF, LY - TS / 2 - 0.3, 0]) for k in range(3)])
        head_h = label("Hopfield model", size=SIZE_LABEL, color=INK_2).to_corner(UL, buff=0.55)

        self.play_at("hopfield",
                     FadeOut(VGroup(cw_node, cw_edges, cw_p, cw_m, cw_lab, one), shift=0.3 * UP),
                     ReplacementTransform(head_cw, head_h), run_time=0.8)
        self.play(LaggedStart(*[LaggedStart(*[FadeIn(sq, scale=0.6) for sq in tiles[k][0]], lag_ratio=0.003,
                                            run_time=1.3) for k in range(3)], lag_ratio=0.3),
                  LaggedStart(*[Create(tiles[k][1]) for k in range(3)], lag_ratio=0.3),
                  LaggedStart(*[FadeIn(l) for l in plabs], lag_ratio=0.3), run_time=2.2)

        # "at low temperature, configurations settle near one pattern or its negative"
        self.play_at("pattern-modes",
                     LaggedStart(*[TransformFromCopy(tiles[k], negs[k]) for k in range(3)], lag_ratio=0.2),
                     LaggedStart(*[FadeIn(l) for l in nlabs], lag_ratio=0.2), run_time=1.8)

        tag = tex(r"$p = 3$ patterns, $N = 2048$ spins; each tile shows 256 of them", size=SIZE_SMALL + 2,
                  color=INK_3).to_corner(DL, buff=0.4)
        self.play_at("p3", FadeIn(tag), run_time=0.8)

        # "a run makes two choices: which pattern, and which sign."
        root = Dot(ROOT, radius=0.08, color=INK)
        nodes = VGroup(*[Dot([XS[k], NY, 0], radius=0.07, color=INK_2) for k in range(3)])
        e1 = VGroup(*[Line(ROOT, nodes[k].get_center(), stroke_color=INK_3, stroke_width=2.5) for k in range(3)])
        e2 = VGroup(*[Line(nodes[k].get_center(), t.get_top(), stroke_color=INK_3, stroke_width=2.5)
                      for k in range(3) for t in (tiles[k], negs[k])])
        e1[0].set_stroke(MODE_A, width=3.5)
        e2[0].set_stroke(MODE_A, width=3.5)
        e2[1].set_stroke(MODE_B, width=3.5)
        q1 = label("which pattern?", size=SIZE_LABEL, color=INK_2)
        q1.move_to([2.0, (ROOT[1] + NY) / 2 + 0.1, 0], aligned_edge=LEFT)
        q2 = label("which sign?", size=SIZE_LABEL, color=INK_2)
        q2.move_to([XS[2] + 0.95, (NY + LY + TS / 2) / 2 - 0.05, 0], aligned_edge=LEFT)
        self.play_at("two-choices", GrowFromCenter(root), run_time=0.5)
        self.wait_until(self.word("which") - 0.1)
        self.play(Create(e1), FadeIn(nodes), FadeIn(q1, shift=0.1 * RIGHT), run_time=0.9)
        self.wait_until(self.word("which", 2) - 0.1)
        self.play(Create(e2), FadeIn(q2, shift=0.1 * RIGHT), run_time=0.9)

        # "Does competition change how much preference survives?"
        qy = -2.35
        c_cw = mtex("c", "=", "0.120", size=SIZE_BODY, color=GILT_TEXT)
        c_cw[1].set_color(INK)
        c_cw_l = label("Curie–Weiss", size=SIZE_SMALL, color=INK_3)
        arr = mtex(r"\longrightarrow", size=SIZE_BODY, color=INK_3)
        c_q = mtex("c", "=", r"\mathord{?}", size=SIZE_BODY, color=GILT_TEXT)
        c_q[1].set_color(INK)
        c_q[2].set_color(INK_2)
        c_q_l = label("three patterns", size=SIZE_SMALL, color=INK_3)
        qrow = VGroup(c_cw, arr, c_q).arrange(RIGHT, buff=0.5).move_to([-0.2, qy, 0])
        c_cw_l.next_to(c_cw, DOWN, buff=0.14)
        c_q_l.next_to(c_q, DOWN, buff=0.14)
        self.play_at("question", FadeIn(c_cw), FadeIn(c_cw_l), run_time=0.8)
        self.play(GrowFromEdge(arr, LEFT), FadeIn(c_q, shift=0.2 * LEFT), FadeIn(c_q_l), run_time=0.9)

        # ============================================================ overlap
        ML, LYO = 4.0, -1.3
        sample_v = pats[0].copy()
        flip = np.random.default_rng(11).choice(SIDE * SIDE, size=77, replace=False)  # m1 = 1 - 154/256 = 0.40
        sample_v[flip] *= -1
        mvals = [float(np.mean(sample_v * pats[k])) for k in range(3)]
        assert abs(mvals[0] - 0.398) < 0.001 and max(abs(mvals[1]), abs(mvals[2])) < 0.15
        ST = 1.75
        sample = make_tile(sample_v, ST).move_to([0, 1.45, 0])
        s_lab = label("a sample", size=SIZE_SMALL, color=INK_2).next_to(sample, UP, buff=0.14)
        line = Line([-ML, LYO, 0], [ML, LYO, 0], stroke_color=INK_3, stroke_width=3)
        ticks = VGroup(*[Line([x, LYO - 0.12, 0], [x, LYO + 0.12, 0], stroke_color=INK_3, stroke_width=2)
                         for x in (-ML, 0, ML)])
        zero_n = mtex("0", size=SIZE_LABEL, color=INK_3).next_to([0, LYO - 0.12, 0], DOWN, buff=0.12)
        m1n = mtex("-1", size=SIZE_LABEL, color=MODE_B).next_to([-ML, LYO - 0.12, 0], DOWN, buff=0.12)
        p1n = mtex("+1", size=SIZE_LABEL, color=MODE_A).next_to([ML, LYO - 0.12, 0], DOWN, buff=0.12)
        ov_lab = label("overlap with pattern 1", size=SIZE_LABEL, color=INK_2).move_to([0, LYO + 0.55, 0])
        mv = ValueTracker(mvals[0])
        mdot = always_redraw(lambda: Dot([mv.get_value() * ML, LYO, 0], radius=0.12, color=INK).set_stroke(BG, 2))

        self.play_at("overlap",
                     FadeOut(VGroup(root, nodes, e1, e2, q1, q2, qrow, c_cw_l, c_q_l, head_h, *tiles[1:], *negs[1:],
                                    plabs, nlabs)),
                     tiles[0].animate.move_to([ML, 0.15, 0]), negs[0].animate.move_to([-ML, 0.15, 0]),
                     run_time=1.0)
        self.play(FadeIn(sample, shift=0.2 * UP), FadeIn(s_lab), Create(line), FadeIn(ticks), FadeIn(zero_n),
                  FadeIn(ov_lab), run_time=0.8)
        self.add(mdot)
        self.play(FadeIn(mdot, scale=0.4), run_time=0.3)
        # "its average agreement with it": an agreeing spin counts +1 (blue), a disagreeing one -1 (orange)
        s = ST / SIDE
        agree = VGroup(*[Square(side_length=s * 0.92, stroke_width=0,
                                fill_color=MODE_A if sample_v[k] == pats[0][k] else MODE_B, fill_opacity=0.8)
                         .move_to(sq) for k, sq in enumerate(sample[0])])
        n_ag = int(np.sum(sample_v == pats[0]))
        assert n_ag == 179 and abs(mvals[0] - (179 - 77) / 256) < 1e-12
        frac = VGroup(VGroup(label("179 agree", size=SIZE_SMALL, color=MODE_A),
                             mtex("(+1)", size=SIZE_SMALL + 2, color=MODE_A)).arrange(RIGHT, buff=0.15),
                      VGroup(label("77 disagree", size=SIZE_SMALL, color=MODE_B),
                             mtex("(-1)", size=SIZE_SMALL + 2, color=MODE_B)).arrange(RIGHT, buff=0.15))
        frac.arrange(DOWN, buff=0.18, aligned_edge=LEFT).next_to(sample, RIGHT, buff=0.35)
        arith = mtex(r"\frac{179-77}{256}", "=", "0.40", size=SIZE_LABEL, color=INK)
        arith.next_to([mvals[0] * ML, LYO - 0.2, 0], DOWN, buff=0.15).shift(0.9 * RIGHT)
        self.play(LaggedStart(*[FadeIn(a) for a in agree], lag_ratio=0.003), FadeIn(frac), run_time=0.8)
        self.play(Write(arith), run_time=0.8)

        # "from minus one at the negative to plus one at the pattern"
        neg_full = make_tile(-pats[0], ST).move_to(sample)
        pos_full = make_tile(pats[0], ST).move_to(sample)
        keep = sample.copy()
        self.play_at("ends", FadeOut(agree), FadeIn(m1n), FadeIn(p1n), run_time=0.5)
        self.wait_until(self.word("minus") - 0.05)
        # only the reference tile's frame pulses; its pixels keep their values
        self.play(Transform(sample, neg_full), mv.animate.set_value(-1.0), FadeOut(frac), FadeOut(arith),
                  Indicate(negs[0][1], color=MODE_B, scale_factor=1.08), run_time=1.1)
        self.wait_until(self.word("plus") - 0.05)
        self.play(Transform(sample, pos_full), mv.animate.set_value(1.0),
                  Indicate(tiles[0][1], color=MODE_A, scale_factor=1.08), run_time=1.1)
        self.wait_until(self.word("pattern", 3) + 0.4)
        self.play(Transform(sample, keep), mv.animate.set_value(mvals[0]), run_time=0.7)

        # ============================================================ label: three overlaps
        LX0, LX1 = -3.0, 5.0
        LC, LH = (LX0 + LX1) / 2, (LX1 - LX0) / 2
        RY = [1.2, -0.6, -2.4]
        lines = VGroup(*[Line([LX0, y, 0], [LX1, y, 0], stroke_color=INK_3, stroke_width=3) for y in RY])
        mini = [make_tile(pats[k], 0.62, border=MODE_A if k == 0 else RULE, bw=3 if k == 0 else 1.2)
                .move_to([-4.1, RY[k], 0]) for k in range(3)]
        mini_l = VGroup(*[label(f"{k + 1}", size=SIZE_SMALL, color=MODE_A if k == 0 else INK_2)
                          .next_to(mini[k], LEFT, buff=0.12) for k in range(3)])
        rticks = VGroup(*[Line([LC + x * LH, y - 0.1, 0], [LC + x * LH, y + 0.1, 0], stroke_color=INK_3,
                               stroke_width=2) for y in RY for x in (-1, 0, 1)])
        rnums = VGroup(mtex("-1", size=SIZE_SMALL + 2, color=INK_3).next_to([LX0, RY[2] - 0.1, 0], DOWN, buff=0.1),
                       mtex("0", size=SIZE_SMALL + 2, color=INK_3).next_to([LC, RY[2] - 0.1, 0], DOWN, buff=0.1),
                       mtex("+1", size=SIZE_SMALL + 2, color=INK_3).next_to([LX1, RY[2] - 0.1, 0], DOWN, buff=0.1))
        top_lab = label("overlap with each pattern", size=SIZE_LABEL, color=INK_2).move_to([LC, 2.2, 0])
        sm = make_tile(sample_v, 1.3).move_to([-6.0, -0.3, 0])
        sm_lab = label("a sample", size=SIZE_SMALL, color=INK_2).next_to(sm, UP, buff=0.14)
        dots = VGroup(*[Dot([LC + mvals[k] * LH, RY[k], 0], radius=0.11, color=INK).set_stroke(BG, 2)
                        for k in range(3)])

        self.play_at("label",
                     FadeOut(VGroup(m1n, p1n, zero_n, ticks, ov_lab, s_lab)), FadeOut(mdot),
                     ReplacementTransform(line, lines[0]), ReplacementTransform(sample, sm),
                     ReplacementTransform(tiles[0], mini[0]), FadeOut(negs[0], scale=0.5),
                     run_time=1.1)
        self.remove(mdot)
        self.add(dots[0])
        self.play(Create(lines[1:]), FadeIn(rticks), FadeIn(rnums), FadeIn(mini[1]), FadeIn(mini[2]),
                  FadeIn(mini_l), FadeIn(top_lab), FadeIn(sm_lab), *[FadeIn(d, scale=0.4) for d in dots[1:]], run_time=1.0)
        # "largest overlap in size": the other rows dim, pattern 1's dot is ringed
        self.wait_until(self.word("largest") - 0.05)
        ring = Circle(radius=0.22, stroke_color=MODE_A, stroke_width=4).move_to(dots[0])
        def dim(op, with_dots=True):
            return [lines[1:].animate.set_stroke(opacity=op), mini_l[1:].animate.set_opacity(op),
                    *([dots[1:].animate.set_opacity(op)] if with_dots else []),
                    *[m.animate.set_fill(opacity=op) for k in (1, 2) for m in mini[k][0]],
                    *[mini[k][1].animate.set_stroke(opacity=op) for k in (1, 2)]]

        self.play(*dim(0.3), Create(ring), run_time=0.8)
        # "and its sign": the label (1, +)
        self.wait_until(self.word("sign") - 0.1)
        tag1 = mtex("(1,+)", size=SIZE_BODY, color=MODE_A).next_to(sm, DOWN, buff=0.25)
        self.play(TransformFromCopy(dots[0], tag1), run_time=0.8)

        # ============================================================ condition
        G_ARR_Y = RY[0] + 0.42
        g_arr = Arrow([LC - 0.1, G_ARR_Y, 0], [LC + 1.5, G_ARR_Y, 0], buff=0, color=BIAS, stroke_width=5,
                      max_tip_length_to_length_ratio=0.18)
        g_lab = VGroup(label("field", size=SIZE_SMALL, color=BIAS), mtex("G", size=SIZE_LABEL, color=BIAS))
        g_lab.arrange(RIGHT, buff=0.1).next_to(g_arr, RIGHT, buff=0.15)
        self.play_at("field1", GrowArrow(g_arr), FadeIn(g_lab), FadeOut(tag), run_time=0.8)

        # population: 60 seeded draws of the independent-Gaussian picture; only the labels are shown
        pz = np.random.default_rng(2).standard_normal((60, 3))
        pz[:, 0] += 0.208
        pw = np.abs(pz).argmax(1)
        psg = np.sign(pz[np.arange(60), pw])
        n1, n1p = int((pw == 0).sum()), int(((pw == 0) & (psg > 0)).sum())
        assert (n1, n1p) == (19, 12)
        PDX, PDY, PDS = -2.55, 3.55, 0.31      # population rows: left x, top y, spacing (free space above the rows)
        pdots = VGroup()
        for i in range(60):
            col = (MODE_A if psg[i] > 0 else MODE_B) if pw[i] == 0 else NOISE
            rr, cc = divmod(i, 30)
            pdots.add(Dot([PDX + PDS * cc, PDY - 0.34 * rr, 0], radius=0.11, color=col))
        all_l = VGroup(label("all samples", size=SIZE_SMALL, color=INK_2),
                       label("illustration", size=SIZE_SMALL - 4, color=INK_3)).arrange(DOWN, buff=0.08,
                                                                                       aligned_edge=RIGHT)
        all_l.next_to(pdots, LEFT, buff=0.35)
        self.play_at("condition", LaggedStart(*[FadeIn(d, scale=0.3) for d in pdots], lag_ratio=0.02),
                     FadeIn(all_l), FadeOut(rnums), run_time=1.0)
        # "only among samples labelled pattern one": the others dim, pattern 1's labels drop to their own row
        self.wait_until(self.word("only") - 0.05)
        others = VGroup(*[pdots[i] for i in range(60) if pw[i] != 0])
        ones = [i for i in range(60) if pw[i] == 0]
        self.play(others.animate.set_opacity(0.15), run_time=0.6)
        order = sorted(ones, key=lambda i: -psg[i])
        SY, SS = 2.72, 0.36
        x0 = LC + 0.4 - SS * (len(order) - 1) / 2
        moves = [pdots[i].animate.move_to([x0 + SS * j, SY, 0]).scale(1.15) for j, i in enumerate(order)]
        lp = mtex("(1,+)", size=SIZE_SMALL + 4, color=MODE_A).next_to([x0, SY, 0], LEFT, buff=0.3)
        lm = mtex("(1,-)", size=SIZE_SMALL + 4, color=MODE_B).next_to([x0 + SS * (len(order) - 1), SY, 0], RIGHT,
                                                                      buff=0.3)
        one_l = label("labelled pattern 1", size=SIZE_SMALL, color=MODE_A).next_to(lp, LEFT, buff=0.3)
        self.play(*moves, FadeIn(lp), FadeIn(lm), FadeIn(one_l), run_time=1.0)

        # ============================================================ assume: independent Gaussians
        U = LH / 3.4                      # units per sd on the overlap axes
        zt = [ValueTracker(mvals[k] * LH / U) for k in range(3)]
        shift1 = ValueTracker(0.0)
        bell_on = ValueTracker(0.0)
        win_on = ValueTracker(0.0)
        BH = 1.15

        def zdots():
            v = np.array([t.get_value() for t in zt])
            w = int(np.argmax(np.abs(v)))
            g = VGroup()
            for k in range(3):
                if win_on.get_value() > 0.5 and k == w:
                    col = (MODE_A if v[0] >= 0 else MODE_B) if k == 0 else INK
                    d = Dot([LC + v[k] * U, RY[k], 0], radius=0.13, color=col).set_stroke(BG, 2)
                    g.add(d, Circle(radius=0.24, stroke_color=col, stroke_width=3.5).move_to(d))
                else:
                    op = 0.45 if win_on.get_value() > 0.5 else 1.0
                    g.add(Dot([LC + v[k] * U, RY[k], 0], radius=0.1, color=INK_2, fill_opacity=op))
            return g

        def bells():
            a = bell_on.get_value()
            g = VGroup()
            for k in range(3):
                sh = shift1.get_value() if k == 0 else 0.0
                xs = np.linspace(-3.4, 3.4, 90)
                pts = [[LC + x * U, RY[k] + BH * a * phi(x - sh) / phi(0), 0] for x in xs]
                g.add(DashedVMobject(VMobject().set_points_smoothly(pts).set_stroke(THEORY, 2.2), num_dashes=46,
                                     dashed_ratio=0.55))
            return g

        a_tag = label("approximation", size=SIZE_SMALL, color=INK_3).move_to([LC, 3.5, 0])
        b_tag = VGroup(mtex(r"\beta \approx 1", size=SIZE_LABEL, color=INK_2),
                       label("where the choice is made", size=SIZE_SMALL, color=INK_3)).arrange(RIGHT, buff=0.25)
        b_tag.move_to([LC, 2.95, 0])
        ind_tag = label("independent Gaussian overlaps", size=SIZE_LABEL, color=INK_2).move_to([LC, 2.95, 0])
        zlab = label("early overlap, in units of its spread", size=SIZE_SMALL, color=INK_3)
        zlab.move_to([LC, RY[2] - 0.55, 0])
        znums = VGroup(*[mtex(str(v) if v <= 0 else f"+{v}", size=SIZE_SMALL + 2, color=INK_3)
                         .next_to([LC + v * U, RY[2] - 0.1, 0], DOWN, buff=0.08) for v in (-3, -2, -1, 0, 1, 2, 3)])
        zlab.next_to(znums, DOWN, buff=0.1)
        zticks = VGroup(*[Line([LC + v * U, y - 0.08, 0], [LC + v * U, y + 0.08, 0], stroke_color=INK_3,
                               stroke_width=2) for y in RY for v in (-3, -2, -1, 0, 1, 2, 3)])
        zdots_m = always_redraw(zdots)
        bells_m = always_redraw(bells)

        self.play_at("assume",
                     FadeOut(VGroup(pdots, all_l, one_l, lp, lm, tag1, sm, sm_lab, g_arr, g_lab, ring, top_lab, rticks)),
                     FadeOut(dots), *dim(1.0, with_dots=False), FadeIn(a_tag), run_time=1.0)
        self.remove(dots, *dots)
        self.add(zdots_m)
        # "Near beta equal to one, where the choice is made": early overlaps are small, near zero
        self.wait_until(self.word("near") - 0.05)
        early = np.random.default_rng(12).standard_normal(3)          # sd 1: the axis is in units of the spread
        self.play(FadeIn(b_tag), FadeIn(zticks), FadeIn(znums), FadeIn(zlab),
                  *[zt[k].animate.set_value(early[k]) for k in range(3)], run_time=1.4)
        # "the three overlaps are independent Gaussians"
        self.wait_until(self.word("three", 2) - 0.05)
        self.add(bells_m)
        self.play(bell_on.animate.set_value(1.0), run_time=1.4)
        self.wait_until(self.word("independent") - 0.05)
        self.play(FadeOut(b_tag, shift=0.15 * UP), FadeIn(ind_tag, shift=0.15 * UP), run_time=0.8)

        # "Pattern one's is shifted by the field, as in Curie-Weiss."
        g2 = Arrow([LC, RY[0] - 0.4, 0], [LC + DELTA * U + 0.05, RY[0] - 0.4, 0], buff=0, color=BIAS,
                   stroke_width=5, max_tip_length_to_length_ratio=0.35)
        g2l = mtex("G", size=SIZE_LABEL, color=BIAS).next_to(g2, RIGHT, buff=0.12)
        enl = label("shift enlarged to be visible", size=SIZE_SMALL - 2, color=INK_3)
        enl.next_to(g2l, RIGHT, buff=0.35)
        self.play_at("shift", shift1.animate.set_value(DELTA), GrowArrow(g2), FadeIn(g2l), run_time=1.2)
        self.play(FadeIn(enl), run_time=0.5)

        # "The largest in size decides the final pattern." seeded draws; the winner is ringed
        # four draws with a clear winner (margin in size > 0.4 sd), so no draw reads as a tie
        draws = np.random.default_rng(36).standard_normal((40, 3))
        draws[:, 0] += DELTA
        srt = np.sort(np.abs(draws), 1)
        draws = draws[srt[:, -1] - srt[:, -2] > 0.4][:4]
        assert list(np.abs(draws).argmax(1)) == [1, 2, 2, 0] and draws[3, 0] > 0
        self.mark("winner")
        win_on.set_value(1.0)
        for i in range(4):
            self.play(*[zt[k].animate.set_value(draws[i, k]) for k in range(3)], run_time=0.4)
            self.wait(0.5)

        # ============================================================ pull: one shifted Gaussian
        PX0, PW = -5.9, 6.4               # big panel: x from PX0 to PX0 + PW, 1 unit per sd
        PC = PX0 + PW / 2
        BY = -2.0                          # baseline
        HS = 2.7 / phi(0)                  # density -> height
        LY0, LS = 1.15, 0.42               # log panel: apex height and units per log unit
        r = ValueTracker(1.3)
        mode = ValueTracker(0.0)           # 0 = density, 1 = log density

        def fx(x):
            return PC + x * 1.0

        def dens_y(x, sh=DELTA):
            return BY + HS * phi(x - sh)

        def log_y(x, sh=DELTA):
            return LY0 - LS * (x - sh) ** 2 / 2

        def curve_y(x):
            t = mode.get_value()
            return (1 - t) * dens_y(x) + t * log_y(x)

        def curve():
            # the Gaussian picture is the paper's approximation: theory, so dashed light grey
            xs = np.linspace(-3.2, 3.2, 120)
            return DashedVMobject(VMobject().set_points_smoothly([[fx(x), curve_y(x), 0] for x in xs])
                                  .set_stroke(THEORY, 3.5), num_dashes=70, dashed_ratio=0.6)

        def marks():
            rv = r.get_value()
            g = VGroup()
            for sgn, col in ((1, MODE_A), (-1, MODE_B)):
                x = sgn * rv
                top = curve_y(x)
                g.add(DashedLine([fx(x), BY, 0], [fx(x), top, 0], dash_length=0.07, stroke_color=col,
                                 stroke_width=3, stroke_opacity=1 - mode.get_value()))
                g.add(Dot([fx(x), top, 0], radius=0.09, color=col))
            return g

        curve_m = always_redraw(curve)
        marks_m = always_redraw(marks)
        base = Line([PX0, BY, 0], [PX0 + PW, BY, 0], stroke_color=RULE, stroke_width=2)
        zero_t = VGroup(Line([PC, BY - 0.1, 0], [PC, BY + 0.1, 0], stroke_color=INK_3, stroke_width=2),
                        mtex("0", size=SIZE_SMALL + 2, color=INK_3).next_to([PC, BY - 0.1, 0], DOWN, buff=0.08))
        rlabs = always_redraw(lambda: VGroup(
            mtex("+r", size=SIZE_SMALL + 4, color=MODE_A).next_to([fx(r.get_value()), BY - 0.1, 0], DOWN, buff=0.08),
            mtex("-r", size=SIZE_SMALL + 4, color=MODE_B).next_to([fx(-r.get_value()), BY - 0.1, 0], DOWN, buff=0.08)))
        hlab = always_redraw(lambda: VGroup(
            mtex("h_+", size=SIZE_LABEL, color=MODE_A).next_to([fx(r.get_value()), curve_y(r.get_value()), 0],
                                                              UR, buff=0.16),
            mtex("h_-", size=SIZE_LABEL, color=MODE_B).next_to([fx(-r.get_value()), curve_y(-r.get_value()), 0],
                                                              UL, buff=0.16)))
        size_lab = VGroup(label("size", size=SIZE_SMALL, color=INK_3), mtex("r", size=SIZE_SMALL + 4, color=INK_3),
                          label(": the same on both sides", size=SIZE_SMALL, color=INK_3)).arrange(RIGHT, buff=0.1)
        size_lab[2].shift(0.06 * LEFT)
        size_lab.next_to(base, DOWN, buff=0.55)
        p_tag = label("illustration: one Gaussian, field shift enlarged to be visible", size=SIZE_SMALL - 2,
                      color=INK_3).move_to([PC, 3.45, 0])

        rows_rest = VGroup(lines, mini[0], mini[1], mini[2], mini_l, zticks, znums, zlab, g2, g2l, enl, a_tag, ind_tag)
        start_curve = DashedVMobject(VMobject().set_points_smoothly(
            [[LC + x * U, RY[0] + BH * phi(x - DELTA) / phi(0), 0] for x in np.linspace(-3.2, 3.2, 120)]
        ).set_stroke(THEORY, 3.5), num_dashes=70, dashed_ratio=0.6)
        self.play_at("magnitude", FadeOut(rows_rest), FadeOut(zdots_m), FadeOut(bells_m), FadeIn(start_curve),
                     run_time=0.6)
        self.remove(zdots_m, bells_m)
        self.play(Transform(start_curve, curve()), Create(base), FadeIn(zero_t), FadeIn(p_tag), run_time=1.1)
        self.remove(start_curve)
        self.add(curve_m)
        self.play(FadeIn(marks_m), FadeIn(rlabs), FadeIn(hlab), FadeIn(size_lab), run_time=1.0)

        # "The log of their ratio is the sign's log-odds."
        RX = 3.9
        f1 = mtex(r"\log\frac{h_+}{h_-}", "=", r"\text{log-odds of the sign}", size=SIZE_LABEL + 4, color=INK)
        f1.move_to([RX, 2.45, 0])
        self.play_at("log-ratio", Write(f1), run_time=1.3)

        # "On a log scale the Gaussian is a parabola,": the same Gaussian without the field is the
        # symmetric reference (dimmer dashes)
        def cen_y(x):
            return log_y(x, 0.0)

        cen = DashedVMobject(VMobject().set_points_smoothly(
            [[fx(x), cen_y(x), 0] for x in np.linspace(-3.2, 3.2, 120)]).set_stroke(THEORY, 2.2, opacity=0.55),
            num_dashes=60, dashed_ratio=0.45)
        log_lab = label("log scale", size=SIZE_SMALL, color=INK_3).move_to([PX0 + 0.8, LY0 + 0.25, 0])
        with_l = label("with field", size=SIZE_SMALL - 2, color=THEORY).next_to([PX0 + PW, log_y(3.2), 0], RIGHT,
                                                                                buff=0.15)
        cen_l = label("no field", size=SIZE_SMALL - 2, color=INK_3).next_to([PX0 + PW, cen_y(3.2), 0], RIGHT, buff=0.15)
        self.play_at("parabola", mode.animate.set_value(1.0), FadeOut(size_lab), FadeIn(log_lab), run_time=1.6)
        self.wait_until(self.word("parabola") - 0.1)
        self.play(Create(cen), FadeIn(cen_l), FadeIn(with_l), run_time=0.8)

        # "and the shift adds a straight tilt.": with field = no field + a straight line in r
        # (log phi(x - d) = log phi(x) + d x - d^2/2); the line is drawn through the parabola's centre and the
        # two BIAS arrows at +r and -r are its value there, from the dashed reference to the curve
        def tilt_y(x):
            return LY0 + LS * (DELTA * x - DELTA ** 2 / 2)

        tilt = DashedLine([fx(-3.2), tilt_y(-3.2), 0], [fx(3.2), tilt_y(3.2), 0], dash_length=0.1,
                          stroke_color=BIAS, stroke_width=3)
        tilt_l = VGroup(mtex("+", size=SIZE_LABEL, color=BIAS),
                        label("a straight tilt from the field", size=SIZE_SMALL, color=BIAS)).arrange(RIGHT, buff=0.15)
        tilt_l.next_to([fx(3.2), tilt_y(3.2), 0], DR, buff=0.12).shift(0.25 * DOWN)
        RT = 2.0

        def bias_arrow(x):
            return Arrow([fx(x), cen_y(x), 0], [fx(x), log_y(x), 0], buff=0, color=BIAS, stroke_width=5,
                         max_tip_length_to_length_ratio=0.32, max_stroke_width_to_length_ratio=20)

        # the faint horizontal reference is the added term's zero, the no-field apex level LY0: the arrows
        # measure the line against it (LS*(d x - d^2/2)), which is exactly the parabola gap at x, so the
        # move onto the gap is a pure translation
        ref = DashedLine([fx(-3.2), LY0, 0], [fx(3.2), LY0, 0], dash_length=0.06,
                         stroke_color=BIAS, stroke_width=1.6, stroke_opacity=0.45)

        def line_arrow(x):
            return Arrow([fx(x), LY0, 0], [fx(x), tilt_y(x), 0], buff=0, color=BIAS, stroke_width=5,
                         max_tip_length_to_length_ratio=0.32, max_stroke_width_to_length_ratio=20)

        self.play_at("tilt", r.animate.set_value(RT), Create(tilt), FadeIn(ref), FadeIn(tilt_l), run_time=0.9)
        arr_p, arr_m = line_arrow(RT), line_arrow(-RT)
        self.play(GrowArrow(arr_p), GrowArrow(arr_m), run_time=0.7)
        self.play(Transform(arr_p, bias_arrow(RT)), Transform(arr_m, bias_arrow(-RT)),
                  tilt.animate.set_stroke(opacity=0.35), ref.animate.set_stroke(opacity=0.15), run_time=0.8)

        # "So the gap between the two sides grows in proportion to the size.": the parabola is symmetric, so the
        # arrow at -r slides across under the one at +r; together they span the gap between the two heights
        BAR_Y, BAR_X0, BAR_S = 0.55, 1.2, 1.3     # units per log unit for the log-odds bar
        GX = 0.25

        def gap():
            rv = r.get_value()
            yp, ym = log_y(rv), log_y(-rv)
            return VGroup(DashedLine([fx(-rv), ym, 0], [fx(rv) + GX, ym, 0], dash_length=0.07, stroke_color=INK_3,
                                     stroke_width=2),
                          Line([fx(rv) + GX, ym, 0], [fx(rv) + GX, yp, 0], stroke_color=MODE_A, stroke_width=7))

        def lo_bar():
            L = 2 * DELTA * r.get_value() * BAR_S
            return Rectangle(width=max(L, 1e-3), height=0.3, stroke_width=0, fill_color=MODE_A,
                             fill_opacity=0.95).move_to([BAR_X0, BAR_Y, 0], aligned_edge=LEFT)

        bar_m = always_redraw(lo_bar)
        bar_l = f1[2].copy().scale(0.8).next_to([BAR_X0, BAR_Y + 0.15, 0], UP, buff=0.12)
        bar_l.align_to([BAR_X0, 0, 0], LEFT)
        prop = mtex(r"\propto", r"r", size=SIZE_LABEL + 2, color=INK).next_to(bar_l, RIGHT, buff=0.25)
        self.play_at("grows", arr_m.animate.shift((2 * RT + GX) * RIGHT), arr_p.animate.shift(GX * RIGHT),
                     run_time=0.9)
        gap_static = gap()
        self.play(FadeIn(gap_static), FadeOut(VGroup(arr_p, arr_m)), FadeIn(bar_m),
                  TransformFromCopy(f1[2], bar_l), FadeOut(VGroup(tilt, ref, tilt_l, cen, cen_l, with_l)), run_time=0.7)
        self.remove(gap_static)
        gap_m = always_redraw(gap)
        self.add(gap_m)
        # small size first, so that the gap and the bar grow on "grows"
        self.play(r.animate.set_value(0.6), run_time=0.3)
        self.wait_until(self.word("grows") - 0.1)
        self.play(r.animate.set_value(2.7), run_time=1.3)
        self.wait_until(self.word("proportion") + 0.3)
        self.play(Write(prop), run_time=0.8)

        # "Pattern one wins only by beating the other two, so its winning overlaps are larger than usual."
        # The grey reference is one overlap with no rival: the one-pattern (Curie-Weiss-like) case whose
        # c the prediction multiplies.
        ax = Axes(x_range=[0, 3.6, 1], y_range=[0, 0.85, 0.85], x_length=6.2, y_length=3.0,
                  axis_config={"color": RULE, "stroke_width": 2, "include_ticks": True, "tick_size": 0.05},
                  tips=False)
        ax.shift(np.array([-6.3, -2.4, 0]) - ax.c2p(0, 0))
        xnums = VGroup(*[mtex(str(v), size=SIZE_SMALL + 2, color=INK_3).next_to(ax.c2p(v, 0), DOWN, buff=0.1)
                         for v in range(4)])
        xlab = label("size of overlap 1, in units of its spread", size=SIZE_SMALL, color=INK_3).next_to(xnums, DOWN,
                                                                                                    buff=0.08)
        xlab.set_x(ax.c2p(1.8, 0)[0])
        mc = np.abs(np.random.default_rng(3).standard_normal((60000, 3)))
        won = mc.argmax(1) == 0
        edges = np.linspace(0, 3.6, 19)
        h_all, _ = np.histogram(mc[:, 0], bins=edges, density=True)
        h_won, _ = np.histogram(mc[won, 0], bins=edges, density=True)

        def hist(h, col, op):
            g = VGroup()
            for i, hv in enumerate(h):
                bl, tr = ax.c2p(edges[i], 0), ax.c2p(edges[i + 1], min(hv, 0.85))
                g.add(Rectangle(width=tr[0] - bl[0] - 0.03, height=max(tr[1] - bl[1], 1e-3), stroke_width=0,
                                fill_color=col, fill_opacity=op).move_to((np.array(bl) + np.array(tr)) / 2))
            return g

        hg_all, hg_won = hist(h_all, NOISE, 0.55), hist(h_won, MODE_A, 0.45)
        cur_all = theory(ax.plot(lambda t: 2 * phi(t), x_range=[0, 3.6]).set_stroke(width=2.5), num_dashes=40)
        cur_won = theory(ax.plot(lambda t: 6 * phi(t) * (2 * float(Phi(t)) - 1) ** 2, x_range=[0, 3.6])
                         .set_stroke(width=2.5), num_dashes=40)
        lab_all = label("one pattern (no competition)", size=SIZE_SMALL, color=INK_2)
        lab_all.move_to(ax.c2p(1.85, 0.78), aligned_edge=LEFT)
        lab_won = label("pattern 1 won, against two rivals", size=SIZE_SMALL, color=MODE_A)
        lab_won.move_to(ax.c2p(1.85, 0.62), aligned_edge=LEFT)
        ml_all = Line(ax.c2p(MEAN_ALL, 0), ax.c2p(MEAN_ALL, 0.85), stroke_color=INK_2, stroke_width=3.5)
        ml_won = Line(ax.c2p(MEAN_WIN, 0), ax.c2p(MEAN_WIN, 0.85), stroke_color=MODE_A, stroke_width=3.5)
        log_panel = VGroup(base, zero_t, f1, log_lab, p_tag)
        at_avg = label("at the average size", size=SIZE_SMALL - 2, color=INK_3)
        at_avg.next_to([BAR_X0 + 2 * DELTA * MEAN_WIN * BAR_S, BAR_Y, 0], RIGHT, buff=0.3)
        # first clear the log panel completely; only the log-odds bar carries over
        self.play_at("winners", FadeOut(log_panel), FadeOut(curve_m), FadeOut(marks_m), FadeOut(rlabs),
                     FadeOut(hlab), FadeOut(gap_m), run_time=0.5)
        self.remove(curve_m, marks_m, rlabs, hlab, gap_m)
        self.play(Create(ax), FadeIn(xnums), FadeIn(xlab),
                  LaggedStart(*[GrowFromEdge(b, DOWN) for b in hg_all], lag_ratio=0.04), Create(cur_all),
                  FadeIn(lab_all), r.animate.set_value(MEAN_ALL), FadeIn(at_avg), run_time=1.2)
        self.wait_until(self.word("beating") - 0.1)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in hg_won], lag_ratio=0.04), Create(cur_won),
                  FadeIn(lab_won), run_time=1.4)
        self.wait_until(self.word("larger") - 0.1)
        self.play(Create(ml_all), Create(ml_won), run_time=0.8)

        # "On average they are one point six six two times larger,": the ratio of the two mean lines
        mean_arrow = Arrow(ax.c2p(MEAN_ALL, 0.9), ax.c2p(MEAN_WIN, 0.9), buff=0, stroke_width=3, color=INK_2,
                           max_tip_length_to_length_ratio=0.3)
        avg_l = label("average size", size=SIZE_SMALL, color=INK_2).next_to(mean_arrow, UP, buff=0.12)
        amp_v = mtex(r"\times\,1.662", size=SIZE_BODY, color=THEORY).next_to(avg_l, UP, buff=0.14)
        self.play_at("amp", FadeOut(prop), GrowArrow(mean_arrow), FadeIn(avg_l), run_time=1.0)
        ul = theory(Line(amp_v.get_corner(DL) + 0.08 * DOWN, amp_v.get_corner(DR) + 0.08 * DOWN)
                    .set_stroke(width=2.5), num_dashes=10)
        self.play_at("amp-value", Write(amp_v), Create(ul), run_time=1.0)

        # "so their sign's log-odds are too."
        x_lab = mtex(r"\times\,1.662", size=SIZE_LABEL + 2, color=INK).next_to([BAR_X0, BAR_Y, 0], DOWN, buff=0.3)
        x_lab.align_to([BAR_X0, 0, 0], LEFT)
        avg_tick = always_redraw(lambda: Line([BAR_X0 + 2 * DELTA * MEAN_ALL * BAR_S, BAR_Y - 0.25, 0],
                                              [BAR_X0 + 2 * DELTA * MEAN_ALL * BAR_S, BAR_Y + 0.25, 0],
                                              stroke_color=INK_2, stroke_width=2.5))
        self.add(avg_tick)
        self.play_at("scales", r.animate.set_value(MEAN_WIN), Indicate(amp_v, color=THEORY, scale_factor=1.12), Write(x_lab), run_time=1.3)

        # ============================================================ assumptions
        hist_grp = VGroup(ax, xnums, xlab, hg_all, hg_won, cur_all, cur_won, lab_all, lab_won, ml_all, ml_won,
                          mean_arrow)
        fy = 1.35
        f_c = mtex("c", "=", size=SIZE_HEAD, color=GILT_TEXT)
        f_c[1].set_color(INK)
        f_num = label("log-odds, sampler", size=SIZE_BODY, color=INK)
        f_den = label("log-odds, model", size=SIZE_BODY, color=INK)
        f_ln = Line(LEFT, RIGHT, stroke_color=INK, stroke_width=2.5).set_width(f_num.width + 0.3)
        f_frac = VGroup(f_num, f_ln, f_den).arrange(DOWN, buff=0.18)
        c_frac = VGroup(f_c, f_frac).arrange(RIGHT, buff=0.3).move_to([-0.6, fy, 0])
        f_c.set_y(fy)
        den_note = label("≈ Curie–Weiss", size=SIZE_LABEL, color=INK_2).next_to(f_den, RIGHT, buff=0.35)
        num_note = mtex(r"\times\,1.662", size=SIZE_BODY, color=INK).next_to(f_num, RIGHT, buff=0.5)
        t1 = label("independent Gaussian overlaps", size=SIZE_LABEL, color=INK_3)
        t2 = label("Hopfield preference ≈ Curie–Weiss", size=SIZE_LABEL, color=INK_3)
        tags = VGroup(t1, t2).arrange(DOWN, buff=0.16, aligned_edge=LEFT).move_to([-0.6, -2.55, 0])
        brace = Brace(tags, RIGHT, color=INK_3, buff=0.2)
        ng = label("not guaranteed", size=SIZE_LABEL, color=INK_2).next_to(brace, RIGHT, buff=0.2)

        self.play_at("assumptions", FadeOut(hist_grp), FadeOut(VGroup(amp_v, ul, avg_l, x_lab, bar_l, at_avg)),
                     FadeOut(bar_m), FadeOut(avg_tick), run_time=0.9)
        self.remove(bar_m, avg_tick)
        self.play(Write(f_c), FadeIn(f_frac), FadeIn(t1), run_time=1.0)
        # "the Hopfield model's own preference to be the Curie-Weiss one"
        self.wait_until(self.word_after("curie–weiss", self.t("assumptions")) - 0.4)
        self.play(FadeIn(den_note, shift=0.1 * LEFT), FadeIn(t2), Indicate(f_den, color=INK, scale_factor=1.08),
                  run_time=0.9)
        # "so the gain changes only the sampler's side of c"
        self.wait_until(self.word("gain") - 0.1)
        self.play(FadeIn(num_note, shift=0.1 * LEFT), run_time=0.7)
        self.wait_until(self.word("sampler's") - 0.1)
        self.play(Indicate(VGroup(f_num, num_note), color=INK, scale_factor=1.08), run_time=0.9)
        # "Neither assumption is guaranteed."
        self.wait_until(self.word("neither") - 0.05)
        self.play(GrowFromCenter(brace), FadeIn(ng, shift=0.1 * LEFT), run_time=0.8)

        # ============================================================ predict
        py_ = 0.85
        cpart = mtex("0.120", size=58, color=GILT_TEXT)
        times = mtex(r"\times", size=58, color=INK)
        apart = mtex("1.662", size=58, color=INK)
        eq = mtex("=", size=58, color=INK)
        res = mtex("0.199", size=58, color=GILT_TEXT)
        row = VGroup(cpart, times, apart, eq, res).arrange(RIGHT, buff=0.45).move_to([0, py_, 0])
        # the name sits under the number, not left of an '=' (0.120 is c for Curie-Weiss; 0.199 is the prediction);
        # set in one TeX line so the italic c shares the words' baseline, scaled to the other tags' word size
        cw_tag = mtex(r"\text{Curie--Weiss}", r"\ c", color=INK_3)
        cw_tag.scale(label("Curie–Weiss", size=SIZE_SMALL).width / cw_tag[0].width)
        cw_tag.next_to(cpart, DOWN, buff=0.2)
        gain_tag = label("competition", size=SIZE_SMALL, color=INK_3).next_to(apart, DOWN, buff=0.2)
        self.play_at("predict", FadeOut(VGroup(f_c, f_frac, den_note)), ReplacementTransform(num_note, apart),
                     FadeIn(cpart, shift=0.3 * RIGHT), VGroup(tags, brace, ng).animate.shift(0.15 * DOWN),
                     run_time=1.2)
        self.play(FadeIn(cw_tag), run_time=0.4)
        self.wait_until(self.word("times") - 0.1)
        self.play(Write(times), FadeIn(gain_tag), run_time=0.7)
        box = theory(SurroundingRectangle(res, buff=0.18).set_stroke(width=2.5), num_dashes=36)
        pred_lab = label("prediction, three patterns", size=SIZE_SMALL, color=THEORY).next_to(box, UP, buff=0.15)
        self.play_at("pred-value", Write(eq), Write(res), Create(box), FadeIn(pred_lab), run_time=1.0)

        # "Competition keeps more of the preference."
        BS = 20.0
        by1, by2 = -0.75, -1.3
        bx = -2.4
        b1 = Rectangle(width=C_CW * BS, height=0.32, stroke_width=0, fill_color=GILT, fill_opacity=0.95)
        b1.move_to([bx, by1, 0], aligned_edge=LEFT)
        b2 = theory(Rectangle(width=C_PRED * BS, height=0.32).set_stroke(GILT, 2.5), num_dashes=40)
        b2.set_stroke(GILT)
        b2.move_to([bx + C_PRED * BS / 2, by2, 0])
        b1l = label("Curie–Weiss", size=SIZE_SMALL, color=INK_2).next_to(b1, LEFT, buff=0.25)
        b2l = label("three patterns", size=SIZE_SMALL, color=INK_2).next_to([bx, by2, 0], LEFT, buff=0.25)
        b1l.align_to(b2l, RIGHT)
        self.play_at("answer", GrowFromEdge(b1, LEFT), FadeIn(b1l), run_time=0.6)
        self.play(Create(b2), FadeIn(b2l), run_time=0.9)
