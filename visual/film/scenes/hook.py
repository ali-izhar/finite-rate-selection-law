"""Scene hook: "The wrong proportions" (script.md, scene hook; page chapter overview).

Every number on screen is the paper's own or derived from it (script.md, Sources):
  * model 97 of 100 and sampler 61 of 100: q_eq = 0.97274 and q_chain = 0.60546 of the exact
    count calculation for the Curie-Weiss model at N = 2,048, G = 0.0006125, through the paper's
    protocol (visual/data/c_cw.json);
  * the cooling staircase: the paper's ten levels beta = 0.6 ... 1.6, drawn as temperature 1/beta,
    each step as wide as its share of the sampling time, in sweeps (15, 30, 59, 107, 355, 145, 65, 31, 16, 11 = 834);
  * 59 and 63 of 100: the same calculation with every level's sweeps x1/4 and x4
    (visual/data/cw_scaling.json, q 0.5875 and 0.6281).
As on the page (odds.js, paintArray), a grid of 100 shows round(100 q) blue squares first, in
reading order, so the samples the sampler loses are cells 61-96.

The digit tiles are a picture of the question (the paper's hypothetical "3 twice as likely as 8"),
drawn here from seeded strokes; they are tagged as not data.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from style import *  # noqa: E402,F403

DATA_DIR = HERE.parent / "data"

# ------------------------------------------------------------------ the paper's numbers
_ccw = json.loads((DATA_DIR / "c_cw.json").read_text())
_scal = {r["scale"]: r for r in json.loads((DATA_DIR / "cw_scaling.json").read_text())["schedule"]
         if r["N"] == 2048}
Q_EQ = _ccw["runs"][-1]["q_eq"]            # 0.9727
Q_CHAIN = _ccw["runs"][-1]["q_chain"]      # 0.6055
LADDER = _ccw["ladder"]                    # beta levels 0.6 ... 1.6
SWEEPS = _ccw["sweeps_per_level"]          # 15 ... 11
N_MODEL = round(100 * Q_EQ)                # 97
N_CHAIN = round(100 * Q_CHAIN)             # 61
N_QUARTER = round(100 * _scal[0.25]["q_model"])  # 59
N_FOUR = round(100 * _scal[4.0]["q_model"])      # 63
assert (N_MODEL, N_CHAIN, N_QUARTER, N_FOUR, sum(SWEEPS)) == (97, 61, 59, 63, 834)
assert abs(_scal[1.0]["q_model"] - Q_CHAIN) < 1e-9  # the x1 row is the paper's schedule

RNG = np.random.default_rng(1983)


# ------------------------------------------------------------------ digit tiles (a picture)
def digit_bitmap(d: str, rng: np.random.Generator, n: int = 28) -> np.ndarray:
    """A 28x28 stroke drawing of "3" or "8" with seeded handwriting jitter, values in [0, 1]."""
    slant, sx, sy = rng.uniform(-0.22, 0.12), rng.uniform(0.88, 1.06), rng.uniform(0.92, 1.04)
    w, dx = rng.uniform(1.05, 1.35), rng.uniform(-0.8, 0.8)
    pts = []
    if d == "3":
        for cx, cy, r, a0, a1 in ((13.6, 9.4, 4.8, 2.6, -1.57), (13.6, 18.6, 5.5, 1.57, -2.6)):
            t = np.linspace(a0, a1, 400)
            pts.append(np.c_[cx + r * np.cos(t), cy - r * np.sin(t)])
        pts.append(np.c_[np.linspace(11.2, 13.6, 40), np.full(40, 14.1)])
    else:
        th = np.linspace(0, 2 * np.pi, 720)
        for cx, cy, rx, ry in ((14.0, 9.3, 4.0, 4.3), (14.0, 18.7, 5.0, 5.0)):
            pts.append(np.c_[cx + rx * np.cos(th), cy + ry * np.sin(th)])
    P = np.vstack(pts)
    y = (P[:, 1] - 14) * sy
    x = (P[:, 0] - 14) * sx - slant * y
    P = np.c_[x + 14 + dx, y + 14]
    yy, xx = np.mgrid[0:n, 0:n] + 0.5
    dist = np.min(np.hypot(xx[..., None] - P[:, 0], yy[..., None] - P[:, 1]), axis=-1)
    return np.clip(1.6 * w - dist / 1.1, 0, 1)


def digit_tile(d: str, height: float, rng: np.random.Generator) -> Group:
    """A digit as a pixel image (nearest-neighbour, so the 28x28 grid shows) in a thin frame."""
    v = digit_bitmap(d, rng)
    rgb = np.array(ManimColor(MODE_A if d == "3" else MODE_B).to_rgb()) * 255
    img = np.zeros((28, 28, 4), np.uint8)
    img[..., :3] = rgb.astype(np.uint8)
    img[..., 3] = (255 * v).astype(np.uint8)
    im = ImageMobject(img)
    im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    im.height = height
    frame = SurroundingRectangle(im, buff=0.04 * height, stroke_color=RULE, stroke_width=1.5,
                                 corner_radius=0.03 * height)
    return Group(frame, im)


# ------------------------------------------------------------------ count grids (page, odds panel a)
CELL = 0.30
GAP = 0.055
STEP = CELL + GAP
FILL = 0.92  # near-full fill: at low opacity the orange turns brown on the dark ground


def make_grid(center):
    g = VGroup()
    for i in range(100):
        r, c = divmod(i, 10)
        g.add(Square(CELL, stroke_width=0).move_to(center + np.array([(c - 4.5) * STEP, -(r - 4.5) * STEP, 0])))
    return g


def paint(grid, n_pos):
    for i, sq in enumerate(grid):
        sq.set_fill(MODE_A if i < n_pos else MODE_B, opacity=FILL)
    return grid


# ------------------------------------------------------------------ the cooling staircase
def staircase_points(width: float, height: float, t_lo=0.55, t_hi=1.75):
    """Corner points of the paper's schedule: x = sweeps (scaled to width), y = temperature 1/beta."""
    temps = [1 / b for b in LADDER]
    xs = np.concatenate([[0], np.cumsum(SWEEPS)]) / sum(SWEEPS) * width
    ys = [(T - t_lo) / (t_hi - t_lo) * height for T in temps]
    pts = []
    for k, y in enumerate(ys):
        pts += [np.array([xs[k], y, 0]), np.array([xs[k + 1], y, 0])]
    return pts


def staircase(width, height, color=INK_2, stroke=3):
    return VMobject(stroke_color=color, stroke_width=stroke).set_points_as_corners(staircase_points(width, height))


class Hook(NarratedScene):
    scene_id = "hook"

    def word(self, w: str, n: int = 1) -> float:
        """Start time of the n-th spoken word that matches w (punctuation ignored)."""
        k = 0
        for x in self.timing["words"]:
            if x["w"].strip(".,:;!?").lower() == w.lower():
                k += 1
                if k == n:
                    return x["start"]
        raise KeyError(w)

    def construct(self):
        LC = np.array([-3.0, 0.6, 0])
        RC = np.array([3.0, 0.6, 0])

        # ---------------------------------------------------- beat digits
        big3 = digit_tile("3", 2.2, np.random.default_rng(3)).move_to(np.array([-2.3, 1.25, 0]))
        big8 = digit_tile("8", 2.2, np.random.default_rng(8)).move_to(np.array([2.3, 1.25, 0]))
        self.wait_until(0.25)
        self.play(FadeIn(big3, scale=0.95), FadeIn(big8, scale=0.95), run_time=1.0)

        ratio = mtex("2", ":", "1", size=60, color=INK_2).move_to(np.array([0, 1.25, 0]))
        ratio[0].set_color(MODE_A)
        ratio[2].set_color(MODE_B)
        self.play_at("twice", Write(ratio), run_time=0.8)

        # a correct sampler: about two 3s for every 8 (8 : 4 of 12), in a seeded order
        rng = np.random.default_rng(1983)
        order = ["3"] * 8 + ["8"] * 4
        rng.shuffle(order)
        H, X0, DXT = 0.62, -3.05, 0.76
        row1 = [digit_tile(d, H, rng).move_to(np.array([X0 + k * DXT, -1.05, 0])) for k, d in enumerate(order)]
        lab1 = label("correct sampler", size=SIZE_SMALL, color=INK_2)
        lab1.next_to(row1[0], LEFT, buff=0.35)
        tag_q = label("a picture of the question, not data", size=SIZE_SMALL, color=INK_3)
        tag_q.to_corner(DL, buff=0.45)
        self.play_at("should",
                     LaggedStart(*[FadeIn(t, shift=UP * 0.1) for t in row1], lag_ratio=0.12),
                     FadeIn(lab1), FadeIn(tag_q), run_time=1.6)

        # another sampler: clean digits again, but two of the 3s come out as 8s (6 : 6)
        row2 = [digit_tile(d, H, rng).move_to(np.array([X0 + k * DXT, -2.05, 0])) for k, d in enumerate(order)]
        lab2 = label("a sampler", size=SIZE_SMALL, color=INK_2)
        lab2.next_to(row2[0], LEFT, buff=0.35).align_to(lab1, RIGHT)
        self.play_at("wrong-ratio",
                     LaggedStart(*[FadeIn(t, shift=UP * 0.1) for t in row2], lag_ratio=0.12),
                     FadeIn(lab2), run_time=1.6)
        # the ratio of each row, at its end: 8 : 4 = 2 : 1 for both, until the flip makes it 6 : 6
        def row_ratio(a, b, row):
            r = mtex(a, ":", b, size=SIZE_LABEL, color=INK_3).next_to(row[-1], RIGHT, buff=0.3)
            r[0].set_color(MODE_A)
            r[2].set_color(MODE_B)
            return r
        rr1, rr2 = row_ratio("2", "1", row1), row_ratio("2", "1", row2)
        self.play(FadeIn(rr1), FadeIn(rr2), run_time=0.4)
        flip = [k for k, d in enumerate(order) if d == "3"][1::4][:2]   # two 3s, seeded positions
        new8 = [digit_tile("8", H, rng).move_to(row2[k]) for k in flip]
        self.wait_until(self.word("wrong") - 0.1)   # "... and still get that ratio wrong"
        rr2b = row_ratio("1", "1", row2)
        self.play(*[FadeOut(row2[k], scale=0.9) for k in flip], *[FadeIn(n, scale=1.1) for n in new8],
                  ReplacementTransform(rr2, rr2b), run_time=0.9)
        for k, n in zip(flip, new8):
            row2[k] = n

        # ---------------------------------------------------- beat grids
        # the digits give way to the model's two outcomes: positive (blue) and negative (orange)
        sq_pos = Square(1.5, stroke_width=0).set_fill(MODE_A, FILL).move_to(np.array([-1.7, 1.0, 0]))
        sq_neg = Square(1.5, stroke_width=0).set_fill(MODE_B, FILL).move_to(np.array([1.7, 1.0, 0]))
        self.play_at("exact-model",
                     FadeOut(Group(*row1, *row2, lab1, lab2, tag_q, ratio, rr1, rr2b)),
                     FadeOut(big3, scale=1.5 / 2.2), FadeIn(sq_pos, scale=2.2 / 1.5),
                     FadeOut(big8, scale=1.5 / 2.2), FadeIn(sq_neg, scale=2.2 / 1.5), run_time=1.0)
        lab_pos = label("positive", color=MODE_A).next_to(sq_pos, DOWN, buff=0.3)
        lab_neg = label("negative", color=MODE_B).next_to(sq_neg, DOWN, buff=0.3)
        self.wait_until(self.word("positive"))
        self.play(FadeIn(lab_pos, shift=UP * 0.1), run_time=0.5)
        self.wait_until(self.word("negative"))
        self.play(FadeIn(lab_neg, shift=UP * 0.1), run_time=0.5)
        tag_m = label("Curie–Weiss model · N = 2,048 · proportions computed exactly",
                      size=SIZE_SMALL, color=INK_3).to_corner(DL, buff=0.45)
        self.wait_until(self.word("exactly"))
        self.play(FadeIn(tag_m), run_time=0.7)

        # "Its sampler starts hot and lowers the temperature step by step": the paper's schedule,
        # each step as wide as its share of the sampling time; a dot rides it at a constant rate
        SW, SH = 4.2, 2.7
        o = np.array([1.2, -1.15, 0])
        ax_x = Line(o, o + RIGHT * (SW + 0.15), stroke_color=RULE, stroke_width=2)
        ax_y = Line(o, o + UP * (SH + 0.15), stroke_color=RULE, stroke_width=2)
        stairs = staircase(SW, SH).shift(o)
        lab_t = label("temperature", size=SIZE_SMALL, color=INK_3).next_to(ax_y, UP, buff=0.12).align_to(ax_y, LEFT)
        lab_hot = label("hot", size=SIZE_SMALL, color=INK_3).next_to(stairs.get_start(), LEFT, buff=0.15)
        lab_s = label("sampling time", size=SIZE_SMALL, color=INK_3).next_to(ax_x.get_end(), DOWN, buff=0.15).align_to(ax_x, RIGHT)
        two = Group(sq_pos, sq_neg, lab_pos, lab_neg)
        self.play_at("cools",
                     two.animate.scale(0.7).move_to(np.array([-3.6, 0.6, 0])),
                     Create(ax_x), Create(ax_y), FadeIn(lab_t), FadeIn(lab_hot), FadeIn(lab_s),
                     run_time=1.0)
        dot = Dot(stairs.get_start(), radius=0.07, color=INK)
        trace = stairs.copy().set_stroke(INK_2)
        self.play(Create(trace, rate_func=linear), MoveAlongPath(dot, stairs, rate_func=linear),
                  run_time=self.word("cooling") - self.now)
        lab_cool = label("cooling", size=SIZE_SMALL, color=INK_2).next_to(trace, UP, buff=0.12).shift(RIGHT * 0.8)
        self.play(FadeIn(lab_cool, shift=DOWN * 0.1), run_time=0.6)

        # "Of a hundred samples, ninety-seven should be positive"
        grid_m = paint(make_grid(LC), N_MODEL)
        head_m = text("model", size=SIZE_LABEL, color=INK_2).next_to(grid_m, UP, buff=0.3)
        self.play_at("model-grid",
                     ReplacementTransform(sq_pos, VGroup(*grid_m[:N_MODEL])),
                     ReplacementTransform(sq_neg, VGroup(*grid_m[N_MODEL:])),
                     FadeOut(lab_pos), FadeOut(lab_neg), FadeIn(head_m), run_time=1.1)
        n97 = mtex("97", size=60, color=MODE_A).next_to(grid_m, DOWN, buff=0.3)
        of97 = label("of 100", size=SIZE_SMALL, color=INK_3).next_to(n97, RIGHT, buff=0.15, aligned_edge=DOWN)
        self.wait_until(self.word("ninety-seven"))
        self.play(FadeIn(n97, shift=UP * 0.12), FadeIn(of97), run_time=0.5)

        # "the sampler delivers sixty-one": the temperature plot moves, axes and all, into a band
        # under the grids (it stays readable there); the sampler's 100 runs finish one by one
        grid_s = make_grid(RC)
        for sq in grid_s:
            sq.set_fill(NOISE, opacity=0.35)
        # "sampler" has a descender: align its top (ascender) with "model", not its bottom
        head_s = text("sampler", size=SIZE_LABEL, color=INK_2).next_to(grid_s, UP, buff=0.3).align_to(head_m, UP)
        BO = np.array([-5.2, -3.2, 0])     # band origin (labels clear the caption strip); the paper's schedule is BW wide
        BW, BH = 2.8, 0.9

        def band_axis(k, origin=BO):
            return Line(origin, origin + RIGHT * (max(k, 1.0) * BW + 0.15), stroke_color=RULE, stroke_width=2)

        def band_stairs(k, color=INK_2, origin=BO):
            return staircase(BW * k, BH, color=color).shift(origin)

        def band_parts(origin):
            ax = band_axis(1.0, origin)
            ay = Line(origin, origin + UP * (BH + 0.1), stroke_color=RULE, stroke_width=2)
            st = band_stairs(1.0, origin=origin)
            hot = label("hot", size=SIZE_SMALL, color=INK_3).next_to(st.get_start(), LEFT, buff=0.15)
            return ax, ay, st, hot

        # two stages, so the plot never crosses the grids or the numbers: first it shrinks straight
        # down into the empty strip under the right half (its "temperature" label goes: "hot" is
        # enough in the band, and the label would sit under "97 of 100"), then it slides left
        # along the bottom while the sampler grid fades in above it
        mid = band_parts(np.array([o[0], BO[1], 0]))
        bax_x, bax_y, bstairs, blab_hot = band_parts(BO)
        self.play_at("sampler-grid",
                     FadeOut(VGroup(lab_s, lab_cool, dot, lab_t, tag_m)),
                     ReplacementTransform(ax_x, mid[0]), ReplacementTransform(ax_y, mid[1]),
                     ReplacementTransform(trace, mid[2]), ReplacementTransform(lab_hot, mid[3]),
                     run_time=0.45)
        self.play(ReplacementTransform(mid[0], bax_x), ReplacementTransform(mid[1], bax_y),
                  ReplacementTransform(mid[2], bstairs), ReplacementTransform(mid[3], blab_hot),
                  FadeIn(grid_s, lag_ratio=0.005), FadeIn(head_s), run_time=0.5)
        finish = RNG.permutation(100)
        self.play(LaggedStart(*[sq.animate.set_fill(MODE_A if i < N_CHAIN else MODE_B, FILL)
                                for i, sq in sorted(enumerate(grid_s), key=lambda p: finish[p[0]])],
                              lag_ratio=0.03),
                  run_time=self.word("sixty-one") + 0.35 - self.now)
        n61 = mtex("61", size=60, color=MODE_A).next_to(grid_s, DOWN, buff=0.3)
        n61.align_to(n97, DOWN)
        of61 = label("of 100", size=SIZE_SMALL, color=INK_3).next_to(n61, RIGHT, buff=0.15, aligned_edge=DOWN)
        self.play(FadeIn(n61, shift=UP * 0.12), FadeIn(of61), run_time=0.5)

        # ---------------------------------------------------- beat belief
        y_ar = LC[1]
        arrow = Arrow(np.array([1.1, y_ar, 0]), np.array([-1.1, y_ar, 0]), buff=0, stroke_width=3,
                      color=INK_3, tip_length=0.2, max_tip_length_to_length_ratio=0.2)
        lab_ar = label("cool more slowly", size=SIZE_SMALL, color=INK_2).next_to(arrow, DOWN, buff=0.2)
        self.play_at("slower", GrowArrow(arrow), FadeIn(lab_ar, shift=LEFT * 0.1), run_time=1.0)

        def tick_label(s, x, is_tex=False):
            m = mtex(s, size=SIZE_SMALL, color=INK_3) if is_tex else label(s, size=SIZE_SMALL, color=INK_3)
            return m.next_to(np.array([x, BO[1], 0]), DOWN, buff=0.14)

        # "With a quarter of the paper's sampling time at every cooling step" -> every step x 1/4,
        # against a fixed tick at the paper's length
        x_paper = BO[0] + BW
        tick = Line(np.array([x_paper, BO[1] - 0.07, 0]), np.array([x_paper, BO[1] + 0.07, 0]),
                    stroke_color=INK_3, stroke_width=2)
        lab_tick = tick_label("paper's", x_paper)
        lab_q = tick_label(r"\times\tfrac14", BO[0] + BW / 4, is_tex=True)
        X59, X61, X63 = RC[0] - 1.35, RC[0], RC[0] + 1.35
        y_num = n61.get_center()[1]
        n61_small = mtex("61", size=SIZE_BODY, color=MODE_A).move_to(np.array([X61, y_num, 0]))
        # "x1": the band's tick already names the paper's schedule, so the row reads x1/4 . x1 . x4
        sub61 = mtex(r"\times 1", size=SIZE_SMALL, color=INK_3).next_to(n61_small, DOWN, buff=0.14)
        # the scaled time is the final cooling only (paper App. A): the brief cooling and reheating stay
        q_tag = label("final cooling only · brief cooling and reheating unchanged (App. A)", size=SIZE_SMALL,
                      color=INK_3).to_edge(UP, buff=0.3)
        self.play_at("quarter",
                     FadeIn(tick), FadeIn(lab_tick), Transform(bstairs, band_stairs(0.25)), FadeIn(lab_q),
                     ReplacementTransform(n61, n61_small), FadeOut(of61), FadeIn(sub61), FadeIn(q_tag), run_time=0.9)
        # two runs change sign: cells 59 and 60 turn negative (a small change, shown as it is)
        ring = lambda idx: VGroup(*[SurroundingRectangle(grid_s[i], buff=0.0, stroke_color=INK, stroke_width=2.5)
                                    for i in idx])
        r1 = ring(range(N_QUARTER, N_CHAIN))
        self.wait_until(self.word("fifty-nine") - 0.1)
        self.play(Create(r1), *[grid_s[i].animate.set_fill(MODE_B, FILL) for i in range(N_QUARTER, N_CHAIN)],
                  run_time=0.8)
        n59 = mtex(str(N_QUARTER), size=SIZE_BODY, color=MODE_A).move_to(np.array([X59, y_num, 0]))
        sub59 = mtex(r"\times\tfrac14", size=SIZE_SMALL, color=INK_3).next_to(n59, DOWN, buff=0.14)
        self.play(FadeIn(n59, shift=UP * 0.1), FadeIn(sub59), FadeOut(r1), run_time=0.6)

        # "With four times the paper's time, sixteen times the first" -> every step x 4; the
        # quarter schedule stays as a faint ghost so both lengths are on screen
        ghost = bstairs.copy()
        self.add(ghost)
        lab_4 = tick_label(r"\times 4", BO[0] + 4 * BW, is_tex=True)
        self.play_at("four", ghost.animate.set_stroke(INK_3, opacity=0.5),
                     Transform(bstairs, band_stairs(4.0)), Transform(bax_x, band_axis(4.0)),
                     FadeIn(lab_4, shift=RIGHT * 0.2), run_time=1.3)
        lab16 = mtex(r"\times 4 \;=\; 16\times\text{ the time of }\times\tfrac14", size=SIZE_SMALL, color=INK_2)
        lab16.next_to(np.array([BO[0] + 4 * BW, BO[1], 0]), DOWN, buff=0.14).align_to(lab_4, RIGHT)
        self.wait_until(self.word("sixteen"))
        self.play(ReplacementTransform(lab_4, lab16), run_time=0.9)
        # "it delivers about sixty-three": cells 59-62 turn positive
        r2 = ring(range(N_QUARTER, N_FOUR))
        n63 = mtex(str(N_FOUR), size=SIZE_BODY, color=MODE_A).move_to(np.array([X63, y_num, 0]))
        sub63 = mtex(r"\times 4", size=SIZE_SMALL, color=INK_3).next_to(n63, DOWN, buff=0.14)
        self.play_at("sixty-three",
                     Create(r2), *[grid_s[i].animate.set_fill(MODE_A, FILL) for i in range(N_QUARTER, N_FOUR)],
                     FadeIn(n63, shift=UP * 0.1), FadeIn(sub63), run_time=0.8)

        # ---------------------------------------------------- beat question
        # back to the paper's schedule (61); the model's preference is named, and the 36 samples
        # the model makes positive and the sampler makes negative move across and turn orange
        n61_back = mtex("61", size=60, color=MODE_A).move_to(n61_small).align_to(n97, DOWN)
        of61b = label("of 100", size=SIZE_SMALL, color=INK_3).next_to(n61_back, RIGHT, buff=0.15, aligned_edge=DOWN)
        pref = label("the model's preference", size=SIZE_SMALL, color=INK_2).next_to(VGroup(n97, of97), DOWN, buff=0.2)
        band = VGroup(bax_x, bax_y, bstairs, ghost, blab_hot, tick, lab_tick, lab_q, lab16)
        # the 59 / x1/4 and 63 / x4 row leaves first, so "of 100" never overlaps a fading "63"
        self.play_at("question",
                     FadeOut(VGroup(arrow, lab_ar, q_tag, n59, sub59, n63, sub63, sub61, r2, band)),
                     *[grid_s[i].animate.set_fill(MODE_B, FILL) for i in range(N_CHAIN, N_FOUR)],
                     run_time=0.35)
        self.play(ReplacementTransform(n61_small, n61_back), FadeIn(of61b),
                  FadeIn(pref, shift=UP * 0.08), run_time=0.4)
        # the 36 blue cells lift off the model grid as one rigid block (the grids have the same
        # layout, so the block lands exactly on the sampler's cells 61-96), stay blue in flight,
        # and turn orange only once they sit on the sampler's cells
        lost = range(N_CHAIN, N_MODEL)
        # (opaque, with a rim in the ground colour, so each moving square stays a separate square
        # where it passes over the cells beneath it)
        movers = VGroup(*[grid_m[i].copy().set_fill(MODE_A, 1.0).set_stroke(BG, 3) for i in lost])
        self.add(movers)
        self.play(*[m.animate.scale(1.15) for m in movers], run_time=0.3)
        self.play(movers.animate.shift(RC - LC), run_time=0.8)
        self.play(*[m.animate.scale(1 / 1.15).set_fill(MODE_B, FILL) for m in movers], run_time=0.3)
        self.remove(movers, *movers)
        # an outline of exactly those 36 cells (61-69, 70-89, 90-96), not whole rows
        def edge(c, r):
            return RC + np.array([(c - 5) * STEP, -(r - 5) * STEP, 0])
        lost_box = Polygon(*[edge(c, r) for c, r in
                             ((1, 6), (10, 6), (10, 9), (7, 9), (7, 10), (0, 10), (0, 7), (1, 7))],
                           stroke_color=INK, stroke_width=2.5)
        lab36 = label("36 lost", size=SIZE_SMALL, color=INK_2).next_to(edge(10, 7.5), RIGHT, buff=0.18)
        # ends on "where the preference is lost", when the grids rise for the title
        self.play_until("title", Create(lost_box), FadeIn(lab36), min_run_time=0.3)

        # "where the preference is lost" -> the grids rise and shrink first, then the title writes on
        top = VGroup(head_m, grid_m, n97, of97, pref, head_s, grid_s, n61_back, of61b, lost_box, lab36)
        title = tex(r"A Finite-Rate Selection Law for Annealed Gibbs Sampling", size=SIZE_HEAD, color=INK)
        title.move_to(np.array([0, -0.35, 0]))
        venue = text("DynaFront @ NeurIPS 2026", size=SIZE_LABEL, color=INK_3).next_to(title, DOWN, buff=0.3)
        self.play_at("title", top.animate.scale(0.56).move_to(np.array([0, 2.35, 0])), run_time=0.8)
        self.play(Write(title), run_time=1.05)
        self.play_until("route", FadeIn(venue), min_run_time=0.3)

        # "and how much of it survives" -> the route through the film, set as one line of text
        route = label("where the choice is made  ·  how much survives  ·  why slower barely helps  ·  "
                      "competing patterns  ·  the test", size=SIZE_SMALL, color=INK_2)
        if route.width > 12.6:
            route.scale_to_fit_width(12.6)
        route.move_to(np.array([0, -2.5, 0]))
        self.play_at("route", FadeIn(route, shift=UP * 0.08), run_time=1.6)
