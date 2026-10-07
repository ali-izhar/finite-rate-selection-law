"""Scene 9 — What the law does and does not say (chapter: limits).

Beats (script.md):
  size      log-log plot of measured c against N (solid ink, 95% intervals, fitted N^-0.36) and the
            stepped prediction (dashed light grey, N^-0.46); a grey tag names the candidate cause
            (pattern correlations ~ N^-1/2) as not established.
  rate      the N axis becomes a cooling-time axis: only the N = 2,048 point stays, above the one
            schedule every Hopfield run used (834 sweeps); at 1/4x and 4x an empty dashed ring with "?"
            and the tag "no Hopfield runs here".
  scope     a box "annealed Gibbs sampling", then two boxes behind a dashed wall: annealed
            importance sampling and diffusion models: future work.
  return    the hook's two 10x10 grids (97 and 61). The 36 squares the sampler lost fly into the
            grey selection interval of a small fork plot (paper Fig. 1a paths, no field), where the
            field arrow (length = the push) tilts a much wider Gaussian fluctuation spread
            (sd = push / a, a = 0.208, so the part above m = 0 is about Phi(a) = 58%).
            Log-odds 3.57 and 0.43; c = 0.43/3.57 = 0.120 in gold.
            A ladder: sixteen times the cooling time on a ramp -> twice c (16^(1/4) = 2). The
            staircase: the paper's schedule (61) shrinks to 1/4 of its final cooling sweeps (59), then
            stretches to 4 times (63); c = 0.120 steps aside for c: 0.099 -> 0.147 (x1.48).
            Three competing patterns: c ~ 1.662 x 0.120 = 0.199 (dashed), approximation, misses up to 11%.

Data (all seeded or computed, none invented):
  size scan   ../data/size_scan.json (results/experiments/size_scan_p3_analysis.json; paper
              Table 1, Sec. 5): measured 0.263 [0.247, 0.283], 0.201 [0.194, 0.210],
              0.160 [0.155, 0.165]; predicted 0.284, 0.199, 0.150.
  grids       ../data/c_cw.json: q_eq 0.97274 -> 97, q_chain 0.60546 -> 61; logits 3.57 and 0.43.
  staircase   ../data/c_cw.json ladder and sweeps (834 in all); ../data/cw_scaling.json scale 0.25
              and 4: q 0.5875 -> 59, 0.6281 -> 63; c 0.098978 -> 0.146625 = x1.48 (paper App. A).
  fork        ../data/fork_trajectories.json (paper Fig. 1a: N = 1024, no field, seed 4).
  a = 0.208   paper App. A (Fig. 1b setting); Phi(0.208) = 0.582.
  0.199       paper Eq. 6 / Table 1 (A_3 c_CW, A_3 = 1.662); "misses by up to 11%" paper Sec. 5.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import *  # noqa: E402,F403

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _load(name):
    return json.loads((DATA_DIR / name).read_text())


C_CW = _load("c_cw.json")
Q_EQ, Q_CHAIN = C_CW["runs"][-1]["q_eq"], C_CW["runs"][-1]["q_chain"]
N_MODEL = int(round(100 * Q_EQ))      # 97
N_SAMPLER = int(round(100 * Q_CHAIN))  # 61
LOGIT_EQ = math.log(Q_EQ / (1 - Q_EQ))        # 3.57
LOGIT_CHAIN = math.log(Q_CHAIN / (1 - Q_CHAIN))  # 0.43
SWEEPS = C_CW["sweeps_per_level"]             # 15, 30, ..., 11 (834)
LADDER = C_CW["ladder"]                       # beta levels 0.6 ... 1.6
_SCALE = {(r["N"], r["scale"]): r for r in _load("cw_scaling.json")["schedule"]}
Q_QUARTER = _SCALE[(2048, 0.25)]["q_model"]   # 0.5875
Q_FOUR = _SCALE[(2048, 4.0)]["q_model"]       # 0.6281
C_RATIO = _SCALE[(2048, 4.0)]["c"] / _SCALE[(2048, 0.25)]["c"]  # 1.48
N_QUARTER = int(round(100 * Q_QUARTER))       # 59
N_FOUR = int(round(100 * Q_FOUR))             # 63
SIZE_SCAN = _load("size_scan.json")["rows"]
FORK = _load("fork_trajectories.json")
A_FIG1B = 0.208                                # paper App. A, Fig. 1b

assert (N_MODEL, N_SAMPLER, N_QUARTER, N_FOUR) == (97, 61, 59, 63)
assert sum(SWEEPS) == 834 and round(C_RATIO, 2) == 1.48

# ------------------------------------------------------------------ count grids (as in the hook)
CELL = 0.30
GAP = 0.055
FILL_A = 0.9
FILL_B = 0.92
LC = np.array([-4.6, 0.45, 0])
RC = np.array([4.6, 0.45, 0])


def make_grid(center, n_a):
    g = VGroup()
    step = CELL + GAP
    for i in range(100):
        r, c = divmod(i, 10)
        sq = Square(CELL, stroke_width=0).move_to(center + np.array([(c - 4.5) * step, -(r - 4.5) * step, 0]))
        paint(sq, i < n_a)
        g.add(sq)
    return g


def paint(sq, positive):
    return sq.set_fill(MODE_A if positive else MODE_B, opacity=FILL_A if positive else FILL_B)


def plate(mob, buff=0.05):
    """A ground-coloured plate behind a label that sits over lines."""
    mob.add_to_back(BackgroundRectangle(mob, color=BG, fill_opacity=0.9, buff=buff))
    return mob


class Limits(NarratedScene):
    scene_id = "limits"
    TAIL = 2.9   # the last scene of the film: sync_check allows it up to 3.0 s, so the close holds before the end
    FADE = 0.4   # the final fade to the ground

    def at(self, t: float, *anims, **kw):
        """Un-logged timed play at a word inside a cue's sentence."""
        self._wait_to(t)
        self.play(*anims, **kw)

    def word_after(self, w: str, t0: float) -> float:
        return next(x["start"] for x in self.timing["words"]
                    if x["w"].strip(".,;:").lower() == w.lower() and x["start"] >= t0 - 1e-6)

    def construct(self):
        # ============================================================ beat: size
        LO, HI = math.log(0.115), math.log(0.31)
        ax = Axes(x_range=[9.7, 12.3, 1], y_range=[0, 1, 1], x_length=6.2, y_length=4.4,
                  axis_config={"stroke_color": RULE, "stroke_width": 2, "include_tip": False,
                               "include_ticks": False})
        ax.move_to(np.array([-1.7, 0.05, 0]))

        def P(x, c):
            return ax.c2p(x, (math.log(c) - LO) / (HI - LO))

        xs = [math.log2(r["N"]) for r in SIZE_SCAN]
        x_ticks = VGroup()
        for r, x in zip(SIZE_SCAN, xs):
            tk = Line(ax.c2p(x, 0), ax.c2p(x, 0) + 0.1 * DOWN, stroke_color=RULE, stroke_width=2)
            x_ticks.add(VGroup(tk, label(f"{r['N']:,}", size=SIZE_SMALL, color=INK_3).next_to(tk, DOWN, buff=0.08)))
        x_name = mtex("N", size=SIZE_LABEL, color=INK_3).next_to(ax.x_axis, RIGHT, buff=0.15)
        y_ticks = VGroup()
        for v in (0.15, 0.20, 0.25, 0.30):
            p = P(9.7, v)
            tk = Line(p, p + 0.1 * LEFT, stroke_color=RULE, stroke_width=2)
            y_ticks.add(VGroup(tk, label(f"{v:.2f}", size=SIZE_SMALL, color=INK_3).next_to(tk, LEFT, buff=0.08)))
        c_name = mtex("c", size=SIZE_BODY, color=GILT_TEXT).next_to(ax.y_axis, UP, buff=0.12)
        frame_parts = VGroup(ax, x_ticks, y_ticks, x_name, c_name)

        def meas_point(x, c, lo, hi):
            bar = Line(P(x, lo), P(x, hi), stroke_color=INK, stroke_width=2.5)
            caps = VGroup(*[Line(P(x, v) + 0.07 * LEFT, P(x, v) + 0.07 * RIGHT, stroke_color=INK, stroke_width=2.5)
                            for v in (lo, hi)])
            return VGroup(bar, caps, Dot(P(x, c), radius=0.065, color=INK))

        meas = VGroup(*[meas_point(x, r["c_measured"], *r["ci95"]) for r, x in zip(SIZE_SCAN, xs)])
        # least-squares line through the three measured points in log-log (slope -0.36)
        lnN = np.log([r["N"] for r in SIZE_SCAN])
        lnc = np.log([r["c_measured"] for r in SIZE_SCAN])
        slope, icpt = np.polyfit(lnN, lnc, 1)
        assert round(-slope, 2) == 0.36
        fx = (9.85, 12.15)
        fit = Line(*[P(x, math.exp(icpt + slope * x * math.log(2))) for x in fx], stroke_color=INK, stroke_width=2)
        fit.set_stroke(opacity=0.6)
        pred_line = theory(VMobject().set_points_as_corners([P(x, r["c_predicted"]) for r, x in zip(SIZE_SCAN, xs)]),
                           num_dashes=26).set_stroke(width=2.5)
        pred_pts = VGroup(*[Circle(radius=0.07, stroke_color=THEORY, stroke_width=2.5).move_to(P(x, r["c_predicted"]))
                            for r, x in zip(SIZE_SCAN, xs)])
        lnp = np.log([r["c_predicted"] for r in SIZE_SCAN])
        assert round(-np.polyfit(lnN, lnp, 1)[0], 2) == 0.46

        # legend to the right of the plot: a line sample, the word, the power
        lx = ax.get_right()[0] + 0.9
        leg_m = VGroup(Line(ORIGIN, 0.5 * RIGHT, stroke_color=INK, stroke_width=2.5),
                       label("measured", size=SIZE_LABEL, color=INK),
                       mtex(r"N^{-0.36}", size=SIZE_BODY, color=INK)).arrange(RIGHT, buff=0.2)
        leg_p = VGroup(theory(Line(ORIGIN, 0.5 * RIGHT), num_dashes=4).set_stroke(width=2.5),
                       label("predicted", size=SIZE_LABEL, color=THEORY),
                       mtex(r"N^{-0.46}", size=SIZE_BODY, color=THEORY)).arrange(RIGHT, buff=0.2)
        leg_m.move_to(np.array([lx, 1.0, 0]), aligned_edge=LEFT)
        leg_p.move_to(np.array([lx, 0.2, 0]), aligned_edge=LEFT)
        leg_p_sub = label("for this schedule · two-point slopes", size=SIZE_SMALL, color=INK_3)
        leg_p_sub.next_to(leg_p, DOWN, buff=0.12, aligned_edge=LEFT)
        setting = label("Hopfield, three patterns · field G = 0.0006125 and the same schedule at every N",
                        size=SIZE_SMALL, color=INK_3).move_to(np.array([0, 3.45, 0]))
        cand = tex(r"possible cause: pattern correlations of order $N^{-1/2}$ \;$\cdot$\; not established",
                   size=SIZE_SMALL + 2, color=INK_3).move_to(np.array([0, -3.2, 0]))

        self.play_at("size", Create(ax), FadeIn(x_ticks), FadeIn(y_ticks), FadeIn(x_name), FadeIn(c_name),
                     FadeIn(setting), run_time=1.0)
        self.play(LaggedStart(*[FadeIn(m, scale=0.6) for m in meas], lag_ratio=0.35), run_time=1.2)
        t_minus = self.word_after("minus", 0)
        self.at(t_minus, Create(fit), FadeIn(leg_m, shift=0.1 * LEFT), run_time=1.2)

        self.play_at("exponents", Create(pred_line), LaggedStart(*[Create(c) for c in pred_pts], lag_ratio=0.3),
                     run_time=1.2)
        self.at(self.word_after("four", self.t("exponents")), FadeIn(leg_p, shift=0.1 * LEFT),
                FadeIn(leg_p_sub), run_time=0.8)
        self.play_at("candidate", FadeIn(cand, shift=0.05 * UP), run_time=0.8)

        # ============================================================ beat: rate
        # the N axis becomes a cooling-time axis (log scale, one unit = four times the sweeps). Every
        # Hopfield run used one schedule (834 sweeps); the N = 2,048 point already sits at that tick.
        # Other cooling times were never run on the Hopfield model: an empty marker with "?" at 1/4x and 4x.
        X_COL = 11.0
        x_name2 = label("cooling time", size=SIZE_LABEL, color=INK_3)
        x_arrow = Arrow(ORIGIN, 0.6 * RIGHT, buff=0, stroke_width=2.5, color=INK_3, tip_length=0.14)
        x_lab2 = VGroup(x_name2, x_arrow).arrange(RIGHT, buff=0.12)
        x_lab2.next_to(ax.x_axis, RIGHT, buff=0.15)

        def x_tick(x, s, col):
            tk = Line(ax.c2p(x, 0), ax.c2p(x, 0) + 0.1 * DOWN, stroke_color=col, stroke_width=2)
            return VGroup(tk, label(s, size=SIZE_SMALL, color=col).next_to(tk, DOWN, buff=0.08))
        col_tick = x_tick(X_COL, "834 sweeps", INK_2)
        side_ticks = VGroup(x_tick(X_COL - 1, "¼ ×", INK_3), x_tick(X_COL + 1, "4 ×", INK_3))
        col_head = label("every Hopfield run", size=SIZE_LABEL, color=INK)
        col_head.next_to(P(X_COL, 0.29), UP, buff=0.05)
        # an empty dashed column (no height is implied: c there is unknown) with "?" at its top
        unknown = VGroup()
        for x in (X_COL - 1, X_COL + 1):
            col = DashedLine(ax.c2p(x, 0.04), ax.c2p(x, 0.86), stroke_color=NOISE, stroke_width=2.5,
                             dash_length=0.1)
            q = text("?", size=SIZE_BODY + 4, color=INK_2).next_to(col, UP, buff=0.1)
            unknown.add(VGroup(col, q))
        never = label("no Hopfield runs here", size=SIZE_LABEL, color=INK_2)
        never.next_to(unknown[1][1], RIGHT, buff=0.25)  # names both empty columns; under the axis it
        # would sit below "834 sweeps" and read as if that schedule was never run
        others = VGroup(meas[0], meas[2])
        self.play_at("rate", FadeOut(VGroup(fit, pred_line, pred_pts, leg_m, leg_p, leg_p_sub, cand, x_ticks,
                                            x_name, setting, others)), run_time=0.8)
        self.play(FadeIn(x_lab2), Create(col_tick[0]), FadeIn(col_tick[1]), FadeIn(col_head), run_time=0.7)
        self.at(self.word_after("never", self.t("rate")), FadeIn(side_ticks),
                LaggedStart(*[AnimationGroup(Create(u[0]), FadeIn(u[1])) for u in unknown], lag_ratio=0.3), FadeIn(never), run_time=0.9)
        self.at(self.word_after("test", self.t("rate")),
                *[Indicate(u[1], color=INK, scale_factor=1.3) for u in unknown], run_time=0.8)

        # ============================================================ beat: scope
        plot_all = VGroup(ax, y_ticks, c_name, meas[1], x_lab2, col_tick, side_ticks, col_head, unknown, never)

        def box(title, sub, color, width=4.6):
            t = text(title, size=SIZE_LABEL + 2, color=color)
            s = label(sub, size=SIZE_SMALL + 2, color=INK_3) if sub else VGroup()
            body = VGroup(t, s).arrange(DOWN, buff=0.14) if sub else VGroup(t)
            r = RoundedRectangle(width=max(width, body.width + 0.6), height=body.height + 0.55, corner_radius=0.14,
                                 stroke_color=color, stroke_width=2)
            r.move_to(body)
            return VGroup(r, body)

        gibbs = box("annealed Gibbs sampling", None, INK, width=4.6).move_to(np.array([-3.6, 0, 0]))
        ais = box("annealed importance sampling", None, INK_2, width=5.2).move_to(np.array([3.3, 1.2, 0]))
        diff = box("diffusion models", None, INK_2, width=5.2).move_to(np.array([3.3, -1.2, 0]))
        assert max(ais.get_right()[0], diff.get_right()[0]) <= config.frame_width / 2 - 0.4
        wall = DashedLine(np.array([0.0, 2.3, 0]), np.array([0.0, -2.3, 0]), stroke_color=NOISE,
                          stroke_width=3, dash_length=0.12)
        wall_lab = label("future work", size=SIZE_LABEL, color=INK_2).next_to(wall, DOWN, buff=0.15)
        self.play_at("scope", FadeOut(plot_all, run_time=0.5),
                     Succession(Wait(0.3), FadeIn(gibbs, shift=0.1 * RIGHT, run_time=0.9)))
        self.play_at("ais", FadeIn(ais, shift=0.1 * LEFT), run_time=0.8)
        self.play_at("diffusion", FadeIn(diff, shift=0.1 * LEFT), run_time=0.8)
        self.at(self.word_after("future", self.t("diffusion")), Create(wall), FadeIn(wall_lab),
                *[b[1].animate.set_opacity(0.6) for b in (ais, diff)], *[b[0].animate.set_stroke(opacity=0.5) for b in (ais, diff)],
                run_time=0.8)

        # ============================================================ beat: return
        grid_m = make_grid(LC, N_MODEL)
        grid_s = make_grid(RC, N_SAMPLER)
        head_m = text("model", size=SIZE_LABEL, color=INK_2).next_to(grid_m, UP, buff=0.3)
        head_s = text("sampler", size=SIZE_LABEL, color=INK_2).next_to(grid_s, UP, buff=0.3)
        head_s.align_to(head_m, UP)  # same cap line ("sampler" has a descender, "model" none)
        num_m = mtex(str(N_MODEL), size=60, color=MODE_A).next_to(grid_m, DOWN, buff=0.3)
        num_s = mtex(str(N_SAMPLER), size=60, color=MODE_A).next_to(grid_s, DOWN, buff=0.3)
        tag = label("Curie–Weiss · N = 2,048 · the paper's 834-sweep schedule · computed exactly",
                    size=SIZE_SMALL, color=INK_3).move_to(np.array([0, -3.62, 0]))
        self.play_at("return", FadeOut(VGroup(gibbs, ais, diff, wall, wall_lab), run_time=0.45),
                     Succession(Wait(0.25), AnimationGroup(FadeIn(grid_m, lag_ratio=0.01), FadeIn(head_m),
                                                           Write(num_m), run_time=0.9)))
        self.at(self.word_after("sixty-one", self.t("return")),
                FadeIn(grid_s, lag_ratio=0.01), FadeIn(head_s), Write(num_s), FadeIn(tag), run_time=1.0)

        # -- where: the 36 lost positive samples fly into the selection interval of a fork
        fax = Axes(x_range=[0.7, 1.35, 0.5], y_range=[-1, 1, 1], x_length=4.5, y_length=3.0,
                   axis_config={"stroke_color": RULE, "stroke_width": 2, "include_tip": False,
                                "include_ticks": False})
        fax.move_to(np.array([0, 0.8, 0]))
        beta = np.asarray(FORK["beta"])
        # selection interval: weight exp(-u^2/2), u = (beta - 1)/sqrt(gamma) (gamma per sweep)
        gamma = float(np.mean(np.diff(beta))) / 6.0
        sig_b = math.sqrt(gamma)
        band = VGroup()
        edges = np.linspace(-2.8, 2.8, 41)
        for a, b in zip(edges[:-1], edges[1:]):
            uc = 0.5 * (a + b)
            x0, x1 = fax.c2p(1 + a * sig_b, 0)[0], fax.c2p(1 + b * sig_b, 0)[0]
            rect = Rectangle(width=x1 - x0 + 0.002, height=fax.y_length, stroke_width=0)
            rect.set_fill(NOISE, opacity=0.55 * math.exp(-uc ** 2 / 2))
            rect.move_to(np.array([(x0 + x1) / 2, fax.c2p(1, 0)[1], 0]))
            band.add(rect)
        runs = VGroup()
        for m in FORK["m"]:
            m = np.asarray(m)
            k1 = int(np.searchsorted(beta, 1.0))
            col = MODE_A if m[-1] > 0 else MODE_B
            g1 = fax.plot_line_graph(beta[:k1 + 1], m[:k1 + 1], add_vertex_dots=False, line_color=INK_3,
                                     stroke_width=1.3)
            g2 = fax.plot_line_graph(beta[k1:], m[k1:], add_vertex_dots=False, line_color=col, stroke_width=1.3)
            g1.set_stroke(opacity=0.6)
            g2.set_stroke(opacity=0.8)
            runs.add(VGroup(g1, g2))
        f_beta = mtex(r"\beta", size=SIZE_LABEL, color=INK_3).next_to(fax.x_axis, RIGHT, buff=0.1)
        f_m = mtex("m", size=SIZE_LABEL, color=INK_3).next_to(fax.y_axis, UP, buff=0.06)
        f_tag = label("paths: no field · illustration", size=SIZE_SMALL - 2, color=INK_3)
        f_tag.next_to(fax, DOWN, buff=0.5)
        lost = list(range(N_SAMPLER, N_MODEL))
        ghosts = VGroup(*[grid_s[i].copy().set_fill(MODE_A, opacity=0).set_stroke(MODE_A, width=2.5, opacity=1)
                          .set_z_index(3) for i in lost])
        self.play_at("where", LaggedStart(*[Create(g) for g in ghosts], lag_ratio=0.02, run_time=0.8),
                     Create(fax, run_time=0.8), FadeIn(band, run_time=0.8), FadeIn(f_beta), FadeIn(f_m),
                     LaggedStart(*[Create(r) for r in runs], lag_ratio=0.04, run_time=1.2), FadeIn(f_tag))
        target = VGroup(*[Square(0.07) for _ in lost]).arrange_in_grid(rows=12, cols=3, buff=0.035)
        target.move_to(fax.c2p(1, 0))
        t_near = self.word_after("near", self.t("where"))
        self.at(t_near, LaggedStart(*[g.animate.set_fill(MODE_A, opacity=0.95).set_stroke(width=0)
                                      .scale(0.07 / CELL).move_to(tg) for g, tg in zip(ghosts, target)],
                                    lag_ratio=0.03), run_time=1.2)
        b1_tick = Line(fax.c2p(1, -1), fax.c2p(1, -1) + 0.1 * DOWN, stroke_color=INK, stroke_width=2)
        b1_lab = mtex(r"\beta = 1", size=SIZE_LABEL, color=INK).next_to(b1_tick, DOWN, buff=0.05)
        win_lab = plate(label("selection interval", size=SIZE_SMALL, color=INK_2))
        win_lab.next_to(fax.c2p(1, 1), UP, buff=0.3)
        self.at(self.word_after("one", t_near), ghosts.animate.set_fill(NOISE, opacity=0).scale(0.5),
                Create(b1_tick), FadeIn(b1_lab), FadeIn(win_lab), runs.animate.set_stroke(opacity=0.22),
                run_time=0.6)
        self.remove(ghosts)

        # the field: a small upward push G; the fluctuations: a Gaussian spread 1/a = 4.8 times wider
        # (Fig. 1b, a = 0.208). The arrow's length is the push; the spread's centre moves up by it,
        # and the part above m = 0 (the positive mode) is then Phi(a) = 58% of the spread.
        x_r, y_r = fax.c2p(1, 0)[0], fax.c2p(1, 0)[1]
        push = 0.3
        sd = push / A_FIG1B
        half_w, reach = 0.32, 1.0 * sd

        def half(mu, upper, col, op):
            ys = np.linspace(0.0, mu + reach, 40) if upper else np.linspace(mu - reach, 0.0, 40)
            ws = half_w * np.exp(-((ys - mu) / sd) ** 2 / 2)
            right = [np.array([x_r + w, y_r + y, 0]) for w, y in zip(ws, ys)]
            left = [np.array([x_r - w, y_r + y, 0]) for w, y in zip(ws[::-1], ys[::-1])]
            pts = right + left + [right[0]]
            return VMobject(stroke_width=0).set_points_as_corners(pts).set_fill(col, opacity=op)

        def spread_at(mu, col_up, col_lo, op):
            return VGroup(half(mu, True, col_up, op), half(mu, False, col_lo, op)).set_z_index(5)
        spread = spread_at(0.0, NOISE, NOISE, 0.8)
        tilted = spread_at(push, MODE_A, MODE_B, 0.85)
        ax_A = x_r - half_w - 0.2
        arrow_G = Arrow(np.array([ax_A, y_r, 0]), np.array([ax_A, y_r + push, 0]), buff=0, stroke_width=10,
                        color=BIAS, tip_length=0.13, max_stroke_width_to_length_ratio=60,
                        max_tip_length_to_length_ratio=0.5).set_z_index(6)
        g_lab = plate(VGroup(label("field", size=SIZE_SMALL, color=BIAS),
                             mtex("G", size=SIZE_LABEL, color=BIAS)).arrange(RIGHT, buff=0.1))
        g_lab.next_to(np.array([ax_A, y_r + push / 2, 0]), LEFT, buff=0.12).set_z_index(7)
        n_lab = plate(label("fluctuations", size=SIZE_SMALL, color=INK_2))
        n_lab.move_to(np.array([x_r + half_w + 0.12, y_r - 0.75 * sd, 0]), aligned_edge=LEFT).set_z_index(7)
        t_weak = self.word_after("weak", t_near)
        self.at(t_weak, GrowArrow(arrow_G), FadeIn(g_lab), FadeIn(spread, scale=0.6), FadeIn(n_lab), run_time=0.7)
        # "tilts the choice": the spread's centre rises by the push; above m = 0 the positive mode
        self.at(self.word_after("tilts", t_weak), Transform(spread, tilted), run_time=0.8)

        # -- c: log-odds 3.57 and 0.43, and their ratio
        fork = VGroup(fax, band, runs, f_beta, f_m, b1_tick, b1_lab, arrow_G, spread)
        lo_m = VGroup(label("log-odds", size=SIZE_SMALL, color=INK_3),
                      mtex(f"{LOGIT_EQ:.2f}", size=SIZE_BODY, color=INK)).arrange(RIGHT, buff=0.15)
        lo_s = VGroup(label("log-odds", size=SIZE_SMALL, color=INK_3),
                      mtex(f"{LOGIT_CHAIN:.2f}", size=SIZE_BODY, color=INK)).arrange(RIGHT, buff=0.15)
        lo_m.next_to(num_m, DOWN, buff=0.15)
        lo_s.next_to(num_s, DOWN, buff=0.15)
        self.play_at("c", FadeOut(VGroup(win_lab, g_lab, n_lab, f_tag)), run_time=0.6)
        self.at(self.word_after("log-odds", self.t("c")), FadeIn(lo_m, shift=0.05 * UP), FadeIn(lo_s, shift=0.05 * UP),
                run_time=0.8)
        c_eq = mtex("c", "=", r"\frac{%.2f}{%.2f}" % (LOGIT_CHAIN, LOGIT_EQ), "=", r"0.120", size=46)
        c_eq[0].set_color(GILT_TEXT)
        c_eq[4].set_color(GILT_TEXT)
        c_eq.move_to(np.array([0, -2.35, 0]))
        t_c = self.word_after("c", self.t("c") + 1.0)
        self.at(t_c, Write(c_eq[:3]), run_time=0.7)
        self.at(self.word_after("zero", t_c), Write(c_eq[3:]), run_time=0.6)

        # ============================================================ slow: the ramp ladder
        u = 0.42
        y0 = -0.45
        rails = VGroup(*[Line(np.array([x, y0 - 0.12, 0]), np.array([x, y0 + 4 * u + 0.12, 0]), stroke_color=RULE,
                              stroke_width=3) for x in (-0.95, 0.95)])
        l_name = label("cooling time", size=SIZE_SMALL, color=INK_2).next_to(rails[0], UP, buff=0.12)
        r_name = mtex("c", size=SIZE_BODY, color=GILT_TEXT).next_to(rails[1], UP, buff=0.12)

        def rung(x, col):
            return Line(np.array([x - 0.22, y0, 0]), np.array([x + 0.22, y0, 0]), stroke_color=col, stroke_width=5)
        base = VGroup(rung(-0.95, INK_3), rung(0.95, INK_3))
        mL, mR = rung(-0.95, INK), rung(0.95, GILT)
        t16 = mtex(r"\times 16", size=SIZE_LABEL, color=INK).move_to(np.array([-1.65, y0 + 4 * u, 0]))
        t2 = mtex(r"\times 2", size=SIZE_LABEL, color=GILT_TEXT).move_to(np.array([1.55, y0 + u, 0]))
        ramp_tag = label("formula · linear ramp", size=SIZE_SMALL - 2, color=THEORY)
        ramp_tag.move_to(np.array([0, y0 - 0.4, 0]))
        quarter = mtex(r"16^{1/4} = 2", size=SIZE_BODY, color=INK).move_to(np.array([0, y0 + 2 * u, 0]))
        ladder = VGroup(rails, l_name, r_name, base, mL, mR, t16, t2, ramp_tag, quarter)
        self.play_at("slow", FadeOut(fork, run_time=0.5),
                     Succession(Wait(0.3), FadeIn(VGroup(rails, l_name, r_name, base, mL, mR, ramp_tag), run_time=0.5)))
        self.play(mL.animate.shift(4 * u * UP), mR.animate.shift(u * UP), run_time=1.2)
        self.play(FadeIn(t16), FadeIn(t2), run_time=0.4)
        self.at(self.word_after("quarter", self.t("slow")), Write(quarter), run_time=0.8)

        # ============================================================ stair: the paper's staircase
        temps = [1 / b for b in LADDER]
        T_HI, T_LO = temps[0], temps[-1]
        W1 = 1.15  # width of the paper's schedule (834 sweeps)
        SY0, SH = -0.1, 1.5

        def staircase(scale):
            w = W1 * scale
            xs_ = np.concatenate([[0], np.cumsum(SWEEPS)]) / sum(SWEEPS) * w - w / 2
            pts = []
            for i, T in enumerate(temps):
                y = SY0 + SH * (T - T_LO) / (T_HI - T_LO)
                pts += [np.array([xs_[i], y, 0]), np.array([xs_[i + 1], y, 0])]
            return VMobject(stroke_color=INK_2, stroke_width=3).set_points_as_corners(pts).move_to(np.array([0, 0.65, 0]))

        # The staircase first appears at the paper's width (834 sweeps, grid 61), then shrinks to a
        # quarter of the sweeps at every final cooling step (grid 59): the viewer sees why 61 became 59.
        # Then it stretches to four times (sixteen times the quarter) and the grid gains four (63).
        # c = 0.120 and the 834-sweep tag belong to the paper's schedule: they step aside here, and the
        # gold line shows c for these two schedules (0.099 -> 0.147, x1.48), at the same place.
        stair = staircase(1.0)
        st_name = label("temperature", size=SIZE_SMALL - 2, color=INK_3).next_to(stair, UP, buff=0.1).align_to(stair, LEFT)
        st_name.add_updater(lambda m: m.next_to(stair, UP, buff=0.1).align_to(stair, LEFT))
        st_lab1 = label("the paper's 834 sweeps", size=SIZE_SMALL, color=INK_2).move_to(np.array([0, -0.65, 0]))
        st_lab = label("¼ × the final cooling's sweeps", size=SIZE_SMALL, color=INK_2).move_to(st_lab1)
        st_lab4 = label("4 × the final cooling's sweeps: 16 × more", size=SIZE_SMALL, color=INK_2).move_to(st_lab1)
        n59 = mtex(str(N_QUARTER), size=60, color=MODE_A).move_to(num_s)
        n63 = mtex(str(N_FOUR), size=60, color=MODE_A).move_to(num_s)
        tag_st = label("Curie–Weiss · N = 2,048 · final cooling sweeps × ¼ and × 4 · computed exactly",
                       size=SIZE_SMALL, color=INK_3).move_to(tag)
        c_q, c_4 = _SCALE[(2048, 0.25)]["c"], _SCALE[(2048, 4.0)]["c"]
        c_st = mtex("c", r"\colon", f"{c_q:.3f}", r"\to", f"{c_4:.3f}", r"\;(\times %.2f)" % C_RATIO, size=46,
                    color=GILT_TEXT).move_to(c_eq)
        assert (f"{c_q:.3f}", f"{c_4:.3f}") == ("0.099", "0.147")
        # Timing: the staircase enters at the end of "ramp" at the paper's width, shrinks to a quarter on
        # "staircase" and holds there (grid 59, readable label) until "the sweeps", where it stretches to
        # four times. Labels, tag and numbers cross-fade (FadeTransform), so no text morphs glyph by glyph.
        t_ramp = self.word_after("ramp", self.t("slow"))
        t_ramp_end = next(x["end"] for x in self.timing["words"] if x["start"] == t_ramp)
        t_case = self.word_after("schedule", t_ramp)
        t_sweeps = self.word_after("the", self.word_after("sixteen", t_case))
        # the log-odds 0.43 and c = 0.120 belong to the paper's schedule: they step aside until it returns
        self.at(t_ramp_end, FadeOut(ladder, run_time=0.35), FadeOut(lo_s, run_time=0.35),
                FadeOut(c_eq, run_time=0.35),
                Succession(Wait(0.2), AnimationGroup(FadeIn(stair), FadeIn(st_name), FadeIn(st_lab1),
                                                     run_time=0.35)))
        self.at(t_case, Transform(stair, staircase(0.25)), FadeTransform(st_lab1, st_lab),
                FadeTransform(tag, tag_st),
                grid_s[N_QUARTER].animate.set_fill(MODE_B, opacity=FILL_B),
                grid_s[N_QUARTER + 1].animate.set_fill(MODE_B, opacity=FILL_B),
                FadeTransform(num_s, n59), run_time=0.5)
        # the stretch ends before "about four", where the grid gains its four samples
        self.at(t_sweeps, Transform(stair, staircase(4.0)), FadeTransform(st_lab, st_lab4),
                run_time=min(1.2, self.t("stair") - t_sweeps - 0.05))
        st_name.clear_updaters()
        gain = list(range(N_QUARTER, N_FOUR))
        plus4 = mtex(r"+%d" % (N_FOUR - N_QUARTER), size=SIZE_BODY, color=MODE_A).next_to(n63, RIGHT, buff=0.25)
        # the gold c line arrives with the +4, so it stays readable for about three seconds
        self.play_at("stair", LaggedStart(*[Indicate(grid_s[i], color=MODE_A, scale_factor=1.3) for i in gain],
                                          lag_ratio=0.25),
                     *[grid_s[i].animate.set_fill(MODE_A, opacity=FILL_A) for i in gain],
                     FadeTransform(n59, n63), FadeIn(plus4, shift=0.05 * UP),
                     Succession(Wait(0.35), Write(c_st, run_time=0.85)), run_time=1.2)

        # ============================================================ patterns: competition keeps more
        pat_head = label("three competing patterns", size=SIZE_LABEL, color=INK_2)
        # paper Eq. 6: c_pred = A_3 c_CW, A_3 = 1.662; the gold 0.120 of the paper's schedule carries into it
        pat_val = mtex("c", r"\approx", "1.662", r"\times", "0.120", "=", "0.199", size=38)
        pat_val[0].set_color(GILT_TEXT)
        pat_val[4].set_color(GILT_TEXT)
        pat_val[6].set_color(THEORY)
        assert round(1.662 * C_CW["runs"][-1]["c_cw"], 3) == 0.199
        pat_ul = theory(Line(LEFT, RIGHT).set_width(pat_val[6].width + 0.15), num_dashes=6).set_stroke(width=2.5)
        pat = VGroup(pat_head, pat_val).arrange(DOWN, buff=0.3).move_to(np.array([0, 0.75, 0]))
        pat_ul.next_to(pat_val[6], DOWN, buff=0.1)
        # two short lines: one line would run into the grids on both sides
        pat_tag = VGroup(label("prediction · an approximation", size=SIZE_SMALL - 2, color=INK_3),
                         label("misses up to 11% elsewhere", size=SIZE_SMALL - 2, color=INK_3))
        pat_tag.arrange(DOWN, buff=0.1).next_to(pat_val, DOWN, buff=0.3)
        # back to the paper's schedule (61, log-odds 0.43, c = 0.120, its tag), where the prediction applies
        n61 = mtex(str(N_SAMPLER), size=60, color=MODE_A).move_to(n63)
        tag_back = label("Curie–Weiss · N = 2,048 · the paper's 834-sweep schedule · computed exactly",
                         size=SIZE_SMALL, color=INK_3).move_to(tag)
        back = [grid_s[i].animate.set_fill(MODE_B, opacity=FILL_B) for i in range(N_SAMPLER, N_FOUR)]
        # three plays in sequence, so nothing overlaps: (1) the staircase and its gold c line leave while
        # the grid returns to 61; (2) the log-odds, c = 0.120 and the head come in on an empty middle;
        # (3) only then the gold 0.120 copy flies. A TransformFromCopy behind a Wait inside one play
        # draws its target at full opacity during the Wait, so it gets its own play.
        # Budget: 0.4 + 0.5 + 0.45 = 1.35 s, which ends before 'keep'.
        self.play_at("patterns", FadeOut(VGroup(stair, st_name, st_lab4, plus4, c_st)), *back,
                     FadeTransform(n63, n61), FadeTransform(tag_st, tag_back), run_time=0.4)
        self.play(FadeIn(lo_s), FadeIn(c_eq), FadeIn(pat_head), Write(pat_val[:4]), run_time=0.5)
        self.play(TransformFromCopy(c_eq[4], pat_val[4]), run_time=0.45)
        # "keep more": the value, its dashed underline and the tag complete together, early enough
        # that the whole close holds for about 3.5 s (TAIL 2.9 s) before a short fade
        self.at(self.word_after("keep", self.t("patterns")),
                AnimationGroup(Write(pat_val[5:]), Create(pat_ul), FadeIn(pat_tag)), run_time=0.4)

        # the last frame of the film: hold, then a short fade to the ground at the very end
        end = self.audio_duration + self.TAIL
        self._wait_to(end - self.FADE)
        rt = round((end - self.now) * config.frame_rate) / config.frame_rate - 1e-6
        self.play(FadeOut(Group(*self.mobjects)), run_time=rt)
