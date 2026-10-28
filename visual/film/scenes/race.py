"""Scene "race" — Field against fluctuations (script.md, scene race; the page's chapter "race").

Beats: forces -> interval -> compete -> gaussian -> formula -> example.

Real shapes and numbers (paper §3, Fig. 1, App. A):
  * the fork: the fourteen runs of paper Fig. 1a (../data/fork_trajectories.json: no field,
    N = 1,024, start m = 0, 6 sweeps at each of 360 temperatures 0.7..1.35, seed 4), with the
    dashed mean-field branches m = tanh(beta m). Its grey band is read from those runs: from the
    earliest to the latest last change of sign (beta 0.96..1.09).
  * the staircase inset: the paper's schedule (Table 3; ../data/schedule.json), ten levels, 834
    sweeps; the dashed line is a straight ramp over the same range ("simplification").
  * the growth panel: rate beta - 1 = gamma t, area gamma t^2 / 2, area 1 at t = sqrt(2/gamma).
  * field against fluctuations: field part G T and fluctuation size sqrt(2 T / N) (variance 2/N
    per sweep), N = 2,048, G = 0.0006125, T from 0 to 16 gamma^(-1/2) at gamma = 0.001.
  * the runs: the linearized dynamics near m = 0 on a ramp, dm = (gamma t m + G) dt + sqrt(2/N) dW,
    at the paper's Fig. 1b setting (N = 2,048, gamma = 0.001, G = 0.0006125), exact growth per
    step plus Euler field and noise, dt = 0.25 sweep, start m = 0 at t = -5 gamma^(-1/2), seed 7. Dividing by exp(gamma t^2 / 2)
    gives y = m exp(-gamma t^2/2), which keeps the sign of m and settles to a Gaussian with
    mean G sqrt(2 pi / gamma) and sd sqrt(2/N) (pi/gamma)^(1/4); a = mean / sd = 0.2075 (0.208),
    Phi(a) = 0.582. 4,000 runs to t = 4 gamma^(-1/2) (120 drawn, up to 1.5 gamma^(-1/2)); the
    histogram holds their settled y (58.05% positive); the dashed curve is the limiting Gaussian.
  * "staircase: 61%": q_chain = 0.60546 of the exact count calculation (../data/c_cw.json).
  * the formula: q_sel ~ Phi(a), a = G sqrt(N) (pi/gamma)^(1/4)  (paper Eq. 3; Kondepudi & Nelson
    1983; Kondepudi & Gao 1987).

Hand-off to "scaling" (in the silent tail, after the last word): everything clears except the
formula, which rises to UP * 2.75 at SIZE_HEAD in THEORY colour with G in BIAS:
    mtex(r"q_{\\text{sel}}", r"\\approx", r"\\Phi(a),", r"\\quad a =", "G", r"\\sqrt{N}\\,",
         r"(\\pi", r"/\\gamma)^{1/4}", size=SIZE_HEAD, color=THEORY, colors={"G": BIAS})
with a dashed THEORY underline 0.12 below it. scaling.py opens on exactly that frame.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import *  # noqa: E402,F403

VISUAL = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------------ Fig. 1b setting
N_EX, GAMMA, G_EX = 2048, 0.001, 0.0006125
T0 = GAMMA ** -0.5                                   # gamma^(-1/2) = 31.6 sweeps
MEAN = G_EX * math.sqrt(2 * math.pi / GAMMA)         # 0.0486
SD = math.sqrt(2 / N_EX) * (math.pi / GAMMA) ** 0.25  # 0.234
A_EX = MEAN / SD                                     # 0.2075
assert abs(A_EX - G_EX * math.sqrt(N_EX) * (math.pi / GAMMA) ** 0.25) < 1e-12
assert f"{A_EX:.3f}" == "0.208"
PHI_A = 0.5 * (1 + math.erf(A_EX / math.sqrt(2)))
assert f"{100 * PHI_A:.0f}" == "58"


def simulate_ramp(n, seed, t_end, dt=0.25, t_start=-5 * T0):
    """Linearized runs near m = 0 on a ramp: dm = (gamma t m + G) dt + sqrt(2/N) dW."""
    rng = np.random.default_rng(seed)
    ts = np.arange(t_start, t_end + dt / 2, dt)
    m = np.zeros(n)
    rec = np.empty((len(ts), n))
    rec[0] = m
    for j in range(1, len(ts)):
        # the growth over the step exactly (Euler would lose about 8% of it by t = 4 gamma^(-1/2)),
        # then the field and the fluctuations of the step
        grow = math.exp(GAMMA * (ts[j] ** 2 - ts[j - 1] ** 2) / 2)
        m = m * grow + G_EX * dt + math.sqrt(2 * dt / N_EX) * rng.standard_normal(n)
        rec[j] = m
    return ts, rec


def gauss_pdf(z):
    return np.exp(-z * z / 2) / math.sqrt(2 * math.pi)


def mf_branch(beta):
    if beta <= 1:
        return 0.0
    m = 0.9
    for _ in range(300):
        m = math.tanh(beta * m)
    return m


class Race(NarratedScene):
    scene_id = "race"

    # ---------------------------------------------------------------- timing helpers
    def wt(self, word, after=0.0):
        """Start time of the first spoken word that starts with `word`, at or after `after`."""
        for w in self.timing["words"]:
            if w["start"] >= after - 1e-6 and w["w"].strip(".,:;!?'").lower().startswith(word.lower()):
                return w["start"]
        raise KeyError(word)

    def at(self, t, *anims, run_time=1.0, **kw):
        """Start animations at time t (seconds), not bound to a cue; warn if late."""
        if self.now > t + 0.07:
            print(f"[race] late: wanted {t:.2f}, now {self.now:.2f}")
        self._wait_to(t)
        self.play(*anims, run_time=run_time, **kw)

    # ---------------------------------------------------------------- scene
    def construct(self):
        cue = self.t
        rng_vis = np.random.default_rng(11)   # only for the jitter of the illustrative dot

        # ============================================================ the fork (from commit)
        fork_d = json.loads((VISUAL / "data" / "fork_trajectories.json").read_text())
        betas = np.array(fork_d["beta"])
        lasts = []
        for mm in fork_d["m"]:
            mm = np.array(mm)
            sc = np.where(np.sign(mm[1:]) * np.sign(mm[:-1]) < 0)[0]
            lasts.append(betas[sc[-1] + 1] if len(sc) else 1.0)
        W_LO = math.floor((min(lasts) - 0.005) * 100) / 100      # 0.96
        W_HI = math.ceil((max(lasts) + 0.002) * 100) / 100       # 1.09

        B_LO, B_HI = 0.7, 1.35
        fork_ax = Axes(x_range=[B_LO, B_HI, 0.1], y_range=[-1, 1, 0.5], x_length=10.6, y_length=4.8,
                       axis_config={"color": RULE, "stroke_width": 2, "include_ticks": False},
                       tips=False).move_to(DOWN * 0.45)
        f_xl = mtex(r"\beta", size=SIZE_BODY, color=INK_2).next_to(fork_ax.c2p(B_HI, -1), RIGHT, buff=0.2)
        f_yl = mtex("m", size=SIZE_BODY, color=INK_2).next_to(fork_ax.c2p(B_LO, 1), UP, buff=0.15)
        f_ticks = VGroup(*[mtex(s, size=SIZE_SMALL, color=INK_3).next_to(fork_ax.c2p(b, -1), DOWN, buff=0.15)
                           for s, b in (("0.7", 0.7), ("1", 1.0), ("1.3", 1.3))])
        f_mt = VGroup(*[mtex(s, size=SIZE_SMALL, color=INK_3).next_to(fork_ax.c2p(B_LO, v), LEFT, buff=0.15)
                        for s, v in (("+1", 1), ("0", 0), ("-1", -1))])
        bb = np.linspace(1.0, B_HI, 80)
        br_up = theory(VMobject(stroke_width=2).set_points_smoothly(
            [fork_ax.c2p(b, mf_branch(b)) for b in bb]), num_dashes=40)
        br_dn = theory(VMobject(stroke_width=2).set_points_smoothly(
            [fork_ax.c2p(b, -mf_branch(b)) for b in bb]), num_dashes=40)
        br_up.set_stroke(opacity=0.6)
        br_dn.set_stroke(opacity=0.6)
        band = Rectangle(width=fork_ax.c2p(W_HI, 0)[0] - fork_ax.c2p(W_LO, 0)[0], height=fork_ax.y_length,
                         stroke_width=0, fill_color=NOISE, fill_opacity=0.28)
        band.move_to([(fork_ax.c2p(W_LO, 0)[0] + fork_ax.c2p(W_HI, 0)[0]) / 2, fork_ax.c2p(0, 0)[1], 0])
        i_hi = int(np.searchsorted(betas, W_HI))
        pre_paths, post_paths, ends = VGroup(), VGroup(), []
        for mm in fork_d["m"]:
            mm = np.array(mm)
            pts = [fork_ax.c2p(b, v) for b, v in zip(betas, mm)]
            pre_paths.add(VMobject(stroke_width=2.2, stroke_color=INK_3).set_points_as_corners(pts[:i_hi + 1]))
            post_paths.add(VMobject(stroke_width=2.2, stroke_color=INK_3).set_points_as_corners(pts[i_hi:]))
            ends.append(mm[-1])
        fork_tag = label("drift-plus-noise paths · no field · N = 1,024 · illustration",
                         size=SIZE_SMALL, color=INK_3).to_corner(DL, buff=0.35)
        fork = VGroup(fork_ax, band, br_up, br_dn, pre_paths, post_paths, f_xl, f_yl, f_ticks, f_mt)
        self.add(fork, fork_tag)

        # the gold reminder of what is being explained
        ctag = mtex("c", "=", "0.120", size=SIZE_LABEL, color=GILT_TEXT).to_corner(UR, buff=0.4)

        # ============================================================ beat: forces
        self.play_at("where", FadeIn(ctag), run_time=0.8)
        fork_home = fork.copy()                       # the full-size fork, for its return

        # the zoom: one run's m at beta = 1 is a point on a vertical slice of the fork; the slice
        # turns into the horizontal m line and the point becomes the dot
        M_R = 0.3
        i_one = int(np.argmin(np.abs(betas - 1.0)))
        m_at_one = [mm[i_one] for mm in fork_d["m"]]
        k_run = int(np.argmin([abs(v - 0.11) if v > 0 else 9 for v in m_at_one]))
        M0 = float(m_at_one[k_run])                   # 0.111 (run 10 of seed 4)
        slice_ = Line(fork_ax.c2p(1.0, -M_R), fork_ax.c2p(1.0, M_R), color=INK, stroke_width=3)
        fdot = Dot(fork_ax.c2p(1.0, M0), radius=0.09, color=INK)
        self.at(self.wt("zoom"), Create(slice_), FadeIn(fdot, scale=0.5),
                fork[4][k_run].animate.set_stroke(color=INK, width=3), run_time=0.7)

        mline = NumberLine(x_range=[-M_R, M_R, 0.1], length=9.0, color=RULE, stroke_width=2,
                           include_ticks=False).move_to(DOWN * 0.55)
        pos_half = Line(mline.n2p(0), mline.n2p(M_R), color=MODE_A, stroke_width=5, stroke_opacity=0.55)
        neg_half = Line(mline.n2p(-M_R), mline.n2p(0), color=MODE_B, stroke_width=5, stroke_opacity=0.55)
        zero_t = Line(mline.n2p(0) + DOWN * 0.12, mline.n2p(0) + UP * 0.12, color=INK_3, stroke_width=2)
        zero_l = mtex("0", size=SIZE_SMALL, color=INK_3).next_to(zero_t, DOWN, buff=0.12)
        m_l = mtex("m", size=SIZE_BODY, color=INK_2).next_to(mline, RIGHT, buff=0.25)
        illus = label("illustration", size=SIZE_SMALL, color=INK_3).to_corner(DL, buff=0.35)
        map_scale, map_pos = 0.3, np.array([-5.15, 2.75, 0])
        self.at(self.wt("beta"), fork.animate.scale(map_scale).move_to(map_pos), FadeOut(fork_tag),
                Transform(slice_, Line(mline.n2p(-M_R), mline.n2p(M_R), color=INK, stroke_width=3)),
                fdot.animate.move_to(mline.n2p(M0)), run_time=1.1)
        self.play(FadeIn(VGroup(mline, pos_half, neg_half, zero_t, zero_l, m_l)), FadeOut(slice_),
                  FadeIn(illus), run_time=0.5)

        m_val = ValueTracker(M0)                      # the illustrative m
        jit = {"x": 0.0}                              # the fluctuation added from the 'fluct' cue

        def m_now():
            return m_val.get_value() + jit["x"]

        dot = Dot(mline.n2p(M0), radius=0.11, color=INK)
        dot.add_updater(lambda d: d.move_to(mline.n2p(m_now())))
        self.remove(fdot)
        self.add(dot)

        # the drift arrow above the line: its length is (beta - 1) m, so it grows with m
        sgn = ValueTracker(-1.0)                      # sign of beta - 1
        K_ARR = 12.0                                  # scene units per unit of (beta - 1) m

        def drift_arrow():
            x = m_now()
            y0 = mline.n2p(0)[1] + 0.55
            start = np.array([mline.n2p(x)[0], y0, 0])
            length = K_ARR * sgn.get_value() * x
            if abs(length) < 0.04:
                return VMobject()
            end = start + RIGHT * length
            return Arrow(start, end, buff=0, color=INK_2, stroke_width=4, max_tip_length_to_length_ratio=0.22,
                         max_stroke_width_to_length_ratio=12)

        arr = always_redraw(drift_arrow)
        arr_l = mtex(r"(\beta-1)", r"\,m", size=SIZE_LABEL, color=INK_2)
        arr_l.add_updater(lambda l: l.move_to([mline.n2p(m_now())[0] + K_ARR * sgn.get_value() * m_now() / 2,
                                               mline.n2p(0)[1] + 0.98, 0]))
        lab_below = mtex(r"\beta < 1", size=SIZE_LABEL, color=INK_2)
        lab_above = mtex(r"\beta > 1", size=SIZE_LABEL, color=INK_2)
        for l in (lab_below, lab_above):
            l.move_to(mline.n2p(-0.2) + UP * 0.9)
        self.play_at("below", FadeIn(arr), FadeIn(arr_l), FadeIn(lab_below), run_time=0.6)
        self.play(m_val.animate.set_value(0.045), run_time=2.6, rate_func=smooth)

        # above one: the imbalance grows, and with it the arrow (dm/dt = (beta - 1) m, illustrative rate)
        def grow(v, dt):
            v.set_value(v.get_value() * math.exp(0.105 * dt))

        self.play_at("above", sgn.animate.set_value(1.0), FadeOut(lab_below), FadeIn(lab_above), run_time=0.7)
        m_val.add_updater(grow)
        self.add(m_val)

        # the field: a steady arrow toward +m, below the line
        def field_arrow():
            start = np.array([mline.n2p(m_now())[0], mline.n2p(0)[1] - 0.55, 0])
            return Arrow(start, start + RIGHT * 0.9, buff=0, color=BIAS, stroke_width=4,
                         max_tip_length_to_length_ratio=0.25)

        farr = always_redraw(field_arrow)
        g_lab = mtex("G", size=SIZE_LABEL, color=BIAS)
        g_lab.add_updater(lambda l: l.next_to(farr, DOWN, buff=0.1))
        self.play_at("field", FadeIn(farr), FadeIn(g_lab), run_time=0.7)

        # the drift formula is assembled from the two arrow labels
        drift = mtex(r"\text{drift} \approx", r"(\beta-1)", r"\,m", "+", "G", size=SIZE_HEAD,
                     colors={"G": BIAS})
        drift.move_to(UP * 1.75)
        self.play_at("drift", Write(drift[0]), run_time=0.6)
        arr_copy = arr_l.copy().clear_updaters()
        self.at(self.wt("beta", cue("drift")), ReplacementTransform(arr_copy, drift[1:3]), run_time=1.0)
        g_copy = g_lab.copy().clear_updaters()
        self.at(self.wt("plus", cue("drift")), Write(drift[3]), ReplacementTransform(g_copy, drift[4]),
                run_time=0.7)

        # fluctuations: the dot jitters (seeded), variance written in grey
        def jitter(d, dt):
            if dt <= 0:
                return
            jit["x"] += -1.5 * jit["x"] * dt + 0.06 * math.sqrt(dt) * rng_vis.standard_normal()

        var = mtex(r"\text{variance} \approx 2/N \text{ per sweep}", size=SIZE_BODY, color=INK_3)
        var.move_to(DOWN * 2.1)
        dot.add_updater(jitter)
        self.play_at("fluct", Write(var), run_time=2.0)

        # ============================================================ beat: interval
        forces = VGroup(mline, pos_half, neg_half, zero_t, zero_l, m_l, dot, arr, arr_l, lab_above, farr, g_lab, var,
                        illus)
        sched = json.loads((VISUAL / "data" / "schedule.json").read_text())
        levels, counts = self._schedule(sched)
        total = sum(counts)
        assert total == 834, total
        st_ax = Axes(x_range=[0, total, 200], y_range=[0.6, 1.6, 0.2], x_length=5.0, y_length=3.0,
                     axis_config={"color": RULE, "stroke_width": 2, "include_ticks": False}, tips=False)
        st_ax.move_to(LEFT * 3.55 + DOWN * 0.35)
        st_pts, x = [], 0
        for lv, k in zip(levels, counts):
            st_pts += [st_ax.c2p(x, lv), st_ax.c2p(x + k, lv)]
            x += k
        stairs = VMobject(stroke_width=3, stroke_color=INK_2).set_points_as_corners(st_pts)
        st_yl = mtex(r"\beta", size=SIZE_LABEL, color=INK_2).next_to(st_ax.c2p(0, 1.6), UP, buff=0.12)
        st_xl = label("sweeps", size=SIZE_SMALL, color=INK_3).next_to(st_ax.c2p(total, 0.6), DOWN, buff=0.15)
        st_xl.align_to(st_ax.c2p(total, 0.6), RIGHT)
        st_name = label("the paper's cooling schedule", size=SIZE_SMALL, color=INK_3).next_to(st_ax, UP, buff=0.45)
        ramp = theory(Line(st_ax.c2p(0, 0.6), st_ax.c2p(total, 1.6), stroke_width=3), num_dashes=34)
        tri_a, tri_b = st_ax.c2p(470, 1.164), st_ax.c2p(650, 1.164)
        tri_c = st_ax.c2p(650, 0.6 + 650 / total)
        tri = VGroup(Line(tri_a, tri_b, color=THEORY, stroke_width=2), Line(tri_b, tri_c, color=THEORY, stroke_width=2))
        g_sym = mtex(r"\gamma", size=SIZE_LABEL, color=THEORY).next_to(Line(tri_b, tri_c), RIGHT, buff=0.1)
        simp = label("simplification", size=SIZE_SMALL, color=INK_3).next_to(st_ax, DOWN, buff=0.5)

        # the forces picture leaves first; the drift formula stays, small, at the top
        self.play_at("ramp", FadeOut(forces), FadeOut(fork), drift.animate.scale(0.62).move_to(UP * 3.3),
                     run_time=0.7)
        m_val.clear_updaters()
        self.remove(m_val)
        self.play(FadeIn(VGroup(st_ax, stairs, st_yl, st_xl, st_name)), run_time=0.6)
        self.at(self.wt("ramp", cue("ramp")) - 0.1, Create(ramp), FadeIn(simp), run_time=1.0)
        self.at(self.wt("gamma", cue("ramp")), Create(tri), Write(g_sym), run_time=0.6)

        # growth-rate panel: beta - 1 = gamma t, from the crossing
        TM = 2.0 * T0
        gr_ax = Axes(x_range=[0, TM, 10], y_range=[0, GAMMA * TM, 0.01], x_length=5.0, y_length=3.0,
                     axis_config={"color": RULE, "stroke_width": 2, "include_ticks": False}, tips=False)
        gr_ax.move_to(RIGHT * 3.55 + DOWN * 0.35)
        gr_yl = mtex(r"\beta - 1", size=SIZE_LABEL, color=INK_2).next_to(gr_ax.c2p(0, GAMMA * TM), UP, buff=0.12)
        gr_line = Line(gr_ax.c2p(0, 0), gr_ax.c2p(TM, GAMMA * TM), color=INK, stroke_width=3)
        gr_ll = mtex(r"\gamma\, t", size=SIZE_LABEL, color=INK).next_to(gr_ax.c2p(TM, GAMMA * TM), RIGHT, buff=0.12)
        gr_name = label("growth rate past one", size=SIZE_SMALL, color=INK_3).next_to(gr_ax, UP, buff=0.45)
        gr_xl = mtex("t", size=SIZE_LABEL, color=INK_2).next_to(gr_ax.c2p(TM, 0), RIGHT, buff=0.15)
        gr_xn = mtex(r"\text{sweeps since }\beta = 1", size=SIZE_LABEL, color=INK_3).next_to(gr_ax, DOWN, buff=0.18)
        self.play_at("rate", FadeIn(VGroup(gr_ax, gr_name)), TransformFromCopy(drift[1], gr_yl), run_time=0.9)
        self.at(self.wt("gamma", cue("rate")), Create(gr_line), FadeIn(gr_ll), run_time=0.6)
        # t = 0 is where the ramp crosses beta = 1 (left), the origin of the growth panel (right)
        B1 = (1.0 - 0.6) * total                     # sweeps of the straight ramp at beta = 1
        one_line = DashedLine(st_ax.c2p(0, 1.0), st_ax.c2p(total, 1.0), color=RULE, stroke_width=1.5,
                              dash_length=0.08)
        one_l = mtex(r"\beta = 1", size=SIZE_SMALL, color=INK_3).next_to(st_ax.c2p(0, 1.0), LEFT, buff=0.12)
        cross = Dot(st_ax.c2p(B1, 1.0), radius=0.07, color=INK)
        t0_l = mtex("t = 0", size=SIZE_SMALL, color=INK_2).next_to(cross, UL, buff=0.08)
        o_dot = Dot(gr_ax.c2p(0, 0), radius=0.07, color=INK)
        self.play_at("t", Write(gr_xl), FadeIn(gr_xn), run_time=0.7)
        self.at(self.wt("crossed", cue("t")) - 0.3, Create(one_line), FadeIn(one_l), FadeIn(cross), FadeIn(t0_l),
                TransformFromCopy(cross, o_dot), run_time=0.9)

        t_cur = ValueTracker(0.0)

        def area_poly():
            tc = max(t_cur.get_value(), 1e-3)
            return Polygon(gr_ax.c2p(0, 0), gr_ax.c2p(tc, 0), gr_ax.c2p(tc, GAMMA * tc),
                           stroke_width=0, fill_color=INK_3, fill_opacity=0.4)

        area = always_redraw(area_poly)
        self.add(area)
        self.bring_to_front(gr_line)
        mult = mtex(r"\text{imbalance} \times e^{", r"\gamma t^2/2", "}", size=SIZE_BODY, color=INK)
        mult.move_to(RIGHT * 3.55 + UP * 2.35)
        T1 = math.sqrt(2 / GAMMA)                    # the area reaches 1 here
        area_full = mtex(r"\gamma t^2/2", r"= 1", size=SIZE_LABEL, color=INK)
        area_full.move_to(gr_ax.c2p(0.68 * T1, 0.2 * GAMMA * T1))
        area_l = area_full[0].copy()
        self.play_at("total", Write(mult[0]), run_time=1.2)
        self.at(self.wt("area", cue("total")), t_cur.animate.set_value(1.05 * T0), run_time=1.0)
        area_l.move_to(gr_ax.c2p(0.72 * T0, 0.2 * GAMMA * T0))
        self.at(self.wt("gamma", cue("total") + 1.0), Write(mult[1:]), FadeIn(area_l), run_time=1.0)

        # the area reaches 1 at t = sqrt(2/gamma): the label sits inside the triangle, the interval
        # is a brace under the t axis (the axis name moves down to make room)
        self.play_at("interval", t_cur.animate.set_value(T1), Transform(area_l, area_full[0]), run_time=1.5)
        self.at(self.wt("one", cue("interval")) - 0.1, FadeIn(area_full[1]), run_time=0.4)
        tick = Line(gr_ax.c2p(T1, 0) + DOWN * 0.1, gr_ax.c2p(T1, 0) + UP * 0.1, color=INK_2, stroke_width=2.5)
        int_brace = BraceBetweenPoints(gr_ax.c2p(0, 0), gr_ax.c2p(T1, 0), direction=DOWN, color=INK_2, buff=0.12)
        tick_l = mtex(r"T \sim \gamma^{-1/2}", size=SIZE_LABEL, color=INK_2).next_to(int_brace, DOWN, buff=0.08)
        self.at(self.wt("gamma", cue("interval") + 2.0), Create(tick), GrowFromCenter(int_brace), Write(tick_l),
                gr_xn.animate.next_to(tick_l, DOWN, buff=0.15).align_to(gr_ax, RIGHT), run_time=0.8)
        sel_name = label("selection interval", size=SIZE_SMALL, color=INK_2).next_to(tick_l, RIGHT, buff=0.25)

        # the fork returns; its band becomes the selection interval
        panels = VGroup(st_ax, stairs, st_yl, st_xl, st_name, ramp, tri, g_sym, simp, one_line, one_l, cross, t0_l,
                        gr_ax, gr_yl, gr_line, gr_ll, gr_name, gr_xl, gr_xn, o_dot, area, mult, area_l, area_full[1],
                        tick, int_brace, tick_l, drift)
        area.clear_updaters()
        self.play_at("selection", FadeIn(sel_name), run_time=0.4)
        # the fork returns (full size); its grey band is where the runs settle their sign
        fork = fork_home
        fork_ax, band, pre_paths, post_paths = fork[0], fork[1], fork[4], fork[5]
        self.play(FadeOut(panels), FadeOut(sel_name), run_time=0.45)
        self.play(FadeIn(fork), FadeIn(fork_tag), run_time=0.5)
        sel_brace = BraceBetweenPoints(fork_ax.c2p(W_LO, 1) + UP * 0.08, fork_ax.c2p(W_HI, 1) + UP * 0.08,
                                       direction=UP, color=INK_2)
        sel_l = label("where the runs settle their sign", size=SIZE_SMALL, color=INK_2)
        sel_l.next_to(sel_brace, UP, buff=0.1)
        self.play(GrowFromCenter(sel_brace), FadeIn(sel_l), band.animate.set_fill(opacity=0.42), run_time=0.45)
        self.play_at("open", pre_paths.animate.set_stroke(color=INK_2, width=2.4),
                     post_paths.animate.set_stroke(opacity=0.25), run_time=0.9)
        self.play_at("runaway", *[p.animate.set_stroke(color=MODE_A if e > 0 else MODE_B, opacity=1, width=2.6)
                                  for p, e in zip(post_paths, ends)], run_time=1.4)

        # ============================================================ beat: compete
        TMAX = 16 * T0
        cp_ax = Axes(x_range=[0, TMAX, 100], y_range=[0, 0.8, 0.2], x_length=7.6, y_length=4.4,
                     axis_config={"color": RULE, "stroke_width": 2, "include_ticks": False}, tips=False)
        cp_ax.move_to(LEFT * 2.3 + DOWN * 0.45)
        cp_xl = mtex("T", size=SIZE_BODY, color=INK_2).next_to(cp_ax.c2p(TMAX, 0), RIGHT, buff=0.15)
        cp_xn = label("length of the selection interval", size=SIZE_SMALL, color=INK_3).next_to(cp_ax, DOWN, buff=0.2)
        cp_yl = label("size of the imbalance added", size=SIZE_SMALL, color=INK_3).next_to(cp_ax.c2p(0, 0.8), UP, buff=0.15)
        cp_yl.align_to(cp_ax.c2p(0, 0.8), LEFT)
        f_line = Line(cp_ax.c2p(0, 0), cp_ax.c2p(TMAX, G_EX * TMAX), color=BIAS, stroke_width=4)
        Ts = np.linspace(0, TMAX, 200)
        n_curve = VMobject(stroke_width=4, stroke_color=NOISE).set_points_smoothly(
            [cp_ax.c2p(t, math.sqrt(2 * t / N_EX)) for t in Ts])
        f_lab = mtex("G", r"\,T \sim", "G", r"\,\gamma^{-1/2}", size=SIZE_BODY, colors={"G": BIAS})
        f_lab.next_to(cp_ax.c2p(TMAX, G_EX * TMAX), RIGHT, buff=0.25)
        n_lab = mtex(r"\sqrt{T/N} \sim \gamma^{-1/4}/\sqrt{N}", size=SIZE_BODY, color=INK_3)
        n_lab.next_to(cp_ax.c2p(TMAX, math.sqrt(2 * TMAX / N_EX)), RIGHT, buff=0.25)
        f_name = label("field", size=SIZE_SMALL, color=BIAS).next_to(f_lab, UP, buff=0.1).align_to(f_lab, LEFT)
        n_name = label("fluctuations", size=SIZE_SMALL, color=INK_3).next_to(n_lab, UP, buff=0.1).align_to(n_lab, LEFT)
        cp_tag = label("N = 2,048 · G = 0.0006125", size=SIZE_SMALL, color=INK_3).to_corner(DL, buff=0.35)

        fork_all = VGroup(fork, sel_brace, sel_l)
        self.play_at("field-grows", FadeOut(fork_all), FadeOut(fork_tag),
                     FadeIn(VGroup(cp_ax, cp_xl, cp_xn, cp_yl)), FadeIn(cp_tag), run_time=0.9)
        self.play(Create(f_line), run_time=2.6, rate_func=linear)
        self.play(FadeIn(f_name), Write(f_lab), run_time=1.0)
        self.play_at("fluct-grows", Create(n_curve), run_time=2.0, rate_func=linear)
        self.play(FadeIn(n_name), Write(n_lab), run_time=1.0)

        # the ratio
        num = mtex("G", r"\,\gamma^{-1/2}", size=SIZE_BODY, colors={"G": BIAS})
        den = mtex(r"\gamma^{-1/4}/\sqrt{N}", size=SIZE_BODY, color=INK_3)
        bar = Line(LEFT, RIGHT, color=INK_2, stroke_width=2)
        bar.set_width(max(num.width, den.width) + 0.2)
        frac = VGroup(num, bar, den).arrange(DOWN, buff=0.12)
        res = mtex("=", "G", r"\sqrt{N}", r"\,\gamma^{-1/4}", size=SIZE_HEAD, colors={"G": BIAS})
        ratio = VGroup(frac, res).arrange(RIGHT, buff=0.25)
        ratio.move_to(RIGHT * 3.1 + UP * 2.45)
        ratio_name = label("field ÷ fluctuations", size=SIZE_SMALL, color=INK_2).next_to(ratio, UP, buff=0.2)
        self.play_at("ratio", TransformFromCopy(f_lab[2:], num), TransformFromCopy(n_lab[0][-12:], den),
                     Create(bar), FadeIn(ratio_name), run_time=1.1)
        self.at(self.wt("G", cue("ratio")) - 0.05, Write(res), run_time=1.6)

        # slower cooling: a longer interval; the field gains on the fluctuations
        Tm = ValueTracker(T0)
        mk_f = always_redraw(lambda: Dot(cp_ax.c2p(Tm.get_value(), G_EX * Tm.get_value()), radius=0.08, color=BIAS))
        mk_n = always_redraw(lambda: Dot(cp_ax.c2p(Tm.get_value(), math.sqrt(2 * Tm.get_value() / N_EX)),
                                         radius=0.08, color=INK_2))
        mk_v = always_redraw(lambda: DashedLine(cp_ax.c2p(Tm.get_value(), 0),
                                                cp_ax.c2p(Tm.get_value(), math.sqrt(2 * Tm.get_value() / N_EX)),
                                                color=INK_3, stroke_width=1.5, dash_length=0.06))
        slow = VGroup(Arrow(cp_ax.c2p(0.35 * TMAX, 0) + DOWN * 0.3, cp_ax.c2p(0.9 * TMAX, 0) + DOWN * 0.3,
                            buff=0, color=INK_2, stroke_width=3, max_tip_length_to_length_ratio=0.06),
                      label("slower cooling", size=SIZE_SMALL, color=INK_2))
        slow[1].next_to(slow[0], DOWN, buff=0.08)
        # the read-out of the ratio at the marker: a field-coloured bar that grows like sqrt(T)
        def ratio_now():
            tv = Tm.get_value()
            return G_EX * tv / math.sqrt(2 * tv / N_EX)

        RB = 5.0                                      # scene units per unit of the ratio
        ro_name = tex(r"ratio at the marker: grows like $\sqrt{T}$", size=SIZE_SMALL, color=INK_2)
        ro_name.move_to(RIGHT * 3.6 + DOWN * 2.35).align_to(RIGHT * 2.55, LEFT)
        ro_x0 = np.array([2.55, -2.8, 0])
        ro_bar = always_redraw(lambda: Rectangle(width=max(RB * ratio_now(), 1e-3), height=0.2, stroke_width=0,
                                                 fill_color=BIAS, fill_opacity=0.85)
                               .move_to(ro_x0 + RIGHT * RB * ratio_now() / 2))
        # no number on the bar: the ratio is an order-of-magnitude relation ('~', no prefactors), and the
        # exact field-over-fluctuations value comes later as a = 0.208
        self.play_at("slower", FadeIn(VGroup(mk_v, mk_f, mk_n)), FadeOut(cp_xn), GrowArrow(slow[0]), FadeIn(slow[1]),
                     FadeIn(ro_name), FadeIn(ro_bar), run_time=0.6)
        self.play(Tm.animate.set_value(TMAX), run_time=2.2, rate_func=smooth)

        # ============================================================ beat: gaussian
        N_RUNS, N_DRAWN = 4000, 120
        ts, rec = simulate_ramp(N_RUNS, seed=7, t_end=4 * T0)
        y_all = rec[-1] * math.exp(-GAMMA * ts[-1] ** 2 / 2)      # settled: w(4 T0) = e^-8
        show = (ts >= -2.5 * T0 - 1e-9) & (ts <= 1.5 * T0 + 1e-9)
        ts_s, rec_s = ts[show][::4], rec[show][::4, :N_DRAWN]
        w_s = np.exp(-GAMMA * ts_s ** 2 / 2)
        Y_R = 1.6
        # time t is plotted at t + 2.5 gamma^(-1/2), so the m axis sits at the left edge, not at beta = 1
        rn_ax = Axes(x_range=[0, 4 * T0, 10], y_range=[-Y_R, Y_R, 0.5], x_length=10.0, y_length=4.4,
                     axis_config={"color": RULE, "stroke_width": 2, "include_ticks": False}, tips=False)
        rn_ax.move_to(DOWN * 0.45)

        def rp(t, v):
            return rn_ax.c2p(t + 2.5 * T0, v)

        rn_yl = mtex("m", size=SIZE_BODY, color=INK_2).next_to(rp(-2.5 * T0, Y_R), UP, buff=0.12)
        rn_xl = label("sweeps", size=SIZE_SMALL, color=INK_3).next_to(rp(1.5 * T0, -Y_R), DOWN, buff=0.15)
        rn_xl.align_to(rp(1.5 * T0, -Y_R), RIGHT)
        rn_one = VGroup(DashedLine(rp(0, -Y_R), rp(0, Y_R), color=RULE, stroke_width=1.5, dash_length=0.08),
                        mtex(r"\beta = 1", size=SIZE_SMALL, color=INK_3))
        rn_one[1].next_to(rp(0, -Y_R), DOWN, buff=0.15)
        rn_tag = mtex(r"\text{linear ramp} \cdot N = 2{,}048 \cdot \gamma = 0.001 \cdot G = 0.0006125 \cdot "
                      r"\text{simulated, linearized near } m = 0", size=SIZE_LABEL, color=INK_3).to_corner(DL, buff=0.35)

        def path_of(vals):
            pts = [rp(t, float(np.clip(v, -Y_R, Y_R))) for t, v in zip(ts_s, vals)]
            return VMobject(stroke_width=1.4, stroke_color=INK_3, stroke_opacity=0.7).set_points_as_corners(pts)

        paths_m = VGroup(*[path_of(rec_s[:, i]) for i in range(rec_s.shape[1])])
        y_end = y_all[:N_DRAWN]
        paths_y = VGroup(*[path_of(rec_s[:, i] * w_s) for i in range(rec_s.shape[1])])
        for p, ye in zip(paths_y, rec_s[-1]):
            p.set_stroke(color=MODE_A if ye > 0 else MODE_B, opacity=0.75)
        undo = mtex(r"\div\, e^{\gamma t^2/2}", size=SIZE_BODY, color=INK).next_to(rp(1.5 * T0, Y_R), DL, buff=0.1)
        undo.shift(UP * 0.35 + LEFT * 0.2)
        compete = VGroup(cp_ax, cp_xl, cp_yl, f_line, n_curve, f_lab, n_lab, f_name, n_name, mk_f, mk_n, mk_v, slow,
                         frac, res[0], ratio_name, ro_name, ro_bar)
        for m_ in (mk_f, mk_n, mk_v, ro_bar):
            m_.clear_updaters()
        self.play_at("gaussian", FadeOut(compete), FadeOut(cp_tag),
                     res[1:].animate.scale(0.75).to_corner(UL, buff=0.45),
                     FadeIn(VGroup(rn_ax, rn_yl, rn_xl, rn_one)), FadeIn(rn_tag), run_time=0.8)
        self.play(LaggedStart(*[Create(p) for p in paths_m], lag_ratio=0.004), run_time=1.3, rate_func=linear)
        # undo the growth: every run divided by the same factor, so each keeps its sign
        yl_div = mtex("m", r"\div e^{\gamma t^2/2}", size=SIZE_BODY, color=INK_2)
        yl_div.move_to(rn_yl, aligned_edge=LEFT)
        self.at(self.wt("undo", cue("gaussian")), Write(undo),
                *[Transform(a, b.copy().set_stroke(color=INK_3, opacity=0.7)) for a, b in zip(paths_m, paths_y)],
                ReplacementTransform(rn_yl, yl_div[0]), TransformFromCopy(undo, yl_div[1]), run_time=1.6)
        self.at(self.wt("each", cue("gaussian")), *[a.animate.set_stroke(color=b.get_stroke_color(), opacity=0.75)
                                                    for a, b in zip(paths_m, paths_y)], run_time=1.0)

        # the settled values pile into a histogram (4,000 runs); the limiting Gaussian lays over it
        BASE, H = -1.3, 2.3              # baseline; height of the Gaussian's peak
        BW = 0.5 * SD                     # bin width in y (edges at multiples, so 0 is an edge)
        g_ax = NumberLine(x_range=[-3.3 * SD, 3.3 * SD, SD], length=6.6 * 1.25, color=RULE, stroke_width=2,
                          include_ticks=False)
        g_ax.shift(np.array([0, BASE, 0]) - g_ax.n2p(0))
        zero_g = VGroup(Line(g_ax.n2p(0) + DOWN * 0.1, g_ax.n2p(0) + UP * 0.1, color=INK_3, stroke_width=2),
                        mtex("0", size=SIZE_SMALL, color=INK_3))
        zero_g[1].next_to(zero_g[0], DOWN, buff=0.1)

        def gy(yv):
            return BASE + H * gauss_pdf((yv - MEAN) / SD) / gauss_pdf(0)

        edges = np.arange(-6, 7) * BW
        cnt, _ = np.histogram(y_all, bins=edges)
        bars = VGroup()
        for lo, k in zip(edges[:-1], cnt):
            h = H * (k / (N_RUNS * BW)) * SD / gauss_pdf(0)
            if h <= 0:
                continue
            r = Rectangle(width=g_ax.n2p(lo + BW)[0] - g_ax.n2p(lo)[0] - 0.04, height=h, stroke_width=0,
                          fill_color=MODE_A if lo >= 0 else MODE_B, fill_opacity=0.55)
            r.move_to([g_ax.n2p(lo + BW / 2)[0], BASE + h / 2, 0])
            bars.add(r)
        runs_l = label("4,000 runs", size=SIZE_SMALL, color=INK_3).move_to([g_ax.n2p(-2.6 * SD)[0], BASE + 1.0, 0])
        drop = VGroup(*[Dot(rp(ts_s[-1], float(np.clip(rec_s[-1, i] * w_s[-1], -Y_R, Y_R))), radius=0.04,
                            color=MODE_A if y_end[i] > 0 else MODE_B) for i in range(N_DRAWN)])
        ygrid = np.linspace(-3.3 * SD, 3.3 * SD, 181)
        g_curve = theory(VMobject(stroke_width=3).set_points_smoothly([[g_ax.n2p(v)[0], gy(v), 0] for v in ygrid]),
                         num_dashes=60)
        self.at(self.wt("the", cue("gaussian") + 9), FadeIn(drop), FadeOut(paths_m), FadeOut(undo),
                FadeOut(VGroup(rn_ax, rn_xl, rn_one)), FadeIn(g_ax), FadeIn(zero_g),
                yl_div.animate.scale(0.8).next_to(g_ax, RIGHT, buff=0.2), run_time=0.35)
        self.play(*[d.animate.move_to([g_ax.n2p(float(np.clip(ye, -3.3 * SD, 3.3 * SD)))[0], BASE + 0.04, 0])
                    for d, ye in zip(drop, y_end)], run_time=0.45)
        self.play(FadeOut(drop), LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.05),
                  FadeIn(runs_l), run_time=0.65)
        self.at(self.wt("approx", cue("gaussian")), Create(g_curve), bars.animate.set_fill(opacity=0.3),
                run_time=1.1)

        # mean (field) and spread (fluctuations)
        top = gy(MEAN)
        mean_line = Line([g_ax.n2p(MEAN)[0], BASE, 0], [g_ax.n2p(MEAN)[0], top, 0], color=BIAS, stroke_width=3)
        mean_l = label("field", size=SIZE_SMALL, color=BIAS).next_to(mean_line, UP, buff=0.1)
        self.play_at("mean", Create(mean_line), FadeIn(mean_l), run_time=0.8)
        h1 = gy(MEAN + SD)
        spread = DoubleArrow([g_ax.n2p(MEAN - SD)[0], h1, 0], [g_ax.n2p(MEAN + SD)[0], h1, 0], buff=0,
                             color=INK_2, stroke_width=3, tip_length=0.18)
        spread_l = label("fluctuations", size=SIZE_SMALL, color=INK_2).next_to(spread, RIGHT, buff=0.35)
        self.play_at("spread", GrowFromCenter(spread), FadeIn(spread_l), run_time=0.8)
        a_def = mtex("a", "=", r"\frac{\text{mean}}{\text{sd}}", size=SIZE_BODY, color=INK)
        a_def.move_to(LEFT * 4.9 + UP * 0.55)
        self.play_at("a", Write(a_def), run_time=1.2)

        def area_piece(lo, hi, color):
            vs = ygrid[(ygrid >= lo) & (ygrid <= hi)]
            vs = np.concatenate([[lo], vs, [hi]])
            pts = [[g_ax.n2p(v)[0], gy(v), 0] for v in vs]
            return Polygon(*pts, [pts[-1][0], BASE, 0], [pts[0][0], BASE, 0], stroke_width=0,
                           fill_color=color, fill_opacity=0.45)

        pos_area = area_piece(0.0, ygrid[-1], MODE_A)
        neg_area = area_piece(ygrid[0], 0.0, MODE_B)
        phi_l = mtex(r"\Phi(a)", size=SIZE_HEAD, color=INK).move_to([g_ax.n2p(1.0 * SD)[0], BASE + 0.5, 0])
        self.play_at("area", FadeIn(pos_area), FadeIn(neg_area), FadeOut(bars), FadeOut(runs_l), run_time=1.0)
        self.bring_to_front(g_curve, mean_line, spread)
        self.at(self.wt("phi", cue("area")), Write(phi_l), run_time=0.8)

        # ============================================================ beat: formula
        formula = mtex(r"q_{\text{sel}}", r"\approx", r"\Phi(a),", r"\quad a =", "G", r"\sqrt{N}\,",
                       r"(\pi", r"/\gamma)^{1/4}", size=SIZE_HEAD, color=THEORY, colors={"G": BIAS})
        formula.move_to(UP * 2.95)
        f_under = theory(Line(formula.get_corner(DL) + DOWN * 0.12, formula.get_corner(DR) + DOWN * 0.12,
                              stroke_width=2), num_dashes=40)
        cite = label("Kondepudi & Nelson 1983 · Kondepudi & Gao 1987", size=SIZE_SMALL, color=INK_3)
        cite.next_to(formula, DOWN, buff=0.3)
        ratio_g, ratio_rest = res[1], res[2:]
        q_name = label("chance of the positive sign", size=SIZE_SMALL, color=INK_3)
        q_name.next_to(formula[0], UP, buff=0.22).align_to(formula[0], LEFT)
        self.play_at("classical", Write(formula[:2]), TransformFromCopy(phi_l, formula[2]), Write(formula[3]),
                     ReplacementTransform(ratio_g, formula[4]), ReplacementTransform(ratio_rest, formula[5:]),
                     Create(f_under), FadeIn(cite), FadeIn(q_name), run_time=1.2)
        # the transforms left the formula's parts on stage as separate mobjects: put the whole back
        fam = set(formula.get_family())
        self.remove(*[m for m in self.mobjects if m is not formula and m in fam])
        self.add(formula)
        noswitch = mtex(r"\text{little switching later: final share} \approx \Phi(a)", size=SIZE_LABEL, color=INK_2)
        noswitch.next_to(cite, DOWN, buff=0.18)
        self.play_at("no-switch", FadeIn(noswitch), run_time=0.8)
        self.at(self.wt("phi", cue("no-switch")), Indicate(formula[2], color=INK, scale_factor=1.2),
                Indicate(phi_l, color=INK, scale_factor=1.15), run_time=0.9)

        # ============================================================ beat: example
        self.play_at("example", Indicate(rn_tag, color=INK_2, scale_factor=1.04), run_time=1.0)
        a_val = mtex("=", "0.208", size=SIZE_BODY, color=INK).next_to(a_def, RIGHT, buff=0.15)
        self.play_at("a-value", Indicate(a_def[0], color=INK, scale_factor=1.3), run_time=0.6)
        self.at(self.wt("zero", cue("a-value")), Write(a_val), run_time=0.9)

        # the share of the positive sign on a 0..100% bar
        BAR_Y, BAR_W = BASE - 1.15, 6.0
        x0 = -BAR_W / 2

        def bx(p):
            return x0 + BAR_W * p

        bar_bg = Rectangle(width=BAR_W, height=0.22, stroke_width=0, fill_color=MODE_B, fill_opacity=0.55)
        bar_bg.move_to([0, BAR_Y, 0])
        share = ValueTracker(0.0)
        bar_fg = always_redraw(lambda: Rectangle(width=max(BAR_W * share.get_value(), 1e-3), height=0.22,
                                                 stroke_width=0, fill_color=MODE_A, fill_opacity=0.9)
                               .move_to([x0 + BAR_W * share.get_value() / 2, BAR_Y, 0]))
        half = Line([bx(0.5), BAR_Y - 0.16, 0], [bx(0.5), BAR_Y + 0.16, 0], color=INK_3, stroke_width=1.5)
        b0 = label("0%", size=SIZE_SMALL, color=INK_3).next_to(bar_bg, LEFT, buff=0.15)
        b1 = label("100%", size=SIZE_SMALL, color=INK_3).next_to(bar_bg, RIGHT, buff=0.15)
        bname = label("positive sign", size=SIZE_SMALL, color=MODE_A).next_to(b0, LEFT, buff=0.3)
        self.play_at("share", FadeIn(VGroup(bar_bg, half, b0, b1, bname)), FadeIn(bar_fg),
                     FadeOut(zero_g[1]), run_time=0.6)
        self.play(share.animate.set_value(PHI_A), run_time=1.4, rate_func=smooth)
        p58 = label("ramp: 58%", size=SIZE_SMALL, color=MODE_A).next_to([bx(PHI_A), BAR_Y, 0], DOWN, buff=0.22)
        self.at(self.wt("fifty", cue("share")), Write(p58), run_time=0.6)
        q_chain = json.loads((VISUAL / "data" / "c_cw.json").read_text())
        qc = self._q_chain(q_chain)
        st_tick = Line([bx(qc), BAR_Y - 0.11, 0], [bx(qc), BAR_Y + 0.25, 0], color=INK_2, stroke_width=2.5)
        st_lab = label("the paper's schedule: 61%", size=SIZE_SMALL, color=INK_2).next_to(st_tick, UP, buff=0.1)
        st_lab.align_to(st_tick, LEFT).shift(RIGHT * 0.05)
        self.at(self.wt("sixty", cue("share")) - 0.1, Create(st_tick), FadeIn(st_lab), run_time=0.6)

        # the answer: a small mean shift against a wide spread
        self.play_at("answer", bar_fg.animate.set_opacity(0.9), run_time=0.3)
        self.at(self.wt("lost", cue("answer")), Indicate(ctag, color=GILT_TEXT, scale_factor=1.25), run_time=0.8)
        self.at(self.wt("weak", cue("answer")), Indicate(mean_line, color=BIAS, scale_factor=1.1),
                Indicate(mean_l, color=BIAS), run_time=1.1)
        self.at(self.wt("fluct", cue("answer")), Indicate(spread, color=INK, scale_factor=1.08),
                Indicate(spread_l, color=INK), run_time=1.2)

        # ============================================================ hand-off to scaling (tail)
        nxt = mtex(r"q_{\text{sel}}", r"\approx", r"\Phi(a),", r"\quad a =", "G", r"\sqrt{N}\,",
                   r"(\pi", r"/\gamma)^{1/4}", size=SIZE_HEAD, color=THEORY, colors={"G": BIAS}).move_to(UP * 2.75)
        nxt_under = theory(Line(nxt.get_corner(DL) + DOWN * 0.12, nxt.get_corner(DR) + DOWN * 0.12,
                                stroke_width=2), num_dashes=40)
        self.wait_until(self.timing["speech_end"] + 0.05)
        bar_fg.clear_updaters()
        others = [m for m in self.mobjects if m is not formula and m is not f_under]
        self.play(Transform(formula, nxt), Transform(f_under, nxt_under), *[FadeOut(m) for m in others],
                  run_time=0.6, rate_func=smooth)
        self.remove(*[m for m in self.mobjects if m is not formula and m is not f_under])

    # ---------------------------------------------------------------- data readers
    @staticmethod
    def _schedule(d):
        """Levels and sweep counts of the paper's ten-level schedule (../data/schedule.json)."""
        return list(map(float, d["ladder"])), list(map(int, d["K"]))

    @staticmethod
    def _q_chain(d):
        """q_chain of the exact count calculation at the finest time step (../data/c_cw.json)."""
        return float(d["runs"][-1]["q_chain"])
