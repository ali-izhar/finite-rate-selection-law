"""Scene "scaling" — Why slower barely helps (script.md, scene: scaling; page chapter "scaling").

Beats: derive (small a: positive area linear in a; sampler log-odds ~ G sqrt(N) gamma^(-1/4),
model log-odds ~ NG; c ~ (N^2 gamma)^(-1/4)) -> ladder (16x slower -> 2x c; the regime) ->
check (paper Fig. 2a: nine linear ramps on the formula lines) -> not-forever (the law stops at
fixed N; the formula line would cross c = 1; waiting at beta = 1.6 is a separate knob, drawn as a
vertical arrow, not a gamma; illustration) -> staircase (the paper's ten-level schedule, final
cooling x1/4 -> x4 in one stretch that also drives the grid 59 -> 63 and c 0.099 -> 0.147, x1.48;
the ramp's x2 for contrast) ->
answer (the ladder again: 16x buys 2x on a ramp, 1.48x on the staircase).

Hand-off: the scene opens on the selection formula as the race scene leaves it,
  q_sel ~ Phi(a),  a = G sqrt(N) (pi/gamma)^(1/4)   (THEORY grey, G in BIAS, SIZE_HEAD, at y = 3.05).

Numbers (no invented values; every source is in the repo or the paper):
  nine linear ramps, c_num and c_sel        ../data/cw_scaling.json "linear_sweep"
                                            (= results/experiments/cw_linear_sweep.json; paper Table 2)
  formula lines c_sel(gamma) = logit Phi(G sqrt(N)(pi/gamma)^(1/4)) / logit q_eq, G = G0*2048/N;
    q_eq per N read back from the same rows (logit q_KN / c_KN)
  c over the eight fields (0.119498-0.119808) results/experiments/exact_protocol.json (paper App. A)
  staircase levels and sweeps               ../data/schedule.json (paper Table 3)
  final cooling x1/4, x1, x4: q 0.5875, 0.6055, 0.6281; c 0.098978, 0.119808, 0.146625
                                            ../data/cw_scaling.json "schedule" (paper App. A: x1.4814)
  a = 0.208 (Fig. 1b setting)               paper App. A
The "not-forever" picture is an illustration (tagged): the paper states the limit, not a curve;
the gamma axis carries no numbers past the measured 0.00025 (axis break), so no location is implied.

Render: ./build.sh render -l scaling
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import *  # noqa: E402,F403

DATA = Path(__file__).resolve().parents[2] / "data"
SCAL = json.loads((DATA / "cw_scaling.json").read_text())
SCHED = json.loads((DATA / "schedule.json").read_text())

LIN = SCAL["linear_sweep"]
STAIR = {r["scale"]: r for r in SCAL["schedule"] if r["N"] == 2048}
G0 = 6.125e-4
NS = (1024, 2048, 4096)
GAMMAS = (0.004, 0.001, 0.00025)

# c over the paper's eight fields (results/experiments/exact_protocol.json, rounded to 6 d.p.)
FIELD_C = [(6.125e-4, 0.119808), (4.375e-4, 0.119653), (2.625e-4, 0.119550), (8.75e-5, 0.119498),
           (-8.75e-5, 0.119498), (-2.625e-4, 0.119550), (-4.375e-4, 0.119653), (-6.125e-4, 0.119808)]


def phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))


def logit(q):
    return math.log(q / (1 - q))


def run(N, gamma):
    return next(r for r in LIN if r["N"] == N and abs(r["gamma"] - gamma) < 1e-12)


# equilibrium log-odds at beta = 1.6 for each N, read back from the formula column
LO_EQ = {N: logit(run(N, 0.001)["q_KN"]) / run(N, 0.001)["c_KN"] for N in NS}


def c_sel(N, gamma):
    a = G0 * 2048 / N * math.sqrt(N) * (math.pi / gamma) ** 0.25
    return logit(phi(a)) / LO_EQ[N]


# sanity: the formula reproduces the paper's column and the data are the paper's Table 2
for _r in LIN:
    assert abs(c_sel(_r["N"], _r["gamma"]) / _r["c_KN"] - 1) < 1e-6
assert max(abs(r["c_exact"] / r["c_KN"] - 1) for r in LIN) < 0.01815
assert all(r["c_exact"] > r["c_KN"] for r in LIN)
assert abs(STAIR[4.0]["c"] / STAIR[0.25]["c"] - 1.4814) < 1e-4


class Scaling(NarratedScene):
    scene_id = "scaling"

    def at(self, t, *anims, run_time=1.0, **kw):
        """A non-cue animation at a fixed time; warns if the clock is already past it."""
        if self.now > t + 0.07:
            print(f"[scaling] late: wanted {t:.2f}, now {self.now:.2f}")
        self._wait_to(t)
        self.play(*anims, run_time=run_time, **kw)

    def word_at(self, w, after=0.0):
        """Start time of the first spoken word equal to w (punctuation ignored) at or after `after`."""
        for x in self.timing["words"]:
            if x["start"] >= after - 1e-6 and x["w"].strip(".,:;").lower() == w.lower():
                return x["start"]
        raise KeyError(w)

    # ------------------------------------------------------------------ ladder (used twice)
    def make_ladder(self, cx, H=4.2, yb=-2.35, gap=3.0, left="cooling time"):
        lx, rx = cx - gap / 2, cx + gap / 2

        def y(f):
            return yb + H * math.log(f, 16)

        rail_l = Line([lx, y(1), 0], [lx, y(16), 0], stroke_width=4, color=RULE)
        rail_r = Line([rx, y(1), 0], [rx, y(16), 0], stroke_width=4, color=RULE)
        facs = (1, 2, 4, 8, 16)
        tl = VGroup(*[VGroup(Line([lx - 0.1, y(f), 0], [lx + 0.1, y(f), 0], stroke_width=2, color=INK_3),
                             mtex(rf"{f}\times", size=SIZE_SMALL, color=INK_3).next_to([lx - 0.12, y(f), 0], LEFT, buff=0.1))
                      for f in facs])
        tr = VGroup(*[VGroup(Line([rx - 0.1, y(f), 0], [rx + 0.1, y(f), 0], stroke_width=2, color=INK_3),
                             mtex(rf"{f}\times", size=SIZE_SMALL, color=INK_3).next_to([rx + 0.12, y(f), 0], RIGHT, buff=0.1))
                      for f in facs])
        hl = label(left, color=INK_2, size=SIZE_SMALL + 2).next_to(rail_l, DOWN, buff=0.22)
        hr = mtex("c", size=SIZE_BODY, color=GILT_TEXT).next_to(rail_r, DOWN, buff=0.18)
        return dict(lx=lx, rx=rx, y=y, rail_l=rail_l, rail_r=rail_r, tl=tl, tr=tr, hl=hl, hr=hr)

    def construct(self):
        T = self.t

        # ============================================================ derive
        law = mtex(r"q_{\text{sel}}", r"\approx", r"\Phi(a),", r"\quad a", "=", "G",
                   r"\sqrt{N}\,(\pi/\gamma)^{1/4}", size=SIZE_HEAD, color=THEORY, colors={"G": BIAS})
        law.move_to(UP * 3.05)
        self.add(law)

        # left: the Gaussian of the race scene, N(a, 1); right: its positive area against a
        av = ValueTracker(0.0)
        gx = Axes(x_range=[-3.2, 3.2, 1], y_range=[0, 0.45, 0.1], x_length=5.0, y_length=2.2,
                  axis_config={"color": RULE, "stroke_width": 2, "include_ticks": False}, tips=False)
        gx.move_to(LEFT * 3.4 + DOWN * 0.2)
        g_axis = gx.x_axis
        zero = DashedLine(gx.c2p(0, 0), gx.c2p(0, 0.47), stroke_width=2, color=INK_3, dash_length=0.06)
        zero_l = mtex("0", size=SIZE_SMALL, color=INK_3).next_to(gx.c2p(0, 0), DOWN, buff=0.12)

        def dens(x, a):
            return math.exp(-(x - a) ** 2 / 2) / math.sqrt(2 * math.pi)

        def bell():
            a = av.get_value()
            return gx.plot(lambda x: dens(x, a), x_range=[-3.2, 3.2, 0.02], color=INK_2, stroke_width=3)

        def area(lo, hi, col):
            a = av.get_value()
            xs = np.linspace(lo, hi, 80)
            pts = [gx.c2p(lo, 0)] + [gx.c2p(x, dens(x, a)) for x in xs] + [gx.c2p(hi, 0)]
            return Polygon(*pts, stroke_width=0, fill_color=col, fill_opacity=0.55)

        curve = always_redraw(bell)
        pos_area = always_redraw(lambda: area(0, 3.2, MODE_A))
        neg_area = always_redraw(lambda: area(-3.2, 0, MODE_B))
        mean_ln = always_redraw(lambda: Line(gx.c2p(av.get_value(), 0), gx.c2p(av.get_value(), 0.43),
                                             stroke_width=3, color=BIAS))
        a_lab = always_redraw(lambda: mtex("a", size=SIZE_LABEL, color=BIAS)
                              .next_to(gx.c2p(av.get_value(), 0.43), UP, buff=0.08))

        px = Axes(x_range=[0, 0.6, 0.2], y_range=[0.5, 0.75, 0.05], x_length=4.0, y_length=2.8,
                  axis_config={"color": RULE, "stroke_width": 2, "include_ticks": False}, tips=False)
        px.move_to(RIGHT * 3.3 + DOWN * 0.05)
        px_xl = mtex("a", size=SIZE_BODY, color=BIAS).next_to(px.c2p(0.6, 0.5), RIGHT, buff=0.15)
        px_yl = label("positive area", color=MODE_A, size=SIZE_SMALL + 2).next_to(px.c2p(0, 0.75), UP, buff=0.12)
        px_ticks = VGroup(*[mtex(s, size=SIZE_SMALL, color=INK_3).next_to(px.c2p(0, v), LEFT, buff=0.1)
                            for v, s in ((0.5, "0.5"), (0.75, "0.75"))],
                          *[mtex(s, size=SIZE_SMALL, color=INK_3).next_to(px.c2p(v, 0.5), DOWN, buff=0.12)
                            for v, s in ((0.0, "0"), (0.6, "0.6"))])
        trace = always_redraw(lambda: px.plot(phi, x_range=[0, max(av.get_value(), 1e-3), 0.005],
                                              color=MODE_A, stroke_width=4))
        tdot = always_redraw(lambda: Dot(px.c2p(av.get_value(), phi(av.get_value())), radius=0.07, color=MODE_A))
        tangent = theory(Line(px.c2p(0, 0.5), px.c2p(0.6, 0.5 + 0.6 / math.sqrt(2 * math.pi)))
                         .set_stroke(width=3), num_dashes=22)
        near = tex(r"near $a = 0$: a straight line", size=SIZE_SMALL + 2, color=INK_3)
        near.next_to(px.c2p(0.3, 0.5), DOWN, buff=0.55)
        ex_dot = Dot(px.c2p(0.208, phi(0.208)), radius=0.06, color=INK_2)
        ex_lab = tex(r"Fig.~1b: $a = 0.208$", size=SIZE_SMALL, color=INK_3).next_to(ex_dot, RIGHT, buff=0.15).shift(DOWN * 0.12)

        # "For small a, the positive area grows in a straight line with a."
        self.play_at("weak", Create(g_axis), FadeIn(zero, zero_l), Create(curve),
                     Create(VGroup(px.x_axis, px.y_axis)), FadeIn(px_xl, px_yl, px_ticks), run_time=1.2)
        self.add(curve)
        self.at(T("weak", 1.3), FadeIn(neg_area, pos_area), FadeIn(mean_ln, a_lab), run_time=0.7)
        self.add(trace, tdot)
        # on "grows in a straight line"; the last fade ends just before "So the sampler's log-odds"
        self.at(self.word_at("grows"), av.animate.set_value(0.6), Create(tangent), run_time=1.6, rate_func=smooth)
        self.at(T("sampler-odds", -0.7), FadeIn(near), FadeIn(ex_dot, ex_lab), run_time=0.6)

        # "So the sampler's log-odds go like a, the ratio of field to fluctuations found above."
        lab1 = label("sampler's log-odds", color=INK_2, size=SIZE_LABEL)
        r1a = mtex(r"\propto", "a", size=SIZE_HEAD, colors={"a": BIAS})
        r1b = mtex(r"\propto", "G", r"\sqrt{N}\,\gamma^{-1/4}", size=SIZE_HEAD, colors={"G": BIAS})
        lab2 = label("model's log-odds", color=INK_2, size=SIZE_LABEL)
        r2 = mtex(r"\propto", "N", "G", size=SIZE_HEAD, colors={"G": BIAS})
        lab1.move_to(LEFT * 1.0 + UP * 1.2, aligned_edge=RIGHT)
        lab2.move_to(LEFT * 1.0 + DOWN * 0.05, aligned_edge=RIGHT)
        for lab, r in ((lab1, r1a), (lab1, r1b), (lab2, r2)):
            r.next_to(lab, RIGHT, buff=0.35)
            lab.match_y(r[0])
        r1sub = label("field / fluctuations", size=SIZE_SMALL, color=INK_3).next_to(r1b[1:], DOWN, buff=0.14)
        derive_plots = VGroup(g_axis, zero, zero_l, curve, pos_area, neg_area, mean_ln, a_lab,
                              px.x_axis, px.y_axis, px_xl, px_yl, px_ticks, trace, tdot, tangent, near, ex_dot, ex_lab)
        self.play_at("sampler-odds", FadeOut(derive_plots), FadeIn(lab1, shift=RIGHT * 0.2),
                     Write(r1a[0]), TransformFromCopy(px_xl, r1a[1]), run_time=1.0)
        self.at(T("sampler-odds", 2.92), Indicate(law[5:7], color=BIAS, scale_factor=1.12),
                ReplacementTransform(r1a[0], r1b[0]), FadeTransform(r1a[1], r1b[1:]), run_time=1.1)
        self.at(T("sampler-odds", 4.3), FadeIn(r1sub, shift=UP * 0.1), run_time=0.6)

        # "The model's log-odds go like N times G, since the field acts on all N spins."
        self.play_at("model-odds", FadeIn(lab2, shift=RIGHT * 0.2), Write(r2), run_time=1.3)
        n_sp = 16
        spins = VGroup(*[Dot(radius=0.08, color=INK_2) for _ in range(n_sp)])
        spins.arrange(RIGHT, buff=0.36).move_to(DOWN * 2.0)
        pushes = VGroup(*[Arrow(s.get_top() + UP * 0.08, s.get_top() + UP * 0.5, buff=0, stroke_width=3,
                                max_tip_length_to_length_ratio=0.35, color=BIAS) for s in spins])
        gsym = mtex("G", size=SIZE_LABEL, color=BIAS).next_to(pushes[-1], RIGHT, buff=0.15)
        nbrace = Brace(spins, DOWN, buff=0.12, color=INK_3)
        nsym = mtex("N", r"\text{ spins}", size=SIZE_LABEL, color=INK_2).next_to(nbrace, DOWN, buff=0.08)
        self.at(T("model-odds", 3.25), LaggedStart(*[FadeIn(s, shift=UP * 0.1) for s in spins], lag_ratio=0.04),
                LaggedStart(*[GrowArrow(p) for p in pushes], lag_ratio=0.04), FadeIn(gsym), run_time=1.1)
        self.at(T("model-odds", 4.95), GrowFromCenter(nbrace), FadeIn(nsym), run_time=0.6)

        # "Divide, and G cancels, as the flat c showed:"
        num = mtex("G", r"\sqrt{N}\,\gamma^{-1/4}", size=SIZE_HEAD, colors={"G": BIAS})
        den = mtex("N", "G", size=SIZE_HEAD, colors={"G": BIAS})
        bar = Line(LEFT, RIGHT, stroke_width=3, color=INK).set_width(max(num.width, den.width) + 0.25)
        num.next_to(bar, UP, buff=0.14)
        den.next_to(bar, DOWN, buff=0.14)
        cprop = mtex("c", r"\propto", size=SIZE_HEAD, colors={"c": GILT_TEXT})
        whole = VGroup(cprop, VGroup(num, bar, den)).arrange(RIGHT, buff=0.3).move_to(LEFT * 1.6 + UP * 0.75)
        cprop.match_y(bar)
        self.play_at("divide",
                     FadeOut(VGroup(lab1, lab2, r1b[0], r2[0], r1sub, spins, pushes, gsym, nbrace, nsym)),
                     ReplacementTransform(r1b[1:], num), ReplacementTransform(r2[1:], den),
                     GrowFromCenter(bar), FadeIn(cprop), run_time=0.9)
        g_num, g_den = num[0], den[1]
        strikes = VGroup(*[Line(g.get_corner(DL) + DL * 0.04, g.get_corner(UR) + UR * 0.04,
                                stroke_width=4, color=BIAS) for g in (g_num, g_den)])
        self.at(T("divide", 0.95), Create(strikes), run_time=0.5)

        # inset: c across the eight fields (the odds scene's flat line)
        fa = Axes(x_range=[-7e-4, 7e-4, 3.5e-4], y_range=[0, 0.2, 0.1], x_length=3.4, y_length=1.7,
                  axis_config={"color": RULE, "stroke_width": 2, "include_ticks": False}, tips=False)
        fa.move_to(RIGHT * 3.9 + UP * 0.55)
        f_dots = VGroup(*[Dot(fa.c2p(g, c), radius=0.055, color=GILT) for g, c in FIELD_C])
        f_line = Line(fa.c2p(-7e-4, 0.1198), fa.c2p(7e-4, 0.1198), stroke_width=2, color=GILT).set_opacity(0.6)
        f_xl = mtex("G", size=SIZE_SMALL + 2, color=BIAS).next_to(fa.c2p(7e-4, 0), RIGHT, buff=0.1)
        f_yl = mtex("c", size=SIZE_SMALL + 2, color=GILT_TEXT).next_to(fa.c2p(-7e-4, 0.2), LEFT, buff=0.1)
        f_yt = mtex("0.12", size=SIZE_SMALL, color=GILT_TEXT).next_to(fa.c2p(-7e-4, 0.1198), LEFT, buff=0.1)
        f_yl.next_to(f_yt, UP, buff=0.25)
        f_tag = label("c over the eight fields", size=SIZE_SMALL, color=INK_3).next_to(fa, DOWN, buff=0.15)
        inset = VGroup(fa.x_axis, fa.y_axis, f_line, f_dots, f_xl, f_yl, f_yt, f_tag)
        self.at(T("divide", 1.65), FadeIn(VGroup(fa.x_axis, fa.y_axis, f_xl, f_yl, f_yt, f_tag)),
                LaggedStart(*[FadeIn(d, scale=0.5) for d in f_dots], lag_ratio=0.08), Create(f_line), run_time=0.9)

        # "c goes like N squared gamma, to the minus one quarter."
        step = mtex("c", r"\propto", r"N^{-1/2}\,\gamma^{-1/4}", size=SIZE_HEAD, colors={"c": GILT_TEXT})
        step.move_to(whole)
        self.play_at("law", FadeOut(VGroup(g_num, g_den, strikes), scale=0.6),
                     ReplacementTransform(cprop, step[:2]),
                     ReplacementTransform(VGroup(num[1], bar, den[0]), step[2]), run_time=0.9)
        claw = mtex("c", r"\propto", r"\left(N^2\gamma\right)^{-1/4}", size=SIZE_TITLE, colors={"c": GILT_TEXT})
        claw.move_to(LEFT * 1.6 + UP * 0.75)
        self.at(T("law", 1.0), ReplacementTransform(step[:2], claw[:2]), ReplacementTransform(step[2], claw[2]),
                run_time=1.1)
        claw_ul = Underline(claw, color=GILT, stroke_width=3, buff=0.12)
        self.at(T("law", 2.6), Create(claw_ul), run_time=0.6)

        # ============================================================ ladder
        L = self.make_ladder(cx=-3.3)
        lx, rx, y = L["lx"], L["rx"], L["y"]
        header = VGroup(claw, claw_ul)
        # "So cooling sixteen times more slowly only doubles c."
        self.play_at("sixteen", FadeOut(inset), FadeOut(law, shift=UP * 0.3),
                     header.animate.scale(0.6).to_corner(UL, buff=0.4),
                     Create(L["rail_l"]), Create(L["rail_r"]), FadeIn(L["tl"], L["tr"], L["hl"], L["hr"]),
                     run_time=0.75)
        lv, rv = ValueTracker(1.0), ValueTracker(1.0)
        fill_l = always_redraw(lambda: Line([lx, y(1), 0], [lx, y(lv.get_value()) + 1e-3, 0], stroke_width=8, color=INK))
        fill_r = always_redraw(lambda: Line([rx, y(1), 0], [rx, y(rv.get_value()) + 1e-3, 0], stroke_width=8, color=GILT))
        mk_l = always_redraw(lambda: Dot([lx, y(lv.get_value()), 0], radius=0.11, color=INK))
        mk_r = always_redraw(lambda: Dot([rx, y(rv.get_value()), 0], radius=0.11, color=GILT))
        rung = always_redraw(lambda: Line([lx, y(lv.get_value()), 0], [rx, y(rv.get_value()), 0],
                                          stroke_width=2, color=INK_3).set_stroke(opacity=0.35))
        self.add(fill_l, fill_r, rung, mk_l, mk_r)
        self.at(T("sixteen", 0.76), lv.animate.set_value(16), run_time=1.4, rate_func=smooth)
        x2 = mtex(r"\times 2", size=SIZE_HEAD, color=GILT_TEXT).next_to([rx, y(2), 0], RIGHT, buff=0.75)
        root = mtex(r"16^{1/4} = 2", size=SIZE_BODY, color=INK_2).next_to(x2, DOWN, buff=0.3, aligned_edge=LEFT)
        self.play_at("doubles", rv.animate.set_value(2), FadeIn(x2, shift=UP * 0.15), run_time=0.9, rate_func=smooth)
        self.at(T("doubles", 0.97), FadeIn(root), run_time=0.5)

        # "This holds for a slow ramp starting well below beta equal to one, with little switching later."
        ra = Axes(x_range=[0, 10, 1], y_range=[0.6, 1.7, 0.1], x_length=3.6, y_length=2.1,
                  axis_config={"color": RULE, "stroke_width": 2, "include_ticks": False}, tips=False)
        ra.move_to(RIGHT * 3.7 + UP * 1.1)
        r_bl = mtex(r"\beta", size=SIZE_BODY, color=INK_2).next_to(ra.c2p(0, 1.7), LEFT, buff=0.12)
        r_tl = label("sweeps", size=SIZE_SMALL, color=INK_3).next_to(ra.c2p(10, 0.6), DOWN, buff=0.12)
        one = DashedLine(ra.c2p(0, 1.0), ra.c2p(10, 1.0), stroke_width=2, color=NOISE, dash_length=0.08)
        one_l = mtex(r"\beta = 1", size=SIZE_SMALL, color=INK_3).next_to(ra.c2p(10, 1.0), RIGHT, buff=0.1)
        ramp = theory(Line(ra.c2p(0, 0.7), ra.c2p(10, 1.6)).set_stroke(width=3), num_dashes=24)
        slope_l = mtex(r"\text{slope } \gamma \ll 1", size=SIZE_SMALL + 2, color=THEORY)
        slope_l.next_to(ra.c2p(6.2, 1.26), UL, buff=0.08)
        start = Dot(ra.c2p(0, 0.7), radius=0.07, color=INK)
        start_l = tex(r"start $\beta = 0.7$", size=SIZE_SMALL + 2, color=INK_2).next_to(start, RIGHT, buff=0.15).shift(DOWN * 0.08)
        pdot = Dot(radius=0.13, color=MODE_A).move_to(RIGHT * 2.75 + DOWN * 1.55)
        ndot = Dot(radius=0.13, color=MODE_B).move_to(RIGHT * 4.65 + DOWN * 1.55)
        sw = CurvedArrow(pdot.get_center() + RIGHT * 0.22 + UP * 0.08, ndot.get_center() + LEFT * 0.22 + UP * 0.08,
                         angle=-1.0, color=NOISE, stroke_width=3, tip_length=0.16)
        sw = DashedVMobject(sw, num_dashes=14)
        sw_l = label("little switching later", size=SIZE_SMALL + 2, color=INK_2).next_to(VGroup(pdot, ndot), DOWN, buff=0.28)
        regime = VGroup(ra.x_axis, ra.y_axis, r_bl, r_tl, one, one_l, ramp, slope_l, start, start_l, pdot, ndot, sw, sw_l)
        self.play_at("regime", Create(VGroup(ra.x_axis, ra.y_axis)), FadeIn(r_bl, r_tl), run_time=0.7)
        self.at(T("regime", 0.95), Create(ramp), FadeIn(slope_l), run_time=1.0)
        self.at(T("regime", 1.98), FadeIn(start, start_l), Create(one), FadeIn(one_l), run_time=0.8)
        self.at(T("regime", 4.7), FadeIn(pdot, ndot), Create(sw), FadeIn(sw_l), run_time=0.9)

        # ============================================================ check (paper Fig. 2a)
        def build_plot(xmax, ymax, x_len, center):
            """c against the ramp rate (log axes; slower to the right: x = log10(1/gamma))."""
            x0v, y0v = 2.0, math.log10(0.035)
            y1v = math.log10(ymax)
            o = np.array(center) - np.array([x_len / 2, 4.6 / 2, 0])

            def P(inv_g_log, c):
                return o + np.array([(inv_g_log - x0v) / (xmax - x0v) * x_len,
                                     (math.log10(c) - y0v) / (y1v - y0v) * 4.6, 0])

            parts = {}
            parts["P"] = P
            parts["xax"] = Line(P(x0v, 0.035), P(xmax, 0.035), stroke_width=3, color=RULE)
            parts["yax"] = Line(P(x0v, 0.035), P(x0v, ymax), stroke_width=3, color=RULE)
            parts["xt"] = VGroup(*[VGroup(Line(P(math.log10(1 / g), 0.035), P(math.log10(1 / g), 0.035) + DOWN * 0.1,
                                               stroke_width=2, color=INK_3),
                                          mtex(s, size=SIZE_SMALL, color=INK_3)
                                          .next_to(P(math.log10(1 / g), 0.035), DOWN, buff=0.17))
                                   for g, s in ((0.004, "0.004"), (0.001, "0.001"), (0.00025, "0.00025"))])
            parts["yt"] = VGroup(*[VGroup(Line(P(x0v, v), P(x0v, v) + LEFT * 0.1, stroke_width=2, color=INK_3),
                                          mtex(s, size=SIZE_SMALL, color=INK_3).next_to(P(x0v, v), LEFT, buff=0.15))
                                   for v, s in ((0.05, "0.05"), (0.1, "0.1"), (0.2, "0.2"))])
            parts["lines"] = VGroup(*[theory(VMobject().set_points_smoothly(
                [P(lx_, c_sel(N, 10 ** -lx_)) for lx_ in np.linspace(2.2, 3.8, 40)]).set_stroke(width=3),
                num_dashes=26) for N in NS])
            parts["nl"] = VGroup(*[mtex(rf"N = {N // 1000}{{,}}{N % 1000:03d}", size=SIZE_SMALL, color=INK_3)
                                   .next_to(P(3.8, c_sel(N, 10 ** -3.8)), RIGHT, buff=0.15) for N in NS])
            parts["syms"] = VGroup(*[Circle(radius=0.085, stroke_color=INK, stroke_width=2.6, fill_color=BG,
                                            fill_opacity=1).move_to(P(math.log10(1 / r["gamma"]), r["c_exact"]))
                                     for r in LIN])
            return parts

        A = build_plot(4.15, 0.3, 8.0, (0.5, -0.45, 0))
        P = A["P"]
        xl = tex(r"ramp rate $\gamma$ \;(slower $\rightarrow$)", size=SIZE_SMALL + 2, color=INK_2)
        xl.next_to(A["xax"], DOWN, buff=0.62)
        yl = mtex("c", size=SIZE_BODY, color=GILT_TEXT).next_to(A["yax"], UP, buff=0.12)
        key = VGroup(
            VGroup(Circle(radius=0.085, stroke_color=INK, stroke_width=2.6), label("exact CW chain", size=SIZE_SMALL, color=INK_2)).arrange(RIGHT, buff=0.15),
            VGroup(theory(Line(LEFT * 0.25, RIGHT * 0.25).set_stroke(width=3), num_dashes=3), label("formula", size=SIZE_SMALL, color=INK_2)).arrange(RIGHT, buff=0.15),
        ).arrange(RIGHT, buff=0.5, aligned_edge=DOWN)
        key.to_edge(UP, buff=0.42).to_edge(RIGHT, buff=0.5)
        set_tag = tex(r"linear ramps $\beta = 0.7 \to 1.6$ $\cdot$ $NG$ fixed", size=SIZE_SMALL, color=INK_3)
        set_tag.next_to(key, DOWN, buff=0.12).align_to(key, RIGHT)

        # "The exact chain agrees."
        ladder_rest = VGroup(L["tl"], L["tr"], L["hl"], L["hr"], fill_l, fill_r, rung, mk_l, mk_r, x2, root)
        self.play_at("check", FadeOut(ladder_rest), FadeOut(regime),
                     ReplacementTransform(L["rail_l"], A["yax"]), ReplacementTransform(L["rail_r"], A["xax"]),
                     run_time=0.85)
        self.remove(fill_l, fill_r, rung, mk_l, mk_r)
        self.at(T("check", 0.87), FadeIn(A["xt"], A["yt"], xl, yl), Create(A["lines"]), FadeIn(A["nl"]),
                FadeIn(key[1]), run_time=0.65)
        self.at(T("check", 1.55), LaggedStart(*[GrowFromCenter(s) for s in A["syms"]], lag_ratio=0.08),
                FadeIn(key[0]), FadeIn(set_tag), run_time=0.7)

        # "At the smallest size, sixteen times slower takes c from zero point zero nine four to zero point one eight nine."
        ia, ib = LIN.index(run(1024, 0.004)), LIN.index(run(1024, 0.00025))
        sa, sb = A["syms"][ia], A["syms"][ib]
        others = VGroup(*[s for i, s in enumerate(A["syms"]) if i not in (ia, ib)])
        pa, pb = sa.get_center(), sb.get_center()
        d = (pb - pa) / np.linalg.norm(pb - pa)
        nrm = np.array([-d[1], d[0], 0])
        slow = Arrow(pa + nrm * 0.4, pb + nrm * 0.4, buff=0.12, stroke_width=4, color=INK_2,
                     max_tip_length_to_length_ratio=0.06)
        slow_l = tex(r"$16\times$ slower", size=SIZE_SMALL + 2, color=INK_2)
        slow_l.move_to(slow.get_center() + nrm * 0.32).rotate(math.atan2(d[1], d[0]))
        va = mtex("0.094", size=SIZE_BODY - 4, color=GILT_TEXT).next_to(sa, DOWN, buff=0.18)
        vb = mtex("0.189", size=SIZE_BODY - 4, color=GILT_TEXT).next_to(sb, DOWN, buff=0.18).shift(LEFT * 0.1)
        x201 = mtex(r"\times 2.01", size=SIZE_BODY - 4, color=GILT_TEXT).next_to(vb, RIGHT, buff=0.3).shift(UP * 0.12)
        self.play_at("check-values", others.animate.set_stroke(opacity=0.35),
                     A["lines"][1:].animate.set_stroke(opacity=0.3), A["nl"][1:].animate.set_opacity(0),
                     sa.animate.set_stroke(GILT, 3.4), sb.animate.set_stroke(GILT, 3.4), run_time=0.8)
        # each value writes on as it is said; the x2.01 follows the second one
        t_16 = self.word_at("sixteen", T("check-values"))
        t_va = self.word_at("zero", T("check-values"))
        t_vb = self.word_at("zero", self.word_at("to", t_va))
        self.at(t_16 - 0.05, GrowArrow(slow), FadeIn(slow_l), run_time=0.9)
        self.at(t_va - 0.05, Write(va), run_time=0.7)
        self.at(t_vb - 0.05, Write(vb), run_time=0.7)
        self.at(t_vb + 0.8, FadeIn(x201, shift=UP * 0.1), run_time=0.5)

        # "Over all nine ramps, the exact chain lies slightly above the full formula, within two percent."
        # two lines under the legend, clear of the bottom caption band
        within = VGroup(tex(r"all nine within 2\%:", size=SIZE_SMALL + 2, color=INK_2),
                        tex(r"the exact chain lies above the law by 0.23--1.81\%", size=SIZE_SMALL + 2,
                            color=INK_2)).arrange(DOWN, buff=0.1, aligned_edge=RIGHT)
        within.next_to(set_tag, DOWN, buff=0.22).align_to(set_tag, RIGHT)
        self.play_at("max", others.animate.set_stroke(opacity=1.0), A["lines"][1:].animate.set_stroke(opacity=1.0),
                     A["nl"][1:].animate.set_opacity(1.0), FadeOut(slow, slow_l, va, vb, x201),
                     sa.animate.set_stroke(INK, 2.6), sb.animate.set_stroke(INK, 2.6), run_time=0.6)
        self.at(T("max", 0.65), LaggedStart(*[Indicate(s, color=INK, scale_factor=1.6) for s in A["syms"]],
                                            lag_ratio=0.1), run_time=1.4)
        self.at(T("max", 3.45), FadeIn(within, shift=UP * 0.1), run_time=0.8)

        # ============================================================ not-forever (illustration)
        # Two knobs, drawn apart: the ramp rate (the x axis, no numbers past the measured ramps) and
        # waiting at the final temperature (a vertical arrow at one fixed ramp, off the gamma axis).
        B = build_plot(6.9, 1.5, 8.0, (0.2, -0.45, 0))
        PB = B["P"]
        ytB = VGroup(*[VGroup(Line(PB(2.0, v), PB(2.0, v) + LEFT * 0.1, stroke_width=2, color=INK_3),
                              mtex(s, size=SIZE_SMALL, color=INK_3).next_to(PB(2.0, v), LEFT, buff=0.15))
                       for v, s in ((0.5, "0.5"), (1.0, "1"))])
        # the N = 1,024 formula line continues through c = 1; the other two fade as before
        ext = VGroup()
        for N in NS:
            x_end = 6.6 if N == 1024 else 5.0
            pts = [PB(x, c_sel(N, 10 ** -x)) for x in np.linspace(3.8, x_end, 40)]
            seg = theory(VMobject().set_points_smoothly(pts).set_stroke(width=3), num_dashes=34 if N == 1024 else 14)
            for k, dash in enumerate(seg):
                if N == 1024:
                    yv = dash.get_center()[1]
                    over = (yv - PB(2.0, 1.0)[1]) / (PB(2.0, 1.3)[1] - PB(2.0, 1.0)[1])
                    dash.set_stroke(opacity=float(np.clip(1 - over, 0.15, 1)))
                else:
                    dash.set_stroke(opacity=1 - k / len(seg))
            ext.add(seg)
        # axis break after the slowest measured ramp: no numeric location past 0.00025
        bx = PB(4.05, 0.035)
        brk = VGroup(Rectangle(width=0.16, height=0.3, stroke_width=0, fill_color=BG, fill_opacity=1).move_to(bx),
                     *[Line(bx + np.array([dx - 0.07, -0.13, 0]), bx + np.array([dx + 0.07, 0.13, 0]),
                            stroke_width=2.5, color=RULE) for dx in (-0.08, 0.08)])
        nf_l = tex(r"the law: not for arbitrarily slow cooling at fixed $N$", size=SIZE_SMALL + 2, color=THEORY)
        nf_l.move_to(PB(4.3, 0.05), aligned_edge=LEFT)
        nf_t = tex(r"\mbox{at $N = 2{,}048$: one mode switch takes about $5\times10^{7}$ expected sweeps at $\beta = 1.1$ (zero field)}",
                   size=SIZE_SMALL, color=INK_3)
        eq_line = theory(Line(PB(2.0, 1.0), PB(6.9, 1.0)).set_stroke(width=3), num_dashes=46)
        eq_l2 = tex(r"the model's proportions: $c = 1$", size=SIZE_SMALL + 2, color=INK)
        eq_l2.next_to(PB(2.1, 1.0), UP, buff=0.14, aligned_edge=LEFT)
        s_wait = B["syms"][LIN.index(run(1024, 0.00025))]
        eq_arrow = Arrow(s_wait.get_top() + UP * 0.05, PB(3.6, 1.0) + DOWN * 0.04, buff=0, stroke_width=4,
                         color=INK_2, max_tip_length_to_length_ratio=0.1)
        eq_l1 = VGroup(tex(r"then wait at $\beta = 1.6$,", size=SIZE_SMALL + 2, color=INK_2),
                       tex(r"very long", size=SIZE_SMALL + 2, color=INK_2)).arrange(DOWN, buff=0.1, aligned_edge=RIGHT)
        eq_l1.next_to(eq_arrow, LEFT, buff=0.18).shift(UP * 0.35)
        illu = label("fixed N · illustration, not a measurement", size=SIZE_SMALL, color=INK_3)
        illu.to_edge(UP, buff=0.45).to_edge(RIGHT, buff=0.5)
        nf_t.next_to(illu, DOWN, buff=0.14).align_to(illu, RIGHT)
        xlB = tex(r"ramp rate $\gamma$ \;(slower $\rightarrow$)", size=SIZE_SMALL + 2, color=INK_2)
        xlB.next_to(B["xax"], DOWN, buff=0.62)

        # "The law does not hold for arbitrarily slow cooling at fixed size:"
        self.play_at("not-forever", FadeOut(within, key, set_tag),
                     *[ReplacementTransform(A[k], B[k]) for k in ("xax", "yax", "xt", "yt", "lines", "nl", "syms")],
                     ReplacementTransform(xl, xlB), yl.animate.next_to(B["yax"], UP, buff=0.12), run_time=1.5)
        self.at(T("not-forever", 1.55), FadeIn(ytB, brk), FadeOut(B["nl"]), Create(eq_line), FadeIn(eq_l2),
                Create(ext), run_time=1.6)
        self.at(T("not-forever", 3.25), FadeIn(nf_l), FadeIn(illu), FadeIn(nf_t), run_time=0.8)
        # "long enough sampling at the final temperature recovers the model's proportions."
        self.play_at("equilibrium", Indicate(s_wait, color=INK, scale_factor=1.5), run_time=0.7)
        self.at(T("equilibrium", 0.75), GrowArrow(eq_arrow), FadeIn(eq_l1), run_time=1.2)
        self.at(T("equilibrium", 2.85), Indicate(eq_l2, color=INK, scale_factor=1.08), run_time=0.9)

        # ============================================================ staircase (the paper's schedule)
        ladder_b, K = SCHED["ladder"], SCHED["K"]
        KT = sum(K)
        u = 7.0 / (4 * KT)                  # x4 final cooling = 7 units
        sx0, sy0, sy1 = -5.9, -1.75, 0.95

        def sby(b):
            return sy0 + (b - 0.6) * (sy1 - sy0)

        sc = ValueTracker(0.25)             # opens on the x1/4 state (59 of 100, c = 0.099)

        def stair():
            pts, x = [], sx0
            for b, k in zip(ladder_b, K):
                w = k * u * sc.get_value()
                pts += [[x, sby(b), 0], [x + w, sby(b), 0]]
                x += w
            return VMobject(stroke_color=INK_2, stroke_width=4).set_points_as_corners(pts)

        stairs = always_redraw(stair)
        s_ax = always_redraw(lambda: Line([sx0, sy0 - 0.25, 0], [sx0 + KT * u * sc.get_value() + 0.15, sy0 - 0.25, 0],
                                          stroke_width=2, color=RULE))
        b_ticks = VGroup(*[mtex(s, size=SIZE_SMALL, color=INK_3).next_to([sx0, sby(v), 0], LEFT, buff=0.12)
                           for v, s in ((0.6, "0.6"), (1.0, "1"), (1.6, "1.6"))])
        b_name = mtex(r"\beta", size=SIZE_BODY, color=INK_2).next_to(b_ticks, LEFT, buff=0.18)
        st_title = label("the paper's schedule: ten temperatures", size=SIZE_SMALL + 2, color=INK_2)
        st_title.next_to([sx0, sy1, 0], UP, buff=0.35, aligned_edge=LEFT)
        sw_lab = {s: tex(t, size=SIZE_SMALL + 2, color=INK_2) for s, t in
                  ((0.25, r"834 sweeps $\times\tfrac14$"), (4.0, r"834 sweeps $\times 4$"))}
        for m in sw_lab.values():
            m.next_to([sx0, sy0 - 0.25, 0], DOWN, buff=0.15, aligned_edge=LEFT)
        st_tag = tex(r"final cooling stretched $\cdot$ brief cooling and reheating unchanged $\cdot$ $N = 2{,}048$",
                     size=SIZE_SMALL, color=INK_3).to_edge(DOWN, buff=0.3).align_to([sx0 - 0.5, 0, 0], LEFT)

        # the hook's grid: 59 -> 63 positive of 100 (x1/4 -> x4; the paper's own x1 gives 61, not shown)
        CELL, GAP = 0.2, 0.04
        gc = np.array([4.35, 1.35, 0])
        grid = VGroup(*[Square(CELL, stroke_width=0).move_to(gc + np.array([(i % 10 - 4.5) * (CELL + GAP),
                                                                          -(i // 10 - 4.5) * (CELL + GAP), 0]))
                        for i in range(100)])

        def paint(k):
            return [sq.animate.set_fill(MODE_A if i < k else MODE_B, opacity=0.9) for i, sq in enumerate(grid)]

        n_pos = {s: round(100 * STAIR[s]["q_model"]) for s in (0.25, 1.0, 4.0)}   # 59, 61, 63
        assert (n_pos[0.25], n_pos[1.0], n_pos[4.0]) == (59, 61, 63)
        for i, sq in enumerate(grid):
            sq.set_fill(MODE_A if i < n_pos[0.25] else MODE_B, opacity=0.9)
        cnt = {s: mtex(str(n_pos[s]), size=SIZE_HEAD, color=MODE_A).next_to(grid, LEFT, buff=0.3) for s in (0.25, 4.0)}
        g_tag = label("positive, of 100", size=SIZE_SMALL, color=INK_3).next_to(grid, UP, buff=0.12)

        # c meter: a gold bar on a 0-0.25 scale; the ramp law's x2 as a dashed tick
        mx0, my, mlen = 2.55, -1.45, 3.6

        def mxp(c):
            return np.array([mx0 + c / 0.25 * mlen, my, 0])

        cv = ValueTracker(STAIR[0.25]["c"])
        m_base = Line(mxp(0), mxp(0.25), stroke_width=2, color=RULE)
        m_ticks = VGroup(*[mtex(s, size=SIZE_SMALL, color=INK_3).next_to(mxp(v), DOWN, buff=0.22)
                           for v, s in ((0, "0"), (0.1, "0.1"))])
        m_bar = always_redraw(lambda: Rectangle(width=max(cv.get_value() / 0.25 * mlen, 1e-3), height=0.22,
                                                stroke_width=0, fill_color=GILT, fill_opacity=0.95)
                              .move_to(mxp(0), aligned_edge=LEFT))
        m_read = always_redraw(lambda: mtex("c", "=", f"{cv.get_value():.3f}", size=SIZE_BODY,
                                            color=GILT_TEXT).next_to(mxp(0), UP, buff=0.3, aligned_edge=LEFT))
        x148 = mtex(r"\times 1.48", size=SIZE_HEAD, color=GILT_TEXT).move_to(mxp(0.03) + DOWN * 1.05, aligned_edge=LEFT)
        c2 = 2 * STAIR[0.25]["c"]
        law_tick = theory(Line(mxp(c2) + DOWN * 0.24, mxp(c2) + UP * 0.24).set_stroke(width=4), num_dashes=4)
        law_tick_l = tex(r"$\times 2$ (ramp)", size=SIZE_SMALL + 2, color=THEORY).next_to(law_tick, UP, buff=0.1)
        # "On the paper's schedule, the opening's fifty-nine to sixty-three"
        plotB = VGroup(B["xax"], B["yax"], B["xt"], B["yt"], B["lines"], B["syms"], ytB, brk, ext, nf_l, eq_line,
                       eq_arrow, eq_l1, eq_l2, illu, nf_t, xlB, yl)
        stairs_still = stair()
        s_ax_still = s_ax.copy()
        stretch_tag = tex(r"$\times\tfrac14 \to \times 4$ $=$ $16\times$", size=SIZE_SMALL + 2, color=INK_3)
        stretch_tag.next_to(sw_lab[4.0], RIGHT, buff=0.45)
        # the ramp plot leaves first (0.45 s); everything new starts at 0.4 s, so no two layers of text overlap
        # (one AnimationGroup without an outer run_time, so each part keeps its own timing; 1.3 s in all)
        self.play_at("staircase", AnimationGroup(
            FadeOut(plotB, run_time=0.45),
            Succession(Wait(run_time=0.4), Create(stairs_still, run_time=0.9)),
            Succession(Wait(run_time=0.4), AnimationGroup(
                FadeIn(s_ax_still, b_ticks, b_name, st_title, sw_lab[0.25]),
                FadeIn(grid, lag_ratio=0.004), FadeIn(cnt[0.25], g_tag), run_time=0.9))))
        self.remove(stairs_still, s_ax_still)
        self.add(stairs, s_ax)
        # the c meter shows the same x1/4 state as the grid and the staircase
        self.at(T("staircase", 1.45), FadeIn(m_base, m_ticks), FadeIn(m_bar, m_read), FadeIn(st_tag), run_time=0.6)
        self.add(m_bar, m_read)
        t_59 = self.word_at("fifty-nine", T("staircase"))
        t_63 = self.word_at("sixty-three", t_59)
        self.at(t_59 - 0.1, Indicate(cnt[0.25], color=MODE_A, scale_factor=1.15), run_time=0.5)
        # one stretch drives all three: staircase x1/4 -> x4, grid 59 -> 63, c 0.099 -> 0.147
        self.at(t_63 - 0.35, sc.animate.set_value(4.0), cv.animate.set_value(STAIR[4.0]["c"]),
                FadeTransform(sw_lab[0.25], sw_lab[4.0]), *paint(n_pos[4.0]), FadeTransform(cnt[0.25], cnt[4.0]),
                FadeIn(stretch_tag), run_time=1.0, rate_func=smooth)

        # "is c growing only one point four eight times."
        self.play_at("stair-c", Indicate(m_read, color=GILT_TEXT, scale_factor=1.15), run_time=0.8)
        self.play_at("stair-result", Write(x148), run_time=0.8)
        self.at(T("stair-result", 1.25), Create(law_tick), FadeIn(law_tick_l), run_time=0.7)

        # "The law was derived for a ramp; the paper does not derive this smaller gain."
        ramp2 = theory(Line([sx0, sby(0.6), 0], [sx0 + KT * u * 4, sby(1.6), 0]).set_stroke(width=3), num_dashes=40)
        ramp_l = label("ramp", size=SIZE_SMALL + 2, color=THEORY).next_to([sx0 + KT * u * 4, sby(1.6), 0], UP, buff=0.15).shift(LEFT * 0.4)
        not_der = label("computed, not derived", size=SIZE_SMALL, color=INK_3).next_to(x148, DOWN, buff=0.15, aligned_edge=LEFT)
        self.play_at("caveat", Indicate(law_tick_l, color=THEORY, scale_factor=1.3), run_time=0.8)
        self.at(T("caveat", 1.75), Create(ramp2), FadeIn(ramp_l), run_time=1.1)
        self.at(self.word_at("derive", T("caveat")) - 0.05, FadeIn(not_der, shift=UP * 0.1), Indicate(x148, color=GILT_TEXT, scale_factor=1.1),
                run_time=0.9)

        # "Either way, slower cooling buys little."  (the ladder again: 16x slower -> x2 ramp, x1.48 staircase)
        L2 = self.make_ladder(cx=-1.0, H=4.2, yb=-2.35, gap=3.4)
        y2 = L2["y"]
        lx2, rx2 = L2["lx"], L2["rx"]
        st_all = VGroup(stairs, s_ax, b_ticks, b_name, st_title, sw_lab[4.0], stretch_tag, st_tag, grid, cnt[4.0], g_tag,
                        m_base, m_ticks, m_bar, m_read, law_tick, law_tick_l, ramp2, ramp_l, not_der)
        x148_end = mtex(r"\times 1.48", size=SIZE_HEAD, color=GILT_TEXT).next_to([rx2, y2(1.4814), 0], RIGHT, buff=0.7)
        lv2 = ValueTracker(1.0)
        rv2 = ValueTracker(1.0)
        fl2 = always_redraw(lambda: Line([lx2, y2(1), 0], [lx2, y2(lv2.get_value()) + 1e-3, 0], stroke_width=8, color=INK))
        fr2 = always_redraw(lambda: Line([rx2, y2(1), 0], [rx2, y2(rv2.get_value()) + 1e-3, 0], stroke_width=8, color=GILT))
        ml2 = always_redraw(lambda: Dot([lx2, y2(lv2.get_value()), 0], radius=0.11, color=INK))
        mr2 = always_redraw(lambda: Dot([rx2, y2(rv2.get_value()), 0], radius=0.11, color=GILT))
        rg2 = always_redraw(lambda: Line([lx2, y2(lv2.get_value()), 0], [rx2, y2(rv2.get_value()), 0],
                                         stroke_width=2, color=INK_3).set_stroke(opacity=0.35))
        ramp_mark = theory(Line([rx2 - 0.3, y2(2), 0], [rx2 + 0.3, y2(2), 0]).set_stroke(width=4), num_dashes=5)
        ramp_mark_l = tex(r"$\times 2$ on a ramp", size=SIZE_SMALL + 2, color=THEORY).next_to(ramp_mark, RIGHT, buff=0.6)
        stair_l = label("the paper's schedule", size=SIZE_SMALL + 2, color=INK_2).next_to(x148_end, DOWN, buff=0.15, aligned_edge=LEFT)
        self.play_at("answer", FadeOut(st_all), ReplacementTransform(x148, x148_end),
                     Create(L2["rail_l"]), Create(L2["rail_r"]), FadeIn(L2["tl"], VGroup(*[t[0] for t in L2["tr"]]), L2["hl"], L2["hr"]),
                     run_time=0.9)
        self.add(fl2, fr2, rg2, ml2, mr2)
        self.at(T("answer", 1.0), lv2.animate.set_value(16), run_time=1.2, rate_func=smooth)
        self.at(T("answer", 2.3), Create(ramp_mark), FadeIn(ramp_mark_l), rv2.animate.set_value(1.4814),
                FadeIn(stair_l), run_time=1.2, rate_func=smooth)
