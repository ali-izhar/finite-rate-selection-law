"""Pipeline test scene: the landscape morphing from one valley to two (script.md, commit/landscape).

Render:  PYTHONNOUSERSITE=1 manim -ql --media_dir media tests/synctest.py SyncTest
Check:   python sync_check.py synctest
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import *  # noqa: E402,F403


def landscape(m, beta):
    """-log p(m) per variable for the Curie-Weiss model (large N), shifted so f(0) = 0."""
    m = np.clip(m, -0.999, 0.999)
    ent = (1 + m) / 2 * np.log((1 + m) / 2) + (1 - m) / 2 * np.log((1 - m) / 2)
    return -beta * m**2 / 2 + ent + np.log(2)


class SyncTest(NarratedScene):
    scene_id = "synctest"

    def construct(self):
        ax = Axes(x_range=[-1, 1, 0.5], y_range=[-0.2, 0.5, 0.1], x_length=9, y_length=4.2,
                  axis_config={"color": RULE, "stroke_width": 2, "include_ticks": False},
                  tips=False).shift(DOWN * 0.4)
        xl = mtex("m", size=SIZE_BODY).next_to(ax.x_axis, RIGHT, buff=0.2)
        ticks = VGroup(*[mtex(s, size=SIZE_SMALL, color=INK_3).next_to(ax.c2p(x, -0.2), DOWN, buff=0.15)
                         for s, x in (("-1", -1), ("0", 0), ("+1", 1))])
        yl = label("height = −log probability", size=SIZE_SMALL).next_to(ax, UP, buff=0.2).align_to(ax, LEFT)

        beta = ValueTracker(0.6)
        xs = np.linspace(-0.97, 0.97, 241)

        def curve_part(lo, hi, color):
            sel = xs[(xs >= lo) & (xs <= hi)]
            pts = [ax.c2p(x, landscape(x, beta.get_value())) for x in sel]
            return VMobject(stroke_width=5, color=color).set_points_smoothly(pts)

        def tint():
            # split colour fades in as beta passes 1: below 1 one grey-ink valley
            a = float(np.clip((beta.get_value() - 0.95) / 0.25, 0, 1))
            return interpolate_color(ManimColor(INK_2), ManimColor(MODE_B), a), \
                interpolate_color(ManimColor(INK_2), ManimColor(MODE_A), a)

        curve = always_redraw(lambda: VGroup(curve_part(-0.97, 0.0, tint()[0]),
                                             curve_part(0.0, 0.97, tint()[1])))
        ball = always_redraw(lambda: Dot(ax.c2p(0, landscape(0.0, beta.get_value())), radius=0.11,
                                         color=NOISE).shift(UP * 0.11))
        bl = always_redraw(lambda: mtex(r"\beta = %.2f" % beta.get_value(), size=SIZE_LABEL,
                                        color=INK_2).to_corner(UR, buff=0.6))

        # "Plot how unlikely each value of m is."
        self.play(Create(ax), FadeIn(xl), FadeIn(ticks), run_time=1.4)
        self.play(Write(yl), run_time=1.0)
        # "At low beta, it has a single valley, at zero."
        self.play_at("one-well", Create(curve), FadeIn(ball), FadeIn(bl), run_time=1.6)
        # "Turn beta past one, and the valley splits in two."
        self.play_at("two-wells", beta.animate.set_value(1.6), run_time=2.6, rate_func=smooth)
