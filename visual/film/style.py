"""Visual identity and the narrated-scene base class for every film scene.

Look: 3Blue1Brown (dark ground, Computer-Modern-family LaTeX that writes itself on, calm motion).
Meaning: the visual edition's colours (the dark values in ../lib/theme.css), never re-assigned:

    mode A (m > 0, favoured, the "3")   MODE_A  cobalt          #7EA3F0
    mode B (m < 0, the "8")             MODE_B  signal orange   #EE8A50
    bias push, G (model's preference)   BIAS    deep cobalt lt  #B3C7F6
    sampling noise, undecided window    NOISE   grey            #6F7177 (strokes, bands)
    kept fraction c                     GILT    #D9B350 strokes; GILT_TEXT #E4C372 as text
    law, prediction (always dashed)     THEORY  light grey      #C9C8C3
    measurement (solid)                 INK     #E8E7E2

Usage in a scene file (scenes/<id>.py):

    from style import *

    class Hook(NarratedScene):  # text(), label(), mtex(), tex(), theory() build the parts
        scene_id = "hook"
        def construct(self):
            self.play_at("modes-appear", FadeIn(tile))       # starts on the cue word
            self.play_until("model-grid", grid.animate.shift(LEFT))  # ends on the cue word
            self.wait_until("gap")
            self.mark("gap")                                 # an event with no play()

The scene adds build/audio/<id>.mp3 at t = 0, ends at audio length + TAIL, and writes every
cue-bound event (target and actual time) to build/sync/<id>.events.json for sync_check.py.
"""
from __future__ import annotations

import json
import os
import subprocess

from pathlib import Path

from manim import *  # noqa: F401,F403  (re-exported for scene files)
from manim import config, Scene, TexTemplate, MathTex, Tex, Text, VMobject, DashedVMobject

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
AUDIO_DIR = BUILD / "audio"
SYNC_DIR = BUILD / "sync"

# ------------------------------------------------------------------ palette
BG = "#111214"
INK = "#E8E7E2"
INK_2 = "#BDBCB6"
INK_3 = "#8C8B86"
RULE = "#4C4D50"
GRID = "#26272A"
MODE_A = "#7EA3F0"
MODE_B = "#EE8A50"
BIAS = "#B3C7F6"
NOISE = "#6F7177"
THEORY = "#C9C8C3"
DATA = INK
GILT = "#D9B350"
GILT_TEXT = "#E4C372"

config.background_color = BG
# Manim's partial-movie cache skips frames without advancing the renderer clock, which would
# break cue timing; every render draws every frame.
config.disable_caching = True

# ------------------------------------------------------------------ type
TEXT_FONT = "Latin Modern Roman"  # the Computer Modern family, same as the LaTeX below
# manim's conda Pango does not see the system/TeX fonts: register the Latin Modern files.
# Search order: $FILM_LM_FONTS, the folder kpsewhich reports, the Debian/Ubuntu TeX path.
def _lm_dirs() -> list[Path]:
    dirs = [Path(os.environ["FILM_LM_FONTS"])] if os.environ.get("FILM_LM_FONTS") else []
    try:
        found = subprocess.run(["kpsewhich", "lmroman10-regular.otf"], capture_output=True, text=True).stdout.strip()
        if found:
            dirs.append(Path(found).parent)
    except OSError:
        pass
    return dirs + [Path("/usr/share/texmf/fonts/opentype/public/lm")]


_LM_DIRS = _lm_dirs()


def _register_fonts() -> None:
    import manimpango

    for d in _LM_DIRS:
        files = [d / f"lmroman10-{w}.otf" for w in ("regular", "italic", "bold", "bolditalic")]
        if all(f.exists() for f in files):
            for f in files:
                manimpango.register_font(str(f))
            return
    raise FileNotFoundError("Latin Modern Roman .otf files not found; Text would fall back to sans")


_register_fonts()
TEX = TexTemplate()
TEX.add_to_preamble(r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath,amssymb}")

SIZE_TITLE = 56
SIZE_HEAD = 44
SIZE_BODY = 36
SIZE_LABEL = 28
SIZE_SMALL = 22

# symbols that always carry their colour, wherever they appear in LaTeX
SYMBOL_COLORS = {"c": GILT_TEXT, "G": BIAS}


TEXT_OVERSAMPLE = 8  # Pango renders at this multiple of the size, then the text is scaled down


class _OversampledText(Text):
    """Text whose Pango layout is never wrapped.

    manim gives Pango a layout as wide as the frame in pixels (854 at -ql, 1920 at -qh), so a
    line set at TEXT_OVERSAMPLE x the size would wrap; here the layout is 100,000 units wide.
    """

    def _text2svg(self, color):
        import manimpango
        from manim.mobject.text.text_mobject import START_X, START_Y, TEXT2SVG_ADJUSTMENT_FACTOR

        dir_name = config.get_dir("text_dir")
        dir_name.mkdir(parents=True, exist_ok=True)
        file_name = dir_name / ("wide_" + self._text2hash(color) + ".svg")
        if file_name.exists():
            return str(file_name.resolve())
        return manimpango.text2svg(
            self._text2settings(color), self._font_size / TEXT2SVG_ADJUSTMENT_FACTOR,
            self.line_spacing / TEXT2SVG_ADJUSTMENT_FACTOR, self.disable_ligatures,
            str(file_name.resolve()), START_X, START_Y, 100_000, 4_000, self.text)


def text(s: str, size: float = SIZE_BODY, color: str = INK, **kw) -> Text:
    """Prose in Latin Modern Roman (Pango).

    Pango hints glyph advances to whole units of the size it renders at. At manim's sizes
    (22-56) that rounding breaks word spacing ("prop osed", "neighb ouring"). The text is
    therefore set at TEXT_OVERSAMPLE x the size, where the rounding is invisible, and scaled to
    the width of the nominal-size setting, so every layout keeps its measured extent.
    """
    ref = Text(s, font=TEXT_FONT, font_size=size, color=color, **kw)
    t = _OversampledText(s, font=TEXT_FONT, font_size=size * TEXT_OVERSAMPLE, color=color, **kw)
    if t.width > 0 and ref.width > 0:
        t.scale(ref.width / t.width)
    else:
        t.scale(1 / TEXT_OVERSAMPLE)
    t.move_to(ref)
    return t


def label(s: str, color: str = INK_3, size: float = SIZE_LABEL, **kw) -> Text:
    return text(s, size=size, color=color, **kw)


def mtex(*s: str, size: float = SIZE_BODY, color: str = INK, colors: dict | None = None, **kw) -> MathTex:
    """MathTex with the project template.

    ``colors`` maps a whole part (one of the ``*s`` strings, exact match) to a colour, so
    ``mtex("c", r"\\propto", r"(N^2\\gamma)^{-1/4}", colors={"c": GILT_TEXT})`` colours only c.
    """
    m = MathTex(*s, tex_template=TEX, font_size=size, color=color, **kw)
    if colors:
        for k, v in colors.items():
            m.set_color_by_tex(k, v, substring=False)
    return m


def tex(*s: str, size: float = SIZE_BODY, color: str = INK, **kw) -> Tex:
    return Tex(*s, tex_template=TEX, font_size=size, color=color, **kw)


def theory(mob: VMobject, num_dashes: int | None = None, dashed_ratio: float = 0.55) -> VMobject:
    """Theory/prediction stroke: light grey and always dashed."""
    mob.set_stroke(THEORY)
    if num_dashes is None:
        num_dashes = max(8, int(mob.get_arc_length() / 0.12)) if hasattr(mob, "get_arc_length") else 30
    return DashedVMobject(mob, num_dashes=num_dashes, dashed_ratio=dashed_ratio)


# ------------------------------------------------------------------ narrated scene
def load_timing(scene_id: str) -> dict:
    p = AUDIO_DIR / f"{scene_id}.timing.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `python tts.py {scene_id}` first")
    return json.loads(p.read_text())


class NarratedScene(Scene):
    """A scene driven by its narration's word timings.

    Clock = ``self.renderer.time`` (seconds of video written so far). Waits are quantised to
    whole frames and self-correcting, so a cue lands within one frame of its word.
    """

    scene_id: str = ""
    TAIL = 0.85  # seconds of picture after the narration ends (sync_check allows 0.5-1.0)

    # -- lifecycle
    def setup(self):
        if not self.scene_id:
            raise ValueError(f"{type(self).__name__}.scene_id is not set")
        self.timing = load_timing(self.scene_id)
        self.cues: dict[str, float] = self.timing["cues"]
        self.audio_duration: float = self.timing["duration"]
        self.events: list[dict] = []
        self.add_sound(str(HERE / self.timing["audio"]), time_offset=0)

    def tear_down(self):
        end = self.audio_duration + self.TAIL
        self._wait_to(end)
        SYNC_DIR.mkdir(parents=True, exist_ok=True)
        fw = self.renderer.file_writer
        out = {
            "id": self.scene_id,
            "class": type(self).__name__,
            "frame_rate": config.frame_rate,
            "pixel_height": config.pixel_height,
            "movie_file": str(getattr(fw, "movie_file_path", "")),
            "audio": self.timing["audio"],
            "audio_duration": self.audio_duration,
            "tail": self.TAIL,
            "end_time": round(self.now, 4),
            "events": self.events,
        }
        (SYNC_DIR / f"{self.scene_id}.events.json").write_text(json.dumps(out, indent=1))

    # -- clock
    @property
    def now(self) -> float:
        return float(self.renderer.time)

    @property
    def dt(self) -> float:
        return 1.0 / config.frame_rate

    def t(self, cue: str | float, offset: float = 0.0) -> float:
        """Time of a cue (or a number passed through), plus an offset."""
        if isinstance(cue, (int, float)):
            return float(cue) + offset
        try:
            return self.cues[cue] + offset
        except KeyError:
            raise KeyError(f"scene {self.scene_id}: no cue {cue!r}; cues are {list(self.cues)}") from None

    def _wait_to(self, target: float) -> None:
        frames = round((target - self.now) * config.frame_rate)
        if frames >= 1:
            # frozen_frame=False: the animated path counts frames as ceil(run_time * fps)
            Scene.wait(self, frames / config.frame_rate - 1e-6, frozen_frame=False)

    # -- cue helpers (the names the brief asks for)
    def wait_until(self, cue, max_time: float = 60):  # noqa: D401
        """Wait until the cue (name or seconds). A callable keeps manim's own meaning."""
        if callable(cue):
            return super().wait_until(cue, max_time)
        self._wait_to(self.t(cue))

    def play_at(self, cue: str, *anims, **kw) -> None:
        """Wait for the cue, then start the animations on the cue word."""
        target = self.t(cue)
        self._wait_to(target)
        start = self.now
        self.play(*anims, **kw)
        self._log(cue, "start", target, start, self.now)

    def play_until(self, cue: str, *anims, min_run_time: float = 0.25, **kw) -> None:
        """Play now with a run_time chosen so the animations end on the cue word."""
        target = self.t(cue)
        rt = round((target - self.now) * config.frame_rate) / config.frame_rate
        if rt < min_run_time:
            raise RuntimeError(
                f"scene {self.scene_id}: play_until({cue!r}) has only {rt:.2f} s left "
                f"(now {self.now:.2f}, cue {target:.2f})")
        start = self.now
        self.play(*anims, run_time=rt - 1e-6, **kw)
        self._log(cue, "end", target, start, self.now)

    def mark(self, cue: str) -> None:
        """Log an event bound to a cue without playing (e.g. an updater switched on at the cue)."""
        target = self.t(cue)
        self._wait_to(target)
        self._log(cue, "mark", target, self.now, self.now)

    def _log(self, cue, kind, target, start, end):
        actual = start if kind in ("start", "mark") else end
        self.events.append({"cue": cue, "kind": kind, "target": round(target, 4),
                            "actual": round(actual, 4), "start": round(start, 4),
                            "end": round(end, 4), "delta": round(actual - target, 4)})
