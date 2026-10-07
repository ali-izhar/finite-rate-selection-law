# Narrated film

Nine manim scenes (about 14 minutes) that follow the paper in order. The committed web files are in
`../media/`; rebuilding them is needed only after a change to the script or a scene.

| # | Scene | Title | Page chapter |
|---|---|---|---|
| 1 | `hook` | The wrong proportions | overview |
| 2 | `commit` | Where the choice is made | commit |
| 3 | `odds` | How much preference survives | odds |
| 4 | `race` | Field against fluctuations | race |
| 5 | `scaling` | Why slower barely helps | scaling |
| 6 | `modes` | Competing patterns | modes |
| 7 | `reheat` | Where the runs start | reheat |
| 8 | `test` | The test | test |
| 9 | `limits` | What the law does and does not say | limits |

`script.md` holds the narration, the on-screen direction and the source of every number; its
"Conventions" section defines the cue syntax. `scenes.json` sets the scene order and posters.

## Environment

```bash
micromamba create -f visual/film/environment.yml      # or: conda env create -f ...
export MANIM_ENV=<prefix of that env>                 # default: manim on PATH
export TEXLIVE_BIN=<TeX Live bin folder>              # latex and dvisvgm; default: PATH
```

Narration calls the ElevenLabs API. `tts.py` reads `ELEVEN_LABS_API_KEY` from the repository's
untracked `.env` (override the path with `FILM_ENV_FILE`) and caches audio in `build/audio/`.
The voice settings are in `voice.json`. Fonts: Latin Modern from `kpsewhich`, or set `FILM_LM_FONTS`.

## Build

`visual/film/build.sh <command>` runs from any folder; outputs go to `build/` and `media/` (both untracked).

| Command | What it does |
|---|---|
| `tts [ids]` | narration and word timings for every scene, or the named ones |
| `render [-l\|-h] [ids]` | render scenes (`-l` 480p preview, `-h` 1080p release), then check cue sync |
| `check [ids]` | `sync_check.py`: every animation cue lands on its spoken word |
| `assemble` | join the scenes, normalize loudness, write captions and the 720p web files |
| `all [-l\|-h]` | `tts`, `render`, `assemble` |
| `test` | the pipeline test in `tests/` |

Then run `node visual/tools/media.mjs` to copy the web files into `visual/media/`.

Back to the [root README](../../README.md#visual-edition-and-film).
