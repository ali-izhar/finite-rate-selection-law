# Poster

The poster for the DynaFront Workshop at NeurIPS 2026: [`poster.pdf`](poster.pdf), 24 × 36 in,
portrait. [`poster.png`](poster.png) is a 1600 × 2400 px preview.

<img src="poster.png" width="420" alt="The poster: a dark hero band with cooling paths and proportion grids, then three Curie–Weiss panels and three Hopfield panels.">

## Layout

1. **Header.** Title, authors, venue, logos, and one QR code to this repository.
2. **Hero.** Annealed Gibbs samples keep only a fraction *c* of the model's log-odds between
   modes: runs commit near *β* = 1 (Fig. 1a paths), and in the Hopfield model the samples favor a
   pattern 67 times in 100 where the model gives 97 (*c* ≈ 1/5).
3. **I. Curie–Weiss.** (1) A weak field races noise: *a* = 0.208, 58%. (2) Sixteen times slower
   cooling only doubles *c*. (3) Reheated runs select again and keep *c*<sub>CW</sub> = 0.120.
4. **II. Hopfield.** (4) Competition among patterns raises *c* by *A*<sub>3</sub> = 1.662.
   (5) 0.120 × 1.662 = 0.199 predicted, 0.199 [0.1908, 0.2095] measured. (6) Table 1: the
   prediction misses by up to 11%.

Every printed number comes from [`paper/main.tex`](../paper/main.tex); `make_figures.py` asserts
each one against the paper source and stops the build if one differs.

## Rebuild

```sh
./poster/build.sh
```

1. `make_figures.py` (numpy, scipy) writes the vector figures `figures/V*.svg` from
   `visual/data/*.json` and analytic curves. It uses `.venv/bin/python` if present; set
   `PYTHON=...` to choose another interpreter.
2. `build.mjs` writes the QR code, inlines the figures, renders the math with KaTeX, checks every
   block for overflow, and prints `poster.pdf` and `poster.png` with Playwright and Chromium
   (from `visual/node_modules`; set `CHROME_PATH=...` to use another browser).
3. `check_qr.mjs` decodes the QR code from the preview and compares it with the repository URL.

The first run installs the Node dependencies into `poster/node_modules`. Fonts are local
(`visual/fonts/` and `fonts/katex/`), so the render needs no network.

## Files

| File | Purpose |
|---|---|
| `poster.html`, `poster.css` | page source and styles |
| `make_figures.py` | the vector figures, with a check of every number against the paper |
| `build.sh`, `build.mjs`, `check_qr.mjs`, `package*.json` | build and checks |
| `figures/`, `assets/qr-repo.svg` | generated figures and QR code |
| `assets/logos/` | workshop and institution logos; sources and brand rules in `SOURCES.md` |
| `fonts/katex/` | KaTeX stylesheet and fonts |

The logos are trademarks of their owners and are not covered by the repository licenses.
