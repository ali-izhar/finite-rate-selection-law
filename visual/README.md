# Visual edition

The interactive page and the 14-minute narrated film of the paper, live at
<https://ali-izhar.github.io/finite-rate-selection-law/> (the film opens at `#watch`). Every number on the
page comes from `../results/` or `../src/`; where the page and the paper differ, the paper is correct.

| Goal | Command (from the repository root) |
|---|---|
| View it locally | `node visual/tools/serve.mjs`, then open http://127.0.0.1:8765 |
| Test the simulation against `src/` | `node visual/lib/sim.test.mjs` |
| Build the site GitHub Pages deploys (on every push to `visual/`) | `sh visual/tools/build-site.sh` |
| Regenerate the page data | `python visual/data/make_exports.py` |
| Rebuild the film | see [film/README.md](film/README.md) |

The page is static files: classic scripts under `window.VE`, no build step, no dependencies, relative URLs
only. KaTeX 0.16.22 loads from jsDelivr; everything else is local. Node 18 or later runs the tools;
`python -m http.server -d visual` also serves the page, without video seeking.

| Path | Content |
|---|---|
| `index.html` | the page: chapters, film player, references |
| `lib/` | `sim.js` (simulation kernels), `ui.js` (controls, plots), `shell.js` (chapters, film player), `theme.css`, `shell.css`; `sim.test.mjs` |
| `components/` | one interactive figure per chapter: `hero`, `speciation`, `slowing`, `odds`, `race`, `scaling`, `amplify`, `reheat`, `comparison` |
| `data/` | the numbers the page loads; [data/README.md](data/README.md) gives each file's source |
| `assets/` | digit sprite sheet of the MNIST illustration |
| `media/` | the film, nine chapter clips, captions, posters and `media.json` |
| `fonts/` | self-hosted fonts and KaTeX fonts, with licenses ([fonts/LICENSES.md](fonts/LICENSES.md)) |
| `film/` | film sources: script, manim scenes, narration, assembly ([film/README.md](film/README.md)) |
| `tools/` | `serve.mjs` (local server with range requests), `build-site.sh` (Pages build into `_site/`), `media.mjs` (copies a film build into `media/`) |
