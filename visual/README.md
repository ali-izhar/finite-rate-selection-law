# Visual edition

The interactive page of the paper, live at <https://ali-izhar.github.io/finite-rate-selection-law/>.
Every number on the page comes from `../results/` or `../src/`; where the page and the paper differ,
the paper is correct.

| Goal | Command (from the repository root) |
|---|---|
| View it locally | `node visual/tools/serve.mjs`, then open http://127.0.0.1:8765 |
| Test the simulation against `src/` | `node visual/lib/sim.test.mjs` |
| Build the site GitHub Pages deploys (on every push to `visual/`) | `sh visual/tools/build-site.sh` |
| Regenerate the page data | `python visual/data/make_exports.py` ([data/README.md](data/README.md)) |

The page is static files: classic scripts under `window.VE`, no build step, no dependencies, relative URLs
only. KaTeX 0.16.22 loads from jsDelivr; everything else is local. Node 18 or later runs the tools.

| Path | Content |
|---|---|
| `index.html` | the page: the overview, six chapters and the limits |
| `lib/` | `sim.js` (simulation kernels and the test's ramp), `ui.js` (controls, plots), `shell.js` (chapters, chapter list, the live frontispiece), `theme.css`, `shell.css`; `sim.test.mjs` |
| `components/` | one figure per chapter: `speciation`, `odds`, `race`, `scaling`, `amplify`, `comparison` |
| `data/` | the numbers the page loads, their generators and the test oracle ([data/README.md](data/README.md)) |
| `fonts/` | self-hosted fonts and KaTeX fonts, with licenses ([fonts/LICENSES.md](fonts/LICENSES.md)) |
| `tools/` | `serve.mjs` (local server), `build-site.sh` (Pages build into `_site/`) |
