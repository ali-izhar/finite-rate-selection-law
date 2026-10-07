# A Finite-Rate Selection Law for Annealed Gibbs Sampling

[![checks](https://github.com/ali-izhar/finite-rate-selection-law/actions/workflows/checks.yml/badge.svg)](https://github.com/ali-izhar/finite-rate-selection-law/actions/workflows/checks.yml)
[![pages](https://github.com/ali-izhar/finite-rate-selection-law/actions/workflows/pages.yml/badge.svg)](https://ali-izhar.github.io/finite-rate-selection-law/)
[![license: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)

Izhar Ali, Muhammad Haris Khan, Saeed Ahmad · DynaFront Workshop at NeurIPS 2026 (poster)

[**Paper**](paper/main.pdf) · [**Visual edition**](https://ali-izhar.github.io/finite-rate-selection-law/) · [**Film**](https://ali-izhar.github.io/finite-rate-selection-law/#watch) (14 min) · [**Docs**](docs/README.md)

<a href="https://ali-izhar.github.io/finite-rate-selection-law/"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/visual-edition-dark.jpg">
  <img src="docs/images/visual-edition-light.jpg" alt="The visual edition: Gibbs-sampler paths wander together while hot, then split into two modes at the transition, beta = 1.">
</picture></a>

A sampler can draw realistic samples in the wrong proportions. In annealed Gibbs sampling, the mode
is chosen early in cooling, near a transition, where a weak preference competes with random
fluctuations; afterwards, switching modes is slow (at N = 2048, a switch takes about 5×10⁷ sweeps,
against an 834-sweep schedule). The samples then keep only a fraction **c** of the model's log-odds:
with c = 0.2, model odds of 4:1 become 1.32:1. We give a law for c, built on a classical result for
this selection. It explains why slower cooling helps little: on a linear ramp, sixteen times more
sweeps only doubles c (a sweep is N single-variable updates).

## Results

| Test | Predicted | Measured | In the paper |
|---|---|---|---|
| Exact Curie–Weiss chain, 3 sizes × 3 cooling rates | the law | c above the law by 0.23–1.81% | Sec. 3, Table 2 |
| Hopfield network, 3 patterns, N = 2048 | 0.199 | 0.199 [0.1908, 0.2095] | Sec. 4, Fig. 2b |
| 1, 2, 3, 4 patterns | 0.120, 0.169, 0.199, 0.220 | 0.127, 0.188, 0.201, 0.211 | Table 1 |
| N = 1024, 4096, 3 patterns | 0.284, 0.150 | 0.263, 0.160 | Table 1 |

The exact check uses the Curie–Weiss model, whose Gibbs chain reduces to N + 1 counts and is evolved
without sampling noise. The Hopfield tests reheat runs from a filtered start, and their predictions
use no fitted parameter. The 0.199 match may come partly from cancelling approximation errors; away
from three patterns at N = 2048 the prediction misses by up to 11%. The results concern Gibbs
cooling, not reverse diffusion.

## Quick start

Python 3.12 or 3.13 and a CPU are enough to recheck every number the paper prints:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src              # the exact sampler: "82/82 checks passed" (seconds)
python -m checks.claims    # every printed number: "75 PASS, 0 FAIL, 4 UNVERIFIED" (2 min)
```

The four UNVERIFIED rows are GPU and MNIST reruns; their saved outputs are checked.
[checks/claims.md](checks/claims.md) lists every number with its location in the paper.

## Reproduce

| Goal | Command |
|---|---|
| Build the PDF (TeX Live) | `(cd paper && latexmk -pdf -outdir=build main.tex)` |
| Check the PDF against the saved results (after the build; poppler) | `python -m checks.paper_consistency` |
| Redraw Figures 1–2 | `python paper/figures/make_figures.py` |
| Recompute c on the exact Curie–Weiss chain (0.120) | `python -m src.experiments.exact_protocol` |
| Recompute c = 0.199 and its interval | `python -m src.hopfield.seed_bootstrap` |
| Rerun the Hopfield experiments (GPU) | see [docs/reproduce.md](docs/reproduce.md) |

[docs/reproduce.md](docs/reproduce.md) gives one command for every table and figure, and the MNIST runs.

## Visual edition and film

The [visual edition](https://ali-izhar.github.io/finite-rate-selection-law/) explains the paper with
interactive simulations and a narrated film.

| Goal | Command |
|---|---|
| View it locally | `node visual/tools/serve.mjs`, then open http://127.0.0.1:8765 |
| Rebuild the film | see [visual/film/README.md](visual/film/README.md) |

## Layout

| Folder | Content |
|---|---|
| [`paper/`](paper/) | the paper: `main.tex`, `main.pdf`, bibliography, DynaFront template, figures |
| [`src/`](src/) | the exact Curie–Weiss sampler; [`hopfield/`](src/hopfield/) and [`experiments/`](src/experiments/) |
| [`results/`](results/README.md) | saved runs, each with the script that wrote it and its place in the paper |
| [`checks/`](checks/README.md) | checks of every number, table and caption in the paper |
| [`docs/`](docs/README.md) | details the paper points to: protocol (schedules, estimators, seeds), derivations, full reproduction steps |
| [`models/`](models/README.md) | the Boltzmann machine and classifier of the MNIST illustration |
| [`visual/`](visual/README.md) | the visual edition and the [film](visual/film/README.md) |

## Citation

```bibtex
@inproceedings{ali2026finiterate,
  title     = {A Finite-Rate Selection Law for Annealed Gibbs Sampling},
  author    = {Ali, Izhar and Khan, Muhammad Haris and Ahmad, Saeed},
  booktitle = {DynaFront Workshop at NeurIPS 2026},
  year      = {2026},
  url       = {https://github.com/ali-izhar/finite-rate-selection-law}
}
```

## License

Code: [MIT](LICENSE). Paper, figures and media: [CC BY 4.0](LICENSE-CC-BY-4.0).
Fonts: SIL Open Font License ([visual/fonts/LICENSES.md](visual/fonts/LICENSES.md)).
`paper/jmlr.cls` and `paper/jmlrutils.sty`: LaTeX Project Public License 1.3;
`paper/dynafront2026.cls`: the DynaFront 2026 template, unmodified.
The digit images derive from MNIST (LeCun, Cortes and Burges).
