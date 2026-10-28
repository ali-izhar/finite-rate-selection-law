# A Finite-Rate Selection Law for Annealed Gibbs Sampling

[![checks](https://github.com/ali-izhar/finite-rate-selection-law/actions/workflows/checks.yml/badge.svg)](https://github.com/ali-izhar/finite-rate-selection-law/actions/workflows/checks.yml)
[![pages](https://github.com/ali-izhar/finite-rate-selection-law/actions/workflows/pages.yml/badge.svg)](https://ali-izhar.github.io/finite-rate-selection-law/)
[![license: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)

Izhar Ali, Muhammad Haris Khan, Saeed Ahmad · DynaFront Workshop at NeurIPS 2026 (poster)

[**Paper**](paper/main.pdf) · [**Poster**](poster/poster.pdf) · [**Visual edition**](https://ali-izhar.github.io/finite-rate-selection-law/) · [**Film**](https://ali-izhar.github.io/finite-rate-selection-law/#watch) (14 min) · [**Docs**](docs/README.md)

<a href="https://ali-izhar.github.io/finite-rate-selection-law/"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/visual-edition-dark.jpg">
  <img src="docs/images/visual-edition-light.jpg" alt="The visual edition: Gibbs-sampler paths wander together while hot, then split into two modes at the transition, beta = 1.">
</picture></a>

A sampler can draw realistic samples in the wrong proportions. In annealed Gibbs sampling, the mode
is chosen early in cooling, near a transition, where a weak preference competes with random
fluctuations; afterwards, switching modes is slow (at N = 2048 and β = 1.1, a switch takes about
5×10⁷ sweeps, against at most 3600 on our ramps). The samples then keep only a fraction **c** of the
model's log-odds: with c = 0.2, model odds of 4:1 become 1.32:1. We give a law for c, built on a
classical result for this selection. It explains why slower cooling helps little: on a linear ramp,
sixteen times more sweeps only doubles c (a sweep is N single-variable updates).

## Results

| Test | Predicted | Measured | In the paper |
|---|---|---|---|
| Exact Curie–Weiss chain, 3 sizes × 3 cooling rates | the law | c above the law by 0.23–1.81% | Sec. 3, Table 2 |
| Hopfield, 1 pattern, N = 1024, 2048, 4096 (sampler check) | exact chain: 0.0937, 0.0659, 0.0464 | ratio 0.990, 1.001, 1.015 | Sec. 4, Table 1 |
| Hopfield, 2 patterns, N = 2048 | exact two-block chain: 0.0936 | 0.0946 [0.0928, 0.0965] | Sec. 4, Table 1 |
| Hopfield, 3 patterns, N = 2048 (primary) | A₃ c_CW = 0.1095 | 0.1090 [0.1059, 0.1123] | Sec. 4, Table 1 |
| Hopfield, 3 patterns, N = 1024, 4096 | 0.1558, 0.0772 | 0.1518, 0.0758 | Sec. 4, Table 1 |
| Hopfield, 4 patterns, N = 2048 | A₄ c_CW = 0.1209 | 0.1137 [0.1069, 0.1210] (6% low) | Sec. 5, Table 1 |

The exact check uses the Curie–Weiss model, whose Gibbs chain reduces to N + 1 counts and is evolved
without sampling noise. The Hopfield test is pre-registered ([docs/protocol-simple.md](docs/protocol-simple.md)):
every run cools on one linear ramp (β = 0.7 to 1.6 at γ = 0.004) from independent unbiased spins, with
random-site heat-bath updates, and one estimator serves every row; intervals are 95% and resample the 50
pattern sets. With one pattern the Hopfield sampler is exactly the Curie–Weiss chain, so those rows check
the sampler; with two patterns the prediction is exact; for three and four patterns it multiplies the
exact Curie–Weiss fraction by a Gaussian competition factor A_p. Three patterns agree to within 3% at
three sizes; four patterns fall 6% below the prediction, and the paper states the expected cause and
why it is untested. The results concern Gibbs cooling, not reverse diffusion.

## Quick start

Python 3.12 or 3.13 and a CPU are enough to recheck every number the paper prints:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src                       # the exact sampler: "82/82 checks passed" (seconds)
python -m pytest -q src/simple/tests   # the Hopfield-test code; the GPU test is skipped without torch
python -m checks.claims             # every printed number: "107 PASS, 0 FAIL, 2 UNVERIFIED" (about 4 min)
```

The two UNVERIFIED rows are the GPU runs; their saved outputs are checked.
[checks/claims.md](checks/claims.md) lists every number with its location in the paper.

## Reproduce

| Goal | Command |
|---|---|
| Build the PDF (TeX Live) | `(cd paper && latexmk -pdf -outdir=build main.tex)` |
| Check the PDF against the saved results (after the build; poppler) | `python -m checks.paper_consistency` |
| Redraw Figures 1–2 | `python paper/figures/make_figures.py` |
| Table 2: the exact Curie–Weiss ramps | `python -m src.experiments.cw_linear_sweep` |
| Table 1 predictions (exact chains, A_p) | `python -m src.simple predict` |
| Table 1 measured c and intervals, from the saved counts | `python -m src.simple analyse` |
| Rerun the Hopfield test (GPU, about 3 h) | see [docs/reproduce.md](docs/reproduce.md) |

[docs/reproduce.md](docs/reproduce.md) gives one command for every table and figure.

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
| [`src/`](src/) | the exact Curie–Weiss sampler; [`simple/`](src/simple/) (the pre-registered Hopfield test) and [`experiments/`](src/experiments/); [`hopfield/`](src/hopfield/) holds the earlier protocol, kept for the visual edition |
| [`results/`](results/README.md) | saved runs, each with the script that wrote it and its place in the paper |
| [`checks/`](checks/README.md) | checks of every number, table and caption in the paper |
| [`docs/`](docs/README.md) | details the paper points to: protocol and pre-registration, full reproduction steps |
| [`models/`](models/README.md) | the Boltzmann machine and classifier of the MNIST illustration (not used by the paper) |
| [`visual/`](visual/README.md) | the visual edition and the [film](visual/film/README.md) |
| [`poster/`](poster/README.md) | the workshop poster (24 × 36 in) and its build |

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
The logos in `poster/assets/logos/` are trademarks of their owners and are used with permission.
