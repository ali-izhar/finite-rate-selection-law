# Reproduce the paper

Every table, figure and number, one command each. Run every command from the repository root,
after the [quick start](../README.md#quick-start).

| Item | Command | Needs |
|---|---|---|
| PDF | `(cd paper && latexmk -pdf -outdir=build main.tex)` | TeX Live |
| Paper against saved results | `python -m checks.paper_consistency` (after the PDF build) | poppler |
| Every printed number | `python -m checks.claims` | CPU, about 4 min |
| Exact-chain implementation | `python -m src` | CPU, seconds |
| Hopfield-test code | `python -m pytest -q src/simple/tests` | CPU, 15 s; the GPU test needs torch |
| Fig. 1a input | `python paper/figures/make_fork.py` | CPU, seconds |
| Figures 1–2 | `python paper/figures/make_figures.py` | CPU, seconds |
| Table 2, Fig. 2a | `python -m src.cw_ramp` | CPU, 2 min |
| Table 1, predictions | `python -m src.simple predict` | CPU, 5 min |
| Table 1, measured, from the saved counts | `python -m src.simple analyse` | CPU, seconds |
| Table 4, post-hoc per-set analysis | `python -m src.posthoc` | CPU, about 10 s |
| App. B, $p=3$ equilibrium reference | `python -m src.simple.eq_reference` | CPU, about 6 min on 12 cores |
| Gate G0 | `python -m src.simple g0` | GPU (torch), seconds |
| Table 1 and Table 3, measured | `python -m src.simple rows --batch 131072`, then `python -m src.simple analyse` | GPU, about 3 h |

`predict` and `analyse` rewrite `results/simple/predictions.json` and `analysis.json` with the same
numbers and a new provenance block. A GPU rerun reproduces the counts bit for bit only with the same
seeds, batch size and hardware; `--seed-offset 1` reruns every row with new seeds
(`row_N*_p*_rerun1.json`, `analysis_rerun1.json`) and leaves the saved files alone. The GPU steps run
in the NGC container `nvcr.io/nvidia/pytorch:26.04-py3`; the command line is in
[`src/simple/__main__.py`](../src/simple/__main__.py). `git restore results` brings back the saved files.

[results/README.md](../results/README.md) names the writer and paper location of every saved file.
