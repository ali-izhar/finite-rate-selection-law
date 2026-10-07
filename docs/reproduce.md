# Reproduce the paper

Every table, figure and number, one command each. Run every command from the repository root,
after the [quick start](../README.md#quick-start).

| Item | Command | Needs |
|---|---|---|
| PDF | `(cd paper && latexmk -pdf -outdir=build main.tex)` | TeX Live |
| Paper against saved results | `python -m checks.paper_consistency` (after the PDF build) | poppler |
| Figures 1–2 | `python paper/figures/make_figures.py` | CPU, seconds |
| c = 0.199 and its interval | `python -m src.hopfield.seed_bootstrap` | CPU, seconds |
| Fits, bootstraps, p = 2 blocks, pattern geometry | `python -m checks.reanalysis` | CPU, seconds |
| Table 1, predicted and intervals | `python -m src.experiments.exact_protocol_sizes`; `python -m src.experiments.draw_level_analysis`; `python -m src.experiments.size_scan_p3_analysis` | CPU, 20 s |
| Sec. 5, one-pattern ratios | `python -m src.experiments.sampler_clock` | CPU, seconds |
| Table 2 | `python -m src.experiments.cw_linear_sweep` | CPU, 2 min |
| Table 3, c_CW | `python -m src.experiments.exact_protocol` (c_CW = 0.120); `python -m src.experiments.cw_scaling` | CPU, 2 min each |
| Table 1, measured | `HOPFIELD_OUT=rerun python -m src.experiments.hopfield_runs draws P` (P = 1..4); `... draws 3 1024,4096` | GPU, 10–48 min each |
| Table 1, measured, analysed | `HOPFIELD_OUT=rerun python -m src.experiments.draw_level_analysis`; `HOPFIELD_OUT=rerun python -m src.experiments.size_scan_p3_analysis` | CPU, seconds |
| Sec. 5, one-pattern runs | `HOPFIELD_OUT=rerun python -m src.experiments.hopfield_runs size-p1 1024` | GPU, 24 min per size |
| Table 4, Fig. 2b | `export HOPFIELD_OUT=rerun`, then `python -m src.hopfield static`, `pre`, `grid`, `analyse` | GPU, 37 min for `grid` |
| Figures in `docs/` | `python docs/figures/make_relaxation_times.py`; `python docs/figures/make_mnist_illustration.py` | CPU, seconds |
| MNIST illustration | `python -m src.experiments.dbm_reheat`; `python -m src.experiments.dbm_control_arm` | CPU, MNIST |

CPU commands rewrite their file in `results/` with the same numbers. GPU commands need
`pip install -r requirements-gpu.txt` and write to `$HOPFIELD_OUT`; the shell lines above put that in
`rerun/` (ignored), so the saved results stay. GPU and MNIST reruns reproduce counts statistically, not
bit for bit; the MNIST runs overwrite their saved file, and `git restore results` brings it back.
[results/README.md](../results/README.md) names the writer and paper location of every saved file.

## MNIST data

The MNIST runs read the raw training files from `~/.cache/mnist_data/MNIST/raw` (set `MNIST_RAW` to
change it):

```bash
d=~/.cache/mnist_data/MNIST/raw && mkdir -p $d
for f in train-images-idx3-ubyte train-labels-idx1-ubyte; do
  curl -fsS https://ossci-datasets.s3.amazonaws.com/mnist/$f.gz | gunzip > $d/$f
done
```
