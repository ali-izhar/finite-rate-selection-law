# Saved results

Every file the paper's numbers come from, with its writer and paper location. The checks read these
files and rerun no GPU job. Commands: [docs/reproduce.md](../docs/reproduce.md).

## `hopfield/`: the field grid (paper Sec. 4, Fig. 2b, Table 4)

Written by `python -m src.hopfield <step>` (GPU, about 37 min for `grid`).

| File | Step | Content |
|---|---|---|
| `sampler_checks_static.json` | `static` | sampler check G1 and the schedule cross-check |
| `overlap_run.json` | `pre` | equilibrium-reference runs: per-set mean overlap and its SE (App. B) |
| `pilot_run.json` | `pre` | run-budget pilot at G = 0 |
| `field_grid.json` | `grid` | counts per field and pattern set (20 sets, 2000 initial runs each) |
| `sampler_checks.json` | `analyse` | checks G2-G4, including the G = 0 row of Table 4 |
| `field_grid_analysis.json` | `analyse` | per-field intervals and per-set counts: Table 4, Fig. 2b, the slope c |

## `experiments/`: predictions, scans and controls

| File | Writer (`python -m src.experiments.<name>`) | Paper | Hardware |
|---|---|---|---|
| `cw_linear_sweep.json` | `cw_linear_sweep` | Table 2, Fig. 2a | CPU |
| `exact_protocol.json` | `exact_protocol` | c_CW over the eight fields (Table 3) | CPU |
| `exact_protocol_sizes.json` | `exact_protocol_sizes` | c_CW at N = 512-4096 (Table 1) | CPU |
| `cw_scaling.json` | `cw_scaling` | schedule scaled by 0.25 and 4 (Table 3) | CPU |
| `pattern_count_p{1..4}.json` | `hopfield_runs draws P` | Table 1, p = 1..4 | GPU |
| `size_scan_p3.json` | `hopfield_runs draws 3 1024,4096` | Table 1, N = 1024, 4096 | GPU |
| `size_scan_p1.json` | `hopfield_runs size-p1` | one-pattern runs (Sec. 5, App. B, App. C) | GPU |
| `draw_level_analysis.json` | `draw_level_analysis` | Table 1 intervals | CPU |
| `size_scan_p3_analysis.json` | `size_scan_p3_analysis` | Table 1 size rows, Sec. 5 exponents | CPU |
| `sampler_clock.json` | `sampler_clock` | one-pattern ratios 1.065-1.020 (Sec. 5, App. B) | CPU |
| `dbm_reheat.npz` | `dbm_reheat` | docs/mnist-illustration.md (figure ensemble) | CPU, MNIST |
| `dbm_control_arm.json` | `dbm_control_arm` | docs/mnist-illustration.md (counts) | CPU, MNIST |

## `figures/`: figure inputs

| File | Content | Readers |
|---|---|---|
| `mechanism.json` | Fig. 1a settings (14 paths, N = 1024) and relaxation-time peaks | `visual/data/make_exports.py`, which writes `visual/data/fork_trajectories.json` (read by `paper/figures/make_figures.py`) and `visual/data/tracking_time.json` (read by `docs/figures/make_relaxation_times.py`) |
| `field_grid_points.json` | Fig. 2b fitted slope and band | `paper/figures/make_figures.py`, `visual/data/make_exports.py` |
| `switching_times.json` | exact mean switching times versus β | the visual edition |
| `mnist_illustration.json` | panel layout of the MNIST illustration | `docs/figures/make_mnist_illustration.py` |

GPU runs used the NGC container `nvcr.io/nvidia/pytorch:26.04-py3` on one NVIDIA GPU; reruns reproduce
counts statistically, not bit for bit.
