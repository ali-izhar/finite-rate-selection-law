# Saved results

Every file the paper's numbers come from, with its writer and paper location. The checks read these
files and rerun no GPU job. Commands: [docs/reproduce.md](../docs/reproduce.md).

## `simple/`: the pre-registered Hopfield test (paper Sec. 4-5, Table 1, Table 3, Fig. 2b, App. B-C)

Protocol: [docs/protocol-simple.md](../docs/protocol-simple.md). Writer: `python -m src.simple <step>`.

| File | Step | Content | Hardware |
|---|---|---|---|
| `predictions.json` | `predict` | frozen predictions: c_CW, the exact two-block value, A_p c_CW, L_eq, run counts (Table 1, App. B-C) | CPU |
| `g0.json` | `g0` | gate G0: GPU sampler equals the NumPy reference bit for bit (App. B) | GPU |
| `row_N{N}_p{p}.json` | `rows` | per pattern set and field sign: runs, pattern-1 runs, correct sign, ties, m^1 = 0 (Table 1, Table 3) | GPU |
| `analysis.json` | `analyse` | gates G0-G2 (gate rows at 98.75%), the competition rows (95%), the size exponent, reading (a) | CPU |
| `eq_reference_p3.json` | `python -m src.simple.eq_reference` | exact p = 3 Hopfield/CW equilibrium log-odds at N = 512, 1024 (App. B) | CPU |
| `pilot*.json` | `pilot` | the GPU pilot behind the two amendments; never pooled with the rows | GPU |

Table 1 prints 95% intervals for every row; `analysis.json` stores the gate rows (p = 1 and p = 2) at
98.75%, the level the gates were read at, so `checks/claims` recomputes their 95% intervals from the
counts.

## `experiments/cw_linear_sweep.json`: the exact ramps (paper Sec. 3, Table 2, Fig. 2a)

Written by `python -m src.experiments.cw_linear_sweep` (CPU, 2 min).

## `figures/mechanism.json`: Fig. 1a settings (14 paths, N = 1024)

Read by `visual/data/make_exports.py`, which writes `visual/data/fork_trajectories.json` (read by
`paper/figures/make_figures.py`).

## Not used by the paper

The files below belong to the earlier protocol (filtered start, reheat, ten-temperature schedule,
field grid, permutation sweeps) and the MNIST illustration. The paper no longer reads them. They are
kept because the visual edition and the poster read them, and `checks.reanalysis` still audits them.

### `hopfield/`: the field grid (earlier Sec. 4, Fig. 2b, Table 4)

Written by `python -m src.hopfield <step>` (GPU, about 37 min for `grid`).

| File | Step | Content |
|---|---|---|
| `sampler_checks_static.json` | `static` | sampler check G1 and the schedule cross-check |
| `overlap_run.json` | `pre` | equilibrium-reference runs: per-set mean overlap and its SE (App. B) |
| `pilot_run.json` | `pre` | run-budget pilot at G = 0 |
| `field_grid.json` | `grid` | counts per field and pattern set (20 sets, 2000 initial runs each) |
| `sampler_checks.json` | `analyse` | checks G2-G4, including the G = 0 row of Table 4 |
| `field_grid_analysis.json` | `analyse` | per-field intervals and per-set counts: Table 4, Fig. 2b, the slope c |

### `experiments/`: predictions, scans and controls of the earlier protocol

| File | Writer (`python -m src.experiments.<name>`) | Earlier paper location | Hardware |
|---|---|---|---|
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

### `figures/`: other figure inputs

| File | Content | Readers |
|---|---|---|
| `field_grid_points.json` | earlier Fig. 2b fitted slope and band | `visual/data/make_exports.py` |
| `switching_times.json` | exact mean switching times versus β | the visual edition |
| `mnist_illustration.json` | panel layout of the MNIST illustration | `docs/figures/make_mnist_illustration.py` |

GPU runs used the NGC container `nvcr.io/nvidia/pytorch:26.04-py3` on one NVIDIA GPU; reruns reproduce
counts statistically, not bit for bit.
