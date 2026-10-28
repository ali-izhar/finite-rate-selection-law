# Page data

The numbers the page loads. Each file carries `source` and `paper` keys naming where its values come
from; nothing is typed by hand. Regenerate from the repository root
([visual/README.md](../README.md)):

```bash
python visual/data/make_exports.py        # every data/*.json and assets/mnist_illustration_frames.png (about 20 s)
python visual/data/ref/make_reference.py  # ref/reference.json, the oracle of lib/sim.test.mjs
node visual/data/odds/make_c_vs_bias.mjs  # odds/c_vs_bias.json, with lib/sim.js (about 30 s)
```

Three files are also figure inputs: `fork_trajectories.json` (paper Fig. 1a) and `tracking_time.json`
and `schedule.json` (docs/relaxation-times.md). `checks/paper_consistency.py` reads `schedule.json`.

| File | Content | Source | Paper |
|---|---|---|---|
| `c_cw.json` | c_CW = 0.119808 at N = 2048 on the paper's schedule | `src/experiments/exact_protocol.py` | Sec. 4, Table 1 |
| `fork_trajectories.json` | 14 drift-plus-noise paths, N = 1024, NumPy seed 4 | recomputed; `results/figures/mechanism.json` | Fig. 1a, Table 5 |
| `tracking_time.json` | reflected-chain relaxation time τ(β), N = 256-16384, zero field | `src.gap.gap_well` | docs/relaxation-times.md |
| `schedule.json` | ten temperatures, τ, dwells ⌈6τ⌉ (834 sweeps), reheat 30 sweeps at β = 0.7 | `src/hopfield/protocol.py`, `src.gap` | Sec. 4, Table 3 |
| `field_grid_points.json` | the eight field-grid points with intervals; c = 0.1994 [0.1908, 0.2095] | `results/figures/field_grid_points.json` | Fig. 2b, Table 4 |
| `table1.json` | p = 1..4: A_p, predicted and measured c with intervals | `results/experiments/draw_level_analysis.json` | Table 1 |
| `size_scan.json` | p = 3 at N = 1024, 2048, 4096: measured and predicted c | `results/experiments/size_scan_p3_analysis.json` | Table 1, Sec. 5 |
| `sampler_clock.json` | one-pattern ratios r(N), N = 512-4096 | `results/experiments/sampler_clock.json` | Sec. 5, App. B |
| `cw_scaling.json` | exact-chain c against N and schedule scale; linear ramps | `results/experiments/cw_scaling.json`, `cw_linear_sweep.json` | Table 2, Table 3 |
| `odds/c_vs_bias.json` | exact-chain c at 16 field values, N = 2048 | `lib/sim.js` (tested against `src/`) | Sec. 4 |
| `mnist_illustration.json` | the MNIST ensemble: schedule, classifier probabilities, labels; the control counts | `results/experiments/dbm_reheat.npz`, `dbm_control_arm.json` | docs/mnist-illustration.md |
| `switching_times.json` | exact mean switching times against β, h = 0.05 | `results/figures/switching_times.json` | visual edition only |
| `ref/reference.json` | test oracle: rates, log π, q_eq, exact evolution, A_p, Φ | `ref/make_reference.py` (mpmath) | none |
| `../assets/mnist_illustration_frames.png` | sprite sheet: 24 runs x 141 snapshots, 28 x 28 tiles | `dbm_reheat.npz` | docs/mnist-illustration.md |
