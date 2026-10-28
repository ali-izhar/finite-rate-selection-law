# Saved results

Every file the paper's numbers come from, with its writer and paper location. The checks read these
files and rerun no GPU job. Commands: [docs/reproduce.md](../docs/reproduce.md).

## `simple/`: the Hopfield test of the written protocol (paper Sec. 4-5, Table 1, Table 3, Fig. 2b, App. B-C)

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

## `posthoc/per_set.json`: the post-hoc per-set analysis (paper Sec. 5, App. B, Table 4)

Written by `python -m src.posthoc` (CPU, about 10 s) from `simple/row_N*_p*.json` and
`simple/predictions.json`. Not in the protocol. Per row: pattern 1's win share per set, a chi-square test
that all sets share one sign fraction, the Spearman correlation of per-set c with the win share, the
spread of per-set c / c_pred beyond binomial noise, the equal-weight ratio, and the 95% interval of the
protocol's estimator under bootstrap seeds 0-199. Read by `checks.claims` rows B70-B79c.

## `experiments/cw_linear_sweep.json`: the exact ramps (paper Sec. 3, Table 2, Fig. 1b, Fig. 2a)

Written by `python -m src.cw_ramp` (CPU, 2 min).

## `figures/`: inputs of Fig. 1a

Written by `python paper/figures/make_fork.py` (CPU, seconds). `fork_trajectories.json` holds the
14 paths (N = 1024, 9 end at m > 0) drawn by `paper/figures/make_figures.py` and read by
`checks.claims` row B27.
