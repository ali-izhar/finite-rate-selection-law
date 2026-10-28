# Saved results

Every file the paper's numbers come from, with its writer and paper location. The checks read these
files and rerun no GPU job. Commands: [docs/reproduce.md](../docs/reproduce.md).

## `simple/`: the Hopfield test of the written protocol (paper Sec. 4, Table 3, Fig. 2b, App. B-C)

Protocol: [docs/protocol-simple.md](../docs/protocol-simple.md). Writer: `python -m src.simple <step>`.

| File | Step | Content | Hardware |
|---|---|---|---|
| `predictions.json` | `predict` | frozen predictions: c_CW, the exact two-block value, A_p c_CW, L_eq, run counts (Table 3, App. B-C) | CPU |
| `g0.json` | `g0` | gate G0: GPU sampler equals the NumPy reference bit for bit (protocol Sec. 3; not in the paper) | GPU |
| `row_N{N}_p{p}.json` | `rows` | per pattern set and field sign: runs, pattern-1 runs, correct sign, ties, m^1 = 0 (Table 3, Table 4) | GPU |
| `analysis.json` | `analyse` | gates G0-G2 (gate rows at 98.75%), the competition rows (95%), the size exponent, reading (a) | CPU |
| `eq_reference_p3.json` | `python -m src.simple.eq_reference` | exact p = 3 Hopfield/CW equilibrium log-odds at N = 512, 1024 (App. C) | CPU |
| `pilot*.json` | `pilot` | the GPU pilot behind the two amendments; never pooled with the rows | GPU |

Table 3 prints 95% intervals for every row; `analysis.json` stores the gate rows (p = 1 and p = 2) at
98.75%, the level the gates were read at, so `checks/claims` recomputes their 95% intervals from the
counts.

## `posthoc/per_set.json`: the post-hoc per-set analysis (paper Sec. 4, App. C, Table 4)

Written by `python -m src.posthoc` (CPU, about 10 s) from `simple/row_N*_p*.json` and
`simple/predictions.json`. Not in the protocol. Per row: pattern 1's win share per set, a chi-square test
that all sets share one sign fraction, the Spearman correlation of per-set c with the win share, the
spread of per-set c / c_pred beyond binomial noise, the equal-weight ratio, and the 95% interval of the
protocol's estimator under bootstrap seeds 0-199. Also `paired_N2048_p3_minus_p4`: the difference of
c / c_pred between the three- and four-pattern rows at N = 2048, which share their pattern sets, with a
paired cluster bootstrap under the same seeds. Read by `checks.claims` rows B70-B82.

## `experiments/cw_linear_sweep.json`: the exact ramps (paper Sec. 3, Table 1, Fig. 1b, Fig. 1c)

Written by `python -m src.cw_ramp` (CPU, 2 min).

## `experiments/switching.json`: sign changes after the selection window (paper Sec. 3, App. A)

Written by `python -m src.switching` (CPU, 3 min). For each of the nine ramps: the fraction of runs of
the exact chain whose sign of m at β = 1 + 3√γ, and at β = 1.1, differs from the final sign. After the
later of the two points at most 0.54% of runs switch; after 1 + 3√γ on the slowest ramp, 11.9% (N = 1024)
and 2.7% (N = 2048) still do. Read by `checks.claims` rows A25-A26.

## `figures/`: inputs of Fig. 1a

Written by `python paper/figures/make_fork.py` (CPU, seconds). `fork_trajectories.json` holds the
14 paths (N = 1024, 9 end at m > 0) drawn by `paper/figures/make_figures.py`.
