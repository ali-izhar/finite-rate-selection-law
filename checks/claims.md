# Claims ledger

Every number the paper prints, where it is printed, and how `python -m checks.claims` recomputes it.
A row passes when the recomputed value rounds to the printed one; no tolerance is wider than the
printed precision unless the row says why. Locations refer to [paper/main.pdf](../paper/main.pdf).

Current result: **150 PASS, 0 FAIL, 2 UNVERIFIED** (the UNVERIFIED rows need a GPU).

Categories: **a** recomputed live from exact chains in `src/`; **b** recomputed from saved results
in `results/` or from `src/simple/design.py`, with no rerun (including the post-hoc per-set analysis,
`src/posthoc.py`); **c** needs a GPU rerun.

## Exact chain (a)

| id | Location | Printed | How it is checked |
|---|---|---|---|
| A00 | Sec. 2 | 1.32 | 4^0.2 (odds 4:1 at c = 0.2) |
| A14 | Sec. 3, Fig. 2a caption, Table 2 | exact c exceeds the law by 0.23-1.81% on all nine ramps (within 2%) | nine linear ramps of the exact chain (`src.cw_ramp`) vs logit Phi(a) / logit q_eq |
| A15 | Sec. 3 | 0.256 | smallest N gamma over the nine ramps |
| A16a-c | Table 2 | 0.09370, 0.09372, 0.09361 | exact chain at N^2 gamma fixed |
| A16d | Abstract, Sec. 3 | for a weak field (N G = 1.2544), sixteen times slower cooling only doubles the fraction | c(gamma=0.00025) / c(gamma=0.004) at each N rounds to 2 |
| A16e | Sec. 3 | 58% | Phi(a) at N=2048, gamma=0.001, G=0.0006125 |
| A16f | Sec. 3 | a = 0.208 | G sqrt(N) (pi/gamma)^1/4 for Fig. 1b |
| A16g | Sec. 3 | noise about five times stronger than the field | 1/a rounds to 5 |
| A16h | App. A | the excess about halves when N doubles at gamma = 0.004 and 0.001, not at 0.00025 | ratios of successive Table 2 excesses |
| A21 | Sec. 3 | about 5×10^7 expected sweeps at N = 2048, beta = 1.1 | `src.barrier.log_mfpt` between mode centres, zero field |
| A22 | Sec. 3 | at most 3600 sweeps | (1.6 - 0.7) / 0.00025 |
| A23 | App. A | zero-magnetization mass below 10^-36 | `src.barrier.log_pi_closed_form`, beta = 1.6, N = 512..16384 |
| A08-A11 | Sec. 4 | A_1 = 1, A_2 = 1.414, A_3 = 1.662, A_4 = 1.836 | `src.simple.predict.amplification` (quadrature) |
| A30 | Table 1 (p = 1) | 0.0937, 0.0659, 0.0464 | `src.simple.predict.ramp_laws`: the CW chain of the sampler, advanced update by update |
| A31 | App. A, Table 1 caption | the discrete-update chain reproduces the gamma = 0.004 column of Table 2 | A30 vs `cw_linear_sweep` at 5 dp |
| A32 | App. B | the two start laws give the same c to six digits | independent spins vs equilibrium at beta = 0.7, N = 2048 |
| A33 | App. B | L_eq = 3.5742, 3.5747, 3.5749 | exact CW equilibrium sign log-odds at beta_f = 1.6 |
| A34 | App. B | a = 0.21, 0.15, 0.10 | G sqrt(N) (pi/gamma)^1/4 at gamma = 0.004 |
| A35 | App. B | exact CW exponent 0.506 | log2(c_CW(1024)/c_CW(4096)) / 2 |
| A36 | Table 1 (p >= 3) | 0.1558, 0.1095, 0.0772, 0.1209 | A_p c_CW |
| A37live | App. C | the exact two-block chain | the two pattern sets with the extreme block sizes, recomputed live, equal the stored per-set values |
| A37, A37t | App. C, Table 1 | 0.09363, 0.0936 | pooled exact two-block prediction (`results/simple/predictions.json`) |
| A38 | App. C | per-set values 0.09360-0.09400 | predictions.json `checks.p2.c_per_set` |
| A39, A39b | Sec. 4, App. C | A_2 c_CW = 0.09316, 0.5% below the exact value | A_2 x A30(2048), against A37 |
| A40 | App. C | blocks of 944-1104 spins | both blocks of the 50 sets at N = 2048, from the pattern sets |
| A41 | App. B | p = 2 Hopfield/CW equilibrium log-odds 0.9993 | exact two-block equilibrium, mean over the 50 sets |
| A42 | App. C | a brute-force check at N = 10 agrees to 1e-5 relative | full 2^10-state chain over the design ramp vs independent blocks |
| A20 | App. A | rates, detailed balance, drift and noise of the exact chain | `python -m src` (82 checks) |

## Saved results (b)

| id | Location | Printed | How it is checked |
|---|---|---|---|
| B40 | Table 1, App. B | predictions fixed before the runs | predictions.json equals the live chains of A30, A33, A37 |
| B41p/m/i/r | Table 1, all eight rows | predicted c, measured c, 95% interval, ratio | `src.simple.estimate.estimate` on `results/simple/row_N*_p*.json` (95%, 10,000 resamples, seed 7) |
| B42 | Table 1 | point estimates | equal `results/simple/analysis.json` |
| B43 | Sec. 4 | one-pattern ratios 0.990-1.015 | B41r, p = 1 |
| B44 | Sec. 4 | two-pattern runs within 1.1% of the exact two-block value | B41r, p = 2 |
| B45, B45i | Sec. 4 | 0.995 [0.967, 1.026] | c / (A_3 c_CW) at N = 2048 |
| B46 | Sec. 4 | 0.975 and 0.982; every three-pattern interval contains 1 and allows deficits up to 6% | B41, p = 3 (lowest ratio bound 0.939) |
| B47, B47i | Sec. 4 | N^-0.50, interval [0.47, 0.53] | `src.simple.analyse` estimator and bootstraps |
| B48, B48i, B48j, B48k | Abstract, Intro, Sec. 5, App. B | pooled four-pattern value 6% below A_4 c_CW; interval [0.884, 1.00]; upper end 1.0005 at seed 7 | B41, p = 4 |
| B49 | Abstract, Intro | averaged over pattern sets, within 3% for three patterns at three sizes | max \|ratio - 1\| over p = 3 rows |
| B50 | Sec. 5, App. C | all four A_p c_CW rows below, by 0.5-6% | B41r, p >= 3 |
| B51 | Sec. 5 | p/(N gamma) from 0.18 to 0.73 | A_p c_CW rows |
| B52 | Sec. 5 | largest deficit at p = 4, N = 2048, not at p = 3, N = 1024 | B41r and p/(N gamma) |
| B53 | Sec. 5, App. B | deficits do not order with p/(N gamma) (Spearman 0.6), so reading (c); reading (a) only at seed 7 | Spearman rho and the seed-7 mechanical reading (a) in analysis.json |
| B53b | App. B | three of the four A_p c_CW rows missed the 3% half-width target | 95% ratio half-widths of the p >= 3 rows |
| B54 | App. B | G0 at N = 16, p = 3, 50 updates, 400 runs; overlaps carried = recomputed; all gates passed | g0.json, row drift_max, analysis.json gates |
| B55 | App. B | gate rows at 98.75%, each contains its exact value | `estimate(level=0.9875)` |
| B55b | App. B | each gate row's 95% interval also contains its exact value | B41, p = 1, 2 |
| B63, B63i, B63f | App. B | pilot with a tenth of the planned runs; gate rows failed at 95%: 0.903 [0.819, 0.987], 1.087 [1.007, 1.166] | `estimate` on `results/simple/pilot_N*_p1.json` |
| B64, B64b | App. B | four diagnostic replicates, new seeds and more runs, passed; ratios 0.970-1.016 | `estimate` on `results/simple/pilot_diag*_N*_p1.json` |
| B56, B56b | Table 3, App. B | runs, pattern-1 runs, m^1 = 0 counts, GPU seeds; no tie; no pattern-1 run excluded | row_*.json counts; `design.torch_seed` |
| B57, B57h | App. B | 4.1×10^12 spin updates, about 2.9 GPU hours | row_*.json runs x updates, wall_seconds |
| B58 | Sec. 4, App. B | beta 0.7 to 1.6, gamma = 0.004, M = 225 N, N G = 1.2544, 50 sets, seed 20261027, 10,000 resamples (seed 7), counts doubled | `src/simple/design.py`, row_*.json |
| B59 | App. B | half-width at most 3% of the predicted c | `design.HALF_WIDTH` |
| B60 | Sec. 5, App. C | expected deficits of 0.5-3% | the pre-stated expectation in docs/protocol-simple.md Sec. 6 |
| B61, B61m | App. B | p = 3 Hopfield/CW equilibrium log-odds 0.987-0.997 at N = 512 and 0.991-0.998 at N = 1024 (means 0.994, 0.997) | `results/simple/eq_reference_p3.json` (`python -m src.simple.eq_reference`, exact four-block sums over the 50 pattern sets) |
| B62 | Sec. 4 | the CW denominator is within 1.3% of the Hopfield one where both were computed | largest 1 - ratio over B61 and the p = 2 mean |
| B27 | Fig. 1 caption, App. A | fourteen paths at N = 1024 | results/figures/fork_trajectories.json |

## Post-hoc per-set analysis (b, not in the protocol)

Recomputed live by `src.posthoc` from `results/simple/row_*.json` and `predictions.json`, and compared
with the saved `results/posthoc/per_set.json` (`python -m src.posthoc`, CPU, about 10 s).

| id | Location | Printed | How it is checked |
|---|---|---|---|
| B70 | Table 4 | saved per-set analysis | equals the live recomputation |
| B71w/p/s/d/e | Table 4, four A_p rows | win share range, chi-square P (49 df), Spearman, excess SD, equal-weight ratio | `src.posthoc.per_set` |
| B72 | Sec. 5 | four-pattern win share 0.09-0.71 | B71w, p = 4 |
| B73 | Sec. 5 | per-set c falls as the win share rises in every A_p row | every Spearman rho < 0 |
| B74 | Sec. 5 | per-set c / (A_p c_CW) varies by 7-13% beyond sampling noise | B71d |
| B75 | Sec. 5 | sets weighted equally: four-pattern ratio 1.00 | B71e, p = 4 |
| B76 | App. B | no excess spread in the one- and two-pattern rows, P = 0.56-0.94 | chi-square on p = 1, 2 rows |
| B76b | Abstract | single pattern sets depart from the prediction by more than sampling noise | largest chi-square P over the A_p rows < 1e-4 |
| B77, B78, B79, B79b | App. B | upper end of the four-pattern interval 0.997-1.002 over bootstrap seeds 0-199, below 1 for 139 of them; the other A_p intervals contain 1 at every seed | `src.posthoc.boot_seeds` (the paper's estimator, 10,000 resamples, 95%) |
| B79c | Sec. 5, App. B | the analysis is labeled post hoc | text match in paper/main.tex |

## Reruns (c)

| id | Location | Printed | How it is checked |
|---|---|---|---|
| C10 | Table 1, Table 3, Fig. 2b | the eight protocol rows | rerun: `python -m src.simple rows --batch 131072` (GPU, about 2.9 h), then `analyse`; the saved counts are checked in B41-B57 |
| C11 | App. B | gate G0 | rerun: `python -m src.simple g0`; `python -m pytest -q src/simple/tests` with torch; the saved result is checked in B54 |

## Checked by `checks.paper_consistency`

- Table 1: every cell, against `results/simple/` (prediction kind, predicted, measured, interval, ratio).
- Table 2: all nine entries (`cw_linear_sweep.json`); Fig. 1b, a = 0.208.
- Table 3: runs, pattern-1 runs, m^1 = 0 counts and GPU seeds, against `results/simple/row_*.json`.
- Table 4: every cell, against `results/posthoc/per_set.json`; the paper states reading (c) and that
  reading (a) holds only at bootstrap seed 7.
- The printed design constants and seeds, against `src/simple/design.py`; 5×10^7 sweeps; 10^-36.
- The elements of the earlier protocol (reheat, filtered start, dwells, 834 sweeps, field grid, 0.199,
  1.063, the reflected chain) are absent; permutation sweeps appear once, in App. B.
- The compiled PDF: six main-text pages, references on page 7, appendix from page 8, at most 14 pages,
  no unresolved reference, all fonts embedded, authors in `paper/main.tex` order.

## Cited, not computed

The selection law q = Φ(G√N(π/γ)^¼) (Eq. 3; Kondepudi and Nelson 1983, Kondepudi and Gao 1987).
