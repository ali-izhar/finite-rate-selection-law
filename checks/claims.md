# Claims ledger

Every number the paper prints, where it is printed, and how `python -m checks.claims` recomputes it.
A row passes when the recomputed value rounds to the printed one; no tolerance is wider than the
printed precision unless the row says why. Locations refer to [paper/main.pdf](../paper/main.pdf);
a `docs/` location is a repository page the paper points to.

Current result: **78 PASS, 0 FAIL, 4 UNVERIFIED** (the UNVERIFIED rows need a GPU or MNIST).

Categories: **a** recomputed live from the exact chain in `src/`; **b** recomputed from saved results
in `results/` or `models/`, with no rerun; **c** needs a GPU rerun or external data.

## Exact chain (a)

| id | Location | Printed | How it is checked |
|---|---|---|---|
| A01 | docs/relaxation-times.md | 76.94 | max over beta of 1/gap_well(16384, beta), bounded maximiser |
| A02 | docs/relaxation-times.md | 1.012 | argmax of the same |
| A03 | App. B | 1.6946 | 1/gap_well(2048, 1.6): reflected-chain relaxation time |
| A04 | App. A | 42.2, 59.0, 82.7 sweeps | 1/gap_full(N, 1.0) for N = 1024, 2048, 4096 |
| A05 | Sec. 4, Table 1 caption | 0.120 | src.experiments.exact_protocol.protocol(G0, dt=0.02): logit(q_model)/logit(q_eq) |
| A06 | Table 3 caption | 2.8e-11 | \|c_CW(dt=0.02) - c_CW(dt=0.01)\|, src.experiments.exact_protocol.protocol(G0, dt) |
| A07 | App. C | 3.575 | logit of the exact CW q_eq at N=2048, beta=1.6, G=0.0006125 |
| A08 | Sec. 4 | 1.414 | E[max_2\|Z\|]/E\|Z\| (= sqrt 2) |
| A09 | Sec. 4 | 1.662 | E[max_3\|Z\|]/E\|Z\| by quadrature |
| A10 | Sec. 4 | 1.836 | E[max_4\|Z\|]/E\|Z\| |
| A11 | Table 1 | 0.169 | c_CW x A_2 |
| A12 | Sec. 4, Table 1 | 0.199 | c_CW x A_3 |
| A13 | Table 1 | 0.220 | c_CW x A_4 |
| A14 | Sec. 3, Table 2 | exact c exceeds the law by 0.23-1.81% on all nine ramps | nine linear ramps of the exact chain vs logit Phi(G sqrt N (pi/gamma)^1/4) / logit q_eq |
| A15 | Sec. 3 | 0.256 | smallest N gamma over the nine ramps |
| A16a | Table 2 | 0.09370 | exact chain, N=1024, gamma=0.004 |
| A16b | Table 2 | 0.09372 | exact chain, N=2048, gamma=0.001 |
| A16c | Table 2 | 0.09361 | exact chain, N=4096, gamma=0.00025 |
| A16d | Abstract, Sec. 3 | sixteen times slower cooling only doubles the fraction | exact chain: c(gamma=0.00025) / c(gamma=0.004) at N = 1024, 2048, 4096; each ratio rounds to 2 |
| A16e | Sec. 3 | 58% of runs end in the favored mode | q_sel = Phi(a) at N=2048, gamma=0.001, G=0.0006125 (Fig. 1b), percent |
| A17 | Sec. 4, Table 3 | dwell ceil(6 tau) sweeps, 834 in total | tau = 1/gap_full for beta <= 1, 1/gap_well above (N = 2048, G = 0) |
| A18 | Table 4 caption | 1.2e-4 | exp(-30/tau), tau = 1/gap_full(2048, 0.7); tau = 3.327 sweeps |
| A19 | Sec. 4, Table 4 | eight nonzero fields, four on each side of q_eq = 1/2 | src.hopfield.equilibrium.qeq_route_a(2048, 1.6, G) over src.hopfield.protocol.G_GRID |
| A20 | App. A | rates, detailed balance, drift and noise of the exact chain | python -m src (src/validate.py) |

## Saved results (b)

| id | Location | Printed | How it is checked |
|---|---|---|---|
| B01 | Sec. 4, App. B | 0.1994 | src/hopfield/seed_bootstrap.py on results/hopfield/field_grid_analysis.json; sample-weighted mean of per-set slopes |
| B02 | Sec. 4 | 0.2101 | src/hopfield/seed_bootstrap.py on results/hopfield/field_grid_analysis.json; equal-weight mean of per-set slopes |
| B03 | Sec. 4, App. B | [0.1908, 0.2095] | src/hopfield/seed_bootstrap.py on results/hopfield/field_grid_analysis.json; 20,000 pattern-set resamples, seed 20260825 |
| B31 | record (no longer printed) | 66.9% sampled, 97.26% reference at G = 0.0006125 | results/hopfield/field_grid_analysis.json, strongest positive field |
| B32 | record (no longer printed) | c = 0.70/3.57 | logits of B31 |
| B33 | Sec. 4, App. B | [0.1994, 0.1997] | checks/reanalysis.py: three alternative estimators |
| B34 | App. B | 0.000944 | checks/reanalysis.py: intercept of the pooled logistic fit |
| B35 | App. B | 2.108e-8 | results/hopfield/sampler_checks.json G4: max \|exact - weak-field q_eq\| over the eight fields |
| B36 | App. B | [0.85, 1.05] | results/hopfield/overlap_run.json: SE of the 256 run means / reported SE, per set |
| B37 | Table 4 caption | 32,467-34,178 reheated (33,331 at G = 0); ties 101 initial, 0 final | results/hopfield/field_grid.json |
| B04 | Table 1, p=1 | 0.127 | pattern_count_p1.json: pooled c = logit(q)/logit(q_eq); bootstrap over the 12 pattern sets (20,000 resamples, seed 2026) |
| B04i | Table 1, p=1 | [0.124, 0.131] | pattern_count_p1.json: pooled c = logit(q)/logit(q_eq); bootstrap over the 12 pattern sets (20,000 resamples, seed 2026) |
| B05 | Table 1, p=2 | 0.188 | pattern_count_p2.json: pooled c = logit(q)/logit(q_eq); bootstrap over the 12 pattern sets (20,000 resamples, seed 2026) |
| B05i | Table 1, p=2 | [0.184, 0.191] | pattern_count_p2.json: pooled c = logit(q)/logit(q_eq); bootstrap over the 12 pattern sets (20,000 resamples, seed 2026) |
| B06 | Table 1, p=3 | 0.201 | pattern_count_p3.json: pooled c = logit(q)/logit(q_eq); bootstrap over the 12 pattern sets (20,000 resamples, seed 2026) |
| B06i | Table 1, p=3 | [0.194, 0.210] | pattern_count_p3.json: pooled c = logit(q)/logit(q_eq); bootstrap over the 12 pattern sets (20,000 resamples, seed 2026) |
| B07 | Table 1, p=4 | 0.211 | pattern_count_p4.json: pooled c = logit(q)/logit(q_eq); bootstrap over the 12 pattern sets (20,000 resamples, seed 2026) |
| B07i | Table 1, p=4 | [0.191, 0.239] | pattern_count_p4.json: pooled c = logit(q)/logit(q_eq); bootstrap over the 12 pattern sets (20,000 resamples, seed 2026) |
| B08 | Sec. 1, Sec. 5 | 11 | largest \|c_meas / (c_CW x A_p) - 1\| over p = 1..4, percent |
| B08a | Abstract | within 11% | largest \|c_meas / c_pred - 1\| over all Table 1 rows (pattern counts and sizes), percent |
| B09 | Sec. 1 | closely matches three-pattern runs | c_meas(p=3) / (c_CW x A_3) - 1 |
| B10m1024 | Table 1 | 0.263 | size_scan_p3.json N=1024: logit q / logit q_eq(N) |
| B10p1024 | Table 1 | 0.284 | exact_protocol_sizes.json: c_CW(1024) x A_3 |
| B10m4096 | Table 1 | 0.160 | size_scan_p3.json N=4096: logit q / logit q_eq(N) |
| B10p4096 | Table 1 | 0.150 | exact_protocol_sizes.json: c_CW(4096) x A_3 |
| B11_512 | Table 1 caption, App. B | 0.227 | exact_protocol_sizes.json: c_CW at N=512 |
| B11_1024 | Table 1 caption, App. B | 0.171 | exact_protocol_sizes.json: c_CW at N=1024 |
| B11_2048 | Table 1 caption, App. B | 0.120 | exact_protocol_sizes.json: c_CW at N=2048 |
| B11_4096 | Table 1 caption, App. B | 0.090 | exact_protocol_sizes.json: c_CW at N=4096 |
| B12m | Sec. 5 | -0.36 | log(c_meas(4096)/c_meas(1024)) / log 4 |
| B12p | Sec. 5 | -0.46 | log(c_pred(4096)/c_pred(1024)) / log 4 |
| B13_512 | App. B | 1.065 | sampler_clock.json: c_meas(p=1, N=512) / c_CW(N) |
| B13_1024 | Sec. 5, App. B | 1.126 | sampler_clock.json: c_meas(p=1, N=1024) / c_CW(N) |
| B13_2048 | Sec. 5, App. B | 1.063 | sampler_clock.json: c_meas(p=1, N=2048) / c_CW(N) |
| B13_4096 | Sec. 5, App. B | 1.020 | sampler_clock.json: c_meas(p=1, N=4096) / c_CW(N) |
| B14 | Sec. 5 | 0.212 | c_CW x A_3 x r(2048) |
| B15_1024 | App. B | 0.319 | c_pred(1024) x r(1024) |
| B15_4096 | App. B | 0.153 | c_pred(4096) x r(4096) |
| B16_1024 | Sec. 5 | 18 | sampler_clock.json: \|c_meas / (c_pred x r) - 1\| at N=1024, percent |
| B16_4096 | Sec. 5 | 4 | sampler_clock.json: \|c_meas / (c_pred x r) - 1\| at N=4096, percent |
| B17 | App. C | 0.343384 | size_scan_p1.json['1024'].logit: pooled one-pattern logit at N=1024 |
| B18 | App. C | 0.19212 | 2 x B17 / logit q_eq (A07) |
| B19 | App. C | 0.18783 | pattern_count_p2.json, as B05 |
| B20 | App. C | blocks of 980-1068 spins | pattern sets s = 3..14 at N=2048: counts of xi^1 xi^2 = +1 and -1 |
| B21 | Table 3 caption | 0.098978 | cw_scaling.json: N=2048, scale 0.25 |
| B22 | Table 3 caption | 0.146625 | cw_scaling.json: N=2048, scale 4 |
| B23 | Sec. 4, Table 3 caption | 1.4814 | B22 / B21 |
| B24 | Table 3 caption | [0.119498, 0.119808] | exact_protocol.json: c_CW over the eight fields |
| B25 | Sec. 4, Table 4, Table 5 | eight nonzero fields; twenty pattern sets; 2000 initial runs per set and field | results/hopfield/field_grid_analysis.json, field_grid.json |
| B26 | Sec. 4, App. B | ten temperatures 0.6 to 1.6; reheat 30 sweeps at beta = 0.7; N=2048, p=3, beta_f=1.6 | src/hopfield/protocol.py |
| B27 | Fig. 1 caption, Table 5 | fourteen paths at N = 1024 | results/figures/mechanism.json |
| B28 | docs/mnist-illustration.md | reheated runs end as 3s in 20-24 of 24 over ten seeds | dbm_control_arm.json, reheated arm (src/experiments/dbm_control_arm.py) |
| B29 | docs/mnist-illustration.md | random-pixel starts end as 3s in 21-24 of 24 (both controls) | dbm_control_arm.json, fresh and fresh_matched arms |
| B30 | docs/mnist-illustration.md | 24 runs; reheat at beta = 0.55 | results/experiments/dbm_reheat.npz |

## Reruns (c)

| id | Location | Printed | How it is checked |
|---|---|---|---|
| C01 | Table 1 | measured c for p = 1..4 at N = 2048, and p = 3 at N = 1024, 4096 | rerun: `python -m src.experiments.hopfield_runs draws P (P = 1..4; 10-21 min GPU each); ... draws 3 1024,4096 (about 48 min GPU)`. needs a GPU; the saved counts reproduce Table 1 in B04-B07i and B10 |
| C02 | Sec. 4, Fig. 2b, Table 4 | the field grid (eight fields, twenty pattern sets) | rerun: `python -m src.hopfield static \| pre \| grid \| analyse (about 37 min GPU for grid)`. needs a GPU; the saved per-set counts reproduce B01-B03 and Table 4 |
| C03 | Sec. 5, App. B, App. C | one-pattern runs at N = 512..4096 | rerun: `python -m src.experiments.hopfield_runs size-p1 N (about 24 min GPU at N = 1024)`. needs a GPU; the saved counts reproduce B13 and B17 |
| C04 | docs/mnist-illustration.md | the 24-run MNIST ensemble of the figure | rerun: `python -m src.experiments.dbm_reheat (needs raw MNIST)`. src.dbm re-implements the original sampler, so a rerun reproduces the ensemble statistically, not byte for byte; the saved run is checked in B30 |

## Checked by `checks.paper_consistency`

These printed values are compared directly with saved results or recomputed from the exact chain:

- Table 1: all six rows and intervals (`draw_level_analysis.json`, `size_scan_p3_analysis.json`).
- Table 2: all nine entries (`cw_linear_sweep.json`); Fig. 1b, a = 0.208.
- Table 3: the ten temperatures and dwells (834 sweeps), against `src/hopfield/protocol.py` and `visual/data/schedule.json`.
- Table 4: every count, both q columns, the G = 0 Wilson interval and the four paired-sum intervals (`results/hopfield/`).
- Table 5: every seed rule and run count.
- App. A: switching times 10^7.7, 10^22.9, 10^137.8 sweeps and 5×10^7 at β = 1.1; zero-magnetization mass below 10^-36; inverse gaps 42.2, 59.0, 82.7; the p = 2 arithmetic of App. C.
- The compiled PDF: six main-text pages, appendix from page 8, references on page 7, no unresolved reference, all fonts embedded, authors in `paper/main.tex` order.

## Cited, not computed

The selection law q = Φ(G√N(π/γ)^¼) (Eq. 3; Kondepudi and Nelson 1983, Kondepudi and Gao 1987).
