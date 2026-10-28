# Pre-registration: the simplified Hopfield test

Locked 2026-10-27, before any run of this design. No GPU run of this protocol exists at the time of
writing. The predictions below were computed on CPU and saved before any sampler code was written.

- Constants: `src/simple/design.py`
- Predictions: `python -m src.simple.predict` (CPU, about 5 min) writes `results/simple/predictions.json`
- Estimator: `src/simple/estimate.py`
- Inputs: the simplification audit and the theory check (scratch `simplify/audit.md`, `simplify/theory.md`)

## 1. Questions

1. Does competition among p stored patterns multiply the CW selection log-odds by
   A_p = E max_{mu<=p}|Z_mu| / E|Z| (paper Eq. amplification)? This is the only new claim the Hopfield
   runs test.
2. The selection law itself is not retested here. It is tested exactly on the CW chain (paper Table 2).
   With coupling 1/(N-1), heat-bath updates and state-independent site choice, the p = 1 Hopfield
   sampler is the CW chain exactly, for every pattern (gauge y = xi^1 * x; theory.md Sec. 1). So p = 1
   Hopfield runs test the sampler, not the law. The law is needed beyond CW only as the explanation of
   c_CW, and c_CW is computed exactly.

## 2. The protocol (one protocol for every row)

| item | value | why this and not something else |
|---|---|---|
| model | E(x) = -(1/(2(N-1))) sum_{i!=j} sum_mu xi^mu_i xi^mu_j x_i x_j - G sum_i xi^1_i x_i | coupling 1/(N-1) makes p = 1 identical to paper Eq. (1); with 1/N it is CW at beta(N-1)/N |
| patterns | i.i.d. +-1; set s at size N from `default_rng([20261027, N, s])`, first p of 4 rows (nested in p) | the paper's model; one seed scheme |
| pattern sets | K = 50 per row, s = 0..49 | disorder enters the interval; 50 keeps its share below half the variance at the assumed spread |
| start | every spin independent and unbiased | exact equilibrium at beta = 0 for every model; no burn-in, filter or reheat. Exact CW: same c as Table 2's equilibrium start at 0.7 to 6 digits (0.065874 both) |
| update | one uniformly random spin per update (independent per chain), heat bath: P(x_i=+1) = sigmoid(2 beta h_i), h_i = sum_{j!=i} J_ij x_j + G xi^1_i, float64 | random-site heat bath is the chain the exact code solves; permutation sweeps were the source of the 1.063 one-pattern ratio |
| clock | one sweep = N updates | |
| schedule | one linear ramp, beta_n = 0.7 + 0.9 (n + 1/2)/M for update n = 0..M-1, M = N * 0.9/gamma | the selection law is derived for a linear ramp; the same ramp as paper Table 2 |
| rate | gamma = 0.004 per sweep (225 sweeps) at every N | A_p is derived for N gamma >> p (theory.md 2c); N gamma = 4.1, 8.2, 16.4 here. gamma = 0.001 gives N gamma = 1-4, where the linear-Gaussian model predicts a 5% or larger bias. Also 2x cheaper for the same precision |
| field | +|G| and -|G|, half the runs each, N G = 1.2544 fixed (G = 6.125e-4 at N = 2048) | fixed N G makes the equilibrium log-odds equal at every N (Table 2). Both signs cancel any sampler sign bias to first order; no field grid (proportionality in G is the linear-response regime, a = 0.15, where logit Phi(a) is linear to 0.2%); no G = 0 run |
| sizes and p | N = 2048: p = 1, 2, 3, 4; N = 1024 and 4096: p = 1, 3 | p = 1 at each N is the sampler check; p = 3 across N is the size trend |
| label | at the end of the ramp: mu-hat = argmax_mu |m^mu|, sign of m^{mu-hat}. Runs with a tie in max |m^mu|, or m^1 = 0, are excluded and counted | A_p is a statement about the sign given that pattern 1 wins |
| seeds | GPU stream of row (N, p): 20261027 + 10 N + p | one scheme |

Removed relative to the paper's protocol, each with the evidence that it does not matter or confounds:
the scarce descent and the removal of (1,+) runs, the 30-sweep reheat (exact CW: no-op to 6 digits),
the 10-temperature stepped schedule with ceil(6 tau) dwells (replaced by the ramp the law is derived
for), permutation sweeps (clock mismatch), coupling 1/N, the zero-field Hopfield equilibrium reference
(differs by 0.15%, see 4), the nine-field grid, three estimators, the 12-of-20 pattern sets, and the
seed zoo.

## 3. The one estimator

For each row, pool the runs labelled pattern 1 over all sets and both signs:

    q-hat = (pattern-1 runs whose m^1 has the sign of the field) / (pattern-1 runs)
    c-hat = logit(q-hat) / L_eq,   L_eq = exact CW equilibrium sign log-odds at beta = 1.6

Interval: 95% percentile cluster bootstrap over the 50 pattern sets (10,000 resamples, seed 7; a
resampled set brings both signs). The same estimator and interval serve every row, including p = 1.

The denominator cancels in every test: c-hat / c_pred = logit(q-hat) / (A_p logit q_CW). So no
equilibrium quantity of the Hopfield model enters any pass/fail reading. L_eq only puts c-hat on the
scale of the paper's c. Reading c-hat as a fraction of the Hopfield equilibrium log-odds assumes
L_eq(Hopfield) = L_eq(CW): exact at p = 1; ratio 0.9993 at p = 2 (exact, these 50 sets); 0.996-0.998 at
p = 3, N = 512-1024 (exact enumeration, theory.md 2d), smaller at N = 2048.

## 4. Frozen predictions (exact chains, `results/simple/predictions.json`)

c_CW is the exact law of the sampler's own chain: the CW birth-death chain advanced update by update
(random site, heat bath, beta at the midpoint of the update, independent start), no time step. The
p = 2 value is the exact two-block chain for the 50 frozen pattern sets (blocks of 944-1074 spins,
coupling 2/(N-1); each block's marginal propagated exactly; blocks treated as independent, which a
brute-force 2^10-state check reproduces to 1e-5 relative). For p >= 3 the prediction is A_p c_CW.

| N | p | N gamma | prediction | c_pred | logit q_pred | q_pred | A_p c_sel (law, secondary) |
|---|---|---|---|---|---|---|---|
| 2048 | 1 | 8.2 | c_CW, exact | 0.065874 | 0.2355 | 0.5586 | 0.06557 |
| 2048 | 2 | 8.2 | two-block chain, exact | 0.093634 | 0.3347 | 0.5829 | 0.09273 |
| 2048 | 3 | 8.2 | A_3 c_CW | 0.109508 | 0.3915 | 0.5966 | 0.10900 |
| 2048 | 4 | 8.2 | A_4 c_CW | 0.120929 | 0.4323 | 0.6064 | 0.12037 |
| 1024 | 1 | 4.1 | c_CW, exact | 0.093699 | 0.3349 | 0.5830 | 0.09283 |
| 1024 | 3 | 4.1 | A_3 c_CW | 0.155764 | 0.5567 | 0.6357 | 0.15432 |
| 4096 | 1 | 16.4 | c_CW, exact | 0.046446 | 0.1660 | 0.5414 | 0.04634 |
| 4096 | 3 | 16.4 | A_3 c_CW | 0.077211 | 0.2760 | 0.5686 | 0.07703 |

A_2 = 1.414214, A_3 = 1.662379, A_4 = 1.835764. L_eq = 3.574249 (N = 1024), 3.574685 (2048),
3.574900 (4096). The c_CW values equal paper Table 2 (gamma = 0.004) to the printed digits:
the start law and discrete-versus-continuous time do not matter at this precision.

Side results already fixed by the computation (no run needed): at p = 2 the Gaussian value A_2 c_CW =
0.093160 lies 0.5% below the exact 0.093634; the per-set spread of the exact p = 2 c is 0.06%.

## 5. Run counts and budget (from the precision target, fixed now)

Target: 95% half-width <= 3% of c_pred in every row. Binomial variance of logit q-hat plus the assumed
between-set spread of c (p = 1: 0 by the gauge identity; p = 2: 0.06%, exact; p >= 3: the linear-Gaussian
model of theory.md 2c, 4% at N gamma = 8.2 scaled as (8.2/(N gamma))^0.65 (p/3)^0.5, i.e. 6.3%, 4.0%,
4.6%, 2.6% for (1024,3), (2048,3), (2048,4), (4096,3)). Pattern 1 wins in about 1/p of the runs.

| N | p | runs per set per sign | runs | spin updates | GPU h at 1.5e8 updates/s |
|---|---|---|---|---|---|
| 2048 | 1 | 3,200 | 320,000 | 1.47e11 | 0.27 |
| 2048 | 2 | 3,200 | 320,000 | 1.47e11 | 0.27 |
| 2048 | 3 | 4,100 | 410,000 | 1.89e11 | 0.35 |
| 2048 | 4 | 4,700 | 470,000 | 2.17e11 | 0.40 |
| 1024 | 1 | 1,600 | 160,000 | 3.7e10 | 0.07 |
| 1024 | 3 | 2,700 | 270,000 | 6.2e10 | 0.12 |
| 4096 | 1 | 6,300 | 630,000 | 5.8e11 | 1.08 |
| 4096 | 3 | 7,300 | 730,000 | 6.7e11 | 1.25 |
| total | | | 3.31e6 | 2.06e12 | 3.8 |

Throughput 1.5e8 updates/s is the old GB10 sampler's measured range (0.9-2.3e8 spin updates/s,
results/experiments/size_scan_p1.json and pattern_count_p3.json walls); the budget is 2.5-6.5 GPU h
over that range. The counts are fixed: no top-up and no optional stopping. If the realised half-width
exceeds 3% (because the true between-set spread is larger than assumed), the interval is reported as
it is.

## 6. Gates and readings (stated before running)

Order: G0, then G1 and G2 analysed; the A_p rows are analysed only after G0-G2 pass.

- **G0, code.** The GPU sampler reproduces a NumPy reference bit for bit on a supplied random stream
  (N = 16, p = 3, 50 updates), and the carried projections equal recomputed ones (drift 0) at the end of
  every row.
- **G1, sampler (p = 1, N = 1024, 2048, 4096).** PASS if each 95% interval contains c_CW. This tests the
  whole pipeline against an exact answer; under the old permutation sampler the ratio was 1.03-1.13.
- **G2, multi-pattern code path (p = 2, N = 2048).** PASS if the interval contains 0.093634.
- **Stop rule.** If G0, G1 or G2 fails: no A_p row is interpreted. Find the cause, fix it, rerun all rows
  with new seeds (20261027 + 10 N + p + 1), and record the failure and the fix.

- **T, competition (A_p rows; primary (2048, 3), secondary (2048, 4), (1024, 3), (4096, 3)).**
  Report R = c-hat / (A_p c_CW) with its interval.
  Pre-stated expectation (speculative model, theory.md 2c): R slightly below 1, with the deficit growing
  with p/(N gamma): about -3% at (1024, 3), -1% at (2048, 3), -1.5% at (2048, 4), -0.5% at (4096, 3).
  Readings:
  - (a) Every interval contains 1: competition multiplies the CW log-odds by A_p to within +-3% at
    N gamma >= 4.
  - (b) Some intervals lie below 1 and the deficits order with p/(N gamma): A_p is the N gamma -> infinity
    limit; report the finite-rate deficit with its size.
  - (c) Any interval lies above 1, or the deficits do not order with p/(N gamma): an assumption of the
    Gaussian argument other than common variance fails (independence, or winner = largest early
    amplitude); A_p is not supported as stated.
  Every outcome is reported. The primary row decides the abstract sentence about the Hopfield test.

- **Secondary, no gate.** q-hat at +|G| and at -|G| separately (sign-bias diagnostic); the count of
  excluded ties; c-hat / (A_p c_sel); the size exponent of c-hat at p = 3 over N = 1024-4096,
  log2(c-hat(1024)/c-hat(4096))/2, against the exact CW value 0.506 (c_CW proportional to N^-0.506).

## 7. Not decided by data

No parameter of the protocol is chosen after a run. Not part of this pre-registration: orthogonal
(Hadamard) patterns, other rates gamma, a field grid, reheating experiments. Any of them would be a new,
separately pre-registered arm.

## 8. Amendments (2026-10-27, after the GPU pilot, before any pre-registered row ran)

The pilot (`results/simple/pilot_*.json`) is not part of the experiment and is never pooled with it.

1. **Run counts doubled.** Every row runs `2 x` the count in `results/simple/predictions.json`
   (`design.RUN_FACTOR`). Reason: at p = 3 the pilot showed between-set spread that could push the
   half-width past the 3% target; the measured throughput makes the doubled design about 3 GPU h.
2. **Joint gate level.** G1 (three rows) and G2 (one row) are read at a joint 95% level: each gate row
   uses a 98.75% cluster-bootstrap interval (Bonferroni over four rows, `design.GATE_LEVEL`). Reason: four
   separate 95% intervals fail by chance with probability 1 - 0.95^4 = 0.19. The T rows keep 95%.
