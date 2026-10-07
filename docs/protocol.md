# Experimental protocol

This page collects the full protocol of the paper's experiments in one place: the model and update rule, the
cooling schedule, the reheating protocol, the estimators, and every run with its seeds. Section and table
numbers refer to the paper.

## Hopfield model and update rule

- $N$ spins $x_i\in\{-1,+1\}$; $p$ patterns $\xi^\mu\in\{-1,+1\}^N$ with independent equiprobable entries.
- Energy (Eq. 5): $E_G(x)=-\frac1{2N}\sum_{i\ne j}\sum_\mu\xi_i^\mu\xi_j^\mu x_ix_j-G\sum_i\xi_i^1x_i$; probability
  $\propto e^{-\beta E_G(x)}$.
- Heat-bath update: $\Pr(x_i=+1\mid x_{-i})=[1+\tanh(\beta H_i)]/2$, with
  $H_i=N^{-1}\sum_\mu\xi_i^\mu\sum_j\xi_j^\mu x_j-(p/N)x_i+G\xi_i^1$. The term $-(p/N)x_i$ removes self-interaction.
- Permutation sweep: every spin updated once in a fresh random order ($N$ updates).
- Label: each sample is labeled $(\mu,\pm)$ by the pattern with the largest $|m^\mu|$, $m^\mu=N^{-1}\sum_i\xi_i^\mu x_i$,
  and the sign of that overlap. Ties go to the first pattern index.

## Cooling schedule

Ten inverse temperatures with dwells $\lceil 6\tau_{\mathrm{des}}\rceil$ sweeps (Table 3; derivation in
[relaxation-times.md](relaxation-times.md)):

| $\beta$ | 0.60 | 0.80 | 0.90 | 0.95 | 1.00 | 1.05 | 1.10 | 1.20 | 1.40 | 1.60 |
|---|---|---|---|---|---|---|---|---|---|---|
| sweeps | 15 | 30 | 59 | 107 | 355 | 145 | 65 | 31 | 16 | 11 |

834 sweeps in total. The counts are computed at $N=2048$ and reused at every size. Hopfield runs apply them as
permutation sweeps; the exact CW evolution applies them as expected sweeps.

## Reheating after a filtered start

Every Hopfield run starts from independent unbiased spins. The field $G$ is present in all three stages.

1. **Brief cooling**: the schedule at two sweeps per temperature. Runs labeled $(1,+)$ are removed (the
   *filtered start*), at either sign of $G$.
2. **Reheat**: 30 sweeps at $\beta=0.7$.
3. **Final cooling**: the full schedule.

Only final samples labeled pattern 1 with nonzero sign (*retained* samples) enter $q_{\mathrm{chain}}$.

The exact CW mirror (Appendix A) starts from $\mathrm{Binomial}(N,1/2)$ counts, runs two sweeps at each schedule
temperature, removes $k\ge N/2$ (including $m=0$) and renormalizes, reheats for 30 sweeps at $\beta=0.7$, and
applies the schedule. It gives $c_{\mathrm{CW}}=0.120$ at $N=2048$.

## Equilibrium reference (field grid)

For each pattern set $s$: 256 runs start at $+\xi^1$ at $\beta_f=1.6$, $G=0$; 60 equilibration sweeps, then
overlaps recorded for 200 sweeps. Their mean $\bar m^1_s$ gives $\operatorname{logit}q_{\mathrm{eq},s}(G)\approx2\beta_fNG\bar m^1_s$.
The standard error uses the correlation correction $\tau_{\mathrm{ref}}=1.6946$ (reflected CW chain, $N=2048$,
$\beta=1.6$), fixed before the runs. Appendix B gives the error envelope and the pooling weights.

## Estimators

- **Field grid** (Fig. 2b, Sec. 4): per-pattern-set weighted slopes $c_s$ of sampled against reference log-odds,
  weights = inverse squared logit half-width of the 95% Wilson interval; $\widehat c_{\mathrm{grid}}$ = retained-count-weighted
  mean of $c_s$. All eight nonzero fields enter the fit; none was excluded. Interval: 20,000 pattern-set resamples
  with $\bar m^1_s$ held fixed.
- **Pattern-count and size scans** (Table 1): $\widehat c_{\mathrm{scan}}=\operatorname{logit}(\sum_sS_s/\sum_sR_s)/\operatorname{logit}q^{\mathrm{CW}}_{\mathrm{eq}}(N)$,
  with $S_s=n_{+,s}(+G)+n_s(-G)-n_{+,s}(-G)$ and $R_s=n_s(+G)+n_s(-G)$: positive outcomes at $+G$ pooled with
  negative outcomes at $-G$, against the exact CW reference.

## Runs and seeds

Pattern set $s$ is drawn with NumPy `default_rng(20260823 + 1009 s)`. Each Hopfield run uses its own segment of a
seeded PyTorch GPU generator. Bootstraps draw 20,000 pattern-set resamples and keep both field signs of a set
together.

| Experiment | Pattern sets | Runs | Fields | Seeds |
|---|---|---|---|---|
| Field grid (Table 4, Fig. 2b) | $s=0$-$19$, $p=3$, $N=2048$ | 2000 initial runs per set and field | $10^4G=\pm0.875,\pm2.625,\pm4.375,\pm6.125$ and $0$ | dynamics and bootstrap 20260825 |
| Equilibrium reference | $s=0$-$19$ | 256 per set | $G=0$ | dynamics 20260823 |
| Pattern-count scan (Table 1, $N=2048$) | $s=3$-$14$, $p=1$-$4$ | 8000 initial runs per set and field sign | $\lvert G\rvert=0.0006125$ | dynamics $9000+p$; bootstrap 2026 |
| Size scan (Table 1, $p=3$) | $s=3$-$14$, $N=1024, 4096$ | 8000 initial runs per set and field sign | $\lvert G\rvert=0.0006125$ | dynamics $4000+N$; bootstrap 7 |
| One-pattern runs (Sec. 5) | $s=0$, $p=1$, $N=512$-$4096$ | 150,000 initial runs per field sign | $\lvert G\rvert=0.0006125$ | dynamics $31337+N$ |
| CW linear ramps (Table 2) | exact chain, no sampling | $\beta$ piecewise constant at the midpoint of each $\Delta\beta=0.005$ interval, held $0.005/\gamma$ expected sweeps; $0.25/\gamma$ (rounded) Richardson-extrapolated implicit Euler steps of about 0.02 per interval | $G=0.0006125\times2048/N$ | none |
| Fig. 1a paths | drift-plus-noise equation | 14 paths from $m=0$ at $N=1024$; 6 sweeps at each of 360 equally spaced $\beta$ from 0.7 to 1.35; Euler step 0.02 | $G=0$ | NumPy 4 |
| MNIST illustration | see [mnist-illustration.md](mnist-illustration.md) | 24 training "8"s per seed | - | NumPy 0-9 |

The one-pattern runs give $c$ equal to 1.065, 1.126, 1.063 and 1.020 times $c_{\mathrm{CW}}=0.227$, 0.171, 0.120,
0.090 at $N=512$, 1024, 2048, 4096 (Appendix B).
