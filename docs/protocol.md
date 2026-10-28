# Experimental protocol

The protocol of the paper's experiments in one place. Section and table numbers refer to the paper.
The Hopfield test was pre-registered before any run: [protocol-simple.md](protocol-simple.md) is the
locked pre-registration, with its predictions, gates, readings and two amendments. Constants live in
[`src/simple/design.py`](../src/simple/design.py).

## Exact Curie–Weiss ramps (Sec. 3, Table 2, Fig. 2a)

- The count $k$ of positive spins evolves exactly (Appendix A): no sampling.
- Start: the equilibrium distribution at $\beta=0.7$. End: $\beta=1.6$.
- Ramps: $N=1024, 2048, 4096$ and $\gamma=0.004, 0.001, 0.00025$, with $G=0.0006125\times2048/N$ ($NG$ fixed).
- $\beta$ is piecewise constant at the midpoint of each $\Delta\beta=0.005$ interval, held $0.005/\gamma$
  expected sweeps, with $0.25/\gamma$ (rounded) Richardson-extrapolated implicit Euler steps of about 0.02.
- Writer: `python -m src.experiments.cw_linear_sweep` → `results/experiments/cw_linear_sweep.json`.

## Hopfield test (Sec. 4–5, Table 1, Table 3, Fig. 2b)

| Item | Value |
|---|---|
| Energy | $E_G(x)=-\frac1{2(N-1)}\sum_{i\ne j}\sum_\mu\xi_i^\mu\xi_j^\mu x_ix_j-G\sum_i\xi_i^1x_i$ (Eq. 5); with one pattern this is the CW model |
| Patterns | set $s=0,\dots,49$ at size $N$: the first $p$ of four rows of `default_rng([20261027, N, s])`, so the sets are nested in $p$ |
| Rows | $N=2048$: $p=1,2,3,4$; $N=1024$ and $4096$: $p=1,3$ |
| Start | every spin independent and unbiased |
| Update | one uniformly chosen spin per update, heat bath, double precision; a sweep is exactly $N$ updates |
| Ramp | update $n=0,\dots,M-1$ uses $\beta_n=0.7+0.9(n+\tfrac12)/M$, $M=225N$ ($\gamma=0.004$, 225 sweeps) |
| Field | half the runs $+\lvert G\rvert$, half $-\lvert G\rvert$; $NG=1.2544$ ($\lvert G\rvert=0.0006125$ at $N=2048$) |
| Label | at $\beta=1.6$: the pattern with the largest $\lvert m^\mu\rvert$ and the sign of $m^1$; ties and $m^1=0$ excluded and counted |
| Estimator | $\widehat c=\operatorname{logit}\widehat q/\operatorname{logit}q^{\mathrm{CW}}_{\mathrm{eq}}(N)$, $\widehat q$ = fraction of pattern-1 runs with the field's sign, pooled over sets and signs |
| Interval | 95% percentile cluster bootstrap over the 50 sets, 10,000 resamples, seed 7; the gate rows were read at 98.75% each |
| Predictions | $p=1$: the exact CW chain of this sampler; $p=2$: the exact two-block chain; $p\ge3$: $A_pc_{\mathrm{CW}}$ |
| GPU streams | row $(N,p)$: seed $20261027+10N+p$ |

Run counts are in Table 3 of the paper and in `results/simple/row_*.json` (twice the counts of
`results/simple/predictions.json`, by amendment 1). The equilibrium reference is the exact CW value; the
Hopfield value differs by 0.07% on average at $p=2$ (exact two-block calculation) and is computed exactly
at $p=3$, $N=512$ and $1024$ by `python -m src.simple.eq_reference` (`results/simple/eq_reference_p3.json`).

## Figure 1a

14 paths of the drift-plus-noise equation from $m=0$ at $N=1024$, $G=0$; 6 sweeps at each of 360 equally
spaced $\beta$ from 0.7 to 1.35; Euler step 0.02; NumPy seed 4.

## Earlier protocol

The camera-ready paper of October 2026 used a filtered start, a reheat, a ten-temperature schedule, a
nine-field grid and permutation sweeps (`src/hopfield/`, `results/hopfield/`, `results/experiments/`).
The paper no longer uses them; the reasons are in [protocol-simple.md](protocol-simple.md) Sec. 2. The
files are kept because the visual edition reads them.
