# The two-pattern case: an exact split into two Curie-Weiss blocks

> Not used by the current paper. This page describes the earlier protocol and is kept for the visual
> edition; [protocol.md](protocol.md) describes the protocol the paper uses.

With $p=2$ patterns the Gaussian prediction misses most (paper, Table 1: predicted $c=0.169$, measured
$0.188$ $[0.184, 0.191]$). For $p=2$ an exact rewriting of the energy gives a second estimate of $c$.
The paper states the result in one sentence (Appendix C); this page gives the derivation.

## The split

Measure each spin relative to pattern 1: $y_i=\xi_i^1x_i$. Pattern 1 becomes all positive. Pattern 2 has
entries $\eta_i=\xi_i^1\xi_i^2$, equal to $+1$ where the two patterns agree.

The interaction coefficient of the Hopfield energy (paper, Eq. 5) becomes $(1+\eta_i\eta_j)/N$: it is $2/N$
between two spins with equal $\eta$ and zero otherwise. The model therefore splits exactly into two
Curie-Weiss (CW) blocks that do not interact: $A$ ($\eta_i=+1$) and $B$ ($\eta_i=-1$).

With block sums $S_A$ and $S_B$ of the $y_i$,

$$Nm^1 = S_A + S_B, \qquad Nm^2 = S_A - S_B.$$

If $S_A$ and $S_B$ have the same sign, $|m^1|>|m^2|$ and pattern 1 wins; opposite signs select pattern 2.
A zero block sum gives a tie; we condition on neither sum being zero.

## The odds multiply

Suppose the block signs are independent, with positive probabilities $q_A$ and $q_B$. The agreeing
outcomes have probabilities $q_Aq_B$ and $(1-q_A)(1-q_B)$. Pattern 1 wins only on agreement, so the sign odds
given that pattern 1 wins are their ratio, and

$$\operatorname{logit} q_{\mathrm{chain}} = \operatorname{logit} q_A + \operatorname{logit} q_B.$$

Independence requires that the two blocks start and evolve independently. The filtered start removes runs
labeled $(1,+)$, which excludes the joint positive outcome of the two blocks. This removal can make their signs
dependent, although the energy has no interaction between the blocks.

## The equal-block estimate

Over the twelve scanned pattern sets ($s=3$-$14$, $N=2048$) the blocks have $980$ to $1068$ spins. A block of
$n$ spins matches the paper's CW convention at $\beta_{\mathrm{eff}}=2\beta(n-1)/N$, with field $h=\beta G$
rather than $\beta_{\mathrm{eff}}G$.

For equal blocks with independent, matched dynamics in the selection regime, the log-odds scale as $\sqrt n$.
Two blocks of $N/2$ spins then give the asymptotic amplification $A_2=\sqrt2$ of the paper's Eq. 6.

Each block has about $N/2=1024$ spins. The one-pattern runs at $N=1024$ (paper, Appendix B) pool positive signs
at $+G$ with negative signs at $-G$; the pooled fraction has logit $0.343384$. Doubling it and dividing by the
exact CW equilibrium log-odds at $N=2048$, $\operatorname{logit} q_{\mathrm{eq}}=3.575$, gives

$$c = 2\times0.343384/3.574685 = 0.19212,$$

against the measured $0.18783$ $[0.184, 0.191]$ at $p=2$ and the Gaussian prediction $0.169$.

The estimate uses Hopfield one-pattern runs, so it includes their update rule and their ratio to the CW chain
at $N=1024$ ($1.126$, paper Appendix B). It does not match the actual
block sizes ($980$-$1068$, not $1024$) or the initialization: the $p=2$ runs remove the joint outcome $(1,+)$,
while each one-pattern run removes its own positive outcome. The block sizes and initialization remain unmatched.

## Data and code

- `results/experiments/size_scan_p1.json`, key `"1024"`: the pooled one-pattern logit $0.3433842$.
- `q_eq(0.0006125)` in `src/experiments/exact_protocol.py`: the exact CW $q_{\mathrm{eq}}=0.972740$ at $N=2048$, $\beta=1.6$ (logit $3.574685$).
- Block sizes: draw the patterns with `patterns(seed_of(s), 2048, 2)` (both in `src/hopfield/`) for $s=3,\dots,14$ and count $\eta_i=+1$.
- `results/experiments/pattern_count_p2.json`: the measured $p=2$ row of Table 1.
