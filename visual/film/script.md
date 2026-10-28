# Narration script — "A Finite-Rate Selection Law for Annealed Gibbs Sampling", the film

The video edition of `../index.html`, grounded in the paper. Nine scenes in the
paper's order: the problem, the Curie–Weiss model, where the sign is chosen, how much preference
survives, the selection formula, the scaling, competing patterns, how the test runs start, the
test, and what stays open. 3Blue1Brown's look and motion (dark ground, manim, LaTeX that writes
itself on, objects that transform into each other). The paper
(`../../paper/main.tex`) is binding for content; where this script and the paper disagree,
the paper wins.

## Conventions (binding for the parser, TTS and `sync_check.py`)

- A scene starts at `## scene: <id>`. The next lines give `title:` and `chapter:` (a visual-edition
  chapter id: overview, commit, odds, race, scaling, modes, reheat, test, limits; the page's
  `slowdown` chapter has no clip).
- A beat starts at `### beat: <name>`. `**Visual.**` is the direction for the animator (not spoken).
- The spoken text is every line that starts with `> ` inside the scene, in order, joined by single
  spaces. Paragraph breaks inside a scene are a `>` line on its own (spoken as a sentence break).
- `[[cue:<name>]]` marks the word where an animation event must land: the event starts at the
  first character of the word that follows the marker. Cue names are unique inside a scene.
- `[[pause:<seconds>]]` is a deliberate pause. The TTS helper sends it as
  `<break time="<seconds>s" />`; `sync_check.py` allows a silence at that point up to
  `<seconds> + 0.6` s.
- Both markers are stripped from the text before TTS (pauses become break tags).
- Numbers and symbols are written the way they are said. On screen they appear as numerals and
  LaTeX (the `**Visual.**` lines give the exact on-screen form), at the moment they are said.

## Palette and words (fixed meanings; constants in `style.py`)

| meaning | colour | word used, never a synonym |
|---|---|---|
| positive mode, m > 0, favoured | `MODE_A` `#7EA3F0` | the positive mode |
| negative mode, m < 0 | `MODE_B` `#EE8A50` | the negative mode |
| the field (the model's preference) | `BIAS` `#B3C7F6` | field, G |
| random fluctuations, the selection interval | `NOISE` `#6F7177` | fluctuations, selection interval |
| c only | `GILT` `#D9B350` (text `#E4C372`) | c |
| theory and prediction, always dashed | `THEORY` `#C9C8C3` | formula, prediction |
| measurement, solid | `INK` `#E8E7E2` | measured |
| ground | `BG` `#111214` | — |

Fixed terms: spin, configuration, Curie–Weiss model, inverse temperature β, cooling, sweep,
magnetization m, mode, field G, log-odds, c, the exact CW chain, selection interval,
overlap, pattern, reheating. No "noise level", "denoise", "commit point", "speciation", "bias push",
"kept fraction", "clock".

Visual rules for every scene: at most one line of prose on screen besides labels and LaTeX; a
setting (N, γ, seeds, run counts) is a small grey tag, never a card; every illustration that is not
data carries a grey tag "illustration"; gold is only c.

## Sources for every spoken or shown number

"paper" is the paper (`../../paper/main.tex`); a "derived" number gives its
arithmetic in the row.

| number | source |
|---|---|
| 97.3% (q_eq), 60.5% (q_chain), log-odds 3.57 and 0.43, grids 97 vs 61 of 100 | `../data/c_cw.json` (0.97274, 0.60546); derived: log-odds = logit q, their ratio = printed c_CW |
| c_CW = 0.120 (0.119808), N = 2,048, G = 0.0006125, computed through removal, reheating, cooling | paper §4; App. A |
| c_CW varies < 0.3% over the eight fields (0.119498–0.119808) | paper Table 3 caption |
| ten inverse temperatures 0.6–1.6; sweeps 15, 30, 59, 107, 355, 145, 65, 31, 16, 11 = 834; ⌈6 τ_des⌉ at N = 2,048 | paper Table 3 |
| final cooling ×¼ → ×4 (sixteen times), every temperature of the 834-sweep final cooling; brief first cooling and reheating unchanged: c 0.098978 → 0.146625, ×1.48 | paper §4, Table 3 caption; `src/experiments/exact_protocol.py` (`K * scale` at every temperature); `results/experiments/cw_scaling.json` |
| hook: same change in grid units, about 59 → 63 positive of 100 (q 0.5875 → 0.6281); the paper's schedule (×1) is the 61 tick; closing "about four positive samples in a hundred" = 62.81 − 58.75 | derived: `results/experiments/cw_scaling.json` (scale 0.25 and 4.0) |
| odds 36:1 → 1.5:1 (0.97274/0.02726 = 35.7; 0.60546/0.39454 = 1.53 = 35.7^0.1198) | derived from `../data/c_cw.json`; paper §2 reading of c (4:1 → 1.32:1) |
| drift (β − 1)m + G; fluctuation variance 2/N per expected sweep; T ~ γ^(−1/2); field ~ GT; fluctuations ~ √(T/N); ratio G√N γ^(−1/4) | paper §3 |
| q_sel ≈ Φ(a), a = G√N (π/γ)^(1/4) | paper Eq. 3; Kondepudi & Nelson 1983; Kondepudi & Gao 1987 |
| a = 0.208 at N = 2,048, γ = 0.001, G = 0.0006125; Φ(a) = 0.582 | paper App. A (Fig. 1b); Φ(0.208) = 0.582 derived |
| Fig. 1a paths: no field, N = 1,024, start m = 0, 6 sweeps at each of 360 temperatures 0.7–1.35, seed 4 | paper App. A; N from `paper/figures/make_figures.py`; `../data/fork_trajectories.json` must be regenerated to this setting or tagged with its own |
| accumulated growth γt²/2, appreciable at t ~ γ^(−1/2) | paper §3 |
| c ∝ (N²γ)^(−1/4); sixteen times slower doubles c | paper Eq. 4 |
| N = 1,024 ramp: 0.094 (γ = 0.004) → 0.189 (γ = 0.00025); nine ramps within 1.82% (spoken "within two percent") | paper Table 2, Fig. 2a |
| A_3 = 1.662; 0.120 × 1.662 = 0.199 | paper Eq. 6 |
| brief cooling 2 sweeps per temperature; remove (1,+) at either field sign; reheat 30 sweeps at β = 0.7 | paper §4, App. B |
| zero field 0.49554, Wilson [0.48630, 0.50480] (screen only) | paper App. B |
| CW reheating memory e^(−30/τ_full) ≈ 1.2×10⁻⁴ ("about one part in ten thousand"), CW only, does not bound Hopfield memory | paper Table 4 caption |
| test runs start from the removal and reheating protocol (the test of "can reheating restore the proportions?") | paper §4, "Reheating after a filtered start"; App. B |
| MNIST: reheated digit-8 runs, a digit-3 bias, ten seeds (picture only; no tallies spoken or shown) | docs/mnist-illustration.md |
| 66.9% vs reference 97.3% at G = 0.0006125 (n = 11,085, n₊ = 7,417) | paper §4, Table 4 |
| predicted 67.1% at that field: logistic(0.199 × logit 0.97274) = 0.6707 | derived |
| c = 0.199 [0.191, 0.210]; twenty pattern sets; eight fields; 2,000 runs per set and field | paper §4, App. B |
| one-pattern ratio 1.063 → 0.212 outside | paper §5 |
| one pattern = Curie–Weiss after flipping the spins where the pattern is −1 | derived: substitute y_i = ξ_i x_i in the paper's Hopfield energy (p = 1); the remaining difference is 1/N vs 1/(N−1), paper §5 |
| interval excludes reference uncertainty and the 1% allowance | paper Fig. 2 caption, App. B |
| Table 1 rows, "misses by up to 11%" | paper §5, Table 1 |
| N^(−0.36) measured vs N^(−0.46) stepped prediction | paper §5 |

---

## scene: hook
title: The wrong proportions
chapter: overview

### beat: digits
**Visual.** Dark ground. Two 28×28 digit tiles fade in side by side: a "3" in the
positive blue, an "8" in the negative orange. Between them, the LaTeX label `2 : 1` writes on at
"twice". Below, a row of twelve small tiles labelled "correct sampler" fills in at "should": eight
3s and four 8s. A second row, "a sampler", fills in with the same mix; each row ends in a small
ratio label `2 : 1`. At "still get that ratio wrong", two 3s of the second row become 8s and its
label becomes `1 : 1`; every tile stays a clean digit. Grey tag bottom-left: "a picture of the
question, not data".

> Suppose a model of handwritten digits makes a three [[cue:twice]] twice as likely as an eight. A correct sampler [[cue:should]] draws about twice as many threes. [[cue:wrong-ratio]] A sampler can draw clear threes and clear eights, and still get that ratio wrong.

### beat: grids
**Visual.** The digit tiles give way to two large squares, blue "positive" and orange
"negative". Grey tag bottom-left: "Curie–Weiss model · N = 2,048 · proportions computed exactly".
At "cools", the squares move left and a temperature plot draws on the right: the paper's ten-step
schedule as a staircase (y "temperature", "hot" at the top; x "sampling time"; each step as wide as
its share of the time); a dot rides it and the grey word "cooling" appears. At "ninety-seven" the
two squares become the 10×10 "model" grid, 97 blue and 3 orange, "97 of 100" under it. At
"sampler", the plot moves, axes and all, into a band under the grids in two stages (first down
under the right half, then left along the bottom, so it never crosses a grid or a number; the band
keeps only "hot"), the tag fades, and the "sampler" grid fills cell by cell to 61 blue and 39
orange; "61 of 100" writes on.

> [[cue:exact-model]] The paper measures this error in a model with two outcomes, positive and negative, whose correct proportions are known exactly. [[cue:cools]] Its sampler starts hot and lowers the temperature step by step: cooling. [[cue:model-grid]] Of a hundred samples, ninety-seven should be positive; [[cue:sampler-grid]] the sampler delivers sixty-one. [[pause:0.6]]

### beat: belief
**Visual.** A solid grey arrow labelled "cool more slowly" points from the sampler
grid to the model grid. At "a quarter", a tick marks the paper's length on the band's time axis
("paper's"), every step of the staircase shrinks to a quarter of its width ("×¼"), two sampler cells
turn orange and `59` (×¼) appears beside `61` (×1, the
band's "paper's" tick). At "With four times", the
quarter staircase stays as a faint ghost and every step stretches to four times the paper's width
("×4"); at "sixteen times" the label becomes "×4 = 16× the time of ×¼". At "sixty-three" four cells
turn blue and `63` (×4) appears. The row 59 · 61 · 63 sits far short of the model's 97.

> [[cue:slower]] The natural remedy is to cool more slowly. [[cue:quarter]] With a quarter of the paper's sampling time at every cooling step, the sampler delivers about fifty-nine. [[cue:four]] With four times the paper's time, sixteen times the first, it delivers about [[cue:sixty-three]] sixty-three.

### beat: question
**Visual.** The band and the arrow fade; the sampler grid returns to 61, and "the
model's preference" appears under the model's 97. Copies of the model's 36 blue cells that the
sampler gets wrong lift off as one block, move across onto the sampler's cells 61–96 and turn orange
only there; an outline of exactly those 36 cells is labelled "36 lost". At "where", the grids rise and shrink first, then the title writes on: "A Finite-Rate
Selection Law for Annealed Gibbs Sampling", with "DynaFront @ NeurIPS 2026" under it in grey. At
"how much", one grey route line fades in under the title: "where the choice is made · how much
survives · why slower barely helps · competing patterns · the test".

> [[cue:question]] So two things need explaining: [[cue:title]] where the preference is lost, and [[cue:route]] how much of it survives.

---

## scene: commit
title: Where the choice is made
chapter: commit

### beat: spins
**Visual.** The sampler grid from the hook dissolves into a row of 64 small squares, each light
(+1) or dark (−1). Label "N spins, each ±1" writes on. Grey tag "64 shown". At "configuration",
a bracket spans the whole row with the grey word "configuration".

> [[cue:spins]] The model is a row of N spins, each plus or minus one. [[cue:config]] One setting of all of them is a configuration.

### beat: couplings
**Visual.** The squares arrange on a circle (16 shown); faint grey lines join every pair, then fade
to a hint. "Curie–Weiss model" writes on in grey. A few configurations flash by: the mostly-agreeing
ones glow brighter than the mixed ones.

> [[cue:couplings]] In the Curie–Weiss model, every spin favours agreeing with every other, [[cue:agree]] so mostly-agreeing configurations are more probable.

### beat: beta
**Visual.** A horizontal dial labelled β, ticks at 0.6, 1, 1.6. Above it a small bar chart of
probabilities over a handful of configurations: flat when the needle is at the left, peaked on
the agreeing configurations when it moves right. Labels "broad" and "concentrated" in grey at the
two ends. At "Cooling", an arrow along the dial points right with the word "cooling".

> [[cue:beta]] The inverse temperature, beta, sets how strongly agreement is favoured, like softmax sharpness: [[cue:beta-low]] small beta gives a broad distribution, [[cue:beta-high]] large beta a concentrated one. [[cue:cooling]] Cooling is raising beta.

### beat: m
**Visual.** The row of squares collapses into a horizontal number line from −1 to +1; the dot that
lands is their average. LaTeX writes on: `m = \frac{1}{N}\sum_i x_i`. The right half of the line
tints blue at "positive", the left half orange at "negative".

> [[cue:m-meter]] The average spin is the magnetization m: [[cue:pos]] positive for mostly plus ones, [[cue:neg]] negative for mostly minus.

### beat: landscape
**Visual.** Above the dial, a curve over m (x from −1 to 1; grey y label "−log probability"). At
β = 0.6 one smooth valley at m = 0. As the needle moves past 1 the curve morphs, without a cut,
into two valleys; the right valley tints blue, the left orange, at the cue "the two modes". Grey
tag "no field".

> [[cue:landscape]] Plot minus the log-probability of m, so likely values sit in valleys. [[cue:one-well]] At small beta there is one valley, at zero; [[cue:two-wells]] past beta equal to one it splits into [[cue:modes]] the two modes.

### beat: gibbs
**Visual.** Back to the row of 64 squares: a highlight hops to one square, which flickers and
redraws; then another; the hops speed up. A counter "N updates = 1 sweep" writes on. The β needle
ticks rightward in small steps while the hops continue.

> [[cue:gibbs]] Annealed Gibbs sampling redraws one spin at a time from its probability given all the others. [[cue:sweep]] N such updates make one sweep, the unit of sampling time. [[cue:anneal]] Step by step, beta rises.

### beat: fork
**Visual.** The landscape slides up and fades; its two valley positions become the ends of two
faint dashed light-grey branches of a plot with β on x (0.7 to 1.35) and m on y. Fourteen grey
paths draw left to right from m = 0 at the cue "runs"; near β = 1 each takes the colour of the
side it ends on, at "falls". Grey tag bottom-left: "simulated magnetization · no field · N = 1,024 · illustration".

> [[cue:fork]] Here are several runs. [[cue:runs]] Each wanders near zero; [[cue:split]] near beta equal to one, a small random imbalance grows, [[cue:falls]] and the run falls into one mode.

### beat: locked
**Visual.** One coloured path is nudged toward the other branch; a tall grey barrier rises between
the branches and the nudge fails. A small vertical band around β = 1 shades grey.

> [[cue:locked]] Once the modes form, switching between them can take far longer than settling in one. [[cue:decides]] So the sign chosen near beta equal to one can decide the final mode.

---

## scene: odds
title: How much preference survives
chapter: odds

### beat: field
**Visual.** The two-valley curve returns. A row of small arrows in the field colour presses every
spin toward +1; at the cue "tilts" the curve tilts so the blue valley is deeper. LaTeX `G` writes
on beside the arrows in the field colour at "strength".

> [[cue:field]] Now give the model a preference: a small bias toward plus one on every spin, a field [[cue:G]] of strength G. [[cue:tilts]] It tilts the landscape toward the positive valley.

### beat: schedule
**Visual.** The paper's schedule as a staircase: ten steps at the true, unequally spaced inverse temperatures
β = .60, .80, .90, .95, 1.00, 1.05, 1.10, 1.20, 1.40, 1.60, each step's width proportional to its
sweeps (15, 30, 59, 107, 355, 145, 65, 31, 16, 11), the β = 1 step widest and highlighted at the
cue "with the most sweeps". Grey tag: "834 sweeps · sweeps per temperature = 6 × relaxation time, rounded up · a
design choice".

> [[cue:schedule]] The paper's schedule cools in ten steps, [[cue:longest]] with the most sweeps at beta equal to one, where the distribution takes longest to settle.

### beat: counts
**Visual.** The staircase shrinks to the top. The row of spins from the commit scene returns
and sorts itself: plus ones slide right, minus ones left; a brace "k = number of positive spins"
writes on. At "track" the sorted row shrinks into one tick at k = N/2 on a new k axis and a
histogram with N + 1 bars over k grows from that tick (one bump); grey LaTeX
`N+1` probabilities, `\text{instead of}\ 2^N` at "instead". At "watch" a playhead runs along the
shrunken staircase while the histogram reshapes step by step (the bump splits into a blue and an
orange spike) and a blue "positive: x%" share (conditioned on m ≠ 0) climbs and freezes at 60.5%.
At "model's" the model's own equilibrium histogram at β = 1.6 draws on top as a dashed light-grey
outline, almost all of it in the positive spike; it collapses into the model's grid (left) with
`97.3\%` at "ninety-seven"; at "sixty" the sampler's histogram pours into the sampler's grid
(right) with `60.5\%`. Grey tag: "N = 2,048 · G = 0.0006125 · computed exactly from the
probability of each count k".

> [[cue:count]] In this model, every arrangement with the same number k of positive spins is equally likely, and the sampler treats them alike. [[cue:track]] So we can track N plus one probabilities, one for each k, [[cue:exact]] instead of two to the N. [[cue:watch]] Watch the positive share: it moves while the bump splits in two, then it freezes. [[cue:q-eq]] The model's own distribution is ninety-seven point three percent positive; [[cue:q-chain]] the sampler ends at sixty point five, with no counting noise.

### beat: logodds
**Visual.** The percentages dim. LaTeX writes on: `\log\frac{q}{1-q}`, with a vertical axis beside
it marked "0 = equal modes". Each grid collapses into a horizontal blue bar: the model bar grows to
3.57, the sampler bar to 0.43, each number writing on at its cue. At "share cannot double", grey
LaTeX `2\times 97.3\% = 194.6\%\,?` writes on and is struck out. At "exponent" the tilted
landscape from the field beat returns small beside the bars with its arrows and `G`; at
"doubling", `G` becomes `2G`, the tilt doubles, and both bars stretch to twice their length
together (grey tag "weak field · illustration"); at "ratio", the fraction
`\frac{2\times0.43}{2\times3.57} = \frac{0.43}{3.57}` writes on, then both bars return to 3.57
and 0.43.

> [[cue:logit]] Call q the positive share. Measure each share in log-odds: the log of q over one minus q. [[cue:bar-model]] The model has three point five seven; [[cue:bar-chain]] the sampler, zero point four three. [[cue:why-logit]] A share cannot double past one hundred percent; log-odds can. [[cue:exponent]] The field sits in the exponent of the model, so for a weak field the log-odds grow in proportion to G. [[cue:doubling]] Doubling G roughly doubles both, [[cue:ratio-fixed]] so their ratio stays the same.

### beat: c
**Visual.** The left part of the fraction fades; `\frac{0.43}{3.57}` stays and gains `c =`. The
short bar lifts onto the start of the long bar and turns gold: 12% of the long bar; faint outlines
of further copies tick along the long bar (about eight fit). Then the ratio becomes `c = 0.120` in
gold. A dial from 0 ("equal modes") to 1
("the model's proportions") appears, its gold needle at 0.120. At "power", LaTeX writes on under
the dial: `36:1 \;\to\; 36^{\,c} : 1 \approx 1.5:1`, the exponent `c` in gold.

> [[cue:ratio]] That ratio, [[cue:c]] zero point one two zero, is called c: the fraction of the log-odds that survives. [[pause:0.8]] [[cue:power]] Equivalently, the sampler's odds are the model's odds to the power c: [[cue:odds-example]] about thirty-six to one becomes about one and a half to one.

### beat: flat
**Visual.** Beside the dial, a small plot of c against the field G: four gold dots on each side of
zero (four strengths, both directions), on a thin gold line at 0.120, with "8 fields: c varies by
< 0.3%". At "answer" the dial's needle, its `c = 0.120` and the flat line pulse together.

> [[cue:flat]] And c barely changes across the paper's eight fields. [[cue:answer]] So at these weak fields, c does not depend on the field: on this schedule, about an eighth of the model's log-odds survives.

---

## scene: race
title: Field against fluctuations
chapter: race

### beat: forces
**Visual.** The fork plot from the commit scene returns in grey, with the grey band around β = 1;
the gold `c = 0.120` sits small in a corner (cue "where"). Then zoom into the band. A dot on the
m axis near zero. Two arrows act on it: at "Below", a restoring arrow pointing back to zero
(labelled `\beta < 1`); at "above", the same arrow flips to point away from zero (labelled
`\beta > 1`). At "field", a short constant arrow in the field colour points toward +m. At "so m
drifts", LaTeX writes on: `\text{drift} \approx (\beta-1)\,m + G`. At "Random", the dot starts
to jitter and a grey LaTeX `\text{variance} \approx 2/N \text{ per sweep}` writes on.

> [[cue:where]] To see where the preference is lost, zoom in near beta equal to one. [[cue:below]] Below beta equal to one, an imbalance in m shrinks; [[cue:above]] above one, it grows. [[cue:field]] The field adds a steady push, [[cue:drift]] so m drifts at beta minus one, times m, plus G. [[cue:fluct]] Random updates add fluctuations of variance two over N per sweep.

### beat: interval
**Visual.** The shrunken staircase from the odds scene appears in a small inset (β against
sweeps); at "To calculate" a dashed light-grey line of slope γ lays itself over the steps (label
`\gamma` on a slope triangle; grey tag "simplification"). Then a second inset: the growth rate
`\beta - 1 = \gamma t` as a rising line from t = 0, where β crosses one; the t axis writes on at
"t, the sweeps". At "So it is multiplied", the triangle under the line shades grey, `\gamma t^2/2` writes on
inside it, and `\times\, e^{\gamma t^2/2}` writes on beside the triangle. At "about one", a
horizontal mark at area 1 meets the time axis, and on the fork plot the grey band becomes the
selection interval with a brace on the sweeps axis: `T \sim \gamma^{-1/2}`. At "still open",
the fork paths inside the band stay close together around zero; at "runs away", they leave the
band toward the two branches.

> [[cue:ramp]] To calculate, the paper takes the simplest schedule: a ramp of slope gamma. [[cue:rate]] Past one, an imbalance grows at rate beta minus one, which is gamma times [[cue:t]] t, the sweeps since beta crossed one. [[cue:total]] So it is multiplied by e to the area under that line, gamma t squared over two. [[cue:interval]] That area reaches about one after roughly gamma to the minus one half sweeps: [[cue:selection]] the selection interval. [[cue:open]] Before then, the sign is still open; [[cue:runaway]] after it, the imbalance runs away.

### beat: compete
**Visual.** Inside the interval, against time: a straight line in the field colour rising
linearly, with `G\,T \sim G\,\gamma^{-1/2}` writing on; and a grey band widening like a square root,
with `\sqrt{T/N} \sim \gamma^{-1/4}/\sqrt{N}` writing on. A cloud of grey dots spreads around the
line as the band grows. At "Their ratio", the two expressions slide into a fraction that becomes
`G\sqrt{N}\,\gamma^{-1/4}`.

> [[cue:field-grows]] Over that interval, the field's push adds up in proportion to its length, [[cue:fluct-grows]] the fluctuations only like its square root. [[cue:ratio]] Their ratio is G root N, times gamma to the minus one quarter: [[cue:slower]] slower cooling favours the field.

### beat: gaussian
**Visual.** The cloud of dots straightens: their linear growth is divided out (each dot slides
back toward a common time, keeping its side of zero). They pile into a histogram that morphs into
a Gaussian curve. A vertical line at its mean in the field colour ("set by the field") and a grey
brace for its width ("set by the fluctuations"). LaTeX `a = \text{mean} / \text{sd}`. At "positive
area", the area right of zero shades blue, left of zero orange; `\Phi(a)` writes on over the blue
area.

> [[cue:gaussian]] To turn this race into a probability, undo the growth. It stretches every run by the same factor, so each keeps its sign. The runs then form, approximately, a Gaussian: [[cue:mean]] the field sets its mean, [[cue:spread]] the fluctuations its spread. [[cue:a]] Call a the mean over the standard deviation. [[cue:area]] The positive area, Phi of a, is the chance of the positive sign.

### beat: formula
**Visual.** A small grey citation tag writes on: "Kondepudi & Nelson 1983 · Kondepudi & Gao 1987".
Then, dashed light grey: `q_{\text{sel}} \approx \Phi(a), \quad a = G\sqrt{N}\,(\pi/\gamma)^{1/4}`;
the ratio from the compete beat slides into `a`, and the new factor `\pi^{1/4}` flashes at
"constant". The full formula is shown, not spoken.

> [[cue:classical]] This is a classical result; [[cue:no-switch]] with little switching later, Phi of a is the final share.

### beat: example
**Visual.** The Gaussian redraws at a = 0.208 (grey tag: "linear ramp · N = 2,048 · γ = 0.001 ·
G = 0.0006125 · Fig. 1b setting"); `a = 0.208` writes on at "a is"; the blue area writes `58\%` at
"So". At "sixty-one", a small grey tick labelled "the paper's schedule: 61%" sits just right of it. The mean
line sits close to zero; the spread brace is wide.

> [[cue:example]] On a ramp at the opening's size and field, [[cue:a-value]] a is zero point two zero eight. [[cue:share]] So the positive sign wins only fifty-eight percent of the time, near the sixty-one of the paper's schedule. [[cue:answer]] That is where the preference is lost: the weak field only tilts a choice the fluctuations mostly decide.

---

## scene: scaling
title: Why slower barely helps
chapter: scaling

### beat: derive
**Visual.** Opens on `\Phi(a)` lifted from the race scene (the hand-off frame). At "small", the
view zooms on the curve Φ(a) near a = 0: it straightens into a line through ½ (grey tag "near
a = 0"). Under it, two lines write on: "sampler's log-odds `\propto G\sqrt{N}\,\gamma^{-1/4}`" and
"model's log-odds `\propto N G`". At "Divide" they form a fraction; the two G's flash and cancel;
the fraction transforms into `c \propto (N^2\gamma)^{-1/4}` in gold.

> [[cue:weak]] For small a, the positive area grows in a straight line with a. [[cue:sampler-odds]] So the sampler's log-odds go like a, the ratio of field to fluctuations found above. [[cue:model-odds]] The model's log-odds go like N times G, since the field acts on all N spins. [[cue:divide]] Divide, and G cancels, as the flat c showed: [[cue:law]] c goes like N squared gamma, to the minus one quarter.

### beat: ladder
**Visual.** Two vertical rails: "cooling time" on the left (log scale) and "c" on the right. A rung
at 1× on both. The left rail stretches to 16× at "sixteen"; the matching rung on the right climbs
only to 2× at "doubles", with a gold `\times 2`.

> [[cue:sixteen]] So cooling sixteen times more slowly [[cue:doubles]] only doubles c. [[cue:regime]] This holds for a slow ramp starting well below beta equal to one, with little switching later.

### beat: check
**Visual.** The ladder becomes the paper's Fig. 2a: c against γ (log axis), three dashed
light-grey formula lines for N = 1,024, 2,048, 4,096, and nine solid ink symbols from the exact
count calculation. The two N = 1,024 symbols at γ = 0.004 and 0.00025 light up with labels `0.094`
and `0.189`, with a grey tag "N = 1,024 · 16× slower". At "within two percent", a grey tag writes on: "all nine within 2%
(largest gap 1.82%)". While the N = 1,024 pair is lit, the other rows dim and their N labels leave.

> [[cue:check]] The exact chain agrees. [[cue:check-values]] At the smallest size, sixteen times slower takes c from zero point zero nine four to zero point one eight nine. [[cue:max]] Over all nine ramps, the exact chain lies slightly above the full formula, within two percent.

### beat: not-forever
**Visual.** The Fig. 2a plot widens to the right, past an axis break after γ = 0.00025 (no numbers
beyond it). The N = 1,024 formula line keeps climbing and crosses a dashed light-grey reference
line "the model's proportions: c = 1", fading above it; the other two lines fade out. The ramp
rate and the final-temperature wait are two knobs, drawn apart: at "long enough sampling", a
vertical arrow rises from the slowest N = 1,024 point straight to c = 1, labelled "then wait at
β = 1.6, very long". Grey tags "fixed N · illustration, not a measurement" and "the law: not for
arbitrarily slow cooling at fixed N".

> [[cue:not-forever]] The law does not hold for arbitrarily slow cooling at fixed size: [[cue:equilibrium]] long enough sampling at the final temperature recovers the model's proportions.

### beat: staircase
**Visual.** The staircase from the odds scene returns at ×¼ of its 834 sweeps, beside the hook's
grid at 59 positive of 100 and the gold dial at `c = 0.099` (grey tag "final cooling stretched ·
N = 2,048"). At "sixty-three", one stretch drives all three:
every step widens to ×4 (sixteen times the width on screen), the grid moves to 63 and the dial to
`0.147`; a grey tag "×¼ → ×4 = 16×". At "is c growing" the dial flashes; a gold `\times 1.48`
writes on at "one point four eight". A small dashed mark shows where the ramp law's ×2 would be,
for contrast. At "derived", the dashed straight ramp from the race scene lays over the staircase
and visibly does not fit it.

> [[cue:staircase]] On the paper's schedule, the opening's fifty-nine to sixty-three [[cue:stair-c]] is c growing only [[cue:stair-result]] one point four eight times. [[pause:0.8]] [[cue:caveat]] The law was derived for a ramp; the paper does not derive this smaller gain. [[cue:answer]] Either way, slower cooling buys little.

---

## scene: modes
title: Competing patterns
chapter: modes

### beat: hopfield
**Visual.** Three random ±1 patterns as 16×16 tiles write on in a row, labelled "pattern 1, 2,
3"; each gets its negative under it, so six tiles. Pattern 1 is outlined blue, its negative orange.
Label "Hopfield model"; grey tag "far fewer patterns than spins". At "two choices", a small
two-level tree writes on beside the tiles: "which pattern?" then "which sign?". At "Does", the
gold `c = 0.120` from the odds scene appears small with a grey "?" beside it.

> Curie–Weiss has only two modes, so a run makes one choice. [[cue:hopfield]] A Hopfield model stores random patterns of plus and minus ones; [[cue:pattern-modes]] at low temperature, configurations settle near one pattern or its negative. [[cue:p3]] With three patterns, [[cue:two-choices]] a run makes two choices: which pattern, and which sign. [[cue:question]] Does competition change how much preference survives?

### beat: overlap
**Visual.** The m line from the commit scene returns, relabelled "overlap with pattern 1"; pattern
1's tile sits at the +1 end, its negative at the −1 end. Then three such lines stack, one per
pattern. At "labelled", the line with the largest |overlap| lights up and the sample gets a tag
like "(1, +)".

> [[cue:overlap]] A sample's overlap with a pattern is its average agreement with it, [[cue:ends]] from minus one at the negative to plus one at the pattern. [[cue:label]] Each sample is labelled by its largest overlap in size, and its sign.

### beat: condition
**Visual.** Field arrows (field colour) point at pattern 1. Samples labelled pattern 2 or 3 fade to
grey; the ones labelled pattern 1 stay, split into blue (+) and orange (−).

> [[cue:field1]] With the field favouring pattern one, [[cue:condition]] the paper compares signs only among samples labelled pattern one.

### beat: assume
**Visual.** Three vertical bars (overlaps 1, 2, 3), each with a small Gaussian beside it; pattern
1's Gaussian is shifted slightly toward + (field colour arrow). Grey tag "approximation:
independent Gaussian overlaps". Several draws flicker past; the bar largest in size lights up as
the winner; when pattern 1 wins the draw is kept (blue or orange by sign), otherwise it fades.

> [[cue:assume]] The paper then approximates. Near beta equal to one, where the choice is made, the three overlaps are independent Gaussians. [[cue:shift]] Pattern one's is shifted by the field, as in Curie–Weiss. [[cue:winner]] The largest in size decides the final pattern.

### beat: pull
**Visual.** One shifted Gaussian. Mirror points at +r and −r mark the curve's heights. At
"parabola", the curve redraws as its log: a downward parabola, with the centred parabola in
dashed light grey behind it. At "tilt", a dashed straight line through the origin writes on: the
difference between the shifted and the centred parabola. At "grows", the log of the ratio of the
heights at ±r is drawn as a short blue bar that lengthens in proportion as r moves outward,
along that line. Then two distributions of |overlap 1|: grey
over all draws, blue over the draws pattern 1 won, with vertical lines at their means; the blue
mean is further right.

> [[cue:magnitude]] Compare the curve's heights at the same size on either side of zero. [[cue:log-ratio]] The log of their ratio is the sign's log-odds. [[cue:parabola]] On a log scale the Gaussian is a parabola, [[cue:tilt]] and the shift adds a straight tilt. [[cue:grows]] So the gap between the two sides grows in proportion to the size. [[cue:winners]] Pattern one wins only by beating the other two, so its winning overlaps are larger than usual.

### beat: amp
**Visual.** The two mean lines lift off and form a ratio, which writes on as
`A_3 = \frac{\mathbb{E}\max_{\mu\le 3}|Z_\mu|}{\mathbb{E}|Z|} = 1.662` (dashed light grey; the
definition is shown, not spoken). At "so their", the blue log-odds bar from the pull beat
stretches by the same ratio.

> [[cue:amp]] On average they are [[cue:amp-value]] one point six six two times larger, [[cue:scales]] so their sign's log-odds are too.

### beat: predict
**Visual.** At "also takes", two grey tags write on in a column: "independent Gaussian overlaps"
and "Hopfield equilibrium preference ≈ Curie–Weiss", with "not guaranteed" beside them. Then the
gold `c = 0.120` from the odds scene flies in from the left, `\times\,1.662` from the right; they
combine into `= 0.199` in a dashed light-grey box labelled "prediction", the two tags under it.

> [[cue:assumptions]] The paper also takes the Hopfield model's own preference to be the Curie–Weiss one, so the gain changes only the sampler's side of c. Neither assumption is guaranteed. [[cue:predict]] Under both, the prediction is the Curie–Weiss c times this gain: [[cue:pred-value]] zero point one nine nine. [[cue:answer]] Competition keeps more of the preference.

---

## scene: reheat
title: Where the runs start
chapter: reheat

### beat: protocol
**Visual.** At "leave", the two-valley curve from the commit scene flashes: a dot sits in one
valley, is lifted as the curve flattens to one valley, and drops into either valley at random
(shown twice, once landing in each). Then one line plot of β against sweeps: a short quick
staircase up to 1.6 ("brief cooling", grey tag "2 sweeps per temperature"); at "removal" a cut mark
where blue (1,+) dots are removed (at "removed") from a cloud of samples, at either field sign (two small field
arrows, opposite ways, both with (1,+) removed); a drop to β = 0.7 held flat ("reheating", grey
tag "30 sweeps at β = 0.7"; at "one valley", the one-valley curve flashes beside it); then the 834-sweep final cooling, drawn as a staircase.
At "The Curie–Weiss numbers", a small `61\%` tick and the gold `c = 0.120` write on with the grey tag "computed through the
same steps"; at "almost none", a grey tag: "e^{−30/τ} ≈ 1.2 × 10⁻⁴ · Curie–Weiss only · does not
bound Hopfield memory".

> [[cue:leave]] Reheating means raising the temperature again, so a run can leave its mode and choose again. [[cue:imbalance]] The paper tests its prediction on such Hopfield runs, from a deliberately imbalanced filtered start. [[cue:brief]] After a brief cooling, [[cue:remove]] every run labelled pattern one, positive, is removed. [[cue:reheat]] The rest are reheated below beta equal to one, where there is one valley, [[cue:recool]] then cooled on the paper's schedule. [[cue:same]] The Curie–Weiss numbers, sixty-one percent and c, came from these same steps; [[cue:memory]] there, almost none of that imbalance survives reheating.

### beat: controls
**Visual.** At "both directions", the field arrows flip to point toward −1, and the same (1,+)
removal mark repeats. Then two mirrored bars for opposite fields, each showing the share of the
sign that field favours; at "mirror" one bar reflects onto the other, and their sum writes on near 1 with an interval that contains 1. A zero-field bar at 0.4955 with its
interval around 0.5 (grey tag "zero field: 0.4955 [0.4863, 0.5048]"). At "does not prove", a grey
line "≠ fresh starts" writes on.

> [[cue:controls]] Every field strength is run in both directions, with the same removal. [[cue:mirror]] If the removal left a memory, the two directions would not mirror each other; [[cue:mirrored]] within the comparisons' precision, they do. [[cue:not-proof]] That does not prove the runs match fresh starts.

### beat: mnist
**Visual.** A strip of MNIST tiles from the paper's deep Boltzmann machine (`../assets/mnist_illustration_frames.png`):
an "8" blurs during reheating and re-forms as a "3". Grey tag "illustration · deep Boltzmann machine
trained on MNIST · one stored run · bias toward 3 added after reheating". No tallies on screen.

> [[cue:mnist]] In a deep Boltzmann machine trained on digits, [[cue:eights]] a reheated eight can become a three under a bias toward threes. [[cue:answer]] So reheating can free a run, but the run then cools and chooses again; [[cue:sets]] the sign chosen in that cooling can decide the proportions.

---

## scene: test
title: The test
chapter: test

### beat: run
**Visual.** Top: the dashed prediction box `0.120 \times 1.662 = 0.199` from the modes scene. A
grid of twenty small pattern tiles fills in at "three patterns" (grey tag "20 pattern sets · 3
patterns each"); eight field arrows at "eight": four lengths, each pointing both ways. Grey tag: "p = 3 · N = 2,048 · 2,000 runs per pattern set
and field".

> [[cue:run]] The test uses three patterns [[cue:fields]] and eight fields: four strengths, each pointing both ways.

### beat: share
**Visual.** Two bars: "sampled" (solid ink) at `66.9\%` and "model's reference" (dashed light
grey) at `97.3\%`. Grey tag "G = 0.0006125". At "approximate", the reference value becomes
`\approx 97.3\%` and the tag "approximate: from the mean overlap inside the positive mode" fades in. At "still
far", the gap between the two bars shades grey.

> [[cue:share]] At the strongest field toward pattern one, sixty-six point nine percent of samples labelled pattern one are positive. [[cue:reference]] The model's own share, ninety-seven point two six, [[cue:approx]] is only approximate here. [[cue:still-far]] So after reheating, the proportions are still far from the model's.

### beat: fit
**Visual.** The paper's Fig. 2b re-made: sampled log-odds (y) against reference log-odds (x) for
the eight fields, solid ink points with vertical intervals and grey horizontal intervals; the
dashed "model's proportions (c = 1)" diagonal; a solid gold slope line with its band. At "zero point one nine
nine", `c = 0.199` writes on in gold; then the interval `[0.191,\,0.210]` with a small grey tag
"from resampling the 20 pattern sets". At "fit", the share beat's "sampled" bar
(blue positive, orange negative, 0–100%) shrinks into the lower right and stays. At "prediction", the
dashed light-grey prediction line lands on top of the gold one, and "predicted 0.199" writes on in a
dashed box. At "at the strongest field", the bar gets its head "positive share at G = 0.0006125": a
dashed mark at `67.1\%` ("predicted") lands on its `66.9\%` boundary ("measured").

> [[cue:fit]] Across the eight fields, [[cue:slope]] the slope of sampled against the model's log-odds is the measured c. [[cue:measured]] It is zero point one nine nine, [[cue:interval]] with a ninety-five percent interval up to zero point two zero nine five. [[cue:agree]] The prediction is also zero point one nine nine. [[pause:0.8]] [[cue:units]] At the strongest field, sixty-seven point one percent positive, against sixty-six point nine measured.

### beat: one-pattern
**Visual.** A horizontal c axis from 0.18 to 0.22 with the measured interval as a solid ink bar and
the dashed prediction mark at 0.199 inside it. At "no competition", a single pattern tile and a
row of spins: the spins under the tile's −1 entries flip, and the label "Hopfield, one pattern"
becomes "Curie–Weiss" at "Curie–Weiss", and at "small differences" comes a grey footnote "model: 1/N vs
1/(N−1) · sampler: update order". At "Yet", the pattern, the spins and the footnote leave, and the label
"Curie–Weiss" becomes the c_CW of `c = 1.063 \times c_{\mathrm{CW}}`. At "Use that larger c", the
header equation comes down and its base is multiplied: `1.063 \times 0.120 \times 1.662`, brace
"one-pattern c"; at "zero point two one two" its 0.199 becomes 0.212 while the dashed mark slides
right to 0.212, just past the bar's end, leaving a ghost at 0.199. At "cancel", a brace from the ghost
to the new mark: "the match at 0.199 can come from approximations that cancel"; a second brace under
1.662: "competition: approximate".

> [[cue:one-pattern]] But a single pattern [[cue:no-competition]] has no competition. [[cue:flip]] Flipping the spins where it is minus one turns the Hopfield model into Curie–Weiss, up to small differences. [[cue:ratio]] Yet its measured c is one point zero six three times the Curie–Weiss c. [[cue:shift]] Use that larger c as the base, and the prediction becomes zero point two one two, [[cue:outside]] outside the interval. [[pause:0.8]] [[cue:cancel]] So the close match can come from approximations that cancel.

### beat: rows
**Visual.** The axis grows into the paper's Table 1 as rows; the measured interval and the ghost
mark at 0.199 become the p = 3 row, so the screen never empties: p = 1, 2, 3, 4 at N = 2,048, then
N = 1,024 and 4,096 at p = 3. Each row has a dashed prediction mark and a solid measured interval
(p = 1: 0.120 vs 0.127 [0.124, 0.131]; p = 2: 0.169 vs 0.188 [0.184, 0.191]; p = 3: 0.199 vs
0.201 [0.194, 0.210]; p = 4: 0.220 vs 0.211 [0.191, 0.239]; N = 1,024: 0.284 vs 0.263 [0.247,
0.283]; N = 4,096: 0.150 vs 0.160 [0.155, 0.165]). Grey tag: "separate runs · both field directions
pooled · Curie–Weiss model's share as reference · so p = 3 differs slightly from the fit". Column head
"measured vs predicted" (+ = measured above). At "up to eleven percent", the p = 2 row's gap highlights with `11\%`. At "captures the gain from competition", the predicted and measured marks for p = 1..4 join into two lines (down and to the right: c grows as the rows go to more patterns), labelled "more patterns → larger c".

> [[cue:rows]] In separate runs at other pattern counts and sizes, [[cue:misses]] the prediction misses by up to eleven percent. [[cue:answer]] It captures the gain from competition, with errors that depend on pattern count, size and update rule.

---

## scene: limits
title: What the law does and does not say
chapter: limits

### beat: size
**Visual.** Log-log plot of c against N (1,024, 2,048, 4,096): solid ink measured points from the
size scan (0.263, 0.201, 0.160) with intervals and a fitted line labelled `N^{-0.36}`; dashed
light-grey prediction for the staircase (0.284, 0.199, 0.150) labelled `N^{-0.46}`. At
"candidate", a small grey tag "pattern correlations ~ N^{-1/2} · not established".

> [[cue:size]] Across sizes, c falls as N to the minus zero point three six, [[cue:exponents]] not the minus zero point four six predicted for this schedule. [[cue:candidate]] The paper has not established why.

### beat: rate
**Visual.** A small axis labelled "cooling rate" with a grey cross over it and the tag "fixed
schedule in every Hopfield run".

> [[cue:rate]] The Hopfield runs never vary the cooling rate, so they do not test the rate law.

### beat: scope
**Visual.** Two grey arrows leave a box "annealed Gibbs sampling": one to "annealed importance
sampling · corrects estimates with weights", one to "diffusion models · different dynamics". Both
marked "future work".

> [[cue:scope]] The law is about annealed Gibbs sampling; [[cue:ais]] comparing it with annealed importance sampling, [[cue:diffusion]] or extending it to diffusion models, is future work.

### beat: return
**Visual.** Everything clears. The two 10×10 grids from the hook return as they were: 97 and 61.
At "near beta equal to one", the 36 missing squares fly into the grey selection interval of a small
fork plot between the grids, where a field arrow and a grey fluctuation band compete. At "c", the
gold `c = 0.120` writes on. At "quarter power", the 16×-to-2× ladder flashes in miniature; at
"about four", the sampler grid shows 59 → 63 with `\times 1.48` beside it. At "Competing
patterns", `c \approx 1.662 \times 0.120 = 0.199` with the tag "approximation; misses up to 11% elsewhere". Hold about three seconds (the film's last scene has a longer tail), then fade to
the ground.

> [[cue:return]] Ninety-seven asked for; sixty-one delivered. [[cue:where]] The sign is chosen near beta equal to one, where the weak field only tilts the choice. [[cue:c]] About an eighth of the model's log-odds survives: c, zero point one two. [[cue:slow]] Slower cooling raises it only through a quarter power on a ramp. On the paper's schedule, sixteen times the final cooling buys [[cue:stair]] about four positive samples in a hundred. [[cue:patterns]] Competing patterns keep more.
