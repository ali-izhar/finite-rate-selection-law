/* components/race.js — "Why a fraction is lost: field against fluctuations in the selection interval"
 *
 * Registers VE.components['race'] = { mount(el, opts) }.
 *
 * What it shows (paper ../paper/main.tex, Sec. 3 and App. B):
 *   Many runs follow the drift-and-fluctuation approximation (paper Eq. 7) of the Curie–Weiss chain near β = 1,
 *       dm = [(β−1) m − m³/3 + G] dt + sqrt(2/N) dW,      β = 1 + γ t   (t in sweeps),
 *   simulated live (Euler–Maruyama, seeded) from m = 0 at t = −4 γ^(−1/2).
 *   Panel 1: the runs' m(t), coloured by the mode they end in, plus a side histogram "now".
 *   Panel 2: field against fluctuations, in the linearised dynamics dm = (γ t m + G) dt + sqrt(2/N) dW. The rescaled
 *            variable y = m·exp(−γt²/2) obeys dy = w(t)(G dt + sqrt(2/N) dW) with w = exp(−γt²/2):
 *            w is the "selection interval" (std γ^(−1/2)), y is Gaussian with
 *              mean   B(t) = G ∫ w ds            → G sqrt(2π/γ)          (field's push ∝ G γ^(−1/2))
 *              sd     S(t) = sqrt(2/N ∫ w² ds)   → sqrt(2/N)(π/γ)^(1/4)   (fluctuation spread ∝ N^(−1/2) γ^(−1/4))
 *            and sign(y) = sign(m). The panel draws B(t) and S(t) as two growing curves. The share
 *            in mode A is Φ(B/S); at the end B/S = G sqrt(N) (π/γ)^(1/4) = x: the
 *            selection formula q_sel = Φ(a)  (paper Eq. 3, eq:law).
 *   Panel 3: the live share of runs with m > 0 vs time (faded before the interval, where it is not
 *            final), against the selection formula and the model's own equilibrium share q_eq; the gap is "lost".
 *
 * Validity of the formula in the drift-and-fluctuation approximation: the simulation depends only on x and Nγ.
 * Checked with this page's kernel (40,000 runs per setting) at the slider corners:
 * |sim − formula| ≤ 0.7 points; the largest gap (sim below the formula by 0.4–0.7 points) is at the slowest
 * schedule with the smallest N (N = 1024, γ = 2.5×10⁻⁴, Nγ = 0.256). Elsewhere within ±0.35 points.
 * The slider ranges keep Nγ ≥ 0.256 (N²γ = 262 … 4.3×10⁶). The paper's exact count checks
 * (Table 2, App. A) cover N = 1,024–4,096, γ = 0.00025–0.004 only.
 *
 * Numbers shown and their sources:
 *   formula q_sel = Φ(G√N(π/γ)^¼)           Eq. (eq:law); VE.sim.knX / VE.sim.Phi
 *   q_eq at β = 1.6                  exact Curie–Weiss, VE.sim.qEq / logOddsEq (tested against ../src)
 *   c = logit q / logit q_eq         paper Sec. 3 definition; VE.sim.knC
 *   exact count q at check points    data/cw_scaling.json "linear_sweep" (results/experiments/cw_linear_sweep.json)
 *   simulated share                  live, on this page (run count and settings shown)
 *   one-pattern ratio 1.063 at N = 2048   data/sampler_clock.json (paper Sec. 5, App. B)
 *   Kondepudi & Nelson 1983; Kondepudi & Gao 1987 (paper bib: kondepudinelson1983, kondepudigao1987)
 */
(function () {
  'use strict';
  var VE = (window.VE = window.VE || {});
  VE.components = VE.components || {};

  var BATCH = 6000;          // runs per batch
  var N_TRACE = 90;          // runs drawn as lines
  var T_START = -4;          // in window units σ = γ^(−1/2)
  var T_SHOW = 4;            // display ends here
  var T_END = 6;             // the tally is taken here (runs have chosen a sign by then)
  var STEPS_PER_SIGMA = 120; // Euler steps per σ
  var REC_EVERY = 3;         // record every 3 steps → 40 samples per σ
  var HFINE = 240;           // fine histogram bins over ±mHist; regrouped for display
  var HSHOW = 41;            // about this many bars in the "now" column
  var REF = { G: 0.001225, N: 1024, gamma: 0.001 };   // an exact count check point (Table 2, App. A)

  function mount(root, opts) {
    var ui = VE.ui, sim = VE.sim, el = ui.el, fmt = ui.fmt;
    // the c tile's hover, in words: no tile at rest names q_chain or q_eq
    if (ui.defineTerm && ui.glossary && !ui.glossary.raceKept) {
      ui.defineTerm('raceKept', { word: 'Retained, in log-odds', sym: 'c', ml: '<strong>The share of the model’s log-odds the sampler keeps:</strong> c = logit(the sampler’s share of A) ÷ logit(the model’s share of A). c = 1 is faithful; c = 0 is a coin flip.' });
    }
    opts = opts || {};

    /* ------------------------------------------------------------ state */
    var P = { G: REF.G, N: REF.N, gamma: REF.gamma };
    var run = null;        // current simulation result (accumulated over batches)
    var job = null;        // {cancelled} signal of the running simulation
    var pending = false;   // sliders moved; simulation stale
    var tNow = T_SHOW;     // playhead in σ units (rest frame = end of the run)
    var seedCounter = 1;
    var queuedAdd = false;
    var checks = [];       // exact count check points from data/cw_scaling.json

    /* ------------------------------------------------------------ maths */
    var SQ2 = Math.SQRT2;
    function erf(z) { return z >= 0 ? 1 - 2 * sim.Phi(-z * SQ2) : 2 * sim.Phi(z * SQ2) - 1; }
    function sigmaT(g) { return 1 / Math.sqrt(g); }
    // B(t), S(t) from the run start t0 (t in sweeps)
    function pushAt(t, p) {
      var g = p.gamma, t0 = T_START * sigmaT(g), a = Math.sqrt(g / 2);
      return p.G * Math.sqrt(Math.PI / (2 * g)) * (erf(a * t) - erf(a * t0));
    }
    function spreadAt(t, p) {
      var g = p.gamma, t0 = T_START * sigmaT(g), a = Math.sqrt(g);
      var I = Math.sqrt(Math.PI / (4 * g)) * (erf(a * t) - erf(a * t0));
      return Math.sqrt(Math.max(0, 2 / p.N * I));
    }
    function pushFinal(p) { return p.G * Math.sqrt(2 * Math.PI / p.gamma); }
    function spreadFinal(p) { return Math.sqrt(2 / p.N) * Math.pow(Math.PI / p.gamma, 0.25); }
    function lawX(p) { return sim.knX(p.G, p.N, p.gamma); }
    function lawQ(p) { return sim.Phi(lawX(p)); }
    var lqCache = {};
    function logOddsEq16(p) {
      var k = p.N + '|' + p.G;
      if (!(k in lqCache)) lqCache[k] = sim.logOddsEq(p.N, 1.6, p.G);
      return lqCache[k];
    }
    function qEq16(p) { return sim.expit(logOddsEq16(p)); }
    function cLaw(p) { return sim.logitPhi(lawX(p)) / logOddsEq16(p); }
    function cOf(q, p) { return Math.log(q / (1 - q)) / logOddsEq16(p); }
    function matchCheck(p) {
      for (var i = 0; i < checks.length; i++) {
        var c = checks[i];
        if (Math.abs(c.N / p.N - 1) < 0.01 && Math.abs(c.gamma / p.gamma - 1) < 0.01 && Math.abs(c.G / p.G - 1) < 0.01) return c;
      }
      return null;
    }
    // a probability as a percentage, never rounding a near-certainty up to 100%
    function pctStr(q) { return q > 0.999 ? '&gt;99.9%' : fmt(100 * q, { digits: 1 }) + '%'; }
    function probStr(q) { return q > 0.9995 ? '&gt;0.999' : fmt(q, { digits: 3 }); }

    /* ------------------------------------------------------- simulation */
    function* simulate(p, seed, n) {
      var sg = sigmaT(p.gamma);
      var dt = sg / STEPS_PER_SIGMA;
      var nShow = (T_SHOW - T_START) * STEPS_PER_SIGMA;
      var nAll = (T_END - T_START) * STEPS_PER_SIGMA;
      var M = nShow / REC_EVERY + 1;
      var r = sim.rng(seed), nrm = r.normal;
      var m = new Float64Array(n);
      var sd = Math.sqrt(2 * dt / p.N);
      var nT = Math.min(N_TRACE, n);
      var traces = new Float32Array(nT * M);
      var plus = new Float64Array(M);         // count of m > 0 at each record
      var hist = new Uint32Array(M * HFINE);  // fine histogram of m at each record, range ±mHist
      // histogram range: the stable state at the last shown level plus room for noise (a bound,
      // not the display range; the display range comes from the recorded traces)
      var bEnd = T_SHOW * sg * p.gamma;
      var mHist = Math.max(0.15, 1.18 * Math.sqrt(3 * bEnd) + 2.5 * Math.sqrt(1 / p.N));
      var t0 = T_START * sg;
      function record(j) {
        var c = 0;
        for (var i = 0; i < n; i++) {
          var x = m[i];
          if (x > 0) c++;
          var b = Math.floor((x / mHist + 1) / 2 * HFINE);
          if (b < 0) b = 0; else if (b >= HFINE) b = HFINE - 1;
          hist[j * HFINE + b]++;
        }
        plus[j] = c;
        for (var k = 0; k < nT; k++) traces[k * M + j] = m[k];
      }
      record(0);
      for (var s = 0; s < nAll; s++) {
        var beta = 1 + p.gamma * (t0 + (s + 0.5) * dt);
        var a = beta - 1;
        for (var i = 0; i < n; i++) {
          var x = m[i];
          m[i] = x + (a * x - x * x * x / 3 + p.G) * dt + sd * nrm();
        }
        if (s + 1 <= nShow && (s + 1) % REC_EVERY === 0) record((s + 1) / REC_EVERY);
        if (s % 8 === 7) yield s / nAll;
      }
      var fin = 0, fate = new Int8Array(nT);
      for (var q = 0; q < n; q++) if (m[q] > 0) fin++;
      for (var k2 = 0; k2 < nT; k2++) fate[k2] = m[k2] > 0 ? 1 : -1;
      // display range for m: 1.15 × the 99th percentile of |m| over the drawn traces
      var abs = new Float32Array(traces.length);
      for (var a2 = 0; a2 < traces.length; a2++) abs[a2] = Math.abs(traces[a2]);
      abs.sort();
      var mShow = Math.min(mHist, Math.max(0.04, 1.15 * abs[Math.floor(0.99 * (abs.length - 1))]));
      return { n: n, M: M, nT: nT, traces: traces, plus: plus, hist: hist, mHist: mHist, mShow: mShow, fate: fate, fin: fin };
    }

    function startRun(add) {
      if (job) job.cancelled = true;
      var sig = { cancelled: false };
      job = sig;
      var p = { G: P.G, N: P.N, gamma: P.gamma };
      var seed = (Math.imul(seedCounter++, 0x9e3779b1) ^ 0x5bd1e995) | 0;
      root.classList.add('is-running');
      sim.runChunked(simulate(p, seed, BATCH), { budgetMs: 6, signal: sig }).then(function (res) {
        if (sig.cancelled) return;
        job = null;
        root.classList.remove('is-running');
        if (add && run && run.p.G === p.G && run.p.N === p.N && run.p.gamma === p.gamma) {
          // pool the tally; keep the first batch's traces for the drawing
          for (var j = 0; j < run.M; j++) run.plus[j] += res.plus[j];
          for (var h = 0; h < run.hist.length; h++) run.hist[h] += res.hist[h];
          run.n += res.n; run.fin += res.fin; run.batches++;
        } else {
          res.p = p; res.batches = 1;
          run = res;
        }
        pending = false;
        root.classList.remove('is-stale');
        refresh();
        if (queuedAdd) { queuedAdd = false; startRun(true); }
      }, function () { /* cancelled */ });
    }

    /* ------------------------------------------------------------- DOM */
    var fig = ui.figure({
      plate: opts.plate != null ? opts.plate : 4,
      title: 'Why a fraction is lost: field against fluctuations',
      className: 've-race'
    });
    root.appendChild(fig.el);
    // terse canvas lettering and a short legend; the long labels live, verbatim, in .ve-race__readp
    // paragraphs that the chapter's reading card mirrors. race.css scopes its plate rules to .is-v2.
    fig.el.classList.add('is-v2');

    var intro = el('div', { class: 've-race__intro' },
      el('p', { html: 'Each line is one sampler run as <span data-term="beta">β</span> rises at a steady rate. In the <span class="ve-race__win">selection interval</span> the <span class="ve-c-bias ve-strong">field’s push</span> toward A competes with random <span class="ve-race__noise ve-strong">fluctuations</span>. Press play.' })
    );
    fig.stage.appendChild(intro);
    // the answer first: the share of runs in A against the model's share, and the one-line reason
    var headline = el('div', { class: 've-race__headline', 'aria-live': 'polite' });
    var headWhy = el('p', { class: 've-race__headwhy', html: '<b>Why:</b> inside the interval the field’s push grows like the interval’s length, the fluctuations only like its square root, so slower cooling helps the field. But the push starts tiny (G&nbsp;≈&nbsp;0.0012), and at the starting settings the fluctuation spread is still about 3.4× the push. So the field tilts a coin rather than deciding it.' });
    fig.stage.appendChild(headline);
    fig.stage.appendChild(headWhy);

    fig.setLegend([
      { role: 'mode-a', mark: 'line', label: 'ends in A' },
      { role: 'mode-b', mark: 'line', label: 'ends in B' },
      { role: 'ink-3', mark: 'wash', label: 'selection interval' }
    ]);
    var legendNote = el('p', { class: 've-race__legendnote', text: 'Each run is coloured by the mode it ends in (A: m > 0), so you can follow it back into the interval.' });
    fig.stage.appendChild(fig.legend);
    fig.stage.appendChild(legendNote);
    // how to read each panel: hidden in the plate, shown in the v2 "Reading the figure" card
    var READ = [
      'Each line is one sampler run as β rises at a steady rate, coloured by the mode it ends in: blue ends in A (m > 0, the preferred mode), orange in B. The grey shading is the selection interval around β = 1, darker where a kick still counts more. Press play, or drag across the plot, to move through time.',
      'Panel 1, runs (simulated, drift-plus-noise equation): m is the average of the N spins (±1); its sign says which mode. The column at the right shows where all runs are now.',
      'Panel 2, field against fluctuations (linearised theory): the field’s push against the fluctuation spread, in units of m, each kick weighted by how much it still counts (the shaded interval). The growth after the interval scales both alike. Its readout gives push ÷ spread and the chance of A = Φ(push ÷ spread), Φ the standard normal CDF.',
      'Panel 3, share of runs in mode A: the share with m > 0 at each moment, faded before the interval, where it is not final yet. The flat solid line at the top is the model’s own share of A: its equilibrium at the final β = 1.6 (right of this plot), which a perfectly mixed sampler would reach. The flat dashed line is the selection formula (Eq. 3). The shaded gap between these two flat lines is lost: the model wants this much more. The solid curve that rises through the interval is the simulated share.',
      'Before the interval, the pull back toward m = 0 still erases the kicks, so the side a run is on is not final yet. Inside the interval the kicks count, and push and fluctuations pile up. After the interval the drift carries each run away from m = 0, and switching sides becomes rare.'
    ];
    var readBox = el('div', { class: 've-race__reading', hidden: true });
    READ.forEach(function (t) { readBox.appendChild(el('p', { class: 've-race__readp', text: t })); });
    fig.stage.appendChild(readBox);

    var cv = ui.canvas(fig.stage, {
      height: function (w) { return w < 560 ? 780 : Math.round(Math.min(640, Math.max(580, w * 0.56))); },
      label: 'Three aligned panels over time: sampler runs splitting into two modes; the field’s push and the fluctuation spread growing inside the selection interval; and the share of runs in mode A settling on the selection formula, below the model’s own preference.',
      draw: draw
    });

    // controls: the timeline (aligned with the plot's time axis), then the sliders
    // the window edges as the narration sees them: the phase switches where the displayed (rounded)
    // sweep count reaches ±round(1/√γ), so '−32 sweeps' already reads 'window opens' (exact at the start γ)
    var sg0 = sigmaT(P.gamma);
    var W_OPEN = -(Math.round(sg0) + 0.5) / sg0, W_CLOSE = (Math.round(sg0) - 0.5) / sg0;
    var player = ui.player({
      duration: T_SHOW - T_START, value: T_SHOW - T_START, speed: (T_SHOW - T_START) / 8,
      label: 'Time through the cooling run',
      chapters: [
        { t: 0, label: 'Before the interval: the pull back to m = 0 erases kicks' },
        { t: W_OPEN - T_START, label: 'The selection interval opens' },
        { t: -T_START, label: 'β = 1: the transition' },
        { t: W_CLOSE - T_START, label: 'The interval closes (+1/√γ): new kicks barely count' },
        { t: 2.5 - T_START, label: 'After the interval: runs rarely switch sides' }
      ],
      onTime: function (t) { tNow = t + T_START; cv.redraw(); updateNow(); }
    });
    var fmtG = function (v) { return fmt(v, { sci: true, sig: 3 }); };
    var fmtGam = function (v) { return fmt(v, { sci: true, sig: 2 }); };
    var slG = ui.slider({ label: 'Field', sym: 'G', min: 1e-4, max: 1e-2, value: P.G, log: true, tone: 'bias',
      format: fmtG, hint: 'The model’s preference for mode A, per spin: tiny, but it acts on all N spins. Scales the push; it barely changes the retained share c.',
      marks: [{ value: REF.G, label: '' }],
      onInput: function (v) { P.G = v; changed(); } });
    var slN = ui.slider({ label: 'Model size', sym: 'N', min: 1024, max: 16384, value: P.N, log: true, step: 1, tone: 'noise',
      format: function (v) { return fmt(Math.round(v)); }, hint: 'Number of spins. More spins, smaller fluctuations per sweep.',
      marks: [{ value: 1024, label: '1k' }, { value: 2048, label: '2k' }, { value: 4096, label: '4k' }, { value: 16384, label: '16k' }],
      onInput: function (v) { P.N = Math.round(v); changed(); } });
    var slGam = ui.slider({ label: 'Cooling rate', sym: 'γ', min: 2.5e-4, max: 1.6e-2, value: P.gamma, log: true, tone: 'theory',
      format: fmtGam, unit: 'per sweep', hint: ' ',
      marks: [{ value: 2.5e-4, label: '' }, { value: 1e-3, label: '' }, { value: 4e-3, label: '' }, { value: 1.6e-2, label: '' }],
      onInput: function (v) { P.gamma = v; changed(); } });
    [slG, slN, slGam].forEach(function (s) { s.el.classList.add('has-marklabels'); });
    var gamHint = slGam.el.querySelector('.ve-slider__hint');

    var btnMore = ui.button({ label: 'Add ' + fmt(BATCH) + ' runs', icon: 'step', onClick: function () { if (job) queuedAdd = true; else startRun(true); } });
    var btnReset = ui.button({ label: 'Reset to paper check point', title: 'A setting the paper also checks with the exact CW chain (Table 2, App. A)', kind: 'ghost', icon: 'reset', onClick: function () {
      slG.set(REF.G, { silent: true }); slN.set(REF.N, { silent: true }); slGam.set(REF.gamma, { silent: true });
      P = { G: REF.G, N: REF.N, gamma: REF.gamma }; changed(true);
    } });
    var rowPlay = el('div', { class: 've-race__playrow' }, player.el);
    var nowLine = el('p', { class: 've-race__nowline', 'aria-live': 'polite' });
    var rowSl = el('div', { class: 've-race__sliders' }, slG.el, slN.el, slGam.el);
    var rowBtns = el('div', { class: 've-race__btns' }, btnMore, btnReset);
    fig.stage.appendChild(rowPlay);
    fig.stage.appendChild(nowLine);
    fig.controls.remove();

    // the result: end sentence, then the strip of end-of-window numbers, then the law's origin
    var endLine = el('p', { class: 've-race__end' });
    function box(label, tone, sub, area) {
      var v = el('span', { class: 've-race__v' });
      var s = el('span', { class: 've-race__sub', html: sub || '' });
      var b = el('div', { class: 've-race__box', style: { '--tone': tone ? 'var(--' + tone + ')' : null, gridArea: area } },
        el('span', { class: 've-race__lab', html: label }), v, s);
      return { el: b, v: v, sub: s };
    }
    var nw = function (s) { return '<span class="ve-race__nw">' + s + '</span>'; };
    var bxPush = box('Field’s push', 'bias', nw('G·√(2π/γ)') + ' ' + nw('∝ G γ<sup>−1/2</sup>'), 'push');
    var bxSpread = box('Fluctuation spread', 'noise', nw('√(2/N)·(π/γ)<sup>1/4</sup>') + ' ' + nw('∝ N<sup>−1/2</sup>&#8201;γ<sup>−1/4</sup>'), 'spread');
    var bxX = box('Push ÷ spread', null, nw('a = G√N&#8201;(π/γ)<sup>1/4</sup>'), 'x');
    var bxLaw = box('Selection formula', 'theory', nw('q<sub>sel</sub> = Φ(a)'), 'law');
    var bxSim = box('Simulated', 'data', '', 'sim');
    var bxEq = box('Model’s own', 'bias', nw('equilibrium (a perfectly') + ' ' + nw('mixed sampler) at the') + ' ' + nw('final β = 1.6'), 'eq');
    var bxC = box('Retained, <i>c</i>', 'highlight', '', 'c');
    bxC.el.classList.add('is-c');
    var op = function (s, area) { return el('span', { class: 've-race__op', 'aria-hidden': 'true', text: s, style: { gridArea: area } }); };
    var strip = el('div', { class: 've-race__strip' },
      el('div', { class: 've-race__rowlab', text: 'At the end of the interval', style: { gridArea: 'h1' } }),
      bxPush.el, op('÷', 'o1'), bxSpread.el, op('=', 'o2'), bxX.el,
      el('div', { class: 've-race__rowlab', text: 'Share of runs in mode A', style: { gridArea: 'h2' } }),
      bxLaw.el, bxSim.el, op('vs', 'o3'), bxEq.el, op('→', 'o4'), bxC.el);
    var why = el('p', { class: 've-race__why', html: '<span class="ve-label">Why Φ</span> At the interval’s end a run sits at the push plus Gaussian fluctuations of this spread, so it lands on the A side with probability Φ(push ÷ spread), Φ the Gaussian CDF. ' +
      'The share is counted at +6/√γ sweeps, when every run is far from m = 0.' });
    var result = el('div', { class: 've-race__result' }, endLine, why, strip);
    // sliders, formulas and the law's origin fold under one disclosure; the plate reads without them
    var explore = el('details', { class: 've-race__explore' },
      el('summary', { html: '<span>Explore</span><b>Change the field, size and cooling rate</b><em>and see the push, the spread and the formula</em>' }),
      rowSl, rowBtns, result);
    fig.stage.appendChild(explore);

    fig.setCaption(
      'Slower cooling lengthens the interval, so it helps, but by less than it costs. Chapter 5 asks how much less.');
    fig.setTakeaway('The sign is chosen in an interval of about γ<sup>−1/2</sup> sweeps, where the field only tilts a coin tossed by the fluctuations.');
    ui.terms(intro);

    /* ------------------------------------------------------------ update */
    var debounce = 0;
    function changed(now) {
      pending = true;
      queuedAdd = false;
      root.classList.add('is-stale');
      refresh();
      clearTimeout(debounce);
      debounce = setTimeout(function () { startRun(false); }, now ? 0 : 180);
    }

    // the playhead narration (updates while scrubbing)
    function updateNow() {
      var p = run ? run.p : P;
      var sg = sigmaT(p.gamma);
      nowLine.innerHTML = '<span class="ve-race__now">' + fmt(Math.round(tNow * sg)) + ' sweeps from β = 1 · β = ' + fmt(1 + p.gamma * tNow * sg, { digits: 3 }) + '</span>';
    }

    // the end-of-window result (constants of the settings; the simulated box waits for the run)
    function updateResult() {
      var p = P;
      var x = lawX(p), q = lawQ(p), qe = qEq16(p), cl = cLaw(p);
      bxPush.v.textContent = fmt(pushFinal(p), { digits: 3 });
      bxSpread.v.textContent = fmt(spreadFinal(p), { digits: 3 });
      bxX.v.textContent = fmt(x, { digits: 3 });
      bxLaw.v.innerHTML = probStr(q);
      bxEq.v.innerHTML = probStr(qe);
      bxC.v.textContent = fmt(cl, { digits: 2 });
      var fresh = run && !pending;
      var share = fresh ? run.fin / run.n : NaN;
      if (fresh) {
        var ci = wilson(share, run.n);
        bxSim.v.innerHTML = probStr(share);
        bxSim.sub.innerHTML = nw('95%: ' + fmt(ci[0], { digits: 3 }) + '–' + fmt(ci[1], { digits: 3 })) + ' ' + nw(fmt(run.n) + ' runs');
      } else {
        bxSim.v.textContent = '…';
        bxSim.sub.textContent = 'simulating';
      }
      var cs = fresh && share > 0 && share < 1 ? cOf(share, p) : NaN;
      bxC.sub.innerHTML = nw('c = logit q ÷ logit q<sub>eq</sub>') + (isFinite(cs) ? ' ' + nw('simulated: ' + fmt(cs, { digits: 2 })) : '');

      var lead = x < 1
        ? 'The fluctuation spread is ' + fmt(1 / x, { digits: 1 }) + '× the push, so the field only tilts the coin: '
        : 'The push is ' + fmt(x, { digits: 1 }) + '× the fluctuation spread, so most runs follow the field: ';
      endLine.innerHTML = '<span class="ve-label">' + 'Now' + '</span> <span>' + lead + pctStr(q) + ' of runs end in A. A perfectly mixed sampler would let the field act until the shares match the model; here it gets only the interval.</span>';
      headline.innerHTML =
        '<span class="ve-race__hl"><span class="ve-race__hlab">Runs ending in mode A</span><span class="ve-race__hv">' + pctStr(q) + '</span><span class="ve-race__hsub">the formula' + (fresh ? '; simulated ' + pctStr(share) : '') + '</span></span>' +
        '<span class="ve-race__hl"><span class="ve-race__hlab">The model’s own share</span><span class="ve-race__hv ve-c-bias">' + pctStr(qe) + '</span><span class="ve-race__hsub">a perfectly mixed sampler, β&nbsp;=&nbsp;1.6</span></span>' +
        '<span class="ve-race__hl is-c"><span class="ve-race__hlab"><span data-term="raceKept">Retained, in log-odds</span></span><span class="ve-race__hv">c&nbsp;=&nbsp;' + fmt(cl, { digits: 2 }) + '</span><span class="ve-race__hsub">the formula' + (isFinite(cs) ? '; simulated ' + fmt(cs, { digits: 2 }) : '') + ' · ' + nw('N&nbsp;=&nbsp;' + fmt(p.N, { group: true }) + ', γ&nbsp;=&nbsp;' + fmt(p.gamma, { sig: 2 })) + '</span></span>';
      if (ui.terms) ui.terms(headline);

      var chk = matchCheck(p);
      bxLaw.sub.innerHTML = nw('q<sub>sel</sub> = Φ(a)') + (chk ? ' ' + nw('exact CW chain: ' + fmt(chk.q_exact, { digits: 3 })) : '');
    }

    function refresh() {
      var p = P;
      var sg = sigmaT(p.gamma), k = Math.sqrt(REF.gamma / p.gamma);
      gamHint.innerHTML = 'Cooling rate: rise in β per sweep. ' + (Math.abs(k - 1) < 0.02
        ? 'Slower stretches the interval: the push grows as γ<sup>−1/2</sup>, the spread only as γ<sup>−1/4</sup>.'
        : 'Against γ = 10<sup>−3</sup>: interval ×' + fmt(k, { digits: 2 }) + ', push ×' + fmt(k, { digits: 2 }) + ', spread ×' + fmt(Math.sqrt(k), { digits: 2 }) + '.');
      fig.setNote('Live simulation of the drift-plus-noise equation (4.2; paper Eq. 7), with drift (β−1)m − m³/3 + G and fluctuation variance 2/N per sweep · linear schedule β = 1 + γt · N = ' + fmt(p.N) +
        ', γ = ' + fmtGam(p.gamma) + ', G = ' + fmtG(p.G) + ' · ' + fmt(run && !pending ? run.n : BATCH) + ' runs from m = 0 at β = ' + fmt(1 - 4 * Math.sqrt(p.gamma), { digits: 3 }) +
        ' · share counted ' + fmt(Math.round(T_END * sg)) + ' sweeps after β = 1 · one sweep = N single-spin Gibbs updates at random times (the paper’s one-pattern Hopfield runs, which differ in update order and interaction strength, give 1.063 times the Curie–Weiss c at N = 2,048; Sec. 5) · the approximation holds near β = 1 only' +
        ' · panels 1 and 3 are the live simulation; panel 2 is the linearised theory (it drops the m³ term, which is small inside the interval)' +
        ' · across the slider range the simulated share stays within 0.7 points of the formula, slightly below it at the slowest schedules with N = 1,024' +
        ' · exact-chain check points: paper Table 2 (App. A) · Kondepudi and Nelson 1983; Kondepudi and Gao 1987', 'live');
      updateResult();
      updateNow();
      cv.redraw();
    }

    /* ------------------------------------------------------------- draw */
    function geom(w) {
      var phone = w < 560;
      return { phone: phone, L: phone ? 38 : 50, R: phone ? 68 : 96 };
    }
    function draw(ctx, w, h, c) {
      var p = run ? run.p : P;
      var g = geom(w), phone = g.phone, L = g.L, R = g.R;
      // keep the timeline's track exactly over the plot's time axis (the range thumb is 18 px wide)
      root.style.setProperty('--race-l', (L - 9) + 'px');
      root.style.setProperty('--race-r', (R - 9) + 'px');
      var T = 70, Bm = 44, gap = 38;   // no panel subtitles: the reading card carries them
      var H = h - T - Bm - 2 * gap;
      var h1 = Math.round(H * 0.37), h2 = Math.round(H * 0.28), h3 = H - h1 - h2;
      var P1 = { y0: T, y1: T + h1 }, P2 = { y0: P1.y1 + gap, y1: P1.y1 + gap + h2 }, P3 = { y0: P2.y1 + gap, y1: h - Bm };
      var x0 = L, x1 = w - R;
      var sg = sigmaT(p.gamma);
      var sx = ui.scale.linear([T_START, T_SHOW], [x0, x1]);  // σ units
      var tN = Math.max(T_START, Math.min(T_SHOW, tNow));
      var px = sx(tN);
      var fs = phone ? 11.5 : 12;   // canvas text: no readable label below 11 px (fs1); on a phone the panel subtitles are dropped instead
      var fs1 = Math.max(11, fs - 1);
      var stale = pending || !run;
      var dark = ui.theme() === 'dark';

      // window shading: weight w(t) = exp(−γ t² / 2) = exp(−u²/2), u = t/σ
      ctx.save();
      ctx.fillStyle = ui.alpha(c.ink3, dark ? 0.2 : 0.14);
      for (var X = x0; X < x1; X += 2) {
        var u = sx.invert(X + 1);
        ctx.globalAlpha = Math.min(1, Math.exp(-u * u / 2) * (dark ? 1.5 : 1));
        ctx.fillRect(X, P1.y0, 2, P3.y1 - P1.y0);
      }
      ctx.restore();
      [-1, 1].forEach(function (u) { ui.plot.vline(ctx, u, sx, P1.y0, P3.y1, { color: c.ink3, dash: [3, 4], width: 1 }); });
      ui.plot.vline(ctx, 0, sx, P1.y0, P3.y1, { color: c.ruleStrong, width: 1 });

      // bracket over the window, and a ghost bracket for the paper check point's window
      var kRef = Math.sqrt(p.gamma / REF.gamma);  // reference half-width in σ units
      var bx0 = sx(-1), bx1 = sx(1), by = 18;
      bracket(ctx, bx0, bx1, by, c.ink2, 1.2, null);
      var winTxt = 'interval ±' + fmt(Math.round(sg)) + ' sweeps';
      var showGhost = Math.abs(kRef - 1) > 0.02;
      if (showGhost && kRef >= T_SHOW - 0.5) {
        // the reference window would span the whole axis: say it in words instead of drawing it
        // the current window is kRef times SHORTER than the reference one (kRef = reference half-width in σ units)
        winTxt += phone ? ' (' + fmt(kRef, { digits: 1 }) + '× shorter than at 10⁻³)' : ' (' + fmt(kRef, { digits: 1 }) + '× shorter than at γ = 10⁻³)';
        showGhost = false;
      }
      boxText(ctx, winTxt, (bx0 + bx1) / 2, by - 5, { c: c, size: fs, align: 'center', color: c.ink2, weight: 600, halo: true });
      if (showGhost) {
        // dashed reference bracket, below the main bracket's legs; its label sits in a gap of the line
        var gx0 = sx(-kRef), gx1 = sx(kRef), gy = by + 14;
        var gl = 'at γ = 10⁻³';
        ctx.save(); ctx.font = ui.font('mono', fs1); var glw = ctx.measureText(gl).width; ctx.restore();
        var inside = glw < gx1 - gx0 - 24, gm = (gx0 + gx1) / 2;
        ctx.save(); ctx.strokeStyle = c.ink3; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(gx0, gy + 6); ctx.lineTo(gx0, gy);
        if (inside) { ctx.lineTo(gm - glw / 2 - 4, gy); ctx.moveTo(gm + glw / 2 + 4, gy); }
        ctx.lineTo(gx1, gy); ctx.lineTo(gx1, gy + 6);
        ctx.stroke(); ctx.restore();
        var ty = gy + 3.5;
        if (inside) boxText(ctx, gl, gm, ty, { c: c, size: fs1, color: c.ink3, align: 'center' });
        else if (gx1 + 6 + glw < w - 2) boxText(ctx, gl, gx1 + 6, ty, { c: c, size: fs1, color: c.ink3, halo: true });
        else boxText(ctx, gl, gx0 - 6, ty, { c: c, size: fs1, color: c.ink3, align: 'right', halo: true });
      }

      // ---------- panel 1: runs
      var my = run ? run.mShow : 0.5;
      var sy1 = ui.scale.linear([-my, my], [P1.y1, P1.y0]);
      ui.plot.hline(ctx, 0, sy1, x0, x1, { color: c.ruleStrong, width: 1 });
      ui.plot.axis(ctx, sy1, { side: 'left', at: x0, ticks: tickSym(my), format: function (v) { return fmt(v, { digits: tickDigits(my) }); }, c: c, size: fs - 0.5 });
      panelTitle(ctx, '1 · Runs (simulated)', x0, P1.y0, c, fs);
      boxText(ctx, '↑ m > 0: mode A', x0 + 6, P1.y0 + fs + 3, { c: c, size: fs - 0.5, color: c.modeA, weight: 600, halo: true });
      boxText(ctx, '↓ m < 0: mode B', x0 + 6, P1.y1 - 5, { c: c, size: fs - 0.5, color: c.modeB, weight: 600, halo: true });
      if (run) {
        var jN = Math.round((tN - T_START) * STEPS_PER_SIGMA / REC_EVERY);
        ctx.save();
        ctx.beginPath(); ctx.rect(x0, P1.y0, x1 - x0, P1.y1 - P1.y0); ctx.clip();
        if (stale) ctx.globalAlpha = 0.3;
        ctx.lineWidth = phone ? 0.9 : 1.1;
        ctx.lineJoin = 'round';
        var du = (T_SHOW - T_START) / (run.M - 1);
        for (var k = 0; k < run.nT; k++) {
          ctx.strokeStyle = ui.alpha(run.fate[k] > 0 ? c.modeA : c.modeB, 0.55);
          ctx.beginPath();
          for (var j = 0; j <= jN; j++) {
            var xx = sx(T_START + j * du), yy = sy1(run.traces[k * run.M + j]);
            if (j === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy);
          }
          ctx.stroke();
        }
        ctx.restore();
        // histogram "now" in the right gutter: regroup the fine bins to about HSHOW bars over ±my
        var fw = 2 * run.mHist / HFINE, kk = Math.max(1, Math.round((2 * my / HSHOW) / fw)), half = HFINE / 2;
        var groups = {}, maxc = 0, base = jN * HFINE;
        for (var b = 0; b < HFINE; b++) {
          var cnt = run.hist[base + b]; if (!cnt) continue;
          var gi = Math.floor((b - half) / kk);
          groups[gi] = (groups[gi] || 0) + cnt;
          if (groups[gi] > maxc) maxc = groups[gi];
        }
        var hx0 = x1 + 8, hw = R - 14, bh = Math.abs(sy1(kk * fw) - sy1(0));
        ctx.save();
        ctx.globalAlpha = (stale ? 0.3 : 1) * 0.75;
        Object.keys(groups).forEach(function (key) {
          var gi2 = +key, lo = gi2 * kk * fw, mid = lo + kk * fw / 2;
          if (Math.abs(mid) > my) return;
          ctx.fillStyle = mid > 0 ? c.modeA : c.modeB;
          ctx.fillRect(hx0, sy1(lo + kk * fw) + 0.5, Math.max(1, hw * groups[key] / maxc), Math.max(1, bh - 1));
        });
        ctx.restore();
        boxText(ctx, phone ? 'now' : 'all runs now', hx0, P1.y0 - 6, { c: c, size: fs1, color: c.ink3 });
      }

      // ---------- panel 2: the race (linearised theory), push and spread as two growing curves
      var Bf = pushFinal(p), Sf = spreadFinal(p);
      // headroom: the top curve must stay below the readout line drawn at P2.y0 + fs + 3
      var M2 = Math.max(Bf, Sf), H2 = P2.y1 - P2.y0;
      var ymax2 = Math.max(M2 * 1.12, M2 * H2 / Math.max(1, H2 - fs - 12));
      var sy2 = ui.scale.linear([0, ymax2], [P2.y1, P2.y0]);
      var nS = 200, xsA = [], bA = [], sA = [];
      for (var i = 0; i <= nS; i++) {
        var uu = T_START + (T_SHOW - T_START) * i / nS, tt = uu * sg;
        xsA.push(uu); bA.push(pushAt(tt, p)); sA.push(spreadAt(tt, p));
      }
      var cut = Math.round(nS * (tN - T_START) / (T_SHOW - T_START));
      ui.plot.axis(ctx, sy2, { side: 'left', at: x0, ticks: [0, niceTop(ymax2)], format: function (v) { return fmt(v, { digits: tickDigits(ymax2) }); }, c: c, size: fs - 0.5 });
      ui.plot.hline(ctx, 0, sy2, x0, x1, { color: c.ruleStrong, width: 1 });
      // the future, faint; the past up to the playhead, strong
      ui.plot.line(ctx, xsA.slice(cut), sA.slice(cut), sx, sy2, { color: c.noise, width: 1.4, alpha: 0.35, dash: [3, 3] });
      ui.plot.line(ctx, xsA.slice(cut), bA.slice(cut), sx, sy2, { color: c.bias, width: 1.4, alpha: 0.35, dash: [3, 3] });
      var lo0 = xsA.slice(0, cut + 1).map(function () { return 0; });
      ui.plot.band(ctx, xsA.slice(0, cut + 1), lo0, sA.slice(0, cut + 1), sx, sy2, { fill: c.noiseWash });
      ui.plot.line(ctx, xsA.slice(0, cut + 1), sA.slice(0, cut + 1), sx, sy2, { color: c.noise, width: 2.6 });
      ui.plot.line(ctx, xsA.slice(0, cut + 1), bA.slice(0, cut + 1), sx, sy2, { color: c.bias, width: 2.6 });
      panelTitle(ctx, phone ? '2 · Field vs fluctuations (theory)' : '2 · The field’s push against the fluctuation spread (linearised theory)', x0, P2.y0, c, fs);
      var Bn = pushAt(tN * sg, p), Sn = spreadAt(tN * sg, p);
      // gutter labels at the current heights, nudged apart
      var labs = [
        { y: sy2(Sn), name: 'spread', v: Sn, color: c.noise },
        { y: sy2(Bn), name: 'push', v: Bn, color: c.bias }
      ].sort(function (a, b2) { return a.y - b2.y; });
      var lh = 2 * (fs + 2);
      labs[1].y = Math.max(labs[1].y, labs[0].y + lh);
      if (labs[1].y > P2.y1 + 6) { labs[1].y = P2.y1 + 6; labs[0].y = Math.min(labs[0].y, labs[1].y - lh); }
      labs.forEach(function (lb) {
        boxText(ctx, lb.name, x1 + 8, lb.y - 2, { c: c, size: fs - 0.5, color: lb.color, weight: 600 });
        boxText(ctx, fmt(lb.v, { digits: 3 }), x1 + 8, lb.y + fs, { c: c, size: fs - 0.5, color: lb.color });
      });
      if (Sn > 1e-12) {
        // three digits on both sides, so Φ(arg) = value checks by hand and matches the law tile (Φ(0.293) = 0.615)
        var rTx = fmt(Bn / Sn, { digits: 3 }), pTx = fmt(sim.Phi(Bn / Sn), { digits: 3 });
        var ratioTxt = phone ? 'push ÷ spread = ' + rTx + ', P(A) = ' + pTx
          : 'push ÷ spread = ' + rTx + '   P(A) = Φ(' + rTx + ') = ' + pTx;
        boxText(ctx, ratioTxt, x0 + 6, P2.y0 + fs + 3, { c: c, size: fs - 0.5, color: c.ink, weight: 600, halo: true });
      }

      // ---------- panel 3: share in mode A, against the law and the model's own share
      var q = lawQ(p), qe = qEq16(p);
      var simEnd = run && !stale ? run.fin / run.n : q;
      var ylo = 0.45, yhi = Math.min(1.03, Math.max(0.6, qe + 0.05, q + 0.08, simEnd + 0.08));
      var sy3 = ui.scale.linear([ylo, yhi], [P3.y1, P3.y0]);
      var t3 = (yhi < 0.75 ? [0.5, 0.6, 0.7] : [0.5, 0.75, 1]).filter(function (v) { return v <= yhi + 1e-9; });
      ui.plot.axis(ctx, sy3, { side: 'left', at: x0, ticks: t3, format: function (v) { return fmt(v, { digits: 2 }); }, c: c, size: fs - 0.5, grid: [x0, x1] });
      var yM = sy3(qe), yL = sy3(q), y5 = sy3(0.5), yS = sy3(simEnd), lostY = null;
      // the lost gap between the law and the model's own share
      if (qe > q) {
        ctx.save(); ctx.fillStyle = ui.alpha(c.bias, dark ? 0.16 : 0.09);
        ctx.fillRect(x0, sy3(Math.min(qe, yhi)), x1 - x0, yL - sy3(Math.min(qe, yhi)));
        ctx.restore();
        if (yL - yM > 2 * fs + 12) {
          lostY = (yM + yL) / 2 + 4;   // drawn after the names are placed
        } else if (qe > 0.98) {
          // two short right-aligned lines in the empty space between the coin-flip line and the curves
          var gyN = (y5 + Math.max(yL, yS)) / 2;
          var l1 = phone ? 'gap small in probability,' : 'the gap looks small in probability,';
          var l2 = phone ? 'large in log-odds' : 'but is large in log-odds: only c = ' + fmt(cLaw(p), { digits: 2 }) + ' is retained';
          boxText(ctx, l1, x1 - 6, gyN - 1, { c: c, size: fs - 0.5, align: 'right', color: c.bias, weight: 600, halo: true });
          boxText(ctx, l2, x1 - 6, gyN + fs + 1, { c: c, size: fs - 0.5, align: 'right', color: c.bias, weight: 600, halo: true });
        }
      }
      ui.plot.hline(ctx, 0.5, sy3, x0, x1, { color: c.ruleStrong, width: 1, dash: [1, 3] });
      // the coin-flip name only when the law and the run curve end clear of the 0.5 line
      var showCoin = Math.min(Math.abs(yL - y5), Math.abs(yS - y5)) > fs + 6;
      if (showCoin) boxText(ctx, 'coin flip', x1 - 6, y5 - 4, { c: c, size: fs1, color: c.ink3, align: 'right', halo: true });
      ui.plot.hline(ctx, qe, sy3, x0, x1, { color: c.bias, width: 1.6, dash: false });
      ui.plot.hline(ctx, q, sy3, x0, x1, { color: c.theory, width: 1.6, dash: [6, 4] });
      panelTitle(ctx, '3 · Share of runs in mode A', x0, P3.y0, c, fs);
      if (run) {
        var jN3 = Math.round((tN - T_START) * STEPS_PER_SIGMA / REC_EVERY), dxs = [], dys = [];
        var du3 = (T_SHOW - T_START) / (run.M - 1), jOpen = Math.round((-1 - T_START) / du3);
        for (var j3 = 1; j3 <= jN3; j3++) { dxs.push(T_START + j3 * du3); dys.push(run.plus[j3] / run.n); }
        var cutD = Math.min(dxs.length, jOpen);
        ui.plot.line(ctx, dxs.slice(0, cutD + 1), dys.slice(0, cutD + 1), sx, sy3, { color: c.data, width: 1.6, alpha: stale ? 0.15 : 0.3 });
        if (dxs.length > cutD) ui.plot.line(ctx, dxs.slice(cutD), dys.slice(cutD), sx, sy3, { color: c.data, width: 2, alpha: stale ? 0.3 : 1 });
      }
      // in-plot names. A name's text band is [b − fs, b + 3] for baseline b. A position is valid when the
      // band stays inside the panel, crosses neither the model line nor the law line (and, if possible,
      // not the coin-flip line), and does not overlap the other name. Try the natural positions first.
      var mName = phone ? 'model’s own share' : 'model’s own share (β = 1.6)';
      var lName = phone ? 'selection formula' : 'selection formula (Eq. 3)';
      var lh3 = fs + 3;
      function band(b) { return [b - fs, b + 3]; }
      function hits(bd, y) { return y >= bd[0] - 1 && y <= bd[1] + 1; }
      function ok(b, strict) {
        var bd = band(b);
        if (bd[0] < P3.y0 + 1 || bd[1] > P3.y1 - 2) return false;
        if (hits(bd, yM) || hits(bd, yL)) return false;
        if (strict && hits(bd, y5)) return false;
        return true;
      }
      function cands(y, other) { return [y - 5, y + lh3, other - 5, other + lh3, y + 2 * lh3, y - 5 - lh3]; }
      var place = null;
      [true, false].some(function (strict) {
        var cm = cands(yM, yL), cl = cands(yL, yM);
        for (var a = 0; a < cm.length; a++) {
          if (!ok(cm[a], strict)) continue;
          for (var b4 = 0; b4 < cl.length; b4++) {
            if (!ok(cl[b4], strict)) continue;
            var A = band(cm[a]), B = band(cl[b4]);
            if (A[1] + 1 < B[0] || B[1] + 1 < A[0]) { place = [cm[a], cl[b4]]; return true; }
          }
        }
        return false;
      });
      if (!place) {
        // fallback: one stacked block under the lower line (or above the upper one)
        var lowY = Math.max(yM, yL), upY = Math.min(yM, yL);
        place = lowY + 2 * lh3 + 3 < P3.y1 ? [lowY + lh3, lowY + 2 * lh3] : [upY - 5 - lh3, upY - 5];
      }
      // the names go on top of the run curve, so their halo masks it
      boxText(ctx, mName, x0 + 6, place[0], { c: c, size: fs - 0.5, color: c.bias, weight: 600, halo: true });
      boxText(ctx, lName, x0 + 6, place[1], { c: c, size: fs - 0.5, color: c.theory, weight: 600, halo: true });
      // the 'lost' name sits in the gap, below the model's name if that name hangs under its line
      if (lostY != null) {
        var mB = band(place[0]);
        if (place[0] > yM && mB[0] < lostY && lostY - fs - 2 < mB[1]) lostY = mB[1] + fs + 4;
        if (lostY + 3 < yL - 1) boxText(ctx, 'lost', sx(2.5), lostY, { c: c, size: fs, align: 'center', color: c.bias, weight: 600, halo: true });
      }
      // gutter values, nudged apart
      var g3 = [
        { y: yM, t: phone ? probStr(qe).replace('&gt;', '>') : 'model ' + probStr(qe).replace('&gt;', '>'), color: c.bias },
        { y: yL, t: (phone ? '' : 'formula ') + probStr(q).replace('&gt;', '>'), color: c.theory }
      ];
      if (run && !stale) g3.push({ y: yS, t: 'sim ' + probStr(simEnd).replace('&gt;', '>'), color: c.data });
      g3.sort(function (a, b3) { return a.y - b3.y; });
      for (var gi3 = 1; gi3 < g3.length; gi3++) g3[gi3].y = Math.max(g3[gi3].y, g3[gi3 - 1].y + fs + 3);
      g3.forEach(function (lb) { boxText(ctx, lb.t, x1 + 8, lb.y + 4, { c: c, size: fs - 0.5, color: lb.color, weight: 600 }); });

      // playhead
      if (tN < T_SHOW - 1e-6) {
        ctx.save(); ctx.strokeStyle = c.ink; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(Math.round(px) + 0.5, P1.y0); ctx.lineTo(Math.round(px) + 0.5, P3.y1); ctx.stroke(); ctx.restore();
      }

      // bottom axis: sweeps since β = 1, and β at the two ends
      var tks = ui.ticks(T_START * sg, T_SHOW * sg, phone ? 4 : 8).filter(function (v) { return v / sg >= T_START - 1e-9 && v / sg <= T_SHOW + 1e-9; });
      var sxs = ui.scale.linear([T_START * sg, T_SHOW * sg], [x0, x1]);
      ui.plot.axis(ctx, sxs, { side: 'bottom', at: P3.y1, ticks: tks, format: function (v) { return fmt(v); }, c: c, size: fs - 0.5 });
      boxText(ctx, 'sweeps since β = 1', (x0 + x1) / 2, h - 4, { c: c, size: fs - 0.5, align: 'center', color: c.ink2 });
      boxText(ctx, '← β ' + fmt(1 + T_START * sg * p.gamma, { digits: 2 }) + (phone ? '' : ', hotter'), x0 - (phone ? 30 : 0), h - 4, { c: c, size: fs - 0.5, color: c.ink3 });
      boxText(ctx, (phone ? '' : 'colder, ') + 'β ' + fmt(1 + T_SHOW * sg * p.gamma, { digits: 2 }) + ' →', w - 2, h - 4, { c: c, size: fs - 0.5, align: 'right', color: c.ink3 });
    }

    function wilson(q, n) {
      var z = 1.96, d = 1 + z * z / n, c0 = (q + z * z / (2 * n)) / d, hw = z * Math.sqrt(q * (1 - q) / n + z * z / (4 * n * n)) / d;
      return [Math.max(0, c0 - hw), Math.min(1, c0 + hw)];
    }
    // title on one line and a plain-English subtitle under it, both in the gap above the panel
    /* A label with halo:true gets a solid surface-coloured box behind it, so the dashed interval lines
       and the β = 1 line never run through its letters (a glyph outline alone leaves them
       visible between the letters). */
    function boxText(ctx, s, x, y, o) {
      o = o || {};
      if (o.halo) {
        var c = o.c || ui.colors(), size = o.size || 11;
        ctx.save();
        ctx.font = ui.font(o.role || 'mono', size, o.weight || 400);
        var tw = ctx.measureText(s).width, al = o.align || 'left';
        var bx = al === 'center' ? x - tw / 2 : al === 'right' ? x - tw : x;
        ctx.fillStyle = c.surface; ctx.globalAlpha = 0.94;
        ctx.fillRect(bx - 3, y - size * 0.92, tw + 6, size * 1.22);
        ctx.restore();
      }
      ui.plot.text(ctx, s, x, y, o);
    }
    function panelTitle(ctx, s, x, y0, c, fs) {
      boxText(ctx, s, x, y0 - 11, { c: c, size: fs, color: c.ink, weight: 600, halo: true });
    }
    function bracket(ctx, xa, xb, y, color, wdt, dash) {
      ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = wdt; if (dash) ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(xa, y + 6); ctx.lineTo(xa, y); ctx.lineTo(xb, y); ctx.lineTo(xb, y + 6); ctx.stroke(); ctx.restore();
    }
    function niceTop(mx) { var t = ui.ticks(0, mx * 0.95, 3).filter(function (v) { return v > 0; }); return t[t.length - 1] || mx; }
    function tickSym(mx) { var top = niceTop(mx); return [-top, 0, top]; }
    function tickDigits(mx) { var top = niceTop(mx); return top >= 1 ? 0 : top >= 0.1 ? 1 : top >= 0.01 ? 2 : 3; }

    /* --------------------------------------------- canvas seek by pointer */
    var dragging = false;
    function seekFromEvent(e) {
      var w = cv.width, g = geom(w);
      var pt = cv.local(e);
      if (pt.x > w - g.R + 4 || pt.x < g.L - 4) return;
      var u = T_START + (pt.x - g.L) / (w - g.R - g.L) * (T_SHOW - T_START);
      player.pause();
      player.seek(Math.max(0, Math.min(T_SHOW - T_START, u - T_START)));
    }
    cv.canvas.addEventListener('pointerdown', function (e) {
      seekFromEvent(e);
      if (e.pointerType !== 'touch') { dragging = true; try { cv.canvas.setPointerCapture(e.pointerId); } catch (er) { /* ignore */ } }
    });
    cv.canvas.addEventListener('pointermove', function (e) { if (dragging) seekFromEvent(e); });
    cv.canvas.addEventListener('pointerup', function () { dragging = false; });
    cv.canvas.addEventListener('pointercancel', function () { dragging = false; });
    cv.canvas.style.cursor = 'ew-resize';

    /* ---------------------------------------------------------- start */
    try {
      sim.loadJSON('cw_scaling.json').then(function (d) {
        checks = (d.linear_sweep || []).map(function (r) { return { N: r.N, gamma: r.gamma, G: r.G, q_exact: r.q_exact, q_KN: r.q_KN }; });
        refresh();
      }).catch(function () { /* the page still works without the check points */ });
    } catch (e) { /* no fetch */ }
    refresh();
    startRun(false);

    /* the chapter's 'Reading the figure' card scrolls; while text remains below the cut, a sticky veil
       fades the last lines and says so (class ve-race-card--more, race.css). Cleared at the end. */
    function markMore(card) {
      card.classList.toggle('ve-race-card--more', card.scrollTop + card.clientHeight < card.scrollHeight - 2);
    }
    function onCardToggle(e) {
      var card = e.target;
      if (!card || card.id !== 'race-read' || e.newState !== 'open') return;
      if (!card.__raceMore) { card.__raceMore = true; card.addEventListener('scroll', function () { markMore(card); }, { passive: true }); }
      requestAnimationFrame(function () { markMore(card); });
    }
    document.addEventListener('toggle', onCardToggle, true);

    return {
      destroy: function () { if (job) job.cancelled = true; clearTimeout(debounce); document.removeEventListener('toggle', onCardToggle, true); player.destroy(); cv.destroy(); fig.el.remove(); },
      set: function (o) {
        if (o.G) slG.set(o.G, { silent: true });
        if (o.N) slN.set(o.N, { silent: true });
        if (o.gamma) slGam.set(o.gamma, { silent: true });
        P = { G: slG.value, N: Math.round(slN.value), gamma: slGam.value }; changed(true);
      },
      seek: function (u) { player.seek(u - T_START); },
      _state: function () { return { P: P, run: run && { n: run.n, fin: run.fin, mShow: run.mShow }, law: lawQ(P), qEq: qEq16(P), c: cLaw(P) }; }
    };
  }

  VE.components['race'] = { mount: mount };
})();
