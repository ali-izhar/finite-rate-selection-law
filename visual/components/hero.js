/* components/hero.js — "The thesis, live" (Visual Edition opening plate).
 *
 * What it shows: Gibbs-sampler runs on an exact two-mode model (Curie–Weiss, N = 256 binary
 * variables) cool from hot (near fair coins, beta = 0.6) to cold (beta = 1.6) on a
 * linear schedule. The model prefers mode A by exactly 9 to 1 at the final level (q_eq = 0.9; the
 * bias G is solved for at mount with VE.sim.logOddsEq). A tally compares the model's odds with the
 * runs' odds and shows c = logit(q_chain) / logit(q_eq), the share of the model's log-odds the
 * sampler keeps, next to the selection formula's prediction (VE.sim.knC, Eq. 3; nothing fitted).
 * One control: sweeps per descent (250, 1,000, 4,000), i.e. gamma = 1/250, 1/1000, 1/4000.
 *
 * Every number on screen is either
 *   - live: counted from VE.sim.CWChains (exact continuous-time lumped chain, 'poisson' clock).
 *     One descent of 300 runs is drawn; further descents of 300 runs (same kernel, same schedule,
 *     not drawn) are added to the tally, up to WARM descents per setting, so the first reading is
 *     not a single noisy batch. The runs are computed in two Web Workers (a blob that loads
 *     lib/sim.js), so the main thread only draws; if a Worker cannot start, the same kernel runs on
 *     the main thread in short slices;
 *   - the law: VE.sim.knC(G, N, gamma, 1.6);
 *   - or the paper (p = 3, N = 2,048: 0.120 x 1.662 = 0.199 predicted, 0.199 [0.1908, 0.2095]
 *     measured; p = 1, p = 2, N = 1,024, N = 4,096 off by 6-11%; main.tex abstract, Sec. 4;
 *     data/field_grid_points.json, data/table1.json, data/size_scan.json).
 * Offline check (node, same kernels): exact c = 0.1927, 0.2745, 0.3635 for 250, 1,000, 4,000
 * sweeps; law 0.186, 0.264, 0.375.
 */
(function () {
  'use strict';
  var VE = (window.VE = window.VE || {});
  VE.components = VE.components || {};

  /* Terms used by the overview chapter's shell text and by this plate. They have their own keys,
     so the shared glossary in lib/ui.js is not overwritten (the paper's wording, Eq. 3 and Eq. 6). */
  if (VE.ui && VE.ui.defineTerm) {
    VE.ui.defineTerm('selectionFormula', { word: 'Selection formula', sym: 'Eq. 3', ml: '<strong>A classical result</strong> (Kondepudi and Nelson 1983; Kondepudi and Gao 1987) for crossing a symmetry-breaking transition at a finite rate. It gives the chance that a run picks the favoured sign from three numbers: the field, the model size and the cooling rate. That chance is the final share of runs in mode A only if later switching between modes is negligible. The paper’s selection law is this formula converted to c by Eq. 2.' });
    VE.ui.defineTerm('fluctuations', { word: 'Random fluctuations', ml: '<strong>The run-to-run randomness of the Gibbs updates.</strong> Near the transition (β = 1) it can push a run toward either mode.' });
    VE.ui.defineTerm('weakField', { word: 'Field', sym: 'G', ml: '<strong>A small bias toward mode A:</strong> a push on every spin toward +1, fixed during a cooling run. In Fig. 0 it is set so that the model gives 9 to 1 odds for A.' });
    VE.ui.defineTerm('exactCount', { word: 'Exact CW chain', ml: '<strong>The sampler’s outcome probabilities, computed rather than sampled.</strong> In the two-mode Curie–Weiss model, every configuration with the same number of +1 spins behaves alike, so the paper tracks just N + 1 probabilities and evolves them numerically through the schedule: the numerical evolution of the exact CW chain. No random runs and no fitting (paper Sec. 2, App. A).' });
    VE.ui.defineTerm('cPred', { word: 'Predicted c', ml: '<strong>c<sub>CW</sub> = 0.120:</strong> the retained share computed numerically for the two-mode Curie–Weiss model on the Hopfield protocol (filtered start, reheating, cooling schedule). <strong>A<sub>3</sub> = 1.662:</strong> the predicted gain from competition among three patterns, under an independent-Gaussian approximation (Eq. 6).' });
  }

  var N = 256;                 // binary variables (paper: 2,048)
  var CHAINS = 300;            // runs per descent
  var B0 = 0.6, B1 = 1.6;      // the paper's first and last inverse temperatures (hot, cold)
  var STEPS = 400;             // staircase steps for the linear schedule
  var DB = (B1 - B0) / STEPS;
  var ODDS = 9;                // model odds for mode A at beta = 1.6
  var SETTINGS = [
    { key: 's1', sweeps: 250, label: '250', mult: '×1' },
    { key: 's4', sweeps: 1000, label: '1,000', mult: '×4' },
    { key: 's16', sweeps: 4000, label: '4,000', mult: '×16' }
  ];
  var TRAILS = 72;             // runs drawn with a full trail
  var LEVELS_PER_S = 95;       // visual pace: about 4.2 s per descent if compute allows
  var FRAME_BUDGET_MS = 6;     // main-thread fallback only: compute per animation frame
  var SLICE_SWEEPS = 1;        // fallback: longest single advance() call (300 runs x 1 sweep ~ 1-2 ms)
  var HOLD_S = 2.6;            // pause at the end of a descent before the next one
  var WARM = 12;               // descents per setting that the background may add (3,600 runs)
  var MIN_DESC = 10;           // descents (3,000 runs) before c is shown: a first small batch is too noisy to read
  var SMOOTH = 6;              // trail smoothing window, in levels (6 x 0.0025 = 0.015 in beta)

  function solveG() {
    var S = VE.sim, lo = 1e-6, hi = 0.1, target = Math.log(ODDS);
    for (var i = 0; i < 70; i++) { var mid = (lo + hi) / 2; if (S.logOddsEq(N, B1, mid) < target) lo = mid; else hi = mid; }
    return (lo + hi) / 2;
  }

  // Wilson 95% interval for a binomial proportion
  function wilson(a, n) {
    if (!n) return [NaN, NaN];
    var z = 1.959964, p = a / n, d = 1 + z * z / n;
    var ctr = (p + z * z / (2 * n)) / d, hw = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
    return [ctr - hw, ctr + hw];
  }

  function parseColor(s) {
    s = (s || '').trim();
    if (s[0] === '#') {
      if (s.length === 4) s = '#' + s[1] + s[1] + s[2] + s[2] + s[3] + s[3];
      return [parseInt(s.slice(1, 3), 16), parseInt(s.slice(3, 5), 16), parseInt(s.slice(5, 7), 16)];
    }
    var m = s.match(/[\d.]+/g);
    return m ? [+m[0], +m[1], +m[2]] : [128, 128, 128];
  }
  function mix(a, b, t, al) {
    return 'rgba(' + Math.round(a[0] + (b[0] - a[0]) * t) + ',' + Math.round(a[1] + (b[1] - a[1]) * t) + ',' +
      Math.round(a[2] + (b[2] - a[2]) * t) + ',' + (al == null ? 1 : al) + ')';
  }
  function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
  function smooth(e0, e1, x) { var t = clamp01((x - e0) / (e1 - e0)); return t * t * (3 - 2 * t); }

  function oddsText(a, b) {
    if (!a && !b) return '–';
    if (!b) return 'all mode A';
    if (!a) return 'all mode B';
    var r = a / b;
    if (r >= 1) return VE.ui.fmt(r, { digits: r >= 10 ? 0 : 1 }) + ' to 1';
    return '1 to ' + VE.ui.fmt(1 / r, { digits: 1 });
  }

  function mount(host, opts) {
    opts = opts || {};
    var ui = VE.ui, S = VE.sim, el = ui.el, fmt = ui.fmt;
    var now = function () { return performance.now(); };
    var G = solveG();
    var LOGIT_EQ = S.logOddsEq(N, B1, G);   // = ln 9 by construction
    var qEq = S.expit(LOGIT_EQ);

    var store = {};   // per-setting accumulated tallies
    SETTINGS.forEach(function (s) {
      s.gamma = 1 / s.sweeps;
      s.law = S.knC(G, N, s.gamma, B1);
      store[s.key] = { a: 0, b: 0, runs: 0 };
    });
    var cur = SETTINGS[0];
    var visited = { s1: true };   // the law's value for a budget shows once the reader has tried it (step 5 asks for a guess)

    /* ------------------------------------------------------------ DOM
       The plate carries numbers, not sentences. The two "how to read" captions stay in the
       DOM (hidden by lib/shell.css inside .v2-plate) because the chapter's "Reading the figure"
       card mirrors them. The "why" paragraph and the "is this the paper's c?" note live in the
       chapter's HTML (a drawer and a margin note), not here. Explanations of the readouts sit
       behind info chips (ui.info). */
    var root = el('div', { class: 've-hero' });
    host.appendChild(root);

    var fig = ui.figure({
      plate: 'Live',
      title: 'Sampler runs choose a mode',
      className: 've-figure--wide',
      // the colour words (mode A, mode B, grey = undecided, dashed = theory) are in the
      // "Reading the figure" and "Colours and marks" cards; only the outline key is not obvious
      legend: [
        { role: 'ink', mark: 'outline', label: 'model’s split at this β' }
      ]
    });
    root.appendChild(fig.el);

    var stage = el('div', { class: 've-hero__stage' });
    fig.stage.appendChild(stage);
    var plotWrap = el('div', { class: 've-hero__plot' });
    stage.appendChild(plotWrap);

    // an info chip (lib/ui.js disclosure primitive)
    function markCard(b, where) {
      var card = b && document.getElementById(b.getAttribute('popovertarget'));
      if (card) card.dataset.heroPlace = where;
    }
    function infoChip(title, node, label) {
      var b = el('button', { type: 'button', class: 'v2-info ve-hero__i', 'aria-label': label || title, html: ui.INFO_SVG || 'i' });
      ui.info(b, { title: title, node: node, parent: root });
      return b;
    }

    // ---- tally panel
    var tally = el('div', { class: 've-hero__tally', 'aria-live': 'polite' });
    stage.appendChild(tally);

    function splitBar() {
      var a = el('i', { class: 've-hero__seg ve-hero__seg--a' }), b = el('i', { class: 've-hero__seg ve-hero__seg--b' });
      return { el: el('div', { class: 've-hero__split', 'aria-hidden': 'true' }, a, b), a: a, b: b };
    }
    function oddsHTML(t) {   // "1.5 to 1" with a quiet italic "to" (the numerals carry the reading)
      return String(t).replace(' to ', ' <i>to</i> ');
    }
    var modelBar = splitBar();
    var modelOdds = el('span', { class: 've-hero__odds', html: oddsHTML('9 to 1') });
    var modelSec = el('section', { class: 've-hero__row ve-hero__row--model' },
      el('div', { class: 've-label', text: 'Model: odds for A' }),
      modelOdds,
      modelBar.el,
      el('div', { class: 've-hero__sub', text: fmt(qEq * 100, { digits: 0 }) + '% end in A' }));

    var chainBar = splitBar();
    var chainOdds = el('span', { class: 've-hero__odds', text: '–' });
    var chainSub = el('div', { class: 've-hero__sub' });
    var chainSec = el('section', { class: 've-hero__row ve-hero__row--runs' },
      el('div', { class: 've-label', text: 'Runs: odds for A' }),
      chainOdds,
      chainBar.el, chainSub);

    // ---- c: readout with the gilt reticle, a hairline meter, and the guess-first table
    var cVal = el('span', { class: 've-hero__cval', text: '–' });
    var cFill = el('i', { class: 've-hero__cfill' });
    var cDot = el('i', { class: 've-hero__cdot' });
    var cCI = el('i', { class: 've-hero__cci' });
    var cLaw = el('i', { class: 've-hero__claw' });
    var cLawLab = el('span', { class: 've-hero__clawlab' });
    var tbl = el('table', { class: 've-hero__tbl' });
    var thead = el('tr', null, el('th', { text: 'sweeps' }));
    var lawRow = el('tr', null, el('th', { class: 've-hero__tbl-law', text: 'formula' }));
    var measRow = el('tr', null, el('th', { text: 'here' }));
    var measCells = [], cols = [];
    SETTINGS.forEach(function (s) {
      var h = el('td', { text: s.mult }), l = el('td', { class: 've-hero__tbl-law', text: fmt(s.law, { digits: 2 }) }), m = el('td', { text: '–' });
      thead.appendChild(h); lawRow.appendChild(l); measRow.appendChild(m);
      measCells.push(m); cols.push([h, l, m]);
    });
    tbl.appendChild(thead); tbl.appendChild(lawRow); tbl.appendChild(measRow);
    var tblHint = el('div', { class: 've-hero__tblhint', text: '? = try it first' });
    var cSub = el('div', { class: 've-hero__sub ve-hero__csub' });
    var cOdds = el('div', { class: 've-hero__codds' });

    // what the c readout means: one interaction away (the law's odds line in it is live)
    var cardLaw = el('span');
    var cCard = el('div', null,
      el('p', { html: '<i>c</i> = logit(<i>q</i><sub>runs</sub>) / logit(<i>q</i><sub>model</sub>): the share of the model’s <span data-term="logOdds">log-odds</span> that the runs keep. 0 is a coin flip; 1 keeps the model’s preference whole.' }),
      el('p', { html: 'On odds it reads as a power: the runs’ odds are 9<sup><i>c</i></sup>. The dashed mark is the <span data-term="selectionFormula">selection formula</span> (Eq. 3). It gives the chance of the favoured sign from the field, the size and the cooling rate; divided by the model’s own log-odds it gives <i>c</i>, with nothing fitted.' }),
      el('p', null, cardLaw),
      el('p', null, 'The whisker under the meter is the 95% interval of the measured ', el('i', { text: 'c' }), '. The value settles as more runs finish; it is shown once 3,000 runs are in, because a first few hundred runs are too noisy to compare with the formula.'));
    var cInfo = infoChip('The retained share <i>c</i>', cCard, 'What c means');
    markCard(cInfo, 'side');   // opens beside the tally column, so the meter it explains stays visible
    var cSec = el('section', { class: 've-hero__row ve-hero__row--c' },
      el('div', { class: 've-label ve-hero__clab' }, el('span', { text: 'Retained share' }), cInfo),
      el('div', { class: 've-hero__cline' },
        el('span', { class: 've-hero__cbox' }, el('span', { class: 've-hero__ceq', html: '<i>c</i>' }), cVal),
        cLawLab),
      cOdds,
      el('div', { class: 've-hero__cmeter', 'aria-hidden': 'true' },
        el('div', { class: 've-hero__ctrack' }, cFill, cCI, cLaw, cDot),
        el('div', { class: 've-hero__cscale' }, el('span', { text: '0 · coin flip' }), el('span', { text: '1 · keeps it all' }))),
      tbl,
      tblHint,
      cSub);
    tally.appendChild(modelSec);
    tally.appendChild(chainSec);
    tally.appendChild(cSec);

    // ---- controls
    var toggle = ui.toggle({
      id: ui.uid('hero-speed'),
      label: 'Sweeps per run',
      options: SETTINGS.map(function (s) { return { value: s.key, label: s.label + ' · ' + s.mult, title: s.sweeps + ' sweeps from β = 0.6 to 1.6' }; }),
      value: cur.key,
      onChange: function (v) { setSetting(v); }
    });
    var sweepInfo = infoChip('Sweeps per run', el('div', null,
      el('p', { text: 'One sweep is N single-spin updates, about one per variable. Each cooling run spends this many sweeps on a linear ramp from β\u00a0=\u00a00.6 (hot) to β\u00a0=\u00a01.6 (cold).' }),
      el('p', { text: '×4 and ×16 are four and sixteen times the compute. The formula’s value for a setting stays hidden until you try it, so you can guess first how much slower cooling helps.' })), 'About sweeps per run');
    markCard(sweepInfo, 'above');   // opens above its trigger when it fits, not over the plot it explains
    var tLab = toggle.el.querySelector('.ve-toggle__label');
    if (sweepInfo && tLab) { tLab.classList.add('ve-hero__tlab'); tLab.appendChild(sweepInfo); }
    var playBtn = ui.button({ icon: 'pause', kind: 'icon', title: 'Pause', onClick: function () { togglePlay(); } });
    playBtn.classList.add('ve-hero__pbtn');
    var status = el('span', { class: 've-hero__status' });
    var ctl = el('div', { class: 've-hero__controls' }, toggle.el, el('div', { class: 've-hero__play' }, playBtn, status));
    fig.controls.remove();

    // the "Reading the figure" text, one paragraph per element: the chapter's card mirrors each
    // .ve-hero__caption into its own paragraph, as plain text (so no sub/sup here). Only the two
    // "how to read" paragraphs live here; "why the preference is lost" is the chapter's Why drawer.
    var NB = '\u00a0';
    var captions = [
      'Each line is one sampler run, read left to right: β is the inverse temperature, from hot (0.6) to cold (1.6), and m says which mode the run is in. ' +
        'Runs stay grey while undecided and take their mode’s colour once they choose a sign near β' + NB + '=' + NB + '1. The dashed curves are the model’s stable states; mode B’s branch appears only after the split.',
      'The bars at the right edge count where the runs are now. The dotted outline is where the same number of runs would sit if they followed the model exactly at this β; after a cooling run, “model 270” is that count. The model sends nine runs in ten to mode A.'
    ].map(function (t) { return el('p', { class: 've-hero__caption', text: t }); });
    var caption = el('div', { class: 've-hero__captions' });
    captions.forEach(function (p) { caption.appendChild(p); });
    fig.caption.replaceWith(caption);
    fig.caption = caption;

    var takeKey = '';
    function paintTakeaway() {   // bound to the live count, so the text never disagrees with the readout above it
      var st = store[cur.key], c = cOf(st), lawTxt = fmt(cur.law, { digits: 2 });
      var key = cur.key + '|' + (isFinite(c) ? fmt(c, { digits: 2 }) : '') + '|' + oddsText(st.a, st.b);
      if (key === takeKey) return; takeKey = key;
      fig.setTakeaway('The model says 9 to 1; runs of ' + fmt(cur.sweeps) + ' sweeps ' +
        (isFinite(c) ? 'end about ' + oddsText(st.a, st.b) + '. The formula predicts they keep <i>c</i> = ' + lawTxt + ' of its <span data-term="logOdds">log-odds</span>; the runs so far measure ' + fmt(c, { digits: 2 }) + '.'
                     : 'are still being counted. The formula predicts they keep only <i>c</i> = ' + lawTxt + ' of its <span data-term="logOdds">log-odds</span>.'));
    }
    // every item is "key: value", so the spec card renders one even key/value table
    fig.setNote('Model: two-mode Curie–Weiss, N = 256 (the paper uses 2,048) · Sampler: exact Gibbs, continuous time (one sweep = N single-spin updates at random times), live in your browser · Schedule: linear ramp, β = 0.6 → 1.6 · ' + 'Runs: ' + CHAINS + ' cooling runs per batch; one batch is drawn, and undrawn batches are added in the background until ' + WARM + ' batches (' + fmt(WARM * CHAINS, { group: true }) + ' runs) are counted, and each drawn batch adds ' + CHAINS + ' more · ' + 'c shown: once ' + fmt(MIN_DESC * CHAINS, { group: true }) + ' runs are counted; the count keeps growing with each batch · ' + 'Lines: smoothed over 0.015 in β; dots and bars are exact', 'live');
    // the "is this the paper's c?" note is the chapter's margin note (index.html), visible at rest

    ui.terms(root);

    /* ------------------------------------------------------- simulation */
    // hist holds m of every drawn run at every level; avail = last level present (-1: none yet).
    // L = level on screen (it follows a fixed visual pace and never runs ahead of avail).
    var hist = new Float32Array(CHAINS * (STEPS + 1));
    var L = 0, avail = -1, phase = 'run', hold = 0, levelClock = 0, seed = 1001, runId = 0;
    var stillMode = false;   // reduced motion: show the finished descent without animating it

    function countDescent(a, key) {
      var st = store[key];
      st.a += a; st.b += CHAINS - a; st.runs++;
      updateTally();
    }
    function finishRun() {
      var a = 0;
      for (var c = 0; c < CHAINS; c++) if (hist[c * (STEPS + 1) + STEPS] > 0) a++;
      countDescent(a, cur.key); phase = 'hold'; hold = HOLD_S;
    }
    function putLevel(l, m) {
      for (var c = 0; c < CHAINS; c++) hist[c * (STEPS + 1) + l] = m[c];
      avail = l;
    }

    // ---- Worker engine: the shown descent and the background descents each run as PARTS Workers of
    // CHAINS / PARTS runs, so the main thread only draws
    var PARTS = 2, PER = CHAINS / PARTS;
    var simSrc = (function () { var sc = document.querySelector('script[src*="lib/sim.js"]'); return sc && sc.src; })();
    var WORKER_JS =
      'self.onmessage=function(e){var d=e.data,i;if(d.sim){importScripts(d.sim);return;}' +
      'var ch=new self.VE.sim.CWChains(d.n,d.N,{seed:d.seed}),per=d.db*d.sweeps,a=0;' +
      'function rec(l){if(!d.rec)return;var m=new Float32Array(d.n);for(i=0;i<d.n;i++)m[i]=2*ch.k[i]/d.N-1;self.postMessage({id:d.id,l:l,m:m},[m.buffer]);}' +
      'rec(0);for(var l=0;l<d.steps;l++){var b=d.b0+(l+0.5)*d.db;ch.advance(b,b*d.G,per);rec(l+1);}' +
      'for(i=0;i<d.n;i++)if(2*ch.k[i]>d.N)a++;self.postMessage({id:d.id,done:1,a:a});};';
    var wURL = null, useWorkers = false, wDraw = [], wBg = [], drawLvl = [], bgJob = null;
    try {
      if (simSrc && window.Worker && window.Blob && window.URL) {
        wURL = URL.createObjectURL(new Blob([WORKER_JS], { type: 'text/javascript' }));
        useWorkers = true;
      }
    } catch (e) { useWorkers = false; }
    function job(id, sd, rec) { return { id: id, n: PER, N: N, seed: sd, sweeps: cur.sweeps, db: DB, b0: B0, steps: STEPS, G: G, rec: rec }; }
    function spawn(onmsg) {
      var w = new Worker(wURL);
      w.onmessage = onmsg;
      w.onerror = function (e) { if (e && e.preventDefault) e.preventDefault(); fallBack(); };
      w.postMessage({ sim: simSrc });
      return w;
    }
    function onDrawMsg(part, e) {
      var d = e.data;
      if (d.id !== runId || d.done) return;
      for (var c = 0; c < PER; c++) hist[(part * PER + c) * (STEPS + 1) + d.l] = d.m[c];
      drawLvl[part] = d.l;
      avail = Math.min.apply(null, drawLvl);
      if (stillMode && avail === STEPS && phase === 'run') { L = STEPS; finishRun(); cv.redraw(); paintStatus(); }
    }
    function onBgMsg(e) {
      var d = e.data;
      if (!d.done || !bgJob || d.id !== bgJob.id) return;
      bgJob.a += d.a;
      if (--bgJob.left > 0) return;
      var key = bgJob.key, a = bgJob.a; bgJob = null;
      countDescent(a, key);
      pumpBg();
    }
    function bgWanted() { return store[cur.key].runs + (phase === 'run' ? 1 : 0) + (bgJob ? 1 : 0) < WARM; }
    function pumpBg() {
      if (!useWorkers || bgJob || !bgWanted()) return;
      bgJob = { id: 'bg' + (seed++), key: cur.key, left: PARTS, a: 0 };
      for (var p = 0; p < PARTS; p++) {
        if (!wBg[p]) wBg[p] = spawn(onBgMsg);
        wBg[p].postMessage(job(bgJob.id, 500000 + 1000 * seed + p, false));
      }
    }
    function stopWorkers() {
      wDraw.concat(wBg).forEach(function (w) { if (w) w.terminate(); });
      wDraw = []; wBg = []; bgJob = null;
    }
    function fallBack() {   // a Worker failed to start: run the same kernel on the main thread
      if (!useWorkers) return;
      useWorkers = false; stopWorkers();
      newRun();
      if (stillMode) runStill();
    }

    // ---- main-thread fallback (same kernel, in short slices)
    var chains = null, rem = 0, bg = null, mBuf = new Float64Array(CHAINS);
    function sliceLive() {
      var b = B0 + (avail + 0.5) * DB, d = Math.min(rem, SLICE_SWEEPS);
      chains.advance(b, b * G, d);
      rem -= d;
      if (rem <= 1e-9) { putLevel(avail + 1, chains.magnetisations(mBuf)); rem = DB * cur.sweeps; }
    }
    function sliceBg() {
      if (!bg || bg.key !== cur.key) bg = { ch: new S.CWChains(CHAINS, N, { seed: 500000 + seed++ }), l: 0, rem: DB * cur.sweeps, key: cur.key };
      var b = B0 + (bg.l + 0.5) * DB, d = Math.min(bg.rem, SLICE_SWEEPS);
      bg.ch.advance(b, b * G, d);
      bg.rem -= d;
      if (bg.rem <= 1e-9) {
        bg.l++; bg.rem = DB * cur.sweeps;
        if (bg.l >= STEPS) { var a = 0, k = bg.ch.k, key = bg.key; bg = null; for (var c = 0; c < CHAINS; c++) if (2 * k[c] > N) a++; countDescent(a, key); }
      }
    }

    function newRun() {
      runId++; L = 0; avail = -1; levelClock = 0; phase = 'run';
      if (useWorkers) {
        try {
          for (var p = 0; p < PARTS; p++) {
            if (!wDraw[p]) wDraw[p] = spawn(onDrawMsg.bind(null, p));
            drawLvl[p] = -1;
            wDraw[p].postMessage(job(runId, 1000 * seed + p, true));
          }
          seed++;
          pumpBg();
          return;
        } catch (e) { useWorkers = false; stopWorkers(); }   // no Workers here: use the main thread
      }
      chains = new S.CWChains(CHAINS, N, { seed: seed++ });
      rem = DB * cur.sweeps;
      putLevel(0, chains.magnetisations(mBuf));
    }

    /* ------------------------------------------------------------ plot */
    // [{b, up, dn, z}] stable states for the dashed guides; filled in short slices after mount
    var stable = [], stableSignal = { cancelled: false };
    S.runChunked((function* () {
      for (var i = 0; i <= 200; i++) {
        var b = B0 + (B1 - B0) * i / 200, pts = S.stationaryPoints(b, b * G), up = null, dn = null, z = null;
        pts.forEach(function (p) { if (!p.stable) return; if (p.m > 0.02) up = p.m; else if (p.m < -0.02) dn = p.m; else z = p.m; });
        stable.push({ b: b, up: up, dn: dn, z: z });
        yield 0;
      }
    })(), { budgetMs: 4, signal: stableSignal }).then(function () { cv.redraw(); }).catch(function () {});
    function mStar(b) {   // distance of the stable branches from 0 at this beta (0 before the split)
      var i = Math.max(0, Math.min(200, Math.round((b - B0) / (B1 - B0) * 200))), s = stable[i];
      return s && s.up != null && s.dn != null ? (s.up - s.dn) / 2 : 0;
    }
    // how committed a run at m is at beta: 0 = undecided (grey), 1 = in a mode (full colour)
    function commitW(m, b) {
      if (b < 1) return 0;
      var ms = mStar(b);
      return smooth(1, 1.1, b) * clamp01((Math.abs(m) - 0.1) / Math.max(0.15, 0.6 * ms - 0.1));
    }

    // created detached and inserted afterwards, so its first size and draw run in the next
    // animation frame instead of forcing a layout inside mount
    var cv = ui.canvas(null, {
      height: function (w) { return w < 520 ? Math.round(w * 0.86) : Math.round(Math.min(440, w * 0.6)); },
      label: 'Sampler runs: which mode each run is in, m, against beta, from 0.6 (hot) to 1.6 (cold). Runs start at m near 0 and split into mode A (m > 0) and mode B (m < 0) near beta = 1.',
      draw: draw
    });
    plotWrap.appendChild(cv.wrap);
    fig.legend.classList.add('ve-hero__legend');
    plotWrap.appendChild(fig.legend);
    plotWrap.appendChild(ctl);

    /* phone: the chapter's "Reading the figure" card opens as a bottom sheet (capped at 45vh in
       hero.css). Before it covers the page, scroll the plot into the free space above it, so the
       reader can look at the curves while reading how to read them. */
    (function () {
      var ch = host.closest && host.closest('.v2-ch');
      var card = ch && ch.querySelector('.v2-card[id$="-read"]');
      if (!card) return;
      card.addEventListener('toggle', function (e) {
        if (e.newState !== 'open' || window.innerWidth >= 900) return;
        requestAnimationFrame(function () {
          var r = cv.wrap.getBoundingClientRect();
          var top = 60, sheetTop = card.getBoundingClientRect().top || window.innerHeight * 0.55;
          if (r.top >= top && r.bottom <= sheetTop - 8) return;
          window.scrollBy({ top: r.top - top, behavior: ui.reducedMotion ? 'auto' : 'smooth' });
        });
      });
    })();

    // the model's own split at level l: CHAINS x the exact stationary law, binned like the histogram
    var ghostCache = {};
    function ghostAt(l) {
      if (ghostCache[l]) return ghostCache[l];
      var b = B0 + l * DB, pi = S.cwPi(N, b, b * G), Hg = new Float64Array(40), a = 0;
      for (var k = 0; k <= N; k++) {
        var mk = 2 * k / N - 1, bi = Math.min(39, Math.max(0, Math.floor((mk + 1) / 2 * 40)));
        Hg[bi] += CHAINS * pi[k];
        if (2 * k > N) a += CHAINS * pi[k];
      }
      return (ghostCache[l] = { H: Hg, a: a });
    }
    var layer = { cv: document.createElement('canvas'), key: '', upTo: 0 };
    layer.ctx = layer.cv.getContext('2d');
    function draw(ctx, w, h, col, api) {
      var narrow = w < 520;
      var P = ui.inset(w, h, { top: 40, right: narrow ? 84 : 124, bottom: 40, left: narrow ? 34 : 44 });
      var sx = ui.scale.linear([B0, B1], [P.x0, P.x1]);
      var sy = ui.scale.linear([-1, 1], [P.y1, P.y0]);
      var cA = parseColor(col.modeA), cB = parseColor(col.modeB), cN = parseColor(col.noise);
      var T = function (str, x, y, o) { o.c = col; ui.plot.text(ctx, str, x, y, o); };
      var bNow = B0 + L * DB, bx = sx(bNow);

      // undecided region wash (solid to beta = 1, then fading out over the commit window)
      var wash = ctx.createLinearGradient(sx(1), 0, sx(1.08), 0);
      ctx.save();
      ctx.globalAlpha = 0.55;
      ctx.fillStyle = col.noiseWash;
      ctx.fillRect(P.x0, P.y0, sx(1) - P.x0, P.y1 - P.y0);
      ctx.restore();
      var wc = parseColor(col.noise);
      wash.addColorStop(0, mix(wc, wc, 0, 0.11)); wash.addColorStop(1, mix(wc, wc, 0, 0));
      ctx.fillStyle = wash;
      ctx.fillRect(sx(1), P.y0, sx(1.08) - sx(1), P.y1 - P.y0);

      ui.plot.axis(ctx, sx, { side: 'bottom', at: P.y1, ticks: [0.6, 0.8, 1, 1.2, 1.4, 1.6], format: function (v) { return fmt(v, { digits: 1 }); },
        label: narrow ? 'β  (hot ← → cold)' : 'β, inverse temperature   (hot on the left, cold on the right)', grid: [P.y0, P.y1], c: col });
      ui.plot.axis(ctx, sy, { side: 'left', at: P.x0, ticks: [-1, -0.5, 0, 0.5, 1], format: function (v) { return v === 0 ? '0' : fmt(v, { digits: 1 }); }, grid: [P.x0, P.x1], c: col });
      T(narrow ? 'm: which mode (+1 = A, −1 = B)' : 'm: which mode the sample is in  (+1 = all A, −1 = all B)', P.x0 - (narrow ? 30 : 40), P.y0 - 28, { size: 11, color: col.ink2, align: 'left', baseline: 'middle' });

      // the transition, beta = 1
      ui.plot.vline(ctx, 1, sx, P.y0, P.y1, { color: col.ink3, dash: [4, 4], width: 1 });
      // right-aligned against the line, in the empty top-left of the undecided region; the longer
      // version is used only where every line fits between the y-axis and the line
      var cLines = ['transition, β = 1', 'modes split near here;', 'runs choose near it,', 'then mostly stay'];
      ctx.save(); ctx.font = ui.font('mono', 11, 600);
      var room = sx(1) - 6 - (P.x0 + 6), fits = !narrow && cLines.every(function (s) { return ctx.measureText(s).width <= room; });
      ctx.restore();
      if (!fits) cLines = ['β = 1'];
      var cMuted = avail >= 0 && bNow > 1.04;   // once the traces have passed the transition, the label steps back

      cLines.forEach(function (s, i) {
        T(s, sx(1) - 6, sy(0.94) + 14 * i, { size: 11, weight: i ? 400 : 600, color: cMuted ? col.ink3 : col.ink2, align: 'right', baseline: 'middle', halo: true });
      });

      // stable-state guides
      ctx.save();
      ctx.lineWidth = 1.3; ctx.strokeStyle = col.theory; ctx.globalAlpha = 0.8; ctx.setLineDash([5, 4]);   // theory is always dashed
      ['up', 'dn', 'z'].forEach(function (k) {
        ctx.beginPath(); var on = false;
        stable.forEach(function (s) {
          var v = s[k]; if (v == null) { on = false; return; }
          var x = sx(s.b), y = sy(v);
          if (!on) { ctx.moveTo(x, y); on = true; } else ctx.lineTo(x, y);
        });
        ctx.stroke();
      });
      ctx.restore();

      var stride = STEPS + 1;
      // a trail point is the run's m averaged over the last SMOOTH levels (at most 0.015 in β), so a slow
      // schedule (many sweeps per pixel) draws the run's path, not a burst of per-sweep jitter;
      // the dots and the side histogram always show the exact current values
      var sm = function (off, l) { var l0 = Math.max(0, l - SMOOTH + 1), a = 0; for (var q = l0; q <= l; q++) a += hist[off + q]; return a / (l - l0 + 1); };
      if (avail >= 0) {
        // trails: each segment is coloured by how committed the run is at that point (grey while
        // undecided). Finished segments live in an offscreen layer; only new ones are added per frame.
        var tstep = narrow ? 3 : 2, lw = narrow ? 0.9 : 1.1;
        var a0 = cur.sweeps > 250 ? 0.14 : 0.26;   // slow settings: many independent updates per level read as spikes, so undecided trails are lighter
        var shades = [cN, cA, cB].map(function (c0, i) { return [0, 1, 2, 3, 4].map(function (q) {
          return i === 0 ? mix(cN, cN, 0, a0) : mix(cN, c0, q / 4, a0 + (0.55 - a0) * q / 4); }); });
        var bucket = function (m, b) { var q = Math.round(commitW(m, b) * 4); return q === 0 ? [0, 0] : [m > 0 ? 1 : 2, q]; };
        var key = [w, h, api.dpr, col.modeA, col.modeB, col.noise, runId, a0].join('|');
        if (layer.key !== key) {
          layer.key = key; layer.upTo = 0;
          layer.cv.width = Math.round(w * api.dpr); layer.cv.height = Math.round(h * api.dpr);
          layer.ctx.setTransform(api.dpr, 0, 0, api.dpr, 0, 0);
        }
        var lastFull = Math.floor(L / tstep) * tstep;
        if (lastFull > layer.upTo) {
          var paths = {}, lc = layer.ctx;
          for (var t = 0; t < TRAILS; t++) {
            var off = Math.floor(t * CHAINS / TRAILS) * stride;
            for (var l1 = layer.upTo + tstep; l1 <= lastFull; l1 += tstep) {
              var m1 = sm(off, l1), bk = bucket(m1, B0 + l1 * DB), pk = bk[0] + ':' + bk[1];
              var pth = paths[pk] || (paths[pk] = { p: new Path2D(), c: shades[bk[0]][bk[1]] });
              pth.p.moveTo(sx(B0 + (l1 - tstep) * DB), sy(sm(off, l1 - tstep)));
              pth.p.lineTo(sx(B0 + l1 * DB), sy(m1));
            }
          }
          lc.lineWidth = lw; lc.lineCap = 'round';
          Object.keys(paths).forEach(function (pk) { lc.strokeStyle = paths[pk].c; lc.stroke(paths[pk].p); });
          layer.upTo = lastFull;
        }
        ctx.drawImage(layer.cv, 0, 0, w, h);
        if (L > layer.upTo) {   // the short live tail since the last finished segment
          ctx.lineWidth = lw; ctx.lineCap = 'round';
          for (var t2 = 0; t2 < TRAILS; t2++) {
            var o2 = Math.floor(t2 * CHAINS / TRAILS) * stride, mt = sm(o2, L), bt = bucket(mt, bNow);
            ctx.strokeStyle = shades[bt[0]][bt[1]];
            ctx.beginPath(); ctx.moveTo(sx(B0 + layer.upTo * DB), sy(sm(o2, layer.upTo))); ctx.lineTo(bx, sy(mt)); ctx.stroke();
          }
        }
        // heads: coloured only once a run has committed
        var r = narrow ? 1.8 : 2.2, heads = {};
        for (var k = 0; k < CHAINS; k++) {
          var m = hist[k * stride + L], hk = bucket(m, bNow), hkey = hk[0] + ':' + hk[1];
          var hp = heads[hkey] || (heads[hkey] = { p: new Path2D(), c: mix(cN, hk[0] === 2 ? cB : cA, hk[1] / 4, 0.9) });
          hp.p.moveTo(bx + r, sy(m)); hp.p.arc(bx, sy(m), r, 0, 6.2832);
        }
        Object.keys(heads).forEach(function (hk2) { ctx.fillStyle = heads[hk2].c; ctx.fill(heads[hk2].p); });
        // cursor
        ctx.strokeStyle = col.ink; ctx.globalAlpha = 0.35; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(bx + 0.5, P.y0); ctx.lineTo(bx + 0.5, P.y1); ctx.stroke(); ctx.globalAlpha = 1;
        var sweepsDone = Math.round(L * DB * cur.sweeps);
        var curLab = narrow ? 'β ' + fmt(bNow, { digits: 2 }) + ' · ' + fmt(sweepsDone) + '/' + fmt(cur.sweeps) + ' sweeps'
          : 'β = ' + fmt(bNow, { digits: 2 }) + ' · sweep ' + fmt(sweepsDone) + ' of ' + fmt(cur.sweeps);
        ctx.save(); ctx.font = ui.font('mono', 11, 400); var cw = ctx.measureText(curLab).width; ctx.restore();
        var right = bx > (P.x0 + P.x1) / 2;
        var lo = 2 + (right ? cw : 0), hi = w - 2 - (right ? 0 : cw);
        var curX = Math.min(Math.max(bx + (right ? -4 : 4), lo), hi);
        T(curLab, curX, P.y0 - 10, { size: 11, color: col.ink, align: right ? 'right' : 'left', baseline: 'middle', halo: true });
      }

      // "undecided" label, bottom-left where trails do not reach; drawn last, at full strength,
      // so its halo covers the cursor line when the cursor passes under it
      var uo = { size: 11, color: col.ink2, align: 'left', baseline: 'middle', halo: true };
      if (narrow) T('undecided', P.x0 + 8, sy(-0.9), uo);
      else { T('undecided:', P.x0 + 8, sy(-0.9) - 14, uo); T('either mode possible', P.x0 + 8, sy(-0.9), uo); }

      if (avail < 0) return;
      // side histogram of where the runs are now (right margin)
      var bins = 40, H = new Float64Array(bins), mx = 0;
      for (var q = 0; q < CHAINS; q++) {
        var mq = hist[q * stride + L], bi = Math.min(bins - 1, Math.max(0, Math.floor((mq + 1) / 2 * bins)));
        H[bi]++; if (H[bi] > mx) mx = H[bi];
      }
      var gh = ghostAt(L), GH = gh.H, gmx = 0;
      for (var g = 0; g < bins; g++) if (GH[g] > gmx) gmx = GH[g];
      var hx0 = P.x1 + 6, hw = (narrow ? 20 : 52), bh = (P.y1 - P.y0) / bins, hscale = hw / Math.max(mx, gmx, 30);
      for (var j = 0; j < bins; j++) {
        if (!H[j]) continue;
        var mid = -1 + (j + 0.5) * 2 / bins;
        ctx.fillStyle = mix(cN, mid > 0 ? cA : cB, commitW(mid, bNow), 0.85);
        ctx.fillRect(hx0, sy(-1 + (j + 1) * 2 / bins) + 0.5, H[j] * hscale, Math.max(1, bh - 1));
      }
      // ghost outline: where the same number of runs would sit if they followed the model exactly
      // at this beta (exact stationary law, VE.sim.cwPi)
      ctx.save();
      ctx.strokeStyle = col.ink; ctx.globalAlpha = 0.7; ctx.lineWidth = 1; ctx.setLineDash([2, 2]);
      ctx.beginPath();
      var open = false;
      for (var j2 = 0; j2 <= bins; j2++) {   // one outline per run of occupied bins, no line along empty ones
        var gv = j2 < bins ? GH[j2] : 0, yb = sy(-1 + j2 * 2 / bins);
        if (gv >= 0.5) {
          var nx = hx0 + gv * hscale;
          if (!open) { ctx.moveTo(hx0, yb); open = true; }
          ctx.lineTo(nx, yb); ctx.lineTo(nx, sy(-1 + (j2 + 1) * 2 / bins));
        } else if (open) { ctx.lineTo(hx0, yb); open = false; }
      }
      ctx.stroke();
      ctx.restore();
      // counts by mode: only once the drawn descent has finished, so 'model' is the final split
      if (L >= STEPS) {
        var nA = 0; for (var u = 0; u < CHAINS; u++) if (hist[u * stride + L] > 0) nA++;
        var lx = hx0 + hw + 4;
        ctx.save();
        T((narrow ? 'A ' : 'A: ') + nA, lx, sy(0.8), { size: 11, weight: 600, color: col.modeA, align: 'left', baseline: 'middle' });
        T((narrow ? 'B ' : 'B: ') + (CHAINS - nA), lx, sy(-0.8), { size: 11, weight: 600, color: col.modeB, align: 'left', baseline: 'middle' });
        var go = { size: narrow ? 10 : 11, color: col.ink2, align: 'left', baseline: 'middle' };
        T('model ' + Math.round(gh.a), lx, sy(0.8) + 15, go);
        T('model ' + Math.round(CHAINS - gh.a), lx, sy(-0.8) + 15, go);
        ctx.restore();
      }
    }

    /* ----------------------------------------------------------- tally */
    // c is shown only from MIN_DESC descents on: the first few hundred runs are too noisy to compare with the law
    function cOf(st) { return (st.runs >= MIN_DESC && st.a && st.b) ? S.logit(st.a / (st.a + st.b)) / LOGIT_EQ : NaN; }
    function updateTally() {
      var st = store[cur.key], n = st.a + st.b;
      modelBar.a.style.width = (qEq * 100) + '%'; modelBar.b.style.width = ((1 - qEq) * 100) + '%';
      var lawOdds = Math.pow(ODDS, cur.law);
      if (st.runs < MIN_DESC) {
        chainBar.a.style.width = '50%'; chainBar.b.style.width = '50%';
        chainBar.el.classList.add('is-empty');
        chainOdds.textContent = 'counting…';
        chainOdds.classList.add('is-wait');
        chainSub.textContent = fmt(n, { group: true }) + ' of ' + fmt(MIN_DESC * CHAINS, { group: true }) + ' runs';
        cVal.textContent = '…'; cVal.classList.add('is-wait');
        cFill.style.width = '0%'; cCI.style.display = 'none'; cDot.style.display = 'none';
        cSub.textContent = 'The formula needs no runs.';
        cSub.hidden = false;
        cOdds.innerHTML = 'runs’ odds = 9<sup><i>c</i></sup>';
      } else {
        chainBar.el.classList.remove('is-empty');
        chainBar.a.style.width = (st.a / n * 100) + '%'; chainBar.b.style.width = (st.b / n * 100) + '%';
        chainOdds.innerHTML = oddsHTML(oddsText(st.a, st.b));
        chainOdds.classList.remove('is-wait');
        chainSub.textContent = fmt(st.a, { group: true }) + ' / ' + fmt(n, { group: true }) + ' in A';
        var c = cOf(st);
        var ci = wilson(st.a, n).map(function (q) { return S.logit(q) / LOGIT_EQ; });
        cVal.textContent = isFinite(c) ? fmt(c, { digits: 2 }) : '–'; cVal.classList.remove('is-wait');
        cFill.style.width = (clamp01(c) * 100) + '%';
        cDot.style.display = ''; cDot.style.left = (clamp01(c) * 100) + '%';
        cCI.style.display = '';
        cCI.style.left = (Math.max(0, ci[0]) * 100) + '%';
        cCI.style.width = (Math.max(0, Math.min(1, ci[1]) - Math.max(0, ci[0])) * 100) + '%';
        cSub.textContent = ''; cSub.hidden = true;
        cOdds.innerHTML = 'runs’ odds = 9<sup>' + fmt(c, { digits: 2 }) + '</sup> ≈ ' + oddsText(st.a, st.b);
      }
      cardLaw.innerHTML = 'At ' + fmt(cur.sweeps) + ' sweeps the formula gives <i>c</i> = ' + fmt(cur.law, { digits: 2 }) +
        ', so odds of 9<sup>' + fmt(cur.law, { digits: 2 }) + '</sup> ≈ ' + fmt(lawOdds, { digits: 1 }) + ' to 1.';
      cLaw.style.left = (cur.law * 100) + '%';
      cLawLab.innerHTML = '<i class="ve-hero__dash" aria-hidden="true"></i>formula ' + fmt(cur.law, { digits: 2 });
      paintTakeaway();
      visited[cur.key] = true;
      SETTINGS.forEach(function (s, i) {
        cols[i][1].textContent = visited[s.key] ? fmt(s.law, { digits: 2 }) : '?';
        var c = cOf(store[s.key]);
        measCells[i].textContent = isFinite(c) ? fmt(c, { digits: 2 }) : (store[s.key].runs ? '…' : '–');
        cols[i].forEach(function (td) { td.classList.toggle('is-on', s === cur); });
      });
      // the sweeps/law/here table shows once a second setting is tried: at rest the column ends on
      // the c meter, and the ×1 law value is already beside the readout
      var tried = SETTINGS.filter(function (s) { return visited[s.key]; }).length;
      tbl.hidden = tried < 2;
      tblHint.hidden = tried < 2 || tried === SETTINGS.length;
    }

    /* ------------------------------------------------------------ loop */
    var playing = !ui.reducedMotion;
    var loop = ui.loop(function (dt) {
      var deadline = now() + FRAME_BUDGET_MS, behind = false;
      if (phase === 'hold') {
        hold -= dt;
        if (hold <= 0) newRun();
      } else {
        levelClock += dt * LEVELS_PER_S;
        var target = Math.min(STEPS, Math.floor(levelClock));
        if (!useWorkers) while (avail < target && now() < deadline) sliceLive();
        if (avail < target) { target = Math.max(L, avail); levelClock = target + 0.999; behind = true; }  // compute-limited: hold the clock
        L = target;
        if (L >= STEPS) finishRun();
      }
      if (!useWorkers && !behind) while (bgWanted() && now() < deadline) sliceBg();
      cv.redraw(); paintStatus();
    }, { el: root });

    function paintStatus() {
      var txt;
      if (!playing) txt = 'paused';
      else if (phase === 'hold') txt = 'done · next in ' + Math.max(1, Math.ceil(hold)) + ' s';
      else txt = 'cooling · ' + Math.round(L / STEPS * 100) + '%';
      if (status.textContent !== txt) status.textContent = txt;
    }
    function setPlayIcon() {
      playBtn.replaceChildren(ui.icon(playing ? 'pause' : 'play'));
      playBtn.title = playing ? 'Pause' : 'Play';
      playBtn.setAttribute('aria-label', playBtn.title);
    }
    function togglePlay() {
      playing = !playing;
      if (playing) { stillMode = false; loop.start(); } else loop.stop();
      setPlayIcon(); paintStatus();
    }
    var stillSignal = { cancelled: false };
    function runStill() {   // reduced motion, main-thread fallback: one finished descent, in slices
      var gen = (function* () {
        while (avail < STEPS) { sliceLive(); yield 0; }
        L = STEPS; finishRun(); cv.redraw();
        while (bgWanted()) { sliceBg(); yield 0; }
      })();
      S.runChunked(gen, { budgetMs: 10, signal: stillSignal }).then(function () { cv.redraw(); paintStatus(); }).catch(function () {});
    }
    function setSetting(key) {
      stillSignal.cancelled = true; stillMode = false;
      cur = SETTINGS.filter(function (s) { return s.key === key; })[0];
      bg = null;
      stopWorkers();
      newRun();
      updateTally();
      playing = true; loop.start(); setPlayIcon();
      cv.redraw(); paintStatus();
    }

    /* The first descent starts after the page's first frame, in its own task, so starting the
       Workers never shares a task with the page's first layout. Until then the plot shows its
       axes and the clock waits (avail = -1). */
    if (useWorkers) requestAnimationFrame(function () { setTimeout(function () { if (runId === 0) newRun(); }, 0); });
    else newRun();
    updateTally();
    setPlayIcon(); paintStatus();
    if (playing) loop.start();   // the first descent animates live from the first frame
    else {                       // reduced motion: no animation; show one finished descent
      stillMode = true;
      if (!useWorkers) runStill();
    }

    return {
      el: root,
      destroy: function () { stillSignal.cancelled = true; stableSignal.cancelled = true; stopWorkers(); if (wURL) URL.revokeObjectURL(wURL); loop.destroy(); cv.destroy(); root.remove(); },
      _state: function () { return { setting: cur.key, store: store, L: L, avail: avail, workers: useWorkers, phase: phase, law: SETTINGS.map(function (s) { return s.law; }), G: G, qEq: qEq }; }
    };
  }

  /* ------------------------------------------------------------------
     Overview chapter helpers (they act only when #ch-overview exists).

     1. Card fit (desktop). lib/ui.js places a card under its trigger and caps its height to the
        room left, so at 1280 x 900 the "What runs here" card lost its last row with no cue.
        After that placement (next frame, before paint) a card opened from this chapter is
        (a) moved beside the tally column or above its trigger when it asks for that, and
        (b) otherwise, if it still overflows, the page scrolls by the missing height (keeping the
            trigger on screen) and the card moves up with it, so the whole card shows.
        hero.css adds a bottom scroll shadow for the case the page cannot scroll far enough.
     2. Frontispiece first paint. lib/shell.js fast-forwards the frontispiece emblem 10 cycles
        in idle slices and paints only at the end (3-5 s of blank canvas). Here the emblem is
        painted at once (axes, commit tick, spin strip) and then every frame while the
        fast-forward runs, so the traces build up visibly. With reduced motion it is painted at
        once and then at most twice a second (discrete frames, no animation).
     ------------------------------------------------------------------ */
  function heroTriggerOf(card) {
    return document.querySelector('#ch-overview [popovertarget="' + card.id + '"]');
  }
  function fitCard(card) {
    if (!card.matches(':popover-open') || card.classList.contains('is-sheet')) return;
    var inv = heroTriggerOf(card);
    if (!inv) return;
    var vh = window.innerHeight, r = inv.getBoundingClientRect();
    var where = card.dataset.heroPlace;
    if (where) {
      card.style.maxHeight = '';
      var w = card.offsetWidth, h = Math.min(card.scrollHeight, vh - 32);
      if (where === 'side') {
        var col = inv.closest('.ve-hero__tally');
        var cr = col && col.getBoundingClientRect();
        if (cr && cr.left - w - 20 >= 16) {
          card.style.left = (cr.left - w - 20) + 'px';
          card.style.top = Math.max(16, Math.min(vh - h - 16, r.top - 14)) + 'px';
          if (card.scrollHeight > vh - 32) card.style.maxHeight = (vh - 32) + 'px';
          return;
        }
      } else if (where === 'above' && r.top - 8 - 16 >= card.scrollHeight) {
        card.style.top = (r.top - 8 - card.scrollHeight) + 'px';
        return;
      }
      card.style.maxHeight = parseFloat(card.dataset.heroMax || '') ? card.dataset.heroMax : '';
    }
    var full = card.scrollHeight;
    if (full <= card.clientHeight + 1) return;   // fits as placed
    // prefer below the trigger: scroll the page up to the missing height, keeping the trigger
    // at least 24 px from the top; then place below, or above if that has more room
    var below = vh - r.bottom - 8 - 16;
    if (full > below) window.scrollTo({ top: window.scrollY + Math.min(full - below, Math.max(0, r.top - 24)), behavior: 'instant' });
    r = inv.getBoundingClientRect();
    below = vh - r.bottom - 8 - 16;
    var above = r.top - 8 - 16;
    if (full <= below || below >= above) {
      card.style.top = (r.bottom + 8) + 'px';
      card.style.maxHeight = full <= below ? '' : Math.max(120, below) + 'px';
    } else {
      var h2 = Math.min(full, above);
      card.style.top = (r.top - 8 - h2) + 'px';
      card.style.maxHeight = full <= above ? '' : above + 'px';
    }
  }
  function heroPageHelpers() {
    if (!document.getElementById('ch-overview')) return;
    /* A popover's 'toggle' event is async, so the browser can paint one frame of the open card
       before lib/ui.js placeCard runs, and a second before fitCard runs. With normal motion those
       frames sit at opacity 0; with reduced motion they flash. So: hide the card synchronously in
       'beforetoggle' (fired inside showPopover, before any paint), then in the card's own 'toggle'
       listener (added after lib/ui.js's, so it runs after placeCard) fit it and reveal it in the
       same task. Layout still works under visibility:hidden, so the measurements are exact. */
    document.addEventListener('beforetoggle', function (e) {
      var card = e.target;
      if (!card.classList || !card.classList.contains('v2-card') || !heroTriggerOf(card)) return;
      if (e.newState !== 'open') { card.style.visibility = ''; return; }
      card.style.visibility = 'hidden';
      if (card._heroFit) return;
      card._heroFit = true;
      card.addEventListener('toggle', function (ev) {
        if (ev.newState === 'open') {
          card.dataset.heroMax = card.style.maxHeight || '';
          try { fitCard(card); } catch (err) { console.error(err); }
        }
        card.style.visibility = '';
      });
    }, true);

    var fc = document.querySelector('canvas[data-emblem="front"]');
    var em = fc && VE.shell && VE.shell.emblems && VE.shell.emblems.filter(function (e) { return e.o && e.o.canvases && e.o.canvases.indexOf(fc) >= 0; })[0];
    if (!em || typeof em.paint !== 'function') return;
    var started = false, start0 = em.start;
    em.start = function () { started = true; return start0.apply(this, arguments); };
    var still = !!(VE.ui && VE.ui.reducedMotion), last = 0, t0 = performance.now();
    em.paint();
    (function frame(t) {
      if (started || em.on || performance.now() - t0 > 20000) return;
      if (fc.offsetParent !== null && (!still || t - last >= 500)) { last = t || 0; em.paint(); }
      requestAnimationFrame(frame);
    })(0);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', heroPageHelpers);
  else setTimeout(heroPageHelpers, 0);

  VE.components.hero = { mount: mount };
})();
