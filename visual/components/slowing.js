/* components/slowing.js — "Relaxation is slowest near the transition"
 *
 * Panels in one plate:
 *   1. Kick: a chain released from m = 0.9 slides downhill on the large-N potential
 *      F(m) = m²/2 − log cosh(βm)/β (its slope is the large-N drift −m + tanh βm of App. A),
 *      next to its distance-to-rest on a log scale. Live, deterministic, no sampling noise.
 *   2. Relaxation time τ(β) for N = 256 … 16,384: inverse spectral gap of the exact chain held on
 *      one sign (reflected at m = 0), zero field; data/tracking_time.json (docs/relaxation-times.md).
 *      Peak height and half-height width are read off the data.
 * One β state is shared by panels 1 and 2.
 */
(function () {
  'use strict';
  var VE = (window.VE = window.VE || {});
  VE.components = VE.components || {};

  var M0 = 0.9;          // release point (same for every β)
  var T_MAX = 40;        // sweeps shown in panel 1
  var TOL = 0.01;        // "settled" threshold on |m − m_rest|
  var SPEED = 8;         // sweeps per second of animation
  var REFS = [0.7, 1.0, 1.3];
  var NS = [256, 1024, 4096, 16384];

  function restPoint(sim, beta) {
    if (beta <= 1) return 0;
    var pts = sim.stationaryPoints(beta, 0).filter(function (p) { return p.stable && p.m > 0; });
    return pts.length ? pts[0].m : 0;
  }
  function pullRate(beta, ms) { var t = Math.tanh(beta * ms); return 1 - beta * (1 - t * t); }

  function rk4(sim, m, beta, dt) {
    var f = sim.drift;
    var k1 = f(m, beta), k2 = f(m + 0.5 * dt * k1, beta), k3 = f(m + 0.5 * dt * k2, beta), k4 = f(m + dt * k3, beta);
    return m + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
  }
  // Trajectory sampled every 0.05 sweeps on [0, T_MAX].
  function trajectory(sim, beta) {
    var ms = restPoint(sim, beta), m = M0, dt = 0.01, n = Math.round(T_MAX / 0.05);
    var ts = new Float64Array(n + 1), mm = new Float64Array(n + 1), dd = new Float64Array(n + 1);
    for (var i = 0; i <= n; i++) {
      ts[i] = i * 0.05; mm[i] = m; dd[i] = Math.abs(m - ms);
      for (var j = 0; j < 5; j++) m = rk4(sim, m, beta, dt);
    }
    return { beta: beta, rest: ms, t: ts, m: mm, d: dd, settle: settleTime(sim, beta, ms) };
  }
  // Sweeps until |m − m_rest| < TOL, adaptive RK4 (step limited to ~5% change of the distance).
  function settleTime(sim, beta, ms) {
    var m = M0, t = 0, guard = 0;
    while (Math.abs(m - ms) > TOL && t < 1e7 && guard++ < 200000) {
      var d = Math.abs(m - ms), f = Math.abs(sim.drift(m, beta)) || 1e-12;
      var dt = Math.max(0.005, Math.min(0.05 * d / f, 1e5));
      var mNew = rk4(sim, m, beta, dt), dNew = Math.abs(mNew - ms);
      // Stop inside the last step: interpolate the crossing of TOL instead of counting the whole step.
      if (dNew <= TOL && d > dNew) { t += dt * (d - TOL) / (d - dNew); m = mNew; break; }
      m = mNew; t += dt;
    }
    return t;
  }
  function interp(xs, ys, x) {
    if (x <= xs[0]) return ys[0];
    var n = xs.length;
    if (x >= xs[n - 1]) return ys[n - 1];
    var lo = 0, hi = n - 1;
    while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (xs[mid] <= x) lo = mid; else hi = mid; }
    var u = (x - xs[lo]) / (xs[hi] - xs[lo]);
    return ys[lo] + u * (ys[hi] - ys[lo]);
  }
  function curveStats(beta, tau) {
    var i0 = 0;
    for (var i = 1; i < tau.length; i++) if (tau[i] > tau[i0]) i0 = i;
    var peak = tau[i0], half = peak / 2;
    var a = i0, b = i0;
    while (a > 0 && tau[a - 1] >= half) a--;
    while (b < tau.length - 1 && tau[b + 1] >= half) b++;
    // linear interpolation of the half-height crossings
    var xa = a > 0 ? beta[a - 1] + (half - tau[a - 1]) / (tau[a] - tau[a - 1]) * (beta[a] - beta[a - 1]) : beta[a];
    var xb = b < tau.length - 1 ? beta[b] + (tau[b] - half) / (tau[b] - tau[b + 1]) * (beta[b + 1] - beta[b]) : beta[b];
    return { peak: peak, peakBeta: beta[i0], half: half, xa: xa, xb: xb, width: xb - xa, left: tau[0] };
  }
  function nfmt(ui, N) { return ui.fmt(N, { group: true }); }

  VE.components['slowing'] = {
    mount: function (host, opts) {
      opts = opts || {};
      var ui = VE.ui, sim = VE.sim, el = ui.el, plot = ui.plot, fmt = ui.fmt;
      // At rest N = 4,096: the N = 16,384 peak values are not stated in the paper,
      // so they appear only after the reader picks that size
      var state = { beta: opts.beta != null ? opts.beta : 1.0, N: opts.N || 4096, tAnim: T_MAX, data: null, sched: null };
      var traj = trajectory(sim, state.beta);
      var ghosts = REFS.map(function (b) { return trajectory(sim, b); });
      var cleanup = [];

      var root = el('div', { class: 've-slowing' });
      host.appendChild(root);
      // No plate label on the figure (the shell's plate bar names it); the two panel notes
      // carry "Panel a/b" as their first item, so the shell's spec card stays sorted by panel
      var fig = ui.figure({ title: 'Relaxation is slowest near the transition' });
      root.appendChild(fig.el);
      var stageAB = fig.stage;

      // disclosure: an ⓘ button whose card holds the explanation; the text is not on the page at rest
      var cards = [];
      function info(label, title, html) {
        var btn = el('button', { class: 'v2-info ve-slowing__i', type: 'button', 'aria-label': label, html: ui.INFO_SVG || 'i' });
        var api = ui.info ? ui.info(btn, { title: title, html: html, kind: 'wide' }) : null;
        if (api) cards.push(api.el);
        return { btn: btn, api: api, set: function (h) {
          if (!api) return;
          var body = api.el.querySelector('.v2-card__body');
          if (body) { body.innerHTML = h; if (ui.terms) ui.terms(body); }
        } };
      }
      function head(letter, title, ib, extra) {
        return el('div', { class: 've-slowing__head' },
          el('h4', { class: 've-slowing__step' }, el('span', { class: 've-slowing__k', text: letter }), title, '\u00a0', ib.btn), extra || null);
      }
      function sayRow(ib) {
        var s = el('span', { class: 've-slowing__saytxt' });
        var row = el('p', { class: 've-slowing__say' }, el('span', { 'aria-live': 'polite' }, s), '\u00a0', ib.btn);
        return { el: row, txt: s };
      }

      // the figure's intro: hidden in the plate, mirrored into the chapter's reading card
      if (opts.intro != null) { if (opts.intro) stageAB.appendChild(el('p', { class: 've-slowing__intro', html: opts.intro })); }
      else stageAB.appendChild(el('p', { class: 've-slowing__intro', html:
        'A cooling schedule is a list of inverse temperatures β, with some ' +
        'sweeps (N single-spin Gibbs updates each) at every temperature before moving on. Small β is hot, a broad distribution; large β is cold, a concentrated one; ' +
        'the sampler cools from small β to large β. The paper sets the sweeps at each temperature from how fast the chain ' +
        'relaxes there. That speed collapses at one place: the transition β = 1, where a run picks the sign of m, the mean of its N binary units (mode A or mode B).' }));

      /* ------------------------------------------------------------ panel 1 */
      var p1 = el('section', { class: 've-slowing__panel', 'aria-label': 'Panel a: kick the chain and watch it settle' });
      stageAB.appendChild(p1);
      var i1 = info('How to read panel a', 'Panel a · release and settle',
        '<p>The model has N binary units (±1) that all prefer to agree, and <span data-term="magnetisation">m</span> is their average: m near +1 is mode A, m near −1 is mode B, m ≈ 0 is undecided. ' +
        'The curve is a landscape over m. On average the chain slides downhill at a speed equal to the slope, with no momentum, like a bead in syrup. ' +
        'Release it from m\u00a0=\u00a00.9 and track how far it still is from its resting point. This panel is the infinite-N limit: the chain follows the average slope with no random kicks.</p>' +
        '<p>Left: the landscape at the chosen β, one dot per sweep. Right: the distance to rest on a log scale, where a straight line means exponential decay; the grey lines show the values among β\u00a0=\u00a00.7, 1.0 and 1.3 that are not the live β, for comparison. Move β with the slider (left: hotter; right: colder).</p>' +
        '<dl class="v2-dl">' +
        '<dt>Pull-back rate</dt><dd>The linear pull toward rest, per sweep: 1 − β below the transition, the curvature of the well inside the mode above it, and 0 at β = 1 (flat bottom, no linear pull).</dd>' +
        '<dt>Linear decay time, infinite N</dt><dd>1 / pull-back rate: the time to shrink by a factor e under the linear pull alone. At β = 1 there is no exponential decay, so it reads ∞. This is not the relaxation time τ of panel b, which is a property of the finite chain.</dd>' +
        '<dt>Sweeps to settle, infinite N</dt><dd>From 0.9 to within 0.01 of rest, with no random kicks (the infinite-N limit). At β = 1 it reads “> 1,000”, off the plot: the large-N drift has no exponential decay there (see “More on this β”); with a finite N, random kicks end the crawl far sooner (panel b).</dd>' +
        '</dl>');
      p1.appendChild(head('a', 'Release the chain, watch it settle', i1));
      var pair = el('div', { class: 've-slowing__pair' });
      p1.appendChild(pair);
      var cellA = el('div', { class: 've-slowing__cell' }, el('div', { class: 've-slowing__cellhead', text: 'Landscape' }));
      var cellB = el('div', { class: 've-slowing__cell' }, el('div', { class: 've-slowing__cellhead', text: 'Distance to rest' }));
      pair.appendChild(cellA); pair.appendChild(cellB);
      var cvPot = ui.canvas(cellA, { aspect: 1.55, minHeight: 210, maxHeight: 300, label: 'Potential landscape over m with the released chain', draw: drawPot });
      var cvDec = ui.canvas(cellB, { aspect: 1.55, minHeight: 210, maxHeight: 300, label: 'Distance to rest against sweeps, log scale', draw: drawDec });

      var ctl1 = el('div', { class: 've-slowing__ctl' });
      p1.appendChild(ctl1);
      var betaSlider = ui.slider({
        label: 'inverse temperature', sym: 'β', min: 0.7, max: 1.35, step: 0.01, value: state.beta,
        format: function (v) { return fmt(v, { digits: 2 }); },
        marks: [{ value: 0.7, label: '0.7' }, { value: 1.0, label: '1 · transition' }, { value: 1.35, label: '1.35' }],
        hint: ' ',
        onInput: function (v) { setBeta(v, true); }
      });
      ctl1.appendChild(betaSlider.el);
      // The direction words (left: hotter) are in the panel's ⓘ card and on the panel b axis
      var hintEl = betaSlider.el.querySelector('.ve-slider__hint');
      if (hintEl) hintEl.remove();
      var replay = ui.button({ label: 'Release', icon: 'reset', onClick: function () { restart(); } });
      ctl1.appendChild(replay);

      var ro1 = el('div', { class: 've-slowing__readouts' });
      p1.appendChild(ro1);
      var rRate = ui.readout({ label: 'Pull-back rate', unit: 'per sweep' });
      var rRelax = ui.readout({ label: 'Linear decay time, infinite N', unit: 'sweeps' });
      var rSettle = ui.readout({ label: 'Sweeps to settle, infinite N', unit: 'sweeps' });
      ro1.appendChild(rRate.el); ro1.appendChild(rRelax.el); ro1.appendChild(rSettle.el);
      var i1s = info('More on this β', 'At this β', '');
      var say1 = sayRow(i1s);
      p1.appendChild(say1.el);
      p1.appendChild(el('p', { class: 've-figure__note', dataset: { kind: 'live' }, text:
        'Panel a: live, infinite-N limit, no random kicks · Model: the large-N drift of App. A, dm/dt = −m + tanh(βm), zero field, integrated here · Settled: within 0.01 of rest' }));

      /* ------------------------------------------------------------ panel 2 */
      var p2 = el('section', { class: 've-slowing__panel', 'aria-label': 'Panel b: relaxation time against β' });
      stageAB.appendChild(p2);
      var i2 = info('How to read panel b', 'Panel b · relaxation time <span class="gr">τ(β)</span>',
        '<p>With N finite, random kicks cap the crawl of panel a, so the wait at β = 1 is finite. ' +
        'The plot shows the relaxation time τ(β): the inverse spectral gap of the exact chain held on one sign (reflected at m = 0), zero field, for four model sizes N. ' +
        'The gap is the slowest decay rate of a deviation from equilibrium, so τ is the time for that deviation to shrink by a factor e. ' +
        'Tap or drag on the plot to move β.</p>' +
        '<p>Bold: τ(β) for the chosen N; grey: the other N, with a ring at each peak. The bracket is the peak width at half height.' +
        ' Squares: the time the schedule is designed with, at N = 2,048 and the schedule temperatures in view (docs/relaxation-times.md). Six times each square, rounded up, gives the sweeps of Table 3: ⌈6 × 59.002⌉ = ⌈354.01⌉ = 355 at β = 1.</p>' +
        '<dl class="v2-dl">' +
        '<dt>Peak width</dt><dd>Measured in β, at half height.</dd>' +
        '<dt>One-sign τ</dt><dd>τ of the chain held in one mode (one sign of m), read off the bold curve at the slider’s β.</dd>' +
        '<dt>Schedule</dt><dd>The paper’s dwell at each temperature is ⌈6τ⌉ sweeps, with τ from this one-sign chain above β = 1 and from the full chain (both signs) at β ≤ 1, at N = 2,048 (Table 3).</dd>' +
        '<dt>Why two chains</dt><dd>Below β = 1, holding the chain on one sign removes the slow relaxation of the balance between the two signs. So at β ≤ 1 the curves sit well below the full-chain time, and the schedule uses the full chain there (App. A). That is why the squares at β ≤ 1 lie above the curves.</dd>' +
        '</dl>' +
        '<p><b>Why √N.</b> At β = 1 fluctuations spread m over about N<sup>−1/4</sup> (App. A), and the peak of τ grows as √N (App. A). The quadratic term matters only when |β − 1| ~ N<sup>−1/2</sup> (App. A), and in the Fig. 3 data each 4× in N halves the peak width.</p>');
      p2.appendChild(head('b', 'Relaxation time peaks near the transition', i2, el('span', { class: 've-slowing__hint2', text: 'drag to move β' })));
      var cv2wrap = el('div', { class: 've-slowing__plot2' });
      p2.appendChild(cv2wrap);
      var cvTau = ui.canvas(cv2wrap, { aspect: 2.3, minHeight: 330, maxHeight: 400, label: 'Relaxation time against inverse temperature β for four model sizes', draw: drawTau });
      var ctl2 = el('div', { class: 've-slowing__ctl ve-slowing__ctl--n' });
      p2.appendChild(ctl2);
      var nToggle = ui.toggle({
        label: 'Model size N',
        options: NS.map(function (n) { return { value: n, label: nfmt(ui, n) }; }),
        value: state.N,
        onChange: function (v) { state.N = v; refreshData(); }
      });
      ctl2.appendChild(nToggle.el);
      ctl2.appendChild(ui.legend([
        { color: 'var(--ink)', mark: 'line', label: 'chosen N' },
        { color: 'var(--ink-3)', mark: 'line', label: 'other N' },
        { color: 'var(--ink-2)', mark: 'bracket', label: 'half-height width' },
        { color: 'var(--theory)', mark: 'square', label: 'schedule τ, N = 2,048 (full chain at β ≤ 1)' }
      ]));
      var ro2 = el('div', { class: 've-slowing__readouts' });
      p2.appendChild(ro2);
      var rTauHere = ui.readout({ label: 'τ at β = 1.00', unit: 'sweeps' });
      var rPeak = ui.readout({ label: 'Peak height', unit: 'sweeps' });
      var rWidth = ui.readout({ label: 'Peak width', unit: 'in β' });
      [rTauHere, rPeak, rWidth].forEach(function (r) { ro2.appendChild(r.el); });
      var i2s = info('More on this model size', 'At this model size', '');
      var say2 = sayRow(i2s);
      p2.appendChild(say2.el);
      p2.appendChild(el('p', { class: 've-figure__note', dataset: { kind: 'paper' }, text:
        'Panel b: relaxation times (paper App. A; docs/relaxation-times.md) · Curves: τ, inverse spectral gap of the exact chain held on one sign (reflected at m = 0), zero field · Squares (data/schedule.json): the schedule-design τ at N = 2,048, full chain at β ≤ 1 and one-sign chain above · Sweep: N single-spin Gibbs updates' }));

      fig.setTakeaway('Relaxation time peaks near β = 1, the peak growing as √N; the schedule is set at N = 2,048.');

      /* ------------------------------------------------------------ logic */
      var loop = ui.loop(function (dt) {
        state.tAnim = Math.min(T_MAX, state.tAnim + dt * SPEED);
        cvPot.redraw(); cvDec.redraw();
        if (state.tAnim >= T_MAX) loop.stop();
      }, { el: root });
      cleanup.push(function () { loop.destroy(); });

      function restart() {
        if (ui.reducedMotion) { state.tAnim = T_MAX; cvPot.redraw(); cvDec.redraw(); return; }
        state.tAnim = 0; loop.start();
      }
      function setBeta(v, fromSlider) {
        v = Math.round(v * 100) / 100;
        state.beta = v;
        if (!fromSlider) betaSlider.set(v, { silent: true });
        traj = trajectory(sim, v);
        paint1();
        restart();
        paint2();
      }

      function paint1() {
        var b = state.beta, k = pullRate(b, traj.rest);
        var atOne = Math.abs(b - 1) < 0.005;
        rRate.set(atOne ? 0 : k);
        rRate.el.querySelector('.ve-readout__value span').textContent = atOne ? '0' : fmt(k, { digits: 2 });
        rRelax.set(atOne ? 0 : 1 / k);
        rRelax.el.querySelector('.ve-readout__value span').textContent = atOne ? '∞' : fmt(1 / k, { digits: 1 });
        var s = traj.settle;
        rSettle.set(s);
        rSettle.el.querySelector('.ve-readout__value span').textContent = (s >= 1000 ? '>\u00a0' + fmt(1000, { group: true }) : fmt(s, { digits: s < 10 ? 1 : 0 }));
        var bs = fmt(b, { digits: 2 });
        var txt;
        if (atOne) {
          txt = '<b>β = 1, the transition.</b> The bottom of the landscape is flat: near rest the slope is only m³/3, so the pull fades as the chain approaches. ' +
            'In this large-N drift (no field, no fluctuations) the distance shrinks like t<sup>−1/2</sup> instead of exponentially, and settling takes about ' + fmt(+s.toPrecision(2), { group: true }) + ' sweeps instead of ' + fmt(ghosts[0].settle, { digits: 0 }) + ' at β = 0.7. That figure is the infinite-N limit, with no random kicks; with a finite N, random kicks end the crawl far sooner (panel b).';
        } else if (b < 1) {
          txt = '<b>β = ' + bs + ', hotter than the transition.</b> One resting point, m = 0: the sample is still undecided. The bottom is curved, so the distance shrinks by the same factor every sweep, a straight line on the log scale' +
            (b > 0.9 ? '. But the curvature 1 − β = ' + fmt(1 - b, { digits: 2 }) + ' is small this close to β = 1, so the line is shallow.' : '.');
        } else {
          txt = '<b>β = ' + bs + ', colder than the transition.</b> Two resting points, m = ±' + fmt(traj.rest, { digits: 2 }) + ', one per mode; the chain settles in mode A, where it was released, so its run starts only ' + fmt(M0 - traj.rest, { digits: 2 }) + ' from rest. ' +
            (b < 1.1 ? 'The wells are still shallow, so the pull is weak and settling is slow.' : 'The well is curved at its bottom, so the decay is exponential again: a straight line on the log scale.');
        }
        i1s.set('<p>' + txt + '</p>');
        say1.txt.innerHTML = atOne ? 'Flat bottom: no linear pull, so no exponential decay. Finite N’s random kicks end the crawl (panel\u00a0b).'
          : b < 1 ? 'One rest point: exponential decay' + (b > 0.9 ? ', but slow so close to β\u00a0=\u00a01.' : '.')
          : b < 1.1 ? 'Two shallow wells: a weak pull and slow settling.' : 'Two wells with curved bottoms: exponential decay again.';
        rTauHere.el.querySelector('.ve-readout__label').innerHTML = 'τ, chain held in one mode, at <span class="ve-slowing__sym">β</span> = ' + bs;
      }

      function paint2() {
        if (!state.data) return;
        var d = state.data, tau = d.tau[String(state.N)];
        var st = state.stats[state.N];
        rTauHere.set(interp(d.beta, tau, state.beta));
        rTauHere.el.querySelector('.ve-readout__value span').textContent = fmt(interp(d.beta, tau, state.beta), { digits: 1 });
        rPeak.set(st.peak, 'at β = ' + fmt(st.peakBeta, { digits: 3 }));
        rPeak.el.querySelector('.ve-readout__value span').textContent = fmt(st.peak, { digits: st.peak < 20 ? 1 : 0 });
        rWidth.set(st.width, '');
        rWidth.el.querySelector('.ve-readout__value span').textContent = fmt(st.width, { digits: 3 });
        var s0 = state.stats[256], sN = st;
        if (state.N === 16384) {
          say2.long = 'At N = 16,384, the largest size shown, the peak is the tallest and narrowest, and it sits closest to β = 1 (the offset shrinks as N grows). Far from the peak (β = 0.7 or 1.35) the curves for all N nearly coincide: the model size matters most near the transition.';
        } else if (state.N === 256) {
          say2.long = 'At N = 256 the peak is low (' + fmt(st.peak, { digits: 1 }) + ' sweeps) and broad (' + fmt(st.width, { digits: 2 }) +
            ' wide in β). Step up N: each 4× in N doubles the height, halves the width, and moves the peak closer to β = 1.';
        } else {
          say2.long = 'At N = ' + nfmt(ui, state.N) + ': peak ' + fmt(st.peak, { digits: st.peak < 20 ? 1 : 0 }) + ' sweeps, ' + fmt(st.peak / s0.peak, { digits: 1 }) + '× the N = 256 peak; width ' +
            fmt(st.width, { digits: 3 }) + ', ' + fmt(s0.width / st.width, { digits: 1 }) + '× narrower. The height grows as √N and the width shrinks as 1/√N.';
        }
        i2s.set('<p>' + say2.long + ' Here τ at the slider’s β is interpolated between the computed points.</p>');
        say2.txt.textContent = state.N === 16384
          ? 'Each 4× in N doubles the peak and halves its width.'
          : state.N === 256 ? 'A low, broad peak. Step up N to see it sharpen.'
          : 'Peak ' + fmt(st.peak / s0.peak, { digits: 1 }) + '× the N\u00a0=\u00a0256 height, ' + fmt(s0.width / st.width, { digits: 1 }) + '× narrower.';
        cvTau.redraw();
      }
      function refreshData() { paint2(); }

      /* ------------------------------------------------------------ drawing */
      function trajAt(tr, t) {
        var i = Math.min(tr.t.length - 1, Math.max(0, t / 0.05));
        var i0 = Math.floor(i), u = i - i0, i1 = Math.min(tr.t.length - 1, i0 + 1);
        return { m: tr.m[i0] + u * (tr.m[i1] - tr.m[i0]), d: tr.d[i0] + u * (tr.d[i1] - tr.d[i0]) };
      }
      function ballColor(c) { return c.modeA; }   // one accent for the reader's run, every β

      function drawPot(ctx, w, h, c) {
        var b = state.beta;
        var box = ui.inset(w, h, { top: 30, right: 12, bottom: 42, left: 12 });
        var sx = ui.scale.linear([-1, 1], [box.x0, box.x1]);
        // the top follows β so the release point and its first dots stay on the plot (β = 0.7: F(0.9) ≈ 0.14)
        var sy = ui.scale.linear([-0.1, Math.max(0.16, sim.landauF(M0, b) + 0.06)], [box.y1, box.y0]);
        // ground washes for the two modes
        ctx.fillStyle = c.modeBWash; ctx.fillRect(box.x0, box.y1, sx(0) - box.x0, 3);
        ctx.fillStyle = c.modeAWash; ctx.fillRect(sx(0), box.y1, box.x1 - sx(0), 3);
        plot.axis(ctx, sx, { side: 'bottom', at: box.y1, ticks: [-1, -0.5, 0, 0.5, 1], c: c, format: function (v) { return fmt(v); } });
        plot.text(ctx, '← mode B', box.x0, h - 6, { color: c.modeB, size: 11, weight: 500, c: c });
        plot.text(ctx, 'm', (box.x0 + box.x1) / 2, h - 6, { color: c.ink2, size: 11, weight: 500, align: 'center', c: c });
        plot.text(ctx, 'mode A →', box.x1, h - 6, { color: c.modeA, size: 11, weight: 500, align: 'right', c: c });
        // guide at m = 0
        plot.vline(ctx, 0, sx, box.y0, box.y1, { color: c.rule, dash: [2, 3] });
        ctx.save();
        ctx.beginPath(); ctx.rect(box.x0 - 2, box.y0 - 8, box.w + 4, box.h + 8); ctx.clip();
        plot.fn(ctx, function (m) { return sim.landauF(m, b); }, sx, sy, { n: 240, color: c.ink, width: 2.25 });
        // resting points
        var rests = b > 1.005 ? [-traj.rest, traj.rest] : [0];
        rests.forEach(function (r) {
          var px = sx(r), py = sy(sim.landauF(r, b));
          ctx.fillStyle = c.ink3;
          ctx.beginPath(); ctx.moveTo(px, py + 6); ctx.lineTo(px - 4, py + 12); ctx.lineTo(px + 4, py + 12); ctx.closePath(); ctx.fill();
          plot.text(ctx, 'rest', px, py + 24, { size: 11, color: c.ink3, align: 'center', halo: true, c: c });
        });
        // strobe: one dot per whole sweep
        var col = ballColor(c);
        for (var k = 1; k <= Math.floor(state.tAnim + 1e-9); k++) {
          var mk = trajAt(traj, k).m;
          ctx.beginPath(); ctx.arc(sx(mk), sy(sim.landauF(mk, b)) - 6, 2.4, 0, Math.PI * 2);
          ctx.fillStyle = ui.alpha(col, 0.55); ctx.fill();
        }
        // release mark
        plot.vline(ctx, M0, sx, sy(sim.landauF(M0, b)) - 18, box.y1, { color: c.ink3, dash: [2, 3] });
        // ball
        var now = trajAt(traj, state.tAnim).m;
        ctx.beginPath(); ctx.arc(sx(now), sy(sim.landauF(now, b)) - 7.5, 7, 0, Math.PI * 2);
        ctx.fillStyle = col; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = c.surface; ctx.stroke();
        ctx.restore();
        plot.text(ctx, 'release', sx(M0) - 4, sy(sim.landauF(M0, b)) - 12, { size: 11, color: c.ink3, align: 'right', halo: true, c: c });
        plot.text(ctx, 'β = ' + fmt(b, { digits: 2 }), box.x0, 16, { size: 13, weight: 600, color: c.ink, c: c });
        plot.text(ctx, 't = ' + fmt(Math.floor(state.tAnim), { digits: 0 }) + ' sweeps', box.x1, 16, { size: 11, color: c.ink3, align: 'right', c: c });
      }

      // Text on a surface-coloured box: stays legible over any curve.
      function tag(ctx, str, x, y, o, c) {
        o = o || {};
        var size = o.size || 11;
        ctx.save();
        ctx.font = ui.font('mono', size, o.weight || 400);
        var tw = ctx.measureText(str).width, pad = 3;
        var left = o.align === 'right' ? x - tw : o.align === 'center' ? x - tw / 2 : x;
        var rect = { left: left - pad, right: left + tw + pad, top: y - size * 0.85 - pad + 1, bottom: y + size * 0.15 + pad };
        if (o.measure) { ctx.restore(); return rect; }
        ctx.fillStyle = c.surface; ctx.globalAlpha = o.opaque ? 1 : 0.92;
        ctx.fillRect(left - pad, y - size * 0.85 - pad + 1, tw + 2 * pad, size + 2 * pad - 1);
        ctx.restore();
        plot.text(ctx, str, x, y, { size: size, weight: o.weight, color: o.color, align: o.align, c: c });
        return rect;
      }

      function drawDec(ctx, w, h, c) {
        var box = ui.inset(w, h, { top: 30, right: 14, bottom: 42, left: 44 });
        var sx = ui.scale.linear([0, T_MAX], [box.x0, box.x1]);
        var sy = ui.scale.log([1e-3, 1], [box.y1, box.y0]);
        plot.axis(ctx, sy, { side: 'left', at: box.x0, ticks: [1e-3, 1e-2, 1e-1, 1], c: c, grid: [box.x0, box.x1],
          format: function (v) { return v >= 1 ? '1' : v >= 0.1 ? '0.1' : v >= 0.01 ? '0.01' : '0.001'; } });
        plot.axis(ctx, sx, { side: 'bottom', at: box.y1, ticks: [0, 10, 20, 30, 40], c: c, label: 'sweeps →', labelAlign: 'right' });
        plot.text(ctx, '|m − rest|', box.x0 - 40, 16, { size: 11, weight: 500, color: c.ink2, c: c });
        // threshold
        plot.hline(ctx, TOL, sy, box.x0, box.x1, { color: c.ink3, dash: [1, 3] });
        plot.text(ctx, 'settled', box.x1 - 2, sy(TOL) - 4, { size: 11, color: c.ink3, align: 'right', halo: true, c: c });
        var b = state.beta, atOne = Math.abs(b - 1) < 0.005, col = ballColor(c);
        var cubic = function (t) { return M0 / Math.sqrt(1 + 2 * M0 * M0 * t / 3); };
        ctx.save(); ctx.beginPath(); ctx.rect(box.x0, box.y0 - 4, box.w + 8, box.h + 4); ctx.clip();
        // ghosts
        ghosts.forEach(function (g) {
          if (Math.abs(g.beta - b) < 0.005) return;
          var ys = Array.prototype.map.call(g.d, function (v) { return v > 1e-3 ? v : null; });
          plot.line(ctx, g.t, ys, sx, sy, { color: c.ink3, width: 1.25, alpha: 0.5 });
        });
        // current run, drawn up to tAnim, in the accent colour
        var n = Math.min(traj.t.length, Math.floor(state.tAnim / 0.05) + 1);
        var xs = [], ys = [];
        for (var i = 0; i < n; i++) { if (traj.d[i] < 1e-3) break; xs.push(traj.t[i]); ys.push(traj.d[i]); }
        plot.line(ctx, xs, ys, sx, sy, { color: col, width: atOne ? 3.5 : 2.5 });
        var head = trajAt(traj, state.tAnim);
        if (head.d >= 1e-3) { ctx.beginPath(); ctx.arc(sx(state.tAnim), sy(head.d), 4.5, 0, Math.PI * 2); ctx.fillStyle = col; ctx.fill(); }
        // theory guide on top, thin and without a halo, so the run shows on both sides of it
        if (atOne) plot.fn(ctx, cubic, sx, sy, { color: c.theory, width: 1.5, dash: [5, 4] });
        ctx.restore();
        // ghost labels (hidden when the reader's run is close to that ghost)
        ghosts.forEach(function (g) {
          if (Math.abs(g.beta - b) < 0.06) return;
          var lab = 'β = ' + fmt(g.beta, { digits: 1 });
          if (g.d[g.d.length - 1] >= 1e-3) plot.text(ctx, lab, box.x1 - 2, sy(g.d[g.d.length - 1]) - 6, { size: 11, color: c.ink3, align: 'right', halo: true, c: c });
          else {
            var j = 0; while (j < g.d.length && g.d[j] >= 1.3e-3) j++;
            plot.text(ctx, lab, sx(g.t[j]) + 4, box.y1 - 6, { size: 11, color: c.ink3, halo: true, c: c });
          }
        });
        if (atOne) tag(ctx, 'cubic pull only: not exponential', sx(24), sy(cubic(24)) - 12, { size: 11, color: c.theory, align: 'center' }, c);
        plot.text(ctx, 'β = ' + fmt(b, { digits: 2 }), box.x1, 16, { size: 12, weight: 600, color: col, align: 'right', c: c });
      }

      var TAU_INSET = { top: 30, right: 14, bottom: 44, left: 44 };
      function drawTau(ctx, w, h, c) {
        var box = ui.inset(w, h, TAU_INSET);
        var sx = ui.scale.linear([0.7, 1.35], [box.x0, box.x1]);
        var sy = ui.scale.linear([0, 85], [box.y1, box.y0]);
        plot.axis(ctx, sy, { side: 'left', at: box.x0, ticks: [0, 20, 40, 60, 80], c: c, grid: [box.x0, box.x1] });
        plot.axis(ctx, sx, { side: 'bottom', at: box.y1, ticks: [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3], c: c, format: function (v) { return fmt(v, { digits: 1 }); } });
        plot.text(ctx, 'τ, sweeps', box.x0 - 40, 16, { size: 11, weight: 500, color: c.ink2, c: c });
        if (w >= 560) {
          plot.text(ctx, '← hotter', box.x0, h - 6, { size: 11, color: c.ink3, c: c });
          plot.text(ctx, 'colder →', box.x1, h - 6, { size: 11, color: c.ink3, align: 'right', c: c });
        }
        plot.text(ctx, 'inverse temperature β', (box.x0 + box.x1) / 2, h - 6, { size: 11, weight: 500, color: c.ink2, align: 'center', c: c });
        plot.vline(ctx, 1, sx, box.y0, box.y1, { color: c.ink3, dash: [4, 3] });
        if (!state.data) {
          plot.text(ctx, 'transition', sx(1) - 5, box.y0 + 10, { size: 11, color: c.ink3, align: 'right', halo: true, c: c });
          plot.text(ctx, 'loading data…', (box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2, { size: 12, color: c.ink3, align: 'center', c: c }); return;
        }
        var d = state.data;
        var tau = d.tau[String(state.N)], st = state.stats[state.N], col = ballColor(c);
        var hit = function (a, b) { return a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom; };
        // does the selected curve pass through a pixel rectangle?
        var crosses = function (r) {
          var prev = null;
          for (var px = Math.floor(r.left) - 2; px <= r.right + 2; px++) {
            var bv = 0.7 + (px - box.x0) / box.w * 0.65, yv = sy(interp(d.beta, tau, bv));
            if (prev != null && Math.min(prev, yv) <= r.bottom + 2 && Math.max(prev, yv) >= r.top - 2) return true;
            prev = yv;
          }
          return false;
        };
        // 1. curves: other N faint, selected bold
        NS.forEach(function (n) {
          if (n === state.N) return;
          plot.line(ctx, d.beta, d.tau[String(n)], sx, sy, { color: c.ink3, width: 1.25, alpha: 0.55 });
        });
        plot.line(ctx, d.beta, tau, sx, sy, { color: c.ink, width: 2.5 });
        // the β cursor line goes under every label drawn below, so the opaque labels mask it
        plot.vline(ctx, state.beta, sx, box.y0, box.y1, { color: col, dash: false, width: 1.5 });
        // 2. place the selected curve's label where it does not cover its own curve:
        //    the right column at peak height, else just above the peak, else left of the peak
        var colX = box.x1 - 2, px0 = sx(st.peakBeta), py0 = sy(st.peak);
        var selLab = 'N = ' + nfmt(ui, state.N) + ' · peak ' + fmt(st.peak, { digits: st.peak < 20 ? 1 : 0 });
        var selO = { size: 11, weight: 600, color: c.ink, opaque: true };
        var cands = [
          { x: colX, y: py0 + 4, align: 'right', leader: true },
          { x: px0, y: py0 - 11, align: 'center' },
          { x: px0 - 10, y: py0 + 4, align: 'right' }
        ];
        var sel = null;
        for (var ci = 0; ci < cands.length && !sel; ci++) {
          var cd = cands[ci], o = Object.assign({ align: cd.align, measure: true }, selO);
          var r = tag(ctx, selLab, cd.x, cd.y, o, c);
          if (cd.align === 'center') {   // keep inside the plot box
            var sh = Math.max(0, box.x0 + 2 - r.left) - Math.max(0, r.right - (box.x1 + 2));
            cd.x += sh; r = tag(ctx, selLab, cd.x, cd.y, o, c);
          }
          if (r.left < box.x0 || r.top < box.y0 - 4) continue;
          if (cd.leader && r.left < px0 + 12) continue;
          if (crosses(r)) continue;
          cd.rect = r; sel = cd;
        }
        if (!sel) { sel = cands[1]; sel.rect = tag(ctx, selLab, sel.x, sel.y, Object.assign({ align: 'center', measure: true }, selO), c); }
        // transition label, moved below the selected label if they meet
        var cy = box.y0 + 10, cr = tag(ctx, 'transition', sx(1) - 5, cy, { size: 11, align: 'right', measure: true }, c);
        if (hit(cr, sel.rect)) cy = sel.rect.bottom + 12;
        tag(ctx, 'transition', sx(1) - 5, cy, { size: 11, color: c.ink3, align: 'right', opaque: true }, c);
        // 3. peak marks of the other N, with a right-hand label column and dotted leaders
        NS.slice().reverse().forEach(function (n) {
          if (n === state.N) return;
          var s2 = state.stats[n], y = sy(s2.peak), lab = 'N = ' + nfmt(ui, n);
          var o2 = { size: 11, color: c.ink3, align: 'right', opaque: true };
          var r2 = tag(ctx, lab, colX, y + 4, Object.assign({ measure: true }, o2), c);
          var x0 = sx(s2.peakBeta) + 6, x1 = r2.left - 3;
          // a leader at the bracket's height starts past the bracket, so it does not seem to come from it
          if (Math.abs(y - sy(st.half)) < 8 && sx(st.xb) + 8 > x0) x0 = sx(st.xb) + 8;
          if (hit(r2, sel.rect)) return;
          if (x1 > x0) plot.line(ctx, [x0, x1], [y, y], function (v) { return v; }, function (v) { return v; }, { color: c.ink3, width: 1, dash: [1, 3], alpha: 0.6 });
          tag(ctx, lab, colX, y + 4, o2, c);
        });
        // 4. half-height width bracket on a surface halo
        var yh = sy(st.half);
        var bracket = function () {
          ctx.beginPath(); ctx.moveTo(sx(st.xa), yh); ctx.lineTo(sx(st.xb), yh);
          ctx.moveTo(sx(st.xa), yh - 5); ctx.lineTo(sx(st.xa), yh + 5); ctx.moveTo(sx(st.xb), yh - 5); ctx.lineTo(sx(st.xb), yh + 5); ctx.stroke();
        };
        ctx.save(); ctx.lineCap = 'round';
        ctx.strokeStyle = c.surface; ctx.lineWidth = 5; bracket();
        ctx.strokeStyle = c.ink2; ctx.lineWidth = 2; bracket(); ctx.restore();
        // the other peaks' rings go on top of the bracket, so a ring at half height stays visible
        NS.forEach(function (n) {
          if (n !== state.N) plot.dots(ctx, [{ x: state.stats[n].peakBeta, y: state.stats[n].peak }], sx, sy, { r: 3, ring: true, color: c.ink3 });
        });
        // 5b. the schedule-design τ at N = 2,048 (paper Fig. 3b): squares at the schedule temperatures in
        //     view, joined by grey connectors; the β = 1 square carries the Table 3 arithmetic
        if (state.sched) {
          var lad = state.sched.ladder, tds = state.sched.tau, sq = [];
          lad.forEach(function (b, i) { if (b >= 0.7 && b <= 1.35) sq.push({ x: b, y: tds[i], K: state.sched.K[i] }); });
          plot.line(ctx, sq.map(function (p) { return p.x; }), sq.map(function (p) { return p.y; }), sx, sy, { color: c.ink3, width: 1, dash: [2, 3], alpha: 0.7 });
          sq.forEach(function (p) {
            var qx = sx(p.x), qy = sy(p.y);
            ctx.fillStyle = c.theory; ctx.strokeStyle = c.surface; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.rect(qx - 4, qy - 4, 8, 8); ctx.fill(); ctx.stroke();
          });
          var s1 = sq.filter(function (p) { return Math.abs(p.x - 1) < 1e-9; })[0];
          if (s1) {
            // drop the unit on narrow plots so the label clears the y-axis ticks
            var lab = '⌈6 × ' + s1.y.toFixed(3) + '⌉ = ' + s1.K;
            ctx.font = '11px sans-serif';
            if (ctx.measureText(lab + ' sweeps').width < sx(1) - 9 - box.x0 - 8) lab += ' sweeps';
          }
          if (s1) tag(ctx, lab, sx(1) - 9, sy(s1.y) + 4, { size: 11, color: c.theory, align: 'right', opaque: true }, c);
        }
        // 5. the selected peak and its label
        if (sel.leader) {
          var x1s = sel.rect.left - 3;
          if (x1s > px0 + 6) plot.line(ctx, [px0 + 6, x1s], [py0, py0], function (v) { return v; }, function (v) { return v; }, { color: c.ink2, width: 1, dash: [1, 3], alpha: 0.9 });
        }
        plot.dots(ctx, [{ x: st.peakBeta, y: st.peak }], sx, sy, { r: 4.5, color: c.ink });
        tag(ctx, selLab, sel.x, sel.y, Object.assign({ align: sel.align }, selO), c);
        // (no left-edge value: the β = 0.7 figure is not in the paper)
        // 6. cursor at the shared β, labelled in the top margin
        var tb = interp(d.beta, tau, state.beta), cx = sx(state.beta);
        plot.dots(ctx, [{ x: state.beta, y: tb }], sx, sy, { r: 5, color: col, stroke: c.surface });
        var cl = 'β = ' + fmt(state.beta, { digits: 2 }) + ': τ = ' + fmt(tb, { digits: tb < 20 ? 1 : 0 }) + ' sweeps';
        ctx.save(); ctx.font = ui.font('mono', 11, 600); var cw = ctx.measureText(cl).width; ctx.restore();
        var lx = Math.max(box.x0 + 56, Math.min(box.x1 - cw, cx - cw / 2));
        plot.text(ctx, cl, lx, 16, { size: 11, weight: 600, color: col, c: c });
      }

      /* ------------------------------------------------------------ canvas drag (panel 2) */
      var dragging = false;
      function pick(e) {
        var r = cvTau.canvas.getBoundingClientRect();
        var box = ui.inset(r.width, r.height, TAU_INSET);
        var u = (e.clientX - r.left - box.x0) / box.w;
        var v = 0.7 + Math.max(0, Math.min(1, u)) * 0.65;
        setBeta(v, false);
      }
      function onDown(e) { dragging = true; pick(e); }
      function onMove(e) { if (dragging) pick(e); }
      function onUp() { dragging = false; }
      cvTau.canvas.addEventListener('pointerdown', onDown);
      cvTau.canvas.addEventListener('pointermove', onMove);
      cvTau.canvas.addEventListener('pointerup', onUp);
      cvTau.canvas.addEventListener('pointercancel', onUp);
      cvTau.canvas.addEventListener('pointerleave', onUp);

      /* ------------------------------------------------------------ data */
      Promise.all([sim.loadJSON('tracking_time.json'), sim.loadJSON('schedule.json')]).then(function (res) {
        state.data = res[0]; state.sched = res[1];
        state.stats = {};
        NS.forEach(function (n) { state.stats[n] = curveStats(state.data.beta, state.data.tau[String(n)]); });
        paint2();
      }).catch(function (e) {
        console.error(e);
        say2.txt.textContent = 'Could not load the relaxation-time data (' + e.message + ').';
      });

      paint1();
      if (ui.reducedMotion) state.tAnim = T_MAX; else { state.tAnim = 0; loop.start(); }

      return {
        setBeta: function (v) { setBeta(v, false); },
        setN: function (n) { if (NS.indexOf(n) >= 0) { nToggle.set(n, { silent: true }); state.N = n; paint2(); } },
        destroy: function () {
          cleanup.forEach(function (f) { f(); });
          cards.forEach(function (c) { c.remove(); });
          [cvPot, cvDec, cvTau].forEach(function (cv) { if (cv) cv.destroy(); });
          root.remove();
        }
      };
    }
  };
})();
