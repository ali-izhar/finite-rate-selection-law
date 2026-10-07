/* components/comparison.js — "The test: prediction vs measurement"
 *
 * Registers VE.components['comparison'] = { mount(el, opts) }.
 * opts.plate  plate number or label (default 8)
 *
 * Every number shown comes from these files (loaded at mount with VE.sim.loadJSON):
 *   data/field_grid_points.json   Fig. 2b: 8 bias points (q_eq, q_chain + Wilson 95%, logits),
 *                            c measured 0.199 [0.1908, 0.2095] over 20 pattern draws, predicted
 *   data/table1.json         Table 1: p = 1..4 at N = 2048, amplification, c predicted / measured
 *   data/size_scan.json      App. A size scan at p = 3: N = 1024, 2048, 4096
 *   data/sampler_clock.json  one-pattern ratio r(N) = c_meas(p=1) / c_CW(N) (Sec. 5, App. B); p=3 prediction x r = 0.212
 *   data/c_cw.json           c_CW = 0.1198 at N = 2048 (numerical evolution of the exact CW chain)
 * Numbers computed on the page (and labelled so): predictions × the one-pattern ratio r(N),
 * the relative gap (measured − predicted) / predicted, and one dot's own ratio logit q_chain / logit q_eq.
 * No live simulation. No global listeners: all events are on this component's own nodes.
 */
(function () {
  'use strict';
  var VE = (window.VE = window.VE || {});
  VE.components = VE.components || {};

  var MINUS = '−';

  function mount(root, opts) {
    opts = opts || {};
    var ui = VE.ui, el = ui.el;
    var fmt = ui.fmt;
    var f3 = function (x) { return fmt(x, { digits: 3 }); };
    var f2 = function (x) { return fmt(x, { digits: 2 }); };
    var pct1 = function (x) { return (100 * x).toFixed(1) + '%'; };
    var signedPct = function (x) {
      var v = Math.round(100 * x);
      if (v === 0) return '0%';
      return (v > 0 ? '+' : MINUS) + Math.abs(v) + '%';
    };
    var reduced = function () { return !!ui.reducedMotion; };

    var wrap = el('div', { class: 've-comparison' });
    root.appendChild(wrap);

    var fig = ui.figure({
      plate: opts.plate != null ? opts.plate : 8,
      title: 'The test: prediction against measurement',
      note: 'Source: the paper (Fig. 2b, Table 1, Sec. 5, App. B) and its result files · Model: Hopfield model, N = 2,048 unless stated · Simulation: none on this page · Runs: brief cooling, removal of (pattern 1, +) runs at both field signs (the filtered start), reheating at β = 0.7, then the paper’s schedule · Computed here: predictions × the one-pattern ratio, and % gaps, from those files',
      noteKind: 'paper'
    });
    wrap.appendChild(fig.el);

    /* Explanations, sources and secondary caveats live in info cards made with
       VE.ui.info / VE.ui.specCard; the figure keeps one short line per panel. Every card this
       component creates is removed on destroy. comparison.css scopes its plate rules to .ve-comparison--v2. */
    wrap.classList.add('ve-comparison--v2');
    var cards = [], specOff = null, chOff = null, drawerOff = null;
    function infoBtn(label, title, html) {
      var b = el('button', { type: 'button', class: 'v2-info ve-comparison__i', 'aria-label': label, html: ui.INFO_SVG || '' });
      var api = ui.info(b, { title: title, html: html, kind: 'wide' });
      cards.push(api.el);
      b.setBody = function (h) { var body = api.el.querySelector('.v2-card__body'); if (body.innerHTML !== h) { body.innerHTML = h; ui.terms(body); } };
      return b;
    }
    var nb = function (s) { return s.replace(/ = /g, '\u00a0=\u00a0'); };
    var loading = el('p', { class: 've-comparison__loading', text: 'Loading the measured numbers…' });
    fig.stage.appendChild(loading);

    var destroyed = false;
    var canvases = [];
    var tweenRaf = 0, domRaf = 0;

    Promise.all([
      VE.sim.loadJSON('field_grid_points.json'),
      VE.sim.loadJSON('table1.json'),
      VE.sim.loadJSON('size_scan.json'),
      VE.sim.loadJSON('sampler_clock.json'),
      VE.sim.loadJSON('c_cw.json')
    ]).then(function (d) {
      if (destroyed) return;
      loading.remove();
      build(d[0], d[1], d[2], d[3], d[4]);
      ui.terms(wrap);
    }).catch(function (e) {
      console.error(e);
      loading.textContent = 'Could not load the data files: ' + (e && e.message ? e.message : e);
      loading.className = 've-error';
    });

    function build(F, T1, SS, CLK, CCW) {
      /* ------------------------------------------------------------ numbers */
      var cCW = CCW.c_cw;                                   // 0.1198
      var p3 = T1.rows.filter(function (r) { return r.p === 3; })[0];
      var A3 = p3.amplification;                            // 1.662
      var predMain = F.c_predicted;                         // 0.199
      var measMain = F.c_measured;                          // 0.199 [0.1908, 0.2095]
      var rByN = {};
      CLK.rows.forEach(function (r) { rByN[r.N] = r.r; });
      var r2048 = rByN[2048];
      var pts = F.points.slice().sort(function (a, b) { return a.logit_q_eq - b.logit_q_eq; });

      var setP = T1.rows.map(function (r) {
        return { key: 'p' + r.p, label: 'p = ' + r.p, sub: 'patterns', pred: r.c_predicted, meas: r.c_measured, lo: r.ci95[0], hi: r.ci95[1], N: 2048, n: r.n_chains, amp: r.amplification, ccw: cCW };
      });
      var setN = SS.rows.map(function (r) {
        return { key: 'N' + r.N, label: 'N = ' + fmt(r.N), sub: 'variables', pred: r.c_predicted, meas: r.c_measured, lo: r.ci95[0], hi: r.ci95[1], N: r.N, n: r.n_chains, amp: A3, ccw: r.c_CW };
      });

      /* ------------------------------------------------------------ state */
      var S = { sel: pts.length - 1, set: 'p', clock: false, k: 0, hl: null, dom: [0.11, 0.25] };

      var p3clk = CLK.p3_clock_corrected_prediction_N2048;   // 0.212

      /* ------------------------------------------------ panel 1: Fig. 2b */
      var P1_TEXT = 'Each dot is one field strength on a <span data-term="hopfield">Hopfield model</span> (N = 2,048, p = 3) whose weights are written down, not trained; the model’s log-odds are estimated from the mean overlap in the positive mode (approximate). ' +
        'Mode A is stored pattern 1 with one sign (its sign-flipped copy counts as a different mode); runs that end on another pattern are set aside, leaving 11,033 to 11,801 runs per dot. ' +
        'Each run begins from the filtered start and reheating, then cools on the paper’s schedule. ' +
        'A faithful sampler would put every dot on the diagonal (c = 1). The dots lie on a line one fifth as steep.';
      var P1_NOTE = 'Panel 1’s slope is fitted per pattern set, then averaged over twenty sets weighted by the number of runs assigned to pattern 1. Panel 2’s rows are separate runs at one field strength, both signs pooled, twelve pattern sets each. All intervals are 95%.';
      var CLOSE = 'The weights are written down, not trained, yet the sampler keeps only about a fifth of the model’s preference. c<sub>CW</sub> × A<sub>3</sub>, with nothing fitted to these runs, predicts that fifth ' +
        'closely at p\u00a0=\u00a03, N\u00a0=\u00a02,048; other settings miss by up to 11%, and the one-pattern ratio does not explain the trend with size.';
      var P2_TEXT = 'Each dot is a separate run: predicted c across, measured c up, with its 95% interval. ' +
        'A dot on the dashed line is an exact hit; an interval that crosses the line agrees.';
      function panelHead(n, title, readHtml) {
        return el('header', { class: 've-comparison__ph ve-comparison__ph--v2' },
          el('span', { class: 've-comparison__pk', text: 'Panel ' + n }),
          el('h4', { class: 've-comparison__pq' }, title, infoBtn('How to read panel ' + n, 'Reading panel ' + n, readHtml)));
      }
      var p1 = el('section', { class: 've-comparison__panel' });
      p1.appendChild(panelHead(1, 'One setting, eight field strengths',
        '<p>' + P1_TEXT + '</p><p>' + P1_NOTE + '</p><p class="v2-card__k">In short</p><p>' + CLOSE + '</p>'));
      p1.appendChild(el('p', { class: 've-comparison__pt', html:
        'A faithful sampler puts every dot on the diagonal, c = 1. These dots lie on a line one fifth as steep.' }));

      var cv1 = ui.canvas(p1, { height: function (w) { return Math.min(Math.round(w * 0.92), 470); }, label: 'Sampler log-odds against model log-odds at eight field strengths; slope 0.199, against 1 for a faithful sampler.', draw: drawFit });
      canvases.push(cv1);
      // legend under the chart (never collides with the dot labels at any width)
      p1.appendChild(el('div', { class: 've-comparison__key' },
        el('span', { class: 've-comparison__keyItem' }, el('i', { class: 've-comparison__sw ve-comparison__sw--pred' }), 'predicted slope ' + f3(predMain)),
        el('span', { class: 've-comparison__keyItem' }, el('i', { class: 've-comparison__sw ve-comparison__sw--fit' }), 'fitted slope ' + f3(measMain.point) + ', 95% band [' + fmt(measMain.lower, { digits: 4 }) + ', ' + fmt(measMain.upper, { digits: 4 }) + ']'),
        el('span', { class: 've-comparison__keyNote', text: 'The two lines coincide at this setting.' })));

      var biasSlider = ui.slider({
        label: 'Field strength (the model’s preference for pattern 1, +)',
        min: 0, max: pts.length - 1, step: 1, value: S.sel,
        marks: pts.map(function (p, i) { return { value: i }; }),
        format: function (v) { return fmt(100 * pts[Math.round(v)].q_eq.point, { digits: 2 }) + '%'; },
        tone: 'bias',
        onInput: function (v) { select(Math.round(v), true); }
      });
      p1.appendChild(el('div', { class: 've-comparison__ctl' }, biasSlider.el));

      // linked probability bars for the selected dot
      function barRow(label) {
        var fill = el('span', { class: 've-comparison__barA' });
        var val = el('span', { class: 've-comparison__barVal' });
        var lab = el('span', { class: 've-comparison__barLab' });
        var row = el('div', { class: 've-comparison__barRow' },
          el('div', { class: 've-comparison__barHead' }, lab, val),
          el('div', { class: 've-comparison__bar', 'aria-hidden': 'true' }, fill, el('span', { class: 've-comparison__barB' }),
            el('span', { class: 've-comparison__barHalf' })));
        lab.innerHTML = label;
        return { el: row, fill: fill, val: val, lab: lab };
      }
      var barModel = barRow('<b class="ve-comparison__who" data-who="model">Model:</b> share of samples in mode A');
      barModel.el.setAttribute('data-who', 'model');
      var barChain = barRow('Sampler: share of runs that end in mode A');
      var ratioLine = el('p', { class: 've-comparison__ratio' });
      p1.appendChild(el('div', { class: 've-comparison__bars', 'aria-live': 'polite' }, barModel.el, barChain.el, ratioLine));

      /* ------------------------------------------ panel 2: other settings */
      var p2 = el('section', { class: 've-comparison__panel' });
      p2.appendChild(panelHead(2, 'Other settings, same prediction rule', '<p>' + P2_TEXT + '</p><p>' + P1_NOTE + '</p>'));
      p2.appendChild(el('p', { class: 've-comparison__pt', html:
        'Predicted c across, measured c up. An interval that crosses the dashed line agrees.' }));

      var tSet = ui.toggle({
        label: 'Vary',
        options: [{ value: 'p', label: 'Pattern count p' }, { value: 'N', label: 'Size N' }],
        value: 'p', onChange: function (v) { S.set = v; S.hl = null; refresh2(); tweenDom(); }
      });
      var tClock = ui.toggle({
        label: 'One-pattern ratio',
        options: [{ value: 'off', label: 'As predicted' }, { value: 'on', label: '× one-pattern ratio' }],
        value: 'off', onChange: function (v) { S.clock = v === 'on'; tween(); pushClock(); }
      });
      p2.appendChild(el('div', { class: 've-comparison__ctl ve-comparison__ctl--row' }, tSet.el, tClock.el));

      var cv2 = ui.canvas(p2, { height: function (w) { return Math.min(Math.round(w * 0.86), 430); }, label: 'Measured c against predicted c, with 95% intervals and the line measured equals predicted.', draw: drawScatter });
      canvases.push(cv2);

      // the view line is a tail that spans both columns under the panels, so the two columns
      // end close together (one column on a phone: the tail follows panel 2)
      var tail = el('div', { class: 've-comparison__tail' });
      var table = el('table', { class: 've-comparison__table' });
      p2.appendChild(el('div', { class: 've-comparison__tablewrap' }, table));
      var rowNote = el('p', { class: 've-comparison__rowNote' });
      var rowNoteText = el('span');
      var rowNoteBtn = infoBtn('About the starred row', 'Two estimates of the same c', '');
      rowNote.appendChild(rowNoteText); rowNote.appendChild(rowNoteBtn);
      p2.appendChild(rowNote);
      var explain = el('p', { class: 've-comparison__explain', 'aria-live': 'polite' });
      var explainSpan = el('span');
      var explainBtn = infoBtn('More on this view', 'About this view', '');
      explain.appendChild(explainSpan); explain.appendChild(explainBtn);
      tail.appendChild(explain);

      fig.stage.appendChild(el('div', { class: 've-comparison__panels' }, p1, p2, tail));

      // inside the plate the figure head and note are hidden: a small bar names the figure
      // and its spec card mirrors the note (sources, what a run is, what is computed here)
      var specChip = el('button', { type: 'button', class: 'v2-chip ve-comparison__spec', html: (ui.INFO_SVG || '') + '<span>What runs here</span>' });
      var sc = ui.specCard(specChip, fig.note, { title: 'What runs here' });
      cards.push(sc.el); specOff = sc.disconnect;
      var plab = opts.plate == null ? 'Fig.' : (typeof opts.plate === 'number' || /^\d+$/.test(opts.plate) ? 'Fig. ' + opts.plate : String(opts.plate));
      fig.stage.insertBefore(el('div', { class: 've-comparison__top' },
        el('span', { class: 've-comparison__topk', text: plab }),
        el('span', { class: 've-comparison__tops', text: 'paper Fig. 2b, Table 1, Sec. 5' }), specChip), fig.stage.firstChild);

      fig.setTakeaway('Three patterns at N = 2,048: predicted 0.199, measured 0.199 [0.1908, 0.2095]; other pattern counts and sizes miss by up to 11%.');

      /* ------------------------------------------------------- behaviour */
      function select(i, fromSlider) {
        i = Math.max(0, Math.min(pts.length - 1, i));
        if (i === S.sel && fromSlider) { paintBars(); return; }
        S.sel = i;
        if (!fromSlider) biasSlider.set(i, { silent: true });
        paintBars();
        cv1.redraw();
      }
      function paintBars() {
        var p = pts[S.sel];
        var qe = p.q_eq.point, qc = p.q_chain.point;
        barModel.fill.style.width = (100 * qe) + '%';
        barChain.fill.style.width = (100 * qc) + '%';
        barModel.val.innerHTML = fmt(100 * qe, { digits: 2 }) + '% <i>log-odds ' + fmt(p.logit_q_eq, { digits: 2 }) + '</i>';
        barChain.val.innerHTML = pct1(qc) + ' <i>log-odds ' + fmt(p.logit_q_chain, { digits: 2 }) + '</i>';
        barChain.lab.innerHTML = '<b class="ve-comparison__who" data-who="sampler">Sampler:</b> share of ' + fmt(p.n_retained, { group: true }) + ' runs that end in mode A';
        var ratio = p.logit_q_chain / p.logit_q_eq;
        ratioLine.innerHTML = 'This dot alone: ' + f2(p.logit_q_chain) + ' ÷ ' + f2(p.logit_q_eq) + ' = <b>' + f3(ratio) + '</b>. ' +
          'The slope through all eight dots is <b>' + f3(measMain.point) + '</b>.';
      }

      function predOf(row) { return row.pred * (1 + S.k * (rByN[row.N] - 1)); }
      function rows() { return S.set === 'p' ? setP : setN; }

      function refresh2() {
        var R = rows();
        table.replaceChildren();
        table.appendChild(el('thead', null, el('tr', null,
          el('th', { text: 'Setting' }), el('th', { html: 'Predicted<br><span class="ve-comparison__thf">= <i>c</i><sub>CW</sub> \u00d7 <i>A<sub>p</sub></i></span>' }), el('th', { text: 'Measured [95%]' }), el('th', { text: 'Gap' }))));
        var tb = el('tbody');
        R.forEach(function (r) {
          var pr = predOf(r);
          var inside = pr >= r.lo && pr <= r.hi;
          var gap = (r.meas - pr) / pr;
          var predCell = el('td', { class: 've-num' }, el('span', { class: 've-comparison__predv', text: f3(pr) }),
            el('span', { class: 've-comparison__subv ve-comparison__rule', text: '=\u00a0' + f3(r.ccw) + '\u00a0×\u00a0' + f3(r.amp) + (S.clock ? ' ×\u00a0' + fmt(rByN[r.N], { digits: 3 }) : '') }));
          var tr = el('tr', { dataset: { key: r.key, hl: S.hl === r.key ? 'true' : 'false' } },
            el('th', { scope: 'row' }, r.label, (r.key === 'p3' || r.key === 'N2048') ? el('sup', { class: 've-comparison__star', text: '*' }) : null),
            predCell,
            el('td', { class: 've-num', html: f3(r.meas) + ' <span class="ve-comparison__subv">[' + f3(r.lo) + ', ' + f3(r.hi) + ']</span>' }),
            el('td', { class: 've-num ve-comparison__gap', dataset: { ok: inside ? 'true' : 'false' }, text: inside ? 'inside' : signedPct(gap) }));
          tr.addEventListener('pointerenter', function () { S.hl = r.key; mark(); cv2.redraw(); });
          tr.addEventListener('pointerleave', function () { S.hl = null; mark(); cv2.redraw(); });
          tb.appendChild(tr);
        });
        table.appendChild(tb);
        var rn = 'Two estimates of the same c: this row is a run at one field strength, both signs pooled, with the Curie–Weiss reference (paper Table 1), ' + f3(p3.c_measured) + '; the headline ' + f3(measMain.point) + ' is panel 1’s slope (paper Fig. 2b). Their intervals nearly coincide' + (S.clock ? '; the prediction × the one-pattern ratio, ' + f3(p3clk) + ', lies above both.' : '.');
        rowNoteText.innerHTML = '* A Table 1 run, ' + f3(p3.c_measured) + '; the headline ' + f3(measMain.point) + ' is panel\u00a01’s slope.';
        rowNoteBtn.setBody('<p>' + rn + '</p>');
        var ex = explainText();
        explainSpan.innerHTML = nb(ex.s) + ' ';
        explainBtn.setBody('<p>' + ex.l + '</p>');
        ui.terms(explain);
        if (!booting) cv2.redraw();
      }
      function mark() {
        table.querySelectorAll('tbody tr').forEach(function (tr) { tr.dataset.hl = tr.dataset.key === S.hl ? 'true' : 'false'; });
      }
      // s: the one line shown under the table; l: the full text, in the card behind its info button
      function explainText() {
        if (!S.clock && S.set === 'p') return {
          s: '<b>At N = 2,048,</b> p = 3 and p = 4 agree within their intervals; p = 1 and p = 2 measure 6% and 11% high.',
          l: '<b>At N = 2,048.</b> p = 3 and p = 4 agree within their intervals. ' +
          'p = 1 and p = 2 measure 6% and 11% above the prediction. With the two neighbouring sizes (switch to Size N), the prediction misses by up to 11% away from the paper’s main setting.' };
        if (!S.clock && S.set === 'N') return {
          s: '<b>At p = 3,</b> N = 2,048 agrees; N = 1,024 measures 7% low and N = 4,096 6% high.',
          l: '<b>At p = 3.</b> The model’s log-odds grow in proportion to N, the sampler’s only as √N for a linear ramp (Eq. 4), so a larger network keeps a smaller share. ' +
          'Each prediction uses c<sub>CW</sub> recomputed at that N by the exact CW chain, with the same schedule and field. ' +
          'N = 2,048, where the schedule’s sweep counts were set, agrees. N = 1,024 measures 7% below the prediction and N = 4,096 6% above it. ' +
          'Measured c falls about as N<sup>−0.36</sup>, against N<sup>−0.46</sup> for the prediction c<sub>pred</sub> (two-point slopes, no uncertainties).' };
        var clockWhy = 'The Curie–Weiss reference updates at random times (expected sweeps) with interaction strength 1/(N−1); Hopfield runs use permutation sweeps (each spin once, in a fresh random order), with 1/N. ' +
          'At p = 1, where no patterns compete, the measured c is r(N) times c<sub>CW</sub>; the ratio mixes both differences. ';
        if (S.set === 'p') return {
          s: '<b>Ratio applied, r = ' + fmt(r2048, { digits: 3 }) + '.</b> p = 1 now agrees, but p = 3 moves to ' + f3(p3clk) + ', outside its interval, and p = 2 still misses by 4%.',
          l: clockWhy + '<b>At N = 2,048, r = ' + fmt(r2048, { digits: 3 }) + '.</b> Multiplying by it brings p = 1 into agreement (r comes from separate one-pattern runs; the Table 1 p = 1 row itself gives about 1.06, 0.127/0.120), ' +
          'but moves p = 3 to ' + f3(p3clk) + ', outside its interval, and p = 2 still misses by 4%. Errors in the approximations may therefore cancel (paper Sec. 5).' };
        return {
          s: '<b>Ratio applied.</b> N = 1,024 still measures 18% below and N = 4,096 4% above; the trend with size is not explained.',
          l: clockWhy + '<b>r = ' + fmt(rByN[1024], { digits: 3 }) + ' at N = 1024 and ' + fmt(rByN[4096], { digits: 3 }) + ' at N = 4096.</b> ' +
          'Ratio applied: the predictions become 0.319 and 0.153, and measured c is 18% below at N = 1024 and 4% above at N = 4096; the trend with size is not explained.' };
      }

      function tweenDom() {
        var to = DOM[S.set];
        if (domRaf) cancelAnimationFrame(domRaf);
        if (reduced()) { S.dom = to.slice(); cv2.redraw(); return; }
        var from = S.dom.slice(), t0 = 0, dur = 450;
        function step(now) {
          if (destroyed) return;
          if (!t0) t0 = now;
          var u = Math.min(1, (now - t0) / dur), e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
          S.dom = [from[0] + (to[0] - from[0]) * e, from[1] + (to[1] - from[1]) * e];
          cv2.redraw();
          if (u < 1) domRaf = requestAnimationFrame(step); else domRaf = 0;
        }
        domRaf = requestAnimationFrame(step);
      }

      function tween() {
        var target = S.clock ? 1 : 0;
        if (tweenRaf) cancelAnimationFrame(tweenRaf);
        if (reduced()) { S.k = target; refresh2(); return; }
        var from = S.k, t0 = 0, dur = 520;
        refreshTableOnly();
        function step(now) {
          if (destroyed) return;
          if (!t0) t0 = now;
          var u = Math.min(1, (now - t0) / dur);
          var e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
          S.k = from + (target - from) * e;
          cv2.redraw();
          if (u < 1) tweenRaf = requestAnimationFrame(step);
          else { tweenRaf = 0; S.k = target; refresh2(); }
        }
        tweenRaf = requestAnimationFrame(step);
      }
      function refreshTableOnly() { var k = S.k; S.k = S.clock ? 1 : 0; refresh2(); S.k = k; }

      /* ---------------------------------------------------------- drawing */
      function drawFit(ctx, w, h, c) {
        var small = w < 420;
        var padL = small ? 30 : 38, padR = 12, padT = 22, padB = 44;
        var side = Math.min(w - padL - padR, h - padT - padB);
        var x0 = padL + Math.max(0, (w - padL - padR - side) / 2), y0 = padT;
        var D = 4;
        var sx = ui.scale.linear([-D, D], [x0, x0 + side]);
        var sy = ui.scale.linear([-D, D], [y0 + side, y0]);
        var tk = [-4, -3, -2, -1, 0, 1, 2, 3, 4];
        ui.plot.axis(ctx, sx, { side: 'bottom', at: y0 + side, ticks: tk, grid: [y0, y0 + side], format: function (v) { return v === 0 ? '0' : fmt(v); }, c: c,
          label: small ? 'model log-odds →' : 'model log-odds for mode A →', labelAlign: 'right' });
        ui.plot.axis(ctx, sy, { side: 'left', at: x0, ticks: tk, grid: [x0, x0 + side], format: function (v) { return v === 0 ? '0' : fmt(v); }, c: c });
        ctx.save();
        ctx.beginPath(); ctx.rect(x0, y0, side, side); ctx.clip();
        // zero lines
        ctx.strokeStyle = c.ruleStrong; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x0, Math.round(sy(0)) + 0.5); ctx.lineTo(x0 + side, Math.round(sy(0)) + 0.5);
        ctx.moveTo(Math.round(sx(0)) + 0.5, y0); ctx.lineTo(Math.round(sx(0)) + 0.5, y0 + side); ctx.stroke();
        // measured interval wedge
        ctx.fillStyle = ui.alpha(c.data, 0.18);
        ctx.beginPath();
        ctx.moveTo(sx(-D), sy(-D * measMain.lower)); ctx.lineTo(sx(D), sy(D * measMain.upper));
        ctx.lineTo(sx(D), sy(D * measMain.lower)); ctx.lineTo(sx(-D), sy(-D * measMain.upper)); ctx.closePath(); ctx.fill();
        // c = 1
        ui.plot.line(ctx, [-D, D], [-D, D], sx, sy, { color: c.ink3, width: 1 });   // solid hairline: only the prediction is dashed
        // measured fit
        ui.plot.line(ctx, [-D, D], [-D * measMain.point, D * measMain.point], sx, sy, { color: ui.alpha(c.data, 0.35), width: 7 });
        // prediction
        ui.plot.line(ctx, [-D, D], [-D * predMain, D * predMain], sx, sy, { color: c.theory, width: 2.5, dash: [7, 6] });

        // selected construction: one column; teal up to the model's log-odds (on the c = 1 line),
        // ink up to the sampler's log-odds (the dot)
        var p = pts[S.sel];
        var X = p.logit_q_eq, Y = p.logit_q_chain;
        var px = Math.round(sx(X)) + 0.5;
        ctx.lineCap = 'butt';
        ctx.strokeStyle = ui.alpha(c.bias, 0.55); ctx.lineWidth = 6;
        ctx.beginPath(); ctx.moveTo(px, sy(0)); ctx.lineTo(px, sy(X)); ctx.stroke();
        ctx.fillStyle = c.bias; ctx.beginPath(); ctx.arc(px, sy(X), 4, 0, 7); ctx.fill();
        ctx.strokeStyle = c.data; ctx.lineWidth = 6;
        ctx.beginPath(); ctx.moveTo(px, sy(0)); ctx.lineTo(px, sy(Y)); ctx.stroke();
        ctx.restore();

        var tsz = 11;
        // faithful line label: in the half of the dashed line opposite the selected column, on the side
        // away from the dots (below the line on the left, above it on the right), so nothing crosses it
        var fa = small ? 2.05 : 2.3, fl = pts[S.sel].logit_q_eq > 0 ? -fa : fa;
        ctx.save();
        ctx.translate(sx(fl), sy(fl));
        ctx.rotate(-Math.PI / 4);
        ui.plot.text(ctx, small ? 'faithful, c = 1' : 'faithful sampler, c = 1', 0, fl < 0 ? 13 : -8, { c: c, color: c.ink3, size: tsz, align: 'center', halo: true });
        ctx.restore();
        // coin-flip label on the zero line, at the end away from the selected column's labels
        if (pts[S.sel].logit_q_eq > 0) ui.plot.text(ctx, 'c = 0: coin flip', sx(D) - 4, sy(0) + 16, { c: c, color: c.ink3, size: tsz, align: 'right', halo: true });
        else ui.plot.text(ctx, 'c = 0: coin flip', sx(-D) + 4, sy(0) - 8, { c: c, color: c.ink3, size: tsz, align: 'left', halo: true });
        ui.plot.text(ctx, '\u2191 sampler log-odds', x0 + 2, y0 - 8, { c: c, color: c.ink2, size: 11, weight: 500, align: 'left' });

        // points with Wilson intervals
        pts.forEach(function (q, i) {
          var lo = logit(q.q_chain.lower), hi = logit(q.q_chain.upper);
          ui.plot.errorbar(ctx, q.logit_q_eq, lo, hi, sx, sy, { color: c.data, width: 1.5, cap: 3 });
          var on = i === S.sel;
          ctx.beginPath(); ctx.arc(sx(q.logit_q_eq), sy(q.logit_q_chain), on ? 5.5 : 4, 0, 7);
          ctx.fillStyle = c.data; ctx.fill();
          ctx.lineWidth = 1.5; ctx.strokeStyle = c.surface; ctx.stroke();
          if (on) {
            ctx.beginPath(); ctx.arc(sx(q.logit_q_eq), sy(q.logit_q_chain), 10, 0, 7);
            ctx.lineWidth = 2; ctx.strokeStyle = c.ink; ctx.stroke();
          }
        });
        // column labels on the side away from the neighbouring dots and the fit line:
        // X > 0: left of the column, model at the tip, sampler just above its dot (neighbours to the left sit lower);
        // X < 0: mirrored, right of the column and below.
        var right = X > 0, al = right ? 'right' : 'left', lx = px + (right ? -11 : 11);
        if (right) {
          var yM = Math.min(sy(X) - 3, sy(0) - 36), yS = Math.max(Math.min(sy(Y) - 9, sy(0) - 22), yM + 15);
          ui.plot.text(ctx, 'model ' + f2(X), lx, yM, { c: c, color: c.bias, size: tsz, weight: 600, align: al, baseline: 'bottom', halo: true });
          ui.plot.text(ctx, 'sampler ' + f2(Y), lx, yS, { c: c, color: c.data, size: tsz, weight: 600, align: al, baseline: 'bottom', halo: true });
        } else {
          var yM2 = Math.max(sy(X) + 3, sy(0) + 36), yS2 = Math.min(Math.max(sy(Y) + 9, sy(0) + 22), yM2 - 15);
          ui.plot.text(ctx, 'model ' + f2(X), lx, yM2, { c: c, color: c.bias, size: tsz, weight: 600, align: al, baseline: 'top', halo: true });
          ui.plot.text(ctx, 'sampler ' + f2(Y), lx, yS2, { c: c, color: c.data, size: tsz, weight: 600, align: al, baseline: 'top', halo: true });
        }
        cv1.hit = { sx: sx, sy: sy };
      }

      // per-set axis domains, tweened when the Vary toggle changes; each holds the clock-corrected values too
      var DOM = { p: [0.11, 0.25], N: [0.12, 0.33] };
      function drawScatter(ctx, w, h, c) {
        var small = w < 420;
        var padL = small ? 40 : 46, padR = 14, padT = 22, padB = 44;
        var side = Math.min(w - padL - padR, h - padT - padB);
        var x0 = padL + Math.max(0, (w - padL - padR - side) / 2), y0 = padT;
        var lo = S.dom[0], hi = S.dom[1];
        var sx = ui.scale.linear([lo, hi], [x0, x0 + side]);
        var sy = ui.scale.linear([lo, hi], [y0 + side, y0]);
        var step = hi - lo < 0.17 ? (small ? 0.04 : 0.02) : 0.05, tk = [];
        for (var t = Math.ceil(lo / step - 1e-9) * step; t <= hi + 1e-9; t += step) tk.push(+t.toFixed(2));
        ui.plot.axis(ctx, sx, { side: 'bottom', at: y0 + side, ticks: tk, grid: [y0, y0 + side], format: function (v) { return v.toFixed(2); }, c: c, label: 'predicted c →', labelAlign: 'right' });
        ui.plot.axis(ctx, sy, { side: 'left', at: x0, ticks: tk, grid: [x0, x0 + side], format: function (v) { return v.toFixed(2); }, c: c });
        ui.plot.text(ctx, '↑ measured c', x0 + 2, y0 - 8, { c: c, color: c.ink2, size: 11, weight: 500 });
        ctx.save(); ctx.beginPath(); ctx.rect(x0, y0, side, side); ctx.clip();
        ui.plot.line(ctx, [lo, hi], [lo, hi], sx, sy, { color: c.ink3, width: 1.5, dash: [5, 4] });
        ctx.restore();
        var tsz = 11;
        // equality label rotated along its line, in a part of the line no row uses
        var eqAt = S.set === 'p' ? (small ? 0.159 : 0.152) : 0.288, eqBelow = S.set === 'p';
        var eqU = lo + (eqAt - DOM[S.set][0]) / (DOM[S.set][1] - DOM[S.set][0]) * (hi - lo);
        ctx.save();
        ctx.translate(sx(eqU), sy(eqU)); ctx.rotate(-Math.PI / 4);
        ui.plot.text(ctx, 'measured = predicted', 0, eqBelow ? 12 : -8, { c: c, color: c.ink3, size: tsz, align: 'center', halo: true });
        ctx.restore();

        var R = rows();
        var hit = [];
        var axisY = y0 + side;
        var obst = [];   // boxes the row labels must not cover: dots, intervals, ghosts, the equality label
        var drawn = [];
        R.forEach(function (r) {
          var base = r.pred, now = predOf(r);
          var dim = S.hl && S.hl !== r.key;
          ctx.save();
          ctx.globalAlpha = dim ? 0.3 : 1;
          // the prediction itself, in theory purple, as a tick on the predicted axis
          var tx = sx(now);
          ctx.fillStyle = c.theory;
          ctx.beginPath(); ctx.moveTo(tx, axisY - 1); ctx.lineTo(tx - 4.5, axisY - 9); ctx.lineTo(tx + 4.5, axisY - 9); ctx.closePath(); ctx.fill();
          ctx.strokeStyle = ui.alpha(c.theory, 0.35); ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
          ctx.beginPath(); ctx.moveTo(tx + 0.5, axisY - 9); ctx.lineTo(tx + 0.5, sy(r.lo) + 2); ctx.stroke(); ctx.setLineDash([]);
          var moved = S.k > 0.001;
          if (moved) {
            // ghost of the uncorrected prediction on the axis, and the move along it
            ctx.strokeStyle = c.theory; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.moveTo(sx(base), axisY - 1); ctx.lineTo(sx(base) - 4.5, axisY - 9); ctx.lineTo(sx(base) + 4.5, axisY - 9); ctx.closePath(); ctx.stroke();
            // the move is an annotation, not data: a thin, faded arrow
            if (tx - sx(base) > 6) ui.plot.arrow(ctx, sx(base), axisY - 14, tx, axisY - 14, { color: ui.alpha(c.theory, 0.5), width: 1, head: 4 });
          }
          // haloed interval, then the dot, then the interval line again on top, so short intervals stay visible
          ui.plot.errorbar(ctx, now, r.lo, r.hi, sx, sy, { color: c.surface, width: 5, cap: 6 });
          ui.plot.errorbar(ctx, now, r.lo, r.hi, sx, sy, { color: c.data, width: 2, cap: 4 });
          ctx.beginPath(); ctx.arc(tx, sy(r.meas), 4.5, 0, 7);
          ctx.fillStyle = c.data; ctx.fill(); ctx.lineWidth = 1.5; ctx.strokeStyle = c.surface; ctx.stroke();
          ui.plot.errorbar(ctx, now, r.lo, r.hi, sx, sy, { color: c.data, width: 1.5, cap: 4 });
          if (moved) {
            // ghost dot where the row sat before the correction, joined to the dot
            ctx.globalAlpha = (dim ? 0.3 : 1) * 0.75;
            ctx.strokeStyle = c.theory; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(sx(base), sy(r.meas)); ctx.lineTo(tx, sy(r.meas)); ctx.stroke(); ctx.setLineDash([]);
            ctx.beginPath(); ctx.arc(sx(base), sy(r.meas), 4, 0, 7); ctx.stroke();
            obst.push([Math.min(sx(base), tx) - 6, sy(r.meas) - 6, Math.max(sx(base), tx) + 6, sy(r.meas) + 6]);
          }
          if (S.hl === r.key) { ctx.beginPath(); ctx.arc(tx, sy(r.meas), 10, 0, 7); ctx.lineWidth = 2; ctx.strokeStyle = c.ink; ctx.stroke(); }
          ctx.restore();
          obst.push([tx - 7, sy(r.hi) - 3, tx + 7, sy(r.lo) + 3]);
          drawn.push({ r: r, now: now, tx: tx, dim: dim });
          hit.push({ key: r.key, x: tx, y: sy(r.meas) });
        });
        // the rotated equality label, as a chain of small boxes along it
        (function () {
          ctx.save(); ctx.font = ui.font('mono', tsz, 400);
          var L = ctx.measureText('measured = predicted').width; ctx.restore();
          var off = eqBelow ? 9 : -11, cx = sx(eqU) + off / Math.SQRT2, cy = sy(eqU) + off / Math.SQRT2;
          for (var t = -L / 2; t <= L / 2; t += 6) obst.push([cx + t / Math.SQRT2 - 5, cy - t / Math.SQRT2 - 5, cx + t / Math.SQRT2 + 5, cy - t / Math.SQRT2 + 5]);
        })();

        // label placement: try the hand-set side first, then the others, and keep the first box that
        // stays inside the plot and clear of the dashed line, every mark and every earlier label
        var PREF = { p1: 't', p2: 't', p3: 'b', p4: w < 520 ? 'b' : 'r', N1024: 'b', N2048: S.clock ? 'r' : 'tl', N4096: 't' };
        var ORDER = ['tl', 't', 'tr', 'ul', 'r', 'l', 'b', 'bl', 'br'];
        function hitsLine(b) {
          // the y = x line in pixels is (px - x0) + (py - y0 - side) = 0, since both axes share one scale
          var f = function (px, py) { return px - x0 + py - y0 - side; };
          var v = [f(b[0], b[1]), f(b[2], b[1]), f(b[0], b[3]), f(b[2], b[3])];
          return Math.min.apply(null, v) < 0 && Math.max.apply(null, v) > 0;
        }
        function overlap(a, b) { return a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3]; }
        var placed = [];
        drawn.forEach(function (d) {
          var r = d.r, tx = d.tx, dy = sy(r.meas), hy = sy(r.hi), ly0 = sy(r.lo);
          var gap = (r.meas - d.now) / d.now, inside = d.now >= r.lo && d.now <= r.hi;
          var sub = inside ? 'inside' : signedPct(gap);
          ctx.save();
          ctx.font = ui.font('mono', tsz + 0.5, 600); var w1 = ctx.measureText(r.label).width;
          ctx.font = ui.font('mono', tsz, inside ? 600 : 400); var w2 = ctx.measureText(sub).width;
          ctx.restore();
          var tw = Math.max(w1, w2);
          var G = {
            ul: [tx - 10, 'right', dy - 9], l: [tx - 10, 'right', dy + 4], r: [tx + 10, 'left', dy + 4],
            t: [tx, 'center', hy - 22], tl: [tx + 4, 'right', hy - 22], tr: [tx - 4, 'left', hy - 22],
            b: [tx, 'center', ly0 + 16], bl: [tx + 4, 'right', ly0 + 16], br: [tx - 4, 'left', ly0 + 16]
          };
          var tries = [PREF[r.key] || 'ul'].concat(ORDER), best = null;
          tries.forEach(function (k) {
            if (best && best.cost === 0) return;
            var g = G[k], x = g[0], al = g[1], ly = g[2];
            var left = al === 'right' ? x - tw : al === 'center' ? x - tw / 2 : x;
            // slide a little to stay clear of the y tick labels and the canvas edge, never far from the dot
            var shift = Math.max(0, x0 + 3 - left) - Math.max(0, left + tw - (w - 2));   // the right margin may hold a label
            if (Math.abs(shift) > 26) return;
            x += shift; left += shift;
            var box = [left - 3, ly - tsz - 2, left + tw + 3, ly + 16];
            if (box[1] < y0 - 4 || box[3] > axisY - 16) return;
            var cost = (hitsLine(box) ? 2 : 0);
            obst.forEach(function (o) { if (overlap(box, o)) cost += 1; });
            placed.forEach(function (o) { if (overlap(box, o)) cost += 3; });
            if (!best || cost < best.cost) best = { cost: cost, x: x, al: al, ly: ly, box: box };
          });
          if (!best) { var g = G[PREF[r.key] || 'ul']; best = { x: g[0], al: g[1], ly: g[2], box: [0, 0, 0, 0] }; }
          placed.push(best.box);
          ctx.save();
          ctx.globalAlpha = d.dim ? 0.3 : 1;
          ui.plot.text(ctx, r.label, best.x, best.ly, { c: c, color: c.ink, size: tsz + 0.5, weight: 600, align: best.al, halo: true, haloWidth: 6 });
          ui.plot.text(ctx, sub, best.x, best.ly + 13, { c: c, color: inside ? c.ink : c.ink2, size: tsz, weight: inside ? 600 : 400, align: best.al, halo: true, haloWidth: 6 });
          ctx.restore();
        });
        cv2.hit = hit;
      }

      function logit(q) { return Math.log(q / (1 - q)); }

      /* ------------------------------------------------- canvas pointers */
      cv1.canvas.addEventListener('click', function (e) { pick1(e); });
      cv1.canvas.addEventListener('pointermove', function (e) { if (e.pointerType === 'mouse') { var i = nearest1(e); cv1.canvas.style.cursor = i >= 0 ? 'pointer' : ''; } });
      function nearest1(e) {
        if (!cv1.hit) return -1;
        var q = cv1.local(e), best = -1, bd = 22 * 22;
        pts.forEach(function (p, i) {
          var dx = cv1.hit.sx(p.logit_q_eq) - q.x, dy = cv1.hit.sy(p.logit_q_chain) - q.y, d = dx * dx + dy * dy;
          if (d < bd) { bd = d; best = i; }
        });
        if (best < 0) {
          // tap anywhere in a column also selects the dot in that column
          var bx = 1e9;
          pts.forEach(function (p, i) { var dx = Math.abs(cv1.hit.sx(p.logit_q_eq) - q.x); if (dx < 16 && dx < bx) { bx = dx; best = i; } });
        }
        return best;
      }
      function pick1(e) { var i = nearest1(e); if (i >= 0) select(i, false); }

      cv2.canvas.addEventListener('pointermove', function (e) {
        if (!cv2.hit) return;
        var q = cv2.local(e), best = null, bd = 24 * 24;
        cv2.hit.forEach(function (p) { var dx = p.x - q.x, dy = p.y - q.y, d = dx * dx + dy * dy; if (d < bd) { bd = d; best = p.key; } });
        if (best !== S.hl) { S.hl = best; mark(); cv2.redraw(); }
      });
      cv2.canvas.addEventListener('pointerleave', function () { if (S.hl) { S.hl = null; mark(); cv2.redraw(); } });

      // first draw: the canvases' own resize observers paint them on the next frame, so mounting
      // never forces a synchronous page layout (that was a 50-70 ms long task on the assembled page)
      // one clock state per chapter: start from the chapter figure's switch (Fig. 7) and keep the two
      // switches in step both ways, so the reader never sees the clock on above and off here
      var hostCh = root.closest ? root.closest('[data-chapter]') : null;
      var chBtns = hostCh ? Array.prototype.slice.call(hostCh.querySelectorAll('[data-vernier] [data-clock]')) : [];
      function chClockOn() { return chBtns.some(function (b) { return b.dataset.clock === '1' && b.getAttribute('aria-checked') === 'true'; }); }
      function pushClock() {
        if (!chBtns.length || chClockOn() === S.clock) return;
        chBtns.forEach(function (b) { if ((b.dataset.clock === '1') === S.clock) b.click(); });
      }
      function pullClock(e) {
        var on = e.currentTarget.dataset.clock === '1';
        if (on === S.clock) return;
        S.clock = on; tClock.set(on ? 'on' : 'off', { silent: true }); tween();
      }
      chBtns.forEach(function (b) { b.addEventListener('click', pullClock); });
      chOff = function () { chBtns.forEach(function (b) { b.removeEventListener('click', pullClock); }); };
      if (chClockOn()) { S.clock = true; S.k = 1; tClock.set('on', { silent: true }); }

      // closing the Plot drawer drops its deep link, so a reload or a shared URL does not reopen it
      var drawerEl = root.closest ? root.closest('details') : null;
      function onDrawerToggle() {
        if (drawerEl.open || !drawerEl.id || location.hash !== '#' + drawerEl.id) return;
        try { history.replaceState(history.state, '', '#' + (hostCh && hostCh.dataset.chapter ? hostCh.dataset.chapter : '')); } catch (e) {}
      }
      if (drawerEl) { drawerEl.addEventListener('toggle', onDrawerToggle); drawerOff = function () { drawerEl.removeEventListener('toggle', onDrawerToggle); }; }

      var booting = true;
      paintBars();
      refresh2();
      booting = false;
    }

    return {
      destroy: function () {
        destroyed = true;
        if (tweenRaf) cancelAnimationFrame(tweenRaf);
        if (domRaf) cancelAnimationFrame(domRaf);
        canvases.forEach(function (cv) { cv.destroy(); });
        if (specOff) specOff();
        if (chOff) chOff();
        if (drawerOff) drawerOff();
        cards.forEach(function (c) { c.remove(); });
        wrap.remove();
      }
    };
  }

  VE.components['comparison'] = { mount: mount };

  /* ---------------------------------------------------------------------------------------
   * Fig. 7 vernier guard (chapter 7). The vernier itself is drawn by lib/shell.js testFigure
   * on a fixed 0.10-0.30 axis. With the one-pattern ratio applied, the N = 1,024 prediction is
   * 0.284 × 1.126 = 0.319: past the axis end, where its triangle would land on the Gap column.
   * This guard watches the vernier's own nodes and, for any prediction past the axis end,
   * holds the triangle at the end of the track, adds an off-scale arrow and prints the value
   * (read from the row's own title, which shell.js writes). It also stars each Gap value while
   * the clock is on, because those gaps are computed on this page.
   * When the axis in shell.js covers every prediction, the clamp never fires and only the
   * stars remain; both steps are idempotent, so a later shell.js fix cannot double them.
   * ------------------------------------------------------------------------------------- */
  function guardVernier(fig) {
    var NS = 'http://www.w3.org/2000/svg', PAD = 10, queued = false;
    function clockOn() { var s = fig.querySelector('[data-vstar]'); return !!(s && s.textContent.trim()); }
    function fix() {
      queued = false;
      var on = clockOn();
      fig.querySelectorAll('.v2-vrow').forEach(function (row) {
        var svg = row.querySelector('svg'), tri = svg && svg.querySelector('.v-pred'), gapEl = row.querySelector('.v2-vrow__g');
        if (gapEl) {
          var t = gapEl.textContent, starred = t.slice(-1) === '*';
          if (on && t && !starred) gapEl.textContent = t + '*';
          else if (!on && starred) gapEl.textContent = t.slice(0, -1);
        }
        if (!tri) return;
        var vb = (svg.getAttribute('viewBox') || '').split(/\s+/), W = +vb[2];
        var m = /translate\(\s*(-?[\d.]+)[\s,]+(-?[\d.]+)\s*\)/.exec(tri.getAttribute('transform') || '');
        var off = svg.querySelector('.vd-off');
        if (!m || !W) return;
        var x = +m[1], y = +m[2], end = W - PAD;
        if (x > end + 0.5) {
          var want = 'translate(' + end.toFixed(1) + 'px, ' + y + 'px)';
          if (tri.style.transform !== want) tri.style.transform = want;
          var pv = /predicted ([\d.]+)/.exec(row.title || ''), txt = (pv ? pv[1] : '') + (on ? '*' : '');
          if (!off) {
            off = document.createElementNS(NS, 'g'); off.setAttribute('class', 'vd-off');
            var lab = document.createElementNS(NS, 'text'); lab.setAttribute('text-anchor', 'end');
            var arr = document.createElementNS(NS, 'path'); arr.setAttribute('fill', 'none');
            off.appendChild(lab); off.appendChild(arr); svg.appendChild(off);
          }
          var lab2 = off.firstChild, arr2 = off.lastChild;
          lab2.setAttribute('x', (end - 9).toFixed(1)); lab2.setAttribute('y', String(y - 6.5));
          if (lab2.textContent !== txt) lab2.textContent = txt;
          /* a small chevron just past the track end: the value lies beyond the scale */
          arr2.setAttribute('d', 'M ' + (end + 7.5).toFixed(1) + ' ' + (y - 11.5) + ' l 3 3 l -3 3');
          if (!off.__shown) { off.__shown = true; requestAnimationFrame(function () { off.classList.add('is-on'); }); }
        } else {
          if (tri.style.transform) tri.style.transform = '';
          if (off) off.remove();
        }
      });
    }
    new MutationObserver(function () { if (!queued) { queued = true; Promise.resolve().then(fix); } })
      .observe(fig, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ['transform', 'viewBox', 'title'] });
    fix();
  }
  function guardAll() { document.querySelectorAll('[data-vernier]').forEach(guardVernier); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', guardAll); else guardAll();
})();
