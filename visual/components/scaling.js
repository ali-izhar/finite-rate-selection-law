/* components/scaling.js — "Bigger and slower: c falls as (N²γ)^(−1/4)"
 *
 * Two plates in one component:
 *   Plate A  the law. Log-log plot of the kept fraction c against N²γ: the Kondepudi–Nelson
 *            curve (computed on the page with VE.sim.knC, the same formula as
 *            src/experiments/cw_linear_sweep.py) and the nine exact-chain runs under linear
 *            sweeps (data/cw_scaling.json "linear_sweep" <- results/experiments/cw_linear_sweep.json).
 *            Controls: sweep budget (1×, 2×, 4×, 16×, 256×) and model size N (1,024 / 2,048 / 4,096).
 *            A slide-rule "ladder" maps sweeps to sampler log-odds (×16 sweeps -> ×2 log-odds).
 *   Plate B  the paper's stepped schedule as a staircase (data/schedule.json: ladder, tau,
 *            K = ceil(6 tau), 834 sweeps), c_CW = 0.120 (data/c_cw.json), the budget-scale runs
 *            and the size scan on that schedule (data/cw_scaling.json "schedule").
 *
 * Linear-sweep protocol constants (not stored in the JSON; read from
 * ../src/experiments/cw_linear_sweep.py, linear_sweep(): b0 = 0.7, b1 = 1.6, start in
 * equilibrium at b0; G = G0·2048/N with G0 = 6.125e-4):  sweeps per descent = 0.9 / γ.
 * The budget stops are γ = 0.004 / k, so k = 1, 4, 16 are exactly the computed runs.
 */
(function () {
  'use strict';
  var VE = (window.VE = window.VE || {});
  VE.components = VE.components || {};

  // Glossary terms this chapter links (gamma, schedule, trackingTime, sweep), in the paper's
  // wording. Registered here, as in speciation.js, so the chapter's tips are correct
  // whatever the state of the shared glossary in lib/ui.js.
  if (VE.ui && VE.ui.defineTerm) {
    VE.ui.defineTerm('gamma', { word: 'Cooling rate', sym: 'γ', ml: '<strong>How fast β rises:</strong> the increase per sweep for a linear ramp β = 1 + γt.' });
    VE.ui.defineTerm('schedule', { word: 'Schedule', ml: '<strong>The cooling schedule: the inverse temperatures β and how many sweeps to spend at each.</strong> Faster schedules cost fewer sweeps.' });
    VE.ui.defineTerm('trackingTime', { word: 'Relaxation time', sym: 'τ(β)', ml: '<strong>The inverse spectral gap:</strong> the time for the slowest deviation from equilibrium to shrink by a factor e. The schedule’s dwell at each temperature is ⌈6τ⌉ sweeps.' });
    VE.ui.defineTerm('sweep', { word: 'Sweep', ml: '<strong>N single-spin Gibbs updates:</strong> an expected sweep (Curie–Weiss: random times, one update per variable on average) or a permutation sweep (Hopfield: each variable once, in a fresh random order). Sweeps measure sampling time.' });
  }

  var B0 = 0.7, B1 = 1.6;            // cw_linear_sweep.py linear_sweep(b0, b1)
  var GAMMA0 = 0.004;                // fastest computed sweep; budget k -> gamma = GAMMA0 / k
  var G0 = 6.125e-4;                 // src/hopfield/protocol.py G0 (paper Sec. 4: G = ±6.125e-4)
  var STOPS = [1, 2, 4, 16, 256];
  var SIZES = [1024, 2048, 4096];
  var SCALES = [0.25, 0.5, 1, 2, 4, 8];

  function gammaOf(k) { return GAMMA0 / k; }
  function Gof(N) { return G0 * 2048 / N; }

  VE.components.scaling = {
    mount: function (root, opts) {
      opts = opts || {};
      var ui = VE.ui, sim = VE.sim, el = ui.el, fmt = ui.fmt, plot = ui.plot;
      var reduced = ui.reducedMotion;

      var state = { N: 2048, k: 1, scale: 1, hover: null, hoverLevel: null, guessed: null };
      // the answer (the 16× arrow, the ladder's right rail, the ×2 note) stays hidden until the reader
      // guesses or moves the budget away from 1× themselves
      function revealed() { return state.guessed != null || state.k !== 1; }
      var data = null;           // cw_scaling.json
      var sched = null;          // schedule.json
      var ccw = null;            // c_cw.json
      var anim = { from: null, t: 1, raf: 0 };
      var disposers = [];

      // The figure keeps its controls, marks and live readouts; explanations, sentences and
      // secondary readouts stay in the DOM (hidden by scaling.css) so the shell's "Reading the
      // figure" card mirrors them verbatim. scaling.css scopes those rules to .ve-scaling--v2.
      var wrap = el('div', { class: 've-scaling ve-scaling--v2' });
      root.appendChild(wrap);

      /* ------------------------------------------------------------ law helpers */
      // Model log-odds do not depend on the schedule speed, so compute them once per N.
      var loeCache = {};
      function modelLO(N) { return loeCache[N] != null ? loeCache[N] : (loeCache[N] = sim.logOddsEq(N, B1, Gof(N))); }
      function lawCX(N, X) { return sim.logitPhi(sim.knX(Gof(N), N, X / (N * N))) / modelLO(N); }   // = sim.knC
      function lawC(N, k) { return lawCX(N, N * N * gammaOf(k)); }
      function lawLogOdds(N, k) { return sim.logitPhi(sim.knX(Gof(N), N, gammaOf(k))); }
      function exactRow(N, k) {
        if (!data) return null;
        var g = gammaOf(k);
        for (var i = 0; i < data.linear_sweep.length; i++) {
          var r = data.linear_sweep[i];
          if (r.N === N && Math.abs(r.gamma - g) < 1e-12) return r;
        }
        return null;
      }
      function sweepsOf(k) { return (B1 - B0) / gammaOf(k); }

      /* ======================================================== PLATE A: the law */
      var figA = ui.figure({
        plate: opts.plate || 'Scaling',
        title: 'Sixteen times the sweeps: how much more preference?',
        className: 've-scaling__fig'
      });
      wrap.appendChild(figA.el);
      // panel head as a small-caps letter plus a serif title (one style on the page)
      function letterHead(fig, k, t) {
        var pl = fig.head.querySelector('.ve-figure__plate');
        if (!pl || !k) return;
        pl.textContent = '';
        pl.classList.add('scl-ph');
        pl.appendChild(el('span', { class: 'scl-ph__k', text: k }));
        pl.appendChild(el('span', { class: 'scl-ph__t', text: t }));
      }
      letterHead(figA, opts.plateK, opts.plateTitle);

      // Lead: what is measured, in one breath.
      var lead = el('p', { class: 'scl-lead', html:
        'Make a guess first. The ladder then gives the answer in one picture: sweeps on one rail, the log-odds the sampler keeps on the other. ' +
        'The plot under it checks the same rule with the exact CW chain over many model sizes and ramp rates.' });
      figA.head.appendChild(lead);

      // Guess first (Nicky Case). The figure is complete without it.
      var guessQ = el('div', { class: 'scl-guess', role: 'group', 'aria-label': 'Guess first' });
      var guessText = el('p', { class: 'scl-guess__q', html:
        '<span class="scl-guess__k">Guess first</span><span class="scl-guess__s">16× the sweeps multiplies the sampler’s log-odds by&nbsp;…</span>' });
      var guessBtns = el('div', { class: 'scl-guess__btns' });
      var guessOut = el('p', { class: 'scl-guess__a', 'aria-live': 'polite' });
      [16, 4, 2].forEach(function (f) {
        var b = ui.button({ label: '×' + f, onClick: function () { answer(f); } });
        b.dataset.f = String(f);
        guessBtns.appendChild(b);
      });
      guessQ.appendChild(guessText); guessQ.appendChild(guessBtns); guessQ.appendChild(guessOut);
      figA.head.appendChild(guessQ);

      function answer(f) {
        state.guessed = f;
        guessBtns.querySelectorAll('button').forEach(function (b) {
          b.dataset.state = +b.dataset.f === 2 ? 'right' : (+b.dataset.f === f ? 'wrong' : '');
          b.setAttribute('aria-pressed', +b.dataset.f === f ? 'true' : 'false');
        });
        var r1 = exactRow(state.N, 1), r16 = exactRow(state.N, 16);
        var ex = (r1 && r16) ? sim.logit(r16.q_exact) / sim.logit(r1.q_exact) : null;
        guessOut.innerHTML = (f === 2 ? '<b>Yes: only ×2.</b> ' : '<b>Only ×2.</b> ') +
          'Log-odds grow as the fourth root of the sweeps, and 16<sup>1/4</sup>\u00a0=\u00a02. ' +
          (ex ? 'In the exact CW chain at N\u00a0=\u00a0' + fmt(state.N) + ' the factor is ' + fmt(ex, { digits: 2 }) + '. ' : '') +
          '<span class="scl-guess__now">The plot now shows the 16× budget.</span>';
        lead.innerHTML = 'The ladder gives the answer in one picture: sweeps on one rail, the log-odds the sampler keeps on the other. ' +
          'The plot under it checks the same rule with the exact CW chain over many model sizes and ramp rates.';
        setK(16, true);
      }

      // Stage: plot (left) + ladder (right)
      var grid = el('div', { class: 'scl-grid' });
      var plotCol = el('div', { class: 'scl-col scl-col--plot' });
      var ladCol = el('div', { class: 'scl-col scl-col--ladder' });
      grid.appendChild(ladCol); grid.appendChild(plotCol);   // the ladder (the answer) before the plot and its formula
      figA.stage.appendChild(grid);

      plotCol.appendChild(el('div', { class: 'scl-panelhead' },
        el('span', { class: 've-label', html: '<span class="scl-nc">c</span> against <span class="scl-nc">N²<span data-term="gamma">γ</span></span>' })));
      plotCol.appendChild(el('p', { class: 'scl-sub scl-sub--a', html:
        'Each ring is the exact CW chain at one model size <i>N</i> and one <span data-term="gamma">ramp rate γ</span> (a linear ramp; more sweeps means a smaller γ); the line is the selection formula (Eq.\u00a03). ' +
        'The x-axis is <i>N</i>²γ because, by the linear-ramp law, <i>c</i> depends on <i>N</i> and γ only through that product; the chapter’s math drawer shows why.' }));
      var plotBox = el('div', { class: 'scl-plotbox' });
      plotCol.appendChild(plotBox);
      var tipA = el('div', { class: 'scl-tip', hidden: true, role: 'status' });
      plotBox.appendChild(tipA);

      ladCol.appendChild(el('div', { class: 'scl-panelhead scl-panelhead--lad' },
        el('span', { class: 've-label', text: 'The ladder: sweeps → log-odds' })));
      var ladBox = el('div', { class: 'scl-ladbox' });
      ladCol.appendChild(ladBox);

      // Plot geometry (shared by draw and hit-testing)
      var geomA = null;
      var XD = [12, 1.6e5], YD = [0.035, 0.5];

      var cvA = ui.canvas(plotBox, {
        height: function (w) { return Math.max(w < 520 ? 340 : 300, Math.min(430, w * 0.68)); },
        label: 'Log-log plot of c, the retained share of the log-odds, against N squared times gamma. The selection formula falls with slope minus one quarter; nine exact-chain runs lie on it.',
        draw: drawA
      });

      function drawA(ctx, w, h, c) {
        var narrow = w < 520;
        var box = ui.inset(w, h, { top: 26, right: narrow ? 10 : 18, bottom: narrow ? 52 : 50, left: narrow ? 44 : 52 });
        var sx = ui.scale.log(XD, [box.x0, box.x1]);
        var sy = ui.scale.log(YD, [box.y1, box.y0]);
        geomA = { sx: sx, sy: sy, box: box };
        var fs = 11;
        var N = state.N;

        // checked range band
        var xa = sx(262.144), xb = sx(67108.864);
        ctx.fillStyle = c.noiseWash;
        ctx.fillRect(xa, box.y0, xb - xa, box.h);

        var xt = [10, 100, 1000, 1e4, 1e5];
        plot.axis(ctx, sx, { side: 'bottom', at: box.y1, ticks: xt, grid: [box.y0, box.y1], c: c, size: fs,
          format: function (v) { return v >= 1e4 ? '10' + ui.sup(Math.round(Math.log10(v))) : fmt(v); } });
        plot.axis(ctx, sy, { side: 'left', at: box.x0, ticks: [0.05, 0.1, 0.2, 0.5], grid: [box.x0, box.x1], c: c, size: fs,
          format: function (v) { return fmt(v); } });
        plot.text(ctx, narrow ? 'range checked' : 'range checked by the exact chain', (xa + xb) / 2, box.y0 + 17,
          { align: 'center', size: fs, color: c.ink3, c: c, halo: true, haloColor: c.surface });
        plot.text(ctx, 'c', box.x0 - 8, box.y0 - 16, { align: 'right', baseline: 'middle', size: fs + 3, weight: 600, color: c.ink2, c: c, role: 'text' });
        plot.text(ctx, 'N²γ  →', box.x1, box.y1 + 34, { align: 'right', size: fs, weight: 500, color: c.ink2, c: c });
        plot.text(ctx, narrow ? '← slower or smaller' : '← more sweeps, or a smaller model', box.x0, box.y1 + 34,
          { align: 'left', size: fs, color: c.ink3, c: c });

        // law curve for the current N (a 120-point polyline; model log-odds cached per N)
        plot.fn(ctx, function (X) { return lawCX(N, X); }, sx, sy, { n: 120, color: c.theory, width: 2.5 });
        // law label: top right, above the curve, away from the arrow (which hugs the curve)

        // budget stops along the curve; labels below-left, away from the arrow and the rings
        STOPS.forEach(function (k) {
          var X = N * N * gammaOf(k), y = lawC(N, k);
          var px = sx(X), py = sy(y), on = k === state.k;
          ctx.save(); ctx.fillStyle = c.theory;
          ctx.beginPath(); ctx.arc(px, py, 2.4, 0, 6.2832); ctx.fill(); ctx.restore();
          // below-left by default; near the y-axis gutter flip to above-right, which on a falling line
          // is also clear of the curve
          ctx.font = ui.font('mono', fs, on ? 700 : 400);
          var flip = px - 9 - ctx.measureText(k + '×').width < box.x0 + 4;
          plot.text(ctx, k + '×', flip ? px + 9 : px - 9, flip ? py - 10 : py + 17, { align: flip ? 'left' : 'right', size: fs, weight: on ? 700 : 400, color: on ? c.ink : c.ink3, c: c, halo: true });
        });

        // current marker (tweened); the rings are drawn on top so a measured run stays visible
        var cur = currentPoint();
        var px0 = sx(cur.X), py0 = sy(cur.c);
        ctx.save();
        // your setting is a selection, so it is grey (the page keeps yellow for c itself)
        ctx.fillStyle = c.ink3; ctx.strokeStyle = c.ink; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(px0, py0, 6, 0, 6.2832); ctx.fill(); ctx.stroke();
        ctx.restore();

        if (data) {
          data.linear_sweep.forEach(function (r) {
            var rr = r.N === 1024 ? 3.5 : r.N === 2048 ? 6 : 8.5;
            ctx.save();
            ctx.strokeStyle = c.data; ctx.lineWidth = state.hover === r ? 2.5 : 1.25;
            ctx.beginPath(); ctx.arc(sx(r.N2gamma), sy(r.c_exact), rr, 0, 6.2832); ctx.stroke();
            ctx.restore();
          });
        }

        // ghost of the previous model size at the same budget: a hollow dashed ring, so it cannot be
        // read as the current (filled) dot
        if (state.prev) {
          var gx = sx(state.prev.X), gy = sy(state.prev.c);
          ctx.save();
          ctx.fillStyle = c.surface; ctx.strokeStyle = c.ink2; ctx.lineWidth = 1.75; ctx.setLineDash([3, 2.5]);
          ctx.beginPath(); ctx.arc(gx, gy, 6, 0, 6.2832); ctx.fill(); ctx.stroke();
          ctx.restore();
        }

        // the 16× arrow, always attached to your setting: when the setting is already 16× or more, the
        // arrow arrives at it from 1/16 of its sweeps (so a guess that moves the budget from 1× to 16×
        // shows exactly that jump); otherwise it leaves the setting toward 16× its sweeps if that fits.
        var mx = box.x0, my = box.y0, wAr = 0;
        if (revealed()) {
        var Xs = N * N * gammaOf(state.k), XA, XB, kA, kB;
        if (state.k < 16 && Xs / 16 >= XD[0] * 1.2) { XA = Xs; XB = Xs / 16; kA = state.k; kB = state.k * 16; }
        else { XA = Xs * 16; XB = Xs; kA = state.k / 16; kB = state.k; }
        var ax = sx(XA), ay = sy(lawCX(N, XA)), bx = sx(XB), by = sy(lawCX(N, XB));
        var off = narrow ? 18 : 22;
        var dx = bx - ax, dy = by - ay, L = Math.hypot(dx, dy) || 1;
        var nx = dy / L, ny = -dx / L;
        if (ny > 0) { nx = -nx; ny = -ny; }
        var ox = nx * off, oy = ny * off;
        // shorten a little at both ends so the arrow does not touch the markers
        var ux = dx / L, uy = dy / L;
        plot.arrow(ctx, ax + ox + ux * 6, ay + oy + uy * 6, bx + ox - ux * 4, by + oy - uy * 4, { color: c.highlightInk, width: 2, head: 8 });
        var rA = exactRow(N, kA), rB = exactRow(N, kB);
        var fac = (rA && rB) ? (sim.logit(rB.q_exact) / sim.logit(rA.q_exact)) : (lawCX(N, XB) / lawCX(N, XA));
        var arT1 = '16× the sweeps', arT2 = '→ log-odds ×' + fmt(fac, { digits: 2 });
        ctx.font = ui.font('mono', fs, 600);
        var wAr1 = ctx.measureText(arT1).width;
        ctx.font = ui.font('mono', fs, 400);
        wAr = Math.max(wAr1, ctx.measureText(arT2).width);
        var cxm = (ax + bx) / 2, cym = (ay + by) / 2;
        mx = cxm + ox * 1.6; my = cym + oy * 1.6;
        // keep the label inside the plot; if the clamp would push it onto the arrow, put it on the
        // other side of the curve (below-left) instead
        if (mx + wAr > box.x1 - 2) {
          var mxC = box.x1 - 2 - wAr;
          if (mxC < cxm + ox + 4) { mx = Math.max(box.x0 + 4, cxm - ox * 1.6 - wAr); my = cym - oy * 1.6 + 6; }
          else mx = mxC;
        }
        plot.text(ctx, arT1, mx, my - 4, { align: 'left', size: fs, weight: 600, color: c.highlightInk, c: c, halo: true });
        plot.text(ctx, arT2, mx, my + 10, { align: 'left', size: fs, color: c.highlightInk, c: c, halo: true });
        }

        // law label: top right, above the curve; pushed below the arrow label if the two would meet
        var lawT1 = 'selection formula';
        var lawT2 = (narrow || !revealed()) ? 'slope −1/4' : 'slope −1/4: c halves each time N²γ grows 16×';
        ctx.font = ui.font('mono', fs, 600);
        var wL = Math.max(ctx.measureText(lawT1).width, ctx.measureText(lawT2).width);
        var lx = box.x1 - 4, ly = box.y0 + (narrow ? 36 : 40);
        var hitX = wAr > 0 && mx < lx + 6 && mx + wAr > lx - wL - 6, hitY = my - 16 < ly + 20 && my + 14 > ly - 12;
        if (hitX && hitY) ly = my + 40;
        plot.text(ctx, lawT1, lx, ly, { align: 'right', size: fs, weight: 600, color: c.theory, c: c, halo: true });
        plot.text(ctx, lawT2, lx, ly + 14, { align: 'right', size: fs, color: c.theory, c: c, halo: true });
      }

      function targetPoint() {
        var r = exactRow(state.N, state.k);
        return { X: state.N * state.N * gammaOf(state.k), c: r ? r.c_exact : lawC(state.N, state.k) };
      }
      function currentPoint() {
        var t = targetPoint();
        if (!anim.from || anim.t >= 1) return t;
        var e = 1 - Math.pow(1 - anim.t, 3);
        return {
          X: Math.exp(Math.log(anim.from.X) + e * (Math.log(t.X) - Math.log(anim.from.X))),
          c: Math.exp(Math.log(anim.from.c) + e * (Math.log(t.c) - Math.log(anim.from.c)))
        };
      }

      // Hover / tap on rings: show the run behind the point; click loads it into the controls.
      function nearestRing(evt) {
        if (!data || !geomA) return null;
        var p = cvA.local(evt), best = null, bd = 1e9;
        data.linear_sweep.forEach(function (r) {
          var d = Math.hypot(geomA.sx(r.N2gamma) - p.x, geomA.sy(r.c_exact) - p.y);
          if (d < bd) { bd = d; best = r; }
        });
        if (bd > 16) return null;
        // several rings share a point: pick the one matching the current N if possible
        var same = data.linear_sweep.filter(function (r) { return Math.abs(r.N2gamma - best.N2gamma) < 1e-6; });
        var pick = same.filter(function (r) { return r.N === state.N; })[0] || same[0];
        return { r: pick, same: same, p: p };
      }
      function showTipA(hit) {
        if (!hit) { tipA.hidden = true; if (state.hover) { state.hover = null; cvA.redraw(); } return; }
        state.hover = hit.r;
        var rows = hit.same.map(function (r) {
          return '<tr><td>N\u00a0=\u00a0' + fmt(r.N) + '</td><td>γ\u00a0=\u00a0' + fmt(r.gamma) + '</td>' + (plotBox.clientWidth < 520 ? '' : '<td>' + fmt(Math.round(sweepsOf(GAMMA0 / r.gamma))) + ' sweeps</td>') + '<td>c\u00a0=\u00a0' + fmt(r.c_exact, { digits: 4 }) + '</td></tr>';
        }).join('');
        var r0 = hit.r;
        tipA.innerHTML = '<div class="scl-tip__h">Exact CW chain · N²γ\u00a0=\u00a0' + fmt(Math.round(r0.N2gamma)) + '</div>' +
          '<table>' + rows + '</table>' +
          '<div class="scl-tip__f">Law: c\u00a0=\u00a0' + fmt(r0.c_KN, { digits: 4 }) + ' · exact / law\u00a0=\u00a0' + fmt(r0.ratio, { digits: 3 }) +
          (hit.same.length > 1 ? '<br>Different N and γ, same N²γ, same c.' : '') + '</div>';
        tipA.hidden = false;
        var bw = plotBox.clientWidth, tw = tipA.offsetWidth, th = tipA.offsetHeight;
        var x = hit.p.x + 14, y = hit.p.y - th - 10;
        if (x + tw > bw - 4) x = hit.p.x - tw - 14;
        if (x < 4) x = 4;
        if (y < 4) y = hit.p.y + 16;
        tipA.style.left = Math.round(x) + 'px'; tipA.style.top = Math.round(y) + 'px';
        cvA.redraw();
      }
      function onMoveA(e) { if (e.pointerType === 'touch') return; showTipA(nearestRing(e)); }
      function onDownA(e) {
        var hit = nearestRing(e);
        showTipA(hit);
        if (hit) {
          var k = Math.round(GAMMA0 / hit.r.gamma);
          sizeT.set(hit.r.N, { silent: true }); state.N = hit.r.N; state.prev = null;
          setK(k, true);
        }
      }
      function onLeaveA() { showTipA(null); }
      cvA.canvas.addEventListener('pointermove', onMoveA);
      cvA.canvas.addEventListener('pointerdown', onDownA);
      cvA.canvas.addEventListener('pointerleave', onLeaveA);
      disposers.push(function () {
        cvA.canvas.removeEventListener('pointermove', onMoveA);
        cvA.canvas.removeEventListener('pointerdown', onDownA);
        cvA.canvas.removeEventListener('pointerleave', onLeaveA);
      });

      /* ---------------------------------------------------------------- ladder */
      var cvL = ui.canvas(ladBox, {
        height: function (w) { return plotBox.clientWidth > w * 1.3 ? (cvA.height || 340) : (revealed() ? 260 : 200); },
        label: 'Two logarithmic scales. Left: sweep budget 1, 2, 4, 16, 256 times. Right: sampler log-odds 1, 1.19, 1.41, 2, 4 times. Eight doublings of sweeps give two doublings of log-odds.',
        draw: drawL
      });
      function drawL(ctx, w, h, c) {
        var narrow = w < 300;
        var fs = 11;
        var top = 30, bot = h - 48;
        var xl = Math.round(w * (narrow ? 0.30 : 0.28)), xr = Math.round(w * (narrow ? 0.70 : 0.70));
        var y = function (oct) { return bot - (bot - top) * oct / 8; };   // log2 scale, 8 doublings
        // column titles
        plot.text(ctx, 'sweeps', xl, top - 16, { align: 'center', size: fs, weight: 600, color: c.ink2, c: c });
        plot.text(ctx, 'log-odds (law)', xr, top - 16, { align: 'center', size: fs, weight: 600, color: c.ink2, c: c });
        // rails
        ctx.save();
        ctx.strokeStyle = c.ruleStrong; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(xl + 0.5, y(0)); ctx.lineTo(xl + 0.5, y(8)); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(xr + 0.5, y(0)); ctx.lineTo(xr + 0.5, y(8)); ctx.stroke();
        // octave ticks on both rails
        for (var o = 0; o <= 8; o++) {
          ctx.beginPath(); ctx.moveTo(xl - 3, Math.round(y(o)) + 0.5); ctx.lineTo(xl + 4, Math.round(y(o)) + 0.5); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(xr - 3, Math.round(y(o)) + 0.5); ctx.lineTo(xr + 4, Math.round(y(o)) + 0.5); ctx.stroke();
        }
        ctx.restore();
        if (!revealed()) {
          STOPS.forEach(function (k) {
            var yl = y(Math.log2(k));
            ctx.save(); ctx.fillStyle = c.ink3; ctx.beginPath(); ctx.arc(xl, yl, 2.5, 0, 6.2832); ctx.fill(); ctx.restore();
            plot.text(ctx, '×' + k, xl - 10, yl, { align: 'right', baseline: 'middle', size: fs, color: c.ink3, c: c });
          });
          plot.text(ctx, '?', xr + 13, (top + bot) / 2, { align: 'left', baseline: 'middle', size: 22, weight: 600, color: c.ink3, c: c });
          plot.text(ctx, 'guess first', w / 2, bot + 20, { align: 'center', size: 11, color: c.ink2, c: c });
          return;
        }
        // right-label positions, spread so they never collide
        var ys = STOPS.map(function (k) { return y(Math.log2(k) / 4); });
        var lab = ys.slice();
        for (var i = 1; i < lab.length; i++) if (lab[i - 1] - lab[i] < 16) lab[i] = lab[i - 1] - 16;
        STOPS.forEach(function (k, i) {
          var on = k === state.k;
          var yl = y(Math.log2(k)), yr = ys[i];
          ctx.save();
          ctx.strokeStyle = on ? c.ink : c.rule;
          ctx.lineWidth = on ? 2.5 : 1.25;
          ctx.beginPath(); ctx.moveTo(xl, yl); ctx.lineTo(xr, yr); ctx.stroke();
          ctx.fillStyle = on ? c.ink : c.ink3;
          ctx.beginPath(); ctx.arc(xl, yl, on ? 4 : 2.5, 0, 6.2832); ctx.fill();
          ctx.beginPath(); ctx.arc(xr, yr, on ? 4 : 2.5, 0, 6.2832); ctx.fill();
          ctx.restore();
          plot.text(ctx, '×' + k, xl - 10, yl, { align: 'right', baseline: 'middle', size: fs, weight: on ? 600 : 400, color: on ? c.ink : c.ink3, c: c });
          var f = Math.pow(k, 0.25);
          var s = '×' + (k === 1 || k === 16 || k === 256 ? fmt(f) : fmt(f, { digits: 2 }));
          if (Math.abs(lab[i] - yr) > 1) {
            ctx.save(); ctx.strokeStyle = c.rule; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(xr + 5, yr); ctx.lineTo(xr + 11, lab[i]); ctx.stroke(); ctx.restore();
          }
          plot.text(ctx, s, xr + 13, lab[i], { align: 'left', baseline: 'middle', size: fs, weight: on ? 600 : 400, color: on ? c.ink : c.ink3, c: c });
        });
        plot.text(ctx, '8 doublings', xl, bot + 18, { align: 'center', size: fs, color: c.ink2, c: c });
        plot.text(ctx, '2 doublings', xr, bot + 18, { align: 'center', size: fs, color: c.ink2, c: c });
        plot.text(ctx, 'fourth root of the sweeps', w / 2, bot + 36, { align: 'center', size: fs, color: c.ink3, c: c });
      }

      /* -------------------------------------------------------------- controls */
      var budgetT = ui.toggle({
        label: 'Sweep budget',
        options: STOPS.map(function (k) { return { value: k, label: k + '×', title: fmt(Math.round(sweepsOf(k))) + ' sweeps' }; }),
        value: state.k,
        onChange: function (v) { setK(v, true); }
      });
      var sizeT = ui.toggle({
        label: 'Model size N',
        options: SIZES.map(function (n) { return { value: n, label: fmt(n) }; }),
        value: state.N,
        onChange: function (v) {
          var from = currentPoint(), t0 = targetPoint();
          state.prev = v === state.N ? state.prev : { N: state.N, X: t0.X, c: t0.c };
          state.N = v; startAnim(from); refreshA();
        }
      });
      function legendA() {
        var items = [
          { role: 'theory', mark: 'line', label: 'selection formula (Eq. 3)' },
          { role: 'data', mark: 'ring', label: 'exact chain, ring size ∝ N' },
          { role: 'ink-3', mark: 'dot', label: 'you' }
        ];
        if (state.prev) items.push({ role: 'ink-2', mark: 'ghost', label: 'same budget at N\u00a0=\u00a0' + fmt(state.prev.N) });
        if (revealed()) items.push({ role: 'highlight-ink', mark: 'arrow', label: '16× the sweeps' });
        var at = figA.legend.parentNode === plotCol;
        figA.setLegend(items);
        if (!at) plotCol.appendChild(figA.legend);
      }
      legendA();
      plotCol.appendChild(el('div', { class: 'scl-ctl' }, budgetT.el, sizeT.el));

      var roSweeps = ui.readout({ label: 'Sweeps' });
      var roC = ui.readout({ label: 'Retained share c', tone: 'highlight-ink' });
      roC.el.classList.add('scl-ro--c');
      var roLO = ui.readout({ label: 'Sampler log-odds' });
      roSweeps.el.classList.add('scl-ro--aux'); roLO.el.classList.add('scl-ro--lo');
      var roWrap = el('div', { class: 've-readouts scl-readouts' }, roC.el, roSweeps.el, roLO.el);
      // The chapter's "Reading the figure" card shows these sentences live. The shell's generic mirror
      // (ui.mirror, for .v2-mirror[data-src]) collapses whitespace with /\s+/, and JS \s matches U+00A0,
      // so the card lost every no-break space and broke 'c =' / '0.133' on a phone. This chapter's
      // live lines carry data-scl-src instead, and this copy collapses only breaking whitespace.
      var cardLines = {};
      function syncCard(key, srcs) {
        var ch = root.closest && root.closest('.v2-ch');
        var p0 = ch && ch.querySelector('.v2-card .v2-mirror[data-scl-src="' + key + '"]');
        if (!p0) return;
        var tg = cardLines[key] || (cardLines[key] = [p0]);
        srcs.forEach(function (src, i) {
          if (!tg[i]) { tg[i] = el('p', { class: 'v2-mirror' }); tg[i - 1].parentNode.insertBefore(tg[i], tg[i - 1].nextSibling); }
          tg[i].textContent = src.textContent.replace(/[^\S\u00a0]+/g, ' ').trim();
        });
      }
      var words = el('p', { class: 'scl-words', 'aria-live': 'polite' });
      figA.controls.appendChild(roWrap);
      figA.controls.appendChild(words);

      // Why: the two log-odds and their ratio
      var whyNote = el('span', { class: 'scl-why__note', text: '16× the sweeps doubles c; 4× the model size halves it.' });
      var why = el('div', { class: 'scl-why' },
        el('div', { class: 'scl-why__cell' },
          el('span', { class: 've-label', text: 'Sampler log-odds' }),
          el('span', { class: 'scl-why__eq', html: '∝ <i>G</i>·√<i>N</i>·γ<sup>−1/4</sup>' }),
          el('span', { class: 'scl-why__note', html: '<i>G</i>\u00a0=\u00a0the field per spin toward the positive mode. Near β\u00a0=\u00a01 the chain stays in the selection interval for about γ<sup>−1/2</sup> sweeps. In that interval the mean spin, which says which mode the chain leans to, gets two pushes:' }),
          el('span', { class: 'scl-why__lines', html:
            '<span>field</span><span>∼ <i>G</i>·γ<sup>−1/2</sup></span>' +
            '<span>fluctuations</span><span>∼ <i>N</i><sup>−1/2</sup>·γ<sup>−1/4</sup></span>' +
            '<span>ratio</span><span>= <i>G</i>√<i>N</i>·γ<sup>−1/4</sup></span>' })),
        el('div', { class: 'scl-why__op', 'aria-hidden': 'true', text: '÷' }),
        el('div', { class: 'scl-why__cell' },
          el('span', { class: 've-label', text: 'Model log-odds' }),
          el('span', { class: 'scl-why__eq', html: '∝ <i>N</i>·<i>G</i>' }),
          el('span', { class: 'scl-why__note', html: 'the two modes differ in log-probability by an amount proportional to <i>N</i>·<i>G</i>: a push of <i>G</i> on each of <i>N</i> spins' })),
        el('div', { class: 'scl-why__op', 'aria-hidden': 'true', text: '=' }),
        el('div', { class: 'scl-why__cell scl-why__cell--res' },
          el('span', { class: 've-label', text: 'Retained share' }),
          el('span', { class: 'scl-why__eq', html: '<i>c</i> ∝ (<i>N</i>²γ)<sup>−1/4</sup>' }),
          whyNote));
      figA.controls.appendChild(why);

      figA.setCaption('Each ring is one exact-chain run; rings of different <i>N</i> at the same <i>N</i>²γ overlap, so only <i>N</i>²γ matters. Tap a ring to load that run.' +
        '<details class="scl-how"><summary>How the rings were computed</summary>' +
        'The line is the selection formula, Eq. 3 (Kondepudi &amp; Nelson 1983; Kondepudi &amp; Gao 1987), computed on this page. Each ring is the continuous-time Gibbs dynamics, reduced exactly to <i>N</i> + 1 counts (one time unit\u00a0=\u00a0one expected sweep: <i>N</i> updates on average), ' +
        'under a linear schedule from β\u00a0=\u00a00.7 to 1.6 at rate γ. The field is scaled with size (<i>G</i>·<i>N</i>\u00a0=\u00a01.254) so the model’s own odds are nearly the same at every <i>N</i>: ' +
        'the positive mode has probability 0.973, log-odds 3.57.</details>');
      figA.setTakeaway('On a linear ramp, 16× the sweeps only doubles c: c ∝ (N²γ)<sup>−1/4</sup>.');
      var noteA = 'Exact CW chain, 9 runs: results/experiments/cw_linear_sweep.json (paper Table 2, App. A) · curve: q\u00a0=\u00a0Φ(G√N·∜(π/γ)), Eq. 3';

      function refreshA() {
        var N = state.N, k = state.k, r = exactRow(N, k);
        var c = r ? r.c_exact : lawC(N, k);
        var lo = r ? sim.logit(r.q_exact) : lawLogOdds(N, k);
        var mlo = r ? sim.logit(r.q_exact) / r.c_exact : sim.logOddsEq(N, B1, Gof(N));
        roSweeps.set(fmt(Math.round(sweepsOf(k))), 'β rises ' + fmt(gammaOf(k), { sig: 3 }) + ' per sweep');
        roC.set(fmt(c, { digits: 3 }), (r ? '' : 'law\u00a0· ') + 'sampler\u00a0' + fmt(lo, { digits: 3 }) + ' ÷\u00a0model\u00a0' + fmt(mlo, { digits: 2 }));
        roLO.set(fmt(lo, { digits: 3 }), 'model’s log-odds: ' + fmt(mlo, { digits: 2 }));
        var c1 = exactRow(N, 1) ? exactRow(N, 1).c_exact : lawC(N, 1);
        words.innerHTML = 'At ' + k + '×, a linear ramp of ' + fmt(Math.round(sweepsOf(k)), { group: true }) + ' sweeps (β rises ' + fmt(gammaOf(k), { sig: 3 }) +
          ' per sweep), at N\u00a0=\u00a0' + fmt(N) + '. ' + (r ? 'Exact CW chain c\u00a0=\u00a0' + fmt(c, { digits: 3 }) + '; the law gives ' + fmt(r.c_KN, { digits: 4 }) + '.'
            : 'Law only, not in the exact CW chain: c\u00a0=\u00a0' + fmt(c, { digits: 3 }) + '.') +
          (function () { var X = N * N * gammaOf(k); return (X < 262 * 0.99 || X > 67109 * 1.01) ? ' This point is outside the range the paper checks (N²γ\u00a0=\u00a0262 to 67,109).' : ''; })() +
          (k === 1 ? '' : ' That is ×' + fmt(c / c1, { digits: 2 }) + ' the 1× value.');
        if (state.prev) {
          var nr = N / state.prev.N, sw = nr * nr;
          words.innerHTML += ' Going from N&nbsp;=&nbsp;' + fmt(state.prev.N) + ' to ' + fmt(N) + ' moved c from ' + fmt(state.prev.c, { digits: 3 }) + ' to ' + fmt(c, { digits: 3 }) +
            ' (×' + fmt(c / state.prev.c, { digits: 2 }) + '): ' + (nr > 1
              ? 'by the law, the same loss as cutting the sweeps by ' + sw + '×.'
              : 'by the law, the same gain as ' + (1 / sw) + '× the sweeps.');
        }
        syncCard('a', [words]);
        if (data) {
          var mx = 0; data.linear_sweep.forEach(function (q) { mx = Math.max(mx, Math.abs(q.ratio - 1)); });
          figA.setNote(noteA + ' · largest gap from Eq. 3: ' + fmt(100 * mx, { digits: 2 }) + '% over N²γ\u00a0=\u00a0262 to 67,109', 'paper');
        } else figA.setNote(noteA, 'paper');
        legendA();
        whyNote.hidden = !revealed();
        var now = guessOut.querySelector('.scl-guess__now');
        if (now) now.hidden = state.k !== 16;
        cvA.redraw();
        // the phone ladder grows once the answer is revealed (its height depends on revealed())
        if (state.wasRevealed !== revealed()) { state.wasRevealed = revealed(); cvL.resize(); } else cvL.redraw();
      }

      function setK(k, animate) {
        var from = currentPoint();
        if (k !== state.k) state.prev = null;
        state.k = k;
        budgetT.set(k, { silent: true });
        if (animate) startAnim(from);
        refreshA();
      }
      function startAnim(from) {
        if (reduced || ui.reducedMotion) { anim.from = null; anim.t = 1; return; }
        anim.from = from; anim.t = 0;
        cancelAnimationFrame(anim.raf);
        var t0 = performance.now();
        (function step(now) {
          anim.t = Math.min(1, (now - t0) / 380);
          cvA.redraw();
          if (anim.t < 1) anim.raf = requestAnimationFrame(step);
        })(t0);
      }

      /* =============================================== PLATE B: the paper schedule */
      var figB = ui.figure({
        plate: opts.plateB || 'Scaling · the paper’s schedule',
        title: 'The schedule the paper runs: ten steps, 834 sweeps',
        className: 've-scaling__fig'
      });
      wrap.appendChild(figB.el);
      letterHead(figB, opts.plateBK, opts.plateBTitle);
      figB.head.appendChild(el('p', { class: 'scl-lead', html:
        'The paper does not use a linear ramp. Its cooling schedule holds β at ten inverse temperatures from 0.6 to 1.6 and dwells ⌈6τ⌉ sweeps at each, where τ is the ' +
        '<span data-term="trackingTime">relaxation time</span> (Table 3), the e-fold time of the slowest deviation from equilibrium. τ peaks at the transition β&nbsp;=&nbsp;1, so the ' +
        'schedule spends most of its budget near it (β\u00a0=\u00a00.95 to 1.05 take 73%). The closed-form law assumes β rises at a constant rate; this schedule dwells at each temperature, ' +
        'longest at the transition, so here the exact CW chain is used instead.' }));
      // one line under the 5b label says why this plate exists (the value chapter 7 builds on)
      figB.head.appendChild(el('p', { class: 'scl-platesub', html: 'reference <i>c</i><sub>CW</sub>\u00a0=\u00a00.120 used in chapter\u00a07' }));

      var gridB = el('div', { class: 'scl-grid scl-grid--b' });
      var stairCol = el('div', { class: 'scl-col scl-col--plot' });
      var sizeCol = el('div', { class: 'scl-col scl-col--ladder' });
      gridB.appendChild(stairCol); gridB.appendChild(sizeCol);
      figB.stage.appendChild(gridB);
      stairCol.appendChild(el('div', { class: 'scl-panelhead' }, el('span', { class: 've-label', html: '<span class="scl-nc">β</span> against sweeps, <span class="scl-nc">N</span>\u00a0=\u00a02,048' })));
      var selLine = el('p', { class: 'scl-sel', 'aria-live': 'polite' });
      stairCol.appendChild(selLine);
      var stairBox = el('div', { class: 'scl-plotbox' });
      stairCol.appendChild(stairBox);
      var tipB = el('div', { class: 'scl-tip', hidden: true, role: 'status' });
      stairBox.appendChild(tipB);
      var strip = el('div', { class: 'scl-strip', role: 'list', 'aria-label': 'Sweeps per temperature' });
      stairCol.appendChild(strip);

      sizeCol.appendChild(el('div', { class: 'scl-panelhead' }, el('span', { class: 've-label', html: '<span class="scl-nc">c</span><sub>CW</sub> by <span class="scl-nc">N</span> · paper Table 1 ÷ A<sub>3</sub>' })));
      sizeCol.appendChild(el('p', { class: 'scl-sub scl-sub--s', text: '834 sweeps at every N.' }));
      var sizeDim = el('p', { class: 'scl-sub scl-sub--dim', hidden: true, text: 'This panel shows the 1× schedule only.' });
      sizeCol.appendChild(sizeDim);
      var sizeBox = el('div', { class: 'scl-ladbox' });
      sizeCol.appendChild(sizeBox);

      var geomB = null;
      var cvB = ui.canvas(stairBox, {
        height: function (w) { return Math.max(280, Math.min(380, w * 0.58)); },
        label: 'The paper’s cooling schedule: ten inverse temperatures beta from 0.6 to 1.6, drawn as a staircase; the width of each step is its number of sweeps. The step at beta 1.0 takes 355 of 834 sweeps. Left and right arrow keys move the highlighted temperature.',
        draw: drawB
      });

      function levelsScaled() {
        var K = sched ? sched.K : sim.PAPER.K;
        return K.map(function (x) { return x * state.scale; });
      }
      function drawB(ctx, w, h, c) {
        var narrow = w < 520, fs = 11;
        var ladder = sched ? sched.ladder : sim.PAPER.ladder;
        var Ks = levelsScaled();
        var total = Ks.reduce(function (a, b) { return a + b; }, 0);
        var base = (sched ? sched.K_total : 834);
        var xmax = Math.max(base, total);
        var box = ui.inset(w, h, { top: 16, right: narrow ? 12 : 20, bottom: 40, left: narrow ? 40 : 48 });
        var sx = ui.scale.linear([0, xmax], [box.x0, box.x1]);
        var sy = ui.scale.linear([0.5, 1.7], [box.y1, box.y0]);
        geomB = { sx: sx, sy: sy, box: box, Ks: Ks, ladder: ladder };

        // undecided region (β < 1) wash
        ctx.fillStyle = c.noiseWash;
        ctx.fillRect(box.x0, sy(1), box.w, box.y1 - sy(1));

        plot.axis(ctx, sx, { side: 'bottom', at: box.y1, ticks: sx.ticks(narrow ? 4 : 6), grid: [box.y0, box.y1], c: c, size: fs,
          format: function (v) { return fmt(v); } });
        plot.axis(ctx, sy, { side: 'left', at: box.x0, ticks: [0.6, 0.8, 1.0, 1.2, 1.4, 1.6], grid: [box.x0, box.x1], c: c, size: fs,
          format: function (v) { return fmt(v, { digits: 1 }); } });
        plot.text(ctx, 'sweeps spent →', box.x1, box.y1 + 30, { align: 'right', size: fs, weight: 500, color: c.ink2, c: c });
        plot.text(ctx, 'β', box.x0 - (narrow ? 30 : 36), box.y0 + 4, { align: 'left', size: fs + 2, weight: 600, color: c.ink2, c: c, role: 'text' });

        // highlight tread (β = 1.0 by default, or the hovered level)
        var hi = state.hoverLevel != null ? state.hoverLevel : 4;
        var x0 = 0; for (var j = 0; j < hi; j++) x0 += Ks[j];
        ctx.save(); ctx.globalAlpha = 1; ctx.fillStyle = ui.alpha(c.ink3, ui.theme() === 'dark' ? 0.22 : 0.15);
        // the band fills the area under the tread only, so it stays clear of the region labels at the top
        var yT = sy(ladder[hi]);
        ctx.fillRect(sx(x0), yT, sx(x0 + Ks[hi]) - sx(x0), box.y1 - yT); ctx.restore();

        // transition rule and the region labels
        plot.hline(ctx, 1, sy, box.x0, box.x1, { color: c.ink2, dash: [5, 4] });
        plot.text(ctx, 'β < 1: one mode', box.x1 - 4, box.y1 - 8, { align: 'right', size: fs, color: c.ink3, c: c, halo: true });
        plot.text(ctx, 'β > 1: two modes', box.x0 + 6, box.y0 + 12, { align: 'left', size: fs, color: c.ink3, c: c, halo: true });
        // the transition label sits under the dashed line at the right end, where the staircase is high
        plot.text(ctx, 'transition β\u00a0=\u00a01', box.x1 - 4, sy(1) + 14, { align: 'right', size: fs, color: c.ink2, c: c, halo: true });

        function stairPath(Kv) {
          ctx.beginPath();
          var x = 0;
          for (var i = 0; i < Kv.length; i++) {
            var yv = sy(ladder[i]);
            if (i === 0) ctx.moveTo(sx(0), yv); else ctx.lineTo(sx(x), yv);
            x += Kv[i];
            ctx.lineTo(sx(x), yv);
          }
        }
        // ghost of the 1× staircase when scaled (named in the legend)
        if (state.scale !== 1) {
          ctx.save(); ctx.strokeStyle = c.ink3; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
          stairPath(sched ? sched.K : sim.PAPER.K); ctx.stroke(); ctx.restore();
          if (state.scale > 1) plot.text(ctx, '1×', sx(base) + 5, sy(ladder[ladder.length - 1]) + 8, { align: 'left', size: fs, weight: 600, color: c.ink3, c: c, halo: true });
        }

        ctx.save();
        ctx.strokeStyle = c.ink; ctx.lineWidth = 2.5; ctx.lineJoin = 'miter';
        stairPath(Ks); ctx.stroke();
        ctx.restore();

        // tread labels: sweeps at each level when there is room. Levels below β = 1 are labelled
        // under their tread so no label lands on the dashed commit line.
        var xx = 0;
        ctx.font = ui.font('mono', fs);
        for (var i = 0; i < Ks.length; i++) {
          var wpx = sx(xx + Ks[i]) - sx(xx);
          var s = sweepTxt(Ks[i]);
          if (wpx > ctx.measureText(s).width + 6) {
            var below = ladder[i] < 1;
            plot.text(ctx, s, (sx(xx) + sx(xx + Ks[i])) / 2, sy(ladder[i]) + (below ? 14 : -6),
              { align: 'center', size: fs, weight: i === hi ? 700 : 400, color: i === hi ? c.ink : c.ink2, c: c, halo: true });
          }
          xx += Ks[i];
        }
        selLine.textContent = 'β\u00a0=\u00a0' + fmt(ladder[hi], { digits: 2 }) + ' · ' + sweepTxt(Ks[hi]) + ' sweeps · ' +
          Math.round(100 * Ks[hi] / total) + '% of the budget';
      }

      function levelAt(evt) {
        if (!geomB) return null;
        var p = cvB.local(evt), x = geomB.sx.invert(p.x), acc = 0;
        if (p.x < geomB.box.x0 || p.x > geomB.box.x1) return null;
        for (var i = 0; i < geomB.Ks.length; i++) { if (x <= acc + geomB.Ks[i]) return { i: i, p: p }; acc += geomB.Ks[i]; }
        return null;
      }
      function showTipB(hit) {
        if (!hit) { tipB.hidden = true; setLevel(null); return; }
        setLevel(hit.i);
        var i = hit.i, tau = sched ? sched.tau[i] : null, K = sched ? sched.K[i] : sim.PAPER.K[i];
        tipB.innerHTML = '<div class="scl-tip__h">Temperature ' + (i + 1) + ' of 10 · β\u00a0=\u00a0' + fmt((sched ? sched.ladder : sim.PAPER.ladder)[i], { digits: 2 }) + '</div>' +
          (tau != null ? '<div>relaxation time τ\u00a0=\u00a0' + fmt(tau, { digits: 1 }) + ' sweeps</div>' : '') +
          '<div>6τ, rounded up: <b>' + K + '</b> sweeps' + (state.scale !== 1 ? ' · ×' + fmtScale(state.scale) + '\u00a0=\u00a0' + fmt(K * state.scale, { sig: 4 }) : '') + '</div>';
        tipB.hidden = false;
        var bw = stairBox.clientWidth, tw = tipB.offsetWidth, th = tipB.offsetHeight;
        var x = hit.p.x + 14, y = hit.p.y - th - 10;
        if (x + tw > bw - 4) x = hit.p.x - tw - 14;
        if (x < 4) x = 4; if (y < 4) y = hit.p.y + 16;
        tipB.style.left = Math.round(x) + 'px'; tipB.style.top = Math.round(y) + 'px';
      }
      function onMoveB(e) { if (e.pointerType === 'touch') return; showTipB(levelAt(e)); }
      function onDownB(e) { showTipB(levelAt(e)); }
      function onLeaveB() { showTipB(null); }
      cvB.canvas.addEventListener('pointermove', onMoveB);
      cvB.canvas.addEventListener('pointerdown', onDownB);
      cvB.canvas.addEventListener('pointerleave', onLeaveB);
      // keyboard: the staircase itself is the level picker (the plate hides the strip of buttons)
      cvB.canvas.tabIndex = 0;
      function onKeyB(e) {
        if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight' && e.key !== 'Home' && e.key !== 'End') return;
        e.preventDefault(); e.stopPropagation();
        var n = (sched ? sched.ladder : sim.PAPER.ladder).length;
        var cur = state.hoverLevel != null ? state.hoverLevel : 4;
        var i = e.key === 'Home' ? 0 : e.key === 'End' ? n - 1 : Math.max(0, Math.min(n - 1, cur + (e.key === 'ArrowRight' ? 1 : -1)));
        state.keyLevel = i; setLevel(i);
      }
      function onBlurB() { if (state.keyLevel != null) { state.keyLevel = null; setLevel(null); } }
      cvB.canvas.addEventListener('keydown', onKeyB);
      cvB.canvas.addEventListener('blur', onBlurB);
      disposers.push(function () { cvB.canvas.removeEventListener('keydown', onKeyB); cvB.canvas.removeEventListener('blur', onBlurB); });
      disposers.push(function () {
        cvB.canvas.removeEventListener('pointermove', onMoveB);
        cvB.canvas.removeEventListener('pointerdown', onDownB);
        cvB.canvas.removeEventListener('pointerleave', onLeaveB);
      });

      function setLevel(i) {
        if (state.hoverLevel === i) return;
        state.hoverLevel = i;
        strip.querySelectorAll('.scl-strip__cell').forEach(function (b, j) { b.dataset.on = (i == null ? j === 4 : j === i) ? 'true' : 'false'; });
        cvB.redraw();
      }
      function buildStrip() {
        strip.replaceChildren();
        var ladder = sched ? sched.ladder : sim.PAPER.ladder;
        var Ks = levelsScaled();
        ladder.forEach(function (b, i) {
          var cell = el('button', { type: 'button', class: 'scl-strip__cell', role: 'listitem',
            'aria-label': 'β ' + b + ': ' + sweepTxt(Ks[i]) + ' sweeps', dataset: { on: (state.hoverLevel == null ? i === 4 : i === state.hoverLevel) ? 'true' : 'false' } },
            el('span', { class: 'scl-strip__b', text: 'β ' + fmt(b, { digits: 2 }) }),
            el('span', { class: 'scl-strip__k', text: sweepTxt(Ks[i]) }));
          cell.addEventListener('mouseenter', function () { setLevel(i); });
          cell.addEventListener('mouseleave', function () { setLevel(null); });
          cell.addEventListener('focus', function () { setLevel(i); });
          cell.addEventListener('blur', function () { setLevel(null); });
          cell.addEventListener('click', function () { setLevel(i); });
          strip.appendChild(cell);
        });
      }

      // size panel: c vs N on the paper schedule (1×)
      var cvS = ui.canvas(sizeBox, {
        height: function () { return cvB.height || 300; },
        label: 'c, the retained share of the log-odds, against model size N on the paper schedule, log-log: 0.171 at N 1024 down to 0.090 at N 4096. A dashed line shows the law for a linear ramp with the same sweeps, 0.091 at N 2048.',
        draw: drawS
      });
      function drawS(ctx, w, h, c) {
        var fs = 11;
        var box = ui.inset(w, h, { top: 22, right: 16, bottom: 40, left: 44 });
        var sx = ui.scale.log([780, 5400], [box.x0, box.x1]);
        var sy = ui.scale.log([0.04, 0.32], [box.y1, box.y0]);
        plot.axis(ctx, sx, { side: 'bottom', at: box.y1, ticks: [1024, 2048, 4096], grid: [box.y0, box.y1], c: c, size: fs,
          format: function (v) { return fmt(v); } });
        plot.axis(ctx, sy, { side: 'left', at: box.x0, ticks: [0.05, 0.1, 0.2, 0.3], grid: [box.x0, box.x1], c: c, size: fs,
          format: function (v) { return fmt(v); } });
        plot.text(ctx, 'N →', box.x1, box.y1 + 30, { align: 'right', size: fs, weight: 500, color: c.ink2, c: c });
        plot.text(ctx, 'c', box.x0 - 36, box.y0 - 6, { align: 'left', size: fs + 1, weight: 600, color: c.ink2, c: c, role: 'text' });
        if (!data) return;
        // N = 1,024 to 4,096 only: the range the paper reports the size trend over (Sec. 4)
        var rows = data.schedule.filter(function (r) { return r.scale === 1 && r.N >= 1024 && r.N <= 4096; });
        // the law for a linear ramp with the same 834 sweeps (beta 0.7 -> 1.6), with the bias scaled as in
        // the scan; it passes through 0.091 at N = 2,048, the value the comparison line below quotes
        var gLin = (B1 - B0) / (sched ? sched.K_total : 834);
        plot.fn(ctx, function (N) { N = Math.round(N); return lawCX(N, N * N * gLin); }, sx, sy, { n: 40, color: c.theory, width: 1.5, dash: [5, 4] });
        // label bottom left, under the dashed line and clear of the point labels (which sit above the dots)
        plot.text(ctx, 'linear ramp, same 834 sweeps', box.x0 + 6, box.y1 - 24, { align: 'left', size: fs, color: c.theory, c: c, halo: true });
        plot.text(ctx, '(Eq. 3, computed here)', box.x0 + 6, box.y1 - 10, { align: 'left', size: fs, color: c.theory, c: c, halo: true });
        plot.line(ctx, rows.map(function (r) { return r.N; }), rows.map(function (r) { return r.c; }), sx, sy, { color: c.data, width: 1.25, alpha: 0.5 });
        rows.forEach(function (r) {
          var on = r.N === 2048;
          plot.dots(ctx, [{ x: r.N, y: r.c }], sx, sy, { r: on ? 5.5 : 4, color: on ? c.ink3 : c.data, stroke: on ? c.ink : c.surface });
          var s = fmt(r.c, { digits: 3 });
          // the last point is labelled straight above its dot, clear of the dashed guide below it
          var last = r.N === 4096;
          plot.text(ctx, s, sx(r.N) + (last ? 0 : 8), sy(r.c) + (last ? -12 : -9), { align: last ? 'center' : 'left', size: fs, weight: on ? 600 : 400, color: on ? c.ink : c.ink2, c: c, halo: true });
        });
      }

      // controls B
      // Fractions of a sweep are real in the continuous-time chain; show them as decimals (88.75), since
      // vulgar-fraction glyphs in the mono face read as a percent sign.
      function sweepTxt(v) {
        var f = v - Math.floor(v);
        if (f < 1e-9) return fmt(v);
        return fmt(v, { digits: Math.abs(f * 2 - Math.round(f * 2)) < 1e-9 ? 1 : 2 });
      }
      function fmtScale(s) { return String(s); }
      var scaleT = ui.toggle({
        label: 'Sweeps per temperature',
        options: SCALES.map(function (s) { return { value: s, label: fmtScale(s) + '×' }; }),
        value: 1,
        onChange: function (v) { state.scale = v; buildStrip(); refreshB(); }
      });
      var fracNote = el('p', { class: 'scl-sub', hidden: true, text: 'The exact CW chain runs in continuous time, so a fraction of a sweep is a real amount of time.' });
      stairCol.appendChild(el('div', { class: 'scl-ctl' }, scaleT.el));
      stairCol.appendChild(fracNote);
      var roBs = ui.readout({ label: 'Sweeps' });
      var roBq = ui.readout({ label: 'Sampler log-odds' });
      var roBe = ui.readout({ label: 'Model log-odds' });
      var roBc = ui.readout({ label: 'Retained share c', tone: 'highlight-ink' });
      roBc.el.classList.add('scl-ro--c'); roBs.el.classList.add('scl-ro--aux'); roBq.el.classList.add("scl-ro--lob"); roBe.el.classList.add("scl-ro--lob");
      figB.controls.appendChild(el('div', { class: 've-readouts scl-readouts' }, roBc.el, roBs.el, roBq.el, roBe.el));
      var wordsB = el('p', { class: 'scl-words', 'aria-live': 'polite' });
      figB.controls.appendChild(wordsB);
      var cmpB = el('p', { class: 'scl-words scl-cmp', 'aria-live': 'polite' });
      figB.controls.appendChild(cmpB);
      function legendB() {
        var items = [];
        if (state.scale !== 1) items.push({ role: 'ink-3', mark: 'dash', label: '1× schedule, 834 sweeps' });
        var at = figB.legend.parentNode === stairCol;
        figB.setLegend(items);
        figB.legend.hidden = !items.length;
        if (!at) stairCol.insertBefore(figB.legend, strip.nextSibling);
      }
      legendB();
      figB.setCaption('In the size scan, at a fixed 834-sweep budget, c falls as about <i>N</i><sup>−0.47</sup> between <i>N</i>&nbsp;=&nbsp;1,024 and 4,096 (field rescaled with <i>N</i>; the paper’s fixed-field prediction falls as <i>N</i><sup>−0.46</sup>, and the linear-ramp law gives −1/2), because γ stays fixed and so <i>N</i>²γ grows as <i>N</i>². Every number comes from the exact CW chain at <i>N</i>&nbsp;=&nbsp;2,048 unless marked; the size scan is the repository’s (results/experiments/cw_scaling.json), not the paper’s.' +
        '<details class="scl-how"><summary>How these numbers were computed</summary>' +
        'The paper first runs a brief cooling (2 sweeps per temperature) and keeps the runs that ended in the negative mode (the filtered start). It reheats them at β\u00a0=\u00a00.7 for 30 sweeps; then the final cooling follows this schedule. ' +
        'The scale toggle multiplies only the sweeps of this final cooling (repository scan; the paper reports ×¼ and ×4). The field is <i>G</i>\u00a0=\u00a06.125×10<sup>−4</sup>, and the chain is the continuous-time Gibbs dynamics, reduced exactly to 2,049 counts. ' +
        'One-pattern Hopfield runs (permutation sweeps, interaction strength 1/<i>N</i>; 150,000 runs per field sign) measure c\u00a0=\u00a00.1274 at <i>N</i>\u00a0=\u00a02,048, 1.063 times c<sub>CW</sub>; the ratio includes both update order and interaction strength (Sec. 5, App. B). ' +
        'In the size scan the field is rescaled so the model’s odds stay nearly fixed. Between <i>N</i>\u00a0=\u00a01,024 and 4,096 the exact chain’s exponent is about −0.47 with the field rescaled, −0.46 with the paper’s fixed field (Table 1), and the sampled three-pattern Hopfield model’s about −0.36 (paper Sec. 5), against −1/2 for the linear-ramp law.</details>');
      function takeawayB() {
        var pick = function (k) { return data && data.schedule.filter(function (r) { return r.N === 2048 && r.scale === k; })[0]; };
        var rq = pick(0.25), r4 = pick(4);
        var g = (rq && r4) ? fmt(r4.c / rq.c, { digits: 2 }) : '1.48';
        figB.setTakeaway('The paper’s schedule keeps c\u00a0=\u00a00.120; sixteen times the final cooling (×¼ → ×4) raises c only ' + g + '× (Sec. 4, App. A).');
      }
      takeawayB();

      function refreshB() {
        var row = null;
        if (data) data.schedule.forEach(function (r) { if (r.N === 2048 && r.scale === state.scale) row = r; });
        var base = sched ? sched.K_total : 834;
        var c = row ? row.c : (state.scale === 1 ? (ccw ? ccw.c_cw : sim.PAPER.c_CW) : null);
        var q = row ? row.q_model : null, qe = row ? row.q_eq : null;
        var lq = q == null ? null : sim.logit(q), le = qe == null ? null : sim.logit(qe);
        var repoOnly = [0.25, 1, 4].indexOf(state.scale) < 0;
        roBc.set(c == null ? '–' : fmt(c, { digits: 3 }), repoOnly ? 'repository scan, not in the paper' : lq == null ? '' : 'sampler\u00a0' + fmt(lq, { digits: 3 }) + ' ÷\u00a0model\u00a0' + fmt(le, { digits: 2 }));
        roBq.set(lq == null ? '–' : fmt(lq, { digits: 3 }), q == null ? '' : 'favoured mode in ' + fmt(100 * q, { digits: 1 }) + '% of chains');
        roBe.set(le == null ? '–' : fmt(le, { digits: 2 }), qe == null ? '' : 'favoured mode has probability ' + fmt(qe, { digits: 3 }));
        roBs.set(sweepTxt(base * state.scale), state.scale === 1 ? '6τ per temperature, rounded up, over 10 temperatures' : (state.scale === 0.25 || state.scale === 4) ? fmtScale(state.scale) + '× of 834 · paper Table 3' : fmtScale(state.scale) + '× of 834 · repository scan, not in the paper');
        fracNote.hidden = state.scale >= 1;
        sizeCol.dataset.dim = state.scale !== 1 ? 'true' : 'false';
        sizeDim.hidden = state.scale === 1;
        legendB();
        if (data && row) {
          var r1 = data.schedule.filter(function (r) { return r.N === 2048 && r.scale === 1; })[0];
          wordsB.innerHTML = state.scale === 1
              ? 'At 1× (834 sweeps), ' + fmt(100 * q, { digits: 1 }) + '% of the chains land in the favoured mode; the model says ' + fmt(100 * qe, { digits: 1 }) + '%. In log-odds, ' +
                fmt(lq, { digits: 2 }) + ' against ' + fmt(le, { digits: 2 }) + ': c\u00a0=\u00a0' + fmt(c, { digits: 3 }) + '.'
              : 'At ' + fmtScale(state.scale) + '× the sweeps at every temperature, c moves from 0.120 to ' + fmt(c, { digits: 3 }) + ', ×' + fmt(c / r1.c, { digits: 2 }) +
                '; the linear-ramp law predicts ×' + fmt(Math.pow(state.scale, 0.25), { digits: 2 }) + ', the fourth root of ' + fmtScale(state.scale) + '. Repository scan' +
                (state.scale === 0.25 || state.scale === 4 ? '; the paper reports this value (Table 3).' : '.');
          var swB = base * state.scale, cLin = lawCX(2048, 2048 * 2048 * (B1 - B0) / swB);
          cmpB.innerHTML = 'A linear ramp with the same ' + sweepTxt(swB) + ' sweeps keeps c\u00a0=\u00a0' + fmt(cLin, { digits: 3 }) + ' (Eq. 3, computed on this page); the schedule keeps ' + (c > cLin ? 'more' : 'less') + '.';
          syncCard('b', [wordsB, cmpB]);
        }
        figB.setNote('Exact CW chain: data/c_cw.json (reference c\u00a0=\u00a0' + fmt(ccw ? ccw.c_cw : sim.PAPER.c_CW, { digits: 4 }) + '; dt\u00a0=\u00a00.01 and 0.02 agree) · schedule: data/schedule.json (src/hopfield/protocol.py) · budget and size scans: results/experiments/cw_scaling.json', 'paper');
        cvB.redraw(); cvS.redraw();
      }

      // A deep link to one of this chapter's drawers (#scaling-math) scrolls before the plates have
      // built and grown, which pushes the drawer far down. Once both plates have their final height,
      // land on the drawer again (once, and only within a few seconds of the mount).
      var mountedAt = performance.now();
      function relandDeepLink() {
        if (performance.now() - mountedAt > 6000) return;
        var ch = root.closest && root.closest('[data-chapter]');
        var id = decodeURIComponent((location.hash || '').slice(1));
        var d = ch && id && document.getElementById(id);
        if (!d || d.tagName !== 'DETAILS' || !ch.contains(d) || !d.open) return;
        requestAnimationFrame(function () { d.scrollIntoView({ block: 'start', behavior: 'auto' }); });
      }

      /* ------------------------------------------------------------------ boot */
      /* the first frame needs plate A only; plate B's first fill runs in its own task (each under 50 ms) */
      refreshA();
      var cancelled = false;
      setTimeout(function () { if (!cancelled && !data) { buildStrip(); refreshB(); } }, 0);
      Promise.all([sim.loadJSON('cw_scaling.json'), sim.loadJSON('schedule.json'), sim.loadJSON('c_cw.json')]).then(function (res) {
        if (cancelled) return;
        data = res[0]; sched = res[1]; ccw = res[2];
        takeawayB();
        refreshA();
        // plate B redraws in its own task, so no single task carries both plates
        setTimeout(function () { if (!cancelled) { buildStrip(); refreshB(); relandDeepLink(); } }, 0);
        if (state.guessed) answer(state.guessed);
      }).catch(function (e) {
        console.error(e);
        figA.setNote('Could not load data/cw_scaling.json: ' + e.message, 'paper');
      });

      return {
        destroy: function () {
          cancelled = true;
          cancelAnimationFrame(anim.raf);
          disposers.forEach(function (f) { f(); });
          [cvA, cvL, cvB, cvS].forEach(function (c) { c.destroy(); });
          wrap.remove();
        }
      };
    }
  };
})();
