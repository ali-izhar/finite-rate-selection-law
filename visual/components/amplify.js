/* components/amplify.js — "Competing patterns: the winners have larger overlaps"
 *
 * Teaches the Hopfield amplification A_p = E[max_p |Z|] / E|Z| (paper Sec. 4, Eq. 6).
 *   Panel a  near beta = 1 the p overlaps are APPROXIMATED by p independent Gaussian draws, and the
 *            largest in size is ASSUMED to win (paper Sec. 4); a run counts (is kept) only when
 *            pattern 1 wins. One draw is shown at a time.
 *   Panel b  the size of overlap 1 over all runs and over the kept runs, with the Gaussian-theory
 *            densities; the ratio of the two means is A_p.
 * Controls: p (1-4, the range the paper tests) and one pause button.
 *
 * Sources: A_p exact = VE.sim.ampP(p) (Simpson integral, tested against scipy in lib/sim.test.mjs);
 * the draws are live (seeded RNG).
 */
(function () {
  'use strict';
  var VE = (window.VE = window.VE || {});
  VE.components = VE.components || {};

  var XMAX = 4, NBIN = 40;            // histogram range |Z1| and bins
  var DRAW_EVERY = 0.9, ANIM = 0.35;  // seconds between shown draws; tween length

  function phi(x) { return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI); }

  function mount(root, opts) {
    opts = opts || {};
    var ui = VE.ui, S = VE.sim, el = ui.el, fmt = ui.fmt;
    var AP = [NaN, 1, 2, 3, 4].map(function (p) { return p ? S.ampP(p) : NaN; });
    var state = { p: 3 }, mc = null, anim = 1, running = !ui.reducedMotion, clock = 0, destroyed = false;

    var fig = ui.figure({ plate: opts.plate != null ? opts.plate : 5, title: 'Competing patterns', className: 've-amplify' });

    /* ------------------------------------------------------------ p */
    var pTg = ui.toggle({
      label: 'Stored patterns, p',
      options: [1, 2, 3, 4].map(function (p) { return { value: p, label: String(p) }; }),
      value: state.p,
      onChange: function (v) { setP(+v); }
    });
    fig.stage.appendChild(el('div', { class: 've-amplify__pctl' }, pTg.el));

    function panel(letter, title) {
      var body = el('div', { class: 've-amplify__body' });
      var s = el('section', { class: 've-amplify__step' },
        el('h4', { class: 've-amplify__h' }, el('span', { class: 've-amplify__n', text: letter }), el('span', { html: title })), body);
      return { el: s, body: body };
    }
    var steps = el('div', { class: 've-amplify__steps' });
    fig.stage.appendChild(steps);

    /* ------------------------------------------------- panel a: one draw */
    var pa = panel('a', 'Assume the largest early overlap wins');
    steps.appendChild(pa.el);
    var drawCv = ui.canvas(pa.body, {
      label: 'The latest draw: the size of one Gaussian overlap per pattern; the largest wins',
      height: function () { return 10 + state.p * 30 + 44; },
      draw: drawDraw
    });
    var playBtn = ui.button({ label: 'Pause', icon: 'pause', onClick: function () { running = !running; syncBtn(); kick(); } });
    pa.body.appendChild(el('div', { class: 've-amplify__btns' }, playBtn));
    function syncBtn() {
      playBtn.replaceChildren(ui.icon(running ? 'pause' : 'play'), el('span', { text: running ? 'Pause' : 'Play' }));
      playBtn.setAttribute('aria-label', (running ? 'Pause' : 'Play') + ' the draws');
    }

    /* ------------------------------------------ panel b: kept against all */
    var pb = panel('b', 'The winners have larger early overlaps');
    steps.appendChild(pb.el);
    var histCv = ui.canvas(pb.body, {
      label: 'Histogram of the size of overlap 1 for all draws and for kept draws, with Gaussian-theory curves and both means',
      height: 238, draw: drawHist
    });
    var legend = ui.legend([
      { role: 'noise', mark: 'area', label: 'all runs' },
      { role: 'mode-a', mark: 'square', label: 'kept runs (pattern 1 won)' },
      { role: 'theory', mark: 'dash', label: 'Gaussian theory' }
    ]);
    var apRead = ui.readout({ label: 'A_p', tone: 'highlight-ink' });
    apRead.el.classList.add('ve-amplify__ap');
    pb.body.appendChild(el('div', { class: 've-amplify__foot' }, legend, apRead.el));

    /* ------------------------------------------------------- the draws */
    function ease() { return anim < 1 ? 1 - Math.pow(1 - anim, 3) : 1; }
    function resetMC() {
      mc = { r: S.rng(900 + state.p), z: new Float64Array(state.p), zFrom: new Float64Array(state.p), win: 0, winFrom: 0,
        n: 0, nKept: 0, sumAll: 0, sumKept: 0, hAll: new Float64Array(NBIN), hKept: new Float64Array(NBIN) };
      draw(state.p === 1 ? 2000 : 1000 * Math.max(3, state.p), false);   // about 1,000 kept runs at every p
      // a clear example at rest: keep drawing (every draw counts) until pattern 1 wins with a visible lead
      var guard = 0;
      while (state.p > 1 && guard++ < 500 && !(mc.win === 0 && Math.abs(mc.z[0]) > 1.4)) draw(1, false);
      mc.zFrom.set(mc.z); mc.winFrom = mc.win; anim = 1;
    }
    function draw(k, animate) {
      var p = state.p, r = mc.r, z = mc.z;
      if (animate) {   // the bars move from what is on screen to the new draw
        var e = ease();
        for (var j0 = 0; j0 < p; j0++) mc.zFrom[j0] += (z[j0] - mc.zFrom[j0]) * e;
        if (e >= 0.5) mc.winFrom = mc.win;
      }
      for (var t = 0; t < k; t++) {
        var best = -1, arg = 0;
        for (var j = 0; j < p; j++) { z[j] = r.normal(); var a = Math.abs(z[j]); if (a > best) { best = a; arg = j; } }
        var a1 = Math.abs(z[0]), bi = Math.min(NBIN - 1, Math.floor(a1 / XMAX * NBIN));
        mc.n++; mc.sumAll += a1; mc.hAll[bi]++;
        if (arg === 0) { mc.nKept++; mc.sumKept += a1; mc.hKept[bi]++; }
        mc.win = arg;
      }
      if (animate) { if (ui.reducedMotion) { mc.zFrom.set(z); mc.winFrom = mc.win; anim = 1; } else anim = 0; }
      else if (anim >= 1) { mc.zFrom.set(z); mc.winFrom = mc.win; }
    }

    function drawDraw(ctx, w, h, c) {
      if (!mc) return;
      var p = state.p, e = ease(), win = e >= 0.5 ? mc.win : mc.winFrom;
      var lx = 74, top = 10, rowH = 30;
      var sx = ui.scale.linear([0, 3.6], [lx + 4, w - 70]);
      var vs = [];
      for (var j = 0; j < p; j++) vs[j] = Math.abs(mc.zFrom[j] + (mc.z[j] - mc.zFrom[j]) * e);
      var yb0 = top - 4, yb1 = top + p * rowH;
      for (var mu = 0; mu < p; mu++) {
        var y = top + mu * rowH, cy = y + rowH / 2 - 2, on = mu === win && p > 1;
        if (on) { ctx.fillStyle = ui.alpha(c.ink3, 0.16); ctx.fillRect(0, y - 2, w, rowH - 2); }
        ui.plot.text(ctx, 'pattern ' + (mu + 1), 10, cy + 4, { c: c, size: 11, color: mu === 0 ? c.modeA : c.ink2, weight: on ? 600 : 400 });
        ctx.fillStyle = mu === 0 ? c.modeA : c.ink2;
        ctx.fillRect(sx(0), cy - 7, Math.max(1, sx(vs[mu]) - sx(0)), 14);
        if (on) ui.plot.text(ctx, 'wins', sx(vs[mu]) + 8, cy + 4, { c: c, size: 11, weight: 600, color: c.ink });
      }
      ctx.fillStyle = c.ruleStrong; ctx.fillRect(Math.round(sx(0)), yb0, 1, yb1 - yb0);
      ui.plot.axis(ctx, sx, { side: 'bottom', at: yb1 + 2, ticks: [0, 1, 2, 3], c: c, label: 'size of early overlap |Z| (units of its spread)' });
    }
    function drawHist(ctx, w, h, c) {
      if (!mc) return;
      var p = state.p, box = ui.inset(w, h, { top: 48, right: 8, bottom: 36, left: 8 });
      var sx = ui.scale.linear([0, XMAX], [box.x0, box.x1]), bw = XMAX / NBIN;
      var fAll = function (x) { return 2 * phi(x); };
      var fKept = function (x) { var t = 1 - 2 * S.Phi(-x); return p * 2 * phi(x) * Math.pow(t, p - 1); };
      var ymax = 0;
      for (var i = 0; i <= 80; i++) { var x = XMAX * i / 80; ymax = Math.max(ymax, fAll(x), fKept(x)); }
      ymax *= 1.08;
      var sy = ui.scale.linear([0, ymax], [box.y1, box.y0]);
      var nA = Math.max(1, mc.n), nK = Math.max(1, mc.nKept), b;
      ctx.beginPath(); ctx.moveTo(sx(0), sy(0));
      for (b = 0; b < NBIN; b++) { var d = Math.min(ymax, mc.hAll[b] / (nA * bw)); ctx.lineTo(sx(b * bw), sy(d)); ctx.lineTo(sx((b + 1) * bw), sy(d)); }
      ctx.lineTo(sx(XMAX), sy(0)); ctx.closePath();
      ctx.fillStyle = c.noiseWash; ctx.fill(); ctx.strokeStyle = c.noise; ctx.lineWidth = 1; ctx.stroke();
      if (mc.nKept > 0 && p > 1) {
        ctx.fillStyle = ui.alpha(c.modeA, ui.theme() === 'dark' ? 0.72 : 0.55);
        for (b = 0; b < NBIN; b++) {
          var dk = Math.min(ymax, mc.hKept[b] / (nK * bw)), x0 = sx(b * bw) + 1, x1 = sx((b + 1) * bw) - 1;
          ctx.fillRect(x0, sy(dk), Math.max(1, x1 - x0), sy(0) - sy(dk));
        }
      }
      ui.plot.fn(ctx, fAll, sx, sy, { color: c.theory, width: 1.4, dash: [4, 3], n: 120 });
      if (p > 1) ui.plot.fn(ctx, fKept, sx, sy, { color: c.theory, width: 1.6, dash: [4, 3], n: 120 });
      ui.plot.axis(ctx, sx, { side: 'bottom', at: box.y1, ticks: [0, 1, 2, 3, 4], c: c, label: 'size of early overlap 1, |Z₁|' });
      // the two means: the all-runs label reads leftward on the top row, the kept label rightward below it
      var mAll = mc.sumAll / nA, mKept = mc.nKept ? mc.sumKept / nK : NaN;
      ctx.font = ui.font('mono', 11, 600);
      function mean(v, col, label, rowY, left) {
        if (!isFinite(v)) return;
        var X = Math.round(sx(v)), tw = ctx.measureText(label).width;
        ctx.fillStyle = col; ctx.fillRect(X - 1, rowY + 3, 2, box.y1 - rowY - 3);
        ui.plot.text(ctx, label, left ? Math.max(box.x0, X - 5 - tw) : Math.min(box.x1 - tw, X + 5), rowY, { c: c, size: 11, weight: 600, color: col });
      }
      if (p === 1) mean(mAll, c.ink2, 'mean ' + fmt(mAll, 2), 14, false);
      else { mean(mAll, c.ink2, 'mean of all ' + fmt(mAll, 2), 14, true); mean(mKept, c.modeA, 'mean of kept ' + fmt(mKept, 2), 30, false); }
    }

    function render() {
      apRead.el.querySelector('.ve-readout__label').innerHTML = '<i>A</i><sub>' + state.p + '</sub>, Gaussian theory';
      apRead.set(fmt(AP[state.p], { digits: 3 }), state.p === 2 ? '√2' : '');
      drawCv.redraw(); histCv.redraw();
    }
    function setP(p) {
      state.p = p;
      resetMC();
      drawCv.resize(); histCv.resize();
      render(); kick();
    }

    var lp = ui.loop(function (dt) {
      if (destroyed) return;
      if (anim < 1) { anim = Math.min(1, anim + dt / ANIM); render(); }
      if (running) {
        clock += dt;
        if (clock >= DRAW_EVERY) {
          clock = 0;
          draw(Math.min(60, 4 + Math.floor(mc.n / 400)), false);   // the histogram fills faster than the shown draw
          draw(1, true);
          render();
        }
      }
      if (!running && anim >= 1) lp.stop();
    }, { el: root });
    function kick() { if (running || anim < 1) lp.start(); }
    var offRM = ui.onReducedMotion(function (rm) { if (rm) { running = false; syncBtn(); } });

    syncBtn();
    resetMC();
    root.appendChild(fig.el);
    render(); kick();

    return {
      destroy: function () {
        destroyed = true; lp.destroy(); offRM();
        drawCv.destroy(); histCv.destroy();
        fig.el.remove();
      }
    };
  }

  VE.components.amplify = { mount: mount };
})();
