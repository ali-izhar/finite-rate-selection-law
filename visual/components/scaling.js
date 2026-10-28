/* components/scaling.js — "Bigger and slower: c falls as (N²γ)^(−1/4)"
 *
 * One plate: the law. Log-log plot of the kept fraction c against N²γ: the selection-law curve
 * (computed on the page with VE.sim.knC, the same formula as src/cw_ramp.py) and the nine
 * exact-chain runs on linear ramps (data/cw_linear.json <- results/experiments/cw_linear_sweep.json;
 * paper Table 2, Fig. 2a). One control: a guess (16× the sweeps multiplies the log-odds by ×16, ×4
 * or ×2?), which moves the reader's point from the test's ramp (1×) to 16× its sweeps at N = 2,048.
 * Hovering a ring shows its run.
 *
 * Linear-ramp constants (src/cw_ramp.py linear_sweep(): b0 = 0.7, b1 = 1.6, start in equilibrium
 * at b0; G = G0·2048/N with G0 = 6.125e-4, so N G = 1.2544):  sweeps per descent = 0.9 / γ.
 * The budget k sets γ = 0.004 / k, so k = 1 and 16 are exactly computed runs.
 */
(function () {
  'use strict';
  var VE = (window.VE = window.VE || {});
  VE.components = VE.components || {};

  var B0 = 0.7, B1 = 1.6;            // src/cw_ramp.py linear_sweep(b0, b1)
  var GAMMA0 = 0.004;                // fastest computed sweep; budget k -> gamma = GAMMA0 / k
  var G0 = 6.125e-4;                 // the field at N = 2048 (N G = 1.2544; paper Sec. 4)

  function gammaOf(k) { return GAMMA0 / k; }
  function Gof(N) { return G0 * 2048 / N; }

  VE.components.scaling = {
    mount: function (root, opts) {
      opts = opts || {};
      var ui = VE.ui, sim = VE.sim, el = ui.el, fmt = ui.fmt, plot = ui.plot;
      var reduced = ui.reducedMotion;

      var state = { N: 2048, k: 1, hover: null, guessed: null };
      // the answer (the 16× arrow) stays hidden until the reader guesses
      function revealed() { return state.guessed != null; }
      var data = null;           // cw_linear.json
      var anim = { from: null, t: 1, raf: 0 };
      var disposers = [];

      var wrap = el('div', { class: 've-scaling ve-scaling--v2' });
      root.appendChild(wrap);

      /* ------------------------------------------------------------ law helpers */
      // Model log-odds do not depend on the cooling rate, so compute them once per N.
      var loeCache = {};
      function modelLO(N) { return loeCache[N] != null ? loeCache[N] : (loeCache[N] = sim.logOddsEq(N, B1, Gof(N))); }
      function lawCX(N, X) { return sim.logitPhi(sim.knX(Gof(N), N, X / (N * N))) / modelLO(N); }   // = sim.knC
      function lawC(N, k) { return lawCX(N, N * N * gammaOf(k)); }
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
        var ch = root.closest && root.closest('.v2-ch');
        if (ch) ch.querySelectorAll('[data-after-guess]').forEach(function (n) { n.hidden = false; });
        guessBtns.querySelectorAll('button').forEach(function (b) {
          b.dataset.state = +b.dataset.f === 2 ? 'right' : (+b.dataset.f === f ? 'wrong' : '');
          b.setAttribute('aria-pressed', +b.dataset.f === f ? 'true' : 'false');
        });
        var r1 = exactRow(state.N, 1), r16 = exactRow(state.N, 16);
        var ex = (r1 && r16) ? sim.logit(r16.q_exact) / sim.logit(r1.q_exact) : null;
        guessOut.innerHTML = (f === 2 ? '<b>Yes: only ×2.</b> ' : '<b>Only ×2.</b> ') +
          'Log-odds grow as the fourth root of the sweeps: 16<sup>1/4</sup>\u00a0=\u00a02' +
          (ex ? '; the exact chain gives ×' + fmt(ex, { digits: 2 }) : '') + '.';
        setK(16, true);
      }

      var plotCol = el('div', { class: 'scl-col scl-col--plot' });
      figA.stage.appendChild(plotCol);
      var plotBox = el('div', { class: 'scl-plotbox' });
      plotCol.appendChild(plotBox);
      var tipA = el('div', { class: 'scl-tip', hidden: true, role: 'status' });
      plotBox.appendChild(tipA);

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
        plot.text(ctx, narrow ? 'band: checked range' : 'band: checked against the exact chain', (xa + xb) / 2, box.y0 + 17,
          { align: 'center', size: fs, color: c.ink3, c: c, halo: true, haloColor: c.surface });
        plot.text(ctx, narrow ? 'rings: exact runs' : 'rings: exact-chain runs, size ∝ N', xa + 6, box.y1 - 10,
          { align: 'left', size: fs, color: c.ink3, c: c, halo: true, haloColor: c.surface });
        plot.text(ctx, 'c', box.x0 - 8, box.y0 - 16, { align: 'right', baseline: 'middle', size: fs + 3, weight: 600, color: c.ink2, c: c, role: 'text' });
        plot.text(ctx, 'N²γ  →', box.x1, box.y1 + 34, { align: 'right', size: fs, weight: 500, color: c.ink2, c: c });
        plot.text(ctx, narrow ? '← slower or smaller' : '← more sweeps, or a smaller model', box.x0, box.y1 + 34,
          { align: 'left', size: fs, color: c.ink3, c: c });

        // law curve for the current N (a 120-point polyline; model log-odds cached per N)
        plot.fn(ctx, function (X) { return lawCX(N, X); }, sx, sy, { n: 120, color: c.theory, width: 2.5 });
        // law label: top right, above the curve, away from the arrow (which hugs the curve)

        // current marker (tweened); the rings are drawn on top so a measured run stays visible
        var cur = currentPoint();
        var px0 = sx(cur.X), py0 = sy(cur.c);
        ctx.save();
        // your setting is a selection, so it is grey (the page keeps yellow for c itself)
        ctx.fillStyle = c.ink3; ctx.strokeStyle = c.ink; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(px0, py0, 6, 0, 6.2832); ctx.fill(); ctx.stroke();
        ctx.restore();
        // one label, on the reader's point only: its sweep count (below-left, clear of the falling line)
        if (!anim.from || anim.t >= 1) {
          var you = (state.k === 1 ? 'the test’s ramp: ' : fmt(state.k) + '× slower: ') + fmt(Math.round(sweepsOf(state.k))) + ' sweeps';
          plot.text(ctx, you, px0 - 10, py0 + 20, { align: 'right', size: fs, weight: 700, color: c.ink, c: c, halo: true });
        }

        if (data) {
          data.linear_sweep.forEach(function (r) {
            var rr = r.N === 1024 ? 3.5 : r.N === 2048 ? 6 : 8.5;
            ctx.save();
            ctx.strokeStyle = c.data; ctx.lineWidth = state.hover === r ? 2.5 : 1.25;
            ctx.beginPath(); ctx.arc(sx(r.N2gamma), sy(r.c_exact), rr, 0, 6.2832); ctx.stroke();
            ctx.restore();
          });
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
        tipA.innerHTML = '<div class="scl-tip__h">Exact Curie–Weiss chain · N²γ\u00a0=\u00a0' + fmt(Math.round(r0.N2gamma)) + '</div>' +
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
      function onDownA(e) { showTipA(nearestRing(e)); }
      function onLeaveA() { showTipA(null); }
      cvA.canvas.addEventListener('pointermove', onMoveA);
      cvA.canvas.addEventListener('pointerdown', onDownA);
      cvA.canvas.addEventListener('pointerleave', onLeaveA);
      disposers.push(function () {
        cvA.canvas.removeEventListener('pointermove', onMoveA);
        cvA.canvas.removeEventListener('pointerdown', onDownA);
        cvA.canvas.removeEventListener('pointerleave', onLeaveA);
      });

      function refreshA() { cvA.redraw(); }

      function setK(k, animate) {
        var from = currentPoint();
        state.k = k;
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

      /* ------------------------------------------------------------------ boot */
      refreshA();
      var cancelled = false;
      sim.loadJSON('cw_linear.json').then(function (d) {
        if (cancelled) return;
        data = { linear_sweep: d.rows };
        refreshA();
        if (state.guessed) answer(state.guessed);
      }).catch(function (e) {
        console.error(e);
      });

      return {
        destroy: function () {
          cancelled = true;
          cancelAnimationFrame(anim.raf);
          disposers.forEach(function (f) { f(); });
          cvA.destroy();
          wrap.remove();
        }
      };
    }
  };
})();
