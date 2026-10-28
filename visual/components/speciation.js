/* components/speciation.js — "Where a run chooses its sign" (chapter "Where the choice is made")
 *
 * Story beat 1 of visual/README.md. Curie–Weiss, N = 1024 spins (the N of the paper's Fig. 1a paths).
 *   Left  : the model's landscape  −log p(m)  (exact, finite N, from VE.sim.cwLogWeights),
 *           with a pile of live chains (exact lumped Gibbs chains, VE.sim.CWChains) sitting in it.
 *   Right : the fork. m against β. h = 0: the paper's 14 runs (data/fork_trajectories.json,
 *           Fig. 1a). With a field: 14 runs recomputed live with the same equation and step.
 *           Stable states ±m*(β) from VE.sim.stationaryPoints (theory).
 * Controls: Run the cooling, and the field toward mode A. Drag across the fork to scrub β.
 * Rate of the "Run the cooling" button: β rises 0.65/359 every 6 sweeps (Fig. 1's grid and step).
 */
(function () {
  'use strict';
  var VE = (window.VE = window.VE || {});
  VE.components = VE.components || {};

  var N = 1024;                 // paper Fig. 1a paths
  var NCH = 120;                // live chains (page choice)
  var B0 = 0.70, B1 = 1.35;     // Fig. 1 beta axis
  var GAMMA = (B1 - B0) / 359 / 6;   // Fig. 1: 360-point grid, 6 sweeps per step  (≈ 3.0e-4 per sweep)
  var SWEEPS_PER_SEC = 180;     // animation speed of the live chains (page choice)
  var MAX_SWEEPS_FRAME = 4;     // keeps a frame under ~8 ms of simulation
  var BIAS = { off: 0, small: 0.002, large: 0.02 };   // G; the field is h = βG (page choices)
  var START_BETA = 0.8;         // the plate rests at β = 0.8: one valley

  function clamp(x, a, b) { return x < a ? a : x > b ? b : x; }
  function pct(q, fmt) { return q > 0.9999 ? '>99.99%' : q < 0.0001 ? '<0.01%' : fmt(100 * q, { digits: q > 0.995 || q < 0.005 ? 2 : 1 }) + '%'; }
  function smooth(a, b, x) { var t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }
  function hexToRgb(c) {
    if (!c) return [128, 128, 128];
    c = c.trim();
    if (c[0] === '#') {
      if (c.length === 4) c = '#' + c[1] + c[1] + c[2] + c[2] + c[3] + c[3];
      return [parseInt(c.slice(1, 3), 16), parseInt(c.slice(3, 5), 16), parseInt(c.slice(5, 7), 16)];
    }
    var m = c.match(/[\d.]+/g);
    return m ? [+m[0], +m[1], +m[2]] : [128, 128, 128];
  }
  function mix(a, b, t) {
    return 'rgb(' + Math.round(a[0] + (b[0] - a[0]) * t) + ',' + Math.round(a[1] + (b[1] - a[1]) * t) + ',' + Math.round(a[2] + (b[2] - a[2]) * t) + ')';
  }

  // exact draw of k (number of +1 spins) from the stationary law pi_k at (beta, h)
  function sampleK(sim, beta, h, n, r) {
    var p = sim.cwPi(N, beta, h), cdf = new Float64Array(N + 1), s = 0;
    for (var k = 0; k <= N; k++) { s += p[k]; cdf[k] = s; }
    var out = new Int32Array(n);
    for (var i = 0; i < n; i++) {
      var u = r.uniform() * s, lo = 0, hi = N;
      while (lo < hi) { var mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
      out[i] = lo;
    }
    return out;
  }
  // 14 runs of Fig. 1's equation with a bias: dm = (−m + tanh(βm + βG)) dt + sqrt(2dt/N) dW,
  // same beta grid, dt = 0.02, 6 sweeps per step. Generator so it can be chunked.
  function* forkGen(sim, betas, G, seed) {
    var r = sim.rng(seed), n = 14, m = new Float64Array(n), dt = 0.02, steps = Math.round(6 / dt);
    var sd = Math.sqrt(2 * dt / N), out = [];
    for (var c = 0; c < n; c++) out.push(new Float64Array(betas.length));
    for (var j = 0; j < betas.length; j++) {
      var b = betas[j], hb = b * G;
      for (var s = 0; s < steps; s++) for (var i = 0; i < n; i++) m[i] += (Math.tanh(b * m[i] + hb) - m[i]) * dt + sd * r.normal();
      for (var i2 = 0; i2 < n; i2++) out[i2][j] = m[i2];
      if (j % 20 === 19) yield j / betas.length;
    }
    return out;
  }

  // centred canvas text: pre + value + post, where value is 10^e written as a×10^E with a raised exponent
  function supText(ctx, c, pre, e, post, x, y, align) {
    var ui = VE.ui, E = Math.floor(e), mt = Math.round(Math.pow(10, e - E));
    if (mt >= 10) { E += 1; mt = 1; }
    var big = ui.font('mono', 11), small = ui.font('mono', 9);
    var head = e < 4 ? pre + ui.fmt(+Math.pow(10, e).toPrecision(2), { group: true }) : pre + (mt === 1 ? '' : mt + '\u00d7') + '10';
    var ex = e < 4 ? '' : String(E);
    ctx.save();
    ctx.font = big; var w1 = ctx.measureText(head).width, w3 = ctx.measureText(post).width;
    ctx.font = small; var w2 = ex ? ctx.measureText(ex).width + 1.5 : 0;
    var W = w1 + w2 + w3, x0 = align === 'left' ? x : align === 'right' ? x - W : x - W / 2;
    ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left';
    ctx.lineJoin = 'round'; ctx.lineWidth = 4; ctx.strokeStyle = c.surface; ctx.fillStyle = c.ink3;
    ctx.font = big;
    ctx.strokeText(head, x0, y); ctx.strokeText(post, x0 + w1 + w2, y);
    ctx.fillText(head, x0, y); ctx.fillText(post, x0 + w1 + w2, y);
    if (ex) { ctx.font = small; ctx.strokeText(ex, x0 + w1 + 0.5, y - 5); ctx.fillText(ex, x0 + w1 + 0.5, y - 5); }
    ctx.restore();
  }

  VE.components['speciation'] = {
    mount: function (host, opts) {
      opts = opts || {};
      var ui = VE.ui, sim = VE.sim, el = ui.el, plot = ui.plot, fmt = ui.fmt;

      /* ---------------------------------------------------------- state */
      var S = {
        beta: START_BETA, biasKey: 'off', G: 0, playing: false,
        fork: null, forkBias: {}, forkPending: null, run: null,
        lastDom: 0, idleUntil: 0, destroyed: false
      };
      var r = sim.rng(20261);
      var chains = new sim.CWChains(NCH, N, { seed: 11, init: sampleK(sim, START_BETA, 0, NCH, r) });

      /* ---------------------------------------------------------- frame */
      var fig = ui.figure({
        plate: opts.plate != null ? opts.plate : 1,
        title: 'Where a run chooses its sign',
        className: 've-speciation'
      });
      host.appendChild(fig.el);
      var grid = el('div', { class: 'sp-grid' });
      var leftCol = el('div', { class: 'sp-panel' }, el('div', { class: 'sp-panel__head' }, el('span', { class: 'sp-panel__k', text: 'a' }), 
        el('span', { class: 've-label', text: 'The landscape · now' })));
      var rightCol = el('div', { class: 'sp-panel' }, el('div', { class: 'sp-panel__head' }, el('span', { class: 'sp-panel__k', text: 'b' }), 
        el('span', { class: 've-label', text: 'The fork · whole cooling run' })));
      grid.appendChild(leftCol); grid.appendChild(rightCol);
      fig.stage.appendChild(grid);

      /* ---------------------------------------------------------- caches */
      var land = { key: null, V: null, vmin: 0 };
      function landscape() {
        var key = S.beta.toFixed(5) + ',' + S.G;
        if (land.key === key) return land;
        var ell = sim.cwLogWeights(N, S.beta, S.beta * S.G), ref = ell[N / 2];
        var V = new Float64Array(N + 1), vmin = Infinity;
        for (var k = 0; k <= N; k++) { V[k] = -(ell[k] - ref); if (V[k] < vmin) vmin = V[k]; }
        land.key = key; land.V = V; land.vmin = vmin;
        // barrier: the highest point between the two deepest minima on each side of m = 0
        var kL = 0, kR = N, i;
        for (i = 0; i <= N / 2; i++) if (V[i] < V[kL]) kL = i;
        for (i = N / 2; i <= N; i++) if (V[i] < V[kR]) kR = i;
        var kS = kL; for (i = kL; i <= kR; i++) if (V[i] > V[kS]) kS = i;
        land.kL = kL; land.kR = kR; land.kS = kS;
        // the escape hill out of the shallower valley (under a bias: the unfavoured mode)
        land.kW = V[kL] >= V[kR] ? kL : kR;
        land.barrier = V[kS] - V[land.kW];
        land.two = kL < kS && kS < kR && V[kS] > V[kL] && V[kS] > V[kR];
        return land;
      }
      // the y range follows the current landscape: its floor sits near the bottom of the plot at every
      // beta, so the hot single valley fills the box (the curve's scale changes as the wells deepen)
      function yLoFor(L) { return Math.min(-6, L.vmin - 0.08 * (30 - L.vmin)); }
      var branchCache = {};
      function branches(G) {
        if (branchCache[G]) return branchCache[G];
        var bs = [], up = [], dn = [], mid = [];
        for (var i = 0; i <= 130; i++) {
          var b = B0 + (B1 - B0) * i / 130, pts = sim.stationaryPoints(b, b * G);
          bs.push(b);
          var st = pts.filter(function (p) { return p.stable; }), un = pts.filter(function (p) { return !p.stable; });
          up.push(st.length ? st[st.length - 1].m : NaN);
          dn.push(st.length > 1 ? st[0].m : NaN);
          mid.push(un.length ? un[0].m : NaN);
        }
        return (branchCache[G] = { b: bs, up: up, dn: dn, mid: mid });
      }

      /* ---------------------------------------------------------- landscape canvas */
      function chainColor(m, c, rgb) {
        var t = smooth(0.09, 0.21, Math.abs(m));   // grey below |m| = 0.15 (UND), mode colour above
        return mix(rgb.noise, m > 0 ? rgb.a : rgb.b, t);
      }
      function rgbs(c) { return { noise: hexToRgb(c.noise), a: hexToRgb(c.modeA), b: hexToRgb(c.modeB) }; }

      var landCv = ui.canvas(leftCol, {
        height: function (w) { return w < 520 ? Math.round(w * 0.86) : Math.round(clamp(w * 0.66, 300, 400)); },
        label: 'The model’s landscape over m with the live chains as dots',
        draw: function (ctx, w, h, c) {
          var L = landscape(), V = L.V, rgb = rgbs(c);
          var narrow = w < 420;
          var box = ui.inset(w, h, { top: narrow ? 38 : 42, right: 18, bottom: 40, left: 18 });
          var yLo = yLoFor(L), yHi = 30;
          var sx = ui.scale.linear([-1, 1], [box.x0, box.x1]);
          var sy = ui.scale.linear([yLo, yHi], [box.y1, box.y0]);
          // mode washes
          ctx.globalAlpha = 0.6;
          ctx.fillStyle = c.modeBWash; ctx.fillRect(box.x0, box.y0, sx(0) - box.x0, box.h);
          ctx.fillStyle = c.modeAWash; ctx.fillRect(sx(0), box.y0, box.x1 - sx(0), box.h);
          ctx.globalAlpha = 1;
          plot.axis(ctx, sx, { side: 'bottom', at: box.y1, ticks: [-1, -0.5, 0, 0.5, 1], c: c, format: function (v) { return v === 0 ? '0' : fmt(v, { digits: 1, sign: v > 0 }); } });
          plot.text(ctx, '← mode B', box.x0, box.y1 + 30, { c: c, size: 11, color: c.modeB, weight: 600 });
          plot.text(ctx, 'mode A →', box.x1, box.y1 + 30, { c: c, size: 11, color: c.modeA, weight: 600, align: 'right' });
          plot.text(ctx, 'm (mean spin)', sx(0), box.y1 + 30, { c: c, size: 11, color: c.ink2, align: 'center' });

          // terrain
          ctx.save();
          ctx.beginPath(); ctx.rect(box.x0, box.y0 - 2, box.w, box.h + 2); ctx.clip();
          ctx.beginPath();
          var k, step = N > 600 ? 2 : 1;
          for (k = 0; k <= N; k += step) { var X = sx(2 * k / N - 1), Y = sy(V[k]); if (k === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); }
          ctx.lineTo(box.x1, box.y1); ctx.lineTo(box.x0, box.y1); ctx.closePath();
          ctx.fillStyle = c.surface2; ctx.globalAlpha = 0.85; ctx.fill(); ctx.globalAlpha = 1;
          ctx.beginPath();
          for (k = 0; k <= N; k += step) { var X2 = sx(2 * k / N - 1), Y2 = sy(V[k]); if (k === 0) ctx.moveTo(X2, Y2); else ctx.lineTo(X2, Y2); }
          ctx.strokeStyle = c.ink; ctx.lineWidth = 2; ctx.stroke();

          // barrier bracket (only once two valleys exist and the hill is visible)
          var barrierLabel = null;
          if (S.beta > 1.03 && L.two && L.barrier > 3) {
            var kW = L.kW;
            var bx = sx(2 * L.kS / N - 1), yTop = sy(V[L.kS]), yBot = sy(V[kW]);
            ctx.strokeStyle = c.ink3; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
            ctx.beginPath(); ctx.moveTo(sx(2 * kW / N - 1), yBot); ctx.lineTo(bx + (kW > L.kS ? 10 : -10), yBot); ctx.stroke();
            ctx.setLineDash([]);
            plot.arrow(ctx, bx, yBot, bx, yTop + 2, { color: c.ink2, width: 1.2, head: 5 });
            var e = L.barrier / Math.LN10;
            var floorName = S.G === 0 ? 'the floor' : 'the mode ' + (kW < N / 2 ? 'B' : 'A') + ' floor';
            var nats = fmt(L.barrier, { digits: 0 }) + ' nats';
            // narrow plots: three short lines, so the label fits between the walls and the pile
            var shortFloor = S.G === 0 ? 'the floor' : (kW < N / 2 ? 'B' : 'A') + '\u2019s floor';
            var lines = narrow ? ['hill above ' + shortFloor, (e >= 1 ? nats + ': the top is' : nats)] : ['hill above ' + floorName + ': ' + nats];
            var rarePre = narrow ? '' : 'so the hilltop is ', nL = lines.length + (e >= 1 ? 1 : 0);
            var ty = Math.max(box.y0 + 12, nL === 1 ? yTop - 12 : yTop - 30 - 13 * (nL - 2));
            ctx.save(); ctx.font = ui.font('mono', 11);
            var lw = 0;
            lines.concat([rarePre + '6\u00d710' + Math.floor(e) + ' times rarer']).forEach(function (t) { lw = Math.max(lw, ctx.measureText(t).width); });
            lw = lw / 2 + 4;
            ctx.restore();
            // The label never covers a chain or the curve. It goes above the hill when there is a
            // clear spot (the centred one first, then shifts sideways, then raised step by step),
            // else under the valley floors, where neither the curve nor the chains are. The side
            // it used last frame wins while it stays clear, so the label does not jump.
            var al = 'center';
            barrierLabel = function (pile) {
              var dir = kW < L.kS ? 1 : -1, xs = [], f;
              for (f = 0; f <= 40; f++) {
                xs.push(clamp(bx + dir * f * 6, box.x0 + lw, box.x1 - lw));
                if (f) xs.push(clamp(bx - dir * f * 6, box.x0 + lw, box.x1 - lw));
              }
              var hTop = 12, hBot = 13 * (nL - 1) + 5;   // text box around the first baseline
              function curveSpan(x0, x1) {              // highest and lowest screen y of the curve in [x0, x1]
                var t = Infinity, b = -Infinity;
                for (var x = x0; x <= x1; x += 2) {
                  var yy = sy(V[clamp(Math.round((sx.invert(x) + 1) * N / 2), 0, N)]);
                  if (yy < t) t = yy; if (yy > b) b = yy;
                }
                return [t, b];
              }
              function pileTop(x0, x1) {
                var t = Infinity;
                pile.forEach(function (p) { if (p.x > x0 - 4 && p.x < x1 + 4) t = Math.min(t, p.top); });
                return t;
              }
              function above() {
                var best = null;
                for (var y = ty; y >= box.y0 + hTop - 0.5; y -= 4) {
                  for (var ci = 0; ci < xs.length; ci++) {
                    var x = xs[ci], o = Math.max(0, y + hBot - Math.min(curveSpan(x - lw, x + lw)[0], pileTop(x - lw, x + lw)));
                    if (!best || o < best.o) best = { x: x, y: y, o: o };
                    if (o === 0) return best;
                  }
                }
                return best;
              }
              function below() {
                for (var ci = 0; ci < xs.length; ci++) {
                  var x = xs[ci], y = Math.max(curveSpan(x - lw, x + lw)[1], yBot) + hTop + 6;
                  if (y + hBot <= box.y1 - 2) return { x: x, y: y, o: 0 };
                }
                return null;
              }
              var order = S.labSide === 'below' ? [below, above] : [above, below], spot = null, side;
              for (var oi = 0; oi < 2 && !(spot && spot.o === 0); oi++) {
                var sp = order[oi]();
                if (sp && (!spot || sp.o < spot.o)) { spot = sp; side = order[oi] === below ? 'below' : 'above'; }
              }
              if (!spot) return;
              S.labSide = side;
              lines.forEach(function (t, i) { plot.text(ctx, t, spot.x, spot.y + 13 * i, { c: c, size: 11, color: c.ink2, halo: true, align: al }); });
              if (e >= 1) supText(ctx, c, rarePre, e, ' times rarer', spot.x, spot.y + 13 * lines.length, al);
            };
          }
          ctx.restore();

          // chains as a pile of balls sitting on the curve (each ball = one chain)
          var rb = clamp(w / 220, 2.2, 3.1), binW = 2 * rb + 0.6;
          var bins = {}, maxC = 0, cIdx;
          for (cIdx = 0; cIdx < NCH; cIdx++) {
            var kc = chains.k[cIdx], b = Math.round(sx(2 * kc / N - 1) / binW);
            (bins[b] = bins[b] || []).push(kc);
            if (bins[b].length > maxC) maxC = bins[b].length;
          }
          // the pile never rises past the top of the plot: squeeze the stack instead
          var baseOf = {}, roomMin = Infinity;
          Object.keys(bins).forEach(function (bk) {
            var arr = bins[bk], kMean = 0; for (var q = 0; q < arr.length; q++) kMean += arr[q]; kMean /= arr.length;
            var base = Math.min(box.y1 - rb, sy(V[Math.round(kMean)]) - rb - 1);
            baseOf[bk] = base; roomMin = Math.min(roomMin, (base - box.y0 - rb - 4) / Math.max(1, arr.length - 1));
          });
          var dy = Math.max(0.6, Math.min(2 * rb * 0.92, box.h * 0.28 / Math.max(1, maxC), roomMin));
          var pile = [];
          Object.keys(bins).forEach(function (bk) {
            pile.push({ x: +bk * binW, top: baseOf[bk] - (bins[bk].length - 1) * dy - rb });
            var arr = bins[bk].sort(function (a, b2) { return Math.abs(a - N / 2) - Math.abs(b2 - N / 2); });
            var x = +bk * binW, base = baseOf[bk];
            for (var j = 0; j < arr.length; j++) {
              var mm = 2 * arr[j] / N - 1;
              ctx.beginPath(); ctx.arc(x, base - j * dy, rb, 0, 2 * Math.PI);
              ctx.fillStyle = chainColor(mm, c, rgb); ctx.fill();
            }
          });

          if (barrierLabel) barrierLabel(pile);
          // mode labels after the transition ('undecided' is named once, in the fork plot)
          var sAfter = smooth(1.04, 1.09, S.beta), ty = 14;
          if (sAfter > 0.5) {
            plot.text(ctx, 'mode A', sx(0.55), ty, { c: c, size: 11, color: c.modeA, baseline: 'middle', halo: true, align: 'center', weight: 600 });
            plot.text(ctx, 'mode B', sx(-0.55), ty, { c: c, size: 11, color: c.modeB, baseline: 'middle', halo: true, align: 'center', weight: 600 });
          }
        }
      });

      /* ---------------------------------------------------------- fork canvas */
      var forkBox = null;
      var forkCv = ui.canvas(rightCol, {
        height: function (w) { return w < 520 ? Math.round(w * 0.86) : Math.round(clamp(w * 0.9, 300, 400)); },
        label: 'Magnetisation m of 14 runs against the inverse temperature beta: one branch splits into two at the transition, beta = 1',
        draw: function (ctx, w, h, c) {
          var rgb = rgbs(c), narrow = w < 420;
          var box = forkBox = ui.inset(w, h, { top: 20, right: 16, bottom: 40, left: narrow ? 40 : 40 });
          var sx = ui.scale.linear([B0, B1], [box.x0, box.x1]), sy = ui.scale.linear([-1, 1], [box.y1, box.y0]);
          forkBox.sx = sx;
          ctx.globalAlpha = 0.6;
          ctx.fillStyle = c.modeBWash; ctx.fillRect(box.x0, sy(0), box.w, box.y1 - sy(0));
          ctx.fillStyle = c.modeAWash; ctx.fillRect(box.x0, box.y0, box.w, sy(0) - box.y0);
          ctx.globalAlpha = 1;
          plot.axis(ctx, sy, { side: 'left', at: box.x0, ticks: [-1, -0.5, 0, 0.5, 1], c: c, format: function (v) { return v === 0 ? '0' : fmt(v, { digits: 1, sign: v > 0 }); } });
          plot.axis(ctx, sx, { side: 'bottom', at: box.y1, ticks: narrow ? [0.7, 0.9, 1.1, 1.3] : [0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3], c: c, format: function (v) { return fmt(v, { digits: 1 }); } });
          plot.text(ctx, 'm', box.x0 - 4, box.y0 - 8, { c: c, size: 11, color: c.ink3, align: 'right' });
          plot.text(ctx, '← hot', box.x0, box.y1 + 30, { c: c, size: 11, color: c.ink2 });
          plot.text(ctx, 'cold →', box.x1, box.y1 + 30, { c: c, size: 11, color: c.ink2, align: 'right' });
          plot.text(ctx, 'β', sx(1.025), box.y1 + 30, { c: c, size: 11, color: c.ink2, align: 'center' });
          // transition line
          plot.vline(ctx, 1, sx, box.y0, box.y1, { color: c.ink3, dash: [4, 4], width: 1 });

          // theory: stable states
          var br = branches(S.G);
          ctx.save();
          ctx.beginPath(); ctx.rect(box.x0, box.y0, box.w, box.h); ctx.clip();
          plot.line(ctx, br.b, br.up, sx, sy, { color: c.theory, width: 1.6, alpha: 0.9 });
          plot.line(ctx, br.b, br.dn, sx, sy, { color: c.theory, width: 1.6, alpha: 0.9 });
          plot.line(ctx, br.b, br.mid, sx, sy, { color: c.theory, width: 1.2, dash: [3, 4], alpha: 0.8 });
          // with a bias, mode B is only a side valley that opens late: say where, so the late branch
          // and its dashed hilltop do not read as a drawing glitch
          var sideLab = null;
          if (forkSide) forkSide.hidden = !(S.G > 0);
          if (S.G > 0) {
            for (var ib = 1; ib < br.b.length; ib++) if (isFinite(br.dn[ib]) && !isFinite(br.dn[ib - 1])) { sideLab = { b: br.b[ib], m: br.dn[ib] }; break; }
          }

          // runs, revealed up to the cursor, coloured by their own m
          var F = S.G === 0 ? (S.fork && { beta: S.fork.beta, m: S.fork.m }) : (S.forkBias[S.G] && { beta: S.fork.beta, m: S.forkBias[S.G] });
          if (F) {
            var bs = F.beta, jEnd = 0;
            while (jEnd < bs.length - 1 && bs[jEnd + 1] <= S.beta + 1e-9) jEnd++;
            ctx.lineWidth = narrow ? 1 : 1.25; ctx.lineJoin = 'round';
            for (var t = 0; t < F.m.length; t++) {
              var mt = F.m[t];
              for (var j = 1; j <= jEnd; j++) {
                ctx.beginPath(); ctx.moveTo(sx(bs[j - 1]), sy(mt[j - 1])); ctx.lineTo(sx(bs[j]), sy(mt[j]));
                ctx.strokeStyle = chainColor(mt[j], c, rgb); ctx.globalAlpha = 0.85; ctx.stroke();
              }
            }
            ctx.globalAlpha = 1;
          }
          ctx.restore();

          // cursor + the live chains at this beta
          var xc = sx(S.beta);
          ctx.fillStyle = c.noiseWash; ctx.fillRect(xc - 7, box.y0, 14, box.h);
          plot.vline(ctx, S.beta, sx, box.y0, box.y1, { color: c.ink, width: 1.5, dash: false });
          for (var q = 0; q < NCH; q++) {
            var mq = 2 * chains.k[q] / N - 1;
            ctx.beginPath(); ctx.arc(xc, sy(mq), 2, 0, 2 * Math.PI);
            ctx.fillStyle = chainColor(mq, c, rgb); ctx.globalAlpha = 0.7; ctx.fill();
          }
          ctx.globalAlpha = 1;
          // knob
          ctx.beginPath(); ctx.arc(xc, box.y1, 5, 0, 2 * Math.PI); ctx.fillStyle = c.ink; ctx.fill();
          // drawn after the scrub band, so the band never covers it
          plot.text(ctx, 'transition, β = 1', sx(1) - 5, box.y0 + 12, { c: c, size: 11, color: c.ink2, align: 'right', halo: true });

          if (sideLab) {
            var px = sx(sideLab.b), py = sy(sideLab.m);
                        ctx.beginPath(); ctx.arc(px, py, 3.5, 0, 2 * Math.PI); ctx.fillStyle = c.theory; ctx.fill();
            var sideTxt = 'Dot: mode B’s valley first appears, β ≈ ' + fmt(sideLab.b, { digits: 2 });
            if (forkSide.textContent !== sideTxt) forkSide.textContent = sideTxt;
          }
          // labels
          // labels sit in empty regions (top-left / bottom-left), never on the data
          plot.text(ctx, 'undecided', sx(0.715), sy(0.34), { c: c, size: 11, color: c.ink2, halo: true });
          plot.text(ctx, 'mode A', sx(0.715), sy(0.72), { c: c, size: 11, color: c.modeA, weight: 600, halo: true });
          plot.text(ctx, 'mode B', sx(0.715), sy(-0.72), { c: c, size: 11, color: c.modeB, weight: 600, halo: true });
        }
      });
      var forkSide = el('p', { class: 'sp-forknote' });
      rightCol.appendChild(forkSide);
      forkCv.canvas.style.cursor = 'ew-resize';
      function betaFromEvent(e) {
        if (!forkBox) return null;
        var p = forkCv.local(e);
        return clamp(forkBox.sx.invert(p.x), B0, B1);
      }
      var dragging = false;
      function onDown(e) { var b = betaFromEvent(e); if (b == null) return; dragging = true; try { forkCv.canvas.setPointerCapture(e.pointerId); } catch (x) {} setBeta(b, true); }
      function onMove(e) { if (!dragging) return; var b = betaFromEvent(e); if (b != null) setBeta(b, true); }
      function onUp() { dragging = false; }
      forkCv.canvas.addEventListener('pointerdown', onDown);
      forkCv.canvas.addEventListener('pointermove', onMove);
      forkCv.canvas.addEventListener('pointerup', onUp);
      forkCv.canvas.addEventListener('pointercancel', onUp);

      /* ---------------------------------------------------------- controls */
      var playBtn = ui.button({ label: 'Run the cooling', icon: 'play', kind: 'primary', onClick: function () { togglePlay(); } });
      var biasTg = ui.toggle({
        label: 'Field toward mode A',
        options: [
          { value: 'off', label: 'None' },
          { value: 'small', label: 'Small', tone: 'bias', title: 'G = 0.002, field h = βG' },
          { value: 'large', label: 'Large', tone: 'bias', title: 'G = 0.02, field h = βG' }
        ],
        value: 'off',
        onChange: function (v) { S.biasKey = v; S.G = BIAS[v]; if (!S.G) forkSide.textContent = ''; if (S.run) { S.run = null; if (S.playing) setPlaying(false); } S.stale = S.beta > 1; playBtn.classList.toggle('sp-pulse', S.stale); ensureFork(); kick(); render(true); }
      });
      var biasHint = el('p', { class: 'sp-hint', html: 'Try <b>Small</b>, then run the cooling.' });
      fig.controls.appendChild(el('div', { class: 'sp-biasrow' }, playBtn, biasTg.el, biasHint));
      // the end of a run with a field: the model's share of mode A against the chains'
      var gapNote = el('p', { class: 'sp-gap', 'aria-live': 'polite', hidden: true });
      function showGap(main) { gapNote.hidden = false; gapNote.textContent = main; }

      fig.stage.appendChild(fig.controls);
      fig.stage.appendChild(gapNote);

      /* ---------------------------------------------------------- data */
      sim.loadJSON('fork_trajectories.json').then(function (d) {
        if (S.destroyed) return;
        S.fork = d; render(true); ensureFork();
      }).catch(function (e) { console.error(e); });
      var cancel = { cancelled: false };
      function ensureFork() {
        if (!S.fork || S.G === 0 || S.forkBias[S.G] || S.forkPending === S.G) return;
        var G = S.G; S.forkPending = G;
        sim.runChunked(forkGen(sim, S.fork.beta, G, 4), { budgetMs: 6, signal: cancel }).then(function (m) {
          S.forkBias[G] = m; S.forkPending = null; render(true);
        }).catch(function () { S.forkPending = null; });
      }

      /* ---------------------------------------------------------- dynamics */
      function setBeta(b, user) {
        S.beta = clamp(b, B0, B1);
        if (S.stale && S.beta < 0.98) { S.stale = false; playBtn.classList.remove('sp-pulse'); }
        if (user) { S.run = null; if (S.playing) setPlaying(false); }
        kick(); render(true);
      }
      function setPlaying(p) {
        S.playing = p;
        var lab = p ? 'Pause' : S.run && S.beta < B1 - 1e-6 ? 'Resume' : S.run ? 'Run it again' : 'Run the cooling';
        playBtn.replaceChildren(ui.icon(p ? 'pause' : 'play'), el('span', { text: lab }));
        if (p) kick();
      }
      function restart(play) {
        chains.k.set(sampleK(sim, B0, B0 * S.G, NCH, r));
        S.beta = B0;
        S.run = play ? { G: S.G } : null;
        if (play && S.G > 0) biasHint.hidden = true;
        S.stale = false; playBtn.classList.remove('sp-pulse');
        setPlaying(!!play); kick(); render(true);
      }
      function togglePlay() {
        if (S.playing) { setPlaying(false); return; }
        if (S.run && S.beta < B1 - 1e-6) setPlaying(true);   // resume the paused cooling run
        else restart(true);
      }

      function step(dt) {
        var sweeps = Math.min(MAX_SWEEPS_FRAME, dt * SWEEPS_PER_SEC);
        if (!(sweeps > 0)) return;
        if (S.playing) {
          S.beta = Math.min(B1, S.beta + GAMMA * sweeps);
          if (S.beta >= B1) setPlaying(false);
        }
        var h = S.beta * S.G;
        chains.advance(S.beta, h, sweeps);
      }

      function render(force) {
        landCv.redraw(); forkCv.redraw();
        var now = performance.now();
        if (!force && now - S.lastDom < 120) return;
        S.lastDom = now;
        // tie k = N/2 excluded in both shares, as in q_eq
        var up = 0, dn = 0; for (var i = 0; i < NCH; i++) { if (2 * chains.k[i] > N) up++; else if (2 * chains.k[i] < N) dn++; }
        var p = up / Math.max(1, up + dn), q = sim.qEq(N, S.beta, S.G);
        if (S.stale && S.G > 0) {
          showGap('These chains chose their mode before the field was on, and the hill now keeps them there. Run the cooling to start again.');
        } else if (S.G > 0 && S.run && S.run.G === S.G && S.beta >= B1 - 1e-6) {
          var main = 'The model puts ' + pct(q, fmt) + ' of samples in mode A; the ' + (up + dn) + ' chains reach ' + Math.round(100 * p) + '%.';
          if (S.G === BIAS.large) main += ' A large field decides nearly every run; the paper studies weak fields.';
          else main += ' The field acted only near β = 1. Run it again to see the spread.';
          showGap(main);
          biasHint.hidden = true;
        } else gapNote.hidden = true;
      }

      var loop = ui.loop(function (dt) {
        step(dt);
        render(false);
        if (ui.reducedMotion && !S.playing && performance.now() > S.idleUntil) loop.stop();
      }, { el: fig.el, autostart: true });
      function kick() {
        S.idleUntil = performance.now() + 1500;
        if (!loop.running) loop.start();
      }

      setPlaying(false);
      render(true);

      return {
        destroy: function () {
          S.destroyed = true; cancel.cancelled = true;
          loop.destroy(); landCv.destroy(); forkCv.destroy();
          fig.el.remove();
        }
      };
    }
  };
})();
