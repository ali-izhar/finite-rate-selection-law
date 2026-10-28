/* components/race.js — "Why a fraction is lost: field against fluctuations in the selection window"
 *
 * Registers VE.components['race'] = { mount(el, opts) }.
 *
 * What it shows (paper ../paper/main.tex, Sec. 3 and App. B):
 *   Many runs follow the drift-and-fluctuation approximation (paper Eq. 7) of the Curie–Weiss chain near β = 1,
 *       dm = [(β−1) m − m³/3 + G] dt + sqrt(2/N) dW,      β = 1 + γ t   (t in sweeps),
 *   simulated live (Euler–Maruyama, seeded) from m = 0 at t = −4 γ^(−1/2).
 *   Panel A: the runs' m(t), coloured by the mode they end in.
 *   In the linearised dynamics dm = (γ t m + G) dt + sqrt(2/N) dW, y = m·exp(−γt²/2) is Gaussian with
 *   mean (the field's push) G sqrt(2π/γ) and sd (the fluctuation spread) sqrt(2/N)(π/γ)^(1/4), and
 *   sign(y) = sign(m); so the share in mode A is Φ(push ÷ spread) = Φ(G√N(π/γ)^¼), the selection
 *   formula (paper Eq. 3). The push ÷ spread value is printed in the formula's label.
 *   Panel B: the live share of runs with m > 0 vs time (faded before the window, where it is not
 *            final), against the selection formula and the model's own equilibrium share q_eq; the gap is "lost".
 *
 * Validity of the formula in the drift-and-fluctuation approximation: the simulation depends only on x and Nγ.
 * Checked with this page's kernel (40,000 runs per setting) over N = 1,024–16,384 and the γ range:
 * |sim − formula| ≤ 0.7 points, the largest gap at N = 1024, γ = 2.5×10⁻⁴ (Nγ = 0.256).
 * At the fixed N = 2,048 the γ slider keeps Nγ ≥ 0.512. The paper's exact count checks
 * (Table 2, App. A) cover N = 1,024–4,096, γ = 0.00025–0.004 only.
 *
 * Numbers shown and their sources:
 *   formula q_sel = Φ(G√N(π/γ)^¼)           Eq. (eq:law); VE.sim.knX / VE.sim.Phi
 *   q_eq at β = 1.6                  exact Curie–Weiss, VE.sim.qEq / logOddsEq (tested against ../src)
 *   simulated share                  live, on this page
 *   Kondepudi & Nelson 1983; Kondepudi & Gao 1987 (paper bib: kondepudinelson1983, kondepudigao1987)
 * Controls: the cooling rate γ only, starting on the test's ramp (γ = 0.004, NG = 1.2544, the
 * setting of chapter 2); each new run plays once by itself. G and N stay fixed.
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
  var REF = { G: 0.0006125, N: 2048, gamma: 0.004 };   // the test's ramp (NG = 1.2544; a = 0.147), as in chapter 2

  function mount(root, opts) {
    var ui = VE.ui, sim = VE.sim, el = ui.el, fmt = ui.fmt;
    opts = opts || {};

    /* ------------------------------------------------------------ state */
    var P = { G: REF.G, N: REF.N, gamma: REF.gamma };
    var run = null;        // current simulation result (accumulated over batches)
    var job = null;        // {cancelled} signal of the running simulation
    var pending = false;   // sliders moved; simulation stale
    var tNow = T_SHOW;     // playhead in σ units (rest frame = end of the run)
    var seedCounter = 1;

    /* ------------------------------------------------------------ maths */
    function sigmaT(g) { return 1 / Math.sqrt(g); }
    function lawX(p) { return sim.knX(p.G, p.N, p.gamma); }
    function lawQ(p) { return sim.Phi(lawX(p)); }
    var lqCache = {};
    function logOddsEq16(p) {
      var k = p.N + '|' + p.G;
      if (!(k in lqCache)) lqCache[k] = sim.logOddsEq(p.N, 1.6, p.G);
      return lqCache[k];
    }
    function qEq16(p) { return sim.expit(logOddsEq16(p)); }
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
      // a bound on |m|: the stable state at the last shown level plus room for noise
      var bEnd = T_SHOW * sg * p.gamma;
      var mHist = Math.max(0.15, 1.18 * Math.sqrt(3 * bEnd) + 2.5 * Math.sqrt(1 / p.N));
      var t0 = T_START * sg;
      function record(j) {
        var c = 0;
        for (var i = 0; i < n; i++) if (m[i] > 0) c++;
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
      return { n: n, M: M, nT: nT, traces: traces, plus: plus, mHist: mHist, mShow: mShow, fate: fate, fin: fin };
    }

    function startRun() {
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
        res.p = p;
        run = res;
        pending = false;
        root.classList.remove('is-stale');
        playOnce();
      }, function () { /* cancelled */ });
    }

    /* ------------------------------------------------------------- DOM */
    var fig = ui.figure({
      plate: opts.plate != null ? opts.plate : 4,
      title: 'Why a fraction is lost: field against fluctuations',
      className: 've-race'
    });
    root.appendChild(fig.el);
    fig.el.classList.add('is-v2');

    var cv = ui.canvas(fig.stage, {
      height: function (w) { return w < 560 ? 560 : Math.round(Math.min(500, Math.max(440, w * 0.44))); },
      label: 'Two aligned panels over time: sampler runs splitting into two modes inside the selection window, and the share of runs in mode A settling on the selection formula, below the model’s own preference.',
      draw: draw
    });

    // one control: the cooling rate. Each new run plays once from the hot end to the cold end.
    var slGam = ui.slider({ label: 'Cooling rate', sym: 'γ', min: 2.5e-4, max: 1.6e-2, value: P.gamma, log: true, tone: 'theory',
      format: function (v) { return fmt(v, { sci: true, sig: 2 }); }, unit: 'per sweep',
      onInput: function (v) { P.gamma = v; changed(); } });
    fig.stage.appendChild(el('div', { class: 've-race__sliders' }, slGam.el));
    fig.controls.remove();

    var anim = 0, PLAY_MS = 4500;
    function playOnce() {
      if (anim) cancelAnimationFrame(anim);
      anim = 0;
      if (ui.reducedMotion) { tNow = T_SHOW; refresh(); return; }
      var t0 = performance.now();
      (function f(now) {
        var k = Math.min(1, (now - t0) / PLAY_MS);
        tNow = T_START + (T_SHOW - T_START) * (1 - Math.pow(1 - k, 1.6));
        refresh();
        anim = k < 1 ? requestAnimationFrame(f) : 0;
      })(t0);
    }

    /* ------------------------------------------------------------ update */
    var debounce = 0;
    function changed(now) {
      pending = true;
      root.classList.add('is-stale');
      refresh();
      clearTimeout(debounce);
      debounce = setTimeout(startRun, now ? 0 : 180);
    }

    function refresh() { cv.redraw(); }

    /* ------------------------------------------------------------- draw */
    function geom(w) {
      var phone = w < 560;
      return { phone: phone, L: phone ? 38 : 50, R: phone ? 62 : 72 };
    }
    function draw(ctx, w, h, c) {
      var p = run ? run.p : P;
      var g = geom(w), phone = g.phone, L = g.L, R = g.R;
      var T = 62, Bm = 44, gap = 38;
      var H = h - T - Bm - gap;
      var h1 = Math.round(H * 0.5);
      var P1 = { y0: T, y1: T + h1 }, P3 = { y0: P1.y1 + gap, y1: h - Bm };
      var x0 = L, x1 = w - R;
      var sg = sigmaT(p.gamma);
      var sx = ui.scale.linear([T_START, T_SHOW], [x0, x1]);  // σ units
      var tN = Math.max(T_START, Math.min(T_SHOW, tNow));
      var px = sx(tN);
      var fs = phone ? 11.5 : 12;   // canvas text: no readable label below 11 px
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

      // bracket over the window
      var bx0 = sx(-1), bx1 = sx(1), by = 18;
      bracket(ctx, bx0, bx1, by, c.ink2, 1.2, null);
      boxText(ctx, 'window ±' + fmt(Math.round(sg)) + ' sweeps', (bx0 + bx1) / 2, by - 5, { c: c, size: fs, align: 'center', color: c.ink2, weight: 600, halo: true });

      // ---------- panel 1: runs
      var my = run ? run.mShow : 0.5;
      var sy1 = ui.scale.linear([-my, my], [P1.y1, P1.y0]);
      ui.plot.hline(ctx, 0, sy1, x0, x1, { color: c.ruleStrong, width: 1 });
      ui.plot.axis(ctx, sy1, { side: 'left', at: x0, ticks: tickSym(my), format: function (v) { return fmt(v, { digits: tickDigits(my) }); }, c: c, size: fs - 0.5 });
      panelTitle(ctx, 'A · Runs', x0, P1.y0, c, fs);
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
        }
      }
      ui.plot.hline(ctx, 0.5, sy3, x0, x1, { color: c.ruleStrong, width: 1, dash: [1, 3] });
      ui.plot.hline(ctx, qe, sy3, x0, x1, { color: c.bias, width: 1.6, dash: false });
      ui.plot.hline(ctx, q, sy3, x0, x1, { color: c.theory, width: 1.6, dash: [6, 4] });
      panelTitle(ctx, 'B · Share of runs in mode A', x0, P3.y0, c, fs);
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
      var mName = 'model’s own share';
      var lName = 'selection formula';
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
        { y: yM, t: probStr(qe).replace('&gt;', '>'), color: c.bias },
        { y: yL, t: probStr(q).replace('&gt;', '>'), color: c.theory }
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
      boxText(ctx, '← β ' + fmt(1 + T_START * sg * p.gamma, { digits: 2 }), x0 - (phone ? 30 : 0), h - 4, { c: c, size: fs - 0.5, color: c.ink3 });
      boxText(ctx, 'β ' + fmt(1 + T_SHOW * sg * p.gamma, { digits: 2 }) + ' →', w - 2, h - 4, { c: c, size: fs - 0.5, align: 'right', color: c.ink3 });
    }

    // title on one line and a plain-English subtitle under it, both in the gap above the panel
    /* A label with halo:true gets a solid surface-coloured box behind it, so the dashed window lines
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

    /* ---------------------------------------------------------- start */
    refresh();
    startRun();

    return {
      destroy: function () { if (job) job.cancelled = true; clearTimeout(debounce); if (anim) cancelAnimationFrame(anim); cv.destroy(); fig.el.remove(); }
    };
  }

  VE.components['race'] = { mount: mount };
})();
