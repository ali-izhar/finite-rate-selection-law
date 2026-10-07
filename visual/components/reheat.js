/* components/reheat.js — "Where the test runs start: reheating"  (paper Sec. 4, App. B; docs/mnist-illustration.md)
 *
 * Three plates:
 *   a  Waiting (repository computation, not in the paper): landscape at beta = 2,
 *      h = 0.05; expected wait to switch modes for N = 256 … 16,384 (exact count calculation,
 *      repository data); a wall of digits for a felt sense.
 *   b  Reheating, an illustration: drag beta down, the negative well flattens and vanishes; the
 *      dot follows the large-N drift dm/dt = -m + tanh(beta m + h) with an exaggerated field.
 *      Play reheats at beta = 0.7, then cools.
 *   c  MNIST Boltzmann machine illustration (docs/mnist-illustration.md): 24 runs, 141 frames,
 *      classifier p(3) - p(8) vs sweep, ten-seed control arms; shading marks the reheating window.
 *
 * Data (all relative, read once):
 *   data/switching_times.json   results/figures/switching_times.json  (log10 wait vs beta, N)
 *   data/mnist_illustration.json       results/experiments/dbm_reheat.npz + results/experiments/dbm_control_arm.json
 *   assets/mnist_illustration_frames.png  sprite of the 24 x 141 frames (28 px tiles, white = on; turned
 *                                 into alpha = pixel on once, at load)
 * Computed on the page: the mean-field landscape F(m) = -beta m^2/2 - h m - S(m) and its
 * stationary points (spinodal 1.18408 checks against the data file's spinodal_beta).
 */
(function () {
  'use strict';
  var VE = (window.VE = window.VE || {});
  VE.components = VE.components || {};

  var H = 0.05;                 // exaggerated field of the illustration (the paper's G is 0.0006125)
  var BETA_LO = 0.6, BETA_HI = 2.0;
  var BETA_RH = 0.7;            // the paper's reheating inverse temperature (Sec. 4, App. B)
  var REPO = 'From the repository’s computations, not in the paper';

  /* ------------------------------------------------------------ mean-field model */
  function F(m, b, h) {
    var a = (1 + m) / 2, c = (1 - m) / 2;
    var s = (a > 0 ? a * Math.log(a) : 0) + (c > 0 ? c * Math.log(c) : 0);
    return -b * m * m / 2 - h * m + s;
  }
  function atanh(x) { return 0.5 * Math.log((1 + x) / (1 - x)); }
  function g(m, b, h) { return atanh(m) - b * m - h; }        // dF/dm
  function rootIn(lo, hi, b, h) {                              // bisection for g = 0 on [lo, hi]
    var glo = g(lo, b, h);
    if ((glo < 0) === (g(hi, b, h) < 0)) return null;
    for (var k = 0; k < 52; k++) { var mid = (lo + hi) / 2; if ((g(mid, b, h) < 0) === (glo < 0)) lo = mid; else hi = mid; }
    return (lo + hi) / 2;
  }
  function stationary(b, h) {                                  // sorted stationary points of F
    // g' = 1/(1-m^2) - b vanishes at m = +-sqrt(1 - 1/b), so g is monotone between those points
    var E = 1 - 1e-12, cuts = b > 1 ? [-E, -Math.sqrt(1 - 1 / b), Math.sqrt(1 - 1 / b), E] : [-E, E], out = [];
    for (var i = 0; i + 1 < cuts.length; i++) { var r = rootIn(cuts[i], cuts[i + 1], b, h); if (r !== null) out.push(r); }
    return out;
  }
  function spinodal(h) {                                       // beta where the wrong well and the barrier top merge
    function q(b) { var mc = -Math.sqrt(1 - 1 / b); return g(mc, b, h); }   // > 0 while the wrong well exists
    var lo = 1.0001, hi = 3;
    for (var k = 0; k < 60; k++) { var mid = (lo + hi) / 2; if (q(mid) > 0) hi = mid; else lo = mid; }
    return (lo + hi) / 2;
  }
  function drift(m, b, h) { return -m + Math.tanh(b * m + h); }
  function stepDrift(m, b, h, sweeps) {                        // RK4, dt <= 0.05 sweep
    var n = Math.max(1, Math.ceil(sweeps / 0.05)), dt = sweeps / n;
    for (var i = 0; i < n; i++) {
      var k1 = drift(m, b, h), k2 = drift(m + dt * k1 / 2, b, h), k3 = drift(m + dt * k2 / 2, b, h), k4 = drift(m + dt * k3, b, h);
      m += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6;
    }
    return Math.max(-0.9999, Math.min(0.9999, m));
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath(); ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h); ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r); ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y); ctx.closePath();
  }

  /* --------------------------------------------- landscape drawing (plates a, b) */
  var YMAX = 0.5;
  function drawLandscape(ctx, w, h, c, o) {
    var ui = VE.ui, plot = ui.plot;
    var narrow = w < 420;
    var box = ui.inset(w, h, { top: 30, right: 12, bottom: 44, left: narrow ? 34 : 44 });
    var sx = ui.scale.linear([-1, 1], [box.x0, box.x1]);
    var sy = ui.scale.linear([-0.03, YMAX], [box.y1, box.y0]);
    var b = o.beta, st = stationary(b, H);
    var fmin = Infinity, n = 360, xs = [], ys = [];
    for (var i = 0; i <= n; i++) { var m = -0.9995 + 1.999 * i / n; var f = F(m, b, H); xs.push(m); ys.push(f); if (f < fmin) fmin = f; }
    for (i = 0; i <= n; i++) ys[i] -= fmin;
    // axes
    plot.axis(ctx, sx, { side: 'bottom', at: box.y1, ticks: [-1, -0.5, 0, 0.5, 1], c: c,
      format: function (v) { return ui.fmt(v, v === 0 ? 0 : (Math.abs(v) === 1 ? 0 : 1)); }, grid: [box.y0, box.y1] });
    plot.axis(ctx, sy, { side: 'left', at: box.x0, ticks: [0, 0.1, 0.2, 0.3, 0.4], c: c, format: function (v) { return ui.fmt(v, 1); } });
    plot.text(ctx, 'm: which mode the sample is in', (box.x0 + box.x1) / 2, box.y1 + 32, { align: 'center', color: c.ink2, size: 11, weight: 500, c: c });
    plot.text(ctx, narrow ? 'cost: −log p(m) / N' : 'cost of state m:  −log p(m) / N', box.x0 - (narrow ? 30 : 40), box.y0 - 14, { color: c.ink2, size: 11, weight: 500, c: c });
    // fill under curve, coloured by side
    ctx.save();
    ctx.beginPath(); ctx.rect(box.x0, box.y0 - 2, box.w, box.h + 2); ctx.clip();
    ['B', 'A'].forEach(function (side) {
      ctx.beginPath();
      var started = false;
      for (var j = 0; j <= n; j++) {
        if (side === 'B' ? xs[j] > 0 : xs[j] < 0) continue;
        var X = sx(xs[j]), Y = sy(Math.min(ys[j], YMAX + 0.1));
        if (!started) { ctx.moveTo(X, sy(-0.03)); started = true; }
        ctx.lineTo(X, Y);
      }
      ctx.lineTo(side === 'B' ? sx(0) : sx(1), sy(-0.03));
      ctx.closePath();
      ctx.fillStyle = side === 'B' ? c.modeBWash : c.modeAWash; ctx.fill();
    });
    var yc = ys.map(function (v) { return Math.min(v, YMAX + 0.1); });
    plot.line(ctx, xs, yc, sx, sy, { color: c.ink, width: 2 });
    ctx.restore();
    // side labels
    plot.text(ctx, narrow ? '← negative mode' : '← negative mode (m < 0)', box.x0 + 4, box.y0 + 2, { color: c.modeB, size: 11.5, weight: 600, baseline: 'top', halo: true, c: c });
    plot.text(ctx, narrow ? 'favoured mode →' : 'favoured mode (m > 0) →', box.x1 - 4, box.y0 + 2, { color: c.modeA, size: 11.5, weight: 600, align: 'right', baseline: 'top', halo: true, c: c });

    // barrier annotation (when the wrong well exists)
    if (o.barrier && st.length >= 3) {
      var mw = st[0], ms = st[1];
      var fw = F(mw, b, H) - fmin, fs = F(ms, b, H) - fmin;
      var xA = sx(ms), y0 = sy(fw), y1 = sy(fs);
      ctx.save();
      ctx.setLineDash([3, 3]); ctx.strokeStyle = c.ink3; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(sx(mw) + 8, y0); ctx.lineTo(xA + 6, y0); ctx.stroke();
      ctx.restore();
      var dF = fs - fw;
      if (dF < 0.04) {                       // too small for an arrow: label above the saddle with a leader line
        var ly = Math.min(y1, y0) - 34;
        ctx.strokeStyle = c.ink; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(xA, y1 - 3); ctx.lineTo(xA, ly + 4); ctx.stroke();
        plot.text(ctx, 'barrier ' + ui.fmt(dF, 3) + ' per variable', xA, ly, { align: 'center', color: c.ink, size: 11, weight: 600, halo: true, c: c });
      } else if (dF < 0.15) {                // mid-size barrier: arrow, label centred above the flat top
        plot.arrow(ctx, xA, y0, xA, y1 + 3, { color: c.ink, width: 2, head: 7 });
        plot.text(ctx, 'barrier ' + ui.fmt(dF, 3) + ' per variable', xA, y1 - 12, { align: 'center', color: c.ink, size: 11, weight: 600, halo: true, c: c });
      } else {
        plot.arrow(ctx, xA, y0, xA, y1 + 3, { color: c.ink, width: 2, head: 7 });
        // right of the arrow, just above the dashed level of the wrong well: the clear area under the
        // curve's right flank at every width (left of the arrow the rising flank runs through it)
        var lx = xA + 7, la = 'left', ly0 = y0;
        plot.text(ctx, 'barrier ' + ui.fmt(dF, 3), lx, ly0 - 20, { align: la, color: c.ink, size: 11.5, weight: 600, halo: true, c: c });
        plot.text(ctx, 'per variable', lx, ly0 - 6, { align: la, color: c.ink2, size: 11, halo: true, c: c });
      }
    }
    // the wrong well's last position when it is gone
    if (o.ghost != null) {
      ctx.save(); ctx.setLineDash([2, 3]); ctx.strokeStyle = c.modeB; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(sx(o.ghost), sy(F(o.ghost, b, H) - fmin) - 9, 7, 0, 2 * Math.PI); ctx.stroke(); ctx.restore();
    }
    // the chain
    if (o.ball != null) {
      var bm = o.ball, by = F(bm, b, H) - fmin;
      var X = sx(bm), Y = sy(Math.min(by, YMAX)) - 9;
      ctx.beginPath(); ctx.arc(X, Y, 8, 0, 2 * Math.PI);
      ctx.fillStyle = bm < 0 ? c.modeB : c.modeA; ctx.fill();
      ctx.lineWidth = 2; ctx.strokeStyle = c.surface; ctx.stroke();
      if (o.ballLabel) {
        // near a wall: put the label on the open side, above the dot, clear of the steep curve
        var edge = bm < -0.8 ? 'left' : (bm > 0.8 ? 'right' : 'center');
        plot.text(ctx, o.ballLabel, edge === 'left' ? X + 12 : (edge === 'right' ? X - 4 : X), edge === 'left' ? Y - 10 : (edge === "right" ? Y - 34 : Y - 15),
          { align: edge === 'left' ? 'left' : (edge === 'right' ? 'right' : 'center'), color: c.ink, size: 11, weight: 600, halo: true, c: c });
      }
    }
    // beta tag
    plot.text(ctx, 'β = ' + ui.fmt(b, 2) + (narrow ? '' : '   h = 0.05 (exaggerated)'), box.x1, box.y0 - 14, { align: 'right', color: c.ink3, size: 11, c: c });
  }

  /* ================================================================ component */
  function mount(root, opts) {
    var ui = VE.ui, el = ui.el, fmt = ui.fmt;
    opts = opts || {};
    var plateBase = opts.plate != null ? String(opts.plate) : '9';
    var sub = /\d$/.test(plateBase) ? '' : '·';            // "Fig. 8a" but "Fig. M·a"
    var prefix = opts.prefix != null ? String(opts.prefix) : 'Plate ';
    /* part: 'all' (dev/reheat.html?all only: B on top, A and C folded under it), or one plate
       alone, so that index.html can mount B in the chapter and A ('wait') and C ('dbm') in lazy
       Plot drawers. */
    var part = opts.part || 'all';
    var want = { wait: part === 'all' || part === 'wait', reheat: part === 'all' || part === 'reheat', dbm: part === 'all' || part === 'dbm' };
    var host = el('div', { class: 've-reheat' + (part !== 'all' ? ' ve-reheat--part ve-reheat--' + part : '') });
    root.appendChild(host);
    var loading = el('p', { class: 've-small', text: 'Loading the reheat data…' });
    host.appendChild(loading);
    var cleanups = [];
    function defer(fn) { var id = setTimeout(fn, 0); cleanups.push(function () { clearTimeout(id); }); }   // one plate (or one canvas) per task, so no task blocks for long

    var spritePromise = !want.dbm ? null : new Promise(function (resolve, reject) {
      var img = new Image();
      img.onload = function () {
        /* the sprite is white = pixel on; turn it once into black with alpha = R, so that plate C
           can tint a tile with 'source-in' (alpha = pixel on) */
        var cv = document.createElement('canvas'); cv.width = img.width; cv.height = img.height;
        var x = cv.getContext('2d');
        x.drawImage(img, 0, 0);
        var d = x.getImageData(0, 0, cv.width, cv.height), px = d.data;
        for (var i = 0; i < px.length; i += 4) { px[i + 3] = px[i]; px[i] = px[i + 1] = px[i + 2] = 0; }
        x.putImageData(d, 0, 0);
        resolve(cv);
      };
      img.onerror = function () { reject(new Error('assets/mnist_illustration_frames.png did not load')); };
      img.src = VE.sim.dataUrl('../assets/mnist_illustration_frames.png');
    });
    Promise.all([want.wait ? VE.sim.loadJSON('switching_times.json') : null,
      want.dbm ? VE.sim.loadJSON('mnist_illustration.json') : null, spritePromise])
      .then(function (r) { loading.remove(); build(r[0], r[1], r[2]); ui.terms(host); })
      .catch(function (e) { console.error(e); loading.textContent = 'Could not load the reheat data: ' + e.message; });

    function build(mfpt, dbm, sprite) {
      function cv(parent, o) { var c = ui.canvas(parent, o); cleanups.push(function () { c.destroy(); }); return c; }
      var SPIN = spinodal(H);

      if (part === 'all' && opts.intro !== false) host.appendChild(el('div', { class: 've-prose ve-reheat__intro' },
        el('p', { html: 'The paper tests its prediction on runs from a <span data-term="filteredStart">filtered start</span>, which choose a mode again after reheating. ' +
          'After a brief cooling, runs labelled (pattern 1, +) are removed for both signs of the field; the rest are <span data-term="reheating">reheated</span> for 30 sweeps at β = 0.7, then cooled on the paper’s schedule.' })));

      /* Order on screen: plate B (the reheat) first; A (why not wait) and C (a trained network)
         fold away under it, so the method takes one screen unless the reader asks for more. */
      var slotB = host, moreA = host, moreC = host;
      function fold(summary) {
        var d = el('details', { class: 've-reheat__more' }, el('summary', { html: summary }));
        host.appendChild(d);
        return d;
      }
      if (part === 'all') {
        slotB = el('div', { class: 've-reheat__slot' });
        host.appendChild(slotB);
        moreA = fold('<span>Optional</span><b>From the repository: waiting for a switch</b><em>(plate ' + plateBase + sub + 'a)</em>');
        moreC = fold('<span>Optional</span><b>A trained network (illustration)</b><em>An MNIST deep Boltzmann machine, reheated (plate ' + plateBase + sub + 'b)</em>');
      }
      /* a plate's "how it is set" text: a fold inside the caption when the plates share one figure
         (part 'all'), a plain paragraph when the plate already sits inside a drawer (one level only) */
      function how(summary, body) {
        if (Array.isArray(body)) body = body.join(part === 'all' ? ' ' : '</span><span class="ve-reheat__howb">');
        return part === 'all'
          ? '<details class="ve-reheat__how"><summary>' + summary + '</summary><span class="ve-reheat__howb">' + body + '</span></details>'
          : '<span class="ve-reheat__howb">' + body + '</span>';
      }

      if (want.wait) plateA();
      if (want.reheat) defer(function () { plateB(); if (want.dbm) defer(plateC); });
      else if (want.dbm) defer(plateC);

      function plateA() {
      /* ========================== PLATE A: waiting ========================== */
      var fa = ui.figure({ plate: prefix + plateBase + sub + 'a', title: 'From the repository: how long a switch takes at β = 2', className: 've-reheat__plate' });
      moreA.appendChild(fa.el);
      var solo = part === 'wait';          // the chapter's figure above already shows this landscape
      var aGrid = el('div', { class: 've-reheat__two' + (solo ? ' ve-reheat__two--solo' : '') });
      fa.stage.appendChild(aGrid);
      var aLeft = el('div', { class: 've-reheat__cell' }), aRight = el('div', { class: 've-reheat__cell ve-reheat__waitcol' });
      if (solo) aGrid.append(aRight); else aGrid.append(aLeft, aRight);
      if (!solo) cv(aLeft, { aspect: 1.55, minHeight: 230, maxHeight: 320,
        label: 'Cost landscape of the two-mode model at beta = 2 with field h = 0.05. The run sits in the negative well at m = -0.95; a barrier of 0.280 per variable separates it from the favoured well at m = +0.96.',
        draw: function (ctx, w, h, c) { drawLandscape(ctx, w, h, c, { beta: 2.0, ball: stationary(2.0, H)[0], ballLabel: 'run', barrier: true }); } });

      var NS = ['256', '1024', '4096', '16384'];
      var nSel = '16384';
      var tN = ui.toggle({ id: ui.uid('rh-n'), label: 'Model size N (variables)', value: nSel,
        options: NS.map(function (n) { return { value: n, label: fmt(+n) }; }),
        onChange: function (v) { nSel = v; updateWait(); } });
      var rWait = ui.readout({ label: 'Expected wait to switch', html: true, unit: 'sweeps', format: function (v) { return v; } });
      var seqLine = el('p', { class: 've-reheat__seq' });
      aRight.append(tN.el, el('div', { class: 've-readouts' }, rWait.el), seqLine);

      var wallWrap = el('div', { class: 've-reheat__wall' });
      fa.stage.appendChild(wallWrap);
      var wallHead = el('p', { class: 've-reheat__wallhead' });
      wallWrap.appendChild(wallHead);
      var wallCv = null;
      var wallLegend = el('div', { class: 've-legend ve-reheat__walllegend' });
      wallWrap.appendChild(wallLegend);
      defer(function () {                   // its own task: each canvas forces a layout when it is made
        wallCv = cv(null, {
          height: function (w) { var L = wallLayout(w, digitsOf(nSel)); return L.h; },
          label: 'One square per digit of the expected wait.',
          draw: function (ctx, w, h, c) { drawWall(ctx, w, c); } });
        wallWrap.insertBefore(wallCv.wrap, wallLegend);
        wallCv.resize();
      });

      function digitsOf(n) { return Math.floor(mfpt.log10_T_at_beta2[n]) + 1; }
      function wallLayout(w, nd) {
        var cell = w < 520 ? 6 : 8, gap = w < 520 ? 1 : 2;
        var per = Math.max(10, Math.floor((w + gap) / (cell + gap)));
        var rows = Math.ceil(nd / per);
        return { cell: cell, gap: gap, per: per, rows: rows, h: rows * (cell + gap) - gap + 2 };
      }
      function drawWall(ctx, w, c) {
        var nd = digitsOf(nSel), L = wallLayout(w, nd);
        for (var i = 0; i < nd; i++) {
          var r = Math.floor(i / L.per), k = i % L.per;
          var x = k * (L.cell + L.gap), y = 1 + r * (L.cell + L.gap);
          ctx.fillStyle = c.ink3; ctx.globalAlpha = 0.7;
          ctx.fillRect(x, y, L.cell, L.cell);
        }
        ctx.globalAlpha = 1;
      }
      function updateWait() {
        var lg = mfpt.log10_T_at_beta2[nSel];
        rWait.set(ui.fmtPow10Html(Math.round(lg)), 'exact CW chain · β = 2 · h = 0.05 · N = ' + fmt(+nSel));
        var nd = digitsOf(nSel);
        wallHead.innerHTML = 'Written out, the wait has <b>' + fmt(nd) + ' digits</b>. Each square below is one digit.';
        wallLegend.replaceChildren();
        seqLine.innerHTML = 'Each 4× in N multiplies the digits by about 4: ' + NS.map(function (n) {
          var d = Math.round(mfpt.log10_T_at_beta2[n]);
          return n === nSel ? '<b>10<sup>' + d + '</sup></b>' : '10<sup>' + d + '</sup>';
        }).join(' → ') + '. The number of digits grows in proportion to N, so the wait grows exponentially.';
        if (wallCv) wallCv.resize();
      }
      updateWait();
      fa.setCaption('<b>' + REPO + '.</b> The model has two modes, m &lt; 0 and m &gt; 0. A field h = 0.05, much larger than the paper’s, tilts the landscape' + (solo ? ' (as in the chapter’s figure at β = 2)' : '') + ' so that the <b class="ve-c-mode-a">favoured mode</b> (m &gt; 0) is slightly more likely than the <b class="ve-c-mode-b">negative mode</b>. ' +
        'At β = 2 a run in the negative well reaches the favoured mode only by climbing the <span data-term="barrier">barrier</span> by chance. ' +
        'The barrier per variable stays the same as the model grows, so the expected wait grows like e<sup>0.280·N</sup>. Long sampling does reach equilibrium, but only after this wait. ' +
        'A <span data-term="sweep">sweep</span> is N single-spin Gibbs updates.');
      fa.setTakeaway('At β = 2, h = 0.05 and N = 16,384, the expected wait to switch modes is 10<sup>1994</sup> sweeps.');
      fa.setNote(REPO + ' · source: data/switching_times.json · exact CW chain (N + 1 states) · β = 2, h = 0.05', 'paper');
      }

      function plateB() {
      /* ========================== PLATE B: reheating ========================== */
      var fb = ui.figure({ plate: prefix + plateBase, title: 'Reheating: lower β until one well remains', className: 've-reheat__plate' });
      slotB.appendChild(fb.el);
      var bGrid = el('div', { class: 've-reheat__two' });
      fb.stage.appendChild(bGrid);
      var bLeft = el('div', { class: 've-reheat__cell' }), bRight = el('div', { class: 've-reheat__cell' });
      bGrid.append(bLeft, bRight);

      var stB = { beta: 2.0, m: stationary(2.0, H)[0], trail: [], play: null, ghost: null };
      stB.trail.push([stB.beta, stB.m]);
      var landB = cv(bLeft, { aspect: 1.45, minHeight: 240, maxHeight: 340,
        label: 'Cost landscape at the selected inverse temperature, with the run as a dot.',
        draw: function (ctx, w, h, c) {
          drawLandscape(ctx, w, h, c, { beta: stB.beta, ball: stB.m, ballLabel: 'run', barrier: true, ghost: stB.ghost });
        } });

      // branch diagram
      var branchCache = null;
      function branches() {
        if (branchCache) return branchCache;
        var top = [], wrong = [], sad = [];
        for (var i = 0; i <= 420; i++) {
          var b = BETA_LO + (BETA_HI - BETA_LO) * i / 420, s = stationary(b, H);
          if (s.length >= 3) { top.push([b, s[2]]); wrong.push([b, s[0]]); sad.push([b, s[1]]); }
          else if (s.length) top.push([b, s[s.length - 1]]);
        }
        var sS = stationary(SPIN + 1e-6, H);
        wrong.unshift([SPIN, (sS[0] + sS[1]) / 2]); sad.unshift([SPIN, (sS[0] + sS[1]) / 2]);
        branchCache = { top: top, wrong: wrong, sad: sad, tip: [SPIN, (sS[0] + sS[1]) / 2] };
        return branchCache;
      }
      var branchLayout = null;
      var brCv = cv(bRight, { aspect: 1.25, minHeight: 300, maxHeight: 400,
        label: 'Stable states of m against the inverse temperature beta.',
        draw: function (ctx, w, h, c) { drawBranch(ctx, w, h, c); } });
      function drawBranch(ctx, w, h, c) {
        var plot = ui.plot, narrow = w < 420;
        var box = ui.inset(w, h, { top: 30, right: 12, bottom: 44, left: narrow ? 34 : 40 });
        var sx = ui.scale.linear([BETA_LO, BETA_HI], [box.x0, box.x1]);
        var sy = ui.scale.linear([-1, 1], [box.y1, box.y0]);
        branchLayout = { sx: sx, box: box };
        var B = branches();
        // region with no wrong well (top panel only)
        ctx.fillStyle = c.surface2 || c.noiseWash; ctx.fillRect(box.x0, box.y0, sx(B.tip[0]) - box.x0, box.y1 - box.y0);
        var bandHalo = c.surface2 || c.noiseWash;   // labels inside the band get a halo of the band's colour
        plot.axis(ctx, sy, { side: 'left', at: box.x0, ticks: [-1, 0, 1], c: c, format: function (v) { return v === 0 ? '0' : (v > 0 ? '+1' : '−1'); }, grid: [box.x0, box.x1] });
        plot.text(ctx, narrow ? 'where m can rest' : 'where m can rest: well bottoms and the barrier top', box.x0 - (narrow ? 30 : 36), box.y0 - 14, { color: c.ink2, size: 11, weight: 500, c: c });
        // current beta: drawn first, so every label sits on top of it
        plot.vline(ctx, stB.beta, sx, box.y0, box.y1 + 8, { color: c.ink, width: 1.5 });
        // transition rule beta = 1
        plot.vline(ctx, 1, sx, box.y0, box.y1 + 8, { color: c.ruleStrong, dash: [4, 3], width: 1 });
        // trail of the chain: thin and faint, under the branches
        if (stB.trail.length > 1) {
          ctx.save(); ctx.globalAlpha = 0.55;
          plot.line(ctx, stB.trail.map(function (p) { return p[0]; }), stB.trail.map(function (p) { return p[1]; }), sx, sy, { color: c.ink3, width: 1 });
          ctx.restore();
        }
        // branches
        function path(pts, col, wd, dash) { plot.line(ctx, pts.map(function (p) { return p[0]; }), pts.map(function (p) { return p[1]; }), sx, sy, { color: col, width: wd, dash: dash }); }
        path(B.sad, c.ink3, 1.5, [4, 4]);
        path(B.top, c.modeA, 2.5);
        path(B.wrong, c.modeB, 2.5);
        // labels (all with a halo)
        if (narrow) {
          plot.text(ctx, 'one well only', box.x0 + 5, box.y1 - 20, { color: c.ink2, size: 10, c: c, halo: true, haloColor: bandHalo });
          plot.text(ctx, '(with this field)', box.x0 + 5, box.y1 - 8, { color: c.ink2, size: 10, c: c, halo: true, haloColor: bandHalo });
        } else plot.text(ctx, 'one well only (with this field)', box.x0 + 5, box.y1 - 8, { color: c.ink2, size: 10, c: c, halo: true, haloColor: bandHalo });
        plot.text(ctx, narrow ? 'β = 1' : 'transition β = 1', sx(1) + 4, box.y0 + 13, { color: c.ink3, size: 10, c: c, halo: true });
        plot.text(ctx, 'favoured well', box.x1 - 2, sy(B.top[B.top.length - 1][1]) + 16, { align: 'right', color: c.modeA, size: 10.5, weight: 600, c: c, halo: true });
        plot.text(ctx, narrow ? 'barrier top' : 'barrier top (unstable)', box.x1 - 2, sy(B.sad[B.sad.length - 1][1]) - 7, { align: 'right', color: c.ink3, size: 10, c: c, halo: true });
        // the chain
        ctx.beginPath(); ctx.arc(sx(stB.beta), sy(stB.m), 6, 0, 2 * Math.PI);
        ctx.fillStyle = stB.m < 0 ? c.modeB : c.modeA; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = c.surface; ctx.stroke();

        plot.axis(ctx, sx, { side: 'bottom', at: box.y1 + 8, ticks: [0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2], c: c, format: function (v) { return fmt(v, 1); } });
        plot.text(ctx, '← hotter', box.x0, h - 4, { color: c.ink3, size: 10.5, c: c });
        plot.text(ctx, 'inverse temperature β', (box.x0 + box.x1) / 2, h - 4, { align: 'center', color: c.ink2, size: 11, weight: 500, c: c });
        plot.text(ctx, 'colder →', box.x1, h - 4, { align: 'right', color: c.ink3, size: 10.5, c: c });
      }
      // drag on the branch diagram sets beta
      (function () {
        var cv = brCv.canvas, down = false;
        cv.style.cursor = 'ew-resize';
        function at(e) {
          if (!branchLayout) return;
          var p = brCv.local(e), b = branchLayout.sx.invert(p.x);
          setBeta(Math.max(BETA_LO, Math.min(BETA_HI, Math.round(b * 100) / 100)), true);
        }
        cv.addEventListener('pointerdown', function (e) { down = true; try { cv.setPointerCapture(e.pointerId); } catch (x) {} at(e); });
        cv.addEventListener('pointermove', function (e) { if (down) at(e); });
        cv.addEventListener('pointerup', function () { down = false; });
        cv.addEventListener('pointercancel', function () { down = false; });
      })();

      var sB = ui.slider({ id: ui.uid('rh-beta'), sym: 'β', label: 'Inverse temperature', min: BETA_LO, max: BETA_HI, value: 2.0, step: 0.01,
        format: function (v) { return fmt(v, 2); },
        marks: [{ value: BETA_RH, label: '0.7' }, { value: 1, label: '1' }, { value: 2, label: '2' }],
        onInput: function (v) { setBeta(v, true); } });
      var PLAY_LABEL = 'Play: reheat at β = 0.7, then cool';
      var btnPlay = ui.button({ label: PLAY_LABEL, icon: 'play', kind: 'primary', onClick: function () { onPlayButton(); } });
      function setPlayButton(icon, text) {
        var sp = btnPlay.querySelector('span'), sv = btnPlay.querySelector('svg');
        if (sp && sp.textContent !== text) sp.textContent = text;
        if (btnPlay.dataset.icon !== icon) { btnPlay.dataset.icon = icon; if (sv) sv.replaceWith(ui.icon(icon)); }
      }
      function onPlayButton() {
        if (!stB.play) { startPlay(); return; }
        if (stB.paused) { stB.paused = false; loopB.start(); } else { stB.paused = true; loopB.stop(); }
        updateB();
      }
      var btnReset = ui.button({ label: 'Back to the negative mode', icon: 'reset', onClick: function () { reset(); } });
      fb.controls.append(sB.el, el('div', { class: 've-reheat__btns' }, btnPlay, btnReset));
      // keep the play button at the width of its longest label, so the slider does not jump while it plays
      requestAnimationFrame(function () { if (btnPlay.offsetWidth) btnPlay.style.minWidth = btnPlay.offsetWidth + 'px'; });

      var rState = ui.readout({ label: 'Wells at this β', html: true, format: function (v) { return v; } });
      var rSpent = ui.readout({ label: 'Magnetization m', unit: '', format: function (v) { return v; } });
      var status = el('p', { class: 've-reheat__status', 'aria-live': 'polite' });
      fb.controls.after(el('div', { class: 've-reheat__rbwrap' }, el('div', { class: 've-readouts ve-reheat__rb' }, rState.el, rSpent.el), status));

      function wellsAt(b) { return stationary(b, H); }
      function updateB() {
        var s = wellsAt(stB.beta);
        var inWrong = s.length >= 3 && stB.m < s[1];
        rState.set(s.length >= 3 ? 'two' : 'one', s.length >= 3 ? 'a negative and a favoured well' : (stB.beta < 1 ? 'below the transition' : 'the negative well is gone'));
        rSpent.set(fmt(stB.m, 2), inWrong ? 'in the negative well' : (s.length >= 3 ? 'in the favoured well' : 'in the only well'));
        // play button state
        if (!stB.play) setPlayButton('play', PLAY_LABEL);
        else if (stB.paused) setPlayButton('play', 'Resume');
        else setPlayButton('pause', stB.play.phase === 'reheat' ? 'Pause · reheating' : 'Pause · cooling');
        var msg;
        var slid = stB.trail.some(function (p) { return p[1] < 0; }) && stB.m > 0;
        var P = stB.play;
        if (P && P.phase === 'reheat') msg = 'Reheating at β = 0.7, below the transition: the model has one well here, so the run can leave its mode.';
        else if (P) msg = 'Cooling again. In this noiseless illustration the field’s push picks the side; in a finite run, fluctuations compete with it (chapter 4).';
        else if (inWrong && stB.beta > 1.6) msg = 'The run is in the negative mode. Drag β to the left to heat.';
        else if (inWrong) msg = 'The negative well is getting shallower, but the run is still in it. Keep lowering β.';
        else if (stB.m < -0.05) msg = 'The negative well is gone. Nothing holds the run, and it slides to the other side.';
        else if (stB.beta < 1) msg = 'Below the transition β = 1 the model has one well; the field only shifts it (here m = ' + fmt(stB.m, 2) + '). Now drag β to the right to cool again.';
        else if (slid && stB.beta >= 1.9) msg = 'The run left the negative mode and cooled into the favoured one. In real runs fluctuations compete with the field, so not every run does this.';
        else if (slid) msg = 'The run has left the negative well. Drag β to the right to cool again.';
        else msg = 'The run is in the favoured mode.';
        status.textContent = msg;
      }
      function pushTrail() {
        var last = stB.trail[stB.trail.length - 1];
        if (!last || Math.abs(last[0] - stB.beta) > 0.002 || Math.abs(last[1] - stB.m) > 0.004) stB.trail.push([stB.beta, stB.m]);
        if (stB.trail.length > 4000) stB.trail.splice(0, stB.trail.length - 4000);
      }
      function noteGhost(prevB) {
        var sOld = wellsAt(prevB), sNew = wellsAt(stB.beta);
        if (sOld.length >= 3 && sNew.length < 3 && stB.m < 0) stB.ghost = stB.m;
        if (sNew.length >= 3 || stB.m > 0) stB.ghost = null;
      }
      function setBeta(b, user) {
        if (user && stB.play) stopPlay();
        var prev = stB.beta;
        stB.beta = b;
        if (user) sB.set(b, { silent: true });
        noteGhost(prev);
        if (ui.reducedMotion) { stB.m = stepDrift(stB.m, stB.beta, H, 300); pushTrail(); }
        else loopB.start();
        redrawB();
      }
      function redrawB() { landB.redraw(); brCv.redraw(); updateB(); }
      function reset() {
        stopPlay();
        stB.beta = 2.0; stB.m = stationary(2.0, H)[0]; stB.trail = [[2.0, stB.m]]; stB.ghost = null;
        sB.set(2.0, { silent: true }); redrawB();
      }
      // live drift: 25 sweeps per second of real time
      var SPS = 25;
      var loopB = ui.loop(function (dt) {
        var sw = dt * SPS;
        if (stB.play) advancePlay(sw);
        else {
          var before = stB.m;
          stB.m = stepDrift(stB.m, stB.beta, H, sw);
          if (stB.m > 0) stB.ghost = null;
          pushTrail();
          if (Math.abs(drift(stB.m, stB.beta, H)) < 1e-4 && Math.abs(stB.m - before) < 1e-5) loopB.stop();
        }
        redrawB();
      }, { el: fb.el });
      cleanups.push(function () { loopB.destroy(); });

      function startPlay() {
        reset();
        stB.play = { phase: 'reheat', left: 30 };
        var prevB = stB.beta;
        stB.beta = BETA_RH; sB.set(stB.beta, { silent: true });
        noteGhost(prevB); pushTrail();
        if (ui.reducedMotion) { while (stB.play) advancePlay(5); redrawB(); return; }
        loopB.start();
      }
      function stopPlay() { stB.play = null; stB.paused = false; }
      function advancePlay(sw) {
        var P = stB.play;
        while (sw > 1e-9 && P) {
          if (P.phase === 'reheat') {
            var d = Math.min(sw, P.left);
            stB.m = stepDrift(stB.m, stB.beta, H, d); P.left -= d; sw -= d; pushTrail();
            if (P.left <= 1e-9) P.phase = 'descend';
          } else {                                   // cool at an illustrative pace, 0.015 beta per sweep
            var d3 = Math.min(sw, 0.5);
            stB.beta = Math.min(BETA_HI, stB.beta + 0.015 * d3);
            stB.m = stepDrift(stB.m, stB.beta, H, d3); sw -= d3; pushTrail();
            sB.set(stB.beta, { silent: true });
            if (stB.beta >= BETA_HI - 1e-9) { stopPlay(); P = null; loopB.stop(); }
          }
        }
        if (stB.ghost != null && stB.m > 0) stB.ghost = null;
      }
      redrawB();

      fb.setCaption('<span class="ve-reheat__cap">Lowering β (heating) makes the <b class="ve-c-mode-b">negative well</b> shallower until it disappears, and the run slides toward the <b class="ve-c-mode-a">favoured mode</b>. ' +
        'Below the transition β&nbsp;=&nbsp;1 one well remains, so a run can leave its mode; raising β again forms the two modes anew.</span> ');
      fb.setTakeaway('Below β = 1 one well remains, so a run can leave its mode; the final cooling then chooses again.');
      /* one keyed row per item (the spec card splits at " · " and keys at ":"); the drift has a
         row of its own, so the formula never wraps after an operator */
      fb.setNote(['Illustration: large-N drift, integrated live',
        'Drift: dm/dt = −m + tanh(βm + h)',
        'Field: h = 0.05, exaggerated so the tilt is visible (the paper’s field is G = 0.0006125, which enters as h = βG)',
        'Not the paper’s reheating protocol: that is 30 sweeps at β = 0.7 in the Hopfield model',
        'Display only: the reheat’s length and the cooling pace'].join(' · '), 'live');
      }

      function plateC() {
      /* ========================== PLATE C: MNIST DBM ========================== */
      var fc = ui.figure({ plate: prefix + plateBase + sub + 'b', title: 'A trained network: a reheated “8” can become a “3”', className: 've-reheat__plate' });
      moreC.appendChild(fc.el);
      var NC = dbm.p3.length, NF = dbm.snap_sweep.length, SW = dbm.beta_per_sweep.length - 1;    // 24, 141, 700
      var RH0 = dbm.protocol.sweeps[0], RH1 = RH0 + dbm.protocol.sweeps[1];   // the reheating window, sweeps 300–400 (shaded in the illustration)
      function inReheat(sw) { return sw >= RH0 && sw <= RH1; }
      var tints = {};                       // one tinted 28 x 672 column per frame, made when first needed
      function tintCol(c, frame) {
        var key = c.ink + '|' + frame;
        if (tints[key]) return tints[key];
        var cv = document.createElement('canvas'); cv.width = 28; cv.height = sprite.height;
        var x = cv.getContext('2d');
        x.drawImage(sprite, frame * 28, 0, 28, sprite.height, 0, 0, 28, sprite.height);   // alpha = pixel on
        x.globalCompositeOperation = 'source-in'; x.fillStyle = c.ink; x.fillRect(0, 0, 28, sprite.height);
        return (tints[key] = cv);
      }
      function tile(ctx, c, chain, frame, X, Y, S) {
        ctx.fillStyle = c.surface2; ctx.fillRect(X, Y, S, S);
        ctx.imageSmoothingEnabled = false;
        ctx.drawImage(tintCol(c, frame), 0, chain * 28, 28, 28, X, Y, S, S);
      }
      function labelColor(c, lab) { return lab === 3 ? c.modeA : (lab === 8 ? c.modeB : c.ink3); }
      function frameAt(sweep) { return Math.max(0, Math.min(NF - 1, Math.round(sweep / dbm.protocol.snap_every))); }
      var stC = { sweep: 0 };

      // filmstrip: two chains at chosen moments
      var FILM_W = [0, 300, 305, 320, 350, 400, 420, 440, 480, 700], FILM_N = [0, 305, 350, 400, 425, 450, 700];
      function film(w) { return w < 520 ? FILM_N : FILM_W; }
      function filmS(w) { return Math.min(58, Math.floor((w - 40) / film(w).length) - 4); }
      function filmX0(w) { return Math.max(40, (w - film(w).length * (filmS(w) + 4)) / 2); }
      fc.stage.appendChild(el('p', { class: 've-reheat__sub', text: 'Two of the 24 runs at chosen sweeps (click a frame to jump there)' }));
      var filmCv = cv(fc.stage, {
        height: function (w) { return 2 * (filmS(w) + 4) + 44; },
        label: 'Two runs over time: an 8 dissolves into noise at beta 0.55 and a 3 forms during the biased cooling.',
        draw: function (ctx, w, h, c) {
          var plot = ui.plot, S = filmS(w), gap = 4;
          var x0 = filmX0(w), FILM = film(w), near = 0;
          FILM.forEach(function (sw, k) { if (Math.abs(sw - stC.sweep) < Math.abs(FILM[near] - stC.sweep)) near = k; });
          FILM.forEach(function (sw, k) {
            var f = frameAt(sw), X = x0 + k * (S + gap);
            for (var r = 0; r < 2; r++) tile(ctx, c, r, f, X, 2 + r * (S + gap), S);
            if (k === near && frameAt(sw) === frameAt(stC.sweep)) {
              ctx.strokeStyle = c.ink; ctx.lineWidth = 2; ctx.strokeRect(X - 1.5, 0.5, S + 3, 2 * (S + gap) + 1);
            }
            var b = dbm.beta_per_sweep[sw];
            var hot = inReheat(sw) && sw > RH0;
            plot.text(ctx, fmt(b, 2), X + S / 2, 2 * (S + gap) + 14, { align: 'center', size: S < 36 ? 9.5 : 10.5, color: hot ? c.ink : c.ink3, weight: hot ? 600 : 400, c: c });
            plot.text(ctx, String(sw), X + S / 2, 2 * (S + gap) + 28, { align: 'center', size: S < 36 ? 9 : 10, color: c.ink3, c: c });
          });
          plot.text(ctx, 'β', x0 - 6, 2 * (S + gap) + 14, { align: 'right', size: 10.5, color: c.ink2, c: c });
          plot.text(ctx, 'sweep', x0 - 6, 2 * (S + gap) + 28, { align: 'right', size: 9.5, color: c.ink3, c: c });
        } });
      filmCv.canvas.style.cursor = 'pointer';
      filmCv.canvas.addEventListener('click', function (e) {
        var w = filmCv.width, S = filmS(w), x0 = filmX0(w), FILM = film(w);
        var k = Math.floor((filmCv.local(e).x - x0) / (S + 4));
        if (k >= 0 && k < FILM.length) { player.pause(); player.seek(FILM[k]); }
      });

      var player = ui.player({ id: ui.uid('rh-pl'), duration: SW, speed: 70, label: 'Sweep',
        format: function (t) { return fmt(Math.round(t)); },
        chapters: [{ t: 0, label: 'settle as “8” at β = 2' }, { t: 300, label: 'reheat at β = 0.55' }, { t: 400, label: 'cool, biased toward “3”' }, { t: 600, label: 'settle at β = 2' }],
        onTime: function (t) { stC.sweep = t; redrawC(); } });
      fc.stage.appendChild(el('div', { class: 've-reheat__player' }, player.el));
      cleanups.push(function () { player.destroy(); });

      var cGrid = el('div', { class: 've-reheat__two ve-reheat__two--c' });
      fc.stage.appendChild(cGrid);
      var cLeft = el('div', { class: 've-reheat__cell' }), cRight = el('div', { class: 've-reheat__cell' });
      cGrid.append(cLeft, cRight);
      cLeft.appendChild(el('p', { class: 've-reheat__sub', text: 'All 24 runs at this sweep, with the classifier’s label' }));
      var gridCv = cv(cLeft, {
        height: function (w) { var L = gridLayout(w); return L.h; },
        label: 'All 24 runs at the current sweep with their classifier labels.',
        draw: function (ctx, w, h, c) {
          var L = gridLayout(w), f = frameAt(stC.sweep);
          for (var i = 0; i < NC; i++) {
            var r = Math.floor(i / L.cols), k = i % L.cols, X = L.x0 + k * (L.S + L.gx), Y = r * (L.S + L.gy);
            tile(ctx, c, i, f, X, Y, L.S);
            var lab = dbm.label_at_snap[i][f], col = labelColor(c, lab);
            ctx.fillStyle = col; ctx.fillRect(X, Y + L.S + 2, L.S, 3);
            ui.plot.text(ctx, '“' + lab + '”', X + L.S / 2, Y + L.S + 17, { align: 'center', size: 11, weight: 600, color: lab === 3 || lab === 8 ? col : c.ink3, c: c });
          }
        } });
      function gridLayout(w) {
        var cols = w >= 330 ? 6 : 4, gx = 6, gy = 26;
        var S = Math.min(56, Math.floor((w - (cols - 1) * gx) / cols));
        var rows = Math.ceil(NC / cols);
        return { cols: cols, S: S, gx: gx, gy: gy, x0: 0, h: rows * (S + gy) };
      }
      var countLine = el('p', { class: 've-reheat__count', 'aria-live': 'polite' });
      cLeft.appendChild(countLine);

      cRight.appendChild(el('p', { class: 've-reheat__sub', html: 'Classifier <span class="ve-c-mode-a">p(“3”)</span> − <span class="ve-c-mode-b">p(“8”)</span> for each run' }));
      var probeCv = cv(cRight, { aspect: 1.25, minHeight: 300, maxHeight: 380,
        label: 'Top: the inverse temperature beta over 700 sweeps. Bottom: classifier probability difference p(3) minus p(8) for 24 runs, bold line the median over the 24 runs at each sweep; shading marks the reheating window, sweeps 300 to 400 at beta 0.55.',
        drag: null,
        draw: function (ctx, w, h, c) { drawProbe(ctx, w, h, c); } });
      var probeLayout = null;
      var medianCache = null;
      function medianTrace() {
        if (medianCache) return medianCache;
        medianCache = dbm.p3[0].map(function (_, j) {
          var v = dbm.p3.map(function (p, i) { return p[j] - dbm.p8[i][j]; }).sort(function (x, y) { return x - y; });
          return (v[11] + v[12]) / 2;
        });
        return medianCache;
      }
      function drawProbe(ctx, w, h, c) {
        var plot = ui.plot, narrow = w < 420;
        var left = narrow ? 34 : 40;
        // two stacked panels on one sweep axis: the beta schedule (top strip) and the classifier vote (main)
        var bTop = 16, bBot = 74;                                   // beta strip
        var box = ui.inset(w, h, { top: bBot + 26, right: 10, bottom: 34, left: left });
        var sx = ui.scale.linear([0, SW], [box.x0, box.x1]);
        var sy = ui.scale.linear([-1, 1], [box.y1, box.y0]);
        var sb = ui.scale.linear([0.45, 2.1], [bBot, bTop]);
        probeLayout = { sx: sx };
        var lo = RH0, hi = RH1, s;
        ctx.fillStyle = c.surface2 || c.noiseWash;
        ctx.fillRect(sx(lo), bTop - 4, sx(hi) - sx(lo), bBot - bTop + 8);
        ctx.fillRect(sx(lo), box.y0, sx(hi) - sx(lo), box.y1 - box.y0);
        // beta strip
        plot.text(ctx, 'inverse temperature β', box.x0, 2, { size: 10.5, weight: 500, color: c.ink2, baseline: 'top', c: c });
        ctx.strokeStyle = c.rule; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(box.x0, bBot + 4.5); ctx.lineTo(box.x1, bBot + 4.5); ctx.stroke();
        var xs = [], ys = [];
        for (s = 0; s <= SW; s++) { xs.push(s); ys.push(dbm.beta_per_sweep[s]); }
        plot.line(ctx, xs, ys, sx, sb, { color: c.ink2, width: 1.75 });
        plot.text(ctx, '2', box.x0 - 6, sb(2), { align: 'right', baseline: 'middle', size: 10, color: c.ink3, c: c });
        plot.text(ctx, '0.55', box.x0 - 3, sb(0.55), { align: 'right', baseline: 'middle', size: 9.5, color: c.ink3, c: c });
        // classifier panel
        plot.text(ctx, 'classifier vote', box.x0, box.y0 - 18, { size: 10.5, weight: 500, color: c.ink2, baseline: 'top', c: c });
        plot.axis(ctx, sy, { side: 'left', at: box.x0, ticks: [-1, 0, 1], c: c, format: function (v) { return v === 0 ? '0' : (v > 0 ? '+1' : '−1'); }, grid: [box.x0, box.x1] });
        var step = dbm.probe_sweep_step, cur = stC.sweep, med = medianTrace();
        ['A', 'B'].forEach(function (side) {
          ctx.save();
          ctx.beginPath();
          if (side === 'A') ctx.rect(box.x0, box.y0 - 2, box.w, sy(0) - box.y0 + 2); else ctx.rect(box.x0, sy(0), box.w, box.y1 - sy(0) + 2);
          ctx.clip();
          for (var i = 0; i < NC; i++) {
            ctx.beginPath();
            var p3 = dbm.p3[i], p8 = dbm.p8[i];
            for (var j = 0; j < p3.length; j++) { var X = sx(j * step), Y = sy(p3[j] - p8[j]); if (j) ctx.lineTo(X, Y); else ctx.moveTo(X, Y); }
            ctx.strokeStyle = side === 'A' ? c.modeA : c.modeB; ctx.globalAlpha = 0.14; ctx.lineWidth = 1; ctx.stroke();
          }
          ctx.globalAlpha = 1;
          ctx.beginPath();
          for (j = 0; j < med.length; j++) { X = sx(j * step); Y = sy(med[j]); if (j) ctx.lineTo(X, Y); else ctx.moveTo(X, Y); }
          ctx.strokeStyle = side === 'A' ? c.modeA : c.modeB; ctx.lineWidth = 3; ctx.stroke();
          ctx.restore();
        });
        plot.text(ctx, 'votes “3”', box.x1 - 2, box.y0 + 14, { align: 'right', color: c.modeA, size: 10.5, weight: 600, halo: true, c: c });
        plot.text(ctx, 'votes “8”', box.x1 - 2, box.y1 - 8, { align: 'right', color: c.modeB, size: 10.5, weight: 600, halo: true, c: c });
        // cursor through both panels
        plot.vline(ctx, cur, sx, bTop - 4, box.y1, { color: c.ink, width: 1.5 });
        plot.axis(ctx, sx, { side: 'bottom', at: box.y1, ticks: [0, 100, 200, 300, 400, 500, 600, 700], c: c, format: function (v) { return narrow && v % 200 ? '' : String(v); } });
        plot.text(ctx, 'sweep', box.x1, h - 2, { align: 'right', color: c.ink2, size: 10.5, c: c });
        plot.text(ctx, 'shaded: reheating', box.x0, h - 2, { color: c.ink3, size: 10.5, c: c });
      }
      (function () {
        var cv = probeCv.canvas, down = false;
        cv.style.cursor = 'ew-resize';
        function at(e) { if (!probeLayout) return; var x = probeLayout.sx.invert(probeCv.local(e).x); player.pause(); player.seek(Math.max(0, Math.min(SW, x))); }
        cv.addEventListener('pointerdown', function (e) { down = true; try { cv.setPointerCapture(e.pointerId); } catch (x) {} at(e); });
        cv.addEventListener('pointermove', function (e) { if (down) at(e); });
        cv.addEventListener('pointerup', function () { down = false; });
        cv.addEventListener('pointercancel', function () { down = false; });
      })();

      // ten seeds
      var reps = dbm.control_arm.repair.map(function (r) { return r.end_3; });
      var fresh = dbm.control_arm.fresh.map(function (r) { return r.end_3; });
      var freshM = dbm.control_arm.fresh_matched.map(function (r) { return r.end_3; });
      var seedsWrap = el('div', { class: 've-reheat__seeds' });
      fc.stage.appendChild(seedsWrap);
      seedsWrap.appendChild(el('p', { class: 've-reheat__sub', html: 'Ten seeds of 24 runs each: how many end labelled “3”' }));
      defer(function () { cv(seedsWrap, { height: 160,
        label: 'Reheated runs: ' + Math.min.apply(null, reps) + ' to ' + Math.max.apply(null, reps) + ' of 24 end labelled 3 across ten seeds; fresh random pixels: ' + Math.min.apply(null, fresh) + ' to ' + Math.max.apply(null, fresh) + '; fresh pixels with a matched 700-sweep budget: ' + Math.min.apply(null, freshM) + ' to ' + Math.max.apply(null, freshM) + '.',
        draw: function (ctx, w, h, c) {
          var plot = ui.plot, narrow = w < 480;
          var lw = narrow ? 92 : 150;
          var sx = ui.scale.linear([19.5, 24.5], [lw + 16, w - 64]);
          var rows = [{ name: narrow ? 'reheated' : 'reheated runs', v: reps, col: c.data }, { name: narrow ? 'fresh pixels' : 'fresh random pixels', v: fresh, col: c.ink3 },
            { name: narrow ? 'fresh, 700' : 'fresh, 700-sweep budget', v: freshM, col: c.ink3 }];
          rows.forEach(function (row, r) {
            var y = 24 + r * 38;
            plot.text(ctx, row.name, lw, y + 6, { align: 'right', size: 11, color: c.ink2, weight: 500, c: c });
            var cnt = {};
            row.v.forEach(function (v) {
              var k = cnt[v] = (cnt[v] || 0) + 1;
              ctx.beginPath(); ctx.arc(sx(v), y + 6 - (k - 1) * 8, 3.5, 0, 2 * Math.PI);
              ctx.fillStyle = row.col; ctx.fill();
            });
            var mn = Math.min.apply(null, row.v), mx = Math.max.apply(null, row.v);
            plot.text(ctx, mn + '–' + mx, w - 2, y + 6, { align: 'right', size: 11.5, weight: 600, color: c.ink, c: c });
          });
          plot.axis(ctx, sx, { side: 'bottom', at: h - 26, ticks: [20, 21, 22, 23, 24], c: c, format: function (v) { return String(v); } });
          plot.text(ctx, 'of 24', w - 2, h - 10, { align: 'right', size: 10, color: c.ink3, c: c });
        } }); });

      function redrawC() {
        filmCv.redraw(); gridCv.redraw(); probeCv.redraw();
        var f = frameAt(stC.sweep), n3 = 0, n8 = 0;
        for (var i = 0; i < NC; i++) { var l = dbm.label_at_snap[i][f]; if (l === 3) n3++; else if (l === 8) n8++; }
        var b = dbm.beta_per_sweep[Math.round(stC.sweep)];
        countLine.innerHTML = 'Sweep ' + Math.round(stC.sweep) + ', β = ' + fmt(b, 2) + ': <b class="ve-c-mode-b">' + n8 + ' “8”</b> · <b class="ve-c-mode-a">' + n3 + ' “3”</b> · ' + (NC - n3 - n8) + '&nbsp;other';
      }
      redrawC();

      var nCommit8 = dbm.label_at_snap.filter(function (L) { return L[frameAt(299)] === 8; }).length;
      var nEnd3 = dbm.label_after.filter(function (l) { return l === 3; }).length;
      fc.setCaption('A <span data-term="dbmTwoLayer">deep Boltzmann machine</span> with two hidden layers of 500 units, trained on MNIST: 24 runs started from images of an “8” are reheated, then cooled under an added bias toward “3”. ' +
        'In this stored ensemble ' + nEnd3 + ' of 24 end labelled “3”; the dot plot compares ten seeds of reheated runs with two fresh-start controls. ' +
        how('The protocol, step by step', ['24 runs start from images of an “8” and run at β = 2 for 300 sweeps; at sweep 300, ' + nCommit8 + ' are labelled “8” and 2 “9”. ' +
        'Then they are reheated to β = 0.55 for 100 sweeps, and the digit dissolves into noise. ' +
        'They cool over 200 sweeps to β = 2 under an added bias toward “3”, then settle for 100 sweeps.', 'A linear classifier saved with the network labels each image; the thin lines are single runs, and the bold line is the median over the 24 runs at each sweep. ' +
        'Before the reheat, single runs wobble from sweep to sweep, but the median and the label grid show that ' + nCommit8 + ' of 24 hold an “8”. ' +
        'During the reheat the images are noise, so the classifier’s votes there carry little meaning. ' +
        'The two controls in the dot plot start from fresh random pixels under the same bias and the same ten seeds; one of them also gets a matched 700-sweep budget. ' +
        'Ten seeds of 24 runs detect only large differences, and the classifier’s outputs are not equilibrium digit probabilities.']));
      fc.setTakeaway('After reheating, the network’s runs chose again under the new bias; the illustration does not test the selection law.');
      fc.setNote('Repository data (docs/mnist-illustration.md), not in the paper · two hidden layers of 500 units · the pictured ensemble is stored from an earlier implementation with unrecorded update order and image-selection rule; the dot-plot counts come from separate runs (App. D)', 'paper');
      ui.terms(host);
      }
    }

    if (VE.ui.defineTerm && !VE.ui.glossary.dbmTwoLayer) {
      VE.ui.defineTerm('dbmTwoLayer', { word: 'Deep Boltzmann machine', ml: '<strong>An energy-based generative model with hidden layers.</strong> It is sampled by Gibbs updates, layer by layer, and β plays the same role of inverse temperature.' });
    }

    return { destroy: function () { cleanups.forEach(function (f) { f(); }); host.remove(); } };
  }

  // Glossary terms this chapter links, in the paper's wording. Registered here so the
  // chapter's popovers are correct whatever the state of the shared glossary (as speciation.js does).
  if (VE.ui && VE.ui.defineTerm) {
    VE.ui.defineTerm('reheating', { word: 'Reheating', ml: '<strong>Raising the temperature again</strong> (β below 1) so a run that settled in one mode can choose again (reselection). In the paper: 30 sweeps at β = 0.7, after the filtered start.' });
    VE.ui.defineTerm('sweep', { word: 'Sweep', ml: '<strong>N single-spin Gibbs updates:</strong> an expected sweep (Curie–Weiss: random times, one update per variable on average) or a permutation sweep (Hopfield: each variable once, in a fresh random order). Sweeps measure sampling time.' });
    VE.ui.defineTerm('barrier', { word: 'Energy barrier', ml: '<strong>The low-probability region between two modes.</strong> To switch modes at large β the sampler must cross it, which takes time that grows exponentially with N.' });
    VE.ui.defineTerm('trackingTime', { word: 'Relaxation time', sym: 'τ(β)', ml: '<strong>The inverse spectral gap:</strong> the time for the slowest deviation from equilibrium to shrink by a factor e. The schedule’s dwell at each temperature is ⌈6τ⌉ sweeps.' });
  }

  VE.components['reheat'] = { mount: mount };
})();
