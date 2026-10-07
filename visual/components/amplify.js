/* components/amplify.js — "Competing patterns, stronger votes"
 *
 * Teaches the Hopfield amplification A_p = E[max_p |Z|] / E|Z| (paper Sec. 4, App. C, Table 1).
 *   Step 1  live Hopfield (N = 256, one chain, no field): patterns are modes; the sample chooses
 *           one stored pattern or its negative.
 *   Step 2  near beta = 1 the p overlaps are APPROXIMATED by p independent Gaussian draws, and the
 *           largest in size is ASSUMED to win (paper Sec. 4); we keep only runs where pattern 1 wins.
 *   Step 3  a larger overlap is a stronger vote: log-odds of its sign = 2 mu a / sigma^2.
 *   Step 4  live Monte Carlo: kept |m1| vs all |m1|; running A_p estimate -> exact (VE.sim.ampP).
 *   Step 5  predicted c = c_CW x A_p next to measured c (data/table1.json, data/c_cw.json);
 *           the p = 2 check (two CW blocks whose log-odds add IF their signs are independent, App. C).
 *
 * Sources of every number:
 *   A_p exact                   AP_EXACT below = VE.sim.ampP(p), Simpson integral of 1 - (2Phi(x) - 1)^p
 *                               (App. C); hard-coded to keep the load fast, checked against ampP
 *                               in dev/amplify.html (console.error on a mismatch)
 *   size scan (p = 3)           data/size_scan.json (paper App. A)
 *   c_CW = 0.1198 (0.120)       data/table1.json "c_CW" (= data/c_cw.json 0.119808)
 *   c predicted, c measured, CI data/table1.json rows (paper Table 1), p <= 4
 *   1.063 one-pattern ratio     data/sampler_clock.json (paper Sec. 5: Hopfield p = 1 c / c_CW)
 *   live numbers                computed on the page (seeded RNG), labelled "live"
 */
(function () {
  'use strict';
  var VE = (window.VE = window.VE || {});
  VE.components = VE.components || {};

  var NH = 256, SIDE = 16;           // live Hopfield size (16 x 16 image)
  var RAMP = 90;                      // sweeps for the beta ramp 0.6 -> 1.6
  var B0 = 0.6, B1 = 1.6;
  var SPS = 34;                       // sweeps per second shown
  var HOLD = 1.8;                     // seconds to hold the committed frame
  var MU = 0.4;                       // exaggerated bias for step 3 (in units of sigma)
  var XMAX = 4;                       // histogram range |m1|/sigma
  var NBIN = 40;
  var ROWH = 40, TILE = 32;           // step 1 pattern rows
  // A_p = E[max_p |Z|] / E|Z|, p = 0..6 (VE.sim.ampP to 1e-10)
  var AP_EXACT = [NaN, 1, 1.4142135624, 1.6623792719, 1.8357642865, 1.9674947910, 2.0729769600];

  //  for the terms this chapter opens (same text as lib/ui.js).
  if (VE.ui && VE.ui.defineTerm) {
    VE.ui.defineTerm('amplification', { word: 'Amplification', sym: 'A_p', ml: '<strong>The predicted gain in c from competition among p patterns,</strong> under an independent-Gaussian approximation: A_p = E[max |Z|] / E|Z| (Eq. 6).' });
    if (!VE.ui.glossary.transition) {
      var TRANSITION = { word: 'Transition', sym: 'β = 1', ml: '<strong>Where the single mode at m = 0 splits into two.</strong> Near it, each run’s sign is chosen.' };
      VE.ui.defineTerm('transition', TRANSITION);
      VE.ui.defineTerm('speciation', TRANSITION);
    }
    VE.ui.defineTerm('bias', { word: 'Field', sym: 'G (h = βG)', ml: '<strong>The model’s preference</strong> for mode A: a small push on every spin toward +1, fixed during a cooling run.' });
  }

  function mount(root, opts) {
    opts = opts || {};
    var ui = VE.ui, S = VE.sim, el = ui.el, fmt = ui.fmt;
    var AP = AP_EXACT;
    var state = { p: Math.max(1, Math.min(6, opts.p || 3)) };
    // opts.measured === false: the assembled page keeps the measurements for its last step,
    // so this plate ends at the prediction
    var SHOW_MEAS = opts.measured !== false;
    var destroyed = false;
    var cleanups = [];

    /* ------------------------------------------------------------ frame */
    var fig = ui.figure({
      plate: opts.plate != null ? opts.plate : 7,
      title: 'Competing patterns, stronger votes',
      className: 've-amplify'
    });
    var leadHtml = 'A <span data-term="hopfield">Hopfield model</span> stores <i>p</i> patterns, and each pattern and its negative is a <span class="ve-amplify__nw"><span data-term="mode">mode</span>.</span> ' +
        'We track pattern 1: the model’s preference <span class="ve-amplify__nw">(the <span data-term="bias">field</span>)</span> tilts it toward +pattern 1 over its negative, −pattern 1. ' +
        'The sampler passes on only a share <i>c</i> of that preference (1 = all of it, 0 = a coin flip). With more patterns, the runs that end in ±pattern 1 are predicted to keep a larger share, by a factor <i>A<sub>p</sub></i> that follows from a Gaussian approximation.';
    // the figure is built detached and attached once at the end of mount, so the canvases
    // size and draw together in the next frame instead of forcing a layout each
    if (opts.intro != null) leadHtml = opts.intro;
    if (leadHtml) fig.stage.appendChild(el('p', { class: 've-amplify__lead', html: leadHtml }));

    var pSlider = ui.slider({
      id: ui.uid('amp-p'), label: 'stored patterns', sym: 'p', min: 1, max: 6, step: 1, value: state.p,
      format: function (v) { return String(Math.round(v)); },
      marks: [1, 2, 3, 4, 5, 6].map(function (v) { return { value: v, label: String(v) }; }),
      tone: 'mode-a',
      onInput: function (v) { v = Math.round(v); if (v !== state.p) setP(v); }
    });
    var pHint = el('p', { class: 've-amplify__phint' });
    var pWrap = el('div', { class: 've-amplify__pctl' }, pSlider.el, pHint);
    fig.stage.appendChild(pWrap);

    var steps = el('div', { class: 've-amplify__steps' });
    fig.stage.appendChild(steps);

    var PANEL = { 1: 'a', 2: 'b', 4: 'c', 5: 'd', 3: 'e' };   // on-screen order: a b c d, then e folded below
    /* Each panel shows a short title; its explanation lives in
       an info card behind the ⓘ next to the title. With "All details" on, the same node moves
       inline under the title, so nothing is duplicated and nothing is two clicks away. */
    var explains = [];
    function step(n, title, text, o) {
      o = o || {};
      var body = el('div', { class: 've-amplify__body' });
      var letter = PANEL[n];   // panels are lettered, so they never read as the page's numbered steps
      var titleEl = el('span', { class: 've-amplify__ht', html: o.short || title });
      var h = el('h4', { class: 've-amplify__h' }, el('span', { class: 've-amplify__n', text: letter }), titleEl);
      var textEl = el('p', { class: 've-amplify__text', html: text });
      var slot = el('div', { class: 've-amplify__xin' });
      var s = el('section', { class: 've-amplify__step', dataset: { step: String(n) }, 'aria-label': 'Panel ' + letter + ': ' + title.replace(/<[^>]+>/g, '') },
        h, slot, body);
      if (o.inline) slot.appendChild(textEl);
      else {
        var x = el('div', { class: 've-amplify__x' }, textEl);
        var btn = el('button', { type: 'button', class: 'v2-info ve-amplify__i', 'aria-label': 'Explain panel ' + letter, html: ui.INFO_SVG || 'i' });
        h.appendChild(btn);
        var card = ui.info(btn, { id: ui.uid('amp-x' + letter), title: 'Panel ' + letter + ' · ' + title, node: x, kind: 'wide', parent: fig.el });
        explains.push({ x: x, card: card, slot: slot });
      }
      steps.appendChild(s);
      return { el: s, body: body, h: h, x: x, textEl: textEl, titleEl: titleEl };
    }
    function placeExplains(on) {
      explains.forEach(function (e) {
        if (on) { e.card.close(); e.slot.appendChild(e.x); }
        else e.card.el.querySelector('.v2-card__body').appendChild(e.x);
        e.slot.hidden = !on;
      });
      window.dispatchEvent(new Event('resize'));
    }

    /* ------------------------------------------------- helpers: images */
    var tileCv = document.createElement('canvas');
    tileCv.width = SIDE; tileCv.height = SIDE;
    var tileCtx = tileCv.getContext('2d');
    var tileImg = tileCtx.createImageData(SIDE, SIDE);
    function rgb(col) {
      var c = ui.alpha(col, 1).match(/[\d.]+/g) || [0, 0, 0];
      return [+c[0], +c[1], +c[2]];
    }
    // draw N=256 spins (values +-1, times sign) as a 16x16 image, crisp
    function drawSpins(ctx, arr, off, sign, x, y, size, onCol, offCol) {
      var on = rgb(onCol), of = rgb(offCol), d = tileImg.data;
      for (var i = 0; i < NH; i++) {
        var v = arr[off + i] * sign > 0 ? on : of;
        d[4 * i] = v[0]; d[4 * i + 1] = v[1]; d[4 * i + 2] = v[2]; d[4 * i + 3] = 255;
      }
      tileCtx.putImageData(tileImg, 0, 0);
      ctx.save();
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(tileCv, Math.round(x), Math.round(y), Math.round(size), Math.round(size));
      ctx.restore();
    }
    function roundRect(ctx, x, y, w, h, r) {
      ctx.beginPath();
      ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
      ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
      ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r);
      ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y);
      ctx.closePath();
    }
    function patColor(c, mu) { return mu === 0 ? c.modeA : c.ink2; }
    function phi(x) { return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI); }

    /* ================================================== STEP 1: Hopfield */
    var s1 = step(1, 'Patterns are modes',
      'Each small tile is a stored pattern of 256 spins, the network’s ±1 units, drawn as 16&nbsp;×&nbsp;16 squares. A sampler run starts from random spins and cools step by step. ' +
      'The bar beside each tile is that pattern’s <span data-term="overlap">overlap</span> <i>m</i>: how much the sample looks like it, from −1 (its negative) to +1 (the pattern itself). ' +
      'Past the <span data-term="transition">transition</span> at β&nbsp;=&nbsp;1 one overlap grows to about ±0.9 and the rest stay small: the sample has become that pattern or its negative. ' +
      'With one pattern every run ends in pattern 1 or its negative; with more, a run counts only when pattern 1 won.');
    var H = null, hopT = 0, hopAcc = 0, hopHold = 0, hopRuns = 0;
    var hopStatus = el('p', { class: 've-amplify__status', 'aria-live': 'polite' });
    function hopImg(w) { return Math.max(64, Math.min(128, Math.floor((w - 40) / 2 / 16) * 16)); }
    var hopCanvas = ui.canvas(s1.body, {
      label: 'A live Hopfield sample next to the closest stored pattern, then one overlap bar per stored pattern',
      height: function (w) { return 20 + hopImg(w) + 44 + Math.max(1, state.p) * ROWH + 18 + 44; },
      draw: drawHop
    });
    var hopPlay = ui.button({ label: 'Pause', icon: 'pause', onClick: function () { hopRunning = !hopRunning; syncHopBtn(); kick(); } });
    var hopNew = ui.button({ label: 'New run', icon: 'reset', kind: 'ghost', onClick: function () { newHopRun(); hopCanvas.redraw(); kick(); } });
    s1.body.appendChild(hopStatus);
    s1.body.appendChild(el('div', { class: 've-amplify__btns' }, hopPlay, hopNew));
    var hopRunning = !ui.reducedMotion;
    function syncHopBtn() {
      hopPlay.replaceChildren(ui.icon(hopRunning ? 'pause' : 'play'), el('span', { text: hopRunning ? 'Pause' : 'Play' }));
      hopPlay.setAttribute('aria-label', (hopRunning ? 'Pause' : 'Play') + ' the Hopfield run');
    }
    syncHopBtn();

    function hopBeta() { return B0 + (B1 - B0) * Math.min(1, hopT / RAMP); }
    function buildHop() {
      H = new S.Hopfield(NH, state.p, { seed: 4100 + 17 * state.p });
      hopRuns = 0;
      newHopRun();
    }
    function newHopRun() {
      hopRuns++;
      H.randomize();
      hopT = 0; hopAcc = 0; hopHold = 0;
    }
    function hopFinishInstantly() { while (hopT < RAMP) { H.sweep(hopBeta(), 0, 1); hopT++; } }
    function hopAdvance(dt) {
      if (hopT >= RAMP) {
        hopHold += dt;
        if (hopHold > HOLD) newHopRun();
        return;
      }
      hopAcc += dt * SPS;
      while (hopAcc >= 1 && hopT < RAMP) { H.sweep(hopBeta(), 0, 1); hopT++; hopAcc -= 1; }
    }
    function hopInfo() {
      var m = H.overlaps(0), b = H.basin(0);
      return { m: m, b: b, sign: m[b] >= 0 ? 1 : -1, done: hopT >= RAMP, beta: hopBeta() };
    }
    function patName(sign, b) { return (sign < 0 ? '−' : '+') + 'pattern ' + (b + 1); }
    function updateHopStatus(info) {
      // A short state line; the full sentences are in panel a's card
      var txt;
      if (info.done) {
        txt = 'Chose <b class="' + (info.b === 0 ? 've-c-mode-a' : '') + '">' + patName(info.sign, info.b) + '</b>' +
          (state.p === 1 ? '' : info.b === 0 ? ' · counted' : ' · not counted');
      } else if (info.beta < 1) {
        txt = 'Undecided: β below 1';
      } else {
        txt = 'Past β = 1';
      }
      if (hopStatus._t !== txt) { hopStatus.innerHTML = txt; hopStatus._t = txt; }
    }
    function drawHop(ctx, w, h, c) {
      if (!H) return;
      var info = hopInfo();
      updateHopStatus(info);
      var p = state.p;
      var img = hopImg(w), top = 20, gx = img + 32;
      var lit = info.done || info.beta > 1.02;
      // the sample and the closest stored pattern, same size
      ui.plot.text(ctx, 'the sample', 0, 12, { c: c, size: 11, weight: 500, color: c.ink3 });
      ui.plot.text(ctx, info.done ? 'the pattern it became' : lit ? 'closest stored pattern' : 'closest so far (undecided)', gx, 12, { c: c, size: 11, weight: 500, color: c.ink3 });
      ctx.fillStyle = c.rule; ctx.fillRect(0, top - 1, img + 2, img + 2);
      drawSpins(ctx, H.s, 0, 1, 1, top, img, c.ink, c.surface2);
      ui.plot.text(ctx, '≈', img + 16, top + img / 2 + 6, { c: c, size: 18, color: c.ink3, align: 'center' });
      ctx.save();
      ctx.globalAlpha = info.done ? 1 : lit ? 0.7 : 0.18;
      ctx.fillStyle = c.rule; ctx.fillRect(gx - 1, top - 1, img + 2, img + 2);
      drawSpins(ctx, H.xi, info.b * NH, info.sign, gx, top, img, info.b === 0 ? c.modeA : c.ink, c.surface2);
      ctx.restore();
      ui.plot.text(ctx, patName(info.sign, info.b), gx, top + img + 16,
        { c: c, size: 11, weight: lit ? 600 : 400, color: !lit ? c.ink3 : info.b === 0 ? c.modeA : c.ink2 });
      // one row per stored pattern
      var rt = top + img + 44;
      var bx0 = TILE + 16, bx1 = w - 46;
      var bmid = (bx0 + bx1) / 2, bhalf = (bx1 - bx0) / 2;
      ui.plot.text(ctx, 'stored', 0, rt - 8, { c: c, size: 11, weight: 500, color: c.ink3 });
      ui.plot.text(ctx, 'overlap m with the sample', bmid, rt - 8, { c: c, size: 11, weight: 500, color: c.ink3, align: 'center' });
      for (var mu = 0; mu < p; mu++) {
        var y = rt + mu * ROWH;
        var lead = mu === info.b && lit;
        if (lead) {
          ctx.fillStyle = info.done ? ui.alpha(c.ink3, 0.16) : ui.alpha(c.ink3, 0.08);
          ctx.fillRect(TILE + 6, y, w - TILE - 6, ROWH - 4);
        }
        drawSpins(ctx, H.xi, mu * NH, 1, 0, y + (ROWH - 4 - TILE) / 2, TILE, patColor(c, mu), c.surface2);
        var cy = y + (ROWH - 4) / 2;
        ui.plot.text(ctx, String(mu + 1), bx0 - 4, cy + 4, { c: c, size: 11, color: mu === 0 ? c.modeA : c.ink3, align: 'right', weight: 600 });
        ctx.fillStyle = c.surface2; ctx.fillRect(bx0 + 4, cy - 1, bx1 - bx0 - 4, 2);
        ctx.fillStyle = c.ruleStrong; ctx.fillRect(Math.round(bmid), cy - 8, 1, 16);
        var v = info.m[mu];
        ctx.fillStyle = patColor(c, mu);
        var bw = v * (bhalf - 2);
        ctx.fillRect(Math.min(bmid, bmid + bw), cy - 5, Math.abs(bw), 10);
        ui.plot.text(ctx, (v < 0 ? '−' : '+') + fmt(Math.abs(v), 2), w, cy + 4, { c: c, size: 11, align: 'right', color: mu === 0 ? c.modeA : c.ink2, weight: lead ? 600 : 400 });
      }
      var ly = rt + p * ROWH + 8;
      ui.plot.text(ctx, '−1', bx0 + 4, ly, { c: c, size: 11, color: c.ink3, align: 'center' });
      ui.plot.text(ctx, '0', bmid, ly, { c: c, size: 11, color: c.ink3, align: 'center' });
      ui.plot.text(ctx, '+1', bx1, ly, { c: c, size: 11, color: c.ink3, align: 'center' });
      // inverse-temperature progress (a plain progress line, not a control)
      var trackY = h - 20, tx0 = 0, tx1 = w;
      var sx = ui.scale.linear([B0, B1], [tx0, tx1]);
      ctx.fillStyle = c.surface2; ctx.fillRect(tx0, trackY - 1, tx1 - tx0, 2);
      ctx.fillStyle = c.ink2; ctx.fillRect(tx0, trackY - 1, sx(info.beta) - tx0, 2);
      ctx.fillStyle = c.ruleStrong; ctx.fillRect(Math.round(sx(1)), trackY - 6, 1, 12);
      ui.plot.text(ctx, 'β = 1', sx(1), trackY - 10, { c: c, size: 11, color: c.ink3, align: 'center' });
      ui.plot.text(ctx, 'β 0.6: hot', tx0, trackY + 15, { c: c, size: 11, color: c.ink3 });
      ui.plot.text(ctx, '1.6: cold', tx1, trackY + 15, { c: c, size: 11, color: c.ink3, align: 'right' });
    }

    /* =============================================== STEP 2: one draw */
    var s2 = step(2, 'Near β = 1: assume the largest overlap wins',
      'Just past β&nbsp;=&nbsp;1 the <i>p</i> overlaps <i>m</i><sub>1</sub> … <i>m<sub>p</sub></i> are approximated by <i>p</i> independent Gaussian draws with a common spread σ (an assumption: small <i>p</i>/<i>N</i> does not guarantee it). ' +
      'Write <i>Z</i> = <i>m</i>/σ for an overlap in units of σ. The paper assumes that whichever overlap is largest in size, regardless of sign, grows into the final pattern. ' +
      'The field acts on pattern 1, so we measure only the runs that end in pattern 1 or its negative: the runs where <b class="ve-c-mode-a">pattern 1</b> wins, about 1 in <i>p</i>. We call these the kept runs. ' +
      'Each step draws one run; the counter compares the kept share with 1/<i>p</i>.', { short: 'Assume the largest overlap wins' });
    var mc = null;
    var anim = 1;
    var drawCanvas = ui.canvas(s2.body, {
      label: 'The latest draw: one Gaussian overlap per pattern; the largest in size is highlighted',
      height: function () { return 24 + state.p * 30 + 44; },
      draw: drawDraw
    });
    var drawStatus = el('p', { class: 've-amplify__status', 'aria-live': 'polite' });
    var mcPlay = ui.button({ label: 'Pause', icon: 'pause', onClick: function () { mcRunning = !mcRunning; syncMcBtn(); kick(); } });
    var mcOne = ui.button({ label: 'Draw 1', icon: 'step', onClick: function () { mcDraw(1, true); renderMC(); kick(); } });
    var mcMany = ui.button({ label: 'Draw 1,000', kind: 'primary', onClick: function () { mcDraw(1000, true); renderMC(); kick(); } });
    var mcReset = ui.button({ icon: 'reset', kind: 'icon', title: 'Reset the draws', ariaLabel: 'Reset the draws', onClick: function () { resetMC(0); renderMC(); } });
    s2.body.appendChild(drawStatus);
    s2.body.appendChild(el('div', { class: 've-amplify__btns' }, mcPlay, mcOne, mcMany, mcReset));
    var mcRunning = !ui.reducedMotion, mcClock = 0;
    function syncMcBtn() { mcPlay.replaceChildren(ui.icon(mcRunning ? 'pause' : 'play'), el('span', { text: mcRunning ? 'Pause' : 'Play' })); mcPlay.setAttribute('aria-label', (mcRunning ? 'Pause' : 'Play') + ' the draws'); }
    syncMcBtn();

    function ease() { return anim < 1 ? 1 - Math.pow(1 - anim, 3) : 1; }
    function resetMC(prefill) {
      // at p >= 5 the kept runs spread over fewer bins, so wider bins keep about 1,000 kept runs from showing seed noise as spikes
      var nb = state.p >= 5 ? 27 : NBIN;
      mc = {
        nb: nb, r: S.rng(900 + state.p), z: new Float64Array(state.p), zFrom: new Float64Array(state.p), win: 0, winFrom: 0,
        n: 0, nKept: 0, sumAll: 0, sumKept: 0,
        hAll: new Float64Array(nb), hKept: new Float64Array(nb),
        hist: [], nextRec: 1
      };
      if (prefill) {
        mcDraw(prefill, false);
        // show a clear example at rest: keep drawing (all draws are counted) until pattern 1 wins with a visible lead
        var guard = 0;
        while (state.p > 1 && guard++ < 500 && !(mc.win === 0 && Math.abs(mc.z[0]) > 1.4)) mcDraw(1, false);
      }
      mc.zFrom.set(mc.z); mc.winFrom = mc.win;
      anim = 1;
    }
    function mcDraw(k, animate) {
      var p = state.p, r = mc.r, z = mc.z;
      if (animate) {
        // the bars move from what is on screen now to the new draw (no blank frame)
        var e = ease();
        for (var j0 = 0; j0 < p; j0++) mc.zFrom[j0] += (z[j0] - mc.zFrom[j0]) * e;
        if (e >= 0.5) mc.winFrom = mc.win;
      }
      for (var t = 0; t < k; t++) {
        var best = -1, arg = 0;
        for (var j = 0; j < p; j++) { z[j] = r.normal(); var a = Math.abs(z[j]); if (a > best) { best = a; arg = j; } }
        var a1 = Math.abs(z[0]);
        mc.n++; mc.sumAll += a1;
        var bi = Math.min(mc.nb - 1, Math.floor(a1 / XMAX * mc.nb));
        mc.hAll[bi]++;
        if (arg === 0) { mc.nKept++; mc.sumKept += a1; mc.hKept[bi]++; }
        mc.win = arg;
        if (mc.n >= mc.nextRec && mc.nKept >= 10) {
          mc.hist.push([mc.n, (mc.sumKept / mc.nKept) / (mc.sumAll / mc.n)]);
          mc.nextRec = mc.n < 40 ? mc.n + 1 : Math.ceil(mc.n * 1.04);
        }
      }
      if (animate) { if (ui.reducedMotion) { mc.zFrom.set(z); mc.winFrom = mc.win; anim = 1; } else anim = 0; }
      else if (anim >= 1) { mc.zFrom.set(z); mc.winFrom = mc.win; }
    }
    function drawDraw(ctx, w, h, c) {
      if (!mc) return;
      var p = state.p, z = mc.z, win = mc.win;
      var lx = 74, top = 24, rowH = 30;
      var sx = ui.scale.linear([-3.5, 3.5], [lx + 4, w - 8]);
      var e = ease();
      var vs = [], zmax = 0;
      for (var j = 0; j < p; j++) { vs[j] = mc.zFrom[j] + (z[j] - mc.zFrom[j]) * e; zmax = Math.max(zmax, Math.abs(vs[j])); }
      var yb0 = top - 4, yb1 = top + p * rowH;
      // band of sizes below the winner's (only meaningful with competitors)
      if (p > 1) {
        ctx.fillStyle = ui.alpha(c.noise, 0.08);
        ctx.fillRect(sx(-zmax), yb0, sx(zmax) - sx(-zmax), yb1 - yb0);
        [-zmax, zmax].forEach(function (v) {
          ctx.save(); ctx.setLineDash([3, 3]); ctx.strokeStyle = c.ink2; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(Math.round(sx(v)) + 0.5, yb0); ctx.lineTo(Math.round(sx(v)) + 0.5, yb1); ctx.stroke(); ctx.restore();
        });
        var lxr = sx(zmax), lab = '± largest size';
        ctx.font = ui.font('mono', 11, 500);
        var tw = ctx.measureText(lab).width;
        var tx = Math.min(w - tw, Math.max(sx(0) + 2, lxr - tw / 2));
        ui.plot.text(ctx, lab, tx, yb0 - 6, { c: c, size: 11, weight: 500, color: c.ink2 });
      }
      for (var mu = 0; mu < p; mu++) {
        var y = top + mu * rowH, cy = y + rowH / 2 - 2;
        var wa = mu === (e >= 0.5 ? win : mc.winFrom) ? 1 : 0;   // one highlighted row at a time
        if (wa) { ctx.fillStyle = ui.alpha(c.ink3, 0.16); ctx.fillRect(0, y - 2, w, rowH - 2); }
        if (mu === 0 && p > 1) { ctx.fillStyle = c.modeA; ctx.fillRect(0, y - 2, 3, rowH - 2); }
        ui.plot.text(ctx, 'pattern ' + (mu + 1), 10, cy + 4, { c: c, size: 11, color: mu === 0 ? c.modeA : c.ink2, weight: wa > 0.5 ? 600 : 400 });
        var v = vs[mu];
        ctx.fillStyle = patColor(c, mu);
        ctx.fillRect(Math.min(sx(0), sx(v)), cy - 7, Math.abs(sx(v) - sx(0)), 14);
      }
      ctx.fillStyle = c.ruleStrong; ctx.fillRect(Math.round(sx(0)), yb0, 1, yb1 - yb0);
      ui.plot.axis(ctx, sx, { side: 'bottom', at: yb1 + 2, ticks: [-3, -2, -1, 0, 1, 2, 3], c: c, format: function (v) { return v < 0 ? '−' + (-v) : String(v); }, label: 'overlap Z = m / σ' });
    }
    function updateDrawStatus() {
      var p = state.p, txt;
      if (p === 1) txt = 'Pattern 1 always wins · kept';
      else txt = (mc.win === 0
        ? '<b class="ve-c-mode-a">Pattern 1</b> largest · kept'
        : '<b>Pattern ' + (mc.win + 1) + '</b> largest · not counted');
      txt += ' <span class="ve-amplify__count">kept ' + fmt(mc.nKept, { group: true }) + ' / ' + fmt(mc.n, { group: true }) + ' · ' + ui.pct(mc.nKept / Math.max(1, mc.n)) + ' (1/' + p + ' = ' + ui.pct(1 / p) + ')</span>';
      if (drawStatus._t !== txt) { drawStatus.innerHTML = txt; drawStatus._t = txt; }
    }

    /* =============================================== STEP 3: the vote */
    var s3 = step(3, 'A larger overlap is a stronger vote',
      'The field shifts overlap 1 by a small amount μ (in units of σ) toward the favoured sign. Given only the size <i>a</i> = |<i>Z</i><sub>1</sub>| the overlap ended up with, ' +
      'its <b>vote</b> is how strongly that size predicts the sign the run ends with: the <span data-term="logOdds">log-odds</span> log P(+|<i>a</i>)/P(−|<i>a</i>). For a Gaussian this is exactly 2μ<i>a</i>: proportional to the size. Near 0 it is a coin flip; far out it is nearly sure.', { inline: true });
    var voteA = 1.5;
    var voteCanvas = ui.canvas(s3.body, {
      label: 'Density of overlap 1 shifted by the field, with its heights at plus and minus a, and the log-odds of the sign against a',
      height: function (w) { return w > 520 ? 220 : 350; },
      draw: drawVote
    });
    var aSlider = ui.slider({
      id: ui.uid('amp-a'), label: 'size of the overlap', sym: 'a', min: 0, max: 3, step: 0.05, value: voteA, unit: 'σ',
      format: function (v) { return fmt(v, 2); }, tone: 'bias',
      onInput: function (v) { voteA = v; voteCanvas.redraw(); }
    });
    s3.body.appendChild(aSlider.el);
    s3.body.appendChild(el('p', { class: 've-amplify__fine', text: 'μ is drawn at 0.4 so the shift is visible. The real field is far smaller; that only lowers the slope.' }));

    function drawVote(ctx, w, h, c) {
      var side = w > 520;
      var A = side ? { x: 0, y: 0, w: w * 0.58, h: h } : { x: 0, y: 0, w: w, h: h * 0.52 };
      var B = side ? { x: w * 0.58 + 24, y: 0, w: w * 0.42 - 24, h: h } : { x: 0, y: h * 0.52 + 10, w: w, h: h * 0.48 - 10 };
      var a = voteA;
      // panel A: density
      var bA = { x0: A.x + 8, x1: A.x + A.w - 8, y0: A.y + 34, y1: A.y + A.h - 34 };
      var sx = ui.scale.linear([-3.8, 3.8], [bA.x0, bA.x1]);
      var sy = ui.scale.linear([0, 0.5], [bA.y1, bA.y0]);
      ui.plot.text(ctx, side || w > 400 ? 'where overlap 1 lands' : 'overlap 1', bA.x0, A.y + 12, { c: c, size: 11, weight: 500, color: c.ink3 });
      ctx.save();
      [[-3.8, 0, c.modeBWash], [0, 3.8, c.modeAWash]].forEach(function (s) {
        ctx.beginPath(); ctx.moveTo(sx(s[0]), sy(0));
        for (var i = 0; i <= 60; i++) { var x = s[0] + (s[1] - s[0]) * i / 60; ctx.lineTo(sx(x), sy(phi(x - MU))); }
        ctx.lineTo(sx(s[1]), sy(0)); ctx.closePath(); ctx.fillStyle = s[2]; ctx.fill();
      });
      ctx.restore();
      ui.plot.fn(ctx, function (x) { return phi(x - MU); }, sx, sy, { color: c.ink, width: 1.6, n: 160 });
      ui.plot.axis(ctx, sx, { side: 'bottom', at: bA.y1, ticks: [-3, -2, -1, 0, 1, 2, 3], c: c, format: function (v) { return v < 0 ? '−' + (-v) : String(v); } });
      // bias arrow, above the curve's peak
      var ay = sy(phi(0)) - 12;
      ctx.fillStyle = c.ruleStrong; ctx.fillRect(Math.round(sx(0)), ay - 6, 1, bA.y1 - ay + 6);
      ui.plot.arrow(ctx, sx(0), ay, sx(MU), ay, { color: c.bias, width: 2, head: 6 });
      ui.plot.text(ctx, 'field μ', sx(MU) + 6, ay + 4, { c: c, size: 11, color: c.bias, weight: 600, halo: true });
      // stems at +-a
      var hp = phi(a - MU), hm = phi(-a - MU);
      [[a, hp, c.modeA], [-a, hm, c.modeB]].forEach(function (s) {
        ctx.strokeStyle = s[2]; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(sx(s[0]), sy(0)); ctx.lineTo(sx(s[0]), sy(s[1])); ctx.stroke();
        ui.plot.dots(ctx, [{ x: s[0], y: s[1] }], sx, sy, { r: 4.5, color: s[2] });
      });
      ui.plot.text(ctx, '+a', sx(a), bA.y1 + 28, { c: c, size: 11, color: c.modeA, align: 'center', weight: 600, halo: true });
      ui.plot.text(ctx, '−a', sx(-a), bA.y1 + 28, { c: c, size: 11, color: c.modeB, align: 'center', weight: 600, halo: true });
      ui.plot.text(ctx, 'favoured sign →', bA.x1, sy(0.30), { c: c, size: 11, color: c.modeA, align: 'right', weight: 600 });
      ui.plot.text(ctx, '← other sign', bA.x0, sy(0.30), { c: c, size: 11, color: c.modeB, weight: 600 });
      var ratio = hp / hm;
      ui.plot.text(ctx, 'height at +a : at −a = ' + fmt(ratio, ratio < 10 ? 2 : 1) + ' : 1', bA.x1, A.y + 12, { c: c, size: 11, weight: 600, color: c.ink, align: 'right' });

      // panel B: log-odds vs a; the readout sits above the plot so no line crosses it
      var bB = { x0: B.x + 34, x1: B.x + B.w - 8, y0: B.y + 46, y1: B.y + B.h - 34 };
      var sxa = ui.scale.linear([0, 3], [bB.x0, bB.x1]);
      var sya = ui.scale.linear([0, 2 * MU * 3], [bB.y1, bB.y0]);
      ui.plot.text(ctx, 'log-odds of the favoured sign', B.x, B.y + 12, { c: c, size: 11, weight: 500, color: c.ink3 });
      var lo = 2 * MU * a, odds = Math.exp(lo);
      ui.plot.text(ctx, '2μa = ' + fmt(lo, 2) + ', odds ' + fmt(odds, odds < 10 ? 2 : 1) + ' : 1', B.x, B.y + 30, { c: c, size: 11, weight: 600, color: c.bias });
      ui.plot.axis(ctx, sya, { side: 'left', at: bB.x0, ticks: bB.y1 - bB.y0 < 110 ? [0, 1.2, 2.4] : [0, 0.6, 1.2, 1.8, 2.4], c: c, format: function (v) { return fmt(v, 1); }, grid: [bB.x0, bB.x1] });
      ui.plot.axis(ctx, sxa, { side: 'bottom', at: bB.y1, ticks: [0, 1, 2, 3], c: c, label: 'size a' });
      ui.plot.fn(ctx, function (x) { return 2 * MU * x; }, sxa, sya, { color: c.bias, width: 2 });
      ctx.save(); ctx.setLineDash([3, 3]); ctx.strokeStyle = c.ink3; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(sxa(a), bB.y1); ctx.lineTo(sxa(a), sya(lo)); ctx.lineTo(bB.x0, sya(lo)); ctx.stroke(); ctx.restore();
      ui.plot.dots(ctx, [{ x: a, y: lo }], sxa, sya, { r: 5, color: c.bias });
    }

    /* ============================================ STEP 4: kept vs all */
    var s4 = step(4, 'Kept runs have larger overlaps, so they vote harder',
      'Pattern 1 wins only when its overlap is the largest of <i>p</i>, so <b class="ve-c-mode-a">kept</b> runs have a larger |<i>Z</i><sub>1</sub>| than <b class="ve-c-noise-text">all</b> runs. ' +
      'The field shifts overlap 1 by a small amount μ (in units of σ) toward +pattern 1, so each run’s vote, the log-odds of the sign it ends with, is 2μ × its size (bars below). The field is small, so a group’s <span data-term="logOdds">log-odds</span> equal its average vote to first order in μ, and the kept runs’ log-odds are <span class="ve-amplify__nw"><i>A<sub>p</sub></i> = E[max of <i>p</i> |<i>Z</i>|] / E|<i>Z</i>|</span> times those of all runs.', { short: 'Kept runs vote harder' });
    /* why size matters, in one picture: the bias moves every overlap by the same small amount, so the
       further a run's overlap sits from 0, the more surely its sign is the favoured one.
       P(favoured sign | size a) = 1 / (1 + exp(−2μa)), at the exaggerated μ = 0.4 used in panel e. */
    var VOTE_MU = 0.4;
    var voteMini = el('div', { class: 've-amplify__votes', role: 'img', 'aria-label': 'Three runs with overlap sizes 0.5, 1.5 and 2.5: the share of runs with the favoured sign rises from 60% to 77% to 88%.' },
      el('p', { class: 've-amplify__votes-h', html: 'Share of runs whose sign is the favoured one, by the size of the overlap (field exaggerated, μ&nbsp;=&nbsp;0.4, so the effect is visible; see panel e in the drawer below):' }));
    [0.5, 1.5, 2.5].forEach(function (a) {
      var q = 1 / (1 + Math.exp(-2 * VOTE_MU * a));
      voteMini.appendChild(el('div', { class: 've-amplify__vrow' },
        el('span', { class: 've-amplify__vlab', html: 'size |<i>Z</i>| = ' + fmt(a, 1) }),
        el('span', { class: 've-amplify__vbar' },
          el('i', { class: 've-amplify__vfav', style: { width: (100 * q) + '%' } }),
          el('i', { class: 've-amplify__vmid', 'aria-hidden': 'true' })),
        el('span', { class: 've-amplify__vval', text: Math.round(100 * q) + '%' })));
    });
    voteMini.appendChild(el('p', { class: 've-amplify__votes-f', html: 'The tick marks 50/50. A larger size moves the vote further from a tie.' }));
    // The three vote bars explain the title, so they sit in panel c's card with its paragraph
    (s4.x || s4.body).appendChild(voteMini);
    var histCanvas = ui.canvas(s4.body, {
      label: 'Histogram of the size of overlap 1 for all draws and for kept draws, with Gaussian-theory curves and both means',
      height: 238, draw: drawHist
    });
    var worked = el('p', { class: 've-amplify__worked', 'aria-live': 'polite' });
    (s4.x || s4.body).appendChild(worked);
    var histLegend = ui.legend([
      { role: 'noise', mark: 'area', label: 'all runs' },
      { role: 'mode-a', mark: 'square', label: 'kept runs' },
      { role: 'theory', mark: 'dash', label: 'Gaussian theory' }
    ]);
    s4.body.appendChild(histLegend);
    var estRead = ui.readout({ label: 'A_p estimate (live)', tone: 'data', format: function (v) { return fmt(v, 2); } });
    var exactRead = ui.readout({ label: 'A_p, Gaussian theory', tone: 'theory', format: function (v) { return fmt(v, 3); } });
    estRead.el.querySelector('.ve-readout__label').innerHTML = '<i>A<sub>p</sub></i> from the draws';
    exactRead.el.querySelector('.ve-readout__label').innerHTML = '<i>A<sub>p</sub></i>, Gaussian theory';
    s4.body.appendChild(el('div', { class: 've-readouts ve-amplify__reads' }, estRead.el, exactRead.el));
    /* the running estimate folds under the readouts, so the 1.66 / 1.662 pair leads at rest */
    var convFold = el('details', { class: 've-amplify__more ve-amplify__conv' },
      el('summary', { text: 'How the estimate settles as draws accumulate' }));
    var convHost = el('div');
    convFold.appendChild(convHost);
    s4.body.appendChild(convFold);
    convFold.addEventListener('toggle', function () { if (convFold.open) window.dispatchEvent(new Event('resize')); });
    var convCanvas = ui.canvas(convHost, {
      label: 'Running estimate of A_p against the number of draws, converging to the Gaussian-theory value',
      height: 132, draw: drawConv
    });

    function drawHist(ctx, w, h, c) {
      if (!mc) return;
      var p = state.p;
      // a label band above the plot holds both mean labels; the data never reach it
      var box = ui.inset(w, h, { top: 48, right: 8, bottom: 36, left: 8 });
      var sx = ui.scale.linear([0, XMAX], [box.x0, box.x1]);
      var NB = mc.nb, bw = XMAX / NB;
      var fAll = function (x) { return 2 * phi(x); };
      var fKept = function (x) { var t = 1 - 2 * S.Phi(-x); return p * 2 * phi(x) * Math.pow(t, p - 1); };
      var ymax = 0;
      for (var i = 0; i <= 80; i++) { var x = XMAX * i / 80; ymax = Math.max(ymax, fAll(x), fKept(x)); }
      ymax *= 1.08;
      var sy = ui.scale.linear([0, ymax], [box.y1, box.y0]);
      var nA = Math.max(1, mc.n), nK = Math.max(1, mc.nKept);
      ctx.beginPath(); ctx.moveTo(sx(0), sy(0));
      for (var b = 0; b < NB; b++) { var d = mc.hAll[b] / (nA * bw); ctx.lineTo(sx(b * bw), sy(Math.min(d, ymax))); ctx.lineTo(sx((b + 1) * bw), sy(Math.min(d, ymax))); }
      ctx.lineTo(sx(XMAX), sy(0)); ctx.closePath();
      ctx.fillStyle = c.noiseWash; ctx.fill(); ctx.strokeStyle = c.noise; ctx.lineWidth = 1; ctx.stroke();
      if (mc.nKept > 0 && p > 1) {
        ctx.fillStyle = ui.alpha(c.modeA, ui.theme() === 'dark' ? 0.72 : 0.55);
        for (b = 0; b < NB; b++) {
          var dk = mc.hKept[b] / (nK * bw);
          var x0 = sx(b * bw) + 1, x1 = sx((b + 1) * bw) - 1;
          ctx.fillRect(x0, sy(Math.min(dk, ymax)), Math.max(1, x1 - x0), sy(0) - sy(Math.min(dk, ymax)));
        }
      }
      ui.plot.fn(ctx, fAll, sx, sy, { color: c.theory, width: 1.4, dash: [4, 3], n: 120 });
      if (p > 1) ui.plot.fn(ctx, fKept, sx, sy, { color: c.theory, width: 1.6, dash: [4, 3], n: 120 });
      ui.plot.axis(ctx, sx, { side: 'bottom', at: box.y1, ticks: [0, 1, 2, 3, 4], c: c, label: 'size of overlap 1, |Z₁| = |m₁| / σ' });
      // means: the all-runs label sits on the top row and reads leftward from its line;
      // the kept label sits on the second row and reads rightward, so neither crosses the other line
      var mAll = mc.sumAll / nA, mKept = mc.nKept ? mc.sumKept / nK : NaN;
      ctx.font = ui.font('mono', 11, 600);
      function meanMark(v, col, label, rowY, leftward) {
        if (!isFinite(v)) return;
        var X = Math.round(sx(v)), tw = ctx.measureText(label).width;
        ctx.fillStyle = col; ctx.fillRect(X - 1, rowY + 3, 2, box.y1 - rowY - 3);
        var tx = leftward ? Math.max(box.x0, X - 5 - tw) : Math.min(box.x1 - tw, X + 5);
        ui.plot.text(ctx, label, tx, rowY, { c: c, size: 11, weight: 600, color: col });
      }
      if (p === 1) meanMark(mAll, c.ink2, 'mean ' + fmt(mAll, 2), 14, false);
      else {
        meanMark(mAll, c.ink2, 'mean of all ' + fmt(mAll, 2), 14, true);
        meanMark(mKept, c.modeA, 'mean of kept ' + fmt(mKept, 2), 30, false);
      }
    }
    function updateWorked() {
      var p = state.p, mAll = mc.sumAll / Math.max(1, mc.n), mKept = mc.nKept ? mc.sumKept / mc.nKept : NaN;
      var txt;
      if (p === 1) txt = 'With one pattern every run is kept, so the two histograms are the same and <i>A</i><sub>1</sub> = 1.';
      else txt = 'Average log-odds of kept runs ÷ average of all runs = <span class="ve-amplify__nw">(2μ × ' + fmt(mKept, 3) + ')</span> ÷ <span class="ve-amplify__nw">(2μ × ' + fmt(mAll, 3) +
        ')</span> = <b>' + fmt(r3(mKept) / r3(mAll), 2) + '</b>, close to <i>A</i><sub>' + p + '</sub> = ' + fmt(AP[p], 3) + '. In this Gaussian picture, all runs are what a one-pattern model gives (set <i>p</i> = 1), so this ratio is the gain over the one-pattern case.';
      if (worked._t !== txt) { worked.innerHTML = txt; worked._t = txt; }
    }
    function r3(v) { return Math.round(v * 1000) / 1000; }   // the ratio is computed from the means as shown
    function drawConv(ctx, w, h, c) {
      if (!mc) return;
      var ex = AP[state.p];
      var box = ui.inset(w, h, { top: 22, right: 26, bottom: 34, left: 44 });
      var nmax = Math.max(10000, mc.n * 1.3);
      // points start once 10 runs are kept; the x range starts at the first point, so no empty strip
      var pts = mc.hist.concat(mc.nKept >= 10 ? [[mc.n, curEst()]] : []).filter(function (q) { return isFinite(q[1]); });
      var n0 = pts.length ? Math.max(10, pts[0][0]) : 10;
      var sx = ui.scale.log([n0, nmax], [box.x0, box.x1]);
      var lo = ex - 0.3, hi = ex + 0.3;
      pts.forEach(function (q) { lo = Math.min(lo, q[1] - 0.05); hi = Math.max(hi, q[1] + 0.05); });
      var sy = ui.scale.linear([lo, hi], [box.y1, box.y0]);
      ui.plot.text(ctx, w > 520 ? 'estimate as draws accumulate (early values are noisy)' : 'estimate (early values are noisy)', box.x0 - 40, 12, { c: c, size: 11, weight: 500, color: c.ink3 });
      ctx.save(); ctx.setLineDash([5, 3]); ctx.strokeStyle = c.theory; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(w - 70, 8.5); ctx.lineTo(w - 50, 8.5); ctx.stroke(); ctx.restore();
      ui.plot.text(ctx, 'theory', w - 50, 12, { c: c, size: 11, weight: 600, color: c.theory });
      var dec = [10, 100, 1000, 10000, 100000, 1000000].filter(function (v) { return v >= n0 && v <= nmax; });
      if (!dec.length || dec[0] / n0 > 3) dec.unshift(n0);   // label the left end when no decade is near it
      ui.plot.axis(ctx, sx, { side: 'bottom', at: box.y1, ticks: dec, c: c, format: function (v) { return fmt(v, { group: true, sci: false }); }, label: 'draws' });
      // ticks follow the live range, so the labels cover the whole plot at every p; the exact value is the dashed line
      // at most 4 labels in the ~76 px plot height
      var yst = [0.1, 0.2, 0.25, 0.5, 1].filter(function (st) { return Math.floor(hi / st) - Math.ceil(lo / st) + 1 <= 4; })[0] || 1, yt = [];
      for (var yv = Math.ceil(lo / yst - 1e-9) * yst; yv <= hi + 1e-9; yv += yst) yt.push(+yv.toFixed(4));
      var ydig = yst === 0.25 ? 2 : 1;
      ui.plot.axis(ctx, sy, { side: 'left', at: box.x0, ticks: yt, c: c, format: function (v) { return fmt(v, ydig); }, grid: [box.x0, box.x1] });
      ui.plot.hline(ctx, ex, sy, box.x0, box.x1, { color: c.theory, dash: [5, 3], width: 1.5 });
      if (pts.length) {
        ctx.save();
        ctx.lineWidth = 1.5; ctx.strokeStyle = c.data;
        ctx.beginPath();
        pts.forEach(function (q, k) { if (k) ctx.lineTo(sx(q[0]), sy(q[1])); else ctx.moveTo(sx(q[0]), sy(q[1])); });
        ctx.stroke();
        ctx.restore();
        var last = pts[pts.length - 1];
        ui.plot.dots(ctx, [{ x: last[0], y: last[1] }], sx, sy, { r: 3.5, color: c.data });
      }
    }
    function curEst() { return mc.nKept ? (mc.sumKept / mc.nKept) / (mc.sumAll / mc.n) : NaN; }

    /* ======================================== STEP 5: the prediction */
    var s5 = step(5, 'The prediction: <i>c</i> = <i>c</i><sub>CW</sub> × <i>A<sub>p</sub></i>', '', { short: 'The prediction' });
    s5.textEl.replaceWith(el('ol', { class: 've-amplify__text ve-amplify__list', html:
      '<li><i>c</i> = the kept runs’ <span data-term="logOdds">log-odds</span> of +pattern 1 over −pattern 1, divided by the model’s own log-odds for the same pair.</li>' +
      '<li>At <i>p</i>&nbsp;=&nbsp;1 the prediction is <i>c</i><sub>CW</sub> = 0.120 (CW for Curie–Weiss, the two-mode model): the paper’s numerical value on its protocol at <i>N</i>&nbsp;=&nbsp;2,048 spins' + (SHOW_MEAS ? '; one-pattern Hopfield runs measure 1.063 times this (below)' : '; one-pattern Hopfield runs measure 1.063 times this (chapter 7)') + '. It is small because the field has only a short interval near the transition to push before the sign is chosen' + (SHOW_MEAS ? '' : ' (chapter 4)') + '.</li>' +
      '<li>Under the Gaussian approximation, kept runs vote <i>A<sub>p</sub></i> times harder, so, to first order in the small field, the predicted <span class="ve-amplify__nw"><i>c</i> = <i>c</i><sub>CW</sub> × <i>A<sub>p</sub></i></span>; nothing is fitted to the network runs. The prediction also takes the Hopfield model’s own preference to be the Curie–Weiss one, an approximation.</li>' }));
    s5.el.classList.add('ve-amplify__step--wide');
    var eqLine = el('div', { class: 've-amplify__eq', 'aria-live': 'polite' });
    var table = el('table', { class: 've-amplify__table' });
    var tableNote = el('p', { class: 've-amplify__fine' });
    // No disclosure inside a disclosure; the "where the prediction misses" text is a plain block of the card
    var tableMore = el('div', { class: 've-amplify__more' });
    var p2box = el('div', { class: 've-amplify__p2' });
    var predCol = el('div', { class: 've-amplify__pred' }, eqLine);
    s5.body.appendChild(predCol);
    // the table of every p, its note and the misses go to panel d's card, under the three-line argument
    var tblHost = s5.x || predCol;
    tblHost.appendChild(el('div', { class: 've-amplify__tablewrap' }, table));
    tblHost.appendChild(tableNote);
    tblHost.appendChild(tableMore);

    function renderP2() {
      var r2 = data.t1 && data.t1.rows.filter(function (r) { return r.p === 2; })[0];
      p2box.innerHTML =
        '<h5 class="ve-amplify__h5">A check at <i>p</i> = 2: the network splits exactly in two</h5>' +
        '<p>Spins where patterns 1 and 2 agree form half A; the rest form half B. Each half acts as its own two-mode model and picks a sign: + means the half matches pattern 1, − means it matches the negative of pattern 1. Pattern 2 agrees with pattern 1 on half A and is its negative on half B, so mixed signs give ±pattern 2.</p>' +
        '<div class="ve-amplify__grid2" role="table" aria-label="Outcome for each pair of signs of the two halves">' +
        '<span></span><span class="ve-amplify__gh" role="columnheader">half B +</span><span class="ve-amplify__gh" role="columnheader">half B −</span>' +
        '<span class="ve-amplify__gh" role="rowheader">half A +</span><span class="ve-amplify__cell is-kept is-fav">+pattern 1</span><span class="ve-amplify__cell">+pattern 2</span>' +
        '<span class="ve-amplify__gh" role="rowheader">half A −</span><span class="ve-amplify__cell">−pattern 2</span><span class="ve-amplify__cell is-kept is-oth">−pattern 1</span>' +
        '</div>' +
        '<p>Pattern 1 wins only when both halves agree, so, if the halves choose independently, their votes add. Each half holds about half the spins, and a vote grows like the square root of the spin count, so each half’s vote is about 1/√2 of a full one. Two of them give 2 × 1/√2 = √2: for equal, independent halves the amplification tends to √2 = ' + fmt(AP[2], 3) + ' at large <i>N</i>.' +
        (r2 ? ' With the large-<i>N</i> factor √2, the prediction at <i>N</i>&nbsp;=&nbsp;2,048 is <i>c</i>&nbsp;=&nbsp;' + fmt(data.t1.c_CW, 3) + '&nbsp;×&nbsp;' + fmt(AP[2], 3) + '&nbsp;=&nbsp;' + fmt(r2.c_predicted, 3) + ' (with the unrounded <i>c</i><sub>CW</sub>&nbsp;=&nbsp;' + fmt(data.t1.c_CW, 4) + '); the halves hold 980–1,068 spins, not exactly <i>N</i>/2, and the paper’s protocol can make the halves’ signs dependent (below), so this is a check, not a proof' + (SHOW_MEAS ? '. Measured <i>c</i> = ' + fmt(r2.c_measured, 3) + ' (' + Math.round(100 * (r2.c_measured / r2.c_predicted - 1)) + '% above).' : '.') : '') + '</p>' +
        '<h5 class="ve-amplify__h6">Why the halves do not interact, and why the square root</h5>' +
        '<p>The interaction coefficient between two spins is a sum over patterns. Between a spin where the patterns agree and one where they disagree, the two terms cancel exactly, so the energy has no interaction between the halves. Their signs can still be dependent through how the runs start: the paper’s protocol discards the runs that chose +pattern 1 early and reheats the rest (see <a href="#reheat">Where the runs start</a>), and that step can make the two halves’ signs dependent (App. C). ' +
        'The sampler’s log-odds grow as √<i>N</i> (the model’s grow as <i>N</i>), so a half of <i>N</i>/2 spins keeps 1/√2 of the full amount. The halves are only roughly equal in size, and that difference is not resolved here.</p>';
    }

    var data = { t1: null, clock: null, scan: null };
    function renderTable() {
      var p = state.p, t1 = data.t1;
      var ccw = t1 ? t1.c_CW : NaN;
      var rows = '<thead><tr><th scope="col"><i>p</i></th><th scope="col"><i>A<sub>p</sub></i> (Gaussian theory)</th><th scope="col"><i>c</i> predicted</th>' + (SHOW_MEAS ? '<th scope="col"><i>c</i> measured [95%]</th>' : '') + '</tr></thead><tbody>';
      for (var k = 1; k <= 6; k++) {
        var row = t1 && t1.rows.filter(function (r) { return r.p === k; })[0];
        var pred = row ? row.c_predicted : ccw * AP[k];
        var meas = row ? fmt(row.c_measured, 3) + ' <span class="ve-amplify__ci">[' + fmt(row.ci95[0], 3) + ', ' + fmt(row.ci95[1], 3) + ']</span>' : '<span class="ve-amplify__na">not run</span>';
        rows += '<tr' + (k === p ? ' class="is-cur" aria-current="true"' : '') + '><td>' + k + '</td><td class="ve-c-theory">' + fmt(AP[k], 3) + '</td><td class="ve-c-theory">' + fmt(pred, 3) + '</td>' + (SHOW_MEAS ? '<td>' + meas + '</td>' : '') + '</tr>';
      }
      table.innerHTML = rows + '</tbody>';
      var row = t1 && t1.rows.filter(function (r) { return r.p === p; })[0];
      var predP = row ? row.c_predicted : ccw * AP[p];
      // Three instruments on one hairline: Inter label over a light serif numeral; c in gilt, the predicted c in the reticle
      eqLine.innerHTML =
        '<span class="ve-amplify__eqterm"><span class="ve-amplify__eqlab">one pattern, <i>c</i><sub>CW</sub></span><span class="ve-amplify__eqv">' + fmt(ccw, 3) + '</span></span>' +
        '<span class="ve-amplify__op" aria-label="times">×</span>' +
        '<span class="ve-amplify__eqterm"><span class="ve-amplify__eqlab">amplification, <i>A</i><sub>' + p + '</sub></span><span class="ve-amplify__eqv">' + fmt(AP[p], 3) + '</span></span>' +
        '<span class="ve-amplify__op" aria-label="equals">=</span>' +
        '<span class="ve-amplify__eqterm is-c ve-amplify__eqres"><span class="ve-amplify__eqlab">predicted <i>c</i></span><span class="ve-amplify__eqv">' + fmt(predP, 3) + '</span></span>' +
        (!SHOW_MEAS ? '' : row ? '<span class="ve-amplify__eqterm ve-amplify__eqmeas"><span class="ve-amplify__eqlab">measured (Table 1 run)</span><span class="ve-amplify__eqv">' + fmt(row.c_measured, 3) + ' <span class="ve-amplify__ci">[' + fmt(row.ci95[0], 3) + ', ' + fmt(row.ci95[1], 3) + ']</span></span></span>'
          : '<span class="ve-amplify__eqterm ve-amplify__eqmeas"><span class="ve-amplify__eqlab">measured</span><span class="ve-amplify__na">not run for p = ' + p + '</span></span>');
      var clk = data.clock ? data.clock.rows.filter(function (r) { return r.N === 2048; })[0] : null;
      var clkR = clk ? clk.r : 1.063, clkP3 = data.clock && data.clock.p3_clock_corrected_prediction_N2048 || 0.212;
      var r1 = t1 && t1.rows.filter(function (r) { return r.p === 1; })[0];
      var r2 = t1 && t1.rows.filter(function (r) { return r.p === 2; })[0];
      tableNote.innerHTML = 'Measured: Hopfield model, <i>N</i>&nbsp;=&nbsp;2,048, 95% intervals over pattern sets (paper Table 1). ' +
        '<i>p</i> = 3 and 4 agree within their intervals; <i>p</i> = 1 and 2 miss by 6% and 11%. ' +
        'Multiplying by ' + fmt(clkR, 3) + ' (the one-pattern ratio at <i>N</i>&nbsp;=&nbsp;2,048) moves the <i>p</i>&nbsp;=&nbsp;3 prediction to ' + fmt(clkP3, 3) + ', outside its interval, so errors in the approximations may cancel; ' +
        'at <i>p</i>&nbsp;=&nbsp;3 the neighbouring sizes <i>N</i>&nbsp;=&nbsp;1024 and 4096 miss by 6–7% (details below). ' +
        'The paper’s headline <i>p</i>&nbsp;=&nbsp;3 value, a separate run over eight field values and twenty pattern sets, is 0.199 [0.1908, 0.2095]. <i>p</i> = 5 and 6 are not measured.';
      var scan = data.scan ? data.scan.rows : [];
      var s1024 = scan.filter(function (r) { return r.N === 1024; })[0], s4096 = scan.filter(function (r) { return r.N === 4096; })[0];
      var more = '<h5 class="ve-amplify__h6">Where the prediction misses, and the one-pattern ratio</h5>';
      if (r1 && r2) {
        // r = c_measured(p = 1) / c_CW: a ratio of c values, not of time rates (data/sampler_clock.json)
        more += '<p>One-pattern Hopfield runs keep ' + fmt(clkR, 3) + ' times the Curie–Weiss log-odds at <i>N</i>&nbsp;=&nbsp;2,048 (paper Sec.&nbsp;5); the ratio includes both the update order and the interaction strength (1/<i>N</i> against 1/(<i>N</i>−1)).' +
          ' This factor is the size of the <i>p</i>&nbsp;=&nbsp;1 miss (' + fmt(r1.c_measured, 3) + ' against ' + fmt(ccw, 3) + '). Applied to <i>p</i>&nbsp;=&nbsp;2, it gives about ' + fmt(r2.c_predicted * clkR, 3) + ', still below the measured 95% interval [' + fmt(r2.ci95[0], 3) + ', ' + fmt(r2.ci95[1], 3) + '] (paper Table 1), so it explains only part of that miss.</p>';
      }
      if (s1024 && s4096) {
        more += '<p>Paper Table&nbsp;1: at <i>p</i> = 3, the neighbouring sizes miss by 6–7%: <i>N</i>&nbsp;=&nbsp;1024 measures ' + fmt(s1024.c_measured, 3) + ' against ' + fmt(s1024.c_predicted, 3) +
          ' predicted, and <i>N</i>&nbsp;=&nbsp;4096 measures ' + fmt(s4096.c_measured, 3) + ' against ' + fmt(s4096.c_predicted, 3) + '. After the one-pattern ratio at each size, 18% and 4% remain.</p>';
      }
      tableMore.innerHTML = more;
      tableMore.hidden = !((r1 && r2) || (s1024 && s4096));
      if (!SHOW_MEAS) {
        tableNote.innerHTML = 'Predictions only, at <i>N</i>&nbsp;=&nbsp;2,048: <i>c</i><sub>CW</sub> = ' + fmt(ccw, 3) + ' from the numerical Curie–Weiss evolution on the paper’s protocol (chapter 3), times <i>A<sub>p</sub></i> from the Gaussian approximation. ' +
          'No sampler run enters them. Chapter 7 sets them against the measurements.';
        tableMore.innerHTML = '';
        tableMore.hidden = true;
      }
      p2box.classList.toggle('is-cur', p === 2);
    }

    /* ------------------------------------------------- figure footer */
    fig.setTakeaway('If the largest of <i>p</i> overlaps wins, kept runs vote harder: predicted <span class="ve-amplify__nw"><i>c</i> = <i>c</i><sub>CW</sub> × <i>A<sub>p</sub></i></span>.');
    fig.setNote('Panel a: Hopfield model, N = 256 (paper: 2,048), β 0.6 → 1.6 over 90 sweeps, no field · Panels b, c: seeded independent Gaussian draws (the paper’s approximation) · Panel d: predicted c from paper Table 1 for p = 1–4; p = 5, 6 computed here by the same rule · Aₚ (Gaussian theory): numerical integration of E[max of p |Z|] / E|Z| (√2 at p = 2)', 'live');

    /* ------------------------------------------------------ p updates */
    function prefillN(p) { return p === 1 ? 2000 : Math.max(3000, 1000 * p); }   // about 1,000 kept runs at every p
    function setP(p) {
      state.p = p;
      pSlider.set(p, { silent: true });
      updatePHint();
      buildHop();
      if (!hopRunning) hopFinishInstantly();
      resetMC(prefillN(p));
      hopCanvas.resize(); drawCanvas.resize(); histCanvas.resize(); convCanvas.resize();
      renderMC();
      renderTable();
      kick();
    }
    function updatePHint() {
      var p = state.p;
      // One short line; "each pattern and its negative" is said in panel a's card
      pHint.innerHTML = (2 * p) + ' modes · the paper tests <i>p</i>&nbsp;=&nbsp;3';
    }
    function renderMC() {
      if (ease() >= 0.5) updateDrawStatus();   // the text switches with the row highlight, half-way through the tween
      updateWorked();
      histLegend.children[1].style.display = state.p === 1 ? 'none' : '';
      drawCanvas.redraw(); histCanvas.redraw(); convCanvas.redraw();
      estRead.set(curEst(), mc && mc.nKept ? 'kept mean ' + fmt(mc.sumKept / mc.nKept, 2) + ' ÷ all ' + fmt(mc.sumAll / mc.n, 2) : '');
      exactRead.set(AP[state.p], state.p === 2 ? '√2' : '');
    }

    /* --------------------------------------------------------- loop */
    var DRAW_EVERY = 0.9, ANIM_DUR = 0.35;
    var lp = ui.loop(function (dt) {
      if (destroyed) return;
      var any = false;
      if (hopRunning) { hopAdvance(dt); hopCanvas.redraw(); any = true; }
      if (mcRunning || anim < 1) {
        if (anim < 1) { var e0 = ease(); anim = Math.min(1, anim + dt / ANIM_DUR); drawCanvas.redraw(); if (e0 < 0.5 && ease() >= 0.5) updateDrawStatus(); if (anim >= 1) renderMC(); }
        if (mcRunning) {
          mcClock += dt;
          if (mcClock >= DRAW_EVERY) {
            mcClock = 0;
            mcDraw(1, true);
            // keep the aggregate moving faster than the one shown draw
            mcDraw(Math.min(60, 4 + Math.floor(mc.n / 400)), false);
            mcDraw(1, true);
            renderMC();
          }
        }
        any = true;
      }
      if (!any) lp.stop();
    }, { el: root });
    function kick() { if (hopRunning || mcRunning || anim < 1) lp.start(); }
    var offRM = ui.onReducedMotion(function (rm) { if (rm) { hopRunning = false; mcRunning = false; syncHopBtn(); syncMcBtn(); } });
    cleanups.push(offRM);

    /* --------------------------------------------------------- init */
    updatePHint();
    buildHop();
    // first frame shows a committed sample, then (with motion) a fresh run starts after the hold
    hopFinishInstantly();
    resetMC(0);
    renderMC();
    renderP2();
    renderTable();
    // fill the Monte Carlo after the first paint, so mounting stays short
    var prefillT = setTimeout(function () {
      if (destroyed) return;
      resetMC(prefillN(state.p));
      renderMC();
      kick();
    }, 0);
    cleanups.push(function () { clearTimeout(prefillT); });

    /* On-screen order: a (patterns), b (largest overlap wins), c (kept runs vote harder), d (the
       prediction). Panel e (why the vote is 2μa) and the p = 2 cross-check are not needed for the
       test, so they fold away under the prediction. */
    steps.appendChild(s4.el);
    steps.appendChild(s5.el);
    s4.el.classList.add('ve-amplify__step--wide');
    /* The two folds are drawers of the chapter's own group (plate, caveat,
       Result, then the drawers). Panel e is a Plot drawer, so its figure may span the plate while
       its prose keeps the drawer measure. Outside a chapter (dev harness) the same drawers sit
       under the plate. */
    var chEl = root.closest ? root.closest('.v2-ch') : null;
    var dHost = chEl && chEl.querySelector('.v2-drawers');
    var ownHost = !dHost;
    if (ownHost) dHost = el('div', { class: 'v2-drawers ve-amplify__drawers' });
    var idBase = chEl && chEl.dataset.chapter ? chEl.dataset.chapter : ui.uid('amp');
    function drawer(id, kind, q, plot, node) {
      var inner = el('div', { class: 've-amplify ve-amplify--drawer' + (plot ? ' v2-plate' : ''), dataset: plot ? { state: 'mounted' } : {} }, node);
      return el('details', { class: 'v2-drawer ve-amplify__drawer' + (plot ? ' v2-drawer--plot' : ''), id: id },
        el('summary', {}, el('span', { class: 'v2-drawer__k', text: kind }), el('span', { class: 'v2-drawer__q', html: q })),
        el('div', { class: 'v2-drawer__body' }, inner));
    }
    var foldE = drawer(idBase + '-vote', 'Panel e', 'Why is a larger overlap a stronger vote?', true, s3.el);
    var foldP2 = drawer(idBase + '-p2', 'Cross-check', 'At <i>p</i>&nbsp;=&nbsp;2 the network splits in two: √2 if they are independent', false, p2box);
    var whyD = !ownHost && dHost.querySelector(':scope > details[id$="-why"]');
    if (whyD) whyD.after(foldE); else dHost.appendChild(foldE);
    dHost.appendChild(foldP2);
    if (ownHost) fig.stage.appendChild(dHost);
    if (typeof ui.drawers === 'function') ui.drawers(dHost);
    [foldE, foldP2].forEach(function (d) {
      d.addEventListener('toggle', function () { if (d.open) window.dispatchEvent(new Event('resize')); });
    });
    cleanups.push(function () { foldE.remove(); foldP2.remove(); });

    root.appendChild(fig.el);

    // "All details" (shell switch): explanations move inline and the two folds open; off puts them back
    function allDetails(on) {
      placeExplains(on);
      foldE.open = on; foldP2.open = on; convFold.open = on;
    }
    if (ui.allDetails()) allDetails(true); else placeExplains(false);
    cleanups.push(ui.onAllDetails(allDetails));

    Promise.all([
      S.loadJSON('table1.json').then(function (d) { data.t1 = d; }),
      S.loadJSON('sampler_clock.json').then(function (d) { data.clock = d; }).catch(function () {}),
      S.loadJSON('size_scan.json').then(function (d) { data.scan = d; }).catch(function () {})
    ]).then(function () { if (!destroyed) { renderP2(); renderTable(); } }).catch(function (e) {
      console.error(e);
      tableNote.textContent = 'Could not load data/table1.json: ' + e.message;
    });

    return {
      el: fig.el,
      setP: setP,
      destroy: function () {
        destroyed = true;
        lp.destroy();
        cleanups.forEach(function (f) { f(); });
        [hopCanvas, drawCanvas, voteCanvas, histCanvas, convCanvas].forEach(function (cv) { cv.destroy(); });
        fig.el.remove();
      }
    };
  }

  VE.components.amplify = { mount: mount, AP_EXACT: AP_EXACT };
})();
