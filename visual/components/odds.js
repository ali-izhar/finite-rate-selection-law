/* components/odds.js — "Measuring the preference: the kept fraction c"  (visual/README.md story item 3)
 *
 * Teaches what c is: the model's preference for one mode (q_eq) against the fraction of sampler
 * chains that land in it (q_chain), shown as counts out of 100 and as log-odds arrows;
 * c = logit(q_chain) / logit(q_eq). A third panel shows why log-odds is the right scale: on it,
 * the exact chain keeps the same share at every bias; on probabilities the "share" drifts.
 *
 * Every number comes from EXACT below (copied from data/odds/c_vs_bias.json, which records how
 * it was computed: VE.sim.protocolGen(2048, G) — the paper's full protocol on the exact two-mode
 * chain, no sampling; its G = 6.125e-4 row is the paper's c_CW = 0.120, data/c_cw.json).
 * Controls: model preference (presets + slider + drag), sampler (presets + slider + drag).
 */
(function () {
  'use strict';
  var VE = (window.VE = window.VE || {});
  VE.components = VE.components || {};

  /* [logit q_eq, logit q_chain] on the exact two-mode chain, N = 2048, paper protocol.
     Source: data/odds/c_vs_bias.json (rows sorted by bias G). Row 10 is the paper's G0. */
  var EXACT = [
    [0.1, 0.011949], [0.25, 0.029873], [0.693147, 0.082834], [1.0, 0.119516],
    [1.5, 0.179321], [2.0, 0.239182], [2.197225, 0.262813], [2.5, 0.299117],
    [3.0, 0.359144], [3.5, 0.419284], [3.574685, 0.428277], [3.583519, 0.429341],
    [4.0, 0.479553], [4.59512, 0.551483], [5.0, 0.600555], [5.5, 0.661325]
  ];
  var L_PAPER = 3.574685;           // logit q_eq at the paper's G0 = 6.125e-4 (q_eq = 0.97274)
  var L_MIN = 0.1, L_MAX = 5.5;     // the range covered by EXACT
  var X_MAX = 5.6;                  // right end of the panel-3 log-odds axis (panel 2 zooms to the model)

  function expit(L) { return 1 / (1 + Math.exp(-L)); }
  function logit(q) { return Math.log(q / (1 - q)); }
  /* the exact chain's logit q_chain at a model log-odds L (linear interpolation in EXACT) */
  function paperChain(L) {
    if (L <= EXACT[0][0]) return L * EXACT[0][1] / EXACT[0][0];
    for (var i = 1; i < EXACT.length; i++) {
      if (L <= EXACT[i][0]) {
        var a = EXACT[i - 1], b = EXACT[i], t = (L - a[0]) / (b[0] - a[0]);
        return a[1] + t * (b[1] - a[1]);
      }
    }
    var z = EXACT[EXACT.length - 1];
    return L * z[1] / z[0];
  }
  function oddsStr(q) {
    var o = q / (1 - q);
    if (o >= 9.95) return Math.round(o) + '\u202F:\u202F1';
    if (Math.abs(o - Math.round(o)) < 0.005) return Math.round(o) + '\u202F:\u202F1';
    if (o >= 2.995) return o.toFixed(1) + '\u202F:\u202F1';
    return o.toFixed(2) + '\u202F:\u202F1';
  }

  /* canvas text on a solid surface-coloured box, so lines and dots behind it never show through */
  function boxText(ui, ctx, str, x, y, o, c) {
    ctx.save(); ctx.font = ui.font('mono', o.size || 11, o.weight || 400);
    var w = ctx.measureText(str).width, al = o.align || 'left';
    var x0 = al === 'right' ? x - w : al === 'center' ? x - w / 2 : x;
    ctx.fillStyle = c.surface; ctx.fillRect(x0 - 3, y - 8, w + 6, 16); ctx.restore();
    ui.plot.text(ctx, str, x, y, { size: o.size || 11, weight: o.weight || 400, color: o.color, align: al, baseline: 'middle' });
  }

  var MODEL_PRESETS = [
    { value: 'two', L: Math.log(2), label: '2 : 1', title: 'The model says “3” twice as often as “8”' },
    { value: 'paper', L: L_PAPER, label: '36 : 1 (paper)', title: 'The paper’s field, G = 6.125×10⁻⁴: q_eq = 0.973' },
    { value: 'ninetynine', L: Math.log(99), label: '99 : 1', title: 'The model says “3” 99 times as often as “8”' }
  ];
  /* probability ticks on the log-odds axes */
  var P_TICKS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99];
  var P_TICKS_FINE = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85];
  var NB = '\u202F';   // narrow no-break space, keeps "99 : 1" on one line

  /* draw a run of text parts on a canvas; a part [str, true] is a subscript. Returns the width. */
  function subText(ui, ctx, parts, x, y, o) {
    var size = o.size || 11, weight = o.weight || 400, w = 0;
    function font(sub) { return ui.font('mono', sub ? Math.round(size * 0.78) : size, weight); }
    parts.forEach(function (pt) { ctx.font = font(pt[1]); w += ctx.measureText(pt[0]).width; });
    if (o.measureOnly) return w;
    var cx = o.align === 'right' ? x - w : x;
    ctx.save(); ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
    parts.forEach(function (pt) {
      ctx.font = font(pt[1]);
      var yy = pt[1] ? y + size * 0.32 : y;
      if (o.halo) { ctx.lineJoin = 'round'; ctx.lineWidth = 4; ctx.strokeStyle = o.haloColor; ctx.strokeText(pt[0], cx, yy); }
      ctx.fillStyle = o.color; ctx.fillText(pt[0], cx, yy);
      cx += ctx.measureText(pt[0]).width;
    });
    ctx.restore();
    return w;
  }

  VE.components['odds'] = {
    mount: function (root, opts) {
      opts = opts || {};
      var ui = VE.ui, el = ui.el, fmt = ui.fmt;
      root.classList.add('ve-odds');

      /* ---------------------------------------------------------- state */
      var S = { L: L_PAPER, mode: 'paper', cCustom: 0.12 };
      function cOf() {
        if (S.mode === 'paper') return paperChain(S.L) / S.L;
        if (S.mode === 'faithful') return 1;
        if (S.mode === 'coin') return 0;
        return S.cCustom;
      }
      function Lc() { return cOf() * S.L; }

      /* ---------------------------------------------------------- frame */
      var fig = ui.figure({
        plate: opts.plate != null ? opts.plate : 3,
        title: 'Measuring the preference: the share c the sampler keeps',
        noteKind: 'paper',
        note: 'Sampler: the exact CW chain for the Curie–Weiss model (N + 1 counts of +1 spins, evolved numerically, not sampled; paper Sec. 2, App. A) · Model size: N = 2,048 spins · ' +
          'Protocol: brief cooling (2 sweeps per temperature), a filtered start that keeps only runs that ended negative, reheat 30 sweeps at β = 0.7, then the 834-sweep final cooling (paper Sec. 4, App. A; one expected sweep is N single-spin updates on average, random scan) · ' +
          'c = 0.120 at the paper’s field (paper Sec. 4); 0.119498–0.119808 over the paper’s eight fields; other fields here use the same code (data/odds/c_vs_bias.json) · ' +
          'One-pattern Hopfield runs (Table 1), which differ in update order and interaction strength (1/N against 1/(N−1)), measure 0.127 [0.124, 0.131]; separate one-pattern runs give 1.063 × c_CW (paper Sec. 5)'
      });
      root.appendChild(fig.el);
      var stage = fig.stage;

      /* v2 disclosure: an info chip opens a card. Each card also has an inline
         copy (.odds-more) that shows only when the reader turns on "All details" (html.v2-all),
         so that switch opens everything in this figure too. */
      var cards = [];
      function infoChip(key, title, html, inlineParent) {
        var btn = el('button', { type: 'button', class: 'v2-info odds-i', html: ui.INFO_SVG, 'aria-label': title.replace(/<[^>]+>/g, '') });
        var api = ui.info(btn, { id: ui.uid('odds-' + key), title: title, html: html });
        cards.push(api.el);
        var more = el('div', { class: 'odds-more' }, el('p', { class: 'odds-more__k', html: title }), el('div', { class: 'odds-more__b', html: html }));
        if (inlineParent) inlineParent.appendChild(more);
        function setHtml(h) {
          api.el.querySelector('.v2-card__body').innerHTML = h;
          more.lastChild.innerHTML = h;
          ui.terms(api.el); ui.terms(more);
        }
        return { btn: btn, more: more, setHtml: setHtml, card: api.el };
      }
      function stepHead(text, extra) {
        // One panel-head style on the page: a small-caps letter, then the title in serif
        var m = /^(\w)\s·\s(.*)$/.exec(text);
        return el('div', { class: 'odds-step' }, m ? el('span', { class: 'odds-step__k', text: m[1] }) : null,
          el('span', { class: 'odds-step__t', text: m ? m[2] : text }), extra);
      }

      /* No lede inside the figure; the chapter lede and the panel-1 info card carry it */

      /* ----------------------------------------------- panel 1: counts */
      var i1 = infoChip('count', 'Count the outcomes',
        '<p>Suppose a model says “3” twice as often as “8”. Sampled many times, it should give twice as many 3s. ' +
        'Two numbers check this: <b class="ve-c-bias">q<sub>eq</sub></b>, the probability the model gives the favoured mode, and ' +
        '<b class="ve-c-data">q<sub>chain</sub></b>, the fraction of sampler runs (chains) that end in it.</p>' +
        '<p>Each square is one sample out of 100: a <b class="ve-c-mode-a">blue</b> square is a “3”, mode A, the mode the model favours; ' +
        'an <b class="ve-c-mode-b">orange</b> square is an “8”, the other. The paper’s Curie–Weiss model has no digits; they are drawn here to fix ideas.</p>' +
        '<p>The figure opens on the paper’s own test case, a preference of 36' + NB + ':' + NB + '1; the 2' + NB + ':' + NB + '1 button shows the twice-as-often case.</p>');
      var key1 = el('span', { class: 'odds-key', html: '<i class="odds-sw odds-sw--a" aria-hidden="true"></i>“3”, mode A (favoured)<i class="odds-sw odds-sw--b" aria-hidden="true"></i>“8”, the other' });
      var p1 = el('section', { class: 'odds-panel' },
        stepHead('a · Count the outcomes', [i1.btn, key1]));
      stage.appendChild(p1);
      function makeArray(kind, title, sym) {
        var cells = [];
        var grid = el('div', { class: 'odds-grid', role: 'img' });
        for (var i = 0; i < 100; i++) { var c = el('span', { class: 'odds-cell' }); cells.push(c); grid.appendChild(c); }
        var head = el('div', { class: 'odds-arr__head' },
          el('span', { class: 'odds-arr__title', html: title }),
          el('span', { class: 'odds-arr__sym ve-num odds-k-' + kind, html: sym }));
        var foot = el('div', { class: 'odds-arr__foot ve-num' });
        var box = el('div', { class: 'odds-arr odds-arr--' + kind }, head, grid, foot);
        return { el: box, cells: cells, grid: grid, foot: foot, sym: head.lastChild, last: -1 };
      }
      var arrModel = makeArray('model', 'The model says · <i>q</i><sub>eq</sub>', '');
      var arrChain = makeArray('chain', 'The sampler gives · <i>q</i><sub>chain</sub>', '');
      var arrows = el('div', { class: 'odds-arrays' }, arrModel.el, arrChain.el);
      p1.appendChild(el('div', { class: 'odds-p1' }, arrows));
      p1.appendChild(i1.more);

      function paintArray(a, q, label) {
        var n = Math.round(100 * q);
        if (n !== a.last) {
          for (var i = 0; i < 100; i++) {
            var on = i < n;
            a.cells[i].dataset.d = on ? '3' : '8';   // drawn by CSS: the grid is a picture (role img, aria-label), not text
            a.cells[i].className = 'odds-cell ' + (on ? 'is-a' : 'is-b');
          }
          a.last = n;
        }
        a.grid.setAttribute('aria-label', n + ' threes and ' + (100 - n) + ' eights out of 100');
        a.sym.textContent = q.toFixed(3);
        a.sym.setAttribute('aria-label', label.replace(/<[^>]+>/g, '') + ' = ' + q.toFixed(3));
        a.foot.innerHTML = '<span class="ve-c-mode-a" title="' + n + ' threes">' + n + '</span><span class="odds-foot__sep">:</span><span class="ve-c-mode-b" title="' + (100 - n) + ' eights">' + (100 - n) +
          '</span><span class="odds-odds">odds ' + oddsStr(q) + '</span>';
      }

      /* -------------------------------------------- panel 2: log-odds */
      var i2 = infoChip('logodds', 'Compare them as log-odds',
        '<p>Put each probability on the <span data-term="logOdds">log-odds</span> line, <span class="odds-nw">ln(q / (1 − q))</span>: a coin flip sits at 0, ' +
        'and each step of 1 multiplies the odds by e ≈ 2.7. The model’s bar is its preference; the sampler’s bar is what survives. ' +
        '<b>c is the ratio of the two lengths.</b> Drag the end of either bar.</p>');
      var p2 = el('section', { class: 'odds-panel' },
        stepHead('b · Compare them as log-odds', [i2.btn, el('span', { class: 'odds-key', text: 'drag either bar’s end' })]),
        i2.more);
      stage.appendChild(p2);
      var p2grid = el('div', { class: 'odds-p2' });
      p2.appendChild(p2grid);
      var cvWrap = el('div', { class: 'odds-cv' });
      p2grid.appendChild(cvWrap);

      var sentence = el('p', { class: 'odds-sentence' });
      var formula = el('div', { class: 'odds-formula' });
      var status = el('p', { class: 'odds-status', 'aria-live': 'polite' });
      var iS = infoChip('sampler', 'The sampler', '');
      var statusRow = el('div', { class: 'odds-statusrow' }, status, iS.btn);
      var side = el('div', { class: 'odds-side' }, formula, sentence, statusRow);
      p2grid.appendChild(side);
      p2.appendChild(iS.more);

      var drag = null;   // 'model' | 'chain'
      var geo = null;    // layout of the last draw
      /* the log-odds axis zooms to the model's preference, so a weak preference (2 : 1) stays readable.
         It eases to the new range, and holds still while the reader drags the model's bar. */
      var axis = { cur: Math.max(1.25 * S.L, 1.2), raf: 0 };
      function axisTarget() { return Math.max(1.25 * S.L, 1.2); }
      function updateAxis() {
        if (drag === 'model') return;
        var target = axisTarget();
        if (ui.reducedMotion) { axis.cur = target; return; }
        if (axis.raf) return;
        var t0 = null, from = axis.cur;
        function step(ts) {
          if (t0 == null) t0 = ts;
          var to = axisTarget(), u = Math.min(1, (ts - t0) / 320), e = 1 - Math.pow(1 - u, 3);
          axis.cur = from + (to - from) * e;
          cv.redraw();
          if (u < 1) axis.raf = requestAnimationFrame(step); else { axis.raf = 0; axis.cur = to; }
        }
        if (Math.abs(target - axis.cur) > 1e-4) axis.raf = requestAnimationFrame(step);
      }
      var cv = ui.canvas(cvWrap, {
        height: 236,
        label: 'Two bars on a log-odds axis: the model’s preference and the sampler’s.',
        draw: drawArrows
      });

      function drawArrows(ctx, w, h, c) {
        var narrow = w < 520;
        var xMax = axis.cur;
        var x0 = narrow ? 14 : 20, x1 = w - (narrow ? 16 : 24);
        var sx = ui.scale.linear([0, xMax], [x0, x1]);
        var yTop = 36, yM = 82, yC = 158, yBot = 194;
        geo = { sx: sx, yM: yM, yC: yC };
        var L = S.L, lc = Lc();

        // top axis: probability of the favoured mode
        ctx.strokeStyle = c.rule; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x0, yTop + 0.5); ctx.lineTo(x1, yTop + 0.5); ctx.stroke();
        ui.plot.text(ctx, 'probability of “3”', x0, 9, { size: 11, color: c.ink3, baseline: 'middle' });
        var lastX = -1e9;
        (xMax < 2.2 ? P_TICKS_FINE : P_TICKS).forEach(function (p) {
          if (logit(p) > xMax + 1e-9) return;
          var x = Math.round(sx(logit(p))) + 0.5;
          if (x - lastX < (narrow ? 28 : 32)) return;
          ctx.strokeStyle = c.ruleStrong; ctx.beginPath(); ctx.moveTo(x, yTop - 4); ctx.lineTo(x, yTop); ctx.stroke();
          ctx.strokeStyle = c.grid; ctx.beginPath(); ctx.moveTo(x, yTop + 1); ctx.lineTo(x, yBot); ctx.stroke();
          var s = p === 0.5 ? '0.5' : String(p).replace(/^0/, '');
          ui.plot.text(ctx, s, x, yTop - 11, { size: 11, color: c.ink2, align: 'center', baseline: 'middle' });
          lastX = x;
        });
        // bottom axis: log-odds
        var stepT = xMax <= 1.6 ? 0.25 : xMax <= 3.2 ? 0.5 : 1;
        ctx.strokeStyle = c.ruleStrong; ctx.beginPath(); ctx.moveTo(x0, yBot + 0.5); ctx.lineTo(x1, yBot + 0.5); ctx.stroke();
        var lastT = -1e9;
        for (var t = 0; t <= xMax + 1e-9; t += stepT) {
          var xt = Math.round(sx(t)) + 0.5;
          ctx.beginPath(); ctx.moveTo(xt, yBot); ctx.lineTo(xt, yBot + 4); ctx.stroke();
          if (xt - lastT >= 30) {
            ui.plot.text(ctx, String(+t.toFixed(2)), xt, yBot + 13, { size: 11, color: c.ink2, align: 'center', baseline: 'middle' });
            lastT = xt;
          }
        }
        ui.plot.text(ctx, '0 = coin flip', x0, yBot + 30, { size: 11, color: c.ink3, baseline: 'middle' });
        ui.plot.text(ctx, narrow ? 'log-odds' : 'log-odds  ln(q / (1 − q))', x1, yBot + 30, { size: 11, color: c.ink3, align: 'right', baseline: 'middle' });
        // coin-flip line at 0
        var xz = sx(0);
        ctx.save(); ctx.strokeStyle = c.noise; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(Math.round(xz) + 0.5, yTop); ctx.lineTo(Math.round(xz) + 0.5, yBot); ctx.stroke(); ctx.restore();

        var xm = Math.min(sx(L), x1), xc = Math.min(sx(lc), x1);
        // kept part of the model's bar: a wash with an outline
        if (xc - xz > 0.5) {
          ctx.fillStyle = c.highlightWash;
          ctx.fillRect(xz, yM - 12, xc - xz, 24);
          ctx.strokeStyle = c.highlightInk; ctx.lineWidth = 1;
          ctx.strokeRect(Math.round(xz) + 0.5, yM - 11.5, Math.max(0, Math.round(xc - xz)), 23);
        }
        // bars, each with a drag grip at its end (the grip sits past the end so the bar keeps its full length)
        function bar(y, xEnd, col, active) {
          ctx.fillStyle = col;
          if (xEnd - xz > 0.5) ctx.fillRect(xz, y - 5, xEnd - xz, 10);
          var gw = active ? 9 : 7, gh = active ? 28 : 24;
          var gx = xEnd + 1;
          ctx.beginPath();
          if (ctx.roundRect) ctx.roundRect(gx, y - gh / 2, gw, gh, 3); else ctx.rect(gx, y - gh / 2, gw, gh);
          ctx.fillStyle = c.surface; ctx.fill();
          ctx.lineWidth = 2; ctx.strokeStyle = col; ctx.stroke();
          ctx.lineWidth = 1;
        }
        bar(yM, xm, c.bias, drag === 'model');
        bar(yC, xc, c.data, drag === 'chain');
        // row labels with real subscripts, placed above each bar
        function rowLabel(y, who, sub, val, col) {
          var parts = [[who + '  logit q', false], [sub, true], [' = ' + val, false]];
          var wd = subText(ui, ctx, parts, 0, 0, { size: 12, weight: 600, measureOnly: true });
          ctx.fillStyle = c.surface;
          ctx.fillRect(x0 + 1, y - 34, wd + 7, 18);
          subText(ui, ctx, parts, x0 + 4, y - 25, { size: 12, weight: 600, color: col, halo: true, haloColor: c.surface });
          return x0 + 8 + wd;
        }
        rowLabel(yM, 'Model', 'eq', L.toFixed(3), c.bias);
        var labC = rowLabel(yC, 'Sampler', 'chain', lc.toFixed(3), c.ink);
        // dashed drop line from the sampler's end up to the model's row; it starts below the kept/lost
        // labels and stops above the sampler's label when that label lies under it
        if (xc - xz > 2) {
          var xd = Math.round(xc) + 0.5, yEnd = (xc <= labC + 6) ? yC - 38 : yC - 7;
          if (yEnd > yM + 44) {
            ctx.save(); ctx.strokeStyle = c.highlightInk; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(xd, yM + 36); ctx.lineTo(xd, yEnd); ctx.stroke(); ctx.restore();
          }
        }
        // "kept" / "lost" labels under the model's bar; a label moves aside when its span is short
        var cc = cOf(), yl = yM + 26, gap = 10;
        ctx.font = ui.font('mono', 12, 600);
        var tKeep = 'retained ' + Math.round(100 * cc) + '%', tLost = 'lost ' + Math.round(100 * (1 - cc)) + '%';
        var wK = ctx.measureText(tKeep).width, wL = ctx.measureText(tLost).width;
        var kRight = xz;
        if (cc > 0.0005) {
          var kx = (xc - xz >= wK + 8) ? (xz + xc) / 2 - wK / 2 : xz + 2;
          ui.plot.text(ctx, tKeep, kx, yl, { size: 12, weight: 600, color: c.highlightInk, baseline: 'middle', halo: true });
          kRight = kx + wK;
        }
        if (cc < 0.9995) {
          var lx = (xc + xm) / 2 - wL / 2;
          if (lx < kRight + gap) lx = kRight + gap;
          if (lx + wL > x1) lx = x1 - wL;
          ui.plot.text(ctx, tLost, lx, yl, { size: 12, weight: 600, color: c.ink3, baseline: 'middle', halo: true });
          // a bracket ties "lost" to its span when the label had to move
          ctx.strokeStyle = c.ruleStrong; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(Math.round(xc) + 0.5, yM + 15); ctx.lineTo(Math.round(xc) + 0.5, yM + 18);
          ctx.lineTo(Math.round(xm) + 0.5, yM + 18); ctx.lineTo(Math.round(xm) + 0.5, yM + 15); ctx.stroke();
        }
      }

      /* drag on the canvas: pick the nearer row; the model row moves q_eq, the sampler row moves q_chain */
      var canvasEl = cv.canvas;
      canvasEl.style.cursor = 'pointer';
      function lFromEvt(e) {
        var p = cv.local(e);
        return { L: geo.sx.invert(p.x), y: p.y, x: p.x };
      }
      function onDown(e) {
        if (!geo) return;
        var p = lFromEvt(e);
        drag = Math.abs(p.y - geo.yM) < Math.abs(p.y - geo.yC) ? 'model' : 'chain';
        try { canvasEl.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
        applyDrag(p.L);
        e.preventDefault();
      }
      function onMove(e) {
        if (!geo) return;
        if (!drag) {
          var p0 = lFromEvt(e);
          var near = Math.abs(p0.x - geo.sx(S.L)) < 16 && Math.abs(p0.y - geo.yM) < 16 ||
            Math.abs(p0.x - geo.sx(Lc())) < 16 && Math.abs(p0.y - geo.yC) < 16;
          canvasEl.style.cursor = near ? 'ew-resize' : 'pointer';
          return;
        }
        applyDrag(lFromEvt(e).L);
      }
      function onUp() { if (drag) { drag = null; updateAxis(); cv.redraw(); } }
      canvasEl.addEventListener('pointerdown', onDown);
      canvasEl.addEventListener('pointermove', onMove);
      canvasEl.addEventListener('pointerup', onUp);
      canvasEl.addEventListener('pointercancel', onUp);
      function applyDrag(L) {
        if (drag === 'model') setModel(Math.max(L_MIN, Math.min(L_MAX, L)), true);
        else setChain(Math.max(0, Math.min(S.L, L)));
      }

      /* ------------------------------------------------------- controls */
      var modelToggle = ui.toggle({
        label: 'Model, “3” : “8”',
        options: MODEL_PRESETS.map(function (p) { return { value: p.value, label: p.label, title: p.title, tone: 'bias' }; }),
        value: 'paper',
        onChange: function (v) {
          var p = MODEL_PRESETS.filter(function (x) { return x.value === v; })[0];
          if (p) setModel(p.L, false);
        }
      });
      var modelSlider = ui.slider({
        label: 'probability of “3”', sym: 'q', min: L_MIN, max: L_MAX, value: S.L, tone: 'bias',
        format: function (L) { return expit(L).toFixed(3); },
        onInput: function (L) { setModel(L, true, true); }
      });
      var SAMPLERS = [
        { value: 'paper', label: 'The paper’s', title: 'The exact CW chain on the paper’s protocol, N = 2,048 (details in the note below)' },
        { value: 'faithful', label: 'Faithful (c = 1)' },
        { value: 'coin', label: 'Coin flip (c = 0)' }
      ];
      var samplerToggle = ui.toggle({
        label: 'Sampler',
        options: SAMPLERS.map(function (s) { return { value: s.value, label: s.label.replace(/\(c = (\d)\)/, '(c\u00a0=\u00a0$1)'), title: s.title }; }),   // '(c = 1)' never breaks inside
        value: 'paper',
        onChange: function (v) { S.mode = v; render(); }
      });
      var chainSlider = ui.slider({
        label: 'chains ending in “3”', sym: 'q', min: 0, max: L_MAX, value: Lc(),
        format: function (L) { return expit(L).toFixed(3); },
        onInput: function (L) { setChain(Math.min(L, S.L), true); }
      });
      /* typeset the slider symbols with real subscripts, as in the prose */
      modelSlider.el.querySelector('.ve-sym').innerHTML = 'q<sub>eq</sub>';
      chainSlider.el.querySelector('.ve-sym').innerHTML = 'q<sub>chain</sub>';
      var grpModel = el('div', { class: 'odds-ctl odds-ctl--model' }, modelToggle.el, modelSlider.el);
      var grpChain = el('div', { class: 'odds-ctl odds-ctl--chain' }, samplerToggle.el, chainSlider.el);
      p2.appendChild(el('div', { class: 'odds-ctls' }, grpModel, grpChain));
      fixToggleFocus(modelToggle); fixToggleFocus(samplerToggle);
      /* the bar canvas takes no keyboard focus; the two sliders are its keyboard control */
      var slIds = [modelSlider.input, chainSlider.input].map(function (inp) {
        if (!inp.id) inp.id = ui.uid('odds-sl');
        return inp.id;
      });
      canvasEl.setAttribute('aria-describedby', slIds.join(' '));

      /* Fit this chapter's cards to the viewport (desktop only; the phone uses the bottom sheet).
         The shared positioner caps a card at the room below (or above) its trigger and lets it
         scroll. When the card's full text fits on screen, shift it so all of it shows; the reader
         then never sees a cut card that looks complete. Runs after the shared positioner. */
      function fitCard(card) {
        if (!card || !card.matches(':popover-open') || window.innerWidth < 900 || card.classList.contains('is-sheet')) return;
        var vh = window.innerHeight, r = card.getBoundingClientRect();
        if (card.scrollHeight <= card.clientHeight + 1 && r.top >= 16 && r.bottom <= vh - 16) return;
        card.style.maxHeight = '';
        var h = card.offsetHeight;                // natural height, still under the CSS cap
        if (h > vh - 32) { card.style.maxHeight = (vh - 32) + 'px'; h = vh - 32; }
        card.style.top = Math.max(16, Math.min(r.top, vh - 16 - h)) + 'px';
      }
      /* a card that still scrolls fades its last lines (class odds-card--more, odds.css) until the end is reached */
      function markMore(card) {
        card.classList.toggle('odds-card--more', card.scrollTop + card.clientHeight < card.scrollHeight - 2);
        if (!card.__oddsMore) { card.__oddsMore = true; card.addEventListener('scroll', function () { markMore(card); }, { passive: true }); }
      }
      function isOurs(card) { return card && (card.id === 'odds-spec' || card.id === 'odds-read' || cards.indexOf(card) >= 0); }
      function onToggle(e) {
        if (e.newState !== 'open' || !isOurs(e.target)) return;
        var card = e.target;
        requestAnimationFrame(function () { fitCard(card); markMore(card); });
      }
      var refitRaf = 0;
      function onViewport() {
        if (refitRaf) return;
        refitRaf = requestAnimationFrame(function () {
          refitRaf = 0;
          document.querySelectorAll('.v2-card:popover-open').forEach(function (k) { if (isOurs(k)) { fitCard(k); markMore(k); } });
        });
      }
      document.addEventListener('toggle', onToggle, true);
      window.addEventListener('resize', onViewport);
      window.addEventListener('scroll', onViewport, { passive: true, capture: true });

      /* A toggle whose value is not one of its options has no tabbable button; keep the first one tabbable. */
      function fixToggleFocus(t) {
        var btns = t.el.querySelectorAll('.ve-toggle__opt');
        var any = false;
        btns.forEach(function (b) { if (b.getAttribute('aria-checked') === 'true') any = true; });
        if (!any && btns[0]) btns[0].tabIndex = 0;
      }

      function setModel(L, fromContinuous, fromSlider) {
        S.L = L;
        if (fromContinuous) {
          var match = MODEL_PRESETS.filter(function (p) { return Math.abs(p.L - L) < 1e-6; })[0];
          modelToggle.set(match ? match.value : 'custom', { silent: true });
          fixToggleFocus(modelToggle);
        }
        if (!fromSlider) modelSlider.set(L, { silent: true });
        render();
      }
      function setChain(Lchain, fromSlider) {
        S.cCustom = S.L > 0 ? Lchain / S.L : 0;
        if (S.cCustom > 0.9995) S.mode = 'faithful';
        else if (S.cCustom < 0.0005) S.mode = 'coin';
        else S.mode = 'custom';
        samplerToggle.set(S.mode, { silent: true });
        fixToggleFocus(samplerToggle);
        if (fromSlider) paintChainFill(); // keep the thumb where the reader holds it
        render(!fromSlider);
      }
      /* the chain slider runs from 0 up to the model's log-odds; ui.slider was built with max L_MAX, so set the
         input's range and fill directly */
      function paintChainFill() {
        var lc = Lc();
        chainSlider.input.max = String(S.L);
        chainSlider.input.value = String(lc);
        chainSlider.el.style.setProperty('--p', (100 * (S.L > 0 ? lc / S.L : 0)).toFixed(2) + '%');
        var s = expit(lc).toFixed(3);
        chainSlider.el.querySelector('.ve-readout-box > span').textContent = s;
        chainSlider.input.setAttribute('aria-valuetext', s);
      }

      /* ------------------------------------------- panel 3: why log-odds */
      var i3 = infoChip('why', 'Why log-odds, not probability',
        '<p>In an energy-based model a preference between two modes (summed over their states) is a difference in log-probability, a log-odds. ' +
        'The chart runs the paper’s Curie–Weiss sampler at 16 fields: the same sampler should give a flat line, and only the log-odds share c is flat.</p>' +
        '<p><b class="ve-c-highlight">Solid</b>: measured in log-odds, c = logit q<sub>chain</sub> / logit q<sub>eq</sub>.<br>' +
        '<b>Dashed</b>: measured in probability, <span class="odds-nw">(q<sub>chain</sub>&nbsp;−&nbsp;½)&nbsp;/&nbsp;(q<sub>eq</sub>&nbsp;−&nbsp;½)</span>.</p>' +
        '<p>The dotted line marks the model you picked above. The retained share rises slowly at stronger preferences than this page shows, so “flat” holds over the range drawn here.</p>');
      var p3 = el('section', { class: 'odds-panel' },
        stepHead('c · Why log-odds, not probability', [i3.btn]), i3.more);
      stage.appendChild(p3);
      var cv3Wrap = el('div', { class: 'odds-cv3' });
      p3.appendChild(cv3Wrap);
      var cv3 = ui.canvas(cv3Wrap, {
        height: function (w) { return w < 520 ? 230 : 220; },
        label: 'Share retained by the paper’s Curie–Weiss sampler, against the model’s preference, measured two ways.',
        draw: drawShares
      });

      function drawShares(ctx, w, h, c) {
        var narrow = w < 520;
        var box = ui.inset(w, h, { top: 30, right: narrow ? 12 : 24, bottom: 46, left: narrow ? 40 : 48 });
        var sx = ui.scale.linear([0, X_MAX], [box.x0, box.x1]);
        var sy = ui.scale.linear([0, 0.4], [box.y1, box.y0]);
        // grid + y axis
        [0, 0.1, 0.2, 0.3, 0.4].forEach(function (v) {
          var y = Math.round(sy(v)) + 0.5;
          ctx.strokeStyle = v === 0 ? c.ruleStrong : c.grid; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(box.x0, y); ctx.lineTo(box.x1, y); ctx.stroke();
          ui.plot.text(ctx, v.toFixed(1), box.x0 - 6, y, { size: 11, color: c.ink2, align: 'right', baseline: 'middle' });
        });
        ui.plot.text(ctx, 'share retained', box.x0 - (narrow ? 36 : 44), box.y0 - 20, { size: 11, color: c.ink3, baseline: 'middle' });
        // x axis: the model's preference, probability ticks at log-odds positions
        var lastX = -1e9;
        P_TICKS.forEach(function (p) {
          var x = Math.round(sx(logit(p))) + 0.5;
          ctx.strokeStyle = c.ruleStrong; ctx.beginPath(); ctx.moveTo(x, box.y1); ctx.lineTo(x, box.y1 + 4); ctx.stroke();
          if (x - lastX >= (narrow ? 26 : 30)) {
            ui.plot.text(ctx, p === 0.5 ? '0.5' : String(p).replace(/^0/, ''), x, box.y1 + 14, { size: 11, color: c.ink2, align: 'center', baseline: 'middle' });
            lastX = x;
          }
        });
        subText(ui, ctx, narrow ? [['model’s q', false], ['eq', true], [' (log-odds spacing)', false]]
          : [['model’s probability of “3”, q', false], ['eq', true], [' (spaced as log-odds)', false]],
          box.x1, box.y1 + 33, { size: 11, color: c.ink3, align: 'right' });

        var xs = EXACT.map(function (r) { return r[0]; });
        var cLog = EXACT.map(function (r) { return r[1] / r[0]; });
        var cProb = EXACT.map(function (r) { return (expit(r[1]) - 0.5) / (expit(r[0]) - 0.5); });
        // current model preference
        // (the marker skips the band of the dashed curve's two-line label so it never strikes through it)
        var xNow = Math.round(sx(S.L)) + 0.5;
        var tx = box.x0 + 8, ty = sy(0.355);
        var tA1 = narrow ? 'in probability: drifts with the' : 'in probability: same sampler, but the share drifts with the preference,';
        var tA2 = narrow ? 'preference, so not the sampler' : 'so it cannot describe the sampler';
        ctx.font = ui.font('mono', 11, 400);
        var wA = Math.max(ctx.measureText(tA1).width, ctx.measureText(tA2).width);
        var hitsLabel = xNow > tx - 4 && xNow < tx + wA + 4;
        ctx.save(); ctx.strokeStyle = c.bias; ctx.setLineDash([2, 3]); ctx.beginPath();
        if (hitsLabel) {
          ctx.moveTo(xNow, box.y0); ctx.lineTo(xNow, ty - 11);
          ctx.moveTo(xNow, ty + 25); ctx.lineTo(xNow, box.y1);
        } else { ctx.moveTo(xNow, box.y0); ctx.lineTo(xNow, box.y1); }
        ctx.stroke(); ctx.restore();
        ui.plot.line(ctx, xs, cProb, sx, sy, { color: c.noise, width: 2, dash: [5, 4] });
        ui.plot.line(ctx, xs, cLog, sx, sy, { color: c.highlightInk, width: 2.5 });
        ui.plot.dots(ctx, xs.map(function (x, i) { return { x: x, y: cProb[i] }; }), sx, sy, { r: 2.5, color: c.noise });
        ui.plot.dots(ctx, xs.map(function (x, i) { return { x: x, y: cLog[i] }; }), sx, sy, { r: 2.5, color: c.highlightInk });
        // direct labels on the two curves, in the empty upper-left and lower bands
        ui.plot.text(ctx, tA1, tx, ty,
          { size: 11, color: c.ink2, baseline: 'middle', halo: true });
        ui.plot.text(ctx, tA2, tx, ty + 15,
          { size: 11, color: c.ink2, baseline: 'middle', halo: true });
        boxText(ui, ctx, 'in log-odds: flat at c = 0.12', sx(1.0), sy(0.045),
          { weight: 600, color: c.highlightInk }, c);
        // values at the current preference
        var lcNow = paperChain(S.L);
        var vLog = lcNow / S.L, vProb = (expit(lcNow) - 0.5) / (expit(S.L) - 0.5);
        [[vLog, c.highlightInk], [vProb, c.noise]].forEach(function (d) {
          ctx.beginPath(); ctx.arc(sx(S.L), sy(d[0]), 5, 0, 2 * Math.PI);
          ctx.fillStyle = c.surface; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = d[1]; ctx.stroke(); ctx.lineWidth = 1;
        });
        // value labels, on the side of the marker with more room
        var right = sx(S.L) < (box.x0 + box.x1) / 2;
        var lx = sx(S.L) + (right ? 9 : -9), al = right ? 'left' : 'right';
        var yLog = sy(vLog), yProb = sy(vProb);
        if (Math.abs(yLog - yProb) < 16) yProb = yLog - 16;
        boxText(ui, ctx, vLog.toFixed(3), lx, yLog + 11, { weight: 600, color: c.highlightInk, align: al }, c);
        // the probability label must not run into the two-line annotation (phone width, strong preferences)
        ctx.font = ui.font('mono', 11, 600);
        var wP = ctx.measureText(vProb.toFixed(3)).width;
        var pX = lx, pY = yProb - 9, pAl = al;
        var hitsA = function (x, y, a) {
          var x0 = a === 'left' ? x : x - wP;
          return x0 - 3 < tx + wA + 4 && x0 + wP + 3 > tx - 4 && y - 9 < ty + 22 && y + 9 > ty - 9;
        };
        if (hitsA(pX, pY, pAl)) {
          var rx = sx(S.L) + 9;
          if (!right && rx + wP <= box.x1 && !hitsA(rx, pY, 'left')) { pX = rx; pAl = 'left'; }
          else if (yProb + 12 < yLog - 4) { pY = yProb + 12; }
        }
        boxText(ui, ctx, vProb.toFixed(3), pX, pY, { weight: 600, color: c.ink2, align: pAl }, c);
      }

      /* --------------------------------------------------------- render */
      function render(syncChainSlider) {
        var c = cOf(), L = S.L, lc = c * L;
        var qe = expit(L), qc = expit(lc);
        paintArray(arrModel, qe, 'q<sub>eq</sub>');
        paintArray(arrChain, qc, 'q<sub>chain</sub>');
        if (syncChainSlider !== false) paintChainFill();
        formula.innerHTML =
          '<span class="odds-f__lab">retained share</span>' +
          '<span class="odds-f__row"><span class="odds-f__lhs">c</span><span class="odds-f__eq">=</span>' +
          '<span class="odds-f__frac"><span class="odds-f__num">' + lc.toFixed(3) + '</span><span class="odds-f__den">' + L.toFixed(3) + '</span></span>' +
          '<span class="odds-f__eq">=</span><span class="odds-f__val v2-reticle">' + c.toFixed(3) + '</span></span>';
        formula.setAttribute('aria-label', 'c equals ' + lc.toFixed(3) + ' over ' + L.toFixed(3) + ', which is ' + c.toFixed(3));
        sentence.innerHTML =
          '<b class="ve-c-bias">' + oddsStr(qe) + '</b><span class="odds-arrowto" aria-hidden="true">→</span><b>' + oddsStr(qc) + '</b>';
        sentence.setAttribute('aria-label', 'The model says ' + oddsStr(qe) + ' for 3. The chains give ' + oddsStr(qc) + '.');
        var full = '<p>The model says <b class="ve-c-bias">' + oddsStr(qe) + '</b> for “3”. ' +
          'The chains give <b>' + oddsStr(qc) + '</b>. ' +
          'In log-odds that is ' + lc.toFixed(3) + ' of ' + L.toFixed(3) + ', so the sampler keeps <b>' + Math.round(100 * c) + '%</b> of the preference.</p>';
        var msg, more;
        if (S.mode === 'faithful') { msg = '<b>c = 1, faithful</b>: the chains reproduce the model’s odds.'; more = 'This is what an infinitely slow sampler would do.'; }
        else if (S.mode === 'coin') { msg = '<b>c = 0, coin flip</b>: the preference is lost.'; more = 'Chains land in either mode equally often; the model’s preference is lost.'; }
        else if (S.mode === 'paper') {
          msg = 'The paper’s sampler keeps about <b>12%</b> at every field here.';
          more = '<b>The paper’s Curie–Weiss sampler</b>: the exact CW chain on the paper’s protocol, at N = 2,048. ' +
            'It keeps about 12% (one eighth) at every field on this page (panel c): drag the model’s bar and the sampler’s bar follows at 12% of its length. ' +
            'This 12% belongs to two modes at this size and protocol; with competing patterns the predicted share rises (chapter 6).';
        } else {
          msg = 'Your sampler keeps c = ' + c.toFixed(3) + '. <button type="button" class="odds-reset">Back to the paper’s</button>';
          more = '<b>Your sampler</b> keeps c = ' + c.toFixed(3) + '. Change the model now and the sampler keeps this same share.';
        }
        status.innerHTML = msg;
        iS.setHtml(full + '<p>' + more + '</p>');
        status.dataset.mode = S.mode;
        var rb = status.querySelector('.odds-reset');
        if (rb) rb.addEventListener('click', function () { samplerToggle.set('paper'); });
        updateAxis();
        cv.redraw(); cv3.redraw();
      }

      fig.setTakeaway('On the paper’s protocol the Curie–Weiss sampler keeps c<sub>CW</sub> = 0.120 of the model’s log-odds (N = 2,048), nearly the same at every field.');
      ui.terms(stage);
      render();

      return {
        destroy: function () {
          canvasEl.removeEventListener('pointerdown', onDown);
          canvasEl.removeEventListener('pointermove', onMove);
          canvasEl.removeEventListener('pointerup', onUp);
          canvasEl.removeEventListener('pointercancel', onUp);
          if (axis.raf) cancelAnimationFrame(axis.raf);
          if (refitRaf) cancelAnimationFrame(refitRaf);
          document.removeEventListener('toggle', onToggle, true);
          window.removeEventListener('resize', onViewport);
          window.removeEventListener('scroll', onViewport, { capture: true });
          cv.destroy(); cv3.destroy();
          cards.forEach(function (k) { k.remove(); });
          root.replaceChildren();
          root.classList.remove('ve-odds');
        }
      };
    }
  };
})();
