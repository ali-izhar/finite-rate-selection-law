/* components/odds.js — "Measuring the preference: the kept fraction c"  (the "odds" chapter of index.html)
 *
 * Teaches what c is: the model's preference for one mode (q_eq) against the fraction of sampler
 * chains that land in it (q_chain), shown as counts out of 100, and
 * c = logit(q_chain) / logit(q_eq) on one line. A second panel shows why log-odds is the right scale: on it,
 * the exact chain keeps the same share at every bias; on probabilities the "share" drifts.
 *
 * Every number comes from EXACT below (printed by data/odds/make_c_vs_bias.mjs, which writes
 * data/odds/c_vs_bias.json: VE.sim.rampGen(2048, G), the exact Curie–Weiss chain on the test ramp,
 * no sampling; its G = 6.125e-4 row is the paper's c_CW = 0.0659, Table 1).
 * No controls: the figure shows the paper's sampler.
 */
(function () {
  'use strict';
  var VE = (window.VE = window.VE || {});
  VE.components = VE.components || {};

  /* [logit q_eq, logit q_chain] on the exact Curie–Weiss chain, N = 2048, the test ramp.
     Source: data/odds/c_vs_bias.json (rows sorted by field G). Row 10 is the test's G = 6.125e-4. */
  var EXACT = [
    [0.1, 0.006581], [0.25, 0.016452], [0.693147, 0.045617], [1, 0.065814],
    [1.5, 0.09873], [2, 0.131658], [2.197225, 0.144651], [2.5, 0.164602],
    [3, 0.197564], [3.5, 0.23055], [3.574685, 0.235479], [3.583519, 0.236062],
    [4, 0.263562], [4.59512, 0.302896], [5, 0.329684], [5.5, 0.3628]
  ];
  var L_PAPER = 3.574685;           // logit q_eq at the test's G = 6.125e-4, beta = 1.6 (q_eq = 0.97274)
  var X_MAX = 5.6;                  // right end of the panel-b log-odds axis
  var Y_MAX = 0.2;                  // top of the panel-b share axis

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

  /* probability ticks on the log-odds axes */
  var P_TICKS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99];

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
      var S = { L: L_PAPER };
      function cOf() { return paperChain(S.L) / S.L; }

      /* ---------------------------------------------------------- frame */
      var fig = ui.figure({
        plate: opts.plate != null ? opts.plate : 3,
        title: 'Measuring the preference: the fraction c the sampler keeps',

      });
      root.appendChild(fig.el);
      var stage = fig.stage;

      function stepHead(text, extra) {
        // One panel-head style on the page: a small-caps letter, then the title in serif
        var m = /^(\w)\s·\s(.*)$/.exec(text);
        return el('div', { class: 'odds-step' }, m ? el('span', { class: 'odds-step__k', text: m[1] }) : null,
          el('span', { class: 'odds-step__t', text: m ? m[2] : text }), extra);
      }

      var key1 = el('span', { class: 'odds-key', html: '<i class="odds-sw odds-sw--a" aria-hidden="true"></i>mode A (favoured)<i class="odds-sw odds-sw--b" aria-hidden="true"></i>mode B' });
      var p1 = el('section', { class: 'odds-panel' },
        stepHead('a · Count the outcomes', [key1]));
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

      function paintArray(a, q, label) {
        var n = Math.round(100 * q);
        if (n !== a.last) {
          for (var i = 0; i < 100; i++) {
            var on = i < n;
            a.cells[i].className = 'odds-cell ' + (on ? 'is-a' : 'is-b');
          }
          a.last = n;
        }
        a.grid.setAttribute('aria-label', n + ' in mode A and ' + (100 - n) + ' in mode B, out of 100');
        a.sym.textContent = q.toFixed(3);
        a.sym.setAttribute('aria-label', label.replace(/<[^>]+>/g, '') + ' = ' + q.toFixed(3));
        a.foot.innerHTML = '<span class="ve-c-mode-a">' + n + '</span><span class="odds-foot__sep">:</span><span class="ve-c-mode-b">' + (100 - n) +
          '</span><span class="odds-odds">odds ' + oddsStr(q) + '</span>';
      }

      /* the kept fraction, one line under the counts */
      var formula = el('p', { class: 'odds-cline ve-num' });
      p1.appendChild(formula);

      /* ------------------------------------------- panel b: why log-odds */
      var p3 = el('section', { class: 'odds-panel' },
        stepHead('b · Why log-odds, not probability', [el('span', { class: 'odds-key', text: 'the same sampler at 16 fields' })]));
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
        var sy = ui.scale.linear([0, Y_MAX], [box.y1, box.y0]);
        // grid + y axis
        [0, 0.05, 0.1, 0.15, 0.2].forEach(function (v) {
          var y = Math.round(sy(v)) + 0.5;
          ctx.strokeStyle = v === 0 ? c.ruleStrong : c.grid; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(box.x0, y); ctx.lineTo(box.x1, y); ctx.stroke();
          ui.plot.text(ctx, v === 0 ? '0' : v.toFixed(2), box.x0 - 6, y, { size: 11, color: c.ink2, align: 'right', baseline: 'middle' });
        });
        ui.plot.text(ctx, 'fraction kept', box.x0 - (narrow ? 36 : 44), box.y0 - 20, { size: 11, color: c.ink3, baseline: 'middle' });
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
          : [['model’s probability of mode A, q', false], ['eq', true], [' (spaced as log-odds)', false]],
          box.x1, box.y1 + 33, { size: 11, color: c.ink3, align: 'right' });

        var xs = EXACT.map(function (r) { return r[0]; });
        var cLog = EXACT.map(function (r) { return r[1] / r[0]; });
        var cProb = EXACT.map(function (r) { return (expit(r[1]) - 0.5) / (expit(r[0]) - 0.5); });
        // current model preference
        // (the marker skips the band of the dashed curve's label so it never strikes through it)
        var xNow = Math.round(sx(S.L)) + 0.5;
        var tx = box.x0 + 8, ty = sy(0.178);
        var tA1 = 'in probability: drifts';
        ctx.font = ui.font('mono', 11, 400);
        var wA = ctx.measureText(tA1).width;
        var hitsLabel = xNow > tx - 4 && xNow < tx + wA + 4;
        ctx.save(); ctx.strokeStyle = c.bias; ctx.setLineDash([2, 3]); ctx.beginPath();
        if (hitsLabel) {
          ctx.moveTo(xNow, box.y0); ctx.lineTo(xNow, ty - 11);
          ctx.moveTo(xNow, ty + 10); ctx.lineTo(xNow, box.y1);
        } else { ctx.moveTo(xNow, box.y0); ctx.lineTo(xNow, box.y1); }
        ctx.stroke(); ctx.restore();
        ui.plot.line(ctx, xs, cProb, sx, sy, { color: c.noise, width: 2, dash: [5, 4] });
        ui.plot.line(ctx, xs, cLog, sx, sy, { color: c.highlightInk, width: 2.5 });
        ui.plot.dots(ctx, xs.map(function (x, i) { return { x: x, y: cProb[i] }; }), sx, sy, { r: 2.5, color: c.noise });
        ui.plot.dots(ctx, xs.map(function (x, i) { return { x: x, y: cLog[i] }; }), sx, sy, { r: 2.5, color: c.highlightInk });
        // direct labels on the two curves, in the empty upper-left and lower bands
        ui.plot.text(ctx, tA1, tx, ty, { size: 11, color: c.ink2, baseline: 'middle', halo: true });
        boxText(ui, ctx, 'in log-odds: flat', sx(1.0), sy(0.035),
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
        boxText(ui, ctx, vLog.toFixed(4), lx, yLog + 11, { weight: 600, color: c.highlightInk, align: al }, c);
        // the probability label must not run into the two-line annotation (phone width, strong preferences)
        ctx.font = ui.font('mono', 11, 600);
        var wP = ctx.measureText(vProb.toFixed(3)).width;
        var pX = lx, pY = yProb - 9, pAl = al;
        var hitsA = function (x, y, a) {
          var x0 = a === 'left' ? x : x - wP;
          return x0 - 3 < tx + wA + 4 && x0 + wP + 3 > tx - 4 && y - 9 < ty + 8 && y + 9 > ty - 9;
        };
        if (hitsA(pX, pY, pAl)) {
          var rx = sx(S.L) + 9;
          if (!right && rx + wP <= box.x1 && !hitsA(rx, pY, 'left')) { pX = rx; pAl = 'left'; }
          else if (yProb + 12 < yLog - 4) { pY = yProb + 12; }
        }
        boxText(ui, ctx, vProb.toFixed(3), pX, pY, { weight: 600, color: c.ink2, align: pAl }, c);
      }

      /* --------------------------------------------------------- render */
      function render() {
        var c = cOf(), L = S.L, lc = c * L;
        var qe = expit(L), qc = expit(lc);
        paintArray(arrModel, qe, 'q<sub>eq</sub>');
        paintArray(arrChain, qc, 'q<sub>chain</sub>');
        formula.innerHTML = '<i>c</i> = log-odds ' + lc.toFixed(4) + ' ÷ ' + L.toFixed(4) + ' = <b class="v2-c">' + c.toFixed(4) + '</b>';
        cv3.redraw();
      }

      render();

      return {
        destroy: function () {
          cv3.destroy();
          root.replaceChildren();
          root.classList.remove('ve-odds');
        }
      };
    }
  };
})();
