/* components/comparison.js — "The test: prediction against measurement" (chapter "The test")
 *
 * Registers VE.components['comparison'] = { mount(el, opts) }.   opts.plate: plate label.
 * One panel from data/table1.json (paper Table 1, results/simple/): measured ÷ predicted c for the
 * eight protocol rows, with 95% intervals, grouped by the kind of prediction (paper Fig. 2b).
 * No simulation and no number computed here except the axis placement.
 */
(function () {
  'use strict';
  var VE = (window.VE = window.VE || {});
  VE.components = VE.components || {};

  var NS = 'http://www.w3.org/2000/svg';
  var R0 = 0.86, R1 = 1.06;                       // ratio axis of panel a
  var GROUPS = [
    { p: [1], k: 'One pattern · exact chain' },
    { p: [2], k: 'Two patterns · exact: two Curie–Weiss blocks' },
    { p: [3, 4], k: 'Competition · A<sub>p</sub>c<sub>CW</sub>' }
  ];

  function mount(root, opts) {
    opts = opts || {};
    var ui = VE.ui, el = ui.el, fmt = ui.fmt;
    var f3 = function (x) { return fmt(x, { digits: 3 }); };
    var wrap = el('div', { class: 've-comparison' });
    root.appendChild(wrap);
    var fig = ui.figure({
      plate: opts.plate != null ? opts.plate : 'Fig. 6',
      title: 'The test: eight rows'
    });
    wrap.appendChild(fig.el);
    var destroyed = false, offTheme = null, rt = 0, redraw = null;
    var loading = el('p', { class: 've-comparison__loading', text: 'Loading Table 1…' });
    fig.stage.appendChild(loading);

    VE.sim.loadJSON('table1.json').then(function (T) {
      if (destroyed) return;
      loading.remove();
      build(T);
    }).catch(function (e) {
      console.error(e);
      loading.textContent = 'Could not load data/table1.json: ' + (e && e.message ? e.message : e);
      loading.className = 've-error';
    });

    function label(r) { return 'p = ' + r.p + ', N = ' + fmt(r.N); }

    function build(T) {
      var rows = T.rows;
      var panels = el('div', { class: 've-comparison__panels' });
      fig.stage.appendChild(panels);

      /* ------------------------------------------------ panel a: ratio rows */
      var pa = el('section', { class: 've-comparison__panel', 'aria-label': 'Measured ÷ predicted c, 95% intervals' });
      var host = el('div', { class: 've-cmp-rows', role: 'list', 'aria-label': 'Measured over predicted c, one row per setting' });
      pa.appendChild(host);
      panels.appendChild(pa);
      var items = [];
      GROUPS.forEach(function (g) {
        host.appendChild(el('div', { class: 've-cmp-group', html: g.k }));
        rows.filter(function (r) { return g.p.indexOf(r.p) >= 0; }).forEach(function (r) {
          var svg = document.createElementNS(NS, 'svg'); svg.setAttribute('aria-hidden', 'true');
          var b = el('div', { class: 've-cmp-row' + (r.p === 3 && r.N === 2048 ? ' is-main' : ''), role: 'listitem',
            'aria-label': label(r) + ': measured over predicted ' + f3(r.ratio) + ', 95% ' + f3(r.ratio_lo) + ' to ' + f3(r.ratio_hi) },
            el('span', { class: 've-cmp-row__l', text: r.p > 2 ? label(r) : 'N = ' + fmt(r.N) }),
            el('span', { class: 've-cmp-row__t' }, svg),
            el('span', { class: 've-cmp-row__v', text: f3(r.ratio) }));
          host.appendChild(b);
          items.push({ r: r, b: b, svg: svg });
        });
      });
      var axis = document.createElementNS(NS, 'svg'); axis.setAttribute('aria-hidden', 'true');
      host.appendChild(el('div', { class: 've-cmp-axis' }, el('span'), el('span', { class: 've-cmp-row__t' }, axis), el('span')));

      function E(tag, at, txt) { var n = document.createElementNS(NS, tag); for (var k in at) n.setAttribute(k, at[k]); if (txt != null) n.textContent = txt; return n; }
      function drawRows() {
        var c = ui.colors();
        items.concat([{ axis: true, svg: axis }]).forEach(function (it) {
          var svg = it.svg, W = svg.clientWidth || svg.getBoundingClientRect().width || 300, pad = 10;
          var sx = function (v) { return pad + (v - R0) / (R1 - R0) * (W - 2 * pad); };
          svg.replaceChildren();
          if (it.axis) {
            svg.setAttribute('viewBox', '0 0 ' + W + ' 22');
            [0.90, 0.95, 1.00, 1.05].forEach(function (v) { svg.appendChild(E('text', { x: sx(v), y: 15, 'text-anchor': 'middle' }, v.toFixed(2))); });
            return;
          }
          var r = it.r, y = 18;
          svg.setAttribute('viewBox', '0 0 ' + W + ' 36');
          svg.appendChild(E('line', { x1: sx(1), x2: sx(1), y1: 2, y2: 34, stroke: c.theory, 'stroke-width': 1.2, 'stroke-dasharray': '3 3' }));
          svg.appendChild(E('line', { x1: sx(r.ratio_lo), x2: sx(r.ratio_hi), y1: y, y2: y, stroke: c.data, 'stroke-width': 1.4 }));
          [r.ratio_lo, r.ratio_hi].forEach(function (v) { svg.appendChild(E('line', { x1: sx(v), x2: sx(v), y1: y - 4, y2: y + 4, stroke: c.data, 'stroke-width': 1.2 })); });
          svg.appendChild(E('circle', { cx: sx(r.ratio), cy: y, r: 3.8, fill: c.data }));
        });
      }
      redraw = drawRows;
      drawRows();
      offTheme = ui.onTheme(drawRows);
      window.addEventListener('resize', onResize);
    }
    function onResize() { cancelAnimationFrame(rt); rt = requestAnimationFrame(function () { if (!destroyed && redraw) redraw(); }); }

    return {
      destroy: function () {
        destroyed = true;
        window.removeEventListener('resize', onResize);
        if (offTheme) offTheme();
        wrap.remove();
      }
    };
  }

  VE.components['comparison'] = { mount: mount };
})();
