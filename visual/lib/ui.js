/* ==========================================================================
   Visual Edition — window.VE.ui   (lib/ui.js, plain classic script, no deps)
   Load order on any page:  fonts link -> lib/theme.css -> lib/ui.js -> lib/sim.js
                            -> components/<id>.js
   Every helper here works in light and dark themes and at 400 px width.

   ---------------------------------------------------------------- API ----
   NAMESPACE
     VE.components            registry: VE.components[id] = { mount(el, opts) }
     VE.ui.mountComponent(id, el, opts)   mount with a visible error box on failure

   DOM
     el(tag, attrs?, ...children) -> Element
        attrs: class, text, html, style (object or string), on: {event: fn},
        dataset: {...}, any other key -> setAttribute (false/null skipped).
        children: nodes, strings, arrays, null (skipped).
     uid(prefix?) -> unique id string
     icon(name) -> <svg> for 'play' | 'pause' | 'next' | 'prev' | 'reset' | 'step'

   NUMBERS  (minus sign is U+2212; output is meant for .ve-num / mono text)
     fmt(x, opts?)  opts: number = fixed digits, or {digits, sig, unit, sign,
                    sci: 'auto'|true|false, group: true}. NaN -> "–".
                    fmt(0.1996,{digits:3}) "0.200"; fmt(-0.5,{digits:2}) "−0.50";
                    fmt(16384) "16,384"; fmt(1.2e-4) "1.2×10⁻⁴"; fmt(77,{unit:'sweeps'}) "77 sweeps"
     fmtPow10Html(exp, mant?) -> "10<sup>1994</sup>" (for HTML readouts; mono superscript
                    glyphs are full-width, so prefer this in large type)
     sup(n) -> superscript digits string
     pct(x, digits=0) -> "12%"

   THEME AND COLOUR
     colors(scopeEl?) -> object of resolved token values, camelCased:
        bg surface surface2 ink ink2 ink3 rule ruleStrong grid focus
        modeA modeB noise bias theory data highlight highlightInk
        modeAWash modeBWash noiseWash biasWash theoryWash highlightWash
        fontDisplay fontText fontMono
        Cached; the cache is cleared automatically when the theme changes.
     theme() -> 'light' | 'dark'  (resolved: data-theme first, then the OS)
     setTheme('light'|'dark'|'system')   sets/removes data-theme on <html>
     onTheme(cb) -> unsubscribe   cb(theme) after any theme change
     alpha(color, a) -> 'rgba(...)' from '#rrggbb' or 'rgb(...)'

   MOTION
     reducedMotion -> boolean (live getter)
     onReducedMotion(cb) -> unsubscribe
     loop(fn, {el?, autostart?}) -> {start(), stop(), toggle(), running, visible}
        fn(dtSeconds, tSeconds) each animation frame; dt is clamped to 0.1 s.
        The loop sleeps while `el` is off-screen (IntersectionObserver) or the
        tab is hidden, and wakes when it comes back. autostart:true starts it
        unless the reader prefers reduced motion.

   CANVAS AND PLOTTING
     canvas(parent, {height?, aspect?, minHeight?, maxHeight?, draw(ctx,w,h,c),
            label?, drag?: 'x'|'xy'}) -> {canvas, ctx, width, height, dpr,
            redraw(), local(evt) -> {x,y}, destroy()}
        Sizes the backing store to devicePixelRatio, draws in CSS pixels,
        redraws on resize, on DPR change, on theme change and when fonts load.
        height: px; or aspect: width/height (with min/maxHeight clamps).
        drag:'xy' sets touch-action:none (for canvases you drag in 2D).
     scale.linear([d0,d1],[r0,r1]) and scale.log(...) -> f(x)
        f.invert(px), f.ticks(n), f.domain, f.range, f.clamp(true) -> f
     ticks(a, b, n) -> nice linear ticks; logTicks(a, b) -> powers of 10 (and 2,5)
     inset(w, h, {top,right,bottom,left}) -> {x0,x1,y0,y1,w,h} plot box
     font(role, sizePx, weight?) -> canvas font string; role: 'mono'|'text'|'display'
     plot.axis(ctx, scale, {side:'bottom'|'left'|'top'|'right', at, ticks,
               format, label, grid:[from,to], c})
     plot.line(ctx, xs, ys, sx, sy, {color, width, dash, alpha})
     plot.fn(ctx, f, sx, sy, {n, color, width, dash})       sample y = f(x)
     plot.band(ctx, xs, lo, hi, sx, sy, {fill})
     plot.dots(ctx, pts:[{x,y}], sx, sy, {r, color, stroke, ring})
     plot.errorbar(ctx, x, lo, hi, sx, sy, {color, width, cap, horizontal})
     plot.arrow(ctx, x0, y0, x1, y1, {color, width, head})   pixel coords
     plot.text(ctx, str, x, y, {role, size, weight, color, align, baseline,
               halo})  halo:true draws a surface-coloured outline for legibility
     plot.vline / plot.hline(ctx, v, scale, from, to, {color, dash, width})

   CONTROLS (all return {el, ...}; all keyboard and touch accessible)
     slider({id, label, sym?, min, max, value, step?, log?, unit?, format?,
             marks?:[{value,label}], tone?, onInput(v), onChange(v)})
        -> {el, input, value (get/set), set(v, {silent}), refresh()}
        log:true maps the track logarithmically (min > 0). tone: a role key
        ('mode-a','bias',...) to tint the fader. Readout shows format(v)+unit.
     toggle({id, label, options:[{value,label,title?,tone?}], value, onChange})
        -> {el, value (get), set(v, {silent})}   radiogroup with arrow keys
     button({label, icon?, kind?:'primary'|'ghost'|'icon', onClick, title?})
     player({id, duration, value?, speed?, loop?, chapters?:[{t,label}],
             format?(t), label?, autoplay?, onTime(t, playing), onEnd()})
        -> {el, play(), pause(), toggle(), seek(t), time, playing, destroy()}
        speed = timeline units per second. Space/K toggles while focused.
     readout({label, value?, unit?, tone?, sub?, format?, html?}) -> {el, set(v, sub?)}
        html:true sets the formatted value as HTML (e.g. fmtPow10Html for 10<sup>1994</sup>)
     legend(items:[{role|color, label, mark:'line'|'dash'|'dot'|'ring'|'square'|
             'band'|'area'|'arrow'|'wash'}]) -> element

   FIGURE FRAME
     figure({plate?, title, caption?, takeaway?, legend?, note?, noteKind?,
             paper?}) -> {el, head, stage, controls, legend, caption,
             takeaway, note, setCaption(html), setTakeaway(text),
             setNote(text, kind?)}
        note: the instrument line under a plate, e.g. "Live simulation · N = 256 ·
        linear schedule · 400 chains". noteKind 'live' (green dot) | 'paper'
        (square: numbers from the paper).

   GLOSSARY
     glossary: {key: {word, sym?, ml, phys?}}  ML meanings of physics words
     term(text, key) -> <button class="ve-term"> with a tap/hover definition
     terms(root) -> upgrades every [data-term="key"] element under root
     defineTerm(key, {word, sym, ml, phys})   add or replace an entry

   DISCLOSURE PRIMITIVES. Nothing nests deeper than one
   level. All work with JS off for reading (drawers are native <details>).
     info(trigger, {id, title, html|node, kind?}) -> {el, open(), close()}
     specCard(trigger, noteEl, {id?, title?}) -> {el, disconnect()}
     mirror(sourceEl, targetEl) -> disconnect()
     drawers(root)            enhance details.v2-drawer (icon, resize on open)
     openDrawer(id)           open one drawer and scroll it into view
     notes(root)              aside.v2-note[data-q] -> toggle below 1180 px
     lazyPlate(details, componentId, opts)   mount on first open
     allDetails(on?) -> bool  open every drawer, note and provenance block
     onAllDetails(cb) -> unsubscribe
     math(root)               KaTeX for [data-tex] and [data-tex-display]

   =========================================================================== */
(function () {
  'use strict';
  var VE = (window.VE = window.VE || {});
  VE.components = VE.components || {};
  var ui = (VE.ui = VE.ui || {});


  /* ------------------------------------------------------------------ DOM */
  function el(tag, attrs) {
    var node = document.createElement(tag);
    if (attrs) {
      for (var k in attrs) {
        if (!Object.prototype.hasOwnProperty.call(attrs, k)) continue;
        var v = attrs[k];
        if (v === null || v === undefined || v === false) continue;
        if (k === 'class') node.className = v;
        else if (k === 'text') node.textContent = v;
        else if (k === 'html') node.innerHTML = v;
        else if (k === 'style') {
          if (typeof v === 'string') node.style.cssText = v;
          else for (var s in v) { if (s.indexOf('--') === 0) node.style.setProperty(s, v[s]); else node.style[s] = v[s]; }
        } else if (k === 'on') { for (var e in v) node.addEventListener(e, v[e]); }
        else if (k === 'dataset') { for (var d in v) node.dataset[d] = v[d]; }
        else node.setAttribute(k, v === true ? '' : v);
      }
    }
    for (var i = 2; i < arguments.length; i++) append(node, arguments[i]);
    return node;
  }
  function append(node, c) {
    if (c === null || c === undefined || c === false) return;
    if (Array.isArray(c)) { c.forEach(function (x) { append(node, x); }); return; }
    node.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  ui.el = el;
  var _uid = 0;
  ui.uid = function (p) { return (p || 've') + '-' + (++_uid); };

  var ICONS = {
    play: 'M4 2.5v11l9.5-5.5z',
    pause: 'M3.5 2.5h3v11h-3zM9.5 2.5h3v11h-3z',
    next: 'M2.5 2.5v11l7-5.5zM10.5 2.5h2.5v11h-2.5z',
    prev: 'M13.5 2.5v11l-7-5.5zM3 2.5h2.5v11H3z',
    step: 'M3 2.5v11l7-5.5zM10 2.5v11l5-5.5z',
    reset: 'M8 2a6 6 0 1 1-5.66 8h1.8A4.3 4.3 0 1 0 8 3.7V6L4.5 3 8 0z'
  };
  ui.icon = function (name) {
    var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 16 16');
    svg.setAttribute('aria-hidden', 'true');
    var p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    p.setAttribute('d', ICONS[name] || ICONS.play);
    svg.appendChild(p);
    return svg;
  };

  /* -------------------------------------------------------------- NUMBERS */
  var SUP = { '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻', '+': '' };
  var MINUS = '−';
  ui.sup = function (n) { return String(n).split('').map(function (ch) { return SUP[ch] !== undefined ? SUP[ch] : ch; }).join(''); };
  function group(s) {
    var parts = s.split('.');
    parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    return parts.join('.');
  }
  function fmt(x, opts) {
    if (typeof opts === 'number') opts = { digits: opts };
    opts = opts || {};
    var unit = opts.unit ? ' ' + opts.unit : '';
    if (x === null || x === undefined || isNaN(x)) return '–' + unit;
    if (!isFinite(x)) return (x < 0 ? MINUS : '') + '∞' + unit;
    var ax = Math.abs(x);
    var neg = x < 0;
    var sci = opts.sci === undefined ? 'auto' : opts.sci;
    var useSci = sci === true || (sci === 'auto' && ax !== 0 && (ax >= 1e7 || ax < 1e-3));
    var body;
    if (useSci) {
      var e = Math.floor(Math.log10(ax));
      var m = ax / Math.pow(10, e);
      var sig = opts.sig || 2;
      var ms = m.toPrecision(sig);
      if (parseFloat(ms) >= 10) { e += 1; ms = (m / 10).toPrecision(sig); }
      ms = ms.replace(/\.?0+$/, '');
      body = (ms === '1' ? '' : ms + '×') + '10' + ui.sup(e);
    } else if (opts.digits !== undefined) {
      body = ax.toFixed(opts.digits);
    } else if (opts.sig !== undefined) {
      body = String(parseFloat(ax.toPrecision(opts.sig)));
      if (/e/.test(body)) body = ax.toFixed(Math.max(0, opts.sig - 1));
    } else {
      body = Number.isInteger(x) ? String(ax) : String(parseFloat(ax.toPrecision(4)));
    }
    if (!useSci && opts.group !== false) body = group(body);
    if (parseFloat(body.replace(/,/g, '')) === 0 && !useSci) neg = false;
    var sign = neg ? MINUS : (opts.sign && x > 0 ? '+' : '');
    return sign + body + unit;
  }
  ui.fmt = fmt;
  ui.pct = function (x, d) { return fmt(100 * x, { digits: d || 0 }) + '%'; };
  ui.fmtPow10Html = function (exp, mant) { return (mant ? mant + '×' : '') + '10<sup>' + String(exp).replace('-', '\u2212') + '</sup>'; };

  /* -------------------------------------------------------- THEME / COLOR */
  var TOKENS = ['bg', 'surface', 'surface-2', 'ink', 'ink-2', 'ink-3', 'rule', 'rule-strong', 'grid', 'focus',
    'mode-a', 'mode-b', 'noise', 'bias', 'theory', 'data', 'highlight', 'highlight-ink',
    'mode-a-wash', 'mode-b-wash', 'noise-wash', 'bias-wash', 'theory-wash', 'highlight-wash',
    'font-display', 'font-text', 'font-mono'];
  function camel(s) { return s.replace(/-([a-z0-9])/g, function (_, c) { return c.toUpperCase(); }); }
  var colorCache = new WeakMap();
  ui.colors = function (scopeEl) {
    var node = scopeEl || document.documentElement;
    var hit = colorCache.get(node);
    if (hit) return hit;
    var cs = getComputedStyle(node);
    var out = {};
    TOKENS.forEach(function (t) { out[camel(t)] = cs.getPropertyValue('--' + t).trim(); });
    if (node.isConnected && out.bg) colorCache.set(node, out); // never cache a detached read
    return out;
  };
  var darkMQ = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : { matches: false };
  ui.theme = function () {
    var t = document.documentElement.getAttribute('data-theme');
    if (t === 'light' || t === 'dark') return t;
    return darkMQ.matches ? 'dark' : 'light';
  };
  ui.setTheme = function (t) {
    if (t === 'light' || t === 'dark') document.documentElement.setAttribute('data-theme', t);
    else document.documentElement.removeAttribute('data-theme');
  };
  var themeSubs = new Set();
  function themeChanged() {
    colorCache = new WeakMap();
    var t = ui.theme();
    themeSubs.forEach(function (cb) { try { cb(t); } catch (e) { console.error(e); } });
  }
  ui.onTheme = function (cb) { themeSubs.add(cb); return function () { themeSubs.delete(cb); }; };
  if (darkMQ.addEventListener) darkMQ.addEventListener('change', themeChanged);
  if (window.MutationObserver) {
    new MutationObserver(themeChanged).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'class'] });
  }

  ui.alpha = function (color, a) {
    color = (color || '').trim();
    var r, g, b;
    if (color[0] === '#') {
      var h = color.slice(1);
      if (h.length === 3) h = h.split('').map(function (c) { return c + c; }).join('');
      r = parseInt(h.slice(0, 2), 16); g = parseInt(h.slice(2, 4), 16); b = parseInt(h.slice(4, 6), 16);
    } else {
      var m = color.match(/[\d.]+/g);
      if (!m) return color;
      r = +m[0]; g = +m[1]; b = +m[2];
    }
    return 'rgba(' + r + ',' + g + ',' + b + ',' + a + ')';
  };

  /* --------------------------------------------------------------- MOTION */
  var rmMQ = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : { matches: false };
  Object.defineProperty(ui, 'reducedMotion', { get: function () { return !!rmMQ.matches; }, enumerable: true });
  var rmSubs = new Set();
  if (rmMQ.addEventListener) rmMQ.addEventListener('change', function () { rmSubs.forEach(function (cb) { cb(!!rmMQ.matches); }); });
  ui.onReducedMotion = function (cb) { rmSubs.add(cb); return function () { rmSubs.delete(cb); }; };

  ui.loop = function (fn, opts) {
    opts = opts || {};
    var wanted = false, visible = true, raf = 0, last = 0, t = 0;
    var api = {};
    function frame(now) {
      raf = 0;
      if (!wanted || !visible || document.hidden) { last = 0; return; }
      var dt = last ? Math.min(0.1, (now - last) / 1000) : 0;
      last = now; t += dt;
      try { fn(dt, t); } catch (e) { console.error(e); wanted = false; return; }
      if (wanted) raf = requestAnimationFrame(frame);
    }
    function kick() { if (wanted && visible && !document.hidden && !raf) { last = 0; raf = requestAnimationFrame(frame); } }
    api.start = function () { wanted = true; kick(); return api; };
    api.stop = function () { wanted = false; if (raf) cancelAnimationFrame(raf); raf = 0; last = 0; return api; };
    api.toggle = function () { return wanted ? api.stop() : api.start(); };
    Object.defineProperty(api, 'running', { get: function () { return wanted; } });
    Object.defineProperty(api, 'visible', { get: function () { return visible; } });
    if (opts.el && window.IntersectionObserver) {
      var io = new IntersectionObserver(function (es) {
        visible = es[es.length - 1].isIntersecting;
        kick();
      }, { rootMargin: '80px' });
      io.observe(opts.el);
      api.destroy = function () { api.stop(); io.disconnect(); };
    } else api.destroy = api.stop;
    document.addEventListener('visibilitychange', kick);
    if (opts.autostart && !ui.reducedMotion) api.start();
    return api;
  };

  /* --------------------------------------------------------------- CANVAS */
  var canvases = new Set();
  ui.font = function (role, size, weight) {
    var c = ui.colors();
    var fam = role === 'display' ? c.fontDisplay : role === 'text' ? c.fontText : c.fontMono;
    return (weight || 400) + ' ' + (size || 12) + 'px ' + (fam || 'monospace');
  };
  ui.canvas = function (parent, o) {
    o = o || {};
    var wrap = el('div', { class: 've-canvas', dataset: o.drag === 'xy' ? { drag: 'xy' } : {} });
    var cv = el('canvas', { role: 'img', 'aria-label': o.label || '' });
    wrap.appendChild(cv);
    if (parent) parent.appendChild(wrap);
    var ctx = cv.getContext('2d');
    var api = { canvas: cv, ctx: ctx, wrap: wrap, width: 0, height: 0, dpr: 1 };
    var pending = 0;
    function size() {
      var w = Math.max(1, Math.round(wrap.clientWidth));
      var h;
      if (o.height) h = typeof o.height === 'function' ? o.height(w) : o.height;
      else h = w / (o.aspect || 1.6);
      if (o.minHeight) h = Math.max(o.minHeight, h);
      if (o.maxHeight) h = Math.min(o.maxHeight, h);
      h = Math.round(h);
      var dpr = Math.min(3, window.devicePixelRatio || 1);
      wrap.style.height = h + 'px';
      if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
        cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
      }
      api.width = w; api.height = h; api.dpr = dpr;
    }
    api.redraw = function () {
      if (!wrap.isConnected) { schedule(); return; }
      if (!api.width) size();
      ctx.setTransform(api.dpr, 0, 0, api.dpr, 0, 0);
      ctx.clearRect(0, 0, api.width, api.height);
      if (o.draw) o.draw(ctx, api.width, api.height, ui.colors(), api);
    };
    function schedule() {
      if (pending) return;
      pending = requestAnimationFrame(function () { pending = 0; if (!wrap.isConnected) return; size(); api.redraw(); });
    }
    api.resize = schedule;
    api.local = function (evt) {
      var r = cv.getBoundingClientRect();
      var p = evt.touches ? evt.touches[0] : evt;
      return { x: p.clientX - r.left, y: p.clientY - r.top };
    };
    var ro = window.ResizeObserver ? new ResizeObserver(schedule) : null;
    if (ro) ro.observe(wrap); else window.addEventListener('resize', schedule);
    var offTheme = ui.onTheme(schedule);
    var dprMQ, onDpr = function () { schedule(); watchDpr(); };
    function watchDpr() {
      if (!window.matchMedia) return;
      if (dprMQ) dprMQ.removeEventListener('change', onDpr);
      dprMQ = window.matchMedia('(resolution: ' + (window.devicePixelRatio || 1) + 'dppx)');
      dprMQ.addEventListener('change', onDpr, { once: true });
    }
    watchDpr();
    api.destroy = function () { if (ro) ro.disconnect(); offTheme(); canvases.delete(api); wrap.remove(); };
    canvases.add(api);
    if (parent && wrap.isConnected) { size(); api.redraw(); }
    else schedule();
    return api;
  };
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(function () { canvases.forEach(function (c) { c.resize(); }); });
  }

  /* --------------------------------------------------------------- SCALES */
  function niceStep(span, n) {
    var raw = span / Math.max(1, n);
    var p = Math.pow(10, Math.floor(Math.log10(raw)));
    var f = raw / p;
    return (f >= 7.5 ? 10 : f >= 3.5 ? 5 : f >= 1.5 ? 2 : 1) * p;
  }
  ui.ticks = function (a, b, n) {
    var lo = Math.min(a, b), hi = Math.max(a, b);
    if (lo === hi) return [lo];
    var st = niceStep(hi - lo, n || 5);
    var out = [];
    var start = Math.ceil(lo / st - 1e-9) * st;
    for (var v = start; v <= hi + st * 1e-9; v += st) out.push(Math.abs(v) < st * 1e-9 ? 0 : +v.toPrecision(12));
    return out;
  };
  ui.logTicks = function (a, b, withSub) {
    var lo = Math.min(a, b), hi = Math.max(a, b), out = [];
    for (var e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
      [1, 2, 5].forEach(function (m) {
        if (m !== 1 && !withSub) return;
        var v = m * Math.pow(10, e);
        if (v >= lo * (1 - 1e-9) && v <= hi * (1 + 1e-9)) out.push(v);
      });
    }
    return out;
  };
  function makeScale(kind, domain, range) {
    var d0 = domain[0], d1 = domain[1], r0 = range[0], r1 = range[1], clampOn = false;
    var tf = kind === 'log' ? Math.log : function (x) { return x; };
    var ti = kind === 'log' ? Math.exp : function (x) { return x; };
    function f(x) {
      var u = (tf(x) - tf(d0)) / (tf(d1) - tf(d0));
      if (clampOn) u = Math.max(0, Math.min(1, u));
      return r0 + u * (r1 - r0);
    }
    f.invert = function (px) {
      var u = (px - r0) / (r1 - r0);
      if (clampOn) u = Math.max(0, Math.min(1, u));
      return ti(tf(d0) + u * (tf(d1) - tf(d0)));
    };
    f.ticks = function (n) { return kind === 'log' ? ui.logTicks(d0, d1, n > 4) : ui.ticks(d0, d1, n); };
    f.domain = [d0, d1];
    f.range = [r0, r1];
    f.kind = kind;
    f.clamp = function (on) { clampOn = on !== false; return f; };
    return f;
  }
  ui.scale = {
    linear: function (d, r) { return makeScale('linear', d, r); },
    log: function (d, r) { return makeScale('log', d, r); }
  };
  ui.inset = function (w, h, m) {
    m = m || {};
    var t = m.top != null ? m.top : 12, r = m.right != null ? m.right : 12, b = m.bottom != null ? m.bottom : 34, l = m.left != null ? m.left : 44;
    return { x0: l, x1: w - r, y0: t, y1: h - b, w: w - l - r, h: h - t - b };
  };

  /* ---------------------------------------------------------------- PLOT */
  var plot = (ui.plot = {});
  plot.text = function (ctx, str, x, y, o) {
    o = o || {};
    var c = o.c || ui.colors();
    ctx.save();
    ctx.font = ui.font(o.role || 'mono', o.size || 11, o.weight || 400);
    ctx.textAlign = o.align || 'left';
    ctx.textBaseline = o.baseline || 'alphabetic';
    if (o.halo) {
      ctx.lineJoin = 'round';
      ctx.lineWidth = o.haloWidth || 4;
      ctx.strokeStyle = o.haloColor || c.surface;
      ctx.strokeText(str, x, y);
    }
    ctx.fillStyle = o.color || c.ink2;
    ctx.fillText(str, x, y);
    ctx.restore();
  };
  plot.axis = function (ctx, sc, o) {
    o = o || {};
    var c = o.c || ui.colors();
    var side = o.side || 'bottom';
    var horiz = side === 'bottom' || side === 'top';
    var ticks = Array.isArray(o.ticks) ? o.ticks : sc.ticks(o.ticks || 5);
    var format = o.format || function (v) { return fmt(v); };
    var at = o.at;
    var r = sc.range;
    ctx.save();
    if (o.grid) {
      ctx.strokeStyle = c.grid; ctx.lineWidth = 1;
      ticks.forEach(function (v) {
        var p = Math.round(sc(v)) + 0.5;
        ctx.beginPath();
        if (horiz) { ctx.moveTo(p, o.grid[0]); ctx.lineTo(p, o.grid[1]); } else { ctx.moveTo(o.grid[0], p); ctx.lineTo(o.grid[1], p); }
        ctx.stroke();
      });
    }
    ctx.strokeStyle = c.ruleStrong; ctx.lineWidth = 1;
    ctx.beginPath();
    var a = Math.round(at) + 0.5;
    if (horiz) { ctx.moveTo(Math.min(r[0], r[1]), a); ctx.lineTo(Math.max(r[0], r[1]), a); }
    else { ctx.moveTo(a, Math.min(r[0], r[1])); ctx.lineTo(a, Math.max(r[0], r[1])); }
    ctx.stroke();
    var dir = side === 'bottom' || side === 'right' ? 1 : -1;
    ctx.font = ui.font('mono', o.size || 10.5);
    ctx.fillStyle = c.ink3;
    ticks.forEach(function (v) {
      var p = Math.round(sc(v)) + 0.5;
      ctx.beginPath();
      if (horiz) { ctx.moveTo(p, a); ctx.lineTo(p, a + 5 * dir); } else { ctx.moveTo(a, p); ctx.lineTo(a + 5 * dir, p); }
      ctx.stroke();
      var s = format(v);
      if (s === '' || s == null) return;
      if (horiz) { ctx.textAlign = 'center'; ctx.textBaseline = dir > 0 ? 'top' : 'bottom'; ctx.fillText(s, p, a + 8 * dir); }
      else { ctx.textAlign = dir > 0 ? 'left' : 'right'; ctx.textBaseline = 'middle'; ctx.fillText(s, a + 8 * dir, p); }
    });
    if (o.label) {
      ctx.fillStyle = c.ink2;
      ctx.font = ui.font('mono', o.labelSize || 11, 500);
      if (horiz) {
        ctx.textAlign = o.labelAlign || 'right'; ctx.textBaseline = dir > 0 ? 'top' : 'bottom';
        var lx = o.labelAlign === 'center' ? (r[0] + r[1]) / 2 : o.labelAlign === 'left' ? Math.min(r[0], r[1]) : Math.max(r[0], r[1]);
        ctx.fillText(o.label, lx, a + 22 * dir);
      } else {
        ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
        ctx.fillText(o.label, o.labelX != null ? o.labelX : Math.max(2, a - 40), Math.min(r[0], r[1]) - 12);
      }
    }
    ctx.restore();
  };
  plot.line = function (ctx, xs, ys, sx, sy, o) {
    o = o || {};
    ctx.save();
    ctx.strokeStyle = o.color || ui.colors().ink;
    ctx.lineWidth = o.width || 2;
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    if (o.alpha != null) ctx.globalAlpha = o.alpha;
    if (o.dash) ctx.setLineDash(o.dash);
    ctx.beginPath();
    var started = false;
    for (var i = 0; i < xs.length; i++) {
      var y = ys[i];
      if (y == null || !isFinite(y)) { started = false; continue; }
      var px = sx(xs[i]), py = sy(y);
      if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.restore();
  };
  plot.fn = function (ctx, f, sx, sy, o) {
    o = o || {};
    var n = o.n || 200, d = sx.domain, xs = [], ys = [];
    for (var i = 0; i <= n; i++) {
      var u = i / n;
      var x = sx.kind === 'log' ? Math.exp(Math.log(d[0]) + u * (Math.log(d[1]) - Math.log(d[0]))) : d[0] + u * (d[1] - d[0]);
      xs.push(x); ys.push(f(x));
    }
    plot.line(ctx, xs, ys, sx, sy, o);
  };
  plot.band = function (ctx, xs, lo, hi, sx, sy, o) {
    o = o || {};
    ctx.save();
    ctx.fillStyle = o.fill || ui.colors().noiseWash;
    ctx.beginPath();
    for (var i = 0; i < xs.length; i++) { var px = sx(xs[i]), py = sy(hi[i]); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
    for (var j = xs.length - 1; j >= 0; j--) ctx.lineTo(sx(xs[j]), sy(lo[j]));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  };
  plot.dots = function (ctx, pts, sx, sy, o) {
    o = o || {};
    var c = ui.colors();
    var r = o.r || 3.5;
    ctx.save();
    ctx.lineWidth = o.strokeWidth || 1.5;
    pts.forEach(function (p) {
      ctx.beginPath();
      ctx.arc(sx(p.x), sy(p.y), p.r || r, 0, Math.PI * 2);
      if (o.ring) { ctx.fillStyle = c.surface; ctx.fill(); ctx.strokeStyle = p.color || o.color || c.data; ctx.stroke(); }
      else {
        ctx.fillStyle = p.color || o.color || c.data; ctx.fill();
        if (o.stroke !== false) { ctx.strokeStyle = o.stroke || c.surface; ctx.stroke(); }
      }
    });
    ctx.restore();
  };
  plot.errorbar = function (ctx, x, lo, hi, sx, sy, o) {
    o = o || {};
    var cap = o.cap == null ? 4 : o.cap;
    ctx.save();
    ctx.strokeStyle = o.color || ui.colors().data;
    ctx.lineWidth = o.width || 1.5;
    ctx.beginPath();
    if (o.horizontal) {
      var y = sy(x), a = sx(lo), b = sx(hi);
      ctx.moveTo(a, y); ctx.lineTo(b, y);
      ctx.moveTo(a, y - cap); ctx.lineTo(a, y + cap); ctx.moveTo(b, y - cap); ctx.lineTo(b, y + cap);
    } else {
      var px = sx(x), p0 = sy(lo), p1 = sy(hi);
      ctx.moveTo(px, p0); ctx.lineTo(px, p1);
      ctx.moveTo(px - cap, p0); ctx.lineTo(px + cap, p0); ctx.moveTo(px - cap, p1); ctx.lineTo(px + cap, p1);
    }
    ctx.stroke();
    ctx.restore();
  };
  plot.arrow = function (ctx, x0, y0, x1, y1, o) {
    o = o || {};
    var head = o.head || 8, w = o.width || 2;
    var ang = Math.atan2(y1 - y0, x1 - x0), len = Math.hypot(x1 - x0, y1 - y0);
    if (len < 0.5) return;
    var hl = Math.min(head, len);
    ctx.save();
    ctx.strokeStyle = ctx.fillStyle = o.color || ui.colors().bias;
    ctx.lineWidth = w; ctx.lineCap = 'round';
    if (o.dash) ctx.setLineDash(o.dash);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1 - Math.cos(ang) * hl * 0.8, y1 - Math.sin(ang) * hl * 0.8);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - hl * Math.cos(ang - 0.42), y1 - hl * Math.sin(ang - 0.42));
    ctx.lineTo(x1 - hl * Math.cos(ang + 0.42), y1 - hl * Math.sin(ang + 0.42));
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  };
  plot.vline = function (ctx, v, sc, from, to, o) {
    o = o || {};
    var p = Math.round(sc(v)) + 0.5;
    ctx.save();
    ctx.strokeStyle = o.color || ui.colors().ink3; ctx.lineWidth = o.width || 1;
    if (o.dash !== false) ctx.setLineDash(o.dash || [4, 3]);
    ctx.beginPath(); ctx.moveTo(p, from); ctx.lineTo(p, to); ctx.stroke();
    ctx.restore();
  };
  plot.hline = function (ctx, v, sc, from, to, o) {
    o = o || {};
    var p = Math.round(sc(v)) + 0.5;
    ctx.save();
    ctx.strokeStyle = o.color || ui.colors().ink3; ctx.lineWidth = o.width || 1;
    if (o.dash !== false) ctx.setLineDash(o.dash || [4, 3]);
    ctx.beginPath(); ctx.moveTo(from, p); ctx.lineTo(to, p); ctx.stroke();
    ctx.restore();
  };

  /* ------------------------------------------------------------- CONTROLS */
  function toneVar(tone) { return tone ? 'var(--' + tone + ')' : null; }
  ui.button = function (o) {
    o = o || {};
    var cls = 've-btn' + (o.kind === 'primary' ? ' ve-btn--primary' : o.kind === 'ghost' ? ' ve-btn--ghost' : '') + (o.kind === 'icon' || (!o.label && o.icon) ? ' ve-btn--icon' : '');
    var b = el('button', { type: 'button', class: cls, id: o.id, title: o.title, 'aria-label': o.ariaLabel || (!o.label ? o.title : null) },
      o.icon ? ui.icon(o.icon) : null, o.label ? el('span', { text: o.label }) : null);
    if (o.onClick) b.addEventListener('click', o.onClick);
    return b;
  };

  ui.slider = function (o) {
    o = o || {};
    var id = o.id || ui.uid('sl');
    var log = !!o.log;
    var min = o.min, max = o.max;
    var RES = 1000;
    var format = o.format || function (v) { return fmt(v, { sig: 3 }); };
    var value = o.value != null ? o.value : min;
    var input = el('input', { type: 'range', class: 've-range', id: id });
    if (log) { input.min = 0; input.max = RES; input.step = 1; }
    else { input.min = min; input.max = max; input.step = o.step || 'any'; }
    var labelEl = el('label', { class: 've-slider__label', for: id });
    if (o.sym) append(labelEl, [el('span', { class: 've-sym', text: o.sym }), ' ']);
    append(labelEl, o.label || '');
    var out = el('output', { class: 've-readout-box', for: id, 'aria-hidden': 'true' });
    var valSpan = el('span'), unitSpan = o.unit ? el('span', { class: 've-unit', text: o.unit }) : null;
    append(out, [valSpan, unitSpan]);
    var marksEl = el('div', { class: 've-slider__marks', 'aria-hidden': 'true' });
    var body = el('div', { class: 've-slider__body' }, input, el('div', { class: 've-slider__scale', 'aria-hidden': 'true' }), marksEl);
    var root = el('div', { class: 've-slider', dataset: o.tone ? { tone: o.tone } : {}, style: o.tone ? { '--tone': toneVar(o.tone) } : null },
      el('div', { class: 've-slider__top' }, labelEl, out), body,
      o.hint ? el('div', { class: 've-slider__hint', text: o.hint }) : null);
    function toPos(v) { return log ? Math.round(RES * Math.log(v / min) / Math.log(max / min)) : v; }
    function fromPos(p) {
      if (!log) return +p;
      var v = min * Math.pow(max / min, p / RES);
      if (o.step) v = Math.round(v / o.step) * o.step;
      return +v.toPrecision(6);
    }
    function frac(v) { return log ? Math.log(v / min) / Math.log(max / min) : (v - min) / (max - min); }
    function paint() {
      var s = format(value);
      valSpan.textContent = s;
      input.setAttribute('aria-valuetext', s + (o.unit ? ' ' + o.unit : ''));
      root.style.setProperty('--p', (100 * Math.max(0, Math.min(1, frac(value)))).toFixed(2) + '%');
    }
    (o.marks || []).forEach(function (m) {
      var u = Math.max(0, Math.min(1, frac(m.value)));
      var mk = el('span', { class: 've-slider__mark', style: { left: (100 * u) + '%' }, dataset: u < 0.06 ? { edge: 'start' } : u > 0.94 ? { edge: 'end' } : {} },
        m.label ? el('span', { class: 've-slider__marklabel', text: m.label }) : null);
      marksEl.appendChild(mk);
      if (m.label) root.classList.add('has-marklabels');
    });
    input.value = toPos(value);
    paint();
    input.addEventListener('input', function () {
      value = fromPos(input.value);
      paint();
      if (o.onInput) o.onInput(value);
    });
    input.addEventListener('change', function () { if (o.onChange) o.onChange(value); });
    var api = {
      el: root, input: input,
      set: function (v, opt) {
        value = Math.max(min, Math.min(max, v));
        input.value = toPos(value);
        paint();
        if (!(opt && opt.silent) && o.onInput) o.onInput(value);
      },
      refresh: paint
    };
    Object.defineProperty(api, 'value', { get: function () { return value; }, set: function (v) { api.set(v); } });
    return api;
  };

  ui.toggle = function (o) {
    o = o || {};
    var id = o.id || ui.uid('tg');
    var value = o.value != null ? o.value : o.options[0].value;
    var labelId = id + '-label';
    var group = el('div', { class: 've-toggle__group', role: 'radiogroup', 'aria-labelledby': o.label ? labelId : null, 'aria-label': o.label ? null : (o.ariaLabel || 'Options'), id: id });
    var btns = o.options.map(function (opt, i) {
      var b = el('button', {
        type: 'button', class: 've-toggle__opt', role: 'radio', title: opt.title,
        dataset: opt.tone ? { tone: opt.tone, value: String(opt.value) } : { value: String(opt.value) },
        style: opt.tone ? { '--tone': toneVar(opt.tone) } : null
      });
      append(b, opt.label);
      b.addEventListener('click', function () { set(opt.value); });
      b.addEventListener('keydown', function (e) {
        var k = e.key, j = -1;
        if (k === 'ArrowRight' || k === 'ArrowDown') j = (i + 1) % btns.length;
        else if (k === 'ArrowLeft' || k === 'ArrowUp') j = (i - 1 + btns.length) % btns.length;
        else if (k === 'Home') j = 0; else if (k === 'End') j = btns.length - 1;
        if (j >= 0) { e.preventDefault(); set(o.options[j].value); btns[j].focus(); }
      });
      group.appendChild(b);
      return b;
    });
    function paint() {
      btns.forEach(function (b, i) {
        var on = o.options[i].value === value;
        b.setAttribute('aria-checked', on ? 'true' : 'false');
        b.tabIndex = on ? 0 : -1;
      });
    }
    function set(v, opt) {
      var changed = v !== value;
      value = v; paint();
      if (changed && !(opt && opt.silent) && o.onChange) o.onChange(value);
    }
    paint();
    var root = el('div', { class: 've-toggle' }, o.label ? el('span', { class: 've-toggle__label', id: labelId, text: o.label }) : null, group);
    var api = { el: root, set: set };
    Object.defineProperty(api, 'value', { get: function () { return value; } });
    return api;
  };

  ui.player = function (o) {
    o = o || {};
    var id = o.id || ui.uid('pl');
    var dur = o.duration || 1;
    var t = o.value || 0;
    var speed = o.speed || dur / 8;
    var format = o.format || function (x) { return fmt(x, { digits: 1 }); };
    var chapters = (o.chapters || []).slice().sort(function (a, b) { return a.t - b.t; });
    var playBtn = ui.button({ icon: 'play', kind: 'primary', title: 'Play', id: id + '-play' });
    playBtn.classList.add('ve-btn--icon');
    var input = el('input', { type: 'range', class: 've-range ve-player__input', id: id, min: 0, max: dur, step: 'any', 'aria-label': o.label || 'Timeline' });
    input.value = t;
    var chapWrap = el('div', { class: 've-player__chapters' });
    var chapBtns = chapters.map(function (c) {
      var b = el('button', { type: 'button', class: 've-player__chap', title: c.label, 'aria-label': 'Jump to: ' + c.label, style: { left: (100 * c.t / dur) + '%' } });
      b.addEventListener('click', function () { api.seek(c.t); });
      chapWrap.appendChild(b);
      return b;
    });
    var nameEl = el('span', { class: 've-player__chapname', 'aria-live': 'polite' });
    var timeEl = el('span', { class: 've-player__time' });
    var root = el('div', { class: 've-player', role: 'group', 'aria-label': o.label || 'Playback' },
      playBtn,
      el('div', { class: 've-player__track' }, chapWrap, el('div', { class: 've-slider ve-player__range' }, el('div', { class: 've-slider__body' }, input))),
      el('div', { class: 've-player__meta' }, nameEl, timeEl));
    var rangeRoot = root.querySelector('.ve-player__range');
    var loopH = ui.loop(function (dt) {
      t += dt * speed;
      if (t >= dur) {
        if (o.loop) t = t % dur;
        else { t = dur; paint(); emit(); api.pause(); if (o.onEnd) o.onEnd(); return; }
      }
      paint(); emit();
    }, { el: root });
    function chapterAt(x) { var c = null; chapters.forEach(function (ch) { if (ch.t <= x + 1e-9) c = ch; }); return c; }
    var lastChap = null;
    function paint() {
      input.value = t;
      rangeRoot.style.setProperty('--p', (100 * t / dur).toFixed(2) + '%');
      timeEl.textContent = format(t) + ' / ' + format(dur);
      var c = chapterAt(t);
      if (c !== lastChap) {
        lastChap = c;
        nameEl.textContent = c ? c.label : '';
        chapBtns.forEach(function (b, i) { b.dataset.active = chapters[i] === c ? 'true' : 'false'; });
      }
      input.setAttribute('aria-valuetext', format(t) + (c ? ', ' + c.label : ''));
    }
    function emit() { if (o.onTime) o.onTime(t, loopH.running); }
    function syncBtn() {
      var on = loopH.running;
      playBtn.replaceChildren(ui.icon(on ? 'pause' : 'play'));
      playBtn.title = on ? 'Pause' : 'Play';
      playBtn.setAttribute('aria-label', on ? 'Pause' : 'Play');
    }
    playBtn.addEventListener('click', function () { api.toggle(); });
    input.addEventListener('input', function () { t = +input.value; paint(); emit(); });
    root.addEventListener('keydown', function (e) {
      if ((e.key === ' ' || e.key === 'k') && e.target === input) { e.preventDefault(); api.toggle(); }
    });
    var api = {
      el: root, input: input,
      play: function () { if (t >= dur - 1e-9) t = 0; loopH.start(); syncBtn(); emit(); return api; },
      pause: function () { loopH.stop(); syncBtn(); emit(); return api; },
      toggle: function () { return loopH.running ? api.pause() : api.play(); },
      seek: function (x) { t = Math.max(0, Math.min(dur, x)); paint(); emit(); return api; },
      setDuration: function (d) { dur = d; input.max = d; paint(); },
      destroy: function () { loopH.destroy(); root.remove(); }
    };
    Object.defineProperty(api, 'time', { get: function () { return t; } });
    Object.defineProperty(api, 'playing', { get: function () { return loopH.running; } });
    paint();
    syncBtn();
    if (o.autoplay && !ui.reducedMotion) api.play();
    return api;
  };

  ui.readout = function (o) {
    o = o || {};
    var format = o.format || function (v) { return typeof v === 'number' ? fmt(v, { sig: 3 }) : String(v); };
    var val = el('span');
    var sub = el('div', { class: 've-readout__sub' });
    var root = el('div', { class: 've-readout', style: o.tone ? { '--tone': toneVar(o.tone) } : null },
      el('div', { class: 've-readout__label', text: o.label || '' }),
      el('div', { class: 've-readout__value' }, val, o.unit ? el('span', { class: 've-unit', text: o.unit }) : null),
      sub);
    function set(v, s) { var t = v == null ? '–' : format(v); if (o.html) val.innerHTML = t; else val.textContent = t; if (s !== undefined) sub.textContent = s; }
    set(o.value, o.sub || '');
    return { el: root, set: set };
  };

  ui.legend = function (items) {
    var root = el('div', { class: 've-legend' });
    (items || []).forEach(function (it) {
      var color = it.color || (it.role ? 'var(--' + it.role + ')' : 'var(--ink)');
      root.appendChild(el('span', { class: 've-key' },
        el('i', { class: 've-key__mark', 'aria-hidden': 'true', dataset: { mark: it.mark || 'line' }, style: { '--c': color } }),
        el('span', { html: it.label })));
    });
    return root;
  };

  /* --------------------------------------------------------------- FIGURE */
  ui.figure = function (o) {
    o = o || {};
    var titleId = ui.uid('fig');
    var head = el('header', { class: 've-figure__head' },
      o.plate != null ? el('div', { class: 've-figure__plate', text: typeof o.plate === 'number' || /^\d+$/.test(o.plate) ? 'Plate ' + o.plate : o.plate }) : null,
      o.title ? el('h3', { class: 've-figure__title', id: titleId, text: o.title }) : null);
    var stage = el('div', { class: 've-figure__stage' + (o.paper ? ' ve-paper' : '') });
    var controls = el('div', { class: 've-figure__controls' });
    var legend = o.legend ? ui.legend(o.legend) : el('div', { class: 've-legend' });
    var caption = el('figcaption', { class: 've-figure__caption' });
    var takeaway = el('p', { class: 've-figure__takeaway' });
    var note = el('p', { class: 've-figure__note' });
    var root = el('figure', { class: 've-figure' + (o.className ? ' ' + o.className : ''), 'aria-labelledby': o.title ? titleId : null },
      head, stage, controls, legend, caption, takeaway, note);
    var api = {
      el: root, head: head, stage: stage, controls: controls, legend: legend, caption: caption, takeaway: takeaway, note: note,
      setCaption: function (html) { caption.innerHTML = html || ''; ui.terms(caption); },
      setTakeaway: function (text) {
        takeaway.replaceChildren();
        if (text) append(takeaway, [el('span', { class: 've-label', text: 'Takeaway' }), el('span', { html: text })]);
        ui.terms(takeaway);
      },
      setNote: function (text, kind) { note.textContent = text || ''; note.dataset.kind = kind || 'live'; },
      setLegend: function (items) { var l = ui.legend(items); legend.replaceWith(l); legend = api.legend = l; }
    };
    if (o.caption) api.setCaption(o.caption);
    if (o.takeaway) api.setTakeaway(o.takeaway);
    if (o.note) api.setNote(o.note, o.noteKind);
    return api;
  };

  /* ------------------------------------------------------------- GLOSSARY */
  ui.glossary = {
    beta: { word: 'Inverse temperature', sym: 'β', ml: '<strong>How sharply the model favours likely configurations,</strong> like softmax sharpness: small β is hot and broad, large β cold and concentrated. Cooling raises β.', phys: 'Inverse temperature, 1/T.' },
    spin: { word: 'Spin', sym: 'xᵢ = ±1', ml: '<strong>One binary variable</strong>, like one black-or-white pixel. The model has N of them.' },
    magnetisation: { word: 'Magnetisation', sym: 'm', ml: '<strong>Which mode the sample is in.</strong> m is the mean of all spins, from −1 to +1. m > 0 is mode A, m < 0 is mode B, m ≈ 0 is undecided.' },
    mode: { word: 'Mode', ml: '<strong>One of the distinct outcomes the model can generate</strong>, like an “8” versus a “3”. Here: all spins mostly +1, or all mostly −1.' },
    transition: { word: 'Transition', sym: 'β = 1', ml: '<strong>Where the single mode at m = 0 splits into two.</strong> Near it, each run’s sign is chosen.' },
    sweep: { word: 'Sweep', ml: '<strong>N single-spin Gibbs updates:</strong> an expected sweep (Curie–Weiss: random times, one update per variable on average) or a permutation sweep (Hopfield: each variable once, in a fresh random order). Sweeps measure sampling time.' },
    bias: { word: 'Field', sym: 'G (h = βG)', ml: '<strong>The model’s preference</strong> for mode A: a small push on every spin toward +1, fixed during a cooling run.' },
    criticalSlowing: { word: 'Critical slowing down', ml: '<strong>Relaxation is slowest at the transition.</strong> Near β = 1 the pull back to equilibrium almost vanishes, so settling takes many more sweeps.' },
    trackingTime: { word: 'Relaxation time', sym: 'τ(β)', ml: '<strong>The inverse spectral gap:</strong> the time for the slowest deviation from equilibrium to shrink by a factor e. The schedule’s dwell at each temperature is ⌈6τ⌉ sweeps.' },
    curieWeiss: { word: 'Curie–Weiss model', ml: '<strong>The simplest model with two modes:</strong> N binary variables that all attract each other equally. Its probabilities and its sampler reduce exactly to the count of +1 spins, so both can be computed.' },
    hopfield: { word: 'Hopfield model', ml: '<strong>A model that stores p patterns.</strong> At low temperature and small p/N its likely configurations lie near a stored pattern or its negative, so it has many modes.' },
    logOdds: { word: 'Log-odds', sym: 'logit q', ml: '<strong>ln(q / (1 − q))</strong> for a probability q. Log-odds add up like energies do, so they measure a preference on its natural scale.' },
    retained: { word: 'Retained share', sym: 'c', ml: '<strong>The share of the model’s log-odds the sampler keeps:</strong> c = logit(q_chain) / logit(q_eq). c = 1 is faithful; c = 0 is a coin flip.' },
    schedule: { word: 'Schedule', ml: '<strong>The cooling schedule: the inverse temperatures β and how many sweeps to spend at each.</strong> Faster schedules cost fewer sweeps.' },
    gamma: { word: 'Cooling rate', sym: 'γ', ml: '<strong>How fast β rises:</strong> the increase per sweep for a linear ramp β = 1 + γt.' },
    kondepudiNelson: { word: 'Selection formula', sym: 'Eq. 3', ml: '<strong>A classical result</strong> (Kondepudi and Nelson 1983; Kondepudi and Gao 1987) for crossing a symmetry-breaking transition at a finite rate. It gives the chance of the favoured sign, Φ(G√N (π/γ)^¼); that is q_chain if later switching is negligible. The paper’s selection law is this formula converted to c by Eq. 2.' },
    reheating: { word: 'Reheating', ml: '<strong>Raising the temperature again</strong> (β below 1) so a run that settled in one mode can choose again (reselection). In the paper: 30 sweeps at β = 0.7, after the filtered start.' },
    filteredStart: { word: 'Filtered start', ml: '<strong>Where the paper’s Hopfield test runs begin:</strong> after a brief cooling, runs labelled (pattern 1, +) are removed, at both signs of the field. For G > 0 this imitates a sampler that settled away from the favoured mode; reheating must undo that choice.' },
    barrier: { word: 'Energy barrier', ml: '<strong>The low-probability region between two modes.</strong> To switch modes at large β the sampler must cross it, which takes time that grows exponentially with N.' },
    overlap: { word: 'Overlap', ml: '<strong>How much the sample looks like one stored pattern</strong>, from −1 to +1. In a Hopfield model it plays the role of m for each pattern.' },
    amplification: { word: 'Amplification', sym: 'A_p', ml: '<strong>The predicted gain in c from competition among p patterns,</strong> under an independent-Gaussian approximation: A_p = E[max |Z|] / E|Z| (Eq. 6).' }
  };
  ui.glossary.speciation = ui.glossary.transition;   // alias: older markup may still use this key
  ui.glossary.selectionFormula = ui.glossary.kondepudiNelson;
  ui.defineTerm = function (key, def) { ui.glossary[key] = def; };

  var tip = null, tipOwner = null, tipPinned = false, hoverTimer = 0;
  function ensureTip() {
    if (tip) return tip;
    tip = el('div', { class: 've-tip', role: 'tooltip', id: 've-tip', hidden: true });
    tip.addEventListener('mouseenter', function () { clearTimeout(hoverTimer); });
    tip.addEventListener('mouseleave', function () { if (!tipPinned) hideTip(); });
    document.body.appendChild(tip);
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && tipOwner) { var o = tipOwner; hideTip(); o.focus(); } });
    document.addEventListener('pointerdown', function (e) {
      if (tipOwner && !tip.contains(e.target) && !tipOwner.contains(e.target)) hideTip();
    }, true);
    window.addEventListener('scroll', function () { if (tipOwner) place(); }, { passive: true });
    window.addEventListener('resize', function () { if (tipOwner) place(); });
    return tip;
  }
  function place() {
    var r = tipOwner.getBoundingClientRect();
    var tw = tip.offsetWidth, th = tip.offsetHeight;
    var vw = document.documentElement.clientWidth, vh = window.innerHeight;
    var x = Math.max(8, Math.min(vw - tw - 8, r.left + r.width / 2 - tw / 2));
    var y = r.bottom + 8;
    if (y + th > vh - 8 && r.top - th - 8 > 8) y = r.top - th - 8;
    tip.style.left = Math.round(x) + 'px';
    tip.style.top = Math.round(y) + 'px';
  }
  function showTip(owner, pin) {
    ensureTip();
    var key = owner.dataset.term, d = ui.glossary[key];
    if (!d) return;
    if (tipOwner && tipOwner !== owner) tipOwner.setAttribute('aria-expanded', 'false');
    tipOwner = owner; tipPinned = !!pin;
    tip.replaceChildren(
      el('div', { class: 've-tip__word' }, el('span', { text: d.word }), d.sym ? el('i', { text: d.sym }) : null),
      el('div', { class: 've-tip__ml', html: d.ml }));
    tip.hidden = false;
    owner.setAttribute('aria-expanded', 'true');
    owner.setAttribute('aria-describedby', 've-tip');
    place();
  }
  function hideTip() {
    clearTimeout(hoverTimer);
    if (!tip) return;
    tip.hidden = true;
    if (tipOwner) { tipOwner.setAttribute('aria-expanded', 'false'); tipOwner.removeAttribute('aria-describedby'); }
    tipOwner = null; tipPinned = false;
  }
  function wire(b) {
    if (b.__veTerm) return b;
    b.__veTerm = true;
    b.setAttribute('aria-expanded', 'false');
    var fine = window.matchMedia && window.matchMedia('(hover: hover) and (pointer: fine)').matches;
    b.addEventListener('click', function (e) {
      e.preventDefault();
      if (tipOwner === b && tipPinned) hideTip(); else showTip(b, true);
    });
    if (fine) {
      b.addEventListener('mouseenter', function () { clearTimeout(hoverTimer); hoverTimer = setTimeout(function () { if (!tipPinned) showTip(b, false); }, 90); });
      b.addEventListener('mouseleave', function () { clearTimeout(hoverTimer); if (!tipPinned) hoverTimer = setTimeout(hideTip, 140); });
    }
    b.addEventListener('focus', function () { if (b.matches(':focus-visible')) showTip(b, false); });
    b.addEventListener('blur', function () { if (!tipPinned) hideTip(); });
    return b;
  }
  ui.term = function (text, key) {
    var b = el('button', { type: 'button', class: 've-term', dataset: { term: key } });
    append(b, text);
    return wire(b);
  };
  ui.terms = function (root) {
    (root || document).querySelectorAll('[data-term]').forEach(function (n) {
      if (n.__veTerm) return;
      if (n.tagName !== 'BUTTON') {
        var b = el('button', { type: 'button', class: 've-term ' + (n.className || ''), dataset: { term: n.dataset.term } });
        while (n.firstChild) b.appendChild(n.firstChild);
        n.replaceWith(b);
        wire(b);
      } else { n.classList.add('ve-term'); wire(n); }
    });
  };

  /* ---------------------------------------------------------------- MOUNT */
  ui.mountComponent = function (id, mountEl, opts) {
    try {
      var c = VE.components[id];
      if (!c || typeof c.mount !== 'function') throw new Error('Component "' + id + '" is not registered. Expected VE.components["' + id + '"] = { mount(el, opts) } in components/' + id + '.js');
      var r = c.mount(mountEl, opts || {});
      ui.terms(mountEl);
      return r;
    } catch (e) {
      console.error(e);
      mountEl.appendChild(el('pre', { class: 've-error', text: 'Mount failed: ' + (e && e.message ? e.message : e) }));
    }
  };

  /* ======================================================================
     DISCLOSURE PRIMITIVES
     ====================================================================== */
  var store = {
    get: function (k) { try { return window.localStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { window.localStorage.setItem(k, v); } catch (e) { /* private mode */ } }
  };
  ui.store = store;
  var INFO_SVG = '<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="6.5"/><path d="M8 7.2v4.3M8 4.6v.1"/></svg>';
  ui.INFO_SVG = INFO_SVG;
  var lastInvoker = null;
  document.addEventListener('click', function (e) {
    var t = e.target && e.target.closest && e.target.closest('[popovertarget]');
    if (t) lastInvoker = t;
  }, true);
  function placeCard(card) {
    var phone = window.innerWidth < 900;
    card.classList.toggle('is-sheet', phone);
    if (phone) { card.style.left = ''; card.style.top = ''; card.style.maxHeight = ''; markMore(card); return; }
    var inv = lastInvoker && lastInvoker.getAttribute('popovertarget') === card.id ? lastInvoker : document.querySelector('[popovertarget="' + card.id + '"]');
    if (!inv) return;
    /* below the trigger; if it does not fit, flip above only when there is more room
       above, and otherwise scroll inside. The card never covers its own trigger. */
    card.style.maxHeight = '';
    var r = inv.getBoundingClientRect(), w = card.offsetWidth, h = card.offsetHeight, vh = window.innerHeight;
    var x = Math.max(16, Math.min(window.innerWidth - w - 16, r.left - 12));
    var below = vh - r.bottom - 8 - 16, above = r.top - 8 - 16, y;
    /* a card that must scroll stays under its trigger when a usable height
       (280 px) is free there, so it does not cover the number it explains */
    if (h <= below || below >= above || below >= 280) {
      y = r.bottom + 8;
      if (h > below) card.style.maxHeight = Math.max(120, below) + 'px';
    } else {
      if (h > above) { card.style.maxHeight = above + 'px'; h = above; }
      y = r.top - 8 - h;
    }
    card.style.left = x + 'px'; card.style.top = y + 'px';
    markMore(card);
  }
  /* every scrolling card says "more below" while text remains under the cut */
  function markMore(card) {
    requestAnimationFrame(function () {
      card.classList.toggle('v2-card--more', card.scrollTop + card.clientHeight < card.scrollHeight - 2);
    });
  }
  function wireCard(card) {
    if (card.__v2card) return card;
    card.__v2card = true;
    if (!card.hasAttribute('popover')) card.setAttribute('popover', 'auto');
    card.classList.add('v2-card');
    if (!card.querySelector('.v2-card__x')) {
      card.insertBefore(el('button', { type: 'button', class: 'v2-card__x', 'aria-label': 'Close', html: '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M3 3l6 6M9 3l-6 6"/></svg>',
        on: { click: function () { try { card.hidePopover(); } catch (e) { /* none */ } } } }), card.firstChild);
    }
    card.addEventListener('toggle', function (e) {
      if (e.newState === 'open') placeCard(card);
    });
    card.addEventListener('scroll', function () { markMore(card); }, { passive: true });
    return card;
  }
  ui.cards = function (root) { (root || document).querySelectorAll('[popover].v2-card, [popover][data-v2-card]').forEach(wireCard); };
  window.addEventListener('resize', function () {
    document.querySelectorAll('.v2-card:popover-open').forEach(placeCard);
  });

  ui.info = function (trigger, o) {
    o = o || {};
    var id = o.id || ui.uid('v2card');
    var card = document.getElementById(id);
    if (!card) {
      card = el('div', { id: id, class: 'v2-card' + (o.kind ? ' v2-card--' + o.kind : ''), popover: 'auto', role: 'dialog', 'aria-label': o.title || null });
      if (o.title) card.appendChild(el('p', { class: 'v2-card__k', html: o.title }));
      var body = el('div', { class: 'v2-card__body' });
      if (o.node) body.appendChild(o.node); else if (o.html) body.innerHTML = o.html;
      card.appendChild(body);
      (o.parent || document.body).appendChild(card);
    }
    wireCard(card);
    if (trigger) {
      trigger.setAttribute('popovertarget', id);
      if (!trigger.getAttribute('aria-label') && !trigger.textContent.trim() && o.title) trigger.setAttribute('aria-label', o.title.replace(/<[^>]+>/g, ''));
    }
    ui.terms(card);
    return {
      el: card,
      open: function () { try { card.showPopover(); } catch (e) { /* none */ } },
      close: function () { try { card.hidePopover(); } catch (e) { /* none */ } }
    };
  };

  /* Splits a component's instrument note at " · " into a definition list. */
  function noteItems(text) {
    return String(text || '').replace(/\s+/g, ' ').trim().split(' · ').map(function (s) { return s.trim(); }).filter(Boolean);
  }
  function itemKey(s) {
    var m = s.match(/^([^:=]{1,28})(:|\s=)\s*(.+)$/);
    if (!m) return [null, s];
    var key = m[1].trim(), words = key.split(/\s+/);
    /* an "=" item is a label only when it reads as one: no digits, no comma, at most three
       words, and not a phrase ending in a symbol ("linear schedule β = 1 + γt" is an equation) */
    /* "N = 1,024, γ = 10⁻³, G = …" is a list of settings, not one label and its value */
    if (m[2] !== ':' && /=/.test(m[3])) return [null, s];
    if (m[2] !== ':' && (/\d/.test(key) || /,/.test(key) || words.length > 3 || (words.length > 1 && words[words.length - 1].length <= 2))) return [null, s];
    return [m[1].trim(), m[3].trim()];
  }
  /* noteEl: one element, or a list of {el, label} for a component with several plates. */
  ui.specCard = function (trigger, noteEl, o) {
    o = o || {};
    var groups = !noteEl ? [] : (noteEl.nodeType === 1 ? [{ el: noteEl }] : Array.prototype.slice.call(noteEl).map(function (g) { return g.nodeType === 1 ? { el: g } : g; }));
    var box = el('div', { class: 'v2-specs' });
    var api = ui.info(trigger, { id: o.id, title: o.title || 'What runs here', node: box, kind: 'spec', parent: o.parent });
    var mos = [];
    groups.forEach(function (g) {
      if (g.label && groups.length > 1) box.appendChild(el('p', { class: 'v2-spec__h', text: g.label }));
      var dl = el('dl', { class: 'v2-spec' });
      box.appendChild(dl);
      function copy() {
        var items = noteItems(g.el.textContent);
        dl.replaceChildren.apply(dl, items.map(function (s) {
          var kv = itemKey(s);
          return kv[0] ? el('div', { class: 'v2-spec__row' }, el('dt', { text: kv[0] }), el('dd', { text: kv[1] }))
                       : el('div', { class: 'v2-spec__row v2-spec__row--full' }, el('dd', { text: kv[1] }));
        }));
      }
      copy();
      if (window.MutationObserver) { var mo = new MutationObserver(copy); mo.observe(g.el, { childList: true, characterData: true, subtree: true }); mos.push(mo); }
    });
    api.disconnect = function () { mos.forEach(function (m) { m.disconnect(); }); };
    return api;
  };
  ui.mirror = function (sourceEl, targetEl) {
    if (!sourceEl || !targetEl) return function () {};
    function copy() { targetEl.textContent = sourceEl.textContent.replace(/\s+/g, ' ').trim(); }
    copy();
    if (!window.MutationObserver) return function () {};
    var mo = new MutationObserver(copy);
    mo.observe(sourceEl, { childList: true, characterData: true, subtree: true });
    return function () { mo.disconnect(); };
  };

  /* Drawers */
  ui.drawers = function (root) {
    (root || document).querySelectorAll('details.v2-drawer').forEach(function (d) {
      if (d.__v2dr) return;
      d.__v2dr = true;
      var s = d.querySelector(':scope > summary');
      if (s && !s.querySelector('.v2-drawer__i')) s.appendChild(el('span', { class: 'v2-drawer__i', 'aria-hidden': 'true' }));
      d.addEventListener('toggle', function () {
        if (!d.open) return;
        ui.math(d);
        requestAnimationFrame(function () { window.dispatchEvent(new Event('resize')); });
      });
      if (allOn) d.open = true;
    });
  };
  ui.openDrawer = function (id, opts) {
    var d = document.getElementById(id);
    if (!d || d.tagName !== 'DETAILS') return false;
    d.open = true;
    if (!opts || opts.scroll !== false) setTimeout(function () { d.scrollIntoView({ block: 'start', behavior: ui.reducedMotion ? 'auto' : 'smooth' }); }, 60);
    return true;
  };

  /* Margin notes: always visible in the wide margin; a labelled toggle below it. */
  ui.notes = function (root) {
    (root || document).querySelectorAll('aside.v2-note').forEach(function (n) {
      if (n.__v2note) return;
      n.__v2note = true;
      var bodyId = n.id ? n.id + '-b' : ui.uid('v2note');
      var body = el('div', { class: 'v2-note__b', id: bodyId });
      while (n.firstChild) body.appendChild(n.firstChild);
      var btn = el('button', { type: 'button', class: 'v2-note__t', 'aria-expanded': 'false', 'aria-controls': bodyId },
        el('span', { class: 'v2-note__g', 'aria-hidden': 'true' }), el('span', { text: n.dataset.q || 'Note' }));
      btn.addEventListener('click', function () {
        var open = btn.getAttribute('aria-expanded') !== 'true';
        btn.setAttribute('aria-expanded', String(open));
        n.classList.toggle('is-open', open);
      });
      n.appendChild(btn); n.appendChild(body);
      if (allOn) { btn.setAttribute('aria-expanded', 'true'); n.classList.add('is-open'); }
    });
  };

  /* Lazy plate: a drawer that mounts a heavy component on first open. */
  ui.lazyPlate = function (details, componentId, opts) {
    if (!details) return;
    function go() {
      if (!details.open || details.__v2mounted) return;
      details.__v2mounted = true;
      var target = details.querySelector('[data-lazy-target]') || details.querySelector('.v2-drawer__body');
      ui.mountComponent(componentId, target, opts || {});
    }
    details.addEventListener('toggle', go);
    go();
  };

  /* "All details": every drawer, note and provenance block of every chapter. */
  var allOn = store.get('ve2.details') === '1';
  var allSubs = new Set();
  ui.allDetails = function (on) {
    if (on === undefined) return allOn;
    allOn = !!on;
    store.set('ve2.details', allOn ? '1' : '0');
    document.documentElement.classList.toggle('v2-all', allOn);
    document.querySelectorAll('details.v2-drawer').forEach(function (d) { d.open = allOn; });
    document.querySelectorAll('aside.v2-note').forEach(function (n) {
      n.classList.toggle('is-open', allOn);
      var b = n.querySelector('.v2-note__t'); if (b) b.setAttribute('aria-expanded', String(allOn));
    });
    allSubs.forEach(function (cb) { try { cb(allOn); } catch (e) { console.error(e); } });
    return allOn;
  };
  ui.onAllDetails = function (cb) { allSubs.add(cb); return function () { allSubs.delete(cb); }; };
  if (allOn) document.documentElement.classList.add('v2-all');

  /* Math: KaTeX (loaded with defer from jsDelivr); plain source text until it arrives. */
  var mathWait = [];
  ui.math = function (root) {
    root = root || document;
    if (!window.katex) {
      mathWait.push(root);
      if (mathWait.length === 1) {
        var tries = 0;
        (function poll() {
          if (window.katex) { var q = mathWait.splice(0); q.forEach(function (r) { ui.math(r); }); return; }
          if (++tries < 100) setTimeout(poll, 100);
        })();
      }
      return;
    }
    root.querySelectorAll('[data-tex]:not([data-tex-done]), [data-tex-display]:not([data-tex-done])').forEach(function (n) {
      var display = n.hasAttribute('data-tex-display');
      var src = n.getAttribute(display ? 'data-tex-display' : 'data-tex') || n.textContent;
      try {
        if (display) {
          var body = el('span', { class: 'v2-eq__m' });
          window.katex.render(src, body, { displayMode: true, throwOnError: false, output: 'htmlAndMathml' });
          n.replaceChildren(body);
          if (n.dataset.eq) n.appendChild(el('span', { class: 'v2-eq__n', text: '(' + n.dataset.eq + ')' }));
          n.classList.add('v2-eq');
        } else {
          window.katex.render(src, n, { displayMode: false, throwOnError: false, output: 'htmlAndMathml' });
        }
        n.setAttribute('data-tex-done', '');
      } catch (e) { /* keep the source text */ }
    });
  };
})();
