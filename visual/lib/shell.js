/* ==========================================================================
   Visual Edition — window.VE.shell   (lib/shell.js, plain classic script)
   Owns: chapter routing (one chapter at a time, URL hash), the chapter list,
   prev / next, the phone drawer, the theme switch, mounting each chapter's
   component on first visit, and the live frontispiece on the overview.
   Never edits a component's DOM: components mount with VE.ui.mountComponent.
   ========================================================================== */
(function () {
  'use strict';
  var VE = window.VE, ui = VE.ui;
  var doc = document, root = doc.documentElement;
  var $ = function (s, r) { return (r || doc).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || doc).querySelectorAll(s)); };
  var reduced = function () { return ui.reducedMotion; };
  var idle = window.requestIdleCallback ? function (fn) { window.requestIdleCallback(fn, { timeout: 200 }); } : function (fn) { setTimeout(fn, 16); };
  var PAPER = 'https://github.com/ali-izhar/finite-rate-selection-law/blob/main/paper/main.pdf';

  /* ------------------------------------------------------------ chapters */
  var chapters = $$('[data-chapter]').map(function (el, i) {
    /* one name per chapter: its h1 (the overview carries data-title) */
    var h = el.querySelector('.v2-title');
    return { id: el.dataset.chapter, el: el, i: i, n: el.dataset.n || '', title: el.dataset.title || (h ? h.textContent.trim() : ''), sec: el.dataset.sec || '' };
  });
  var byId = {}; chapters.forEach(function (c) { byId[c.id] = c; });
  var cur = null;
  function neighbours(id) {
    var k = byId[id] ? byId[id].i : -1;
    return k < 0 ? {} : { prev: chapters[k - 1], next: chapters[k + 1] };
  }
  function label(c) { return (c.n ? c.n + ' · ' : '') + c.title; }

  /* the paper link and prev / next at the foot of each chapter */
  chapters.forEach(function (c) {
    if (c.sec) c.el.appendChild(ui.el('p', { class: 'v2-paperlink' }, ui.el('a', { href: PAPER, target: '_blank', rel: 'noopener', text: 'In the paper: ' + c.sec + ' →' })));
    var nb = neighbours(c.id);
    var nav = ui.el('nav', { class: 'v2-pn', 'aria-label': 'Previous and next chapter' });
    var mk = function (t, kind) {
      return ui.el('a', { href: '#' + t.id }, ui.el('span', { class: 'v2-pn__k', text: kind }), ui.el('span', { class: 'v2-pn__t', text: label(t) }));
    };
    if (nb.prev) nav.appendChild(mk(nb.prev, 'Previous'));
    if (nb.next) { var a = mk(nb.next, c.id === 'overview' ? 'Begin' : 'Next'); a.classList.add('v2-pn__nx'); nav.appendChild(a); }
    c.el.appendChild(nav);
  });

  /* ------------------------------------------------------- chapter list */
  var toc = $('[data-toc]');
  chapters.forEach(function (c) {
    if (c.id === 'overview') return;   /* the title block above the list links the overview */
    toc.appendChild(ui.el('li', null, ui.el('a', { href: '#' + c.id, dataset: { goto: c.id } },
      ui.el('span', { class: 'v2-toc__n', text: c.n }), ui.el('span', { class: 'v2-toc__t', text: c.title }))));
  });
  function markToc() {
    $$('[data-goto]', toc).forEach(function (a) {
      if (a.dataset.goto === cur) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    });
  }

  /* ------------------------------------------------------------- routing */
  function parseHash() {
    /* an old drawer link (#race-math, #limits-2) lands on its chapter */
    var h = decodeURIComponent((location.hash || '').slice(1)), base = h.split('-')[0];
    return { id: byId[h] ? h : byId[base] ? base : 'overview' };
  }
  function show(id, o) {
    o = o || {};
    var c = byId[id] || byId.overview, changed = cur !== c.id;
    cur = c.id;
    chapters.forEach(function (x) {
      var on = x === c;
      if (x.el.hidden === on) x.el.hidden = !on;
      if (on && changed && !reduced() && !o.initial) { x.el.classList.remove('is-in'); void x.el.offsetWidth; x.el.classList.add('is-in'); }
    });
    markToc();
    $('[data-bar-t]').textContent = label(c);
    var nx = neighbours(c.id).next, bn = $('[data-bar-next]');
    if (nx) { bn.hidden = false; bn.href = '#' + nx.id; } else bn.hidden = true;
    root.style.setProperty('--bp', (100 * c.i / (chapters.length - 1)).toFixed(1) + '%');
    doc.title = c.id === 'overview' ? 'Finite-Rate Selection Law' : c.title + ' · Finite-Rate Selection Law';
    closeRail();
    if (changed) window.scrollTo(0, 0);
    visit(c);
  }
  function go(id) {
    if (location.hash !== '#' + id) history.pushState(null, '', '#' + id);
    if (id === cur) window.scrollTo(0, 0);
    show(id);
  }
  window.addEventListener('popstate', function () { show(parseHash().id); });
  window.addEventListener('hashchange', function () { show(parseHash().id); });
  doc.addEventListener('click', function (e) {
    var a = e.target.closest && e.target.closest('a[href^="#"]');
    if (!a || e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey) return;
    var id = a.getAttribute('href').slice(1);
    if (byId[id]) { e.preventDefault(); go(id); }
  });
  doc.addEventListener('keydown', function (e) {
    if (e.defaultPrevented || e.altKey || e.metaKey || e.ctrlKey) return;
    if (e.key === 'Escape' && rail.classList.contains('is-open')) { closeRail(); menu.focus(); return; }
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    var t = e.target;
    if (t && t.closest && t.closest('input, textarea, select, button, canvas, [role="radiogroup"], [role="slider"], [contenteditable]')) return;
    var to = neighbours(cur)[e.key === 'ArrowRight' ? 'next' : 'prev'];
    if (to) { e.preventDefault(); go(to.id); }
  });

  /* skip link: focus the current chapter's heading */
  $('.v2-skip').addEventListener('click', function (e) {
    e.preventDefault();
    var h = byId[cur] && $('.v2-title', byId[cur].el);
    if (!h) return;
    h.setAttribute('tabindex', '-1'); h.focus({ preventScroll: true }); window.scrollTo(0, 0);
  }, true);

  /* ------------------------------------------------------------ mounting */
  /* first load: the content after the plate stays invisible until the plate has built
     its figure, so the Result line does not paint in the figure's place and then jump
     down. The hold always ends: after the mount, or after 3 s. */
  var visited = {};
  function visit(c) {
    if (visited[c.id]) return;
    visited[c.id] = true;
    ui.math(c.el);
    var slots = $$('[data-mount]', c.el);
    if (!slots.length) return;
    slots.forEach(function (s) { s.dataset.wait = '1'; s.appendChild(ui.el('div', { class: 'v2-pending', text: 'Loading figure…' })); });
    var t = setTimeout(release, 3000);
    function release() { clearTimeout(t); slots.forEach(function (s) { delete s.dataset.wait; }); }
    idle(function () {
      slots.forEach(function (s) {
        var opts = {}; try { opts = JSON.parse(s.dataset.opts || '{}'); } catch (e) { /* none */ }
        var p = s.querySelector('.v2-pending'); if (p) p.remove();
        ui.mountComponent(s.dataset.mount, s, opts);
        s.dataset.state = 'mounted';
      });
      requestAnimationFrame(release);
    });
  }

  /* ---------------------------------------------------------- phone drawer */
  var rail = $('#rail'), menu = $('.v2-bar__menu'), scrim = $('[data-scrim]');
  function openRail() {
    rail.classList.add('is-open'); scrim.classList.add('is-open');
    menu.setAttribute('aria-expanded', 'true');
    var f = $('[aria-current="page"]', rail) || $('a, button', rail);
    setTimeout(function () { if (f) f.focus(); }, 30);
  }
  function closeRail() {
    if (!rail.classList.contains('is-open')) return;
    rail.classList.remove('is-open'); scrim.classList.remove('is-open');
    menu.setAttribute('aria-expanded', 'false');
  }
  menu.addEventListener('click', function () { rail.classList.contains('is-open') ? closeRail() : openRail(); });
  scrim.addEventListener('click', closeRail);
  rail.addEventListener('keydown', function (e) {   /* focus trap while the drawer is open */
    if (e.key !== 'Tab' || !rail.classList.contains('is-open')) return;
    var f = $$('a[href], button:not([disabled])', rail).filter(function (x) { return x.offsetParent !== null; });
    if (!f.length) return;
    if (e.shiftKey && doc.activeElement === f[0]) { e.preventDefault(); f[f.length - 1].focus(); }
    else if (!e.shiftKey && doc.activeElement === f[f.length - 1]) { e.preventDefault(); f[0].focus(); }
  });

  /* ---------------------------------------------------------------- theme */
  var themeBtns = $$('[data-theme-set]');
  function setTheme(t, save) {
    ui.setTheme(t);
    themeBtns.forEach(function (b) { b.setAttribute('aria-checked', String(b.dataset.themeSet === t)); });
    if (save) ui.store.set('ve2.theme', t);
  }
  themeBtns.forEach(function (b) { b.addEventListener('click', function () { setTheme(b.dataset.themeSet, true); }); });
  setTheme(ui.store.get('ve2.theme') || 'system', false);

  /* ======================================================================
     The frontispiece: a real Curie–Weiss chain, N = 1,024,
       p(x) ∝ exp(β/(2(N−1)) Σ_{i≠j} x_i x_j + h Σ x_i),  h = βG,
     heat-bath updates at random sites, one sweep per frame. Each cycle cools
     (raises β linearly) from 0.7 to 1.6, holds, returns to 0.7. A finished
     trace takes the colour of its final sign. G√N = 0.0067·√256 (page choice).
     ====================================================================== */
  var B0 = 0.7, B1 = 1.6, DOWN = 380, HOLD = 360, BACK = 50, REST = 30;
  function rng(seed) { var s = seed >>> 0 || 1; return function () { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; }
  function Front(cv, N, seed) {
    this.cv = cv; this.N = N; this.G = 0.0067 * Math.sqrt(256 / N);
    this.r = rng(seed); this.x = new Int8Array(N); this.S = 0;
    for (var i = 0; i < N; i++) { this.x[i] = this.r() < 0.5 ? 1 : -1; this.S += this.x[i]; }
    this.traces = []; this.phase = 'down'; this.step = 0; this.beta = B0; this.m = this.S / N;
    this.newTrace(); this.on = false; this.raf = 0; this.vis = false;
  }
  Front.prototype.newTrace = function () { this.cur = { pts: new Float32Array(2 * (DOWN + 2)), n: 0, fate: 0 }; };
  Front.prototype.sweep = function (beta) {
    var N = this.N, x = this.x, r = this.r, h = beta * this.G, J = beta / (N - 1), S = this.S;
    for (var k = 0; k < N; k++) {
      var i = (r() * N) | 0;
      var up = r() < 1 / (1 + Math.exp(-2 * (J * (S - x[i]) + h))) ? 1 : -1;
      if (up !== x[i]) { S += up - x[i]; x[i] = up; }
    }
    this.S = S;
    return S / N;
  };
  Front.prototype.tick = function () {
    if (this.phase === 'down') {
      this.beta = B0 + (B1 - B0) * Math.min(1, this.step / DOWN);
      this.m = this.sweep(this.beta);
      if (this.cur.n < DOWN + 2) { this.cur.pts[2 * this.cur.n] = this.beta; this.cur.pts[2 * this.cur.n + 1] = this.m; this.cur.n++; }
      if (++this.step > DOWN) { this.phase = 'hold'; this.step = 0; }
    } else if (this.phase === 'hold') {
      this.m = this.sweep(B1);
      if (++this.step >= HOLD) {
        this.cur.fate = this.m > 0 ? 1 : -1;
        this.traces.push(this.cur); if (this.traces.length > 24) this.traces.shift();
        this.phase = 'back'; this.step = 0;
      }
    } else {
      this.beta = B1 - (B1 - B0) * Math.min(1, this.step / BACK);
      this.m = this.sweep(this.beta);
      if (++this.step >= BACK + REST) { this.phase = 'down'; this.step = 0; this.newTrace(); }
    }
  };
  Front.prototype.paint = function () {
    var cv = this.cv, W = cv.clientWidth, H = cv.clientHeight; if (!W || !H || cv.offsetParent === null) return;
    var dpr = Math.min(2, window.devicePixelRatio || 1);
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    var ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    var c = ui.colors(), A = c.modeA, B = c.modeB;
    var padL = 22, padR = 4, top = 6, bot = H - 20;
    var sx = function (b) { return padL + (b - B0) / (B1 - B0) * (W - padL - padR); };
    var sy = function (m) { return top + (1 - m) / 2 * (bot - top); };
    ctx.lineWidth = 1; ctx.strokeStyle = c.ruleStrong;
    ctx.globalAlpha = 0.55; ctx.beginPath(); ctx.moveTo(sx(B0), Math.round(sy(0)) + 0.5); ctx.lineTo(sx(B1), Math.round(sy(0)) + 0.5); ctx.stroke();
    ctx.globalAlpha = 0.9; ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(Math.round(sx(1)) + 0.5, top); ctx.lineTo(Math.round(sx(1)) + 0.5, bot + 3); ctx.stroke(); ctx.setLineDash([]);
    ctx.globalAlpha = 1; ctx.fillStyle = c.ink3; ctx.font = '450 11px ' + c.fontMono; ctx.textBaseline = 'top';
    ctx.textAlign = 'center'; ctx.fillText('β = 1', sx(1), bot + 4);
    ctx.textAlign = 'left'; ctx.fillText('β 0.7 · hot', sx(B0), bot + 4);
    ctx.textAlign = 'right'; ctx.fillText('cold · β 1.6', sx(B1), bot + 4);
    ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
    ctx.fillStyle = A; ctx.fillText('A', padL - 14, sy(0.78)); ctx.fillStyle = B; ctx.fillText('B', padL - 14, sy(-0.78));
    var n = this.traces.length;
    ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.lineWidth = 1;
    for (var t = 0; t < n; t++) {
      var tr = this.traces[t];
      ctx.strokeStyle = tr.fate > 0 ? A : B;
      ctx.globalAlpha = t >= n - 4 ? 0.55 + 0.1 * (t - n + 4) : 0.14 + 0.36 * (t + 1) / n;
      path(ctx, tr, sx, sy); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    if (this.cur.n > 1 && this.phase !== 'back') {
      ctx.strokeStyle = c.ink; ctx.lineWidth = 1.35; path(ctx, this.cur, sx, sy); ctx.stroke();
      var k = this.cur.n - 1;
      ctx.fillStyle = c.ink; ctx.beginPath(); ctx.arc(sx(this.cur.pts[2 * k]), sy(this.phase === 'hold' ? this.m : this.cur.pts[2 * k + 1]), 3, 0, 7); ctx.fill();
    }
    ctx.globalAlpha = 1;
  };
  function path(ctx, tr, sx, sy) {
    ctx.beginPath();
    for (var j = 0; j < tr.n; j++) { var X = sx(tr.pts[2 * j]), Y = sy(tr.pts[2 * j + 1]); if (j) ctx.lineTo(X, Y); else ctx.moveTo(X, Y); }
  }
  /* run only while on screen, the tab is visible and motion is allowed */
  Front.prototype.check = function () {
    var self = this, want = this.vis && !doc.hidden && !reduced();
    if (!want) { this.on = false; if (this.raf) cancelAnimationFrame(this.raf); this.raf = 0; this.paint(); return; }
    if (this.on) return;
    this.on = true;
    var f = function () { if (!self.on) return; self.tick(); self.paint(); self.raf = requestAnimationFrame(f); };
    this.raf = requestAnimationFrame(f);
  };
  Front.prototype.start = function () {
    var self = this;
    if (window.IntersectionObserver) new IntersectionObserver(function (es) { self.vis = es[0].isIntersecting; self.check(); }).observe(this.cv);
    else this.vis = true;
    doc.addEventListener('visibilitychange', function () { self.check(); });
    ui.onTheme(function () { self.paint(); });
    window.addEventListener('resize', function () { self.paint(); });
    ui.onReducedMotion(function () { self.check(); });
    this.check();
  };
  /* fast-forward k whole cycles in idle slices of at most 8 ms, so the first frame already
     shows finished traces and no task blocks the page */
  Front.prototype.prefill = function (k, done) {
    var self = this, left = k * (DOWN + 1 + HOLD + BACK + REST);
    (function next(dl) {
      var t0 = performance.now(), budget = 8;
      if (dl && dl.timeRemaining && !dl.didTimeout) budget = Math.max(2, Math.min(8, dl.timeRemaining() - 2));
      while (left > 0 && performance.now() - t0 < budget) { self.tick(); left--; }
      if (left > 0) { idle(next); return; }
      done();
    })();
  };

  /* ---------------------------------------------------------------- go */
  if ('scrollRestoration' in history) history.scrollRestoration = 'manual';
  var p0 = parseHash();
  if (location.hash && location.hash.slice(1) !== p0.id) history.replaceState(null, '', '#' + p0.id);
  show(p0.id, { initial: true });
  var fc = $('canvas[data-emblem="front"]');
  if (fc) { var front = new Front(fc, 1024, 11); front.prefill(10, function () { front.start(); }); }
})();
