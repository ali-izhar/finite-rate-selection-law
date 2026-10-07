/* ==========================================================================
   Visual Edition — window.VE.shell   (lib/shell.js, plain classic script)
   Owns: chapter routing (one chapter at a time, URL hash), the schedule rail,
   the pitchfork emblems (rail, phone bar, frontispiece), theme and "All
   details" switches, the phone drawer, mounting components on first visit,
   the reading / spec cards for each plate, the test chapter's figure, and
   the narrated film: the Watch chapter's player and each chapter's optional
   "Watch this step" clip (both read media/media.json; never autoplayed).
   Never edits a component's DOM: components mount with VE.ui.mountComponent
   and are restyled by lib/shell.css; the shell only reads their text.

   API
     VE.shell.emblems   the pitchfork emblems (components/hero.js reads the
                        frontispiece emblem's state)
   ========================================================================== */
(function () {
  'use strict';
  var VE = window.VE, ui = VE.ui, sim = VE.sim;
  var doc = document, root = doc.documentElement;
  var $ = function (s, r) { return (r || doc).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || doc).querySelectorAll(s)); };
  var reduced = function () { return ui.reducedMotion; };
  var idle = window.requestIdleCallback ? function (fn) { window.requestIdleCallback(fn, { timeout: 200 }); } : function (fn) { setTimeout(fn, 16); };
  var strip = function (h) { var d = doc.createElement('div'); d.innerHTML = h || ''; return d.textContent; };

  /* ------------------------------------------------------------ chapters */
  var chapters = $$('[data-chapter]').map(function (el) {
    var b = el.dataset.beta;
    return { id: el.dataset.chapter, el: el, beta: b === 'off' ? null : +b, n: el.dataset.n || '', title: el.dataset.title || '', takeaway: el.dataset.takeaway || '' };
  });
  var byId = {}; chapters.forEach(function (c, i) { c.i = i; byId[c.id] = c; });
  /* reading order: the scale (by β), then references. The film is off the chain, as it is
     off the rail: its own row reads Previous = the overview, Next = chapter 1, so Begin,
     the phone bar, the arrow keys and every Previous link agree */
  var order = ['overview', 'commit', 'slowdown', 'odds', 'race', 'scaling', 'modes', 'reheat', 'test', 'limits', 'refs'].filter(function (id) { return byId[id]; });
  var REDIRECT = { top: 'overview', ledger: 'odds-ledger', film: 'watch' };
  var cur = null;
  function neighbours(id) {
    if (id === 'watch') return { prev: byId.overview, next: byId.commit };
    var k = order.indexOf(id);
    return k < 0 ? {} : { prev: byId[order[k - 1]], next: byId[order[k + 1]] };
  }

  /* icons for every info chip and dot */
  $$('.v2-chip, .v2-info').forEach(function (b) { if (!b.querySelector('svg')) b.insertAdjacentHTML('afterbegin', ui.INFO_SVG); });

  /* ---------------------------------------------------------- kicker ticks */
  $$('.v2-steps[data-step]').forEach(function (s) {
    var k = +s.dataset.step;
    s.setAttribute('aria-label', 'Step ' + k + ' of 7');
    for (var i = 1; i <= 7; i++) s.appendChild(ui.el('i', { class: i <= k ? 'on' : null }));
  });

  /* ----------------------------------------------------------- prev / next */
  function label(c) { return (c.n ? c.n + ' · ' : '') + strip(c.title); }
  chapters.forEach(function (c) {
    var nb = neighbours(c.id), prev = nb.prev, next = nb.next;
    var nav = ui.el('nav', { class: 'v2-pn', 'aria-label': 'Previous and next chapter' });
    var mk = function (t, kind) {
      var a = ui.el('a', { href: '#' + t.id, dataset: { nav: t.id } }, ui.el('span', { class: 'v2-pn__k', text: kind }), ui.el('span', { class: 'v2-pn__t' }));
      a.lastChild.innerHTML = (t.n ? t.n + ' · ' : '') + t.title;
      return a;
    };
    if (prev) nav.appendChild(mk(prev, 'Previous'));
    /* the overview begins the interactive chapters; the film is one press away in its
       "Watch this step" clip (which links the whole film) and in the rail */
    if (c.id === 'overview' && next) nav.appendChild(ui.el('div', { class: 'v2-pn__next' }, mk(next, 'Begin')));
    else if (next) nav.appendChild(mk(next, 'Next'));
    c.el.appendChild(nav);
  });

  /* --------------------------------------------------------------- the rail */
  var scale = $('[data-scale]'), off = $('[data-off]');
  var needle = ui.el('span', { class: 'v2-scale__needle', 'aria-hidden': 'true' });
  function buildRail(cc) {
    var ladder = cc.ladder, sw = cc.sweeps_per_level, maxSw = Math.max.apply(null, sw);
    var total = sw.reduce(function (a, b) { return a + b; }, 0);
    scale.replaceChildren();
    scale.appendChild(ui.el('li', { class: 'v2-scale__head', 'aria-hidden': 'true' }, ui.el('span', { text: 'sweeps' }), ui.el('span', { html: '<i class="gr">β</i>' }), ui.el('span', { text: 'chapter' })));
    scale.appendChild(ui.el('span', { class: 'v2-scale__spine', 'aria-hidden': 'true' }));
    scale.appendChild(needle);
    var used = {};
    ladder.forEach(function (b, i) {
      var ch = chapters.find(function (c) { return c.beta != null && Math.abs(c.beta - b) < 1e-6; });
      var bar = ui.el('span', { class: 'v2-st__bar', 'aria-hidden': 'true', style: { '--bf': (sw[i] / maxSw).toFixed(3) } });
      var bcell = ui.el('span', { class: 'v2-st__b', 'aria-hidden': 'true', text: b.toFixed(2) });
      if (Math.abs(b - 1) < 1e-6) bcell.appendChild(ui.el('span', { class: 'v2-bc', html: 'β<sub>c</sub>' }));
      var title = 'β = ' + b.toFixed(2) + ' · ' + sw[i] + ' of ' + total + ' sweeps at this temperature';
      if (!ch) {
        scale.appendChild(ui.el('li', { class: 'v2-st v2-st--bare', title: title, 'aria-hidden': 'true' }, bar, ui.el('span', { class: 'v2-st__tick' }), bcell));
        return;
      }
      used[ch.id] = true;
      var aside = ch.el && ch.el.hasAttribute('data-aside');
      var name = ui.el('span', { class: 'v2-st__name' },
        ui.el('span', { class: 'v2-st__t', html: '<span class="v2-st__n">' + ch.n + '</span>' + ch.title + (aside ? '<span class="v2-st__aside">side note</span>' : '') }),
        ch.takeaway ? ui.el('span', { class: 'v2-st__take', html: ch.takeaway }) : null);
      /* one clean accessible name: the visual cells (bar, tick, β) are hidden from AT */
      var speak = function (h) { return strip(String(h).replace(/<sup>(.*?)<\/sup>/g, '^$1').replace(/<sub>(.*?)<\/sub>/g, ' $1')); };
      var aria = (/^\d+$/.test(ch.n) && +ch.n > 0 ? 'Chapter ' + ch.n + ', ' : '') + speak(ch.title) + (aside ? ' (side note)' : '') + ' (beta ' + b.toFixed(2) + ')' + (ch.takeaway ? ': ' + speak(ch.takeaway) : '');
      var a = ui.el('a', { class: 'v2-st' + (aside ? ' v2-st--aside' : ''), href: '#' + ch.id, title: title, 'aria-label': aria, dataset: { goto: ch.id } }, bar, ui.el('span', { class: 'v2-st__tick', 'aria-hidden': 'true' }), bcell, name);
      scale.appendChild(ui.el('li', { style: 'display:contents' }, a));
    });
    /* a chapter whose β is not on the ladder goes off the scale */
    var ICON = { watch: '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M3.5 2.2v7.6L9.6 6z"/></svg>', refs: '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2.5 3h7v1h-7zM2.5 5.5h7v1h-7zM2.5 8h4.5v1h-4.5z"/></svg>' };
    off.replaceChildren();
    chapters.forEach(function (c) {
      if (used[c.id]) return;
      off.appendChild(ui.el('a', { href: '#' + c.id, dataset: { goto: c.id }, html: (ICON[c.id] || '') + strip(c.title) }));
    });
    var info = ui.el('button', { class: 'v2-info', type: 'button', popovertarget: 'card-rail', 'aria-label': 'What this rail shows', html: ui.INFO_SVG, style: 'position:absolute;right:-4px;top:-6px' });
    scale.insertBefore(info, scale.firstChild);   /* first in tab order: it explains the list below */
    railReady = true;
    markRail();
  }
  var railReady = false;
  function markRail() {
    if (!railReady) return;
    $$('[data-goto]', $('#rail')).forEach(function (a) {
      if (a.dataset.goto === cur) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    });
    var st = $('.v2-st[aria-current="page"]', scale);
    if (st) {
      needle.style.opacity = 1;
      needle.style.setProperty('--ny', (st.offsetTop + 7 + 7.5 - 4.5) + 'px');
    } else needle.style.opacity = 0;
  }
  (sim && sim.loadJSON ? sim.loadJSON('c_cw.json') : Promise.reject(new Error('no loader'))).then(buildRail).catch(function (e) {
    console.error(e);
    scale.appendChild(ui.el('li', { class: 've-error', text: 'Schedule data failed to load: ' + e.message }));
  });

  /* ------------------------------------------------------------- routing */
  function parseHash() {
    var h = decodeURIComponent((location.hash || '').slice(1));
    if (REDIRECT[h]) h = REDIRECT[h];
    if (!h) return { id: 'overview' };
    if (byId[h]) return { id: h };
    var d = doc.getElementById(h);
    if (d && d.tagName === 'DETAILS') { var ch = d.closest('[data-chapter]'); if (ch) return { id: ch.dataset.chapter, drawer: h }; }
    var sec = doc.getElementById('ch-' + h);
    if (sec) return { id: h };
    return { id: 'overview' };
  }
  function show(id, o) {
    o = o || {};
    var c = byId[id] || byId.overview;
    var changed = cur !== c.id;
    cur = c.id;
    chapters.forEach(function (x) {
      var on = x === c;
      if (x.el.hidden === on) x.el.hidden = !on;
      if (on && changed && !reduced() && !o.initial) { x.el.classList.remove('is-in'); void x.el.offsetWidth; x.el.classList.add('is-in'); }
    });
    markRail();
    /* phone bar */
    var bt = $('[data-bar-t]');
    if (bt) bt.innerHTML = (c.n ? '<b>' + c.n + '</b>' : '') + c.title;
    var nx = neighbours(c.id).next, bn = $('[data-bar-next]');
    if (bn) { if (nx) { bn.hidden = false; bn.href = '#' + nx.id; bn.dataset.nav = nx.id; } else bn.hidden = true; }
    if (c.beta != null) root.style.setProperty('--bp', ((c.beta - 0.6) / 1.0 * 100).toFixed(1) + '%');
    doc.title = c.id === 'overview' ? 'Finite-Rate Selection Law' : strip(c.title) + ' · Finite-Rate Selection Law';
    if (c.beta != null) emblems.forEach(function (e) { if (e.follow) e.retarget(c.beta); });
    /* a hidden chapter's video stops: sound never plays behind another chapter */
    $$('video').forEach(function (v) { if (!c.el.contains(v) && !v.paused) v.pause(); });
    closeRail();
    if (changed && !o.keepScroll) window.scrollTo(0, 0);
    visit(c);
    if (o.drawer) ui.openDrawer(o.drawer);
  }
  function go(id, drawerId) {
    var h = '#' + (drawerId || id);
    if (location.hash !== h) history.pushState(null, '', h);
    /* a click on the current chapter returns to its top, like a click on any other */
    if (id === cur && !drawerId) window.scrollTo(0, 0);
    show(id, { drawer: drawerId });
  }
  window.addEventListener('popstate', function () { var p = parseHash(); show(p.id, { drawer: p.drawer }); });
  window.addEventListener('hashchange', function () {
    var h = (location.hash || '').slice(1);
    if (REDIRECT[h]) history.replaceState(null, '', '#' + REDIRECT[h]);
    var p = parseHash(); show(p.id, { drawer: p.drawer });
  });
  doc.addEventListener('click', function (e) {
    var a = e.target.closest && e.target.closest('a[href^="#"]');
    if (!a || e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey) return;
    var id = a.getAttribute('href').slice(1);
    if (REDIRECT[id]) id = REDIRECT[id];
    if (byId[id]) { e.preventDefault(); go(id); return; }
    var d = doc.getElementById(id);
    if (d && d.tagName === 'DETAILS') { e.preventDefault(); go(d.closest('[data-chapter]').dataset.chapter, id); }
  });
  doc.addEventListener('keydown', function (e) {
    if (e.defaultPrevented || e.altKey || e.metaKey || e.ctrlKey) return;
    if (e.key === 'Escape' && rail.classList.contains('is-open')) { closeRail(); menu.focus(); return; }
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    var t = e.target;
    if (t && t.closest && t.closest('input, textarea, select, button, canvas, video, [role="radiogroup"], [role="slider"], [contenteditable], summary, [popover]')) return;
    var to = neighbours(cur)[e.key === 'ArrowRight' ? 'next' : 'prev'];
    if (to) { e.preventDefault(); go(to.id); }
  });

  /* skip link: focus the current chapter's heading, without routing through the hash */
  var skip = $('.v2-skip');
  if (skip) skip.addEventListener('click', function (e) {
    e.preventDefault();
    var c = byId[cur], h = c && $('.v2-title', c.el);
    if (!h) return;
    h.setAttribute('tabindex', '-1'); h.focus({ preventScroll: true }); window.scrollTo(0, 0);
  }, true);

  /* ------------------------------------------------------------ mounting */
  var visited = {};
  function visit(c) {
    if (visited[c.id]) return;
    visited[c.id] = true;
    ui.terms(c.el); ui.notes(c.el); ui.drawers(c.el); ui.cards(c.el); ui.math(c.el);
    $$('details[data-lazy]', c.el).forEach(function (d) {
      var opts = {}; try { opts = JSON.parse(d.dataset.opts || '{}'); } catch (e) { /* none */ }
      ui.lazyPlate(d, d.dataset.lazy, opts);
    });
    var slots = $$('[data-mount]', c.el);
    /* first load: hold the content after the main plate invisible until the plate has built
       its figure, so the Result line and drawers do not paint in the figure's place and then
       jump down (CLS). The hold always ends: on the figure, or after 3 s. */
    var main = $('.v2-plate[data-mount]', c.el);
    if (main && !main.dataset.state) {
      main.dataset.wait = '1';
      c.revealT = setTimeout(function () { reveal(c); }, 3000);
    }
    slots.forEach(function (s) { if (!s.dataset.state) s.appendChild(ui.el('div', { class: 'v2-pending', text: 'Loading figure…' })); });
    var mountAll = function () {
      slots.forEach(function (s) {
        if (s.dataset.state) return;
        var opts = {}; try { opts = JSON.parse(s.dataset.opts || '{}'); } catch (e) { /* none */ }
        var p = s.querySelector('.v2-pending'); if (p) p.remove();
        ui.mountComponent(s.dataset.mount, s, opts);
        s.dataset.state = 'mounted';
      });
      wirePlate(c);
      if (ui.allDetails()) $$('.v2-plate details', c.el).forEach(function (d) { d.open = true; });
    };
    if (c.id === 'test') testFigure(c.el);
    if (c.id === 'watch') filmPlayer(c.el);
    else stepClip(c);
    idle(mountAll);
  }

  function reveal(c) {
    clearTimeout(c.revealT);
    var main = $('.v2-plate[data-wait]', c.el);
    if (main) delete main.dataset.wait;
  }

  /* reading card mirrors and the live spec card for one chapter's plate.
     A component can build its figures after a fetch (reheat). Until its first figure note
     exists, the chip stays hidden and a MutationObserver waits; once the notes stop
     arriving for 250 ms (or after 20 s), the plate is wired as if it had mounted at once. */
  function wirePlate(c) {
    var plate = $('.v2-plate[data-mount]', c.el);
    if (!plate) return;
    var chip0 = $('[data-spec]', c.el);
    if (!$('.ve-figure__note', plate) && window.MutationObserver && !c.wireWaited) {
      c.wireWaited = true;
      if (chip0) chip0.hidden = true;
      var t = 0, cap = setTimeout(function () { done(); }, 20000);
      var mo = new MutationObserver(function () {
        if (!$('.ve-figure__note', plate)) return;
        clearTimeout(t); t = setTimeout(done, 250);
      });
      var done = function () { mo.disconnect(); clearTimeout(cap); clearTimeout(t); wirePlate(c); };
      mo.observe(plate, { childList: true, subtree: true });
      return;
    }
    $$('.v2-card .v2-mirror[data-src]', c.el).forEach(function (p) {
      var srcs = $$(p.dataset.src, plate).filter(function (s) { return !s.closest('.v2-plate--nested'); });
      if (!srcs.length) { p.remove(); return; }
      srcs.forEach(function (s, i) {
        var tgt = p;
        if (i) { tgt = ui.el('p', { class: 'v2-mirror' }); p.parentNode.insertBefore(tgt, p.nextSibling); p = tgt; }
        ui.mirror(s, tgt);
      });
    });
    var chip = $('[data-spec]', c.el);
    if (chip) {
      var notes = $$('.ve-figure__note', plate).map(function (n) {
        var f = n.closest('.ve-figure'), lab = f && f.querySelector('.ve-figure__plate');
        var parts = lab ? Array.prototype.map.call(lab.children, function (x) { return x.textContent.trim(); }).filter(Boolean) : [];
        return { el: n, label: lab ? (parts.length > 1 ? parts.join(' \u00b7 ') : lab.textContent) : '' };
      });
      chip.hidden = !notes.length;
      if (notes.length) ui.specCard(chip, notes, { id: chip.dataset.spec, title: 'What runs here' });
    }
    fillProvenance(c);
    reveal(c);
    if (ui.allDetails()) $$('.v2-plate details', c.el).forEach(function (d) { d.open = true; });
  }
  function fillProvenance(c) {
    var box = $('.v2-provenance', c.el);
    if (!box) return;
    if (!ui.allDetails()) { box.replaceChildren(); return; }
    box.replaceChildren();
    $$('.v2-platebar [popovertarget], .v2-platebar [data-spec]', c.el).forEach(function (b) {
      var card = doc.getElementById(b.getAttribute('popovertarget') || b.dataset.spec);
      if (!card) return;
      var copy = ui.el('div');
      copy.innerHTML = card.innerHTML;
      $$('.v2-card__x, button', copy).forEach(function (x) { x.remove(); });
      $$('[id]', copy).forEach(function (x) { x.removeAttribute('id'); });
      box.appendChild(copy);
    });
  }
  ui.onAllDetails(function (on) {
    $$('.v2-plate details').forEach(function (d) { d.open = on; });
    chapters.forEach(function (c) { if (visited[c.id]) fillProvenance(c); });
  });

  /* ---------------------------------------------------------- phone drawer */
  var rail = $('#rail'), menu = $('.v2-bar__menu'), scrim = $('[data-scrim]');
  function openRail() {
    rail.classList.add('is-open'); scrim.classList.add('is-open');
    menu.setAttribute('aria-expanded', 'true');
    markRail();
    var f = $('[aria-current="page"]', rail) || $('a, button', rail);
    setTimeout(function () { if (f) f.focus({ preventScroll: false }); }, 30);
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

  /* ------------------------------------------------------- theme, details */
  var themeBtns = $$('[data-theme-set]');
  function setTheme(t, save) {
    ui.setTheme(t);
    themeBtns.forEach(function (b) { b.setAttribute('aria-checked', String(b.dataset.themeSet === t)); });
    if (save) ui.store.set('ve2.theme', t);
  }
  themeBtns.forEach(function (b) { b.addEventListener('click', function () { setTheme(b.dataset.themeSet, true); }); });
  setTheme(ui.store.get('ve2.theme') || 'system', false);
  var allBtn = $('[data-all-details]');
  allBtn.setAttribute('aria-checked', String(ui.allDetails()));
  allBtn.addEventListener('click', function () { allBtn.setAttribute('aria-checked', String(ui.allDetails(!ui.allDetails()))); });
  ui.cards(doc);

  /* ======================================================================
     The pitchfork emblem: a real Curie–Weiss chain,
       p(x) ∝ exp(β/(2(N−1)) Σ_{i≠j} x_i x_j + h Σ x_i),  h = βG,
     heat-bath updates at random sites, one sweep per frame. Each cycle
     cools (raises β linearly) from 0.6 to a target, holds 6 s, reheats to 0.6.
     A trace that ends above β = 1.05 takes the colour of its final sign;
     otherwise it stays grey. G√N is fixed across sizes (0.0067 · √256).
     ====================================================================== */
  var B0 = 0.6, B1 = 1.6, DFULL = 420, HOLD = 360, REHEAT = 50, REST = 30;
  function rng(seed) { var s = seed >>> 0 || 1; return function () { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; }; }
  function Emblem(o) {
    this.o = o; this.N = o.N; this.G = 0.0067 * Math.sqrt(256 / o.N);
    this.r = rng(o.seed); this.x = new Int8Array(o.N); this.avg = new Float32Array(o.N); this.S = 0;
    for (var i = 0; i < o.N; i++) { this.x[i] = this.r() < 0.5 ? 1 : -1; this.S += this.x[i]; }
    this.follow = !!o.follow; this.target = this.eff(o.target || B1);
    this.traces = []; this.cur = null; this.phase = 'descend'; this.step = 0; this.beta = B0; this.m = this.S / o.N;
    this.D = this.descentLen(); this.newTrace();
    this.on = false; this.raf = 0; this.vis = {};
  }
  /* a chapter at the start level (the overview, β 0.60) would make a zero-length descent and a
     dead dot; the emblem then runs the full descent to β 1.6, as the frontispiece does */
  Emblem.prototype.eff = function (b) { return this.follow && b <= B0 + 0.05 ? B1 : b; };
  Emblem.prototype.descentLen = function () { return Math.max(1, Math.round(DFULL * (this.target - B0) / (B1 - B0))); };
  Emblem.prototype.newTrace = function () { this.cur = { pts: new Float32Array(2 * (this.D + 2)), n: 0, fate: 0 }; };
  Emblem.prototype.sweep = function (beta) {
    var N = this.N, x = this.x, r = this.r, h = beta * this.G, J = beta / (N - 1), S = this.S;
    for (var k = 0; k < N; k++) {
      var i = (r() * N) | 0;
      var up = r() < 1 / (1 + Math.exp(-2 * (J * (S - x[i]) + h))) ? 1 : -1;
      if (up !== x[i]) { S += up - x[i]; x[i] = up; }
    }
    this.S = S;
    var a = this.avg; for (var j = 0; j < N; j++) a[j] += 0.06 * (x[j] - a[j]);
    return S / N;
  };
  Emblem.prototype.tick = function () {
    if (this.phase === 'descend') {
      this.beta = B0 + (this.target - B0) * Math.min(1, this.step / this.D);
      this.m = this.sweep(this.beta);
      if (this.cur.n < this.D + 2) { this.cur.pts[2 * this.cur.n] = this.beta; this.cur.pts[2 * this.cur.n + 1] = this.m; this.cur.n++; }
      if (++this.step > this.D) { this.phase = 'hold'; this.step = 0; }
    } else if (this.phase === 'hold') {
      this.m = this.sweep(this.target);
      if (++this.step >= HOLD) {
        this.cur.fate = this.target > 1.05 ? (this.m > 0 ? 1 : -1) : 0;
        this.traces.push(this.cur); if (this.traces.length > 24) this.traces.shift();
        this.phase = 'reheat'; this.step = 0;
      }
    } else {
      this.beta = this.target - (this.target - B0) * Math.min(1, this.step / REHEAT);
      this.m = this.sweep(this.beta);
      if (++this.step >= REHEAT + REST) { this.phase = 'descend'; this.step = 0; this.D = this.descentLen(); this.newTrace(); }
    }
  };
  Emblem.prototype.cycle = function () { var n = this.D + 1 + HOLD + REHEAT + REST; for (var k = 0; k < n; k++) this.tick(); };
  Emblem.prototype.retarget = function (b) {
    b = this.eff(b);
    if (Math.abs(b - this.target) < 1e-6) return;
    this.target = b;
    /* a running descent bends to the new level; a hold or a reheat finishes first */
    if (this.phase === 'descend') {
      var D = this.descentLen(), done = this.step;
      this.D = D; if (done > D) { this.step = D; }
      var old = this.cur; this.newTrace(); for (var i = 0; i < Math.min(old.n, this.D + 2); i++) { this.cur.pts[2 * i] = old.pts[2 * i]; this.cur.pts[2 * i + 1] = old.pts[2 * i + 1]; } this.cur.n = Math.min(old.n, this.D + 2);
    } else if (this.phase === 'hold') { this.phase = 'reheat'; this.step = 0; this.cur.fate = this.beta > 1.05 ? (this.m > 0 ? 1 : -1) : 0; this.traces.push(this.cur); if (this.traces.length > 24) this.traces.shift(); }
    if (reduced()) { this.settle(); this.paint(); }
  };
  Emblem.prototype.settle = function () {   /* reduced motion: run to a committed frame at the target */
    var guard = 0;
    while (this.phase !== 'hold' && guard++ < 4000) this.tick();
    for (var k = 0; k < 120; k++) this.sweep(this.target);
    this.m = this.S / this.N;
  };
  Emblem.prototype.phaseWord = function () { return this.phase === 'reheat' ? (this.step < REHEAT ? 'reheating' : 'reset') : this.phase === 'hold' ? 'holding' : 'cooling'; };
  Emblem.prototype.paint = function () {
    var self = this;
    this.o.canvases.forEach(function (cv) { if (cv.offsetParent !== null) self.draw(cv); });
    var txt = 'β ' + this.beta.toFixed(2) + ' · m ' + (this.m >= 0 ? '+' : '−') + Math.abs(this.m).toFixed(2) + ' · ' + this.phaseWord();
    if (this.o.line && this.o.line.textContent !== txt) this.o.line.textContent = txt;
  };
  Emblem.prototype.draw = function (cv) {
    var W = cv.clientWidth, H = cv.clientHeight; if (!W || !H) return;
    var mini = W < 60, dpr = Math.min(2, window.devicePixelRatio || 1);
    if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    var ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    var c = ui.colors(), A = c.modeA, B = c.modeB, ink = c.ink, ink3 = c.ink3, rule = c.ruleStrong, noise = c.noise;
    var stripH = mini ? 0 : (this.o.strip ? 12 : 0);
    var padL = mini ? 1 : (this.o.axis ? 22 : 2), padR = mini ? 1 : 4, top = mini ? 2 : 6, bot = H - stripH - (mini ? 2 : (this.o.axis ? 20 : 14));
    var sx = function (b) { return padL + (b - B0) / (B1 - B0) * (W - padL - padR); };
    var sy = function (m) { return top + (1 - m) / 2 * (bot - top); };
    ctx.lineWidth = 1; ctx.strokeStyle = rule;
    ctx.globalAlpha = 0.55; ctx.beginPath(); ctx.moveTo(sx(B0), Math.round(sy(0)) + 0.5); ctx.lineTo(sx(B1), Math.round(sy(0)) + 0.5); ctx.stroke();
    ctx.globalAlpha = 0.9; ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(Math.round(sx(1)) + 0.5, top); ctx.lineTo(Math.round(sx(1)) + 0.5, bot + (mini ? 0 : 3)); ctx.stroke(); ctx.setLineDash([]);
    if (this.follow && !mini) {   /* the chapter's level */
      ctx.globalAlpha = 0.7; ctx.strokeStyle = ink3; ctx.beginPath(); ctx.moveTo(Math.round(sx(this.target)) + 0.5, bot - 4); ctx.lineTo(Math.round(sx(this.target)) + 0.5, bot + 3); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    if (!mini) {
      ctx.fillStyle = ink3; ctx.font = '450 ' + (this.o.axis ? 11 : 10) + 'px ' + c.fontMono; ctx.textBaseline = 'top'; ctx.textAlign = 'center';
      ctx.fillText('β = 1', sx(1), bot + 4);
      if (this.o.axis) {
        ctx.textAlign = 'left'; ctx.fillText('β 0.6 · hot', sx(B0), bot + 4);
        ctx.textAlign = 'right'; ctx.fillText('cold · β 1.6', sx(B1), bot + 4);
        ctx.textBaseline = 'middle'; ctx.fillStyle = A; ctx.fillText('A', padL - 8, sy(0.78)); ctx.fillStyle = B; ctx.fillText('B', padL - 8, sy(-0.78));
      }
    }
    var n = this.traces.length;
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    for (var t = 0; t < n; t++) {
      var tr = this.traces[t];
      ctx.strokeStyle = tr.fate > 0 ? A : tr.fate < 0 ? B : noise;
      /* the last four descents keep near-full mode colour so the rail reads as physics at a glance */
      ctx.globalAlpha = t >= n - 4 ? 0.55 + 0.1 * (t - n + 4) : 0.14 + 0.36 * (t + 1) / n;
      ctx.lineWidth = mini ? 0.8 : (this.o.axis ? 1 : 0.8);
      path(ctx, tr, sx, sy); ctx.stroke();
    }
    ctx.globalAlpha = 1;
    if (this.cur && this.cur.n > 1 && this.phase !== 'reheat') {
      ctx.strokeStyle = ink; ctx.lineWidth = mini ? 1 : (this.o.axis ? 1.35 : 1.1); path(ctx, this.cur, sx, sy); ctx.stroke();
      var k = this.cur.n - 1, hx = sx(this.cur.pts[2 * k]), hy = sy(this.phase === 'hold' ? this.m : this.cur.pts[2 * k + 1]);
      ctx.fillStyle = ink; ctx.beginPath(); ctx.arc(hx, hy, mini ? 1.6 : (this.o.axis ? 3 : 2.2), 0, 7); ctx.fill();
    }
    if (stripH) {   /* one mark per spin (subsampled to 256), coloured by its recent average sign */
      var M = Math.min(256, this.N), step = this.N / M, pitch = (W - padL - padR) / M, y0 = H - stripH + 2;
      for (var j = 0; j < M; j++) {
        var v = this.avg[Math.floor(j * step)], s = Math.abs(v);
        ctx.fillStyle = s < 0.5 ? rule : (v > 0 ? A : B);
        ctx.globalAlpha = s < 0.5 ? 0.6 : 0.35 + 0.6 * s;
        ctx.fillRect(padL + j * pitch, y0 + (v > 0 ? 0 : 3), Math.max(0.7, pitch * 0.55), stripH - 6);
      }
      ctx.globalAlpha = 1;
    }
  };
  function path(ctx, tr, sx, sy) {
    ctx.beginPath();
    for (var j = 0; j < tr.n; j++) { var X = sx(tr.pts[2 * j]), Y = sy(tr.pts[2 * j + 1]); if (j) ctx.lineTo(X, Y); else ctx.moveTo(X, Y); }
  }
  Emblem.prototype.start = function () {
    var self = this;
    if (window.IntersectionObserver) {
      var io = new IntersectionObserver(function (es) { es.forEach(function (e) { self.vis[e.target.dataset.emblem] = e.isIntersecting; }); self.check(); });
      this.o.canvases.forEach(function (cv) { io.observe(cv); });
    } else this.o.canvases.forEach(function (cv) { self.vis[cv.dataset.emblem] = true; });
    doc.addEventListener('visibilitychange', function () { self.check(); });
    ui.onTheme(function () { self.paint(); });
    window.addEventListener('resize', function () { self.paint(); });
    ui.onReducedMotion(function () { if (reduced()) { self.settle(); } self.check(); self.paint(); });
    this.check();
  };
  Emblem.prototype.shown = function () {
    var self = this;
    return !doc.hidden && !this.paused && this.o.canvases.some(function (cv) { return self.vis[cv.dataset.emblem] && cv.offsetParent !== null; });
  };
  Emblem.prototype.check = function () {
    var self = this, want = this.shown() && !reduced();
    if (reduced() && this.shown()) this.paint();
    if (want && !this.on) {
      this.on = true;
      var f = function () {
        if (!self.on) { self.raf = 0; return; }
        if (!self.shown()) { self.on = false; self.raf = 0; return; }
        self.tick(); self.paint();
        self.raf = requestAnimationFrame(f);
      };
      this.raf = requestAnimationFrame(f);
    } else if (!want && this.on) { this.on = false; if (this.raf) cancelAnimationFrame(this.raf); this.raf = 0; }
  };
  /* fast-forward about k whole cycles in idle slices. A slice runs single sweeps until its
     time budget (8 ms, or the idle deadline if that is shorter) is spent, so no task passes
     50 ms even for the 1,024-variable frontispiece (one full cycle there is ~60 ms). */
  Emblem.prototype.prefill = function (k, done) {
    var self = this, left = k * (this.D + 1 + HOLD + REHEAT + REST);
    (function next(dl) {
      var t0 = performance.now(), budget = 8;
      if (dl && dl.timeRemaining && !dl.didTimeout) budget = Math.max(2, Math.min(8, dl.timeRemaining() - 2));
      while (left > 0 && performance.now() - t0 < budget) { self.tick(); left--; }
      if (left > 0) { idle(next); return; }
      if (done) done(); self.paint();
    })();
  };

  var emblems = [];
  var railCanvases = $$('canvas[data-emblem="rail"], canvas[data-emblem="mini"]');
  if (railCanvases.length) {
    var er = new Emblem({ N: 256, seed: 7, follow: true, target: 0.6, strip: true, canvases: railCanvases, line: $('[data-emblem-line="rail"]') });
    emblems.push(er);
  }
  var fc = $('canvas[data-emblem="front"]');
  if (fc) {
    var ef = new Emblem({ N: 1024, seed: 11, follow: false, target: B1, strip: true, axis: true, canvases: [fc], line: $('[data-emblem-line="front"]') });
    emblems.push(ef);
    /* one physics figure leads: while the frontispiece is on screen, the rail emblem
       pauses on its last frame and dims, and it resumes when the frontispiece leaves */
    if (er && window.IntersectionObserver) {
      var railBox = ($('canvas[data-emblem="rail"]') || railCanvases[0]).closest('.v2-emblem');
      new IntersectionObserver(function (es) {
        var on = es.some(function (e) { return e.isIntersecting; }) && fc.offsetParent !== null;
        er.paused = on;
        if (railBox) railBox.classList.toggle('is-yielding', on);
        /* a paused emblem that never ran would show an empty frame: run one cycle offscreen */
        if (on && !er.traces.length && !er.prefilled) { er.prefilled = true; er.prefill(1, function () { er.paint(); }); }
        er.check();
      }, { threshold: 0.15 }).observe(fc);
    }
  }

  /* ======================================================================
     Chapter 7: prediction against measurement. Rows: the p scan
     at N = 2,048 (data/table1.json), then the size scan at p = 3
     (data/size_scan.json). The clock switch multiplies each prediction by
     r(N) from data/sampler_clock.json. Gap = measured / predicted − 1 from
     the unrounded data, or "inside" when the prediction is in the interval.
     ====================================================================== */
  var testBuilt = false;
  function testFigure(sec) {
    if (testBuilt) return; testBuilt = true;
    var fig = $('[data-vernier]', sec), host = $('[data-vrows]', fig), star = $('[data-vstar]', fig);
    var NS = 'http://www.w3.org/2000/svg', state = { clock: false }, rows = null;
    var D0 = 0.10, D1 = 0.30;
    function E(tag, at, txt) { var n = doc.createElementNS(NS, tag); for (var k in at) n.setAttribute(k, at[k]); if (txt != null) n.textContent = txt; return n; }
    Promise.all([sim.loadJSON('table1.json'), sim.loadJSON('size_scan.json'), sim.loadJSON('sampler_clock.json')]).then(function (r) {
      var rN = {}; r[2].rows.forEach(function (x) { rN[x.N] = x.r; });
      var P = r[0].rows.map(function (x) { return { g: 'p', lab: 'p = ' + x.p, sub: x.p === 3 ? 'paper’s setting' : 'A<sub>' + x.p + '</sub> = ' + x.amplification.toFixed(3), main: x.p === 3, pred: x.c_predicted, meas: x.c_measured, ci: x.ci95, r: rN[2048] }; });
      var S = r[1].rows.filter(function (x) { return x.N !== 2048; }).map(function (x) { return { g: 'N', lab: 'N = ' + ui.fmt(x.N), sub: 'c<sub>CW</sub> = ' + x.c_CW.toFixed(3), pred: x.c_predicted, meas: x.c_measured, ci: x.ci95, r: rN[x.N] }; });
      rows = P.concat(S);
      build();
    }).catch(function (e) { host.replaceChildren(ui.el('p', { class: 've-error', text: 'Data failed to load: ' + e.message })); });
    function track(svg) { return svg.clientWidth || svg.getBoundingClientRect().width || 400; }
    function build() {
      host.replaceChildren();
      host.appendChild(ui.el('div', { class: 'v2-vgroup' }, ui.el('span', { text: 'Patterns · N = 2,048' }), ui.el('span'), ui.el('span', { class: 'v2-vgroup__gap', html: 'Gap<small>meas. vs pred.</small>' })));
      rows.forEach(function (rw, i) {
        if (i && rw.g !== rows[i - 1].g) host.appendChild(ui.el('div', { class: 'v2-vgroup' }, ui.el('span', { text: 'Size · p = 3' }), ui.el('span'), ui.el('span')));
        var svg = E('svg', { 'aria-hidden': 'true' });
        var sub = ui.el('small', { html: rw.sub }), lab = ui.el('span', { class: 'v2-vrow__n', text: rw.lab });
        /* the (i) sits inline after "p = 3", on the label line, never on the axis baseline */
        if (rw.main) { var b = ui.el('button', { class: 'v2-info', type: 'button', popovertarget: 'test-p3', 'aria-label': 'Why this row reads 0.201', html: ui.INFO_SVG }); lab.appendChild(b); }
        var row = ui.el('div', { class: 'v2-vrow' + (rw.main ? ' is-main' : '') },
          ui.el('span', { class: 'v2-vrow__l' }, lab, sub),
          ui.el('div', null, svg), ui.el('span', { class: 'v2-vrow__g' }));
        rw.svg = svg; rw.gapEl = row.lastChild; rw.row = row;
        host.appendChild(row);
      });
      var axis = E('svg', { 'aria-hidden': 'true' });
      host.appendChild(ui.el('div', { class: 'v2-vaxis' }, ui.el('span'), ui.el('div', null, axis), ui.el('span')));
      rows.axis = axis;
      draw(true);
      ui.cards(sec);
    }
    function draw(first) {
      if (!rows) return;
      var c = ui.colors();
      rows.forEach(function (rw) {
        var svg = rw.svg, W = track(svg), pad = 10, sx = function (v) { return pad + (v - D0) / (D1 - D0) * (W - 2 * pad); }, y = 23;
        var pv = state.clock ? rw.pred * rw.r : rw.pred;
        if (first || svg.__w !== W) {
          svg.__w = W; svg.replaceChildren();
          svg.setAttribute('viewBox', '0 0 ' + W + ' 46');
          svg.appendChild(E('line', { x1: sx(D0), x2: sx(D1), y1: y + 9, y2: y + 9, stroke: c.rule, 'stroke-width': 1 }));
          for (var t = 0; t <= 20; t++) { var v = D0 + t * 0.01, maj = t % 5 === 0; svg.appendChild(E('line', { x1: sx(v), x2: sx(v), y1: y + 9, y2: y + (maj ? 15 : 12), stroke: maj ? c.ruleStrong : c.rule, 'stroke-width': 1 })); }
          svg.appendChild(E('line', { x1: sx(rw.ci[0]), x2: sx(rw.ci[1]), y1: y, y2: y, stroke: c.data, 'stroke-width': 1.25 }));
          [rw.ci[0], rw.ci[1]].forEach(function (v) { svg.appendChild(E('line', { x1: sx(v), x2: sx(v), y1: y - 4, y2: y + 4, stroke: c.data, 'stroke-width': 1 })); });
          svg.appendChild(E('circle', { cx: sx(rw.meas), cy: y, r: 3.6, fill: c.data }));
          rw.tri = E('path', { class: 'v-pred', d: 'M -5.5 -9 L 5.5 -9 L 0 0 Z', fill: 'none', stroke: c.theory, 'stroke-width': 1.2, transform: 'translate(' + sx(pv).toFixed(1) + ' ' + (y - 6) + ')' });
          svg.appendChild(rw.tri);
        } else rw.tri.setAttribute('transform', 'translate(' + sx(pv).toFixed(1) + ' ' + (y - 6) + ')');
        var inside = pv >= rw.ci[0] && pv <= rw.ci[1], gap = rw.meas / pv - 1, gs = Math.round(100 * gap);
        rw.gapEl.textContent = inside ? 'inside' : (gs > 0 ? '+' : '−') + Math.abs(gs) + '%';
        rw.gapEl.classList.toggle('is-in', inside);
        rw.row.title = rw.lab + ' (' + strip(rw.sub) + ')' + ': predicted ' + pv.toFixed(3) + (state.clock ? '*' : '') + ', measured ' + rw.meas.toFixed(3) + ' [' + rw.ci[0].toFixed(3) + ', ' + rw.ci[1].toFixed(3) + ']';
      });
      var ax = rows.axis, W = track(ax), pad = 10;
      ax.setAttribute('viewBox', '0 0 ' + W + ' 24'); ax.replaceChildren();
      [0.10, 0.15, 0.20, 0.25, 0.30].forEach(function (v) { ax.appendChild(E('text', { x: pad + (v - D0) / (D1 - D0) * (W - 2 * pad), y: 16, 'text-anchor': 'middle' }, v.toFixed(2))); });
      host.setAttribute('aria-label', rows.map(function (rw) { return rw.row.title; }).join('; '));
      star.textContent = state.clock ? '* Each prediction × r(N), the one-pattern ratio (1.063 at N = 2,048), computed on this page. The p = 1 row then nearly agrees: its own ratio is about 1.06 (0.127/0.120, Table 1), against r = 1.063 from separate one-pattern runs.' : '';
    }
    $$('[data-clock]', fig).forEach(function (b) {
      b.addEventListener('click', function () {
        state.clock = b.dataset.clock === '1';
        $$('[data-clock]', fig).forEach(function (x) { x.setAttribute('aria-checked', String(x === b)); });
        draw(false);
      });
    });
    var rt = 0; window.addEventListener('resize', function () { cancelAnimationFrame(rt); rt = requestAnimationFrame(function () { draw(false); }); });
    ui.onTheme(function () { draw(true); });
  }


  /* ======================================================================
     The narrated film (visual/film/, a manim film of the paper in nine
     scenes). media/media.json, written by tools/media.mjs, lists the whole
     film (one 720p file, its captions, poster and scene start times) and one
     clip per chapter. Nothing plays until the reader presses play; captions
     are on by default. If the manifest is missing (a copy without media/),
     the clips stay absent and the Watch chapter says so.
     ====================================================================== */
  var mediaP = null;
  function media() {
    if (!mediaP) mediaP = fetch('media/media.json').then(function (r) {
      if (!r.ok) throw new Error('media/media.json: HTTP ' + r.status);
      return r.json();
    });
    return mediaP;
  }
  function mmss(t) { t = Math.max(0, Math.round(t)); return Math.floor(t / 60) + ':' + ('0' + (t % 60)).slice(-2); }
  var PLAY_SVG = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M5 3.2v9.6L12.8 8z"/></svg>';
  var CLOSE_SVG = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4.5 4.5l7 7M11.5 4.5l-7 7"/></svg>';
  /* one screen: a <video> with native controls, an English captions track shown by default,
     the poster, and a large play mark over the poster until the first play */
  function screen(o) {
    var v = doc.createElement('video');
    v.className = 'v2-screen__v';
    /* at rest: the poster and one play mark; the native controls arrive with the first play */
    v.controls = false; v.preload = o.preload || 'none'; v.playsInline = true;
    v.setAttribute('playsinline', ''); v.setAttribute('aria-label', o.label);
    v.poster = o.poster; v.src = o.src;
    var tr = doc.createElement('track');
    tr.kind = 'captions'; tr.srclang = 'en'; tr.label = 'English'; tr.default = true;
    v.appendChild(tr);
    /* the captions ship as .vtt.txt (the host does not serve text/vtt); give the track a
       text/vtt blob so every browser accepts it */
    fetch(o.vtt).then(function (r) { return r.ok ? r.text() : Promise.reject(r.status); })
      .then(function (t) { tr.src = URL.createObjectURL(new Blob([t], { type: 'text/vtt' })); })
      .catch(function () { /* no captions; the video still plays */ });
    /* captions on at first load; after that the reader's choice in the native menu stands */
    v.addEventListener('loadedmetadata', function () { var t = v.textTracks && v.textTracks[0]; if (t && t.mode !== 'showing') t.mode = 'showing'; }, { once: true });
    var big = ui.el('button', { class: 'v2-screen__play', type: 'button', 'aria-label': 'Play ' + o.label + ' (' + mmss(o.duration) + ')', html: PLAY_SVG });
    big.addEventListener('click', function () { var p = v.play(); if (p && p.catch) p.catch(function () { /* the reader can use the controls */ }); });
    var box = ui.el('div', { class: 'v2-screen' }, v, big);
    /* a clip may place its play mark on an empty patch of its poster (scenes.json "play",
       percentages of the frame); otherwise the mark sits lower left */
    if (o.play && isFinite(o.play.x) && isFinite(o.play.y)) {
      box.classList.add('v2-screen--placed');
      box.style.setProperty('--play-x', o.play.x + '%');
      box.style.setProperty('--play-y', o.play.y + '%');
    }
    /* after the first play the mark is gone for good: move focus to the video (so Space and
       Enter pause it) and take the mark out of the tab order and the accessibility tree */
    v.addEventListener('play', function () {
      box.classList.add('is-played'); v.controls = true;
      if (doc.activeElement === big) { v.setAttribute('tabindex', '0'); v.focus({ preventScroll: true }); }
      big.hidden = true; big.setAttribute('tabindex', '-1'); big.setAttribute('aria-hidden', 'true');
    });
    return { box: box, video: v };
  }

  function filmPlayer(sec) {
    var host = $('[data-film-screen]', sec), toc = $('[data-film-toc]', sec);
    media().then(function (m) {
      var f = m.film, sc = screen({ src: f.src, vtt: f.vtt, poster: f.poster, duration: f.duration, play: f.play, preload: 'metadata', label: 'the narrated film' });
      var v = sc.video;
      host.replaceChildren(sc.box);
      var meta = $('[data-film-meta]', sec);
      if (meta) meta.textContent = Math.round(f.duration / 60) + ' min · narrated · captions on';
      var rows = f.chapters.map(function (ch, i) {
        var c = byId[ch.chapter], end = i + 1 < f.chapters.length ? f.chapters[i + 1].start : f.duration;
        var b = ui.el('button', { class: 'v2-toc__go', type: 'button', 'aria-label': 'Play from ' + mmss(ch.start) + ': ' + ch.title },
          ui.el('span', { class: 'v2-toc__t', text: mmss(ch.start) }),
          ui.el('span', { class: 'v2-toc__n', text: c && c.n !== '' ? c.n : '' }),
          ui.el('span', { class: 'v2-toc__h', html: ch.title.replace(/[&<>]/g, function (x) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[x]; }).replace(/ c$/, ' <i>c</i>') }));
        b.addEventListener('click', function () {
          v.currentTime = ch.start + 0.02;
          var p = v.play(); if (p && p.catch) p.catch(function () { /* paused; the poster stays */ });
          /* a lower row may leave the player above the fold or under the phone bar: bring it
             into view; the focus stays on the row so the keyboard reader can go on in the list */
          var r = sc.box.getBoundingClientRect(), bar = $('.v2-bar'), barH = bar && bar.offsetHeight ? bar.offsetHeight : 0;
          if (r.top < barH || r.bottom > window.innerHeight) sc.box.scrollIntoView({ block: 'start', behavior: reduced() ? 'auto' : 'smooth' });
        });
        var li = ui.el('li', { class: 'v2-toc__row' }, b);
        if (c) li.appendChild(ui.el('a', { class: 'v2-toc__fig', href: '#' + c.id, dataset: { nav: c.id }, title: 'Open the chapter and its figures', 'aria-label': 'Open ' + label(c) + ' and its figures', html: '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2 6h7.2M6.4 3l3 3-3 3"/></svg>' }));
        toc.appendChild(li);
        return { li: li, a: ch.start, b: end };
      });
      var mark = function () {
        var t = v.currentTime;
        rows.forEach(function (r) { var on = t >= r.a - 0.01 && t < r.b - 0.01 && (v.played.length > 0 || t > 0); r.li.classList.toggle('is-now', on); if (on) r.li.setAttribute('aria-current', 'time'); else r.li.removeAttribute('aria-current'); });
      };
      ['timeupdate', 'seeked', 'play'].forEach(function (e) { v.addEventListener(e, mark); });
    }).catch(function () {
      host.replaceChildren(ui.el('p', { class: 'v2-screen__none', text: 'The film is not in this copy of the page (media/ is built from visual/film/ by tools/media.mjs).' }));
    });
  }

  /* "Watch this step · 1:43": a compact control under the chapter's lede that opens the
     chapter's narrated clip above the figure; the live figure stays the chapter's content */
  function stepClip(c) {
    var head = $('.v2-head', c.el), top = $('.v2-top', c.el);
    if (!head || !top) return;
    media().then(function (m) {
      var k = m.clips && m.clips[c.id];
      if (!k) return;
      var id = 'clip-' + c.id;
      var btn = ui.el('button', { class: 'v2-watch', type: 'button', 'aria-expanded': 'false', 'aria-controls': id },
        ui.el('span', { class: 'v2-watch__i', 'aria-hidden': 'true', html: PLAY_SVG }),
        ui.el('span', { class: 'v2-watch__t', text: 'Watch this step' }),
        ui.el('span', { class: 'v2-watch__d', text: mmss(k.duration) }));
      var line = ui.el('p', { class: 'v2-watchline' }, btn);
      head.appendChild(line);
      var panel = null, sc = null;
      /* wide: the clip sits between the header row and the figure, as wide as a plate.
         Single column (the margin note drops under the lede): the clip goes straight under
         its control, so the note never separates the pill from the screen it opens */
      var narrow = window.matchMedia('(max-width: 1179px)');
      var place = function () {
        var want = narrow.matches ? line : top;
        if (panel.previousSibling !== want) want.parentNode.insertBefore(panel, want.nextSibling);
        panel.classList.toggle('v2-clip--inhead', narrow.matches);
      };
      btn.addEventListener('click', function () {
        var open = btn.getAttribute('aria-expanded') !== 'true';
        if (open && !panel) {
          sc = screen({ src: k.src, vtt: k.vtt, poster: k.poster, duration: k.duration, play: k.play, preload: 'metadata', label: 'the narrated clip of this chapter' });
          var foot = ui.el('p', { class: 'v2-clip__foot' },
            ui.el('span', { text: 'Narrated · ' + mmss(k.duration) + ' · from the film' }),
            ui.el('a', { href: '#watch', dataset: { nav: 'watch' }, text: 'The whole film' }));
          panel = ui.el('div', { class: 'v2-clip', id: id, role: 'region', 'aria-label': 'Narrated clip: ' + strip(c.title) }, sc.box, foot);
          place();
          if (narrow.addEventListener) narrow.addEventListener('change', place);
          else if (narrow.addListener) narrow.addListener(place);
        }
        btn.setAttribute('aria-expanded', String(open));
        btn.querySelector('.v2-watch__t').textContent = open ? 'Hide the clip' : 'Watch this step';
        btn.querySelector('.v2-watch__i').innerHTML = open ? CLOSE_SVG : PLAY_SVG;
        panel.hidden = !open;
        if (!open) sc.video.pause();
        else if (!reduced()) { panel.classList.remove('is-in'); void panel.offsetWidth; panel.classList.add('is-in'); }
      });
    }).catch(function () { /* no media in this copy: no clip control */ });
  }

  /* ---------------------------------------------------------------- go */
  VE.shell = { emblems: emblems };
  if ('scrollRestoration' in history) history.scrollRestoration = 'manual';
  var p0 = parseHash();
  if (p0.id && location.hash && !byId[(location.hash || '').slice(1)] && !p0.drawer) history.replaceState(null, '', '#' + p0.id);
  if (REDIRECT[(location.hash || '').slice(1)]) history.replaceState(null, '', '#' + (REDIRECT[location.hash.slice(1)]));
  show(p0.id, { drawer: p0.drawer, initial: true });
  emblems.forEach(function (e) {
    if (reduced()) { e.prefill(e === emblems[0] ? 0 : 10, function () { e.settle(); e.start(); }); }
    else if (e.o.axis) e.prefill(10, function () { e.start(); });
    else e.start();
  });
})();
