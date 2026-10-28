/* lib/sim.js -- window.VE.sim: validated simulation kernels for the Visual Edition.
 *
 * Paper: ../paper/main.tex, "A Finite-Rate Selection Law for Annealed Gibbs Sampling".
 * Every exact Curie-Weiss (CW) routine here mirrors the Python in ../src and is tested
 * against it by lib/sim.test.mjs (reference values: data/ref/reference.json, produced by
 * data/ref/make_reference.py).  Run the tests with:   node lib/sim.test.mjs
 *
 * Conventions (identical to ../src and the paper):
 *   N spins x_i in {-1,+1}; k = number of +1 spins, k = 0..N; m = 2k/N - 1.
 *   p(x) ∝ exp( beta/(2(N-1)) Σ_{i≠j} x_i x_j + h Σ_i x_i )           (Sec. 2)
 *   The bias is written h = beta*G (G is a fixed field in the energy).    (Sec. 3)
 *   Time unit = one sweep = N attempted single-spin updates.               (Sec. 2, App. A)
 *   The favoured mode is m > 0 (G > 0 favours it); q_eq excludes the tie k = N/2.
 *
 * Performance (measured by the BENCH lines of lib/sim.test.mjs, node 24 on an ARM64 desktop CPU;
 * expect a laptop browser within ~2x).  Budget ~8 ms of simulation per 60-fps frame:
 *   CWChains (lumped, exact)     ~2.6e7 jump events/s => ~2e5 events per frame.  Events per
 *                                sweep per chain = W+_k + W-_k: ≈ N/2 near m = 0, ≈ 0.1 N deep
 *                                in a mode at beta = 1.6.  So per frame, near speciation:
 *                                chains × N × sweeps ≲ 4e5, e.g. 400 chains × N=256 × 4 sweeps
 *                                or 1000 × 1024 × 0.4.
 *   CWSpins (one chain, spins)   ~2.5e7 spin updates/s => N=4096: ~50 sweeps per frame.
 *   Hopfield (p = 3)             ~1.9e7 spin updates/s => 64 chains × N=512 × 4.5 sweeps per frame.
 *   CTPropagator (exact law)     ~25(N+1) flops per implicit step; the full paper protocol
 *                                (884 sweeps, dt = 0.02) takes ~2.0 s at N=2048, ~0.25 s at N=256.
 *                                Always drive it through runChunked (longest step 3.7 ms at N=1024).
 * In node, do not load this file into a vm.Context for timing: global lookups through a vm
 * context are ~40x slower; evaluate it with a plain `window` object instead (see the test).
 * Long jobs are generators (protocolGen, CTPropagator.evolveGen) that yield often;
 * VE.sim.runChunked(gen, {budgetMs}) runs them in ≤ budgetMs slices so a component never blocks the main thread for more than ~budgetMs (default 12 ms).
 */
(function (root) {
  'use strict';
  const VE = (root.VE = root.VE || {});

  const LN2 = Math.LN2;
  const SQRT2PI = 2.5066282746310002;
  const now = () => (root.performance && root.performance.now ? root.performance.now() : Date.now());

  // Capture where this script lives, so data/ can be fetched from dev/ or the root page.
  let SCRIPT_BASE = null;
  try {
    if (root.document && root.document.currentScript && root.document.currentScript.src) {
      SCRIPT_BASE = new URL('..', root.document.currentScript.src).href; // visual/
    }
  } catch (e) { /* not in a browser */ }

  /* ================================================================== RNG ===== */

  /**
   * Seeded RNG: xoshiro128** seeded through splitmix32.  Deterministic for a given seed.
   * @param {number} seed  any 32-bit integer
   * @returns {{next:()=>number, uniform:()=>number, normal:()=>number, fillNormal:(a:Float64Array)=>Float64Array, int:(n:number)=>number}}
   *   uniform() is in the open interval (0,1) (never 0, so log(u) is finite);
   *   normal() is a standard Gaussian (Marsaglia polar method).
   */
  function rng(seed) {
    // State lives in an Int32Array: keeping uint32 values in plain closure variables makes V8
    // box them as heap numbers (values ≥ 2^30), which is ~50x slower.
    const st = new Int32Array(4);
    let sm = (seed | 0) || 0x9e3779b9 | 0;
    for (let i = 0; i < 4; i++) {          // splitmix32
      sm = (sm + 0x9e3779b9) | 0;
      let z = sm;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
      st[i] = z ^ (z >>> 16);
    }
    if ((st[0] | st[1] | st[2] | st[3]) === 0) st[0] = 1;
    /** next 32 random bits as a signed int32 (use >>> 0 for unsigned) */
    function next() {
      const s1 = st[1];
      const x = Math.imul(s1, 5);
      const r = Math.imul((x << 7) | (x >>> 25), 9);
      const t = s1 << 9;
      st[2] ^= st[0]; st[3] ^= s1; st[1] ^= st[2]; st[0] ^= st[3]; st[2] ^= t;
      const d = st[3]; st[3] = (d << 11) | (d >>> 21);
      return r;
    }
    // 53-bit uniform in the open interval (0,1)
    function uniform() {
      const hi = next() >>> 5, lo = next() >>> 6;
      return (hi * 67108864 + lo + 0.5) / 9007199254740992;
    }
    let spare = 0, hasSpare = false;
    function normal() {
      if (hasSpare) { hasSpare = false; return spare; }
      let u, v, q;
      do { u = 2 * uniform() - 1; v = 2 * uniform() - 1; q = u * u + v * v; } while (q >= 1 || q === 0);
      const f = Math.sqrt(-2 * Math.log(q) / q);
      spare = v * f; hasSpare = true;
      return u * f;
    }
    function fillNormal(arr) { for (let i = 0; i < arr.length; i++) arr[i] = normal(); return arr; }
    /** uniform integer in [0, n) */
    function int(n) { return Math.floor(uniform() * n); }
    return { next, uniform, normal, fillNormal, int };
  }

  /* ===================================================== special functions ===== */

  /**
   * Standard Gaussian CDF Φ(x) to ~1e-15 relative accuracy in both tails.
   *   |x| < XS: Φ = ½ ± ½ erf(|x|/√2), with the all-positive series
   *             erf(z) = (2/√π) e^{-z²} Σ_n 2^n z^{2n+1} / (2n+1)!!   (no cancellation in the sum);
   *   |x| ≥ XS: tail Q(|x|) = φ(|x|) / (|x| + 1/(|x| + 2/(|x| + 3/(...)))), Laplace's continued
   *             fraction, by modified Lentz to convergence.
   * Φ(-x) is computed directly, never as 1-Φ(x).  Tested against scipy.special.ndtr (rtol 1e-13).
   * @param {number} x
   * @returns {number}
   */
  function Phi(x) {
    const ax = Math.abs(x);
    let tail; // Q(ax) = Φ(-ax)
    if (ax < 2.5) {
      const z = ax / Math.SQRT2, z2 = z * z;
      let term = z, sum = z;
      for (let n = 0; n < 200; n++) { term *= 2 * z2 / (2 * n + 3); sum += term; if (term < 1e-17 * sum) break; }
      const erf = 2 / Math.sqrt(Math.PI) * Math.exp(-z2) * sum;
      return x >= 0 ? 0.5 + 0.5 * erf : 0.5 - 0.5 * erf;
    }
    if (ax > 38.5) tail = 0;
    else {
      // f = ax + 1/(ax + 2/(ax + ...)) by modified Lentz
      const tiny = 1e-300;
      let f = ax, C = ax, D = 0;
      for (let j = 1; j < 5000; j++) {
        D = ax + j * D; if (D === 0) D = tiny; D = 1 / D;
        C = ax + j / C; if (C === 0) C = tiny;
        const delta = C * D; f *= delta;
        if (Math.abs(delta - 1) < 1e-16) break;
      }
      // exp(-ax²/2) with ax split as xh + (ax - xh), xh = ax rounded to 1/16, so the rounding
      // of ax² (condition number ax²) does not cost digits (the trick of Cody's erfc).
      const xh = Math.round(ax * 16) / 16, dl = ax - xh;
      tail = Math.exp(-0.5 * xh * xh) * Math.exp(-0.5 * dl * (ax + xh)) / SQRT2PI / f;
    }
    return x > 0 ? 1 - tail : tail;
  }

  /** logit(q) = log(q/(1-q)).  @param {number} q @returns {number} */
  function logit(q) { return Math.log(q) - Math.log1p(-q); }
  /** expit(x) = 1/(1+e^{-x}), stable at both tails.  @param {number} x @returns {number} */
  function expit(x) { if (x >= 0) return 1 / (1 + Math.exp(-x)); const e = Math.exp(x); return e / (1 + e); }
  /** log Σ exp(a_i) over a[lo..hi).  @param {ArrayLike<number>} a */
  function logsumexp(a, lo = 0, hi = a.length) {
    let mx = -Infinity;
    for (let i = lo; i < hi; i++) if (a[i] > mx) mx = a[i];
    if (mx === -Infinity) return -Infinity;
    let s = 0;
    for (let i = lo; i < hi; i++) s += Math.exp(a[i] - mx);
    return mx + Math.log(s);
  }
  /** logit Φ(x) = log Φ(x) - log Φ(-x), accurate to |x| ≈ 37. */
  function logitPhi(x) { return Math.log(Phi(x)) - Math.log(Phi(-x)); }

  /* ======================================================= Curie-Weiss exact ===== */

  /**
   * Exact leave-one-out Glauber birth/death rates of the lumped CW chain (App. A; src/rates.py
   * flip_rates):
   *   W+_k = (N-k)/2 · [1 + tanh(beta (2k-N+1)/(N-1) + h)]
   *   W-_k =   k/2   · [1 - tanh(beta (2k-N-1)/(N-1) + h)]
   * in jumps per sweep.  Evaluated as (N-k)·expit(2z) and k·expit(-2z), which are the same
   * numbers without the cancellation of 1 - tanh at large z.
   * @param {number} N  number of spins (N ≥ 2)
   * @param {number} beta  inverse temperature (small beta = hot, broad distribution; large beta = cold, concentrated)
   * @param {number} h  bias field (h = beta*G)
   * @returns {{Wp: Float64Array, Wm: Float64Array}}  length N+1, indexed by k
   */
  function cwRates(N, beta, h = 0) {
    if (!(N >= 2)) throw new Error('cwRates: N must be >= 2');
    const Wp = new Float64Array(N + 1), Wm = new Float64Array(N + 1);
    const inv = 1 / (N - 1);
    for (let k = 0; k <= N; k++) {
      Wp[k] = k === N ? 0 : (N - k) * expit(2 * (beta * (2 * k - N + 1) * inv + h));
      Wm[k] = k === 0 ? 0 : k * expit(-2 * (beta * (2 * k - N - 1) * inv + h));
    }
    return { Wp, Wm };
  }

  /**
   * Unnormalised log stationary weights ell_k from the rates (Prop. prop:pi; src/barrier.py
   * log_potential_exact = -ell):  ell_0 = 0, ell_k = ell_{k-1} + log W+_{k-1} - log W-_k.
   * Detailed balance makes this equal, up to a constant, to the closed form of App. A:
   *   log pi_k = log C(N,k) + beta((2k-N)^2 - N)/(2(N-1)) + h(2k-N).
   * @returns {Float64Array} length N+1
   */
  function cwLogWeights(N, beta, h = 0) {
    const { Wp, Wm } = cwRates(N, beta, h);
    const ell = new Float64Array(N + 1);
    for (let k = 1; k <= N; k++) ell[k] = ell[k - 1] + Math.log(Wp[k - 1]) - Math.log(Wm[k]);
    return ell;
  }

  /**
   * Normalised log stationary law log pi_k (App. A), log-space throughout (Σ pi_k = 1).
   * @returns {Float64Array} length N+1
   */
  function cwLogPi(N, beta, h = 0) {
    const ell = cwLogWeights(N, beta, h);
    const z = logsumexp(ell);
    for (let k = 0; k <= N; k++) ell[k] -= z;
    return ell;
  }

  /** Stationary law pi_k (App. A) as probabilities.  @returns {Float64Array} */
  function cwPi(N, beta, h = 0) {
    const lp = cwLogPi(N, beta, h);
    for (let k = 0; k <= N; k++) lp[k] = Math.exp(lp[k]);
    return lp;
  }

  /**
   * Equilibrium log-odds of the favoured mode, logit q_eq =
   *   LSE_{k>N/2} log pi_k - LSE_{k<N/2} log pi_k,  with h = beta*G
   * (Sec. 3 definition of q_eq; src/hopfield/equilibrium.py qeq_route_a; the tie k = N/2 is excluded).
   * Use this instead of logit(qEq(...)) when q_eq is within 1e-16 of 1.
   * @param {number} N @param {number} beta final inverse temperature (paper: 1.6) @param {number} G bias
   */
  function logOddsEq(N, beta, G) {
    const ell = cwLogWeights(N, beta, beta * G);
    const half = N / 2;
    const kLo = Math.ceil(half) - 1;       // last k with k < N/2
    const kHi = Math.floor(half) + 1;      // first k with k > N/2
    return logsumexp(ell, kHi, N + 1) - logsumexp(ell, 0, kLo + 1);
  }

  /**
   * q_eq: equilibrium probability of the favoured mode m > 0 at inverse temperature beta, bias
   * h = beta*G (Sec. 3; src/hopfield/equilibrium.py qeq_route_a).  Paper value: q_eq = 0.97274 at
   * N=2048, beta=1.6, G=6.125e-4.
   * @returns {number}
   */
  function qEq(N, beta, G) { return expit(logOddsEq(N, beta, G)); }

  /**
   * Large-N drift of the magnetisation, Eq. (eq:drift):  dm/dt = -m + tanh(beta m + h).
   * @param {number} m @param {number} beta @param {number} h
   */
  function drift(m, beta, h = 0) { return -m + Math.tanh(beta * m + h); }

  /** Landau free energy F(m) = m²/2 - log cosh(beta m + h)/beta (src/landau.py landau_F). */
  function landauF(m, beta, h = 0) {
    const z = Math.abs(beta * m + h);
    const lc = z < 1 ? Math.log1p(2 * Math.sinh(z / 2) ** 2) : z + Math.log1p(Math.exp(-2 * z)) - LN2;
    return 0.5 * m * m - lc / beta;
  }

  /**
   * Stationary magnetisations: all roots of m = tanh(beta m + h) in (-1,1), ascending
   * (src/landau.py stationary_magnetisations).  One root for beta ≤ 1 (or a large |h|),
   * three for beta > 1 below the spinodal.  Stable ones have beta·sech²(beta m+h) < 1.
   * @returns {{m:number, stable:boolean}[]}
   */
  function stationaryPoints(beta, h = 0) {
    const g = (m) => Math.tanh(beta * m + h) - m;
    const n = 4000, out = [];
    let x0 = -1 + 1e-12, g0 = g(x0);
    for (let i = 1; i <= n; i++) {
      const x1 = -1 + 1e-12 + (2 - 2e-12) * i / n, g1 = g(x1);
      if (g0 === 0) out.push(x0);
      else if (g0 * g1 < 0) {
        let lo = x0, hi = x1, glo = g0;
        for (let it = 0; it < 80; it++) {
          const mid = 0.5 * (lo + hi), gm = g(mid);
          if ((gm < 0) === (glo < 0)) { lo = mid; glo = gm; } else hi = mid;
        }
        out.push(0.5 * (lo + hi));
      }
      x0 = x1; g0 = g1;
    }
    if (h === 0 && beta > 1 && !out.some((r) => Math.abs(r) < 1e-9)) out.push(0); // exact root on a grid node
    out.sort((a, b) => a - b);
    return out.map((m) => {
      const t = Math.tanh(beta * m + h);
      return { m, stable: beta * (1 - t * t) < 1 };
    });
  }

  /* ============================================================ schedules ===== */

  /**
   * The paper's protocol (Sec. 4, Table 3; src/hopfield/protocol.py), N = 2048.
   * Re-descent: 10 levels, sweeps K = ceil(6 tau(beta)) summing to 834.
   */
  const PAPER = Object.freeze({
    N: 2048,
    ladder: Object.freeze([0.6, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.6]),
    K: Object.freeze([15, 30, 59, 107, 355, 145, 65, 31, 16, 11]),
    K_total: 834,
    budgetConstant: 6,
    K_scarce: 2,
    beta_reheat: 0.7,
    K_reheat: 30,
    beta_final: 1.6,
    G0: 6.125e-4,
    c_CW: 0.119808426187369,         // data/c_cw.json (src exact_protocol, dt = 0.02)
    q_eq: 0.9727396877964474,        // src exact_protocol.q_eq(G0)
  });

  /* ======================================= exact continuous-time propagator ===== */

  /** Binomial(N, 1/2) start (cold start, every spin a fair coin), in log space. */
  function binomialStart(N) {
    const p = new Float64Array(N + 1);
    let lc = 0; // log C(N,k)
    for (let k = 0; k <= N; k++) {
      if (k > 0) lc += Math.log((N - k + 1) / k);
      p[k] = Math.exp(lc - N * LN2);
    }
    return p;
  }

  // Factor (I - dt Q^T) once per (beta, h, dt): Thomas algorithm coefficients.
  function factorIE(Wp, Wm, dt) {
    const n = Wp.length;
    const sub = new Float64Array(n), cp = new Float64Array(n), invd = new Float64Array(n);
    // row k: sub (k,k-1) = -dt Wp[k-1], diag 1 + dt(Wp+Wm)[k], super (k,k+1) = -dt Wm[k+1]
    for (let k = 0; k < n; k++) {
      const a = k > 0 ? -dt * Wp[k - 1] : 0;
      const b = 1 + dt * (Wp[k] + Wm[k]);
      const c = k < n - 1 ? -dt * Wm[k + 1] : 0;
      const den = b - (k > 0 ? a * cp[k - 1] : 0);
      invd[k] = 1 / den; cp[k] = c / den; sub[k] = a;
    }
    return { sub, cp, invd, n };
  }
  function solveIE(f, rhs, out) {
    const { sub, cp, invd, n } = f;
    out[0] = rhs[0] * invd[0];
    for (let k = 1; k < n; k++) out[k] = (rhs[k] - sub[k] * out[k - 1]) * invd[k];
    for (let k = n - 2; k >= 0; k--) out[k] -= cp[k] * out[k + 1];
    return out;
  }

  /**
   * Exact propagation of the distribution over k under the continuous-time lumped chain,
   * dp/dt = Q^T p (time in sweeps), by implicit Euler with Richardson extrapolation
   * (second order, L-stable) — the same integrator as src/experiments/exact_protocol.py evolve.
   * No sampling: this is how the paper computes c_CW (Sec. 3).
   */
  class CTPropagator {
    /** @param {number} N @param {Float64Array} [p0] initial law (default binomialStart(N)) */
    constructor(N, p0) {
      this.N = N;
      this.p = p0 ? Float64Array.from(p0) : binomialStart(N);
      this._a = new Float64Array(N + 1); this._b = new Float64Array(N + 1); this._c = new Float64Array(N + 1);
      this.t = 0;
    }
    /**
     * Generator: evolve for t sweeps at fixed (beta, h) with step dt (n = round(t/dt) steps).
     * Yields after every implicit step (≈25(N+1) flops).
     */
    *evolveGen(beta, h, t, dt = 0.02) {
      if (!(t > 0)) return;
      const n = Math.max(1, Math.round(t / dt)); dt = t / n;
      const { Wp, Wm } = cwRates(this.N, beta, h);
      const F1 = factorIE(Wp, Wm, dt), F2 = factorIE(Wp, Wm, dt / 2);
      const p = this.p, a = this._a, b = this._b, c = this._c;
      for (let s = 0; s < n; s++) {
        solveIE(F1, p, a);
        solveIE(F2, p, c); solveIE(F2, c, b);
        for (let k = 0; k <= this.N; k++) p[k] = 2 * b[k] - a[k];
        this.t += dt;
        yield s + 1;
      }
    }
    /** Synchronous evolve; blocks — prefer evolveGen + runChunked for N ≳ 512 or t ≳ 100. */
    evolve(beta, h, t, dt = 0.02) { drain(this.evolveGen(beta, h, t, dt)); return this; }
    /** Probability of the favoured mode, P(m > 0) = Σ_{k>N/2} p_k. */
    qPlus() { let s = 0; for (let k = Math.floor(this.N / 2) + 1; k <= this.N; k++) s += this.p[k]; return s; }
    /** Total mass (should stay 1 to ~1e-12). */
    mass() { let s = 0; for (let k = 0; k <= this.N; k++) s += this.p[k]; return s; }
  }

  /**
   * The paper's full CW protocol on the exact chain (Sec. 4, App. A; src exact_protocol.protocol):
   *  1 cold start Binomial(N,½); 2 scarce descent, K_scarce sweeps at each ladder level;
   *  3 keep the mis-committed mass m < 0 and renormalise (f_mis = its mass);
   *  4 reheat K_reheat sweeps at beta_reheat; 5 re-descent K sweeps (× scale) per level.
   * All levels use h = beta·G.  q_chain = P(m > 0) at the end;  c = logit q_chain / logit q_eq
   * with q_eq = qEq(N, beta_final, G).  At N = 2048, G = 6.125e-4: c = 0.1198 (= c_CW, Sec. 3).
   * Generator yields {phase, frac}; returns {f_mis, q_chain, q_eq, c, p}.
   * @param {number} N
   * @param {number} G
   * @param {{dt?:number, scale?:number, ladder?:number[], K?:number[], K_scarce?:number, beta_reheat?:number, K_reheat?:number}} [o]
   */
  function* protocolGen(N, G, o = {}) {
    const ladder = o.ladder || PAPER.ladder, K = o.K || PAPER.K;
    const dt = o.dt || 0.02, scale = o.scale == null ? 1 : o.scale;
    const Ks = o.K_scarce == null ? PAPER.K_scarce : o.K_scarce;
    const brh = o.beta_reheat == null ? PAPER.beta_reheat : o.beta_reheat;
    const Krh = o.K_reheat == null ? PAPER.K_reheat : o.K_reheat;
    const total = Ks * ladder.length + Krh + scale * K.reduce((a, b) => a + b, 0);
    const P = new CTPropagator(N);
    let done = 0;
    const run = function* (beta, t, phase) {
      const t0 = P.t;
      for (const _ of P.evolveGen(beta, beta * G, t, dt)) yield { phase, frac: (done + (P.t - t0)) / total };
      done += t;
    };
    for (const b of ladder) yield* run(b, Ks, 'scarce descent');
    let f = 0;
    for (let k = 0; k <= N; k++) { if (2 * k < N) f += P.p[k]; else P.p[k] = 0; }
    for (let k = 0; k <= N; k++) P.p[k] /= f;
    yield* run(brh, Krh, 'reheat');
    for (let i = 0; i < ladder.length; i++) yield* run(ladder[i], K[i] * scale, 're-descent');
    let mass = 0; for (let k = 0; k <= N; k++) mass += P.p[k];
    const q_chain = P.qPlus() / mass;
    const lq = logOddsEq(N, o.beta_final || PAPER.beta_final, G);
    return { f_mis: f, q_chain, q_eq: expit(lq), c: logit(q_chain) / lq, p: P.p };
  }

  /* ======================================== many independent CW chains (lumped) ===== */

  /**
   * Many independent Curie-Weiss chains, each simulated exactly on the lumped birth-death chain
   * (state = k, the number of up spins) — statistically identical to running the spins.
   * Continuous time, Gillespie: an exact draw of the paper's continuous-time chain, the one
   * c_CW is computed on.  advance() can be called with any (fractional) number of sweeps; splitting a level across
   * frames is exact (memoryless clock).  Cost ≈ Σ_chains (W+_k + W-_k) events per sweep.
   */
  class CWChains {
    /**
     * @param {number} nChains
     * @param {number} N
     * @param {{seed?:number, init?:'binomial'|number|Int32Array}} [o]
     *   init: 'binomial' (each spin a fair coin; default), a fixed k, or an array of k.
     */
    constructor(nChains, N, o = {}) {
      this.n = nChains; this.N = N;
      this.rng = rng(o.seed == null ? 1 : o.seed);
      this.k = new Int32Array(nChains);
      this.t = 0;
      this._cacheKey = null;
      if (o.init == null || o.init === 'binomial') {
        for (let c = 0; c < nChains; c++) this.k[c] = sampleBinomialHalf(N, this.rng);
      } else if (typeof o.init === 'number') this.k.fill(o.init);
      else this.k.set(o.init);
    }
    _tables(beta, h) {
      const key = beta + ',' + h;
      if (key === this._cacheKey) return this._tab;
      const { Wp, Wm } = cwRates(this.N, beta, h);
      const N = this.N, R = new Float64Array(N + 1), pUp = new Float64Array(N + 1);
      for (let k = 0; k <= N; k++) {
        R[k] = Wp[k] + Wm[k];
        pUp[k] = Wp[k] / R[k];
      }
      this._cacheKey = key;
      this._tab = { R, pUp };
      return this._tab;
    }
    /**
     * Advance every chain by `sweeps` sweeps at fixed (beta, h).
     * @returns {number} number of jump events simulated (cost measure)
     */
    advance(beta, h, sweeps) {
      if (!(sweeps > 0)) return 0;
      const { R, pUp } = this._tables(beta, h);
      const k = this.k, u = this.rng.uniform, n = this.n;
      let events = 0;
      for (let c = 0; c < n; c++) {
        let kc = k[c], t = -Math.log(u()) / R[kc];
        while (t < sweeps) {
          kc += u() < pUp[kc] ? 1 : -1;
          events++;
          t += -Math.log(u()) / R[kc];
        }
        k[c] = kc;
      }
      this.t += sweeps;
      return events;
    }
    /** magnetisation of chain c */
    m(c) { return 2 * this.k[c] / this.N - 1; }
    /** Float64Array of all magnetisations */
    magnetisations(out = new Float64Array(this.n)) { for (let c = 0; c < this.n; c++) out[c] = 2 * this.k[c] / this.N - 1; return out; }
    /** fraction of chains in the favoured mode (m > 0), ties excluded from neither side */
    fracPlus() { let s = 0; for (let c = 0; c < this.n; c++) if (2 * this.k[c] > this.N) s++; return s / this.n; }
    /** histogram of k, Float64Array length N+1 (counts) */
    histogram() { const H = new Float64Array(this.N + 1); for (let c = 0; c < this.n; c++) H[this.k[c]]++; return H; }
  }

  // exact Binomial(N, 1/2) draw (sum of N fair bits, 32 at a time)
  function sampleBinomialHalf(N, r) {
    let s = 0, left = N;
    while (left >= 32) { s += popcount(r.next()); left -= 32; }
    if (left > 0) s += popcount(r.next() & ((1 << left) - 1) >>> 0);
    return s;
  }
  function popcount(x) {
    x = x - ((x >>> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
    return (Math.imul((x + (x >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24);
  }

  /* ================================================ explicit spin-level CW Gibbs ===== */

  /**
   * One Curie-Weiss chain with explicit spins (Int8Array), single-site heat-bath (Gibbs):
   *   P(x_i = +1 | rest) = ½[1 + tanh(beta (M - x_i)/(N-1) + h)],  M = Σ_j x_j
   * (the exact leave-one-out field of App. B).  One sweep = N updates, in a fresh random
   * permutation order (the paper sampler's order) or with N uniform random sites ('random').
   * Both leave pi_k invariant (tested against cwPi).
   */
  class CWSpins {
    /** @param {number} N @param {{seed?:number, init?:'random'|'up'|'down'}} [o] */
    constructor(N, o = {}) {
      this.N = N; this.rng = rng(o.seed == null ? 7 : o.seed);
      this.spins = new Int8Array(N);
      this.order = new Int32Array(N);
      for (let i = 0; i < N; i++) this.order[i] = i;
      this.reset(o.init || 'random');
      this._key = null;
    }
    /** @param {'random'|'up'|'down'} init */
    reset(init = 'random') {
      let M = 0;
      for (let i = 0; i < this.N; i++) {
        const s = init === 'up' ? 1 : init === 'down' ? -1 : (this.rng.next() & 1 ? 1 : -1);
        this.spins[i] = s; M += s;
      }
      this.M = M;
    }
    _table(beta, h) {
      const key = beta + ',' + h;
      if (key !== this._key) {
        const N = this.N, T = new Float64Array(2 * N - 1);   // index (M - x_i) + (N-1)
        for (let j = 0; j < 2 * N - 1; j++) T[j] = expit(2 * (beta * (j - (N - 1)) / (N - 1) + h));
        this._T = T; this._key = key;
      }
      return this._T;
    }
    /**
     * Run `n` sweeps.  @param {'perm'|'random'} [order='perm']
     * @returns {number} number of spins flipped
     */
    sweep(beta, h = 0, n = 1, order = 'perm') {
      const T = this._table(beta, h), s = this.spins, N = this.N, u = this.rng.uniform, off = N - 1, ord = this.order;
      let M = this.M, flips = 0;
      for (let sw = 0; sw < n; sw++) {
        if (order === 'perm') for (let i = N - 1; i > 0; i--) { const j = Math.floor(u() * (i + 1)); const t = ord[i]; ord[i] = ord[j]; ord[j] = t; }
        for (let a = 0; a < N; a++) {
          const i = order === 'perm' ? ord[a] : Math.floor(u() * N);
          const si = s[i];
          const ns = u() < T[M - si + off] ? 1 : -1;
          if (ns !== si) { s[i] = ns; M += ns - si; flips++; }
        }
      }
      this.M = M;
      return flips;
    }
    /** magnetisation m = M/N */
    get m() { return this.M / this.N; }
    /** number of up spins k */
    get k() { return (this.M + this.N) / 2; }
  }

  /* ================================================ the selection law and c ===== */

  /** KN argument x = G √N (π/γ)^{1/4} (Eq. eq:law). */
  function knX(G, N, gamma) { return G * Math.sqrt(N) * Math.pow(Math.PI / gamma, 0.25); }
  /**
   * Kondepudi-Nelson selection law, Eq. (eq:law):  q_chain = Φ(G √N (π/γ)^{1/4}),
   * γ = rise in beta per sweep of a linear schedule beta = 1 + γ t.
   */
  function knQ(G, N, gamma) { return Phi(knX(G, N, gamma)); }
  /**
   * c predicted by the law for a linear sweep ending at beta_final:
   *   c_KN = logit Φ(x) / logit q_eq(N, beta_final, G)   (src/experiments/cw_linear_sweep.py).
   */
  function knC(G, N, gamma, betaFinal = 1.6) { return logitPhi(knX(G, N, gamma)) / logOddsEq(N, betaFinal, G); }

  /* ================================================ Gaussian order statistics ===== */

  /**
   * Amplification A_p = E[max_p |Z|] / E|Z| for p iid standard Gaussians (App. C, Table 1):
   *   E[max_p|Z|] = ∫_0^∞ [1 - (2Φ(x) - 1)^p] dx,  E|Z| = sqrt(2/π).
   * Composite Simpson on [0, 14] with 14,000 panels (integrand < 1e-43 beyond).
   * A_1 = 1, A_2 = √2 = 1.41421, A_3 = 1.66238, A_4 = 1.83576.
   */
  function ampP(p) {
    const a = 0, b = 14, n = 14000, hh = (b - a) / n;
    const f = (x) => { const t = 1 - 2 * Phi(-x); return 1 - Math.pow(t, p); };
    let s = f(a) + f(b);
    for (let i = 1; i < n; i++) s += (i % 2 ? 4 : 2) * f(a + i * hh);
    return (s * hh / 3) / Math.sqrt(2 / Math.PI);
  }

  /* ================================================================ Hopfield ===== */

  /**
   * Small Hopfield network, batched over independent chains (Sec. 4, App. B; src/hopfield/sampler.py):
   *   E(s) = -1/(2N) Σ_{i≠j} Σ_μ ξ^μ_i ξ^μ_j s_i s_j - G Σ_i ξ^1_i s_i
   *   local field H_i = (Σ_μ ξ^μ_i c_μ - p s_i)/N + G ξ^1_i,  c_μ = ξ^μ·s (leave-one-out)
   *   P(s_i = +1) = ½[1 + tanh(beta H_i)] = expit(2 beta H_i).
   * One sweep = N updates in a fresh random permutation per chain.  Overlaps m^μ = c_μ/N.
   * N ≤ 512, p ≤ 6 intended (works beyond, just slower).  At p = 1 the equilibrium of m^1
   * is CW at beta' = beta(N-1)/N, h = beta G (tested).
   */
  class Hopfield {
    /**
     * @param {number} N @param {number} p
     * @param {{seed?:number, nChains?:number, patterns?:Int8Array}} [o] patterns: p×N row-major ±1
     */
    constructor(N, p, o = {}) {
      this.N = N; this.p = p; this.nChains = o.nChains || 1;
      this.rng = rng(o.seed == null ? 13 : o.seed);
      this.xi = o.patterns ? Int8Array.from(o.patterns) : new Int8Array(p * N);
      if (!o.patterns) for (let j = 0; j < p * N; j++) this.xi[j] = this.rng.next() & 1 ? 1 : -1;
      this.s = new Int8Array(this.nChains * N);
      this.c = new Int32Array(this.nChains * p);
      this.order = new Int32Array(N);
      for (let i = 0; i < N; i++) this.order[i] = i;
      this._key = null;
      this.randomize();
    }
    /** every spin of every chain a fair coin */
    randomize() { for (let j = 0; j < this.s.length; j++) this.s[j] = this.rng.next() & 1 ? 1 : -1; this._recompute(); }
    /** set chain `ch` (or all chains if ch < 0) to sign·ξ^mu */
    setToPattern(mu, sign = 1, ch = -1) {
      const N = this.N;
      for (let c = 0; c < this.nChains; c++) if (ch < 0 || c === ch) for (let i = 0; i < N; i++) this.s[c * N + i] = sign * this.xi[mu * N + i];
      this._recompute();
    }
    _recompute() {
      const N = this.N, p = this.p;
      for (let c = 0; c < this.nChains; c++) for (let mu = 0; mu < p; mu++) {
        let acc = 0; for (let i = 0; i < N; i++) acc += this.xi[mu * N + i] * this.s[c * N + i];
        this.c[c * p + mu] = acc;
      }
    }
    _table(beta, G) {
      const key = beta + ',' + G;
      if (key !== this._key) {
        const N = this.N, p = this.p, off = p * N + p, W = 2 * off + 1;
        const T = new Float64Array(2 * W); // [xi1 = -1 | xi1 = +1] × numerator
        for (let j = 0; j < W; j++) {
          const base = (j - off) / N;
          T[j] = expit(2 * beta * (base - G));
          T[W + j] = expit(2 * beta * (base + G));
        }
        this._T = T; this._W = W; this._off = off; this._key = key;
      }
      return this._T;
    }
    /** Run n sweeps of every chain at (beta, G).  @returns {number} flips */
    sweep(beta, G = 0, n = 1) {
      const T = this._table(beta, G), W = this._W, off = this._off;
      const N = this.N, p = this.p, xi = this.xi, s = this.s, cc = this.c, ord = this.order, u = this.rng.uniform;
      let flips = 0;
      for (let ch = 0; ch < this.nChains; ch++) {
        const sb = ch * N, cb = ch * p;
        for (let sw = 0; sw < n; sw++) {
          for (let i = N - 1; i > 0; i--) { const j = Math.floor(u() * (i + 1)); const t = ord[i]; ord[i] = ord[j]; ord[j] = t; }
          for (let a = 0; a < N; a++) {
            const i = ord[a], si = s[sb + i];
            let num = -p * si;
            for (let mu = 0; mu < p; mu++) num += xi[mu * N + i] * cc[cb + mu];
            const pr = T[(xi[i] > 0 ? W : 0) + num + off];
            const ns = u() < pr ? 1 : -1;
            if (ns !== si) {
              s[sb + i] = ns; flips++;
              const d = ns - si;
              for (let mu = 0; mu < p; mu++) cc[cb + mu] += d * xi[mu * N + i];
            }
          }
        }
      }
      return flips;
    }
    /** overlaps m^μ = ξ^μ·s/N of chain ch.  @returns {Float64Array} length p */
    overlaps(ch = 0, out = new Float64Array(this.p)) { for (let mu = 0; mu < this.p; mu++) out[mu] = this.c[ch * this.p + mu] / this.N; return out; }
    /** index μ of the largest |m^μ| of chain ch (which pattern the chain committed to) */
    basin(ch = 0) { let b = 0, best = -1; for (let mu = 0; mu < this.p; mu++) { const v = Math.abs(this.c[ch * this.p + mu]); if (v > best) { best = v; b = mu; } } return b; }
    /** spins of chain ch (a view, Int8Array length N) */
    spinsOf(ch = 0) { return this.s.subarray(ch * this.N, (ch + 1) * this.N); }
  }

  /* ============================================================ scheduling ===== */

  function drain(gen) { let r; do { r = gen.next(); } while (!r.done); return r.value; }

  /**
   * Run a generator in time slices of at most `budgetMs` (default 12 ms), yielding to the
   * browser between slices (requestAnimationFrame when available, else setTimeout 0).
   * @param {Generator} gen
   * @param {{budgetMs?:number, onProgress?:(v:any)=>void, signal?:{cancelled:boolean}}} [o]
   * @returns {Promise<any>} resolves with the generator's return value; rejects with
   *   Error('cancelled') if o.signal.cancelled becomes true.
   */
  function runChunked(gen, o = {}) {
    const budget = o.budgetMs || 12;
    const tick = (f) => (root.requestAnimationFrame ? root.requestAnimationFrame(f) : setTimeout(f, 0));
    return new Promise((resolve, reject) => {
      function slice() {
        if (o.signal && o.signal.cancelled) { reject(new Error('cancelled')); return; }
        const t0 = now();
        let r, last;
        try {
          do { r = gen.next(); if (!r.done) last = r.value; } while (!r.done && now() - t0 < budget);
        } catch (e) { reject(e); return; }
        if (o.onProgress && last !== undefined) o.onProgress(last);
        if (r.done) resolve(r.value); else tick(slice);
      }
      slice();
    });
  }

  /**
   * URL of a data file, resolved against visual/ (works from index.html and from dev/*.html).
   * @param {string} name e.g. 'tracking_time.json'
   */
  function dataUrl(name) { return SCRIPT_BASE ? new URL('data/' + name, SCRIPT_BASE).href : 'data/' + name; }
  /** fetch + parse a JSON file from visual/data/.  @returns {Promise<any>} */
  function loadJSON(name) { return fetch(dataUrl(name)).then((r) => { if (!r.ok) throw new Error(name + ': HTTP ' + r.status); return r.json(); }); }

  VE.sim = {
    // rng + special functions
    rng, Phi, logit, expit, logsumexp, logitPhi,
    // CW exact
    cwRates, cwLogWeights, cwLogPi, cwPi, logOddsEq, qEq, drift, landauF, stationaryPoints,
    // the paper's protocol + exact propagation
    PAPER, binomialStart, CTPropagator, protocolGen,
    // stochastic
    CWChains, CWSpins, Hopfield,
    // law + order statistics
    knX, knQ, knC, ampP,
    // infrastructure
    runChunked, drain, dataUrl, loadJSON,
  };
})(typeof window !== 'undefined' ? window : globalThis);
