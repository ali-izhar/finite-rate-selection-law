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
 *   CTPropagator (exact law)     ~25(N+1) flops per implicit step; the test ramp
 *                                (225 sweeps, dt = 0.02) takes ~0.6 s at N=2048, ~0.08 s at N=256.
 *                                Always drive it through runChunked (longest step 3.7 ms at N=1024).
 * In node, do not load this file into a vm.Context for timing: global lookups through a vm
 * context are ~40x slower; evaluate it with a plain `window` object instead (see the test).
 * Long jobs are generators (rampGen, CTPropagator.evolveGen) that yield often;
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
   * (Sec. 2 definition of q_eq; the tie k = N/2 is excluded; tested against src.barrier.log_pi_closed_form).
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
   * h = beta*G (Sec. 2).  Paper value: q_eq = 0.97274 at
   * N=2048, beta=1.6, G=6.125e-4.
   * @returns {number}
   */
  function qEq(N, beta, G) { return expit(logOddsEq(N, beta, G)); }

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

  /* ============================================================== the ramp ===== */

  /**
   * The paper's test ramp (Sec. 4; src/simple/design.py): beta rises linearly from 0.7 to 1.6 at
   * gamma = 0.004 per sweep (225 sweeps), with N G = 1.2544 at every N.  The exact CW fraction on
   * it is c_CW = 0.0659 at N = 2048 (Table 1; results/simple/predictions.json; tested).
   */
  const RAMP = Object.freeze({
    N: 2048,
    beta0: 0.7,
    beta_final: 1.6,
    gamma: 0.004,
    NG: 1.2544,
    G0: 6.125e-4,                    // = NG / 2048
    c_CW: 0.06587418845290097,       // results/simple/predictions.json, N = 2048
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
   * (second order, L-stable) — the same integrator as src/cw_ramp.py evolve.
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

  // Python's round(): halves go to the even integer (src.cw_ramp.evolve uses it for the step count)
  function pyRound(x) { const f = Math.floor(x), d = x - f; return d > 0.5 || (d === 0.5 && f % 2 !== 0) ? f + 1 : f; }

  /**
   * A linear ramp on the exact chain (src/cw_ramp.py linear_sweep): beta rises from beta0 to
   * beta_final at rate gamma per sweep, in steps of db; each step is evolved at its midpoint beta
   * for db / gamma sweeps, with the field h = beta G.  start: 'binomial' (independent unbiased
   * spins, as in the Hopfield test; default) or 'eq' (equilibrium at beta0, as in Table 2).
   * q_chain = P(m > 0 | m != 0); c = logit q_chain / logit q_eq at beta_final.
   * Generator yields {frac}; returns {q_chain, q_eq, c, p}.
   * @param {number} N @param {number} G
   * @param {{gamma?:number, beta0?:number, beta_final?:number, db?:number, dt?:number, start?:'binomial'|'eq'}} [o]
   */
  function* rampGen(N, G, o = {}) {
    const b0 = o.beta0 == null ? RAMP.beta0 : o.beta0, b1 = o.beta_final == null ? RAMP.beta_final : o.beta_final;
    const gamma = o.gamma || RAMP.gamma, db = o.db || 0.005, dt0 = o.dt || 0.02;
    const P = new CTPropagator(N, o.start === 'eq' ? cwPi(N, b0, b0 * G) : undefined);
    const nSeg = Math.round((b1 - b0) / db), tseg = db / gamma, dt = Math.min(dt0, tseg / 4);
    const nStep = Math.max(1, pyRound(tseg / dt));
    for (let i = 0; i < nSeg; i++) {
      const bb = b0 + i * db + db / 2;
      for (const s of P.evolveGen(bb, bb * G, tseg, tseg / nStep)) yield { frac: (i + s / nStep) / nSeg };
    }
    let up = 0, dn = 0;
    for (let k = 0; k <= N; k++) { if (2 * k > N) up += P.p[k]; else if (2 * k < N) dn += P.p[k]; }
    const q_chain = up / (up + dn), lq = logOddsEq(N, b1, G);
    return { q_chain, q_eq: expit(lq), c: logit(q_chain) / lq, p: P.p };
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
   *   c_KN = logit Φ(x) / logit q_eq(N, beta_final, G)   (src/cw_ramp.py).
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
   * @param {string} name e.g. 'table1.json'
   */
  function dataUrl(name) { return SCRIPT_BASE ? new URL('data/' + name, SCRIPT_BASE).href : 'data/' + name; }
  /** fetch + parse a JSON file from visual/data/.  @returns {Promise<any>} */
  function loadJSON(name) { return fetch(dataUrl(name)).then((r) => { if (!r.ok) throw new Error(name + ': HTTP ' + r.status); return r.json(); }); }

  VE.sim = {
    // rng + special functions
    rng, Phi, logit, expit, logsumexp, logitPhi,
    // CW exact
    cwRates, cwLogWeights, cwLogPi, cwPi, logOddsEq, qEq, stationaryPoints,
    // the test ramp + exact propagation
    RAMP, binomialStart, CTPropagator, rampGen,
    // stochastic
    CWChains,
    // law + order statistics
    knX, knQ, knC, ampP,
    // infrastructure
    runChunked, drain, dataUrl, loadJSON,
  };
})(typeof window !== 'undefined' ? window : globalThis);
