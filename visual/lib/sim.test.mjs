// Tests for lib/sim.js against the paper's Python (../src).
//   node lib/sim.test.mjs            (from visual/)
// Reference values: data/ref/reference.json, produced by
//   .venv/bin/python visual/data/ref/make_reference.py    (from the paper repo root)
// and data/c_cw.json, produced by visual/data/make_exports.py.
// Statistical tests use fixed seeds; their thresholds are 4-sigma (z) or a
// Wilson-Hilferty chi-square z < 4, stated per test.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
// sim.js is a classic browser script: evaluate it with a fake `window` object.  A vm.Context
// sandbox would also work, but global lookups through a vm context are ~40x slower in V8, which
// distorts the benchmarks; a vm context is used once below only to prove the script is
// self-contained (no globals beyond `window`).
const SRC = fs.readFileSync(path.join(HERE, 'sim.js'), 'utf8');
const ctx = { performance, setTimeout };
new Function('window', SRC)(ctx);
const S = ctx.VE.sim;
{
  const sandbox = { performance };
  sandbox.window = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(SRC, sandbox, { filename: 'sim.js' });
  if (typeof sandbox.VE.sim.qEq !== 'function') throw new Error('sim.js did not define window.VE.sim in a vm sandbox');
}
const REF = JSON.parse(fs.readFileSync(path.join(ROOT, 'data/ref/reference.json'), 'utf8'));
const CCW = JSON.parse(fs.readFileSync(path.join(ROOT, 'data/c_cw.json'), 'utf8'));

let pass = 0, fail = 0;
const results = [];
function check(name, ok, detail) {
  (ok ? pass++ : fail++);
  const line = `${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  -- ' + detail : ''}`;
  results.push(line); console.log(line);
}
function maxRel(a, b, atol = 0) {
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]) / (atol + Math.abs(b[i]) || 1);
    if (!(d <= worst)) worst = Number.isNaN(d) ? Infinity : Math.max(worst, d);
  }
  return worst;
}
function allclose(a, b, rtol, atol = 0) {
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    const err = Math.abs(a[i] - b[i]), tol = atol + rtol * Math.abs(b[i]);
    if (!(err <= tol)) return { ok: false, i, a: a[i], b: b[i], err };
    worst = Math.max(worst, tol > 0 ? err / tol : 0);
  }
  return { ok: true, worst };
}
// chi-square of observed counts vs expected probabilities; bins with expected < 5 are pooled.
function chiSquare(obs, prob, n) {
  let chi = 0, df = -1, accO = 0, accE = 0;
  for (let i = 0; i < obs.length; i++) {
    accO += obs[i]; accE += prob[i] * n;
    if (accE >= 5) { chi += (accO - accE) ** 2 / accE; df++; accO = 0; accE = 0; }
  }
  if (accE > 0) { chi += (accO - accE) ** 2 / Math.max(accE, 1e-300); df++; }
  const k = df, z = (Math.cbrt(chi / k) - (1 - 2 / (9 * k))) / Math.sqrt(2 / (9 * k)); // Wilson-Hilferty
  return { chi, df, z };
}
const fmt = (x, d = 6) => Number(x).toPrecision(d);

// ---------------------------------------------------------------- 1. rates, pi
{
  let worstR = 0, worstL = 0, ok = true;
  for (const r of REF.pi) {
    const { Wp, Wm } = S.cwRates(r.N, r.beta, r.h);
    const a = allclose(Wp, r.W_plus, 1e-12, 1e-300), b = allclose(Wm, r.W_minus, 1e-12, 1e-300);
    // log pi: cumulative-sum route vs src closed form (gammaln); abs error grows ~ N eps |log pi|
    const lp = S.cwLogPi(r.N, r.beta, r.h);
    const c = allclose(lp, r.log_pi, 1e-11, 1e-10);
    ok = ok && a.ok && b.ok && c.ok;
    worstR = Math.max(worstR, a.worst || 0, b.worst || 0); worstL = Math.max(worstL, c.worst || 0);
    if (!c.ok) console.log('  log_pi mismatch', r.N, r.beta, r.h, c);
  }
  check('W+_k, W-_k vs src.rates.flip_rates (N=64,256; 4 (beta,h) each)', ok, `assert_allclose rtol=1e-12; worst err/tol ${fmt(worstR, 3)}`);
  check('log pi_k vs src.barrier.log_pi_closed_form (normalised)', ok, `rtol=1e-11, atol=1e-10; worst err/tol ${fmt(worstL, 3)}`);
}
// ---------------------------------------------------------------- 2. q_eq
{
  let ok = true, worst = 0;
  for (const r of REF.q_eq) {
    const q = S.qEq(r.N, r.beta, r.G);
    const e = Math.abs(q - r.q_eq) / Math.min(r.q_eq, 1 - r.q_eq); // relative on the smaller tail
    worst = Math.max(worst, e); if (!(e < 1e-11)) { ok = false; console.log('  q_eq', r, q); }
  }
  check('q_eq vs src.hopfield.equilibrium.qeq_route_a (N=64,256,2048)', ok, `relative error on min(q,1-q) < 1e-11; worst ${fmt(worst, 3)}`);
  const q = S.qEq(2048, 1.6, S.PAPER.G0);
  check('q_eq(N=2048, beta=1.6, G0) = 0.97274 (paper grid point k=0)', Math.abs(q - 0.9727397) < 5e-8, `q_eq = ${fmt(q, 9)}`);
}
// ---------------------------------------------------------------- 3. exact propagation
{
  const r = REF.evolve;
  const P = new S.CTPropagator(r.N);
  P.evolve(r.beta, r.h, r.t, r.dt);
  const c = allclose(P.p, r.p, 1e-9, 1e-14);
  check('CTPropagator one level vs src exact_protocol.evolve (N=128, 25 sweeps)', c.ok, `rtol=1e-9, atol=1e-14; mass-1 = ${(P.mass() - 1).toExponential(2)}`);
  for (const pr of REF.protocol) {
    const res = S.drain(S.protocolGen(pr.N, pr.G, { dt: pr.dt }));
    const ok = Math.abs(res.q_chain - pr.q_chain) < 1e-10 && Math.abs(res.f_mis - pr.f_mis) < 1e-10 && Math.abs(res.c - pr.c) < 1e-8;
    check(`paper protocol on the exact chain, N=${pr.N}: q_chain, f_mis, c vs src`, ok,
      `q_chain ${fmt(res.q_chain, 10)} (src ${fmt(pr.q_chain, 10)}), c ${fmt(res.c, 8)} (src ${fmt(pr.c, 8)})`);
  }
  const t0 = performance.now();
  const res = S.drain(S.protocolGen(2048, S.PAPER.G0, { dt: 0.02 }));
  const ms = performance.now() - t0;
  const ref = CCW.runs.find((x) => x.dt === 0.02);
  check('c_CW at N=2048 on the paper schedule = 0.120 (Sec. 3), vs src', Math.abs(res.c - ref.c_cw) < 1e-8 && Math.abs(res.q_chain - ref.q_chain) < 1e-10,
    `JS c = ${fmt(res.c, 9)}, src c = ${fmt(ref.c_cw, 9)}, q_chain ${fmt(res.q_chain, 9)}; JS wall ${ms.toFixed(0)} ms`);
}
// ---------------------------------------------------------------- 4. special functions, A_p, law
{
  const a = allclose(REF.ndtr.map((x) => S.Phi(x[0])), REF.ndtr.map((x) => x[1]), 1e-13, 1e-310);
  check("Phi vs mpmath ncdf at 40 digits (x in [-38, 9])", a.ok, a.ok ? `rtol=1e-13; worst err/tol ${fmt(a.worst, 3)}` : JSON.stringify(a));
  const A = [1, 2, 3, 4, 5, 6].map((p) => S.ampP(p));
  const c = allclose(A, [1, 2, 3, 4, 5, 6].map((p) => REF.A_p[p]), 1e-10);
  const rounded = A.slice(0, 4).map((x) => x.toFixed(3)).join(', ');
  check('A_p = E[max_p|Z|]/E|Z| vs scipy quad, p=1..6', c.ok && rounded === '1.000, 1.414, 1.662, 1.836', `rtol=1e-10; A_1..4 = ${rounded}`);
  const lin = JSON.parse(fs.readFileSync(path.join(ROOT, '../results/experiments/cw_linear_sweep.json'), 'utf8'));
  let ok = true, worstC = 0;
  for (const r of lin) {
    const q = S.knQ(r.G, r.N, r.gamma), c2 = S.knC(r.G, r.N, r.gamma, 1.6);
    worstC = Math.max(worstC, Math.abs(c2 - r.c_KN) / r.c_KN);
    if (!(Math.abs(q - r.q_KN) < 1e-13 && Math.abs(c2 - r.c_KN) / r.c_KN < 1e-6)) ok = false;
  }
  check('knQ, knC vs results/experiments/cw_linear_sweep.json (9 rows)', ok, `q to 1e-13; c rel ${fmt(worstC, 2)} (src q_eq keeps the tie mass, < 1e-6)`);
  const sp = S.stationaryPoints(1.6, 0).map((x) => x.m);
  check('stationaryPoints(1.6, 0) = {-m*, 0, m*} with m* = tanh(1.6 m*)', sp.length === 3 && Math.abs(Math.tanh(1.6 * sp[2]) - sp[2]) < 1e-12 && Math.abs(sp[1]) < 1e-12, sp.map((x) => fmt(x, 8)).join(', '));
}
// ---------------------------------------------------------------- 5. spin-level Gibbs vs lumped equilibrium
{
  for (const order of ['perm', 'random']) {
    const N = 64, beta = 1.2, h = 0.0, nChains = 3000, burn = 150;
    const pi = S.cwPi(N, beta, h);
    const obs = new Float64Array(N / 2 + 1), prob = new Float64Array(N / 2 + 1); // folded |2k - N|/2
    for (let k = 0; k <= N; k++) prob[Math.abs(2 * k - N) / 2] += pi[k];
    const sp = new S.CWSpins(N, { seed: order === 'perm' ? 31 : 32 });
    for (let c = 0; c < nChains; c++) {
      sp.reset('random'); sp.sweep(beta, h, burn, order);
      obs[Math.abs(2 * sp.k - N) / 2]++;
    }
    const t = chiSquare(obs, prob, nChains);
    check(`CWSpins (${order} order) equilibrium |m| histogram vs pi_k, N=64, beta=1.2`, t.z < 4, `chi2 = ${t.chi.toFixed(1)} on ${t.df} df, WH z = ${t.z.toFixed(2)} (< 4); 3000 chains × 150 sweeps`);
  }
}
// ---------------------------------------------------------------- 6. stochastic lumped chains vs the exact law
{
  const N = 128, G = 0.004;
  const levels = [0.8, 1.0, 1.1, 1.3].map((b, i) => ({ beta: b, h: b * G, sweeps: [3, 20, 7.5, 5][i] }));
  const P = new S.CTPropagator(N);
  for (const L of levels) P.evolve(L.beta, L.h, L.sweeps);
  const pExact = P.p;
  const ch = new S.CWChains(40000, N, { seed: 4242 });
  // split each level in two pieces to also test the memoryless split
  for (const L of levels) { ch.advance(L.beta, L.h, L.sweeps * 0.37); ch.advance(L.beta, L.h, L.sweeps * 0.63); }
  const t = chiSquare(ch.histogram(), pExact, ch.n);
  let qE = 0; for (let k = N / 2 + 1; k <= N; k++) qE += pExact[k];
  const z = (ch.fracPlus() - qE) / Math.sqrt(qE * (1 - qE) / ch.n);
  check('CWChains k-histogram after a 4-level schedule vs CTPropagator', t.z < 4 && Math.abs(z) < 4,
    `chi2 ${t.chi.toFixed(1)}/${t.df} df, z_WH ${t.z.toFixed(2)}; P(m>0): MC ${fmt(ch.fracPlus(), 5)} vs exact ${fmt(qE, 5)}, z ${z.toFixed(2)}`);
}
// ---------------------------------------------------------------- 7. Hopfield
{
  const N = 64, beta = 1.1, G = 0.01;
  // mixing check: the run is >= 15 full-chain relaxation times at these parameters,
  // tauMix = 1 / src.gap.gap_full(64, 1.1 * 63/64, 0.011) = 21.33 sweeps
  const tauMix = 21.33192192727143;
  const nSweeps = Math.ceil(15 * tauMix);
  const hp = new S.Hopfield(N, 1, { seed: 8, nChains: 3000 });
  hp.sweep(beta, G, nSweeps);
  const obs = new Float64Array(N + 1);
  for (let c = 0; c < hp.nChains; c++) obs[Math.round((hp.overlaps(c)[0] + 1) * N / 2)]++;
  const pi = S.cwPi(N, beta * (N - 1) / N, beta * G);
  const t = chiSquare(obs, pi, hp.nChains);
  check('Hopfield p=1 equilibrium overlap law = CW(beta(N-1)/N, h=beta G), N=64, beta=1.1', t.z < 4, `chi2 ${t.chi.toFixed(1)}/${t.df} df, z_WH ${t.z.toFixed(2)}; ${nSweeps} sweeps = 15 x full-chain relaxation time ${tauMix.toFixed(1)}`);
  const h3 = new S.Hopfield(256, 3, { seed: 9, nChains: 8 });
  h3.sweep(1.6, 0, 40);
  const cInc = Int32Array.from(h3.c); h3._recompute();
  check('Hopfield p=3 incremental overlaps equal recomputed xi·s after 40 sweeps', cInc.every((v, i) => v === h3.c[i]), `basins: ${[...Array(8).keys()].map((c) => h3.basin(c)).join('')}`);
}
// ---------------------------------------------------------------- 8. chunking
{
  let maxSlice = 0, last = performance.now(), slices = 0;
  ctx.requestAnimationFrame = (f) => setTimeout(() => { const t = performance.now(); slices++; f(t); }, 0);
  const gen = S.protocolGen(1024, S.PAPER.G0 * 2);
  const wrap = (function* () { let r; while (!(r = gen.next()).done) { const t = performance.now(); maxSlice = Math.max(maxSlice, t - last); last = t; yield r.value; } return r.value; })();
  let sliceStart = 0, worstSlice = 0;
  const res = await S.runChunked(wrap, { budgetMs: 12, onProgress: () => { worstSlice = Math.max(worstSlice, performance.now() - sliceStart); sliceStart = performance.now(); } });
  check('runChunked(protocolGen N=1024): longest single step < 5 ms, returns the result', maxSlice < 5 && res.c > 0 && slices > 5, `longest generator step ${maxSlice.toFixed(2)} ms, ${slices} slices, c = ${fmt(res.c, 6)}`);
}

// ---------------------------------------------------------------- bench (informational)
{
  const bench = (name, f, units) => { const t0 = performance.now(); const u = f(); const s = (performance.now() - t0) / 1000; console.log(`BENCH ${name}: ${(u / s / 1e6).toFixed(1)} M ${units}/s (${(s * 1000).toFixed(0)} ms)`); };
  bench('CWChains, 2000 chains N=1024, 4 sweeps at beta=1', () => { const c = new S.CWChains(2000, 1024, { seed: 1 }); return c.advance(1.0, 0, 4); }, 'events');
  bench('CWSpins N=4096, 200 sweeps', () => { const c = new S.CWSpins(4096); c.sweep(1.0, 0, 200); return 4096 * 200; }, 'spin updates');
  bench('Hopfield p=3, 64 chains N=512, 4 sweeps', () => { const h = new S.Hopfield(512, 3, { nChains: 64 }); h.sweep(1.0, 0, 4); return 64 * 512 * 4; }, 'spin updates');
  bench('protocolGen N=256', () => { S.drain(S.protocolGen(256, 0.0049)); return 1; }, 'protocols');
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
