// Tests for lib/sim.js against the paper's Python (../src).
//   node lib/sim.test.mjs            (from visual/)
// Reference values: data/ref/reference.json, produced by
//   .venv/bin/python visual/data/ref/make_reference.py    (from the paper repo root)
// and the repository's results: results/simple/predictions.json (the test's frozen c_CW) and
// results/experiments/cw_linear_sweep.json (paper Table 2).
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
const PRED = JSON.parse(fs.readFileSync(path.join(ROOT, '../results/simple/predictions.json'), 'utf8'));
const LIN = JSON.parse(fs.readFileSync(path.join(ROOT, '../results/experiments/cw_linear_sweep.json'), 'utf8'));

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
  check('q_eq vs src.barrier.log_pi_closed_form, tie excluded (N=64,256,2048)', ok, `relative error on min(q,1-q) < 1e-11; worst ${fmt(worst, 3)}`);
  let worstL = 0;
  for (const r of PRED.rows.filter((x) => x.p === 1)) worstL = Math.max(worstL, Math.abs(S.logOddsEq(r.N, PRED.design.BETA_F, r.G) - r.L_eq));
  check('logit q_eq at beta = 1.6 vs results/simple/predictions.json L_eq (N=1024,2048,4096)', worstL < 1e-9, `worst abs error ${worstL.toExponential(2)}`);
}
// ---------------------------------------------------------------- 3. exact propagation
{
  const r = REF.evolve;
  const P = new S.CTPropagator(r.N);
  P.evolve(r.beta, r.h, r.t, r.dt);
  const c = allclose(P.p, r.p, 1e-9, 1e-14);
  check('CTPropagator one level vs src.cw_ramp.evolve (N=128, 25 sweeps)', c.ok, `rtol=1e-9, atol=1e-14; mass-1 = ${(P.mass() - 1).toExponential(2)}`);
  for (const rr of REF.ramp) {
    const res = S.drain(S.rampGen(rr.N, rr.G, { start: 'eq' }));
    check(`linear ramp on the exact chain, N=${rr.N}: q_chain vs src.cw_ramp.linear_sweep`, Math.abs(res.q_chain - rr.q_chain) < 1e-12,
      `q_chain ${fmt(res.q_chain, 12)} (src ${fmt(rr.q_chain, 12)})`);
  }
  const d = PRED.design;
  let worst = 0, ms = 0, digits = [];
  for (const r of PRED.rows.filter((x) => x.p === 1).sort((x, y) => x.N - y.N)) {
    const t0 = performance.now();
    const res = S.drain(S.rampGen(r.N, d.NG / r.N, { gamma: d.GAMMA, beta0: d.BETA0, beta_final: d.BETA_F }));
    ms += performance.now() - t0;
    worst = Math.max(worst, Math.abs(res.c / r.c_CW - 1));
    digits.push(res.c.toFixed(4));
  }
  check('c_CW on the test ramp = 0.0937, 0.0659, 0.0464 (Table 1) vs results/simple/predictions.json',
    worst < 1e-5 && digits.join(', ') === '0.0937, 0.0659, 0.0464' && Math.abs(S.RAMP.c_CW - PRED.rows.find((x) => x.N === 2048 && x.p === 1).c_CW) < 1e-15,
    `JS (continuous time) ${digits.join(', ')}; worst relative gap to the update-by-update chain ${worst.toExponential(2)} (< 1e-5); JS wall ${ms.toFixed(0)} ms`);
  let worstT = 0;
  for (const r of LIN.filter((x) => x.gamma === 0.004)) {
    const res = S.drain(S.rampGen(r.N, r.G, { start: 'eq' }));
    worstT = Math.max(worstT, Math.abs(res.c - r.c_exact) / r.c_exact);
  }
  check('Table 2, gamma = 0.004 column vs results/experiments/cw_linear_sweep.json', worstT < 1e-10, `worst relative error ${worstT.toExponential(2)}`);
}
// ---------------------------------------------------------------- 4. special functions, A_p, law
{
  const a = allclose(REF.ndtr.map((x) => S.Phi(x[0])), REF.ndtr.map((x) => x[1]), 1e-13, 1e-310);
  check("Phi vs mpmath ncdf at 40 digits (x in [-38, 9])", a.ok, a.ok ? `rtol=1e-13; worst err/tol ${fmt(a.worst, 3)}` : JSON.stringify(a));
  const A = [1, 2, 3, 4, 5, 6].map((p) => S.ampP(p));
  const c = allclose(A, [1, 2, 3, 4, 5, 6].map((p) => REF.A_p[p]), 1e-10);
  const rounded = A.slice(0, 4).map((x) => x.toFixed(3)).join(', ');
  check('A_p = E[max_p|Z|]/E|Z| vs scipy quad, p=1..6', c.ok && rounded === '1.000, 1.414, 1.662, 1.836', `rtol=1e-10; A_1..4 = ${rounded}`);
    let ok = true, worstC = 0;
  for (const r of LIN) {
    const q = S.knQ(r.G, r.N, r.gamma), c2 = S.knC(r.G, r.N, r.gamma, 1.6);
    worstC = Math.max(worstC, Math.abs(c2 - r.c_KN) / r.c_KN);
    if (!(Math.abs(q - r.q_KN) < 1e-13 && Math.abs(c2 - r.c_KN) / r.c_KN < 1e-6)) ok = false;
  }
  check('knQ, knC vs results/experiments/cw_linear_sweep.json (9 rows)', ok, `q to 1e-13; c rel ${fmt(worstC, 2)} (src q_eq keeps the tie mass, < 1e-6)`);
  const sp = S.stationaryPoints(1.6, 0).map((x) => x.m);
  check('stationaryPoints(1.6, 0) = {-m*, 0, m*} with m* = tanh(1.6 m*)', sp.length === 3 && Math.abs(Math.tanh(1.6 * sp[2]) - sp[2]) < 1e-12 && Math.abs(sp[1]) < 1e-12, sp.map((x) => fmt(x, 8)).join(', '));
}
// ---------------------------------------------------------------- 5. stochastic lumped chains vs the exact law
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
// ---------------------------------------------------------------- 6. chunking
{
  let maxSlice = 0, last = performance.now(), slices = 0;
  ctx.requestAnimationFrame = (f) => setTimeout(() => { const t = performance.now(); slices++; f(t); }, 0);
  const gen = S.rampGen(1024, S.RAMP.NG / 1024);
  const wrap = (function* () { let r; while (!(r = gen.next()).done) { const t = performance.now(); maxSlice = Math.max(maxSlice, t - last); last = t; yield r.value; } return r.value; })();
  let sliceStart = 0, worstSlice = 0;
  const res = await S.runChunked(wrap, { budgetMs: 12, onProgress: () => { worstSlice = Math.max(worstSlice, performance.now() - sliceStart); sliceStart = performance.now(); } });
  check('runChunked(rampGen N=1024): longest single step < 5 ms, returns the result', maxSlice < 5 && res.c > 0 && slices > 5, `longest generator step ${maxSlice.toFixed(2)} ms, ${slices} slices, c = ${fmt(res.c, 6)}`);
}

// ---------------------------------------------------------------- bench (informational)
{
  const bench = (name, f, units) => { const t0 = performance.now(); const u = f(); const s = (performance.now() - t0) / 1000; console.log(`BENCH ${name}: ${(u / s).toPrecision(3)} ${units}/s (${(s * 1000).toFixed(0)} ms)`); };
  bench('CWChains, 2000 chains N=1024, 4 sweeps at beta=1', () => { const c = new S.CWChains(2000, 1024, { seed: 1 }); return c.advance(1.0, 0, 4); }, 'jump events');
  bench('rampGen N=256', () => { S.drain(S.rampGen(256, S.RAMP.NG / 256)); return 1; }, 'ramps');
}

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
