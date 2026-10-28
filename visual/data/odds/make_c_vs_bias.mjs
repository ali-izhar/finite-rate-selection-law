// Writes data/odds/c_vs_bias.json: the exact Curie-Weiss chain on the test ramp (beta 0.7 -> 1.6 at
// gamma = 0.004, independent unbiased spins at the start; paper Sec. 4) at 16 field values G,
// N = 2048, computed with lib/sim.js (which lib/sim.test.mjs tests against ../src).
// It also prints the [logit q_eq, logit q_chain] pairs that components/odds.js carries inline.
// Run from the repository root:  node visual/data/odds/make_c_vs_bias.mjs   (about 10 s)
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');   // visual/
const ctx = {}; new Function('window', fs.readFileSync(ROOT + '/lib/sim.js', 'utf8'))(ctx);
const S = ctx.VE.sim, R = S.RAMP, N = R.N, bf = R.beta_final;
function GforL(L) { let a = 0, b = 0.01; for (let i = 0; i < 80; i++) { const m = (a + b) / 2; if (S.logOddsEq(N, bf, m) < L) a = m; else b = m; } return (a + b) / 2; }
// round model log-odds (ln 2: the model says 2:1; ln 9, ln 36, ln 99), plus the test's own field
const Gs = [0.1, 0.25, Math.log(2), 1, 1.5, 2, Math.log(9), 2.5, 3, Math.log(36), 3.5, 4, Math.log(99), 5, 5.5].map(GforL);
Gs.push(R.G0);
const rows = [];
for (const G of Gs.sort((a, b) => a - b)) {
  const r = S.drain(S.rampGen(N, G));
  rows.push({ G, logit_q_eq: S.logOddsEq(N, bf, G), q_eq: r.q_eq, q_chain: r.q_chain, logit_q_chain: S.logit(r.q_chain), c: r.c });
  console.error(G.toExponential(4), r.q_eq.toFixed(5), r.q_chain.toFixed(5), r.c.toFixed(5));
}
fs.writeFileSync(ROOT + '/data/odds/c_vs_bias.json', JSON.stringify({
  what: 'Exact Curie-Weiss chain, N = 2048, on the test ramp (beta 0.7 -> 1.6, gamma = 0.004, start: independent unbiased spins), at 16 field values G (h = beta G). q_eq = equilibrium P(m > 0) at beta = 1.6; q_chain = P(m > 0) at the end of the ramp; c = logit q_chain / logit q_eq. No sampling.',
  source: 'VE.sim.rampGen(2048, G) in visual/lib/sim.js (mirrors src/cw_ramp.py; tested by lib/sim.test.mjs). G chosen by bisection so logit q_eq hits round targets, plus the test field G = 6.125e-4.',
  paper: 'Sec. 4, Table 1: c_CW = 0.0659 at N = 2048 (the row G = 6.125e-4)',
  command: 'node visual/data/odds/make_c_vs_bias.mjs',
  N, beta_final: bf, gamma: R.gamma, rows,
}, null, 1) + '\n');
console.log('EXACT = [' + rows.map((x) => '[' + +x.logit_q_eq.toFixed(6) + ', ' + +x.logit_q_chain.toFixed(6) + ']').join(', ') + ']');
