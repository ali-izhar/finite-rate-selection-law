// Writes data/odds/c_vs_bias.json: the exact two-mode chain under the paper's full protocol at
// 16 bias values G, computed with lib/sim.js (which lib/sim.test.mjs tests against ../src).
// Run from the paper repo root:  node visual/data/odds/make_c_vs_bias.mjs   (prints one line per G)
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');   // visual/
const ctx={}; new Function('window', fs.readFileSync(ROOT+'/lib/sim.js','utf8'))(ctx);
const S=ctx.VE.sim; const N=2048, bf=1.6;
function GforL(L){let a=0,b=0.01;for(let i=0;i<80;i++){const m=(a+b)/2; if(S.logOddsEq(N,bf,m)<L)a=m;else b=m;}return (a+b)/2;}
const targetsL=[0.1,0.25,Math.log(2),1,1.5,2,Math.log(9),2.5,3,Math.log(36),3.5,4,Math.log(99),5,5.5];
const Gs=targetsL.map(GforL); Gs.push(6.125e-4);
const rows=[];
for(const G of Gs.sort((a,b)=>a-b)){const t=Date.now(); const r=S.drain(S.protocolGen(N,G,{dt:0.02}));
 rows.push({G, logit_q_eq:S.logOddsEq(N,bf,G), q_eq:r.q_eq, q_chain:r.q_chain, logit_q_chain:S.logit(r.q_chain), c:r.c, f_mis:r.f_mis});
 console.error(G, r.q_eq.toFixed(5), r.q_chain.toFixed(5), r.c.toFixed(5), (Date.now()-t)+'ms');}
fs.writeFileSync(ROOT+'/data/odds/c_vs_bias.json', JSON.stringify({
 what:"Exact two-mode (Curie-Weiss) chain, N=2048, the paper's full protocol (scarce descent 2 sweeps/level, keep mis-committed mass, reheat 30 sweeps at beta=0.7, re-descent 834 sweeps on the 10-level ladder 0.6..1.6), at several bias values G (h = beta*G). q_eq = equilibrium P(m>0) at beta=1.6; q_chain = P(m>0) after the re-descent; c = logit q_chain / logit q_eq. No sampling: exact propagation of the law over k.",
 source:"VE.sim.protocolGen(2048, G, {dt:0.02}) in visual/lib/sim.js (mirrors src/experiments/exact_protocol.py; tested by lib/sim.test.mjs against it). G values chosen by bisection so logit q_eq hits round targets (incl. ln 2 = model says 2:1, ln 9, ln 36, ln 99), plus the paper's G0 = 6.125e-4.",
 paper:"Sec. 4: c_CW = 0.120 at N = 2048 (the row G = 6.125e-4 reproduces data/c_cw.json: 0.119808).",
 command:"node visual/data/odds/make_c_vs_bias.mjs",
 script:"visual/data/odds/make_c_vs_bias.mjs",
 N, beta_final:bf, rows},null,1));
