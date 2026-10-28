"""The sampler-clock factor r(N) = c_meas(p=1, N) / c_CW(N): the permutation-sweep
sampler against the continuous-time exact chain on the same schedule, and whether
it accounts for the p=3 size-scan drift.  Reads size_scan_p1.json,
exact_protocol_sizes.json and size_scan_p3_analysis.json; writes sampler_clock.json
(paper Sec. 5, App. B)."""
import json
from scipy.special import logit
ex = {int(k): v for k, v in json.load(open('results/experiments/exact_protocol_sizes.json')).items()}
p1 = {int(k): v for k, v in json.load(open('results/experiments/size_scan_p1.json')).items()}
p3 = json.load(open('results/experiments/size_scan_p3_analysis.json'))
out = {}
for N in sorted(p1):
    c1 = float(logit(p1[N]['q']) / logit(ex[N]['q_eq'])); r = c1 / ex[N]['c_cw']
    out[N] = dict(c_meas_p1=c1, c_cw=ex[N]['c_cw'], r=r, n=p1[N]['n'])
    print(f"N={N}: c_meas(p=1)={c1:.4f} c_CW={ex[N]['c_cw']:.4f} r={r:.3f}")
for N in (1024, 2048, 4096):
    v = p3[str(N)]; out[N]['p3_meas_over_pred'] = v['c_meas'] / v['c_pred']
    out[N]['p3_meas_over_pred_times_r'] = v['c_meas'] / (v['c_pred'] * out[N]['r'])
    print(f"N={N}: p=3 meas/pred={out[N]['p3_meas_over_pred']:.3f}; with r: {out[N]['p3_meas_over_pred_times_r']:.3f}")
json.dump(out, open('results/experiments/sampler_clock.json', 'w'), indent=1)
