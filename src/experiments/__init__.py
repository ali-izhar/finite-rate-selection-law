"""Exact-chain predictions, GPU reruns and analyses of saved runs.

Exact chain (CPU):
  exact_protocol         c_CW: the Hopfield protocol on the exact Curie-Weiss chain
  exact_protocol_sizes   c_CW(N) at several N (Table 1)
  cw_scaling             c(N, schedule scale) on the exact chain (Table 3)
  cw_linear_sweep        linear ramps against the selection law (Table 2)
Sampler reruns (torch + GPU; each calls src.hopfield.pipeline.run_protocol):
  hopfield_runs          pattern counts p = 1..4 and the size scan (Table 1), one-pattern runs
Trained Boltzmann machine (NumPy; needs raw MNIST; docs/mnist-illustration.md):
  dbm_reheat             the 24-run ensemble of the illustration
  dbm_control_arm        the two random-start controls
Analyses of saved results:
  draw_level_analysis, size_scan_p3_analysis, sampler_clock
"""
