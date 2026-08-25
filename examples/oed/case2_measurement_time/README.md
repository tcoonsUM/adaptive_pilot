# OED Case 2: measurement-time design

The sensor is fixed at `(-0.8, -0.2)`, and the design is the measurement time in
`[0.1, 0.6]`. The optimized objective is expected information gain divided by the
experimental cost `0.1 + t`.

Run a small end-to-end demonstration:

```bash
python examples/oed/case2_measurement_time/run_pilot.py \
  --config examples/oed/case2_measurement_time/configs/smoke.yaml \
  --output outputs/oed/case2_pilot.npz

python examples/oed/case2_measurement_time/run_optimization.py \
  --config examples/oed/case2_measurement_time/configs/smoke.yaml \
  --pilot outputs/oed/case2_pilot.npz \
  --method mfeig-gamma-opt \
  --output outputs/oed/case2_optimization.npz
```

Available optimization methods are `mfeig-gamma-opt`, `mfeig-flat`, and `nmc-sf`. The
`paper.yaml` configuration follows the manuscript's 10 active pilot designs, 128 utility
samples per active design, 20 objective-optimization iterations, inner sample sizes
`[1000, 500, 100]`, and nominal total budget of 37.1 hours.

The research source and manuscript use slightly different intermediate budget and cost
conventions. The public driver charges every utility sample according to the explicitly
reported model runtimes and chosen inner-size scaling; the supplied paper arrays are the
stable source for the published runtime-regret curve.

Plot-only reproduction:

```bash
python examples/oed/case2_measurement_time/plot_precomputed.py
```
