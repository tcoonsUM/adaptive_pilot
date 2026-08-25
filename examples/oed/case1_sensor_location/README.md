# OED Case 1: fixed-time sensor location

A sensor observes the concentration field at time `t=0.325`; the design is one of the
`21 x 21` grid locations. One surrogate rollout produces the full field, so utility estimates
can be assembled at all sensor locations at once.

`run_case1.py` performs an offline covariance pilot study and compares three estimators under
the same final-estimator budget:

- `NMC-SF`: high-fidelity nested Monte Carlo;
- `MFEIG-ADAPT`: one ACV allocation with location-dependent covariance weights; and
- `MFEIG-FLAT`: the same allocation with domain-averaged covariance weights.

Run the lightweight workflow with:

```bash
python examples/oed/case1_sensor_location/run_case1.py \
  --config examples/oed/case1_sensor_location/configs/smoke.yaml \
  --output outputs/oed/case1_smoke.npz
```

The `paper.yaml` file exposes the manuscript settings (300 outer pilot samples, 25 NMC outer
samples, 1000 inner samples, and 60 trials). It is computationally expensive. Stable plot-only
reproduction uses the supplied numeric arrays:

```bash
python examples/oed/case1_sensor_location/plot_precomputed.py
```

The public executable uses the self-contained independent-sample ACV family. The original
research script used MXMCPy for allocation search; the supplied paper arrays remain the
reference for the reported aggregate values.
