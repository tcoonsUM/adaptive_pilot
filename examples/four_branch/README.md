# Four-branch benchmark

The benchmark uses the three fidelity parameter sets and costs reported in the manuscript.
The uncertain input is uniform on `[-6,6]` and the design domain is `[-3,3]`.

- `run_pilot.py` trains the active covariance emulator.
- `run_flat_pilot.py` collects an equal-cost, fixed space-filling pilot study for the
  domain-averaged baseline.
- `run_optimization.py` runs one stochastic-optimization trial using single-fidelity MC,
  emulator-informed ACV, domain-averaged ACV, or independent local pilot sampling.
- `run_grid.py` creates or executes the 12-setting, 30-trial experiment grid.

A representative comparison is:

```bash
python examples/four_branch/run_pilot.py \
  --config examples/four_branch/configs/illustrative.yaml \
  --output outputs/four_branch/pilot_active.npz

python examples/four_branch/run_flat_pilot.py \
  --config examples/four_branch/configs/illustrative.yaml \
  --output outputs/four_branch/pilot_space_filling.npz

python examples/four_branch/run_optimization.py \
  --config examples/four_branch/configs/illustrative.yaml \
  --pilot outputs/four_branch/pilot_active.npz \
  --flat-pilot outputs/four_branch/pilot_space_filling.npz \
  --method acv-emulator \
  --output outputs/four_branch/acv_emulator.npz
```

When `optimization.total_budget` is supplied, each method receives the same total budget
after its own initialization/pilot cost is charged. Integer allocation floors can leave a small
unused remainder. `paper_grid.yaml` computes a separate common total for every
`(B_iter, N_pd, N_pilot)` combination.

The core runnable backend is an independent-sample ACV estimator. To search the MXMCPy
allocation families used in the research scripts, install the `paper` optional dependencies and
set `optimization.estimator.backend: mxmcpy`.
