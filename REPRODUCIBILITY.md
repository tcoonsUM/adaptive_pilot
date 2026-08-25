# Reproducibility guide

This repository separates numerical reproduction into three levels so a reader does not need
to rerun costly neural-surrogate and nested-Monte-Carlo calculations merely to inspect the
reported results.

## Level 1: supplied-result reproduction

Run:

```bash
bash scripts/reproduce_precomputed.sh
```

This produces:

- the four-branch high-fidelity mean and pairwise-correlation reference plot;
- the paper's four-branch regret tables in readable CSV form;
- the Case 1 sensor-location oracle surface, utility-correlation fields, and estimator summary;
- the Case 2 objective/correlation/transformed-covariance reference plot; and
- the Case 2 average regret-versus-runtime plot.

The files in `paper_results/` contain only arrays required for these outputs. Legacy pickle
files were converted to CSV or `.npz` before inclusion.

## Level 2: executable algorithm demonstrations

The four-branch workflow is fully self contained and is the recommended starting point for
understanding or adapting the method:

```bash
python examples/four_branch/run_pilot.py \
  --config examples/four_branch/configs/illustrative.yaml \
  --output outputs/four_branch/pilot.npz

python examples/four_branch/run_optimization.py \
  --config examples/four_branch/configs/illustrative.yaml \
  --pilot outputs/four_branch/pilot.npz \
  --method acv-emulator \
  --output outputs/four_branch/acv_emulator.npz
```

The `smoke.yaml` configuration is intended for tests. `illustrative.yaml` corresponds to the
single-run settings in the manuscript: 12 active pilot designs, 32 random-input samples per
active design, 40 mean-optimization iterations, and an ACV budget of 64 per iteration.
`paper_grid.yaml` enumerates the manuscript's 12 hyperparameter combinations and 30 trials.
The executable driver uses explicit, internally consistent cost accounting; the exact reported
aggregate tables are supplied under `paper_results/`.

The default implementation uses scikit-learn Gaussian processes, the self-contained
independent-sample ACV backend, and candidate-grid noisy expected improvement. Set
`estimator.backend: mxmcpy` and install the `paper` extra to use the optional allocation
adapter found in the research code. The public optimization driver intentionally avoids a
second BoTorch implementation; the covariance and estimator interfaces can be connected to
another optimizer without changing the pilot stage.

## Level 3: full paper experiments

### Four-branch grid

```bash
python examples/four_branch/run_grid.py \
  --config examples/four_branch/configs/paper_grid.yaml \
  --output-dir outputs/four_branch/grid
```

This first writes a restartable `manifest.jsonl`. Add `--execute` for sequential execution or
split manifest indices across a scheduler.

The grid contains

- per-iteration ACV budgets 32, 64, and 128;
- active pilot-design counts 6 and 12;
- pilot sample counts 16 and 32;
- four estimator strategies; and
- 30 independent trials.

Use a job scheduler or split the generated manifest across workers. Random seeds and output
metadata make jobs restartable.

### OED Case 1: sensor location

The supplied plot arrays reproduce the reported 60-trial grid-search comparison. To
recompute a reduced utility study from the trained surrogate models:

```bash
python examples/oed/case1_sensor_location/run_case1.py \
  --config examples/oed/case1_sensor_location/configs/smoke.yaml \
  --output outputs/oed/case1_smoke.npz
```

Use `paper.yaml` for the manuscript sample counts. That run is expensive: the single-fidelity
baseline uses 25 outer samples and 1,000 inner samples at every grid point, while the reference
surface was generated with much larger nested sample counts.

### OED Case 2: measurement time

```bash
python examples/oed/case2_measurement_time/run_pilot.py \
  --config examples/oed/case2_measurement_time/configs/smoke.yaml \
  --output outputs/oed/case2_pilot.npz

python examples/oed/case2_measurement_time/run_optimization.py \
  --config examples/oed/case2_measurement_time/configs/smoke.yaml \
  --pilot outputs/oed/case2_pilot.npz \
  --output outputs/oed/case2_optimization.npz
```

The manuscript configuration uses 10 active pilot designs, 20 mean-optimization iterations,
inner-loop sizes 1,000, 500, and 100, a fixed sensor at `(-0.8,-0.2)`, and a total nominal
runtime budget of 37.1 hours. The collaborative source mixes runtime and ensemble-sample
budget conventions; the public driver uses the explicit model costs and records its resulting
cost curve, while `paper_results/oed_case2/regret_curves.npz` reproduces the reported curve.

## Numerical and source discrepancies

### Four-branch reference value

The supplied model source implements

```text
min(branch_1, branch_2, branch_3, branch_4) + 3
```

with the manuscript's fidelity parameters. High-accuracy deterministic quadrature of that
implementation at `xi=0` gives approximately `2.1358`. The manuscript reports `2.159` from
a one-million-sample Monte Carlo calculation. The public model preserves the supplied source
formula and records both values in plot metadata. Regret calculations should use one declared
reference consistently.

### Initial pilot designs

The manuscript states that two space-filling designs with 20 high-fidelity runs initialize the
four-branch optimizer. The principal legacy function uses four initial designs. Public paper
YAML files use two; `legacy_source.yaml` is provided for debugging old result files.

### OED training data

The omitted files are raw finite-difference trajectories or large cached utility evaluations.
They are not needed to load the supplied MLP/FNO checkpoints. Training scripts are not part
of this focused public package because the requested scope is paper reproduction, covariance
emulation, and pilot sampling. The source audit records the omitted filenames and sizes.

## Determinism

Every public script accepts a seed. NumPy generators are passed explicitly, and PyTorch seeds
are set when neural models are used. Exact bitwise equality across BLAS, scikit-learn,
PyTorch, or GPU versions is not guaranteed. The supplied paper-result arrays are the stable
reference for plots and reported aggregate statistics.
