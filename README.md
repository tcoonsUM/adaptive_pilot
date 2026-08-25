# Adaptive Covariance

Publication-oriented code for **active pilot sampling and covariance emulation in
multi-fidelity stochastic optimization**. The repository accompanies the submitted paper

> Thomas E. Coons, Aniket Jivani, and Xun Huan, *Active Pilot Sampling for
> Multi-fidelity Estimation in Stochastic Optimization*.

The code is organized around the two numerical demonstrations in the paper:

1. the one-dimensional three-fidelity four-branch stochastic-optimization benchmark; and
2. the convection–diffusion Bayesian optimal experimental design (OED) examples for
   sensor location and measurement time.

The core package is intentionally problem independent. A user supplies a model ensemble,
a random-input sampler, design candidates, model costs, and pilot-study settings. The
package then

1. evaluates all fidelities on common pilot samples at selected designs;
2. estimates local output covariance matrices;
3. maps each covariance to unconstrained matrix-log correlation parameters and log
   standard deviations;
4. fits independent heteroscedastic Gaussian-process emulators;
5. chooses new pilot designs using covariance uncertainty, optionally weighted by a
   surrogate for the high-fidelity mean objective; and
6. exposes design-specific covariance predictions to a downstream multi-fidelity estimator
   and stochastic optimizer.

## Repository layout

```text
src/adaptive_covariance/
  covariance/        positive-definite transforms, pilot observations, GP emulator
  pilot/             acquisition functions and active pilot loop
  estimators/        single-fidelity MC, self-contained ACV, optional MXMCPy backend
  optimization/      lightweight noisy Bayesian optimization on a candidate set
  models/            the four-branch model ensemble
  oed/               convection–diffusion physics, neural networks, and EIG utilities
examples/
  four_branch/       runnable benchmark and paper configurations
  oed/               Case 1 and Case 2 scripts
paper_results/       compact arrays and CSV tables used by plot-only reproduction
assets/oed/           trained MLP/FNO checkpoints and numeric transform parameters
tests/               fast tests for the reusable numerical components
```

`REPRODUCIBILITY.md` distinguishes plot-only, lightweight executable, and full-paper workflows. Author-controlled licensing and release steps are listed in `PUBLICATION_CHECKLIST.md`.

## Installation

Core covariance emulation and the four-branch example:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[plots,dev]
```

The OED surrogate models require PyTorch:

```bash
python -m pip install -e .[oed,plots]
```

For OED inference together with the MXMCPy allocation families searched by the original
research scripts:

```bash
python -m pip install -e .[paper,dev]
```

A Conda environment is also provided in `environment-paper.yml`.

## Fast start

Run a small end-to-end four-branch pilot study and optimization:

```bash
python examples/four_branch/run_pilot.py \
  --config examples/four_branch/configs/smoke.yaml \
  --output outputs/four_branch/pilot.npz

python examples/four_branch/run_optimization.py \
  --config examples/four_branch/configs/smoke.yaml \
  --pilot outputs/four_branch/pilot.npz \
  --method acv-emulator \
  --output outputs/four_branch/optimization.npz
```

Reproduce the inexpensive reference plots and paper tables from supplied arrays:

```bash
bash scripts/reproduce_precomputed.sh
```

Complete smoke workflows are also available as `scripts/run_four_branch_smoke.sh` and
`scripts/run_oed_smoke.sh`.

For an `acv-flat` comparison, first run `run_flat_pilot.py` and pass its archive with
`--flat-pilot`; the experiment-grid driver does this automatically.

The generated files are placed under `figures/generated/`.

Run the reduced OED examples with the included neural-network assets:

```bash
python examples/oed/case1_sensor_location/run_case1.py \
  --config examples/oed/case1_sensor_location/configs/smoke.yaml \
  --output outputs/oed/case1_smoke.npz

python examples/oed/case2_measurement_time/run_pilot.py \
  --config examples/oed/case2_measurement_time/configs/smoke.yaml \
  --output outputs/oed/case2_pilot.npz

python examples/oed/case2_measurement_time/run_optimization.py \
  --config examples/oed/case2_measurement_time/configs/smoke.yaml \
  --pilot outputs/oed/case2_pilot.npz \
  --method mfeig-gamma-opt \
  --output outputs/oed/case2_optimization.npz
```

## Using the pilot loop on another problem

A model is any callable with the signature

```python
model(design: numpy.ndarray, random_inputs: numpy.ndarray) -> numpy.ndarray
```

where one scalar output is returned for each row of `random_inputs`. All fidelities must be
evaluated on the same random inputs within a pilot batch. A minimal setup is:

```python
import numpy as np
from adaptive_covariance.pilot import PilotStudyConfig, run_pilot_study

models = [high_fidelity, medium_fidelity, low_fidelity]
costs = np.array([1.0, 1.0e-2, 1.0e-4])
candidates = np.linspace(-3.0, 3.0, 401)[:, None]
initial_designs = np.linspace(-3.0, 3.0, 2)[:, None]


def sample_inputs(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-6.0, 6.0, size=(n, 1))

config = PilotStudyConfig(
    n_active_designs=12,
    n_samples_per_design=32,
    n_initial_samples=20,
    acquisition_switch_fraction=0.5,
    wishart_draws=1000,
    random_seed=42,
)

result = run_pilot_study(
    models=models,
    sample_random_inputs=sample_inputs,
    candidate_designs=candidates,
    initial_designs=initial_designs,
    config=config,
)

covariance_at_x = result.emulator.predict_covariance([[0.25]])[0]
```

The principal pilot hyperparameters are explicit inputs: initial designs and sample count,
number of active pilot designs, random-input samples per active design, acquisition-switch
schedule, Wishart pushforward draws, candidate set, GP settings, and random seed.

## Estimator backends

`IndependentSampleACV` is a self-contained ACV implementation suitable for examples and
new applications. It uses a common base sample across all fidelities and independent
additional samples for each low-fidelity model, then numerically optimizes the allocation
under the supplied cost budget.

The research scripts searched the allocation families exposed by MXMCPy. Install the
`mxmcpy` or `paper` extra and choose the `mxmcpy` backend in the four-branch YAML
configuration to use that adapter. The OED executable examples use the self-contained backend; the supplied
paper-result arrays remain the stable reference for the reported aggregate values. The
optional MXMCPy dependency is isolated so the covariance-emulation package remains usable
when it is unavailable.

## Reproducibility tiers

- **Plot-only:** seconds; uses `paper_results/` and does not load neural networks.
- **Executable four-branch:** minutes on a CPU for a single run; the 30-trial, 12-setting
  paper grid is longer and can be parallelized by configuration.
- **OED smoke runs:** use reduced inner/outer sample counts and the included checkpoints.
- **Full OED study:** computationally expensive. The Case 1 reference surface uses very
  large nested sample counts, and Case 2 repeats neural-surrogate utility evaluations inside
  pilot sampling and optimization. The scripts expose manuscript configurations but the
  `smoke.yaml` files are the defaults for code inspection and testing.


## Data and model assets

The archive includes the trained MLP, medium-fidelity FNO, low-fidelity FNO, and numeric
transform parameters needed to execute the OED surrogate ensemble. Raw finite-difference training
trajectories were omitted because they exceeded the upload threshold and are not required
for inference. Compact derived arrays replace large or unsafe research pickle files wherever
possible.
