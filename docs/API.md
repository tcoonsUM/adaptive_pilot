# Core API guide

## Model interface

A stochastic model is a callable

```python
model(design, random_inputs) -> outputs
```

where `design` is a one-dimensional deterministic design vector, `random_inputs` has shape
`(n_samples, n_random_dimensions)`, and `outputs` contains one scalar per row. A model
ensemble is an ordered sequence with the high-fidelity model first. All models are evaluated
on common random inputs within each local covariance pilot batch.

## Covariance coordinates

`adaptive_covariance.covariance.transforms` represents a positive-definite covariance matrix
as

```text
Sigma = D R D,
gamma = vech_offdiag(log(R)),
ell = log(diag(D)).
```

`covariance_to_unconstrained` maps `Sigma` to `(gamma, ell)`, and
`unconstrained_to_covariance` performs the inverse map. The correlation inverse uses the
fixed-point diagonal correction required to make `exp(log(R))` have unit diagonal. Mapping
GP posterior means through this inverse always produces a positive-definite covariance
matrix.

## Finite-pilot observations

`observation_from_outputs(outputs, wishart_draws=...)` computes:

- the unbiased sample covariance;
- transformed `gamma` and log-standard-deviation observations;
- componentwise observation-noise variances from a plug-in Wishart pushforward;
- model sample means and variances of those means; and
- the local pilot sample count.

The Gaussian/Wishart approximation is most appropriate once the local sample count exceeds
the number of models and is large enough for transformed sampling distributions to be
approximately Gaussian.

## Covariance emulator

`CovarianceEmulator` fits one heteroscedastic scikit-learn Gaussian process per transformed
component. Typical use is:

```python
emulator = CovarianceEmulator(settings).fit(
    designs,
    gammas,
    log_standard_deviations,
    gamma_noise_variances,
    log_standard_deviation_noise_variances,
)

sigma = emulator.predict_covariance([[0.25]])[0]
uncertainty = emulator.uncertainty_score(candidate_designs)
```

`predict_transformed` returns posterior means and marginal standard deviations. The active
acquisition score is the sum of those standard deviations. The convenience method
`high_fidelity_standard_deviation_upper` implements the manuscript's conservative high-
fidelity variance inflation heuristic.

## Active pilot study

`run_pilot_study` accepts models, a random-input sampler, a finite acquisition candidate set,
initial designs, and `PilotStudyConfig`. It performs the following steps:

1. evaluate all models at the initial designs;
2. fit the covariance and high-fidelity-mean emulators;
3. maximize covariance uncertainty during the exploratory phase;
4. switch to mean-weighted covariance uncertainty at the configured fraction;
5. collect a new common-input pilot batch and refit; and
6. return a portable `PilotStudyResult` containing all local observations and the frozen
   covariance emulator.

The serialized `.npz` contains numeric arrays and JSON metadata only. Loading refits the GP
models rather than unpickling a Python estimator.

`run_pilot_at_designs` uses the same observation and emulator machinery with a prescribed
space-filling design sequence. It is the reusable building block for a domain-averaged or
otherwise fixed-design pilot baseline.

## Self-contained ACV estimator

`IndependentSampleACV` implements a common-base, independent-extra-sample allocation:

```text
Q_tilde = Q0_base + alpha^T (Q_low_extra - Q_low_base).
```

`design(covariance, costs, budget)` searches the base count and numerically allocates extra
low-fidelity samples. `estimate(...)` then draws exactly that allocation. This backend is
intended as a clear, dependency-light implementation for adaptation and testing. The optional
`MXMCPyBackend` searches the estimator families exposed by MXMCPy when that package is
installed.

## Downstream optimization

`NoisyBayesianOptimizer` is a compact heteroscedastic GP optimizer over a finite candidate
set. It uses expected improvement during the loop and returns the posterior-mean maximizer at
termination. It is intentionally separate from pilot sampling: the pilot stage learns
covariance structure, while the downstream optimizer learns the mean objective.

## OED utilities

`OEDSurrogateEnsemble` loads the included high-fidelity proxy MLP and two FNO checkpoints.
`Case1UtilityFieldEvaluator` evaluates all sensor locations at a fixed time.
`Case2UtilityEnsemble` exposes three scalar cost-aware EIG utility models with the generic
model signature required by the pilot loop.

The OED output transform is reconstructed from numeric-only Yeo--Johnson and
standardization parameters. Identity input transforms are represented directly in code, so
loading the public OED assets does not execute joblib or pickle files.

The NMC utility functions use shared inner prior samples and a log-sum-exp evidence
calculation. Different inner sizes may be selected by fidelity; prefixes of one common prior
sample are used to preserve sample sharing.
