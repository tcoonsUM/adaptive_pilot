# Compact paper results

These files are derived from the supplied research outputs and are intended for inexpensive,
stable plot reproduction.

- `four_branch/reference_quantities.npz` evaluates the supplied four-branch source formula on
  a fixed design grid using 200,000 common random samples. It also records the deterministic
  quadrature value at `xi=0` and the distinct value reported in the manuscript.
- `four_branch/table2_regret_reduction.csv` and `appendix_a_regrets.csv` transcribe the
  manuscript's aggregate and configuration-level regret tables.
- `oed_case1/case1_precomputed.npz` contains the oracle EIG surface, 60 trial-level estimator
  surfaces, regrets, and compact local utility covariance/correlation fields. The much larger
  raw field and training caches are not included.
- `oed_case1/table5_summary.csv` transcribes the reported regret and analytical estimator
  variances.
- `oed_case2/reference_quantities.npz` contains the cost-aware utility means, covariances,
  correlations, gamma parameters, and log standard deviations derived from the supplied
  pilot-utility file using the final manuscript cost scaling.
- `oed_case2/regret_curves.npz` contains the trial-level curves used for the runtime-regret
  comparison.

All NumPy archives use numeric arrays only and can be opened with `allow_pickle=False`.
