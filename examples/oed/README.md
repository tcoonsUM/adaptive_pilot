# Convection–diffusion OED examples

The two examples share one high-fidelity proxy MLP and two autoregressive FNO surrogates.
The checkpoints and portable numeric transform parameters are stored under `assets/oed/`; raw finite-difference
training trajectories are not required for inference and are not included.

- `case1_sensor_location/`: fixed-time, all-at-once spatial sensor placement.
- `case2_measurement_time/`: active covariance pilot sampling followed by iterative
  measurement-time optimization.

Start with each example's `smoke.yaml`. The `paper.yaml` files expose the manuscript sample
counts but can require many hours of computation. The compact arrays under `paper_results/`
reproduce reported plots without rerunning the nested Monte Carlo studies.
