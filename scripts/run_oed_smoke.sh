#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-${ROOT}/outputs/smoke/oed}"
CASE1_CONFIG="${ROOT}/examples/oed/case1_sensor_location/configs/smoke.yaml"
CASE2_CONFIG="${ROOT}/examples/oed/case2_measurement_time/configs/smoke.yaml"

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python "${ROOT}/examples/oed/case1_sensor_location/run_case1.py" \
  --config "${CASE1_CONFIG}" \
  --output "${OUTPUT_DIR}/case1.npz"

python "${ROOT}/examples/oed/case2_measurement_time/run_pilot.py" \
  --config "${CASE2_CONFIG}" \
  --output "${OUTPUT_DIR}/case2_pilot.npz"

for method in mfeig-gamma-opt mfeig-flat nmc-sf; do
  python "${ROOT}/examples/oed/case2_measurement_time/run_optimization.py" \
    --config "${CASE2_CONFIG}" \
    --pilot "${OUTPUT_DIR}/case2_pilot.npz" \
    --method "${method}" \
    --output "${OUTPUT_DIR}/case2_${method}.npz"
done

printf 'OED smoke outputs: %s\n' "${OUTPUT_DIR}"
