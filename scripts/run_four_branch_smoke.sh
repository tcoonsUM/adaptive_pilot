#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-${ROOT}/outputs/smoke/four_branch}"
CONFIG="${ROOT}/examples/four_branch/configs/smoke.yaml"

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python "${ROOT}/examples/four_branch/run_pilot.py" \
  --config "${CONFIG}" \
  --output "${OUTPUT_DIR}/pilot_active.npz"

python "${ROOT}/examples/four_branch/run_flat_pilot.py" \
  --config "${CONFIG}" \
  --output "${OUTPUT_DIR}/pilot_space_filling.npz"

for method in mc-sf acv-emulator acv-flat acv-independent; do
  python "${ROOT}/examples/four_branch/run_optimization.py" \
    --config "${CONFIG}" \
    --pilot "${OUTPUT_DIR}/pilot_active.npz" \
    --flat-pilot "${OUTPUT_DIR}/pilot_space_filling.npz" \
    --method "${method}" \
    --output "${OUTPUT_DIR}/${method}.npz"
done

printf 'Four-branch smoke outputs: %s\n' "${OUTPUT_DIR}"
