#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/adaptive-covariance-release.XXXXXX")"
cleanup() {
  rm -rf \
    "${WORK_DIR}" \
    "${ROOT}/build" \
    "${ROOT}/dist" \
    "${ROOT}/src/adaptive_covariance.egg-info"
}
trap cleanup EXIT
export PYTHONPYCACHEPREFIX="${WORK_DIR}/pycache"

python -m compileall -q src examples scripts tests
pytest -q -p no:cacheprovider
python scripts/reproduce_precomputed.py --output-dir "${WORK_DIR}/figures"
python -m pip wheel . --no-deps --no-build-isolation -w "${WORK_DIR}/dist"

python - <<'PY'
from __future__ import annotations

import hashlib
from pathlib import Path

root = Path.cwd()
checksum_file = root / "CHECKSUMS.sha256"
if not checksum_file.exists():
    raise SystemExit("CHECKSUMS.sha256 is missing; regenerate it before release")

failures: list[str] = []
for line in checksum_file.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    expected, relative = line.split(maxsplit=1)
    path = root / relative
    if not path.is_file():
        failures.append(f"missing: {relative}")
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        failures.append(f"checksum mismatch: {relative}")

if failures:
    raise SystemExit("\n".join(failures))
print("Release checks and asset/result checksums passed.")
PY
