#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  python "$ROOT/scripts/reproduce_precomputed.py" \
  --repository-root "$ROOT" \
  --output-dir "$ROOT/figures/generated"
