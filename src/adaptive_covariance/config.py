"""Small configuration helpers used by the example command-line programs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject non-mapping top-level documents."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}, got {type(data).__name__}")
    return data


def repository_root() -> Path:
    """Locate the source-checkout root used by example and reproduction scripts.

    Core package APIs do not depend on a repository checkout.  Examples additionally
    require ``assets/`` and ``paper_results/``; this helper first searches the current
    working directory and then the installed module path so it works for normal source
    and editable installations.
    """
    candidates = (Path.cwd().resolve(), Path(__file__).resolve())
    for candidate in candidates:
        for parent in (candidate, *candidate.parents):
            if (parent / "pyproject.toml").exists() and (
                parent / "src" / "adaptive_covariance"
            ).exists():
                return parent
    raise RuntimeError(
        "Could not locate the adaptive-covariance source checkout. "
        "Run the example from the repository or pass explicit asset/result paths."
    )
