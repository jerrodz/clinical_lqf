"""Expand data and repository paths in configurations and CSV files."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    raw = os.environ.get("DATA_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (PACKAGE_ROOT / "data").resolve()


def expand_path_string(raw: str) -> str:
    """Expand ${PACKAGE_ROOT}, ${DATA_ROOT}, and $ENV vars in config/CSV paths."""
    s = str(raw or "").strip()
    if not s:
        return s
    mapping = {
        "${PACKAGE_ROOT}": str(PACKAGE_ROOT),
        "${DATA_ROOT}": str(data_root()),
    }
    for key, val in mapping.items():
        s = s.replace(key, val)
    for key, val in os.environ.items():
        s = s.replace(f"${{{key}}}", val)
    return s
