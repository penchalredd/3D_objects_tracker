from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_runtime_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Runtime config must be a mapping")
    return cfg
