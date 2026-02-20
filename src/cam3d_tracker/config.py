from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrackerConfig:
    raw: dict[str, Any]

    @property
    def tracker(self) -> dict[str, Any]:
        return self.raw["tracker"]

    @property
    def association(self) -> dict[str, Any]:
        return self.raw["association"]

    @property
    def noise(self) -> dict[str, Any]:
        return self.raw["noise"]

    @property
    def imm(self) -> dict[str, Any]:
        return self.raw["imm"]


def load_config(path: str | Path) -> TrackerConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return TrackerConfig(raw=data)
