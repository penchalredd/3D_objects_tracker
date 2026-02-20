from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Detection3D:
    x: float
    y: float
    z: float
    yaw: float
    l: float
    w: float
    h: float
    score: float
    label: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def z_vec(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.yaw, self.l, self.w, self.h], dtype=float)


@dataclass
class FrameDetections:
    timestamp_s: float
    detections: list[Detection3D]


@dataclass
class TrackOutput:
    track_id: int
    label: str
    score: float
    state: np.ndarray
    age_s: float
    hits: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        x = self.state
        return {
            "track_id": self.track_id,
            "label": self.label,
            "score": float(self.score),
            "x": float(x[0]),
            "y": float(x[1]),
            "z": float(x[2]),
            "v": float(x[3]),
            "yaw": float(x[4]),
            "yaw_rate": float(x[5]),
            "l": float(x[6]),
            "w": float(x[7]),
            "h": float(x[8]),
            "age_s": float(self.age_s),
            "hits": int(self.hits),
            "status": self.status,
        }
