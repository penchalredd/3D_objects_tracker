from __future__ import annotations

import math


def wrap_angle(theta: float) -> float:
    while theta > math.pi:
        theta -= 2.0 * math.pi
    while theta < -math.pi:
        theta += 2.0 * math.pi
    return theta


def angle_diff(a: float, b: float) -> float:
    return wrap_angle(a - b)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
