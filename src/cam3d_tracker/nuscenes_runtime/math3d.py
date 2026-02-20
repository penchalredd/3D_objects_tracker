from __future__ import annotations

import math


def quat_to_yaw(q_wxyz: list[float]) -> float:
    w, x, y, z = q_wxyz
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def ego_to_global_xyzyaw(x: float, y: float, z: float, yaw: float, ego_pose: dict) -> tuple[float, float, float, float]:
    tx, ty, tz = ego_pose["translation"]
    ego_yaw = float(ego_pose["yaw"])
    c = math.cos(ego_yaw)
    s = math.sin(ego_yaw)
    gx = c * x - s * y + tx
    gy = s * x + c * y + ty
    gz = z + tz
    gyaw = yaw + ego_yaw
    while gyaw > math.pi:
        gyaw -= 2.0 * math.pi
    while gyaw < -math.pi:
        gyaw += 2.0 * math.pi
    return gx, gy, gz, gyaw
