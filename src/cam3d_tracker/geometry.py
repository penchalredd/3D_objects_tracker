from __future__ import annotations

import math

import numpy as np

from .math_utils import angle_diff


def oriented_box_corners_xy(x: float, y: float, yaw: float, l: float, w: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    dx = l / 2.0
    dy = w / 2.0
    local = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]], dtype=float)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    return (local @ rot.T) + np.array([x, y], dtype=float)


def _polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _inside(p: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
    return (edge_end[0] - edge_start[0]) * (p[1] - edge_start[1]) - (edge_end[1] - edge_start[1]) * (p[0] - edge_start[0]) >= 0.0


def _intersection(s: np.ndarray, e: np.ndarray, cp1: np.ndarray, cp2: np.ndarray) -> np.ndarray:
    dc = cp1 - cp2
    dp = s - e
    n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
    n2 = s[0] * e[1] - s[1] * e[0]
    denom = dc[0] * dp[1] - dc[1] * dp[0]
    if abs(denom) < 1e-9:
        return e
    x = (n1 * dp[0] - n2 * dc[0]) / denom
    y = (n1 * dp[1] - n2 * dc[1]) / denom
    return np.array([x, y], dtype=float)


def polygon_clip(subject_polygon: np.ndarray, clip_polygon: np.ndarray) -> np.ndarray:
    output = subject_polygon.copy()
    cp1 = clip_polygon[-1]
    for cp2 in clip_polygon:
        input_list = output
        if len(input_list) == 0:
            return np.empty((0, 2), dtype=float)
        output_list = []
        s = input_list[-1]
        for e in input_list:
            if _inside(e, cp1, cp2):
                if not _inside(s, cp1, cp2):
                    output_list.append(_intersection(s, e, cp1, cp2))
                output_list.append(e)
            elif _inside(s, cp1, cp2):
                output_list.append(_intersection(s, e, cp1, cp2))
            s = e
        output = np.array(output_list, dtype=float) if output_list else np.empty((0, 2), dtype=float)
        cp1 = cp2
    return output


def bev_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    # box = [x, y, z, v, yaw, yaw_rate, l, w, h]
    pa = oriented_box_corners_xy(box_a[0], box_a[1], box_a[4], box_a[6], box_a[7])
    pb = oriented_box_corners_xy(box_b[0], box_b[1], box_b[4], box_b[6], box_b[7])
    inter_poly = polygon_clip(pa, pb)
    inter = _polygon_area(inter_poly)
    if inter <= 0.0:
        return 0.0
    ua = _polygon_area(pa) + _polygon_area(pb) - inter
    if ua <= 1e-9:
        return 0.0
    return float(inter / ua)


def yaw_cost(yaw_a: float, yaw_b: float) -> float:
    return min(abs(angle_diff(yaw_a, yaw_b)) / math.pi, 1.0)
