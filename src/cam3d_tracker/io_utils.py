from __future__ import annotations

import json
from pathlib import Path

from .math_utils import wrap_angle
from .models import Detection3D, FrameDetections, TrackOutput


def load_frames(path: str | Path) -> list[FrameDetections]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames: list[FrameDetections] = []
    for frame in data["frames"]:
        dets: list[Detection3D] = []
        for d in frame.get("detections", []):
            dets.append(
                Detection3D(
                    x=float(d["x"]),
                    y=float(d["y"]),
                    z=float(d["z"]),
                    yaw=wrap_angle(float(d["yaw"])),
                    l=float(d["l"]),
                    w=float(d["w"]),
                    h=float(d["h"]),
                    score=float(d["score"]),
                    label=str(d["label"]),
                    raw=d,
                )
            )
        frames.append(FrameDetections(timestamp_s=float(frame["timestamp_s"]), detections=dets))

    frames.sort(key=lambda x: x.timestamp_s)
    return frames


def save_tracks(path: str | Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"tracks": rows}, f, indent=2)


def flatten_outputs(timestamp_s: float, outputs: list[TrackOutput]) -> list[dict]:
    rows = []
    for out in outputs:
        r = out.to_dict()
        r["timestamp_s"] = float(timestamp_s)
        rows.append(r)
    return rows
