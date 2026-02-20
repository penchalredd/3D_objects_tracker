from __future__ import annotations

from .config import load_config
from .io_utils import flatten_outputs, load_frames, save_tracks
from .tracker import Classical3DTracker


def run_tracking(config_path: str, detections_path: str, output_path: str) -> None:
    cfg = load_config(config_path).raw
    tracker = Classical3DTracker(cfg)
    frames = load_frames(detections_path)

    rows: list[dict] = []
    for frame in frames:
        outputs = tracker.step(frame.timestamp_s, frame.detections)
        rows.extend(flatten_outputs(frame.timestamp_s, outputs))

    save_tracks(output_path, rows)
