from __future__ import annotations

from typing import Any


def build_model_input(frame: dict[str, Any], device: Any, torch: Any) -> dict[str, Any]:
    """
    Example adapter.

    Replace this function with your model-specific preprocessing:
    - Read 6 camera images from frame["camera_paths"]
    - Resize/normalize/stack
    - Add calibration/metadata expected by your model
    - Move tensors to `device`
    """
    return {
        "camera_paths": frame["camera_paths"],
        "timestamp_s": frame["timestamp_s"],
        "sample_token": frame["sample_token"],
    }


def convert_model_output(raw_output: Any, frame: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert raw model output into tracker detection schema.

    Required per detection:
      x, y, z, yaw, l, w, h, score, label

    Return list[dict].
    """
    if isinstance(raw_output, dict) and "detections" in raw_output:
        return raw_output["detections"]
    if isinstance(raw_output, list):
        return raw_output
    raise ValueError("Unsupported model output. Implement convert_model_output for your model format.")
