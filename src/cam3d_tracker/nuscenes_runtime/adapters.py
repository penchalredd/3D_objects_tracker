from __future__ import annotations

from typing import Any

import numpy as np


def default_input_adapter(frame: dict[str, Any], device: Any, torch: Any) -> dict[str, Any]:
    # This default adapter only passes metadata. Replace with your model-specific adapter.
    return {"frame": frame, "device": str(device)}


def default_output_adapter(raw_output: Any, frame: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(raw_output, dict) and "detections" in raw_output:
        return _validate_detection_list(raw_output["detections"])
    if isinstance(raw_output, list):
        if raw_output and isinstance(raw_output[0], dict) and "x" in raw_output[0]:
            return _validate_detection_list(raw_output)
        # Handle MMDet3D common output pattern [Det3DDataSample]
        if raw_output and hasattr(raw_output[0], "pred_instances_3d"):
            return _mmdet3d_like_to_dets(raw_output[0].pred_instances_3d)
    if hasattr(raw_output, "pred_instances_3d"):
        return _mmdet3d_like_to_dets(raw_output.pred_instances_3d)
    raise ValueError(
        "Output adapter could not parse raw model output. Provide detector.output_adapter in runtime YAML."
    )


def _mmdet3d_like_to_dets(pred_instances_3d: Any) -> list[dict[str, Any]]:
    bboxes = None
    if hasattr(pred_instances_3d, "bboxes_3d"):
        b = pred_instances_3d.bboxes_3d
        if hasattr(b, "tensor"):
            bboxes = b.tensor
        else:
            bboxes = b

    scores = getattr(pred_instances_3d, "scores_3d", None)
    labels = getattr(pred_instances_3d, "labels_3d", None)
    if bboxes is None or scores is None or labels is None:
        raise ValueError("MMDet3D-like output missing bboxes_3d/scores_3d/labels_3d")

    to_np = lambda x: x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    b_np = to_np(bboxes)
    s_np = to_np(scores)
    l_np = to_np(labels)

    out: list[dict[str, Any]] = []
    for i in range(len(b_np)):
        bb = b_np[i]
        if bb.shape[0] < 7:
            continue
        out.append(
            {
                "x": float(bb[0]),
                "y": float(bb[1]),
                "z": float(bb[2]),
                "l": float(bb[3]),
                "w": float(bb[4]),
                "h": float(bb[5]),
                "yaw": float(bb[6]),
                "score": float(s_np[i]),
                "label": int(l_np[i]),
            }
        )
    return _validate_detection_list(out)


def _validate_detection_list(dets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = {"x", "y", "z", "yaw", "l", "w", "h", "score", "label"}
    for i, d in enumerate(dets):
        missing = required - set(d.keys())
        if missing:
            raise ValueError(f"Detection at index {i} is missing fields: {sorted(missing)}")
    return dets
