from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cam3d_tracker.config import load_config
from cam3d_tracker.math_utils import wrap_angle
from cam3d_tracker.models import Detection3D
from cam3d_tracker.tracker import Classical3DTracker

from .model_runtime import DetectorRuntime
from .nuscenes_provider import load_nuscenes_frames
from .math3d import ego_to_global_xyzyaw


def run_nuscenes_tracking(runtime_cfg: dict[str, Any]) -> None:
    ncfg = runtime_cfg["nuscenes"]
    dcfg = runtime_cfg["detector"]
    tcfg = runtime_cfg["tracker"]
    ocfg = runtime_cfg["output"]

    tracker_cfg = load_config(tcfg["config_path"]).raw
    tracker = Classical3DTracker(tracker_cfg)

    runtime = DetectorRuntime(
        model_class_path=dcfg["model_class"],
        checkpoint_path=dcfg["checkpoint_path"],
        model_kwargs=dcfg.get("model_kwargs", {}),
        device=dcfg.get("device", "cpu"),
        input_adapter_path=dcfg.get("input_adapter", "cam3d_tracker.nuscenes_runtime.adapters:default_input_adapter"),
        output_adapter_path=dcfg.get("output_adapter", "cam3d_tracker.nuscenes_runtime.adapters:default_output_adapter"),
    )

    frames = load_nuscenes_frames(
        dataroot=ncfg["dataroot"],
        version=ncfg.get("version", "v1.0-trainval"),
        split=ncfg.get("split", "val"),
        max_frames=ncfg.get("max_frames"),
        camera_order=ncfg.get("camera_order"),
    )

    label_map = dcfg.get("label_map", {})
    assume_ego_frame = bool(dcfg.get("detections_in_ego_frame", True))
    to_global = bool(dcfg.get("transform_ego_to_global", True))
    min_score = float(dcfg.get("min_score", 0.0))

    all_rows: list[dict[str, Any]] = []
    for frame in frames:
        det_rows = runtime.infer(frame)
        dets: list[Detection3D] = []
        for d in det_rows:
            score = float(d["score"])
            if score < min_score:
                continue

            label = d["label"]
            if isinstance(label, int):
                label = label_map.get(str(label), label_map.get(int(label), str(label)))
            label = str(label)

            x = float(d["x"])
            y = float(d["y"])
            z = float(d["z"])
            yaw = wrap_angle(float(d["yaw"]))
            if assume_ego_frame and to_global:
                if not frame.get("ego_pose"):
                    raise ValueError("Missing ego pose in frame; cannot transform ego->global")
                x, y, z, yaw = ego_to_global_xyzyaw(x, y, z, yaw, frame["ego_pose"])

            dets.append(
                Detection3D(
                    x=x,
                    y=y,
                    z=z,
                    yaw=yaw,
                    l=float(d["l"]),
                    w=float(d["w"]),
                    h=float(d["h"]),
                    score=score,
                    label=label,
                    raw=d,
                )
            )

        outs = tracker.step(frame["timestamp_s"], dets)
        for out in outs:
            row = out.to_dict()
            row["timestamp_s"] = frame["timestamp_s"]
            row["sample_token"] = frame["sample_token"]
            all_rows.append(row)

    out_path = Path(ocfg["path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"tracks": all_rows}, f, indent=2)
