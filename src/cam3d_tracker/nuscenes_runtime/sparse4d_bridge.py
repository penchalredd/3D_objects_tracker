from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from cam3d_tracker.pipeline import run_tracking


DEFAULT_ALLOWED_LABELS = {
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
}


def run_sparse4d_to_tracker(cfg: dict[str, Any]) -> None:
    scfg = cfg["sparse4d"]
    ccfg = cfg["conversion"]
    tcfg = cfg["tracker"]
    ocfg = cfg["output"]

    if bool(scfg.get("run_inference", True)):
        _run_sparse4d_detection_only(scfg)

    token_to_ts = _load_token_timestamps(ccfg)
    allowed = set(ccfg.get("allowed_labels", sorted(DEFAULT_ALLOWED_LABELS)))

    detections_json = Path(ocfg["detections_path"])
    detections_json.parent.mkdir(parents=True, exist_ok=True)
    _convert_sparse4d_results_to_tracker_input(
        results_nusc_path=Path(scfg["results_nusc_path"]),
        token_to_timestamp_s=token_to_ts,
        output_path=detections_json,
        allowed_labels=allowed,
    )

    run_tracking(
        config_path=tcfg["config_path"],
        detections_path=str(detections_json),
        output_path=ocfg["tracks_path"],
    )


def _run_sparse4d_detection_only(scfg: dict[str, Any]) -> None:
    repo_root = Path(scfg["repo_root"]).resolve()
    results_json = Path(scfg["results_nusc_path"]).resolve()
    format_dir = results_json.parent
    format_dir.mkdir(parents=True, exist_ok=True)

    cfg_options = {
        "data_root": scfg["nuscenes_dataroot"],
        "data.test.data_root": scfg["nuscenes_dataroot"],
        "data.test.ann_file": scfg["ann_file"],
        "data.test.tracking": False,
    }
    cfg_options.update(scfg.get("cfg_options", {}))

    cmd = [
        scfg.get("python_exe", "python"),
        "tools/test.py",
        scfg["config_path"],
        scfg["checkpoint_path"],
        "--format-only",
        "--eval-options",
        f"jsonfile_prefix={format_dir}",
        "--cfg-options",
    ]
    for k, v in cfg_options.items():
        cmd.append(f"{k}={_to_mmcv_value(v)}")

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo_root}:{env.get('PYTHONPATH', '')}".rstrip(":")

    subprocess.run(cmd, cwd=repo_root, env=env, check=True)


def _to_mmcv_value(v: Any) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return "[" + ",".join(_to_mmcv_value(x) for x in v) + "]"
    return str(v)


def _load_token_timestamps(ccfg: dict[str, Any]) -> dict[str, float]:
    if ccfg.get("token_timestamps_json"):
        with open(ccfg["token_timestamps_json"], "r", encoding="utf-8") as f:
            m = json.load(f)
        return {str(k): float(v) for k, v in m.items()}

    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError as exc:
        raise RuntimeError(
            "nuscenes-devkit required for timestamp mapping. Install or provide conversion.token_timestamps_json"
        ) from exc

    nusc = NuScenes(
        version=ccfg.get("nuscenes_version", "v1.0-trainval"),
        dataroot=ccfg["nuscenes_dataroot"],
        verbose=False,
    )
    return {s["token"]: float(s["timestamp"]) * 1e-6 for s in nusc.sample}


def _quat_wxyz_to_yaw(q: list[float]) -> float:
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _convert_sparse4d_results_to_tracker_input(
    results_nusc_path: Path,
    token_to_timestamp_s: dict[str, float],
    output_path: Path,
    allowed_labels: set[str],
) -> None:
    with open(results_nusc_path, "r", encoding="utf-8") as f:
        nusc = json.load(f)

    by_token = nusc["results"]
    frames: list[dict[str, Any]] = []

    for token, annos in by_token.items():
        if token not in token_to_timestamp_s:
            continue
        dets = []
        for a in annos:
            name = a.get("detection_name")
            if not name:
                # Skip tracking-only fields; this bridge is detection-only.
                continue
            if allowed_labels and name not in allowed_labels:
                continue

            t = a["translation"]
            s = a["size"]  # nuScenes order: [w, l, h]
            q = a["rotation"]

            dets.append(
                {
                    "x": float(t[0]),
                    "y": float(t[1]),
                    "z": float(t[2]),
                    "yaw": float(_quat_wxyz_to_yaw(q)),
                    "l": float(s[1]),
                    "w": float(s[0]),
                    "h": float(s[2]),
                    "score": float(a["detection_score"]),
                    "label": str(name),
                }
            )

        frames.append(
            {
                "timestamp_s": token_to_timestamp_s[token],
                "sample_token": token,
                "detections": dets,
            }
        )

    frames.sort(key=lambda x: x["timestamp_s"])
    out = {"schema": "cam3d_detections_v1", "frames": frames}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
