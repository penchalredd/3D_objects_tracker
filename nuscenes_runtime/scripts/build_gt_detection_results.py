#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nuscenes.nuscenes import NuScenes


CATEGORY_TO_DET = {
    "vehicle.car": "car",
    "vehicle.truck": "truck",
    "vehicle.trailer": "trailer",
    "vehicle.construction": "construction_vehicle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "human.pedestrian": "pedestrian",
}


def to_det_name(category_name: str) -> str | None:
    for prefix, det_name in CATEGORY_TO_DET.items():
        if category_name.startswith(prefix):
            return det_name
    return None


def build_results(nusc: NuScenes, scene_name: str) -> tuple[dict[str, list[dict]], dict[str, float], list[str]]:
    scene = next((s for s in nusc.scene if s["name"] == scene_name), None)
    if scene is None:
        raise ValueError(f"scene '{scene_name}' not found for {nusc.version}")

    results: dict[str, list[dict]] = {}
    token_to_ts_s: dict[str, float] = {}
    trace_tokens: list[str] = []

    sample_token = scene["first_sample_token"]
    while sample_token:
        sample = nusc.get("sample", sample_token)
        token = sample["token"]
        trace_tokens.append(token)
        token_to_ts_s[token] = float(sample["timestamp"]) * 1e-6
        annos: list[dict] = []

        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            det_name = to_det_name(ann["category_name"])
            if det_name is None:
                continue
            annos.append(
                {
                    "sample_token": token,
                    "translation": [float(x) for x in ann["translation"]],
                    "size": [float(x) for x in ann["size"]],
                    "rotation": [float(x) for x in ann["rotation"]],
                    "velocity": [0.0, 0.0],
                    "detection_name": det_name,
                    "detection_score": 0.99,
                    "attribute_name": "",
                }
            )

        results[token] = annos
        sample_token = sample["next"]

    return results, token_to_ts_s, trace_tokens


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build scene-ordered nuScenes detection JSON from mini GT (for tracker pipeline testing)."
    )
    p.add_argument("--dataroot", required=True)
    p.add_argument("--version", default="v1.0-mini")
    p.add_argument("--scene-name", required=True)
    p.add_argument("--out-results", required=True)
    p.add_argument("--out-tokens", required=True)
    p.add_argument("--out-timestamps", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_results = Path(args.out_results)
    out_tokens = Path(args.out_tokens)
    out_timestamps = Path(args.out_timestamps)
    out_results.parent.mkdir(parents=True, exist_ok=True)
    out_tokens.parent.mkdir(parents=True, exist_ok=True)
    out_timestamps.parent.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    results, token_to_ts, trace_tokens = build_results(nusc, args.scene_name)

    with out_results.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "use_camera": True,
                    "use_lidar": False,
                    "use_radar": False,
                    "use_map": False,
                    "use_external": False,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    with out_tokens.open("w", encoding="utf-8") as f:
        json.dump({"scene_name": args.scene_name, "sample_tokens": trace_tokens}, f, indent=2)
    with out_timestamps.open("w", encoding="utf-8") as f:
        json.dump(token_to_ts, f, indent=2)

    print(f"Scene trace length: {len(trace_tokens)}")
    print(f"Wrote: {out_results}")
    print(f"Wrote: {out_tokens}")
    print(f"Wrote: {out_timestamps}")


if __name__ == "__main__":
    main()
