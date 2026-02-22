#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ordered sample-token trace for one nuScenes scene")
    p.add_argument("--dataroot", required=True, help="Path to nuScenes root")
    p.add_argument("--version", default="v1.0-mini", help="nuScenes version")
    p.add_argument("--scene-name", required=True, help="Scene name, e.g., scene-0103")
    p.add_argument("--out-tokens", required=True, help="Output JSON path for ordered sample tokens")
    p.add_argument("--out-timestamps", required=True, help="Output JSON path for token->timestamp_s map")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError as exc:
        raise SystemExit("nuscenes-devkit is required: pip install nuscenes-devkit") from exc

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    scene = None
    for s in nusc.scene:
        if s["name"] == args.scene_name:
            scene = s
            break
    if scene is None:
        raise SystemExit(f"Scene '{args.scene_name}' not found in {args.version}")

    sample_tokens: list[str] = []
    token_to_ts: dict[str, float] = {}

    tok = scene["first_sample_token"]
    while tok:
        sample = nusc.get("sample", tok)
        sample_tokens.append(tok)
        token_to_ts[tok] = float(sample["timestamp"]) * 1e-6
        tok = sample["next"]

    tokens_payload = {
        "version": args.version,
        "scene_name": args.scene_name,
        "sample_tokens": sample_tokens,
    }

    Path(args.out_tokens).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_timestamps).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_tokens, "w", encoding="utf-8") as f:
        json.dump(tokens_payload, f, indent=2)
    with open(args.out_timestamps, "w", encoding="utf-8") as f:
        json.dump(token_to_ts, f, indent=2)

    print(f"Wrote {len(sample_tokens)} ordered tokens for {args.scene_name}")
    print(f"tokens: {args.out_tokens}")
    print(f"timestamps: {args.out_timestamps}")


if __name__ == "__main__":
    main()
