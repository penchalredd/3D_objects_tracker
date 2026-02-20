from __future__ import annotations

import argparse

from .pipeline import run_tracking


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Classical camera-based 3D MOT")
    p.add_argument("--config", required=True, help="YAML config path")
    p.add_argument("--detections", required=True, help="Input detections JSON path")
    p.add_argument("--output", required=True, help="Output tracks JSON path")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_tracking(args.config, args.detections, args.output)


if __name__ == "__main__":
    main()
