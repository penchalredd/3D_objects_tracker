from __future__ import annotations

import argparse

from .config import load_runtime_config
from .pipeline import run_nuscenes_tracking


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run detector checkpoint on nuScenes and track outputs")
    p.add_argument("--runtime-config", required=True, help="YAML runtime config path")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_runtime_config(args.runtime_config)
    run_nuscenes_tracking(cfg)


if __name__ == "__main__":
    main()
