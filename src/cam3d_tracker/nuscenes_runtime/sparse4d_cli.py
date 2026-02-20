from __future__ import annotations

import argparse

from .config import load_runtime_config
from .sparse4d_bridge import run_sparse4d_to_tracker


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Sparse4D detection-only and feed classical 3D tracker")
    p.add_argument("--config", required=True, help="Sparse4D bridge runtime YAML")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_runtime_config(args.config)
    run_sparse4d_to_tracker(cfg)


if __name__ == "__main__":
    main()
