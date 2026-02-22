#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for local machine with NVIDIA runtime.
# Usage:
#   bash docker/run_container.sh /absolute/path/to/nuscenes scene-0103

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <NUSCENES_ROOT> [SCENE_NAME]"
  exit 1
fi

NUSC_ROOT="$1"
SCENE_NAME="${2:-scene-0103}"
SPARSE4D_RUN_INFERENCE="${SPARSE4D_RUN_INFERENCE:-0}"
INSTALL_SPARSE4D_DEPS="${INSTALL_SPARSE4D_DEPS:-$SPARSE4D_RUN_INFERENCE}"

GPU_ARGS=""
if docker info --format '{{json .Runtimes}}' | grep -qi nvidia; then
  GPU_ARGS="--gpus all"
fi

docker build \
  --build-arg INSTALL_SPARSE4D_DEPS="$INSTALL_SPARSE4D_DEPS" \
  -t cam3d-tracker:latest .

DOCKER_TTY_ARGS="-i"
if [ -t 0 ] && [ -t 1 ]; then
  DOCKER_TTY_ARGS="-it"
fi

docker run --rm $DOCKER_TTY_ARGS $GPU_ARGS \
  -e NUSCENES_ROOT=/data/nuscenes \
  -e SCENE_NAME="$SCENE_NAME" \
  -e SPARSE4D_RUN_INFERENCE="$SPARSE4D_RUN_INFERENCE" \
  -v "$(pwd)":/workspace \
  -v "$NUSC_ROOT":/data/nuscenes \
  cam3d-tracker:latest \
  bash /workspace/docker/run_sparse4d_mini_trace.sh
