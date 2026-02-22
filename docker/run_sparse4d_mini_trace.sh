#!/usr/bin/env bash
set -euo pipefail

# Runs full pipeline inside container:
# 1) download mini data (if not present)
# 2) fetch Sparse4D repo + checkpoint
# 3) prepare ordered scene trace
# 4) run Sparse4D detection-only -> classical tracker
#
# Required env vars:
#   NUSCENES_ROOT=/data/nuscenes
# Optional env vars:
#   SCENE_NAME=scene-0103
#   PY_BIN=python
#   SPARSE4D_RUN_INFERENCE=0|1

ROOT_DIR="/workspace"
NUSCENES_ROOT="${NUSCENES_ROOT:-/data/nuscenes}"
SCENE_NAME="${SCENE_NAME:-scene-0103}"
PY_BIN="${PY_BIN:-python}"
SPARSE4D_RUN_INFERENCE="${SPARSE4D_RUN_INFERENCE:-0}"

if [ ! -d "$ROOT_DIR" ]; then
  echo "Missing /workspace mount"
  exit 1
fi

cd "$ROOT_DIR"

bash nuscenes_runtime/scripts/download_nuscenes_mini.sh "$NUSCENES_ROOT"

TRACE_DIR="$ROOT_DIR/outputs/scene_traces"
mkdir -p "$TRACE_DIR"
TRACE_TOKENS="$TRACE_DIR/${SCENE_NAME}_tokens.json"
TRACE_TS="$TRACE_DIR/${SCENE_NAME}_timestamps.json"
RESULTS_NUSC="$ROOT_DIR/outputs/sparse4d_detection/results_nusc.json"

if [ "$SPARSE4D_RUN_INFERENCE" = "1" ]; then
  bash nuscenes_runtime/scripts/fetch_sparse4d_assets.sh
  SPARSE4D_REPO="$ROOT_DIR/nuscenes_runtime/third_party/Sparse4D"

  bash nuscenes_runtime/scripts/prepare_nuscenes_mini_trace.sh \
    "$SPARSE4D_REPO" \
    "$NUSCENES_ROOT" \
    "$SCENE_NAME" \
    "$PY_BIN"

  TRACE_TOKENS="$SPARSE4D_REPO/data/scene_traces/${SCENE_NAME}_tokens.json"
  TRACE_TS="$SPARSE4D_REPO/data/scene_traces/${SCENE_NAME}_timestamps.json"
else
  # CPU-friendly local path: build scene-ordered detection JSON from mini GT.
  "$PY_BIN" nuscenes_runtime/scripts/build_gt_detection_results.py \
    --dataroot "$NUSCENES_ROOT" \
    --version v1.0-mini \
    --scene-name "$SCENE_NAME" \
    --out-results "$RESULTS_NUSC" \
    --out-tokens "$TRACE_TOKENS" \
    --out-timestamps "$TRACE_TS"
fi

RUNTIME_CFG="$ROOT_DIR/outputs/sparse4d_mini_trace.runtime.yaml"
mkdir -p "$ROOT_DIR/outputs"

cat > "$RUNTIME_CFG" << YAML
sparse4d:
  run_inference: $([ "$SPARSE4D_RUN_INFERENCE" = "1" ] && echo true || echo false)
  repo_root: ${SPARSE4D_REPO:-$ROOT_DIR/nuscenes_runtime/third_party/Sparse4D}
  python_exe: $PY_BIN
  config_path: projects/configs/sparse4dv3_temporal_r50_1x8_bs6_256x704.py
  checkpoint_path: $ROOT_DIR/nuscenes_runtime/checkpoints/sparse4dv3_r50.pth
  nuscenes_dataroot: $NUSCENES_ROOT
  ann_file: data/nuscenes_anno_pkls/nuscenes-mini_infos_val.pkl
  results_nusc_path: $RESULTS_NUSC
  cfg_options:
    data.test.tracking_threshold: 0.0

conversion:
  nuscenes_dataroot: $NUSCENES_ROOT
  nuscenes_version: v1.0-mini
  token_timestamps_json: $TRACE_TS
  scene_tokens_json: $TRACE_TOKENS
  scene_name: null
  allowed_labels: [car, truck, bus, trailer, construction_vehicle, pedestrian, motorcycle, bicycle]

tracker:
  config_path: $ROOT_DIR/configs/default.yaml

output:
  detections_path: $ROOT_DIR/outputs/sparse4d_mini_scene_detections.json
  tracks_path: $ROOT_DIR/outputs/sparse4d_mini_scene_tracks.json
YAML

$PY_BIN -m cam3d_tracker.nuscenes_runtime.sparse4d_cli --config "$RUNTIME_CFG"

echo "Done"
echo "Detections: $ROOT_DIR/outputs/sparse4d_mini_scene_detections.json"
echo "Tracks: $ROOT_DIR/outputs/sparse4d_mini_scene_tracks.json"
