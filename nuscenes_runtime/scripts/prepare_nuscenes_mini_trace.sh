#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <SPARSE4D_REPO_ROOT> <NUSCENES_DATAROOT> <SCENE_NAME> [PYTHON_BIN]"
  exit 1
fi

SPARSE4D_REPO_ROOT="$1"
NUSCENES_DATAROOT="$2"
SCENE_NAME="$3"
PY_BIN="${4:-python}"

TRACE_DIR="${SPARSE4D_REPO_ROOT%/}/data/scene_traces"
PKL_DIR="${SPARSE4D_REPO_ROOT%/}/data/nuscenes_anno_pkls"

mkdir -p "$TRACE_DIR" "$PKL_DIR"

echo "[1/3] Building nuScenes mini annotation PKLs (if missing)"
if [ ! -f "$PKL_DIR/nuscenes-mini_infos_val.pkl" ]; then
  (
    cd "$SPARSE4D_REPO_ROOT"
    "$PY_BIN" tools/nuscenes_converter.py --version v1.0-mini --info_prefix data/nuscenes_anno_pkls/nuscenes-mini
  )
else
  echo "PKL exists: $PKL_DIR/nuscenes-mini_infos_val.pkl"
fi

echo "[2/3] Building ordered trace token/timestamp JSON for scene $SCENE_NAME"
"$PY_BIN" nuscenes_runtime/scripts/build_nuscenes_scene_trace.py \
  --dataroot "$NUSCENES_DATAROOT" \
  --version v1.0-mini \
  --scene-name "$SCENE_NAME" \
  --out-tokens "$TRACE_DIR/${SCENE_NAME}_tokens.json" \
  --out-timestamps "$TRACE_DIR/${SCENE_NAME}_timestamps.json"

echo "[3/3] Done"
echo "Use these in sparse4d_detection_to_tracker.yaml:"
echo "  sparse4d.ann_file: data/nuscenes_anno_pkls/nuscenes-mini_infos_val.pkl"
echo "  conversion.scene_tokens_json: $TRACE_DIR/${SCENE_NAME}_tokens.json"
echo "  conversion.token_timestamps_json: $TRACE_DIR/${SCENE_NAME}_timestamps.json"
