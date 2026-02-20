#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
THIRD_PARTY_DIR="$ROOT_DIR/nuscenes_runtime/third_party"
CKPT_DIR="$ROOT_DIR/nuscenes_runtime/checkpoints"

SPARSE4D_REPO="https://github.com/HorizonRobotics/Sparse4D.git"
SPARSE4D_DIR="$THIRD_PARTY_DIR/Sparse4D"
SPARSE4D_CKPT_URL="https://github.com/HorizonRobotics/Sparse4D/releases/download/v3.0/sparse4dv3_r50.pth"
SPARSE4D_CKPT_PATH="$CKPT_DIR/sparse4dv3_r50.pth"

mkdir -p "$THIRD_PARTY_DIR" "$CKPT_DIR"

if [ ! -d "$SPARSE4D_DIR/.git" ]; then
  git clone --depth 1 "$SPARSE4D_REPO" "$SPARSE4D_DIR"
else
  echo "Sparse4D repo already exists: $SPARSE4D_DIR"
fi

if [ ! -f "$SPARSE4D_CKPT_PATH" ] || [ "$(stat -f%z "$SPARSE4D_CKPT_PATH" 2>/dev/null || echo 0)" -lt 100000000 ]; then
  curl -L "$SPARSE4D_CKPT_URL" -o "$SPARSE4D_CKPT_PATH"
fi

if file "$SPARSE4D_CKPT_PATH" | grep -q "XML"; then
  echo "Downloaded file is not a valid checkpoint (XML error page)." >&2
  exit 1
fi

echo "Sparse4D assets ready"
echo "Repo: $SPARSE4D_DIR"
echo "Checkpoint: $SPARSE4D_CKPT_PATH"
