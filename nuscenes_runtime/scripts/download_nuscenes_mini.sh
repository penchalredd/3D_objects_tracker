#!/usr/bin/env bash
set -euo pipefail

# Download and extract nuScenes mini data.
# Usage:
#   bash nuscenes_runtime/scripts/download_nuscenes_mini.sh /path/to/nuscenes
# Optional env overrides:
#   NUSC_MINI_URL
#   NUSC_MAPS_URL
#   NUSC_CANBUS_URL

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <NUSCENES_ROOT_DIR>"
  exit 1
fi

ROOT_DIR="$1"
mkdir -p "$ROOT_DIR"
ROOT_DIR="$(cd "$ROOT_DIR" && pwd)"

# Official nuScenes endpoints (can be overridden).
NUSC_MINI_URL="${NUSC_MINI_URL:-https://www.nuscenes.org/data/v1.0-mini.tgz}"
NUSC_MAPS_URL="${NUSC_MAPS_URL:-https://www.nuscenes.org/data/maps.tgz}"
NUSC_CANBUS_URL="${NUSC_CANBUS_URL:-https://www.nuscenes.org/data/can_bus.zip}"

TMP_DIR="$ROOT_DIR/.downloads"
mkdir -p "$TMP_DIR"

MINI_TGZ="$TMP_DIR/v1.0-mini.tgz"
MAPS_TGZ="$TMP_DIR/maps.tgz"
CANBUS_ZIP="$TMP_DIR/can_bus.zip"

download() {
  local url="$1"
  local out="$2"
  echo "Downloading: $url"
  curl -fL "$url" -o "$out"
}

extract_tgz() {
  local file="$1"
  local dst="$2"
  echo "Extracting: $file"
  tar -xzf "$file" -C "$dst"
}

extract_zip() {
  local file="$1"
  local dst="$2"
  echo "Extracting: $file"
  unzip -o "$file" -d "$dst" >/dev/null
}

if [ ! -d "$ROOT_DIR/samples" ] || [ ! -d "$ROOT_DIR/v1.0-mini" ]; then
  download "$NUSC_MINI_URL" "$MINI_TGZ"
  extract_tgz "$MINI_TGZ" "$ROOT_DIR"
else
  echo "Skipping mini download: existing samples/ and v1.0-mini/ found"
fi

if [ ! -d "$ROOT_DIR/maps" ]; then
  download "$NUSC_MAPS_URL" "$MAPS_TGZ"
  extract_tgz "$MAPS_TGZ" "$ROOT_DIR"
else
  echo "Skipping maps download: existing maps/ found"
fi

if [ ! -d "$ROOT_DIR/can_bus" ]; then
  if download "$NUSC_CANBUS_URL" "$CANBUS_ZIP"; then
    if ! extract_zip "$CANBUS_ZIP" "$ROOT_DIR"; then
      echo "Warning: failed to extract can_bus.zip, skipping CAN bus package."
      rm -f "$CANBUS_ZIP"
    fi
  else
    echo "Warning: failed to download can_bus.zip, skipping CAN bus package."
  fi
else
  echo "Skipping can_bus download: existing can_bus/ found"
fi

echo

echo "nuScenes mini setup complete"
echo "Root: $ROOT_DIR"
echo "Contains:"
ls -1 "$ROOT_DIR" | sed 's/^/  - /'
