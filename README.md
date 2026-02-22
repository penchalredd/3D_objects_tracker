# Camera-Only Classical 3D Tracker

A production-style baseline tracker for autonomous driving when you already have 3D detections from multi-view cameras (nuScenes-style).

## What this repo gives you

- Classical tracking-by-detection pipeline
- IMM-EKF state estimator (`CV` + `CTRV`)
- Gated global association (Hungarian)
- BEV IoU + yaw-aware cost
- Track lifecycle (`tentative`, `confirmed`, `lost`, delete)
- Simple JSON input/output for detector integration

## State and measurements

Track state:

`[x, y, z, v, yaw, yaw_rate, l, w, h]`

Detection measurement:

`[x, y, z, yaw, l, w, h]`

## Input format

`data/sample_detections.json` shows the schema:

```json
{
  "schema": "cam3d_detections_v1",
  "frames": [
    {
      "timestamp_s": 0.0,
      "detections": [
        {
          "x": 8.0,
          "y": 1.0,
          "z": 0.2,
          "yaw": 0.02,
          "l": 4.4,
          "w": 1.9,
          "h": 1.6,
          "score": 0.91,
          "label": "car"
        }
      ]
    }
  ]
}
```

Coordinates should be in one consistent frame (ego or global), and yaw in radians.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

## Run

```bash
track3d \
  --config configs/default.yaml \
  --detections data/sample_detections.json \
  --output outputs/tracks.json
```

## Output

Creates JSON:

```json
{
  "tracks": [
    {
      "timestamp_s": 1.0,
      "track_id": 1,
      "label": "car",
      "x": 10.2,
      "y": 1.1,
      "z": 0.2,
      "v": 2.1,
      "yaw": 0.04,
      "l": 4.4,
      "w": 1.9,
      "h": 1.6,
      "score": 0.89,
      "status": "confirmed"
    }
  ]
}
```

## Integration notes

- Keep detector output in world-consistent coordinates per frame.
- Tune `configs/default.yaml` by class for your detector noise.
- Start with current association and lifecycle params, then tighten gates once recall is stable.
- If dense scenes still cause ID switches, the next upgrade is JPDA/MHT-style association.

## Development

```bash
python -m pytest -q
```

## nuScenes Checkpoint Runtime

For real detector + checkpoint usage on nuScenes data, use:

`/Users/bhumireddypenchalareddy/Documents/3d_tracker/nuscenes_runtime`

Run with:

```bash
track3d-nuscenes --runtime-config /Users/bhumireddypenchalareddy/Documents/3d_tracker/nuscenes_runtime/configs/runtime_example.yaml
```

Sparse4D detection-only to classical tracker:

```bash
track3d-sparse4d --config /Users/bhumireddypenchalareddy/Documents/3d_tracker/nuscenes_runtime/configs/sparse4d_detection_to_tracker.yaml
```

To run on a strict nuScenes mini scene trace (ordered, no random frames), use:

`/Users/bhumireddypenchalareddy/Documents/3d_tracker/nuscenes_runtime/scripts/prepare_nuscenes_mini_trace.sh`
