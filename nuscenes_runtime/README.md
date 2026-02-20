# nuScenes Runtime Folder

This folder is for real detector-to-tracker usage with a nuScenes-trained checkpoint.

## What it does

1. Loads nuScenes frames (multi-camera) from `dataroot`
2. Loads your model + checkpoint
3. Runs detector inference per frame
4. Converts detections to tracker schema
5. Runs classical 3D tracker and writes tracked output JSON

## Install extra dependencies

```bash
source /Users/bhumireddypenchalareddy/Documents/3d_tracker/.venv/bin/activate
pip install torch nuscenes-devkit
```

## Configure runtime

Edit:

`/Users/bhumireddypenchalareddy/Documents/3d_tracker/nuscenes_runtime/configs/runtime_example.yaml`

Critical fields:

- `detector.model_class`: import path to your model class (`module:Class`)
- `detector.checkpoint_path`: checkpoint file
- `detector.input_adapter`: function to build model input
- `detector.output_adapter`: function to convert raw output to detection schema

## Detection schema required by tracker

Each detection must contain:

- `x`, `y`, `z`
- `yaw`
- `l`, `w`, `h`
- `score`
- `label`

## Run

```bash
source /Users/bhumireddypenchalareddy/Documents/3d_tracker/.venv/bin/activate
track3d-nuscenes --runtime-config /Users/bhumireddypenchalareddy/Documents/3d_tracker/nuscenes_runtime/configs/runtime_example.yaml
```

Output default path:

`/Users/bhumireddypenchalareddy/Documents/3d_tracker/outputs/nuscenes_tracks.json`

## Notes

- If your detector outputs in ego frame, keep:
  - `detections_in_ego_frame: true`
  - `transform_ego_to_global: true`
- If your detector already outputs global frame, disable transform.

## Sparse4D Detection-Only -> Classical Tracker

Use this when you want Sparse4D only as a detector and keep tracking classical.

Config:

`/Users/bhumireddypenchalareddy/Documents/3d_tracker/nuscenes_runtime/configs/sparse4d_detection_to_tracker.yaml`

Run:

```bash
source /Users/bhumireddypenchalareddy/Documents/3d_tracker/.venv/bin/activate
track3d-sparse4d --config /Users/bhumireddypenchalareddy/Documents/3d_tracker/nuscenes_runtime/configs/sparse4d_detection_to_tracker.yaml
```

Local dry-run test (no Sparse4D inference):

```bash
track3d-sparse4d --config /Users/bhumireddypenchalareddy/Documents/3d_tracker/nuscenes_runtime/configs/sparse4d_detection_to_tracker_localtest.yaml
```
