from __future__ import annotations

from typing import Any

from .math3d import quat_to_yaw

CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def load_nuscenes_frames(
    dataroot: str,
    version: str,
    split: str,
    max_frames: int | None,
    camera_order: list[str] | None = None,
) -> list[dict[str, Any]]:
    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.splits import create_splits_scenes
    except ImportError as exc:
        raise RuntimeError("nuscenes-devkit is required. Install with: pip install nuscenes-devkit") from exc

    cams = camera_order or CAMERAS
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    split_scenes = set(create_splits_scenes().get(split, []))
    if not split_scenes:
        raise ValueError(f"Unknown split '{split}'. Use one of train/val/test/... from nuscenes-devkit")

    frames: list[dict[str, Any]] = []
    for scene in nusc.scene:
        if scene["name"] not in split_scenes:
            continue
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)
            cam_paths: dict[str, str] = {}
            pose_info = None
            for cam in cams:
                sd_token = sample["data"][cam]
                sd = nusc.get("sample_data", sd_token)
                cam_paths[cam] = nusc.get_sample_data_path(sd_token)
                if cam == "CAM_FRONT":
                    pose = nusc.get("ego_pose", sd["ego_pose_token"])
                    pose_info = {
                        "translation": pose["translation"],
                        "rotation": pose["rotation"],
                        "yaw": quat_to_yaw(pose["rotation"]),
                    }

            frames.append(
                {
                    "sample_token": sample_token,
                    "timestamp_s": float(sample["timestamp"]) * 1e-6,
                    "camera_paths": cam_paths,
                    "ego_pose": pose_info,
                }
            )

            if max_frames is not None and len(frames) >= max_frames:
                return frames
            sample_token = sample["next"]

    return frames
