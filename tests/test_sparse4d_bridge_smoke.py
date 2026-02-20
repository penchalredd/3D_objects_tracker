from cam3d_tracker.nuscenes_runtime.config import load_runtime_config
from cam3d_tracker.nuscenes_runtime.sparse4d_bridge import run_sparse4d_to_tracker


def test_sparse4d_bridge_local_dryrun():
    cfg = load_runtime_config(
        "nuscenes_runtime/configs/sparse4d_detection_to_tracker_localtest.yaml"
    )
    run_sparse4d_to_tracker(cfg)

    tracks_path = "outputs/localtest_sparse4d_tracks.json"
    with open(tracks_path, "r", encoding="utf-8") as f:
        txt = f.read()

    assert "tracks" in txt
    assert "track_id" in txt
