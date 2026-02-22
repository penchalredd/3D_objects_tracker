from cam3d_tracker.nuscenes_runtime.config import load_runtime_config
from cam3d_tracker.nuscenes_runtime.sparse4d_bridge import run_sparse4d_to_tracker


def test_sparse4d_trace_filter_single_token():
    cfg = load_runtime_config(
        "nuscenes_runtime/configs/sparse4d_detection_to_tracker_localtest.yaml"
    )
    cfg["conversion"]["scene_tokens_json"] = "data/sample_scene_tokens.json"
    cfg["output"]["detections_path"] = "outputs/localtest_sparse4d_trace_detections.json"
    cfg["output"]["tracks_path"] = "outputs/localtest_sparse4d_trace_tracks.json"

    run_sparse4d_to_tracker(cfg)

    with open("outputs/localtest_sparse4d_trace_detections.json", "r", encoding="utf-8") as f:
        text = f.read()

    assert "sample_token_1" in text
    assert "sample_token_2" not in text
