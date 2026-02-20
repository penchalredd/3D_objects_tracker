from cam3d_tracker.pipeline import run_tracking


def test_pipeline_smoke(tmp_path):
    out = tmp_path / "tracks.json"
    run_tracking("configs/default.yaml", "data/sample_detections.json", str(out))
    text = out.read_text(encoding="utf-8")
    assert "track_id" in text
    assert "confirmed" in text
