from __future__ import annotations

from gemma4_capability_map.models.hf_service import read_service_state, service_id_for, service_paths_for


def test_service_id_for_sanitizes_model_and_device() -> None:
    assert service_id_for("google/gemma-4-E2B-it", "mps") == "google__gemma_4_E2B_it_mps"


def test_service_paths_for_returns_runtime_artifacts_under_results() -> None:
    paths = service_paths_for("google/gemma-4-E2B-it", "mps")
    assert paths["socket_path"].endswith("service.sock")
    assert paths["event_log_path"].endswith("events.jsonl")
    assert "results/runtime/hf_reasoner" in paths["socket_path"]
    assert paths["request_log_path"].endswith("requests.jsonl")


def test_read_service_state_returns_none_for_missing_file(tmp_path) -> None:
    assert read_service_state(tmp_path / "missing.json") is None
