from __future__ import annotations

import pytest

from gemma4_capability_map.models import hf_service


def test_ensure_hf_reasoner_service_reuses_running_ready_service_without_duplicate_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {
        "service_id": "svc",
        "root": "/tmp/svc",
        "socket_path": "/tmp/svc.sock",
        "state_path": "/tmp/state.json",
        "event_log_path": "/tmp/events.jsonl",
        "request_log_path": "/tmp/requests.jsonl",
        "stdout_log_path": "/tmp/service.log",
    }
    monkeypatch.setattr(hf_service, "service_paths_for", lambda model_id, device: paths)
    monkeypatch.setattr(hf_service, "_service_ready", lambda value: None)
    monkeypatch.setattr(hf_service, "read_service_state", lambda path: {"status": "ready", "pid": 4242})
    monkeypatch.setattr(hf_service, "_pid_is_running", lambda pid: True)
    monkeypatch.setattr(hf_service, "_wait_for_existing_ready", lambda value, timeout_seconds: {"ok": True, "paths": value})

    def _unexpected_popen(*args, **kwargs):
        raise AssertionError("duplicate service launch should not happen")

    monkeypatch.setattr(hf_service.subprocess, "Popen", _unexpected_popen)

    result = hf_service.ensure_hf_reasoner_service("google/gemma-4-E4B-it", "auto", 96)

    assert result["ok"] is True
    assert result["paths"] == paths


def test_ensure_hf_reasoner_service_refuses_duplicate_launch_for_unreachable_ready_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = {
        "service_id": "svc",
        "root": "/tmp/svc",
        "socket_path": "/tmp/svc.sock",
        "state_path": "/tmp/state.json",
        "event_log_path": "/tmp/events.jsonl",
        "request_log_path": "/tmp/requests.jsonl",
        "stdout_log_path": "/tmp/service.log",
    }
    monkeypatch.setattr(hf_service, "service_paths_for", lambda model_id, device: paths)
    monkeypatch.setattr(hf_service, "_service_ready", lambda value: None)
    monkeypatch.setattr(hf_service, "read_service_state", lambda path: {"status": "ready", "pid": 4242})
    monkeypatch.setattr(hf_service, "_pid_is_running", lambda pid: True)
    monkeypatch.setattr(hf_service, "_wait_for_existing_ready", lambda value, timeout_seconds: None)

    def _unexpected_popen(*args, **kwargs):
        raise AssertionError("duplicate service launch should not happen")

    monkeypatch.setattr(hf_service.subprocess, "Popen", _unexpected_popen)

    with pytest.raises(TimeoutError, match="refusing to launch a duplicate process"):
        hf_service.ensure_hf_reasoner_service("google/gemma-4-E4B-it", "auto", 96, startup_timeout_seconds=1.0)
