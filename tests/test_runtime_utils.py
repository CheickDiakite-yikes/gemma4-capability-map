from __future__ import annotations

import signal
import subprocess

from gemma4_capability_map.models.runtime_utils import (
    is_offline_mode_enabled,
    probe_llama_cpp_runtime,
    probe_mlx_runtime,
    recommended_local_reasoner_backend,
    recommended_reasoner_backend_from_probe,
    resolve_mlx_model_id,
    resolve_model_source,
    should_use_local_files_only,
)


def test_resolve_mlx_model_id_maps_known_gemma4_edges() -> None:
    assert resolve_mlx_model_id("google/gemma-4-E2B-it") == "mlx-community/gemma-4-e2b-it-4bit"
    assert resolve_mlx_model_id("google/gemma-4-E4B-it") == "mlx-community/gemma-4-e4b-it-4bit"


def test_resolve_mlx_model_id_leaves_explicit_repo_unchanged() -> None:
    explicit = "mlx-community/gemma-4-e2b-it-4bit"
    assert resolve_mlx_model_id(explicit) == explicit


def test_resolve_model_source_uses_local_env_override(tmp_path, monkeypatch) -> None:
    local_model = tmp_path / "gemma-4-E2B-it"
    local_model.mkdir()
    monkeypatch.setenv("GEMMA4_E2B_PATH", str(local_model))
    assert resolve_model_source("google/gemma-4-E2B-it") == str(local_model.resolve())


def test_resolve_model_source_uses_qwen_env_override(tmp_path, monkeypatch) -> None:
    local_model = tmp_path / "Qwen3-8B"
    local_model.mkdir()
    monkeypatch.setenv("QWEN3_8B_PATH", str(local_model))
    assert resolve_model_source("Qwen/Qwen3-8B") == str(local_model.resolve())


def test_resolve_model_source_uses_qwen_mlx_env_override(tmp_path, monkeypatch) -> None:
    local_model = tmp_path / "Qwen3-8B-MLX-4bit"
    local_model.mkdir()
    monkeypatch.setenv("QWEN3_8B_MLX_PATH", str(local_model))
    assert resolve_model_source("Qwen/Qwen3-8B-MLX-4bit") == str(local_model.resolve())


def test_resolve_model_source_uses_gemma31b_gguf_env_override(tmp_path, monkeypatch) -> None:
    local_model = tmp_path / "gemma-4-31b-it.gguf"
    local_model.write_text("fake-gguf", encoding="utf-8")
    monkeypatch.setenv("GEMMA4_31B_GGUF_PATH", str(local_model))
    assert resolve_model_source("google/gemma-4-31b-it") == str(local_model.resolve())


def test_resolve_model_source_uses_local_root(tmp_path, monkeypatch) -> None:
    local_model = tmp_path / "gemma-4-E4B-it"
    local_model.mkdir()
    monkeypatch.setenv("GEMMA_MODEL_ROOT", str(tmp_path))
    assert resolve_model_source("google/gemma-4-E4B-it") == str(local_model.resolve())


def test_resolve_model_source_uses_generic_derived_env_override(tmp_path, monkeypatch) -> None:
    local_model = tmp_path / "Qwen3-32B"
    local_model.mkdir()
    monkeypatch.setenv("LOCAL_MODEL_QWEN_QWEN3_32B_PATH", str(local_model))
    assert resolve_model_source("Qwen/Qwen3-32B") == str(local_model.resolve())


def test_resolve_model_source_uses_generic_model_root(tmp_path, monkeypatch) -> None:
    local_model = tmp_path / "Qwen3-32B"
    local_model.mkdir()
    monkeypatch.setenv("LOCAL_MODEL_ROOT", str(tmp_path))
    assert resolve_model_source("Qwen/Qwen3-32B") == str(local_model.resolve())


def test_offline_mode_flag_and_local_files_only(monkeypatch) -> None:
    monkeypatch.setenv("GEMMA4_OFFLINE", "1")
    assert is_offline_mode_enabled() is True
    assert should_use_local_files_only("google/gemma-4-E2B-it") is True


def test_probe_mlx_runtime_parses_success_payload(monkeypatch) -> None:
    class FakePopen:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            self.returncode = 0
            self.pid = 1234

        def communicate(self, timeout=None):  # noqa: ANN001
            return ('{"ok": true}\n', "")

    monkeypatch.setattr("gemma4_capability_map.models.runtime_utils.subprocess.Popen", FakePopen)
    probe = probe_mlx_runtime()
    assert probe["ok"] is True
    assert probe["returncode"] == 0


def test_probe_llama_cpp_runtime_parses_success_payload(monkeypatch) -> None:
    class FakePopen:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            self.returncode = 0
            self.pid = 5678

        def communicate(self, timeout=None):  # noqa: ANN001
            return ('{"ok": true, "runtime": "llama_cpp"}\n', "")

    monkeypatch.setattr("gemma4_capability_map.models.runtime_utils.subprocess.Popen", FakePopen)
    probe = probe_llama_cpp_runtime()
    assert probe["ok"] is True
    assert probe["runtime"] == "llama_cpp"
    assert probe["returncode"] == 0


def test_probe_mlx_runtime_reports_failures(monkeypatch) -> None:
    class FakePopen:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            self.returncode = 1
            self.pid = 1234

        def communicate(self, timeout=None):  # noqa: ANN001
            return ("", "mlx crash")

    monkeypatch.setattr("gemma4_capability_map.models.runtime_utils.subprocess.Popen", FakePopen)
    probe = probe_mlx_runtime()
    assert probe["ok"] is False
    assert probe["stderr"] == "mlx crash"


def test_probe_mlx_runtime_reports_timeouts(monkeypatch) -> None:
    class FakePopen:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            self.returncode = None
            self.pid = 4321
            self.calls = 0

        def communicate(self, timeout=None):  # noqa: ANN001
            self.calls += 1
            if self.calls == 1:
                error = subprocess.TimeoutExpired(cmd=["python"], timeout=20, output="")
                error.stderr = "stalled"
                raise error
            self.returncode = -9
            return ("", "stalled")

    monkeypatch.setattr("gemma4_capability_map.models.runtime_utils.subprocess.Popen", FakePopen)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr("gemma4_capability_map.models.runtime_utils.os.killpg", lambda pid, sig: killed.append((pid, sig)))
    probe = probe_mlx_runtime()
    assert probe["ok"] is False
    assert probe["error"] == "timeout"
    assert probe["timeout_seconds"] == 20
    assert killed == [(4321, signal.SIGKILL)]


def test_recommended_local_reasoner_backend_prefers_hf_when_mlx_probe_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        "gemma4_capability_map.models.runtime_utils.probe_mlx_runtime",
        lambda timeout_seconds=20: {"ok": False, "returncode": 1},
    )
    assert recommended_local_reasoner_backend() == "hf"


def test_recommended_reasoner_backend_from_probe_prefers_mlx_on_success() -> None:
    assert recommended_reasoner_backend_from_probe({"ok": True}) == "mlx"
