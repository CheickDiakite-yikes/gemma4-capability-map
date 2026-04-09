from __future__ import annotations

from gemma4_capability_map.models.gemma4_runner import Gemma4Runner
from gemma4_capability_map.models.gemma4_runner import (
    _mlx_completion_tokens,
    _mlx_generated_text,
    _mlx_prompt_tokens,
    _parse_model_response,
)
from gemma4_capability_map.schemas import Message, ToolCall


def test_parse_model_response_strips_trailing_control_tokens() -> None:
    final_answer, thinking_text = _parse_model_response(None, "42<turn|>")
    assert final_answer == "42"
    assert thinking_text == ""


def test_parse_model_response_splits_thought_and_final_channels() -> None:
    raw = "<|channel>thought\nPlan first.\n<|channel>final\n42<turn|>"
    final_answer, thinking_text = _parse_model_response(None, raw)
    assert final_answer == "42"
    assert thinking_text == "Plan first."


def test_parse_model_response_handles_thought_only_outputs() -> None:
    raw = "<|channel>thought\nNeed more tokens to finish."
    final_answer, thinking_text = _parse_model_response(None, raw)
    assert final_answer == ""
    assert thinking_text == "Need more tokens to finish."


def test_parse_model_response_handles_gemma_thought_then_answer_format() -> None:
    raw = "<|channel>thought\nWork it out here.<channel|>42<turn|>"
    final_answer, thinking_text = _parse_model_response(None, raw)
    assert final_answer == "42"
    assert thinking_text == "Work it out here."


class _FakeProcessor:
    def __init__(self, payload) -> None:  # noqa: ANN001
        self.payload = payload

    def parse_response(self, text: str):  # noqa: ANN001
        return self.payload


def test_parse_model_response_does_not_fallback_to_raw_text_when_processor_returns_only_thinking() -> None:
    final_answer, thinking_text = _parse_model_response(_FakeProcessor({"thought": "Plan first."}), "raw fallback text")
    assert final_answer == ""
    assert thinking_text == "Plan first."


def test_parse_model_response_can_recover_final_answer_after_processor_only_returns_thinking() -> None:
    raw = "<|channel>thought\nPlan first.<channel|>42<turn|>"
    final_answer, thinking_text = _parse_model_response(_FakeProcessor({"thought": "Plan first."}), raw)
    assert final_answer == "42"
    assert thinking_text == "Plan first."


class _FakeGenerationResult:
    def __init__(self) -> None:
        self.text = "hello"
        self.prompt_tokens = 12
        self.generation_tokens = 5


def test_mlx_generation_helpers_handle_structured_result_objects() -> None:
    generation = _FakeGenerationResult()
    assert _mlx_generated_text(generation) == "hello"
    assert _mlx_prompt_tokens(generation, fallback=0) == 12
    assert _mlx_completion_tokens(generation, fallback=0) == 5


def test_hf_service_backend_uses_service_response(monkeypatch) -> None:
    monkeypatch.setattr(
        "gemma4_capability_map.models.gemma4_runner.ensure_hf_reasoner_service",
        lambda **kwargs: {
            "service_id": "svc-1",
            "status": "ready",
            "load_elapsed_ms": 123,
            "requests_completed": 7,
            "paths": {
                "socket_path": "/tmp/hf.sock",
                "state_path": "/tmp/hf.json",
                "event_log_path": "/tmp/hf.events.jsonl",
                "request_log_path": "/tmp/hf.jsonl",
                "stdout_log_path": "/tmp/hf.log",
            },
            "runtime_info": {
                "runtime_device": "mps",
                "load_mode": "vision",
                "effective_model": "local/gemma-4-E2B-it",
            },
        },
    )
    monkeypatch.setattr(
        "gemma4_capability_map.models.gemma4_runner.request_service",
        lambda socket_path, payload, timeout_seconds=600.0: {
            "ok": True,
            "request_id": "req-1",
            "elapsed_ms": 77,
            "service_id": "svc-1",
            "runtime_info": {"runtime_device": "mps"},
            "turn": {
                "raw_model_output": "Enable two-factor authentication",
                "normalized_tool_call": [],
                "final_answer": "Enable two-factor authentication",
                "thinking_text": "",
                "latency_ms": 0,
                "prompt_tokens": 11,
                "completion_tokens": 5,
                "runtime_metadata": {"source": "service"},
            },
        },
    )
    runner = Gemma4Runner("google/gemma-4-E2B-it", backend="hf_service", device="mps")
    turn = runner.generate(
        messages=[Message(role="user", content="Look at the screenshot and tell me the settings change needed.")],
        media=[],
        tool_specs=[],
        thinking=False,
        max_new_tokens=32,
    )

    assert turn.final_answer == "Enable two-factor authentication"
    assert turn.runtime_metadata["service_request_id"] == "req-1"
    assert turn.runtime_metadata["service_elapsed_ms"] == 77
    assert runner.runtime_info()["service"]["service_id"] == "svc-1"
    assert runner.runtime_info()["service"]["event_log_path"] == "/tmp/hf.events.jsonl"


def test_oracle_single_tool_call_returns_normalized_call() -> None:
    runner = Gemma4Runner("google/gemma-4-E2B-it", backend="oracle")
    turn = runner.generate(
        messages=[
            Message(
                role="system",
                content='ORACLE_NEXT_TOOL_CALL:{"tool_name":"read_repo_file","arguments":{"path":"config/settings.yaml"}}',
            )
        ],
        media=[],
        tool_specs=[],
        thinking=False,
    )

    assert turn.normalized_tool_call == [
        ToolCall(
            name="read_repo_file",
            arguments={"path": "config/settings.yaml"},
            source_format="oracle",
            raw='{"name": "read_repo_file", "arguments": {"path": "config/settings.yaml"}}',
        )
    ]
