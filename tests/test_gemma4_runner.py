from __future__ import annotations

import sys
import types

from gemma4_capability_map.models.gemma4_runner import Gemma4Runner
from gemma4_capability_map.models.gemma4_runner import (
    _mlx_completion_tokens,
    _mlx_generated_text,
    _mlx_prompt_tokens,
    _parse_model_response,
)
from gemma4_capability_map.schemas import Message, ToolCall, ToolSpec


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


class _FakeInputBatch(dict):
    def __init__(self, prompt_length: int) -> None:
        super().__init__(input_ids=_FakeTokenBatch(prompt_length))
        self.last_device = None

    def to(self, device):  # noqa: ANN001
        self.last_device = device
        return self


class _FakeTokenBatch:
    def __init__(self, length: int) -> None:
        self._length = length

    @property
    def shape(self):  # noqa: D401
        return (1, self._length)


class _FakeGeneratedBatch:
    def __init__(self, text: str, length: int = 3) -> None:
        self.text = text
        self._length = length

    @property
    def shape(self):  # noqa: D401
        return (1, self._length)

    def __getitem__(self, item):  # noqa: ANN001
        return self


class _FakeTextTokenizer:
    def __init__(self) -> None:
        self.chat_messages = None
        self.chat_kwargs = None
        self.prompt = None
        self.prompt_kwargs = None

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # noqa: ANN001
        return cls()

    def apply_chat_template(self, messages, **kwargs):  # noqa: ANN001
        self.chat_messages = messages
        self.chat_kwargs = kwargs
        return "QWEN_PROMPT"

    def __call__(self, prompt, return_tensors="pt"):  # noqa: ANN001
        self.prompt = prompt
        self.prompt_kwargs = {"return_tensors": return_tensors}
        return _FakeInputBatch(prompt_length=11)

    def decode(self, generated, skip_special_tokens=False):  # noqa: ANN001
        return generated.text


class _FakeVisionProcessor(_FakeTextTokenizer):
    def apply_chat_template(self, messages, **kwargs):  # noqa: ANN001
        self.chat_messages = messages
        self.chat_kwargs = kwargs
        return _FakeInputBatch(prompt_length=9)


class _FakeModel:
    def __init__(self, text: str) -> None:
        self.text = text
        self.kwargs = None
        self.device = None
        self.generate_calls = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # noqa: ANN001
        return cls(text=kwargs.pop("generated_text", "Qwen answer"))

    def to(self, device):  # noqa: ANN001
        self.device = device
        return self

    def generate(self, **kwargs):  # noqa: ANN001
        self.generate_calls.append(kwargs)
        return [_FakeGeneratedBatch(self.text)]


def _install_fake_hf_stack(monkeypatch, *, text_mode: bool, generated_text: str) -> tuple[_FakeTextTokenizer, _FakeModel]:
    tokenizer = _FakeTextTokenizer() if text_mode else _FakeVisionProcessor()
    model = _FakeModel(generated_text)
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer),
        AutoProcessor=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer),
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: model),
        AutoModelForImageTextToText=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: model),
    )
    fake_torch = types.SimpleNamespace(
        float16="float16",
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr("gemma4_capability_map.models.gemma4_runner.load_hf_auth_token", lambda: None)
    monkeypatch.setattr("gemma4_capability_map.models.gemma4_runner.resolve_model_source", lambda model_id: f"/models/{model_id.lower().replace('/', '-')}")
    monkeypatch.setattr("gemma4_capability_map.models.gemma4_runner.should_use_local_files_only", lambda resolved_source: True)
    return tokenizer, model


def test_hf_text_backend_uses_tokenizer_chat_template(monkeypatch) -> None:
    tokenizer, model = _install_fake_hf_stack(monkeypatch, text_mode=True, generated_text="Qwen answer")
    runner = Gemma4Runner("Qwen/Qwen3-8B", backend="hf", device="cpu")
    turn = runner.generate(
        messages=[Message(role="user", content="Summarize the latest note.")],
        media=[],
        tool_specs=[
            ToolSpec(
                name="read_repo_file",
                description="Read a repository file.",
                schema={"type": "object", "properties": {"path": {"type": "string"}}},
            )
        ],
        thinking=False,
        max_new_tokens=32,
    )

    assert runner.runtime_info()["load_mode"] == "text"
    assert runner.runtime_info()["effective_model"] == "/models/qwen-qwen3-8b"
    assert tokenizer.chat_messages is not None
    assert all(isinstance(message["content"], str) for message in tokenizer.chat_messages)
    assert tokenizer.chat_messages[0]["role"] == "system"
    assert "Available tools:" in tokenizer.chat_messages[1]["content"]
    assert tokenizer.prompt == "QWEN_PROMPT"
    assert model.generate_calls[0]["max_new_tokens"] == 32
    assert turn.final_answer == "Qwen answer"


def test_hf_vision_backend_still_uses_processor_chat_template(monkeypatch, tmp_path) -> None:
    tokenizer, model = _install_fake_hf_stack(monkeypatch, text_mode=False, generated_text="Vision answer")
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"fake")
    runner = Gemma4Runner("google/gemma-4-E2B-it", backend="hf", device="cpu")
    turn = runner.generate(
        messages=[Message(role="user", content="What do you see?")],
        media=[str(image_path)],
        tool_specs=[],
        thinking=False,
        max_new_tokens=24,
    )

    assert runner.runtime_info()["load_mode"] == "vision"
    assert runner.runtime_info()["effective_model"] == "/models/google-gemma-4-e2b-it"
    assert tokenizer.chat_messages is not None
    assert isinstance(tokenizer.chat_messages[0]["content"], list)
    assert any(part.get("type") == "image" for part in tokenizer.chat_messages[0]["content"])
    assert tokenizer.chat_kwargs["tokenize"] is True
    assert tokenizer.chat_kwargs["enable_thinking"] is False
    assert model.generate_calls[0]["max_new_tokens"] == 24
    assert turn.final_answer == "Vision answer"


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
