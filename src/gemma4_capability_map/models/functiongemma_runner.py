from __future__ import annotations

import json
import re
import time
from typing import Any

from gemma4_capability_map.models.base import Runner
from gemma4_capability_map.models.runtime_utils import load_hf_auth_token, resolve_model_source, should_use_local_files_only
from gemma4_capability_map.schemas import Message, ModelTurn, ToolCall, ToolSpec
from gemma4_capability_map.tools.planner import plan_tool_calls, tool_catalog_text
from gemma4_capability_map.tools.validators import normalize_tool_output


_HF_ROUTER_CACHE: dict[tuple[str, str], tuple[object, object]] = {}


class FunctionGemmaRunner(Runner):
    def __init__(
        self,
        model_id: str,
        backend: str = "heuristic",
        max_new_tokens: int = 128,
        device: str = "auto",
    ) -> None:
        super().__init__(model_id=model_id, backend=backend)
        self.requested_model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.device = device
        self._tokenizer = None
        self._model = None
        self._runtime_device: str | None = None
        self._effective_model_id = model_id

    def ensure_loaded(self) -> dict[str, Any]:
        if self.backend == "hf":
            self._ensure_hf_loaded()
        return self.runtime_info()

    def runtime_info(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "requested_model": self.requested_model_id,
            "effective_model": self._effective_model_id,
            "runtime_device": self._runtime_device,
            "configured_device": self.device,
        }

    def generate(
        self,
        messages: list[Message],
        media: list[str],
        tool_specs: list[ToolSpec],
        thinking: bool,
        max_new_tokens: int | None = None,
    ) -> ModelTurn:
        start = time.perf_counter()
        active_max_new_tokens = max_new_tokens or self.max_new_tokens
        if self.backend == "oracle":
            turn = self._generate_oracle(messages)
        elif self.backend == "heuristic":
            turn = self._generate_heuristic(messages, media, tool_specs)
        elif self.backend == "hf":
            turn = self._generate_hf(messages, media, tool_specs, max_new_tokens=active_max_new_tokens)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
        return turn.model_copy(update={"latency_ms": int((time.perf_counter() - start) * 1000)})

    def _generate_oracle(self, messages: list[Message]) -> ModelTurn:
        oracle_message = next((message.content for message in messages if message.role == "system" and message.content.startswith("ORACLE_NEXT_TOOL_CALL:")), "")
        payload = json.loads(oracle_message.split(":", 1)[1]) if oracle_message else {"tool_name": "noop", "arguments": {}}
        if isinstance(payload, list):
            tool_calls = [
                ToolCall(
                    name=item["tool_name"],
                    arguments=item.get("arguments", {}),
                    source_format="functiongemma",
                    raw=self._format_function_call(item["tool_name"], item.get("arguments", {})),
                )
                for item in payload
            ]
            raw = "".join(call.raw for call in tool_calls)
            return ModelTurn(
                raw_model_output=raw,
                normalized_tool_call=tool_calls,
            )
        raw = self._format_function_call(payload["tool_name"], payload.get("arguments", {}))
        return ModelTurn(
            raw_model_output=raw,
            normalized_tool_call=[
                ToolCall(
                    name=payload["tool_name"],
                    arguments=payload.get("arguments", {}),
                    source_format="functiongemma",
                    raw=raw,
                )
            ],
        )

    def _generate_heuristic(self, messages: list[Message], media: list[str], tool_specs: list[ToolSpec]) -> ModelTurn:
        tool_calls = plan_tool_calls(messages, media=media, tool_specs=tool_specs)
        if not tool_calls:
            tool_calls = [ToolCall(name=tool_specs[0].name if tool_specs else "noop", arguments={}, source_format="heuristic", raw="")]
        raw = "".join(self._format_function_call(call.name, call.arguments) for call in tool_calls)
        return ModelTurn(
            raw_model_output=raw,
            normalized_tool_call=normalize_tool_output(raw),
        )

    def _generate_hf(
        self,
        messages: list[Message],
        media: list[str],
        tool_specs: list[ToolSpec],
        max_new_tokens: int,
    ) -> ModelTurn:
        self._ensure_hf_loaded()
        prompt_messages = self._build_prompt_messages(messages, media, tool_specs)
        prompt = self._render_prompt(prompt_messages)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if hasattr(inputs, "to") and self._runtime_device is not None:
            inputs = inputs.to(self._runtime_device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        text = self._tokenizer.decode(generated, skip_special_tokens=False)
        structured_text = _extract_structured_candidate(text)
        normalized = normalize_tool_output(structured_text or text)
        return ModelTurn(
            raw_model_output=structured_text or text,
            normalized_tool_call=normalized,
            prompt_tokens=int(inputs["input_ids"].shape[-1]),
            completion_tokens=int(generated.shape[-1]),
        )

    def _ensure_hf_loaded(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the hf extra to use the FunctionGemma HF backend.") from exc

        if self._tokenizer is not None and self._model is not None:
            return

        token = load_hf_auth_token()
        resolved_source = resolve_model_source(self.requested_model_id)
        local_files_only = should_use_local_files_only(resolved_source)
        runtime_device = _pick_hf_device(torch, preferred=self.device)
        cached = _HF_ROUTER_CACHE.get((resolved_source, runtime_device))
        if cached is None:
            tokenizer = AutoTokenizer.from_pretrained(
                resolved_source,
                token=token,
                local_files_only=local_files_only,
            )
            load_kwargs: dict[str, Any] = {
                "low_cpu_mem_usage": True,
                "token": token,
                "local_files_only": local_files_only,
            }
            if runtime_device == "cuda":
                load_kwargs["torch_dtype"] = "auto"
                load_kwargs["device_map"] = "auto"
                model = AutoModelForCausalLM.from_pretrained(resolved_source, **load_kwargs)
            else:
                load_kwargs["torch_dtype"] = torch.float16 if runtime_device == "mps" else "auto"
                model = AutoModelForCausalLM.from_pretrained(resolved_source, **load_kwargs)
                model.to(runtime_device)
            cached = (tokenizer, model)
            _HF_ROUTER_CACHE[(resolved_source, runtime_device)] = cached
        self._tokenizer, self._model = cached
        self._runtime_device = runtime_device
        self._effective_model_id = resolved_source
        self.model_id = self._effective_model_id

    def _build_prompt_messages(
        self,
        messages: list[Message],
        media: list[str],
        tool_specs: list[ToolSpec],
    ) -> list[dict[str, str]]:
        prompt_messages: list[dict[str, str]] = []
        system_sections = [
            "You are a function routing model.",
            "Choose only from the allowed tools.",
            "Return only function calls.",
            "Preferred format: <start_function_call>call:tool_name{arg:<escape>value<escape>}<end_function_call>.",
            'If multiple independent tools are needed, emit multiple function call blocks or a JSON array of {"name": "...", "arguments": {...}} objects.',
            "Never invent a tool or field name.",
        ]
        catalog = tool_catalog_text(tool_specs)
        if catalog:
            system_sections.append(catalog)
        prompt_messages.append({"role": "system", "content": "\n\n".join(system_sections)})
        for message in messages:
            role = "user" if message.role == "tool" else message.role
            text = message.content
            if message.role == "tool":
                text = f"Tool result:\n{text}"
            image_refs = list(message.image_refs)
            if role == "user" and not image_refs and message is messages[-1]:
                image_refs = list(media)
            if image_refs:
                text = (text + "\n" if text else "") + "Image refs: " + ", ".join(image_refs)
            prompt_messages.append({"role": role, "content": text or ""})
        return prompt_messages

    def _render_prompt(self, prompt_messages: list[dict[str, str]]) -> str:
        if hasattr(self._tokenizer, "apply_chat_template") and getattr(self._tokenizer, "chat_template", None):
            return self._tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        lines: list[str] = []
        for message in prompt_messages:
            lines.append(f"{message['role'].upper()}: {message['content']}")
        lines.append("ASSISTANT:")
        return "\n\n".join(lines)

    def _format_function_call(self, tool_name: str, arguments: dict[str, object]) -> str:
        serialized_parts = []
        for key, value in arguments.items():
            if isinstance(value, str):
                serialized_parts.append(f"{key}:<escape>{value}<escape>")
            else:
                serialized_parts.append(f"{key}:{json.dumps(value, ensure_ascii=False)}")
        return f"<start_function_call>call:{tool_name}{{{','.join(serialized_parts)}}}<end_function_call>"


def _extract_structured_candidate(text: str) -> str:
    if "<start_function_call>" in text:
        matches = re.findall(r"<start_function_call>.*?<end_function_call>", text, flags=re.DOTALL)
        if matches:
            return "".join(matches)
    for pattern in (r"(\[\s*\{.*\}\s*\])", r"(\{.*\})"):
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return match.group(1)
    return text.strip()


def _pick_hf_device(torch: Any, preferred: str = "auto") -> str:
    if preferred != "auto":
        if preferred == "mps":
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            raise RuntimeError("Requested mps device, but MPS is not available.")
        if preferred == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            raise RuntimeError("Requested cuda device, but CUDA is not available.")
        if preferred == "cpu":
            return "cpu"
        raise ValueError(f"Unsupported device selection: {preferred}")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
