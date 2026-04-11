from __future__ import annotations

import json
import re
import time
from ast import literal_eval
from pathlib import Path
from typing import Any, Callable

from gemma4_capability_map.models.base import Runner
from gemma4_capability_map.models.hf_service import ensure_hf_reasoner_service, request_service
from gemma4_capability_map.models.runtime_utils import (
    load_hf_auth_token,
    resolve_mlx_model_id,
    resolve_model_source,
    should_use_local_files_only,
)
from gemma4_capability_map.schemas import Message, ModelTurn, ToolCall, ToolSpec
from gemma4_capability_map.tools.planner import plan_tool_calls, tool_catalog_text
from gemma4_capability_map.tools.validators import normalize_tool_output


class Gemma4Runner(Runner):
    def __init__(
        self,
        model_id: str,
        backend: str = "heuristic",
        max_new_tokens: int = 256,
        device: str = "auto",
        request_timeout_seconds: float = 600.0,
        load_event_hook: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        super().__init__(model_id=model_id, backend=backend)
        self.requested_model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.request_timeout_seconds = request_timeout_seconds
        self._load_event_hook = load_event_hook
        self._processor = None
        self._tokenizer = None
        self._model = None
        self._mlx_config = None
        self._loaded_mode: str | None = None
        self._runtime_device: str | None = None
        self._effective_model_id = model_id
        self._service_info: dict[str, Any] = {}

    def ensure_loaded(self, media: list[str] | None = None) -> dict[str, Any]:
        if self.backend == "hf":
            self._ensure_hf_loaded()
        elif self.backend == "hf_service":
            self._ensure_hf_service_loaded()
        elif self.backend == "mlx":
            self._ensure_mlx_loaded(media or [])
        return self.runtime_info()

    def runtime_info(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "requested_model": self.requested_model_id,
            "effective_model": self._effective_model_id,
            "runtime_device": self._runtime_device,
            "load_mode": self._loaded_mode,
            "configured_device": self.device,
            "request_timeout_seconds": self.request_timeout_seconds,
            "service": self._service_info,
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
            turn = self._generate_oracle(messages, thinking)
        elif self.backend == "heuristic":
            turn = self._generate_heuristic(messages, media, tool_specs, thinking)
        elif self.backend == "hf":
            turn = self._generate_hf(messages, media, tool_specs, thinking, active_max_new_tokens)
        elif self.backend == "hf_service":
            turn = self._generate_hf_service(messages, media, tool_specs, thinking, active_max_new_tokens)
        elif self.backend == "mlx":
            turn = self._generate_mlx(messages, media, tool_specs, thinking, active_max_new_tokens)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
        latency_ms = int((time.perf_counter() - start) * 1000)
        return turn.model_copy(update={"latency_ms": latency_ms})

    def _generate_oracle(self, messages: list[Message], thinking: bool) -> ModelTurn:
        oracle_message = next((message.content for message in messages if message.role == "system" and message.content.startswith("ORACLE_")), "")
        if oracle_message.startswith("ORACLE_NEXT_TOOL_CALL:"):
            payload = json.loads(oracle_message.split(":", 1)[1])
            if isinstance(payload, list):
                tool_calls = [
                    ToolCall(
                        name=item["tool_name"],
                        arguments=item.get("arguments", {}),
                        source_format="oracle",
                        raw=json.dumps(item, ensure_ascii=False),
                    )
                    for item in payload
                ]
                raw = json.dumps(
                    [{"name": call.name, "arguments": call.arguments} for call in tool_calls],
                    ensure_ascii=False,
                )
                return ModelTurn(
                    raw_model_output=raw,
                    normalized_tool_call=tool_calls,
                    final_answer="",
                    thinking_text="oracle-planned step" if thinking else "",
                    runtime_metadata={"backend": "oracle"},
                )
            raw = json.dumps({"name": payload["tool_name"], "arguments": payload.get("arguments", {})})
            return ModelTurn(
                raw_model_output=raw,
                normalized_tool_call=[
                    ToolCall(
                        name=payload["tool_name"],
                        arguments=payload.get("arguments", {}),
                        source_format="oracle",
                        raw=raw,
                    )
                ],
                final_answer="",
                thinking_text="oracle-planned step" if thinking else "",
                runtime_metadata={"backend": "oracle"},
            )
        if oracle_message.startswith("ORACLE_FINAL_ANSWER:"):
            expected = json.loads(oracle_message.split(":", 1)[1])
            final_answer = " ".join(expected) if expected else "Completed."
            return ModelTurn(
                raw_model_output=final_answer,
                final_answer=final_answer,
                thinking_text="oracle-synthesized answer" if thinking else "",
                runtime_metadata={"backend": "oracle"},
            )
        return ModelTurn(raw_model_output="", final_answer="", runtime_metadata={"backend": "oracle"})

    def _generate_heuristic(
        self,
        messages: list[Message],
        media: list[str],
        tool_specs: list[ToolSpec],
        thinking: bool,
    ) -> ModelTurn:
        user_text = "\n".join(message.content for message in messages if message.role in {"user", "tool"})
        if any(message.role == "tool" for message in messages):
            answer = self._synthesize_from_tool_history(messages)
            return ModelTurn(
                raw_model_output=answer,
                final_answer=answer,
                thinking_text="heuristic synthesis" if thinking else "",
                runtime_metadata={"backend": "heuristic"},
            )
        if tool_specs:
            tool_calls = plan_tool_calls(messages, media, tool_specs)
            raw = json.dumps(
                [{"name": call.name, "arguments": call.arguments} for call in tool_calls]
                if len(tool_calls) > 1
                else {"name": tool_calls[0].name, "arguments": tool_calls[0].arguments},
                ensure_ascii=False,
            )
            return ModelTurn(
                raw_model_output=raw,
                normalized_tool_call=normalize_tool_output(raw),
                thinking_text="heuristic tool selection" if thinking else "",
                runtime_metadata={"backend": "heuristic"},
            )
        answer = self._summarize_text(messages, media)
        return ModelTurn(
            raw_model_output=answer,
            final_answer=answer,
            thinking_text="heuristic answer" if thinking else "",
            runtime_metadata={"backend": "heuristic"},
        )

    def _generate_hf(
        self,
        messages: list[Message],
        media: list[str],
        tool_specs: list[ToolSpec],
        thinking: bool,
        max_new_tokens: int,
    ) -> ModelTurn:
        self._ensure_hf_loaded()
        model_mode = self._loaded_mode or ("vision" if _is_edge_multimodal_model(self.requested_model_id) else "text")
        if model_mode == "vision":
            tokenizer = self._processor
            if tokenizer is None:
                raise RuntimeError("HF vision processor was not loaded.")
            prompt_messages = self._build_hf_messages(messages, tool_specs, media, thinking=thinking)
            inputs = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
        else:
            tokenizer = self._tokenizer
            if tokenizer is None:
                raise RuntimeError("HF tokenizer was not loaded.")
            prompt_messages = self._build_hf_text_messages(messages, tool_specs, media, thinking=thinking)
            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = self._build_hf_prompt(prompt_messages)
            inputs = tokenizer(prompt, return_tensors="pt")
        target_device = self._runtime_device or getattr(self._model, "device", None)
        if target_device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(target_device)
        outputs = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(generated, skip_special_tokens=False)
        normalized = normalize_tool_output(text)
        final_answer, thinking_text = _parse_model_response(tokenizer, text)
        return ModelTurn(
            raw_model_output=text,
            normalized_tool_call=normalized,
            final_answer="" if normalized else final_answer,
            thinking_text=thinking_text,
            prompt_tokens=int(inputs["input_ids"].shape[-1]),
            completion_tokens=int(generated.shape[-1]),
            runtime_metadata={"backend": "hf", "runtime_info": self.runtime_info()},
        )

    def _generate_hf_service(
        self,
        messages: list[Message],
        media: list[str],
        tool_specs: list[ToolSpec],
        thinking: bool,
        max_new_tokens: int,
    ) -> ModelTurn:
        self._ensure_hf_service_loaded()
        service = self._service_info
        response = request_service(
            service["socket_path"],
            {
                "type": "generate",
                "messages": [message.model_dump(mode="json") for message in messages],
                "media": media,
                "tool_specs": [spec.model_dump(mode="json", by_alias=True) for spec in tool_specs],
                "thinking": thinking,
                "max_new_tokens": max_new_tokens,
            },
            timeout_seconds=self.request_timeout_seconds,
        )
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "HF reasoner service generation failed."))
        turn = ModelTurn.model_validate(response["turn"])
        metadata = dict(turn.runtime_metadata)
        metadata.update(
            {
                "backend": "hf_service",
                "service_request_id": response.get("request_id"),
                "service_elapsed_ms": response.get("elapsed_ms"),
                "service_id": response.get("service_id"),
                "runtime_info": response.get("runtime_info"),
            }
        )
        return turn.model_copy(update={"runtime_metadata": metadata})

    def _ensure_hf_loaded(self) -> None:
        self._emit_load_event("hf_import_start", detail="Importing HF runtime dependencies.")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the hf extra to use the Hugging Face backend.") from exc
        self._emit_load_event("hf_import_complete", detail="Imported HF runtime dependencies.")

        model_mode = "vision" if _is_edge_multimodal_model(self.requested_model_id) else "text"
        if model_mode == "vision" and self._processor is not None and self._model is not None and self._loaded_mode == model_mode:
            self._emit_load_event("hf_load_cached", detail="Reusing already loaded HF model.", model_mode=model_mode)
            return
        if model_mode == "text" and self._tokenizer is not None and self._model is not None and self._loaded_mode == model_mode:
            self._emit_load_event("hf_load_cached", detail="Reusing already loaded HF model.", model_mode=model_mode)
            return

        self._emit_load_event("hf_source_resolution_start", detail="Resolving model source and auth.")
        token = load_hf_auth_token()
        resolved_source = resolve_model_source(self.requested_model_id)
        local_files_only = should_use_local_files_only(resolved_source)
        self._emit_load_event(
            "hf_load_start",
            detail="Starting HF model load.",
            model_mode=model_mode,
            resolved_source=resolved_source,
            local_files_only=local_files_only,
            auth_token_present=bool(token),
        )
        model_cls = AutoModelForImageTextToText if model_mode == "vision" else AutoModelForCausalLM
        runtime_device = _pick_hf_device(torch, preferred=self.device)
        if model_mode == "vision":
            self._emit_load_event("hf_processor_load_start", detail="Loading HF processor.", resolved_source=resolved_source)
            self._processor = AutoProcessor.from_pretrained(
                resolved_source,
                token=token,
                local_files_only=local_files_only,
            )
            self._tokenizer = None
            self._emit_load_event("hf_processor_load_complete", detail="HF processor loaded.", resolved_source=resolved_source)
        else:
            self._emit_load_event("hf_tokenizer_load_start", detail="Loading HF tokenizer.", resolved_source=resolved_source)
            self._tokenizer = AutoTokenizer.from_pretrained(
                resolved_source,
                token=token,
                local_files_only=local_files_only,
            )
            self._processor = None
            self._emit_load_event("hf_tokenizer_load_complete", detail="HF tokenizer loaded.", resolved_source=resolved_source)
        self._emit_load_event("hf_model_load_start", detail="Loading HF model weights.", runtime_device=runtime_device)
        load_kwargs: dict[str, Any] = {
            "low_cpu_mem_usage": True,
            "token": token,
            "local_files_only": local_files_only,
        }
        if runtime_device == "cuda":
            load_kwargs["torch_dtype"] = "auto"
            load_kwargs["device_map"] = "auto"
            self._model = model_cls.from_pretrained(resolved_source, **load_kwargs)
        else:
            load_kwargs["torch_dtype"] = torch.float16 if runtime_device == "mps" else "auto"
            self._model = model_cls.from_pretrained(resolved_source, **load_kwargs)
            self._emit_load_event(
                "hf_model_to_device_start",
                detail="Moving HF model to runtime device.",
                runtime_device=runtime_device,
            )
            self._model.to(runtime_device)
            self._emit_load_event(
                "hf_model_to_device_complete",
                detail="HF model moved to runtime device.",
                runtime_device=runtime_device,
            )
        self._emit_load_event("hf_model_load_complete", detail="HF model weights loaded.", runtime_device=runtime_device)
        self._loaded_mode = model_mode
        self._runtime_device = runtime_device
        self._effective_model_id = resolved_source
        self.model_id = self._effective_model_id
        self._emit_load_event(
            "hf_ready",
            detail="HF runner is ready.",
            runtime_device=runtime_device,
            load_mode=model_mode,
            effective_model=resolved_source,
        )

    def _ensure_hf_service_loaded(self) -> None:
        service = ensure_hf_reasoner_service(
            model_id=self.requested_model_id,
            device=self.device,
            max_new_tokens=self.max_new_tokens,
        )
        runtime_info = service.get("runtime_info", {})
        self._service_info = {
            "service_id": service.get("service_id"),
            "socket_path": service.get("paths", {}).get("socket_path"),
            "state_path": service.get("paths", {}).get("state_path"),
            "event_log_path": service.get("paths", {}).get("event_log_path"),
            "request_log_path": service.get("paths", {}).get("request_log_path"),
            "stdout_log_path": service.get("paths", {}).get("stdout_log_path"),
            "status": service.get("status"),
            "load_elapsed_ms": service.get("load_elapsed_ms"),
            "requests_completed": service.get("requests_completed"),
        }
        self._runtime_device = str(runtime_info.get("runtime_device") or "service")
        self._loaded_mode = str(runtime_info.get("load_mode") or "service")
        self._effective_model_id = str(runtime_info.get("effective_model") or self.requested_model_id)
        self.model_id = self._effective_model_id

    def _emit_load_event(self, event: str, detail: str = "", **payload: Any) -> None:
        if self._load_event_hook is None:
            return
        body: dict[str, Any] = {"event": event}
        if detail:
            body["detail"] = detail
        body.update(payload)
        self._load_event_hook(body)

    def _generate_mlx(
        self,
        messages: list[Message],
        media: list[str],
        tool_specs: list[ToolSpec],
        thinking: bool,
        max_new_tokens: int,
    ) -> ModelTurn:
        self._ensure_mlx_loaded(media)
        real_media = [ref for ref in media if _is_real_media_ref(ref)]
        if self._loaded_mode == "vision":
            try:
                from mlx_vlm import generate as generate_vlm
                from mlx_vlm.prompt_utils import apply_chat_template
            except ImportError as exc:
                raise RuntimeError("Install the mlx extra to use the MLX backend.") from exc

            prompt_messages = self._build_mlx_messages(messages, tool_specs, thinking=thinking)
            formatted_prompt = apply_chat_template(
                self._processor,
                self._mlx_config or getattr(self._model, "config", None),
                prompt_messages,
                num_images=len(real_media),
                enable_thinking=thinking,
            )
            kwargs: dict[str, Any] = {
                "verbose": False,
                "max_tokens": max_new_tokens,
                "enable_thinking": thinking,
            }
            if real_media:
                generation = generate_vlm(self._model, self._processor, formatted_prompt, real_media, **kwargs)
            else:
                generation = generate_vlm(self._model, self._processor, formatted_prompt, **kwargs)
            text = _mlx_generated_text(generation)
            prompt_tokens = _mlx_prompt_tokens(
                generation,
                fallback=_maybe_count_tokens(_mlx_tokenizer(self._processor), formatted_prompt),
            )
            completion_tokens = _mlx_completion_tokens(
                generation,
                fallback=_maybe_count_tokens(_mlx_tokenizer(self._processor), text),
            )
        else:
            try:
                from mlx_lm import generate as generate_lm
            except ImportError as exc:
                raise RuntimeError("Install the mlx extra to use the MLX backend.") from exc

            prompt_messages = self._build_mlx_messages(messages, tool_specs, thinking=thinking)
            prompt = self._processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            generation = generate_lm(
                self._model,
                self._processor,
                prompt=prompt,
                max_tokens=max_new_tokens,
                verbose=False,
            )
            text = _mlx_generated_text(generation)
            prompt_tokens = _mlx_prompt_tokens(
                generation,
                fallback=_maybe_count_tokens(self._processor, prompt),
            )
            completion_tokens = _mlx_completion_tokens(
                generation,
                fallback=_maybe_count_tokens(self._processor, text),
            )
        normalized = normalize_tool_output(text)
        final_answer, thinking_text = _parse_model_response(self._processor, text)
        return ModelTurn(
            raw_model_output=text,
            normalized_tool_call=normalized,
            final_answer="" if normalized else final_answer,
            thinking_text=thinking_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            runtime_metadata={"backend": "mlx", "runtime_info": self.runtime_info()},
        )

    def _ensure_mlx_loaded(self, media: list[str]) -> None:
        if self.device not in {"auto", "mps"}:
            raise ValueError("The MLX backend only supports auto or mps device selection on Apple Silicon.")
        model_mode = "vision" if _is_edge_multimodal_model(self.requested_model_id) else "text"
        if self._processor is not None and self._model is not None and self._loaded_mode == model_mode:
            return

        effective_model = resolve_mlx_model_id(self.requested_model_id)
        if model_mode == "vision":
            try:
                from mlx_vlm import load as load_vlm
                from mlx_vlm.utils import load_config
            except ImportError as exc:
                raise RuntimeError("Install the mlx extra to use the MLX backend.") from exc

            self._model, self._processor = load_vlm(effective_model)
            self._mlx_config = load_config(effective_model)
        else:
            try:
                from mlx_lm import load as load_lm
            except ImportError as exc:
                raise RuntimeError("Install the mlx extra to use the MLX backend.") from exc

            self._model, self._processor = load_lm(effective_model)
            self._mlx_config = None
        self._loaded_mode = model_mode
        self._runtime_device = "mlx"
        self._effective_model_id = effective_model
        self.model_id = self._effective_model_id

    def _build_hf_messages(
        self,
        messages: list[Message],
        tool_specs: list[ToolSpec],
        media: list[str],
        thinking: bool = False,
    ) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        system_text = self._system_instruction(tool_specs, thinking=thinking, native_thinking=True)
        if system_text:
            prepared.append({"role": "system", "content": [{"type": "text", "text": system_text}]})
        if tool_specs:
            prepared.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "Available tools:\n" + "\n".join(
                                json.dumps(tool.model_dump(mode="json", by_alias=True), ensure_ascii=False) for tool in tool_specs
                            ),
                        },
                    ],
                }
            )
        media_iter = iter(media)
        for message in messages:
            role = self._map_hf_role(message.role)
            content: list[dict[str, str]] = []
            message_media = list(message.image_refs)
            if role == "user" and not message_media and not message.image_refs and message is messages[-1]:
                for ref in media_iter:
                    message_media.append(ref)
            for ref in message_media:
                content.extend(_render_media_content(ref))
            text = message.content
            if message.role == "tool":
                text = f"Tool result:\n{text}"
            if text:
                content.append({"type": "text", "text": text})
            prepared.append({"role": role, "content": content or [{"type": "text", "text": ""}]})
        return prepared

    def _build_hf_text_messages(
        self,
        messages: list[Message],
        tool_specs: list[ToolSpec],
        media: list[str],
        thinking: bool = False,
    ) -> list[dict[str, str]]:
        prepared: list[dict[str, str]] = []
        system_text = self._system_instruction(tool_specs, thinking=thinking, native_thinking=False)
        if system_text:
            prepared.append({"role": "system", "content": system_text})
        if tool_specs:
            prepared.append(
                {
                    "role": "system",
                    "content": "Available tools:\n"
                    + "\n".join(
                        json.dumps(tool.model_dump(mode="json", by_alias=True), ensure_ascii=False) for tool in tool_specs
                    ),
                }
            )
        media_iter = iter(media)
        for message in messages:
            role = self._map_hf_role(message.role)
            text = message.content
            message_media = list(message.image_refs)
            if role == "user" and not message_media and not message.image_refs and message is messages[-1]:
                for ref in media_iter:
                    message_media.append(ref)
            if message_media:
                text = (text + "\n" if text else "") + "Image refs: " + ", ".join(message_media)
            if message.role == "tool":
                text = f"Tool result:\n{text}"
            prepared.append({"role": role, "content": text or ""})
        return prepared

    def _build_hf_prompt(self, messages: list[dict[str, str]]) -> str:
        lines: list[str] = []
        for message in messages:
            role = str(message.get("role", "user")).upper()
            text = str(message.get("content", ""))
            lines.append(f"{role}: {text}".rstrip())
        lines.append("ASSISTANT:")
        return "\n\n".join(lines).strip()

    def _build_mlx_messages(
        self,
        messages: list[Message],
        tool_specs: list[ToolSpec],
        thinking: bool,
    ) -> list[dict[str, str]]:
        prepared: list[dict[str, str]] = []
        system_text = self._system_instruction(tool_specs, thinking=thinking, native_thinking=False)
        if system_text:
            prepared.append({"role": "system", "content": system_text})
        catalog_text = tool_catalog_text(tool_specs)
        if catalog_text:
            prepared.append({"role": "system", "content": catalog_text})
        for message in messages:
            role = self._map_hf_role(message.role)
            text = message.content
            if message.role == "tool":
                text = f"Tool result:\n{text}"
            if message.image_refs:
                text = (text + "\n" if text else "") + "Image refs: " + ", ".join(message.image_refs)
            prepared.append({"role": role, "content": text or ""})
        return prepared

    def _build_mlx_prompt(
        self,
        messages: list[Message],
        tool_specs: list[ToolSpec],
        thinking: bool,
    ) -> str:
        lines: list[str] = []
        system_text = self._system_instruction(tool_specs, thinking=thinking, native_thinking=False)
        if system_text:
            lines.append(f"SYSTEM: {system_text}")
        for message in messages:
            role = self._map_hf_role(message.role).upper()
            text = message.content
            if message.role == "tool":
                text = f"Tool result:\n{text}"
            if message.image_refs:
                text = (text + "\n" if text else "") + "Image refs: " + ", ".join(message.image_refs)
            lines.append(f"{role}: {text}".rstrip())
        lines.append("ASSISTANT:")
        return "\n\n".join(lines).strip()

    def _system_instruction(self, tool_specs: list[ToolSpec], thinking: bool, native_thinking: bool) -> str:
        instructions: list[str] = []
        if thinking and not native_thinking:
            instructions.append("Reason carefully before the final answer, but return only the answer or tool call.")
        if tool_specs:
            instructions.append(
                "You are a structured assistant. "
                'If a tool is needed, respond with a single JSON object or JSON array of objects using {"name": "...", "arguments": {...}}. '
                "Use only the exact tool names listed in the tool catalog. "
                "Never invent a tool name, never rename a field, and prefer a tool call over asking the user for data that is already available through a tool. "
                "If no tool is needed, answer directly."
            )
        return "\n\n".join(instructions)

    def _map_hf_role(self, role: str) -> str:
        if role == "tool":
            return "user"
        return role

    def _pick_tool(self, user_text: str, tool_specs: list[ToolSpec]) -> str:
        lowered = user_text.lower()
        for tool in tool_specs:
            if any(token in lowered for token in tool.name.lower().split("_")):
                return tool.name
        for tool in tool_specs:
            description_words = set(re.findall(r"[a-z0-9]+", tool.description.lower()))
            if description_words.intersection(re.findall(r"[a-z0-9]+", lowered)):
                return tool.name
        return tool_specs[0].name

    def _infer_arguments(self, user_text: str, media: list[str], tool_name: str) -> dict[str, Any]:
        text = user_text.lower()
        if tool_name == "search_events":
            attendee = ""
            if "sarah" in text:
                attendee = "Sarah"
            elif "vendor" in text:
                attendee = "Vendor"
            if "tuesday" in text:
                return {"start_date": "2026-04-15", "end_date": "2026-04-15", "attendee": attendee}
            return {"start_date": "2026-04-10", "end_date": "2026-04-10", "attendee": attendee}
        if tool_name == "update_event":
            event_id_match = re.search(r"(evt-\d+)", text)
            event_id = event_id_match.group(1) if event_id_match else ("evt-009" if "vendor" in text else "evt-001")
            return {"event_id": event_id, "new_start": "2026-04-14T14:00:00", "new_end": "2026-04-14T14:30:00"}
        if tool_name == "find_latest_file":
            return {"directory": "finance", "kind": "budget"}
        if tool_name == "compare_files":
            file_names = re.findall(r"([A-Za-z0-9_./-]+\.csv)", user_text)
            if len(file_names) >= 2:
                return {"file_a": file_names[0], "file_b": file_names[1]}
            if "ops" in text:
                return {"file_a": "ops_budget_mar.csv", "file_b": "ops_budget_apr.csv"}
            return {"file_a": "budget_mar.csv", "file_b": "budget_apr.csv"}
        if tool_name == "find_repo_file":
            if "settings" in text:
                return {"query": "settings"}
            return {"query": "config"}
        if tool_name == "read_repo_file":
            match = re.search(r"([A-Za-z0-9_./-]+\.(?:ya?ml|json|toml|py|md))", user_text)
            return {"path": match.group(1) if match else "config/settings.yaml"}
        if tool_name == "inspect_image":
            match = re.search(r"(img-[a-z0-9-]+)", text)
            if match:
                return {"image_id": match.group(1)}
            if media:
                return {"image_id": media[0]}
            return {"image_id": "img-settings"}
        if tool_name == "create_event":
            return {"title": "Budget review", "start": "2026-04-15T15:00:00", "end": "2026-04-15T15:30:00", "attendees": ["team@example.com"]}
        if tool_name == "propose_patch":
            path_match = re.search(r"([A-Za-z0-9_./-]+\.(?:ya?ml|json|toml))", user_text)
            path = path_match.group(1) if path_match else ("config/billing.yaml" if "billing" in text else "config/settings.yaml")
            if "invoice lock" in text or "billing" in text:
                patch = "invoice_lock: true"
            else:
                patch = "safe_mode: true"
            return {"path": path, "patch": patch}
        return {}

    def _synthesize_from_tool_history(self, messages: list[Message]) -> str:
        tool_messages = [message.content for message in messages if message.role == "tool"]
        if not tool_messages:
            return "No tool results available."
        return "Summary: " + " ".join(tool_messages[-2:])

    def _summarize_text(self, messages: list[Message], media: list[str]) -> str:
        text = messages[-1].content if messages else ""
        if media:
            return f"Processed {len(media)} image inputs. {text}".strip()
        return f"Answer: {text}".strip()


def parse_python_literal(value: str) -> Any:
    try:
        return literal_eval(value)
    except (ValueError, SyntaxError):
        return value.strip("\"'")


def _is_real_media_ref(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://") or Path(value).exists()


def _render_media_content(ref: str) -> list[dict[str, str]]:
    if ref.startswith("http://") or ref.startswith("https://"):
        if ref.lower().endswith((".mp4", ".mov", ".mkv")):
            return [{"type": "video", "video": ref}]
        if ref.lower().endswith((".wav", ".mp3", ".flac")):
            return [{"type": "audio", "audio": ref}]
        return [{"type": "image", "url": ref}]
    path = Path(ref)
    if path.exists():
        if ref.lower().endswith((".wav", ".mp3", ".flac")):
            return [{"type": "audio", "audio": str(path.resolve())}]
        return [{"type": "image", "url": str(path.resolve())}]
    return [{"type": "text", "text": f"[Image ref: {ref}]"}]


def _parse_model_response(processor: Any, text: str) -> tuple[str, str]:
    parsed_thinking = ""
    if hasattr(processor, "parse_response"):
        try:
            parsed = processor.parse_response(text)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            final_answer = _strip_control_tokens(str(parsed.get("answer") or parsed.get("text") or parsed.get("response") or "").strip())
            thinking_text = _strip_control_tokens(str(parsed.get("thinking") or parsed.get("thought") or "").strip())
            if final_answer:
                return final_answer, thinking_text
            parsed_thinking = thinking_text
    thought_with_final = re.search(r"<\|channel\>thought\s*(.*?)<channel\|>(.*)", text, re.DOTALL)
    if thought_with_final:
        return (
            _strip_control_tokens(thought_with_final.group(2).strip()),
            _strip_control_tokens(thought_with_final.group(1).strip()) or parsed_thinking,
        )
    channel_matches = list(re.finditer(r"<\|channel\>([a-zA-Z_]+)\s*", text))
    if channel_matches:
        segments: dict[str, str] = {}
        for index, match in enumerate(channel_matches):
            label = match.group(1).lower()
            start = match.end()
            end = channel_matches[index + 1].start() if index + 1 < len(channel_matches) else len(text)
            segments[label] = text[start:end].strip()
        final_answer = _strip_control_tokens(
            segments.get("final") or segments.get("answer") or segments.get("response") or ""
        )
        thinking_text = _strip_control_tokens(
            segments.get("thought") or segments.get("thinking") or ""
        )
        if final_answer or thinking_text:
            return final_answer, thinking_text or parsed_thinking
    if parsed_thinking:
        return "", parsed_thinking
    return _strip_control_tokens(text.strip()), ""


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


def _is_edge_multimodal_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return "gemma-4-e2b" in lowered or "gemma-4-e4b" in lowered


def _maybe_count_tokens(tokenizer: Any, text: str) -> int:
    if tokenizer is None or not text:
        return 0
    try:
        encoded = tokenizer.encode(text)
    except Exception:
        return 0
    if isinstance(encoded, list):
        return len(encoded)
    if hasattr(encoded, "shape") and encoded.shape:
        return int(encoded.shape[-1])
    return 0


def _mlx_tokenizer(processor: Any) -> Any:
    if hasattr(processor, "tokenizer"):
        return processor.tokenizer
    return processor


def _mlx_generated_text(generation: Any) -> str:
    text = getattr(generation, "text", generation)
    if text is None:
        return ""
    return str(text)


def _mlx_prompt_tokens(generation: Any, fallback: int) -> int:
    prompt_tokens = getattr(generation, "prompt_tokens", None)
    if isinstance(prompt_tokens, int):
        return prompt_tokens
    return fallback


def _mlx_completion_tokens(generation: Any, fallback: int) -> int:
    completion_tokens = getattr(generation, "generation_tokens", None)
    if isinstance(completion_tokens, int):
        return completion_tokens
    return fallback


def _strip_control_tokens(text: str) -> str:
    return re.sub(r"<[^>\n]+>", "", text).strip()
